import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from io import BytesIO
import re
from typing import Dict, List, Optional
import time

# Konfiguration der Streamlit App
st.set_page_config(
    page_title="TimeMoto Zeiterfassung",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für modernes Design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }
    
    .info-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

class DatabaseManager:
    """Robuste Datenbankverbindung mit dauerhafter Speicherung"""
    
    def __init__(self):
        self.connection_string = self._get_connection_string()
        self.max_retries = 3
        self.retry_delay = 2
    
    def _get_connection_string(self) -> str:
        """Erstellt die Verbindungszeichenfolge für neon.tech"""
        # Versuche zuerst DATABASE_URL (empfohlen für Neon.tech)
        database_url = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL"))
        
        if database_url:
            # Bereinige problematische Parameter
            clean_url = database_url.replace("&channel_binding=require", "")
            clean_url = clean_url.replace("channel_binding=require&", "")
            clean_url = clean_url.replace("?channel_binding=require", "?sslmode=require")
            
            # Stelle sicher, dass SSL aktiviert ist
            if "sslmode=" not in clean_url:
                separator = "&" if "?" in clean_url else "?"
                clean_url += f"{separator}sslmode=require"
            
            return clean_url
        
        # Fallback auf einzelne Parameter
        db_host = st.secrets.get("DB_HOST", os.getenv("DB_HOST"))
        db_port = st.secrets.get("DB_PORT", os.getenv("DB_PORT", "5432"))
        db_name = st.secrets.get("DB_NAME", os.getenv("DB_NAME"))
        db_user = st.secrets.get("DB_USER", os.getenv("DB_USER"))
        db_password = st.secrets.get("DB_PASSWORD", os.getenv("DB_PASSWORD"))
        
        if not all([db_host, db_name, db_user, db_password]):
            st.error("⚠️ Datenbankverbindung nicht konfiguriert! Bitte DATABASE_URL oder einzelne DB-Parameter in secrets.toml setzen.")
            st.stop()
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require"
    
    def get_connection(self):
        """Erstellt eine robuste Datenbankverbindung mit Retry-Logik"""
        for attempt in range(self.max_retries):
            try:
                conn = psycopg2.connect(
                    self.connection_string,
                    connect_timeout=30,
                    keepalives=1,
                    keepalives_idle=600,
                    keepalives_interval=30,
                    keepalives_count=3
                )
                
                # Teste die Verbindung
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                
                return conn
                
            except psycopg2.OperationalError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    st.warning(f"⏳ Verbindungsversuch {attempt + 1} fehlgeschlagen, versuche erneut in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error(f"❌ Datenbankverbindung fehlgeschlagen nach {self.max_retries} Versuchen: {e}")
                    return None
            except Exception as e:
                st.error(f"❌ Unerwarteter Datenbankfehler: {e}")
                return None
        
        return None
    
    def create_tables(self) -> bool:
        """Erstellt die erforderlichen Tabellen"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS time_entries (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) NOT NULL,
            entry_date DATE NOT NULL,
            start_time VARCHAR(10),
            end_time VARCHAR(10),
            breaks_duration VARCHAR(50),
            total_duration VARCHAR(10),
            duration_excluding_breaks VARCHAR(10),
            work_schedule VARCHAR(10),
            balance VARCHAR(10),
            absence_name VARCHAR(100),
            remarks TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(username, entry_date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_time_entries_user_date 
        ON time_entries(username, entry_date);
        
        CREATE INDEX IF NOT EXISTS idx_time_entries_date 
        ON time_entries(entry_date);
        
        CREATE INDEX IF NOT EXISTS idx_time_entries_username 
        ON time_entries(username);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_sql)
                    conn.commit()
            return True
        except Exception as e:
            st.error(f"Fehler beim Erstellen der Tabellen: {str(e)}")
            return False
    
    def insert_time_entries(self, df: pd.DataFrame) -> tuple[int, int]:
        """Fügt Zeiterfassungsdaten in die Datenbank ein mit robuster Fehlerbehandlung"""
        inserted_count = 0
        updated_count = 0
        
        if df.empty:
            st.warning("Keine Daten zum Einfügen gefunden.")
            return 0, 0
        
        insert_sql = """
        INSERT INTO time_entries (
            username, entry_date, start_time, end_time, breaks_duration,
            total_duration, duration_excluding_breaks, work_schedule,
            balance, absence_name, remarks
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (username, entry_date) 
        DO UPDATE SET
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time,
            breaks_duration = EXCLUDED.breaks_duration,
            total_duration = EXCLUDED.total_duration,
            duration_excluding_breaks = EXCLUDED.duration_excluding_breaks,
            work_schedule = EXCLUDED.work_schedule,
            balance = EXCLUDED.balance,
            absence_name = EXCLUDED.absence_name,
            remarks = EXCLUDED.remarks,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id, 
        (CASE WHEN created_at = updated_at THEN 'inserted' ELSE 'updated' END) as action
        """
        
        conn = self.get_connection()
        if not conn:
            return 0, 0
        
        try:
            with conn:
                with conn.cursor() as cur:
                    progress_bar = st.progress(0)
                    total_rows = len(df)
                    
                    for index, row in df.iterrows():
                        try:
                            # Überspringe Total-Zeilen
                            if str(row.get('Username', '')).strip().lower() == 'total':
                                continue
                            
                            # Konvertiere Datum
                            entry_date = self._parse_german_date(row.get('Date'))
                            if not entry_date:
                                continue
                            
                            # Bereite Daten vor
                            values = (
                                str(row.get('Username', '')).strip(),
                                entry_date,
                                str(row.get('StartTime', '')),
                                str(row.get('EndTime', '')),
                                str(row.get('Breaks', '')),
                                str(row.get('Duration', '')),
                                str(row.get('DurationExcludingBreaks', '')),
                                str(row.get('WorkSchedule', '')),
                                str(row.get('Balance', '')),
                                str(row.get('AbsenceName', '')) if pd.notna(row.get('AbsenceName')) else None,
                                str(row.get('Remarks', '')) if pd.notna(row.get('Remarks')) else None
                            )
                            
                            # Führe INSERT aus
                            cur.execute(insert_sql, values)
                            result = cur.fetchone()
                            
                            if result:
                                if result[1] == 'inserted':
                                    inserted_count += 1
                                else:
                                    updated_count += 1
                            
                            # Update Progress
                            progress_bar.progress((index + 1) / total_rows)
                            
                        except Exception as row_error:
                            st.warning(f"Zeile {index + 1} übersprungen: {str(row_error)}")
                            continue
                    
                    # Explizit committen
                    conn.commit()
                    progress_bar.empty()
                    
                    # Validiere die Einfügung
                    cur.execute("SELECT COUNT(*) FROM time_entries")
                    total_in_db = cur.fetchone()[0]
                    st.info(f"✅ Daten erfolgreich gespeichert. Gesamt in Datenbank: {total_in_db} Einträge")
                    
        except Exception as e:
            st.error(f"❌ Fehler beim Einfügen der Daten: {str(e)}")
            return 0, 0
        finally:
            conn.close()
        
        return inserted_count, updated_count
    
    def _parse_german_date(self, date_str: str) -> Optional[str]:
        """Konvertiert deutsches Datumsformat zu ISO"""
        if not date_str or pd.isna(date_str):
            return None
        
        # Deutsche Wochentage zu englischen
        german_days = {
            'Montag': 'Monday', 'Dienstag': 'Tuesday', 'Mittwoch': 'Wednesday',
            'Donnerstag': 'Thursday', 'Freitag': 'Friday', 'Samstag': 'Saturday',
            'Sonntag': 'Sunday'
        }
        
        # Deutsche Monate zu englischen
        german_months = {
            'Januar': 'January', 'Februar': 'February', 'März': 'March',
            'April': 'April', 'Mai': 'May', 'Juni': 'June',
            'Juli': 'July', 'August': 'August', 'September': 'September',
            'Oktober': 'October', 'November': 'November', 'Dezember': 'December'
        }
        
        try:
            # Ersetze deutsche Begriffe
            english_date = date_str
            for german, english in german_days.items():
                english_date = english_date.replace(german, english)
            for german, english in german_months.items():
                english_date = english_date.replace(german, english)
            
            # Parse das Datum
            # Format: "Montag, 14. Juli 2025" -> "Monday, 14. July 2025"
            parsed_date = datetime.strptime(english_date, "%A, %d. %B %Y")
            return parsed_date.strftime("%Y-%m-%d")
        except:
            return None
    
    def get_statistics(self) -> Dict:
        """Holt Statistiken aus der Datenbank"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Grundstatistiken
                    cur.execute("SELECT COUNT(*) as total_entries FROM time_entries")
                    total_entries = cur.fetchone()['total_entries']
                    
                    cur.execute("SELECT COUNT(DISTINCT username) as total_users FROM time_entries")
                    total_users = cur.fetchone()['total_users']
                    
                    cur.execute("SELECT COUNT(*) as absences FROM time_entries WHERE absence_name IS NOT NULL")
                    total_absences = cur.fetchone()['absences']
                    
                    # Letzter Import
                    cur.execute("SELECT MAX(created_at) as last_import FROM time_entries")
                    last_import = cur.fetchone()['last_import']
                    
                    return {
                        'total_entries': total_entries,
                        'total_users': total_users,
                        'total_absences': total_absences,
                        'last_import': last_import
                    }
        except Exception as e:
            st.error(f"Fehler beim Abrufen der Statistiken: {str(e)}")
            return {}
    
    def get_time_entries(self, limit: int = 100) -> pd.DataFrame:
        """Holt Zeiterfassungsdaten aus der Datenbank"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT username, entry_date, start_time, end_time, 
                       breaks_duration, total_duration, duration_excluding_breaks,
                       work_schedule, balance, absence_name, remarks, created_at
                FROM time_entries 
                ORDER BY entry_date DESC, username
                LIMIT %s
                """
                df = pd.read_sql(query, conn, params=[limit])
                return df
        except Exception as e:
            st.error(f"Fehler beim Abrufen der Daten: {str(e)}")
            return pd.DataFrame()

class TimeMotoApp:
    """Hauptanwendungsklasse"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def run(self):
        """Startet die Streamlit Anwendung"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>⏰ TimeMoto Zeiterfassung</h1>
            <p>Professionelle Verwaltung von Zeiterfassungsdaten</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar Navigation
        st.sidebar.title("🔧 Navigation")
        page = st.sidebar.selectbox(
            "Seite auswählen:",
            ["📊 Dashboard", "📤 Daten importieren", "📋 Daten anzeigen", "⚙️ Einstellungen"]
        )
        
        # Seiten-Routing
        if page == "📊 Dashboard":
            self.show_dashboard()
        elif page == "📤 Daten importieren":
            self.show_import_page()
        elif page == "📋 Daten anzeigen":
            self.show_data_view()
        elif page == "⚙️ Einstellungen":
            self.show_settings()
    
    def show_dashboard(self):
        """Zeigt das Dashboard"""
        st.header("📊 Dashboard")
        
        # Datenbankverbindung testen
        if not self.test_database_connection():
            return
        
        # Statistiken abrufen
        stats = self.db_manager.get_statistics()
        
        if stats:
            # Metriken anzeigen
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="📝 Gesamte Einträge",
                    value=stats.get('total_entries', 0)
                )
            
            with col2:
                st.metric(
                    label="👥 Mitarbeiter",
                    value=stats.get('total_users', 0)
                )
            
            with col3:
                st.metric(
                    label="🏥 Abwesenheiten",
                    value=stats.get('total_absences', 0)
                )
            
            with col4:
                last_import = stats.get('last_import')
                if last_import:
                    import_text = last_import.strftime("%d.%m.%Y %H:%M")
                else:
                    import_text = "Noch keine Daten"
                
                st.metric(
                    label="📅 Letzter Import",
                    value=import_text
                )
        
        # Visualisierungen
        if stats and stats.get('total_entries', 0) > 0:
            self.show_charts()
    
    def show_import_page(self):
        """Zeigt die Import-Seite"""
        st.header("📤 Daten importieren")
        
        # Datenbankverbindung testen
        if not self.test_database_connection():
            return
        
        # Tabellen erstellen
        if not self.db_manager.create_tables():
            st.error("❌ Fehler beim Erstellen der Datenbanktabellen!")
            return
        
        st.markdown("""
        <div class="info-box">
            <h4>📋 Unterstützte Dateiformate</h4>
            <ul>
                <li><strong>Excel (.xlsx, .xls):</strong> TimeMoto Export Dateien</li>
                <li><strong>CSV (.csv):</strong> Komma-getrennte Werte</li>
            </ul>
            <p><strong>Erwartete Spalten:</strong> Username, Date, StartTime, EndTime, Breaks, Duration, DurationExcludingBreaks, WorkSchedule, Balance, AbsenceName, Remarks</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Datei-Upload
        uploaded_file = st.file_uploader(
            "📁 TimeMoto Export-Datei auswählen",
            type=['xlsx', 'xls', 'csv'],
            help="Unterstützte Formate: Excel (.xlsx, .xls) und CSV (.csv)"
        )
        
        if uploaded_file is not None:
            try:
                # Datei lesen
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Datei erfolgreich gelesen: {len(df)} Zeilen gefunden")
                
                # Datenvorschau
                with st.expander("👀 Datenvorschau", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Validierung
                required_columns = ['Username', 'Date', 'StartTime', 'EndTime', 'Duration']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Fehlende Spalten: {', '.join(missing_columns)}")
                    st.info("💡 Stellen Sie sicher, dass die Datei die richtige Struktur hat.")
                    return
                
                # Import-Button
                if st.button("🚀 Daten importieren", type="primary", use_container_width=True):
                    with st.spinner("Importiere Daten..."):
                        inserted, updated = self.db_manager.insert_time_entries(df)
                    
                    if inserted > 0 or updated > 0:
                        st.markdown(f"""
                        <div class="success-message">
                            <h4>✅ Import erfolgreich!</h4>
                            <ul>
                                <li><strong>Neue Einträge:</strong> {inserted}</li>
                                <li><strong>Aktualisierte Einträge:</strong> {updated}</li>
                                <li><strong>Gesamt verarbeitet:</strong> {inserted + updated}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Cache leeren für Dashboard-Update
                        st.cache_data.clear()
                    else:
                        st.warning("⚠️ Keine Daten wurden importiert. Möglicherweise sind alle Einträge bereits vorhanden.")
            
            except Exception as e:
                st.error(f"❌ Fehler beim Verarbeiten der Datei: {str(e)}")
    
    def show_data_view(self):
        """Zeigt die Datenansicht"""
        st.header("📋 Zeiterfassungsdaten")
        
        if not self.test_database_connection():
            return
        
        # Filter-Optionen
        col1, col2 = st.columns(2)
        
        with col1:
            limit = st.selectbox(
                "📊 Anzahl Einträge:",
                [50, 100, 200, 500, 1000],
                index=1
            )
        
        with col2:
            if st.button("🔄 Daten aktualisieren"):
                st.cache_data.clear()
        
        # Daten laden
        df = self.db_manager.get_time_entries(limit)
        
        if not df.empty:
            st.info(f"📈 {len(df)} Einträge gefunden")
            
            # Interaktive Tabelle
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "entry_date": st.column_config.DateColumn("Datum"),
                    "username": st.column_config.TextColumn("Mitarbeiter"),
                    "start_time": st.column_config.TextColumn("Start"),
                    "end_time": st.column_config.TextColumn("Ende"),
                    "total_duration": st.column_config.TextColumn("Dauer"),
                    "balance": st.column_config.TextColumn("Saldo"),
                    "absence_name": st.column_config.TextColumn("Abwesenheit")
                }
            )
            
            # Export-Option
            if st.button("📥 Als CSV exportieren"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 CSV herunterladen",
                    data=csv,
                    file_name=f"zeiterfassung_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("📭 Keine Daten gefunden. Importieren Sie zuerst TimeMoto Daten.")
    
    def show_settings(self):
        """Zeigt die Einstellungen"""
        st.header("⚙️ Einstellungen")
        
        # Datenbankstatus
        st.subheader("🗄️ Datenbankverbindung")
        
        if self.test_database_connection():
            st.success("✅ Datenbankverbindung aktiv")
            
            # Tabellenstatus
            try:
                stats = self.db_manager.get_statistics()
                if stats:
                    st.info(f"📊 Datenbank enthält {stats.get('total_entries', 0)} Einträge")
            except:
                pass
        else:
            st.error("❌ Datenbankverbindung fehlgeschlagen")
        
        # Konfigurationshilfe
        st.subheader("🔧 Konfiguration")
        
        st.markdown("""
        <div class="info-box">
            <h4>📝 Neon.tech Konfiguration</h4>
            <p><strong>Option 1 (Empfohlen):</strong> Verwenden Sie DATABASE_URL in <code>secrets.toml</code>:</p>
            <pre>
[secrets]
DATABASE_URL = "postgresql://username:password@host/database?sslmode=require"
            </pre>
            
            <p><strong>Option 2:</strong> Einzelne Parameter in <code>secrets.toml</code>:</p>
            <pre>
[secrets]
DB_HOST = "your-neon-host.neon.tech"
DB_PORT = "5432"
DB_NAME = "your_database_name"
DB_USER = "your_username"
DB_PASSWORD = "your_password"
            </pre>
            
            <p><strong>Für Streamlit Cloud:</strong> Setzen Sie die gleichen Werte in den App-Secrets.</p>
            
            <h5>🔍 Troubleshooting:</h5>
            <ul>
                <li>Entfernen Sie <code>channel_binding=require</code> aus der URL</li>
                <li>Verwenden Sie nur <code>sslmode=require</code></li>
                <li>Stellen Sie sicher, dass der Pooler-Endpoint verwendet wird</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def show_charts(self):
        """Zeigt Diagramme und Visualisierungen"""
        st.subheader("📈 Visualisierungen")
        
        # Daten für Charts laden
        df = self.db_manager.get_time_entries(1000)
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Mitarbeiter-Verteilung
                user_counts = df['username'].value_counts()
                fig_users = px.bar(
                    x=user_counts.values,
                    y=user_counts.index,
                    orientation='h',
                    title="Einträge pro Mitarbeiter",
                    labels={'x': 'Anzahl Einträge', 'y': 'Mitarbeiter'}
                )
                fig_users.update_layout(height=400)
                st.plotly_chart(fig_users, use_container_width=True)
            
            with col2:
                # Abwesenheiten
                absence_data = df[df['absence_name'].notna()]
                if not absence_data.empty:
                    absence_counts = absence_data['absence_name'].value_counts()
                    fig_absence = px.pie(
                        values=absence_counts.values,
                        names=absence_counts.index,
                        title="Verteilung der Abwesenheiten"
                    )
                    fig_absence.update_layout(height=400)
                    st.plotly_chart(fig_absence, use_container_width=True)
                else:
                    st.info("Keine Abwesenheitsdaten gefunden")
    
    def test_database_connection(self) -> bool:
        """Testet die Datenbankverbindung"""
        try:
            conn = self.db_manager.get_connection()
            if conn:
                conn.close()
                return True
            return False
        except:
            return False

# Anwendung starten
if __name__ == "__main__":
    app = TimeMotoApp()
    app.run()
