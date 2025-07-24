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
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustDatabaseManager:
    """Robuste Datenbankverbindung mit dauerhafter Speicherung"""
    
    def __init__(self):
        self.connection_string = self._get_connection_string()
        self.max_retries = 3
        self.retry_delay = 2
    
    def _get_connection_string(self) -> str:
        """Erstellt die Verbindungszeichenfolge"""
        # Option 1: DATABASE_URL
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
        
        # Option 2: Einzelne Parameter
        db_host = st.secrets.get("DB_HOST", os.getenv("DB_HOST"))
        db_port = st.secrets.get("DB_PORT", os.getenv("DB_PORT", "5432"))
        db_name = st.secrets.get("DB_NAME", os.getenv("DB_NAME"))
        db_user = st.secrets.get("DB_USER", os.getenv("DB_USER"))
        db_password = st.secrets.get("DB_PASSWORD", os.getenv("DB_PASSWORD"))
        
        if not all([db_host, db_name, db_user, db_password]):
            raise ValueError("Datenbankparameter unvollständig. Bitte DATABASE_URL oder alle DB_* Parameter setzen.")
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require"
    
    def get_connection(self, autocommit=False):
        """Erstellt eine robuste Datenbankverbindung"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Datenbankverbindung Versuch {attempt + 1}/{self.max_retries}")
                
                conn = psycopg2.connect(
                    self.connection_string,
                    connect_timeout=30,
                    keepalives=1,
                    keepalives_idle=600,
                    keepalives_interval=30,
                    keepalives_count=3
                )
                
                if autocommit:
                    conn.autocommit = True
                
                # Teste die Verbindung
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                
                logger.info("Datenbankverbindung erfolgreich")
                return conn
                
            except psycopg2.OperationalError as e:
                error_msg = str(e).lower()
                logger.warning(f"Verbindungsversuch {attempt + 1} fehlgeschlagen: {e}")
                
                if attempt < self.max_retries - 1:
                    if "timeout" in error_msg or "connection" in error_msg:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.info(f"Warte {wait_time} Sekunden vor nächstem Versuch...")
                        time.sleep(wait_time)
                        continue
                
                logger.error("Maximale Anzahl Verbindungsversuche erreicht")
                raise
                
            except Exception as e:
                logger.error(f"Unerwarteter Datenbankfehler: {e}")
                raise
        
        raise psycopg2.OperationalError("Konnte keine Datenbankverbindung herstellen")
    
    def create_tables_with_validation(self) -> bool:
        """Erstellt Tabellen und validiert sie"""
        create_sql = """
        -- Lösche Tabelle falls sie existiert (für Debugging)
        -- DROP TABLE IF EXISTS time_entries CASCADE;
        
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
            
            -- Unique constraint für Duplikate vermeiden
            CONSTRAINT unique_user_date UNIQUE(username, entry_date)
        );
        
        -- Indices für Performance
        CREATE INDEX IF NOT EXISTS idx_time_entries_user_date 
        ON time_entries(username, entry_date);
        
        CREATE INDEX IF NOT EXISTS idx_time_entries_date 
        ON time_entries(entry_date);
        
        CREATE INDEX IF NOT EXISTS idx_time_entries_username 
        ON time_entries(username);
        
        -- Trigger für updated_at
        CREATE OR REPLACE FUNCTION update_modified_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        DROP TRIGGER IF EXISTS update_time_entries_updated_at ON time_entries;
        CREATE TRIGGER update_time_entries_updated_at 
            BEFORE UPDATE ON time_entries 
            FOR EACH ROW EXECUTE FUNCTION update_modified_column();
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Führe CREATE-Statements aus
                    for statement in create_sql.split(';'):
                        statement = statement.strip()
                        if statement and not statement.startswith('--'):
                            cur.execute(statement)
                    
                    conn.commit()
                    logger.info("Tabellen erfolgreich erstellt")
                    
                    # Validiere Tabellenerstellung
                    cur.execute("""
                        SELECT column_name, data_type, is_nullable 
                        FROM information_schema.columns 
                        WHERE table_name = 'time_entries'
                        ORDER BY ordinal_position
                    """)
                    
                    columns = cur.fetchall()
                    if len(columns) >= 10:  # Mindestens 10 Spalten erwartet
                        logger.info(f"Tabelle validiert: {len(columns)} Spalten gefunden")
                        return True
                    else:
                        logger.error("Tabellenerstellung fehlgeschlagen - zu wenige Spalten")
                        return False
                        
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Tabellen: {e}")
            st.error(f"❌ Tabellenerstellung fehlgeschlagen: {str(e)}")
            return False
    
    def insert_time_entries_robust(self, df: pd.DataFrame) -> tuple[int, int, List[str]]:
        """Robuste Dateneinfügung mit detailliertem Logging"""
        inserted_count = 0
        updated_count = 0
        errors = []
        
        if df.empty:
            return 0, 0, ["Keine Daten zum Einfügen"]
        
        # Bereinige DataFrame
        df_clean = self._clean_dataframe(df)
        
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
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for index, row in df_clean.iterrows():
                        try:
                            # Überspringe Total-Zeilen
                            if str(row.get('Username', '')).strip().lower() == 'total':
                                continue
                            
                            # Konvertiere Datum
                            entry_date = self._parse_german_date(row.get('Date'))
                            if not entry_date:
                                errors.append(f"Zeile {index + 1}: Ungültiges Datum '{row.get('Date')}'")
                                continue
                            
                            # Bereite Daten vor
                            values = (
                                str(row.get('Username', '')).strip(),
                                entry_date,
                                self._clean_string(row.get('StartTime')),
                                self._clean_string(row.get('EndTime')),
                                self._clean_string(row.get('Breaks')),
                                self._clean_string(row.get('Duration')),
                                self._clean_string(row.get('DurationExcludingBreaks')),
                                self._clean_string(row.get('WorkSchedule')),
                                self._clean_string(row.get('Balance')),
                                self._clean_string(row.get('AbsenceName')) if pd.notna(row.get('AbsenceName')) else None,
                                self._clean_string(row.get('Remarks')) if pd.notna(row.get('Remarks')) else None
                            )
                            
                            # Führe INSERT aus
                            cur.execute(insert_sql, values)
                            result = cur.fetchone()
                            
                            if result:
                                if result[1] == 'inserted':
                                    inserted_count += 1
                                else:
                                    updated_count += 1
                            
                            logger.debug(f"Zeile {index + 1} erfolgreich verarbeitet: {row.get('Username')} - {entry_date}")
                            
                        except Exception as row_error:
                            error_msg = f"Zeile {index + 1} ({row.get('Username', 'Unbekannt')}): {str(row_error)}"
                            errors.append(error_msg)
                            logger.warning(error_msg)
                            continue
                    
                    # Explizit committen
                    conn.commit()
                    logger.info(f"Daten erfolgreich committet: {inserted_count} eingefügt, {updated_count} aktualisiert")
                    
                    # Validiere Einfügung
                    cur.execute("SELECT COUNT(*) FROM time_entries")
                    total_count = cur.fetchone()[0]
                    logger.info(f"Gesamtzahl Einträge in Datenbank: {total_count}")
                    
        except Exception as e:
            error_msg = f"Datenbankfehler beim Einfügen: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        return inserted_count, updated_count, errors
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereinigt DataFrame vor der Einfügung"""
        df_clean = df.copy()
        
        # Entferne leere Zeilen
        df_clean = df_clean.dropna(how='all')
        
        # Fülle NaN-Werte
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('')
        
        return df_clean
    
    def _clean_string(self, value) -> str:
        """Bereinigt String-Werte"""
        if pd.isna(value):
            return ''
        return str(value).strip()
    
    def _parse_german_date(self, date_str: str) -> Optional[str]:
        """Konvertiert deutsches Datumsformat zu ISO"""
        if not date_str or pd.isna(date_str):
            return None
        
        german_days = {
            'Montag': 'Monday', 'Dienstag': 'Tuesday', 'Mittwoch': 'Wednesday',
            'Donnerstag': 'Thursday', 'Freitag': 'Friday', 'Samstag': 'Saturday',
            'Sonntag': 'Sunday'
        }
        
        german_months = {
            'Januar': 'January', 'Februar': 'February', 'März': 'March',
            'April': 'April', 'Mai': 'May', 'Juni': 'June',
            'Juli': 'July', 'August': 'August', 'September': 'September',
            'Oktober': 'October', 'November': 'November', 'Dezember': 'December'
        }
        
        try:
            english_date = str(date_str)
            for german, english in german_days.items():
                english_date = english_date.replace(german, english)
            for german, english in german_months.items():
                english_date = english_date.replace(german, english)
            
            parsed_date = datetime.strptime(english_date, "%A, %d. %B %Y")
            return parsed_date.strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Datum konnte nicht geparst werden: {date_str} - {e}")
            return None
    
    def test_database_connection(self) -> tuple[bool, str, Dict]:
        """Umfassender Datenbankverbindungstest"""
        try:
            start_time = time.time()
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Grundlegende Verbindung
                    cur.execute("SELECT version()")
                    version = cur.fetchone()['version']
                    
                    # Tabellen prüfen
                    cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = 'time_entries'
                    """)
                    tables = cur.fetchall()
                    
                    # Datenanzahl prüfen
                    if tables:
                        cur.execute("SELECT COUNT(*) as count FROM time_entries")
                        count_result = cur.fetchone()
                        entry_count = count_result['count'] if count_result else 0
                    else:
                        entry_count = 0
                    
                    # Letzter Eintrag
                    last_entry = None
                    if entry_count > 0:
                        cur.execute("""
                            SELECT username, entry_date, created_at 
                            FROM time_entries 
                            ORDER BY created_at DESC 
                            LIMIT 1
                        """)
                        last_entry = cur.fetchone()
                    
                    connection_time = round(time.time() - start_time, 2)
                    
                    info = {
                        'version': version,
                        'connection_time': connection_time,
                        'tables_exist': len(tables) > 0,
                        'entry_count': entry_count,
                        'last_entry': dict(last_entry) if last_entry else None
                    }
                    
                    return True, f"✅ Verbindung erfolgreich ({connection_time}s)", info
                    
        except Exception as e:
            return False, f"❌ Verbindung fehlgeschlagen: {str(e)}", {}
    
    def get_statistics(self) -> Dict:
        """Holt erweiterte Statistiken"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    stats = {}
                    
                    # Grundstatistiken
                    cur.execute("SELECT COUNT(*) as total_entries FROM time_entries")
                    result = cur.fetchone()
                    stats['total_entries'] = result['total_entries'] if result else 0
                    
                    cur.execute("SELECT COUNT(DISTINCT username) as total_users FROM time_entries")
                    result = cur.fetchone()
                    stats['total_users'] = result['total_users'] if result else 0
                    
                    cur.execute("SELECT COUNT(*) as absences FROM time_entries WHERE absence_name IS NOT NULL AND absence_name != ''")
                    result = cur.fetchone()
                    stats['total_absences'] = result['absences'] if result else 0
                    
                    # Zeitstatistiken
                    cur.execute("SELECT MIN(entry_date) as first_date, MAX(entry_date) as last_date FROM time_entries")
                    result = cur.fetchone()
                    if result and result['first_date']:
                        stats['date_range'] = {
                            'first_date': result['first_date'],
                            'last_date': result['last_date']
                        }
                    
                    # Letzter Import
                    cur.execute("SELECT MAX(created_at) as last_import FROM time_entries")
                    result = cur.fetchone()
                    stats['last_import'] = result['last_import'] if result else None
                    
                    # Datenqualität
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(CASE WHEN start_time != '' AND start_time IS NOT NULL THEN 1 END) as with_start_time,
                            COUNT(CASE WHEN end_time != '' AND end_time IS NOT NULL THEN 1 END) as with_end_time
                        FROM time_entries
                    """)
                    result = cur.fetchone()
                    if result:
                        stats['data_quality'] = dict(result)
                    
                    return stats
                    
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Statistiken: {e}")
            return {'error': str(e)}
    
    def get_time_entries(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """Holt Zeiterfassungsdaten mit Pagination"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT 
                    id, username, entry_date, start_time, end_time, 
                    breaks_duration, total_duration, duration_excluding_breaks,
                    work_schedule, balance, absence_name, remarks, 
                    created_at, updated_at
                FROM time_entries 
                ORDER BY entry_date DESC, username
                LIMIT %s OFFSET %s
                """
                df = pd.read_sql(query, conn, params=[limit, offset])
                logger.info(f"Abgerufen: {len(df)} Einträge (Limit: {limit}, Offset: {offset})")
                return df
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Daten: {e}")
            st.error(f"❌ Fehler beim Laden der Daten: {str(e)}")
            return pd.DataFrame()
    
    def delete_all_entries(self) -> bool:
        """Löscht alle Einträge (für Debugging)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM time_entries")
                    deleted_count = cur.rowcount
                    conn.commit()
                    logger.info(f"Alle Einträge gelöscht: {deleted_count}")
                    return True
        except Exception as e:
            logger.error(f"Fehler beim Löschen: {e}")
            return False

# Streamlit App für Datenbanktest
def show_database_diagnostics():
    """Zeigt umfassende Datenbankdiagnostik"""
    st.header("🔧 Datenbankdiagnostik")
    
    try:
        db_manager = RobustDatabaseManager()
        
        # Verbindungstest
        with st.expander("🔗 Verbindungstest", expanded=True):
            if st.button("Verbindung testen"):
                with st.spinner("Teste Datenbankverbindung..."):
                    success, message, info = db_manager.test_database_connection()
                
                if success:
                    st.success(message)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Verbindungszeit", f"{info.get('connection_time', 0)}s")
                        st.metric("Einträge in DB", info.get('entry_count', 0))
                    
                    with col2:
                        st.metric("Tabellen vorhanden", "✅" if info.get('tables_exist') else "❌")
                        if info.get('last_entry'):
                            last = info['last_entry']
                            st.write(f"**Letzter Eintrag:** {last.get('username')} am {last.get('entry_date')}")
                    
                    st.json(info)
                else:
                    st.error(message)
        
        # Tabellenerstellung
        with st.expander("🏗️ Tabellen-Management"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Tabellen erstellen/aktualisieren"):
                    with st.spinner("Erstelle Tabellen..."):
                        success = db_manager.create_tables_with_validation()
                    
                    if success:
                        st.success("✅ Tabellen erfolgreich erstellt/aktualisiert")
                    else:
                        st.error("❌ Fehler beim Erstellen der Tabellen")
            
            with col2:
                if st.button("⚠️ Alle Daten löschen"):
                    if st.session_state.get('confirm_delete'):
                        success = db_manager.delete_all_entries()
                        if success:
                            st.success("Alle Einträge gelöscht")
                        st.session_state.confirm_delete = False
                    else:
                        st.session_state.confirm_delete = True
                        st.warning("Nochmal klicken zum Bestätigen")
        
        # Statistiken
        with st.expander("📊 Datenbankstatistiken"):
            if st.button("Statistiken aktualisieren"):
                stats = db_manager.get_statistics()
                
                if 'error' not in stats:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Gesamteinträge", stats.get('total_entries', 0))
                        st.metric("Mitarbeiter", stats.get('total_users', 0))
                    
                    with col2:
                        st.metric("Abwesenheiten", stats.get('total_absences', 0))
                        if stats.get('last_import'):
                            st.write(f"**Letzter Import:** {stats['last_import']}")
                    
                    with col3:
                        if stats.get('date_range'):
                            st.write(f"**Zeitraum:** {stats['date_range']['first_date']} bis {stats['date_range']['last_date']}")
                        
                        if stats.get('data_quality'):
                            quality = stats['data_quality']
                            completion = round((quality['with_start_time'] / max(quality['total'], 1)) * 100, 1)
                            st.metric("Datenqualität", f"{completion}%")
                    
                    st.json(stats)
                else:
                    st.error(f"Fehler bei Statistiken: {stats['error']}")
        
        # Datenvorschau
        with st.expander("👀 Datenvorschau"):
            col1, col2 = st.columns(2)
            
            with col1:
                limit = st.number_input("Anzahl Einträge", min_value=1, max_value=1000, value=10)
            
            with col2:
                if st.button("Daten laden"):
                    df = db_manager.get_time_entries(limit=limit)
                    
                    if not df.empty:
                        st.success(f"✅ {len(df)} Einträge geladen")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("Keine Daten gefunden")
        
    except Exception as e:
        st.error(f"❌ Fehler bei Datenbankdiagnostik: {str(e)}")
        logger.error(f"Diagnostik-Fehler: {e}")

if __name__ == "__main__":
    st.set_page_config(page_title="Datenbankdiagnostik", layout="wide")
    show_database_diagnostics()
