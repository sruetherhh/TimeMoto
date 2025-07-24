import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional
import numpy as np

# Streamlit Konfiguration
st.set_page_config(
    page_title="TimeMoto Analytics",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für Analytics Dashboard
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .analytics-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .insight-box {
        background: linear-gradient(45deg, #e3f2fd, #f3e5f5);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

class TimeMotoAnalytics:
    """Erweiterte Analytics-Klasse für TimeMoto Daten"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = self.prepare_data(df)
        self.insights = {}
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereitet die Daten für Analysen vor"""
        # Kopie erstellen
        df = df.copy()
        
        # Datum parsen
        df['parsed_date'] = df['Date'].apply(self.parse_german_date)
        df['weekday'] = pd.to_datetime(df['parsed_date']).dt.day_name()
        df['week_number'] = pd.to_datetime(df['parsed_date']).dt.isocalendar().week
        
        # Arbeitscodes aus Remarks extrahieren
        df['work_code'] = df['Remarks'].apply(self.extract_work_code)
        
        # Balance zu numerisch konvertieren
        df['balance_minutes'] = df['Balance'].apply(self.time_to_minutes)
        df['duration_minutes'] = df['DurationExcludingBreaks'].apply(self.time_to_minutes)
        
        # Arbeitszeitttyp klassifizieren
        df['time_type'] = df.apply(self.classify_time_type, axis=1)
        
        # Total-Zeilen entfernen
        df = df[df['Username'] != 'Total'].copy()
        
        return df
    
    def parse_german_date(self, date_str: str) -> Optional[str]:
        """Konvertiert deutsches Datum zu ISO"""
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
            english_date = date_str
            for german, english in german_days.items():
                english_date = english_date.replace(german, english)
            for german, english in german_months.items():
                english_date = english_date.replace(german, english)
            
            parsed_date = datetime.strptime(english_date, "%A, %d. %B %Y")
            return parsed_date.strftime("%Y-%m-%d")
        except:
            return None
    
    def extract_work_code(self, remarks: str) -> Optional[str]:
        """Extrahiert Arbeitscodesaus Remarks"""
        if not remarks or pd.isna(remarks):
            return None
        
        # Suche nach "Arbeitscodes: XXXX"
        match = re.search(r'Arbeitscodes:\s*([^.\r\n]+)', str(remarks))
        if match:
            return match.group(1).strip()
        return None
    
    def time_to_minutes(self, time_str: str) -> int:
        """Konvertiert Zeitstring zu Minuten"""
        if not time_str or pd.isna(time_str) or time_str in ['', '-']:
            return 0
        
        try:
            # Entferne Vorzeichen
            clean_time = str(time_str).replace('+', '').replace('-', '')
            
            # Parse HH:MM Format
            if ':' in clean_time:
                parts = clean_time.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                total_minutes = hours * 60 + minutes
                
                # Berücksichtige negatives Vorzeichen
                if str(time_str).startswith('-'):
                    total_minutes = -total_minutes
                
                return total_minutes
        except:
            pass
        
        return 0
    
    def classify_time_type(self, row) -> str:
        """Klassifiziert den Arbeitszeitttyp"""
        start_time = str(row['StartTime'])
        end_time = str(row['EndTime'])
        absence = row['AbsenceName']
        
        if absence and not pd.isna(absence):
            return 'Abwesenheit'
        elif start_time == '-' or end_time == '-':
            return 'Keine Erfassung'
        elif start_time == '<':
            return 'Früher Beginn'
        elif end_time == '>':
            return 'Spätes Ende'
        elif start_time and end_time and start_time != '' and end_time != '':
            return 'Normal'
        else:
            return 'Unbekannt'
    
    def get_employee_summary(self) -> pd.DataFrame:
        """Erstellt Mitarbeiter-Zusammenfassung"""
        summary = self.df.groupby('Username').agg({
            'parsed_date': 'count',
            'balance_minutes': 'sum',
            'duration_minutes': 'sum',
            'AbsenceName': lambda x: x.notna().sum(),
            'work_code': lambda x: x.notna().sum()
        }).reset_index()
        
        summary.columns = ['Mitarbeiter', 'Arbeitstage', 'Saldo_Minuten', 'Arbeitszeit_Minuten', 'Abwesenheiten', 'Projekte']
        
        # Konvertiere Minuten zurück zu Stunden
        summary['Saldo_Stunden'] = round(summary['Saldo_Minuten'] / 60, 2)
        summary['Arbeitszeit_Stunden'] = round(summary['Arbeitszeit_Minuten'] / 60, 2)
        
        return summary
    
    def get_daily_analysis(self) -> pd.DataFrame:
        """Tägliche Analyse"""
        daily = self.df.groupby('parsed_date').agg({
            'Username': 'nunique',
            'balance_minutes': 'sum',
            'duration_minutes': 'sum',
            'AbsenceName': lambda x: x.notna().sum(),
            'time_type': lambda x: (x == 'Normal').sum()
        }).reset_index()
        
        daily.columns = ['Datum', 'Mitarbeiter_Anzahl', 'Gesamt_Saldo_Min', 'Gesamt_Arbeitszeit_Min', 'Abwesenheiten', 'Normale_Erfassungen']
        daily['Gesamt_Saldo_Std'] = round(daily['Gesamt_Saldo_Min'] / 60, 2)
        daily['Gesamt_Arbeitszeit_Std'] = round(daily['Gesamt_Arbeitszeit_Min'] / 60, 2)
        
        return daily
    
    def get_work_code_analysis(self) -> pd.DataFrame:
        """Analysiert Arbeitscodes/Projekte"""
        work_codes = self.df[self.df['work_code'].notna()]
        
        if work_codes.empty:
            return pd.DataFrame()
        
        analysis = work_codes.groupby(['work_code', 'Username']).agg({
            'parsed_date': 'count',
            'duration_minutes': 'sum'
        }).reset_index()
        
        analysis.columns = ['Projekt', 'Mitarbeiter', 'Tage', 'Arbeitszeit_Minuten']
        analysis['Arbeitszeit_Stunden'] = round(analysis['Arbeitszeit_Minuten'] / 60, 2)
        
        return analysis
    
    def get_absence_analysis(self) -> pd.DataFrame:
        """Analysiert Abwesenheiten"""
        absences = self.df[self.df['AbsenceName'].notna()]
        
        if absences.empty:
            return pd.DataFrame()
        
        analysis = absences.groupby(['AbsenceName', 'Username']).agg({
            'parsed_date': 'count'
        }).reset_index()
        
        analysis.columns = ['Abwesenheitstyp', 'Mitarbeiter', 'Tage']
        
        return analysis
    
    def generate_insights(self) -> Dict:
        """Generiert automatische Insights"""
        insights = {}
        
        # Mitarbeiter mit meisten Überstunden
        summary = self.get_employee_summary()
        if not summary.empty:
            top_overtime = summary.loc[summary['Saldo_Stunden'].idxmax()]
            insights['top_overtime'] = {
                'employee': top_overtime['Mitarbeiter'],
                'hours': top_overtime['Saldo_Stunden']
            }
            
            # Mitarbeiter mit meisten Minusstunden
            bottom_overtime = summary.loc[summary['Saldo_Stunden'].idxmin()]
            insights['most_undertime'] = {
                'employee': bottom_overtime['Mitarbeiter'],
                'hours': bottom_overtime['Saldo_Stunden']
            }
        
        # Häufigste Arbeitscodes
        work_codes = self.df[self.df['work_code'].notna()]['work_code'].value_counts()
        if not work_codes.empty:
            insights['top_project'] = {
                'project': work_codes.index[0],
                'count': work_codes.iloc[0]
            }
        
        # Arbeitszeitmuster
        time_types = self.df['time_type'].value_counts()
        insights['time_patterns'] = time_types.to_dict()
        
        return insights

class StreamlitAnalyticsApp:
    """Hauptanwendung mit Analytics"""
    
    def __init__(self):
        self.analytics = None
    
    def run(self):
        """Startet die Analytics-Anwendung"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>📊 TimeMoto Analytics Dashboard</h1>
            <p>Professionelle Zeiterfassung mit erweiterten Auswertungen</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar Navigation
        st.sidebar.title("📈 Analytics Navigation")
        page = st.sidebar.selectbox(
            "Auswertung wählen:",
            [
                "🏠 Übersicht",
                "👥 Mitarbeiter-Analyse", 
                "📅 Datums-Analyse",
                "🏗️ Projekt-Analyse", 
                "🏥 Abwesenheits-Analyse",
                "⚡ Insights & KPIs",
                "📤 Daten laden"
            ]
        )
        
        # Daten laden falls nicht vorhanden
        if 'analytics_data' not in st.session_state:
            if page != "📤 Daten laden":
                st.warning("⚠️ Bitte laden Sie zuerst Daten über '📤 Daten laden'")
                self.show_data_upload()
                return
        
        # Seiten-Routing
        if page == "🏠 Übersicht":
            self.show_overview()
        elif page == "👥 Mitarbeiter-Analyse":
            self.show_employee_analysis()
        elif page == "📅 Datums-Analyse":
            self.show_date_analysis()
        elif page == "🏗️ Projekt-Analyse":
            self.show_project_analysis()
        elif page == "🏥 Abwesenheits-Analyse":
            self.show_absence_analysis()
        elif page == "⚡ Insights & KPIs":
            self.show_insights()
        elif page == "📤 Daten laden":
            self.show_data_upload()
    
    def show_data_upload(self):
        """Daten-Upload Seite"""
        st.header("📤 TimeMoto Daten laden")
        
        uploaded_file = st.file_uploader(
            "Excel-Datei hochladen",
            type=['xlsx', 'xls', 'csv'],
            help="Laden Sie Ihre TimeMoto Export-Datei hoch"
        )
        
        if uploaded_file is not None:
            try:
                # Datei laden
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Datei erfolgreich geladen: {len(df)} Zeilen")
                
                # Analytics-Objekt erstellen
                self.analytics = TimeMotoAnalytics(df)
                st.session_state.analytics_data = self.analytics
                
                # Schnelle Datenvorschau
                with st.expander("👀 Datenvorschau", expanded=True):
                    st.dataframe(df.head(), use_container_width=True)
                
                st.success("🎉 Daten erfolgreich verarbeitet! Sie können jetzt die Auswertungen verwenden.")
                
            except Exception as e:
                st.error(f"❌ Fehler beim Laden der Datei: {str(e)}")
    
    def show_overview(self):
        """Übersichts-Dashboard"""
        if 'analytics_data' not in st.session_state:
            return
        
        analytics = st.session_state.analytics_data
        
        st.header("🏠 Übersicht Dashboard")
        
        # KPI Metriken
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_employees = analytics.df['Username'].nunique()
            st.metric("👥 Mitarbeiter", total_employees)
        
        with col2:
            total_days = analytics.df['parsed_date'].nunique()
            st.metric("📅 Erfassungstage", total_days)
        
        with col3:
            total_hours = round(analytics.df['duration_minutes'].sum() / 60, 1)
            st.metric("⏰ Gesamtstunden", f"{total_hours:,.1f}")
        
        with col4:
            total_balance = round(analytics.df['balance_minutes'].sum() / 60, 1)
            delta_color = "normal" if total_balance >= 0 else "inverse"
            st.metric("⚖️ Gesamt-Saldo", f"{total_balance:+.1f}h", delta_color=delta_color)
        
        with col5:
            total_absences = analytics.df['AbsenceName'].notna().sum()
            st.metric("🏥 Abwesenheiten", total_absences)
        
        # Visualisierung
        col1, col2 = st.columns(2)
        
        with col1:
            # Arbeitszeit pro Tag
            daily_data = analytics.get_daily_analysis()
            if not daily_data.empty:
                fig = px.bar(
                    daily_data, 
                    x='Datum', 
                    y='Gesamt_Arbeitszeit_Std',
                    title="📊 Tägliche Gesamtarbeitszeit",
                    labels={'Gesamt_Arbeitszeit_Std': 'Stunden'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Saldo pro Tag
            if not daily_data.empty:
                fig = px.line(
                    daily_data, 
                    x='Datum', 
                    y='Gesamt_Saldo_Std',
                    title="📈 Täglicher Gesamt-Saldo",
                    labels={'Gesamt_Saldo_Std': 'Saldo (Stunden)'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Arbeitszeitmuster
        st.subheader("🔍 Arbeitszeitmuster")
        time_patterns = analytics.df['time_type'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=time_patterns.values,
                names=time_patterns.index,
                title="Verteilung der Arbeitszeitttypen"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top Mitarbeiter nach Arbeitszeit
            employee_summary = analytics.get_employee_summary()
            top_employees = employee_summary.nlargest(10, 'Arbeitszeit_Stunden')
            
            fig = px.bar(
                top_employees,
                x='Arbeitszeit_Stunden',
                y='Mitarbeiter',
                orientation='h',
                title="🏆 Top 10 Mitarbeiter (Arbeitszeit)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_employee_analysis(self):
        """Mitarbeiter-spezifische Analyse"""
        if 'analytics_data' not in st.session_state:
            return
        
        analytics = st.session_state.analytics_data
        
        st.header("👥 Mitarbeiter-Analyse")
        
        # Filter
        employees = ['Alle'] + list(analytics.df['Username'].unique())
        selected_employee = st.selectbox("👤 Mitarbeiter auswählen:", employees)
        
        if selected_employee == 'Alle':
            # Alle Mitarbeiter Übersicht
            summary = analytics.get_employee_summary()
            
            st.subheader("📊 Mitarbeiter-Übersicht")
            st.dataframe(
                summary[['Mitarbeiter', 'Arbeitstage', 'Arbeitszeit_Stunden', 'Saldo_Stunden', 'Abwesenheiten', 'Projekte']],
                use_container_width=True
            )
            
            # Visualisierungen
            col1, col2 = st.columns(2)
            
            with col1:
                # Saldo-Vergleich
                fig = px.bar(
                    summary, 
                    x='Mitarbeiter', 
                    y='Saldo_Stunden',
                    title="⚖️ Saldo-Vergleich alle Mitarbeiter",
                    color='Saldo_Stunden',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Arbeitszeit vs Saldo
                fig = px.scatter(
                    summary,
                    x='Arbeitszeit_Stunden',
                    y='Saldo_Stunden',
                    size='Arbeitstage',
                    hover_name='Mitarbeiter',
                    title="💼 Arbeitszeit vs. Saldo",
                    labels={
                        'Arbeitszeit_Stunden': 'Arbeitszeit (Stunden)',
                        'Saldo_Stunden': 'Saldo (Stunden)'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Einzelner Mitarbeiter
            employee_data = analytics.df[analytics.df['Username'] == selected_employee]
            
            # Metriken für den Mitarbeiter
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                work_days = len(employee_data)
                st.metric("📅 Arbeitstage", work_days)
            
            with col2:
                total_hours = round(employee_data['duration_minutes'].sum() / 60, 1)
                st.metric("⏰ Arbeitszeit", f"{total_hours}h")
            
            with col3:
                balance = round(employee_data['balance_minutes'].sum() / 60, 1)
                st.metric("⚖️ Saldo", f"{balance:+.1f}h")
            
            with col4:
                absences = employee_data['AbsenceName'].notna().sum()
                st.metric("🏥 Abwesenheiten", absences)
            
            # Detailanalyse
            col1, col2 = st.columns(2)
            
            with col1:
                # Täglicher Saldo
                daily_balance = employee_data.groupby('parsed_date')['balance_minutes'].sum() / 60
                
                fig = px.bar(
                    x=daily_balance.index,
                    y=daily_balance.values,
                    title=f"📊 Täglicher Saldo - {selected_employee}",
                    labels={'x': 'Datum', 'y': 'Saldo (Stunden)'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Arbeitszeitmuster
                time_patterns = employee_data['time_type'].value_counts()
                
                fig = px.pie(
                    values=time_patterns.values,
                    names=time_patterns.index,
                    title=f"🔍 Arbeitszeitmuster - {selected_employee}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailtabelle
            st.subheader("📋 Detaildaten")
            display_data = employee_data[['parsed_date', 'StartTime', 'EndTime', 'DurationExcludingBreaks', 'Balance', 'work_code', 'AbsenceName', 'time_type']].copy()
            display_data.columns = ['Datum', 'Start', 'Ende', 'Arbeitszeit', 'Saldo', 'Projekt', 'Abwesenheit', 'Typ']
            st.dataframe(display_data, use_container_width=True)
    
    def show_date_analysis(self):
        """Datums-basierte Analyse"""
        if 'analytics_data' not in st.session_state:
            return
        
        analytics = st.session_state.analytics_data
        
        st.header("📅 Datums-Analyse")
        
        daily_data = analytics.get_daily_analysis()
        
        if daily_data.empty:
            st.warning("Keine Daten für Datums-Analyse verfügbar.")
            return
        
        # Datumsbereich
        st.info(f"📊 Analyse für Zeitraum: {daily_data['Datum'].min()} bis {daily_data['Datum'].max()}")
        
        # Tagesvergleich
        col1, col2 = st.columns(2)
        
        with col1:
            # Mitarbeiteranzahl pro Tag
            fig = px.bar(
                daily_data,
                x='Datum',
                y='Mitarbeiter_Anzahl',
                title="👥 Anwesende Mitarbeiter pro Tag"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Abwesenheiten pro Tag
            fig = px.bar(
                daily_data,
                x='Datum',
                y='Abwesenheiten',
                title="🏥 Abwesenheiten pro Tag",
                color='Abwesenheiten',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Wochentagsanalyse
        analytics.df['weekday_num'] = pd.to_datetime(analytics.df['parsed_date']).dt.dayofweek
        weekday_data = analytics.df.groupby(['weekday', 'weekday_num']).agg({
            'Username': 'nunique',
            'balance_minutes': 'sum',
            'duration_minutes': 'sum'
        }).reset_index().sort_values('weekday_num')
        
        weekday_data['balance_hours'] = weekday_data['balance_minutes'] / 60
        weekday_data['duration_hours'] = weekday_data['duration_minutes'] / 60
        
        st.subheader("📊 Wochentagsanalyse")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                weekday_data,
                x='weekday',
                y='duration_hours',
                title="⏰ Durchschnittliche Arbeitszeit pro Wochentag"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                weekday_data,
                x='weekday',
                y='balance_hours',
                title="⚖️ Durchschnittlicher Saldo pro Wochentag",
                color='balance_hours',
                color_continuous_scale='RdYlGn'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailtabelle
        st.subheader("📋 Tägliche Übersicht")
        st.dataframe(
            daily_data[['Datum', 'Mitarbeiter_Anzahl', 'Gesamt_Arbeitszeit_Std', 'Gesamt_Saldo_Std', 'Abwesenheiten', 'Normale_Erfassungen']],
            use_container_width=True
        )
    
    def show_project_analysis(self):
        """Projekt-basierte Analyse"""
        if 'analytics_data' not in st.session_state:
            return
        
        analytics = st.session_state.analytics_data
        
        st.header("🏗️ Projekt-Analyse")
        
        work_analysis = analytics.get_work_code_analysis()
        
        if work_analysis.empty:
            st.warning("⚠️ Keine Projekt-/Arbeitscodedaten in den Remarks gefunden.")
            st.info("💡 Stellen Sie sicher, dass Ihre Daten Arbeitscodes in den Bemerkungen enthalten.")
            return
        
        # Projekt-Übersicht
        project_summary = work_analysis.groupby('Projekt').agg({
            'Mitarbeiter': 'nunique',
            'Tage': 'sum',
            'Arbeitszeit_Stunden': 'sum'
        }).reset_index()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_projects = len(project_summary)
            st.metric("🏗️ Projekte", total_projects)
        
        with col2:
            total_project_hours = round(project_summary['Arbeitszeit_Stunden'].sum(), 1)
            st.metric("⏰ Projektzeit", f"{total_project_hours}h")
        
        with col3:
            avg_project_size = round(project_summary['Arbeitszeit_Stunden'].mean(), 1)
            st.metric("📊 Ø Projektgröße", f"{avg_project_size}h")
        
        # Visualisierungen
        col1, col2 = st.columns(2)
        
        with col1:
            # Projektverteilung nach Zeit
            fig = px.pie(
                project_summary,
                values='Arbeitszeit_Stunden',
                names='Projekt',
                title="⏰ Zeitverteilung nach Projekten"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Mitarbeiter pro Projekt
            fig = px.bar(
                project_summary,
                x='Projekt',
                y='Mitarbeiter',
                title="👥 Mitarbeiter pro Projekt"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailanalyse
        st.subheader("📊 Projekt-Details")
        
        selected_project = st.selectbox(
            "🏗️ Projekt auswählen:",
            ['Alle'] + list(project_summary['Projekt'].unique())
        )
        
        if selected_project == 'Alle':
            st.dataframe(
                project_summary[['Projekt', 'Mitarbeiter', 'Tage', 'Arbeitszeit_Stunden']],
                use_container_width=True
            )
        else:
            project_details = work_analysis[work_analysis['Projekt'] == selected_project]
            
            # Mitarbeiterverteilung im Projekt
            fig = px.bar(
                project_details,
                x='Mitarbeiter',
                y='Arbeitszeit_Stunden',
                title=f"👥 Mitarbeiterzeiten - {selected_project}"
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                project_details[['Mitarbeiter', 'Tage', 'Arbeitszeit_Stunden']],
                use_container_width=True
            )
    
    def show_absence_analysis(self):
        """Abwesenheits-Analyse"""
        if 'analytics_data' not in st.session_state:
            return
        
        analytics = st.session_state.analytics_data
        
        st.header("🏥 Abwesenheits-Analyse")
        
        absence_data = analytics.get_absence_analysis()
        
        if absence_data.empty:
            st.info("ℹ️ Keine Abwesenheiten in den Daten gefunden.")
            return
        
        # Abwesenheits-Übersicht
        absence_summary = absence_data.groupby('Abwesenheitstyp').agg({
            'Mitarbeiter': 'nunique',
            'Tage': 'sum'
        }).reset_index()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_absence_days = absence_summary['Tage'].sum()
            st.metric("📅 Abwesenheitstage", total_absence_days)
        
        with col2:
            affected_employees = absence_data['Mitarbeiter'].nunique()
            st.metric("👥 Betroffene Mitarbeiter", affected_employees)
        
        with col3:
            absence_types = len(absence_summary)
            st.metric("🏥 Abwesenheitstypen", absence_types)
        
        # Visualisierungen
        col1, col2 = st.columns(2)
        
        with col1:
            # Abwesenheitstypen
            fig = px.pie(
                absence_summary,
                values='Tage',
                names='Abwesenheitstyp',
                title="📊 Verteilung der Abwesenheitstypen"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Mitarbeiter mit meisten Abwesenheiten
            employee_absences = absence_data.groupby('Mitarbeiter')['Tage'].sum().reset_index()
            top_absences = employee_absences.nlargest(10, 'Tage')
            
            fig = px.bar(
                top_absences,
                x='Tage',
                y='Mitarbeiter',
                orientation='h',
                title="👥 Mitarbeiter mit meisten Abwesenheitstagen"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailanalyse
        st.subheader("📋 Abwesenheits-Details")
        st.dataframe(absence_data, use_container_width=True)
    
    def show_insights(self):
        """KPIs und automatische Insights"""
        if 'analytics_data' not in st.session_state:
            return
        
        analytics = st.session_state.analytics_data
        insights = analytics.generate_insights()
        
        st.header("⚡ Insights & KPIs")
        
        # Automatische Insights
        st.subheader("🎯 Automatische Insights")
        
        if 'top_overtime' in insights:
            st.markdown(f"""
            <div class="insight-box">
                <h4>🏆 Mitarbeiter mit meisten Überstunden</h4>
                <p><strong>{insights['top_overtime']['employee']}</strong> hat <strong>{insights['top_overtime']['hours']:+.1f} Stunden</strong> Überstunden</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'most_undertime' in insights:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ Mitarbeiter mit meisten Minusstunden</h4>
                <p><strong>{insights['most_undertime']['employee']}</strong> hat <strong>{insights['most_undertime']['hours']:+.1f} Stunden</strong> Minusstunden</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'top_project' in insights:
            st.markdown(f"""
            <div class="insight-box">
                <h4>🏗️ Häufigstes Projekt</h4>
                <p><strong>{insights['top_project']['project']}</strong> wurde <strong>{insights['top_project']['count']} mal</strong> erfasst</p>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI Dashboard
        st.subheader("📊 Key Performance Indicators")
        
        # Berechne KPIs
        total_employees = analytics.df['Username'].nunique()
        total_workdays = analytics.df.groupby('Username')['parsed_date'].nunique().mean()
        avg_daily_hours = analytics.df.groupby(['Username', 'parsed_date'])['duration_minutes'].sum().mean() / 60
        overtime_ratio = (analytics.df['balance_minutes'] > 0).mean() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Ø Arbeitstage/MA", f"{total_workdays:.1f}")
        
        with col2:
            st.metric("⏰ Ø Tägliche Arbeitszeit", f"{avg_daily_hours:.1f}h")
        
        with col3:
            st.metric("📈 Überstunden-Quote", f"{overtime_ratio:.1f}%")
        
        with col4:
            absence_rate = (analytics.df['AbsenceName'].notna().sum() / len(analytics.df)) * 100
            st.metric("🏥 Abwesenheits-Quote", f"{absence_rate:.1f}%")
        
        # Trend-Analyse
        st.subheader("📈 Trend-Analyse")
        
        daily_trends = analytics.get_daily_analysis()
        if len(daily_trends) > 1:
            # Berechne Trends
            recent_avg = daily_trends.tail(2)['Gesamt_Arbeitszeit_Std'].mean()
            overall_avg = daily_trends['Gesamt_Arbeitszeit_Std'].mean()
            trend = ((recent_avg - overall_avg) / overall_avg) * 100
            
            trend_color = "normal" if trend >= 0 else "inverse"
            st.metric(
                "📊 Arbeitszeit-Trend (letzte vs. Durchschnitt)",
                f"{trend:+.1f}%",
                delta_color=trend_color
            )
        
        # Export aller Insights
        if st.button("📥 Insights-Report exportieren"):
            report_data = {
                'Zeitraum': f"{analytics.df['parsed_date'].min()} bis {analytics.df['parsed_date'].max()}",
                'Mitarbeiter_Gesamt': total_employees,
                'Durchschnittliche_Arbeitstage': round(total_workdays, 1),
                'Durchschnittliche_Tagesarbeitszeit': round(avg_daily_hours, 1),
                'Überstunden_Quote_Prozent': round(overtime_ratio, 1),
                'Abwesenheits_Quote_Prozent': round(absence_rate, 1),
                **insights
            }
            
            import json
            report_json = json.dumps(report_data, indent=2, ensure_ascii=False, default=str)
            
            st.download_button(
                label="💾 JSON-Report herunterladen",
                data=report_json,
                file_name=f"timemoto_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Anwendung starten
if __name__ == "__main__":
    app = StreamlitAnalyticsApp()
    app.run()
