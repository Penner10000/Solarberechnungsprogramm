import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pvlib import location, irradiance, pvsystem, temperature, inverter, atmosphere
import uuid
import json
import os

st.set_page_config(
    page_title="PV Ertrags-Simulator",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "plants" not in st.session_state:
    st.session_state.plants = []
if "edit_id" not in st.session_state:
    st.session_state.edit_id = None
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None
if "pending_clear_all" not in st.session_state:
    st.session_state.pending_clear_all = False

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pv_plants.json")
DEFAULT_LAT = 51.16
DEFAULT_LON = 10.45


def load_plants():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            try:
                st.session_state.plants = json.load(f)
            except (json.JSONDecodeError, TypeError):
                st.session_state.plants = []


def save_plants():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.plants, f, indent=2, ensure_ascii=False)


def reset_form():
    st.session_state.edit_id = None


if not st.session_state.loaded:
    if not st.session_state.plants:
        load_plants()
    st.session_state.loaded = True


@st.cache_data(show_spinner=False)
def simulate_plant(plant_identifier, peak_power_wp, azimuth, tilt, latitude, longitude, inverter_limit_w, sim_year, system_loss=0.0, temp_coeff=0.0, albedo=0.2, transposition_model="perez", eta_inv_nom=0.96):
    loc = location.Location(latitude, longitude, tz="UTC")

    start = f"{sim_year}-01-01 00:00:00"
    end = f"{sim_year}-12-31 23:00:00"
    times = pd.date_range(start, end, freq="1h", tz="UTC")

    clearsky = loc.get_clearsky(times)
    solpos = loc.get_solarposition(times)

    airmass = atmosphere.get_relative_airmass(solpos["apparent_zenith"])
    dni_extra = irradiance.get_extra_radiation(times)

    poa = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=albedo,
        model=transposition_model,
    )

    poa_global = poa["poa_global"].clip(lower=0)

    day_of_year = times.dayofyear
    temp_air = 10 + 10 * np.cos(2 * np.pi * (day_of_year - 182) / 365)
    wind_speed = np.full_like(temp_air, 1.0)

    temp_params = temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
    t_cell = temperature.sapm_cell(
        poa_global, temp_air, wind_speed,
        temp_params["a"], temp_params["b"], temp_params["deltaT"],
    )

    p_dc = pvsystem.pvwatts_dc(poa_global, t_cell, peak_power_wp, gamma_pdc=temp_coeff)
    p_ac_raw = inverter.pvwatts(p_dc, peak_power_wp, eta_inv_nom=eta_inv_nom)
    p_ac_unclipped = p_ac_raw * (1 - system_loss)

    if inverter_limit_w is not None and inverter_limit_w > 0:
        p_ac = p_ac_unclipped.clip(upper=inverter_limit_w)
        clipped_wh = (p_ac_unclipped - p_ac).clip(lower=0).sum()
    else:
        p_ac = p_ac_unclipped
        clipped_wh = 0.0

    df = pd.DataFrame({
        "datetime": times,
        "poa_global": poa_global.values,
        "temp_cell": t_cell.values,
        "p_dc": p_dc.values,
        "p_ac": p_ac.values,
        "energy_wh": p_ac.values,
    })
    df.set_index("datetime", inplace=True)

    iso_week = df.index.isocalendar()
    df["week"] = iso_week.week.astype(int)
    df["year_iso"] = iso_week.year.astype(int)

    yearly_total_wh = df["energy_wh"].sum()

    daily = df.resample("D")["energy_wh"].sum()
    daily_weeks = daily.groupby(daily.index.isocalendar().week)
    daily_mean = daily_weeks.mean()
    weekly = daily_mean * 7

    weekly = weekly.reindex(range(1, int(df["week"].max()) + 1), fill_value=0.0)
    weekly.index.name = None

    return weekly, df, yearly_total_wh, clipped_wh


# ===================== SIDEBAR =====================

st.sidebar.title("☀️ PV Simulator")

st.sidebar.divider()
st.sidebar.subheader("Simulationsparameter")

with st.sidebar.expander("Umgebung & Modell", expanded=True):
    sim_year = st.number_input(
        "Simulationsjahr",
        value=2026, min_value=2000, max_value=2100, step=1,
        help="Beeinflusst ISO-Wochen-Layout (52 vs. 53 Wochen).",
        key="sim_year",
    )
    transposition_model = st.selectbox(
        "Transpositionsmodell",
        options=["perez", "haydavies", "isotropic"],
        index=0,
        help="Perez = genaueste Diffusstrahl-Berechnung (empfohlen). Haydavies = vereinfacht. Isotrop = stark vereinfacht.",
        key="trans_model",
    )
    albedo_val = st.number_input(
        "Albedo (Bodenreflexion)",
        value=0.20, min_value=0.0, max_value=1.0, step=0.05,
        help="0,20 = Gras/Asphalt. Höher bei Schnee (0,7) oder hellem Sand (0,4).",
        key="albedo",
    )

with st.sidebar.expander("Elektrische Verluste", expanded=True):
    st.caption("Hinweis: 0 % = optimaler Idealfall ohne Verlust.")
    sys_loss_pct = st.number_input(
        "System-Verlustfaktor (%)",
        value=0.0, min_value=0.0, max_value=50.0, step=0.5,
        help="Pauschalabzug für Kabel, Verschmutzung, Mismatch. Wird NACH AC-Wandlung angewandt.",
        key="sys_loss",
    )
    eta_inv = st.number_input(
        "Wechselrichter-Wirkungsgrad (%)",
        value=96.0, min_value=50.0, max_value=100.0, step=0.5,
        help="PVWatts-Wechselrichtermodell inkl. Schwachlichtverhalten. 96 % = realistisch, 100 % = ideal.",
        key="eta_inv",
    ) / 100.0
    temp_coeff_val = st.number_input(
        "Temperaturkoeffizient (1/K)",
        value=0.0, min_value=-0.0100, max_value=0.0000, step=0.0005, format="%.4f",
        help="Leistungsabfall pro °C über 25 °C. 0 = ideal. Realistisch: -0,004/K für kristalline Module.",
        key="temp_coeff",
    )

sys_loss = sys_loss_pct / 100.0

st.sidebar.divider()
st.sidebar.subheader("Neue Anlage")

edit_plant = None
if st.session_state.edit_id:
    for p in st.session_state.plants:
        if p["id"] == st.session_state.edit_id:
            edit_plant = p
            break

form_title = "Anlage bearbeiten" if edit_plant else "Anlage hinzuf\u00fcgen"

with st.sidebar.form("plant_form", clear_on_submit=False):
    st.caption(form_title)

    name = st.text_input(
        "Name",
        value=edit_plant["name"] if edit_plant else "",
        placeholder="z.B. S\u00fcddach 10kWp",
    )
    wp_val = st.number_input(
        "Nennleistung (Wp)",
        value=edit_plant["peak_power_wp"] if edit_plant else 10000,
        min_value=100, max_value=1_000_000, step=100,
    )
    col_a, col_t = st.columns(2)
    with col_a:
        azimuth = st.number_input(
            "Azimut (\u00b0)",
            value=edit_plant["azimuth"] if edit_plant else 180,
            min_value=0, max_value=360, step=1,
            help="0\u00b0=Nord, 90\u00b0=Ost, 180\u00b0=S\u00fcd, 270\u00b0=West",
        )
    with col_t:
        tilt_val = st.number_input(
            "Neigung (\u00b0)",
            value=edit_plant["tilt"] if edit_plant else 35,
            min_value=0, max_value=90, step=1,
        )
    lat = st.number_input(
        "Breitengrad",
        value=edit_plant["latitude"] if edit_plant else DEFAULT_LAT,
        min_value=-90.0, max_value=90.0, step=0.01,
        key="plant_lat",
    )
    lon = st.number_input(
        "L\u00e4ngengrad",
        value=edit_plant["longitude"] if edit_plant else DEFAULT_LON,
        min_value=-180.0, max_value=180.0, step=0.01,
        key="plant_lon",
    )
    inv_limit = st.number_input(
        "Wechselrichter-Limit (W) \u2014 0 = kein Limit",
        value=edit_plant["inverter_limit_w"] if edit_plant else 0,
        min_value=0, max_value=1_000_000, step=100,
    )
    color_default = edit_plant.get("color", "#1f77b4") if edit_plant else "#1f77b4"
    color = st.color_picker("Farbe im Diagramm", value=color_default)

    submitted = st.form_submit_button(
        "Speichern" if not edit_plant else "\u00c4nderungen speichern",
        type="primary", width="stretch",
    )

    if submitted:
        if not name.strip():
            st.error("Bitte einen Namen vergeben.")
        else:
            plant_data = {
                "id": edit_plant["id"] if edit_plant else str(uuid.uuid4()),
                "name": name.strip(),
                "peak_power_wp": wp_val,
                "azimuth": azimuth,
                "tilt": tilt_val,
                "latitude": lat,
                "longitude": lon,
                "inverter_limit_w": inv_limit if inv_limit > 0 else None,
                "color": color,
            }
            if edit_plant:
                for i, p in enumerate(st.session_state.plants):
                    if p["id"] == edit_plant["id"]:
                        st.session_state.plants[i] = plant_data
                        break
            else:
                st.session_state.plants.append(plant_data)
            save_plants()
            reset_form()
            st.rerun()

    if edit_plant:
        if st.form_submit_button("Abbrechen", width="stretch", type="secondary"):
            reset_form()
            st.rerun()

st.sidebar.divider()
st.sidebar.subheader(f"Gespeicherte Anlagen ({len(st.session_state.plants)})")

if not st.session_state.plants:
    st.sidebar.info("Noch keine Anlagen gespeichert.")
else:
    plants_to_duplicate = None
    for plant in st.session_state.plants:
        with st.sidebar.container():
            cols = st.columns([5, 1, 1, 1])
            limit_str = (
                f" | Limit: {plant['inverter_limit_w']}W"
                if plant.get("inverter_limit_w") else ""
            )
            cols[0].caption(
                f"**{plant['name']}**  \n"
                f"{plant['peak_power_wp']:,} Wp | "
                f"Az:{plant['azimuth']}\u00b0 | Tilt:{plant['tilt']}\u00b0"
                f"{limit_str}"
            )
            if cols[1].button("\u270f\ufe0f", key=f"edit_{plant['id']}", help="Bearbeiten"):
                st.session_state.edit_id = plant["id"]
                st.session_state.pending_delete = None
                st.session_state.pending_clear_all = False
                st.rerun()
            if cols[2].button("\U0001f4cb", key=f"dup_{plant['id']}", help="Duplizieren"):
                plants_to_duplicate = plant
            if cols[3].button("\U0001f5d1\ufe0f", key=f"del_{plant['id']}", help="L\u00f6schen"):
                st.session_state.pending_delete = plant["id"]
                plants_to_duplicate = None
                st.session_state.pending_clear_all = False
                st.rerun()

    if st.session_state.pending_delete:
        st.sidebar.warning("Anlage wirklich l\u00f6schen?")
        col_yd, col_nd = st.sidebar.columns(2)
        if col_yd.button("Ja, l\u00f6schen", key="confirm_del", type="primary", width="stretch"):
            st.session_state.plants = [
                p for p in st.session_state.plants if p["id"] != st.session_state.pending_delete
            ]
            st.session_state.pending_delete = None
            save_plants()
            st.rerun()
        if col_nd.button("Abbrechen", key="cancel_del", width="stretch"):
            st.session_state.pending_delete = None
            st.rerun()

    if plants_to_duplicate:
        new_plant = dict(plants_to_duplicate)
        new_plant["id"] = str(uuid.uuid4())
        new_plant["name"] = new_plant["name"] + " (Kopie)"
        st.session_state.plants.append(new_plant)
        save_plants()
        st.rerun()

if st.session_state.plants:
    st.sidebar.divider()
    col_load, col_clear = st.sidebar.columns(2)

    if col_load.button("\U0001f4c2 Laden", width="stretch"):
        load_plants()
        st.session_state.pending_clear_all = False
        st.rerun()

    if col_clear.button("\U0001f5d1\ufe0f Alle l\u00f6schen", width="stretch"):
        st.session_state.pending_clear_all = True
        st.rerun()

    if st.session_state.pending_clear_all:
        st.sidebar.warning("Wirklich ALLE Anlagen l\u00f6schen?")
        col_ya, col_na = st.sidebar.columns(2)
        if col_ya.button("Ja, alle l\u00f6schen", key="confirm_clear", type="primary", width="stretch"):
            st.session_state.plants = []
            st.session_state.pending_clear_all = False
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            st.rerun()
        if col_na.button("Abbrechen", key="cancel_clear", width="stretch"):
            st.session_state.pending_clear_all = False
            st.rerun()


# ===================== MAIN AREA =====================

st.title("\u2600\ufe0f W\u00f6chentlicher PV-Ertragsvergleich")

if not st.session_state.plants:
    st.info("F\u00fcge links in der Sidebar eine Photovoltaik-Anlage hinzu, um die Simulation zu starten.")
    st.stop()

st.divider()

st.markdown("### Anlagen f\u00fcr Vergleich ausw\u00e4hlen")

plant_lookup = {}
for p in st.session_state.plants:
    key = f"{p['name']} [{p['peak_power_wp']}Wp]"
    idx = 1
    orig_key = key
    while key in plant_lookup:
        key = f"{orig_key} (#{idx})"
        idx += 1
    plant_lookup[key] = p

selected_names = st.multiselect(
    "Welche Anlagen sollen verglichen werden?",
    options=list(plant_lookup.keys()),
    default=list(plant_lookup.keys()),
)

selected_plants = [plant_lookup[name] for name in selected_names]

if not selected_plants:
    st.warning("Bitte mindestens eine Anlage ausw\u00e4hlen.")
    st.stop()

st.divider()

with st.spinner("Berechne PV-Ertr\u00e4ge ..."):
    results = {}
    detail_dfs = {}
    yearly_totals = {}
    clipping_losses = {}
    sim_errors = []
    for plant in selected_plants:
        try:
            weekly, detail, yearly, clipped = simulate_plant(
                plant["id"],
                plant["peak_power_wp"],
                plant["azimuth"],
                plant["tilt"],
                plant["latitude"],
                plant["longitude"],
                plant.get("inverter_limit_w"),
                sim_year,
                sys_loss,
                temp_coeff_val,
                albedo_val,
                transposition_model,
                eta_inv,
            )
            results[plant["id"]] = weekly
            detail_dfs[plant["id"]] = detail
            yearly_totals[plant["id"]] = yearly
            clipping_losses[plant["id"]] = clipped
        except Exception as e:
            sim_errors.append((plant["name"], str(e)))
            continue

    if sim_errors:
        for plant_name, err_msg in sim_errors:
            st.error(f"Fehler bei Anlage '{plant_name}': {err_msg}. Bitte Koordinaten pr\u00fcfen.")

if not results:
    st.error("Keine Anlage konnte simuliert werden.")
    st.stop()

weekly_df = pd.DataFrame(results)
weekly_df.index.name = "KW"

id_to_name = {p["id"]: p["name"] for p in selected_plants if p["id"] in results}
id_to_plant = {p["id"]: p for p in selected_plants if p["id"] in results}
weekly_df = weekly_df.rename(columns=id_to_name)

st.markdown("### Diagramm-Einstellungen")
col_ctl1, col_ctl2, col_ctl3 = st.columns(3)

with col_ctl1:
    unit_mode = st.toggle(
        "Spezifischer Ertrag (kWh/kWp)",
        value=False,
        help="Wenn aktiviert, wird der Ertrag pro kWp statt absolut angezeigt.",
    )
with col_ctl2:
    chart_type = st.radio(
        "Diagramm-Typ",
        options=["Linie", "Balken gruppiert", "Balken gestapelt"],
        horizontal=True,
    )
with col_ctl3:
    display_weeks = st.selectbox(
        "Angezeigte Wochen",
        options=["Alle (1\u201352)", "Sommer (14\u201339)", "Winter (40\u201313)"],
    )

display_df = weekly_df.copy()

if unit_mode:
    for plant in selected_plants:
        wp = plant["peak_power_wp"]
        display_df[plant["name"]] = display_df[plant["name"]] / 1000 / (wp / 1000)
    unit_label = "kWh/kWp"
else:
    display_df = display_df / 1000
    unit_label = "kWh"

if display_weeks == "Sommer (14\u201339)":
    display_df = display_df.loc[14:39]
elif display_weeks == "Winter (40\u201313)":
    max_week = display_df.index.max()
    weeks_40_end = list(range(40, max_week + 1))
    weeks_1_13 = list(range(1, 14))
    winter_weeks = weeks_40_end + weeks_1_13
    display_df = display_df.reindex([w for w in winter_weeks if w in display_df.index])
    offset = max_week + 1
    display_df.index = pd.Index(
        [w + offset if w <= 13 else w for w in display_df.index],
        name="KW",
    )

st.markdown("### Ertragsverlauf")
fig = go.Figure()

for plant in selected_plants:
    name = plant["name"]
    c = plant.get("color")
    if name not in display_df.columns:
        continue

    if chart_type == "Linie":
        fig.add_trace(go.Scatter(
            x=display_df.index, y=display_df[name],
            mode="lines+markers", name=name,
            line=dict(color=c) if c else None,
            marker=dict(color=c) if c else None,
            hovertemplate=f"{name}<br>KW %{{x}}: %{{y:.1f}} {unit_label}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Bar(
            x=display_df.index, y=display_df[name],
            name=name, marker_color=c,
            hovertemplate=f"{name}<br>KW %{{x}}: %{{y:.1f}} {unit_label}<extra></extra>",
        ))

if chart_type == "Linie":
    barmode = None
elif chart_type == "Balken gestapelt":
    barmode = "stack"
else:
    barmode = "group"

fig.update_layout(
    barmode=barmode,
    xaxis_title="Kalenderwoche",
    yaxis_title=f"Ertrag ({unit_label})",
    hovermode="x unified",
    height=500,
    margin=dict(l=20, r=20, t=10, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

if display_weeks == "Winter (40\u201313)":
    tick_vals = list(display_df.index)
    tick_text = [str(v - offset if v >= offset else v) for v in tick_vals]
    fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text)

st.plotly_chart(fig, width="stretch", key="main_chart")

# --- Comparison table ---
st.markdown("### Vergleichstabelle")
kwh_unit = "kWh/kWp" if unit_mode else "kWh"

table_data = []
for plant in selected_plants:
    pid = plant["id"]
    if pid not in results:
        continue
    name = plant["name"]
    raw_weekly = results[pid]
    actual_yearly_wh = yearly_totals[pid]
    clipped_wh = clipping_losses.get(pid, 0)

    if unit_mode:
        wp = plant["peak_power_wp"]
        yearly = actual_yearly_wh / 1000 / (wp / 1000)
        peak_week = raw_weekly.max() / 1000 / (wp / 1000)
        avg_week = raw_weekly.mean() / 1000 / (wp / 1000)
        clipped_display = clipped_wh / 1000 / (wp / 1000)
    else:
        yearly = actual_yearly_wh / 1000
        peak_week = raw_weekly.max() / 1000
        avg_week = raw_weekly.mean() / 1000
        clipped_display = clipped_wh / 1000

    row = {
        "Anlage": name,
        f"Jahresertrag [{kwh_unit}]": round(yearly, 1),
        f"\u00d8 Wochenertrag [{kwh_unit}]": round(avg_week, 1),
        f"Max. KW-Ertrag [{kwh_unit}]": round(peak_week, 1),
        "Nennleistung [kWp]": plant["peak_power_wp"] / 1000,
        "Azimut [\u00b0]": plant["azimuth"],
        "Neigung [\u00b0]": plant["tilt"],
        "WR-Limit [W]": plant.get("inverter_limit_w") or "\u2014",
    }
    if plant.get("inverter_limit_w"):
        row[f"Clipping-Verlust [{kwh_unit}]"] = round(clipped_display, 1)
        row["Clipping (%)"] = round(
            (clipped_wh / (actual_yearly_wh + clipped_wh)) * 100, 1
        ) if (actual_yearly_wh + clipped_wh) > 0 else 0.0
    table_data.append(row)

table_df = pd.DataFrame(table_data)
st.dataframe(
    table_df, width="stretch", hide_index=True,
    column_config={
        f"Jahresertrag [{kwh_unit}]": st.column_config.NumberColumn(format="%.1f"),
        f"\u00d8 Wochenertrag [{kwh_unit}]": st.column_config.NumberColumn(format="%.1f"),
        f"Max. KW-Ertrag [{kwh_unit}]": st.column_config.NumberColumn(format="%.1f"),
        f"Clipping-Verlust [{kwh_unit}]": st.column_config.NumberColumn(format="%.1f"),
        "Clipping (%)": st.column_config.NumberColumn(format="%.1f"),
        "Nennleistung [kWp]": st.column_config.NumberColumn(format="%.2f"),
        "WR-Limit [W]": st.column_config.TextColumn(),
    },
)

# --- Wirtschaftlichkeit ---
st.markdown("### Wirtschaftlichkeit (Sch\u00e4tzung)")
econ_col1, econ_col2, econ_col3 = st.columns(3)
with econ_col1:
    price_per_kwh = st.number_input(
        "Strompreis (\u20ac/kWh)", value=0.35, min_value=0.0, max_value=2.0, step=0.01,
    )
with econ_col2:
    lifespan = st.number_input(
        "Betrachtungszeitraum (Jahre)", value=20, min_value=1, max_value=40, step=1,
    )
with econ_col3:
    degradation_pa = st.number_input(
        "Degradation (% pro Jahr)", value=0.5, min_value=0.0, max_value=5.0, step=0.1,
        help="J\u00e4hrlicher Leistungsverlust der Module. \u00dcblich: 0,5 %/a.",
    ) / 100.0

econ_data = []
for plant in selected_plants:
    pid = plant["id"]
    if pid not in yearly_totals:
        continue
    name = plant["name"]
    yearly_kwh = yearly_totals[pid] / 1000
    annual_savings_base = yearly_kwh * price_per_kwh
    lifetime_savings = 0.0
    for year_idx in range(lifespan):
        lifetime_savings += annual_savings_base * ((1 - degradation_pa) ** year_idx)
    econ_data.append({
        "Anlage": name,
        "Jahresertrag [kWh]": round(yearly_kwh, 0),
        "J\u00e4hrliche Ersparnis (1. Jahr) [\u20ac]": round(annual_savings_base, 2),
        f"Ersparnis \u00fcber {lifespan} J. (inkl. Degradation) [\u20ac]": round(lifetime_savings, 2),
    })

econ_df = pd.DataFrame(econ_data)
st.dataframe(
    econ_df, width="stretch", hide_index=True,
    column_config={
        "Jahresertrag [kWh]": st.column_config.NumberColumn(format="%.0f"),
        "J\u00e4hrliche Ersparnis (1. Jahr) [\u20ac]": st.column_config.NumberColumn(format="%.2f"),
        f"Ersparnis \u00fcber {lifespan} J. (inkl. Degradation) [\u20ac]": st.column_config.NumberColumn(format="%.2f"),
    },
)

# --- Raw data ---
st.divider()
with st.expander("\U0001f4ca W\u00f6chentliche Rohdaten anzeigen", expanded=False):
    display_raw = weekly_df / 1000
    display_raw.index.name = "KW"
    st.dataframe(
        display_raw.style.format("{:.1f}"),
        width="stretch",
        column_config={
            col: st.column_config.NumberColumn(f"{col} [kWh]", format="%.1f")
            for col in display_raw.columns
        },
    )
    csv = display_raw.to_csv(float_format="%.1f")
    st.download_button(
        label="\U0001f4e5 Daten als CSV herunterladen",
        data=csv,
        file_name="pv_weekly_yield.csv",
        mime="text/csv",
    )
