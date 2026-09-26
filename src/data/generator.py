"""
Physics-Informed Smart-Meter Interval Data and Tamper Attack Simulator
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict
from config.settings import SIMULATION_CONFIG, TariffAndCostConfig, TARIFF_CONFIG
from config.logging_config import setup_logger

logger = setup_logger("SmartMeterSimulator")

class SmartMeterSimulator:
    def __init__(self, config=SIMULATION_CONFIG, tariff_config=TARIFF_CONFIG, seed: int = 42):
        self.config = config
        self.tariff_config = tariff_config
        self.seed = seed
        np.random.seed(seed)
        
    def generate_weather(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generate realistic seasonal and diurnal ambient temperature (Celsius)
        """
        n_steps = len(timestamps)
        day_of_year = timestamps.dayofyear.values
        hour = timestamps.hour.values
        
        # Seasonal base: summer peak mid-year
        seasonal_temp = 25.0 + 8.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        # Diurnal fluctuation: lowest at 5am, highest at 3pm (hour 15)
        diurnal_temp = 6.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
        noise = np.random.normal(0, 1.5, n_steps)
        
        temperature = seasonal_temp + diurnal_temp + noise
        
        # Cooling Degree Hours (CDH) and Heating Degree Hours (HDH)
        cdh = np.maximum(0, temperature - 24.0)
        hdh = np.maximum(0, 18.0 - temperature)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "temperature_c": temperature,
            "cooling_degree_hours": cdh,
            "heating_degree_hours": hdh
        })

    def _generate_base_load(self, 
                            meter_id: str, 
                            consumer_type: str, 
                            timestamps: pd.DatetimeIndex, 
                            weather_df: pd.DataFrame) -> np.ndarray:
        """
        Generate normal, un-tampered physics-driven electricity consumption (kWh).
        """
        n_steps = len(timestamps)
        hours = timestamps.hour.values
        is_weekend = (timestamps.dayofweek.values >= 5).astype(int)
        
        cdh = weather_df["cooling_degree_hours"].values
        hdh = weather_df["heating_degree_hours"].values
        
        if consumer_type == "Residential":
            base_kw = np.random.uniform(0.3, 1.2)
            # Diurnal morning peak (7-9am) and evening peak (6-11pm)
            morning = np.exp(-0.5 * ((hours - 8) / 1.5) ** 2) * 1.5
            evening = np.exp(-0.5 * ((hours - 20) / 2.0) ** 2) * 2.8
            weekend_boost = is_weekend * np.random.uniform(0.2, 0.6)
            # Weather sensitivity (AC in summer, heaters in winter)
            weather_load = cdh * np.random.uniform(0.08, 0.18) + hdh * np.random.uniform(0.04, 0.10)
            noise = np.random.gamma(shape=2.0, scale=0.15, size=n_steps)
            
            load = base_kw + morning + evening + weekend_boost + weather_load + noise
            
        elif consumer_type == "Commercial":
            base_kw = np.random.uniform(3.0, 7.0)
            # Active 8am to 8pm on weekdays, lower on weekends
            work_hours = ((hours >= 8) & (hours <= 20)).astype(float)
            weekday_mult = np.where(is_weekend == 1, 0.25, 1.0)
            operating_load = work_hours * weekday_mult * np.random.uniform(5.0, 15.0)
            weather_load = cdh * np.random.uniform(0.4, 0.9)
            noise = np.random.normal(0, 0.5, n_steps)
            
            load = np.maximum(0.5, base_kw + operating_load + weather_load + noise)
            
        else:  # Industrial
            base_kw = np.random.uniform(15.0, 35.0)
            # Two or three operational shifts
            shift_load = np.sin(2 * np.pi * hours / 24.0) * 4.0 + np.random.uniform(10.0, 25.0)
            noise = np.random.normal(0, 2.0, n_steps)
            load = np.maximum(5.0, base_kw + shift_load + noise)
            
        return np.maximum(0.01, load)

    def _inject_tampering(self, 
                          clean_load: np.ndarray, 
                          timestamps: pd.DatetimeIndex, 
                          attack_type: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Inject physically realistic meter tampering attacks.
        Returns:
            reported_load: tampered reading
            is_tampered: binary ground-truth array (1 = tampered, 0 = normal)
            tamper_start_idx: index where tamper commenced
        """
        n_steps = len(clean_load)
        # Tamper starts between 25% and 65% of the total observation period
        start_idx = np.random.randint(int(n_steps * 0.25), int(n_steps * 0.65))
        duration = min(n_steps - start_idx, int(self.config.min_tamper_duration_days * 24))
        end_idx = start_idx + duration
        
        tampered_load = clean_load.copy()
        is_tampered = np.zeros(n_steps, dtype=int)
        is_tampered[start_idx:end_idx] = 1
        
        hours = timestamps.hour.values
        
        if attack_type == "scaling_bypass":
            # Direct partial current bypass (registers 25% to 60% of real consumption)
            scale_alpha = np.random.uniform(0.25, 0.60)
            tampered_load[start_idx:end_idx] = clean_load[start_idx:end_idx] * scale_alpha
            
        elif attack_type == "flatline_floor":
            # Reports fixed low constant standby floor (e.g. 0.1 kWh)
            floor_val = np.percentile(clean_load, 5)
            tampered_load[start_idx:end_idx] = np.minimum(clean_load[start_idx:end_idx], floor_val)
            
        elif attack_type == "peak_selective_bypass":
            # Consumer bypasses meter selectively during expensive peak hours (18:00 - 23:00)
            for t in range(start_idx, end_idx):
                if 18 <= hours[t] <= 23:
                    tampered_load[t] = clean_load[t] * np.random.uniform(0.15, 0.35)
                    
        elif attack_type == "meter_reverse_flow":
            # Manipulation of reverse diodes or spoofed net metering yielding negative/zero readings
            for t in range(start_idx, end_idx):
                tampered_load[t] = np.maximum(0.0, clean_load[t] - np.random.uniform(1.0, 3.0))
                if np.random.rand() < 0.15:
                    tampered_load[t] = 0.0
                    
        elif attack_type == "gradual_drift_attenuation":
            # Slow potentiometer/magnetic attenuation degrading readings over weeks
            attack_len = end_idx - start_idx
            decay_curve = np.linspace(1.0, np.random.uniform(0.2, 0.4), attack_len)
            tampered_load[start_idx:end_idx] = clean_load[start_idx:end_idx] * decay_curve
            
        return tampered_load, is_tampered, start_idx

    def generate_fleet(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate complete fleet dataset:
        1. smart_meter_readings (hourly intervals per meter)
        2. meter_metadata (meter info, consumer type, feeder id, ground truth tamper status)
        3. feeder_supply_readings (transformer bulk metering for energy balance)
        """
        logger.info(f"Generating synthetic fleet with {self.config.num_meters} meters across {self.config.num_days} days...")
        
        start_date = pd.Timestamp("2026-01-01 00:00:00")
        total_hours = self.config.num_days * 24
        timestamps = pd.date_range(start=start_date, periods=total_hours, freq=f"{self.config.frequency_minutes}min")
        
        weather_df = self.generate_weather(timestamps)
        
        meters = []
        readings_list = []
        feeder_names = [f"FEEDER_{chr(65 + i)}" for i in range(self.config.num_feeders)]
        
        num_theft_meters = int(self.config.num_meters * self.config.theft_prevalence)
        tamper_meter_indices = set(np.random.choice(self.config.num_meters, size=num_theft_meters, replace=False))
        
        consumer_types = ["Residential"] * int(0.70 * self.config.num_meters) + \
                         ["Commercial"] * int(0.20 * self.config.num_meters) + \
                         ["Industrial"] * (self.config.num_meters - int(0.90 * self.config.num_meters))
        np.random.shuffle(consumer_types)
        
        for i in range(self.config.num_meters):
            m_id = f"MTR_{i+1:04d}"
            c_type = consumer_types[i]
            f_id = feeder_names[i % self.config.num_feeders]
            
            clean_load = self._generate_base_load(m_id, c_type, timestamps, weather_df)
            
            # Legitimate vacation/away period for ~10% of honest meters
            if i not in tamper_meter_indices and np.random.rand() < 0.12 and c_type == "Residential":
                vac_start = np.random.randint(int(total_hours * 0.3), int(total_hours * 0.7))
                vac_len = np.random.randint(3, 10) * 24
                clean_load[vac_start:min(total_hours, vac_start + vac_len)] = np.random.uniform(0.02, 0.08)
            
            is_tampered_flag = i in tamper_meter_indices
            attack_type = "none"
            start_idx = -1
            
            if is_tampered_flag:
                attack_type = np.random.choice(self.config.attack_types)
                reported_load, tamper_labels, start_idx = self._inject_tampering(clean_load, timestamps, attack_type)
            else:
                reported_load = clean_load.copy()
                tamper_labels = np.zeros(total_hours, dtype=int)
            
            # Meter metadata
            meters.append({
                "meter_id": m_id,
                "feeder_id": f_id,
                "consumer_type": c_type,
                "is_theft_meter": int(is_tampered_flag),
                "attack_type": attack_type,
                "tamper_start_timestamp": timestamps[start_idx] if start_idx != -1 else pd.NaT,
                "base_contract_kw": float(np.percentile(clean_load, 95))
            })
            
            # Time series dataframe chunk
            df_meter = pd.DataFrame({
                "timestamp": timestamps,
                "meter_id": m_id,
                "feeder_id": f_id,
                "actual_consumption_kwh": clean_load,
                "reported_consumption_kwh": reported_load,
                "is_tampered": tamper_labels,
                "temperature_c": weather_df["temperature_c"].values
            })
            readings_list.append(df_meter)
            
        readings_df = pd.concat(readings_list, ignore_index=True)
        meters_df = pd.DataFrame(meters)
        
        # Calculate bulk Feeder Transformer Supply (actual total load + 4-7% normal technical line losses)
        feeder_readings = []
        for f_id in feeder_names:
            feeder_sub = readings_df[readings_df["feeder_id"] == f_id]
            actual_sum = feeder_sub.groupby("timestamp")["actual_consumption_kwh"].sum()
            reported_sum = feeder_sub.groupby("timestamp")["reported_consumption_kwh"].sum()
            
            # Technical line losses: ~5% dissipation
            tech_loss_ratio = 0.05
            transformer_supply = actual_sum * (1.0 + tech_loss_ratio) + np.random.normal(0, 0.5, len(actual_sum))
            
            f_df = pd.DataFrame({
                "timestamp": actual_sum.index,
                "feeder_id": f_id,
                "feeder_supply_kwh": transformer_supply.values,
                "feeder_reported_sum_kwh": reported_sum.values,
                "feeder_actual_sum_kwh": actual_sum.values
            })
            feeder_readings.append(f_df)
            
        feeders_df = pd.concat(feeder_readings, ignore_index=True)
        
        logger.info(f"Generated {len(readings_df):,} smart meter interval records for {len(meters_df)} meters.")
        return readings_df, meters_df, feeders_df
