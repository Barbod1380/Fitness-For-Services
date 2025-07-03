# core/enhanced_growth_modeling.py
"""
Enhanced physics-based growth modeling to replace linear assumptions.
Implements industry-standard corrosion growth models with uncertainty quantification.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

@dataclass
class EnvironmentParams:
    """Environmental parameters affecting corrosion growth."""
    temperature_c: float = 20.0  # Operating temperature (°C)
    ph: float = 7.0  # pH of product/environment
    chloride_ppm: float = 0.0  # Chloride concentration (ppm)
    flow_velocity_ms: float = 1.0  # Flow velocity (m/s)
    pressure_mpa: float = 5.0  # Operating pressure (MPa)
    product_type: str = "natural_gas"  # natural_gas, crude_oil, water, etc.
    coating_condition: str = "good"  # good, fair, poor, none
    cp_potential_mv: float = -850.0  # Cathodic protection potential (mV CSE)

@dataclass
class GrowthModelResult:
    """Result from enhanced growth modeling."""
    growth_rate_mm_per_year: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_used: str
    uncertainty_factors: Dict
    environmental_acceleration: float = 1.0

class EnhancedGrowthModeling:
    """
    Physics-based corrosion growth modeling per NACE/AMPP standards.
    """
    
    def __init__(self):
        # Industry corrosion rate databases (simplified)
        self.base_corrosion_rates = {
            'external_soil': {
                'natural_gas': 0.025,  # mm/year for typical soil
                'crude_oil': 0.030,
                'water': 0.040
            },
            'internal_flow': {
                'natural_gas': 0.015,  # Dry gas is less corrosive
                'crude_oil': 0.050,   # Can be more corrosive due to H2S, CO2
                'water': 0.080        # Most corrosive for steel
            }
        }
        
        # Environmental acceleration factors
        self.acceleration_factors = {
            'temperature': self._temperature_acceleration,
            'ph': self._ph_acceleration,
            'chloride': self._chloride_acceleration,
            'flow': self._flow_acceleration,
            'pressure': self._pressure_acceleration,
            'coating': self._coating_factor,
            'cp': self._cathodic_protection_factor
        }
    
    def calculate_enhanced_growth_rate(self, 
                                     defect_properties: Dict,
                                     environment: EnvironmentParams,
                                     historical_data: Optional[pd.DataFrame] = None,
                                     surface_location: str = "external") -> GrowthModelResult:
        """
        Calculate growth rate using enhanced physics-based models.
        
        Parameters:
        - defect_properties: Dict with current defect size, age, etc.
        - environment: Environmental parameters
        - historical_data: Historical inspection data for calibration
        - surface_location: 'internal' or 'external'
        
        Returns:
        - GrowthModelResult with rate and uncertainty bounds
        """
        
        # Step 1: Select base corrosion model
        if historical_data is not None and len(historical_data) >= 2:
            # Use data-driven model if sufficient history
            base_rate = self._fit_power_law_model(historical_data)
            model_used = "data_driven_power_law"
            base_uncertainty = 0.3  # 30% uncertainty for fitted models
        else:
            # Use physics-based estimation
            base_rate = self._estimate_base_rate(environment, surface_location)
            model_used = "physics_based_estimation"
            base_uncertainty = 0.5  # 50% uncertainty for estimates
        
        # Step 2: Apply environmental acceleration factors
        total_acceleration = 1.0
        acceleration_details = {}
        
        for factor_name, factor_func in self.acceleration_factors.items():
            try:
                acceleration = factor_func(environment)
                total_acceleration *= acceleration
                acceleration_details[factor_name] = acceleration
            except Exception as e:
                warnings.warn(f"Error calculating {factor_name} acceleration: {e}")
                acceleration_details[factor_name] = 1.0
        
        # Step 3: Apply defect-specific factors
        defect_acceleration = self._calculate_defect_specific_factors(defect_properties)
        total_acceleration *= defect_acceleration
        acceleration_details['defect_geometry'] = defect_acceleration
        
        # Step 4: Calculate final growth rate
        enhanced_rate = base_rate * total_acceleration
        
        # Step 5: Calculate uncertainty bounds
        uncertainty_factor = self._calculate_uncertainty_factor(
            base_uncertainty, environment, defect_properties, model_used
        )
        
        lower_bound = enhanced_rate * (1 - uncertainty_factor)
        upper_bound = enhanced_rate * (1 + uncertainty_factor)
        
        # Step 6: Apply physical constraints
        enhanced_rate = max(0.001, enhanced_rate)  # Minimum 0.001 mm/year
        enhanced_rate = min(2.0, enhanced_rate)    # Maximum 2.0 mm/year (very aggressive)
        
        return GrowthModelResult(
            growth_rate_mm_per_year=enhanced_rate,
            confidence_interval_lower=max(0.001, lower_bound),
            confidence_interval_upper=min(2.0, upper_bound),
            model_used=model_used,
            uncertainty_factors=acceleration_details,
            environmental_acceleration=total_acceleration
        )
    
    def _fit_power_law_model(self, historical_data: pd.DataFrame) -> float:
        """
        Fit power law model: depth(t) = a * t^b
        This is more realistic than linear growth for corrosion.
        """
        try:
            # Sort by time
            data = historical_data.sort_values('inspection_year')
            
            if len(data) < 2:
                return 0.05  # Default fallback
            
            # Calculate time differences and depth changes
            times = data['inspection_year'].values
            depths = data['depth_mm'].values
            
            # Remove zero or negative depths
            valid_mask = depths > 0
            times = times[valid_mask]
            depths = depths[valid_mask]
            
            if len(times) < 2:
                return 0.05
            
            # Fit power law using log-linear regression
            # log(depth) = log(a) + b * log(t)
            log_times = np.log(times - times[0] + 1)  # Avoid log(0)
            log_depths = np.log(depths)
            
            # Linear regression on log-transformed data
            coeffs = np.polyfit(log_times, log_depths, 1)
            b = coeffs[0]  # Power law exponent
            log_a = coeffs[1]  # Log of coefficient
            
            # Calculate current growth rate: d(depth)/dt = a * b * t^(b-1)
            current_time = times[-1] - times[0] + 1
            current_rate = np.exp(log_a) * b * (current_time ** (b - 1))
            
            # Convert to mm/year and apply reasonable bounds
            return np.clip(current_rate, 0.005, 1.0)
            
        except Exception as e:
            warnings.warn(f"Power law fitting failed: {e}. Using fallback rate.")
            return 0.05  # Fallback rate
    
    def _estimate_base_rate(self, environment: EnvironmentParams, surface_location: str) -> float:
        """Estimate base corrosion rate from environment."""
        location_key = 'external_soil' if surface_location == 'external' else 'internal_flow'
        
        if location_key in self.base_corrosion_rates:
            rates = self.base_corrosion_rates[location_key]
            return rates.get(environment.product_type, 0.05)  # Default 0.05 mm/year
        else:
            return 0.05
    
    def _temperature_acceleration(self, env: EnvironmentParams) -> float:
        """
        Calculate temperature acceleration factor.
        Rule of thumb: corrosion rate doubles every 25°C increase.
        """
        reference_temp = 20.0  # °C
        temp_diff = env.temperature_c - reference_temp
        
        # Arrhenius-type relationship
        activation_energy = 30000  # J/mol (typical for steel corrosion)
        R = 8.314  # J/mol/K
        
        factor = np.exp(activation_energy / R * (1/(reference_temp + 273) - 1/(env.temperature_c + 273)))
        
        return np.clip(factor, 0.1, 10.0)  # Reasonable bounds
    
    def _ph_acceleration(self, env: EnvironmentParams) -> float:
        """
        Calculate pH acceleration factor.
        Steel corrosion is worst at low pH (acidic conditions).
        """
        if env.ph < 4.0:
            return 5.0  # Very acidic - high acceleration
        elif env.ph < 6.0:
            return 2.0  # Acidic - moderate acceleration
        elif env.ph <= 8.0:
            return 1.0  # Neutral - baseline
        elif env.ph <= 10.0:
            return 0.7  # Slightly alkaline - slight protection
        else:
            return 0.5  # Very alkaline - significant protection
    
    def _chloride_acceleration(self, env: EnvironmentParams) -> float:
        """
        Calculate chloride acceleration factor.
        Chlorides promote pitting and crevice corrosion.
        """
        if env.chloride_ppm < 100:
            return 1.0  # Low chloride
        elif env.chloride_ppm < 1000:
            return 1.5  # Moderate chloride
        elif env.chloride_ppm < 10000:
            return 2.5  # High chloride
        else:
            return 4.0  # Very high chloride (seawater level)
    
    def _flow_acceleration(self, env: EnvironmentParams) -> float:
        """
        Calculate flow velocity acceleration factor.
        Higher flow can increase mass transfer (erosion-corrosion).
        """
        if env.flow_velocity_ms < 1.0:
            return 1.0  # Low flow
        elif env.flow_velocity_ms < 3.0:
            return 1.2  # Moderate flow
        elif env.flow_velocity_ms < 6.0:
            return 1.8  # High flow - erosion-corrosion
        else:
            return 3.0  # Very high flow - significant erosion component
    
    def _pressure_acceleration(self, env: EnvironmentParams) -> float:
        """
        Calculate pressure acceleration factor.
        Higher pressure can affect corrosion kinetics.
        """
        # Simplified model - pressure effect is usually secondary
        if env.pressure_mpa < 1.0:
            return 1.0
        elif env.pressure_mpa < 10.0:
            return 1.1  # Slight acceleration
        else:
            return 1.2  # Higher pressure environments
    
    def _coating_factor(self, env: EnvironmentParams) -> float:
        """
        Calculate coating protection factor.
        """
        coating_factors = {
            'good': 0.1,     # 90% protection
            'fair': 0.3,     # 70% protection
            'poor': 0.6,     # 40% protection
            'none': 1.0      # No protection
        }
        return coating_factors.get(env.coating_condition, 1.0)
    
    def _cathodic_protection_factor(self, env: EnvironmentParams) -> float:
        """
        Calculate cathodic protection effectiveness.
        For external corrosion only.
        """
        # CSE potentials (mV)
        if env.cp_potential_mv < -900:
            return 0.05  # Excellent protection
        elif env.cp_potential_mv < -850:
            return 0.1   # Good protection  
        elif env.cp_potential_mv < -800:
            return 0.3   # Fair protection
        elif env.cp_potential_mv < -750:
            return 0.6   # Poor protection
        else:
            return 1.0   # No effective protection
    
    def _calculate_defect_specific_factors(self, defect_properties: Dict) -> float:
        """
        Calculate defect geometry and stress concentration effects.
        """
        acceleration = 1.0
        
        # Depth effect - deeper defects can grow faster due to stress concentration
        if 'depth_pct' in defect_properties:
            depth_pct = defect_properties['depth_pct']
            if depth_pct > 50:
                acceleration *= 1.3  # Stress concentration factor
            elif depth_pct > 30:
                acceleration *= 1.1
        
        # Length effect - longer defects can have higher stress at tips
        if 'length_mm' in defect_properties:
            length_mm = defect_properties['length_mm']
            if length_mm > 200:  # Long defects
                acceleration *= 1.2
            elif length_mm > 100:
                acceleration *= 1.1
        
        # Defect type effect
        if 'defect_type' in defect_properties:
            defect_type = defect_properties['defect_type'].lower()
            if 'pitting' in defect_type:
                acceleration *= 1.4  # Pitting can accelerate
            elif 'groove' in defect_type:
                acceleration *= 1.2  # Grooves have stress concentration
        
        return acceleration
    
    def _calculate_uncertainty_factor(self, base_uncertainty: float, 
                                    environment: EnvironmentParams,
                                    defect_properties: Dict,
                                    model_used: str) -> float:
        """
        Calculate overall uncertainty factor for growth rate prediction.
        """
        uncertainty = base_uncertainty
        
        # Add uncertainty based on data quality
        if model_used == "physics_based_estimation":
            uncertainty += 0.2  # Higher uncertainty for estimates
        
        # Environmental uncertainty
        if environment.temperature_c > 50 or environment.temperature_c < 0:
            uncertainty += 0.1  # Extreme temperatures
        
        if environment.ph < 5 or environment.ph > 9:
            uncertainty += 0.1  # Extreme pH
        
        # Defect complexity uncertainty
        if defect_properties.get('is_combined', False):
            uncertainty += 0.15  # Combined defects have higher uncertainty
        
        # Cap uncertainty at reasonable bounds
        return np.clip(uncertainty, 0.2, 0.8)  # 20% to 80% uncertainty


# Integration function for use in remaining life analysis
def apply_enhanced_growth_modeling(defects_df: pd.DataFrame, 
                                 environment_params: EnvironmentParams,
                                 historical_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Apply enhanced growth modeling to all defects in a DataFrame.
    
    Parameters:
    - defects_df: DataFrame with current defect data
    - environment_params: Environmental conditions
    - historical_data: Optional historical growth data for calibration
    
    Returns:
    - DataFrame with enhanced growth rate predictions
    """
    modeler = EnhancedGrowthModeling()
    enhanced_df = defects_df.copy()
    
    # Initialize new columns
    enhanced_df['enhanced_growth_rate_mm_year'] = 0.0
    enhanced_df['growth_rate_lower_bound'] = 0.0
    enhanced_df['growth_rate_upper_bound'] = 0.0
    enhanced_df['growth_model_used'] = ''
    enhanced_df['environmental_acceleration'] = 1.0
    enhanced_df['growth_uncertainty_pct'] = 0.0
    
    for idx, defect in enhanced_df.iterrows():
        try:
            # Prepare defect properties
            defect_props = {
                'depth_pct': defect.get('depth [%]', 0),
                'length_mm': defect.get('length [mm]', 0),
                'width_mm': defect.get('width [mm]', 0),
                'defect_type': defect.get('component / anomaly identification', ''),
                'is_combined': defect.get('is_combined', False)
            }
            
            # Determine surface location
            surface_location = "external"
            if 'surface location' in defect.columns:
                if defect['surface location'] == 'INT':
                    surface_location = "internal"
            
            # Calculate enhanced growth rate
            result = modeler.calculate_enhanced_growth_rate(
                defect_properties=defect_props,
                environment=environment_params,
                historical_data=historical_data,
                surface_location=surface_location
            )
            
            # Store results
            enhanced_df.loc[idx, 'enhanced_growth_rate_mm_year'] = result.growth_rate_mm_per_year
            enhanced_df.loc[idx, 'growth_rate_lower_bound'] = result.confidence_interval_lower
            enhanced_df.loc[idx, 'growth_rate_upper_bound'] = result.confidence_interval_upper
            enhanced_df.loc[idx, 'growth_model_used'] = result.model_used
            enhanced_df.loc[idx, 'environmental_acceleration'] = result.environmental_acceleration
            
            # Calculate uncertainty percentage
            uncertainty_pct = (result.confidence_interval_upper - result.confidence_interval_lower) / \
                            (2 * result.growth_rate_mm_per_year) * 100
            enhanced_df.loc[idx, 'growth_uncertainty_pct'] = uncertainty_pct
            
        except Exception as e:
            warnings.warn(f"Error calculating enhanced growth for defect {idx}: {e}")
            # Use fallback values
            enhanced_df.loc[idx, 'enhanced_growth_rate_mm_year'] = 0.05
            enhanced_df.loc[idx, 'growth_rate_lower_bound'] = 0.025
            enhanced_df.loc[idx, 'growth_rate_upper_bound'] = 0.075
            enhanced_df.loc[idx, 'growth_model_used'] = 'fallback'
            enhanced_df.loc[idx, 'environmental_acceleration'] = 1.0
            enhanced_df.loc[idx, 'growth_uncertainty_pct'] = 50.0
    
    return enhanced_df