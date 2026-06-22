"""
Dynamic Optimization Recommender
Generates energy/carbon recommendations based on real ESP32 sensor data with trend analysis
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import statistics


@dataclass
class Recommendation:
    """Single optimization recommendation"""
    title: str
    description: str
    icon: str
    daily_savings: Dict[str, float]  # kWh, $, kg CO2
    timeline: str
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    factors: List[str]  # What triggered this recommendation


class DynamicRecommender:
    """
    Generates optimization recommendations based on real-time ESP32 data
    Maintains 300-reading history for trend analysis
    """
    
    def __init__(self, max_history=300):
        self.history = deque(maxlen=max_history)
        self.carbon_intensity = 150  # gCO2/kWh (default, should be updated)
        self.electricity_cost = 0.12  # $/kWh (default)
    
    def add_reading(self, current_a: float, temp_c: float, humidity_pct: float, power_w: float):
        """Add new ESP32 sensor reading to history"""
        self.history.append({
            "current": current_a,
            "temperature": temp_c,
            "humidity": humidity_pct,
            "power": power_w,
        })
    
    def _get_trend(self, metric: str, window: int = 30) -> str:
        """Analyze trend for a metric (increasing/stable/decreasing)"""
        if len(self.history) < window:
            return "stable"
        
        recent = [r[metric] for r in list(self.history)[-window:]]
        first_half = statistics.mean(recent[:window//2])
        second_half = statistics.mean(recent[window//2:])
        delta = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
        
        if delta > 5:
            return "increasing"
        elif delta < -5:
            return "decreasing"
        return "stable"
    
    def _calc_maintenance_score(self, current_a: float, temp_c: float, humidity_pct: float) -> Tuple[int, str]:
        """
        Multi-factor maintenance scoring (0-12 points)
        Returns (score, confidence_level)
        """
        score = 0
        factors = []
        
        # Factor 1: Current draw stress (0-4 points)
        if current_a > 100:
            score += 4
            factors.append("Current >100A (critical stress)")
        elif current_a > 80:
            score += 3
            factors.append("Current 80-100A (high stress)")
        elif current_a > 60:
            score += 2
            factors.append("Current 60-80A (moderate stress)")
        else:
            score += 1
            factors.append("Current optimal")
        
        # Factor 2: Temperature (0-4 points)
        if temp_c > 55:
            score += 4
            factors.append("Temperature >55°C (critical)")
        elif temp_c > 45:
            score += 3
            factors.append("Temperature 45-55°C (elevated)")
        elif temp_c > 40:
            score += 2
            factors.append("Temperature 40-45°C (watch)")
        else:
            factors.append("Temperature normal")
        
        # Factor 3: Humidity (0-2 points)
        if humidity_pct < 30 or humidity_pct > 80:
            score += 2
            factors.append(f"Humidity {humidity_pct}% (at edge)")
        else:
            factors.append(f"Humidity {humidity_pct}% (optimal)")
        
        # Factor 4: Trend analysis (0-2 points)
        current_trend = self._get_trend("current")
        if current_trend == "increasing":
            score += 2
            factors.append("Current increasing (worsening)")
        elif current_trend == "decreasing":
            factors.append("Current decreasing (improving)")
        
        # Confidence mapping
        if score >= 10:
            confidence = "HIGH"
        elif score >= 6:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return score, confidence, factors
    
    def get_recommendations(self, current_a: float = None, temp_c: float = None, 
                          humidity_pct: float = None, power_w: float = None) -> List[Recommendation]:
        """
        Generate all optimization recommendations based on current ESP32 data
        Falls back to history average if real-time data unavailable
        """
        if current_a is None or temp_c is None or humidity_pct is None or power_w is None:
            if not self.history:
                # No data at all - return empty
                return []
            # Use averages from history
            current_a = statistics.mean([r["current"] for r in self.history])
            temp_c = statistics.mean([r["temperature"] for r in self.history])
            humidity_pct = statistics.mean([r["humidity"] for r in self.history])
            power_w = statistics.mean([r["power"] for r in self.history])
        
        self.add_reading(current_a, temp_c, humidity_pct, power_w)
        recommendations = []
        
        # 1️⃣ LOAD BALANCING & PEAK SHAVING
        if current_a > 50:  # Only if drawing significant power
            target_current = 50
            reduction_pct = ((current_a - target_current) / current_a) * 100
            daily_kwh_saved = (power_w / 1000) * (reduction_pct / 100) * 8  # 8 hours peak
            daily_cost = daily_kwh_saved * self.electricity_cost
            daily_carbon = daily_kwh_saved * self.carbon_intensity / 1000
            
            confidence = "HIGH" if current_a > 80 else "MEDIUM" if current_a > 60 else "LOW"
            
            recommendations.append(Recommendation(
                title="⚡ Load Balancing & Peak Shaving",
                description=f"Reduce current draw from {current_a:.1f}A to {target_current}A through load distribution. Currently drawing {reduction_pct:.0f}% above optimal.",
                icon="⚡",
                daily_savings={"kwh": daily_kwh_saved, "cost": daily_cost, "carbon": daily_carbon},
                timeline="2-4 weeks",
                confidence=confidence,
                factors=[
                    f"Current {current_a:.1f}A (target: {target_current}A)",
                    f"Reduction potential: {reduction_pct:.0f}%",
                    self._get_trend("current")
                ]
            ))
        
        # 2️⃣ THERMAL MANAGEMENT
        if temp_c > 35:  # If elevated temperature
            # Non-linear model: efficiency gain increases with temperature
            if temp_c > 50:
                efficiency_gain_pct = 1.0 * (temp_c - 35)  # 15°C overshoot = 15% gain
            elif temp_c > 40:
                efficiency_gain_pct = 0.7 * (temp_c - 35)  # Moderate gain
            else:
                efficiency_gain_pct = 0.4 * (temp_c - 35)  # Small gain
            
            daily_kwh_saved = (power_w / 1000) * (efficiency_gain_pct / 100) * 16  # 16 hours operation
            daily_cost = daily_kwh_saved * self.electricity_cost
            daily_carbon = daily_kwh_saved * self.carbon_intensity / 1000
            
            confidence = "HIGH" if temp_c > 50 else "MEDIUM" if temp_c > 43 else "LOW"
            
            recommendations.append(Recommendation(
                title="🌡️ Thermal Management",
                description=f"Optimize cooling systems to lower temperature from {temp_c:.1f}°C. Improved cooling efficiency gains {efficiency_gain_pct:.1f}% energy savings.",
                icon="🌡️",
                daily_savings={"kwh": daily_kwh_saved, "cost": daily_cost, "carbon": daily_carbon},
                timeline="1-3 weeks",
                confidence=confidence,
                factors=[
                    f"Temperature {temp_c:.1f}°C (optimal <35°C)",
                    f"Efficiency gain potential: {efficiency_gain_pct:.1f}%",
                    self._get_trend("temperature")
                ]
            ))
        
        # 3️⃣ CARBON-AWARE SCHEDULING
        # Check if currently in high-carbon window
        shift_pct = 0
        if current_a > 40:  # Only if significant load
            # Assume peak hours 8-17, off-peak 18-7
            shift_pct = 30 if current_a < 80 else 50  # More shiftable if not at stress
            daily_kwh_saved = (power_w / 1000) * (shift_pct / 100) * 6  # 6-hour shift window
            daily_carbon = daily_kwh_saved * (self.carbon_intensity * 0.4) / 1000  # Off-peak is 40% of peak carbon
            daily_cost = daily_kwh_saved * (self.electricity_cost * 0.85)  # Off-peak pricing discount
            
            confidence = "HIGH" if humidity_pct > 50 else "MEDIUM"
            
            recommendations.append(Recommendation(
                title="🕐 Carbon-Aware Scheduling",
                description=f"Shift {shift_pct}% of production to off-peak low-carbon hours (18:00-07:00). Grid carbon intensity is currently {self.carbon_intensity} gCO₂/kWh.",
                icon="🕐",
                daily_savings={"kwh": daily_kwh_saved, "cost": daily_cost, "carbon": daily_carbon},
                timeline="Immediate",
                confidence=confidence,
                factors=[
                    f"Shiftable load: {shift_pct}%",
                    "Off-peak pricing: -15% cost",
                    "Off-peak carbon: -60% intensity"
                ]
            ))
        
        # 4️⃣ PREDICTIVE MAINTENANCE
        maintenance_score, maint_confidence, maint_factors = self._calc_maintenance_score(current_a, temp_c, humidity_pct)
        
        if maintenance_score >= 6:  # Only if score warrants action
            efficiency_gain_pct = 8 if maintenance_score >= 10 else 5 if maintenance_score >= 8 else 2
            daily_kwh_saved = (power_w / 1000) * (efficiency_gain_pct / 100) * 24
            daily_cost = daily_kwh_saved * self.electricity_cost
            daily_carbon = daily_kwh_saved * self.carbon_intensity / 1000
            
            recommendations.append(Recommendation(
                title="🔧 Predictive Maintenance",
                description=f"Maintenance score: {maintenance_score}/12. Proactive maintenance can recover {efficiency_gain_pct}% efficiency by addressing wear patterns.",
                icon="🔧",
                daily_savings={"kwh": daily_kwh_saved, "cost": daily_cost, "carbon": daily_carbon},
                timeline="1-2 weeks",
                confidence=maint_confidence,
                factors=maint_factors
            ))
        
        # 5️⃣ HUMIDITY OPTIMIZATION (if needed)
        if humidity_pct < 30 or humidity_pct > 80:
            efficiency_gain_pct = 3
            daily_kwh_saved = (power_w / 1000) * (efficiency_gain_pct / 100) * 24
            daily_cost = daily_kwh_saved * self.electricity_cost
            daily_carbon = daily_kwh_saved * self.carbon_intensity / 1000
            
            recommendations.append(Recommendation(
                title="💨 Humidity Control",
                description=f"Current humidity at {humidity_pct}% is outside optimal 40-70% range. Stabilizing humidity improves component efficiency by ~3%.",
                icon="💨",
                daily_savings={"kwh": daily_kwh_saved, "cost": daily_cost, "carbon": daily_carbon},
                timeline="1 week",
                confidence="MEDIUM",
                factors=[
                    f"Humidity {humidity_pct}% (optimal: 40-70%)",
                    "Efficiency gain: 3%"
                ]
            ))
        
        return recommendations
    
    def get_cumulative_impact(self, recommendations: List[Recommendation]) -> Dict:
        """Calculate total annual savings from all recommendations"""
        total_kwh = sum(r.daily_savings["kwh"] for r in recommendations)
        total_cost = sum(r.daily_savings["cost"] for r in recommendations)
        total_carbon = sum(r.daily_savings["carbon"] for r in recommendations)
        
        annual_kwh = total_kwh * 365
        annual_cost = total_cost * 365
        annual_carbon = total_carbon * 365
        
        # Estimate implementation cost at $50k
        impl_cost = 50000
        payback_months = (impl_cost / annual_cost * 12) if annual_cost > 0 else 999
        
        return {
            "annual_kwh": annual_kwh,
            "annual_cost": annual_cost,
            "annual_carbon": annual_carbon,
            "payback_months": payback_months,
            "num_recommendations": len(recommendations)
        }
