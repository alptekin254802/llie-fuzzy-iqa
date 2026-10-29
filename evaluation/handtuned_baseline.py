"""Hand-tuned Mamdani baseline from the perceptual-gap comparison.

This reference-based baseline uses PSNR, SSIM and entropy, fixed trapezoids,
minimum rule activation, maximum aggregation and centroid defuzzification.
It is distinct from the calibrated singleton models.
"""
from __future__ import annotations
from typing import Dict,Optional,Tuple
import numpy as np
import skfuzzy as fuzz
class FuzzyEnhancementEvaluator:

    def __init__(self, psnr_range: Optional[Tuple[float, float]]=None, ssim_range: Optional[Tuple[float, float]]=None, entropy_range: Optional[Tuple[float, float]]=None, quality_range: Optional[Tuple[float, float]]=None, psnr_params: Optional[Dict]=None, ssim_params: Optional[Dict]=None, entropy_params: Optional[Dict]=None, quality_params: Optional[Dict]=None):
        self.psnr_range = psnr_range or (0.0, 50.0)
        self.ssim_range = ssim_range or (0.0, 1.0)
        self.entropy_range = entropy_range or (0.0, 8.0)
        self.quality_range = quality_range or (0.0, 100.0)
        self.psnr_params = psnr_params or {'low': (0.0, 0.0, 15.0, 25.0), 'medium': (15.0, 25.0, 30.0, 40.0), 'high': (30.0, 40.0, 50.0, 50.0)}
        self.ssim_params = ssim_params or {'low': (0.0, 0.0, 0.3, 0.5), 'medium': (0.3, 0.5, 0.7, 0.85), 'high': (0.7, 0.85, 1.0, 1.0)}
        self.entropy_params = entropy_params or {'low': (0.0, 0.0, 2.0, 4.0), 'medium': (2.0, 4.0, 5.0, 6.5), 'high': (5.0, 6.5, 8.0, 8.0)}
        self.quality_params = quality_params or {'poor': (0.0, 0.0, 20.0, 40.0), 'fair': (20.0, 40.0, 50.0, 65.0), 'good': (50.0, 65.0, 75.0, 90.0), 'excellent': (75.0, 90.0, 100.0, 100.0)}
        self._setup_universe()
        self._setup_membership_functions()
        self._setup_rules()

    def _setup_universe(self) -> None:
        resolution = 1000
        self.psnr_universe = np.linspace(self.psnr_range[0], self.psnr_range[1], resolution)
        self.ssim_universe = np.linspace(self.ssim_range[0], self.ssim_range[1], resolution)
        self.entropy_universe = np.linspace(self.entropy_range[0], self.entropy_range[1], resolution)
        self.quality_universe = np.linspace(self.quality_range[0], self.quality_range[1], resolution)

    def _setup_membership_functions(self) -> None:
        self.psnr_low = fuzz.trapmf(self.psnr_universe, self.psnr_params['low'])
        self.psnr_medium = fuzz.trapmf(self.psnr_universe, self.psnr_params['medium'])
        self.psnr_high = fuzz.trapmf(self.psnr_universe, self.psnr_params['high'])
        self.ssim_low = fuzz.trapmf(self.ssim_universe, self.ssim_params['low'])
        self.ssim_medium = fuzz.trapmf(self.ssim_universe, self.ssim_params['medium'])
        self.ssim_high = fuzz.trapmf(self.ssim_universe, self.ssim_params['high'])
        self.entropy_low = fuzz.trapmf(self.entropy_universe, self.entropy_params['low'])
        self.entropy_medium = fuzz.trapmf(self.entropy_universe, self.entropy_params['medium'])
        self.entropy_high = fuzz.trapmf(self.entropy_universe, self.entropy_params['high'])
        self.quality_poor = fuzz.trapmf(self.quality_universe, self.quality_params['poor'])
        self.quality_fair = fuzz.trapmf(self.quality_universe, self.quality_params['fair'])
        self.quality_good = fuzz.trapmf(self.quality_universe, self.quality_params['good'])
        self.quality_excellent = fuzz.trapmf(self.quality_universe, self.quality_params['excellent'])

    def _setup_rules(self) -> None:
        self.rule1 = np.fmin(np.fmin(fuzz.interp_membership(self.psnr_universe, self.psnr_low, 0), fuzz.interp_membership(self.ssim_universe, self.ssim_low, 0)), fuzz.interp_membership(self.entropy_universe, self.entropy_low, 0))
        self.rules = []

    def evaluate(self, psnr: float, ssim: float, entropy: float) -> Tuple[float, str]:
        if not self.psnr_range[0] <= psnr <= self.psnr_range[1]:
            raise ValueError(f'PSNR value {psnr} is outside the defined range [{self.psnr_range[0]}, {self.psnr_range[1]}].')
        if not self.ssim_range[0] <= ssim <= self.ssim_range[1]:
            raise ValueError(f'SSIM value {ssim} is outside the defined range [{self.ssim_range[0]}, {self.ssim_range[1]}].')
        if not self.entropy_range[0] <= entropy <= self.entropy_range[1]:
            raise ValueError(f'Entropy value {entropy} is outside the defined range [{self.entropy_range[0]}, {self.entropy_range[1]}].')
        psnr_level_low = fuzz.interp_membership(self.psnr_universe, self.psnr_low, psnr)
        psnr_level_medium = fuzz.interp_membership(self.psnr_universe, self.psnr_medium, psnr)
        psnr_level_high = fuzz.interp_membership(self.psnr_universe, self.psnr_high, psnr)
        ssim_level_low = fuzz.interp_membership(self.ssim_universe, self.ssim_low, ssim)
        ssim_level_medium = fuzz.interp_membership(self.ssim_universe, self.ssim_medium, ssim)
        ssim_level_high = fuzz.interp_membership(self.ssim_universe, self.ssim_high, ssim)
        entropy_level_low = fuzz.interp_membership(self.entropy_universe, self.entropy_low, entropy)
        entropy_level_medium = fuzz.interp_membership(self.entropy_universe, self.entropy_medium, entropy)
        entropy_level_high = fuzz.interp_membership(self.entropy_universe, self.entropy_high, entropy)
        poor_activation = np.fmax(np.fmax(np.fmax(np.fmin(np.fmin(psnr_level_low, ssim_level_low), entropy_level_low), np.fmax(np.fmax(np.fmin(np.fmin(psnr_level_low, ssim_level_low), entropy_level_medium), np.fmin(np.fmin(psnr_level_low, ssim_level_medium), entropy_level_low)), np.fmin(np.fmin(psnr_level_medium, ssim_level_low), entropy_level_low))), np.fmin(np.fmin(psnr_level_low, ssim_level_low), entropy_level_high)), 0.0)
        fair_activation = np.fmax(np.fmax(np.fmax(np.fmin(np.fmin(psnr_level_medium, ssim_level_medium), entropy_level_medium), np.fmax(np.fmax(np.fmin(np.fmin(psnr_level_low, ssim_level_medium), entropy_level_medium), np.fmin(np.fmin(psnr_level_medium, ssim_level_low), entropy_level_medium)), np.fmin(np.fmin(psnr_level_medium, ssim_level_medium), entropy_level_low))), np.fmax(np.fmin(np.fmin(psnr_level_medium, ssim_level_low), entropy_level_high), np.fmin(np.fmin(psnr_level_low, ssim_level_medium), entropy_level_high))), 0.0)
        good_activation = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmin(np.fmin(psnr_level_high, ssim_level_medium), entropy_level_medium), np.fmin(np.fmin(psnr_level_medium, ssim_level_high), entropy_level_medium)), np.fmin(np.fmin(psnr_level_medium, ssim_level_medium), entropy_level_high)), np.fmax(np.fmax(np.fmin(np.fmin(psnr_level_high, ssim_level_high), entropy_level_medium), np.fmin(np.fmin(psnr_level_high, ssim_level_medium), entropy_level_high)), np.fmin(np.fmin(psnr_level_medium, ssim_level_high), entropy_level_high))), np.fmin(np.fmin(psnr_level_medium, ssim_level_medium), entropy_level_high)), 0.0)
        excellent_activation = np.fmin(np.fmin(psnr_level_high, ssim_level_high), entropy_level_high)
        quality_poor_clipped = np.fmin(poor_activation, self.quality_poor)
        quality_fair_clipped = np.fmin(fair_activation, self.quality_fair)
        quality_good_clipped = np.fmin(good_activation, self.quality_good)
        quality_excellent_clipped = np.fmin(excellent_activation, self.quality_excellent)
        aggregated = np.fmax(np.fmax(quality_poor_clipped, quality_fair_clipped), np.fmax(quality_good_clipped, quality_excellent_clipped))
        if np.max(aggregated) == 0.0:
            max_act = max(poor_activation, fair_activation, good_activation, excellent_activation)
            if max_act > 0:
                if max_act == poor_activation:
                    aggregated = np.fmin(max_act, self.quality_poor)
                elif max_act == fair_activation:
                    aggregated = np.fmin(max_act, self.quality_fair)
                elif max_act == good_activation:
                    aggregated = np.fmin(max_act, self.quality_good)
                else:
                    aggregated = np.fmin(max_act, self.quality_excellent)
            elif entropy_level_high > 0.5:
                if psnr_level_low > 0.5 and ssim_level_low > 0.5:
                    aggregated = np.fmin(0.5, self.quality_poor)
                else:
                    aggregated = np.fmin(0.5, self.quality_fair)
            else:
                aggregated = np.fmin(0.5, self.quality_fair)
        crisp_quality = fuzz.defuzz(self.quality_universe, aggregated, 'centroid')
        poor_membership = fuzz.interp_membership(self.quality_universe, self.quality_poor, crisp_quality)
        fair_membership = fuzz.interp_membership(self.quality_universe, self.quality_fair, crisp_quality)
        good_membership = fuzz.interp_membership(self.quality_universe, self.quality_good, crisp_quality)
        excellent_membership = fuzz.interp_membership(self.quality_universe, self.quality_excellent, crisp_quality)
        memberships = {'Poor': poor_membership, 'Fair': fair_membership, 'Good': good_membership, 'Excellent': excellent_membership}
        linguistic_label = max(memberships, key=memberships.get)
        return (crisp_quality, linguistic_label)
