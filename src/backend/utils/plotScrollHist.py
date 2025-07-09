# Standard Library
import pdb
import json
import traceback
from pathlib import Path

# 3rd party libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
DIR_FILE        = Path(__file__).resolve().parent # <projectRoot>/src/backend/utils/
DIR_ROOT        = DIR_FILE.parent.parent.parent # <projectRoot>/src/backend/
DIR_EXPERIMENTS         = DIR_ROOT / '_experiments'
DIR_TMP                 = DIR_ROOT / '_tmp'
DIR_EXPERIMENTS_OUTPUTS = DIR_EXPERIMENTS / 'experiment-outputs'

SUFFIX_SLICE_SCROLL_JSON = '__slice-scroll.json'
KEY_PATH_MANUAL_BRUSH = 'manual_brush_path'
KEY_PATH_AI_PENCIL    = 'ai_pencil_path'

PATH_SAVE_EXPERTS    = DIR_TMP / 'scrolls_interaction_experts.json'
PATH_SAVE_NONEXPERTS = DIR_TMP / 'scrolls_interaction_nonexperts.json'
Path(PATH_SAVE_EXPERTS.parent).mkdir(parents=True, exist_ok=True)

KEY_AXIAL   = 'Axial'
KEY_CORONAL = 'Coronal'
KEY_SAGITTAL= 'Sagittal'

COLOR_MANUAL_DICE = 'blue'
COLOR_SEMIAUTO_DICE = 'orange'

VERSION = 'v3' # ['v2', 'v3']
DO_EXPERTS = False
DO_NONEXPERTS = True
print (f'\n - [{VERSION}] Do experts: {DO_EXPERTS} | Do non-experts: {DO_NONEXPERTS}\n')


TICK_FONTSIZE = 26
AXESLABEL_FONTSIZE = 30

def openJSONAndGetCount(pathJSON):

    res, resScrollCount = {},-1

    try:
        with open(pathJSON, 'r') as f:
            res = json.load(f)
            resScrollCount = len(res[KEY_AXIAL]) + len(res[KEY_CORONAL]) + len(res[KEY_SAGITTAL])

    except:
        traceback.print_exc()
    
    return res, resScrollCount

if __name__ == "__main__":
    
    
    try:
        
        ########################################
        # Step 0 - Define paths
        ########################################
        res_patients = {}
        if 1:
            # Patient 1 (P1(CHUP-033)-Experts)
            obj_experts_patient1 = {
                'C1-Martin': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-03 14-40-44 -- pedantic_lehmann__Martin-De Jong-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-12 14-50-57 -- brave_diffie__Martin-De Jong-Expert-AI-based',
                },
                'C1-Mischa': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-06 14-31-12 -- reverent_taussig__Mischa-de Ridder-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-03 08-12-22 -- jolly_hopper__Mischa-de Ridder-Expert-AI-based',
                },
                'C1-Niels': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-12 13-54-50 -- exciting_poincare__Niels-den Haan-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-09 12-15-16 -- modest_lamarr__Niels-den Haan-Expert-AI-based',
                },
                'C1-Jos': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-06 08-47-00 -- jovial_vaughan__Jos-Elbers-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-05 08-28-22 -- fervent_lederberg__Jos-Elbers-Expert-AI-based',
                }
            }

            # Patient 2 (P2(CHUP-059)-Experts)
            obj_experts_patient2 = {
                'C1-Martin': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-11-29 09-34-51 -- magical_mahavira__Martin-De Jong-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-12 15-08-33 -- boring_wilson__Martin-De Jong-Expert-AI-based',
                },
                'C1-Mischa': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-03 08-20-15 -- recursing_shaw__Mischa-de Ridder-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-06 14-25-58 -- gallant_galileo__Mischa-de Ridder-Expert-AI-based',
                },
                'C1-Niels': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-09 12-22-15 -- awesome_shamir__Niels-den Haan-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-12 14-04-32 -- tender_wilbur__Niels-den Haan-Expert-AI-based',
                },
                'C1-Jos': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-05 08-33-14 -- happy_elgamal__Jos-Elbers-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-06 08-53-44 -- clever_bartik__Jos-Elbers-Expert-AI-based',
                }
            }

            # Patient 3 (P3(CHUP-005)-Experts)
            obj_experts_patient3 = {
                'C1-Martin': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-05 13-00-58 -- tender_hugle__Martin-De Jong-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-11-29 09-43-36 -- distracted_feistel__Martin-De Jong-Expert-AI-based',
                },
                'C1-Mischa': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-09 15-03-46 -- dreamy_ritchie__Mischa-de ridder-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-18 15-38-23 -- silly_goodall__Mischa-de Ridder-Expert-AI-based',
                },
                'C1-Niels': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-16 12-00-19 -- great_jennings__Niels -den Haan-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-09 12-26-54 -- crazy_cohen__Niels-den Haan-Expert-AI-based',
                },
                'C1-Jos': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-13 09-05-17 -- gracious_ride__Jos-Elbers-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-05 08-36-55 -- epic_leavitt__Jos-Elbers-Expert-AI-based',
                }
            }

            # Patient 4 (P4(CHUP-064)-Experts)
            obj_experts_patient4 = {
                'C1-Martin': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-11-29 09-57-03 -- fervent_hopper__Martin-De Jong-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-05 13-06-46 -- nervous_taussig__Martin-De Jong-Expert-AI-based',
                },
                'C1-Mischa': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-03 08-41-05 -- gifted_meitner__Mischa-de Ridder-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-09 15-11-38 -- gallant_hellman__Mischa-de Ridder-Expert-AI-based',
                },
                'C1-Niels': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-09 12-31-40 -- frosty_hertz__Niels-den Haan-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-16 12-05-24 -- hopeful_sinoussi__Niels-den Haan-Expert-AI-based',
                },
                'C1-Jos': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-05 08-42-52 -- epic_cartwright__Jos-Elbers-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-13 09-09-11 -- practical_davinci__Jos-Elbers-Expert-AI-based',
                }
            }

            # Patient 5 (P5(CHUP-028)-Experts)
            obj_experts_patient5 = {
                'C1-Martin': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-05 13-12-15 -- zealous_rhodes__Martin-De Jong-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-03 14-10-13 -- objective_austin__Martin-De Jong-Expert-AI-based',
                },
                'C1-Mischa': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-09 15-16-47 -- elastic_thompson__Mischa-de Ridder-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-06 14-03-32 -- gifted_chaum__Mischa-de Ridder-Expert-AI-based',
                },
                'C1-Niels': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-16 12-08-22 -- great_noether__Niels-den Haan-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-12 13-43-11 -- nostalgic_payne__Niels-den Haan-Expert-AI-based',
                },
                'C1-Jos': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-13 09-12-00 -- mystifying_shamir__Jos-Elbers-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-06 08-35-46 -- focused_austin__Jos-Elbers-Expert-AI-based',
                }
            }

            # Patient 6 (P6(CHUP-044)-Experts)
            obj_experts_patient6 = {
                'C1-Martin': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-03 14-15-38 -- compassionate_nightingale__Martin-De Jong-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-05 13-22-27 -- kind_curie__Martin-De Jong-Expert-AI-based',
                },
                'C1-Mischa': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-06 14-08-30 -- confident_lamarr__Mischa-de Ridder-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-09 15-25-59 -- vibrant_lumiere__Mischa-de Ridder-Expert-AI-based',
                },
                'C1-Niels': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-12 13-46-01 -- musing_merkle__Niels-den haan-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-16 12-15-50 -- lucid_fermi__Niels-den Haan-Expert-AI-based',
                },
                'C1-Jos': {
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-06 08-38-29 -- intelligent_sutherland__Jos-Elbers-Expert-Manual',
                    KEY_PATH_AI_PENCIL:     DIR_EXPERIMENTS / '2024-12-13 09-18-35 -- festive_allen__Jos-Elbers-Expert-AI-based',
                }
            }

            # Patient 1 (P1(CHUP-033)-NonExperts)
            obj_nonexperts_patient1 = {
                'U1-Yauheniya':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 10-03-30 -- loving_mestorf__Yauheniya-Makarevich-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-11-29 10-13-51 -- elated_ride__Yauheniya-Makarevich-NonExpert-AI-based',
                },
                'U2-Faeze':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 12-40-37 -- quirky_allen__Faeze-Gholamiankhah-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-04 12-06-33 -- jovial_tu__Faeze-Gholamiankhah-NonExpert-AI-based',
                },
                'U3-Frank':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-03 15-52-24 -- frosty_khorana__Frank-Dankers-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-11-28 13-17-37 -- vibrant_banach__Frank-Dankers-NonExpert-AI-based',
                },
                'U4-Alex':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 09-27-26 -- goofy_sutherland__Alex-Vieth-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-02 09-06-36 -- nostalgic_sammet__Alex-Vieth-NonExpert-AI-based',
                },
                'U5-Patrick':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 13-27-34 -- cranky_kowalevski__patrick-de Koning-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-11-29 15-03-16 -- affectionate_galois__Patrick-de Koning-NonExpert-AI-based',
                },
                'U6-Chinmay':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 16-04-54 -- vigorous_wescoff__Chinmay-Rao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-11-28 12-05-53 -- bold_villani__Chinmay-Rao-NonExpert-AI-based',
                },
                'U7-Ruochen':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-06 11-49-19 -- angry_tesla__Ruochen-Gao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-04 14-26-23 -- exciting_pasteur__Ruochen-Gao-NonExpert-AI-based',
                }
            }

            # Patient 2 (P2(CHUP-059)-NonExperts)
            obj_nonexperts_patient2 = {
                'U1-Yauheniya':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-11-29 10-29-31 -- magical_blackburn__Yauheniya-Makarevich-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-06 15-56-44 -- zen_driscoll__Yauheniya-Makarevich-NonExpert-AI-based',
                },
                'U2-Faeze':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 12-18-46 -- nifty_wu__Faeze-Gholamiankhah-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-05 12-37-15 -- blissful_hopper__Faeze-Gholamiankhah-NonExpert-AI-based',
                },
                'U3-Frank':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-11-28 13-25-01 -- nostalgic_swanson__Frank-Dankers-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-03 15-38-16 -- gifted_murdock__Frank-Dankers-NonExpert-AI-based',
                },
                'U4-Alex':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-02 09-15-07 -- wizardly_pike__Alex-Vieth-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-09 09-06-22 -- condescending_einstein__Alex-Vieth-NonExpert-AI-based',
                },
                'U5-Patrick':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-11-29 15-12-36 -- sweet_sinoussi__Patrick-de Koning-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-06 10-03-09 -- reverent_mirzakhani__Patrick-de Koning-NonExpert-AI-based',
                },
                'U6-Chinmay':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-11-28 12-15-42 -- nice_mccarthy__Chinmay-Rao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-04 16-20-34 -- eloquent_lalande__Chinmay-Rao-NonExpert-AI-based',
                },
                'U7-Ruochen':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 14-37-23 -- mystifying_albattani__Ruochen-Gao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-13 08-11-36 -- quirky_ishizaka__Ruochen-Gao-NonExpert-AI-based',
                }
            }

            # Patient 3 (P3(CHUP-005)-NonExperts)
            obj_nonexperts_patient3 = {
                'U1-Yauheniya':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-06 15-34-30 -- reverent_dhawan__Yauheniya-Makarevich-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-11-29 10-58-32 -- vibrant_villani__Yauheniya-Makarevich-NonExpert-AI-based',
                },
                'U2-Faeze':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-09 14-05-59 -- kind_perlman__Faeze-Gholamiankhah-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-04 12-32-33 -- elegant_turing__Faeze-Gholamiankhah-NonExpert-AI-based',
                },
                'U3-Frank':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-09 13-01-56 -- objective_almeida__Frank-Dankers-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-11-28 13-41-26 -- competent_haibt__Frank-Dankers-NonExpert-AI-based',
                },
                'U4-Alex':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-09 09-16-43 -- compassionate_clarke__Alex-Vieth-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-02 09-27-56 -- boring_bhabha__Alex-Vieth-NonExpert-AI-based',
                },
                'U5-Patrick':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-06 10-13-58 -- quizzical_blackburn__Patrick-de Koning-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-11-29 15-25-20 -- objective_bose__Patrick-de Koning-NonExpert-AI-based',
                },
                'U6-Chinmay':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 16-43-31 -- sweet_boyd__Chinmay-Rao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-11-28 12-33-57 -- vibrant_proskuriakova__Chinmay-Rao-NonExpert-AI-based',
                },
                'U7-Ruochen':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-13 08-23-33 -- elated_hugle__Ruochen-Gao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-04 14-49-01 -- gallant_gould__Ruochen-Gao-NonExpert-AI-based',
                }
            }

            # Patient 4 (P4(CHUP-064)-NonExperts)
            obj_nonexperts_patient4 = {
                'U1-Yauheniya':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 10-24-29 -- elated_sammet__Yauheniya-Makarevich-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-11 12-06-16 -- blissful_gagarin__Yauheniya-Makarevich-NonExpert-AI-based',
                },
                'U2-Faeze':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-05 12-03-21 -- heuristic_wing__Faeze-Gholamiankhah-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-09 14-19-43 -- infallible_galois__Faeze-Gholamiankhah-NonExpert-AI-based',
                },
                'U3-Frank':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-11-28 13-50-05 -- nifty_hodgkin__Frank-Dankers-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-09 13-16-41 -- vigorous_khayyam__Frank-Dankers-NonExpert-AI-based',
                },
                'U4-Alex':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 09-02-17 -- festive_nash__Alex-Vieth-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-09 09-27-30 -- exciting_newton__Alex-Vieth-NonExpert-AI-based',
                },
                'U5-Patrick':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 13-00-30 -- upbeat_bell__Patrick-de Koning-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-06 10-24-44 -- quirky_cohen__Patrick-de Koning-NonExpert-AI-based',
                },
                'U6-Chinmay':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-02 15-30-03 -- romantic_galileo__Chinmay-Rao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-10 16-05-10 -- cranky_jackson__Chinmay-Rao-NonExpert-AI-based',
                },
                'U7-Ruochen':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 14-57-34 -- jovial_carver__Ruochen-Gao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-13 08-36-43 -- relaxed_cori__Ruochen-Gao-NonExpert-AI-based',
                }
            }

            # Patient 5 (P5(CHUP-028)-NonExperts)
            obj_nonexperts_patient5 = {
                'U1-Yauheniya':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-11 12-16-05 -- affectionate_wilson__Yauheniya-Makarevich-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-04 10-39-05 -- jolly_cray__Yuaheniya-Makarevich-NonExpert-AI-based',
                },
                'U2-Faeze':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-09 14-26-21 -- determined_rubin__Faeze-Gholamiankhah-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-05 12-13-17 -- strange_bardeen__Faeze-Gholamiankhah-NonExpert-AI-based',
                },
                'U3-Frank':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-09 13-22-54 -- admiring_darwin__Frank-Dankers-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-03 15-10-43 -- gracious_elbakyan__Frank-Dankers-NonExpert-AI-based',
                },
                'U4-Alex':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-09 09-32-40 -- objective_bell__Alex-Vieth-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-04 09-10-37 -- practical_mestorf__Alex-Vieth-NonExpert-AI-based',
                },
                'U5-Patrick':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-06 10-31-58 -- wonderful_galileo__Patrick-de Koning-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-04 13-11-40 -- exciting_spence__Patrick-de Koning-NonExpert-AI-based',
                },
                'U6-Chinmay':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-10 16-13-52 -- jolly_goldstine__Chinmay-Rao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-02 15-40-11 -- quirky_blackburn__Chinmay-Rao-NonExpert-AI-based',
                },
                'U7-Ruochen':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-13 08-45-25 -- frosty_cartwright__Ruochen-Gao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-06 11-25-53 -- beautiful_joliot__Ruochen-Gao-NonExpert-AI-based',
                }
            }

            # Patient 6 (P6(CHUP-044)-NonExperts)
            obj_nonexperts_patient6 = {
                'U1-Yauheniya':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 10-51-32 -- hungry_archimedes__Yauheniya-Makarevich-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-11 12-26-05 -- great_ellis__Yauheniya-Makarevich-NonExpert-AI-based',
                },
                'U2-Faeze':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-05 12-19-19 -- relaxed_proskuriakova__Faeze-Gholamiankhah-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-09 14-34-13 -- gallant_hoover__Faeze-Gholamiankhah-NonExpert-AI-based',
                },
                'U3-Frank':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-03 15-15-34 -- nice_greider__Frank-Dankers-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-09 13-30-58 -- amazing_lehmann__Frank-Dankers-NonExpert-AI-based',
                },
                'U4-Alex':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 09-15-03 -- focused_fermat__Alex-Vieth-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-09 09-39-20 -- vibrant_carver__Alex-Vieth-NonExpert-AI-based',
                },
                'U5-Patrick':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-04 13-16-27 -- hopeful_kilby__Patrick-de Koning-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-06 10-40-13 -- musing_franklin__Patrick-de Koning-NonExpert-AI-based',
                },
                'U6-Chinmay':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-02 15-45-46 -- jolly_mestorf__Chinmay-Rao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-10 16-23-57 -- xenodochial_cori__Chinmay-Rao-NonExpert-AI-based',
                },
                'U7-Ruochen':{
                    KEY_PATH_MANUAL_BRUSH: DIR_EXPERIMENTS / '2024-12-06 11-33-17 -- nice_kepler__Ruochen-Gao-NonExpert-Manual',
                    KEY_PATH_AI_PENCIL:    DIR_EXPERIMENTS / '2024-12-13 08-55-56 -- optimistic_kare__Ruochen-Gao-NonExpert-AI-based',
                }
            }

            if DO_EXPERTS:
                obj_patients = [obj_experts_patient1, obj_experts_patient2, obj_experts_patient3, obj_experts_patient4, obj_experts_patient5, obj_experts_patient6]
                PATH_SAVE = PATH_SAVE_EXPERTS
            elif DO_NONEXPERTS:
                obj_patients = [obj_nonexperts_patient1, obj_nonexperts_patient2, obj_nonexperts_patient3, obj_nonexperts_patient4, obj_nonexperts_patient5, obj_nonexperts_patient6]
                PATH_SAVE = PATH_SAVE_NONEXPERTS

            patient_names = ['P1(CHUP-033)', 'P2(CHUP-059)', 'P3(CHUP-005)', 'P4(CHUP-064)', 'P5(CHUP-028)', 'P6(CHUP-044)']
            patient_ids = ['CHUP-033-gt-filtered-gausssig2', 'CHUP-059-gt-filtered-gausssig2', 'CHUP-005-gt-filtered-gausssig2',
                           'CHUP-064-gt-filtered-gausssig2', 'CHUP-028-gt-filtered-gausssig2', 'CHUP-044-gt-filtered-gausssig2']
        
        ########################################
        # Step 1 - Make PATH_SAVE
        ########################################
        # Step 1.1 - Loop over patient objs
        if not PATH_SAVE.exists():
            for patientIdx, obj_patient in enumerate(obj_patients):
                patient_name = patient_names[patientIdx]
                res_patients[patient_name] = {}
                for clinician_name in obj_patient:
                    res_patients[patient_name][clinician_name] = {
                            KEY_PATH_MANUAL_BRUSH: [],
                            KEY_PATH_AI_PENCIL: []
                        }
                    for tool_type, tool_path in obj_patient[clinician_name].items():
                        print (f'\n - [patient={patient_name}, Clinician: {clinician_name}] Tool: {tool_type}]')
                        tool_scroll_path = tool_path / f'{patient_ids[patientIdx]}{SUFFIX_SLICE_SCROLL_JSON}'
                        
                        if tool_type == KEY_PATH_MANUAL_BRUSH:
                            
                            manual_scrollcount = openJSONAndGetCount(tool_scroll_path)[-1]
                            res_patients[patient_name][clinician_name][KEY_PATH_MANUAL_BRUSH] = manual_scrollcount
                            print (f'\n   - Manual brush path: {tool_path}')
                            print (f'   - Manual brush scroll count: {manual_scrollcount}')
                            
                        if tool_type == KEY_PATH_AI_PENCIL:
                            aiscrollcount = openJSONAndGetCount(tool_scroll_path)[-1]
                            res_patients[patient_name][clinician_name][KEY_PATH_AI_PENCIL] = aiscrollcount
                            print (f'\n   - AI pencil path: {tool_path}')
                            print (f'   - AI pencil scroll count: {aiscrollcount}')
            
            # Step 2 - Save the results
            with open(PATH_SAVE, 'w') as f:
                json.dump(res_patients, f, indent=4)


        ########################################
        # Step 2 - Plotting
        ########################################
        with open(PATH_SAVE, 'r') as f:
            res_patients = json.load(f)

            for patientId, patient_name in enumerate(res_patients):
                print (f'\n - Patient: {patientId} - {patient_name}')
                manual_vals, ai_vals = [], []
                for clinicianId, clinician_name in enumerate(res_patients[patient_name]):
                    manual_vals_clinician = res_patients[patient_name][clinician_name][KEY_PATH_MANUAL_BRUSH]
                    ai_vals_clinician = res_patients[patient_name][clinician_name][KEY_PATH_AI_PENCIL]
                    diff = manual_vals_clinician - ai_vals_clinician
                    savings_str = f'(Scrolls) AI:{ai_vals_clinician} vs Manual: {manual_vals_clinician} (=> {diff}, {((diff)/manual_vals_clinician*100):03f} %)'
                    print (f'   - Clinician: {clinician_name} - {savings_str}')
    except:
        traceback.print_exc()
        pdb.set_trace()