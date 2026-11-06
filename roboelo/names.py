"""Display names of policies (the rating data uses RoboDojo's ids)."""

NAMES = {"Pi_0": "π0", "Pi_05": "π0.5", "Pi_05_SF": "π0.5-SF", "Pi0_FAST": "π0-FAST", "X_VLA": "X-VLA",
         "GR00T_N17": "GR00T N1.7", "GR00T_N16": "GR00T N1.6", "MolmoACT2": "MolmoAct2", "G05": "G0.5",
         "OpenVLA_OFT": "OpenVLA-OFT", "RDT_1B": "RDT-1B", "Xiaomi_Robotics_1": "Xiaomi-R1",
         "Xiaomi_Robotics_0": "Xiaomi-R0", "LingBot_VA": "LingBot-VA", "LingBot_VLA": "LingBot-VLA",
         "VLA_JEPA": "VLA-JEPA", "VLA_Adapter": "VLA-Adapter", "ME_Dex_1_0": "ME-Dex-1.0",
         "GigaBrain_0_7": "GigaBrain-0.7", "WorldScape_Policy_2_0": "WorldScape-2.0", "Octo_Small": "Octo-Small",
         "OpenDM05": "OpenDM0.5", "4D_WAM": "4D-WAM", "Evo1": "Evo-1", "InternVLA_A1_5": "InternVLA-A1.5",
         "SimpleMemVLA": "SimpleMemVLA", "Abot_M0": "ABot-M0"}


def display_name(policy):
    return NAMES.get(policy, policy.replace("_", "-"))
