# Generated from robosyn_cobotmagic_semantics.json; do not hand-edit.
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

robosyn_cobotmagic_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=['front', 'left_wrist', 'right_wrist']),
    "state": ModalityConfig(delta_indices=[0], modality_keys=['left_arm', 'left_gripper', 'right_arm', 'right_gripper']),
    "action": ModalityConfig(
        delta_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        modality_keys=['left_arm', 'left_gripper', 'right_arm', 'right_gripper'],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT, state_key='left_arm'),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT, state_key='right_arm'),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=['annotation.human.task_description']),
}

register_modality_config(robosyn_cobotmagic_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
