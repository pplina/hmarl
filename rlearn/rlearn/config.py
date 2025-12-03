import os
import confuse
from pydantic import BaseModel


def get_config(filepath: str):
    config = confuse.Configuration("rlearn")
    config.set_file(filepath, base_for_paths=True)
    return config


class DDQN_Config(BaseModel):
    beta_start: float
    beta_frames: int
    gamma_start: float
    gamma_final: float
    epsilon_start: float
    epsilon_final: float
    epsilon_decay: float

    lenBuffer: int    #buffer_size
    num_steps: int
    batchsize: int    #batch_size

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @classmethod
    def parse_config(cls):
        #config = get_config("/home/ubuntu/src/rl-test/rlearn/ddqn_config.yaml")
        config = get_config( (os.getcwd() + '/rlearn/ddqn_config.yaml') )

        return cls.parse_obj(config.flatten(redact=False))
