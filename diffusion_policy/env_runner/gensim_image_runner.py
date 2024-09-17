import torch
import os
import numpy as np
import imageio
import sys
from termcolor import colored
from typing import Dict
from torchvision.transforms import Compose, ToTensor, Resize

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class GensimImageRunner(BaseImageRunner):
    def __init__(
        self,
        task,
        output_dir
    ):
        super().__init__(output_dir)

        self.transform = Compose([ToTensor(), Resize((240, 180))])
        self.mean_score = 0
        self.eval = False

    def set_task(self, task: str):
        import cliport
        from cliport import tasks
        from gen_diversity.dataset import GenSimEnvironment, GENSIM_ROOT
        
        print(f"Create environment with task: {task}")
        
        self.env = GenSimEnvironment(
            os.path.join(os.environ["GENSIM_ROOT"], "cliport", "environments", "assets"),
            disp=False,
            shared_memory=False,
            hz=240,
            record_cfg={"save_video": False}
        )
        self.task: tasks.Task = tasks.names[task]()
        self.task.mode = "train"
        self.agent = self.task.oracle(self.env)

    @torch.no_grad()
    def run(self, policy: BaseImagePolicy) -> Dict:
        if not self.eval:
            self.mean_score += 1
            return {"test_mean_score": self.mean_score}
        else:
            rewards = []
            for i, seed in enumerate([0, 1, 2, 4, 17, 24, 44, 68, 100, 127]):
                self.env.set_task(self.task)
                obs, info = self.env.reset(seed=seed)

                total_reward = 0
                for _ in range(self.task.max_steps):
                    obs = self._process_obs(obs)
                    act = policy.predict_action(obs)
                    act = self._process_act(act)
                    # act_oracle = self.agent.act(obs, info)
                    # print(act)
                    # print(act_oracle)
                    obs, reward, done, info = self.env.step(act)
                    lang_goal = info['lang_goal']
                    total_reward += reward
                    print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
                    if done:
                        break
                    # imageio.imsave(f"{seed}_{_:02}.jpg", obs["color"][0])

                episode_data, ims = self.env.get_episode_data(rgb_array=True)
                gif_path = f"test_{seed}.gif"
                imageio.mimsave(gif_path, ims, format="gif")
                rewards.append(total_reward)

            print(colored(f"Mean score: {np.mean(rewards)}", "green"))
            print(colored(f"Std: {np.std(rewards)}", "green"))
            
            return {"test_mean_score": np.mean(rewards), "std": np.std(rewards)}
    
    def _process_obs(self, obs: dict):
        state = torch.as_tensor(self.env.jstate)
        color = obs["color"][0]
        print(color.shape)
        # depth = obs["depth"][0]
        # image = self.transform(np.concatenate([color, depth.reshape(*color.shape[:2], 1)], axis=2))
        image = torch.as_tensor(color / 255).permute(2, 0, 1)
        return {
            "image": image.reshape(1, 1, *image.shape),
            "state": state.reshape(1, 1, *state.shape)
        }
    
    def _process_act(self, act: dict):
        action = act["action"]
        if not isinstance(action, np.ndarray):
            action = action.cpu().numpy()
        pose0, pose1 = np.split(action[0, 0], 2)
        return {
            "pose0": (pose0[:3], normalize(pose0[3:])),
            "pose1": (pose1[:3], normalize(pose1[3:])),
        }

def normalize(q):
    return q / np.linalg.norm(q).clip(1e-6)