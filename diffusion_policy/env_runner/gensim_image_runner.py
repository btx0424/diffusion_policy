import torch
import os
import numpy as np
import imageio
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
        import cliport
        import os
        import sys
        from cliport import tasks
        from cliport.environments.environment import Environment

        sys.path.append("/home/btx0424/gensim_ws/dreamerv3-torch")
        from dataset import GensimEnvironment

        self.env = GensimEnvironment(
            os.path.join(os.environ["GENSIM_ROOT"], "cliport", "environments", "assets"),
            disp=False,
            shared_memory=False,
            hz=240,
            record_cfg={"save_video": False}
        )
        task = "packing-boxes-pairs-seen-colors"
        self.task: tasks.Task = tasks.names[task]()
        self.task.mode = "train"

        self.agent = self.task.oracle(self.env)
        self.transform = Compose([ToTensor(), Resize((240, 180))])
        self.meas_score = 0

    @torch.no_grad()
    def run(self, policy: BaseImagePolicy) -> Dict:
        # self.meas_score += 1
        # return {"test_mean_score": self.meas_score}
        np.random.seed(0)
        self.env.set_task(self.task)
        obs, info = self.env.reset()
        
        total_reward = 0
        for _ in range(self.task.max_steps):
            obs = self._process_obs(obs)
            act = policy.predict_action(obs)
            act = self._process_act(act)
            act_oracle = self.agent.act(obs, info)
            print(act)
            print(act_oracle)
            obs, reward, done, info = self.env.step(act)
            lang_goal = info['lang_goal']
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break
            imageio.imsave(f"{_:02}.jpg", obs["color"][0])
            imageio.imsave(f"{_:02}_.jpg", self.env._obs["rgb"][-1])

        episode_data, ims = self.env.get_episode_data(rgb_array=True)
        gif_path = "test.gif"
        imageio.mimsave(gif_path, ims, format="gif")
        return {"test_mean_score": total_reward}
    
    # @torch.no_grad()
    # def run(self, policy: BaseImagePolicy) -> Dict:
    #     path = "/home/btx0424/gensim_ws/GenSim/data/train/packing-boxes-pairs-seen-colors/episode_0_2009.pt"
    #     data = torch.load(path)
    #     idx = torch.from_numpy(data["high_action_mask"]).nonzero().squeeze()

    #     for i in idx:
    #         image = torch.from_numpy(data["image"][i])[..., :3].permute(2, 0, 1)
    #         state = torch.from_numpy(data["state"][i])
    #         action = data["high_action"][i]
    #         print(image.shape)
    #         image = image / 255
    #         action_pred = policy.predict_action({
    #             "image": image.reshape(1, 1, *image.shape),
    #             "state": state.reshape(1, 1, *state.shape)
    #         })["action"]
    #         action_pred = action_pred[0, 0].cpu().numpy()
            
    #         print(action.shape, action_pred.shape)
    #         print(np.linalg.norm(action - action_pred))

    #     return {"test_mean_score": 0}
    
    # @torch.no_grad()
    # def run(self, policy: BaseImagePolicy) -> Dict:
    #     from ..dataset.gensim_image_dataset import GensimImageDataset
    #     dataset = GensimImageDataset()

    #     for i in range(len(dataset)):
    #         data = dataset[i]
    #         print(data)
    #         obs = data["obs"]
    #         print(obs["image"].shape)
    #         obs["image"] = obs["image"].unsqueeze(0)
    #         obs["state"] = obs["state"].unsqueeze(0)
    #         action = data["action"]
    #         action_pred = policy.predict_action(obs)["action"].cpu()

    #         print(torch.norm(action - action_pred))
    #         break

    #     return {"test_mean_score": 0}

    @torch.no_grad()
    def run(self, policy: BaseImagePolicy) -> Dict:
        root_path = "/home/btx0424/gensim_ws/GenSim/data/train/packing-boxes-pairs-seen-colors"
        total_rewards = []
        data_paths = []
        
        for file_name in os.listdir(root_path):
            if not (file_name.startswith("episode") and file_name.endswith(".pt")):
                continue
            file_path = os.path.join(root_path, file_name)
            data_paths.append(file_path)

        for file_name in sorted(data_paths):
            if not (file_name.startswith("episode") and file_name.endswith(".pt")):
                continue
            file_path = os.path.join(root_path, file_name)
            data = torch.load(file_path)

            self.env.set_task(self.task)
            seed = data["seed"]
            obs, info = self.env.reset(seed)
            total_reward = 0
            ims = []
            for i in range(self.task.max_steps):
                act = data["action_high"]
                act = {
                    "pose0": np.split(act["pose0"][i], [3]),
                    "pose1": np.split(act["pose1"][i], [3])
                }
                obs, reward, done, info = self.env.step(act)
                total_reward += reward
                print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {info["lang_goal"]}')
                ims.append(obs["color"][0])
                if done:
                    break
            ims = np.stack(ims)
            imageio.mimsave(file_name.replace(".pt", ".gif"), ims, format="gif")
            total_rewards.append(total_reward)

        return {"test_mean_score": np.mean(total_rewards)}
    
    def _process_obs(self, obs: dict):
        state = torch.as_tensor(self.env._obs["state"][-1])
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
        print(action)
        pose0, pose1 = np.split(action[0, 0], 2)
        return {
            "pose0": (pose0[:3], normalize(pose0[3:])),
            "pose1": (pose1[:3], normalize(pose1[3:])),
        }

def normalize(q):
    return q / np.linalg.norm(q).clip(1e-6)