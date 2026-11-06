"""Policy factories for run_simpler_env.py: factory(policy_setup, **kwargs) -> SimplerEnv policy object.

policy_setup is "google_robot" or "widowx_bridge". Each factory builds the policy exactly as the official
SimplerEnv (or model) evaluation entry point does, and is imported only in that model's environment.
"""
import os
import sys

ROOT = os.path.expanduser("~/roboelo")


def _cpu_only_tf_jax():
    """TensorFlow 2.15 and jax 0.4.20 have no RTX 5090 kernels: run them on the CPU (SAPIEN renders on the GPU)."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")


FORK = os.path.join(ROOT, "third_party", "SimplerEnv-OpenVLA")  # DelinQu/SimplerEnv-OpenVLA (policy adapters only)


def _load_fork_module(relative_path, name):
    """Import one adapter file of the fork by path, so that `simpler_env` stays the official SimplerEnv package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, os.path.join(FORK, relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def openvla(policy_setup, ckpt, action_scale=1.0):
    """OpenVLA: the fork's main_inference.py with --policy-model openvla (unnorm_key bridge_orig / fractal20220817_data)."""
    module = _load_fork_module("simpler_env/policies/openvla/openvla_model.py", "fork_openvla_model")
    return module.OpenVLAInference(saved_model_path=ckpt, policy_setup=policy_setup, action_scale=action_scale)


def cogact(policy_setup, ckpt, action_model_type, action_scale=1.0):
    """CogACT: the fork's main_inference.py with --policy-model cogact (cfg_scale 1.5). The fork hard-codes
    action_model_type DiT-L for CogACT-Large; CogACT-Base needs DiT-B (CogACT README)."""
    sys.path.insert(0, os.path.join(FORK, "simpler_env", "policies"))  # sim_cogact.adaptive_ensemble
    sys.path.insert(0, os.path.join(ROOT, "third_party", "CogACT"))  # vla.load_vla
    module = _load_fork_module("simpler_env/policies/sim_cogact/cogact_policy.py", "fork_cogact_policy")
    return module.CogACTInference(saved_model_path=ckpt, policy_setup=policy_setup, action_scale=action_scale,
                                  action_model_type=action_model_type, cfg_scale=1.5)


def spatialvla(policy_setup, ckpt, action_scale=1.0):
    """SpatialVLA: the fork's main_inference.py with --policy-model spatialvla."""
    module = _load_fork_module("simpler_env/policies/spatialvla/spatialvla_model.py", "fork_spatialvla_model")
    return module.SpatialVLAInference(saved_model_path=ckpt, policy_setup=policy_setup, action_scale=action_scale)


def univla(policy_setup, ckpt, action_scale=1.0, pred_action_horizon=10):
    """UniVLA (WidowX only): UniVLABridgeInference as experiments/robot/simpler-bridge/real2sim_eval_maniskill3.py
    builds it (action_scale 1, pred_action_horizon 10, decoder = <ckpt>/action_decoder.pt). UniVLA's own script runs
    the ManiSkill3 port of the Bridge tasks; here the policy runs on the official SimplerEnv (ManiSkill2) episodes."""
    assert policy_setup == "widowx_bridge", "the UniVLA SimplerEnv checkpoint is for WidowX (Bridge) only"
    import importlib.util

    repo = os.path.join(ROOT, "third_party", "UniVLA")
    sys.path.insert(0, repo)  # prismatic of UniVLA
    path = os.path.join(repo, "experiments", "robot", "simpler-bridge", "policies", "univla", "univla_model.py")
    spec = importlib.util.spec_from_file_location("univla_simpler_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UniVLABridgeInference(saved_model_path=ckpt, decoder_path=os.path.join(ckpt, "action_decoder.pt"),
                                        policy_setup=policy_setup, action_scale=action_scale,
                                        pred_action_horizon=pred_action_horizon)


def rt1(policy_setup, ckpt, action_scale=1.0):
    """RT-1 / RT-1-X: simpler_env/main_inference.py with --policy-model rt1."""
    _cpu_only_tf_jax()
    from simpler_env.policies.rt1.rt1_model import RT1Inference

    return RT1Inference(saved_model_path=ckpt, policy_setup=policy_setup, action_scale=action_scale)


def octo(policy_setup, model_type="octo-base", init_rng=0, action_scale=1.0):
    """Octo: simpler_env/main_inference.py with --policy-model octo-base / octo-small (--octo-init-rng 0)."""
    _cpu_only_tf_jax()
    from simpler_env.policies.octo.octo_model import OctoInference

    return OctoInference(model_type=model_type, policy_setup=policy_setup, init_rng=init_rng,
                         action_scale=action_scale)


# ---------------------------------------------------------------------------------------------------------------
# Models whose code needs Python >= 3.12 (GR00T, LeRobot ports) run in their own policy server, started by
# run_simpler_supervisor.sh (SERVER_CMD); the adapters below are the clients. A client waits for its server (it may
# be loading or restarting) instead of failing the episode.
# ---------------------------------------------------------------------------------------------------------------

def _wait(what, ok, wait_s, every_s=5.0):
    import time

    t0 = time.time()
    while not ok():
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"{what} did not answer within {wait_s:.0f} s")
        time.sleep(every_s)


class _Gr00tWire:
    """The wire format of GR00T's PolicyServer (gr00t/policy/server_client.py, N1.6 and N1.7): msgpack, arrays as
    {"__ndarray_class__", "as_npy"} (N1.6 sends and both accept this) or msgpack_numpy's {"nd", "type", "shape",
    "data"} (N1.7 sends this); a ModalityConfig arrives as its JSON form."""

    @staticmethod
    def pack(obj):
        import io

        import msgpack
        import numpy as np

        def default(o):
            if isinstance(o, np.ndarray):
                buf = io.BytesIO()
                np.save(buf, o, allow_pickle=False)
                return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError(f"cannot send {type(o)}")

        return msgpack.packb(obj, default=default)

    @staticmethod
    def unpack(data):
        import io
        import json

        import msgpack
        import numpy as np

        def hook(o):
            # msgpack_numpy writes its keys as bytes (b"nd", b"data" ...), GR00T's own markers as str
            get = lambda k, d=None: o.get(k, o.get(k.encode(), d))  # noqa: E731
            if get("__ndarray_class__"):
                return np.load(io.BytesIO(get("as_npy")), allow_pickle=False)
            if get("nd") is not None and get("type") is not None and get("data") is not None:
                t = get("type")
                dtype = np.dtype(t if isinstance(t, str) else t.decode())
                if get("nd"):
                    return np.frombuffer(get("data"), dtype=dtype).reshape(get("shape"))
                return np.frombuffer(get("data"), dtype=dtype)[0]
            for marker in ("__ModalityConfig__", "__ModalityConfig_class__"):
                if get(marker):
                    cfg = get("as_json")
                    return json.loads(cfg) if isinstance(cfg, (str, bytes)) else cfg
            return o

        return msgpack.unpackb(data, object_hook=hook, raw=False)


class Gr00tSimplerPolicy:
    """GR00T N1.6 / N1.7 on the official SimplerEnv episodes; the policy runs in GR00T's own server
    (gr00t/eval/run_gr00t_server.py --use-sim-policy-wrapper).

    Reproduces what GR00T's evaluation does on the client side (gr00t/eval/sim/SimplerEnv/simpler_env.py, the
    MultiStepWrapper and rollout_policy.py): the image resized to 256x320 (Google) / 256x256 (WidowX); the state from
    the end-effector pose in the robot base frame plus gripper width, as the `eef_pos` that GR00T's SimplerEnv fork adds
    to the agent observation (computed here from the same links, since the official SimplerEnv has no `eef_pos`);
    video / state history by the policy's delta indices; the first n_action_steps actions of each predicted chunk
    executed before the next query (README: 1 for Google, 4 for WidowX); rotation sent to the env as GR00T sends it;
    gripper: Google sticky relative action (15 repeats), WidowX binarized. What stays official SimplerEnv: episodes,
    step limits, instructions, success = `done` at the last step (GR00T's own client stops at the first success and
    allows 300 steps)."""

    needs_obs = True

    def __init__(self, policy_setup, port, n_action_steps, host="127.0.0.1", timeout_ms=300000, wait_s=3600,
                 instructions=None):
        import zmq

        # Own-protocol comparison runs only: GR00T's SimplerEnv fork words one instruction differently
        # ("put the eggplant in the yellow basket"); the uniform runs keep SimplerEnv's own text.
        self.instructions = dict(instructions or {})
        self.google = policy_setup == "google_robot"
        self.image_key = "video.image" if self.google else "video.image_0"
        self.image_hw = (256, 320) if self.google else (256, 256)
        self.ee_link_name = "link_gripper_tcp" if self.google else "ee_gripper_link"
        self.addr, self.timeout_ms = f"tcp://{host}:{port}", timeout_ms
        self.ctx = zmq.Context()
        self._open()
        _wait(f"GR00T server {self.addr}", self._ping, wait_s)
        cfg = self._call("get_modality_config", None, requires_input=False)
        self.video_delta = [int(d) for d in cfg["video"]["delta_indices"]]
        state = cfg.get("state")
        self.state_delta = [int(d) for d in state["delta_indices"]] if state and state.get("delta_indices") else [0]
        horizon = len(cfg["action"]["delta_indices"])
        assert 1 <= n_action_steps <= horizon, (n_action_steps, horizon)
        self.n_action_steps = int(n_action_steps)
        span = lambda d: max(d) - min(d) + 1  # noqa: E731
        self.history_len = max(span(self.video_delta), span(self.state_delta)) + 1
        self.n_infer = 0
        print(f"[gr00t] {self.addr}: video {self.video_delta}, state {self.state_delta}, action horizon {horizon}, "
              f"executing {self.n_action_steps} per query", flush=True)

    # ---- transport ----
    def _open(self):
        import zmq

        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def _call(self, endpoint, data, requires_input=True):
        import zmq

        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            self.sock.send(_Gr00tWire.pack(request))
            message = self.sock.recv()
        except zmq.error.ZMQError:
            self.sock.close()
            self._open()  # a REQ socket that timed out cannot be used again
            raise
        if message == b"ERROR":
            raise RuntimeError(f"GR00T server error on {endpoint}")
        response = _Gr00tWire.unpack(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error on {endpoint}: {response['error']}")
        return response

    def _ping(self):
        try:
            self._call("ping", None, requires_input=False)
            return True
        except Exception:
            return False

    # ---- observation / action conversion (gr00t/eval/sim/SimplerEnv/simpler_env.py) ----
    def _ee_state(self, env):
        """End-effector position and wxyz quaternion in the robot base frame, and the gripper's normalized width."""
        import numpy as np
        from transforms3d import quaternions as tq

        agent = env.unwrapped.agent
        link = [x for x in agent.robot.get_links() if x.name == self.ee_link_name][0]
        ee_in_base = np.linalg.inv(agent.base_pose.to_transformation_matrix()) @ link.get_pose().to_transformation_matrix()
        return ee_in_base[:3, 3], tq.mat2quat(ee_in_base[:3, :3]), 1.0 - float(agent.get_gripper_closedness())

    def _observation(self, image, task_description, env):
        import cv2
        import numpy as np
        from transforms3d import euler as te, quaternions as tq

        pos, quat_wxyz, gripper_nwidth = self._ee_state(env)
        out = {self.image_key: cv2.resize(image, (self.image_hw[1], self.image_hw[0])),
               "state.x": [pos[0]], "state.y": [pos[1]], "state.z": [pos[2]],
               "annotation.human.action.task_description": self.instructions.get(task_description, task_description)}
        if self.google:
            q_xyzw = np.roll(quat_wxyz, -1)
            out.update({"state.rx": [q_xyzw[0]], "state.ry": [q_xyzw[1]], "state.rz": [q_xyzw[2]],
                        "state.rw": [q_xyzw[3]], "state.gripper": [1.0 - gripper_nwidth]})
        else:
            rpy = te.mat2euler(tq.quat2mat(quat_wxyz) @ np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]).T)
            out.update({"state.roll": [rpy[0]], "state.pitch": [rpy[1]], "state.yaw": [rpy[2]], "state.pad": [0.0],
                        "state.gripper": [gripper_nwidth]})
        return out

    def _batch(self):
        import numpy as np

        hist = list(self.history)
        batch = {}
        for key in hist[-1]:
            if key.startswith("video"):
                batch[key] = np.stack([hist[d - 1][key] for d in self.video_delta])[None].astype(np.uint8)
            elif key.startswith("state"):
                batch[key] = np.stack([np.asarray(hist[d - 1][key], dtype=np.float32) for d in self.state_delta])[None]
            else:
                batch[key] = [hist[-1][key]]  # a batch of one instruction, as gym's vector env passes it
        return batch

    def _gripper(self, g):
        import numpy as np

        g = float(np.asarray(g).reshape(-1)[0])
        if not self.google:
            return 2.0 * (g > 0.5) - 1.0
        relative = -((g * 2) - 1)  # [0, 1] -> [-1, 1], then inverted
        if abs(relative) > 0.5 and not self.sticky_on:
            self.sticky_on, self.sticky_action = True, relative
        if self.sticky_on:
            self.sticky_repeat += 1
            relative = self.sticky_action
        if self.sticky_repeat == 15:
            self.sticky_on, self.sticky_repeat, self.sticky_action = False, 0, 0.0
        return relative

    # ---- SimplerEnv policy interface ----
    def reset(self, task_description, obs=None, env=None):
        import collections

        _wait(f"GR00T server {self.addr}", self._ping, 3600)  # it may be restarting after a crash
        self._call("reset", {"options": None})
        first = self._observation(self._image(obs, env), task_description, env)
        self.history = collections.deque([first] * self.history_len, maxlen=self.history_len)
        self.queue, self.fresh = collections.deque(), True
        self.sticky_on, self.sticky_repeat, self.sticky_action = False, 0, 0.0

    @staticmethod
    def _image(obs, env):
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

        return get_image_from_maniskill2_obs_dict(env, obs)

    def step(self, image, task_description, obs=None, env=None):
        import numpy as np

        if not self.fresh:  # MultiStepWrapper appends every new observation
            self.history.append(self._observation(image, task_description, env))
        self.fresh = False
        if not self.queue:
            actions, _ = self._call("get_action", {"observation": self._batch(), "options": None})
            self.n_infer += 1
            for i in range(self.n_action_steps):
                self.queue.append({k: np.asarray(v)[0, i] for k, v in actions.items()})
        act = self.queue.popleft()
        vec = lambda k: float(np.asarray(act[k]).reshape(-1)[0])  # noqa: E731
        action = {"world_vector": np.array([vec("action.x"), vec("action.y"), vec("action.z")]),
                  "rot_axangle": np.array([vec("action.roll"), vec("action.pitch"), vec("action.yaw")]),
                  "gripper": np.array([self._gripper(act["action.gripper"])]),
                  "terminate_episode": np.array([0.0])}
        return act, action


def gr00t(policy_setup, port, n_action_steps, host="127.0.0.1", instructions=None):
    """GR00T N1.6 / N1.7 client (see Gr00tSimplerPolicy); the server is started by run_simpler_supervisor.sh."""
    return Gr00tSimplerPolicy(policy_setup, port=port, n_action_steps=n_action_steps, host=host,
                              instructions=instructions)


class Gr00tN15SimplerPolicy(Gr00tSimplerPolicy):
    """GR00T N1.5 on the official SimplerEnv episodes; the policy runs in Isaac-GR00T n1.5-release's
    scripts/inference_service.py --server (endpoints ping, get_action, get_modality_config; same msgpack wire format).

    Reproduces the client side of N1.5's own evaluation (eval_simpler.py of NVIDIA's SimplerEnv fork: GR00TPolicy,
    WrapSimplerEnv, GoogleSimplerActionWrapper, BridgeSimplerStateWrapper): the image resizing, state and gripper
    conversion of Gr00tSimplerPolicy; one observation without history; one query per step, executing the first action
    of the predicted chunk (eval_simpler.py's default --action_horizon 1); the instruction under
    "annotation.human.task_description". As in GoogleSimplerActionWrapper, which converts the state only in step(),
    the Google Robot state of the first query is the unconverted `eef_pos` (wxyz quaternion, gripper openness). What
    stays official SimplerEnv: episodes, step limits, instructions, success = `done` at the last step (eval_simpler.py
    stops at the first success and allows 120 steps)."""

    def __init__(self, policy_setup, port, host="127.0.0.1", timeout_ms=300000, wait_s=3600):
        import zmq

        self.google = policy_setup == "google_robot"
        self.image_key = "video.image" if self.google else "video.image_0"
        self.image_hw = (256, 320) if self.google else (256, 256)
        self.ee_link_name = "link_gripper_tcp" if self.google else "ee_gripper_link"
        self.addr, self.timeout_ms = f"tcp://{host}:{port}", timeout_ms
        self.ctx = zmq.Context()
        self._open()
        _wait(f"GR00T N1.5 server {self.addr}", self._ping, wait_s)
        cfg = self._call("get_modality_config", None, requires_input=False)
        self.n_infer = 0
        print(f"[gr00t_n15] {self.addr}: video {cfg['video']['delta_indices']}, action horizon "
              f"{len(cfg['action']['delta_indices'])}, executing 1 per query", flush=True)

    def _observation(self, image, task_description, env, first=False):
        import cv2
        import numpy as np
        from transforms3d import euler as te, quaternions as tq

        pos, quat_wxyz, gripper_nwidth = self._ee_state(env)
        state = {"x": pos[0], "y": pos[1], "z": pos[2]}
        if self.google:
            q, g = (quat_wxyz, gripper_nwidth) if first else (np.roll(quat_wxyz, -1), 1.0 - gripper_nwidth)
            state.update(rx=q[0], ry=q[1], rz=q[2], rw=q[3], gripper=g)
        else:
            rpy = te.mat2euler(tq.quat2mat(quat_wxyz) @ np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]).T)
            state.update(roll=rpy[0], pitch=rpy[1], yaw=rpy[2], gripper=gripper_nwidth, pad=0.0)
        out = {self.image_key: cv2.resize(image, (self.image_hw[1], self.image_hw[0]))[None]}  # (T=1, H, W, 3)
        out.update({f"state.{k}": np.array([[v]], dtype=np.float64) for k, v in state.items()})
        out["annotation.human.task_description"] = [task_description]
        return out

    def reset(self, task_description, obs=None, env=None):
        _wait(f"GR00T N1.5 server {self.addr}", self._ping, 3600)  # it may be restarting after a crash
        self.obs = self._observation(self._image(obs, env), task_description, env, first=True)
        self.fresh = True
        self.sticky_on, self.sticky_repeat, self.sticky_action = False, 0, 0.0

    def step(self, image, task_description, obs=None, env=None):
        import numpy as np

        if not self.fresh:
            self.obs = self._observation(image, task_description, env)
        self.fresh = False
        chunk = self._call("get_action", self.obs)
        self.n_infer += 1
        first = lambda k: float(np.atleast_1d(np.asarray(chunk[f"action.{k}"])[0])[0])  # noqa: E731
        action = {"world_vector": np.array([first("x"), first("y"), first("z")]),
                  "rot_axangle": np.array([first("roll"), first("pitch"), first("yaw")]),
                  "gripper": np.array([self._gripper(first("gripper"))]),
                  "terminate_episode": np.array([0.0])}
        return chunk, action


def gr00t_n15(policy_setup, port, host="127.0.0.1"):
    """GR00T N1.5 client (see Gr00tN15SimplerPolicy); the server is started by run_simpler_supervisor.sh."""
    return Gr00tN15SimplerPolicy(policy_setup, port=port, host=host)


class _LeRobotServerClient:
    """Client of scripts/lerobot_policy_server.py (multiprocessing.connection; arrays travel as .npy bytes)."""

    def __init__(self, port, host="127.0.0.1", wait_s=3600):
        self.addr = (host, int(port))
        self.conn = None
        _wait(f"LeRobot policy server {self.addr}", self._connect, wait_s)
        self.info = self.call({"cmd": "info"})

    def _connect(self):
        from multiprocessing.connection import Client

        try:
            self.conn = Client(self.addr, authkey=b"roboelo")
            return True
        except Exception:
            return False

    def call(self, request):
        try:
            self.conn.send(request)
            reply = self.conn.recv()
        except (EOFError, OSError, ConnectionError):
            _wait(f"LeRobot policy server {self.addr}", self._connect, 3600)  # restarted by the supervisor
            raise
        if "error" in reply:
            raise RuntimeError(f"policy server: {reply['error']}")
        return reply

    @staticmethod
    def npy(a):
        import io

        import numpy as np

        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
        return buf.getvalue()

    @staticmethod
    def from_npy(b):
        import io

        import numpy as np

        return np.load(io.BytesIO(b), allow_pickle=False)

    def chunk(self, images, task, state=None, extra=None, postprocess=True):
        request = {"cmd": "chunk", "images": {k: self.npy(v) for k, v in images.items()}, "task": task,
                   "state": None if state is None else self.npy(state), "extra": extra or {},
                   "postprocess": postprocess}
        return self.from_npy(self.call(request)["chunk"])


def _rotate6d_to_euler_xyz(v6):
    """X-VLA's evaluation/simpler clients: 6D rotation (first two matrix columns, interleaved) -> Euler xyz."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    v6 = np.asarray(v6, dtype=np.float64)
    a1, a2 = v6[0:5:2], v6[1:6:2]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    return Rotation.from_matrix(np.stack((b1, b2, np.cross(b1, b2)), axis=-1)).as_euler("xyz")


class XVLASimplerPolicy:
    """X-VLA (LeRobot ports lerobot/xvla-widowx, lerobot/xvla-google-robot) on the official SimplerEnv episodes.

    X-VLA predicts absolute end-effector poses, so it needs the controllers its SimplerEnv fork adds to
    ManiSkill2_real2sim (base-frame absolute pose, Euler xyz); the runner builds the env with `control_modes` and
    SIMPLER_ROOT must be the patched copy (third_party/SimplerEnv-absee, see setup_ladder_simpler_envs.sh). The client
    logic is X-VLA's evaluation/simpler clients: WidowX domain 0, proprio = [EE position in the base frame, 6D identity,
    gripper 0] padded to 20 and then replaced by each executed action, the whole 30-step chunk executed, rotation + (0,
    pi/2, 0), gripper open below 0.7; Google domain 1, proprio zeros, every second action of the chunk up to 10,
    positions relative to the last commanded position, gripper closed above 0.25. Official SimplerEnv step limits and
    success at the last step (X-VLA's clients allow 1200 steps on WidowX / twice the limit on Google and stop at the
    first success; its WidowX eggplant scene also moves the camera and the robot)."""

    needs_obs = True
    control_modes = {
        "google_robot": "arm_pd_ee_base_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
        "widowx": "arm_pd_ee_target_base_pose_gripper_pd_joint_pos",
    }

    def __init__(self, policy_setup, port, host="127.0.0.1"):
        self.google = policy_setup == "google_robot"
        self.client = _LeRobotServerClient(port, host)
        self.n_infer = 0

    @staticmethod
    def _ee_in_base(env):
        u = env.unwrapped
        return u.agent.robot.pose.inv() * u.tcp.pose

    def reset(self, task_description, obs=None, env=None):
        import collections

        import numpy as np

        self.plan = collections.deque()
        ee = self._ee_in_base(env)
        if self.google:
            self.proprio = np.zeros(20, dtype=np.float32)
            self.current_xyz = np.asarray(ee.p, dtype=np.float32)
        else:
            p10 = np.concatenate([ee.p, [1, 0, 0, 1, 0, 0, 0]]).astype(np.float32)
            self.proprio = np.concatenate([p10, np.zeros_like(p10)])

    def step(self, image, task_description, obs=None, env=None):
        import math

        import numpy as np

        if not self.plan:
            # X-VLA's processor (CLIPImageProcessor of the original checkpoints): 224x224 bicubic, no aspect padding;
            # the server's code pipeline then applies ImageNet normalization.
            from PIL import Image

            img = np.asarray(Image.fromarray(image).resize((224, 224), Image.BICUBIC))
            seq = self.client.chunk({"observation.images.image": img}, task_description, state=self.proprio,
                                    extra={"domain_id": 1 if self.google else 0})
            seq = np.asarray(seq, dtype=np.float32).reshape(-1, seq.shape[-1])
            self.n_infer += 1
            if self.google:
                seq = seq[::2][:10].copy()
                seq[:, :3] += self.current_xyz
            self.plan.extend(list(seq))
        a = np.asarray(self.plan.popleft(), dtype=np.float32)
        if self.google:
            euler = _rotate6d_to_euler_xyz(a[3:9])
            gripper = 1.0 if a[9] > 0.25 else -1.0
            self.current_xyz = a[:3].copy()
        else:
            self.proprio[:10] = a[:10]
            euler = _rotate6d_to_euler_xyz(a[3:9]) + np.array([0, math.pi / 2, 0])
            gripper = 1.0 if a[9] < 0.7 else -1.0
        action = {"world_vector": a[:3].astype(np.float64), "rot_axangle": np.asarray(euler, dtype=np.float64),
                  "gripper": np.array([gripper]), "terminate_episode": np.array([0.0])}
        return {"action": a}, action


def xvla(policy_setup, port, host="127.0.0.1"):
    return XVLASimplerPolicy(policy_setup, port=port, host=host)


class VLAJEPASimplerPolicy:
    """VLA-JEPA (LeRobot port lerobot/VLA-JEPA-SimplerEnv) with the client of VLA-JEPA's own SimplerEnv evaluation
    (examples/SimplerEnv/eval_files/model2simpler_interface.py): image resized to 224x224 (INTER_AREA), one view; the
    model's normalized action chunk un-normalized with the q01/q99 statistics of the robot's dataset (oxe_bridge /
    oxe_rt1, from the original checkpoint's dataset_statistics.json: the LeRobot port only carries Bridge statistics),
    gripper snapped at 0.5; adaptive action ensemble (horizon 7 WidowX / 2 Google, alpha 0.1); Euler -> axis-angle;
    gripper: WidowX binarized, Google relative sticky (10 repeats) with the evaluation's own fix of SimplerEnv's code."""

    def __init__(self, policy_setup, port, stats_path, host="127.0.0.1", alpha=0.1):
        import importlib.util
        import json

        import numpy as np

        self.google = policy_setup == "google_robot"
        key = "oxe_rt1" if self.google else "oxe_bridge"
        stats = json.load(open(stats_path))[key]["action"]
        self.q01, self.q99 = np.array(stats["q01"]), np.array(stats["q99"])
        self.mask = np.array(stats.get("mask", np.ones_like(self.q01, dtype=bool)), dtype=bool)
        path = os.path.join(ROOT, "third_party", "VLA-JEPA", "examples", "SimplerEnv", "eval_files", "adaptive_ensemble.py")
        spec = importlib.util.spec_from_file_location("vla_jepa_adaptive_ensemble", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.ensembler = module.AdaptiveEnsembler(2 if self.google else 7, alpha)
        self.sticky_repeat_max = 10 if self.google else 1
        self.client = _LeRobotServerClient(port, host)
        self.n_infer = 0

    def reset(self, task_description):
        self.task_description = task_description
        self.ensembler.reset()
        self.sticky_on, self.repeat, self.sticky_action, self.previous = False, 0, 0.0, None

    def _unnormalize(self, normalized):
        import numpy as np

        a = np.clip(np.asarray(normalized, dtype=np.float64), -1, 1)
        a[:, 6] = np.where(a[:, 6] < 0.5, 0, 1)
        return np.where(self.mask, 0.5 * (a + 1) * (self.q99 - self.q01) + self.q01, a)

    def step(self, image, task_description=None):
        import cv2
        import numpy as np
        from transforms3d.euler import euler2axangle

        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        chunk = self.client.chunk({"observation.images.image": img}, self.task_description, postprocess=False)
        self.n_infer += 1
        raw = self._unnormalize(np.asarray(chunk).reshape(-1, chunk.shape[-1]))
        raw = self.ensembler.ensemble_action(raw)[None]
        raw_action = {"world_vector": raw[0, :3], "rotation_delta": raw[0, 3:6], "open_gripper": raw[0, 6:7]}
        axis, angle = euler2axangle(*np.asarray(raw_action["rotation_delta"], dtype=np.float64))
        action = {"world_vector": raw_action["world_vector"], "rot_axangle": axis * angle}
        if self.google:
            current = raw_action["open_gripper"]
            if self.previous is None:
                relative = np.array([0])
                self.previous = current
            else:
                relative = self.previous - current
            if np.abs(relative) > 0.5 and not self.sticky_on:
                self.sticky_on, self.sticky_action, self.previous = True, relative, current
            if self.sticky_on:
                self.repeat += 1
                relative = self.sticky_action
            if self.repeat == self.sticky_repeat_max:
                self.sticky_on, self.repeat, self.sticky_action = False, 0, 0.0
            action["gripper"] = relative
        else:
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action


def vla_jepa(policy_setup, port, stats_path, host="127.0.0.1"):
    return VLAJEPASimplerPolicy(policy_setup, port=port, stats_path=stats_path, host=host)


class MolmoActVLLMClient:
    """MolmoAct through scripts/molmoact_vllm_server.py (vLLM, batched across the workers): the client half of
    allenai/SimplerEnv's MolmoActInferenceVLLM.step: image resized to 256x256 (INTER_AREA), then the generated action
    (parsed by the server with MolmoActParser, unnorm_key fractal20220817_data) turned into the env action exactly as
    there: world vector, Euler -> axis-angle, Google sticky relative gripper (15 repeats), WidowX binarized gripper."""

    def __init__(self, policy_setup, port, host="127.0.0.1", action_scale=1.0):
        self.google = policy_setup == "google_robot"
        self.sticky_gripper_num_repeat = 15 if self.google else 1
        self.action_scale = action_scale
        self.client = _LeRobotServerClient(port, host)
        self.n_infer = 0
        self.task_description = None

    def reset(self, task_description):
        self.task_description = task_description
        self.sticky_action_is_on, self.gripper_action_repeat = False, 0
        self.sticky_gripper_action, self.previous_gripper_action = 0.0, None

    def step(self, image, task_description=None):
        import cv2
        import numpy as np
        from transforms3d.euler import euler2axangle

        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        reply = self.client.call({"cmd": "infer", "image": self.client.npy(img), "instruction": self.task_description})
        self.n_infer += 1
        if reply["action"] is None:
            raise ValueError(f"no action could be parsed from: {reply['text'][-200:]!r}")
        a = np.asarray(reply["action"], dtype=np.float64).reshape(-1)[:7]
        raw_action = {"world_vector": a[:3], "rotation_delta": a[3:6], "open_gripper": a[6:7]}
        action = {"world_vector": raw_action["world_vector"] * self.action_scale}
        ax, angle = euler2axangle(*raw_action["rotation_delta"])
        action["rot_axangle"] = ax * angle * self.action_scale
        if self.google:
            current = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative = np.array([0])
            else:
                relative = self.previous_gripper_action - current
            self.previous_gripper_action = current
            if np.abs(relative) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on, self.sticky_gripper_action = True, relative
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on, self.gripper_action_repeat, self.sticky_gripper_action = False, 0, 0.0
            action["gripper"] = relative
        else:
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action


def molmoact_vllm(policy_setup, port, host="127.0.0.1"):
    return MolmoActVLLMClient(policy_setup, port=port, host=host)


def molmoact(policy_setup, ckpt):
    """MolmoAct (allenai/MolmoAct-7B-D-Pretrain-RT-1-0812, Google Robot): the Hugging Face inference class of
    allenai/SimplerEnv (simpler_env/policies/molmoact/molmoact_model.py, as main_inference.py --policy-model molmoact
    builds it: image 256x256, unnorm_key fractal20220817_data, sticky gripper 15). Its step() also returns an annotated
    image, dropped here; it reads its dataset statistics relative to the repository root."""
    assert policy_setup == "google_robot", "the MolmoAct SimplerEnv checkpoint is for the Google Robot only"
    import importlib.util

    repo = os.path.join(ROOT, "third_party", "SimplerEnv-MolmoAct")
    spec = importlib.util.spec_from_file_location("molmoact_simpler_model",
                                                  os.path.join(repo, "simpler_env", "policies", "molmoact", "molmoact_model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        inner = module.MolmoActInference(saved_model_path=ckpt, policy_setup=policy_setup)
    finally:
        os.chdir(cwd)

    class _TwoOutputs:
        def __init__(self, model):
            self.model, self.n_infer = model, 0

        def reset(self, task_description):
            self.model.reset(task_description)

        def step(self, image, task_description=None):
            self.n_infer += 1
            raw_action, action, _ = self.model.step(image, task_description)
            return raw_action, action

    return _TwoOutputs(inner)


class XiaomiR0SimplerPolicy:
    """Xiaomi-Robotics-0 through scripts/xiaomi_policy_server.py (deploy/server.py for several clients), with the
    client half of Xiaomi-Robotics-0's eval_simplerenv/main.py run_single_episode, whose functions are used unchanged
    (preprocess_proprio_bridge / _fractal, client_process: 256x256 (WidowX) / 320x256 (Google) image, 0.95 centre crop
    resized back, prompt template, state padded to 32; hash_data_to_seed; postprocess_gripper_bridge / _fractal) and
    its deploy/client.py Client. Proprio = the `eef_pos` of allenzren's SimplerEnv fork (end-effector pose in the robot
    base frame, wxyz quaternion, gripper openness), computed from the same links as for GR00T, since the official
    SimplerEnv has no `eef_pos`. Instruction with a capital first letter and a final "."; the first 4 actions of each
    chunk executed before the next query (replan_steps 4); Euler -> axis-angle. Official SimplerEnv episodes, step
    limits and success at the last step (Xiaomi's client stops at the first success)."""

    needs_obs = True

    def __init__(self, policy_setup, port, host="127.0.0.1", replan_steps=4, repo=None):
        import types

        repo = repo or os.path.join(ROOT, "third_party", "Xiaomi-Robotics-0")
        sys.modules.setdefault("tyro", types.ModuleType("tyro"))  # main.py imports it only for its command line
        import importlib.util

        spec = importlib.util.spec_from_file_location("xiaomi_eval_simplerenv",
                                                      os.path.join(repo, "eval_simplerenv", "main.py"))
        self.main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.main)
        self.google = policy_setup == "google_robot"
        self.ee_link_name = "link_gripper_tcp" if self.google else "ee_gripper_link"
        self.replan_steps = int(replan_steps)
        self.addr = (host, int(port))
        _wait(f"Xiaomi-Robotics-0 server {self.addr}", self._port_open, 3600)
        self.client = self.main.Client(host, int(port))
        self.n_infer = 0

    def _port_open(self):
        import socket

        with socket.socket() as s:
            return s.connect_ex(self.addr) == 0

    def _eef_pos(self, env):
        import numpy as np
        from transforms3d import quaternions as tq

        agent = env.unwrapped.agent
        link = [x for x in agent.robot.get_links() if x.name == self.ee_link_name][0]
        ee_in_base = np.linalg.inv(agent.base_pose.to_transformation_matrix()) @ link.get_pose().to_transformation_matrix()
        openness = 1.0 - float(agent.get_gripper_closedness())
        return np.concatenate([ee_in_base[:3, 3], tq.mat2quat(ee_in_base[:3, :3]), [openness]])

    def reset(self, task_description, obs=None, env=None):
        import collections

        self.plan = collections.deque()

    def step(self, image, task_description, obs=None, env=None):
        import numpy as np
        from PIL import Image
        from transforms3d.euler import euler2axangle

        m = self.main
        if not self.plan:
            base_obs = Image.fromarray(image, mode="RGB")
            if self.google:
                state, task_id = m.preprocess_proprio_fractal(self._eef_pos(env)), "fractal_delta"
            else:
                state, task_id = m.preprocess_proprio_bridge(self._eef_pos(env)), "bridge_delta"
            instruction = task_description[0].upper() + task_description[1:] + "."
            model_inputs = m.client_process(task_id, state, base_obs, instruction)
            model_inputs["seed"] = m.hash_data_to_seed(model_inputs)
            chunk = self.client(**model_inputs)[0]
            self.n_infer += 1
            self.plan.extend(chunk[: self.replan_steps, :7].cpu().numpy())
        raw = self.plan.popleft()
        ax, angle = euler2axangle(*raw[3:6])
        gripper = m.postprocess_gripper_fractal(raw[-1]) if self.google else m.postprocess_gripper_bridge(raw[-1])
        action = {"world_vector": np.asarray(raw[:3]), "rot_axangle": ax * angle,
                  "gripper": np.array([gripper], dtype=np.float64).reshape(-1), "terminate_episode": np.array([0.0])}
        return {"raw": raw}, action


def xiaomi_r0(policy_setup, port, host="127.0.0.1", replan_steps=4):
    return XiaomiR0SimplerPolicy(policy_setup, port=port, host=host, replan_steps=replan_steps)


class ECoTSimplerPolicy:
    """ECoT, Embodied-CoT/ecot-openvla-7b-bridge (WidowX only), as the embodied-CoT README's Hugging Face quickstart
    runs it: AutoModelForVision2Seq in bfloat16 (OpenVLA's remote code, flash_attention_2 as the OpenVLA SimplerEnv
    adapter loads it), prompt "A chat between ... USER: What action should the robot take to {instruction.lower()}?
    ASSISTANT: TASK:", predict_action(unnorm_key="bridge_orig", max_new_tokens=1024, do_sample=False): the reasoning
    (plan, subtask, move, gripper position, objects) is generated before the 7 action tokens at every step. The image is
    prepared as ECoT's own Bridge evaluation does (experiments/bridge/utils.py resize_image: 256x256, then 224x224,
    LANCZOS); the action goes to the env as in the OpenVLA SimplerEnv adapter (Euler -> axis-angle, gripper
    binarized at 0.5). The generated text of each episode's last step is kept in `last_reasoning`."""

    PROMPT = ("A chat between a curious user and an artificial intelligence assistant. "
              "The assistant gives helpful, detailed, and polite answers to the user's questions. "
              "USER: What action should the robot take to {}? ASSISTANT: TASK:")

    def __init__(self, policy_setup, ckpt, max_new_tokens=1024):
        assert policy_setup == "widowx_bridge", "the ECoT Bridge checkpoint is for the WidowX only"
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(ckpt, attn_implementation="flash_attention_2",
                                                          torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                          trust_remote_code=True).cuda()
        self.max_new_tokens = int(max_new_tokens)
        self.n_infer, self.last_reasoning = 0, ""

    def reset(self, task_description):
        self.task_description = task_description

    def step(self, image, task_description=None):
        import numpy as np
        import torch
        from PIL import Image
        from transforms3d.euler import euler2axangle

        if task_description is not None:
            self.task_description = task_description
        img = Image.fromarray(image).resize((256, 256), Image.Resampling.LANCZOS)
        img = img.resize((224, 224), Image.Resampling.LANCZOS).convert("RGB")
        inputs = self.processor(self.PROMPT.format(self.task_description.lower()), img).to("cuda:0", dtype=torch.bfloat16)
        with torch.inference_mode():
            actions, generated_ids = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False,
                                                             max_new_tokens=self.max_new_tokens)
        self.n_infer += 1
        self.last_reasoning = self.processor.batch_decode(generated_ids)[0]
        a = np.asarray(actions, dtype=np.float64).reshape(-1)
        raw_action = {"world_vector": a[:3], "rotation_delta": a[3:6], "open_gripper": a[6:7]}
        ax, angle = euler2axangle(*raw_action["rotation_delta"])
        action = {"world_vector": raw_action["world_vector"], "rot_axangle": ax * angle,
                  "gripper": 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0, "terminate_episode": np.array([0.0])}
        return raw_action, action


def ecot(policy_setup, ckpt, max_new_tokens=1024):
    return ECoTSimplerPolicy(policy_setup, ckpt=ckpt, max_new_tokens=max_new_tokens)


def internvla_m1(policy_setup, ckpt, port, host="127.0.0.1", action_scale=1.0):
    """InternVLA-M1 (InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge) with its own SimplerEnv client, the M1Inference
    class of examples/SimplerEnv/model2simpler_interface.py built as start_simpler_env.py builds it (cfg_scale 1.5;
    class defaults: image 224x224 INTER_AREA, DDIM 10 steps, adaptive action ensembling over 7 (WidowX) / 2 (Google)
    steps with alpha 0.1, unnorm_key bridge_dataset / fractal20220817_data from the checkpoint's statistics, WidowX
    gripper binarized, Google sticky gripper 10 repeats). The model runs in the repository's websocket server
    (deployment/model_server/server_policy_M1.py through scripts/starvla_server_launcher.py), started by
    run_simpler_supervisor.sh."""
    import socket

    sys.path.insert(0, os.path.join(ROOT, "third_party", "InternVLA-M1"))
    from examples.SimplerEnv.model2simpler_interface import M1Inference

    def port_open():
        with socket.socket() as s:
            return s.connect_ex((host, int(port))) == 0

    _wait(f"InternVLA-M1 server {host}:{port}", port_open, 3600)
    return M1Inference(policy_ckpt_path=ckpt, policy_setup=policy_setup, port=int(port), host=host,
                       action_scale=action_scale, cfg_scale=1.5)
