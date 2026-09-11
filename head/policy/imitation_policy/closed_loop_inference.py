"""Single model-independent closed-loop inference entry point."""
import numpy as np
from head.agents import AgentInput, Trajectory, create_agent


class ClosedLoopInference:
    def __init__(self, imitation_config, checkpoint, device="cpu", controller_dt=0.1):
        self.agent = create_agent(imitation_config, checkpoint, device=device)
        self.controller_dt = float(controller_dt)
        self.trajectory = None

    def predict(self, scenario, current_step):
        trajectory = self.agent.compute_trajectory(AgentInput(scenario, int(current_step)))
        if not isinstance(trajectory, Trajectory):
            raise TypeError("Agent.compute_trajectory must return head.agents.Trajectory")
        if not np.isclose(trajectory.dt, self.controller_dt, rtol=0, atol=1e-8):
            raise ValueError("Agent trajectory dt must match controller dt; resample in the agent")
        self.trajectory = trajectory
        return trajectory.samples

    def control_position(self, centre, heading):
        if self.trajectory is None:
            return centre
        return self.trajectory.control_position(centre, heading)

    def reset(self):
        self.trajectory = None
        self.agent.reset()
