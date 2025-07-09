import numpy as np
from typing import Optional, Union
from metadrive.engine.core.engine_core import RENDER_MODE_NONE


class HeadTopDownRenderer:
    """
    This class is used to render the top-down view of the environment.
    It is a pseudo-render function, only used to update onscreen message when using panda3d backend.
    """

    def __init__(self, env):
        self.engine = env.engine
        self.head_top_down_renderer = None

    def _render_topdown(self, text: Optional[Union[dict, str]] = None, *args, **kwargs) -> Optional[np.ndarray]:
        """
        Render the top-down view of the environment.
        :param text: text to show
        :return: top_down image
        """
        if self.head_top_down_renderer is None:
            from head.engine.top_down_renderer import TopDownRenderer
            self.head_top_down_renderer = TopDownRenderer(*args, **kwargs)
        return self.head_top_down_renderer.render(text, *args, **kwargs)

    def render(self, text: Optional[Union[dict, str]] = None, mode=None, *args, **kwargs) -> Optional[np.ndarray]:
        """
        This is a pseudo-render function, only used to update onscreen message when using panda3d backend
        :param text: text to show
        :param mode: start_top_down rendering candidate parameter is ["top_down", "topdown", "bev", "birdview"]
        :return: None or top_down image
        """

        if mode in ["top_down", "topdown", "bev", "birdview"]:
            ret = self._render_topdown(text=text, *args, **kwargs)
            return ret
        return None
