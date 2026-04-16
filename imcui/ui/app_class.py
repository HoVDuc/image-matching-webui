from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
from easydict import EasyDict as edict
from omegaconf import OmegaConf

from .sfm import SfmEngine
from .utils import (
    GRADIO_VERSION,
    MAX_TABS,
    gen_examples,
    generate_warp_images,
    get_matcher_zoo,
    load_config,
    ransac_zoo,
    run_matching,
    run_matching_multi,
    run_ransac,
    send_to_match,
)

DESCRIPTION = """
<div style="text-align: left;">
  <h1 style="font-size: 2rem; font-weight: bold; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 1rem;">
    Image Matching WebUI
  </h1>
</div>

This Space demonstrates [Image Matching WebUI](https://github.com/Vincentqyw/image-matching-webui) by vincent qin. Feel free to play with it, or duplicate to run image matching without a queue!
<br/>
🚀 **Now GPU-accelerated!** Thanks to HuggingFace's community grant, all algorithms run on GPU for fast, responsive inference.

🔎 For more details about supported local features and matchers, please refer to https://github.com/Vincentqyw/image-matching-webui

🐛 Your feedback is valuable to me. Please do not hesitate to report any bugs [here](https://github.com/Vincentqyw/image-matching-webui/issues).
"""

CSS = """
#warning {background-color: #FFCCCB}
.logs_class textarea {font-size: 10px !important}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

"""


class ImageMatchingApp:
    def __init__(self, server_name="0.0.0.0", server_port=7860, **kwargs):
        self.server_name = server_name
        self.server_port = server_port
        self.config_path = kwargs.get("config", Path(__file__).parent / "config.yaml")
        self.cfg = load_config(self.config_path)
        self.matcher_zoo = get_matcher_zoo(self.cfg["matcher_zoo"])
        self.app = None
        self.example_data_root = kwargs.get(
            "example_data_root", Path(__file__).parents[1] / "datasets"
        )
        # final step
        self.init_interface()

    def init_matcher_dropdown(self):
        algos = []
        for k, v in self.cfg["matcher_zoo"].items():
            if v.get("enable", True):
                algos.append(k)
        return algos

    def init_interface(self):
        with gr.Blocks() as self.app:
            with gr.Tab("Image Matching"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Image(
                            str(Path(__file__).parent.parent / "assets/logo.webp"),
                            elem_id="logo-img",
                            show_label=False,
                            buttons=["fullscreen"],
                        )
                    with gr.Column(scale=3):
                        gr.Markdown(DESCRIPTION)
                with gr.Row(equal_height=False):
                    with gr.Column():
                        with gr.Row():
                            matcher_list = gr.Dropdown(
                                choices=self.init_matcher_dropdown(),
                                value="disk+lightglue",
                                label="Matching Model",
                                interactive=True,
                            )
                            match_image_src = gr.Radio(
                                (
                                    ["upload", "webcam", "clipboard"]
                                    if GRADIO_VERSION > "3"
                                    else ["upload", "webcam", "canvas"]
                                ),
                                label="Image Source",
                                value="upload",
                            )
                        with gr.Row():
                            input_candidates = gr.Gallery(
                                label="Candidates (multiple images)",
                                interactive=True,
                                columns=4,
                                height=300 if GRADIO_VERSION > "3" else None,
                            )
                            input_query = gr.Image(
                                label="Query",
                                type="numpy",
                                image_mode="RGB",
                                height=300 if GRADIO_VERSION > "3" else None,
                                interactive=True,
                            )

                        with gr.Row():
                            button_reset = gr.Button(value="Reset")
                            button_run = gr.Button(value="Run Match", variant="primary")
                        with gr.Row():
                            button_stop = gr.Button(value="Force Stop", variant="stop")

                        with gr.Accordion("Advanced Setting", open=False):
                            with gr.Tab("Matching Setting"):
                                with gr.Row():
                                    match_setting_threshold = gr.Slider(
                                        minimum=0.0,
                                        maximum=1,
                                        step=0.001,
                                        label="Match threshold",
                                        value=0.1,
                                    )
                                    match_setting_max_keypoints = gr.Slider(
                                        minimum=10,
                                        maximum=10000,
                                        step=10,
                                        label="Max features",
                                        value=1000,
                                    )
                                # TODO: add line settings
                                with gr.Row():
                                    detect_keypoints_threshold = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        step=0.001,
                                        label="Keypoint threshold",
                                        value=0.015,
                                    )
                                    detect_line_threshold = gr.Slider(  # noqa: F841
                                        minimum=0.1,
                                        maximum=1,
                                        step=0.01,
                                        label="Line threshold",
                                        value=0.2,
                                    )

                            with gr.Tab("RANSAC Setting"):
                                with gr.Row(equal_height=False):
                                    ransac_method = gr.Dropdown(
                                        choices=ransac_zoo.keys(),
                                        value=self.cfg["defaults"]["ransac_method"],
                                        label="RANSAC Method",
                                        interactive=True,
                                    )
                                ransac_reproj_threshold = gr.Slider(
                                    minimum=0.0,
                                    maximum=12,
                                    step=0.01,
                                    label="Ransac Reproj threshold",
                                    value=8.0,
                                )
                                ransac_confidence = gr.Slider(
                                    minimum=0.0,
                                    maximum=1,
                                    step=0.00001,
                                    label="Ransac Confidence",
                                    value=self.cfg["defaults"]["ransac_confidence"],
                                )
                                ransac_max_iter = gr.Slider(
                                    minimum=0.0,
                                    maximum=100000,
                                    step=100,
                                    label="Ransac Iterations",
                                    value=self.cfg["defaults"]["ransac_max_iter"],
                                )
                                button_ransac = gr.Button(
                                    value="Rerun RANSAC", variant="primary"
                                )
                            with gr.Tab("Geometry Setting"):
                                with gr.Row(equal_height=False):
                                    choice_geometry_type = gr.Radio(
                                        ["Fundamental", "Homography"],
                                        label="Reconstruct Geometry",
                                        value=self.cfg["defaults"]["setting_geometry"],
                                    )
                            with gr.Tab("Image Setting"):
                                with gr.Row():
                                    image_force_resize_cb = gr.Checkbox(
                                        label="Force Resize",
                                        value=False,
                                        interactive=True,
                                    )
                                    image_setting_height = gr.Slider(
                                        minimum=48,
                                        maximum=2048,
                                        step=16,
                                        label="Image Height",
                                        value=480,
                                        visible=False,
                                    )
                                    image_setting_width = gr.Slider(
                                        minimum=64,
                                        maximum=2048,
                                        step=16,
                                        label="Image Width",
                                        value=640,
                                        visible=False,
                                    )
                        # image resize
                        image_force_resize_cb.select(
                            fn=self._on_select_force_resize,
                            inputs=image_force_resize_cb,
                            outputs=[image_setting_width, image_setting_height],
                        )
                        # collect inputs
                        inputs = [
                            input_candidates,
                            input_query,
                            match_setting_threshold,
                            match_setting_max_keypoints,
                            detect_keypoints_threshold,
                            matcher_list,
                            ransac_method,
                            ransac_reproj_threshold,
                            ransac_confidence,
                            ransac_max_iter,
                            choice_geometry_type,
                            gr.State(self.matcher_zoo),
                            image_force_resize_cb,
                            image_setting_width,
                            image_setting_height,
                        ]

                        # Examples section removed: gr.Gallery input is not compatible
                        # with the gr.Examples tabular format for multi-image candidates.
                        with gr.Accordion("Supported Algorithms", open=False):
                            # add a table of supported algorithms
                            self.display_supported_algorithms()

                    with gr.Column():
                        with gr.Tabs(elem_id="output_tabs"):
                            tab_objects = []
                            for i in range(MAX_TABS):
                                with gr.Tab(
                                    label=f"Candidate {i + 1}", visible=False
                                ) as tab:
                                    with gr.Accordion("Keypoints", open=True):
                                        out_kpts = gr.Image(
                                            label="Keypoints", type="numpy"
                                        )
                                    with gr.Accordion(
                                        "Raw Matches (Green=good, Red=bad)", open=False
                                    ):
                                        out_raw = gr.Image(
                                            label="Raw Matches", type="numpy"
                                        )
                                    with gr.Accordion(
                                        "RANSAC Matches (Green=good, Red=bad)",
                                        open=True,
                                    ):
                                        out_ransac = gr.Image(
                                            label="RANSAC Matches", type="numpy"
                                        )
                                    with gr.Accordion("Warped Image", open=True):
                                        out_wrap = gr.Image(
                                            label="Warped Pair", type="numpy"
                                        )
                                    with gr.Accordion("Match Statistics", open=False):
                                        out_info = gr.JSON(label="Match Info")
                                    with gr.Accordion("Geometry Info", open=False):
                                        out_geom = gr.JSON(label="Geometry")
                                tab_objects.append(
                                    (tab, out_kpts, out_raw, out_ransac, out_wrap, out_info, out_geom)
                                )

                    # flat list: [tab0, kpts0, raw0, ransac0, wrap0, info0, geom0, tab1, ...]
                    flat_tab_outputs = [comp for items in tab_objects for comp in items]

                    # match_image_src only changes the query image source
                    match_image_src.change(
                        fn=self.ui_change_imagebox,
                        inputs=match_image_src,
                        outputs=input_query,
                    )

                    # Run button
                    click_event = button_run.click(
                        fn=run_matching_multi,
                        inputs=inputs,
                        outputs=flat_tab_outputs,
                    )

                    # Stop button
                    button_stop.click(
                        fn=None, inputs=None, outputs=None, cancels=[click_event]
                    )

                    # Reset button
                    reset_outputs = [
                        input_candidates,
                        input_query,
                        match_setting_threshold,
                        match_setting_max_keypoints,
                        detect_keypoints_threshold,
                        matcher_list,
                        match_image_src,
                        image_force_resize_cb,
                    ] + flat_tab_outputs
                    button_reset.click(
                        fn=self.ui_reset_state,
                        inputs=None,
                        outputs=reset_outputs,
                    )
            with gr.Tab("Structure from Motion(under-dev)"):
                sfm_ui = AppSfmUI(  # noqa: F841
                    {
                        **self.cfg,
                        "matcher_zoo": self.matcher_zoo,
                        "outputs": "experiments/sfm",
                    }
                )
                sfm_ui.call_empty()

    def run(self):
        self.app.queue().launch(
            server_name=self.server_name,
            server_port=self.server_port,
            share=False,
            css=CSS,
            allowed_paths=[
                str(Path(__file__).parents[0]),
                str(Path(__file__).parents[1]),
            ],
        )

    def ui_change_imagebox(self, choice):
        """
        Updates the image box with the given choice.

        Args:
            choice (list): The list of image sources to be displayed in the image box.

        Returns:
            dict: A dictionary containing the updated value, sources, and type for the image box.
        """
        ret_dict = {
            "value": None,  # The updated value of the image box
            "__type__": "update",  # The type of update for the image box
        }
        if GRADIO_VERSION > "3":
            return {
                **ret_dict,
                "sources": choice,  # The list of image sources to be displayed
            }
        else:
            return {
                **ret_dict,
                "source": choice,  # The list of image sources to be displayed
            }

    def _on_select_force_resize(self, visible: bool = False):
        return gr.update(visible=visible), gr.update(visible=visible)

    def ui_reset_state(self, *args: Any):
        """Reset the state of the UI."""
        import gradio as gr

        key: str = list(self.matcher_zoo.keys())[0]
        base = (
            None,  # input_candidates
            None,  # input_query
            self.cfg["defaults"]["match_threshold"],
            self.cfg["defaults"]["max_keypoints"],
            self.cfg["defaults"]["keypoint_threshold"],
            key,
            "upload",  # match_image_src
            False,  # image_force_resize_cb
        )
        # Reset all tab slots: (tab hidden, 6x None) × MAX_TABS
        tab_resets = tuple(
            v
            for _ in range(MAX_TABS)
            for v in (gr.update(visible=False), None, None, None, None, None, None)
        )
        return base + tab_resets

    def display_supported_algorithms(self, style="tab"):
        def get_link(link, tag="Link"):
            return "[{}]({})".format(tag, link) if link is not None else "None"

        data = []
        cfg = self.cfg["matcher_zoo"]
        if style == "md":
            markdown_table = "| Algo. | Conference | Code | Project | Paper |\n"
            markdown_table += "| ----- | ---------- | ---- | ------- | ----- |\n"

            for _, v in cfg.items():
                if not v["info"].get("display", True):
                    continue
                github_link = get_link(v["info"].get("github", ""))
                project_link = get_link(v["info"].get("project", ""))
                paper_link = get_link(
                    v["info"]["paper"],
                    (
                        Path(v["info"]["paper"]).name[-10:]
                        if v["info"]["paper"] is not None
                        else "Link"
                    ),
                )

                markdown_table += "{}|{}|{}|{}|{}\n".format(
                    v["info"].get("name", ""),
                    v["info"].get("source", ""),
                    github_link,
                    project_link,
                    paper_link,
                )
            return gr.Markdown(markdown_table)
        elif style == "tab":
            for k, v in cfg.items():
                if not v["info"].get("display", True):
                    continue
                data.append(
                    [
                        v["info"].get("name", ""),
                        v["info"].get("source", ""),
                        v["info"].get("github", ""),
                        v["info"].get("paper", ""),
                        v["info"].get("project", ""),
                    ]
                )
            tab = gr.Dataframe(
                headers=["Algo.", "Conference", "Code", "Paper", "Project"],
                datatype=["str", "str", "str", "str", "str"],
                column_count=5,
                column_limits=(5, 5),
                value=data,
                # wrap=True,
                # min_width = 1000,
                # height=1000,
            )
            return tab


class AppBaseUI:
    def __init__(self, cfg: Dict[str, Any] = {}):
        self.cfg = OmegaConf.create(cfg)
        self.inputs = edict({})
        self.outputs = edict({})
        self.ui = edict({})

    def _init_ui(self):
        NotImplemented

    def call(self, **kwargs):
        NotImplemented

    def info(self):
        gr.Info("SFM is under construction.")


class AppSfmUI(AppBaseUI):
    def __init__(self, cfg: Dict[str, Any] = None):
        super().__init__(cfg)
        assert "matcher_zoo" in self.cfg
        self.matcher_zoo = self.cfg["matcher_zoo"]
        self.sfm_engine = SfmEngine(cfg)
        self._init_ui()

    def init_retrieval_dropdown(self):
        algos = []
        for k, v in self.cfg["retrieval_zoo"].items():
            if v.get("enable", True):
                algos.append(k)
        return algos

    def _update_options(self, option):
        if option == "sparse":
            return gr.Textbox("sparse", visible=True)
        elif option == "dense":
            return gr.Textbox("dense", visible=True)
        else:
            return gr.Textbox("not set", visible=True)

    def _on_select_custom_params(self, value: bool = False):
        return gr.update(visible=value)

    def _init_ui(self):
        with gr.Row():
            # data settting and camera settings
            with gr.Column():
                self.inputs.input_images = gr.File(
                    label="SfM",
                    interactive=True,
                    file_count="multiple",
                    min_width=300,
                )
                # camera setting
                with gr.Accordion("Camera Settings", open=True):
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                self.inputs.camera_model = gr.Dropdown(
                                    choices=[
                                        "PINHOLE",
                                        "SIMPLE_RADIAL",
                                        "OPENCV",
                                    ],
                                    value="PINHOLE",
                                    label="Camera Model",
                                    interactive=True,
                                )
                            with gr.Column():
                                gr.Checkbox(
                                    label="Shared Params",
                                    value=True,
                                    interactive=True,
                                )
                                camera_custom_params_cb = gr.Checkbox(
                                    label="Custom Params",
                                    value=False,
                                    interactive=True,
                                )
                        with gr.Row():
                            self.inputs.camera_params = gr.Textbox(
                                label="Camera Params",
                                value="0,0,0,0",
                                interactive=False,
                                visible=False,
                            )
                        camera_custom_params_cb.select(
                            fn=self._on_select_custom_params,
                            inputs=camera_custom_params_cb,
                            outputs=self.inputs.camera_params,
                        )

                with gr.Accordion("Matching Settings", open=True):
                    # feature extraction and matching setting
                    with gr.Row():
                        # matcher setting
                        self.inputs.matcher_key = gr.Dropdown(
                            choices=self.matcher_zoo.keys(),
                            value="disk+lightglue",
                            label="Matching Model",
                            interactive=True,
                        )
                    with gr.Row():
                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Column():
                                with gr.Row():
                                    # matching setting
                                    self.inputs.max_keypoints = gr.Slider(
                                        label="Max Keypoints",
                                        minimum=100,
                                        maximum=10000,
                                        value=1000,
                                        interactive=True,
                                    )
                                    self.inputs.keypoint_threshold = gr.Slider(
                                        label="Keypoint Threshold",
                                        minimum=0,
                                        maximum=1,
                                        value=0.01,
                                    )
                                with gr.Row():
                                    self.inputs.match_threshold = gr.Slider(
                                        label="Match Threshold",
                                        minimum=0.01,
                                        maximum=12.0,
                                        value=0.2,
                                    )
                                    self.inputs.ransac_threshold = gr.Slider(
                                        label="Ransac Threshold",
                                        minimum=0.01,
                                        maximum=12.0,
                                        value=4.0,
                                        step=0.01,
                                        interactive=True,
                                    )

                                with gr.Row():
                                    self.inputs.ransac_confidence = gr.Slider(
                                        label="Ransac Confidence",
                                        minimum=0.01,
                                        maximum=1.0,
                                        value=0.9999,
                                        step=0.0001,
                                        interactive=True,
                                    )
                                    self.inputs.ransac_max_iter = gr.Slider(
                                        label="Ransac Max Iter",
                                        minimum=1,
                                        maximum=100,
                                        value=100,
                                        step=1,
                                        interactive=True,
                                    )
                with gr.Accordion("Scene Graph Settings", open=True):
                    # mapping setting
                    self.inputs.scene_graph = gr.Dropdown(
                        choices=["all", "swin", "oneref"],
                        value="all",
                        label="Scene Graph",
                        interactive=True,
                    )

                    # global feature setting
                    self.inputs.global_feature = gr.Dropdown(
                        choices=self.init_retrieval_dropdown(),
                        value="netvlad",
                        label="Global features",
                        interactive=True,
                    )
                    self.inputs.top_k = gr.Slider(
                        label="Number of Images per Image to Match",
                        minimum=1,
                        maximum=100,
                        value=10,
                        step=1,
                    )
                # button_match = gr.Button("Run Matching", variant="primary")

            # mapping setting
            with gr.Column():
                with gr.Accordion("Mapping Settings", open=True):
                    with gr.Row():
                        with gr.Accordion("Buddle Settings", open=True):
                            with gr.Row():
                                self.inputs.mapper_refine_focal_length = gr.Checkbox(
                                    label="Refine Focal Length",
                                    value=False,
                                    interactive=True,
                                )
                                self.inputs.mapper_refine_principle_points = (
                                    gr.Checkbox(
                                        label="Refine Principle Points",
                                        value=False,
                                        interactive=True,
                                    )
                                )
                                self.inputs.mapper_refine_extra_params = gr.Checkbox(
                                    label="Refine Extra Params",
                                    value=False,
                                    interactive=True,
                                )
                    with gr.Accordion("Retriangluation Settings", open=True):
                        gr.Textbox(
                            label="Retriangluation Details",
                        )
                    self.ui.button_sfm = gr.Button("Run SFM", variant="primary")
                self.outputs.model_3d = gr.Model3D(
                    interactive=True,
                )
                self.outputs.output_image = gr.Image(
                    label="SFM Visualize",
                    type="numpy",
                    image_mode="RGB",
                    interactive=False,
                )

    def call_empty(self):
        self.ui.button_sfm.click(fn=self.info, inputs=[], outputs=[])

    def call(self):
        self.ui.button_sfm.click(
            fn=self.sfm_engine.call,
            inputs=[
                self.inputs.matcher_key,
                self.inputs.input_images,  # images
                self.inputs.camera_model,
                self.inputs.camera_params,
                self.inputs.max_keypoints,
                self.inputs.keypoint_threshold,
                self.inputs.match_threshold,
                self.inputs.ransac_threshold,
                self.inputs.ransac_confidence,
                self.inputs.ransac_max_iter,
                self.inputs.scene_graph,
                self.inputs.global_feature,
                self.inputs.top_k,
                self.inputs.mapper_refine_focal_length,
                self.inputs.mapper_refine_principle_points,
                self.inputs.mapper_refine_extra_params,
            ],
            outputs=[self.outputs.model_3d, self.outputs.output_image],
        )
