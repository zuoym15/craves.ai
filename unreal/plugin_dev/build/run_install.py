import os, sys
import opt
from UE4Automation import UE4Automation

# Define the UE4Automation API here
ue4_automation = UE4Automation(engine = opt.UE4)
ue4_automation.build_plugin(
    plugin_descriptor = os.path.join(opt.unrealcv_src_folder, 'unrealcv.uplugin'),
    output_folder = opt.unrealcv_output_folder,
)
ue4_automation.install(
    plugin_folder = opt.unrealcv_output_folder,
)

# ue4_automation.package(
#     project_file = opt.project_file,
#     output_dir = opt.dev_project_output,
# )

# call tox to run test.
