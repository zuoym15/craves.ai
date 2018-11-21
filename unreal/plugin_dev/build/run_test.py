import os, sys, subprocess
import opt
from UnrealcvAutomation import UE4Automation, WindowsBinary

# Define the UE4Automation API here
ue4_automation = UE4Automation(engine = opt.UE4)

# When doing test, there is no need to overwrite previous build, but need to make sure the version I am testing is the version I want to test against.
print('Build plugin')
print('-' * 70)
ue4_automation.build_plugin(
    plugin_descriptor = os.path.join(opt.unrealcv_src_folder, 'unrealcv.uplugin'),
    output_folder = opt.unrealcv_output_folder,
    overwrite = False,
)

print('Install plugin')
print('-' * 70)
ue4_automation.install(
    plugin_folder = opt.unrealcv_output_folder,
    overwrite = False,
)

print('Package UE4 project')
print('-' * 70)
ue4_automation.package(
    project_descriptor = opt.dev_project,
    output_folder = opt.dev_project_output,
)

# call tox to run test.
binary = WindowsBinary(r'D:\temp\dev_project_output\WindowsNoEditor\UnrealcvDevProject.exe')
with binary:
    subprocess.call(
        # ['pytest', os.path.join(opt.unrealcv_src_folder, 'test')]
        ['tox', '-c', os.path.join(opt.unrealcv_src_folder, 'tox.ini')]
    )
