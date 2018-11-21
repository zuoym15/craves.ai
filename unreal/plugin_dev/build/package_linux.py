import sys; sys.path.append('..'); sys.path.append('.')
from UnrealcvAutomation import UE4Automation
import os

ue414 = os.path.expanduser('~/workspace/UE414')
output_folder = os.path.expanduser('~/workspace/UnrealcvTest')
ue = UE4Automation(ue414)
uproject = os.path.abspath('../UnrealcvDevProject.uproject')
ue.package(uproject, output_folder)
