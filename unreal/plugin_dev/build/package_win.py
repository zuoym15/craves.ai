from UnrealcvAutomation import UE4Automation
import os

ue414 = UE4Automation('C:\Program Files\Epic Games\UE_4.14')
uproject = os.path.abspath('../UnrealcvDevProject.uproject')
ue414.package(uproject, 'D:/UnrealcvTest/TestProject')
