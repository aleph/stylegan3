@echo off

echo Tracking Service
start "Tracking Service" "C:\Program Files\Ultraleap\TrackingService\bin\LeapSvc.exe"

timeout 10 > NUL
echo Tracking Control Panel
start "Control Panel" "C:\Program Files\Ultraleap\TrackingControlPanel\bin\TrackingControlPanel.exe"

echo Leap Motion
@REM start "Leap Motion" "C:\Users\aless\Unity\LMT_Headless_3\Leap_Motion_Test.exe"
start "Leap Motion" "C:\Users\aless\Unity\LMT_Build_3\Leap_Motion_Test.exe"


echo MAX
@REM start "Max" "C:\Users\aless\Documents\__Fuse\AB .Interactive\AB_Medulla_MaxMSP\AB.maxpat"


@REM echo Open Frameworks
@REM start "Iris Gallery" "C:\Users\iris\workspace\of_v0.11.2_vs2017_release\apps\_fuse\Iris_screens\bin\Iris_screens.exe"
@REM @REM start "Iris Gallery" "C:\Users\Iris_2\Desktop\Iris_screens\bin\Iris_screens.exe"

@REM echo Ableton
@REM start "Ableton" "C:\Users\iris\OneDrive\Desktop\IRIS_Marble_Monitor_Project\IRIS_Marble_Monitor_Sole_Project\IRIS_Marble_Monitor_Sole Project\IRIS_Marble_Monitor_Sole.als"
@REM @REM start "Ableton" "C:\Users\Iris_2\Desktop\IRIS_Marble_Monitor_Project\IRIS_Marble_Monitor_idroColor_Project\IRIS_Marble_Monitor_idroColor Project\IRIS_Marble_Monitor_idroColor.als"
@REM @REM start "Ableton" "C:\Users\Iris_2\Desktop\IRIS_Marble_Monitor_Project\IRIS_Marble_Monitor_Molecole_Project\IRIS_Marble_Monitor_Molecole Project\IRIS_Marble_Monitor_Molecole.als"
@REM @REM start "Ableton" "C:\Users\Iris_2\Desktop\IRIS_Marble_Monitor_Project\IRIS_Marble_Monitor_acqua_Project\IRIS_Marble_Monitor_acqua Project\IRIS_Marble_Monitor_acqua.als"

echo ok