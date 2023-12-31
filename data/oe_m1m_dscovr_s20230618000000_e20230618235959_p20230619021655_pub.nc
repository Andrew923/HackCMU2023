CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230618000000_e20230618235959_p20230619021655_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-06-19T02:16:55.870Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-06-18T00:00:00.000Z   time_coverage_end         2023-06-18T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx��Ҁ  
(          @5>k���
=��\�9(�C�S3>k���G����dC�@                                     Bx���&  T          @"�\?333���ÿУ��{C���?333��p���Q��Gz�C���                                    Bx����  
�          @%�>�׿�녿��"z�C��>�׿���   �L��C�o\                                    Bx���r  
�          @-p��W
=��ff� ���?ffC��;W
=�����
�j�
C��3                                    Bx��  �          @333��p��޸R�
=q�J{C��R��p���ff����t�C~                                    Bx���  
�          @:=q�
=q��p��G��Mz�C|�q�
=q���\�#�
�v�\Cw�                                    Bx��*d  T          @;�����z��
=�V�
C|������
=�(Q��Cv^�                                    Bx��9
  "          @4zᾞ�R����	���G�C�  ���R��������r=qC���                                    Bx��G�  "          @%�>u��=q����0C���>u���H�
=�[C���                                    Bx��VV  "          @0��?}p���(���
=�ffC�P�?}p���\)�G��;p�C��{                                    Bx��d�  �          @9��?aG��   ��\)�"�C��H?aG���\)�{�I{C�O\                                    Bx��s�  T          @3�
?G����H�����%Q�C��3?G���=q�(��L�C�!H                                    Bx���H  "          @7�?^�R��Q���(\)C��?^�R��ff�  �O{C��H                                    Bx����  {          @6ff?&ff�G�����&�RC���?&ff�У���R�OQ�C��\                                    Bx����  
e          @:=q?B�\�����G��0G�C��H?B�\����ff�WC�"�                                    Bx���:  "          @<(�?�� ����
�2�C��?녿˅����[�C���                                    Bx����  
�          @A�?J=q�33����=qC���?J=q�����9p�C�7
                                    Bx��ˆ  -          @G�?Q��   ��33� Q�C�3?Q��	����(C�o\                                    Bx���,  T          @G�?s33�녿���\)C�J=?s33������<Q�C�P�                                    Bx����  "          @e�?�{�,�Ϳ�
=���HC�s3?�{��	���p�C�N                                    Bx���x  �          @n{?�ff�B�\��(����C�� ?�ff�.{�   ��\C��R                                    Bx��  T          @p  ?��
�DzῚ�H��=qC�\?��
�2�\��  ���C�AH                                    Bx���  �          @qG�?�
=�E�s33�i��C���?�
=�7���  ��=qC���                                    Bx��#j  T          @q�?��I�������C��?��9����������C��                                    Bx��2  
�          @u?���U���G��s33C��?���E������
=C��)                                    Bx��@�  �          @g�?�(��J�H�.{�,(�C�=q?�(��E�@  �?\)C���                                    Bx��O\  "          @vff?�33�]p���\)���C���?�33�Vff�h���Z�\C�!H                                    Bx��^  
Z          @\(�?�\)�H�ý��Ϳ�  C�� ?�\)�Dz�.{�6ffC���                                    Bx��l�  T          @[�?�{�@  >\@�33C�'�?�{�AG��.{�:�HC�{                                    Bx��{N  T          @`��?ٙ��?\)?(��A-G�C�Ф?ٙ��C�
=���?�C���                                    Bx����  
�          @Tz�?�=q�3�
?J=qA[�C��?�=q�9��>��@��C�E                                    Bx����  �          @S33?���*�H?�z�A��C�aH?���8Q�?fffA}G�C��                                    Bx���@  �          @C�
?k��!�?���A���C��?k��0  ?xQ�A��C�G�                                    Bx����  �          @@  ?n{�=q?\A�z�C���?n{�(��?��A�=qC��                                    Bx��Č  
�          @?\)?�G���?�G�A�=qC���?�G��'
=?�ffA��C���                                    Bx���2  �          @B�\?�=q���?�G�A�=qC�  ?�=q�(��?�ffA��C�*=                                    Bx����  
�          @0��?�{��?�B*{C��)?�{��p�?�G�B��C��q                                    Bx���~  �          @{?���Tz�@�Bb{C��=?����
=?���BE��C��                                    Bx���$  
Z          @5?@  ���@.�RB��=C�|)?@  �!G�@*=qB�W
C���                                    Bx���  
�          @)��>��
<�@'�B��H@��R>��
��G�@%�B�\C�
                                    Bx��p  �          @$zὸQ�L��@#33B�Q�CP�3��Q��\@   B���C��q                                    Bx��+  �          @2�\�u�Ǯ@0  B��Cm�3�u�^�R@(��B�(�C~n                                    Bx��9�  
Z          @L(���׿=p�@Dz�B�#�Cm�\��׿��\@8��B��3Cy��                                    Bx��Hb  
�          @`�׿J=q��=q@P  B�W
Ci�J=q���@@  Bkp�Ct^�                                    Bx��W  �          @e��k���@P��B��HCg��k���(�@@  Bc�\CqǮ                                    Bx��e�  
(          @[����Ϳ�=q@N{B��{Cy����ͿУ�@>�RBr=qC��                                    Bx��tT  
�          @]p��L�Ϳ���@N{B�k�C�b��L�Ϳ�  @<��Bl��C��\                                    Bx����  "          @U�8Q쿯\)@AG�B���C�/\�8Q���@.�RB]�C�4{                                    Bx����  
(          @Q녾8Q쿢�\@@��B��3C���8Q���
@/\)BcG�C�q                                    Bx���F  �          @S�
=�Q�u@J=qB�.C��\=�Q쿾�R@<(�B|33C���                                    Bx����  �          @^�R>\)���@O\)B�ffC��f>\)���@@  Bt��C�o\                                    Bx����  T          @^�R��z��p�@@  Bn��C�0���z��{@*=qBGG�C�@                                     Bx���8  
�          @c�
>8Q����@2�\BQz�C�Y�>8Q��&ff@Q�B)�\C��                                    Bx����  �          @l(�>�G��'�@"�\B.�
C��>�G��AG�@33BffC�.                                    Bx���  
�          @z=q>��R�:=q@%�B%�RC��q>��R�Tz�@�\A�(�C��H                                    Bx���*  "          @��\>\�7�@5B1z�C���>\�Tz�@33B

=C�C�                                    Bx���            @}p�=u�S�
@
=qBffC���=u�hQ�?ǮA��C��                                     Bx��v  T          @z�H=L���S33@ffB  C�y�=L���g
=?�  A��HC�o\                                    Bx��$  T          @x��>��Tz�@G�A���C�(�>��g
=?�A�C��                                    Bx��2�  	`          @u���
�^�R?˅A�=qC�� ���
�l��?xQ�Ak
=C��                                    Bx��Ah  
�          @tz�L���dz�?�{A��HC��\�L���p  ?:�HA0��C��{                                    Bx��P  "          @j=q    �XQ�?�z�A��HC�    �dz�?O\)AL��C��                                    Bx��^�  
�          @vff�aG��n�R?k�A]�C�Ff�aG��u�>�=q@�=qC�Q�                                    Bx��mZ  �          @w
=�k��p��?L��A@��C�:�k��u>#�
@�C�E                                    Bx��|   "          @z=q�W
=�s�
?L��A<��C�c׾W
=�x��>\)@C�k�                                    Bx����  
�          @u�>.{�n�R?8Q�A.=qC�H�>.{�r�\=��
?��HC�C�                                    Bx���L  �          @x��?+��^�R?ǮA�=qC�s3?+��l(�?s33AdQ�C�#�                                    Bx����  �          @�ff?B�\�u?�  A�G�C��)?B�\��G�?Tz�A733C�XR                                    Bx����  
Z          @�G�?&ff��Q�?�
=A���C�Q�?&ff��{?333Ap�C�%                                    Bx���>  �          @��
?���(�?�{A���C�k�?�����?(�@��HC�L�                                    Bx����  T          @��?#�
���?}p�AK\)C�33?#�
��
=>��@O\)C��                                    Bx���  �          @��
>���u�u�`  C��{>���hQ��=q��(�C��                                    Bx���0  �          @s33�L���@�����ffC�%�L���%�0  �:\)C��R                                    Bx����  �          @xQ�   �0���*�H�.�C��)�   ���E�SC�Ф                                    Bx��|  �          @u�����!��7��A�C��������G��P  �g=qC���                                    Bx��"  �          @�  =�G��@  �E��6�C��=�G��p��a��\ffC�T{                                    Bx��+�  �          @�ff>��
�<(��Tz��A�C�\>��
�
=�p���f�C��                                    Bx��:n  �          @��R>�(��3�
�[��I=qC�e>�(��{�u�n=qC���                                    Bx��I  
�          @���?8Q��;��L���:z�C���?8Q����hQ��^�\C�~�                                    Bx��W�  T          @��
?5�+��W��Jz�C�z�?5��p���n=qC�n                                    Bx��f`  T          @�\)?+��!��Tz��O
=C�]q?+����H�k��r�RC�`                                     Bx��u  �          @�
=?L���"�\�QG��K=qC�� ?L�Ϳ�p��h���n33C��                                    Bx����  {          @�{?:�H�
=�XQ��W�\C���?:�H���
�n{�zp�C��                                    Bx���R  
�          @�Q�?E�����[��V�HC��?E������qG��yz�C��H                                    Bx����  T          @���?B�\��\�`���]�C�7
?B�\�ٙ��u�C�
                                    Bx����  �          @�ff?:�H�p��_\)�a(�C�.?:�H�У��s33�C�%                                    Bx���D  "          @��
?=p��G��`���i�
C���?=p���Q��q��RC���                                    Bx����  �          @�(�?Q�� ���`���i=qC��?Q녿�
=�r�\
=C��f                                    Bx��ې  �          @�(�?fff�G��^�R�f  C�
=?fff�����p���C��)                                    Bx���6  �          @��H?��Ϳ���_\)�l  C��q?��Ϳ�(��n�R��C���                                    Bx����  �          @�
=>��Ϳ�p��g��s33C��=>��Ϳ���x��.C���                                    Bx���  �          @�{�\)�\)�j�H�_G�C�` �\)�����Q���C���                                    Bx��(  "          @�(��#�
�!G��dz��[
=C���#�
��Q��z�H�~��C�`                                     Bx��$�  T          @�(��L���$z��b�\�W��C�}q�L�Ϳ��R�y���{C�W
                                    Bx��3t  �          @�=q�����#33�_\)�W{C��=���Ϳ�p��u�z�C���                                    Bx��B  
�          @�G��aG��&ff�Z=q�RQ�C����aG���\�qG��u�C��)                                    Bx��P�  �          @�녾\)�%��\(��Tz�C�p��\)�G��s33�w�
C�H                                    Bx��_f  T          @�Q����&ff�W��QG�C�K�����33�n�R�t�C��
                                    Bx��n  �          @�\)�W
=�#�
�W
=�R33C���W
=�G��mp��u=qC�{                                    Bx��|�  
�          @�Q쾅��-p��Q��I{C�5þ�����i���k��C���                                    Bx�X  �          @��׾\)�'��Vff�O�C�� �\)���mp��r�\C�)                                    Bx��  �          @��þ���*=q�U�M{C�4{����Q��l���o�RC��H                                    Bx�¨�  �          @�Q쾊=q�(���Tz��MQ�C���=q�
=�k��o�
C�Q�                                    Bx�·J  T          @��R��\)�   �XQ��Up�C��3��\)���H�n{�w��C��
                                    Bx����  �          @�ff=�G���R�X���W(�C�K�=�G���Q��n{�y��C���                                    Bx��Ԗ  T          @�<��
�p��XQ��W��C�5�<��
���mp��zQ�C�C�                                    Bx���<  �          @��=�Q��p��Vff�V�C��=�Q��
=�k��x�C�ff                                    Bx����  �          @��
<��Q��W
=�Z��C�\)<���{�k��|�RC�w
                                    Bx�� �  "          @��
��Q�����W
=�Z(�C����Q��\)�j�H�|(�C���                                    Bx��.  I          @�����
=�[��]=qC�ٚ�����=q�o\)�~�HC���                                    Bx���  �          @��
���
���W��\=qC�!H���
���j�H�}��C�
=                                    Bx��,z  T          @�33=��
�#�
�L���Mp�C���=��
��
�b�\�o{C��                                    Bx��;   �          @z=q�#�
�\)�=p��G�C��H�#�
���R�\�i(�C�e                                    Bx��I�  -          @vff>8Q��&ff�0���:z�C�f>8Q��
�H�G
=�[��C�l�                                    Bx��Xl  "          @��H>W
=�%�J=q�J(�C�O\>W
=�
=�`  �kQ�C��
                                    Bx��g  "          @|��>�Q�� ���A��H{C��>�Q��33�W
=�h��C��                                    Bx��u�  �          @u>���+��-p��3�C��{>������C�
�S��C��R                                    Bx�Ä^  �          @r�\?��!G��1��<��C���?��ff�G
=�\�C�#�                                    Bx�Ó  �          @�33?   ����QG��T�
C��?   ��33�dz��tC�h�                                    Bx�á�  �          @��?E��'��H���D  C�0�?E��	���^�R�c{C��q                                    Bx�ðP  �          @�33?=p��,���?\)�;�
C���?=p��  �U�Z��C�q                                    Bx�þ�  
(          @~�R>�Q��\)�P  �\=qC��R>�Q��G��a��|�C�Ф                                    Bx��͜  "          @|(�?��=q�C�
�L�C�=q?����H�W
=�kp�C���                                    Bx���B  T          @�G�>��H����L���R33C�Ф>��H��
=�`���q�C�.                                    Bx����  "          @{�>���ff�QG��dG�C��\>���У��a���C���                                    Bx����  
�          @}p�?0�׿��R�e��C�l�?0�׿z�H�p���3C���                                    Bx��4  �          @~{?+���ff�^{�t33C�#�?+���ff�l(���C���                                    Bx���  "          @���>�����(��g
=\)C���>������H�s�
��C��                                    Bx��%�  
�          @���?5���R�j=q�{C���?5�xQ��u�.C�                                    Bx��4&  T          @�  ?��H�����XQ��f�\C�|)?��H����c�
�{��C���                                    Bx��B�  T          @��?�녿˅�\���j
=C��R?�녿�{�hQ���C��H                                    Bx��Qr  "          @|��?�zῷ
=�^�R�xffC���?�z�p���h��=qC�}q                                    Bx��`  
Z          @~{?�Q쿵�Z�H�m�\C���?�Q�s33�e��{C�N                                    Bx��n�  
�          @�Q�?�  ��ff�P���Y=qC�=q?�  �����[��k��C��\                                    Bx��}d  
�          @}p�?�Q��  �P  �\�C�/\?�Q쿆ff�Z�H�offC��                                    Bx�Č
  
Z          @~{@������J=q�R��C�  @��333�Q��^�
C��                                    Bx�Ě�  �          @y��@zῊ=q�G��T��C�+�@z�(���N�R�`p�C�!H                                    Bx�ĩV  "          @x��@G��xQ��J�H�[
=C�0�@G��
=q�QG��e��C�|)                                    Bx�ķ�  
�          @tz�?�(���G��Fff�Yz�C�e?�(�����L���d�HC��                                    Bx��Ƣ  S          @u?�=q�B�\�R�\�kffC��)?�=q�����W
=�s�HC���                                    Bx���H  
Z          @u�?�p��z��Vff�u�
C��{?�p��\)�Y���{�C��                                    Bx����  "          @k�?�ͿxQ��^�RaHC���?�Ϳ��e�=qC�k�                                    Bx���  "          @i��>B�\��  �_\)�{C�N>B�\�
=q�e¡�C��                                    Bx��:  
Z          @l��>�{�xQ��c33��C�� >�{��\�i�� �HC��                                    Bx���  T          @^�R>�p��n{�U�z�C��3>�p��   �Z�H\)C�w
                                    Bx���  
�          @hQ�>���k��_\)Q�C�H>������e�¢�HC��\                                    Bx��-,  
�          @j�H=u�z�H�b�\
=C��f=u���hQ�£�C��=                                    Bx��;�  �          @i�����
����_\)L�C�T{���
�(��e C��
                                    Bx��Jx  
�          @l(��������^{33Cw�R���0���eCl\)                                    Bx��Y  �          @i������  �^{  Cuc׾�����dz�B�Cf��                                    Bx��g�  "          @e��
=�z�H�[�ffCw��
=����aG��
Ch�{                                    Bx��vj  �          @dz�aG��aG��\(�ffC�;aG���ff�a�£��Ct#�                                    Bx�Ņ  "          @a�=��ͿO\)�[�(�C���=��;\�`  §
=C�xR                                    Bx�œ�  �          @aG����
�O\)�Z�HG�C�C׽��
�\�`  §33C�L�                                    Bx�Ţ\  T          @^{�L�Ϳ@  �XQ�  Cc׾L�;�{�\(�§
=Co��                                    Bx�ű  �          @b�\>��
�xQ��XQ�ffC��>��
����^{(�C��                                    Bx�ſ�  "          @`��>����O\)�Y��p�C�
=>��������^{£�qC�e                                    Bx���N  �          @b�\>���Y���Z�H  C�|)>����(��`  £�C�~�                                    Bx����  T          @aG�>\)�O\)�Z�H�C���>\)�Ǯ�`  ¦ffC���                                    Bx���  "          @aG�=��Ϳ=p��\(���C��)=��;��
�`  ¨��C�N                                    Bx���@            @Z=q����
=�U�  Cv\����B�\�W�¨�CX�                                    Bx���  
d          @Vff�.{�   �R�\¡�)Cz���.{��G��U�­
=CS�f                                    Bx���  
�          @S33�L�Ϳ���O\)ffCz
=�L�;.{�Q�ª��C\u�                                    Bx��&2            @QG���녿(���I��Q�CnE��녾�z��Mp�¢B�CW^�                                    Bx��4�  
�          @L�Ϳ�R����Fff��CYY���R��G��H��Q�C>(�                                    Bx��C~            @N�R�Q녾�G��E�CP��Q녽��
�G�k�C9��                                    Bx��R$  ^          @Mp��!G����G
=�\CY33�!G����I��C>��                                    Bx��`�  �          @P�׿+����H��p�CZ8R�+��.{�K�W
CB33                                    Bx��op  �          @N{��\�
=�HQ�Cd���\�k��J�H��CLE                                    Bx��~  
�          @J=q�
=q�+��B�\L�Cg�
=q���
�E�fCR�                                    Bx�ƌ�  "          @@�׿
=q��R�8��L�Cd��
=q��z��<(�k�CPJ=                                    Bx�ƛb  
�          @E���ff��R�>�R{Cj;�ff��z��A�   CTxR                                    Bx�ƪ  �          @H�ÿ
=�.{�@  � CeE�
=��33�C�
CR�)                                    Bx�Ƹ�  
�          @@�׾�G��0���8��=qCm����G���p��<���
C\�                                    Bx���T  
�          @C�
���+��<(�{Ck� ����33�@  G�CY0�                                    Bx����  �          @@  ��G��=p��7
=�Co���G���
=�;�\)C_��                                    Bx���  
�          @9���z�!G��1G��RCc���zᾨ���4z��\CQ��                                    Bx���F  �          @:�H�5��(��2�\\CS��5���4z�p�C=�H                                    Bx���  "          @<�Ϳ
=q��R�4z�\)Cd���
=q���
�7�G�CRaH                                    Bx���  
(          @AG��:�H�Tz��3�
B�Cd�3�:�H���8Q�� CW�\                                    Bx��8  |          @=p���ff�p���0����Ct����ff�&ff�6ff�Ckff                                    Bx��-�  
�          @<(����Ϳs33�1G���C��q���Ϳ(���6ff�RC�y�                                    Bx��<�  T          @;��u�xQ��,���\C�q�u�.{�1�u�C�S3                                    Bx��K*  �          @:=q=L�Ϳ����,(�u�C�=q=L�ͿQ��2�\L�C���                                    Bx��Y�  "          @<��<���p��*�Hk�C�� <��s33�1�8RC��\                                    Bx��hv  �          @8Q�u�������l�HC��H�u��Q��$z�#�C�|)                                    Bx��w  "          @:�H�
=q��=q�Q��[��C{�
=q��=q�!��qffCw�=                                    Bx�ǅ�  "          @=p��fff�޸R����AQ�Cr��fff��G����U�Co�                                    Bx�ǔh  T          @=p��Tz�����	���:�HCu�ÿTz��\)�z��O�\Cr�\                                    Bx�ǣ            @=p���녿�  �ff�Rp�C�]q��녿�G��!G��h�\C~��                                    Bx�Ǳ�  
�          @8Q��녿�
=�=q�iffC~���녿�
=�"�\�
=Cz��                                    Bx���Z  
�          @>�R�p�׿У���\�JG�Cp.�p�׿�33�(��]ffCl@                                     Bx���   
�          @J�H�(�ÿ�p��$z��X�\Cy33�(�ÿ�p��.{�m=qCu�3                                    Bx��ݦ  T          @K��:�H�޸R�#33�U��CwE�:�H���R�-p��j
=Cs�)                                    Bx���L  	�          @J=q�^�R�ٙ�� ���R�HCr�)�^�R�����*=q�f(�Co�                                    Bx����  T          @K��aG���
=�#33�U��Crh��aG����,���h�
Cnk�                                    Bx��	�  "          @L(���  ���
�(��H{CpǮ��  ��ff�&ff�Z�RCm&f                                    Bx��>  "          @K���\)��G�����D  Cms3��\)���
�#�
�U�
Ci�f                                    Bx��&�  
�          @K���녿�  ����C�HCl�q��녿\�#33�Uz�Ci.                                    Bx��5�  �          @H�ÿ����z����?  Cg�q�����Q��{�O=qCc��                                    Bx��D0  
�          @I�����R��\)�Q��E�\Ch}q���R����!G��U�
CdB�                                    Bx��R�  
�          @A녿�z�˅���Dz�Ci޸��z῰����H�U  Ce�
                                    Bx��a|  �          @Dz῝p�����G��?Ci+���p���
=�=q�P  CeJ=                                    Bx��p"  
�          @I���}p��������C(�Cqn�}p������!G��U
=Cn5�                                    Bx��~�  �          @G
=�����z��p��533Cp�ῇ����H���F�Cn#�                                    Bx�ȍn  "          @L�Ϳ�����H��<\)Ch
=�����  ��R�L  CdG�                                    Bx�Ȝ  T          @N{���׿�z�����?Q�Cf5ÿ��׿����!G��N\)Cb=q                                    Bx�Ȫ�  �          @Mp���p���33���9�RCd��p���Q��p��H(�C`�                                    Bx�ȹ`  T          @N�R��z�ٙ��ff�;�Cf\)��z῾�R�\)�J  Cb��                                    Bx���  "          @Mp����Ϳ�33����A�RCf�׿��Ϳ�Q��"�\�P�Cb�R                                    Bx��֬  "          @O\)���Ϳ޸R���<�\ChJ=���Ϳ��
� ���K�Cd��                                    Bx���R  �          @P�׿�  �޸R����C�Cj0���  �\�%�R�HCf��                                    Bx����  |          @N{���H��  ����3
=Cf33���H�Ǯ����Az�Cb�{                                    Bx���  �          @Mp�����޸R���;�Chuÿ������{�I�Ce�                                    Bx��D  �          @P�׿�G���{�
=�9Q�Cl���G���z��   �HCh�                                    Bx���  
�          @L(���ff�������E{Co����ff��=q�"�\�U
=Cl�                                    Bx��.�  
d          @Fff�
=q�޸R��R�VC|�=�
=q����'��hp�Cz��                                    Bx��=6  "          @O\)�k�����5��u{C����k���ff�<����C�                                      Bx��K�  T          @U���Q��\�333�d�\C�=q��Q����;��vp�C�h�                                    Bx��Z�  �          @Vff�#�
��p��7��k�
C���#�
���R�?\)�~  C��                                     Bx��i(  T          @Z�H���Ϳ޸R�<(��mz�C�lͽ��Ϳ�  �Dz��p�C�.                                    Bx��w�  "          @XQ쾀  ���4z��b��C�!H��  ��{�<���t=qC���                                    Bx�Ɇt  
�          @Z=q���
���6ff�b�HC�׾��
��\)�>�R�t=qC�W
                                    Bx�ɕ  �          @Y�������:�H�mQ�C~�=����Q��A��~�C|=q                                    Bx�ɣ�  �          @U�����5�i�C|��������=p��y�\Cz{                                    Bx�ɲf  "          @P�׾��R��33�333�lffC������R��
=�:=q�}=qC��                                    Bx���  �          @S33���R�޸R�2�\�fC������R�\�:=q�wz�C�8R                                    Bx��ϲ  �          @S�
���Ϳ�33�5�l��C�*=���Ϳ�
=�=p��}�C~aH                                    Bx���X  T          @QG��
=q��G��5�rQ�Cz&f�
=q��ff�<���Cw:�                                    Bx����  �          @S33�:�H����4z��k\)Ct�{�:�H��=q�:�H�z{Cq8R                                    Bx����  �          @U��������7��o�HC{E����{�>�R�G�Cx��                                    Bx��
J  �          @[�����Q��Fff�C�Z����(��L��8RC�<)                                    Bx���  �          @Y��<#�
��(��;��n�C�+�<#�
��G��B�\�~��C�0�                                    Bx��'�  
�          @Y���������@  �w��C���������Fff�
C�u�                                    Bx��6<  T          @U���Q쿽p��>�R�~G�C�/\��Q쿢�\�Dz���C���                                    Bx��D�  T          @QG��W
=�����5��r�HC�Ff�W
=�����;�#�C��                                     Bx��S�  
�          @P  ��33��ff�<����C~�׾�33�����A�33C|J=                                    Bx��b.  "          @E��������7
=�C�5þ���fff�;�� C}�f                                    Bx��p�  �          @J�H�#�
�p���AG�� C��)�#�
�=p��E�C�e                                    Bx��z            @I��>�=q���\�=p��RC�s3>�=q�Tz��AG��HC��                                    Bx�ʎ   �          @E>�Q쿯\)�.�R�z  C�\)>�Q쿘Q��3�
�C�ff                                    Bx�ʜ�  �          @9��=��
���R�'�ffC�� =��
�����,(��C�+�                                    Bx�ʫl  �          @\)���Ϳ��
�  ffC�>����Ϳc�
��
�{C��                                    Bx�ʺ  �          @3�
��녿��H�\)�{p�C{c׾�녿�ff�#�
p�Cx�
                                    Bx��ȸ  T          @7��0�׿����$z��~��CmͿ0�׿h���(Q�(�Ch��                                    Bx���^            @=p��8Q쿊=q�)����
Cl\�8Q�h���-p��\Cg��                                    Bx���  
�          @C33���R�L���p��]��CPY����R�&ff�   �d  CK�R                                    Bx����            @E����(����
�B�RCF�R�����ff�FCC33                                    Bx��P  ,          @AG���
�8Q���\�,33CG5���
������0G�CD(�                                    Bx���  �          @<���G�������3�HCA)�G���33�ff�6�\C=��                                    Bx�� �  �          @8���
=q�B�\��z��%ffC8���
=q���
���&33C6                                    Bx��/B  T          @'
=�
=>8Q��  �ffC.�q�
=>�=q���R�
=C,�R                                    Bx��=�  "          @(���>�����R��
C/ff��>u��p���C,��                                    Bx��L�  
�          @)���{>.{���33C/� �{>u��z��
=C-�                                     Bx��[4  �          @+��޸R��G���33�:�CBE�޸R��{���<�
C?�                                    Bx��i�  J          @:=q�u�����%��y\)C���u���H�)����C�}q                                    Bx��x�  �          @8Q�    �����#33�z�C���    ��
=�'
=Q�C��=                                    Bx�ˇ&  �          @�R�n{������
=�O�
ChB��n{������R�Y(�Ce�                                     Bx�˕�  �          @�\��ff�k��ٙ��J{C]B���ff�W
=��  �Q��CZh�                                    Bx�ˤr  
�          @
=>�\)��
=� ���k�RC��f>�\)��=q�z��w  C�C�                                    Bx�˳  
�          @z�>�p��xQ���\�|ffC���>�p��\(��� C���                                    Bx����  
�          @z�>�  ���H�����f��C�� >�  ��{� ���q�
C�`                                     Bx���d  �          @   >�G���\)� ���Y�
C���>�G����\�z��dffC��3                                    Bx���
  �          @=q>��������R�`C���>�����H�33�k�C���                                    Bx����  �          @p���\)��33����j�C�` ��\)�����
=�t�C�<)                                    Bx���V  �          @�<��
���
��G��W��C��\<��
��Q���b��C���                                    Bx��
�  T          @�.{���\��{�N=qC�33�.{��Q���X��C��
                                    Bx���  �          ?�(��aG���G���z��gC��H�aG��p�׿ٙ��q��C�g�                                    Bx��(H  �          ?�{?zῂ�\���I�\C��)?z�u��(��RG�C��=                                    Bx��6�  J          @33?�G���(����
�{C�?�G���zΎ��#�RC�o\                                    Bx��E�  �          @33?��������C�
=?����{��{�%��C��H                                    Bx��T:  T          @{?���������
��RC���?����녿˅��HC�p�                                    Bx��b�            @�?�G���{�����G�C�ff?�G������Q���
C��f                                    Bx��q�  ,          ?���?333��  ��R��C���?333�xQ�(�����HC���                                    Bx�̀,  �          ?�  ?(�ÿ�녿+���p�C��q?(�ÿ�{�8Q���33C�L�                                    Bx�̎�  �          ?�
=?�=q���\��(��z�C�E?�=q�z�H��G��#�\C��
                                    Bx�̝x  
�          @!�?ٙ���  ��z��=qC�Ф?ٙ���Q쿺�H�ffC�n                                    Bx�̬  T          @G�?������
��
=��  C�=q?�����p���p���C��H                                    Bx�̺�  
�          ?���?��H���E��ď\C�  ?��H��녿O\)��=qC�]q                                    Bx���j  
�          ?�\?�z῅��Tz����C�H?�z῁G��^�R��{C�n                                    Bx���  "          @�?��\���
�p���݅C�` ?��\���R�}p���
=C���                                    Bx���  "          @�?�p����H�+����\C�S3?�p���Q�5����C���                                    Bx���\            @33?�녿��H���@Q�C���?�녿������Q�C��                                    Bx��  ,          @p�?�p���{���C�C�� ?�p�����\)�T��C���                                    Bx���  "          @��?�\)���Ϳ�\�F�\C��?�\)��=q����X  C�8R                                    Bx��!N  �          @  ?�ff���\��G��3�C�l�?�ff��  ���Dz�C��{                                    Bx��/�  "          @?У׿�
=��(��@z�C��?У׿�����QG�C�,�                                    Bx��>�  T          @-p�@ff�����!G��VffC�� @ff��
=�(���c
=C���                                    Bx��M@  "          @/\)@�Ϳ�33��ff��C�t{@�Ϳ��׾��!p�C���                                    Bx��[�  T          @&ff@�\���\�!G��aC�  @�\��  �(���lz�C�1�                                    Bx��j�  T          @1G�@'��@  =�\)?��C���@'��@  =L��?���C���                                    Bx��y2  �          @0  @!G���R?h��A���C�@!G��&ff?fffA��C��=                                    Bx�͇�  "          @*�H@��k�?fffA�33C�1�@��s33?^�RA���C���                                    Bx�͖~  
�          @+�@��s33?��HAי�C�@ @��z�H?�
=A���C��                                    Bx�ͥ$  
�          @\)?��Y��?�ffA�ffC�\?��aG�?��
A���C���                                    Bx�ͳ�  ^          @�H?�(��}p�?�  A���C��=?�(���G�?xQ�A��C�b�                                    Bx���p  "          @��?�\)�p��?�Au�C��\?�\)�s33?��Ak33C��                                    Bx���  
�          @\)@ �׿\(�>�
=A.{C�e@ �׿^�R>���A%C�G�                                    Bx��߼  
Z          @@{���H?�AMC��H@{�   ?�\AIG�C��)                                    Bx���b  �          @=q@�R<�?Y��A��?Q�@�R<#�
?\(�A�>�{                                    Bx���  �          @{?�ff?=p�?��\A�(�A��?�ff?5?��A�A���                                    Bx���  
�          @33?˅?W
=?uA�Q�Aݙ�?˅?Q�?z�HA��A��                                    Bx��T  
�          ?�33?�Q�>�{?��B&Q�A|��?�Q�>��R?���B'z�Am��                                    Bx��(�  T          ?�33?
=q��?޸RB�� C��)?
=q�   ?�p�B�\C��                                    Bx��7�  T          @�>��H��G�?�B�  C�)>��H���?�z�B��{C��                                    Bx��FF  
�          @��>��u@Q�B��\C�N>���\)@Q�B���C��\                                    Bx��T�  
�          ?��>���G�?���B��C���>��\)?�B�33C���                                    Bx��c�  ,          ?�Q�?�R�B�\?���B��qC�n?�R�aG�?���B�(�C�:�                                    Bx��r8  
Z          @�>�=q�k�?�(�B���C�E>�=q���?�(�B�u�C��H                                    Bx�΀�  
�          @\)���>8Q�@	��B�u�C �\���>��@
=qB��3C
=                                    Bx�Ώ�  
�          @\)����?W
=@ ��B�W
B�8R����?O\)@G�B���B�u�                                    Bx�Ξ*  �          @&ff�
=q?��@{B���CͿ
=q?�@{B��C��                                    Bx�ά�  
�          @�R���?B�\@�
B�ǮB��þ��?:�H@z�B���B��                                    Bx�λv  
�          @��\)?�@  B��=B�{��\)?
=q@  B��RB�q                                    Bx���  T          @녿Q�?fff?�33BD�C�{�Q�?aG�?�z�BFQ�C�                                    Bx����            ?����p��?��H?�{BG�B����p��?���?���B�C 
                                    Bx���h  
�          ?�33����?Y��?fffA�=qC� ����?Y��?h��A�\C                                      Bx���  �          ?�(��J=q?�\)?���B/�\B���J=q?�{?�=qB1(�B�\                                    Bx���  T          @�\���R>�>�
=A=G�C08R���R>�>�
=A=C0O\                                    Bx��Z  
�          ?�p�����>L��>�=q@�\)C.�����>L��>�=q@���C.\                                    Bx��"   �          ?���\)>�33>L��@��HC)s3��\)>�33>L��@��C)}q                                    Bx��0�  �          ?�Q���>�=q>W
=@�C+�Ϳ��>�=q>aG�@�
=C+�
                                    Bx��?L  T          ?��Ϳ��H?#�
?\(�A�z�C^����H?!G�?^�RA�p�C�                                    Bx��M�  �          ?�녿���?^�R?xQ�A�C(�����?^�R?z�HA���CG�                                    Bx��\�  �          @�
��=q?\(�>�\)@��C�R��=q?\(�>�\)A   C�                                     Bx��k>  	�          @33���?G�=�G�@Dz�Cn���?G�=�G�@HQ�Cp�                                    Bx��y�  �          @�\��=q?c�
��Q��{C(���=q?c�
��Q���HC&f                                    Bx�ψ�  �          @z����?aG�����6�RC8R����?aG�����6{C5�                                    Bx�ϗ0  �          ?�׿��H?.{��Q��1��Ck����H?.{��Q��1p�Ck�                                    Bx�ϥ�  �          ?���\?0�׾L����  C�H��\?0�׾L����  C�H                                    Bx�ϴ|  
�          ?�33���
>\�������C'�
���
>\������RC'ٚ                                    Bx���"  
�          ?�녿��H?�������\C �����H?��������C ��                                    Bx����  �          ?�Q����>�(���p��N{C$������>�(���p��N�\C$�f                                    Bx���n  |          ?����H>�G���ff�g�C%�����H>�G����hQ�C%��                                    Bx���  
d          @G����H?
=�O\)���\C �q���H?
=�O\)���C!{                                    Bx����  T          @(���
?5��������C!  ��
?333��{�י�C!!H                                    Bx��`  T          @   �Q�?G���ff��=qC���Q�?E���ff��
=C )                                    Bx��  T          @���	��?=p��h����G�C!�	��?:�H�k���(�C!&f                                    Bx��)�  
�          @\)����?+��\(���  C!�����?+��^�R����C!(�                                    Bx��8R  
�          @�ÿ���?0�׿W
=���C� ����?.{�W
=���C��                                    Bx��F�  T          ?��R��33?#�
�h����C���33?!G��k���
=C5�                                    Bx��U�  T          ?�ff����?\)��ff��CY�����?�Ϳ����HC��                                    Bx��dD  T          ?�\)���>aG��fff�Z33C�����>W
=�fff�Z�C :�                                    Bx��r�  �          ?k�<����
�fff¯k�C��<����fff®��C��R                                    Bx�Ё�  	`          ?��Ϳ��>�{�0����(�C"^����>��ÿ0�����C"��                                    Bx�А6  "          ?�=q���\>�׿5� �Cn���\>��5�C�\                                    Bx�О�  
�          ?�����
?��\)��\)C�q���
?������C
=                                    Bx�Э�  	.          ?��׿��>�׿:�H�=qCJ=���>��=p��ffC��                                    Bx�м(  "          ?�\)��G�>Ǯ�W
=��RC� ��G�>\�W
=��
CQ�                                    Bx����  
�          ?�(���Q�>�(��=p����C !H��Q�>�
=�@  ��\)C �
                                    Bx���t  
�          ?�G���  ?   �E���\)C!� ��  >��G��хC"(�                                    Bx���  	�          ?�Q쿱�?z�@  ��G�CO\���?녿B�\��{C��                                    Bx����  T          ?޸R��(�?zῃ�
�C\)��(�?녿��
�\)C                                    Bx��f  
�          ?�=q���>�
=�
=��(�C.���>�녿����
=C��                                    Bx��  T          ?녽#�
>��
��G���(�B�k��#�
>��
����ffB                                    Bx��"�  "          ?���=��
�B�\��
=�=C�/\=��
�L�;�
=�}�C��f                                    Bx��1X  
�          ?�Q�?333�Tz῞�R�C��C�<)?333�Y����p��@33C��{                                    Bx��?�  "          ?�(�?=p���R��ff�U��C�R?=p��#�
����R��C���                                    Bx��N�  "          ?�p�?k���=q�aG���G�C�` ?k����ͿY����\)C�1�                                    Bx��]J  T          ?޸R=�Q��׿�33�=C��R=�Q��\����C�5�                                    Bx��k�  �          ?�=q?O\)���
���\�0�C��?O\)�����  �,z�C���                                    Bx��z�  
�          ?�p�?
=�z�H��  �R��C���?
=��G���p��Np�C�{                                    Bx�щ<  �          @(�?&ff��{��
�g�C�+�?&ff��z����c  C��                                    Bx�ї�  
�          @�R?�G���  ����E��C�p�?�G���������A��C���                                    Bx�Ѧ�  "          @(�?�
=��33��{�%ffC�%?�
=��Q��=q�!  C��                                     Bx�ѵ.  T          @��?�(��Ǯ��
=�ffC�\?�(��˅����
C��q                                    Bx����  
�          @�?�  ��p���
=��HC�{?�  �\����33C���                                    Bx���z  T          @�>�Q쿱녿ٙ��G  C�B�>�Q쿸Q��z��A  C�
=                                    Bx���   T          @��?�p����H��z����C��?�p����R��\)��C���                                    Bx����  T          @?�\��
=�z��b�HC��3?�\��Q���R�\C�l�                                    Bx���l  
�          @Q�?���녿��MG�C�l�?���z���=�C�G�                                    Bx��  ^          @�?�׿�
=����Q�C�Z�?�׿�����\�@z�C�4{                                    Bx���  
(          @�?��H��(��@  ����C��?��H���R�8Q����C���                                    Bx��*^  T          @p�?�p������Tz����C�j=?�p���(��J=q��  C�&f                                    Bx��9  "          @=q@ff�}p���R�o�C�^�@ff��G��
=�c
=C�'�                                    Bx��G�  �          @   @�\�aG���(���
C��f@�\�c�
������C�`                                     Bx��VP  
�          @(�@
=�녽#�
�^�RC�*=@
=�zἣ�
��
=C�'�                                    Bx��d�  
�          @!�@=q�&ff��Q��33C��H@=q�(�þ�{��{C�`                                     Bx��s�  T          @'
=@#�
>�>���A=q@;�@#�
>\)>���Az�@O\)                                    Bx�҂B  !          @#33@ �׾�\)=��
?�(�C��
@ �׾�\)=�Q�?�Q�C��                                     Bx�Ґ�  �          @!�@(��녾�=q��\)C�k�@(��zᾀ  ��Q�C�O\                                    Bx�ҟ�  �          @'
=@%����
������C�)@%���Q쾙���ӅC��)                                    Bx�Ү4  
�          @!�@!G�<�<#�
>L��?&ff@!G�<�<#�
>.{?(��                                    Bx�Ҽ�  "          @�@�þk�>aG�@��\C�K�@�þaG�>k�@���C�ff                                    Bx��ˀ  �          @�@
=q>�Q�?��
A�ffA�H@
=q>���?�G�A�\)A'33                                    Bx���&  
�          @\)@
=q>���?�33A�33A)p�@
=q>�ff?��AٮA<                                      Bx����  �          @7�@�R?   ?��A�z�A5�@�R?��?��Aڣ�AH��                                    Bx���r  �          @
=?�Q�>��R?��\BG�A�?�Q�>�Q�?�G�B�A)��                                    Bx��  �          @Q�?�z�>L��?k�A�  @��H?�z�>k�?h��A��
@�
=                                    Bx���  "          @z�?�=�Q�?n{Aי�@*�H?�>�?n{A�z�@y��                                    Bx��#d  �          @��?�=q>B�\?�=qA�
=@���?�=q>k�?���A�R@�=q                                    Bx��2
  �          @�?�{��>�\)Az�C�w
?�{�\)>��RA(�C��                                    Bx��@�  
Z          ?�{?޸R�\)?(�A��RC��q?޸R��G�?�RA�(�C�0�                                    Bx��OV  "          @
�H@z�<#�
?�At��>���@z�=#�
?\)At(�?��                                    Bx��]�  T          @��?У׿��Ϳ��N�RC�?У׿�\)��ff�1G�C��\                                    Bx��l�  T          @ ��>���ff���
��33C�}q>���
=q��
=��{C�e                                    Bx��{H  "          @*=q����
=��=q�-�C������G��޸R�"�HC���                                    Bx�Ӊ�  T          @5��(����{�#��C�%��(�����  �=qC�^�                                    Bx�Ә�  �          @7
=���H����{�"�C�p����H�p���  �Q�C���                                    Bx�ӧ:  T          @3�
�u�Q���!{C�33�u�{�ٙ���
C�<)                                    Bx�ӵ�  �          @4z�>�=q��ÿ����C���>�=q�{�ٙ���C��                                     Bx��Ć  
Z          @8Q�>��ÿ��H�z��8�C��f>����z��(��-�C���                                    Bx���,  T          @3�
�W
=�������R�5�\C�箾W
=�33����)�C�                                    Bx����  �          @/\)��\������-=qCͿ�\�G����
�!C�R                                    Bx���x  
Z          @,�Ϳ���{��\)�0p�C~k������H��\�$�HC&f                                    Bx���  
�          @-p��(���{����/��C{��(��������
�$
=C|��                                    Bx���  
�          @,�Ϳ#�
���H��p����C{�#�
�33��\)�G�C|�                                    Bx��j  "          @/\)���R��(���\)�-{C�����R��
��G�� C���                                    Bx��+  "          @>�R�aG����У��=qCyO\�aG��=q��  ���HCy�q                                    Bx��9�  	�          @0�׾�Q���\���$��C��{��Q��Q��Q��{C�.                                    Bx��H\  "          @�\?��׿
=��ff�JffC�,�?��׿.{��G��D{C���                                    Bx��W  �          ?���?���>��
�p����A8z�?���>�=q�u��ffA�                                    Bx��e�  "          @?�(�?!G��^�R��Q�A��?�(�?z�fff����A���                                    Bx��tN  "          @\)?�=q>�33���H��RA.�R?�=q>�\)��p��  AQ�                                    Bx�Ԃ�  
�          @33?�  ��p���z���C��
?�  ���������C���                                    Bx�ԑ�  T          @G�?������\��
=�,��C�#�?�����ff��Q��\)C��                                     Bx�Ԡ@  
�          @��?����R?&ffA
=C���?�����?:�HA�=qC��                                    Bx�Ԯ�  
�          @�?�\)�\(�?���A��C��f?�\)�J=q?�33A��
C��f                                    Bx�Խ�  
�          ?�׿s33�u?���BW=qCB��s33�#�
?�{BZ{C=W
                                    Bx���2  "          ?��H�˅���
?
=qA�ffC6ٚ�˅�L��?��A��C5�H                                    Bx����  �          ?�ff��  �.{?\(�A��C:aH��  ��G�?^�RA�Q�C8Q�                                    Bx���~  �          ?У׿�
=����?��B!Q�CCs3��
=���?��B$�
C@h�                                    Bx���$  �          ?��
���
���?W
=A�z�C=����
�L��?Y��A��HC;�=                                    Bx���  �          ?�33��\)�#�
?�  B){C:���\)���
?�G�B*Q�C7#�                                    Bx��p  �          ?�
=��ff>�z�?�ffB[��C$\)��ff>Ǯ?��
BWQ�Cc�                                    Bx��$  T          ?�p���p�>�?�B�{C �;�p�?z�?�ffB��B�#�                                    Bx��2�  �          ?У׿=p�>��?�\)Bi�HC�H�=p�?   ?��Ba�HC�=                                    Bx��Ab  T          ?��\���
>�\)?z�A���C$�ῃ�
>��
?\)A�Q�C"                                    Bx��P  "          ?���z�H>�\)>�@�{C$(��z�H>�z�=�G�@˅C#��                                    Bx��^�  �          ?xQ�L��>�����{���
C#׿L��>�\)��Q����\C ��                                    Bx��mT  �          ?�p��p��>�p��&ff�G�C�)�p��>��
�.{�C!�                                    Bx��{�  �          ?��
��ff>��\(��v�C#���ff=�\)�^�R�zffC+�                                    Bx�Պ�  �          ?�G���
=>8Q�c�
�|{C���
==�G��fff��C$��                                    Bx�ՙF  
�          ?��=��;��\§�)C�k�=��;k���G� �
C���                                    Bx�է�  �          ?��
?�;�{����C��{?�;����33C�*=                                    Bx�ն�  �          @�R?�
=����p����Q�C���?�
=��z�\(���  C��H                                    Bx���8  T          @!�@���{>u@��C�p�@���=q>�{@��
C��f                                    Bx����  "          @.{@
�H���þ�{���
C�
=@
�H���;aG���z�C��q                                    Bx���  �          @4z�?�zΰ�
��\)��
C�q?�z῱녿\��C��                                    Bx���*  �          @?\)@�ÿ\�����ݮC�K�@�ÿ�\)��G���p�C�o\                                    Bx����  
�          @B�\?�(���녿У���\C�'�?�(���G���  ���C�+�                                    Bx��v  T          @B�\?�p��\��p��
�RC�,�?�p���33��{���C��                                    Bx��  �          @E�@
=q��33��Q��Q�C�|)@
=q���
��=q����C�Y�                                    Bx��+�  "          @J=q@(����R�Ǯ��{C���@(����Ϳ��H��p�C�xR                                    Bx��:h  "          @P��@0  �u���H��z�C�j=@0  ���ÿ�����=qC�g�                                    Bx��I  
�          @N�R@'�����\���
C��3@'����H��
=��p�C���                                    Bx��W�  "          @L(�@����\��Q�� ��C��f@���33�˅��p�C��R                                    Bx��fZ  
�          @Mp�@)���xQ�\��RC��@)�������Q��ՅC�Ф                                    Bx��u   
�          @L��@ �׿h�ÿ�\�p�C��@ �׿�ff��Q����C���                                    Bx�փ�  �          @L(�@{��G���  �33C��{@{��33�����C�}q                                    Bx�֒L  �          @N{@�ÿfff������RC��f@�ÿ�������C�                                    Bx�֠�  "          @N{@�׿E��
=q�(�C��3@�׿p����"
=C��                                    Bx�֯�            @L(�@\)�fff��#p�C��@\)�����   ��
C�>�                                    Bx�־>  
�          @Mp�@�ÿ�G���\��C���@�ÿ�
=��
=��\C�q                                    Bx����  T          @J�H?�(���(��
=q�,  C�(�?�(�����33�!p�C�aH                                    Bx��ۊ  T          @O\)@(��J=q�  �033C�
=@(��z�H���(��C���                                    Bx���0  T          @P  @z�333�
�H�'ffC��=@z�c�
�ff�!�C��=                                    Bx����  
�          @R�\@��(����!  C���@��J=q��
���C�
=                                    Bx��|  
�          @Tz�@ �׾����z�C���@ �׿(���z��(�C���                                    Bx��"  �          @XQ�@   ��G���R�%(�C��@   �!G����!
=C��                                    Bx��$�  T          @Tz�@���33�z��2\)C��f@�������.��C�ff                                    Bx��3n  �          @E?�׿��\�33�(p�C��q?�׿�Q��
=��C�>�                                    Bx��B  �          @AG�?ٙ����R��p��%(�C�]q?ٙ���z���p�C��{                                    Bx��P�  
�          @E�?��R��
���\)C�y�?��R�{��p��C��=                                    Bx��_`  "          @H��?�33����
�H�.��C�?�33��(��   �=qC���                                    Bx��n  T          @K�?�녿�ff�  �3�C�O\?�녿޸R�ff�%Q�C���                                    Bx��|�  T          @K�?У׿�{���.{C��)?У׿�ff�G��{C�                                    Bx�׋R  
�          @O\)?�{��{�33�5
=C�y�?�{������%�
C���                                    Bx�י�  "          @P  ?��ÿ�33��R�0Q�C�(�?��ÿ�����"��C�P�                                    Bx�ר�  �          @P  @p���\)�� �C��3@p������p���\C��{                                    Bx�׷D  T          @Mp�@���p����%Q�C��3@����   ��C��3                                    Bx����  
�          @S�
@�ÿ�ff�G��.\)C��@�ÿ�G��
�H�$=qC��f                                    Bx��Ԑ  T          @Q�@�
�����\)�,��C��f@�
��33���!Q�C�޸                                    Bx���6  T          @Mp�@ �׿+���H�CG�C���@ �׿fff�ff�;ffC��3                                    Bx����  �          @H��@z�������9��C��
@z�Q��(��2�C�9�                                    Bx�� �  �          @HQ�@	���333�
=q�.�C��\@	���fff���'
=C���                                    Bx��(  T          @Fff@�׿\)���$�C��@�׿@  ��(��
=C��q                                    Bx���  
�          @H��@{��R���*�RC�7
@{�Q��33�#�C��)                                    Bx��,t  "          @J=q@{�!G��	���+�HC�\@{�W
=���$�C���                                    Bx��;  
�          @H��@z�h���
�H�/��C�q@z῏\)�z��%��C�˅                                    Bx��I�  
�          @Fff?�(��p�����3�HC�>�?�(���33�z��)Q�C�޸                                    Bx��Xf  �          @E?У׿z�H�Q��M�C��?У׿��H���A�C���                                    Bx��g  �          @E�@�Ϳ:�H��p��"�C�Ф@�Ϳk���33��C��q                                    Bx��u�  
Z          @C�
@�R�=p����  C��@�R�k����
=C��                                    Bx�؄X  �          @Dz�@33���	���5�HC��3@33�@  ��/
=C��q                                    Bx�ؒ�  �          @C33?���>�33�Q��N��A)G�?���=����=q�Q�R@Mp�                                    Bx�ء�  "          @A�?�p��L�Ϳ��H�2Q�C��?�p����Ϳ�
=�.�HC�T{                                    Bx�ذJ  �          @L��@�Ϳ�=q��p���  C���@�Ϳ�p��������C�w
                                    Bx�ؾ�  T          @I��@
=�\�����Q�C��)@
=��z῝p���(�C�~�                                    Bx��͖  "          @Fff@�
��  ��z��مC�o\@�
��녿�  ���HC�E                                    Bx���<  �          @H��@(��\�Ǯ��\C��R@(���
=����֣�C�K�                                    Bx����  T          @K�@�R��Q쿸Q��مC�p�@�R��=q��G���{C�T{                                    Bx����  T          @Mp�?�=q�
=��
=����C�� ?�=q�{�n{��z�C�>�                                    Bx��.  
�          @N�R@�
������  ��
=C�=q@�
�ff�����G�C�4{                                    Bx���  
�          @P  @33���
�   �ffC�� @33��p���=q�	\)C���                                    Bx��%z  �          @N{@%����������C���@%�������p����HC�^�                                    Bx��4   �          @QG�@7
=���
��
=��
=C�{@7
=��33������C��                                    Bx��B�  
�          @Q�@%���ÿ���  C��=@%���R��ff��{C�4{                                    Bx��Ql  "          @QG�@�
��(�������C��@�
����ff�z�C�7
                                    Bx��`  
Z          @Z=q?У׿���1��\�\C���?У׿���*=q�N
=C�\)                                    Bx��n�  T          @Z=q?�G��h���9���j��C���?�G���(��1��\ffC���                                    Bx��}^  T          @X��?\�p���7��hG�C�9�?\���R�0  �Y��C�h�                                    Bx�ٌ  
�          @Y��?��H�@  �E�C�  ?��H����?\)�u�C�                                      Bx�ٚ�  �          @[�?}p��(���L��#�C�E?}p���  �G
=�C�]q                                    Bx�٩P  
�          @^�R?�ff�=p��N{�=C�c�?�ff����G�B�C���                                    Bx�ٷ�  
�          @^�R?Y���.{�S�
��C�Ǯ?Y������Mp��C��\                                    Bx��Ɯ  �          @^{?G��&ff�Tz�{C��?G���G��N�R��C��3                                    Bx���B  "          @`��>�׿+��Z=q�C�ff>�׿�ff�S�
�qC�                                    Bx����  
�          @^�R>W
=�+��Z=q��C���>W
=����S�
W
C��                                    Bx���  
Z          @`  >Ǯ�O\)�W�.C���>Ǯ��
=�P��33C�1�                                    Bx��4  "          @`  ?��Y���Vff��C��3?���(��N�RG�C��R                                    Bx���  �          @e�>�G��J=q�\���RC��>�G����U�fC�T{                                    Bx���  �          @h��=��
�\)�e¢C�,�=��
�xQ��`  
=C�p�                                    Bx��-&  
(          @hQ���<��
�dz�¢� C2��;Ǯ�c33�CW\)                                    Bx��;�  �          @j=q��������h��¨�{Cuz����@  �e���C�G�                                    Bx��Jr  
(          @k�>.{���hQ�¢�\C��>.{�p���c33�RC�5�                                    Bx��Y  �          @j�H>�=q�!G��g
=u�C��>�=q����`���HC�]q                                    Bx��g�  T          @j�H>�  �+��fff�C�0�>�  ��=q�`  �HC�s3                                    Bx��vd  
�          @p��>�׿Y���g�
=C��)>�׿�G��`  �)C�Ff                                    Bx�څ
  T          @p��>�녿J=q�i����C��f>�녿��H�a�G�C�aH                                    Bx�ړ�  T          @p��>�=q�n{�g��fC��>�=q�����^�RǮC���                                    Bx�ڢV  
Z          @p  >Ǯ�u�eǮC�q>Ǯ�����\����C��{                                    Bx�ڰ�  
Z          @q�>�33�@  �k�C�u�>�33��
=�dz�  C�7
                                    Bx�ڿ�  �          @p��>Ǯ���H�l�� ��C��>Ǯ�n{�g��C�L�                                    Bx���H  "          @tz�>\���R�q�¥33C���>\�@  �n{W
C���                                    Bx����  
�          @|(�>�p��fff�tz�ffC�!H>�p������l(�aHC��H                                    Bx���  
�          @{�?�Ϳh���q��3C���?�Ϳ����i��Q�C�                                      Bx���:  �          @{�?333�=p��s33��C���?333��Q��k��)C�C�                                    Bx���  "          @}p�?��=p��vff�=C���?������o\)W
C��3                                    Bx���  
�          @u�?(��8Q��mp�z�C��?(����fff�
C���                                    Bx��&,  "          @p��>��W
=�h���C�e>����
�`���qC��\                                    Bx��4�  "          @tz�>���:�H�n{8RC�� >����
=�g
=\)C���                                    Bx��Cx  
�          @vff>8Q�333�q�\)C�  >8Q쿔z��j�H�=C�]q                                    Bx��R  �          @s33=��Ϳ
=q�p  £p�C�=��Ϳ}p��i��p�C��                                     Bx��`�  
�          @u�#�
���s33¦W
C�s3�#�
�k��n{(�C���                                    Bx��oj  �          @xQ�=��   �u¤��C���=��xQ��p  ǮC�xR                                    Bx��~  "          @}p�=�Q쾽p��{�¨��C���=�Q�\(��vff�3C��                                    Bx�ی�  "          @tz���H�#�
�q�¥L�C5{���H���p  Q�C`�\                                    Bx�ۛ\  �          @r�\�k��\)�p��«�CSQ�k��(��mp�   Cy\                                    Bx�۪  
�          @y���aG�����vff¦#�Cq�{�aG��c�
�qG���C��                                    Bx�۸�  �          @w
=��G���=q�u«G�Cx�\��G��@  �q�8RC���                                    Bx���N  T          @w���Q�#�
�w�®� Cq\��Q�&ff�tz� � C�\                                    Bx����  T          @x��=���\)�w�ª��C�5�=��E��s33�)C�L�                                    Bx���  �          @y��<#�
��\�w
=¤�
C�p�<#�
��  �p��=qC�9�                                    Bx���@  T          @z=q��G���33�x��©33C|���G��W
=�s�
�
C�U�                                    Bx���  
(          @z=q�.{�
=�w
=¡��C}�R�.{��=q�p  u�C�s3                                    Bx���  
�          @~�R=�Q��ff�|��¦��C��{=�Q�s33�w
==qC���                                    Bx��2  T          @\)>8Q�����}p�§=qC�8R>8Q�h���xQ��C��3                                    Bx��-�            @|(��8Q�&ff�xQ� W
C~��8Q쿑��p���
C���                                    Bx��<~  �          @�  �u�
=�|(�¡�=Cx\�u�����u�L�C���                                    Bx��K$  �          @~�R�\)��
=�|��§#�C|O\�\)�n{�w
=�{C��                                     Bx��Y�  �          @~�R�aG���\�{�£�)Cv���aG����\�u��\C���                                    Bx��hp  "          @~{���Ϳ�R�y��z�CmO\���Ϳ����q��Czp�                                    Bx��w  T          @\)�k����|(�£aHCvO\�k�����u
=C��
                                    Bx�܅�  "          @�  �.{����~{§�Cw��.{�n{�xQ��{C��R                                    Bx�ܔb  �          @���=u��z�����«�\C��
=u�Q��}p��\C��                                    Bx�ܣ  "          @���>�녽�G�����§�
C�=q>�녿#�
�~�R8RC�T{                                    Bx�ܱ�  T          @���?(�=��~�R¢8RA0��?(�����}p���C�'�                                    Bx���T  T          @�=q>����8Q�����©�C�h�>����8Q��~{��C��                                     Bx����  �          @�=q>B�\�\)����­W
C�P�>B�\�.{��  ��C��{                                    Bx��ݠ  �          @�(�>��u���
°.C���>�������\¢ǮC��                                     Bx���F  
�          @�z�>B�\��\)��(�®� C��3>B�\��R���\¢C�`                                     Bx����  "          @��>�zὣ�
����«�qC��>�z�#�
��33 �C�0�                                    Bx��	�  }          @��>Ǯ�����(�¨aHC���>Ǯ�333���\�C��                                    Bx��8  ]          @�z�>�  �u���
ªaHC�Ǯ>�  �L������C���                                    Bx��&�  
(          @�ff>�
=�k���p�¦�)C���>�
=�J=q��33k�C�*=                                    Bx��5�            @�(�>�
=�u���H¦�\C�H�>�
=�J=q������C��                                    Bx��D*  
�          @z�H?h��>u�s33W
Aj�\?h�þ�\)�s33��C�e                                    Bx��R�  �          @z�H@33?�
=�3�
�G��A���@33?G��<(��U
=A��                                    Bx��av  "          @|(�@7
=@��G���
=B  @7
=@�ÿ������B\)                                    Bx��p  �          @~�R@:=q@G���(����B�@:=q@33��\��
=B�                                    Bx��~�  
(          @}p�@5�@�ÿ�33���B!{@5�@����H���B��                                    Bx�ݍh  �          @\)@7
=@(�������
=B!��@7
=@�R������HB��                                    Bx�ݜ  �          @�  @6ff@(���������B"�\@6ff@�R�ٙ��ɮB\)                                    Bx�ݪ�  �          @x��@1�@��������\B#Q�@1�@�Ϳ������HB�                                    Bx�ݹZ  �          @s33@*=q@�
�����ffB#��@*=q@ff�ٙ����HB
=                                    Bx���   �          @qG�@+�@�׿������
B p�@+�@33��
=�ӮBff                                    Bx��֦  �          @r�\@&ff@�ÿ�p���{B
=@&ff?��� ��� �\B��                                    Bx���L  K          @x��@*�H?�
=����p�BG�@*�H?У����ffA��R                                    Bx����  
�          @hQ�@��@���p���G�B&G�@��?�{�   ��Bz�                                    Bx���  
�          @g�@�?�Q��G��{B!Q�@�?���G���
B��                                    Bx��>  �          @mp�@�?��(���B�@�?\���$�B�\                                    Bx���  �          @q�@��?����*�H�4�RBG�@��?�
=�7��F(�A�                                    Bx��.�  
�          @u@�?\�)���0�B  @�?���5�@\)A�                                      Bx��=0  
Z          @s�
@�?�(��!��&��A�z�@�?����-p��5�A��                                    Bx��K�  �          @qG�@  ?�{����33B
=@  ?�  �'��0�\B�
                                    Bx��Z|  �          @r�\@  ?������Q�B�H@  ?��
�(���0�B�R                                    Bx��i"  �          @u@��?��H�G����B�@��?�\)�!G��$
=B�
                                    Bx��w�  T          @x��@{?�
=��
��\B33@{?�=q�#�
�$p�B�H                                    Bx�ކn  
�          @xQ�@!G�@z����\B�R@!G�?�  �
=�Q�Bff                                    Bx�ޕ  }          @|(�@   @ff�(���B 33@   ?�\�{��B{                                    Bx�ޣ�  
�          @�@
=?��:�H�1�B  @
=?����I���D33A�{                                    Bx�޲`  
          @���@33?�\)�G��:  Bff@33?�33�Vff�L��A�\)                                    Bx���  �          @��H@Q�?�{�G
=�7ffB@Q�?���Vff�I�HA��                                    Bx��Ϭ  
�          @��@/\)?��R�E��6��A�G�@/\)?J=q�N{�B  A��R                                    Bx���R  T          @�{@*�H?���XQ��F��A��R@*�H?���`  �P\)A9p�                                    Bx����  �          @��
@   ?��
�Tz��GQ�A���@   ?J=q�^{�S�A�(�                                    Bx����  �          @�z�@
=?����N�R�=Q�B�@
=?�=q�]p��O��A�33                                    Bx��
D  �          @�\)@p�?�Q��U�@�\B
33@p�?����c33�QQ�A�\)                                    Bx���  �          @��@��?�33�S�
�B�B	��@��?�33�`���S33A���                                    Bx��'�  �          @��R@#33@   �*�H�(�B@#33?����<(��1\)B G�                                    Bx��66  �          @�ff@�R?�p��W
=�E33A���@�R?z�H�b�\�S��A�=q                                    Bx��D�  �          @�=q@G�?���s33�a��A̸R@G�?��z�H�l�HAPz�                                    Bx��S�  T          @��@�?�  �r�\�`�HA�Q�@�>�G��y���j�RA+�
                                    Bx��b(  �          @�\)@?��\�j=q�\\)A�ff@>��p���f�A8z�                                    Bx��p�  �          @��
@(��?333�n{�V�An�H@(��>���r�\�\G�@Vff                                    Bx��t  T          @�=q@0��>���hQ��Q��A�@0�׾��i���S��C���                                    Bx�ߎ  T          @�@+�>�(��r�\�YG�A�H@+����s�
�[33C��                                    Bx�ߜ�  �          @��R@(��>�(��c�
�Tp�A��@(�ý�Q��e�V��C���                                    Bx�߫f  �          @�33@���G��n{�oG�C�"�@��z�H�g��d��C���                                    Bx�ߺ  T          @�G�@ff�����n{�qQ�C�޸@ff�Y���h���hp�C�H                                    Bx��Ȳ  T          @�G�?�(��Ǯ�q��x(�C�b�?�(��p���k��mffC�8R                                    Bx���X  �          @�
=?�33��\)�p���{�
C��\?�33�Tz��k��rp�C�33                                    Bx����  �          @��?�ff��=q�o\)33C���?�ff�Q��j=q�v��C��
                                    Bx����  �          @��?�{���
�mp��{�C�)?�{�\(��g��q�RC��\                                    Bx��J  �          @��@33�k��g
=�p�
C���@33�B�\�b�\�h�C��
                                    Bx���  �          @���@�;�Q��`  �e�RC�Z�@�Ϳ^�R�Y���\��C�0�                                    Bx�� �  �          @�
=@{��z��dz��g��C�Q�@{�O\)�_\)�_p�C���                                    Bx��/<  �          @�Q�@{<#�
�i���j�\>�@{�
=q�g
=�f�\C�:�                                    Bx��=�  �          @�
=?���=����s33�)@Fff?��þ���qG��~Q�C�˅                                    Bx��L�  
�          @�
=?�=��
�r�\L�@ ��?����p���|�C��                                     Bx��[.  �          @�\)?޸R��\)�vffp�C�˅?޸R�&ff�s33  C�Ф                                    Bx��i�  �          @���?�(�����e�p��C���?�(������]p��c  C�                                      Bx��xz  �          @�ff@	����
=�dz��iC�� @	���p���]p��_p�C�#�                                    Bx���   �          @��R@�R��(��W��U{C�@�R�n{�QG��L  C��\                                    Bx����  T          @��@)���.{�N{�F\)C�Ф@)����33�E��;(�C�B�                                    Bx��l  �          @��R@(Q�z��N{�Hp�C�˅@(Q쿆ff�Fff�>=qC��                                    Bx��  �          @�\)@1G���\)�?\)�4{C��{@1G��Ǯ�2�\�%�C�P�                                    Bx����  �          @��R@+�����>{�3�C�@+��޸R�/\)�"�\C���                                    Bx���^  �          @�33@\)����>�R�:�RC�q@\)�޸R�0  �(�RC��                                     Bx���  �          @�33@#�
���R�<���8=qC�)@#�
��z��.�R�'=qC�z�                                    Bx����  �          @��@1녿�{�.�R�)C�'�@1녿�  �"�\��C��=                                    Bx���P  �          @��@A녿s33�'
=���C�H�@A녿�=q�(����C�                                      Bx��
�  
�          @�33@%����=p��:p�C��=@%���R�1G��+33C�\                                    Bx���  �          @��H@�׿���I���I��C�
@�׿�  �:�H�6��C�{                                    Bx��(B  �          @��@Q쿳33�Mp��N�C�e@Q��{�=p��9{C�n                                    Bx��6�  �          @��\?�\��33�QG��U��C�xR?�\�Q��?\)�<�C��                                     Bx��E�  �          @��H?�p����U�Z��C�o\?�p��z��AG��>�RC�G�                                    Bx��T4  �          @���?�p���\�K��P\)C�  ?�p���R�7��6{C��q                                    Bx��b�  �          @���@&ff��33�����C���@&ff�  ������HC��3                                    Bx��q�  �          @��\@!녿��
�*=q�#�C�s3@!��
=q�
=��RC��                                     Bx��&  �          @���@{���.{�)C��3@{��
�(��C��                                    Bx���  �          @�  @$z�����'
=�#z�C�C�@$z��������C�h�                                    Bx��r  �          @��\@�\��=q�5�0Q�C���@�\�\)�!���C�˅                                    Bx��  �          @��H@�\����8Q��2�C��)@�\����$z��  C��                                    Bx�Ế  �          @�33@  ����7
=�0�HC��@  �33�"�\��C�33                                    Bx���d  T          @�33@	����=q�?\)�:ffC��{@	������+��"\)C��                                    Bx���
  �          @���@�����Q��Xz�C��@���\)�Dz��E33C��                                    Bx���  �          @�33@���\�QG��T�HC�Q�@��G��B�\�@ffC���                                    Bx���V  �          @���@(���\)�?\)�>�\C���@(���
�-p��({C�b�                                    Bx���  �          @�=q?�׿˅�N{�R(�C�޸?����
�<(��9��C�,�                                    Bx���  �          @�33?�z���R�J�H�J��C���?�z�����5��.��C��                                    Bx��!H  �          @�33?��{�@  �<��C�w
?��)���(Q���C�R                                    Bx��/�  �          @��?�����\�W
=�W�C��f?����!��@���9ffC�{                                    Bx��>�  �          @�
=?�z��
=q�U�Q�C���?�z��)���>{�2�HC�H                                    Bx��M:  �          @���?����
�\(��U  C�u�?���#�
�E�7�HC���                                    Bx��[�  �          @�G�?�=q��z����z�C��H?�=q�G��vff�gffC��H                                    Bx��j�  �          @���?�녿�����RC��?��� ���|���h��C�T{                                    Bx��y,  �          @��
?���Ǯ�����C�˅?���
�H�w
=�cffC�T{                                    Bx���  �          @�?�=q�ٙ������z=qC��q?�=q��
�u�]  C��                                    Bx��x  �          @�p�?�\)���H���
�x(�C�Z�?�\)��
�s�
�[(�C�H�                                    Bx��  �          @��H?�{��ff��33�}
=C���?�{�
=q�s�
�`C��                                    Bx���  �          @���?�G���z���33� C��H?�G��G��u�hC��
                                    Bx���j  �          @��R?��R��{�~{�{�C���?��R�(��k��^33C��=                                    Bx���  �          @�  ?����p���G��z�C�� ?�����p���c=qC��                                    Bx��߶  �          @�  ?����H�\)�y�HC�33?��33�k��[�\C��                                     Bx���\  �          @��?���Q��s33�^�C���?���<(��X���=C���                                    Bx���  �          @��
?}p��1G��c�
�I�\C��{?}p��Q��E�(�C�c�                                    Bx���  �          @�z�?���4z��a��Fz�C�%?���Tz��C�
�%33C��3                                    Bx��N  �          @���?k��B�\�c33�@�C�t{?k��c33�C33���C�L�                                    Bx��(�  �          @���?^�R�HQ��`���=(�C��=?^�R�hQ��@  ��C��H                                    Bx��7�  �          @���?c�
�J=q�^{�:=qC���?c�
�j=q�<(��33C���                                    Bx��F@  �          @�(�?����C�
�fff�?\)C��?����e��E�{C��q                                    Bx��T�  �          @��?��
�"�\��=q�`{C��?��
�HQ��hQ��?  C�R                                    Bx��c�  �          @�  ?\(��:=q�}p��R(�C�:�?\(��^�R�^{�033C��\                                    Bx��r2  �          @��R?aG��E�q��F\)C��?aG��hQ��P���$Q�C���                                    Bx���  �          @�ff?\(��N�R�j=q�>p�C�n?\(��p  �G��G�C�o\                                    Bx��~  �          @�ff?}p��^{�W��,�C��?}p��|���333�
�C��                                    Bx��$  �          @�{?�G��dz��O\)�$G�C��?�G������)���Q�C��                                    Bx���  �          @�  ?G��HQ��s�
�FC�?G��k��Q��$z�C��                                    Bx��p  T          @�(�?+��B�\��(��S�C�.?+��hQ��g
=�1p�C�0�                                    Bx���  �          @�z�?.{��  ��Q�ǮC��
?.{��H��{�l(�C��{                                    Bx��ؼ  
�          @��
>\�   ��p��{C�b�>\�)������bz�C�{                                    Bx���b  �          @�33>�p��
�H���\�|{C���>�p��3�
�|(��YQ�C��q                                    Bx���  �          @���?#�
�9���j=q�KC�33?#�
�[��J�H�)Q�C�E                                    Bx���  �          @��?�p���z��ff��C�@ ?�p���p�������(�C��                                    Bx��T  �          @���?������
��
�ˮC��?�����zῴz�����C�b�                                    Bx��!�  �          @���?�
=��G���R��
=C�(�?�
=���\��=q��p�C���                                    Bx��0�  �          @��H?������������=qC�˅?�����녿�  ����C�'�                                    Bx��?F  �          @��H?�  �}p������C�?�  ���ÿ�ff���C�!H                                    Bx��M�  �          @��H?�{�}p��"�\��  C��\?�{������z���Q�C�8R                                    Bx��\�  �          @��\?^�R�w��/\)�
G�C�T{?^�R��  �
=��  C�˅                                    Bx��k8  �          @�z�?Ǯ�������ָRC�y�?Ǯ��33�Ǯ��
=C���                                    Bx��y�  �          @�(�?�Q���p�����HC���?�Q���{��
=��z�C��R                                    Bx�䈄  �          @�(�?�������	���љ�C�9�?������  ����C��                                    Bx��*  �          @�=q?�=q��z���
��G�C��=?�=q���Ϳ�z���z�C�h�                                    Bx���  �          @��H?�  ���R������p�C��{?�  ��{��
=�b�RC�Z�                                    Bx��v  �          @��H?�  ��(�� �����HC��q?�  ��z΅{���HC�q�                                    Bx���  �          @��?�{�����ff��G�C�T{?�{���H���H����C��{                                    Bx����  �          @�z�?�
=�z=q�����C���?�
=��\)���
��33C��R                                    Bx���h  �          @�p�?ٙ��w
=�"�\��(�C��R?ٙ���ff����C���                                    Bx���  �          @�ff?��\�w
=�3�
�
p�C��?��\��  �(���Q�C�Y�                                    Bx����  �          @���?z�H�o\)�L����C�P�?z�H���&ff��\)C���                                    Bx��Z  �          @���?fff�h���XQ��(G�C���?fff����2�\�\)C�,�                                    Bx��   T          @���?c�
�e�\(��+�C��)?c�
��=q�7
=�	��C�0�                                    Bx��)�  �          @�=q?^�R�b�\�a��0z�C��?^�R�����=p���\C��                                    Bx��8L  �          @���?O\)�W��j=q�:z�C��?O\)�xQ��G
=��\C��f                                    Bx��F�  �          @��?^�R�Mp��n{�@��C���?^�R�o\)�L(����C���                                    Bx��U�  T          @�Q�?�G��S�
�a��3�\C�` ?�G��s�
�?\)��
C�                                      Bx��d>  �          @��?��R�X���]p��,=qC��?��R�w��:=q�(�C��\                                    Bx��r�  �          @�=q?Ǯ�AG��o\)�>�
C��?Ǯ�c33�O\)��C��                                    Bx�偊  �          @���?�ff�%������T�HC���?�ff�I���e�6p�C��                                    Bx��0  �          @��?�G��7��r�\�AffC��)?�G��Z=q�S�
�#=qC���                                    Bx���  �          @�=q?�p��8���q��A{C�g�?�p��[��S33�"C�\)                                    Bx��|  �          @�=q?޸R�7��s33�B(�C���?޸R�Z=q�Tz��#��C�}q                                    Bx��"  �          @�=q@�\�A��`���/�C��R@�\�aG��AG��=qC��                                    Bx����  �          @��\@G��?\)�e��3
=C�H@G��_\)�E���C��                                    Bx���n  T          @�=q?�  �B�\�j=q�8�HC��)?�  �c33�J=q�z�C�!H                                    Bx���  �          @��H?�  �>{�n�R�={C�AH?�  �`  �P  ��
C�L�                                    Bx����  �          @��?ٙ��-p��z�H�K(�C��?ٙ��QG��^{�-G�C��)                                    Bx��`  �          @���?�=q�0  ��Q��R��C���?�=q�Tz��c�
�3�C���                                    Bx��  �          @���?�=q�9���w��I��C�H�?�=q�\���Y���*Q�C���                                    Bx��"�  �          @��
?�p��G
=�q��>��C��{?�p��h���QG��G�C��                                    Bx��1R  �          @�=q?����'
=��p��]�C�Z�?����L���n�R�>{C�H�                                    Bx��?�  �          @�Q�?����<(��qG��E\)C�XR?����^{�R�\�%�HC��f                                    Bx��N�  �          @�?�{�Q��Y���.�C�E?�{�p  �8Q��{C���                                    Bx��]D  �          @�p�?��\�@  �hQ��@G�C�~�?��\�`���I��� �\C���                                    Bx��k�  �          @�Q�?Ǯ�\���P���"�RC�/\?Ǯ�y���-p��\)C��\                                    Bx��z�  �          @���?�Q��'��~�R�Tp�C�o\?�Q��K��c33�6  C�7
                                    Bx��6  �          @�?��H��p���\)�HC��?��H�=q���j�C�\)                                    Bx���  �          @�?��������uz�C�W
?���/\)��{�W�
C��                                    Bx�概  �          @�(�?�ff�����R�qz�C�j=?�ff�3�
���H�Sz�C�h�                                    Bx��(  �          @��?:�H��
=�0���{C���?:�H���\�
=��\)C���                                    Bx����  �          @��?:�H�{��G
=��\C�@ ?:�H���H�   ��\C�                                    Bx���t  �          @���>�Q��x���K����C��>�Q���=q�%���{C�b�                                    Bx���  �          @�33�#�
����;����C��
�#�
��G���\��=qC��q                                    Bx����  �          @���L������)�����\C��f�L����ff���R��G�C���                                    Bx���f  �          @��ͽ�G���  �\)���
C�N��G���=q������C�Z�                                    Bx��  �          @�z�u���\�z���Q�C��H�u���
��\)��\)C��R                                    Bx���  �          @��\�aG����
�%�����C��=�aG���ff����C���                                    Bx��*X  �          @�=q�u��Q��0  �G�C�]q�u����ff��C�}q                                    Bx��8�  �          @��\�k������;����C�o\�k�������\�֏\C���                                    Bx��G�  �          @���.{��=q�E���\C��3�.{��
=�p���z�C��\                                    Bx��VJ  �          @��
�u��  �L�����C����u��p��%����C��
                                    Bx��d�  �          @�����
�|���N�R�{C�⏼��
��(��(Q����C��                                    Bx��s�  \          @�33�����}p��N{�ffC�Ff������(��'
=��z�C�XR                                    Bx��<  �          @��\��Q��y���P  �
=C�O\��Q����\�)�����
C�aH                                    Bx���  �          @�(����
�x���U��!�RC��׾��
���\�.�R� C��                                     Bx�矈  �          @����=q�w
=�Tz��"�\C�����=q�����.�R���C�1�                                    Bx��.  �          @���>�=q�~�R�B�\�=qC��{>�=q��(��(�����C�Ǯ                                    Bx���  �          @���>�  �u�O\)� \)C��)>�  ��Q��)�����C���                                    Bx���z  �          @��H>�33�q��W
=�&G�C��H>�33��
=�2�\��\C�Z�                                    Bx���   �          @�G�?c�
��ff�(Q����C��)?c�
�����   ���C��\                                    Bx����  �          @���?z�H���
�G��ׅC�Y�?z�H��z�У�����C���                                    Bx���l  �          @�  ?O\)��\)��
���C�"�?O\)��
=��z����HC��                                     Bx��  �          @�
=?z�H���ÿ�\���C�
?z�H�����{�N�HC��{                                    Bx���  �          @�
=?u�������C�  ?u��p�������p�C�Ф                                    Bx��#^  �          @�  ?��H��(��z��ÅC���?��H��(���
=����C�L�                                    Bx��2  \          @�  ?�  ��p����H��p�C�� ?�  ���Ϳ����u�C��                                     Bx��@�  �          @���?�=q��33���H���C��
?�=q������ff�@��C�T{                                    Bx��OP  �          @�ff?��H���
�޸R���C�8R?��H��=q��{�QG�C��
                                    Bx��]�  �          @���?�  ��Q����=qC�(�?�  ������(���Q�C���                                    Bx��l�  �          @�ff?������
�   ����C�<)?�����33��\)��\)C��                                     Bx��{B  �          @��R?�33�����  ��33C�8R?�33�������N{C���                                    Bx���  �          @�ff?�Q���p���\)���C��?�Q���z῞�R�g�C�/\                                    Bx�蘎  �          @�
=?xQ����ÿ�ff��=qC��?xQ���\)��z��W�C�˅                                    Bx��4  �          @��?�����\��z���\)C�b�?����Q쿁G��:�\C�%                                    Bx���  �          @�Q�?����녿�ff��ffC�j=?����Q쿓33�T��C�%                                    Bx��Ā  �          @���?����\)��\���C�xR?����
=��Q���
=C���                                    Bx���&  �          @�=q@Q����\��îC��@Q����\��G���G�C��                                    Bx����  �          @��H?������H�{��=qC���?�������������
C�1�                                    Bx���r  �          @�33?������ �����C���?���=q��33��C�&f                                    Bx���  �          @��\?�����p��%���\C�Ф?�����\)���R���\C�<)                                    Bx���  �          @��H?Q����\�:=q��C���?Q�������ffC�B�                                    Bx��d  �          @���?\(��q��N�R�\)C�c�?\(����,(�� �
C��=                                    Bx��+
  �          @�=q?���k��O\)���C���?�����H�-p��=qC��q                                    Bx��9�  �          @��H?�Q��mp��P  �G�C���?�Q�����.{��\C�R                                    Bx��HV  �          @��\?��Tz��c33�2
=C���?��qG��Dz��G�C�K�                                    Bx��V�  �          @���?��H�^{�U��&{C�l�?��H�xQ��5��	\)C�O\                                    Bx��e�  �          @���?���e��Fff���C�O\?���~{�%����C�>�                                    Bx��tH  �          @�G�?�ff�q��2�\��C���?�ff���
�  ���C���                                    Bx���  �          @�Q�?�{��������33C�c�?�{������(���33C��{                                    Bx�鑔  �          @��?\��{��R�ԏ\C���?\��ff��33���HC�o\                                    Bx��:  
�          @���?�{������(�C��f?�{��{���H��
=C���                                    Bx���  �          @���?�  �����
=q��  C��\?�  ���׿�=q��z�C�0�                                    Bx�齆  �          @�{?�z����Ϳ�
=��G�C�T{?�z������W
=���C��                                    Bx���,  �          @�ff?�p���Q쿈���H��C�}q?�p���������
=C�Ff                                    Bx����  �          @��R?�  ��{�
=q���C�Ф?�  ��{�˅��{C�N                                    Bx���x  �          @�ff?��
�g
=�?\)�p�C��H?��
�~{�   ��33C��                                    Bx���  �          @��?xQ��S33�b�\�7ffC�8R?xQ��n�R�E���C�N                                    Bx���  �          @��R?����
=��R�֏\C��)?����\)����\)C�%                                    Bx��j  �          @��R?��
��Q��
=q��G�C�` ?��
��Q�˅��{C���                                    Bx��$  �          @��R?�  ��G������(�C�%?�  ���ÿǮ����C��)                                    Bx��2�  �          @�\)?}p�����(��Џ\C�l�?}p������{���RC�R                                    Bx��A\  �          @�
=?�G���(��
=��G�C���?�G����
���
���C�1�                                    Bx��P  �          @�{?J=q���H�����(�C�/\?J=q���H�У���=qC���                                    Bx��^�  �          @�{?��
��  �  ��  C���?��
��  ��Q����HC�s3                                    Bx��mN  �          @�Q�?�33�����ř�C���?�33��
=�������C�q                                    Bx��{�  �          @���?�Q���\)����ffC�b�?�Q���\)��p���
=C���                                    Bx�ꊚ  �          @���?�{����Q���z�C��
?�{��(�����(�C�q                                    Bx��@  �          @���?��w��(Q����\C��{?���p������C��                                    Bx���  �          @�\)?�33�����=q���
C�'�?�33���������=qC���                                    Bx�권  �          @�{?�
=�qG��!����HC���?�
=�����\��G�C��
                                    Bx���2  �          @��?���s33�!���Q�C���?�����\����(�C���                                    Bx����  �          @��?��H�u�!G����C���?��H��(��G��£�C�AH                                    Bx���~  �          @�p�?��{�����C���?���{�������
C��{                                    Bx���$  �          @�p�?�z��w
=�ff��33C�'�?�z����
��{��{C�l�                                    Bx����  T          @�{@X���J�H?\)@��HC�w
@X���E�?p��A:{C���                                    Bx��p  �          @��@33�fff@�HA�Q�C�O\@33�QG�@5B�\C��\                                    Bx��  �          @��R@���ff?�\)AP��C��R@�����?�\)A��HC�9�                                    Bx��+�  �          @��
@p��{�?�
=A���C�
=@p��n�R?�33A��C���                                    Bx��:b  �          @�(�@���?
=q@�33C��@����\?��AE��C�e                                    Bx��I  �          @�(�@���  �aG��,(�C��
@���  >�=q@U�C�ٚ                                    Bx��W�  �          @��
?�33���\�h���6ffC�w
?�33��������(�C�AH                                    Bx��fT  �          @�Q�@��
=���Ϳ�  C�/\@��ff>���@��C�<)                                    Bx��t�  �          @�\)@	����\)�.{���RC���@	����
=>�Q�@�C��{                                    Bx�냠  �          @�(�?��R��p�>8Q�@p�C���?��R���
?333Az�C���                                    Bx��F  �          @�\)?p����  �&ff�(�C�4{?p�������\)��\)C�#�                                    Bx���  �          @�p�?J=q��z��Tz��hz�C�>�?J=q���Dz��O(�C���                                    Bx�믒  
�          @��þ��H?��H�����B�
=���H?\(���z�B�k�                                    Bx��8  �          @�녿\(�?G����C	�H�\(�>.{��
= ��C)                                      Bx����  �          @�{�W
=?�p���p���B�
=�W
=?�\)���H�HBɔ{                                    Bx��ۄ  �          @��;��H?У���(�W
BՀ ���H?��\����ǮB�=q                                    Bx���*  �          @�G��^�R?:�H��p�(�C�=�^�R>���
= �HC+J=                                    Bx����  �          @�G����
?�����qC.���
��\)��p�C8#�                                    Bx��v  �          @�  �&ff?\(����
#�B�k��&ff>�=q��¤p�C�\                                    Bx��  �          @�
=�(�>�z���p�¥�CaH�(�������p�¥
=CN#�                                    Bx��$�  �          @�Q쾙��>�����¬��C�������ff��
=¨8RCk�q                                    Bx��3h  �          @�녾��?
=q��Q�§Bҙ�����L����G�°�{CG޸                                    Bx��B  �          @�����>���Q�©��B�  ���\)����¯��Cb�                                    Bx��P�  �          @�녾��R�#�
����­=qC5�쾞�R�����Q�¥Q�Cr�=                                    Bx��_Z  �          @��
���������­� C:\�����!G���=q¥\CtJ=                                    Bx��n   �          @��H����#�
���\®z�C5�����������¥��Cv�{                                    Bx��|�  \          @�33���8Q�����¨.CG8R���B�\��� 
=Ck��                                    Bx��L  �          @�(��\(��8Q�����¡.C@!H�\(��E���  B�C]�3                                    Bx���  �          @��
�J=q<��
����¢��C2B��J=q�\)����.CV��                                    Bx�쨘  �          @�33�\)��=q���¦��CN=q�\)�Y����  {Cl�                                    Bx��>  �          @�z���;����33§�qCa�þ��Ϳ}p�����#�Cw�f                                    Bx����  �          @�zᾸQ�u���
ª��CV쾸Q�Q���� �Cvh�                                    Bx��Ԋ  �          @�ff��;W
=��p�§��CIO\��ͿJ=q�����Ck\)                                    Bx���0  �          @��R�O\)�8Q���z�¢z�C@aH�O\)�@  ���H��C^��                                    Bx����  �          @�{�Tz�\)���
¢#�C=Y��Tz�5���\��C\G�                                    Bx�� |  �          @�
=�8Q�B�\��p�¤k�CB� �8Q�B�\���
Cb��                                    Bx��"  �          @��Ϳ�p���\)��{.C6��p���R����� CJ��                                    Bx���  T          @�(����R��  ���z�C=h����R�G������CO��                                    Bx��,n  T          @���׿!G���Q�¢��Ci&f��׿���p�z�Cx\                                    Bx��;  �          @�
=?�R�k������C��?�R��(���\)#�C��f                                    Bx��I�  �          @�ff?aG�������C�#�?aG��33��33�|ffC��H                                    Bx��X`  �          @�\)?Tz῅���  �=C�ff?Tz��������3C�H                                    Bx��g  �          @�\)?�ͿW
=���
�
C���?�Ϳ�����Q�p�C��\                                    Bx��u�  �          @�
=>�  �xQ����
��C�7
>�  ��G���  u�C��{                                    Bx��R  �          @�
=<��
��{�����C���<��
�����
={C�aH                                    Bx���  �          @��k��z�H���H��C�s3�k���G����R\)C��3                                    Bx����  �          @��R�\�8Q���z�¢�RCr�\��G���G�#�C|�R                                    Bx���D  �          @�\)>��ÿ\(���z� ��C��>��ÿ���������C���                                    Bx����  �          @���>�p��z���  ¥ǮC�C�>�p���{����C�&f                                    Bx��͐  �          @�ff>�\)��
=���©p�C��>�\)�p�����H(�C�/\                                    Bx���6  �          @�33>�p������p�k�C�ٚ>�p�����Q���C��                                    Bx����  �          @�>.{��(���
=W
C�8R>.{��(�����W
C�h�                                    Bx����  �          @�{<#�
������ffffC�1�<#�
��
����ffC�&f                                    Bx��(  �          @���Q�����(�aHC�� ��Q��G���u�C��=                                    Bx���  �          @�ff�#�
�������k�C��\�#�
��=q��(���C�z�                                    Bx��%t  �          @���u������H�)C���u���
��{�C��R                                    Bx��4  �          @�  >�  ��G����H�)C��q>�  ��G���ffL�C��                                    Bx��B�  �          @��ü���
=���\��C�~�������W
C��                                     Bx��Qf  �          @�Q��G��˅����Q�C��)��G��z���33ǮC�s3                                    Bx��`  �          @���<������
==qC�}q<������G��qC�c�                                    Bx��n�  �          @�G�=�Q��
=����C�t{=�Q��	�����W
C�"�                                    Bx��}X  �          @���=��
���
�����C�g�=��
� ����z�W
C�3                                    Bx���  �          @�  �aG���Q�����C����aG�����\)\)C�AH                                    Bx�  �          @�ff���R���R��\)� C�����R������=qu�C�w
                                    Bx��J  �          @�p�=u������=qC�1�=u��ff��33�C���                                    Bx���  �          @��>�=q�u����C�Ф>�=q����ff�)C�Y�                                    Bx��Ɩ  �          @�>��ÿTz���33¡�C��q>��ÿ����  ��C�*=                                    Bx���<  �          @�(���G���  ��
=�C��׽�G��ٙ����H
=C�*=                                    Bx����  �          @�z��G���
=��  �{C�g���G���\)���
C�q                                    Bx���  �          @�(�>�p��0�����£.C��>�p������\)#�C��{                                    Bx��.  �          @�>����\�����qC���>����H����\C��                                    Bx���  �          @�����{�k����\ª�RCUp���{�&ff��G�£W
Cr33                                    Bx��z  �          @��
��?�  ����W
B�B���?\)���\§�qB�                                    Bx��-   �          @��\�Ǯ?!G���Q�£�B��H�Ǯ>L����G�ª��C
=                                    Bx��;�  �          @��H��\�����¨�RCB
��\�\)����£��Cc޸                                    Bx��Jl  �          @�녾\�8Q�����ª�fCN{�\�(���  ¤\)Cnk�                                    Bx��Y  T          @�G��B�\���H��Q�©�Cx���B�\�n{��ff 33C�*=                                    Bx��g�  �          @�Q�aG������©Q�Ct}q�aG��fff�� ��C��                                    Bx��v^  �          @�p�?�?z�H��  ��Bw=q?�?\)���£p�B;                                    Bx��  �          @�ff?�p�?�(���=q�=BH�R?�p�?�����p�B$33                                    Bx�  �          @�p������  �i���)��C����������U���RC���                                    Bx��P  �          @�33����|(��e�){C��������R�QG��\)C�=q                                    Bx���  �          @��H��G��fff�z�H�=�RC�&f��G��x���h���,�C�7
                                    Bx�ￜ  �          @��\��G��O\)��{�P{C�#׾�G��c33�{��>��C�xR                                    Bx���B  �          @��\�!G��:=q��p��`=qC�녿!G��N�R���OQ�C���                                    Bx����  �          @��Ϳ#�
�1���=q�h
=C���#�
�G���33�WQ�C�9�                                    Bx���  �          @�����A���=q�a��C�����W
=���H�P�
C��\                                    Bx���4  �          @�{��=q�J�H�����Xp�C��þ�=q�_\)����G�C��                                    Bx���  �          @�33��  �C�
����[\)C��3��  �W���(��J�RC���                                    Bx���  �          @��R���>�R����cQ�C��H���S33���H�R�
C��)                                    Bx��&&  �          @�\)�\)�4z���{�k�HC��
�\)�I����\)�[�\C��)                                    Bx��4�  �          @��=�G��S33�����M�C���=�G��e��y���=C��f                                    Bx��Cr  �          @��\>\)�.�R����l�C�t{>\)�B�\����\�C�N                                    Bx��R  �          @�z�>aG��AG���ff�^��C�  >aG��U���\)�O�C��\                                    Bx��`�  �          @���>�Q��E���p��[��C�U�>�Q��W���ff�L\)C��                                    Bx��od  �          @�p�>��E���{�[�
C�xR>��W���
=�Lz�C��                                    Bx��~
  �          @�p�?
=q�<(������b  C�7
?
=q�O\)����R�HC��                                     Bx����  �          @�p�?&ff�@  ��
=�^{C�#�?&ff�R�\��Q��O33C��q                                    Bx��V  �          @�{?W
=�H����33�U
=C�y�?W
=�Z�H��z��Fz�C���                                    Bx���  �          @��?��Dz���p��[Q�C���?��Vff���R�L��C��                                    Bx��  T          @�ff?#�
�>�R�����`G�C�R?#�
�P�����\�Q�HC��{                                    Bx���H  
�          @�ff?���7
=��33�f�\C�w
?���I������X=qC���                                    Bx����  �          @��?��1G�����i{C�o\?��C33��(��Z�HC���                                    Bx���  �          @��=u�Fff��
=�V�HC��=u�W
=�����H��C���                                    Bx���:  �          @���=�Q��I����\)�Up�C���=�Q��Y�������GffC��=                                    Bx���  "          @��>�=q�P����p��O��C�aH>�=q�`���~{�A�RC�7
                                    Bx���  "          @���?��Q����H�Kp�C��{?��aG��xQ��=�C��                                     Bx��,  �          @�G�?333�Vff�����F33C��?333�e��s�
�8�HC��f                                    Bx��-�  �          @�(�?aG��1G������d�
C���?aG��A���33�X
=C��                                    Bx��<x  �          @��?Q���
����{��C���?Q��%��p��offC���                                    Bx��K  �          @��?G��:=q���_  C���?G��J=q��Q��RQ�C���                                    Bx��Y�  �          @�{?���R�����x��C��3?��0  ����k�C�c�                                    Bx��hj  T          @�{?��'���\)�rQ�C��3?��8Q����\�e��C��                                    Bx��w  �          @�  ?&ff�1G���ff�k33C���?&ff�A���G��^�C�R                                    Bx��  �          @��>8Q���R��\)�=C�W
>8Q��   ��33�z�RC�
                                    Bx��\  �          @��R>����{���H�z��C���>����.�R��ff�n�C�5�                                    Bx��  �          @�{=u�/\)��{�n�HC���=u�>�R��G��b�
C���                                    Bx��  M          @��?��+�����pG�C�w
?��:�H���H�d��C�                                    Bx���N  
a          @�G�?��
�(������l�C��?��
�8Q���33�a�C���                                    Bx����  �          @�\)?(���E��\)�[{C�H?(���S�
����OC��q                                    Bx��ݚ  
�          @��?+��C�
��  �\��C�(�?+��Q����H�Q�C�                                    Bx���@  �          @�z�?.{��\��p��RC�,�?.{������~C�Ff                                    Bx����  "          @��\=�\)�
�H���\��C��f=�\)�����
=�|G�C�Ф                                    Bx��	�  
�          @��
����5�����fz�C������C33��(��[C�,�                                    Bx��2  
�          @��
�\)�AG������\=qC����\)�N{��Q��Q�RC��                                    Bx��&�  
�          @��
�(��@�������\(�C�LͿ(��Mp���  �Q�HC���                                    Bx��5~  
�          @�33�   �<�����_��C�8R�   �I����G��Up�C���                                    Bx��D$  "          @�(������@  ��ff�_�C��쾨���L(���=q�Up�C��                                    Bx��R�  �          @��;L���>�R����a�C���L���J�H����W�\C�:�                                    Bx��ap  �          @��
�B�\�8�������eG�C�R�B�\�E���z��[�C�5�                                    Bx��p  �          @�����
�0�������m\)C�,ͽ��
�<�������c�RC�9�                                    Bx��~�  "          @�p���Q��%��Q��u��C�
=��Q��2�\��z��l(�C��                                    Bx��b  �          @��
��G�������#�C�����G���������{  C���                                    Bx��  
�          @���W
=�ff������C�y��W
=�!����R�vp�C���                                    Bx��  �          @���>�z��.�R��Q��j��C��>�z��9�������a�RC��H                                    Bx��T  �          @�p�=�\)�1������lffC��\=�\)�=p���G��c�C���                                    Bx����  T          @�p�>#�
�7
=��33�hz�C���>#�
�A�����_��C�~�                                    Bx��֠  �          @�G��.{�=p����
�_\)C�ff�.{�G���Q��W  C�z�                                    Bx���F  "          @�ff��  � ����G��s�C�+���  �*�H��ff�k�C�W
                                    Bx����  
�          @��=L���Q���p��|  C��R=L���"�\���\�s��C��                                    Bx���  T          @������H�����v=qC�w
����$z����R�n�C��=                                    Bx��8  T          @�p��
=q�����G��vz�C����
=q�#33���R�o  C�                                      Bx���  �          @��;�ff�p���  �s��C�����ff�'
=��p��lffC��                                    Bx��.�  
�          @�ff���#33�����rG�C��׽��,����{�j��C���                                    Bx��=*  "          @�Q�����!G����H�t  C��H�����*=q��Q��l��C��q                                    Bx��K�  "          @��
��(��6ff�����f
=C��=��(��>�R���_33C��)                                    Bx��Zv  "          @��������,(������l��C�������4z���
=�e�C��                                    Bx��i  
�          @�Q쾨���'������o=qC�` �����0  ��
=�h�RC���                                    Bx��w�  �          @�=q��(��*�H���\�m�
C�w
��(��2�\��  �g�C��=                                    Bx��h  �          @�ff���/\)���m�C�\���7�����g  C�Ff                                    Bx��  �          @�����,����  �p=qC�#׾��4z����jG�C�Y�                                    Bx��  T          @�����#�
��p��r��C��ÿ��*�H����m=qC���                                    Bx��Z  T          @��׿
=�
=q��Q��C~�
�
=�����R�}�HC�{                                    Bx���   �          @��þ��{��Q�{C���������R�|��C�aH                                    Bx��Ϧ  
�          @�Q�!G��'���Q��l��C�8R�!G��.{��ff�gz�C�xR                                    Bx���L  [          @���(���.�R��p��fp�C�@ �(���5�����ap�C�y�                                    Bx����  �          @�녿
=�2�\��
=�e�C�׿
=�8������`�
C�4{                                    Bx����  
�          @����=p������a��C��{���C�
���R�]�C���                                    Bx��
>  T          @�녾�G��:�H����`��C��R��G��@  ��33�\p�C��
                                    Bx���  �          @�33��G��8Q���\)�c�HC��3��G��>{���_�\C���                                    Bx��'�  M          @�z�&ff�>{��{�^ffC��q�&ff�C33��z��ZG�C�f                                    Bx��60  [          @�=q�=p��Dz������V{C�33�=p��I����
=�R(�C�Z�                                    Bx��D�  	�          @��\�0���B�\��=q�Y
=C����0���G
=�����UG�C���                                    Bx��S|  T          @�=q�E��>�R���H�Z�\C���E��C33��G��W
=C��                                    Bx��b"  �          @�
=�=p��C�
���R�[�C�33�=p��HQ���p��W�RC�W
                                    Bx��p�  �          @�z�=p��HQ�����U
=C�T{�=p��L(������Q�
C�t{                                    Bx��n  
�          @���.{�Z=q���\�F{C�^��.{�^{�����C
=C�t{                                    Bx��  "          @��\��Q��^�R��  �CQ�C�f��Q��a��}p��@p�C��                                    Bx����  T          @��ͿaG��X������F\)C���aG��\(���=q�C�RC�3                                    Bx���`  �          @�zῑ��W
=����C\)C{Q쿑��Z=q�����@��C{��                                    Bx���  "          @�ff�u�e�~�R�;�RC~�q�u�hQ��|(��9p�C&f                                    Bx��Ȭ  
�          @�
=��33�^{�~�R�;\)Cx\��33�`  �|���9\)CxE                                    Bx���R  �          @�\)��33�XQ��~�R�:��Cs�R��33�Z=q�|���8��Ct33                                    Bx����  	�          @�Q�Ǯ�U���33�@Q�Ct�q�Ǯ�W���=q�>�RCu33                                    Bx����  
�          @����У��R�\����BG�Cs���У��Tz���(��@�
Cs޸                                    Bx��D  
�          @��ÿ����B�\���F\)Cn�H�����Dz�����E(�Cnٚ                                    Bx���  T          @������9����p��C�Ch� ����:�H�����B�\Ch��                                    Bx�� �  
�          @��H�33�3�
��\)�EffCf�)�33�5���
=�D��Cf��                                    Bx��/6  �          @�=q����8Q����H�=�Ce������9�����\�<p�Ce�\                                    Bx��=�  
�          @�=q�'��1���  �9G�Cb���'��2�\�\)�8��CbǮ                                    Bx��L�  
Z          @�33�J=q�*=q�p  �)�C\\�J=q�*=q�p  �(��C\#�                                    Bx��[(  
�          @�z��\�;�G����R�I�\C;:��\�;�G����R�I�C;O\                                    Bx��i�  
�          @��H��
�����G��d�
CYٚ��
��ff��G��d��CY�H                                    Bx��xt  
�          @�ff?�33�����5�� 33C��?�33�����5�� G�C��                                    Bx���  T          @�=q?У������.�R��33C�n?У���Q��/\)��C�o\                                    Bx����  T          @���?�Q����\�(�����\C�8R?�Q���=q�)����p�C�:�                                    Bx���f  
�          @��R?���Q��)����  C�4{?���  �*=q��G�C�8R                                    Bx���  "          @���?������,(���\)C��?�������,����
=C��                                    Bx����  "          @���?�ff��z��%��(�C�=q?�ff��(��'
=��(�C�B�                                    Bx���X  
�          @�Q�?�  ���
�)����{C��)?�  ��33�*�H��z�C��                                    Bx����  "          @��H?������
�7��(�C��)?�����33�9����\C��                                    Bx����  T          @�p�?fff���
�Dz��
{C���?fff��33�Fff���C��3                                    Bx���J  
�          @���?
=�����N�R�G�C��?
=��  �P���{C��{                                    Bx��
�  T          @�p�?5��
=�Tz��Q�C���?5��{�W
=�G�C�Ǯ                                    Bx���  T          @�z�?��y���k��,ffC���?��w
=�n{�.�\C���                                    Bx��(<  "          @�G�?��j�H����?�
C��?��g������B33C�&f                                    Bx��6�  �          @�{?#�
�q��vff�4ffC���?#�
�o\)�x���6�C���                                    Bx��E�  �          @�z�?(��o\)�tz��4�
C���?(��l���w
=�7�C���                                    Bx��T.  4          @���?:�H�p  �tz��3�
C��f?:�H�l���w��6�RC��R                                    Bx��b�  �          @��?��h����G��>�\C��?��e���H�A��C�"�                                    Bx��qz  �          @�  <��
�w��z=q�533C�"�<��
�s�
�}p��8�C�"�                                    Bx���   �          @�������p  �r�\�4z�C��þ����l(��vff�7��C���                                    Bx����  �          @�p���\)�{��n�R�-�HC�xR��\)�w��r�\�1�\C�u�                                    Bx���l  �          @��H�u�{��g
=�*Q�C��3�u�w��k��.33C���                                    Bx���  �          @�zὸQ���  �fff�'�C�Uý�Q��|(��j�H�,  C�S3                                    Bx����  �          @����\)��33�hQ��%�
C���\)�����mp��*{C�f                                    Bx���^  �          @�����33�����s33�-G�C��f��33�|(��xQ��1�RC�z�                                    Bx���  �          @��ÿ:�H�h�����H�?  C�^��:�H�c�
����C�C�=q                                    Bx���  �          @��R�J=q�j=q�}p��:p�C�箿J=q�e���G��?(�C���                                    Bx���P  �          @�{���\(���p��H�RC��Ϳ��Vff��  �M�C��\                                    Bx���  �          @�\)�333�Q�����P(�C��q�333�L(���(��U=qC��\                                    Bx���  �          @�(���G���Q��dz��&C�.��G��z�H�j=q�,(�C�*=                                    Bx��!B  
�          @�G��.{�����Z=q� �
C��.{�|���`���&p�C���                                    Bx��/�  �          @��
���s33�r�\�3��C��q���l���xQ��9��C��)                                    Bx��>�  �          @�\)�B�\�6ff��{�oC�  �B�\�.{��Q��u�RC��                                    Bx��M4  �          @�z�����9������jC�������1G���z��p�
C��R                                    Bx��[�  �          @��;�\)�!���  �{�HC�ٚ��\)�����=q�C���                                    Bx��j�  �          @�z�=����(���{�~�C�G�=����z���  ��C�Z�                                    Bx��y&  �          @�녾���@9����  �nB��f����@B�\����h{B�#�                                    Bx����  T          @�G�>�
=@P  ����]Q�B�B�>�
=@X����z��Vp�B��)                                    Bx���r  �          @���>#�
@H�����\�c�
B�G�>#�
@Q���\)�\�RB��=                                    Bx���  �          @�
=��Q�@s�
����6  B֣׿�Q�@{��|(��/
=BՔ{                                    Bx����  �          @�Q�n{@{������3z�Bνq�n{@����x���,(�B��                                    Bx���d  �          @��>�G�>�����¥�fB?>�G�?(����  ¢�)Bb{                                    Bx���
  �          @�\)?�ff��\)��G�  C�G�?�ff��
=���\z�C�b�                                    Bx��߰  
�          @�  @
=q�(����  L�C��f@
=q�������L�C���                                    Bx���V  �          @��@ �׿Y����  ��C���@ �׿&ff�����C��                                    Bx����  �          @��@zᾨ������C���@z����=q#�C�P�                                    Bx���  �          @�z�@녾u��p�
=C�@논���p�L�C���                                    Bx��H  �          @�(�?�zᾏ\)���\C�j=?�z�u��{C��f                                    Bx��(�  �          @��>�z�?�ff���.B���>�z�?\����ǮB�33                                    Bx��7�  �          @��\?�{�z�H��{�qC��?�{�B�\��
=�3C�ٚ                                    Bx��F:  �          @�\)@,�Ϳ�=q��ff�\�C��=@,�Ϳ�������b  C�^�                                    Bx��T�  �          @���@HQ��p���{�>=qC�g�@HQ��������C��C���                                    Bx��c�  �          @�=q?�{�fff���
��C�J=?�{�.{����\C���                                    Bx��r,  �          @��H?�p������\�=C�T{?�p��k���33�C��3                                    Bx����  T          @��\>���\)��=q��C�>�>��������
aHC�H                                    Bx���x  T          @����B�\��{��Q�C�E�B�\�aG���� �
C�                                    Bx���  �          @�G�?#�
��(�����{C��?#�
��p����\�
C���                                    Bx����  �          @�(���
=�u������Cv;�
=�8Q����H¡\)Cop�                                    Bx���j  �          @���?�Q������4z���\C��f?�Q�����A����C��                                    Bx���  �          @�(�?��R���0������C���?��R��G��?\)� 
=C��                                    Bx��ض  �          @�=q?��������33���C�
=?�����G���\���C�9�                                    Bx���\  �          @�Q�?���p���
�ǅC��
?������"�\��
=C��                                    Bx���  �          @�Q�?
=�9����33�d��C��q?
=�*=q����p\)C�<)                                    Bx���  �          @��>.{��ff���
��C���>.{�E���p�£ffC�c�                                    Bx��N  �          @�ff�#�
������z�­�\C�논#�
���
����³� Cp��                                    Bx��!�  �          @�Q�?(���g����\�?�C�#�?(���X����Q��K�HC�xR                                    Bx��0�  �          @��H?:�H�h�����A��C��=?:�H�Z=q���
�M�RC�
=                                    Bx��?@  �          @�?W
=�p  ���=�RC�H�?W
=�aG���(��J  C���                                    Bx��M�  �          @�ff?.{�e����
�Hp�C�ff?.{�U����T��C�˅                                    Bx��\�  �          @�\)?c�
��Q��~�R�0\)C�>�?c�
�q���ff�<��C��q                                    Bx��k2  �          @�\)>�z������i�����C��>�z����y���+�RC��                                    Bx��y�  �          @�z����{�n�R�&(�C��R���~{�~{�3ffC���                                    Bx���~  �          @�p��������`���ffC�ٚ�������R�qG��&�
C��q                                    Bx���$  �          @�ff�@  ���H�xQ��+�C��=�@  �vff����9\)C�y�                                    Bx����  �          @��?�������Fff�  C�z�?�������W��{C��                                     Bx���p  �          @���?����
=�8����=qC���?�������J�H�\)C���                                    Bx���  �          @��?޸R����.�R��C�O\?޸R��{�@����C���                                    Bx��Ѽ  �          @��@ ������333��C�>�@ ����\)�Dz���C���                                    Bx���b  �          @��?�G������Dz���HC�Ф?�G�����Vff��C�+�                                    Bx���  T          @��H?�ff�����G
=�ffC���?�ff����Y���C��)                                    Bx����  �          @��?�������H�����C���?��������Z�H�(�C�=q                                    Bx��T  �          @�ff?��R�����N�R�
ffC���?��R�����a��  C��                                    Bx���  �          @���?����G��n�R� p�C�&f?��������Q��/=qC��
                                    Bx��)�  �          @��\?����=q����0�C�O\?���q����\�?{C���                                    Bx��8F  �          @�p�>�=q�Z=q����R�
C�K�>�=q�Fff��
=�bz�C��f                                    Bx��F�  �          @��?333�W���
=�Q��C�ٚ?333�C�
��ff�a�C�o\                                    Bx��U�  �          @�?0���E���_��C�T{?0���0����z��oG�C��                                    Bx��d8  �          @�?(���3�
��(��m��C���?(�������=q�}�\C���                                    Bx��r�  �          @�  >��2�\��Q��r(�C���>���H��ff(�C��f                                    Bx����  �          @���?333�;�����iC��=?333�$z�����y�HC���                                    Bx���*  �          @�G�?=p��3�
��  �oQ�C�]q?=p��(���{�z�C�o\                                    Bx����  �          @�z�?B�\�\)�����~�C�xR?B�\�ff��{ffC��3                                    Bx���v  �          @��
>�{����z��qC�J=>�{��\)����33C�5�                                    Bx���  �          @���?E��'
=���\�w33C�*=?E��{��Q��)C��H                                    Bx����  �          @�>�G�� �����\�}G�C�H>�G�����  =qC��=                                    Bx���h  �          @�(�=��
�
�H��p�� C�
=��
��G���=qW
C�XR                                    Bx���  �          @��ý�G��   ��p�k�C�Z��G��������Q�C���                                    Bx����  �          @��ý����	�����G�C��)���Ϳ�(�����L�C�k�                                    Bx��Z  �          @�\)=�G������\)�C�  =�G��������HC���                                    Bx��   �          @�>�  �P  ��(��[�C�,�>�  �7���(��m�
C�u�                                    Bx��"�  �          @�\)?������^{�G�C�` ?������s�
�'
=C���                                    Bx��1L  �          @��R?�=q��\)�j�H� {C�*=?�=q�z�H��  �233C��
                                    Bx��?�  �          @�ff?}p���  �{��.z�C���?}p��j=q����@��C��=                                    Bx��N�  �          @�>L���u������==qC�}q>L���^{���R�PQ�C��                                    Bx��]>  �          @�z�?�����H�k��#��C�z�?���qG���Q��6\)C�
                                    Bx��k�  �          @��
?����z��3�
��C���?����z��L(��p�C�J=                                    Bx��z�  �          @��?�  ��p��4z���C���?�  ����Mp��C��                                    Bx���0  �          @�z�?��\����>�R��G�C��3?��\���H�W
=��C�(�                                    Bx����  �          @��
?�\)�����*�H��G�C�>�?�\)�����C33�  C��{                                    Bx���|  �          @��?�Q����H�E�(�C���?�Q������]p���
C�Q�                                    Bx���"  �          @�33?�G�����<����
=C��H?�G���(��U��Q�C��=                                    Bx����  �          @���?���}p��l���%ffC��?���g���G��8�C�}q                                    Bx���n  �          @�z�?�33���H�P  ��HC�u�?�33�q��fff�!z�C�W
                                    Bx���  �          @�p�?����s�
�~�R�2z�C���?����[�����F{C�~�                                    Bx���  �          @�ff@33�{��e���
C��\@33�e��{��.Q�C��                                    Bx���`  T          @�ff@���  �U�=qC�"�@��j�H�l(��!\)C�7
                                    Bx��  �          @��@�\�tz��fff�
=C�{@�\�^{�|(��1�C�C�                                    Bx���  �          @�z�@���`  �w
=�,��C��)@���G���p��?
=C�:�                                    Bx��*R  T          @�(�@8Q��Dz��k��$�\C���@8Q��-p��|���3�C�e                                    Bx��8�  �          @�{@(�����=q�Y��C�H@(���p������i(�C���                                    Bx��G�  �          @��?�\���
��(�ǮC�f?�\�B�\��\)B�C�^�                                    Bx��VD  �          @���?�{�b�\�z�H�7��C�}q?�{�H����  �L�\C��                                    Bx��d�  �          @���?�{��p��_\)�G�C��=?�{����z=q�)��C�AH                                    Bx��s�  �          @��?������\�s�
�#�
C���?����l(���{�8�C��                                     Bx���6  �          @��\?ٙ��~�R�y���'�C���?ٙ��e������<�C��                                    Bx����  �          @���?ٙ��}p��u�&�
C�� ?ٙ��c�
��
=�;�RC�Ǯ                                    Bx����  �          @�Q�?�
=��Q��q��#��C�^�?�
=�g
=��p��9{C�z�                                    Bx���(  �          @�
=?�33�b�\��p��;Q�C�u�?�33�Fff��  �P�C��R                                    Bx����  �          @�?��Tz���\)�A
=C�P�?��7������U33C�!H                                    Bx���t  �          @�?�ff�P����z��Jp�C��{?�ff�2�\��ff�_ffC���                                    Bx���  �          @�z�?��\(����\�9�\C��q?��@  ����NG�C��=                                    Bx����  �          @�
=?��P����z��H�\C���?��2�\��ff�]z�C�p�                                    Bx���f  �          @��?�G��Dz���p��ZG�C�'�?�G��$z����R�p=qC�{                                    Bx��  �          @�
=?c�
�HQ���{�]
=C��{?c�
�'���  �t(�C�j=                                    Bx���  T          @�p�?}p��G����H�Z(�C�ٚ?}p��'
=�����q33C�n                                    Bx��#X  �          @��?���O\)��ff�O�\C��H?���0  �����f�C��)                                    Bx��1�  �          @�z�?����J�H����S�C�b�?����*�H����jC�!H                                    Bx��@�  �          @�z�?��
�AG����H�YC�t{?��
� ����z��p\)C�~�                                    Bx��OJ  �          @�z�?\�;���=q�X�
C��\?\�=q���
�n��C�3                                    Bx��]�  �          @�\)?�ff�4z����
�Xp�C�>�?�ff�33�����m
=C��                                    Bx��l�  T          @��?��+���
=�]�C�J=?������\)�q��C�c�                                    Bx��{<  T          @��R?��+���Q��a��C��{?��Q������v�C�                                    Bx����  �          @�  ?���1������X�\C�  ?���\)���m
=C��                                    Bx����  �          @��H@	���<������D��C�@	���p������Y(�C���                                    Bx���.  �          @��?����-p���ff�Tp�C�޸?����(���
=�h��C�ٚ                                    Bx����  �          @���@��p���(��\�RC��@�����(��o��C���                                    Bx���z  �          @�{@�H�"�\��
=�O��C��@�H� ����\)�a��C�#�                                    Bx���   �          @�ff@�H�Dz������:�C�R@�H�$z���\)�O
=C���                                    Bx����  �          @�p�@.{�X���hQ��G�C�^�@.{�<����  �3�C�Y�                                    Bx���l  �          @��R@1��N�R�s�
�'(�C�U�@1��0������:��C��\                                    Bx���  �          @��R@8���[��`���33C�\@8���@  �x���,=qC���                                    Bx���  �          @��@8���^�R�aG��p�C�޸@8���B�\�y���+�RC��=                                    Bx��^  �          @���@Fff�s�
�Dz����C��{@Fff�Z�H�`  ���C�q                                    Bx��+  �          @�Q�@;��xQ��E�z�C���@;��_\)�a���C��                                    Bx��9�  
�          @�\)@6ff�w��G
=���C�7
@6ff�]p��c33��C��q                                    Bx��HP  �          @�
=@6ff�����8Q����C���@6ff�h���U��\C�f                                    Bx��V�  �          @�\)@,���z�H�L����RC�H�@,���`  �i���=qC���                                    Bx��e�  �          @�  @(���\)�L(����C���@(���dz��i�����C�<)                                    Bx��tB  �          @��H@@  �w��J=q�ffC�޸@@  �\���g
=�=qC�xR                                    Bx����  �          @��\@�\��33�XQ��{C���@�\�i���w
=�'z�C��                                    Bx����  �          @��\@  �����Mp����C��@  �w��n{��C��                                    Bx���4  �          @�{@����(��B�\��=qC�` @����
=�e��HC�n                                    Bx����  �          @�{@(�����G
=��z�C��@(���z��i�����C��                                    Bx����  �          @�=q@���Q��&ff��Q�C��@������J�H��C��H                                    Bx���&  �          @���@p���33�.{���\C���@p���
=�Q��
G�C���                                    Bx����  �          @��@��z��&ff��ffC�h�@�����J=q�{C�b�                                    Bx���r  �          @��
@�����'���(�C�@���G��L����HC���                                    Bx���  �          @��
@z���p��+��ٮC�33@z���G��P���{C�33                                    Bx���  �          @�z�@G������G����HC��=@G���
=�9����C��)                                    Bx��d  �          @�@z����\�\)���C�{@z���  �8Q���\C��f                                    Bx��$
  �          @�\)@G����5���{C��3@G������Z=q�C�H                                    Bx��2�  �          @�?�\)�p  ���\�0�C�@ ?�\)�L(���G��KffC�5�                                    Bx��AV  �          @�?�G��0  ����ap�C�T{?�G�����\)�y��C�#�                                    Bx��O�  T          @�p�?�\)�:�H�����X�C�K�?�\)������
�p�RC��H                                    Bx��^�  �          @���@ff�B�\��\)�EffC��@ff��H���H�\�\C��                                    Bx��mH  �          @�
=?����%��Q��f=qC���?��Ϳ�33����}��C�0�                                    Bx��{�  �          @�Q�@���(������g��C�"�@�ÿ\�����|�C�S3                                    Bx����  �          @�ff@(���5���
=�D�C�~�@(�������=q�Y�
C��                                    Bx���:  �          @��R@!��Dz���p��?��C��q@!���������V�
C��q                                    Bx����  �          @�
=@���J�H��=q�H33C��@���!G����R�a�C�/\                                    Bx����  �          @�{@,���1���\)�D��C�
@,�������=q�Y�RC�˅                                    Bx���,  �          @�z�@\)�G���=q�=(�C�J=@\)�   ���R�T�RC�s3                                    Bx����  �          @��@
=�:=q��\)�H=qC���@
=�G����H�_��C��                                    Bx���x  �          @��H@
=�8Q�����I(�C��\@
=��R���H�`z�C�P�                                    Bx���  �          @���@��c33�|���/ffC�7
@��=p������JG�C���                                    Bx����  �          @��@���ff�>{����C�@ @��p  �a��G�C���                                    Bx� j  �          @��@�������?\)���
C�AH@���tz��c�
�ffC���                                    Bx�   �          @���?������\�0  ��C��q?�����z��Y���\)C�Ф                                    Bx� +�  �          @��R@p��`���qG��%z�C��@p��<(���\)�?G�C���                                    Bx� :\  �          @�p�@���q��h����C��q@���N�R��z��;C�                                    Bx� I  �          @��?�G��qG��|(��-��C��=?�G��J�H��{�J��C���                                    Bx� W�  �          @�G�?�=q�8�����[p�C�^�?�=q�(���G��wG�C��                                    Bx� fN  
�          @�Q�?����?\)����Q��C���?����33��{�m�C�"�                                    Bx� t�  �          @��@8Q��\)�+����HC���@8Q��c�
�N�R���C�}q                                    Bx� ��  �          @��H@>{�y���,(���  C��q@>{�^{�N�R��C�>�                                    Bx� �@  �          @��\@%�q��I���	�C�1�@%�Q��j=q�$�
C�q                                    Bx� ��  �          @��@�
��Q��K��
�C��3@�
�`  �n�R�&�C���                                    Bx� ��  �          @�33@ff����?\)�33C��q@ff�hQ��c�
�  C�s3                                    Bx� �2  �          @�
=@���Q��   ��C���@����H�H�����C���                                    Bx� ��  �          @��@
=���׿8Q���  C���@
=������R�qC�4{                                    Bx� �~  �          @�p�@�
����  �vffC��{@�
�����p���z�C�:�                                    Bx� �$  �          @�p�@�R��=q�N{�{C�aH@�R�c33�r�\�(�C��                                    Bx� ��  �          @���?�(�� ������W33C�{?�(���=q���p�C��                                    Bx�p  �          @��H?��׿h����z�u�C���?��׾���
=k�C���                                    Bx�  T          @��R?�z������=qu�C��?�z�Q���Q���C�G�                                    Bx�$�  �          @��?�����������C�4{?��׿�ff��G���C�q�                                    Bx�3b  �          @�
=?���$z����H�gz�C���?����������C���                                    Bx�B  �          @���@��c�
�mp��(p�C���@��=p���ff�F
=C�8R                                    Bx�P�  �          @���@
=q�j=q�aG���\C�Ff@
=q�E���G��<
=C���                                    Bx�_T  �          @�{@
=�i���\(���RC��@
=�E��|���:z�C�33                                    Bx�m�  �          @��?��H�`  �o\)�,{C��
?��H�8������J�C�                                    Bx�|�  �          @��H@�\�j=q�S33�
=C��{@�\�G
=�u��7Q�C���                                    Bx��F  �          @���@(��tz��:�H�
=C�H�@(��U��^{� \)C�#�                                    Bx���  �          @��
@�e��O\)�p�C���@�B�\�p  �1z�C��3                                    Bx���  T          @��H@ff�mp��C33�{C�,�@ff�L���e�(��C�.                                    Bx��8  T          @�33@���s33�9���p�C�'�@���S33�]p��!�C�f                                    Bx���  �          @��
@��fff�QG���HC�%@��C33�r�\�3�\C�`                                     Bx�Ԅ  
�          @�
=@   �c33�R�\���C���@   �?\)�s�
�1G�C��\                                    Bx��*  �          @�33@J=q�u��"�\�ظRC��q@J=q�X���G
=���C�~�                                    Bx���  �          @��\@L(��q��#33��z�C��@L(��U��G��G�C�޸                                    Bx� v  �          @��\@^{�i���
=��z�C��@^{�N�R�9�����
C���                                    Bx�  �          @�  ?�z��������u�C�q�?�zῳ33��=q8RC���                                    Bx��  �          @��R?�33�/\)���\�d
=C�n?�33��p����Rp�C�'�                                    Bx�,h  �          @�?��R�!G������d��C�K�?��R��G���z��C�)                                    Bx�;  �          @�{?�(��(�����sC��?�(���33����{C�!H                                    Bx�I�  �          @��?�{�33��G��{(�C��=?�{��G����C���                                    Bx�XZ  �          @��
?ٙ��5��ff�O=qC�l�?ٙ��Q�����m��C�K�                                    Bx�g   �          @��H?�ff�(������[�C�1�?�ff��33��
=�zQ�C��3                                    Bx�u�  
�          @��H@\)�-p��{��@C�˅@\)������[�
C��q                                    Bx��L  T          @���?Ǯ�Z=q�n�R�3ffC�N?Ǯ�0����\)�TC�                                    Bx���  �          @��\?��W��j=q�-�C���?��.�R����M=qC��                                    Bx���  �          @��H@{�U��I���C���@{�1G��i���7(�C�\)                                    Bx��>  �          @�
=@��Vff�:�H���C�� @��5��[��/p�C��=                                    Bx���  �          @�
=@:=q�AG��#�
��33C��3@:=q�#�
�AG����C�Q�                                    Bx�͊  �          @��@'
=�Q��:�H��C�J=@'
=�0  �Z�H�((�C��                                    Bx��0  �          @�@���]p��<����C�L�@���;��^�R�*�\C���                                    Bx���  �          @�{?�33�?\)�p���:��C�1�?�33����ff�Y��C��
                                    Bx��|  �          @���?�Q��.�R���\�U�RC�޸?�Q��p���
=�t�RC�Ff                                    Bx�"  �          @��?�\)�8����p��I�C�}q?�\)�
=q���H�hffC��H                                    Bx��  �          @��H?�  �L���y���;��C�Q�?�  � ����(��\Q�C�t{                                    Bx�%n  �          @��?�\�B�\��G��C��C��?�\����  �c��C��q                                    Bx�4  �          @�=q?�{�@�����\�Hp�C��?�{�������iffC���                                    Bx�B�  
`          @��H?���J=q��p��MC��?���=q�����r  C���                                    Bx�Q`  �          @�Q�?�{�N�R�\)�E�RC�� ?�{� ����
=�i�HC���                                    Bx�`  �          @��
?�=q�HQ��z�H�GQ�C�~�?�=q��H��z��k�\C��)                                    Bx�n�  �          @�z�?�G��E�y���F  C��?�G��Q����
�iffC��f                                    Bx�}R  �          @�Q�?�Q��C�
�s33�9z�C�7
?�Q��������Y=qC��                                    Bx���  �          @�  ?�
=�C33�s33�9��C�!H?�
=�
=��Q��Y�RC���                                    Bx���  �          @��?�33�G
=�p  �7=qC���?�33����
=�W��C�                                      Bx��D  �          @��?����>{�n�R�;��C�Ǯ?�����\���\�C�G�                                    Bx���  T          @��@5��#�
�R�\�#{C��3@5������k��;�C���                                    Bx�Ɛ  �          @�Q�@e�� ���z��߮C�xR@e����.{�33C���                                    Bx��6  "          @���?�G��K��o\)�:�HC��{?�G��\)��\)�]�
C��                                     Bx���  �          @��?�(��L���a��4z�C�Q�?�(��"�\�����W�
C�                                      Bx��  "          @��@���e��+��Q�C���@���C�
�P���#�C��3                                    Bx�(  �          @�@\)�Z�H�-p��G�C���@\)�9���QG��&��C��H                                    Bx��  �          @�ff@!��J=q�  ��Q�C�^�@!��.{�0���z�C�|)                                    Bx�t  �          @�Q�@G
=�G��޸R�\C��\@G
=��
=�
=��C��                                    Bx�-  "          @��@H����R��ff��C�P�@H�ÿ�����{C�@                                     Bx�;�  T          @�(�@XQ���Ϳ��H���\C��=@XQ�Ǯ��G����HC��
                                    Bx�Jf  �          @��@j�H������ܣ�C�ff@j�H��
=����
=C��                                    Bx�Y  "          @��@L�Ϳ������
C��@L�Ϳ��
��\�	\)C��                                    Bx�g�  T          @��H@333��p��	��� =qC�aH@333�����{��C�]q                                    Bx�vX  �          @���@��Tz��p���C�B�@��8Q��0����\C�5�                                    Bx���  �          @�z�@33�\�Ϳ��R��p�C�Ф@33�C33�$z����C��H                                    Bx���  �          @�33@
=�?\)��33��z�C�+�@
=�&ff����	\)C�)                                    Bx��J  T          @xQ�?�  �L�Ϳ�����p�C���?�  �9����
=� z�C���                                    Bx���  �          @c33@(Q��=q��p�����C��q@(Q���
���
���HC��                                     Bx���  �          @�
=@z��@  �����z�C�P�@z��'�����=qC�/\                                    Bx��<  �          @��B�\������
=�HCm�;B�\�#�
���H¤��C@}q                                    Bx���  �          @'��˅?���Q���(�C���˅?��Ϳ^�R��
=C��                                    Bx��  T          @q녿�p�@5������ffB��f��p�@B�\�\(��T  B�B�                                    Bx��.  �          @�ff�ff@]p��Q��׮B�B��ff@r�\��(���Q�B�{                                    Bx��  �          @�G����@c�
��
��(�B�����@w������
=B��                                    Bx�z  �          @�{�
=@j�H�   ��=qB��
�
=@~{������B�                                    Bx�&   �          @�  �ff@j=q�(��ۮB�R�ff@\)��  ����B�z�                                    Bx�4�  �          @�G����@e��z����B����@|(�������HB�
=                                    Bx�Cl  �          @�{��Q�@C�
�8Q����B���Q�@a��G���p�Bힸ                                    Bx�R  �          @��?:�H�����(��z��C�Y�?:�H������{z�C�o\                                    Bx�`�  T          @�(�?�G��}p��Q���C�k�?�G��S33�|(��=�C�G�                                    Bx�o^  �          @��R?��}p��c33�"C�B�?��P  ���R�J�\C��\                                    Bx�~  �          @�\)?c�
�w
=�o\)�-\)C��f?c�
�G
=���
�U��C��                                    Bx���  T          @�\)?E��g
=�����>p�C�H?E��3�
��z��gG�C���                                    Bx��P  �          @�=q>���x���}p��5ffC���>���Fff��33�_(�C���                                    Bx���  "          @�33>�ff��{�l(��$�RC��>�ff�\����z��N�\C��3                                    Bx���  T          @��H?�R��G��aG��ffC�R?�R�dz���  �F
=C��f                                    Bx��B  T          @�\)?@  ���H�j=q��C���?@  �fff��z��HffC��                                    Bx���  �          @���?E����R�e���C��?E��n�R��33�C=qC��R                                    Bx��  �          @��?0�������Q���
C�O\?0���w�����7�C�                                    Bx��4  �          @�\)?������O\)�	�C�s3?���{������2�HC���                                    Bx��  "          @�z�?��
��(��n{��\C�%?��
�g����R�E
=C��)                                    Bx��  �          @�33?������\(����C��=?���x����\)�9(�C��{                                    Bx�&  T          @�33?��������L(��Q�C�xR?�����33�����-�C�|)                                    Bx�-�  �          @��H?�
=��\)�4z����C��?�
=���
�l(���C���                                    Bx�<r  T          @��\?��
�����7���ffC�T{?��
�����n�R�\)C�T{                                    Bx�K  �          @��?��\��
=�-p��޸RC�#�?��\��z��e���C�                                    Bx�Y�  T          @��H?�\)��  ��R���
C�U�?�\)��  �G����C��                                    Bx�hd  T          @���?k���p�����G�C�?k���{�@��� ��C��\                                    Bx�w
  �          @�p�?L����=q��G����\C�G�?L�������.{��p�C���                                    Bx���  �          @�33?O\)��
=��=q���HC�e?O\)�����1G���C���                                    Bx��V  T          @���?�ff���Ϳ�z���{C��?�ff��  �%��Q�C�=q                                    Bx���  "          @��?\)��33��\)��Q�C��?\)����2�\��  C�j=                                    Bx���  �          @��   ��Q�L�Ϳ�C�C׿   ������
�4(�C�5�                                    Bx��H  �          @���<#�
��{�=p����C��<#�
��ff�޸R��Q�C�
=                                    Bx���  T          @�zᾙ�����
�����W
=C�s3�������R�����b�\C�g�                                    Bx�ݔ  �          @�G�?=p���(��^�R�=qC��=?=p����
��\)���HC�)                                    Bx��:  �          @�Q�?�Q����\�+�����C�Ff?�Q������z�����C��=                                    Bx���  �          @�\)?��
��Q쿂�\�,Q�C���?��
��\)���R��G�C���                                    Bx�	�  �          @�ff?����  ��\)���C�U�?������!�����C��                                    Bx�,  T          @�\)?(����R�����w
=C�\)?(����H�=q���C��)                                    Bx�&�  �          @���?��
��������(�C��?��
���R�$z���ffC��)                                    Bx�5x  �          @��\>W
=��ff��
=���HC�*=>W
=����8Q���  C�G�                                    Bx�D  �          @�(��!G����׿�{����C��
�!G���=q�5���p�C�Ff                                    Bx�R�  T          @��\�k���{��(���z�C��k���
=�:�H���C���                                    Bx�aj  T          @��H�333��z�����
=C�  �333����>{� ��C��)                                    Bx�p  T          @�ff��\)����������RC~�Ϳ�\)�����K��p�C}#�                                    Bx�~�  �          @�
=��=q��\)��(��uG�C�&f��=q�������=qC��
                                    Bx��\  �          @�p��Ǯ���\�\(���C�f�Ǯ��=q��33��G�C��                                    Bx��  �          @��H�c�
�����G��P��C�N�c�
�����  ���HC���                                    Bx���  T          @���\)��p����
�S�C���\)���\��\���C��{                                    Bx��N  �          @�33����(������p(�C�uþ���  ����ϙ�C�Ff                                    Bx���  T          @�(��u�����33�<��C��)�u��p��
�H��=qC���                                    Bx�֚  T          @�(�=�Q���  �����5G�C�s3=�Q���{�Q�����C�y�                                    Bx��@  �          @��?����p���p��J�RC���?�����H�\)��
=C��                                    Bx���  �          @��>����
���R�w�
C���>�����\)��{C��q                                    Bx��  T          @��?u���������HC�k�?u���H�#�
��p�C��                                    Bx�2  �          @�  @ff�����Ǯ��C�P�@ff�����   ����C�*=                                    Bx��  	�          @�p�?s33��33�˅����C��?s33��ff�%�ٙ�C�n                                    Bx�.~  
.          @�ff?�ff��z�Ǯ�~{C�<)?�ff���R���g
=C��f                                    Bx�=$  �          @���?����녾����UC��\?�����Ϳ���\  C�R                                    Bx�K�  �          @��R?�(����H�L���G�C�'�?�(���ff��(��F=qC�ff                                    Bx�Zp  �          @�
=?������aG����C���?����
=��z����\C��                                    Bx�i  �          @�Q�?����G�����%�C���?���������C�&f                                    Bx�w�  �          @�Q�?����Q쿠  �H��C��{?����p���\��p�C��                                    Bx��b  �          @���?aG���Q쿽p��mC��=?aG����
�!G��Ώ\C��                                     Bx��  �          @���?s33��  �Ǯ�z�\C��?s33��33�%�ԸRC�Ff                                    Bx���  �          @�G�>�G���{�(�����C�l�>�G����
�X����C��R                                    Bx��T  �          @�Q�>�33�����\)��p�C��>�33��  �Mp��{C��                                    Bx���  �          @����\)����=q��\)C����\)��
=�6ff��ffC��q                                    Bx�Ϡ  �          @��R<#�
��ff��
=��
=C�<#�
�����,����(�C�\                                    Bx��F  T          @�
=>\)��녿���S�C��R>\)���R�
=����C��                                    Bx���  T          @�  <#�
���\��G��K�C��<#�
��\)�z�����C��                                    Bx���  T          @�Q�?����=q��  �H��C��?����
=�33����C�E                                    Bx�	
8  �          @��?W
=��ff���
�x��C�g�?W
=�����#�
��ffC���                                    Bx�	�  �          @�\)?xQ����
��
=����C�
?xQ���ff�,(���RC��                                    Bx�	'�  �          @�\)?�ff������\�M�C�n?�ff��z���
��{C��=                                    Bx�	6*  �          @���?�p�����  �G�
C��H?�p����H�����C��                                    Bx�	D�  �          @��@����R���<Q�C��H@���z��	�����C�E                                    Bx�	Sv  �          @��׿����������
C}=q������H�HQ���C{xR                                    Bx�	b  �          @��H��
=��  �
=��\)C~� ��
=����E���G�C}5�                                    Bx�	p�  �          @��������z��{�Ǚ�C�\��������Z�H��C}�{                                    Bx�	h  �          @�(���{��������ŅC0���{���\�Y����
C}n                                    Bx�	�  �          @�z����Q�� ���˙�C{�
�����\(��  Cy\)                                    Bx�	��  �          @�p��������������C}+�������Q��HQ���
=C{u�                                    Bx�	�Z  �          @�{��
=��Q��(���\)C|LͿ�
=��\)�J�H��Czz�                                    Bx�	�   �          @����z����\������CyW
��z������QG��ffCw
                                    Bx�	Ȧ  �          @����Q���ff�+��؏\C{&f��Q����H�e�ffCx��                                    Bx�	�L  �          @��
��Q���z��8����z�C}�
��Q���\)�q�� ��C{:�                                    Bx�	��  �          @�zῼ(���Q��*�H��33C}����(���z��fff�\)C{��                                    Bx�	��  �          @���c�
��z��)�����C�ÿc�
�����fff���C�o\                                    Bx�
>  �          @�������33�,(��ڸRC�������\)�hQ���C�P�                                    Bx�
�  �          @��
������(��B�\��Q�C��
������ff�{��(�
C�                                     Bx�
 �  �          @��Ϳ�����=q�E��G�C~T{������(��~{�)�\C{޸                                    Bx�
/0  �          @�zῃ�
��\)�U��
(�C��q���
�\)��{�5��C��                                    Bx�
=�  �          @�p����
��ff�\(��33C��쿃�
�|����G��9�
C\)                                    Bx�
L|  �          @��
�fff��=q�c�
��C�lͿfff�r�\��(��A\)C�Q�                                    Bx�
["  �          @��
�333��G��{��(Q�C�aH�333�\����{�Tp�C�Ff                                    Bx�
i�  �          @�z�@%���33��\����C�f@%�����<(����HC�S3                                    Bx�
xn  �          @�@����������C�1�@������Fff��=qC��                                     Bx�
�  �          @�z�@z���������
C�y�@z�����@����ffC���                                    Bx�
��  �          @�G�@-p���=q��z���{C��=@-p������$z���p�C��=                                    Bx�
�`  �          @���@$z���z�� ����Q�C��R@$z������;����HC�q                                    Bx�
�  �          @��@\)��  �@����z�C�q�@\)�u��u����C�z�                                    Bx�
��  �          @�  @8�����
�=p���C��)@8���mp��p  �Q�C��3                                    Bx�
�R  �          @��R@`���^{�R�\��C���@`���0���y���$�C��                                    Bx�
��  �          @��@w
=�6ff�Z�H��C��f@w
=�Q��z=q�&=qC���                                    Bx�
�  �          @���@�=q�%��Y����
C�Ǯ@�=q��\)�u�"ffC��H                                    Bx�
�D  �          @�p�@H���X���[���C�h�@H���*=q�����1=qC��                                     Bx�
�  �          @��?�����
�Vff�Q�C���?����(�����1G�C���                                    Bx��  �          @\?��
���\�W
=�{C���?��
���H��\)�0p�C�B�                                    Bx�(6  �          @��?�
=��{�hQ��\)C���?�
=�x����
=�:{C��=                                    Bx�6�  �          @��?˅���
�Z=q�p�C��?˅���
��G��0��C���                                    Bx�E�  Y          @���?����Q��N{��ffC���?����G���z��*{C���                                    Bx�T(  T          @�p�?޸R�����S33���C�Ǯ?޸R��p���{�+ffC�W
                                    Bx�b�  "          @�p�@33��\)�O\)���C��{@33��Q���33�&33C��                                    Bx�qt  "          @\?У�����0�����
C�3?У�����k��(�C�G�                                    Bx��  �          @���@8Q���  �4z��޸RC�O\@8Q��w��h���33C�U�                                    Bx���  �          @��@C33����'
=��\)C��@C33�y���[��
�HC�H                                    Bx��f  T          @�G�@?\)��{�-p��ָRC��R@?\)�u��aG����C���                                    Bx��  �          @��H@E�����%�ɮC�H@E��~{�Z�H�	�C��                                    Bx���  �          @�33@C33���\�%���33C��R@C33�\)�Z�H�	  C��{                                    Bx��X  
(          @�
=@R�\��  �H����=qC��H@R�\�c�
�y���G�C�^�                                    Bx���  �          @�  @_\)��\)��\��p�C�<)@_\)����:�H���C��)                                    Bx��  
�          @�p�@>�R����  �`��C��3@>�R�����{��
=C��                                    Bx��J  T          @�z�@=p�����������  C�,�@=p���33�0����(�C�\)                                    Bx��  T          @��@L����ff�����IC�t{@L����33�  ��
=C�l�                                    Bx��  
�          @�
=@���n{�L�Ϳ   C��
@���h�ÿE����C�                                    Bx�!<  
_          @�
=@������W
=��p�C�/\@����  ��  ��C��R                                    Bx�/�  T          @��@z�H���׾����8Q�C�xR@z�H��(����3�
C��                                    Bx�>�  "          @�  @c�
���������
=C�C�@c�
���
����UC�˅                                    Bx�M.  
�          @��@X�����;����\)C�XR@X����
=��33�YG�C��)                                    Bx�[�  T          @�ff@N{��\)�u�C�s3@N{���H�����:=qC��3                                    Bx�jz  "          @��R@	����z῜(��>=qC�ٚ@	������\)��G�C�~�                                    Bx�y   
�          @�G�@!G����\��ff�(z�C�,�@!G���G�� ����\)C��                                     Bx���  T          @\@K���33�:�H��z�C��)@K����
��Q����C��R                                    Bx��l  
�          @�Q�@(������
=��(�C�q@(�����Ϳ�{�x��C���                                    Bx��  "          @���@ ����ff�&ff��C�c�@ ����
=��
=���C��)                                    Bx���  "          @�G�@p���녿O\)��  C��
@p�������\)��{C�P�                                    Bx��^  �          @�Q�?�z�����c�
�z�C�c�?�z����H�������C���                                    Bx��  
(          @�
=?��
��{�!G���z�C���?��
���R���H��ffC���                                    Bx�ߪ  "          @��R?�����{�:�H��G�C��?�����ff��ff���C�l�                                    Bx��P  
�          @���?�=q����Ǯ�u�C���?�=q��\)��(��g33C��                                    Bx���  
�          @�z�?ٙ��s33�%���C�
?ٙ��N�R�QG��'z�C���                                    Bx��  �          @��R?�z��B�\�p���9ffC��?�z���������]�C�)                                    Bx�B  "          @�{?�������33�F�\C��?��������
��33C�33                                    Bx�(�  �          @�  ?�p���녿�\)�k�C���?�p����R��
���C�S3                                    Bx�7�  
�          @��?W
=���R��=q��
=C��R?W
=�����2�\���
C��                                    Bx�F4  �          @�Q�?5���H��z�����C��
?5��(��5��  C�Z�                                    Bx�T�  
�          @�=q?aG������  ����C���?aG���  �J=q�
�C��                                    Bx�c�  
Z          @�(�?=p�����3�
��C�Y�?=p���{�j�H�#33C���                                    Bx�r&  
�          @�?(�����hQ��=qC�
=?(��b�\��(��J�\C��f                                    Bx���  '          @�33?h���xQ��z�H�233C���?h���C�
���H�\��C�<)                                    Bx��r  �          @��?�Q������r�\�(Q�C�@ ?�Q��O\)����Q�HC�                                      Bx��  �          @�33?��l(��}p��4=qC���?��7����H�\\)C�0�                                    Bx���  �          @�{?��H��
=�R�\��C��?��H�qG����\�7p�C��                                    Bx��d  �          @�p�?n{��Q��{��
=C�=q?n{��ff�XQ��\)C��H                                    Bx��
  �          @��R?���n{���\�7�HC���?���8�����R�`�C�                                    Bx�ذ  �          @�
=?��R�S�
����O33C�=q?��R�����G��w(�C��q                                    Bx��V  �          @��?�G��a�����8�C�:�?�G��,(���ff�^�\C���                                    Bx���  T          @�
=?�
=�|(��h�����C��?�
=�K���=q�D�HC���                                    Bx��  
�          @��?��
�^{���
�=��C��?��
�(Q���ff�d\)C�"�                                    Bx�H  
�          @�33?��H�Vff��\)�DC��H?��H�\)��G��k�\C�&f                                    Bx�!�  O          @�\)?���J�H����K�C��?���z���Q��qC�p�                                    Bx�0�  �          @��\?����I������L�
C�XR?�������=q�r��C�0�                                    Bx�?:  
�          @��@Q��U��33�>��C�@ @Q��{�����az�C�ff                                    Bx�M�            @��
?�ff�:=q�����YffC��?�ff���H��\)�|  C�O\                                    Bx�\�  
�          @��H?Q��33��G�ffC���?Q녿��\��(��fC�Y�                                    Bx�k,  �          @�Q�>��H��Q�����k�C�+�>��H�c�
��p���C��                                    Bx�y�  �          @�33?z��{��p��z��C���?z��G�����C���                                    Bx��x            @���}p��AG���z��c�C{޸�}p���
����
CtQ�                                    Bx��  Y          @���L���3�
�����w�C��3�L�Ϳ��
��ff(�C��                                    Bx���  �          @���R�ff������C8R��R��ff��  �
Ct�=                                    Bx��j            @�
==�����p�G�C�j==���\)��Q���C�e                                    Bx��  Y          @�z�@(��:�H�|(��<�RC�j=@(��������]
=C���                                    Bx�Ѷ  �          @��?�  �%��u��KQ�C�  ?�  ��������l��C��{                                    Bx��\  �          @�=q?���H�r�\�P�C�\)?������q�
C��                                    Bx��  
-          @���?�����
�l���Mz�C�XR?��Ϳ�=q����lQ�C��{                                    Bx���  �          @�G�@%����1G��=qC�Ff@%���p��I���5\)C�
                                    Bx�N  �          @�=q@7
=���<��� 33C�G�@7
=�����QG��6\)C���                                    Bx��  �          @��H@3�
���@���#�C��@3�
��Q��U��:33C�k�                                    Bx�)�  �          @���@;�����"�\�=qC��@;���Q��<(����C�E                                    