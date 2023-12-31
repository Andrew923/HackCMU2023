CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230124000000_e20230124235959_p20230125020701_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-25T02:07:01.415Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-24T00:00:00.000Z   time_coverage_end         2023-01-24T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill         �   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx^0�  �          @�  >����u@�
=B��{C�y�>�����@�ffB�RC���                                    Bx^?f  �          @�Q�?p��>��@�z�B���Aw�?p�׿��@��\B���C�Ф                                    Bx^N  �          @��H?���    @���B���=��
?�����@��Bzp�C�                                      Bx^\�  T          @�33?��#�
@�\)B��3C�T{?��   @��Bj�\C��                                    Bx^kX  
�          @��\?�Q�?��@�z�B�\A�G�?�Q쿽p�@��RB�33C�^�                                    Bx^y�  �          @�Q�?�
=?}p�@�Q�B��
B �
?�
=���@�  B��)C�                                      Bx^��  �          @�\)?��?�@���B�B�B!  ?�녿Y��@�
=B��qC�L�                                    Bx^�J  �          @�Q�?�G�?�\@�=qB���A���?�G����
@��B���C�                                    Bx^��  �          @��?�
=>��H@��\B��fA�=q?�
=���@��B�=qC�]q                                    Bx^��  
�          @�  ?��
=�@�(�B�=q@�(�?��
��z�@���B��C�#�                                    Bx^�<  T          @�G�?u>k�@�B�p�AVff?u��=q@�33B���C��)                                    Bx^��  T          @��?�  >�p�@��
B��=A�p�?�  ��@��B�z�C�S3                                    Bx^��  T          @���?��H>Ǯ@�z�B�33A�ff?��H��33@�(�B�p�C�3                                    Bx^�.  �          @�Q�?��>�=q@�z�B�Q�Ahz�?����\@��HB���C�E                                    Bx^��  �          @��R?�>��@���B�u�A�\)?���{@���B�\C��                                    Bx^z  �          @��H?J=q>���@�Q�B��{A�Q�?J=q��z�@�\)B��C���                                    Bx^   "          @��R��p����R@��B���C[B���p���@�33B{��C�#�                                    Bx^)�  �          @�녿!G�>�33@�ffB��=CaH�!G��\@��RB�ffCwh�                                    Bx^8l  �          @����Ǯ=u@�
=B�k�C*�H�Ǯ���@��B�u�C��R                                    Bx^G  
�          @��>aG�<�@�{B��{@�Q�>aG���@��\B�L�C�w
                                    Bx^U�  �          @�\)��{�\@�G�B�L�C?�׿�{��@s33BXG�Cc�\                                    Bx^d^  �          @�ff�<(����\@_\)BA33CG&f�<(��33@3�
B  CZ�                                    Bx^s  T          @�\)�2�\��{@n�RBTG�C:�H�2�\��\@S33B4(�CTh�                                    Bx^��  �          @�G�<#�
?Q�@���B��B�(�<#�
���@�33B�{C�*=                                    Bx^�P  �          @��ÿaG�>�z�@�p�B��C!�=�aG��Ǯ@��B��)Cpz�                                    Bx^��  �          @����E�>�@�(�B�ǮC�3�E���{@�ffB�=qCp��                                    Bx^��  T          @�G�?�Q���@�G�Bg�\C�T{?�Q��_\)@9��B(�C�4{                                    Bx^�B  �          @�\)��zῙ��@�ffB��C�N��z��5@hQ�BO�C�                                      Bx^��  
�          @��ÿ5���
@�(�B���Cu�5�G�@\(�B<33C���                                    Bx^َ  T          @�G���G����
@���B���Cas3��G��-p�@qG�BQ�HCy��                                    Bx^�4  T          @����=p���ff@�
=B�  Cp:�=p��;�@g
=BH{C��                                    Bx^��  �          @���.{���@�\)B��\Cm���.{�.{@mp�BS�C�
=                                    Bx^�  �          @��R>\)�&ff@�33B��{C���>\)�Q�@~�RBl(�C��f                                    Bx^&  �          @����#�
���@p��B�  C��H�#�
�>{@5�B.�C��\                                    Bx^"�  
~          @��R?#�
���@}p�BoC�&f?#�
�c�
@2�\B��C�)                                    Bx^1r  
�          @���=#�
��H@}p�Bj(�C�|)=#�
�qG�@-p�BC�O\                                    Bx^@  �          @��=u�!G�@fffB\
=C��R=u�n{@B C�|)                                    Bx^N�  "          @��\����Q�@�=qBn�HCs�\����b�\@;�B�C}\)                                    Bx^]d  
�          @��	���aG�@���Bp��CJ.�	���Q�@X��B:p�Cc�)                                    Bx^l
  "          @����fff@���Bn�\CJQ������@W�B8��Cc��                                    Bx^z�  "          @�Q��ff�fff@���Bg�RCI��ff���@W
=B4(�Cau�                                    Bx^�V  �          @�  �{�#�
@���Bi��C4=q�{����@mp�BN�HCT^�                                    Bx^��  �          @�ff�4z�>��@mp�BQ�C*�{�4z῀  @fffBI�CG��                                    Bx^��  T          @��\�����\)@uBS�
CU������>{@:=qBG�Cg{                                    Bx^�H  T          @�z��1녿��@vffBQG�CH���1��p�@H��B!
=C]��                                    Bx^��  �          @���$zῈ��@~{B[CJ�{�$z��!G�@P  B(G�C`c�                                    Bx^Ҕ  "          @�G��8Q�fff@n�RBL  CEaH�8Q��G�@EB �\CZ:�                                    Bx^�:  T          @�33�C�
����@G�B+�CL=q�C�
��R@ffA��RCZ�3                                    Bx^��  �          @�  ���=��
@c33B�Q�C0� ��녿��@S�
Bq  C^�3                                    Bx^��  T          @�
=�`  ���@z�A��RCH@ �`  ��?��A���CR�                                    Bx^,  
�          @���Z=q��33@.�RB��CF���Z=q��@�A�33CS��                                    Bx^�  "          @���{���=q@�HA�CF���{��
=q?�(�A�  CP�                                     Bx^*x  
�          @����g
=��@:=qB��CE�q�g
=���@\)A�33CS\)                                    Bx^9  �          @�33�qG���33@5�B�HCD�3�qG��	��@�A��CQ�                                    Bx^G�  �          @��
�k���G�@:=qB�CF��k���@��A׮CSǮ                                    Bx^Vj  
Z          @�\)�)�����@9��BQ�C\c��)���L��?���A��CfQ�                                    Bx^e  	�          @�����Dz�@@  B�RCk����~�R?���A�Q�CrE                                    Bx^s�  	�          @����ff�5�@A�B��Cf^��ff�q�?�(�A��HCn&f                                    Bx^�\  "          @�������33@Z=qB5�RC_&f����[�@G�A���Cjn                                    Bx^�  
�          @�Q��<(����H@\��B5��CR)�<(��8��@!�A��\C`��                                    Bx^��  
�          @��\�I�����R@:�HB��CMQ��I���\)@Q�A��CZQ�                                    Bx^�N  �          @��\�ͿQ�@AG�B!��CAu��\�Ϳ�z�@{B p�CQ�                                    Bx^��  T          @���R�\���H@W
=B5=qC<z��R�\��p�@:�HB��CO��                                    Bx^˚  �          @����W��.{@X��B4�\C6�)�W���Q�@Dz�B��CK=q                                    Bx^�@  �          @�Q��e�>Ǯ@E�B!�C-� �e��O\)@?\)BC@�3                                    Bx^��  
�          @��H�l(�?�R@A�B�
C*ff�l(���@B�\Bp�C<�{                                    Bx^��  
�          @����j�H?�p�@33A���C���j�H?8Q�@2�\B�RC(�H                                    Bx^2  "          @�G��S�
@�?�A�(�C���S�
?�=q@   B�C\                                    Bx^�  T          @�=q�W
=<#�
@\��B6��C3���W
=��ff@L(�B&(�CI
                                    Bx^#~  T          @���(��?#�
@qG�BX�HC&u��(�ÿW
=@n�RBU�\CE�H                                    Bx^2$  �          @���:=q?���@VffB<(�C�q�:=q�aG�@a�BI�RC8Y�                                    Bx^@�  �          @�33�W�?��@3�
BffC"!H�W��L��@@��B&�
C4��                                    Bx^Op  
�          @���S33?��@\)B
=C=q�S33?O\)@AG�B&��C&J=                                    Bx^^  T          @�ff�@�׾�
=@VffB?G�C;��@�׿�33@<(�B"�CP��                                    Bx^l�  T          @��A녿(��@`  BA��C@aH�A녿�
=@?\)B  CT�{                                    Bx^{b  �          @�ff� �׿.{@n{B\{CC8R� ����\@L(�B2p�C[                                    Bx^�  �          @�  ���?(��@~{Bm33C#�{��׿c�
@{�Bh�CIc�                                    Bx^��  �          @�G�����@>�R@R�\B0�RB�Q����?��
@�p�BxQ�CJ=                                    Bx^�T  �          @��;#�
@q�@�A��B��
�#�
@'�@e�BV��B���                                    Bx^��  T          @��Ϳ�33@<��@J�HB0�B�R��33?��@���B{
=C.                                    Bx^Ġ  "          @��
��z�?�{@�{B��
C�\��zᾞ�R@���B�aHCC!H                                    Bx^�F  �          @�p���ff?��@��RB���BΣ׾�ff>.{@��B�33Cu�                                    Bx^��  
�          @�{�#�
@�R@xQ�Bl=qB�\�#�
?�R@�B�=qC�                                    Bx^�  �          @��7�>�\)@dz�BLQ�C.aH�7�����@Z�HB@�CH�                                    Bx^�8  �          @�\)�ff?}p�@x��Bc33C+��ff�
=q@~�RBkG�C@��                                    Bx^�  
Z          @�z����@Dz�@5�B\)B��3���?��@qG�BN�RC)                                    Bx^�  
�          @�
=��Q�@AG�@b�\B6�B�k���Q�?�p�@��B|{C
�q                                    Bx^+*  
(          @��R��(�@S�
@N{B#\)B����(�?�\)@�
=Bk��C�R                                    Bx^9�  "          @�����
?���@vffB\=qC��
���@��HBr�C7�f                                    Bx^Hv  T          @�G��\)?�{@\)Bb�C�R�\)�aG�@��RBwffC9�\                                    Bx^W  
�          @������@=p�@C�
B(�C �����?�{@|(�BW{C�                                    Bx^e�  
�          @�(��(Q�?�z�@`��BJ��CG��(Q�k�@l(�BYz�C8�                                    Bx^th  b          @�(��\)?��@j�HB`�C.�\)����@s33BlQ�C>#�                                    Bx^�  �          @���ff?��@C�
B;z�C)�ff>Ǯ@\��B]p�C*��                                    Bx^��  �          @�\)��\@aG�@(Q�B 
=B�{��\@33@p  BD33C�)                                    Bx^�Z  "          @�  �
=q@aG�@,��BB���
=q@�@s�
BJ=qCp�                                    Bx^�   
�          @���
=@vff@�A�{B�#׿�
=@+�@k�BE
=B�{                                    Bx^��  
Z          @�z��C33���@^�RBA�C?\�C33��@AG�B!33CS{                                    Bx^�L  
�          @���p����
@���B�u�C4���p���Q�@�
=Bp��C`c�                                    Bx^��  S          @����H?���@|(�BZ��C=q��H�#�
@�p�BoffC7ٚ                                    Bx^�  �          @�
=��33@�@ffA�(�B�{��33@Vff@e�B:��B���                                    Bx^�>  "          @�{�^�R@s�
@5B�B��H�^�R@   @�G�Bc
=Bڙ�                                    Bx^�  �          @��R�?G�@�Q�Bo�C!����Tz�@�  Bn��CG��                                    Bx^�  
�          @����
����@�{Bs��C<=q��
��@qG�BO�HCZ��                                    Bx^$0  
g          @��\�{���\@��\Bl\)CL�{���@[�B8(�Cc�                                    Bx^2�  �          @���
=@G�@*=qBB=qB��
=?��@S33B�u�B�aH                                    Bx^A|  
�          @���?�(�@��\?B�\A
{B��?�(�@�
=@p�A�Q�B���                                    Bx^P"  �          @�z�?���@�33?�\@��\B��?���@��H@\)A���B��{                                    Bx^^�  �          @�(�?�(�@^{@o\)B0(�B~��?�(�?�\)@�  Bw�B=�                                    Bx^mn  T          @�z�?Y��@���@\(�Bz�B�{?Y��@   @�p�Bq�B�W
                                    Bx^|  T          @�(�>�p�@|(�@i��B*��B�G�>�p�@ff@��\B(�B�{                                    Bx^��  
�          @��>aG�@hQ�@vffB:\)B��>aG�?��R@���B��\B�u�                                    Bx^�`  �          @�\)��\)@�\@��HBx�\B��q��\)?\)@��
B���B�\                                    Bx^�  �          @�\)�L�Ϳ�p�@`  B3\)CL�
�L���)��@.�RBffC[��                                    Bx^��  
�          @���I���z�H@tz�BDCEE�I����\@Mp�B�HCX\                                    Bx^�R  
�          @����5�8Q�@��HB\��C7���5��33@p  BC(�CR�                                    Bx^��  �          @�{�{?&ff@�ffBj�RC%.�{�fff@���Bf��CH{                                    Bx^�  
�          @�(��1�?!G�@|��BX�C':��1녿Tz�@z=qBU�HCD��                                    Bx^�D  �          @�p��8��>k�@~�RBW�\C/p��8�ÿ��H@r�\BI�RCJ�                                    Bx^��  "          @�
=�G�?�=q@���B^
=C!H�G�<�@��\By=qC38R                                    Bx^�  
�          @��
�Ǯ@#33@r�\BN�
B��H�Ǯ?��\@�ffB��C�R                                    Bx^6  
Z          @��\����>#�
@��\B�u�C/n������33@�33Bn��CW��                                    Bx^+�  
�          @��H��Q�?�G�@��B{��B�
=��Q�>��@���B�L�C,�
                                    Bx^:�  
�          @��ÿTz�@=q@z=qBcz�B�{�Tz�?\(�@���B�W
C+�                                    Bx^I(  /          @��ͽ�Q�@z�@u�BkG�B�����Q�?L��@��B�Q�B�G�                                    Bx^W�  �          @�33�k�@HQ�@C�
B,ffB��f�k�?�@~{BzffB�ff                                    Bx^ft  y          @�G��\)@P��?���A\B��
�\)@��@8��B�
C@                                     Bx^u  �          @��\�J=q?���@��B��C#��J=q>��@.{B"Q�C/\)                                    Bx^��  
�          @���8�ý��
@<��B6Q�C5���8�ÿ�\)@/\)B%��CI&f                                    Bx^�f  �          @����
=��\)@p��B`�CT��
=�'
=@A�B((�Cg                                      Bx^�  
�          @�33���׿��@�  Bw�Ce�ÿ����<��@J=qB0z�Ct�                                    Bx^��  
�          @�
=�n{��@�B{��Ct�n{�QG�@O\)B.Q�C~
                                    Bx^�X  
�          @�\)��p��Ǯ@�  B��)Cg�f��p��=p�@[�B;�RCw\)                                    Bx^��  "          @������
=@uB^p�Cc@ ����J=q@:�HB�Cp�=                                    Bx^ۤ  �          @�����Ϳ��H@x��B]  Cb�H�����L��@<��BQ�Co��                                    Bx^�J  "          @�G���G���@|��B_G�Ce���G��Q�@?\)B��Cq��                                    Bx^��  
�          @�G���{�(�@�=qBkp�Cs&f��{�^�R@B�\B�C|O\                                    Bx^�  �          @�녿����
=@z�HB`ffCiff�����Vff@;�Bz�Ct�f                                    Bx^<            @�Q��z��33@���Bop�C`�ÿ�z��=p�@L��B-\)Cp                                    Bx^$�  /          @�\)�Q��Q�@eBH(�Ca
=�Q��P  @'�B�Cl�=                                    Bx^3�  �          @�G��ff��z�@w
=BV�\C^@ �ff�H��@<��B{Cl&f                                    Bx^B.  �          @�(����R��{@���B_�RC_)���R�H��@HQ�B (�Cm��                                    Bx^P�  
�          @�=q��  ��
=@���BdQ�Cc녿�  �Mp�@FffB!G�CqxR                                    Bx^_z  "          @��׿��˅@��Bs  C_�����:�H@S�
B2�CpG�                                    Bx^n   
Z          @�{��ff��Q�@���B�� CY���ff�#�
@^�RBEG�Cn�                                     Bx^|�  
Z          @�(����׿�@�  BsG�Con�����K�@EB)�Czc�                                    Bx^�l  
�          @�  �X�ÿ333@#33Bp�C?���X�ÿ��@	��A��CLp�                                    Bx^�  "          @���g
=�u@��B G�C5�g
=�W
=@ffA�ffCA�                                    Bx^��  
�          @�(���p���@p��B���CKaH��p����@VffBcCk�                                     Bx^�^  
�          @��>��
��@��HBpG�C��>��
�e@B�\B \)C��=                                    Bx^�  �          @�Q쾀  �{@���Bi{C�'���  �l��@;�B{C�R                                    Bx^Ԫ  T          @�(���
=���H@�G�B�#�CDJ=��
=��@hQ�B]�HCc�                                    Bx^�P  T          @�  �Ǯ@,��@c�
BC33B��ͿǮ?���@���B��C��                                    Bx^��  "          @���k��@xQ�Bn
=Cv33�k��R�\@;�B"��C~W
                                    Bx^ �  �          @�p��:�H��=q@��B�G�Cq(��:�H�.{@fffBP  C~�3                                    Bx^B  
�          @�=q����˅@�33B�W
C�k�����>�R@c33BG��C���                                    Bx^�  �          @�{�У׾��H@�{B���CD�)�У׿�(�@�Q�Be�Cfh�                                    Bx^,�  �          @�z���?
=@���Bt��C%u��녿c�
@�
=Bo�\CIh�                                    Bx^;4  
�          @���?s33@�=qBw��C� ���@�(�B~�HCC&f                                    Bx^I�  
�          @�{���?�{@��B�8RC�Ϳ�녿�@�Q�B�\)CL��                                    Bx^X�  "          @���\)?�@�  B�B��f��\)=L��@���B��{C18R                                    Bx^g&  
�          @��H��(�?�=q@�\)B{�
C녿�(�=u@���B��fC1��                                    Bx^u�  �          @��׿�?�\)@��B|=qC����\)@��RB��HC8�3                                    Bx^�r  �          @������?�  @���BiffC� ��þ.{@��RB{��C8�f                                    Bx^�  �          @����?�33@z=qB_\)C���>u@��B~��C-}q                                    Bx^��  "          @��׿��R?��H@�z�Br�
C�R���R��  @��B�(�C;&f                                    Bx^�d  �          @�=q��=q?��H@��
B��C�\��=q���
@���B���C?}q                                    Bx^�
  �          @�\)��G�?}p�@���B��=C���G����H@��
B��CC��                                    Bx^Ͱ  T          @��Ϳ@  ?�@~{B���B��@  <��
@��RB��
C2�R                                    Bx^�V  T          @�  �?\)>��R@.{B(�\C-�q�?\)�(�@*�HB$��C?z�                                    Bx^��  
�          @�p��   ?.{@b�\BWz�C$ٚ�   �
=q@dz�BY�\C@#�                                    Bx^��  "          @�  ��\?޸R@e�BT�RC����\>�ff@}p�BxffC'�                                    Bx^H  T          @�\)�&ff?�ff@UBC��Ch��&ff=�@e�BX{C1G�                                    Bx^�  �          @�=q�ٙ�?�(�@r�\BeCǮ�ٙ�>�p�@�z�B�W
C'��                                    Bx^%�  �          @���G�@�@Z=qB?��C	�)�G�?W
=@y��Bh��C�                                    Bx^4:  �          @��\�*�H?��@L��B;{Cn�*�H>��@^�RBQ��C.\)                                    Bx^B�  �          @����7�?L��@W�BBQ�C$s3�7���33@\��BH(�C;�                                    Bx^Q�  
�          @�p��9��>�(�@Q�B@C+���9���+�@O\)B=�\CA                                    Bx^`,  
�          @����9��>�@O\)B?G�C*��9����R@Mp�B=33C@{                                    Bx^n�  �          @�p��Z=q?5@'�B�C(G��Z=q�B�\@-p�B�RC7!H                                    Bx^}x  T          @��\�W�?��@+�BG�C���W�>�G�@?\)B%33C,��                                    Bx^�  "          @����E�?�@C33B+  C33�E�=�@P��B:p�C1�{                                    Bx^��  
Z          @�z��k�@Q�?��HA���C޸�k�?��H@�
A�z�CL�                                    Bx^�j  
�          @����_\)@�?�(�A�(�C)�_\)?�=q@�A�G�C\)                                    Bx^�  
Z          @�  �e�?�  @(�A�G�C8R�e�?8Q�@#�
B  C(�3                                    Bx^ƶ  
�          @����a�?�z�@"�\B	z�C!ٚ�a�>�\)@1�Bz�C/�                                    Bx^�\  T          @����Dz�@��@��B��C^��Dz�?�ff@AG�B(��C�                                    Bx^�  T          @���i���#�
@z�A�33C=�q�i�����\?�  A���CG:�                                    Bx^�  �          @��\�g
=�˅?�\A���CK� �g
=�z�?�A{
=CQ�)                                    Bx^N  �          @��
�u�8Q�?���A�ffC>���u���?�\)A��CFٚ                                    Bx^�  �          @���~{�B�\?��
A�p�C6�~{�@  ?У�A�C>��                                    Bx^�  T          @����{��#�
@{A�C4&f�{��8Q�@ffA��C>W
                                    Bx^-@  T          @�ff�g
=>���@<��B�RC/+��g
=�&ff@8��B33C>33                                    Bx^;�  �          @���xQ�#�
@{B C=O\�xQ쿱�@��A�  CG��                                    Bx^J�  
�          @�=q�^{?�G�@A���C)�^{?n{@1�B
=C%�                                    Bx^Y2  "          @����S33?�
=@,(�Bz�C���S33?   @@��B(�C+\)                                    Bx^g�  T          @��
�HQ�@%@
�HA�ffC\)�HQ�?��
@7�B\)Cff                                    Bx^v~  �          @�Q��;�@S33?�Q�A�p�C���;�@&ff@(��B�C
ff                                    Bx^�$  �          @�=q��Q�@��\?�A�
=B�녿�Q�@S33@A�B(�B�33                                    Bx^��  T          @�  ���R@p��?ǮA�G�B�3���R@E@*=qB�B���                                    Bx^�p  �          @����G�@�(�?���A�=qB��{��G�@o\)@)��B��B��=                                    Bx^�  "          @�(�>�@�=q�L�Ϳ
=B���>�@��H?��HA�  B��                                    Bx^��  �          @�
=�C�
@E?���A���C���C�
@ ��@G�A�
=C��                                    Bx^�b  "          @��R�A�@P��?�G�A��C�{�A�@(Q�@��A�G�C
�R                                    Bx^�  T          @��R���@K�@z�A�{B��)���@Q�@<��B$�C)                                    Bx^�  �          @�{�9��@Fff@   AɅC#��9��@�@7
=B{CE                                    Bx^�T  �          @�33�<(�@>{?�\)A��RC���<(�@  @,(�B33C��                                    Bx^�  
�          @����n�R?�{@"�\A��HCu��n�R?}p�@?\)B�C%#�                                    Bx^�  T          @�p��w�?ٙ�@ ��A��C8R�w�?Y��@:=qBQ�C'�)                                    Bx^&F  
�          @���e?�@2�\B
�
C���e?c�
@N{B$33C&\                                    Bx^4�  T          @�\)��
=?���@Q�Aʣ�C�\��
=?W
=@ ��A�{C(�                                    Bx^C�  �          @��R����?333?���A���C+B�����=�G�?���A�\)C2�)                                    Bx^R8  "          @�ff��=q?.{?�  A�  C+}q��=q=�?��A�z�C2�                                    Bx^`�  "          @�  ���R?�?�z�A��C%B����R?
=q@
�HAΏ\C-)                                    Bx^o�  
(          @�����{?���@33A�
=C%�R��{>�ff@33A�  C.(�                                    Bx^~*  �          @�������?\@��AȸRC z�����?O\)@   A�C)aH                                    Bx^��  �          @�  ����?��R@G�A�33C$E����?�@33Aۙ�C,��                                    Bx^�v  "          @�=q���
>�G�?�z�A��RC.�
���
�#�
?���A���C5�3                                    Bx^�  
�          @��H��=q?�G�@�\A��RC�
��=q?���@{A�ffC&#�                                    Bx^��  
�          @��\���?��\@�A�\)C#�����?\)@p�A�C,��                                    Bx^�h  �          @��H��
=?�
=@�Aď\C%@ ��
=>��H@�A�=qC-�q                                    Bx^�  �          @�\)���R?B�\@A�  C*J=���R=�G�@{AӅC2�f                                    Bx^�  �          @�\)��z�>�?�(�A�=qC.\��z�L��?��A��RC4�f                                    Bx^�Z  
�          @�  ���
>��
?�A��C0���
�u?���A���C6��                                    Bx^   T          @��R��{�c�
?�
=A��RC?\)��{��?�=qA��CEǮ                                    Bx^�  "          @����z�=#�
?�\A���C3����z���H?ٙ�A�Q�C9�R                                    Bx^L  T          @�����
=?&ff?˅A��
C,���
=>��?��HA���C2�                                    Bx^-�  
�          @�
=��
=>�Q�?��RA�{C/�f��
=����?��
A��C5+�                                    Bx^<�  �          @�
=����>�{?��@�
=C0�����>��?+�@��\C2B�                                    Bx^K>  �          @�{��G�=L��?�=qAJ�HC3\)��G����?�ffAD��C7(�                                    Bx^Y�  �          @�
=��{���?���A�=qC:����{�n{?�z�AY�C?0�                                    Bx^h�  
�          @�\)���H>��R?�=qAI�C0aH���H���
?�\)AQ�C4E                                    Bx^w0  �          @�
=��z�>���?Q�A�
C/Q���z�>\)?fffA&�HC2O\                                    Bx^��  �          @�{���?+�?O\)A
=C,����>���?uA4��C/5�                                    Bx^�|  �          @�  ���?0��?���AK�C+�����>�{?�  AhQ�C/�R                                    Bx^�"  "          @�Q����?���?n{A+�C!aH���?�p�?��Ay��C%(�                                    Bx^��  "          @�Q����
?�?��\A;�C-Ǯ���
>aG�?���APQ�C1xR                                    Bx^�n  T          @����ff?s33?�ffAD��C(�{��ff?(�?��\Ap��C,��                                    Bx^�  T          @�z���\)?fff?\(�A"ffC)B���\)?(�?�=qAL��C,��                                    Bx^ݺ  �          @�(����R?h��?W
=AffC)
=���R?!G�?��AIC,T{                                    Bx^�`  T          @�z���{?���?\(�A"ffC&����{?O\)?��AW�C*8R                                    Bx^�  �          @�����?��H?#�
@�\C%h����?z�H?uA6=qC()                                    Bx^	�  "          @�����?h��?�@�{C)+����?8Q�?B�\A�
C+Y�                                    Bx^R  
(          @��H���R?}p�>�G�@��RC((����R?Q�?333AG�C*
                                    Bx^&�  "          @������
?���>�33@�\)C&@ ���
?}p�?(��@��C'��                                    Bx^5�  �          @������?���>�\)@U�C&�����?z�H?z�@�C(�                                    Bx^DD  T          @�(���G�?aG������C)�=��G�?c�
=�?�Q�C)��                                    Bx^R�  T          @��\���\?�G�>��@EC!����\?���?+�A Q�C#33                                    Bx^a�  T          @�  ����?�p�?z�@߮C8R����?޸R?�{AW�C�                                    Bx^p6  �          @��R��p�@�
?�@љ�C����p�?�=q?��AV�RCT{                                    Bx^~�  �          @�  �\)@!녾�
=��33C���\)@#33>���@g
=CaH                                    Bx^��  "          @�Q��w
=@*=q�(���RCp��w
=@.�R>��?��
C�=                                    Bx^�(  
(          @�\)�h��@7��8Q����C�h��@=p�=���?�  C�                                    Bx^��  
�          @�\)����?��R>��@�Q�C�����?�ff?p��A8(�C:�                                    Bx^�t  �          @�ff���R?���?�A�(�C!�H���R?n{?޸RA�(�C'z�                                    Bx^�  T          @��
���
@�H?�G�A?
=C�{���
@33?У�A�ffC��                                    Bx^��  "          @�33����?���?
=q@�{C&f����?�(�?�ffAI�C�)                                    Bx^�f  	�          @��H���@  >#�
?�{C����@�?B�\AffC^�                                    Bx^�  
�          @�=q���
@   <��
>B�\CaH���
?�?\)@�C@                                     Bx^ �  �          @�=q��Q�@�=�\)?Tz�C�3��Q�@�?(��@�(�C�                                    Bx^ X  
�          @�G���(�?��\)��C@ ��(�?��#�
���CY�                                    Bx^ �  
�          @�\)���\?����R�p��CJ=���\?�=q>B�\@�C)                                    Bx^ .�  
�          @��\�s�
@<(����
�p��CT{�s�
@:�H>�@���C��                                    Bx^ =J  
(          @��H����@"�\�B�\�p�C������@)���L�Ϳ!G�C��                                    Bx^ K�  
�          @��\����@G��fff�,��CJ=����@�����HQ�C�H                                    Bx^ Z�  
�          @��\�s33@#�
�����]�C�s33@1G��������C�                                    Bx^ i<  "          @���{�@(���{����C�
�{�@'
=���H�f=qCp�                                    Bx^ w�  
Z          @��w
=?��R�	����Q�C��w
=@\)�����ffC(�                                    Bx^ ��  �          @��H�~{?��H�	���ӅC�q�~{@p���\)���C�)                                    Bx^ �.  T          @��
�p��@;��(���\C
�p��@?\)>8Q�@��C�=                                    Bx^ ��  
�          @���E�@dz�<�>�33C�E�@\(�?xQ�A=�C�
                                    Bx^ �z  �          @����~{?xQ��Q���C&L��~{?Ǯ��\�̏\C�
                                    Bx^ �   �          @������?s33��z���Q�C'u�����?��Ϳ�{���RC"p�                                    Bx^ ��  T          @��H�y��?�ff�z���  C33�y��@녿��
���
C�3                                    Bx^ �l  "          @����w�?�
=�'
=�{C�q�w�@�
�	���У�C                                    Bx^ �  T          @�ff�y��?�ff�7
=�(�C$��y��?�G��\)��CǮ                                    Bx^ ��  T          @�ff�p  ?��R�>{��C!��p  ?��H�#33���HCff                                    Bx^!
^  �          @�  �|(�?�z��+��\)C G��|(�@33�{�ԏ\Cz�                                    Bx^!  �          @����w�?!G��:=q�Q�C*� �w�?�{�)����C ��                                    Bx^!'�  
�          @�����ff?Y���%��\)C(�H��ff?�p���\�ٮC �
                                    Bx^!6P  �          @�33��  ?�{�ff����C&.��  ?�{��  ���\C ^�                                    Bx^!D�  "          @�=q��
=?����Q���G�C   ��
=?�p����R��C޸                                    Bx^!S�  �          @�Q����@{?
=q@�G�C\���@ ��?�=qAK�CJ=                                    Bx^!bB  
�          @�p��s33?�
=��=q���C&f�s33@����
���\C�\                                    Bx^!p�  
�          @�ff�e?��
�QG��"\)C h��e@�
�5�	�RC(�                                    Bx^!�  	`          @��R����?�
=��p���G�CL�����@���
=��  C�R                                    Bx^!�4            @�ff�{�?�{�Fff��
C$0��{�?����.{� Q�C�                                     Bx^!��  
�          @�z��
�H?�p�?   A,  C���
�H?��?Y��A��RC��                                    Bx^!��  
�          @��Ϳ�
=@:=q@|��BJz�B���
=?�=q@���B{��C 
=                                    Bx^!�&  T          @���!�@AG�@S�
B��C��!�@
=@|��BH�
C8R                                    Bx^!��  T          @��
�.{@A�@B�\B��C���.{@(�@l(�B:G�C5�                                    Bx^!�r  "          @�(��5�@O\)@/\)B�C��5�@p�@\��B*�C
�                                    Bx^!�  z          @�(��N�R@HQ�@��A��C���N�R@(�@FffBC��                                    Bx^!��  .          @����L(�@Q�@A�=qC:��L(�@'
=@Dz�B��C�                                    Bx^"d  
�          @����3�
@[�@$z�A���Cc��3�
@,(�@U�B"{C=q                                    Bx^"
  "          @���S33@N{@��AˮC���S33@%@:=qB�C�                                    Bx^" �  T          @���L��@[�@�A�
=C��L��@5�@6ffB�HC
�{                                    Bx^"/V  
�          @��
�Vff@#�
@4z�B��C�\�Vff?�@VffB%��C�f                                    Bx^"=�  
�          @����hQ�@!�@'
=A�ffC0��hQ�?���@I��B�CaH                                    Bx^"L�  T          @�{�a�@I��@ffA�G�C
J=�a�@#33@2�\B��C(�                                    Bx^"[H  �          @�p��n{@I��?�\)A�p�C���n{@*�H@�A��CT{                                    Bx^"i�  �          @��\�HQ�@xQ�?s33A,��C �f�HQ�@b�\?���A��C}q                                    Bx^"x�  T          @�33�7
=@xQ�?�(�A�B����7
=@Z�H@ffAޏ\C�                                    Bx^"�:  
�          @�=q�P��@dz�?��HA���Cc��P��@G�@G�Aՙ�C=q                                    Bx^"��  T          @���n�R@G
=?У�A�  C:��n�R@(Q�@�
A��HC�=                                    Bx^"��  �          @���Z=q@Y��?�A��\C\�Z=q@8Q�@#�
A�\C��                                    Bx^"�,  T          @�(��E�@r�\?���A��\C��E�@S�
@=qA���C��                                    Bx^"��  �          @��\�!G�@�=q?Tz�A�B�ff�!G�@�Q�?�A��
B�8R                                    Bx^"�x  T          @��\���\@{?��A�{C�{���\@ ��@
�HA�=qC�3                                    Bx^"�  T          @���tz�@.�R?��A�z�Cs3�tz�@\)@�A�G�C�)                                    Bx^"��  T          @�=q�AG�@e?�{A��\C��AG�@C�
@(��A�{C�)                                    Bx^"�j  �          @���;�@u�?�33A��RB��R�;�@Vff@\)A�C&f                                    Bx^#  �          @��
�L��@p��?��HA\��Cp��L��@XQ�@33A��
Cz�                                    Bx^#�  �          @��
�>�R@w�?�  A��HB�8R�>�R@Z�H@ffA�33C\                                    Bx^#(\  �          @��
�I��@x��?��\A9�C �q�I��@c33?��A�p�C��                                    Bx^#7  �          @�33�:=q@��?uA.{B�G��:=q@n�R?�{A��B��H                                    Bx^#E�  
�          @���33@�p�?�=qAC�
B����33@��@�
A�Q�B�\                                    Bx^#TN  �          @���>�R@j=q?�G�A��C#��>�R@J�H@"�\A�RC@                                     Bx^#b�  �          @���J=q@Vff?ٙ�A�CL��J=q@8Q�@=qA�33C	��                                    Bx^#q�  
Z          @�33�c�
@^�R�aG�� z�C���c�
@e��\)�Y��CǮ                                    Bx^#�@  �          @�G��qG�@J�H�p���-�C���qG�@S33�W
=��C
�
                                    Bx^#��  T          @�p��O\)@^{������RC��O\)@s33����?
=Cs3                                    Bx^#��  
�          @�33�S33@H����\���
C���S33@dz�����z�C�\                                    Bx^#�2  "          @����*�H@7
=>�?�33C�q�*�H@0��?B�\A3\)C�                                    Bx^#��  �          @�\)>#�
@C33@z=qBOB�(�>#�
@�@�\)B�\)B��{                                    Bx^#�~  T          @�{�k�@i��@S�
B(��B���k�@5�@�G�B[ffB�(�                                    Bx^#�$  
�          @�
=�333@xQ�@@��B33B�uÿ333@HQ�@q�BF��B�=q                                    Bx^#��  T          @�{���@s33@2�\B
�HB��f���@Fff@c�
B:  B��)                                    Bx^#�p  T          @�����@��@=qA��BԀ ����@_\)@O\)B%\)B��                                    Bx^$  "          @��׿�Q�@�  ?���A��B�Ǯ��Q�@^�R@1G�BQ�B�q                                    Bx^$�  
Z          @��\���@~{?�p�A�B����@\��@2�\Bz�B�L�                                    Bx^$!b  �          @�����@��?���A�{B����@n{@�A��B�L�                                    Bx^$0  
�          @�Q�޸R@\)?޸RA�  B���޸R@a�@#33B
=B�\                                    Bx^$>�  
�          @�
=��=q@~�R?޸RA��HB�LͿ�=q@`��@#33B
=B�ff                                    Bx^$MT  �          @���z�@s�
?��A��B�\�z�@W�@=qA�G�B��                                    Bx^$[�  �          @�(��H��@S�
�+��p�C�\�H��@W�=L��?(�C�q                                    Bx^$j�  �          @��R�]p�@H�þ�����z�C	�]p�@I��>�=q@VffC	��                                    Bx^$yF  
�          @��
�g
=@&ff��Q��pz�C=q�g
=@2�\��R��\)CE                                    Bx^$��  �          @�33��p�@33>�Q�@���C&f��p�@
�H?Tz�A ��C�                                     Bx^$��  !          @��
����@�ÿ�z��_�C=q����@%��#�
���HCG�                                    Bx^$�8  �          @�=q����@�L���Q�C�����@p���z��e�C�\                                    Bx^$��  
�          @��u�@{�����z�C��u�@'
=�У���33C��                                    Bx^$  -          @�\)�u@z�����ffC�{�u@-p��˅���
C�                                    Bx^$�*  "          @�G��~�R?�
=�1G���C#xR�~�R?�p��p����C��                                    Bx^$��  �          @�(��dz�?�\)�Y���&p�C�3�dz�@�\�B�\���C8R                                    Bx^$�v  
�          @�(��P��?�G��q��?��C"ٚ�P��?�G��_\)�-=qC�f                                    Bx^$�  
Z          @����5@J=q��ff�^�HC��5@S�
�Ǯ��=qC�H                                    Bx^%�  
�          @��
���@��?\A���B������@qG�@�A�p�B��                                    Bx^%h  
�          @����R@��R@33A�(�B�׿��R@j=q@G
=B��B�#�                                    Bx^%)  
Z          @�
=��G�@�?�A���B����G�@�{@4z�B�B�aH                                    Bx^%7�  �          @������@��H?�ffAk�
B�녿���@��@��A�p�B�8R                                    Bx^%FZ  "          @�33�Q�@��?G�A
=B�=q�Q�@��?��A�(�B�\                                    Bx^%U   "          @��Ϳ�ff@���?   @��
B�LͿ�ff@��\?���A��HB��                                    Bx^%c�  �          @�=q�3�
@�(�������B�z��3�
@�(�>�
=@�=qB��                                     Bx^%rL  �          @�����@�(�>�
=@�(�B�����@��R?��
An�HB��H                                    Bx^%��  �          @���(Q�@�(�?5Az�B�\�(Q�@y��?�G�A�z�B���                                    Bx^%��  �          @����P��@c33=�Q�?��C�
�P��@]p�?J=qAffCL�                                    Bx^%�>  T          @����E�@o\)>�ff@�=qC���E�@dz�?�A_\)C�                                     Bx^%��  
�          @��R�j=q@J�H?��\A=p�C��j=q@:=q?�{A�Q�C�                                    Bx^%��  �          @�{�l��@N�R��{�~�RC
޸�l��@O\)>�z�@Y��C
��                                    Bx^%�0  �          @��R��\)@!G�����33C0���\)@$z�<#�
>#�
C��                                    Bx^%��  �          @�
=���H@��=�Q�?��C)���H@�?��@˅C�q                                    Bx^%�|  �          @�{����?B�\?�p�A��C*�����>��?�A�z�C.�{                                    Bx^%�"  "          @�������>���?��
A��HC/�)�������
?�A��C48R                                    Bx^&�  �          @�p�����?0��@   A�C+#�����>�z�@ffA˅C033                                    Bx^&n  T          @�(���(�>��H?�\)A��C-����(�>�?�
=A�(�C2L�                                    Bx^&"  �          @��������\)?�{A�z�C;p������Y��?�(�A�Q�C?=q                                    Bx^&0�  �          @������?�G�?���AV=qC�=����?��
?�z�A�ffC T{                                    Bx^&?`  
�          @�\)�0  @z=q>�
=@�\)B�=q�0  @p��?��A\Q�B�Q�                                    Bx^&N  	�          @���#�
@�z�>u@1�B�B��#�
@�Q�?�33AX��B�G�                                    Bx^&\�  
�          @���^�R@��>��
@hQ�B�Ǯ�^�R@�33?��RAb�HB�\)                                    Bx^&kR  �          @��
�G�@���?(�@ڏ\BŔ{�G�@�=q?��
A�
=B�L�                                    Bx^&y�  T          @�  �}p�@�=q?B�\A�RB�
=�}p�@��H?�Q�A���B�\                                    Bx^&��  T          @�G��\)@��>8Q�?�(�B�LͿ\)@��
?��AH  B���                                    Bx^&�D  �          @�녿�@��ýu�.{B�(���@�ff?fffA��B�Q�                                    Bx^&��  "          @�=q�   @��׽��
�O\)B��f�   @�{?c�
A=qB�\                                    Bx^&��  �          @��\���@���>�z�@FffB�8R���@��?��RAX��B�Q�                                    Bx^&�6  T          @�ff��Q�@�ff?k�A1��B��f��Q�@��R?��HA��B���                                    Bx^&��  
�          @�{��Q�@�G�?�
=A^�HB����Q�@�  ?�p�A�Q�B�Q�                                    Bx^&��  �          @�=q��H@��?uA-B�
=��H@��
?�(�A���B��)                                    Bx^&�(  T          @����:=q@}p�?fffA&�RB��q�:=q@n�R?���A�
=B���                                    Bx^&��  
�          @�z��!�@o\)?�  A�B���!�@W�@��A�\B�Ǯ                                    Bx^'t  
�          @�33�8��@^{?�(�A�33C�q�8��@G
=@�
A�\Cٚ                                    Bx^'  
�          @��R�A�@\(�?��A�\)Cff�A�@C33@{A���C�\                                    Bx^')�  T          @����W�@E�?�(�A��C	�
�W�@.{@\)Aڏ\C                                    Bx^'8f  
�          @�(��u@��?��A�\)C}q�u?�33@  A�C��                                    Bx^'G  �          @�=q�\��@,��?��
A��RC��\��@ff@{A�(�C��                                    Bx^'U�  
�          @�
=�]p�@�?�ffA���C�f�]p�@�@��A�z�C��                                    Bx^'dX  �          @�  �7�@n{?0��A��B�G��7�@b�\?���A�
=C ��                                    Bx^'r�  "          @��H�E�@e�?s33A5�C���E�@W
=?�ffA��
Cz�                                    Bx^'��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^'�J              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^'��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^'��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^'�<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^'��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^'و              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^'�.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^'��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^("�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(1l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(N�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(]^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(�P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(�B              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(Ҏ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(�4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^(��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)*r              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)9              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)G�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)Vd              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)e
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)s�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)�V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)�H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)˔              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^)��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*#x              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*Oj              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*l�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*{\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*Ě              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^*�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^++$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+9�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+Hp              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+W              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+e�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+tb              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^+�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,$*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,2�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,Av              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,mh              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,�               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-:|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-I"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-W�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-fn              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-u              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-�`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-�R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-۞              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-�D              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^-��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.6              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.$�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.3�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.B(              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^._t              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.|�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.�f              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^.��  �          @����
=@7��k��%�C�\��
=@8Q�=L��?�C��                                    Bx^.Ԥ  �          @�{�`  @q녾�Q�����C��`  @s33    <��
C��                                    Bx^.�J  T          @�����
@C33�   ����C�����
@E��L���  C:�                                    Bx^.��  T          @�ff�l��@e��333��  C���l��@hQ쾸Q��|��C�\                                    Bx^/ �  J          @�p��~�R@I���fff� ��C�f�~�R@N{�����p�C�                                    Bx^/<  �          @��
���@ �׿c�
��
C����@%��&ff��Q�C\)                                    Bx^/�  
�          @�����\@$z�J=q��\CG����\@(�ÿ����z�C��                                    Bx^/,�  �          @��
��p�@�H�=p��=qCY���p�@{��\����C�                                    Bx^/;.  T          @��H��z�?��H�.{��
=C&f��z�@ �׿   ���
C�{                                    Bx^/I�  �          @��
��  ?��
�+���G�Ch���  ?��   ����C�
                                    Bx^/Xz  �          @��
��
=@   ��{�Lz�Cc���
=@%�aG�� ��Cz�                                    Bx^/g   T          @�{��=q@���33�~{C@ ��=q@\)��
=�T��C�                                    Bx^/u�  "          @����
=?�ff���\�6=qC#���
=?�׿Y����
C=q                                    Bx^/�l  
�          @�{��{@ �׿Y���
=C��{@��(����33C�                                    Bx^/�  "          @����\@
=q�xQ��.{C� ���\@\)�E��
{C�3                                    Bx^/��  T          @�{��\)?�  ����C
=C�R��\)?��n{�%��C�                                     Bx^/�^  �          @�{��ff@#�
�+���\)C
��ff@'
=��G���z�C��                                   Bx^/�  T          @�p�����@�\�@  ��CL�����@ff�
=q���C�R                                    Bx^/ͪ  
�          @��
����@�Ϳ�=q�r{CB�����@#33��{�H��C+�                                    Bx^/�P  
�          @�����H@�����\�fffC
=���H@   ��ff�>=qC                                      Bx^/��  �          @����@{��=q�A�CǮ���@#33�\(����C�                                    Bx^/��  l          @����33@���(����HCٚ��33@�Ϳ�G��b�HC�H                                   Bx^0B  
�          @���w
=@AG��˅��p�C���w
=@I������l��C�q                                   Bx^0�  
�          @���vff@Dzῢ�\�h(�Cz��vff@J�H��  �5C�
                                    Bx^0%�  
�          @��
��=q@ff�����v{CxR��=q@�Ϳ���O\)C^�                                    Bx^044  
�          @�p���
=@#�
����{\)C�3��
=@*�H��z��Qp�C��                                    Bx^0B�  
�          @�Q��N{@G��'���ffC�3�N{@U����ffC�                                    Bx^0Q�  
Z          @��H��\@�=q�����W�B�(���\@���L����B�aH                                    Bx^0`&  
(          @��\�1G�@_\)��  ���
C p��1G�@hQ쿹�����HB��q                                    Bx^0n�  �          @�����@�
=�!G��ӅB�L���@�  �W
=�
�HB���                                    Bx^0}r  "          @�{�U�@��p���\)C �{�U�@���z���z�C &f                                    Bx^0�  �          @����@��@��H��Q��tQ�B���@��@�ff�����4Q�B�8R                                    Bx^0��  �          @��.�R@�{��=q��{B�ff�.�R@�����
=�A�B�.                                    Bx^0�d  
�          @�{�C33@�z������{B��R�C33@��\��\���HB�W
                                    Bx^0�
  "          @�ff�E�@k��5���ffC�f�E�@z=q�!G��֣�C @                                     Bx^0ư  T          @����Dz�@*�H�z=q�/ffC  �Dz�@>�R�j�H�"z�C�\                                    Bx^0�V  T          @�(��#33@`���a����B��)�#33@r�\�N�R�
=B��R                                    Bx^0��  
�          @��H�   @z�H�B�\��B�{�   @���,����33B�\                                    Bx^0�  �          @���@  @S�
�S33��RC0��@  @dz��AG��z�C�                                    Bx^1H  �          @�
=�J=q@@  �N�R�G�C}q�J=q@P���>�R�{C(�                                    Bx^1�  �          @�Q��J�H@N�R�HQ��
ffCs3�J�H@^�R�6ff����C^�                                    Bx^1�  T          @����W
=@Fff�C33���C	J=�W
=@U�2�\���
C.                                    Bx^1-:  T          @�G��Dz�@?\)�_\)���C���Dz�@P���O\)��C:�                                    Bx^1;�  
�          @�Q��9��@J�H�Z=q�Ch��9��@\(��H�����C#�                                    Bx^1J�  
�          @�  �P  @G��g
=�)33C��P  @#�
�Z�H�33C��                                    Bx^1Y,  �          @�ff�XQ�����ff�K\)C=G��XQ�W
=��\)�Mp�C7�)                                    Bx^1g�  
(          @�(��^�R?���\���(C��^�R?��S�
� ��C��                                    Bx^1vx  "          @����G�?�  ��
=aHC}q��G�?�����(��|��C��                                    Bx^1�  �          @��e�aG������@C7u��e=��������A{C2^�                                    Bx^1��  T          @���a녿aG��}p��=��CA��a녿\)��Q��A=qC=                                    Bx^1�j  T          @�(��=q��ff����e�CT��=q������33�n��CNY�                                    Bx^1�  
�          @����C�
�������J{CK���C�
���\�����P�\CFaH                                    Bx^1��  
�          @��R���>�z��A����C0  ���?��@  �33C,Ǯ                                    Bx^1�\  �          @�(���33?�������  C#p���33?��H�	����33C!aH                                    Bx^1�  "          @�����33?�=q��\C�q��33?޸R���H��C.                                    Bx^1�  "          @�p����?�\)���¸RC�{���?��
���R���C�                                    Bx^1�N  �          @�ff���
?�(���R����C�����
?�����p�C�3                                    Bx^2�  	V          @�  ���?�G��Q���C!�����?�� ����(�C޸                                   Bx^2�  	8          @���mp�@  �9�����C���mp�@{�.{��
=CW
                                   Bx^2&@  
�          @�p��|(�@{�
�H��ffC��|(�@(Q��p���\)CL�                                    Bx^24�  J          @�����  @2�\�   ��{C
=��  @<(����
���C�                                    Bx^2C�  �          @�
=�x��@C�
����G�CǮ�x��@K���
=�
=C��                                    Bx^2R2  
�          @�{��G�@+���G��5��C�3��G�@0  �O\)�z�CJ=                                    Bx^2`�  �          @����  ?�33�޸R��G�C���  @녿�����\)C�R                                    Bx^2o~  K          @�����33@
=��Q��S33CaH��33@(����\�6{C��                                    Bx^2~$  
�          @������R@녿�  �/�
C�����R@�W
=�Q�C�                                    Bx^2��  T          @�=q����?������p�C'Ǯ����?�Q��Q����C&�                                    Bx^2�p  �          @�33��Q�?�������\C'p���Q�?�(�����\C%�H                                    Bx^2�  T          @�����p�?�33�=p���HC����p�?ٙ���R��p�C                                      Bx^2��  }          @����|��@^�R?   @���C
���|��@\(�?@  A33C                                    Bx^2�b  +          @�Q��n{@c33?�G�A2{C\)�n{@]p�?��\A`z�C	�                                    Bx^2�  T          @�=q�mp�@j=q?uA&=qC\)�mp�@e�?�(�AUp�C�q                                    Bx^2�  �          @�{�|(�@Fff?��RA^{C�=�|(�@@  ?��HA��C�3                                    Bx^2�T  �          @��p  @X��?���AH��C	��p  @S33?�\)AuC
��                                    Bx^3�  T          @�ff�g�@`  ?��Af{C���g�@Y��?��A�=qC�
                                    Bx^3�  
�          @�p��q�@9��?�A��C� �q�@0��@33A�33C�H                                    Bx^3F  "          @��
�j=q@-p�?�Q�A��\Cz��j=q@#�
@Q�A�{C�                                    Bx^3-�  T          @�(��n{@G
=?�z�A�33C\�n{@>�R?��A�Q�C@                                     Bx^3<�  
Z          @����c�
@E�?�p�A�p�C��c�
@<��?���A��HCO\                                    Bx^3K8  
�          @��R�x��@�@ ��A���C{�x��@�@
�HAυC�                                     Bx^3Y�  �          @����z�H@�?�p�A�G�C�f�z�H@{@	��A��C�                                    Bx^3h�  "          @���xQ�@'
=?�  A�C��xQ�@�R?�
=A��
Cz�                                    Bx^3w*  
�          @���l(�@QG�?��RA�{C
n�l(�@J=q?�(�A��Cp�                                    Bx^3��  �          @���N{@�  ?z�HA-p�C Ǯ�N{@z�H?��\A`z�CY�                                    Bx^3�v  �          @�{�A�@�33?Y��A  B�  �A�@���?��ALQ�B���                                    Bx^3�  
�          @�z��B�\@}p�?�{AH  B���B�\@xQ�?��A{�C �                                    Bx^3��  �          @��Z�H@n�R?���AD(�C���Z�H@h��?�{As�
C:�                                    Bx^3�h  �          @�p��^{@j�H?���A@Q�CaH�^{@e�?�=qAo\)C�                                    Bx^3�  �          @����^�R@c�
?��Ahz�CL��^�R@]p�?��A�G�C�                                    Bx^3ݴ  �          @�{�j�H@P��?���A��
C
^��j�H@I��?�ffA��Ck�                                    Bx^3�Z  T          @���u@)��@�AȸRC^��u@\)@
=AڸRC                                    Bx^3�   �          @�Q��e�@'�?�Q�A�33C���e�@{@�A��
CW
                                    Bx^4	�  �          @���c33@'�?���A�(�C�
�c33@�R@�A��HC�                                    Bx^4L  "          @�{�9��?k�@c�
BE��C"Q��9��?+�@g�BJ33C&�                                    Bx^4&�  T          @�33�]p�����@vffB8��CH�)�]p���=q@p  B2z�CL�)                                    Bx^45�  �          @����N�R�z�H@���BHQ�CDٚ�N�R��G�@|��BC{CIB�                                    Bx^4D>  "          @����I����ff@y��B?�
CN#��I����@q�B8Q�CQ�                                    Bx^4R�  "          @��R�1녿�G�@�Q�BO33CP� �1녿��
@y��BG{CT�3                                    Bx^4a�  
�          @�33�,�;��@�G�Bf  C<���,�Ϳ5@�  Bb��CB��                                    Bx^4p0  
�          @�33�   ?c�
@~�RBaQ�C ff�   ?(�@���Be�HC&:�                                    Bx^4~�  }          @�z��{��@p  Bbp�C6�{�{�\@n�RB`�
C<�                                    Bx^4�|  
�          @���:�H�h��@n{BJz�CEaH�:�H��@i��BE(�CI��                                    Bx^4�"  T          @��U����@z�HBF
=C9��U��@x��BD  C=��                                    Bx^4��  �          @��H�O\)�:�H@�z�BL�HC@�R�O\)��G�@��\BHCEaH                                    Bx^4�n  �          @�  �mp��J=q@a�B+�RC@��mp����
@^{B'��CC��                                    Bx^4�  �          @�\)�U�����@vffB>�CFE�U���{@p��B9=qCJ:�                                    Bx^4ֺ  �          @��R�z���@�33B{p�C=��z�8Q�@��BwCE=q                                    Bx^4�`  
�          @�ff�0  ��@��Be�C=��0  �G�@�=qBb  CC��                                    Bx^4�  "          @��7���=q@���BV  CH�=�7���{@�=qBO�HCMG�                                    Bx^5�  �          @�ff�"�\��
=@���BcG�CLٚ�"�\��(�@�{B\
=CQ�q                                    Bx^5R  �          @�{�6ff��\@vffBCz�CSٚ�6ff��@n{B:�
CWp�                                    Bx^5�  �          @��
��R��33@�  BP33CY}q��R�
�H@w
=BFG�C]0�                                    Bx^5.�  �          @����=q���@~�RBL  C]�=�=q���@u�BAQ�C`��                                    Bx^5=D  �          @������
=@���B���C�������   @�p�B�B�C�K�                                    Bx^5K�  T          @����33���H@�{B~�Cf�\��33� ��@��Bq�Ck5�                                    Bx^5Z�  �          @��H�E�O\)@z�HBKz�CB���E����@w
=BF��CG&f                                    Bx^5i6  "          @���<(���  @uBA(�CR�\�<(�� ��@n{B8��CVY�                                    Bx^5w�  �          @�\)�Dz��z�@p��B8z�CS���Dz��
=q@hQ�B/�CW&f                                    Bx^5��  �          @���8Q�У�@|(�BG�CQ���8Q���@tz�B?�
CUT{                                    Bx^5�(  "          @��B�\���@|��BG(�CL���B�\��33@vffB@33CP��                                    Bx^5��  "          @��
�N{��\)@hQ�B4��CN���N{��{@`��B-p�CR                                      Bx^5�t  "          @�z��L(��@Z�HB'��CU0��L(���
@Q�B�
CW�3                                    Bx^5�  T          @���QG���
=@Z�HB+�RCO0��QG���z�@S33B$G�CR=q                                    Bx^5��  
�          @����z=q�u@;�B�CA�=�z=q��z�@6ffB�HCDs3                                    Bx^5�f  T          @���{��+�@C33B�RC=�3�{��^�R@@  B�RC@�\                                    Bx^5�  �          @������H��@7�B�C:�����H�+�@5�B	\)C=L�                                    Bx^5��  �          @�
=���H��(�@�\A�C9�����H�z�@��A�(�C;��                                    Bx^6
X  K          @�  ��z�?=p�?��HAf�RC*����z�?&ff?�G�Ao�
C+��                                    Bx^6�  +          @��\��{?�33?:�HA��C ����{?���?W
=A�C!+�                                    Bx^6'�  �          @������R@�þW
=�Q�Ck����R@	���u�5CT{                                    Bx^66J  
�          @�z����@5��O\)�{C�����@8Q��R��ffCG�                                    Bx^6D�  �          @������@;������3\)CG����@?\)�^�R���C�3                                    Bx^6S�  �          @����@3�
����0(�C�q��@8Q�^�R�(�Ch�                                    Bx^6b<  �          @��H��=q@AG�����3�
C���=q@E�aG����C�                                    Bx^6p�  �          @�������@C�
����[�C0�����@H�ÿ����8  C}q                                    Bx^6�  "          @�Q����\@4z῕�D��CaH���\@8�ÿz�H�$  C��                                    Bx^6�.  �          @�����
@(Q쿓33�B�\CJ=���
@-p��xQ��#�C�H                                    Bx^6��  "          @��
�w�@!��,����\C� �w�@-p��!���33C��                                    Bx^6�z  
�          @��q�@\)�L(��
=Ch��q�@���B�\��C)                                    Bx^6�   �          @��Ϳ�
=?��xQ���ffC33��
=?�p��W
=��
=CQ�                                    Bx^6��  �          @�{?�G�@>{@�G�BP��B|
=?�G�@+�@�\)B^(�Br
=                                    Bx^6�l  T          @�z�?�p�@@  @�Q�BR��B�Q�?�p�@-p�@�ffB`��B�                                    Bx^6�  T          @�p�?�33@=p�@��
BW��B�� ?�33@*=q@��Bf{B�33                                    Bx^6��  T          @�{?�\@@�Bn�RBG
=?�\?�\@��Bz�B4{                                    Bx^7^  
�          @�
=@ff@�R@�  BL��B:p�@ff@(�@��BW�B,�                                    Bx^7  
�          @�=q@!�@2�\@n�RB3  B?
=@!�@!�@z=qB>(�B3��                                    Bx^7 �  �          @���?�@-p�@�{BSQ�Bi��?�@�H@��
B`G�B]��                                    Bx^7/P  
�          @�?���@:=q@w
=BE��Bv��?���@(��@�G�BS{Bm\)                                    Bx^7=�  �          @�(���{���
�.{��C?��{�z�H�@  �33C?8R                                    Bx^7L�  "          @�33��(�����\)�nffCF�
��(����ÿ�p�����CE��                                    Bx^7[B  
�          @�
=�����\��\)�P��CIu������Q쿞�R�g
=CH�{                                    Bx^7i�  T          @������\�У׿�z��V�HCG�
���\��ff���\�k33CF��                                    Bx^7x�  
�          @���녿�G��ٙ���p�CFY���녿�녿�����HCE                                      Bx^7�4  T          @����p���H����>{CP�q��p����(��[�CO��                                    Bx^7��  �          @�G����H�{��\)�R�\CS#����H��;�G�����CR��                                    Bx^7��  �          @���n�R�ff�,(�� �RCQp��n�R���5���CO5�                                    Bx^7�&  �          @��n{�G��/\)� �RCSW
�n{�z��8����\CQ#�                                    Bx^7��  �          @�(���G���
=�   ��ffCL33��G�����Q�����CJ��                                    Bx^7�r  �          @�ff��녿���� ����=qCE�f��녿���ff��G�CD
=                                    Bx^7�  �          @�����
��p����R��=qCEǮ���
�������p�CD5�                                    Bx^7��  "          @��R���H�\��
=��  CFaH���H��������CD�
                                    Bx^7�d  	�          @�p����������33��Q�CF� ������z��   ��  CE:�                                    Bx^8
  �          @�����\)�\��p���Q�CF�3��\)��33��=q��{CEL�                                    Bx^8�  
�          @�z����R��Q��\����CE����R���ÿ�\)���HCDz�                                    Bx^8(V  �          @�=q����ff��(����CN�q�����(���{��CM^�                                    Bx^86�  �          @��R��Q�>��ÿ
=q�ϮC/����Q�>�p�����ffC/�=                                    Bx^8E�  �          @�\)��G���Q�(�����C8E��G����R�0���=qC7�R                                    Bx^8TH  �          @�{�����!G��E��  C;u������녿O\)��
C:��                                    Bx^8b�  �          @�{���H?@      =#�
C+&f���H?@  =u?&ffC+.                                    Bx^8q�  "          @������?k�=�G�?��C)T{���?h��>8Q�@G�C)p�                                    Bx^8�:  �          @����ff�B�\�\��z�C6+���ff�#�
�Ǯ����C5�)                                    Bx^8��  T          @�
=��{�aG�������p�C6����{�B�\��
=���HC6=q                                    Bx^8��  
�          @����{��G��!G���
=C5T{��{��\)�!G���G�C4��                                    Bx^8�,  �          @��
���\�W
=�����\)C6ff���\�8Q����(�C6�                                    Bx^8��  T          @��
���\���þ��
�e�C7�����\���R��{�u�C7s3                                    Bx^8�x  �          @����ff�8Q�����ffC6���ff������H���\C5��                                    Bx^8�  �          @�  ��ff�5<#�
>\)C;�=��ff�5�#�
���C;Ǯ                                    Bx^8��  "          @������R�Tz�>aG�@Q�C=\���R�W
=>#�
?޸RC=33                                    Bx^8�j  �          @��\��
=�0��?B�\A\)C;}q��
=�=p�?5@�p�C<
=                                    Bx^9  T          @�����(����?�AK�C:����(��.{?�\)AC�C;�                                    Bx^9�  
�          @�����
=���H?�@���C9Y���
=��>�@�Q�C9�q                                    Bx^9!\  �          @�����G��k���\)�L��C6xR��G��aG���Q�xQ�C6h�                                    Bx^90  �          @�Q���ff�(�>�@�Q�C:����ff�#�
>�
=@���C;                                      Bx^9>�  �          @��R��(���R?�@��C:����(��(��?�@���C;W
                                    Bx^9MN  �          @��R��(����?.{@���C90���(���\?&ff@�z�C9�R                                    Bx^9[�  T          @�ff��녿\(�?(�@ٙ�C=���녿fff?��@�33C>!H                                    Bx^9j�  
�          @�{��{��z�?#�
@�(�CC����{����?
=q@��CDh�                                    Bx^9y@  *          @����
=��z�?=p�A�HCC޸��
=���H?#�
@�G�CDff                                    Bx^9��  T          @��R��zΐ33?�\)As�CA8R��z῞�R?��
Ad  CBE                                    Bx^9��  T          @����(��\?���AB{CEB���(��˅?}p�A.{CF\                                    Bx^9�2  �          @�
=���R���?�ffA��
CC����R��(�?ٙ�A�ffCEW
                                    Bx^9��  "          @��R��p���G�?��AA��CBY���p���=q?}p�A0��CC+�                                    Bx^9�~  �          @����녿�=q?.{@�ffC@\��녿���?��@��C@�\                                    Bx^9�$  �          @�Q������Q�?@  A�HCD�������R?#�
@�Q�CD�f                                    Bx^9��  �          @������ÿQ녿Q����C=B����ÿB�\�^�R�  C<�{                                    Bx^9�p  T          @�z����ÿTz�޸R����C<����ÿ333��ff��p�C;�\                                    Bx^9�  T          @�{���E����
�y��C<����(�ÿ�=q���C:��                                    Bx^:�  �          @����  �@  ���H�k33C;����  �#�
��G��s�C:��                                    Bx^:b  �          @�\)��p��L�Ϳ�����C<k���p��.{��(���(�C;!H                                    Bx^:)  �          @����Q�L�Ϳ�z����C<����Q�(�ÿ��H���\C;!H                                    Bx^:7�  "          @�33����!G���\)��{C:ٚ������H��z����C9T{                                    Bx^:FT  
�          @�=q�������
=����C:������ff��(���{C9�                                    Bx^:T�  T          @�G����;���(����C9{���;��R�   ���C7p�                                    Bx^:c�  �          @�  ��33�#�
��
=��
=C;���33���H��p�����C9�                                     Bx^:rF  �          @�{��(����H����=qCC����(����ÿ�����CAǮ                                    Bx^:��  �          @�=q���
���ÿ�ff��{C@�q���
�n{��\)��G�C?Y�                                    Bx^:��  
�          @�=q��ff�O\)�޸R���RC=����ff�+���ff��{C<&f                                    Bx^:�8  �          @�=q���������t��C:=q�����(���\)�{�C9
=                                    Bx^:��  �          @�����\�
=�\���\C:�3���\��׿Ǯ��ffC9�{                                    Bx^:��  �          @�������z�H���
���HC?������\(��������C>z�                                    Bx^:�*  
�          @�����녿�ff��=q��p�CC�H��녿������\CBk�                                    Bx^:��  T          @�
=��\)�+���{�}�C<���\)�녿�z����C:�
                                    Bx^:�v  
�          @�Q�����>�������  C2������>�\)��Q����\C0��                                    Bx^:�  T          @�����=�������  C2�����>����ff����C1                                      Bx^;�  
�          @������\>\)��G���{C2T{���\>�\)�޸R���\C0��                                    Bx^;h  
�          @������?�\�G���=qC-�����?(�ÿ�p����C+��                                    Bx^;"  
�          @����G�?^�R�33����C))��G�?��
��(�����C':�                                    Bx^;0�  T          @��\���ͽ������ffC5s3����<�������RC3��                                    Bx^;?Z  �          @�\)��(��k����
��{C6�=��(���Q�����G�C5�                                    Bx^;N   �          @�����H>\��33����C/=q���H?���{��p�C-k�                                    Bx^;\�  �          @�Q�����>�33��Q����RC/������>���z���p�C-�R                                    Bx^;kL  �          @����  �z�H=#�
>ǮC?)��  �z�H�#�
��C?�                                    Bx^;y�  �          @�������
=�8Q��p�C:�{����녾aG��\)C:ff                                    Bx^;��  
�          @����G���=q�u�)��C@!H��G�������
�eC?��                                    Bx^;�>  �          @�=q��(���
=?
=@��CA�H��(���p�>��H@��CB)                                    Bx^;��  
�          @�p���z`\)?\(�A ��CDu���zῸQ�?=p�A\)CE.                                    Bx^;��  �          @�����  ��
=?0��@�{CD����  ��p�?z�@�33CE@                                     Bx^;�0  �          @��R��G����
�L�Ϳ�\CBE��G����\������CB.                                    Bx^;��  
�          @�\)��(��xQ�>�=q@HQ�C?:���(��}p�>B�\@{C?p�                                    Bx^;�|  "          @�{���\�\(�>��@��C=�����\�c�
>�{@qG�C=��                                    Bx^;�"  �          @�����p����ü����
C?����p���������\C?�H                                    Bx^;��  T          @�G���p���33�u�&ffC@� ��p���녾����z�C@h�                                    Bx^<n  �          @�=q��p��޸R�(���
=CG����p���Q�B�\�Q�CF�                                    Bx^<  �          @�{�w
=�`�׿��R�~�HC^J=�w
=�XQ��\����C].                                    Bx^<)�  T          @�=q���R�>{��z��K
=CW33���R�7
=��33�v=qCV8R                                    Bx^<8`  �          @�=q��z�k��k���HC>���z�W
=�}p��+�C=B�                                    Bx^<G  �          @�{��
=��Q�aG��G�CAz���
=��{�z�H�.�\C@�H                                    Bx^<U�  �          @�{��  �0�׿����IC;�\��  �
=��
=�S�C:�R                                    Bx^<dR  T          @�ff���׾�{��=q�mC7�f���׾k���{�r{C6�)                                    Bx^<r�  �          @�Q���  ��(������ffC8�f��  �����������C7h�                                    Bx^<��  T          @�\)����.{�����u�C;�R����\)��
=�~�\C:^�                                    Bx^<�D  
�          @�����R���Ϳ����r�HC8�����R��\)�����xQ�C7B�                                    Bx^<��  "          @�  ���\��z῞�R�[�C7O\���\�B�\��G��_33C6)                                    Bx^<��  �          @������\�
=��=q�iC:�
���\��׿����q�C9L�                                    Bx^<�6  
�          @��������   ��(���C9�f������p���G����C85�                                    Bx^<��  �          @�Q���  �����˅���RC7n��  �#�
��{���\C5�)                                    Bx^<ق  �          @����33�녿�
=��Q�C:����33��
=��p���Q�C8��                                    Bx^<�(  T          @�����������z�C9�=�������������C7�H                                    Bx^<��  �          @�=q��=q�O\)����m�C=�\��=q�333��{�y��C<8R                                    Bx^=t  T          @�=q��ff��  ��{��(�C?����ff�Y����Q����C>J=                                    Bx^=  �          @�z����׿�{�޸R���CG�����׿��������ffCE�\                                    Bx^="�  �          @�=q��
=��녿�Q���z�CJ���
=��  ������CIp�                                    Bx^=1f  L          @�33���
�Ǯ���H���CF�����
��
=�˅��{CE(�                                    Bx^=@  �          @��H��33��=q��(����HCF����33��Q��{���CE^�                                    Bx^=N�  
�          @�=q��=q�Ǯ�����33CF�=��=q��z�����
CE+�                                    Bx^=]X  �          @��������
�
=���
CH�������\)��
=CEn                                    Bx^=k�  �          @�G����\�У�����{CI�q���\��33�$z���ffCF��                                    Bx^=z�  T          @�=q��
=�����ff����CG���
=��\)��
=��CE�                                    Bx^=�J  �          @�p���녿��(���(�CK
��녿У���׮CH�q                                    Bx^=��  �          @��
��녿����
����CJ����녿˅�{���
CHB�                                    Bx^=��  T          @���\)�1녿�  �g33CV�H�\)�*=q��  ��33CU��                                    Bx^=�<  �          @���vff�Fff�^�R��CZ�
�vff�@�׿�33�T  CZ                                      Bx^=��  �          @��H�����,�Ϳ\(�� z�CU�f�����'
=��{�N�RCT�                                     Bx^=҈  �          @��R��33�u��p���G�C?�q��33�k�����33C?G�                                    Bx^=�.  T          @����=q�  ����l��CO����=q�����R���CN0�                                    Bx^=��  "          @�(��|(��@�׿333� z�CYaH�|(��;��z�H�4  CX��                                    Bx^=�z  �          @����\)�!�>��@�Q�CTn�\)�#�
>.{@
=CT��                                    Bx^>   �          @�ff�����{�O\)��CR�����Q쿅��D��CQ�)                                    Bx^>�  �          @���������33��G�CQ�{�������  ���COT{                                    Bx^>*l  �          @�����{�   ��{���CMxR��{���33���CK^�                                    Bx^>9  
�          @������\����(���z�CQ����\�	����
=���CO��                                    Bx^>G�  �          @�=q�|���B�\�:�H���CY�{�|���=p���G��9CX�\                                    Bx^>V^  T          @�Q��~{�B�\>W
=@
=CYu��~{�B�\��Q쿀  CY�                                    Bx^>e  �          @�����'��u�/�CT{��� �׿����]�CS
=                                    Bx^>s�  "          @��\���ÿ�G��5� Q�CE�\���ÿ�Q�Y����\CD��                                    Bx^>�P  �          @�����H��������p�CJ�����H��\)� �����
CH�                                     Bx^>��  T          @�G���z��Q쿓33�S�CM���z�� �׿����y�CL��                                    Bx^>��  T          @�����z��33���H�`��CM���z����z����HCK��                                    Bx^>�B  �          @������H��׿s33�/�CO����H�
=q���Xz�CNn                                    Bx^>��  
�          @�����=q��R�=p��33CQ�
��=q��ÿz�H�3�
CP��                                    Bx^>ˎ  "          @������!�=��
?fffCQ�3���!G��.{��CQ�f                                    Bx^>�4  T          @��\�����\)��33���
CD������������
��{CB�{                                    Bx^>��  "          @��\��z��
=�Ǯ��CK����z��G��޸R���CI�)                                    Bx^>��            @�������33��p���\)CP.������ÿ��H���\CN}q                                    Bx^?&  \          @��H������
������CM�=���׿�\)��\����CK�)                                    Bx^?�  �          @�33������{����CN@ ��������
��\)CL�                                    Bx^?#r  �          @��������  �����Q�CF�������=q���
��ffCD}q                                    Bx^?2  �          @�����33���Ϳ�����33CHE��33��z��(�����CE��                                    Bx^?@�  �          @�(������%�>��@I��CT�������%    ��CT�                                    Bx^?Od  �          @����s�
�C33?�=qAH  CZ��s�
�H��?B�\AQ�C[�=                                    Bx^?^
  �          @�G���  �=p�?+�@��RCX����  �@��>�Q�@���CY�                                    Bx^?l�  �          @�����\�7
=?E�Az�CW����\�:�H>��@��\CW��                                    Bx^?{V  �          @�p����H�.�R?�p�A�=qCU�����H�7�?�Q�AVffCW                                    Bx^?��  �          @�p���z��%?�\@�p�CR�\��z��(Q�>k�@%CR�                                    Bx^?��  �          @��\���׿��Ϳ��\�ICH�\���׿��R��
=�j�\CG.                                    Bx^?�H  �          @�(�������G��{��
=CDQ������}p��%��\)C@��                                    Bx^?��  �          @�z���ff��33��R��\)CBz���ff�h�����ffC?�                                     Bx^?Ĕ  �          @�p����ͿTz��ff���
C>#����Ϳ������G�C;c�                                    Bx^?�:  �          @��\����h�ÿ�33��(�C?33����333���R���RC<�                                    Bx^?��  �          @�������&ff��Q����C<\�����G��   ����C9s3                                    Bx^?��  �          @������R�E��
=q�̣�C=�=���R���\)���
C:�=                                    Bx^?�,  �          @�G���
=�z�H�ff����C@T{��
=�=p��(���ffC=s3                                    Bx^@�  �          @�����
=���H���H��{CF\��
=��G���{���RCC�                                    Bx^@x  T          @�������z�H��������C?�����L�Ϳٙ���z�C=Ǯ                                    Bx^@+  �          @�����ff��(�>W
=@�CH&f��ff�޸R<��
>B�\CHG�                                    Bx^@9�  �          @�Q���Q쿾�R�aG��"�\CEp���Q쿺�H�Ǯ��p�CE�                                    Bx^@Hj  �          @�  ��G����
�����(�CC��G���(��(���  CBY�                                    Bx^@W  �          @�G���ff��Ϳ�R��(�C:\)��ff���.{��G�C9��                                    Bx^@e�  �          @�33��p��B�\�k��((�C<����p��&ff��  �6�HC;��                                    Bx^@t\  �          @����z�Ǯ�����
CF����z῱녿˅���CD�R                                    Bx^@�  �          @������ü���\)��p�C4T{����>#�
��{���RC2                                    Bx^@��  �          @��\��Q�>������\)C.}q��Q�?#�
��=q��\)C,Q�                                    Bx^@�N  �          @��
��=q?��R��33���HC$� ��=q?����޸R��{C"Y�                                    Bx^@��  �          @�33���
>�zῸQ���p�C0�����
>���33�~�RC.�f                                    Bx^@��  �          @�(���Q�>�33������
C/�
��Q�?\)��������C-L�                                    Bx^@�@  �          @��H��{?(���z����C,����{?Tz�������
C)�3                                    Bx^@��  �          @�(���?(�ÿ�����33C,���?aG���{����C)ff                                    Bx^@�  �          @��\��녾�Q�����{C8u���녽����p��θRC5@                                     Bx^@�2  �          @�33��녿�
=���\)CCT{��녿fff�p���  C?�=                                    Bx^A�  �          @��\���H?��ÿ�Q���  C@ ���H@   ���H��=qC33                                    Bx^A~  �          @�����?xQ�z���{C(�����?��
�����C'޸                                    Bx^A$$  �          @�����{>8Q����=qC1�3��{>u�   ����C1=q                                    Bx^A2�  
�          @��H��Q�?5��ff���C+����Q�?B�\��Q���=qC+Y�                                    Bx^AAp  �          @�����=q?ٙ���=q�G
=C����=q?��^�R��C^�                                    Bx^AP  �          @�33��녽��
�}p��:=qC4�q���=#�
�}p��:�HC3��                                    Bx^A^�  �          @�����ff��ff���\�733CI���ff����p��]��CG��                                    Bx^Amb  �          @�\)��
=��Ϳ���8��CP��
=��
��=q�mG�COQ�                                    Bx^A|  �          @�{�����=q�+����CO�3������
�s33�*{CN�R                                    Bx^A��  �          @�
=���\�,(���ff�9CSٚ���\�"�\��\)�s�CRk�                                    Bx^A�T  �          @�  ��(��0�׿W
=��CT0���(��(�ÿ��N�\CS                                    Bx^A��  �          @�ff����!G��!G���\)CQJ=�����H�p���&=qCPY�                                    Bx^A��  �          @������ÿ��?0��@�p�CD8R���ÿ�(�?�@�z�CE
=                                    Bx^A�F  �          @��R�����6ff?^�RA�RCU�3�����;�?�\@�CV�                                     Bx^A��  �          @�Q���G��G
=?��A<z�CY�=��G��Mp�?+�@�CZ}q                                    Bx^A�  �          @��R�w��P��?�Q�ATQ�C\)�w��X��?G�A	G�C]+�                                    Bx^A�8  �          @��x���J=q?�{AG
=C[��x���QG�?333@�33C\�                                    Bx^A��  T          @��
��G��녿s33�-�CL!H��G���33�����[\)CJ��                                    Bx^B�  �          @�z���(���Q�z�H�0z�CJ�=��(���ff��(��\Q�CIL�                                    Bx^B*  �          @�ff����	����z��O�
CMB���녿��R��
=��(�CK�=                                    Bx^B+�  �          @����=q����=q�l��CLs3��=q��녿�=q���
CJz�                                    Bx^B:v  �          @����33��
��p����RCO�q��33�z�� ����G�CMu�                                    Bx^BI  �          @��������R�n{�#33CP��������  �Z�\CO�                                    Bx^BW�  
�          @�G���  �,�Ϳ!G���(�CR����  �%�z�H�*�\CQ�R                                    Bx^Bfh  �          @�����p��6ff����CT�\��p��0  �n{�"�HCS��                                    Bx^Bu  �          @��H��
=�p������L(�CO�\��
=�����R��(�CN�q                                    Bx^B��  �          @�G������ͿaG���RCPJ=����zῙ���R�RCN�                                    Bx^B�Z  �          @�G��]p��J�H�z���z�C^}q�]p��7��{��Q�C[��                                    Bx^B�   �          @�Q���  �U������C[�=��  �O\)�xQ��*{CZ�                                    Bx^B��  �          @��\�|(��Z�H<��
>�=qC\�3�|(��Y����(���
=C\��                                    Bx^B�L  �          @��
��p��A녾�=q�:�HCVff��p��=p��+���
=CU�{                                    Bx^B��  �          @�z���G��P  ��\)�>�RCY#���G��K��5��33CX��                                    Bx^Bۘ  �          @�ff����c�
�.{��ffC\������`�׿&ff���HC\s3                                    Bx^B�>  �          @�{�����j=q=#�
>�p�C^@ �����hQ�����HC^                                    Bx^B��  �          @���
=�A녾k���CV���
=�>{�#�
�أ�CU��                                    Bx^C�  �          @�=q��=q���k��"{CO
=��=q�(���p��Y��CM��                                    Bx^C0  �          @�{���Ϳ��z���z�C:�q���;�\)�Q���=qC7xR                                    Bx^C$�  �          @�{��zἣ�
�
=q��\)C48R��z�=L�Ϳ
=q��ffC3c�                                    Bx^C3|  �          @�\)����?Y��������=qC)p�����?�����H����C&u�                                    Bx^CB"  �          @�  ���R?n{�J=q��C)k����R?��
�&ff��\C(L�                                    Bx^CP�  �          @�33��(��#�
��z����
C5�3��(�=��
����ffC3�                                    Bx^C_n  �          @�
=��G����H���H�g33CB޸��G����
��\)����C@�R                                    Bx^Cn  �          @��R���+�������RC<!H����׿�(���z�C9��                                    Bx^C|�  �          @�\)��
=�#�
�����z�C4{��
=>W
=�\��
=C1s3                                    Bx^C�`  �          @�ff��
=>B�\������Q�C1����
=>Ǯ��z���Q�C/G�                                    Bx^C�  �          @�z����?!G���33�~�HC,�H���?Q녿���k�C*k�                                    Bx^C��  �          @������?}p���=q�=�C(�)����?��׿k��"�RC'J=                                    Bx^C�R  �          @����=q?�ff����d  CxR��=q?�(����
�3�
C�q                                    Bx^C��  �          @�{��ff@7���{�=G�C8R��ff@?\)�.{����C�                                    Bx^CԞ  �          @�ff���@QG��n{�=qC^����@W���ff��Q�C��                                    Bx^C�D  �          @������@J=q�\(���RC������@P  �Ǯ��C��                                    Bx^C��  �          @�ff���@Q녿fff��
CE���@XQ�����=qCxR                                    Bx^D �  �          @����(�@aG��O\)��
C�)��(�@fff��z��@  C
��                                    Bx^D6  �          @�33����@l�Ϳ5��C
Q�����@p�׾#�
�˅C	�{                                    Bx^D�  �          @������@C33��G��O�C\����@L�ͿJ=q��\C                                    Bx^D,�  �          @�����  @J�H��G��MCٚ��  @Tz�E����
C�{                                    Bx^D;(  �          @���z�@G��xQ���C���z�@N�R���H��\)C(�                                    Bx^DI�  �          @�������@.{��G��O\)CL�����@8Q�W
=�	C��                                    Bx^DXt  �          @�(���p�@/\)������33C�
��p�@<(�����:�\C                                    Bx^Dg  �          @�{���@;���Q����\CG����@I����p��HQ�C\)                                    Bx^Du�  �          @��R���H@E����U�C����H@P�׿Tz���C��                                    Bx^D�f  �          @��R��  @L�Ϳ��fffC�
��  @XQ�k��  C�                                    Bx^D�  �          @��R��z�@N�R�:�H���C(���z�@S�
�k����C�\                                    Bx^D��  �          @�p����\@X�ÿ��\�Pz�C����\@c33�=p����C�                                    Bx^D�X  �          @�����(�@Y�����\�&{C0���(�@`�׾�����CB�                                    Bx^D��  �          @�{��
=@G��z����C����
=@J�H��Q�n{C{                                    Bx^Dͤ  �          @����=q@R�\����G�C=q��=q@U�u�\)C�{                                    Bx^D�J  �          @����=q@S33��������C.��=q@Tz�=�?�p�C                                      Bx^D��  �          @����ff@HQ��
=��  C\)��ff@I��=�\)?@  C!H                                    Bx^D��  �          @�������@J=q����  CǮ����@L�ͼ��
�#�
Cn                                    Bx^E<  �          @�����@e���
�L��C� ���@fff>�  @#�
Cu�                                    Bx^E�  �          @��\���\@r�\��\)�5C	����\@r�\>���@VffC	{                                    Bx^E%�  T          @�����Q�@^{���R�Mp�C޸��Q�@^{>u@#33C�{                                    Bx^E4.  T          @�  ��G�@+��.{��\C�q��G�@+�>��@-p�C��                                    Bx^EB�  �          @�����G�@1G�=L��?�C�3��G�@.�R>��H@�33CW
                                    Bx^EQz  �          @�����(�@B�\<�>��
C����(�@@  ?�@�(�C�                                    Bx^E`   �          @�=q��  @Q녽�Q�aG�C�H��  @P  >�G�@��\C�                                    Bx^En�  �          @�����\@Z=q?O\)A
�RC)���\@O\)?���Ah��C��                                    Bx^E}l  �          @�(��fff@s33?�(�AQp�CxR�fff@c33?���A�(�Cff                                    Bx^E�  �          @�{����@c�
?^�RAffC
�\����@W�?�
=AtQ�C�                                    Bx^E��  �          @�����{@)��>�{@g�C�=��{@#�
?E�A33Cc�                                    Bx^E�^  �          @��
����@�H<#�
>#�
C(�����@Q�>��@��C}q                                    Bx^E�  �          @���tz�@]p�?�p�AW
=C	޸�tz�@Mp�?��
A�p�C                                      Bx^Eƪ  �          @��H���@\)>��R@S�
C�H���@=q?5@��C�3                                    Bx^E�P  �          @�������@;��#�
��ffC{����@8��>�G�@��CaH                                    Bx^E��  
�          @�����
@+��u�"�\C�
���
@,(�>W
=@33C��                                    Bx^E�  �          @���@$z�xQ��%p�C8R��@,�Ϳ���33C
=                                    Bx^FB  �          @��\���R@1G���ff��ffC33���R@333<��
>8Q�C�)                                    Bx^F�  �          @��\��Q�@J=q>W
=@�
C}q��Q�@Dz�?=p�A ��C33                                    Bx^F�  �          @�p��^�R?��?���A��RC�q�^�R?��?���A�z�C�                                    Bx^F-4  �          @���\?�@��Bw(�CͿ\?��
@���B��
C�=                                    Bx^F;�  �          @����#33?�Q�@|��BQ  C^��#33?�  @�BcG�C��                                    Bx^FJ�  �          @��9��?�(�@c33B9�HCT{�9��?��@r�\BK
=CxR                                    Bx^FY&  �          @�=q�N�R@(�@,��B
�C��N�R?�Q�@A�B�HCc�                                    Bx^Fg�  �          @�(����?�33>���@�Q�C0����?�ff?8Q�A��CQ�                                    Bx^Fvr  T          @����ff?�G���z�����C#J=��ff?��ÿ���
=Cu�                                    Bx^F�  �          @����HQ�?333�qG��FQ�C'k��HQ�?���e�9�CǮ                                    Bx^F��  �          @�\)��{���33��=qC5�
��{>aG���\��\)C0�                                    Bx^F�d  T          @�=q���\>�{@p�A�
=C/@ ���\�L��@\)A��
C4                                    Bx^F�
  �          @�\)�=p�@	��@.�RB
=C��=p�?��@Dz�B(�C�R                                    Bx^F��  �          @���vff@5>�z�@`��C�{�vff@/\)?J=qA�
C�                                    Bx^F�V  �          @����b�\@c�
<#�
=��
C�=�b�\@`��?#�
@陚C@                                     Bx^F��  �          @����G�@��>��@��RB�.�G�@z=q?�
=ATz�C ��                                    Bx^F�  �          @�z��B�\@��\?:�HA�B�\)�B�\@xQ�?���A�33C �                                    Bx^F�H  �          @����C�
@�G�?c�
A33B�B��C�
@tz�?���A���C �R                                    Bx^G�  �          @��I��@{�?�\)AH��C �R�I��@j=q?�A�(�C��                                    Bx^G�  �          @�
=�A�@z=q?ǮA�ffB��{�A�@c�
@\)A�
=Cc�                                    Bx^G&:  �          @��R�C33@~�R?��Ao\)B��
�C33@j�H@�\A�
=C��                                    Bx^G4�  �          @��>�R@���?�Q�ATQ�B��>�R@qG�?�33A���C c�                                    Bx^GC�  T          @�z��U�@k�?��AO33C{�U�@Z=q?��A�
=CB�                                    Bx^GR,  �          @����^{@j=q=��
?n{C� �^{@e�?@  A��C�                                    Bx^G`�  T          @���Vff@p�׿\)��G�C�3�Vff@s33>��?��Ck�                                    Bx^Gox  �          @��\�S�
@tz�>.{?�\)C��S�
@n{?^�RA�C��                                    Bx^G~  �          @�  �Q�@j�H?Tz�Ap�C�
�Q�@\��?�  A��C�{                                    Bx^G��  �          @�\)�@  @p  ?�
=A\��C ���@  @^{?�{A�ffC�)                                    Bx^G�j  �          @����Y��@_\)?��AIp�CJ=�Y��@N�R?�(�A�33C�=                                    Bx^G�  �          @�{�.{@e@G�A���B�k��.{@I��@)��B
=C�H                                    Bx^G��  �          @�  �Tz�@8��=u?:�HC�Tz�@4z�?��@���C�H                                    Bx^G�\  �          @�ff�n�R@H��>�@��C��n�R@?\)?��AHQ�CG�                                    Bx^G�  �          @��R�;�@mp���{��(�C G��;�@mp�>�p�@�{C L�                                    Bx^G�  �          @��H��@��H?�
=A�{B�
=��@x��@.{B
=B��                                    Bx^G�N  T          @�  ��\)@�  @  A��B׸R��\)@o\)@AG�B�\B�.                                    Bx^H�  
�          @�\)��
=@���?�p�A���B���
=@s�
@0��BffB��                                    Bx^H�  �          @�=q�L(�@Q�?
=@�=qC&f�L(�@G
=?��HAr�RC��                                    Bx^H@  
�          @�\)���@�33?&ff@���B����@y��?�
=A��\B�=q                                    Bx^H-�  �          @��þ�(�@��?�
=A�  B�z��(�@��@*=qA���B�L�                                    Bx^H<�  �          @��þL��@��?��
A�{B�B��L��@��@!�AمB���                                    Bx^HK2  �          @��H��\@��@�A���B�녿�\@�p�@<(�B
=B�=q                                    Bx^HY�  �          @�Q쾀  @��?˅A��B��R��  @��\@!�A��B�8R                                    Bx^Hh~  �          @��ý�\)@��\?��Au�B����\)@��R@
=A���B���                                    Bx^Hw$  �          @�G�=��
@�G�@	��A�p�B�\=��
@���@C33B{B���                                    Bx^H��  T          @�p�>�G�@���?�33A�33B�>�G�@���@7
=A��B�                                      Bx^H�p  �          @��R>\)@��R?У�A���B�  >\)@���@(Q�A�Q�B��q                                    Bx^H�  �          @�z�=���@���?�A��
B���=���@���@8Q�BB��\                                    Bx^H��  �          @�
=?z�@G�@W�B;(�B�  ?z�@Q�@{�Bh33B��q                                    Bx^H�b  �          @��
��@�  ?�@�
=B�\)��@�G�?\A�=qB���                                    Bx^H�  T          @��H�Q�@�=q?J=qA��B�R�Q�@��?޸RA���B�{                                    Bx^Hݮ  �          @�=q�p�@�=q>�
=@�G�B�Q��p�@�(�?���AqG�B�
=                                    Bx^H�T  �          @��H��H@�  >�?�{B�����H@�(�?��A;�B�8R                                    Bx^H��  �          @�(���\@�>���@�Q�B����\@�  ?��Ap��B�u�                                    Bx^I	�  �          @�(��
�H@�z�?�@�(�B��
�
�H@�p�?ǮA���B���                                    Bx^IF  �          @����Q�@�  �����B�p���Q�@�p�?fffA=qB�\                                    Bx^I&�  �          @�\)����@�=q�u�"�\B�\)����@�Q�?\(�A��Bˣ�                                    Bx^I5�  �          @�{���@��H�#�
���Bܞ����@��?�  A*ffB�\)                                    Bx^ID8  �          @��   @�Q�>�ff@�G�B�\)�   @��?�At��B�\)                                    Bx^IR�  �          @�p��,(�@�p�>k�@{B���,(�@���?���AL��B�=                                    Bx^Ia�  �          @���/\)@�(�>W
=@p�B�33�/\)@��?�z�AG�
B���                                    Bx^Ip*  �          @���<��@�\)�.{��RB��q�<��@���>�=q@7
=B�=q                                    Bx^I~�  �          @���=p�@�{�aG��B�k��=p�@���=�\)?8Q�B�ff                                    Bx^I�v  �          @���6ff@�
=�8Q���B����6ff@���>u@#�
B�ff                                    Bx^I�  T          @�=q�6ff@�z�p���"�\B����6ff@��<#�
=��
B���                                    Bx^I��  �          @�(��.{@����33�q�B�.�.{@��\?!G�@�
=B�                                     Bx^I�h  �          @���(��@��?(�@�33B�(��@�(�?�=qA�33B�=q                                    Bx^I�  �          @���,(�@��H?(��@�\B�q�,(�@��H?�\)A���B�                                    Bx^Iִ  �          @��\�0  @�G�>�\)@>{B�p��0  @�(�?��RAW\)B�L�                                    Bx^I�Z  �          @����5�@��R>Ǯ@�\)B��q�5�@���?��AiB���                                    Bx^I�   �          @�=q�;�@��>�@�{B�=q�;�@��R?��As
=B��q                                    Bx^J�  �          @����>{@�G�?W
=A�HB�u��>{@�Q�?�  A�ffB�(�                                    Bx^JL  �          @�Q��;�@��\>�
=@��\B�=q�;�@�z�?��Amp�B��                                    Bx^J�  �          @�  �<��@��׿8Q�����B�B��<��@�=q>aG�@�B��{                                    Bx^J.�  �          @�=q�$z�@����aG��z�B��
�$z�@�(�>�?�Q�B���                                    Bx^J=>  �          @�����@�����H�VffB�z���@�Q�.{���B��                                    Bx^JK�  �          @����(�@�z��33��33B��(�@�ff�c�
�  B���                                    Bx^JZ�  
(          @�{��{@���z�����B�{��{@��R����=B�                                      Bx^Ji0  �          @�
=��  @����#33�뙚BڸR��  @�\)��ff��
=B�8R                                    Bx^Jw�  �          @���(�@�\)��=q����B��)�(�@�\)�(��ٙ�B��                                    Bx^J�|  �          @�(��5@vff������Q�B��3�5@�녾��H���
B��
                                    Bx^J�"  �          @��R�2�\@g�������ffB�=q�2�\@vff�����  B��
                                    Bx^J��  �          @��R�5@�{?��A9�B�G��5@w
=?���A��HB���                                    Bx^J�n  �          @�G��{@���>.{?���B��H�{@��
?�p�AXQ�B�aH                                    Bx^J�  T          @�Q��p�@��\����\)B�p���p�@�
=?��A;\)B�p�                                    Bx^JϺ  �          @��R��\@�(�?�R@���B�{��\@��
?�33A�{B�B�                                    Bx^J�`  �          @�\)�,��@�>�Q�@~�RB����,��@�\)?���Ap��B�#�                                    Bx^J�  �          @���8��@��
    <��
B���8��@�  ?}p�A.�\B�W
                                    Bx^J��  �          @���S33@s33���H����C��S33@s�
>Ǯ@�z�C�                                    Bx^K
R  
�          @�����H@E�
=�У�C�����H@H��>�?�p�C}q                                    Bx^K�  �          @�=q�q�@mp��������C�\�q�@n{>Ǯ@�  C}q                                    Bx^K'�  �          @��\�s33@`�׿�G��[33C	=q�s33@mp���
=����C��                                    Bx^K6D  �          @���p��@\�Ϳ�{�n�RC	���p��@j�H������C�                                     Bx^KD�  �          @�=q�c�
@:=q�(���\)C��c�
@X�ÿٙ����\Cff                                    Bx^KS�  �          @�=q��R@p  �<(����B�p���R@��\���R��(�B�z�                                    Bx^Kb6  �          @����Vff@1G��5�p�Ch��Vff@W
=����Q�C��                                    Bx^Kp�  �          @����C33@hQ��
=��{C\�C33@��\�����
=B��{                                    Bx^K�  �          @�\)�|��@0  ������ffC&f�|��@HQ쿠  �_�C�)                                    Bx^K�(  �          @�ff�tz�@%��ff��  C�3�tz�@C�
��
=���CL�                                    Bx^K��  �          @��R�|��@���   ���
C�f�|��@.�R����=qCff                                    Bx^K�t  T          @�G����R@녿���yG�C����R@!녿B�\���C��                                    Bx^K�  
�          @���|(�?Q��Fff���C(J=�|(�?��
�4z��
=C�\                                    Bx^K��  �          @��H��Q�?aG��#33����C(B���Q�?��H�G���C!�                                    Bx^K�f  �          @����qG�?:�H�E�z�C)��qG�?�Q��5��=qC\                                    Bx^K�  �          @�z��w�?����*�H�(�C#�q�w�?�p�����  C�H                                    Bx^K��  �          @��\�~{@p�@
=qA�\)C��~{?У�@(Q�A���C��                                    Bx^LX  �          @�{�:�H@!�@[�B&\)C&f�:�H?У�@{�BFz�C�
                                    Bx^L�  �          @�\)�>{@C�
@A�B\)C+��>{@\)@j�HB2ffC��                                    Bx^L �  �          @�\)�R�\@-p�@@  B�\C� �R�\?�z�@c�
B,��C��                                    Bx^L/J  �          @�
=�8Q�@hQ�@�\A�(�C s3�8Q�@>{@FffBG�C&f                                    Bx^L=�  �          @��R�N{@QG�@A�=qC��N{@'
=@C�
B
=C�q                                    Bx^LL�  �          @��`��@Z=q?޸RA�
=C�\�`��@8Q�@!G�A�ffC�{                                    Bx^L[<  �          @����&ff@mp�@(Q�A�
=B�
=�&ff@=p�@]p�B%{CT{                                    Bx^Li�  @          @��*�H@h��@{A�B����*�H@:�H@Q�B�Cs3                                   Bx^Lx�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^L�.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^L��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^L�z  L          @�  �u@��@%�A�G�C�3�u?�Q�@EB��C=q                                    Bx^L�   
�          @�
=�z=q?�@9��B�C5��z=q?��
@P��B�C%@                                     Bx^L��  
�          @�\)�Y��@��@QG�B�\C�H�Y��?�G�@l��B6=qC�                                     Bx^L�l            @����Z=q@0  @8��B�RC{�Z=q?�Q�@_\)B&�CL�                                   Bx^L�  
�          @�G��X��@I��@   A��C	��X��@�@L��B(�Cp�                                    Bx^L��  "          @����E�@,(�@VffBQ�C
�=�E�?�\@z�HB?ffC!H                                    Bx^L�^  
"          @�
=�E�@(�@^�RB*�RC���E�?��R@z�HBG
=C
=                                    Bx^M  
�          @�33�C33=�@{�BP��C1�3�C33�^�R@uBI�\CC��                                    Bx^M�  �          @���I��?���@mp�B;�
CE�I��>�@|��BL��C+��                                    Bx^M(P  
�          @������H@�H@(�AŮCc����H?��@.�RA��Ch�                                    Bx^M6�  
�          @��\�C33@b�\@Q�A�{C���C33@4z�@L��B�C	5�                                    Bx^ME�  �          @�33���@�ff?�Q�A}��B�{���@|(�@!�A�
=B�3                                    Bx^MTB  �          @�G����?�p�@B\)C	�3���?���@1G�B<�C\                                    Bx^Mb�  
�          @�
=�R�\�\)@\)BHQ�C=�q�R�\����@mp�B6{CM�                                     Bx^Mq�  �          @����U��:�H@w
=BB=qC@p��U��ٙ�@b�\B-��CO
                                    Bx^M�4  �          @��H�,(��\(�@�B_�HCE�q�,(���z�@s�
BD�RCW^�                                    Bx^M��  T          @�
=��\)�.{@���BJ�Ck����\)�j�H@K�B�
Cs�                                    Bx^M��  
�          @�p���G��Dz�@}p�BH33Cw�f��G��\)@A�B��C|s3                                    Bx^M�&  T          @��׿�p���G�@,(�B��C9Ϳ�p��L��@$z�Bp��CU{                                    Bx^M��  
Z          @�Q�p��@�=q?��@�33B̏\�p��@���?�z�A��RB�L�                                    Bx^M�r  
�          @�(���{@���?�ffA�
=B�LͿ�{@�z�@>�RB�Bޏ\                                    Bx^M�  '          @�\)��Q�@��R?��AAG�B��쿘Q�@�\)@{A���B�W
                                    Bx^M�  
�          @��ÿ(��@���?c�
AQ�B���(��@�\)@G�A�
=B�#�                                    Bx^M�d  
�          @����k�@�(�?��A7\)B��f�k�@���@{AծB�aH                                    Bx^N
  "          @�녿�z�@�  ?��A4��B�\��z�@���@(�A�ffB��H                                    Bx^N�  "          @�=q����@���?J=qABսq����@�(�@
=qA�{B�L�                                    Bx^N!V  
�          @�33��(�@�G�?333@�{B����(�@��@�A�=qBڊ=                                    Bx^N/�  "          @�z��
=@��H?(��@أ�B��f��
=@�
=@�
A���B�Q�                                    Bx^N>�  �          @��
��G�@�ff@�HA�ffB�녿�G�@Y��@]p�B1
=B��                                    Bx^NMH  
�          @���W
=@z=q@�=qB8��B�.�W
=@*�H@�
=Bv�B�                                      Bx^N[�  T          @��>�z�@a�@�p�BL��B��{>�z�@{@�ffB�k�B�\                                    Bx^Nj�  
�          @���?
=@K�@��
BV
=B�
=?
=?��@�=qB�\)B�aH                                    Bx^Ny:  
_          @�  ?�@QG�@��HBR�\B��?�?�p�@�=qB��B��=                                    Bx^N��  "          @�\)?O\)@^{@��BE�HB��3?O\)@p�@�ffB�\B���                                    Bx^N��  "          @�>�@g
=@���B?=qB�  >�@��@�33B}Q�B�z�                                    Bx^N�,  
�          @�\)?�R@q�@y��B6
=B�\)?�R@%�@���Bs�HB��                                    Bx^N��  �          @�
=?Tz�@`  @��BC33B�=q?Tz�@  @��B��B�\)                                    Bx^N�x  
�          @�
=?\(�@^�R@��
BC��B�\)?\(�@{@��B�G�B��f                                    Bx^N�  T          @��>���@j=q@���BA�B���>���@Q�@��B�Q�B�.                                    Bx^N��  �          @��>���@X��@��
BP�B�\)>���@�
@�(�B��=B��q                                    Bx^N�j  �          @�{?.{@[�@���BG�B��q?.{@
=q@�B�B�=q                                    Bx^N�  �          @��?(��@&ff@�p�Bu��B���?(��?��@�ffB�ǮBp�                                    Bx^O�  �          @�Q�?�@A�@��B`  B��?�?�33@�
=B�8RB�W
                                    Bx^O\  "          @���@  @���@e�B${B��f�@  @9��@�=qBb�B��f                                    Bx^O)  
�          @��\��@p��@�=qB=G�B�\��@�R@�ffB}�RB���                                    Bx^O7�  T          @��?8Q�@a�@�=qBO{B�?8Q�@�@��
B���B��{                                    Bx^OFN  �          @��?(�@k�@��BE��B��?(�@z�@��RB���B�k�                                    Bx^OT�  �          @��\?E�@j�H@��B>z�B�Q�?E�@�@�
=B}\)B��                                    Bx^Oc�  
�          @�z�?�  @�@��\Bl��Bz��?�  ?�ff@�=qB�L�B�                                    Bx^Or@  
�          @��?��?�@��RB|  BG�H?��>�G�@�G�B���A~�R                                    Bx^O��  
Z          @�(�?=p�@G�@�\)B|�RB��?=p�?Y��@�B�p�BC�R                                    Bx^O��  
�          @�
=���H@O\)@�(�BU
=B�
=���H?�\)@��
B��3B��                                    Bx^O�2  �          @�Q쿴z�@e�@^�RB(��B�𤿴z�@��@�33Bd{B���                                    Bx^O��  T          @���K�@[�@�A�p�C��K�@.{@:�HB��Cs3                                    Bx^O�~  T          @�  �Q�@j=q?�\)A��HC�{�Q�@>�R@5B�\C	��                                    Bx^O�$  �          @��@��@G�@'�A�G�C�q�@��@\)@Y��B(�CQ�                                    Bx^O��  
�          @�Q��"�\@.{@U�B'{C{�"�\?�z�@}p�BR=qC�                                    Bx^O�p  �          @�z��(Q�@8��@W
=B"CB��(Q�?�@�G�BO  Cs3                                    Bx^O�  "          @�z�ٙ�@8��@q�BA��B��f�ٙ�?�
=@�{Bw
=CB�                                    Bx^P�  
�          @�{��  @J=q@u�BA�\B�#׿�  ?�
=@�=qB}Q�B��                                    Bx^Pb  T          @�=q�	��@\��@Z=qB�B��	��@z�@�Q�BU��C�q                                    Bx^P"  �          @�=q�
=@a�@W
=B�HB��
=@=q@�\)BS��C�                                    Bx^P0�  "          @���Q�@^{@dz�B$�B��Q�@�\@�p�BZ�C�f                                    Bx^P?T  
�          @�p��Q�@hQ�@N�RB��B��=�Q�@"�\@���BG��C0�                                    Bx^PM�  T          @�
=�#�
@p  @A�B�B��=�#�
@-p�@\)B;��CY�                                    Bx^P\�  
(          @����"�\@\)@7
=A�G�B����"�\@>�R@y��B3p�Ck�                                    Bx^PkF  �          @�
=��@s�
@HQ�Bp�B����@.�R@�33BBp�C�H                                    Bx^Py�  r          @�\)�\)@s�
@AG�BffB�=q�\)@0��@�Q�B<�C�                                   Bx^P��  T          @��\��@��\@7�A��HB�L���@C�
@|(�B7��B�W
                                    Bx^P�8  �          @�33��\@��@!G�A�z�B�G���\@g
=@p  B(p�B���                                    Bx^P��  �          @��H�
=@��R@
=qA���B�ff�
=@w
=@\��BQ�B�p�                                    Bx^P��  
�          @�����R@�ff@9��A���B���R@I��@�Q�B833B���                                    Bx^P�*  
Z          @�Q���R@u@N{B�B�=q��R@.{@��RBH��C\)                                    Bx^P��  �          @�=q�&ff@��\@p�A��B��H�&ff@^�R@Y��B�B�u�                                    Bx^P�v  "          @�G��)��@�  @33A�p�B����)��@XQ�@]p�B��C )                                    Bx^P�  
�          @��\�&ff@�@'
=A�
=B�R�&ff@N{@n�RB(Q�C �                                    Bx^P��  
�          @�=q���@���@>{Bz�B�k����@=p�@�G�B:��C �R                                    Bx^Qh  	�          @����(Q�@xQ�@:=qA��B�\)�(Q�@5�@|(�B6Q�C��                                    Bx^Q  "          @�G��.�R@y��@333A�p�B���.�R@8Q�@uB0�Cn                                    Bx^Q)�  
Z          @����<��@z�H@!�A�(�B�
=�<��@>�R@eB"=qCǮ                                    Bx^Q8Z  "          @���L��@~{@(�A�C ٚ�L��@HQ�@Q�B
=C��                                    Bx^QG   �          @���,��@��@(�A�
=B���,��@S�
@g
=B �RC8R                                    Bx^QU�  �          @��
�5@�@�RA�p�B�p��5@N�R@hQ�B ��CQ�                                    Bx^QdL  T          @���(Q�@�p�@$z�AܸRB�u��(Q�@L(�@n{B'�
Cu�                                    Bx^Qr�  T          @�Q��9��@k�@7�A���C 5��9��@(��@vffB1�C	��                                    Bx^Q��  
�          @�G��J=q@k�@+�A�G�C���J=q@,(�@j�HB%��C�H                                    Bx^Q�>  T          @�G��(��@\(�@O\)B�B����(��@�\@�z�BGQ�C\                                    Bx^Q��  �          @�G��QG�@q�@Q�A�C�H�QG�@7�@Z�HB�C
��                                    Bx^Q��  �          @����I��@�  @
=A�  C B��I��@J=q@N�RB��C�                                    Bx^Q�0  T          @���I��@z�H@Q�A��C � �I��@E�@N�RB{C�H                                    Bx^Q��  �          @���L��@���@
=A��HC �=�L��@J�H@P  B
=C=q                                    Bx^Q�|  �          @����N{@��?�{A�G�C ��N{@U@B�\B
=C��                                    Bx^Q�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^Q��   i          @���AG�@�\)?޸RA�=qB�#��AG�@^�R@>{B�C�R                                    Bx^Rn  �          @�Q��G
=@��H?��AX��B�W
�G
=@n�R@$z�A�
=C��                                    Bx^R  �          @����B�\@���?c�
A�B����B�\@���@  A���B�.                                    Bx^R"�  �          @��   @���?�z�AJ�HB��q�   @|(�@!G�A�\B��3                                    Bx^R1`  �          @�ff�0��@��\?p��A
=B�#��0��@��@z�A�=qB�ff                                    Bx^R@  
�          @�\)�(Q�@�?��A�\)B�u��(Q�@l(�@<(�B�HB��                                    Bx^RN�  �          @�  ��@��\?�G�A��
B�
=��@s33@FffB
{B�\)                                    Bx^R]R  �          @�  �p�@��?�p�A�p�B�\)�p�@h��@R�\B\)B�                                      Bx^Rk�  �          @�G��
=@�G�?���As
=B�\)�
=@��H@8Q�A��B��
                                    Bx^Rz�  �          @������@�
=?��HAv{B��H���@���@7�A�G�B�q                                    Bx^R�D  �          @�z��G�@�Q�?(�@�=qB���G�@��@
=qA��RB�=q                                    Bx^R��  �          @�(��У�@��H>�@��Bה{�У�@�@�
A���Bڀ                                     Bx^R��  �          @��ͿǮ@���>��
@W�Bծ�Ǯ@�G�?�
=A��B�{                                    Bx^R�6  �          @��Ϳ��@��>�Q�@tz�B�.���@���?�p�A�  B׮                                    Bx^R��  �          @�\)��33@���>�\)@8��Bѳ3��33@�p�?�Q�A�(�B�Ǯ                                    Bx^R҂  �          @�\)��z�@��
>���@EB��ÿ�z�@�Q�?�z�A�\)B�Ǯ                                    Bx^R�(  �          @�\)�Q�@�G�=���?��B��H�Q�@�  ?��HA�G�B�k�                                    Bx^R��  �          @�
=�
=q@��ü#�
�\)B�u��
=q@���?˅A��B�3                                    Bx^R�t  
�          @�ff���@��
?O\)A	��B��)���@��@�A��B��                                    Bx^S  �          @�Q���
@��?z�@�G�B�33��
@��@
�HA�=qB�{                                    Bx^S�  �          @�����
@��>�?��B�����
@�G�?�\A�B�=                                    Bx^S*f  �          @�  ��Q�@�(�>���@_\)B�\)��Q�@�  ?�(�A�  B�\)                                    Bx^S9  �          @��R��ff@���    <��
Bڙ���ff@�(�?�z�A�  Bܙ�                                    Bx^SG�  �          @�Q��G�@��R�L�;��B�LͿ�G�@�{?��A��HB�(�                                    Bx^SVX  �          @������@���>��@���B�녿���@�(�@ffA��RB�33                                    Bx^Sd�  �          @�p��\@��\>�ff@��B�G��\@��@z�A�33B�{                                    Bx^Ss�  �          @��R>�{@��
�L�Ϳ��B��
>�{@��H?�Q�A���B�u�                                    Bx^S�J  �          @�(��^�R@�>8Q�@ ��B�.�^�R@��H?�{A�BȀ                                     Bx^S��  �          @��
��\)@���?��@�
=BѸR��\)@�p�@  A�z�BԞ�                                    Bx^S��  �          @��
�xQ�@�Q�>�Q�@xQ�B�  �xQ�@�33@�
A���BʸR                                    Bx^S�<  T          @�(��W
=@�Q�?(�@�Q�B���W
=@���@33A���B��f                                    Bx^S��  T          @��
���
@�����B��쿣�
@�{?���A��B��                                    Bx^Sˈ  �          @�(���\@��?��@�p�B�z���\@��
@G�A�33B�8R                                    Bx^S�.  �          @����P  @�G�?��RA��\C Ǯ�P  @U�@/\)A�(�CO\                                    Bx^S��  �          @����L��@��?�G�AXQ�B�33�L��@`��@#�
A��C^�                                    Bx^S�z  �          @��R�P��@��R?��AA�B���P��@fff@p�A�G�C.                                    Bx^T   �          @�p��N�R@�{?���AAG�B�W
�N�R@e�@��A�C�                                    Bx^T�  �          @�z��U@}p�?��HA{\)C!H�U@P��@,(�AC��                                    Bx^T#l  �          @�z��XQ�@w�?ǮA�\)C��XQ�@HQ�@0��A�G�C	(�                                    Bx^T2  �          @��H�n�R@Tz�?���A�ffC
J=�n�R@!G�@6ffB=qC�3                                    Bx^T@�  �          @�{��33@   @�A�z�C����33?�z�@.{B\)C$0�                                    Bx^TO^  �          @�
=�\)?�G�@3�
B�C"p��\)>�  @Dz�B{C0^�                                    Bx^T^  �          @�  ��Q�>�p�@'
=A�{C/J=��Q��@%A�p�C:�)                                    Bx^Tl�  
�          @�Q����@G�?��A�  C
=���?�{@�RA���C#h�                                    Bx^T{P  �          @�����@
=?���AfffC�����?��
@�A�ffC��                                    Bx^T��  "          @�G���Q�@3�
?�  A\Q�C����Q�@{@�A�ffCn                                    Bx^T��  "          @�����  @5�?��RAZ=qCW
��  @  @�A�=qC�                                    Bx^T�B  S          @���w�@dz�?�@�{C	T{�w�@J�H?޸RA�\)C��                                    Bx^T��  T          @�{�J�H@�33�L�;��HB�{�J�H@��?�
=Au��B�33                                    Bx^TĎ  T          @���?\)@�33�#�
��
=B���?\)@�(�?�33Al��B���                                    Bx^T�4  T          @����8Q�@��R�aG��G�B����8Q�@�Q�?��Ah��B�{                                    Bx^T��  �          @�G��9��@��R�W
=�(�B�\)�9��@�  ?�33AiB��3                                    Bx^T��  �          @����1�@�G��W
=���B�(��1�@��\?�
=An�RB�p�                                    Bx^T�&  
(          @���,��@�33�B�\���HB���,��@�(�?�(�AuB�u�                                    Bx^U�  T          @�  �
=@�ff�8Q��z�B���
=@�
=?�G�A�(�B��                                    Bx^Ur  T          @���0  @�(�?���A333B�W
�0  @\)@%�A�z�B�{                                    Bx^U+  T          @�
=�Dz�@���?�Q�A�33B��
�Dz�@^{@U�B�RC�                                     Bx^U9�  "          @����i��@dz�@,��A�
=C���i��@�@p��B"ffCaH                                    Bx^UHd  
�          @����g�@Dz�@P  B	�\C���g�?�\@�33B6{C�                                    Bx^UW
  �          @��R����@.{@Z�HB�C�����?���@���B1  C!                                    Bx^Ue�  
Z          @�Q���33@>{@4z�A�G�C�q��33?���@j�HBz�C\)                                    Bx^UtV  T          @�����z�@W�@\)A�{C���z�@33@`  B�HCT{                                    Bx^U��  �          @�\)���
@g�@�A��\C
�����
@%@^{B  C��                                    Bx^U��  "          @�ff���@e�@
=A�z�C����@(Q�@N{B=qC�                                    Bx^U�H  "          @�Q��w
=@���@{A�p�C�R�w
=@AG�@^�RBz�C�R                                    Bx^U��  T          @�z��u�@\)?�33A��Cٚ�u�@E�@J�HB�C0�                                    Bx^U��  �          @��q�@Y��@#33AԸRC
��q�@�
@dz�B\)C�)                                    Bx^U�:  �          @���\)@J=q@Q�A�=qC�H�\)@��@Tz�B33C�{                                    Bx^U��  �          @�����p�@5@:=qA�CW
��p�?�33@mp�B�C��                                    Bx^U�  �          @�=q��@\)@H��A�G�Cu���?�  @n{B\)C'�f                                    Bx^U�,  T          @���z�@
=@H��A�z�C����z�?aG�@k�B=qC)��                                    Bx^V�  T          @�(����
?�p�@S�
B�C k����
>��@mp�B�RC.�=                                    Bx^Vx  
Z          @��
��(�@g
=?��
AL  C}q��(�@;�@{Aȏ\C.                                    Bx^V$  T          @�z���p�@{@#�
A�
=C#���p�?�z�@P��B�C#&f                                    Bx^V2�  T          @�z�����@�\@!�A�CY�����?��
@EA��C((�                                    Bx^VAj  
�          @����Q�?޸R@��A�=qC �{��Q�?E�@:=qA�z�C+@                                     Bx^VP  �          @�����
=@C33@�A���C����
=@@G�B{C��                                    Bx^V^�  T          @����8Q�@��?�G�A���B�\)�8Q�@o\)@R�\B��B��                                    Bx^Vm\  T          @�Q��333@��\@�A��B���333@c33@dz�B  C L�                                    Bx^V|  
�          @������@���?��A���B�����@z=q@`  BB�{                                    Bx^V��  T          @���<(�@�33?��A�(�B�8R�<(�@h��@X��B��C ��                                    Bx^V�N  T          @��\�>�R@�p�?�\A�B����>�R@o\)@S�
B
��C }q                                    Bx^V��  �          @���2�\@���?�Q�A�z�B��\�2�\@q�@`��B  B��H                                    Bx^V��  T          @��
�333@��@�A���B���333@mp�@e�BQ�B�                                    Bx^V�@  P          @�(��{@�z�@ffA�p�B�=�{@u�@l��B\)B���                                    Bx^V��  &          @�{�!G�@�p�?�
=A`  B��!G�@�=q@K�B��B��\                                    Bx^V�  �          @�ff��@�ff?��HAc33B�ff��@�=q@S33BQ�B��                                    Bx^V�2  
�          @�ff��z�@�  ?��HA�Q�B�Ǯ��z�@���@c�
BB֣�                                    Bx^V��  &          @��ÿ�{@�33?�G�A�B�W
��{@�33@i��B�HB���                                    Bx^W~  
Z          @\���@��?��A�
=Bɳ3���@��@r�\B��B�z�                                    Bx^W$  T          @��Ϳ���@���?�
=A}B��H����@���@h��B��B���                                    Bx^W+�  �          @����@�G�?�33Aw\)B��ÿ��@�=q@g�BG�B�.                                    Bx^W:p  �          @�{��@�Q�?�33A�\)B�녿�@�{@uB��B��                                    Bx^WI  T          @�=q�G�@���@�
A�33B��G�@���@}p�B#  Bǔ{                                    Bx^WW�  �          @�Q�=p�@�
=?�{Ay�B�Ǯ�=p�@�  @c�
B(�BŽq                                    Bx^Wfb  T          @��R�O\)@�p�?�G�Ak�B�LͿO\)@�  @\��B�B�k�                                    Bx^Wu  T          @�\)����@�z�?���A;�B�𤿬��@��\@J=qB �B�L�                                    Bx^W��  T          @�ff�p��@��H?��
A��
B�{�p��@��@k�BQ�B�=q                                    Bx^W�T  
�          @������@�z�?�z�A��B�.����@��@tz�Bz�B��                                    Bx^W��  �          @�  �#�
@��R?�z�A�\)B�G��#�
@�
=@g�B�B���                                    Bx^W��  �          @�ff��=q@��H?�
=A�p�B��ÿ�=q@�33@eB  BΊ=                                    Bx^W�F  
�          @�p���{@��H?��HA�Q�BՔ{��{@��@b�\B�Bܞ�                                    Bx^W��  �          @��H��33@\(�>k�@�C5���33@G�?�(�Al(�C�
                                    Bx^Wے  
�          @��H��{@G��B�\����C h���{@
=q<�>��RC0�                                    Bx^W�8  �          @�����
=@�H�333����C)��
=@ ��>k�@�RCY�                                    Bx^W��            @�  ��@
=�=p���z�C����@{>.{?��C�{                                    Bx^X�  
^          @�Q����?������~�HC'����?�����"�\C"+�                                    Bx^X*  �          @����G�?����c�
�
�\C �f��G�@�ý���z�C�f                                    Bx^X$�  �          @�ff��Q�@u��u��C�3��Q�@e�?���AR�RC�\                                    Bx^X3v  �          @�  �u�@�������
=C�R�u�@�33?�ffAp  Cff                                    Bx^XB  T          @�\)����@��ͽ��
�J=qCxR����@��
?��Ao\)CT{                                    Bx^XP�  �          @������@i����Q�Tz�C�{���@Z�H?�G�A@��C�\                                    Bx^X_h  �          @�p�����@s33����\)C�����@g�?�z�A4��CY�                                    Bx^Xn  X          @�����
@]p����R�<(�C�)���
@U�?}p�Az�C�f                                    Bx^X|�  �          @������
@G
=>�=q@$z�C�����
@333?�33AW�
C^�                                    Bx^X�Z  �          @�������@"�\?�  Af{CE����?�@�A��C!{                                    Bx^X�   
�          @�����ff@3�
?���A)�C����ff@��@�
A��\C\                                    Bx^X��  
�          @�ff��G�@`��>L��?�(�C�\��G�@K�?��RAh  Ck�                                    Bx^X�L  X          @��
���@QG����
�L��CW
���@Dz�?��A;�C�                                    Bx^X��  
�          @������@a�=��
?L��CY����@P  ?���Ab�RC�f                                    Bx^XԘ  T          @��H��=q@xQ�?�\@�
=C
\��=q@Z�H?�z�A�
=C��                                    Bx^X�>  �          @����Q�@j=q?333@��C���Q�@H��@G�A��C.                                    Bx^X��  �          @�33���@XQ�?z�HA�C����@1G�@
=qA�  CB�                                    Bx^Y �  T          @��
�u�@���?:�H@�C��u�@dz�@{A�C	�                                    Bx^Y0  
�          @�(��w�@��H?333@�z�C�q�w�@p��@G�A�z�C�\                                    Bx^Y�  "          @����ff@�=q?(��@θRC���ff@a�@Q�A��C�                                    Bx^Y,|  �          @�{��Q�@��?O\)@�(�C���Q�@l(�@
=A�C	W
                                    Bx^Y;"  "          @��R�g�@��H?��\AffC @ �g�@w�@*=qA�\)C
=                                    Bx^YI�  �          @�  �p��@�  ?���A9p�C�H�p��@n{@333A��CJ=                                    Bx^YXn  "          @�  �{�@�p�?�  A�C�{�{�@n�R@%�A�{Cz�                                    Bx^Yg  T          @�\)�u@�
=?�G�A�C�H�u@qG�@'
=A�\)C��                                    Bx^Yu�  
�          @���r�\@�
=?�z�A3
=CL��r�\@l��@0  AۅC��                                    Bx^Y�`  �          @�\)�~�R@�(�?Y��A=qCE�~�R@o\)@�A��CǮ                                    Bx^Y�  "          @�  �Q�@���?���A;�B��)�Q�@~�R@:�HA�  C}q                                    Bx^Y��  �          @���p��@�
=?�=qAL��C
�p��@h��@:=qA��HC�3                                    Bx^Y�R  �          @����G�@x��=�?�{Cu���G�@c�
?���At  C�3                                    Bx^Y��  
Z          @�ff����@s33=#�
>�(�C!H����@`��?�p�Af�RCh�                                    Bx^Y͞  �          @�{��(�@k�>�p�@c�
C� ��(�@QG�?�p�A��C                                    Bx^Y�D  
�          @���G�@qG�>�G�@���CO\��G�@Tz�?�A��\C��                                    Bx^Y��  �          @�  �b�\@�Q�?Y��A�HB�L��b�\@��H@%A��HC�                                    Bx^Y��  "          @�
=�\(�@��\?z�@��B�  �\(�@�  @�A�G�C
=                                    Bx^Z6  
(          @���^{@��>�  @=qB�  �^{@���@�A���C :�                                    Bx^Z�  T          @�Q��_\)@�(���Q�aG�B���_\)@���?�\A�
=B��                                    Bx^Z%�  T          @�Q��g
=@��ý�\)�.{B��g
=@�ff?�  A�  C�                                    Bx^Z4(  �          @�Q��w
=@��H���
���C��w
=@�  ?�p�A�=qC=q                                    Bx^ZB�  "          @����[�@��Ϳ!G�����B�\�[�@���?��
ADQ�B�z�                                    Bx^ZQt  T          @���y��@�{�Tz�� Q�CG��y��@�?k�A=qCff                                    Bx^Z`  �          @�
=�s�
@�
=?B�\@�z�Cu��s�
@u@=qA��CǮ                                    Bx^Zn�  "          @��R�c33@�ff?E�@�B�8R�c33@���@   A�\)CG�                                    Bx^Z}f  T          @�ff�XQ�@��
>8Q�?��HB�z��XQ�@�@G�A��B���                                    Bx^Z�  "          @�{�C�
@�=q=u?��B�8R�C�
@�p�?��RA��B��=                                    Bx^Z��  �          @���J�H@��ÿ�����B�u��J�H@��
?�\)ATQ�B�{                                    Bx^Z�X  "          @�G��xQ�@{��Q���=qC���xQ�@�G��0���ҏ\C}q                                    Bx^Z��  
�          @�
=�u�@�Q���R���C���u�@���������C�                                    Bx^ZƤ  
�          @����s33@�ff�   ���
C
�s33@��;u�G�C:�                                    Bx^Z�J  �          @�{�a�@z�H��p�����C��a�@p��?�z�AJ�RC8R                                    Bx^Z��  
Z          @���P��@w
=@�A��
C&f�P��@0  @\(�Bz�C�\                                    Bx^Z�            @��\�1G�@��Ϳ�����HB����1G�@�  ?�=qAYG�B�z�                                    Bx^[<  
�          @�Q��n{@�33>�p�@j�HC}q�n{@xQ�@ ��A�ffC�=                                    Bx^[�  �          @����u@��<�>��C��u@~{?�
=A��C�                                    Bx^[�  "          @����O\)@��(���ӅB�ff�O\)@�=q?�Q�A@z�B���                                    Bx^[-.  �          @����U�@�{�޸R��Q�C ���U�@�G�<#�
=�\)B���                                    Bx^[;�  
�          @�z��j=q@��
�u��\C���j=q@�p�?333@�ffCG�                                    Bx^[Jz  "          @���4z�@�z��\)���B�u��4z�@��ü�����B�.                                    Bx^[Y   
�          @�{�:�H@��ÿ�����B��\�:�H@�=q>W
=@
=qB�W
                                    Bx^[g�  "          @�
=�Mp�@��Ϳ\(��33B�L��Mp�@��
?�  A!p�B���                                    Bx^[vl  
�          @�
=�@  @��\�����X��B�k��@  @�\)?\)@�{B��3                                    Bx^[�  �          @������H@+��8Q��=qC  ���H@"�\?aG�ACL�                                    Bx^[��  T          @��
���@_\)�W
=�
=C�\���@b�\?z�@�\)C!H                                    Bx^[�^  �          @����`  @�ff��Q��Dz�C� �`  @��H?�@�
=C �H                                    Bx^[�  �          @��R�b�\@��ÿs33� Q�CT{�b�\@��\?0��@���C��                                    Bx^[��  �          @�z�����@g����
�Y��C�\����@W
=?�=qA[\)C�
                                    Bx^[�P  �          @��H��  ?��H�aG��33C&����  ?���>�=q@9��C'{                                    Bx^[��  T          @�33����?�\?�=qA�\)C�����?�G�@��A��HC(\                                    Bx^[�  �          @�z��p��@~{����Cp��p��@w�?��\A+�C)                                    Bx^[�B  T          @�33��
=@@  <�>�=qC����
=@0��?�Q�AEG�C��                                    Bx^\�  �          @�p���ff@H��>.{?�(�CL���ff@4z�?���Aa�C�                                    Bx^\�  �          @�\)��=q@@��?
=q@��RC�q��=q@#�
?�
=A�z�C                                      Bx^\&4  �          @��R���@I��?
=@�
=C�����@*�H?�\A�{C&f                                    Bx^\4�  �          @�z����R@n�R��G����C
�����R@^{?�{A_\)C��                                    Bx^\C�  �          @��H��ff@k�=#�
>��C
����ff@W�?�(�At��CJ=                                    Bx^\R&  �          @�p��c33@����5���C}q�c33@��?��
A(��C��                                    Bx^\`�  �          @�����@����p����
B�#���@���>�Q�@g
=B��
                                    Bx^\or  �          @���'
=@�G����H��33B�G��'
=@�=q>���@AG�B�                                    Bx^\~  T          @�p��:=q@�G��#�
�љ�B��:=q@��?��AR�RB��                                    Bx^\��  �          @�Q��XQ�@������@��B����XQ�@�p�?ǮAy�B��R                                    Bx^\�d  �          @����.{@�zῥ��O\)B�33�.{@�Q�?:�H@��HB�{                                    Bx^\�
  �          @����@�=q���d(�B�\�@�
=?.{@ڏ\B�L�                                    Bx^\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^\�V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^\�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^\�H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]-�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]<�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]K,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]Y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]hx              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]w              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]�j              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]�\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]ݨ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^	�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^&�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^5�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^D2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^R�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^a~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^p$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^~�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^�p              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^�b              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^֮              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^_F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^_�  a          @����  @~{�333��33C�q��  @z=q?s33A��C	^�                                    Bx^_.�  :          @��R��{@hQ쿮{�T��C����{@w�>W
=@33C
��                                    Bx^_=8  
Z          @�{��  @J�H�����dQ�CQ���  @^�R�u�!G�C                                    Bx^_K�  
�          @�
=���R@Y����{�,(�C+����R@c�
>�33@XQ�C�                                    Bx^_Z�  	�          @�p����
@Dz�>�@���C� ���
@(Q�?�33A��RC�)                                    Bx^_i*  
�          @�{��
=?��H�Tz��=qC Q���
=@�<��
>W
=C�                                    Bx^_w�  %          @������þ�(��*=q��33C8�H����?+��'
=����C,��                                    Bx^_�v  �          @�=q��{���  ��=qCJ�{��{�����;���C?�R                                    Bx^_�  �          @��H�����
�����\CL}q��녿���,(���33CBٚ                                    Bx^_��  T          @�������?����33��Q�C.=q����?����  �_�C'�                                     Bx^_�h  �          @ʏ\����@Q��  ��C������@8Q�&ff��ffCs3                                    Bx^_�  �          @�����=q@   �
=��ffC����=q@H�ÿn{�	Cs3                                    Bx^_ϴ  T          @�G���ff@\)��33��z�Cz���ff@B�\�=p��أ�C޸                                    Bx^_�Z  "          @�=q��(�@G��������C� ��(�@h�ÿ���33C�3                                    Bx^_�   �          @�G���  @P  ��p���
=C����  @qG���\��C�R                                    Bx^_��  �          @��H��(�@w���z�����C{��(�@�녾L�Ϳ�ffC	�                                    Bx^`
L  �          @ə���\)@QG���(���=qC����\)@q녾��H���CǮ                                    Bx^`�  "          @�p�����?�33�  ���
C ������@(�ÿ�ff�C�C�                                    Bx^`'�  T          @�p���p�?�ff�33��p�C'���p�@G���{�K\)C h�                                    Bx^`6>  
�          @�
=���R?�p�����33C'�=���R?�(���z��R�HC ��                                    Bx^`D�  
�          @ƸR���?�
=��G���ffC!���@{�W
=����C0�                                    Bx^`S�  
�          @�p���
=?Ǯ��\)�u�C$�R��
=@��\(�� ��C��                                    Bx^`b0  "          @����H?��
�����4(�C%\)���H?�׾��H����C"&f                                    Bx^`p�  
�          @����
=?5���0��C-.��
=?��׿J=q����C)T{                                    Bx^`|  �          @�
=��
=?\�E���(�C%����
=?ٙ���Q�^�RC$
                                    Bx^`�"  �          @ƸR��
=?�{��(��{�C$�3��
=?��>��@�C$�H                                    Bx^`��  �          @�Q���
=?�ff�G���C%xR��
=?�p���Q�Tz�C#ٚ                                    Bx^`�n  �          @�{��
=?�(��(������C&.��
=?�{���
���C$�f                                    Bx^`�  �          @�{��=q@   ������HC!��=q@G�>\@`  C ٚ                                    Bx^`Ⱥ  T          @�\)���R@(������0��C�
���R@
=?5@�G�C��                                    Bx^`�`  T          @Ǯ���H@/\)�#�
��(�C޸���H@%�?uA�CE                                    Bx^`�  �          @�  ���\@W
=����B�HC�=���\@fff>��?��C��                                    Bx^`��  �          @�z���ff@,(�<#�
=�Q�C��ff@p�?��A%�C�q                                    Bx^aR  
�          @�  ��\)?���?!G�@ÅC(�)��\)?G�?��
A ��C,8R                                    Bx^a�  W          @����
>���@A��C0�����
�#�
@G�A�Q�C:�\                                    Bx^a �  �          @�G���  ���H@=p�A�{C9ٚ��  �У�@ ��A�z�CF��                                    Bx^a/D  
�          @�
=���=u@��A��C3O\����h��@�RA���C=�                                    Bx^a=�  �          @У���{>\?��@���C0����{=�G�?333@�C3�                                    Bx^aL�  �          @Ϯ���
��=q=#�
>\C=�����
��G��Ǯ�`  C<��                                    Bx^a[6  �          @�Q����H�u@N�RA��C>�R���H��@$z�A�33CK0�                                    Bx^ai�  T          @�p����Ϳ��@�\A�(�CC������(�?��AvffCK{                                    Bx^ax�  �          @��7��E�@��B8{Cc  �7���=q@0��A�ffCm�H                                    Bx^a�(  
�          @���&ff�vff@�p�B4�Ck���&ff����@.�RAîCtL�                                    Bx^a��  T          @����,���aG�@�p�B?�Ch�\�,����ff@EAޏ\Cr�)                                    Bx^a�t  �          @����aG���
@��
BD�\CUO\�aG����\@a�B��Ce.                                    Bx^a�  T          @�=q�HQ��33@�\)B\{CS=q�HQ��u@���Bp�Cf�{                                    Bx^a��  T          @��
��33��Q�@�z�Bz�CG+���33�Mp�@G�A�G�CUxR                                    Bx^a�f  �          @��
��(���G�@s33B�CF����(��H��@1G�A�=qCSp�                                    Bx^a�  �          @�����\)@n{B��CK�f����c33@ ��A��HCW�                                    Bx^a��  �          @�Q����\�	��@`  A��RCI�����\�W�@ffA�
=CTL�                                    Bx^a�X  �          @�G���\)���
@�=qB+z�CC���\)�:=q@\(�B\)CT�q                                    Bx^b
�  T          @ٙ��c33=���@���Biz�C2T{�c33�{@��\BO33CT
=                                    Bx^b�  "          @׮�w�>��H@�
=BZ
=C,�q�w���  @��RBKG�CLG�                                    Bx^b(J  T          @ָR��p�?��
@�z�BH�C&#���p���\)@��
BGz�CC�                                    Bx^b6�  �          @�{��G�?c�
@��B8\)C(����G���\)@�Q�B5��CAٚ                                    Bx^bE�  �          @�����p�?��@�G�B,��C$W
��p��!G�@�{B3z�C;��                                    Bx^bT<  �          @�z����\?J=q@��B@��C)�����\��  @��B;��CD�                                    Bx^bb�  �          @��H����?(��@�(�B533C+�����ÿ�  @�  B/
=CCxR                                    Bx^bq�  T          @�=q����?�\@E�A�{C":�����>\@a�BG�C0�                                    Bx^b�.  T          @��H�C33?�@���Bw�\C*&f�C33��@���Bd�CS�                                    Bx^b��  �          @��H�:�H?��@��
B|G�C(n�:�H��@��Bj33CS�q                                    Bx^b�z  �          @����8��?5@��B{�\C&#��8�ÿ�
=@��Bl�HCR+�                                    Bx^b�   �          @�  ��=q?�  @���B/��C'���=q�L��@��B1�C>��                                    Bx^b��  �          @�{��\)?������
=C)aH��\)?���#�
����C(J=                                    Bx^b�l  "          @θR����?������R���\C!T{����@%����G�C�H                                    Bx^b�  �          @�ff��Q�@G���  �%Q�C  ��Q�@q��>�R��
=C
s3                                    Bx^b�  �          @�{��G�@>�R����  C�)��G�@�33�!����\C�
                                    Bx^b�^  �          @θR���@.�R�xQ��  C����@���p���p�C	�                                    Bx^c  T          @�G��hQ�@\(������HC���hQ�@�=q�����B�                                    Bx^c�  
�          @���>�R@r�\����(z�C =q�>�R@�
=��R��G�B�\                                    Bx^c!P  
�          @��H��
=?��
?8Q�@љ�C#n��
=?��?�\)AIp�C'W
                                    Bx^c/�  
�          @����ff?Tz�@0  A�z�C,���ff��{@6ffA�Q�C7E                                    Bx^c>�  �          @�ff��{��  @`��B ��C@����{�#33@-p�A�ffCM#�                                    Bx^cMB  �          @�Q��~{�W
=@�G�BR
=C?��~{�8Q�@�Q�B*\)CX�                                    Bx^c[�  
Z          @׮�I��?&ff@��Bt��C(E�I���޸R@�(�Be�CQ                                      Bx^cj�  T          @�  �mp�?��@�(�BV��C {�mp��k�@�
=B\33CA��                                    Bx^cy4  T          @�Q��U@?\)@�G�BA�\C
&f�U?8Q�@�=qBn\)C'Ǯ                                    Bx^c��  �          @����C�
@p��@���B+��C&f�C�
?��@���Bi��C��                                   