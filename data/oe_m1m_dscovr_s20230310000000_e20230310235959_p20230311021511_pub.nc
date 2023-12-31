CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230310000000_e20230310235959_p20230311021511_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-11T02:15:11.513Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-10T00:00:00.000Z   time_coverage_end         2023-03-10T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxl��  
�          @��\@c33��Q�?��HA噚C��@c33����?��HA��\C��H                                    Bxl�"&  "          @�G�@P  �k�?�{A��C��)@P  ��
=?��HA�z�C���                                    Bxl�0�  "          @y��@Fff?��?�ffA�G�A���@Fff>aG�@��B{@�Q�                                    Bxl�?r  T          @s33@Q�?=p�?�G�A݅AL��@Q녾��R?��A��HC�S3                                    Bxl�N  
�          @tz�@E�?
=@ffBz�A-�@E���R@B�
C�P�                                    Bxl�\�  T          @|(�@X��?��?�\)A㙚A ��@X�þ�?�33A�(�C���                                    Bxl�kd  �          @q�@O\)>aG�?�\)A@}p�@O\)�L��?��HA�ffC��                                    Bxl�z
  
�          @w�@%��=q?��
A���C�h�@%�\)?
=qA	G�C�
=                                    Bxl���  �          @��H@(��,��@�
A��HC��\@(��XQ�>��R@�C��                                     Bxl��V  
�          @���?����=q@9��B9p�C���?����e?�
=A��RC�P�                                    Bxl���  �          @��\?����@Dz�BD(�C���?���c�
?�33A��\C���                                    Bxl���  �          @�(�@(�����@E�BB=qC��{@(��)��?�p�A�(�C�XR                                    Bxl��H  �          @�
=@8�ÿ���@-p�B!�C�}q@8���%�?�=qA�G�C�%                                    Bxl���  "          @�\)@_\)��=q@ffB��C��f@_\)��p�?���A�=qC���                                    Bxl���  "          @�ff@Z=q���@
=B	33C��@Z=q��G�?�Q�A��
C�]q                                    Bxl��:  T          @�
=@S�
�!G�@"�\B
=C���@S�
���?���Ạ�C�5�                                    Bxl���  T          @�{@K��0��@'�BffC�ٚ@K���p�?�{A�
=C��                                    Bxl��  �          @��@+���=q@G
=B<p�C�@+��#�
@z�A��C�#�                                    Bxl�,  
�          @�Q�@�����@S�
BM  C�N@���.{@��A��\C���                                    Bxl�)�  
�          @�Q�@?\)��(�@0  B!C��@?\)�\)?�Q�A��C��                                    Bxl�8x  "          @��@R�\�n{@��B��C�#�@R�\��?˅A��
C���                                    Bxl�G  T          @���@`�׿W
=@�RA�G�C�K�@`�׿��?���A��C��                                     Bxl�U�  T          @�  @[��+�@Q�B��C��H@[�����?�A��
C��                                    Bxl�dj  
�          @��H@Mp��
=q@3�
B#z�C�5�@Mp���Q�@ffA�Q�C�aH                                    Bxl�s  
R          @���@0��?Y��@P��BAA�p�@0�׿�  @N{B>��C��                                    Bxl���  
�          @�ff@%?�=q@QG�B<G�A���@%�\@g
=BX
=C���                                    Bxl��\  T          @�ff@"�\?�=q@J=qB4��B  @"�\��G�@i��B\Q�C��                                    Bxl��  T          @�z�@ff@0��@�HB=qBFG�@ff?��@`��BV�A���                                    Bxl���  
(          @�G�@,(�@<��?�
=A�33B>�@,(�?�G�@1G�B"�Bz�                                    Bxl��N  �          @�(�@AG�>aG�@'�B#�R@�p�@AG�����@ffBp�C�1�                                    Bxl���  �          @��@C33?��@�BA�=q@C33��G�@.{B'  C���                                    Bxl�ٚ  
�          @��
@<(�@!G�?�(�A�B"��@<(�?�{@@  B.�A�33                                    Bxl��@  "          @�(�@S�
?�z�@!G�B��A���@S�
��{@0  BQ�C�R                                    Bxl���  T          @���@O\)?�@*�HB
=Aٙ�@O\)<#�
@I��B0�R>B�\                                    Bxl��  �          @��@S�
?���@2�\B33A�\)@S�
��@L��B0{C��H                                    Bxl�2  "          @��@U�?�ff@1G�B
=A�{@U���@J�HB.ffC��R                                    Bxl�"�  "          @���@Z=q?���@,(�B  A���@Z=q�L��@HQ�B*
=C��3                                    Bxl�1~  �          @�ff@e?   @ ��B
��@���@e�fff@��BC��                                    Bxl�@$  �          @���@q녿0��@�A�p�C�Ф@q녿��?�=qA�\)C�\)                                    Bxl�N�  �          @�  @u��\)?��A�=qC�1�@u�	��?c�
A5C�]q                                    Bxl�]p  �          @���@xQ쿚�H?�(�A���C�XR@xQ���
?��AU��C�H                                    Bxl�l  T          @�  @w���
=?�Q�A��C��@w���?��AUG�C�33                                    Bxl�z�  �          @��@{��n{?��HA�33C�T{@{�����?�
=AtQ�C���                                    Bxl��b  
�          @��H@�G��\(�?��RAͅC���@�G���\?��RA{�C�'�                                    Bxl��  �          @��
@vff����@�\A�=qC��@vff�Q�?���A�
=C��H                                    Bxl���  
�          @�p�@{���  @�
A�
=C���@{���
?�Q�A�{C�+�                                    Bxl��T  �          @�@~{�L��@�A�C�E@~{��33?���A��HC�+�                                    Bxl���  �          @���@z�H�p��@�
A�RC�:�@z�H� ��?�p�A�Q�C�g�                                    Bxl�Ҡ  �          @���@l(��^�R@+�B(�C�^�@l(����?�=qA�{C���                                    Bxl��F  T          @���@{���z�@�RA��C��=@{���z�?�G�A���C��                                    Bxl���  
Z          @���@~�R�B�\@Q�A�33C��q@~�R���
?�(�A��RC�)                                    Bxl���  T          @�\)@��H�!G�@G�A�=qC��f@��H��(�?�{A��
C��R                                    Bxl�8  �          @���@��W
=@
=qA�ffC�S3@���?�33A�ffC�!H                                    Bxl��  �          @�G�@�=q��
=@�A�\)C��@�=q�У�?���A��C�)                                    Bxl�*�  "          @�Q�@�  =u@#�
Bz�?k�@�  ���
@{A��HC�"�                                    Bxl�9*  
�          @���@���?��H?�A�G�A��R@���<�@G�A�>�ff                                    Bxl�G�  �          @�{@�Q�?��?��A�33A���@�Q�>�
=?��A���@���                                    Bxl�Vv  T          @��
@�\)>�
=@
=A�ff@��
@�\)�^�R@�RAڸRC�,�                                    Bxl�e  
�          @��@���>���@,(�B@�  @��׿�{@{A��C�C�                                    Bxl�s�  �          @�\)@���>.{@:�HB��@ff@��ÿ���@%�A�{C���                                    Bxl��h  "          @�G�@x��>�=q@K�B��@}p�@x�ÿ�@7
=B
p�C��
                                    Bxl��  "          @���@�G�?=p�@9��BA%�@�G��^�R@7
=B
��C��f                                    Bxl���  �          @�(�@������@��A�=qC��@����{?�A��RC�S3                                    Bxl��Z  
�          @�33@�z��?�(�A�{C��\@�zῼ(�?�z�A�p�C��R                                    Bxl��   
�          @��\@��R�
=?�(�A�C�>�@��R��33?�A`��C�O\                                    Bxl�˦  �          @�@��H�O\)?ǮA��\C���@��H��G�?k�A,Q�C��=                                    Bxl��L  
�          @�=q@�ff��z�?��A�ffC��H@�ff�G�?
=q@�ffC�z�                                    Bxl���  
�          @�\)@mp�� ��@
=qA�C��\@mp��P��?��@�=qC�O\                                    Bxl���  "          @�  @����#33?�
=A�\)C��\@����:�H���Ϳ�\)C��                                    Bxl�>  T          @�@���{?�z�A�\)C��\@���(Q�=u?333C��{                                    Bxl��  "          @���@�33�Q�?�
=A^�HC��@�33�)���aG��(Q�C��                                    Bxl�#�  �          @���@����p�?Q�A{C�O\@����!녿\)��Q�C��                                    Bxl�20  T          @�\)@p  �333?
=@�(�C��q@p  �,�Ϳp���8z�C�)                                    Bxl�@�  �          @���@w��%�?��AM��C�*=@w��0�׾�
=����C�E                                    Bxl�O|  
�          @�\)@vff�\)?�z�Ab�\C��\@vff�.�R��z��aG�C�U�                                    Bxl�^"  
�          @�  @|(��{?aG�A+\)C���@|(��$z����  C�u�                                    Bxl�l�  
�          @�\)@tz��,��?��@�33C�ff@tz��'��^�R�)�C��                                    Bxl�{n  �          @��
@h���5=#�
>��C��@h���p���
=��{C���                                    Bxl��  �          @���@`  �AG���G���z�C���@`  �"�\��33��\)C�f                                    Bxl���  �          @��@\(��?\)�(����C��f@\(��  �33��{C�ff                                    Bxl��`  �          @���@g��Fff�u�:=qC���@g��"�\��ff���C�|)                                    Bxl��  �          @��
@U��L(��.{��C�
@U��*=q���
����C��{                                    Bxl�Ĭ  T          @�ff@=p��Tz�#�
���HC��
@=p��5���p�����C�q                                    Bxl��R  �          @��@<(��XQ�.{�
�HC���@<(��4z������C��                                    Bxl���  
�          @�  @?\)�W
=�B�\�{C��{@?\)�1녿����{C��                                     Bxl��            @��@Vff�E�aG��0  C��H@Vff�#33��\���\C�^�                                    Bxl��D  
�          @�G�@R�\�HQ쾏\)�_\)C�33@R�\�#33������C��                                    Bxl��  T          @���@���e��  �P��C�&f@���=p��33��=qC���                                    Bxl��  S          @�@2�\�P  ?\(�A5C�L�@2�\�O\)�k��BffC�^�                                    Bxl�+6  �          @�{@(��g�������
C��)@(��(��>{�(�C��{                                    Bxl�9�  �          @�?��
�R�\�.�R�=qC��?��
��Q�����k�C���                                    Bxl�H�  �          @��H?�ff�_\)�G���C�<)?�ff��p��aG��`=qC��{                                    Bxl�W(  T          @��
?�ff�|�Ϳp���HQ�C���?�ff�:�H�4z��!�HC��{                                    Bxl�e�  "          @�(�@��s33�����C��3@��B�\��� ffC�p�                                    Bxl�tt  �          @�33@
�H�n�R����=qC�
@
�H�>�R��\���HC�f                                    Bxl��  "          @��\?���G�������\)C���?��+��Y���A33C��
                                    Bxl���  �          @�(�?�=q�����p��x  C�` ?�=q�=p��J�H�-�RC�f                                    Bxl��f  T          @��?�
=���H�ٙ����C���?�
=�*�H�b�\�E�C��                                    Bxl��  �          @��\?�
=���ÿ�33��G�C���?�
=�(���^{�D�C�4{                                    Bxl���  
�          @��H?����  ������z�C�/\?���?\)�R�\�7�
C��q                                    Bxl��X  �          @��
>�  ���R��{�`(�C���>�  �Q��Mp��1Q�C�33                                    Bxl���  �          @�(�?h����
=������  C��?h���<(��U��<C���                                    Bxl��  
�          @�z�?�Q���
=��(���z�C��f?�Q��9���Y���=\)C�33                                    Bxl��J  
�          @��\?O\)���ͽL�Ϳ#�
C��=?O\)�~�R�����C���                                    Bxl��  "          @��
?E�����=��
?c�
C��{?E����
=���
C�=q                                    Bxl��  �          @��?Tz����R�.{� z�C���?Tz��n�R�=p����C�Ff                                    Bxl�$<  
�          @�  ?#�
��p�=�?��C��
?#�
�������\C�:�                                    Bxl�2�  �          @���>��H���.{��G�C��R>��H�z�H�Dz��G�C���                                    Bxl�A�  �          @��?�G���=q��Q���z�C�@ ?�G��N�R�b�\�9(�C���                                    Bxl�P.  
�          @��?�ff�����{��  C���?�ff�.�R�o\)�L{C���                                    Bxl�^�  
�          @��
?�  ���
�\)��{C�w
?�  ����  �^�C��q                                    Bxl�mz  �          @��\?}p���Q쿗
=�dz�C�.?}p��S�
�Q��-��C�O\                                    Bxl�|   �          @��\?aG����R>�p�@�(�C�G�?aG���Q���\��G�C��3                                    Bxl���  
�          @���?�����p������Q�C�R?����o\)�6ff���C���                                    Bxl��l  �          @�{?�=q���׾�
=����C�g�?�=q�z�H�0  �=qC���                                    Bxl��  T          @�
=?k���(�=#�
>�ffC�W
?k���  �=q���
C�q                                    Bxl���  T          @��R?5����=#�
>�G�C�%?5��Q��=q��p�C��H                                    Bxl��^  
�          @��R?^�R���
>\)?���C�
=?^�R��G���
����C���                                    Bxl��  
�          @�?�������>L��@�C�{?�������p����HC��\                                    Bxl��  T          @��?@  ��
=>��R@l��C�}q?@  ��  ���ffC���                                    Bxl��P  "          @�  >���33?s33A;
=C��\>����R���H��33C���                                    Bxl���  T          @��H?���ff?^�RA&�HC�#�?����׿�=q��C�E                                    Bxl��  
�          @�(�@{��=q��
=��
=C��q@{�b�\� ����=qC��                                    Bxl�B  "          @�=q@*=q���׿B�\�	G�C��@*=q�U��1��z�C�J=                                    Bxl�+�  
�          @�
=@����\����  C���@��l(��0��� 
=C��=                                    Bxl�:�  �          @�{@�����;�p����HC��@���w
=�(Q���C�+�                                    Bxl�I4  �          @�{@p���녽�Q쿃�
C�.@p��z�H��֣�C�{                                    Bxl�W�  T          @�ff@Q���33��G���G�C��3@Q��|(�����=qC��
                                    Bxl�f�  �          @�
=@�\��p���{�n�RC�@�\�x���&ff��C�:�                                    Bxl�u&  T          @�{@(����.{��\)C��f@(��\)�p���33C�aH                                    Bxl���  �          @�@��{?(�@�Q�C��q@��(�������C��)                                    Bxl��r  "          @�G�@33��\)?��\A;
=C�N@33���Ϳ����s�C��                                     Bxl��  T          @��R?�����ff?n{A.=qC�)?������\������C�c�                                    Bxl���  
�          @���?�{���
?aG�A%p�C�33?�{��ff��G�����C�}q                                    Bxl��d  "          @�ff?�p����H?���ARffC��
?�p��������\�n�RC��                                    Bxl��
  
�          @���@+���G�?��A�  C��f@+���=q�����C��H                                    Bxl�۰  
�          @��H@
=��ff?���A�p�C�N@
=��=q��ff��p�C�^�                                    Bxl��V  �          @�  @�
��?
=q@��
C��@�
���H��=q���C��q                                    Bxl���  �          @�z�@)����>8Q�?�\)C�Ǯ@)����������\)C�@                                     Bxl��  �          @�@���=q?Y��A\)C�Ff@����Ϳ��R��(�C���                                    Bxl�H  T          @�\)@����(�?�G�A1�C��{@�����ÿ����u�C���                                    Bxl�$�  T          @��@
=��\)?=p�A�C�
=@
=�������z�C��)                                    Bxl�3�  �          @�ff@	����ff?��@�33C�B�@	�����Ϳ�\��\)C�H                                    Bxl�B:  T          @�?��R��  ?\)@��C�` ?��R��p���=q���C�"�                                    Bxl�P�  �          @�\)?�����H?
=q@�ffC���?����������\)C�Y�                                    Bxl�_�  
�          @�(�?��H��\)?�\@�ffC���?��H��33��p�����C���                                    Bxl�n,  �          @���?�\�U@�RB  C��
?�\���
?
=q@�z�C���                                    Bxl�|�  "          @���?�ff�1G�@]p�B9  C��f?�ff��33?�A��C�޸                                    Bxl��x  
�          @��?����8��@aG�B7�C��?������?�
=A�Q�C���                                    Bxl��  "          @�33?�ff�L(�@FffB!33C���?�ff��G�?�Ab{C�c�                                    Bxl���  �          @�z�@�
�_\)@��A���C�>�@�
����>�\)@`  C�/\                                    Bxl��j  �          @��
@��U�@��A�\)C��{@���33?�@��C�k�                                    Bxl��  "          @�ff@	���N�R@'�B�C���@	�����H?:�HA\)C��
                                    Bxl�Զ  T          @�  @33�J=q@8Q�BC�y�@33���?�  AB�RC�)                                    Bxl��\  �          @�=q?��L(�@G
=B �RC�?���G�?�Q�Ad(�C���                                    Bxl��  T          @���?����U�@C33B=qC��\?�����z�?���ANffC�                                      Bxl� �  �          @�ff?�33�xQ�@7
=B��C�J=?�33����?#�
@�p�C�˅                                    Bxl�N  �          @��?�Q��^{@W�B*p�C�w
?�Q����?��Ap(�C�'�                                    Bxl��  
�          @���?�  �S�
@c�
B4��C�Y�?�  ���?��
A�{C��
                                    Bxl�,�  �          @��?\�XQ�@VffB(G�C�!H?\��=q?��At(�C�7
                                    Bxl�;@  �          @���?��mp�@1�B�C�~�?����H?(��@�G�C���                                    Bxl�I�  �          @��?�����z�?�p�A��C�b�?������\�u�5�C�t{                                    Bxl�X�  T          @�{?�=q���@�RA�G�C�?�=q��  >u@0��C��                                    Bxl�g2  "          @�{?���p  @@  B��C�w
?����\)?W
=AG�C��=                                    Bxl�u�  
�          @�  ?У��p  @7�B�C���?У����?:�HA�RC��=                                    Bxl��~  "          @���?�33�0  @p��BA�C�T{?�33���R@   A�\)C�#�                                    Bxl��$  "          @��?�(��,(�@u�BH�RC�Y�?�(���@A�{C�1�                                    Bxl���  �          @�\)?�\)�%@\)BV�HC���?�\)��p�@�A�\)C��                                    Bxl��p  �          @�{?���R@���BZ�\C���?����H@
=A���C��                                    Bxl��  "          @���?����C�
@c�
B:�C�&f?�����z�?�
=A�ffC��H                                    Bxl�ͼ  �          @��H?����"�\@x��BV�\C��)?������\@{A��HC��R                                    Bxl��b  �          @��?�z��'�@p��BN��C�"�?�z����H@z�A̸RC��H                                    Bxl��  T          @���?�Q��7�@^{B=
=C�N?�Q���p�?ٙ�A���C��H                                    Bxl���  
�          @���@=p��l��>�{@�=qC�T{@=p��Z�H������Q�C�p�                                    Bxl�T  �          @�33@HQ��c�
>.{@�\C���@HQ��N{���
����C�q                                    Bxl��  "          @��\?����H@���B��)C��H?��\(�@xQ�B@�C�G�                                    Bxl�%�  
�          @���?����%�@��BdG�C���?�����
=@9��A���C���                                    Bxl�4F  T          @���?�(����@�B��=C�]q?�(��dz�@}p�B6�RC�1�                                    Bxl�B�  �          @��>����@�\)B�ǮC���>��s�
@x��B5z�C��R                                    Bxl�Q�  "          @�ff?(�ÿ��@���B�=qC���?(���Z=q@���BH=qC�xR                                    Bxl�`8  T          @�33>�ff=u@���B�8RA�
>�ff��@��RByffC�<)                                    Bxl�n�  "          @��?\)��z�@�ffB���C�g�?\)�y��@tz�B0z�C�q                                    Bxl�}�  �          @���?333��@�Q�B��C�Z�?333���@N�RBz�C��=                                    Bxl��*  �          @�p�?+��p�@��B��)C�h�?+����@a�B�C�Y�                                    Bxl���  �          @��\?�Ϳ���@�
=B��C�` ?����G�@p��B*�RC��)                                    Bxl��v  �          @��H>�� ��@�B��=C�q�>���{@h��B#G�C�"�                                    Bxl��  �          @�(�>\��{@�{B���C���>\�o\)@�B@�C��f                                    Bxl���  T          @�(�?(�ÿǮ@�=qB���C���?(���w
=@~�RB6
=C��                                    Bxl��h  T          @��R�#�
�#�
@�p�B�\C�\)�#�
�L��@�
=B_z�C���                                    Bxl��  �          @�\)�aG���{@��RB�ǮClٚ�aG��<��@���BkffC��3                                    Bxl��  �          @�z�?L�Ϳ�p�@��
B��)C���?L�����
@g�B#(�C�t{                                    Bxl�Z  �          @�p�>k����\@�G�B���C�N>k��^{@�{BO��C��H                                    Bxl�   �          @�\)>#�
�n{@��\B�C���>#�
�Z=q@���BS�
C�\)                                    Bxl��  
�          @��R?Q녿���@��B�.C�}q?Q��z�H@��B5�C��                                    Bxl�-L  "          @�  >k��+�@�{B��RC���>k��N{@��B^�HC�{                                    Bxl�;�  T          @�G��u��\)@�Q�B��qCd��u�9��@��Bo
=C��                                     Bxl�J�  T          @�G����
��  @���B��fC|Q콣�
�7�@���Bp�RC�8R                                    Bxl�Y>  T          @��׾8Q�>�Q�@�\)B��HB��8Q���
@�Q�B�u�C���                                    Bxl�g�  �          @�녿�\�\@��RB�.CX�q��\�=p�@���Bi�HC�3                                    Bxl�v�  �          @��\���Ϳ�33@��HB�.Cbk������e@�{BG33C|��                                    Bxl��0  T          @��H��Q��@��B��qCFk���Q��?\)@�Q�B\�\CtQ�                                    Bxl���  T          @�=q��G��L��@�33B��
C5�ÿ�G��&ff@��RBj�Co�                                    Bxl��|  T          @�33��Q���@���B�.Cn�쾸Q��K�@�z�Bc  C��q                                    Bxl��"  �          @��
�333��=q@���B�#�CIE�333�8��@�G�Bm��C�&f                                    Bxl���  
�          @�p���
=>u@�G�B�u�C(ff��
=��H@�Q�B{��Cs�R                                    Bxl��n  �          @��Ϳ��\>\)@���B��)C,#׿��\� ��@�\)Bz\)Cw�f                                    Bxl��  �          @�p���=��
@�(�B���C*�R���&ff@���B}\)C�Z�                                    Bxl��  "          @�p���33�W
=@���B�G�C>xR��33�4z�@�=qBl�Cw�H                                    Bxl��`  
�          @��R��  ��Q�@�=qB��qC80���  �.{@���Bo\)CuY�                                   Bxl�	  
�          @��R����u@��B�L�CA(�����7�@��Bl�RCz�                                   Bxl��  �          @�  �u<#�
@�p�B���C3��u�*�H@���Bw(�CzB�                                    Bxl�&R  �          @���Q�=��
@�B�8RC.�R�Q��'
=@��\B{Q�C|�
                                    Bxl�4�  
�          @�\)�fff>u@�z�B���C$�\�fff�(�@��
B�\)Cy�                                    Bxl�C�  
�          @��
��z�&ff@��RB�ǮCvaH��z��I��@���Bb�C�j=                                    Bxl�RD  "          @�?
=q��@��RB�W
C�.?
=q��33@{�B'p�C��                                    Bxl�`�  �          @�p�?�(��"�\@�{Bv{C��?�(���33@`  BQ�C�h�                                    Bxl�o�  T          @�?�G��+�@�(�Bo��C��)?�G���ff@X��B\)C��f                                    Bxl�~6  "          @��R@��E�@�G�BGffC�@ @���G�@+�AָRC���                                    Bxl���  S          @�@�R�dz�@���B*�C�b�@�R���@   A���C�5�                                    Bxl���  �          @�Q�@Q��p  @�{B0C��{@Q���
=@�A��C��                                    Bxl��(  �          @�=q?����c33@���B@��C�aH?�����ff@p�A��C�G�                                    Bxl���  T          @�=q?�{�xQ�@���B3  C�˅?�{���
@A��C��=                                    Bxl��t  
�          @��H@Q����\@tz�B��C�&f@Q����?�{Au�C�                                      Bxl��  �          @��
@{��  @g�BffC�
@{��p�?���AL��C�E                                    Bxl���  "          @��@(�����@`��B��C��@(����?��RA<��C��                                    Bxl��f  �          @��@$z����@U�BffC�=q@$z���z�?�ffA Q�C��q                                    Bxl�  
�          @�z�@ ����(�@^�RB
\)C��@ ����
=?�
=A2{C�Q�                                    Bxl��  
�          @��@33����@]p�B�C�j=@33���?��A#�
C�&f                                    Bxl�X  	�          @���@�R��@R�\Bz�C���@�R����?aG�Az�C���                                    Bxl�-�  �          @ƸR@
=q�n�R@��RB7��C��@
=q���@
=A��HC��                                    Bxl�<�  "          @�p�?#�
=�G�@ÅB��fAz�?#�
�%@���B��\C���                                    Bxl�KJ  �          @�
=?O\)<�@���B��@�?O\)�*�H@�G�B}  C�u�                                    Bxl�Y�  T          @ə�?W
=>��H@�B�#�A�33?W
=�G�@���B�� C�                                      Bxl�h�  "          @ʏ\?G�����@�Q�B��RC�,�?G��5�@��\Bx�HC��=                                    Bxl�w<  �          @�p�>���\@��HB�ffC��q>��N�R@�\)Bl�HC�C�                                    Bxl���  �          @У�?���?O\)@��
B��qB�?����33@\B�
=C�                                      Bxl���  
�          @�?Q녿   @�=qB��qC�Q�?Q��L��@�
=BkQ�C�+�                                    Bxl��.  
�          @��?B�\�W
=@�33B���C�%?B�\�=p�@��
Bu��C�5�                                    Bxl���  
�          @θR?Y��<��
@���B��
?���?Y���0��@���B}z�C��\                                    Bxl��z  "          @�\)?�R>8Q�@�B�u�A�
?�R�(Q�@��
B��C���                                    Bxl��   
�          @�
==�Q�>��
@�B�{B��R=�Q��   @�B�B�C�\                                    Bxl���  	�          @�\)?��Q�@���B��C�b�?��`��@��Bc  C�Ff                                    Bxl��l  �          @�?���ff@�z�B���C���?��J�H@�=qBp  C���                                    Bxl��  �          @�z�?����  @���B�Q�C��q?����=q@�Q�BH{C�P�                                    Bxl�	�  �          @�?��
�
=q@��
B�.C��=?��
���H@��HB-��C�:�                                    Bxl�^  T          @��?�G��{@���B��qC�,�?�G���33@��B)p�C�q�                                    Bxl�'  
�          @Ӆ?�z��Q�@�B��C�q�?�z���ff@��\B${C���                                    Bxl�5�  Z          @�?�(��0  @��Bw��C��?�(���  @��
B�C�*=                                    Bxl�DP  
�          @ָR?����I��@���BoG�C�u�?������\@x��B�RC�]q                                    Bxl�R�  
(          @�ff?�=q�5@��RB|
=C�q�?�=q���
@�BffC���                                    Bxl�a�  �          @Ӆ?Tz��3�
@�p�B~��C�=q?Tz���=q@���B�RC���                                    Bxl�pB  
�          @�  ?�p��6ff@�ffBy�HC��
?�p����@�p�B�\C�Ǯ                                    Bxl�~�  "          @���@��(��@���BrG�C�@�����@��RB��C�<)                                    Bxl���  �          @��@ ����R@�Q�Bx
=C�w
@ ������@�(�B G�C�U�                                    Bxl��4  T          @�G�?��R�)��@�ffBs�\C�xR?��R��p�@�Q�B��C�H                                    Bxl���  N          @���?��@  @��HBn{C��?���{@�Q�BQ�C��                                    Bxl���  �          @�?��\�c�
@��Ba(�C��?��\���\@aG�A���C�0�                                    Bxl��&  
�          @׮?�Q��j=q@���B\�C���?�Q����@^{A�p�C��\                                    Bxl���  "          @��
?����\)@��
BJ��C�(�?�������@=p�A�=qC�o\                                    Bxl��r  T          @��?��H��33@�33BH=qC�>�?��H���
@9��AΏ\C��
                                    Bxl��  �          @��?�(���p�@���B9{C��q?�(�����@\)A�(�C��q                                    Bxl��  �          @�ff?��H��33@�ffB2��C�W
?��H����@�A��C���                                    Bxl�d  �          @���?�G����@�B:p�C�ٚ?�G���z�@%A��C��                                    Bxl� 
  �          @�33?�p���G�@�\)B:z�C��
?�p��ƸR@'�A��C��)                                    Bxl�.�  
T          @�p�?��R��33@���B.p�C�(�?��R���@33A��C�y�                                    Bxl�=V  	�          @�?�=q����@�=qB%=qC�aH?�=q��\)@33A��
C�Ǯ                                    Bxl�K�  "          @�p�?��H���@��RB+�RC��?��H��@�RA�z�C�T{                                    Bxl�Z�  
�          @�\?�33��\)@��\B,G�C��?�33��G�@z�A�33C��                                    Bxl�iH  "          @�p�?��R���\@�{B1�C�'�?��R����@'
=A���C�/\                                    Bxl�w�  "          @�  ?�(�����@�  B1��C���?�(��ۅ@(��A���C�                                    Bxl���  T          @�G�?��R���@��\B4{C��?��R�ۅ@.�RA�  C�q                                    Bxl��:  
�          @��H?��H��{@��HB(G�C��H?��H���@Q�A�  C��
                                    Bxl���  �          @�33?�\)��Q�@�
=B.
=C��\?�\)��ff@%�A�G�C���                                    Bxl���  "          @�33?�{��z�@��\B2��C��3?�{��z�@/\)A�G�C��
                                    Bxl��,  T          @���?�33��=q@��BL��C���?�33��33@c�
A��
C�Q�                                    Bxl���  �          @��H?�(���ff@���BJp�C��3?�(�����@Z�HA�G�C��                                    Bxl��x  T          @�{?�(����@��HB7�C�"�?�(���=q@8��A��HC�U�                                    Bxl��  "          @�G�?����H@��RBffC�b�?���ff?��Ar�\C��3                                    Bxl���  
�          @�p�?�(�����@���B.G�C�~�?�(��ʏ\@��A���C��                                    Bxl�
j  �          @��?���G�@|��B
��C�4{?���p�?��A2=qC�                                    Bxl�  
�          @��H?�
=���@���B
p�C�3?�
=���?�\)A2{C��=                                    Bxl�'�  T          @�\?����@���BG�C��?���Q�@33A�
=C��                                     Bxl�6\  
�          @�
=?�z���\)@�=qB�C��?�z���z�@33A�=qC��                                    Bxl�E  "          @���?�Q�����@���BQ�C�S3?�Q����@z�A��\C�@                                     Bxl�S�  �          @�{?���@���B!{C��q?����
@	��A��C���                                    Bxl�bN  "          @�\@9���8Q�@�
=BO�C��
@9����  @g
=B  C���                                    Bxl�p�  
�          @��H@
�H�mp�@��
BRp�C�(�@
�H���@h��A�  C�y�                                    Bxl��  �          @��@6ff�7
=@�G�Ba
=C�}q@6ff����@�(�B(�C���                                    Bxl��@  �          @��@G��'
=@��B`z�C�f@G����H@�  B  C�l�                                    Bxl���  �          @�=q@K��0��@�
=B[=qC�}q@K���@��BffC�e                                    Bxl���  �          @�{@O\)�/\)@ÅB\��C��@O\)���R@�  B��C���                                    Bxl��2  T          @�=q@hQ��
=q@��RBZ�
C��q@hQ����
@�33B  C�ٚ                                    Bxl���  
�          @�33@e����@�Q�B[z�C�޸@e���
=@��B=qC�Q�                                    Bxl��~  
�          @�\@u����@��BTC�\)@u����@���B\)C��                                    Bxl��$  T          @�=q@w����@��BNQ�C�{@w����@��B  C�]q                                    Bxl���  
�          @�=q@mp��"�\@���BPG�C��@mp���(�@���BG�C�]q                                    Bxl�p  "          @�@����@�z�BE�HC��
@����\)@��RB
(�C�u�                                    Bxl�  
�          @��H@L(��8Q�@��BW�
C���@L(����@���BffC�G�                                    Bxl� �  �          @陚@p��\��@�p�BY�RC�Ǯ@p���Q�@�=qB{C��                                    Bxl�/b  	�          @�
=@#33�S�
@�(�BZQ�C��3@#33���@��HB
33C���                                    Bxl�>  
�          @��
?�Q����@�B2�
C���?�Q���Q�@5A��C�N                                    Bxl�L�  T          @�
=?�z���  @�G�B=  C�xR?�z���\)@EA���C���                                    Bxl�[T  T          @�
=@G���z�@�G�B3�
C�33@G��ҏ\@;�A��RC��f                                    Bxl�i�  
�          @�z�@���z�@�Q�B-�\C�9�@���G�@4z�A��RC��\                                    Bxl�x�  �          @�@  ���
@��B*�C�޸@  ��\)@/\)A�
=C�=q                                    Bxl��F  
�          @�33@�����@�\)B-��C�9�@���p�@5A�C�s3                                    Bxl���  
�          @���@  ��p�@�33B<��C�ٚ@  �Ϯ@U�A��HC��                                    Bxl���  �          @�33?�{��\)@�p�B�C�,�?�{�أ�?�
=Ax  C��                                    Bxl��8  "          @�z�?�\)��Q�@��RBp�C�˅?�\)�ҏ\@�
A��C��                                    Bxl���  	�          @�33?�\)��Q�@���B�C�>�?�\)�ȣ�?��RA��C�L�                                    Bxl�Є  �          @���?�G���33@�33B�HC�~�?�G���G�?�ffAv{C�˅                                    Bxl��*  
�          @�p�?�
=��@vffB{C��
?�
=�ȣ�?��AV{C�|)                                    Bxl���  �          @�Q�?�  ���@n{B�C�
=?�  ��(�?�{A9p�C��=                                    Bxl��v  �          @��H?�
=��\)@k�B�RC��f?�
=��\)?��
A,��C�Ff                                    Bxl�  �          @��?�
=��{@EA���C���?�
=��?\)@�C�3                                    Bxl��  �          @�ff?����H?��\A-G�C�o\?����H���\�-��C�o\                                    Bxl�(h  
�          @��H@
=���
�\)��\)C��=@
=����*�H����C���                                    Bxl�7  �          @�{@ ����
=�
=���C���@ ����  �A��ӅC���                                    Bxl�E�  
�          @�  ?��H��\)��  �\)C�(�?��H�5��  �s  C��f                                    Bxl�TZ  
�          @љ�?�\)�\)���R�FffC�t{?�\)��ff�Åp�C���                                    Bxl�c   T          @�33?�=q�p  ��{�R{C���?�=q���R��\)� C��R                                    Bxl�q�  �          @�\?���G����H��RC�AH?��O\)�����p��C���                                    Bxl��L  
�          @�z�?\(���G��l(���\)C��?\(������R�T��C�˅                                    Bxl���  �          @�z�?E���{�0�����\C�c�?E������  �4�C�o\                                    Bxl���  
�          @��H>Ǯ�ٙ�����v�\C��>Ǯ���H��=q�z�C�H                                    Bxl��>  �          @�G��8Q��ȣ��J=q�ՅC�*=�8Q������G��C��C��q                                    Bxl���  T          @�Q쾅��������
�G�C�������l����\)�l�
C��{                                    Bxl�Ɋ  "          @�
=�\��=q������C�  �\�hQ���Q��r�
C�                                      Bxl��0  
�          @�33�fff��G���\)�6Q�C��ÿfff�-p��׮#�C{��                                    Bxl���  
�          @��Ϳs33��{���
�;��C����s33�#�
�ڏ\z�Cy��                                    Bxl��|  "          @�R>\)��(��9����
=C���>\)��G������3�\C���                                    Bxl�"  
�          @陚?0����=q����P  C�˅?0����{��p��p�C�Q�                                    Bxl��  �          @�=q?(����=q��p��<��C���?(����Q������p�C�#�                                    Bxl�!n  T          @��?z�H�Ӆ�^{�ۙ�C�:�?z�H���H���D{C��q                                    Bxl�0  "          @��?�{�ᙚ��{�H��C�h�?�{��ff���
��C���                                    Bxl�>�  
�          @�(�@G���׾u��ffC��f@G�����:�H��z�C�c�                                    Bxl�M`  
.          @�
=@
=���8Q�����C���@
=����Y����  C��                                    Bxl�\  �          @�z�?�������ff�;�C�?����������
��
C�=q                                    Bxl�j�  
�          @��?�\�陚�������C�Ф?�\���H�tz�����C���                                    Bxl�yR  �          @�Q�@
=q��\)���H�RffC��R@
=q��33�����RC�q                                    Bxl���  �          @��@G���{�.{���C��q@G�����W��θRC�s3                                    Bxl���  �          @�ff?.{��p�?�\A\Q�C���?.{��=q�}p���ffC���                                    Bxl��D  T          @�?��\���H>�@aG�C���?��\�߮�G���p�C�*=                                    Bxl���  "          @�G�?�����?.{@�C�3?�����33�z��\)C�B�                                    Bxl�  
�          @�ff?�(���ff>��?���C��?�(��߮�%����C��                                    Bxl��6  �          @��?�ff��>��@AG�C��R?�ff�����H���C�H                                    Bxl���  T          @��
?������
�
=q�z�HC��
?��������QG���33C�b�                                    Bxl��  
�          @�?�33������U�C��?�33��G��O\)���C���                                    Bxl��(  S          @�z�?�\)����n{��=qC��?�\)�ٙ��i�����
C���                                    Bxl��  �          @�p�?�G���ff��\)��HC��H?�G���������HC�Q�                                    Bxl�t  T          @�z�?����
=��\)�33C��
?���ٙ��u��C�Z�                                    Bxl�)  
�          @�\)?����녿n{��
=C��{?����ff�l(��܏\C�Ff                                    Bxl�7�  
�          A Q�?�ff���ͿL����  C��f?�ff��\�fff��G�C�5�                                    Bxl�Ff  T          AG�?�G���ff�}p����C��q?�G�����q���G�C��                                    Bxl�U  
�          A�?���  �����P��C�N?��ҏ\��\)�=qC�                                    Bxl�c�  
�          @�{?^�R��33��t��C�@ ?^�R��33���z�C��                                    Bxl�rX  
�          @�{?Tz���׿����!p�C�%?Tz���G��|����33C��)                                    Bxl���  �          @�Q�?��\��G���  �2=qC�޸?��\��  ���H���RC�xR                                    Bxl���  �          @�
=?s33��Q���
�1p�C�y�?s33��ff��{��Q�C��                                    Bxl��J  "          AQ�?z�H�ff���\��\C�h�?z�H��Q��vff��  C��{                                    Bxl���  O          A��?�\)�33���R�\C�� ?�\)��  �W
=��
=C�<)                                    Bxl���  
�          A��?������ff�C�
C���?�������U���ffC��=                                    Bxl��<  T          A�\?�  ��=�G�?=p�C���?�  ��\)�3�
��  C�]q                                    Bxl���  T          A33?����H?L��@�C�z�?����{����k\)C���                                    Bxl��  T          A�H@(���{?��RA"ffC���@(����R��z����C���                                    Bxl��.  �          A�@1���p�?޸RA@��C���@1���G��������HC��{                                    Bxl��  
�          A
=@)�����?��
ABffC�aH@)����{��=q��33C�=q                                    Bxl�z  �          Az�@+���?�G�A#33C�U�@+���ff��\)�(�C�N                                    Bxl�"   
�          A  @#�
��p�?�z�A4��C��
@#�
�   ���H�\)C��H                                    Bxl�0�  �          Aff@  ��z�?�ffAF=qC���@  � zῈ����G�C���                                    Bxl�?l  
�          A
=@&ff��\)@
=Ah(�C�J=@&ff��\)�8Q���(�C��                                    Bxl�N  
Z          A��@=p���  @Ab{C�xR@=p�����=p���{C�.                                    Bxl�\�  �          A��@-p����@ ��AYG�C�� @-p����Y����p�C�Ff                                    Bxl�k^  
�          A	�@%���@Q�A�33C�*=@%�ff����J=qC��                                    Bxl�z  
�          A{@���(�@!�A�ffC��)@�� �׾�=q����C�33                                    Bxl���  �          A�\@G����
@z�Ak�C�q�@G�����0�����\C�8R                                    Bxl��P  �          A�R@ �����
@QG�A��\C��q@ �����?(�@��HC�33                                    Bxl���  
�          A�
@��ۅ@��HA�=qC�9�@���33?�p�A%�C�<)                                    Bxl���  �          A(�@   ��p�@��A�C�{@   ��z�?�Q�A ��C�                                      Bxl��B  T          A@\)���@o\)AܸRC�q@\)��?�
=A�HC�#�                                    Bxl���  
�          A
=@\)��=q@z�HA���C��@\)��  ?���A�C�3                                    Bxl���  T          @��?�(���
=@p  A�33C�.?�(���33?�p�A�C�aH                                    Bxl��4  �          @��@����p�@��B	Q�C���@���ᙚ@�Ay�C�]q                                    Bxl���  	�          @�\)?�Q����@�
=B��C��)?�Q���\)?�\)A`��C��H                                    Bxl��  T          @�
=@����=q@�{B�C�33@����(�?��Ab{C�Ǯ                                    Bxl�&  �          @��@#33���@�=qA�(�C�ff@#33��\?��
ATz�C��                                    Bxl�)�  "          @�{@
=���R@�z�B�C��)@
=���H@
=A\)C�E                                    Bxl�8r  
�          @�=q@z���G�@�
=B�\C��@z���G�@\)A��\C�                                      Bxl�G  T          @�ff@���
=@~�RA�p�C�h�@���ff?�33AF=qC�4{                                    Bxl�U�  
(          @�z�@$z���Q�@c�
A��
C�'�@$z����H?��RA�C��{                                    Bxl�dd  �          @��@&ff���
@W�AиRC��@&ff��(�?��
@�Q�C�                                    Bxl�s
  T          @�(�?�z���p�@��
BQ�C���?�z���{?���A]G�C��                                     Bxl���  
�          @��?�\)��33@���B
=C�  ?�\)��p�?�33Ac�C�K�                                    Bxl��V  �          @���?��
����@�p�B�\C���?��
��?�ffAUp�C��{                                    Bxl���  
�          @�z�@ ����z�@�z�A��
C��R@ �����?��
APz�C���                                    Bxl���  	�          @��@(���\)@�\)B�RC��=@(����
@
=qAz�RC�C�                                    Bxl��H  T          @�@G���z�@�Q�B�C��@G���z�@"�\A��C���                                    Bxl���  �          @�Q�@���=q@���B��C��@���G�@p�A�{C�e                                    Bxl�ٔ  
�          @��H@	����  @�B%  C��3@	������@Dz�A�  C���                                    Bxl��:  "          @���?�G����@���B5�C�h�?�G���{@aG�AׅC�Z�                                    Bxl���  �          @��R?�����p�@�{B833C���?�����Q�@j�HA�33C��=                                    Bxl��  "          @���?�\)���H@�33BCz�C�:�?�\)��  @{�A�G�C��q                                    Bxl�,  �          @��H?�(���p�@�p�BDG�C�Q�?�(����H@~�RA�{C�J=                                    Bxl�"�  
�          A{?:�H��Q�@�{B^�C��)?:�H��ff@��HB�HC�<)                                    Bxl�1x  �          @�p�>��H���
@�BXz�C��>��H��
=@��Bp�C�*=                                    Bxl�@  T          A�?������R@ٙ�B\��C��?�����@�
=B�C���                                    Bxl�N�  
�          A	@�s33@�
=Bi33C��\@��{@��HB$C��)                                    Bxl�]j  
�          A	�@'��l��@�Bf�HC��@'����\@��HB$�C�
                                    Bxl�l  T          A	��@
=�H��@�ffB\)C��=@
=��
=@�Q�B;�\C��f                                    Bxl�z�  �          A��?�(��1�@�(�B��\C�� ?�(���33@��BF(�C�Q�                                    Bxl��\  �          A=q?��333@�p�B�33C�8R?���(�@�33BD�
C�C�                                    Bxl��  T          A	?�  ��HAG�B�k�C���?�  ��z�@��
BSz�C��)                                    Bxl���  T          AG�@Q��!G�A�HB�C�#�@Q�����@�BO(�C��                                    Bxl��N  
�          A�
?�ff�(�A(�B�#�C���?�ff��\)@��
BZQ�C�e                                    Bxl���  
�          A
ff@/\)�	��@�p�B��C���@/\)��=q@��BOC�xR                                    Bxl�Қ  �          A{@aG���@�=qBw=qC���@aG����@�p�BCG�C��f                                    Bxl��@  
Z          A�\@r�\���@�Q�Br(�C��{@r�\���@���BA
=C��q                                    Bxl���  �          A�@^�R�J=q@��RB��3C���@^�R�R�\@�G�Bb�HC�S3                                    Bxl���  
�          A
=@;��+�AffB�u�C��@;��N�R@�Q�Bo\)C�q                                    Bxl�2  
�          A\)@/\)�z�A  B�z�C��@/\)�J�H@�(�Bt�C�g�                                    Bxl��  �          A
�\@*=q�#�
A\)B�(�C�4{@*=q�N{@�\Bt�RC�˅                                    Bxl�*~  
�          A\)@ff�.{A��B�L�C��{@ff�R�\@�ffBy=qC��=                                    Bxl�9$  
-          A
=@6ff�L��A�\B�#�C�/\@6ff�U@�Bn�\C�@                                     Bxl�G�  T          AQ�@<�Ϳ#�
A�
B�ǮC�� @<���Mp�@��
Bp�C�K�                                    Bxl�Vp  T          A\)@33�#�
A=qB�Q�C�@33�2�\@��B��fC��R                                    Bxl�e  
�          A(�@  �z�A�HB�.C���@  �L��@�=qB}�C��                                    Bxl�s�  "          A
�R@(�?�p�A(�B�33A��@(���A�B���C�y�                                    Bxl��b  "          Aff@z�@(�A��B��B-��@z��A	�B�  C��
                                    Bxl��  
�          A=q@{@�A�B���B2
=@{��A	p�B�8RC�P�                                    Bxl���  T          A�@��@�A�B�Bz�@�Ϳ#�
A33B���C��                                    Bxl��T  �          A�
@�R?�=qA�RB���BQ�@�R�O\)AG�B�.C��q                                    Bxl���  �          A33@=q?�\)@�
=B���A���@=q��
=A   B��C��q                                    Bxl�ˠ  T          Aff@��?���@��RB��A�{@�����@��B�\)C���                                    Bxl��F  
�          AG�@*=q?=p�@�\)B���Ax  @*=q��Q�@�\B�ǮC���                                    Bxl���  Y          A\)@33?�  @�33B�  A�\@33��z�@��
B���C���                                    Bxl���  
�          A{@  @*=q@陚B���BF��@  >�\)@�Q�B�k�@߮                                    Bxl�8  T          A
=?���@7
=@�B�ffB�\)?���>ǮA ��B�ǮA}p�                                    Bxl��  "          A�?L��@h��@��B|  B�=q?L��?��HA�B���Bb��                                    Bxl�#�  �          A��=u@���@�\Bj�B�L�=u?�A ��B�33B�33                                    Bxl�2*            @����@�\)@�Q�Bg
=B�=q��?���@�\)B���B�=q                                    Bxl�@�  
�          A
=<��
@��@��Be�HB��{<��
@�@�p�B��=B��                                    Bxl�Ov  	�          A   >\)@��@�(�BmB�
=>\)?�G�@�G�B�aHB��                                    Bxl�^  T          @�p�>�Q�@@��
B���B�p�>�Q쾅�@��B�aHC�)                                    Bxl�l�  	�          @���>W
=@|��@ڏ\Bo��B��)>W
=?�
=@�ffB�.B��                                    Bxl�{h  
�          @���>���@aG�@ᙚB}z�B�#�>���?��H@�G�B���B�33                                    Bxl��  �          @�p�>��H@p  @�ffBu�
B�(�>��H?�(�@�Q�B�z�B�#�                                    Bxl���  �          @���?
=@dz�@���B{{B�G�?
=?��\@���B��=B�z�                                    Bxl��Z  T          @��?\)@k�@��Bv�B��3?\)?�@�ffB���B��                                    Bxl��   T          @��>���@e�@�p�Bz�B���>���?�=q@�B�B�u�                                    Bxl�Ħ  �          @��>��
@^{@�=qB{��B�Q�>��
?�G�@�B�k�B��                                    Bxl��L  "          @�p�?n{@S33@�33B}��B��R?n{?��@���B�p�BG�                                    Bxl���  �          @�?c�
@^{@���Bx��B�B�?c�
?��
@�  B��{B\�R                                    Bxl��  "          @�{?�ff@W�@�\)Bt�B��?�ff?���@�B�aHB��                                    Bxl��>  
�          @��?^�R@L��@�G�B�RB��=?^�R?��
@�B�k�BG
=                                    Bxl��  
�          @�33?���@R�\@�Q�B{(�B�8R?���?�\)@�B��
B3�H                                    Bxl��  �          @�=q?n{@Mp�@���B33B���?n{?��@�B��3B@                                    Bxl�+0  T          @�?+�@L(�@�  B�aHB�=q?+�?��@�z�B��Be                                      Bxl�9�  T          @�\)����@{@�
=B���B�� ����>�Q�@���B���B��)                                    Bxl�H|  �          @��B�\?��R@�{B�p�B�uþB�\�5@陚B��\C�                                    Bxl�W"  T          @�33���
@��@�  B�  B��=���
���
@�=qB���CD33                                    Bxl�e�  �          @��>��@ ��@޸RB���B�ff>����@�  B��
C��3                                    Bxl�tn  �          @��?.{@8��@���B��
B�u�?.{?O\)@�RB�Q�BG�                                    Bxl��  �          @�Q�?�@_\)@��HBsffB��?�?�(�@��HB��HB���                                    Bxl���  �          @�>�\)@�=q@�  BMB��H>�\)@,(�@��HB��HB��f                                    Bxl��`  
�          @�?�33@P��@��BhffBo(�?�33?��@�  B���B
=                                    Bxl��  
�          @�?˅@p��@�z�B]G�B�#�?˅?�\)@׮B���BF�                                    Bxl���  "          @�?�p�@��\@�ffBO�B��{?�p�@�
@��B��=Bx\)                                    Bxl��R  
(          @�z�?�33@�p�@��BLG�B��?�33@�@��HB�
=B�B�                                    Bxl���  	.          @�ff?p��@�
=@�BN
=B��f?p��@p�@�B�� B�(�                                    Bxl��  �          @�G�?xQ�@��
@�{B-��B�z�?xQ�@^{@ƸRBo��B���                                    Bxl��D  �          @�{?��@<(�@�(�B{��B�8R?��?u@�ffB���B
�                                    Bxl��  �          @��
@G�?�p�@�=qB���B"33@G���  @�G�B���C��H                                    Bxl��  �          @���@.{?��@љ�B��HAE�@.{����@�{B��C�3                                    Bxl�$6  T          @�33@333?
=@�p�B�  A>=q@333���@��B�
=C��f                                    Bxl�2�  �          @�(�@��?�ff@�B��)A�{@�Ϳk�@�ffB���C��
                                    Bxl�A�  "          @�{@z�?�  @�{B�33B@z���H@��HB�z�C�e                                    Bxl�P(  
�          @��?�=q@p�@љ�B���BvQ�?�=q>�@�\)B��A���                                    Bxl�^�  �          @�@ff?��@���B�W
A�{@ff�+�@׮B�C�(�                                    Bxl�mt  �          @���@U��k�@�33Bq�
C�>�@U��(Q�@��\BS  C��)                                    Bxl�|  �          @�\)@HQ�B�\@�p�By�\C�'�@HQ��\)@�{B[�\C�                                    Bxl���  
�          @�ff@Tz�h��@�  Bp�\C�Y�@Tz��%�@�  BR=qC��                                    Bxl��f  �          @ۅ@B�\�p��@�=qByQ�C�\)@B�\�(Q�@���BXQ�C��                                    Bxl��  �          @��@2�\�\(�@�  B���C�w
@2�\�&ff@�  Ba�HC���                                    Bxl���  "          @�z�@3�
���R@���B}��C�{@3�
�;�@�G�BW�C��                                    Bxl��X  �          @��@k���=q@�\)BY=qC��H@k��XQ�@�
=B3p�C���                                    Bxl���  �          @��?��@,��@\BvQ�B\�
?��?aG�@�33B�#�A��                                    Bxl��  "          @��H?�33@{@��
B��BQ?�33?�@��B��A�(�                                    Bxl��J  
�          @���?\@'�@�ffB��Bo�\?\?E�@�B�u�A�=q                                    Bxl���  �          @ڏ\?�(�@&ff@�ffB�Q�B���?�(�?@  @�B�ǮA��\                                    Bxl��  T          @ᙚ@G�?8Q�@�(�B�z�A�
=@G���@��B��C�c�                                    Bxl�<  �          @��@p�>�@�ffB���A1@p���z�@��B�G�C�\                                    Bxl�+�  
�          @��
@
==�G�@ָRB�.@(Q�@
=��\@�
=B���C��\                                    Bxl�:�  
�          @�ff@�H>.{@�Q�B��@��
@�H��(�@�G�B�.C�L�                                    Bxl�I.  
�          @�Q�@-p�?333@�ffB��Ag�@-p�����@�(�B��HC��                                    Bxl�W�  "          @�R@"�\?�\@�\)B��A5�@"�\����@�33B�C���                                    Bxl�fz  
�          @��@Q�>�=q@�  B��@�(�@Q��\)@��B���C��                                    Bxl�u   �          @��@Q쾀  @��HB�C�@Q���
@�Q�B}�C��\                                    Bxl���  �          @�p�@��>Ǯ@θRB��AQ�@�ÿ�z�@��B��=C���                                    Bxl��l  �          @�{@��?G�@�ffB�L�A���@�����
@�p�B��qC�]q                                    Bxl��  "          @ۅ@녿(�@��HB�33C�xR@��@�Bt��C�q                                    Bxl���  T          @�(�@z῰��@ÅB���C�,�@z��@��@�  BaC�>�                                    Bxl��^  
�          @�p�@0  �Y��@�
=B}(�C�p�@0  ���@�Q�B]G�C��                                    Bxl��  �          @��H@E����@�p�Bp33C�t{@E�%�@��BP�C��                                    Bxl�۪  
V          @Ӆ@0  ?�R@�ffB�\)AL��@0  ���
@�z�B~(�C��{                                    Bxl��P  &          @Ϯ@#�
>�G�@�(�B�\)A�@#�
��
=@���B�  C���                                    Bxl���  
�          @�ff@ff?˅@�33B�B�B�@ff�\)@��B�{C�&f                                    Bxl��  "          @�p�?���@G�@��B�\)B=�H?���>�z�@�(�B�G�Ap�                                    Bxl�B  �          @�\)@z�@333@��RBe�BVp�@z�?��H@���B���A�R                                    Bxl�$�  �          @љ�?�p�@2�\@��\Bj
=BZ�?�p�?�z�@�(�B��HA�R                                    Bxl�3�  
�          @�  @�@3�
@�ffBd��BS�H@�?�(�@�Q�B��A�                                      Bxl�B4  
�          @�{@ff@Vff@��\BP�\Bg@ff?�@���B��\B%Q�                                    Bxl�P�  �          @ƸR@��@s33@�=qB1Q�Bl�
@��@\)@�{Bd33B>                                    Bxl�_�  "          @ȣ�@�H@�Q�@��B&��Bk��@�H@.�R@��HBYz�BB
=                                    Bxl�n&  �          @�ff@'�@�  @s33B�Bo(�@'�@S�
@�(�BD�BNff                                    Bxl�|�  �          @�G�@��@u@��RB8p�Bh�@��@�@�=qBi��B5�\                                    Bxl��r  
�          @�@�@P  @�p�BJ�BU
=@�?�ff@��
Bv�HB�\                                    Bxl��  �          @���@&ff@C�
@�ffBL  BF��@&ff?�{@��HBuQ�A��H                                    Bxl���  "          @�p�@(Q�@^�R@��RB<��BS�@(Q�@@�
=Bi�
B��                                    Bxl��d  "          @ʏ\@   @]p�@��B>�BXp�@   @�@�p�Bl
=B
=                                    Bxl��
  
�          @��H@Dz�@S�
@�=qB;ffB<�@Dz�?�33@���BcQ�A��                                    Bxl�԰  
�          @���@_\)@C�
@��\B2p�B%{@_\)?�p�@�\)BU\)AҸR                                    Bxl��V  T          @�  @^{@:�H@�z�B6��B �\@^{?�=q@�  BX(�A��                                    Bxl���  
�          @Ӆ@&ff@k�@���B;33B[G�@&ff@G�@��Bi�\B$                                    Bxl� �  �          @�@%@�G�@�ffB$��Bep�@%@1�@��
BU�
B<�                                    Bxl�H  
�          @���@�@�\)@��B"�B�33@�@L��@���BX��Bc�R                                    Bxl��  "          @ҏ\@(�@(Q�@��HBj  BH�
@(�?��@��HB�k�Aͮ                                    Bxl�,�  
�          @ָR?�@^�R@��BW��Bx�?�?�@�z�B�33B9�                                    Bxl�;:  �          @�{?���@���@�B?
=B�B�?���@%�@�=qBs�BS�                                    Bxl�I�  �          @�33?�p�@���@��B��B�\?�p�@b�\@�BO�Bs33                                    Bxl�X�  �          @�\)?�=q@�\)@mp�B��B��R?�=q@u�@�z�BDG�B���                                    Bxl�g,  "          @�\)?�z�@��H@�Q�BG�B��?�z�@g
=@�z�BQ{B���                                    Bxl�u�  
�          @�z�?�
=@�p�@S33A��B�
=?�
=@��
@�G�B6Q�B��                                    Bxl��x  �          @�{��\@
=@�=qB��B�(���\>�p�@���B�#�C��                                    Bxl��            @�(��#�
?��
@ʏ\B��3B�z�#�
=u@ҏ\B��
C.�                                    Bxl���  	�          @�  ��p�>u@�33B�k�C)\��p���  @�B�ffCf}q                                    Bxl��j  �          @��H��p���Q�@���B�L�CAٚ��p��G�@\B�=qCi�                                    Bxl��  "          @�G����þ�@��B��)CD\)�����@�
=B���Ci�                                    Bxl�Ͷ  �          @�����aG�@�B�  CQ�����p�@�\)B|�Cn                                      Bxl��\  T          @�ff�+���@��B��=CWٚ�+���\@�
=B��)C{�                                    Bxl��  
�          @���@,(�?��H@�G�Bg  B�@,(�>�{@�=qB|
=@�                                    Bxl���  T          @���@\(�@��@��RB>��A��\@\(�?h��@��BW  Al                                      Bxl�N  	`          @�  @:�H?�Q�@�Bb\)A�ff@:�H=�G�@�z�Bq�\@p�                                    Bxl��  
�          @�=q@�?J=q@�G�B�G�A�33@����@�=qB��C��=                                    Bxl�%�  "          @\�\)?aG�@��B�ffB��\)�   @�G�B�C]�\                                    Bxl�4@  
�          @�\)?��R@!�@�
=BiffBO=q?��R?��@�ffB��=A�z�                                    Bxl�B�  �          @��@��@:�H@�G�BT�BJ�@��?\@�(�B}  B                                    Bxl�Q�  
�          @��H@�H@A�@�Q�BP��BM��@�H?��@��
By��B�                                    Bxl�`2  T          @��@�@E@�  BS�B[�\@�?ٙ�@�(�Bz�B�                                    Bxl�n�  �          @�=q?�ff@Y��@�  BQ��Bx33?�ff@   @�ffB�u�B?��                                    Bxl�}~  T          @�=q?��H@~�R@�G�B;�
B���?��H@+�@��Br33Buff                                    Bxl��$  
�          @��?p��@��@�(�B \)B���?p��@e@�
=BZ\)B��\                                    Bxl���  �          @�ff?�@���@y��B  B��R?�@h��@�  BM33B��
                                    Bxl��p  T          @�\)?��
@���@�p�B �B�\?��
@c33@�  BYQ�B�k�                                    Bxl��  �          @У�?��@��@�p�B,{B�L�?��@Tz�@�ffBe(�B���                                    Bxl�Ƽ  
�          @��H?�p�@Mp�@�33BT=qBv�R?�p�?�{@�  B�ffB<�                                    Bxl��b  X          @��@��@   @�
=Bt33B��@��>�@�G�B�{A/�                                    Bxl��  �          @љ�@��?�33@��B~ffB#G�@��>�33@���B�.A\)                                    Bxl��  "          @љ�@�
@�@��B}�
B5ff@�
?�@�ffB�.Adz�                                    Bxl�T  T          @љ�?��H@�R@��HB|�BBff?��H?+�@ƸRB�
=A�33                                    Bxl��  "          @љ�?�(�@��@�ffB�z�BM{?�(�?\)@ə�B�  A���                                    Bxl��  �          @У�?�\?�G�@���B��HB3ff?�\>8Q�@ȣ�B�B�@�=q                                    Bxl�-F  �          @Ϯ?�=q?�{@�(�B�p�B"�
?�=q�u@���B�k�C��                                     Bxl�;�  �          @�{@
=q?Ǯ@�33B���B
=@
=q<�@���B���?E�                                    Bxl�J�  "          @�ff?}p�?n{@ə�B�ǮB,�\?}p��333@ʏ\B�ffC�c�                                    Bxl�Y8  
�          @θR?B�\?O\)@˅B�.B<ff?B�\�Q�@˅B�#�C�ff                                    Bxl�g�  
�          @ȣ�?�?p��@���B�{Bsp�?��#�
@�{B���C�Ǯ                                    Bxl�v�  P          @�\)>8Q�?�ff@\B��=B���>8Q쾊=q@ƸRB�C�>�                                    Bxl��*  
�          @�  >��þB�\@ƸRB�C��>��ÿ޸R@�
=B��)C�T{                                    Bxl���  T          @ȣ׾8Q�>��@�ffB��RB���8Q쿌��@��
B�L�C�XR                                    Bxl��v  T          @�z�?
=q@1�@�z�B}��B�(�?
=q?��\@��B�B�Q�                                    Bxl��  
�          @�z�?�p�@�p�@��\B:  B�=q?�p�@8��@�\)Bp��B�                                    Bxl���  T          @У׽��
@J�H@�{Bs\)B�
=���
?�33@��B���B��
                                    Bxl��h  �          @У׾k�@#�
@�
=B�ffB�k��k�?}p�@�p�B�{BΨ�                                    Bxl��  
�          @�G��\)@E@�{Bqz�B�\�\)?У�@�G�B���B��=                                    Bxl��  �          @��H�L��@@�ffB�G�B��)�L��?
=q@�G�B�#�B�\                                    Bxl��Z  �          @�\)�z�?��@ʏ\B�\)B���z���@�z�B�ǮC_�                                    Bxl�	   	�          @�z��>���@�33B�\)C�{�녿��@�\)B�aHCv�                                    Bxl��  P          @ʏ\��G���\)@���B��
C7.��G�����@�{B�.CgǮ                                    Bxl�&L  
�          @�G�@\)@j�H@��B3=qB_(�@\)@p�@�B_��B2Q�                                    Bxl�4�  
�          @�33@P��@�G�@j=qB�RBLff@P��@@  @���B6�B*�                                    Bxl�C�  �          @ȣ�@L��@g�@�Q�B�BB33@L��@ ��@���BFp�B�                                    Bxl�R>  
Z          @���@E@P��@���B1�RB:33@E@z�@�=qBW�B(�                                    Bxl�`�  
�          @ə�@=q@L(�@�33BJ  BS��@=q?��@��BsffB�                                    Bxl�o�  �          @�=q@ ��@>{@�B]\)B_�H@ ��?���@�Q�B�  B                                      Bxl�~0  "          @�@Q�@C33@�  BZB\\)@Q�?�z�@��HB�k�B��                                    Bxl���  "          @���@�H@Z=q@��BC�BZz�@�H@�@�  Bn�RB$�                                    Bxl��|  
�          @��@(�@@  @��\BRBK��@(�?�z�@��By�B��                                    Bxl��"  T          @��@��@0��@���BZ�BDQ�@��?�z�@�p�B��A�                                    Bxl���  
�          @�{@0  >�Q�@�Br�\@�{@0  �^�R@��
Bmp�C�AH                                    Bxl��n  
�          @���?������H@��\Bz{C�l�?����N{@�BL�
C��                                    Bxl��  �          @�  @7��h��@��
Bn{C�,�@7����@��BS�\C��                                     Bxl��  �          @�z�@9���&ff@���Br�HC��=@9����Q�@�\)B\  C�%                                    Bxl��`  "          @���@"�\?&ff@�\)Bt
=Ad��@"�\��@�  BuQ�C��                                    Bxl�  
�          @�  ?�(�@Mp�@��BQ��Bw�?�(�?���@�(�B�ffBB�                                    Bxl��  
�          @��
?�z�?�  @���B���B*
=?�z�>�{@�G�B���A"=q                                    Bxl�R  "          @�p�@7�?(��@�p�BlffAP(�@7��\)@�{Bm\)C�u�                                    Bxl�-�  "          @�G�@   ?���@��Bk33B��@   ?�\@���B�Q�A8z�                                    Bxl�<�  
�          @���@4z�@B�\@�\)B6\)B<�@4z�?�z�@��HB[��B�                                    Bxl�KD  
�          @�  @b�\@�(�@�HA��BE�@b�\@Z�H@W
=B	(�B/��                                    Bxl�Y�  
,          @�=q@'
=@o\)@~{B$G�B\�@'
=@*=q@���BP
=B633                                    Bxl�h�  �          @\@��@G�@�Q�BN{B^  @��?�{@��
Bx��B$                                      Bxl�w6  
�          @���?�(�@��@��Bo{BJQ�?�(�?��\@�\)B��qA�(�                                    Bxl���  	�          @���@(Q�@p�@��BU�HB,\)@(Q�?���@��Bu��A�z�                                    Bxl���  
(          @�z�@#�
@   @�Bg��B��@#�
?.{@�Q�B���Al(�                                    Bxl��(  
�          @�@(�@ ��@�=qBnG�B*33@(�?8Q�@��B�A�\)                                    Bxl���  �          @�z�@#33?�=q@��\Bu��A�ff@#33�8Q�@�{B33C��                                    Bxl��t  �          @�{@$z�?fff@�Q�Bzz�A�33@$z����@�=qB�
C��3                                    Bxl��  
�          @��R@333@ff@��\BM�\B��@333?�@�Q�Bk(�A��R                                    Bxl���  !          @�@�@'�@��BV(�BA(�@�?��@��Bzp�A�                                      Bxl��f  �          @�@�R@-p�@��\BR(�BJ{@�R?\@�33Bxz�B�H                                    Bxl��  
�          @���?�=q@_\)@�G�BL�B�Ǯ?�=q@�\@�  B�� B��3                                    Bxl�	�  
�          @�\)?E�@(�@��B�� B���?E�?�=q@��B��BZ=q                                    Bxl�X  W          @�(�?�z�@%@���By{B�Ǯ?�z�?��H@��
B�(�B8��                                    Bxl�&�  T          @�=q?�Q�@'
=@�(�Bk=qBd33?�Q�?��@��B��BG�                                    Bxl�5�  �          @�33?�ff@S�
@���BX=qB�(�?�ff@�@��B�.Beff                                    Bxl�DJ  Q          @���@   @_\)@�z�B=��Bp��@   @�@��Bk�HBE{                                    Bxl�R�  
�          @�@G�@6ff@��BNffBZ�
@G�?ٙ�@�
=Bw��B ��                                    Bxl�a�  
�          @�?��@���@r�\B�HB��H?��@W
=@��BU�RB��f                                    Bxl�p<  %          @��H?�\@�ff@�  B-z�B�33?�\@G
=@�z�Bd��B�p�                                    Bxl�~�  "          @�z�>�G�@�z�@�33B2G�B���>�G�@A�@�
=Bi�B���                                    Bxl���  	�          @�?�\@��H@U�B	ffB��)?�\@y��@���B@��B���                                    Bxl��.  �          @���?�
=@�@��A�z�B�8R?�
=@�  @X��B�RB���                                    Bxl���  T          @�=q?�(�@{�@b�\B(�B}�?�(�@>{@�(�BK�Ba�\                                    Bxl��z  �          @���@p�?���@�  B��HA�(�@p��L��@��B�L�C�ff                                    Bxl��   �          @�@�\@��@���Bm��B,�@�\?L��@��B��{A�{                                    Bxl���  
�          @�@*=q?��
@�Q�Bj�
B�@*=q>�@�G�B�A��                                    Bxl��l  �          @��@)��?�G�@�ffBz
=A���@)������@�G�B�u�C�p�                                    Bxl��  "          @���@ff?B�\@�
=B�G�A�=q@ff�#�
@�\)B�C���                                    Bxl��  
�          @�33?��?��@�p�B�k�BX��?��>\@�{B�k�A�                                      Bxl�^  �          @��H@�
?�  @��RBx��B33@�
>W
=@�p�B��3@��
                                    Bxl�   T          @�ff?��R@��@���Bn(�BI��?��R?��@�
=B�
=A��                                    Bxl�.�  	�          @�  ?���@	��@��B���Bg��?���?:�H@�Q�B�W
A�                                    Bxl�=P  	�          @�G�?���@#�
@�=qBy�RBv�?���?�33@���B�B                                      Bxl�K�  T          @�=q>�Q�@7
=@��\Bvp�B���>�Q�?�  @��B��\B��                                    Bxl�Z�  �          @�33?�ff@p�@�=qBuG�Bgz�?�ff?�\)@�  B�L�Bp�                                    Bxl�iB  
�          @�{@6ff?�\)@���Bl�RA�@6ff=u@�
=Bz  ?��\                                    Bxl�w�  
�          @�
=@  ?�p�@�  B�Bz�@  =�@�ffB�Ǯ@H��                                    Bxl���  
�          @�\)@G�?Tz�@�  Bi��Anff@G���ff@���Bl�C��                                    Bxl��4  �          @ƸR@\��?p��@�=qB[{At  @\�;���@�z�B`(�C���                                    Bxl���  �          @�{@e>�ff@�Q�BX�@�(�@e�B�\@�
=BV�C�                                    Bxl���  
�          @�@X��?��@�(�B`�
A��@X�ÿ0��@��B_C�AH                                    Bxl��&  
�          @�p�@U>��
@�p�Bd  @�\)@U�k�@�33B_{C�E                                    Bxl���  T          @��@k��B�\@�p�BT��C��@k���{@�\)BIQ�C���                                    Bxl��r  "          @�@p�׾���@��
BP�C���@p�׿�ff@�(�BB��C��                                    Bxl��  
Z          @��@fff���@�G�BR��C�@ @fff��p�@�Q�BA�
C�+�                                    Bxl���  
�          @��@Y��>\)@��
Ba��@�@Y�����@�Q�BY��C��                                    Bxl�
d  �          @�z�@2�\��G�@�\)Bq��C��@2�\��R@��HBVQ�C���                                    Bxl�
  T          @�G�@-p���Q�@��\Bk\)C���@-p��'�@��\BJp�C�                                      Bxl�'�  �          @�ff@U����@�ffBS
=C�b�@U��Q�@�Q�B8z�C�:�                                    Bxl�6V  �          @�ff@\)�Ǯ@�z�Bp�RC���@\)�0  @��BL��C�                                      Bxl�D�  
�          @�ff@'��aG�@�  Bx�C��f@'��ff@�z�B]��C���                                    Bxl�S�  �          @���@A��(�@�
=BC�C��3@A��\(�@p��Bz�C��3                                    Bxl�bH  
�          @��@1녿��@��
BQ��C��@1��8Q�@s33B.�C��                                    Bxl�p�  �          @�ff?�p��h��@}p�B.C�Ff?�p���\)@>�RA�=qC���                                    Bxl��  �          @�{@��#�
@�G�B=ffC��)@���\)@���B5�
C��                                     Bxl��:  �          @�\)@�{<�@��HB>G�>\@�{����@��RB7�C��)                                    Bxl���  "          @�{@��H?
=q@��HB3
=@�z�@��H��\@��HB3G�C���                                    Bxl���  �          @���@��>�Q�@�ffB=�
@��@�녿5@���B;z�C�
=                                    Bxl��,  T          @���@�\)?��R@z�HB$�A�33@�\)?�\@��B1\)@���                                    Bxl���  �          @��@�Q�?z�@uB ��@�(�@�Q쾳33@w�B"
=C�˅                                    Bxl��x  �          @�(�@�=q>k�@�=qB&�\@5@�=q�@  @�  B#ffC�W
                                    Bxl��  "          @�
=@�G��B�\@y��Bp�C��R@�G���\)@o\)B��C�p�                                    Bxl���  �          @�Q�@��׿�(�@^{B��C�,�@���� ��@E�A�p�C�{                                    Bxl�j  �          @Å@�  ��G�@aG�B  C�*=@�  ��
@C�
A�RC��                                    Bxl�  
�          @ƸR@�G���@E�A��
C���@�G��&ff@!�A�z�C�W
                                    Bxl� �  �          @ȣ�@�p��@N�RA��C�z�@�p��333@(��A�=qC�(�                                    Bxl�/\  "          @�  @��
��=q@\��B(�C��
@��
�&ff@:=qAޏ\C��
                                    Bxl�>  
�          @�
=@�  ���@HQ�A��C�XR@�  �HQ�@��A���C�Q�                                    Bxl�L�  �          @��@�ff�Q�@S33B��C��@�ff�Fff@(��AυC���                                    Bxl�[N  
�          @�=q@�{�   @B�\A��
C���@�{�I��@ffA���C��)                                    Bxl�i�  �          @�z�@�ff�
=@.�RA�C�:�@�ff�<��@A�p�C��)                                    Bxl�x�  
�          @��H@��ÿ�  @8��A�C�c�@������@Q�A�\)C�H�                                    Bxl��@  "          @���@����Q�@AG�A�C��R@���
=@%A�z�C���                                    Bxl���  	�          @��H@����Q�@J�HA�\)C��R@���	��@/\)A��
C�U�                                    Bxl���  �          @�  @�{���R@C�
A�G�C��R@�{��@'
=AθRC�R                                    Bxl��2  "          @�\)@�33��@J�HB z�C��)@�33�Q�@/\)A���C�!H                                    Bxl���  �          @�Q�@�p����R@@��A��HC���@�p��
=q@$z�A�ffC�&f                                    Bxl��~  "          @�G�@�(��.{@"�\A�z�C�@�(��P  ?�A�ffC�t{                                    Bxl��$  
�          @�Q�@�ff�5�?@  @�C��{@�ff�;�=L��?�\C�K�                                    Bxl���  "          @�(�@�p��9��?\AfffC�^�@�p��K�?J=q@�(�C�8R                                    Bxl��p  �          @�@���@��?�  Aa�C��q@���Q�?=p�@���C���                                    Bxl�  �          @ʏ\@����dz�?�=qAB{C�^�@����q�>�G�@}p�C��3                                    Bxl��  
�          @�33@�  �Mp�?�ffAb�RC�AH@�  �_\)?=p�@�{C�,�                                    Bxl�(b  
�          @ə�@��<��?�p�A4��C���@��J=q>��H@�\)C��                                    Bxl�7  �          @˅@����)��?�AtQ�C�.@����>�R?}p�A\)C�ٚ                                    Bxl�E�  "          @�(�@����/\)?ǮAa�C�ٚ@����A�?\(�@��RC���                                    Bxl�TT  �          @�(�@���(Q�?\A\(�C�u�@���:=q?W
=@�33C�G�                                    Bxl�b�  �          @�33@��H�)��?�33AK�C�W
@��H�9��?8Q�@�G�C�G�                                    Bxl�q�  �          @ʏ\@�p��$z�?�\)A#33C��{@�p��0��>��@�G�C�f                                    Bxl��F  "          @���@���+�?���A>{C�T{@���:�H?#�
@��RC�]q                                    Bxl���  T          @�@�
=�(Q�?�(�A/
=C��f@�
=�6ff?��@�C��                                    Bxl���  �          @���@�G��0��?�=qAd��C�� @�G��C�
?^�R@�=qC���                                    Bxl��8  T          @˅@�(��7�?�ffA���C���@�(��N{?��AG�C��=                                    Bxl���  �          @�=q@���%?���AT��C��H@���7�?J=q@�p�C�`                                     Bxl�Ʉ  
�          @��H@�33�&ff?�
=AQp�C��\@�33�7
=?E�@�\)C�s3                                    Bxl��*  T          @���@�  �p�?��AAG�C�k�@�  �-p�?5@˅C�c�                                    Bxl���  �          @�(�@����)��?��AB�RC�q�@����8��?+�@���C�p�                                    Bxl��v  �          @˅@�\)�7�?���AAp�C�.@�\)�Fff?��@��C�<)                                    Bxl�  T          @�p�@�=q�G�?��A��C���@�=q�]p�?}p�AC��                                     Bxl��  �          @�  @��
�N{?�G�A{�C���@��
�c33?p��A��C�J=                                    Bxl�!h  
Z          @ҏ\@��\�K�?�Q�AI�C�*=@��\�[�?!G�@��C�4{                                    Bxl�0  �          @ҏ\@�  �QG�?��
AW\)C���@�  �b�\?333@�33C��)                                    Bxl�>�  �          @ҏ\@��G
=?�33A!�C���@��R�\>�p�@L��C��3                                    Bxl�MZ  
�          @љ�@��R�E�?Y��@�
=C�Ф@��R�L��=��
?:�HC�`                                     Bxl�\   T          @�  @���G
=?333@�p�C��)@���L(���\)�#�
C�O\                                    Bxl�j�  
�          @��@���S33?Q�@��C��f@���Y��    ���
C�ff                                    Bxl�yL  �          @�=q@��H�X��?.{@�
=C�e@��H�\�;�����C�'�                                    Bxl���  �          @љ�@���Z�H>�@�  C�33@���\(���Q��G�C�#�                                    Bxl���  �          @���@��R�a�?\)@�p�C��{@��R�c�
���R�-p�C�w
                                    Bxl��>  "          @�  @�
=�\��>\@W�C��q@�
=�\(�������
C��                                    Bxl���  
�          @θR@�=q�J=q?G�@�{C�0�@�=q�P�׼#�
���
C��3                                    Bxl�  �          @�ff@����QG�>L��?޸RC�@����N{�����33C��3                                    Bxl��0  "          @�{@�z��E>�?���C��=@�z��A녿�R��  C��f                                    Bxl���  "          @�@��
�G
=�#�
���RC���@��
�@�׿J=q��G�C���                                    Bxl��|  �          @�{@���C�
=#�
>�Q�C�˅@���>�R�333����C��                                    Bxl��"  �          @��@�p��?\)�L�;�G�C�)@�p��8Q�E���p�C���                                    Bxl��  �          @�@����G
=�#�
�ǮC��R@����@  �L����33C�                                      Bxl�n  "          @�p�@�ff�:�H��=q�=qC�n@�ff�1G��xQ���C��                                    Bxl�)  
�          @θR@��H�|(�=���?n{C�)@��H�vff�\(����C�p�                                    Bxl�7�  
�          @�z�@�=q���?���A
=C��{@�=q��{�#�
�\C�L�                                    Bxl�F`  T          @�(�@��\���?5@�C�1�@��\�����p��UC�                                    Bxl�U  �          @���@������?h��A\)C��@����z�aG����RC�Ǯ                                    Bxl�c�  �          @�z�@�ff��\)?L��@�
=C�@�ff������  ��\C��=                                    Bxl�rR  
�          @�33@��H�j�H?333@�33C�)@��H�n�R�aG���p�C���                                    Bxl���  �          @�=q@�\)�h��?�z�A*�RC��@�\)�s�
>u@�C�G�                                    Bxl���  �          @�=q@�Q��G
=?ǮAep�C��3@�Q��Y��?B�\@��
C���                                    Bxl��D  "          @���@����`��?�APz�C���@����p  ?�@��C��3                                    Bxl���  �          @��@�  �\(�?��\A6�HC�` @�  �i��>Ǯ@^{C���                                    Bxl���  �          @�{@�33�^{?^�R@���C���@�33�e��#�
�uC��                                    Bxl��6  "          @�@�{�Vff>��@hQ�C�.@�{�Vff��(��uC�0�                                    Bxl���  T          @�z�@�33�n{?�@���C���@�33�p  ��Q��Q�C��{                                    Bxl��  
Z          @��
@���fff?�\)AF�RC�J=@���tz�>�G�@\)C�u�                                    Bxl��(  �          @���@�
=����?p��A	G�C�� @�
=��(����
�B�\C�aH                                    Bxl��  T          @��@�\)��(�?z�@���C�q�@�\)��z��ff��(�C�c�                                    Bxl�t  
�          @�33@�\)�w
=?!G�@��C�
@�\)�y����{�C33C���                                    Bxl�"  �          @˅@�Q��G
=?(�@�G�C�G�@�Q��J=q�.{���C�\                                    Bxl�0�  
�          @�(�@����G�>��@��RC�E@����H�þ��
�6ffC�33                                    Bxl�?f  �          @��@�G��e>�@��C��\@�G��e��G��x��C��                                    Bxl�N  �          @��@�\)�k�>\@[�C�o\@�\)�j=q�
=q��=qC��H                                    Bxl�\�  �          @�@���dz�>���@fffC��@���c�
���H���C��                                    Bxl�kX  �          @љ�@�
=� ��?�G�AG�C���@�
=�+�>�33@EC��                                    Bxl�y�  �          @�  @���(Q�?k�A�RC��@���1G�>u@	��C�XR                                    Bxl���  2          @���@�
=�E�?!G�@��\C���@�
=�H�þ������C���                                    Bxl��J  �          @�(�@��0��?Tz�@�  C�@��8Q�=�?��C���                                    Bxl���  �          @ʏ\@�z��*�H?Q�@�ffC�S3@�z��2�\>�?�33C���                                    Bxl���  �          @�33@���mp������l(�C�� @���_\)����@��C���                                    Bxl��<  �          @ʏ\@�z��hQ��p���(�C���@�z��>{�B�\��33C��3                                    Bxl���  �          @�33@��\�l(�������C���@��\�>�R�N{��C�u�                                    Bxl���  �          @��
@���l(��ff���C��@���?\)�L����C���                                    Bxl��.  �          @�33@�z��x�ÿ����(�C��@�z��R�\�2�\��33C�L�                                    Bxl���  �          @˅@������
��Q��w�C�޸@����c�
�*�H����C���                                    Bxl�z  T          @˅@�
=�|�Ϳ�=q�f=qC�@�
=�[��!G�����C�                                    Bxl�   �          @�33@�33���Ϳ�33�K�C�T{@�33�z=q�p���=qC�                                    Bxl�)�  �          @˅@�������\(���G�C��R@���l�Ϳ�����\)C�7
                                    Bxl�8l  �          @˅@�(���Q����C�O\@�(��o\)�Ǯ�c�
C�J=                                    Bxl�G  �          @˅@��
���׿+�����C�7
@��
�n�R��z��qC�G�                                    Bxl�U�  �          @ə�@����{�������C�L�@����b�\�   ��=qC���                                    Bxl�d^  �          @�Q�@����fff�8Q����HC�B�@����S�
�����m��C�g�                                    Bxl�s  �          @Ǯ@�p��$z����  C���@�p���R�+��ƸRC�33                                    Bxl���  �          @�  @�=q�5��B�\��  C��=@�=q�,(��fff�Q�C�q                                    Bxl��P  �          @�Q�@����:=q��p��[�C�q@����.{�����!��C���                                    Bxl���  �          @�\)@����P�׿�R��ffC�#�@����?\)���S33C�1�                                    Bxl���  �          @�Q�@�33�x�þ����Dz�C���@�33�k�����B{C�e                                    Bxl��B  �          @�@���\(��xQ��G�C��
@���E������C�'�                                    Bxl���  �          @�@���l�Ϳ   ��p�C��H@���\�Ϳ�
=�UC�s3                                    Bxl�َ  �          @�{@���~{�G���\)C�Ф@���h�ÿ�G���z�C�                                    Bxl��4  �          @��@�\)�k���33�Q�C�
=@�\)�Mp���\���\C���                                    Bxl���  �          @Ǯ@�z�� �׿�
=�{
=C�/\@�z�\�������C�y�                                    Bxl��  �          @�\)@�ff�#�
��Q��W
=C�h�@�ff�
=�33��=qC�h�                                    Bxl�&  �          @�ff@��\���\�J=q�陚C�(�@��\�p  ��ff��p�C�^�                                    Bxl�"�  
�          @�@��N�R�h���Q�C�  @��8�ÿ��H����C�e                                    Bxl�1r  �          @�p�@��R�b�\�u��C�C�@��R�J�H��=q��ffC��{                                    Bxl�@  �          @�{@�=q�\�Ϳ��{�C�4{@�=q�:=q�   ��G�C�t{                                    Bxl�N�  �          @�@���Tz῕�/�C�]q@���:=q���R��ffC�\                                    Bxl�]d  �          @�z�@����6ff��z��/
=C��@�����Ϳ�{����C��
                                    Bxl�l
  �          @�z�@��� �׿�  �=�C���@���
=��{��  C�^�                                    Bxl�z�  �          @Å@��H�Q�����p��C���@��H��33�Q����C�33                                    Bxl��V  �          @�z�@���}p��Y��� (�C��@���g
=������C��{                                    Bxl���  �          @ƸR@~{���R�&ff��  C�3@~{��(�����Q�C�R                                    Bxl���  �          @�
=@�z�����\)��z�C��R@�z���=q���x��C���                                    Bxl��H  �          @�\)@�  ����O\)��ffC�` @�  �xQ�����(�C��q                                    Bxl���  �          @�=q@n�R��녾k��C�33@n�R���\���
�a��C�ٚ                                    Bxl�Ҕ  �          @��H@������
����k�C��@�����33��z��r�RC���                                    Bxl��:  �          @�33@`�����׾�p��U�C��
@`�������p��|z�C��\                                    Bxl���  �          @ʏ\@g����;������C��\@g���33��ff��=qC�`                                     Bxl���  �          @��H@u���  �!G���p�C��q@u���p���z���(�C���                                    Bxl�,  �          @�=q@i�����\�W
=����C�ٚ@i����{�����=qC��
                                    Bxl��  �          @ʏ\@������=L��>��C���@����������/33C��                                    Bxl�*x  �          @�33@|(�����.{��{C�e@|(���녿�Q����HC�o\                                    Bxl�9  �          @��H@qG���ff���*�\C���@qG���
=����Q�C�
=                                    Bxl�G�  �          @�=q@c�
��p���\��=qC��
@c�
�����@  ��z�C��                                    Bxl�Vj  �          @�\)@g
=��33�����C��@g
=�x���P����z�C�o\                                    Bxl�e  �          @��H@Mp���p��+�����C���@Mp��c�
�p  �
=C�                                    Bxl�s�  �          @��
@,���a����
�+G�C���@,���z�����X
=C��=                                    Bxl��\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxl��  ,          @��@g
=�����#33��\)C��@g
=�^{�e���C�3                                    Bxl���  h          @���@O\)���\�5�ݮC�j=@O\)�Z�H�xQ���C��R                                    Bxl��N  �          @�@Mp����H�$z���\)C�}q@Mp��o\)�l(��\)C�N                                    Bxl���  �          @�{@;����
�(����C���@;������h���p�C��                                    Bxl�˚  �          @�ff@;������p���{C��@;������B�\��C�w
                                    Bxl��@  �          @�\)@i������������C�p�@i���p���XQ��Q�C��                                    Bxl���  �          @ƸR@z�H��\)������C�j=@z�H�^{�S�
��C�:�                                    Bxl���  �          @�\)@j�H���������C�k�@j�H�tz��P������C��                                    Bxl�2  �          @�\)@r�\���\�}p��=qC��@r�\��z��\)���C�g�                                    Bxl��  �          @���@qG���p��z����
C��
@qG����H�����ffC���                                    Bxl�#~  �          @�G�@o\)���}p����C��
@o\)����G����C���                                    Bxl�2$  �          @�Q�@k���{���\�{C�T{@k������
��p�C���                                    Bxl�@�  �          @���@[������33��C���@[�����Fff��=qC�)                                    Bxl�Op  �          @�p�@Q���(�������C��@Q����R�E��ffC��
                                    Bxl�^  �          @\@�G����Ϳ�p���\)C�)@�G��a��2�\��C�l�                                    Bxl�l�  �          @\@���{���\)��G�C��
@���Q��7
=���\C�%                                    Bxl�{b  �          @�Q�@�z��z=q��p��<z�C�#�@�z��[��\)����C��R                                    Bxl��  �          @���@���Z=q�(����33C���@���G
=�Ǯ�q�C��{                                    Bxl���  �          @Å@�=q��p�������RC�&f@�=q�a��6ff��\)C���                                    Bxl��T  �          @��
@mp��l(��G���(�C���@mp��-p���  �$=qC��{                                    Bxl���  �          @�G�@}p��o\)�*=q���
C�Q�@}p��8Q��e��z�C�                                      Bxl�Ġ  �          @�  @�=q�`  �����x��C�E@�=q�;��\)��\)C���                                    Bxl��F  �          @��@p  ��p���p��@z�C�(�@p  �z�H������\C��                                    Bxl���  �          @���@_\)����=�G�?�  C�e@_\)������\�?�C���                                    Bxl��  �          @�z�@W����\>k�@��C�˅@W���ff��z��0  C�#�                                    Bxl��8  �          @�(�@X�����\�\�c33C�޸@X�����ÿ�G���\)C���                                    Bxl��  �          @��@S�
��(��333��G�C�h�@S�
�������ffC�t{                                    Bxl��  �          @���@U���(��z����
C�� @U����׿��H���C�s3                                    Bxl�+*  �          @�z�@Q������s33��C��H@Q���33��
���C��                                    Bxl�9�  �          @�(�@]p������  �d��C��@]p���z��0  �׮C��\                                    Bxl�Hv  �          @�p�@g
=��\)��=q�Ip�C���@g
=���%���z�C�n                                    Bxl�W  �          @�p�@j�H����G���G�C��f@j�H���R���p�C��)                                    Bxl�e�  �          @�ff@Z=q���׿����)�C�{@Z=q��Q���R��p�C���                                    Bxl�th  �          @�ff@Z=q������  ���C���@Z=q���\����(�C�U�                                    Bxl��  �          @Å@(����=q�:�H��  C��{@(���U��Q��-(�C�(�                                    Bxl���  �          @�=q@\)����P  �C���@\)�U���33�9�C�c�                                    Bxl��Z  �          @���@~{�q��$z����HC�7
@~{�:=q�aG��=qC��                                     Bxl��   �          @\@������\���R���HC�U�@����W
=�C33��
=C�R                                    Bxl���  �          @��@���y������k�
C��@���S�
�$z�����C�e                                    Bxl��L  �          @���@�p���33��(��b�\C���@�p��aG��#�
��\)C���                                    Bxl���  �          @���@qG���p������r=qC�7
@qG��s33�0  �مC�]q                                    Bxl��  �          @�
=@k���ff�����C���@k��[��Mp��
=C���                                    Bxl��>  �          @���@H����{�=p��뙚C�h�@H���L(������'�C�C�                                    Bxl��  �          @���@~�R�P  �7���C�c�@~�R��
�j�H�Q�C��3                                    Bxl��  �          @��@}p��vff�\)��Q�C���@}p��?\)�^�R��C�z�                                    Bxl�$0  �          @���@H�����\����RC�5�@H���n�R�b�\��C��                                    Bxl�2�  �          @\@����z��*�H���C��@���{��|���"p�C��
                                    Bxl�A|  �          @�33@G���=q�\)��
=C��@G������tz��C�W
                                    Bxl�P"  �          @��
@�����7���
=C���@���z=q�����,
=C�W
                                    Bxl�^�  �          @Å@&ff����!����
C���@&ff�\)�u���C���                                    Bxl�mn  T          @Å@  �����B�\��C��q@  �mp������2\)C���                                    Bxl�|  �          @�(�@*=q��G��E��C�.@*=q�^{��Q��1�C���                                    Bxl���  �          @��@+����
�Y���{C���@+��N{��Q��<p�C��                                     Bxl��`  �          @�=q?�z����O\)�
=C�  ?�z��r�\��Q��@z�C�8R                                    Bxl��  �          @ƸR?�����R�G���p�C��3?�����H��\)�9{C��q                                    Bxl���   2          @ȣ�?u����G
=��{C�3?u��������8�C�aH                                    Bxl��R  �          @��?^�R�����W
=� �C���?^�R���\��\)�B�\C�f                                    Bxl���  �          @��?s33����i���(�C�U�?s33�s33��ff�N��C�                                    Bxl��  �          @�33?�33��
=�r�\�ffC�~�?�33�j=q����SffC���                                    Bxl��D  �          @���?�����{�y���ffC�n?����fff����W�C��{                                    Bxl���  �          @��?s33���R�{����C�n?s33�g
=��ff�Yz�C�e                                    Bxl��  �          @˅?�����R��z��"p�C�<)?���S�
���\�c�HC��                                     Bxl�6  �          @�G�?s33���\��{�'ffC��3?s33�J=q��33�i(�C�P�                                    Bxl�+�  �          @�Q�?�{��(���G��!ffC��)?�{�P  ��
=�b�C�l�                                    Bxl�:�  �          @���?�(���\)�tz��p�C��?�(��Z=q��G��VQ�C��\                                    Bxl�I(  �          @�G�?�33��33����#��C�
=?�33�L�������d�C��                                    Bxl�W�  �          @��H?0����  ����0\)C�Y�?0���@�������sp�C�q�                                    Bxl�ft  �          @˅>��R�����p��0(�C���>��R�C�
����tG�C��                                    Bxl�u  �          @��?@  ��=q�~{���C�k�?@  �\(���
=�`  C�&f                                    Bxl���  �          @��H>�p���\)�y����HC�#�>�p��g
=��{�\=qC��3                                    Bxl��f  �          @�p�����������z�C��R���e���=q�`33C���                                    Bxl��  �          @θR�.{��ff��(��\)C���.{�`����p��d�C��f                                    Bxl���  �          @�{�L����p����-�
C��H�L���I������r�RC�'�                                    Bxl��X  �          @���Q����\��ff�$33C�zὸQ��W
=��ff�iG�C�@                                     Bxl���  �          @�ff?   ��\)��33�)��C��?   �N{����n��C�h�                                    Bxl�ۤ  �          @�녿J=q����^�R�C��׿J=q�x������K
=C�>�                                    Bxl��J  T          @Ǯ�O\)���n�R��C�UÿO\)�e�����W(�C��f                                    Bxl���  �          @�=q��������s33�ffC�������b�\���H�W��C|�                                    Bxl��  �          @�녿����
=��
=�(�\C}O\����@  ���
�j�Cu�                                    Bxl�<  �          @�  �z�H������p��(Q�C�޸�z�H�C�
���H�k�C|33                                    Bxl�$�  �          @�Q�=L����Q���=q�<=qC�S3=L���+���z��C���                                    Bxl�3�  �          @�
=�����~�R���R�F�RC��������Q���{8RC�                                      Bxl�B.  �          @�ff�xQ��l������O\)CJ=�xQ���
��  �)Ct�q                                    Bxl�P�  �          @�  ��\)�j�H��p��P=qC}���\)� ������qCq                                    Bxl�_z  �          @��?xQ����\��ff�A�C���?xQ��p����RQ�C��
                                    Bxl�n   �          @��?�{��z���  �?�C���?�{� ����G�G�C��{                                    Bxl�|�  T          @��H?s33�n{�����R�\C�&f?s33�G���p�C���                                    Bxl��l  �          @�  ?G��J�H�����i\)C��q?G���z���Q�
=C�g�                                    Bxl��  �          @�G�?h���N{���\�g=qC���?h�ÿ������p�C�#�                                    Bxl���  T          @���?+��C33��{�p�C�5�?+����R�Å��C�33                                    Bxl��^  �          @�{?!G��Mp���  �hp�C��R?!G���Q���\)u�C���                                    Bxl��  �          @�p�?��K���  �i��C��?녿���
=L�C��                                    Bxl�Ԫ  �          @��?G��P  ��p��d�C���?G���  ��p��C���                                    Bxl��P  �          @�
=?s33�L����  �f
=C�E?s33��
=��\)�fC��=                                    Bxl���  �          @�  ?W
=�5����v��C�H�?W
=��G����H��C���                                    Bxl� �  �          @�33?��R�Fff����h\)C��?��R������C���                                    Bxl�B  �          @�p�?�(�������5  C�/\?�(��#�
���
�u  C��                                    Bxl��  U          @˅?�\)�fff�����G(�C��q?�\)��
=��G�C��                                    Bxl�,�  �          @�(�?�33�Dz����H�c�C�  ?�33��G�����
=C�Q�                                    Bxl�;4  �          @�33@�
�N�R�����S\)C�C�@�
��  �����C���                                    Bxl�I�  �          @�ff?�=q�N�R�����[�\C��f?�=q�������HC�3                                    Bxl�X�  �          @�ff?��E��p��d(�C�9�?����R�ÅQ�C��{                                    Bxl�g&  �          @��?���?\)���R�i\)C��f?���������
B�C��)                                    Bxl�u�  �          @���?�ff�I����33�b�C�q?�ff��������C��                                     Bxl��r  �          @ʏ\?�{�E��ff�]  C���?�{��ff�����C���                                    Bxl��  �          @�(�?����G
=��z��Z  C��?�����=q����C��                                    Bxl���  �          @��@'��e���
�8z�C�@'�������Q��m�C���                                    Bxl��d  �          @�ff@'��[������@{C��
@'��޸R��z��sG�C�=q                                    Bxl��
  T          @��@>�R�'���G��O
=C�` @>�R�c�
����s��C��3                                    Bxl�Ͱ  T          @���@�R�(Q������^p�C���@�R�Q����H��C���                                    Bxl��V  U          @��?�(��,(���ff�jQ�C�&f?�(��Q�������C��                                    Bxl���  �          @�{?��#�
����p\)C�|)?��&ff���H{C��)                                    Bxl���  �          @�{@��z����R�x��C�8R@��\)�\u�C��                                    Bxl�H  �          @ə�?˅���R���33C�` ?˅�L���\��C�#�                                    Bxl��  �          @���?�\)��\)��33ǮC�p�?�\)?&ff��
=��A��
                                    Bxl�%�  �          @���?�G���Q���33C��
?�G�?Tz�����A�=q                                    Bxl�4:  �          @��H?��Ϳ�33��33�qC���?���?\)���33A��                                    Bxl�B�  �          @���@$zῨ����ff�|p�C�n@$z�?z���=q=qAL��                                    Bxl�Q�  �          @���@(�ÿ�
=��ff�|ffC��@(��?5����\)Arff                                    Bxl�`,  �          @�(�@��aG���=q#�C��@�?����G���A�                                      Bxl�n�  �          @�G�?�(����\��=q�C��?�(�?.{��p�=qA��
                                    Bxl�}x  �          @�{@ff���R�����C�J=@ff>�����G�p�A-                                    Bxl��  �          @��@%��u����{C��H@%�?k�����\)A���                                    Bxl���  T          @�p�@6ff������p��l
=C���@6ff>�\)��(��|�@��                                    Bxl��j  �          @�G�?�=q�w����
�O��C���?�=q���R��33G�C�H�                                    Bxl��  �          @�?��p������]ffC��?���G����C�]q                                    Bxl�ƶ  T          @أ�?�\�W
=��33�o\)C�W
?�\��G���(��qC��                                    Bxl��\  �          @���?J=q�,(���(��fC�4{?J=q����£C��                                    Bxl��  �          @ڏ\>����C33���H�|��C��R>��ͿaG���  £��C�!H                                    Bxl��  �          @�z�?����@  �����u��C�ٚ?��ÿ\(���p�p�C�u�                                    Bxl�N  �          @�z�?��H�9����ff�u��C�U�?��H�G��ҏ\��C���                                    Bxl��  �          @޸R=��
�@  ��
=p�C�� =��
�G��ۅ¦�C��q                                    Bxl��  �          @�Q�G��<(���G�W
C��G��333���¢�qC]�R                                    Bxl�-@  �          @�{��{�8Q��ə��\C�� ��{�!G�����¨�Cq��                                    Bxl�;�  �          @�33>�{�.�R�ǮaHC���>�{��\��G�©�qC��q                                    Bxl�J�  �          @ڏ\?�G��AG���\)�u\)C�Z�?�G��\(���z�#�C��                                    Bxl�Y2  �          @�Q�@G��]p����\�P��C��f@G����R���C�c�                                    Bxl�g�  T          @׮@��^�R���\�Qz�C�H@���G�����C���                                    Bxl�v~  �          @�ff?��
�\(������^�HC�H?��
�����33�HC��{                                    Bxl$  �          @�{?���X�����H�c��C��\?���������=qC��                                     Bxl�  �          @��?��\�E�����r�C�%?��\�xQ���  �=C�C�                                    Bxl¢p  �          @���?@  �7
=��p��}�C�]q?@  �5����¢C�`                                     Bxl±  �          @�{��  ���H��G�Cks3��  ?L���љ��HC�\                                    Bxl¿�  �          @���?@  �
=q��33W
C���?@  =�����ff§.@��                                    Bxl��b  �          @�z�?�{����˅ǮC�s3?�{��G���G�¡G�C�                                      Bxl��  T          @���?����$z��Å(�C��?��;�33���
#�C���                                    Bxl��  �          @�ff?�ff����G��|��C��R?�ff���R��  p�C�Q�                                    Bxl��T  �          @\?����?\)��z��f(�C��)?��ÿ�����33�C�s3                                    Bxl��  �          @ʏ\��\)������33�{C�{��\)?O\)��ff¤=qB�p�                                    Bxl��  �          @�=q�}p������z��qCbG��}p�?��H�˅p�C^�                                    Bxl�&F  �          @�G��=p��޸R��Q�p�Cv�q�=p�?�\��\)¤G�C��                                    Bxl�4�  T          @�녿�=q��(��ə�ǮCiǮ��=q?G���p���C&f                                    Bxl�C�  �          @�33��\)�u��p�(�C\}q��\)?�ff�˅C��                                    Bxl�R8  �          @�녾�=q�J=q��  ¥\)C{\)��=q?��R���
L�B�Q�                                    Bxl�`�  �          @��H����p����Q�¢��C~xR���?�{���RBɸR                                    Bxl�o�  �          @�녽�G���p���B�C�o\��G�?���θR¡8RB��)                                    Bxl�~*  �          @��
=��
��Q������C��=��
?G��ȣ�¥ǮB��)                                    BxlÌ�  �          @�G��^�R��=q���H�)ClǮ�^�R?^�R���\C{                                    BxlÛv  �          @��
@?\)��Ϳ�p�����C�^�@?\)��{�-p��{C��H                                    Bxlê  T          @�
=@�ff�Dz�����HC��
@�ff�.{��  ��
C�L�                                    Bxlø�  �          @�ff@�\)�-p�>�?�{C��
@�\)�%�Q����C�C�                                    Bxl��h  �          @��@��
�?\)>\@p��C�:�@��
�;��0����p�C�xR                                    Bxl��  �          @�ff@��H�@  =�\)?:�HC�R@��H�5�z�H��\C��                                    Bxl��  �          @�{@����8Q쾔z��7�C�� @����&ff���
�O\)C�H                                    Bxl��Z  T          @��@����1녾�G���Q�C�1�@�����Ϳ���d(�C���                                    Bxl�   T          @�z�@����5�5���C���@����=q��z�����C���                                    Bxl��  �          @��@���6ff���H��p�C���@���\)���H�n�RC�k�                                    Bxl�L  �          @���@�{�Dz�(���{C�j=@�{�*=q��33��{C�>�                                    Bxl�-�  T          @�{@����333��  �K33C��q@������
=q��Q�C��=                                    Bxl�<�  �          @�G�@��H�0  ��
���\C��@��H�����9�����\C�xR                                    Bxl�K>  �          @�  @�z��#�
����33C���@�z����8�����C�                                    Bxl�Y�  �          @�33@����=�\)?@  C�t{@���	���=p���ffC�                                    Bxl�h�  �          @��\@��?G�A�C�J=@��{��G����C��{                                    Bxl�w0  �          @�  @�Q��p�>��H@�ffC�U�@�Q��{��
=���C�G�                                    Bxlą�  �          @��H@���� ��?�@���C�u�@����!녾�����C�\)                                    BxlĔ|  T          @�  @��H�*�H?�33AD  C��@��H�9��=�?��C��R                                    Bxlģ"  �          @���@j=q�aG���  �H\)C�=q@j=q?J=q�����I\)AB�H                                    Bxlı�  �          @���@q녿J=q�����I  C�)@q�?n{���
�Gz�A]�                                    Bxl��n  �          @�p�@s�
�W
=��{�B�RC�˅@s�
?O\)��{�C  A@��                                    Bxl��  �          @�p�@u��0�����B�
C��@u�?s33��(��?�A_�
                                    Bxl�ݺ  �          @�@p�׾k�����I�RC�B�@p��?�
=���\�<�A��R                                    Bxl��`  �          @�{@p�׿8Q���Q��F�\C���@p��?u���R�C��Ad                                      Bxl��  �          @��@{��W
=��p��>��C���@{�?Q���p��?  A<Q�                                    Bxl�	�  �          @�
=@��
�+�����5��C�g�@��
?k���{�3ffAI�                                    Bxl�R  �          @�=q@w��Tz���Q��<��C���@w�?G������==qA5                                    Bxl�&�  �          @�33@�(�����{��(��C�T{@�(�>k���(��3�@H��                                    Bxl�5�  �          @���@|(���=q��Q��/��C��=@|(�>aG���
=�;��@N�R                                    Bxl�DD  �          @��@��R���\�vff�$�HC��R@��R>aG������/p�@B�\                                    Bxl�R�  �          @��H@���z�H�xQ��&�
C�~�@��?�\�~{�+��@�=q                                    Bxl�a�  �          @��
@��
�Y����=q�0Q�C�#�@��
?333��33�1�A�                                    Bxl�p6  T          @��
@���xQ������-{C�u�@��?z���33�1�@���                                    Bxl�~�  �          @��@�녿����6p�C�>�@��?�����\�1{Af{                                    Bxlō�  �          @���@�zᾣ�
��{�4��C�@�z�?��R��Q��+z�A��                                    BxlŜ(  �          @���@|�ͽ�G���z��@�C�8R@|��?\���
�0��A��\                                    BxlŪ�  �          @�ff@��;u����6G�C�T{@���?�������*�A�p�                                    BxlŹt  �          @���@|(��u���
�?�HC��=@|(�?Ǯ���\�/��A��H                                    Bxl��  �          @��@}p������\�=z�C��R@}p�?�
=��{�5�\A��                                    Bxl���  T          @���@�Q쾳33��G��;=qC�� @�Q�?��\����1ffA�ff                                    Bxl��f  �          @���@j�H�\(���G��I�C�c�@j�H?c�
�����H��AY�                                    Bxl��  �          @�p�@s33�0����\)�D�C���@s33?������@��As�
                                    Bxl��  �          @�{@��H��������8z�C�'�@��H?�(����
�/�HA���                                    Bxl�X  �          @�p�@o\)��\�����HC�q@o\)?�(���z��@�A���                                    Bxl��  �          @�p�@P  �8Q���z��^�
C��@P  ?������Y  A��                                    Bxl�.�  �          @�ff@B�\�\(������g=qC��@B�\?����\)�d(�A�33                                    Bxl�=J  �          @�ff@q녾��H����H�C�L�@q�?�G�����?�A�                                    Bxl�K�  �          @�@l�Ϳ   ��33�K�
C�+�@l��?��\��ff�B�RA�{                                    Bxl�Z�  �          @�=q@W��@  ���V33C���@W�?�������R33A�z�                                    Bxl�i<  �          @���@l(���{�����@��C��f@l(�?�������FA��                                    Bxl�w�  �          @�\)@n�R�=p������AC�j=@n�R?p������?Q�Ac
=                                    BxlƆ�  T          @�G�@^{�
=���\�Q�RC�,�@^{?�Q����R�J(�A�p�                                    Bxlƕ.  �          @���@8Q�(�����R�l��C��f@8Q�?��\���H�cA���                                    Bxlƣ�  �          @�\)@>�R�������
�i33C�#�@>�R?�p�����YA�\)                                    BxlƲz  �          @��@G
=�
=�����b  C��H@G
=?��
��z��XffA���                                    Bxl��   �          @�{@E���\)��Q��c�RC�l�@E�?Ǯ��  �Q�AָR                                    Bxl���  �          @�33@Y���c�
��  �I�
C��@Y��?O\)�����J�HAV=q                                    Bxl��l  �          @�  @Q�>W
=��p��@�G�@Q�@����R�Z�B$z�                                    Bxl��  �          @�Q�@%��\)��p��nC�s3@%�?�����(��Wp�A�ff                                    Bxl���  T          @���@`  �!G��~{�@��C���@`  ?s33�z=q�<�Arff                                    Bxl�
^  �          @��
@�
=����0  ��C��\@�
=?!G��.{�=qA
=                                    Bxl�  T          @��@{��J=q�"�\�\)C�P�@{�>�  �(�����@i��                                    Bxl�'�  �          @�33@�{�0��� ����z�C�XR@�{>\)����ff?��H                                    Bxl�6P  �          @�33@�녿Y��?8Q�A=qC���@�녿���>�{@�(�C�g�                                    Bxl�D�  �          @�=q@�����\��  �8��C��@���Y���!G���Q�C�                                    Bxl�S�  �          @��@���=�?�(�A���?��H@��;�?�33A�C�&f                                    Bxl�bB  T          @��H@��    �p���+�<#�
@��>�p��^�R�@�
=                                    Bxl�p�  �          @���@��R=�G��E����?��R@��R>��Ϳ(����@��H                                    Bxl��  �          @�{@�����������C�,�@��aG���  �Dz�C��=                                    Bxlǎ4  �          @��@��R�h��>B�\@�C��@��R�h�þ8Q���C��=                                    Bxlǜ�  �          @�  @�{��\)?�@�p�C�]q@�{���>\@�z�C�H�                                    Bxlǫ�  �          @�(�@�
=��?�z�A\��C�J=@�
=�\)?��\AB{C���                                    BxlǺ&  �          @��
@�=q�aG�?�A�=qC���@�=q�\(�?���A�Q�C��                                    Bxl���  �          @��
@�  ��  ?�=qA��C�h�@�  �s33?˅A�=qC��                                    Bxl��r  �          @���@���u?�A��RC�t{@���aG�?�
=A�p�C�t{                                    Bxl��  �          @�{@�녾�\)@��A�  C�)@�녿��H@A��C�+�                                    Bxl���  �          @��@xQ�=#�
@:=qBp�?��@xQ쿎{@,(�B��C��q                                    Bxl�d  �          @��@y��<�@2�\BG�>��H@y����=q@%�B�
C�K�                                    Bxl�
  �          @���@u��.{@6ffB�C���@u����
@#33B  C��R                                    Bxl� �  �          @��@y��<#�
@Mp�B>L��@y����G�@<��B�HC�f                                    Bxl�/V  �          @��\@�
=<��
@hQ�B"�R>�=q@�
=��
=@U�BG�C���                                    Bxl�=�  �          @���@��
��\)@q�B#ffC��f@��
����@[�B��C�
                                    Bxl�L�  T          @�p�@�(��+�@b�\BffC��H@�(���@>{A�
=C��\                                    Bxl�[H  �          @���@���}p�@a�B
=C��@���z�@5A�ffC�S3                                    Bxl�i�  �          @�{@�z���@vffB �HC���@�z��Q�@1G�A�G�C��\                                    Bxl�x�  �          @�@�����H@qG�B
=C��q@���E�@0��A�ffC���                                    Bxlȇ:  �          @�@��\����@u�B!33C�=q@��\�Tz�@.{A��C�h�                                    Bxlȕ�  �          @��@�G��"�\@R�\B
=qC��@�G��i��?��RA�Q�C���                                    BxlȤ�  �          @��
@��H��@1G�A�C�Y�@��H�R�\?�=qAyC�4{                                    Bxlȳ,  �          @�=q@����@E�A�\)C��@��P  ?�
=A��RC��R                                    Bxl���  �          @�33@�33��
=@C33A�G�C��R@�33�@  @   A�=qC�n                                    Bxl��x  �          @��\@����9��@��A���C���@����aG�?G�@�  C�{                                    Bxl��  �          @�=q@�����H?�33A�{C�+�@����8Q�?�@�
=C��                                    Bxl���  �          @���@�
=�޸R?�\A�ffC���@�
=�33?h��A��C�
                                    Bxl��j  �          @�G�@��R��G�@p��B"p�C��f@��R�,(�@:�HA���C��
                                    Bxl�  T          @�Q�@�zῙ��@\��Bz�C�W
@�z�� ��@*=qAݮC�&f                                    Bxl��  �          @�=q@���xQ�@B�\A�z�C�\)@���Q�@Q�A�\)C�+�                                    Bxl�(\  �          @���@��ÿ��@Z=qB\)C���@����(�@(��A�p�C��H                                    Bxl�7  �          @���@�(���Q�@l(�B{C��)@�(��C�
@*�HA�\)C�                                    Bxl�E�  �          @�{@�=q��Q�>���@UC���@�=q�W
=>��@*�HC��f                                    Bxl�TN  �          @��R@�p�>�{>�?��@Z=q@�p�>�  >��@'�@#�
                                    Bxl�b�  �          @�ff@�>�{������@Z=q@�>��
=�G�?���@O\)                                    Bxl�q�  �          @��@��
��p���Q��h��C�!H@��
�B�\����33C��                                    Bxlɀ@  �          @�p�@�\)���׾�\)�4z�C��=@�\)��33�Tz��{C��                                    BxlɎ�  �          @�\)@��\��p�@4z�A�C���@��\�/\)?���A�\)C��\                                    Bxlɝ�  �          @�\)@����(�@��A�C��3@����9��?���A1p�C�\)                                    Bxlɬ2  �          @��@���   @0  A��C�J=@���=p�?�
=A�(�C��                                    Bxlɺ�  �          @���@���ٙ�@'�A�Q�C�aH@���(Q�?�
=A��C���                                    Bxl��~  �          @���@�{��z�@��A��RC�l�@�{�+�?�G�AIC��                                    Bxl��$  �          @��\@�
=��@6ffA�Q�C���@�
=�X��?�=qA{�
C�ff                                    Bxl���  �          @��\@�=q�ff@FffB ��C��
@�=q�[�?�=qA��RC�Ǯ                                    Bxl��p  �          @�33@p���!G�@j�HBQ�C��@p���u�@�A�z�C�=q                                    Bxl�  �          @��
@K��9��@~�RB*��C��
@K�����@��A�33C�B�                                    Bxl��  �          @�(�@\(��%@�  B+�C�}q@\(�����@#33A�G�C�8R                                    Bxl�!b  �          @��@O\)�ff@�Q�B;(�C�f@O\)�z�H@8Q�A�  C�Ǯ                                    Bxl�0  �          @���@c33�(�@�p�B3��C�%@c33�p  @7
=A�{C���                                    Bxl�>�  �          @���@XQ�˅@���BJp�C�t{@XQ��W
=@]p�B�HC��                                     Bxl�MT  �          @���@]p�����@�=qB=��C�J=@]p��e@FffA���C��3                                    Bxl�[�  �          @��@|(���ff@z�HB(��C��{@|(��R�\@2�\A�ffC�\                                    Bxl�j�  �          @��\@�(���@dz�BG�C��@�(��Vff@�A��\C�t{                                    Bxl�yF  T          @�33@n{��@w�B&z�C�E@n{�l��@"�\A��HC��\                                    Bxlʇ�  T          @�(�@e��\)@s33B'�HC�@e��h��@\)A�  C�H�                                    Bxlʖ�  �          @�@��H�\)@<(�A�33C���@��H�Q�?��HA��\C�:�                                    Bxlʥ8  �          @�33@�
=�ff?�
=A�(�C�S3@�
=�6ff?�@�{C�R                                    Bxlʳ�  �          @�@��
���@z�A�=qC��f@��
�L��?�ffA#�C�T{                                    Bxl�  �          @���@����33@  A��C��@����3�
?�33A4��C�n                                    Bxl��*  �          @��\@����
=?�=qA�(�C�n@����:=q?&ff@˅C��                                    Bxl���  �          @��@�\)��\@�\A���C���@�\)�<(�?\(�A�HC���                                    Bxl��v  �          @�=q@W
=����@���BA33C���@W
=�aG�@EB��C�Ф                                    Bxl��  �          @�z�@�ÿ���@�z�Bs  C��@���j=q@|��B(�\C��
                                    Bxl��  �          @��@Q���H@�(�Bq{C�+�@Q��p  @x��B%{C�4{                                    Bxl�h  �          @��@=q��Q�@��
BpG�C�p�@=q�n�R@x��B$�RC�k�                                    Bxl�)  "          @���@-p���@�ffBe{C�33@-p��h��@p  B33C�U�                                    Bxl�7�  �          @�33@�H��z�@���Bo�C���@�H�k�@uB$ffC��f                                    Bxl�FZ  �          @�(�@�ÿ�
=@��RBy�C��R@���qG�@~{B*
=C�Ф                                    Bxl�U   �          @��@
=����@�
=B|p�C�^�@
=�l��@�Q�B-  C��\                                    Bxl�c�  �          @�z�@p��ٙ�@�p�Bv�\C�4{@p��q�@z�HB'\)C�,�                                    Bxl�rL  �          @��?��ÿ�=q@�\)B��)C���?����s33@�  B:G�C��3                                    Bxlˀ�  �          @�
=@  ����@���Bm��C��H@  �qG�@g�B33C�c�                                    Bxlˏ�  �          @�33?�ff��@�ffBx(�C���?�ff�xQ�@g�B ��C�j=                                    Bxl˞>  �          @���?�33�	��@�{B~�C�?�33���H@`��B�C��R                                    Bxlˬ�  �          @�=q?��z�@�  B�33C�h�?�����@_\)B(�C��                                    Bxl˻�  �          @�?�ff��z�@�G�Bz33C���?�ff�z�H@l��B"�C�O\                                    Bxl��0  �          @��?�\��@�{Bzz�C���?�\�tz�@hQ�B#
=C�n                                    Bxl���  �          @���@z���@�p�BZ(�C��
@z���=q@;�B�
C��                                     Bxl��|  �          @�ff@e���{@�  B3  C�AH@e��[�@1G�A��
C�)                                    Bxl��"  �          @�
=@s�
����@}p�B/�C�˅@s�
�I��@7�A��HC�/\                                    Bxl��  �          @�ff@z�H�\@uB)z�C�e@z�H�C33@1G�A�G�C��                                    Bxl�n  �          @�ff@��\��Q�@l��B"33C�B�@��\�;�@,(�A��C�*=                                    Bxl�"  �          @�  @���Ǯ@l��B \)C���@���A�@(Q�A�  C��f                                    Bxl�0�  �          @��@�녿�p�@VffB33C�
@���A�@\)A�Q�C�s3                                    Bxl�?`  �          @�  @��
����@i��B=qC�j=@��
�C33@$z�AԸRC��H                                    Bxl�N  �          @�  @�G���(�@r�\B%��C�H@�G��?\)@/\)A���C���                                    Bxl�\�  T          @���@|�Ϳ˅@x��B)�\C��@|���I��@1�A�=qC��{                                    Bxl�kR  �          @��@����\@S�
B\)C�+�@��Q�@ ��AܸRC�'�                                    Bxl�y�  �          @�Q�@�=q���@4z�A�G�C���@�=q�QG�?\A�p�C���                                    Bxl̈�  T          @�(�@�ff�(�ÿ��DQ�C�]q@�ff���{��=qC�T{                                    Bxl̗D  �          @�z�@�녿�
=?�ffA��C�c�@���(�?#�
@�z�C���                                    Bxl̥�  �          @�\)@n{��  @�p�B;  C��)@n{�>�R@J�HB��C��f                                    Bxl̴�  �          @���@qG��u@��RB=33C�� @qG��/\)@UBz�C���                                    Bxl��6  �          @�=q@��\��  @mp�B=qC�z�@��\�#�
@7
=A��C��R                                    Bxl���  �          @��\@���G�@Q�BC��
@���
�H@%�A�\)C��f                                    Bxl���  �          @���@�����p�@EBQ�C���@������H@%A��C��=                                    Bxl��(  �          @�(�@��@�
=�G�?�\)A��
@��?��?��\A)��A���                                    Bxl���  �          @�=q@���>��R@ ��A�@fff@��Ϳ!G�?�
=A���C�Y�                                    Bxl�t  �          @���@���?�(�?�Q�Au�A�=q@���?(��?���A�{@�                                    Bxl�  �          @�=q@��?�ff?��A�p�Af�\@��>���@�A��@\(�                                    Bxl�)�  �          @��@���?��?���AG\)AB=q@���>��?�=qA���@��
                                    Bxl�8f  �          @��R@��
����@�RA�z�C�U�@��
���@�A���C�h�                                    Bxl�G  �          @�
=@�z�W
=@;�A�G�C���@�z�\@!G�A�33C�]q                                    Bxl�U�  �          @�\)@�녾�(�@E�BQ�C�t{@�녿��
@#33A��
C��{                                    Bxl�dX  �          @��@�\)���@Mp�BffC���@�\)����@+�A޸RC�t{                                    Bxl�r�  �          @���@��H�0��@AG�A��
C��@��H���R@Q�A�\)C��\                                    Bxĺ�  �          @���@��H��Q�@QG�B
�C���@��H�#33@�A\C�y�                                    Bxl͐J  �          @�Q�@�G��\@_\)B��C�B�@�G��<(�@=qA�
=C�Ǯ                                    Bxl͞�  �          @��@�녿�\)@`  B�HC�.@���4z�@\)A�(�C�j=                                    Bxlͭ�  �          @�ff@����(�@`��B
=C�o\@���:=q@��A˅C�                                    Bxlͼ<  T          @�ff@�\)��\)@h��B
=C��)@�\)�*=q@.�RA���C���                                    Bxl���  �          @�
=@��׿�@fffB��C�T{@����,(�@*�HA���C��                                     Bxl�و  �          @��@�p���@@  A��C���@�p��K�?�p�A�
=C�!H                                    Bxl��.  �          @�\)@��R�,��@0��A�G�C���@��R�j�H?��HAD  C�y�                                    Bxl���  �          @�z�@�ff�Vff?�\)A��\C���@�ff�n{���Ϳ�G�C�>�                                    Bxl�z  �          @��R@�  �@�A�ffC�q@�  �<��?��A8Q�C��                                    Bxl�   �          @��R@�=q�J=q@�A���C�3@�=q�xQ�?(�@�
=C�+�                                    Bxl�"�  �          @��@�(��?\)@(�A�(�C��@�(��j=q?\)@��HC�9�                                    Bxl�1l  �          @�@���У�@,��A��C�.@���,��?��A�  C�]q                                    Bxl�@  �          @�z�@�\)>�Q�@mp�B$p�@�@�\)���@\��B(�C���                                    Bxl�N�  �          @��@�G�>���@fffB�H@~�R@�G���z�@U�B�RC���                                    Bxl�]^  �          @��@�p�?!G�@j�HB$
=A
=q@�p����@b�\B\)C�Y�                                    Bxl�l  �          @�z�@�=q>�G�@dz�B�\@�G�@�=q���\@W
=B��C�Ф                                    Bxl�z�  �          @�\)@��>�{@p  B#�R@�=q@�녿�Q�@^�RBC���                                    BxlΉP  �          @�
=@��R�.{@G
=Bz�C�޸@��R�33@�A�z�C�<)                                    BxlΗ�  �          @�
=@�(���{@H��B�
C�C�@�(����@��A�33C��                                    BxlΦ�  �          @��\@��R�u@Mp�BffC���@��R�@��A�
=C�'�                                    BxlεB  �          @�\)@����Q�@VffBG�C��
@���5@�\A��\C��                                     Bxl���  �          @�Q�@�{�\@P��B
�
C��\@�{��{@,��A�p�C�.                                    Bxl�Ҏ  �          @���@�zὸQ�@Y��B��C�j=@�z��
=@=p�A�C��                                    Bxl��4  �          @�
=@���#�
@a�B�C���@����Q�@FffB\)C���                                    Bxl���  �          @�ff@�{=���@e�B��?�p�@�{����@Mp�B�HC��                                    Bxl���            @�ff@�Q�?&ff@|��B0�RA�H@�Q쿣�
@r�\B'��C�                                      Bxl�&  �          @�@`��?���@�
=B@
=A�  @`�׿333@��BK�C�W
                                    Bxl��  �          @�  @z=q?J=q@�(�B7�A7\)@z=q��  @���B1�C�*=                                    Bxl�*r  �          @��@��R�W�?�  AJffC�z�@��R�c�
����=qC��{                                    Bxl�9  �          @�
=@�G��#33@�
A�
=C�Q�@�G��U?Tz�A��C��3                                    Bxl�G�  �          @�  @�{��Q�@L��B	��C�  @�{�1�@��A��C��{                                    Bxl�Vd  �          @�G�@����+�@�=qB3  C�H�@����!G�@QG�B	�C�\                                    Bxl�e
  �          @���@\(��Tz�@���BO�
C�0�@\(��8Q�@eB�RC��                                    Bxl�s�  �          @���@|(��=p�@��B833C���@|(��'�@S�
B�C�,�                                    BxlςV  T          @���@\)�&ff@��
B6G�C�\)@\)�!�@Tz�BffC���                                    Bxlϐ�  �          @���@�\)�Ǯ@xQ�B)��C�e@�\)��@N�RB��C�W
                                    Bxlϟ�  �          @��@e<�@�ffBLp�>��@e�ff@{�B-z�C�޸                                    BxlϮH  �          @��@����z�@i��B=qC�&f@�����R@Dz�B {C��                                    Bxlϼ�  �          @�G�@�G�?�@��HB4��@�G�@�G���  @uB'  C��{                                    Bxl�˔  �          @���@{�>�(�@��B;@�\)@{����@{�B*��C��=                                    Bxl��:  �          @��\@w
=    @�33BA�C���@w
=��@tz�B$=qC���                                    Bxl���  �          @��@qG��0��@��HBBQ�C��
@qG��+�@_\)B
=C�K�                                    Bxl���  �          @�Q�@x�ÿ&ff@���B9�HC�E@x���#33@U�Bp�C�XR                                    Bxl�,  �          @���@{���  @��B5�C�ٚ@{��6ff@HQ�BG�C�                                    Bxl��  �          @���@����B�\@G
=Bz�C�ٚ@�����z�@(Q�A��C�xR                                    Bxl�#x  �          @�
=@����?�A���C��3@�����?��AY��C��                                     Bxl�2  T          @���@�\)���@ ��A�z�C��{@�\)��(�?��HAC�C��\                                    Bxl�@�  
�          @�Q�@�(��k�?���A�\)C�'�@�(���
=?�33A8��C�T{                                    Bxl�Oj  
�          @�=q@S33��p�@��\BQC��\@S33�Q�@Z�HBQ�C��
                                    Bxl�^  �          @��@a녿��@�BH��C�o\@a��E@W
=BffC�e                                    Bxl�l�  �          @��\@�ff=L��>�33@a�>��H@�ff��>���@U�C�\)                                    Bxl�{\  �          @�(�@�\)�L��@?\)A�\)C�l�@�\)�	��@�RA���C�U�                                    BxlЊ  �          @�z�@�\)��
=@_\)B�HC��
@�\)�.{@�RA�z�C�XR                                    BxlИ�  T          @��@�33��(�@Tz�B	Q�C�z�@�33��p�@,��A�=qC��                                     BxlЧN  T          @�@������@(Q�A��C�1�@�����z�@�RA���C��3                                    Bxlе�  T          @�@��þ�=q@H��A��C�p�@��ÿ�G�@'
=A���C�XR                                    Bxl�Ě  �          @�p�@���>aG�@z�A��
@�@��Ϳn{@��A�=qC�R                                    Bxl��@  �          @�33@�G�<��
@{A�{>L��@�G���
=@
=qA�=qC���                                    Bxl���  T          @��
@��þ���@ ��AʸRC��)@��ÿ���@   A��C���                                    Bxl���  �          @��H@�
=�L��@%�A�{C���@�
=��ff@{A�C��                                    Bxl��2  �          @�(�@�  =���@%A�(�?��@�  ��
=@�
A��
C��=                                    Bxl��  T          @��
@��H=���@�HA�
=?��@��H���@
=qA�
=C�4{                                    Bxl�~  �          @���@�(��#�
@=qA���C�  @�(�����@G�A�z�C��                                    Bxl�+$  �          @��@��>L��@5�A�@p�@�����H@$z�A�Q�C�T{                                    Bxl�9�  �          @�p�@�G�>#�
@@  B =q?�@�G���=q@,��A�C�@                                     Bxl�Hp  �          @�(�@��H>\@N�RB
=@�
=@��H��  @@  B �HC�b�                                    Bxl�W  �          @�@���>��
@AG�B ��@q�@��ÿ��H@2�\A�  C��                                    Bxl�e�  �          @��@���>L��@ffA�
=@�@����z�H@	��A���C�Ǯ                                    Bxl�tb  �          @���@�(���=q?��Ax(�C��@�(���z�?0��@ۅC�7
                                    Bxlу  T          @�G�@��׿��?��A��C�� @��׿�ff?��A8z�C���                                    Bxlё�  �          @�Q�@Y���e��.{���HC���@Y���1��ff��C�U�                                    BxlѠT  �          @��@p���G
=�����\��C�1�@p�����$z�����C�L�                                    BxlѮ�  �          @���@c33�#�
�G�����C��@c33��{�N�R�#�C�U�                                    Bxlѽ�  T          @��H@�Q�?B�\����z�A"�\@�Q�?�{�޸R���
A��H                                    Bxl��F  
�          @�G�@C�
?h���E��/��A�z�@C�
@�
�{����B�                                    Bxl���  �          @���@����R�y���OQ�C���@�>L�����
�wG�@��
                                    Bxl��  �          @�G�@{���������e�RC�t{@{?J=q��=q�u�A���                                    Bxl��8  �          @�p�?B�\�#�
��Q��affC�N?B�\�u��   �fC�]q                                    Bxl��  �          @��?ٙ��8Q���(�G�C��{?ٙ�@ ���x���_��BG
=                                    Bxl��  �          @�ff?��R>�����AQ�?��R@(Q��j�H�@G�BS�                                    Bxl�$*  T          @��?���=���z�ff@hQ�?���@!����T��BW
=                                    Bxl�2�  �          @��@>W
=��ff�}�H@��@@!G��~�R�D�B<z�                                    Bxl�Av  �          @�=q@�H?������w�AL��@�H@2�\�qG��6�BD=q                                    Bxl�P  T          @�z�@\)>�����Q��x�Ap�@\)@-p��{��;�\B>{                                    Bxl�^�  �          @��
����p����H�Cy\)��?�(����H8RB�ff                                    Bxl�mh  �          @�(���=q��{�����{�
Ce���=q?E���=q�
C�q                                    Bxl�|  �          @���0�׿�\�����)Cx��0��?}p���Q�(�B��R                                    BxlҊ�  �          @�z�J=q�333��\)33C]�\�J=q@ ����(�\)B���                                    BxlҙZ  �          @�33�Q녾L������¡�
CB�Q�@p���p��s�B�Ǯ                                    BxlҨ   �          @�z�E��B�\���£�CA�{�E�@   ��{�s��B�(�                                    BxlҶ�  �          @��
>���?���������B��{>���@a��z=q�?
=B�L�                                    Bxl��L  �          @��?\(�?�G���Q���Bqff?\(�@tz��e��)��B��{                                    Bxl���  �          @�(�?�  ?\)��33��A�R?�  @B�\��(��P�
B���                                    Bxl��  �          @�
==u������{¬��C�C�=u@(����
�}\)B�#�                                    Bxl��>  �          @�\)?   �aG���¨�=C�+�?   @"�\�����v�
B��q                                    Bxl���  �          @��=�G��B�\��
=¯aHC��\=�G�@%����\�wG�B�33                                    Bxl��  �          @�  >aG������\)®u�C�c�>aG�@(Q���=q�uG�B��=                                    Bxl�0  T          @�  >��=�\)���Rª{A=q>��@3�
���j��B�                                    Bxl�+�  �          @�Q�?�33>L����=q8RA\)?�33@6ff����Z�HBz�                                    Bxl�:|  �          @���?xQ�u��=q��C��
?xQ�@{��
=�rz�B���                                    Bxl�I"  �          @����u>�(���Q�©�
B�\)�u@I����G��\�RB��                                    Bxl�W�  �          @��\?��?^�R��{��B^��?��@`  ��  �H�B���                                    Bxl�fn  �          @����E��z�H����)Cg��E�?�{���
�B��                                    Bxl�u  �          @���>����R��{¥�
C�]q>��@p���Q�B��\                                    BxlӃ�  �          @���>�  ����\)®C�]q>�  @*=q��G��sp�B�{                                    BxlӒ`  T          @��׾�׿   ��{¥�{Cc;��@z���ff�
B��
                                    Bxlӡ  �          @�G���z�+���\)¤�RCvk���z�@����=q�qB�=q                                    Bxlӯ�  �          @��׾8Q�.{��ff¥W
C
=�8Q�@���G�\B�z�                                    BxlӾR  �          @��ÿ�
=�������R�Cd���
=?�G����8RB���                                    Bxl���  �          @�  �Ǯ�Y����(� 33Cu� �Ǯ@   ���B�B�\                                    Bxl�۞  �          @�������Q���\)  CVn����?�����p��B���                                    Bxl��D  �          @�      ���R��{­� C�Ф    @   ���H�z�B��                                    Bxl���  �          @��>u�c�
��z� ��C�xR>u?�p����HG�B�L�                                    Bxl��  �          @�ff>�33�.{��(�£��C�Ǯ>�33@
=q��
=�\B�\)                                    Bxl�6  �          @�
=>��H�W
=��z�¨C�H�>��H@$z���  �t�B��=                                    Bxl�$�  �          @�{���
��{���
©�Cb���
@(���G��{33B�#�                                    Bxl�3�  �          @�ff�L�Ϳn{�����3C��;L��?�z�����B�B��                                    Bxl�B(  �          @�<��
�^�R����¡G�C���<��
?��H��  G�B��)                                    Bxl�P�  �          @���E���  ���H�CnO\�E�?��
��Q�=qB�B�                                    Bxl�_t  �          @��;����33��
=��C��ᾅ�?�\)���
��Bͨ�                                    Bxl�n  �          @���?�=q�����ff#�C�Z�?�=q?���p�33B'�H                                    Bxl�|�  �          @�ff��������R��C�e��?����ff� B�
=                                    Bxlԋf  �          @�ff�(�ÿ+����R�o��CB:��(��?���(��W
=C�                                    BxlԚ  
�          @��R��
=�}p������CR�)��
=?޸R��33�~  C�3                                    BxlԨ�  �          @�\)?��R�&ff��
=ffC�B�?��R@���=q�v�B[�                                    BxlԷX  �          @���>�\)�xQ����ǮC�*=>�\)?�Q���z�.B�W
                                    Bxl���  �          @����G���{���C�]q��G�?��
���H��B��3                                    Bxl�Ԥ  �          @����녿���������Cj\���?�  ��z��{CY�                                    Bxl��J  �          @�{���Ϳ�=q����}�RCd�ÿ���?n{���
=C�                                    Bxl���  �          @�����  �����\)B�Cm\��  ?�\)����aHB�B�                                    Bxl� �  �          @�p���G���33���\#�CnT{��G�?����p��{B�k�                                    Bxl�<  �          @�녾�p���ff��=q�qC�P���p�?����k�B�{                                    Bxl��  �          @�  <��
��ff���R�
C�P�<��
?�{��Q�\B��{                                    Bxl�,�  �          @��׿�\)��\)��  p�Ca{��\)?�  ���\� B�#�                                    Bxl�;.  �          @���333������(��Cq�ÿ333?�33��G��{B�8R                                    Bxl�I�  �          @�����R��(�����¤
=CV�Ϳ�R@�H��33�zB��H                                    Bxl�Xz  T          @�=q�aG���=q����¬��Cgz�aG�@(Q���z��vp�B�p�                                    Bxl�g   �          @�33>\)���R���­�C�f>\)@'���p��w�HB�                                    Bxl�u�  �          @�33��(��h������)CS�=��(�?�(��������B���                                    BxlՄl  �          @���ٙ��!G���33�=CHG��ٙ�@�R����p��B���                                    BxlՓ  �          @���׿����\�3CC�
���@�
���H�i�\C5�                                    Bxlա�  �          @���
���\��=q�~=qCK���
?�G����
�l��C��                                    Bxlհ^  �          @��������H�����CQ
��?˅���R�uC�H                                    Bxlտ  �          @�z῅��z�H��ff� C_T{���?��R��p�
=B�ff                                    Bxl�ͪ  �          @�p����׿�������C_\)����?�z���{ǮB�p�                                    Bxl��P  �          @�{��녿.{��  �)CR�q���@�����|�B�8R                                    Bxl���  �          @�z῾�R��
=��  ��C_����R?�  ��\)�C�                                     Bxl���  �          @�{��녿�33��  p�C\aH���?�����RG�C�                                    Bxl�B  �          @�
=���
=�\)��p��=C1�����
@8�����H�V�B�p�                                    Bxl��  �          @�{���
�aG���
=W
C<
=���
@*=q�����e
=B��                                    Bxl�%�  �          @��H��Q������ffCC=q��Q�@������jQ�B�\                                    Bxl�44  �          @�(���Q�\�����
C?
=��Q�@�H��{�b
=C ��                                    Bxl�B�  �          @��Ϳ��H�Tz�����{CM�ÿ��H@�
���R�v�C�3                                    Bxl�Q�  �          @�(��  ������R�x��CR�q�  ?������v
=CG�                                    Bxl�`&  �          @�G��
=q��ff��33�u
=CW�{�
=q?��H��{�}p�C��                                    Bxl�n�  �          @�Q��G���G������=CN�H�G�?�G�����s�\C
�3                                    Bxl�}r  �          @�G�����\)��ff�3C<����@�R���\�`{B�                                      Bxl֌  �          @������\��\���\�CI�=���\@ff����s��B��H                                    Bxl֚�  �          @��\�8Q쿌��������Cl�f�8Q�?����k�Bݽq                                    Bxl֩d  �          @�ff=�\)=u��±��B�H=�\)@9����33�g�B�\                                    Bxlָ
  �          @��H��  >u��  ¬�{C!H��  @<����33�_  B��                                    Bxl�ư  �          @�  ?Tz�W
=���B�C�O\?Tz�@��  �B�                                      Bxl��V  T          @��?&ff�&ff���� �{C�n?&ff@G���{�)B�                                      Bxl���  �          @�  >�{��������{C��>�{?У���\)��B��                                    Bxl��  �          @�ff��z��(���\)�HCk�῔z�?�33��z��RC)                                    Bxl�H  �          @�\)?����G���
=
=C�˅?��?�����B�8R                                    Bxl��  �          @�\)>��
��G���
=�C���>��
?��H���33B�z�                                    Bxl��  �          @�\)>�Q�ٙ���{C��3>�Q�?��
����
B���                                    Bxl�-:  �          @��R�������H��p��qC��
����?�G������\B�B�                                    Bxl�;�  �          @�\)�
=�  ��ff��C:�
=?0����z� �)C                                    Bxl�J�  �          @��R�����Q����H�z(�Cm������?=p����G�C:�                                    Bxl�Y,  �          @��R���������  �sp�Cn�����?z���\)  CB�                                    Bxl�g�  �          @�ff���\�����{�p(�Cr����\>�G���  G�C ޸                                    Bxl�vx  �          @�
=��(��G����\�Cn�ÿ�(�?Q���p�B�C��                                    Bxlׅ  �          @��ÿ�z��\������Cgs3��z�?n{��33��C�{                                    Bxlד�  T          @�G��B�\�33�����HCy��B�\?G���(�(�C0�                                    Bxlעj  �          @��=�G���33��Q�z�C���=�G�?��R��(��B�                                    Bxlױ  �          @��H?
=q��\)���R�C��?
=q?��
��ffǮBxff                                    Bxl׿�  �          @���?�Q쿋�����u�C�c�?�Q�?�z����L�BC��                                    Bxl��\  �          @��H?&ff��ff��=q�C�� ?&ff?�����B�B��                                    Bxl��  �          @��?���\)���33C���?�@!����R�SBS{                                    Bxl��  �          @��?��H�+���G�
=C��{?��H?�=q���R�s�BM�                                    Bxl��N  �          @���?����������HC�k�?��?����=qG�B�
=                                    Bxl��  �          @�=q?B�\�p������C��?B�\?�Q���z��)B���                                    Bxl��  �          @���>��H�@  ��{ W
C��f>��H@
=����=B�Ǯ                                    Bxl�&@  �          @�G�?aG���ff���W
C�J=?aG�@ff���
�v(�B�                                    Bxl�4�  �          @��?�녾B�\��33=qC�?��@!����cz�Bu�                                    Bxl�C�  �          @��\?���\(���(��RC�+�?��?�{��33�r�
B8{                                    Bxl�R2  �          @��@
�H�8Q���  �qC���@
�H@����
�PG�B=��                                    Bxl�`�  �          @���@ ��>B�\���H�@���@ ��@.{��Q��G33BV�                                    Bxl�o~  �          @�=q@녿5������C�W
@�?�Q���ff�f��B/(�                                    Bxl�~$  �          @�  ?���(���=q�C�G�?�@(���=q�b�BIQ�                                    Bxl،�  �          @���@�R?O\)���H�q�A��@�R@G��]p��#�BN\)                                    Bxl؛p  �          @�ff@(�?k����
�l�\A�33@(�@Fff�Mp���BO                                      Bxlت  �          @�
=@33?�G����\�u=qA���@33@R�\�Vff�z�B\=q                                    Bxlظ�  �          @�Q�?��>�ff��{  At��?��@@  �|���D\)Buff                                    Bxl��b  �          @�Q�?�\)��Q���G��C���?�\)@&ff��=q�b�B��\                                    Bxl��  �          @�z�?�\)?�G���p��A�{?�\)@U�Z�H�'(�BsG�                                    Bxl��  �          @��@9��?k�����W(�A��@9��@?\)�A��(�B7z�                                    Bxl��T  �          @��@E�?��x���F�
A�ff@E�@C�
�*�H����B3(�                                    Bxl��  �          @�33@�R?Tz�����lffA�=q@�R@AG��P  ��BJ�                                    Bxl��  �          @��H@$z�?\(������f\)A��@$z�@@  �J=q��\BE�\                                    Bxl�F  �          @�z�@+�?��
��
=�_G�A�@+�@G
=�A����BEG�                                    Bxl�-�  �          @�z�@*=q<#�
�����kp�>�@*=q@ff�n{�9�B%�R                                    Bxl�<�  �          @��@0  �8Q������d��C�
@0  @��mp��;��B�\                                    Bxl�K8  �          @�  ?�  ����{�33C��\?�  ?=p���=q�\B�R                                    Bxl�Y�  
�          @���?#�
��
���C��q?#�
?Q���p�=qBP��                                    Bxl�h�  T          @�G�?�R��
���R�|
=C�� ?�R?����
=¡��B%                                    Bxl�w*  �          @�G��L���HQ�����VffC����L�;Ǯ��  «�=C�@                                     Bxlم�  �          @�ff��33�\���r�\�=��C�R��33�fff��G�B�Cx                                    Bxlٔv  �          @�  �W
=�0������g�\C��\�W
=<��
���R¯Q�C/�\                                    Bxl٣  �          @�{@Fff�H���.�R��  C�Ff@Fff�����\)�H��C�`                                     Bxlٱ�  �          @�  @P  ����S33��C�~�@P  ���
��33�M�C�:�                                    Bxl��h  �          @���@U���fff�,(�C��@U>�z���=q�J
=@���                                    Bxl��  �          @�?�����������d�\C��f?���>\���\k�AS�                                    Bxl�ݴ  �          @���?��1����R�U�C���?�������G��C��R                                    Bxl��Z  �          @��?�p��(�����H�\�\C���?�p�=�Q����\�\@U�                                    Bxl��   �          @��?:�H�1���Q��e�C�N?:�H=u����¤\)@��
                                    Bxl�	�  �          @��
?k������
=�u�C�}q?k�>��H������A��                                    Bxl�L  �          @��H?��\��=q��z���C���?��\?�������RB8�                                    Bxl�&�  �          @��
?8Q�����z���C��?8Q�?\(�����fBG��                                    Bxl�5�  �          @��>�
=������ff�C���>�
=?�z���33�fB���                                    Bxl�D>  �          @�p�?�p��
=q��\)�t33C�/\?�p�?333����ǮA�p�                                    Bxl�R�  �          @�ff?�����Q����8RC�G�?���?������=qB(�                                    Bxl�a�  �          @�z�>�
=�����  � C�޸>�
=?�z���Q�=qB��                                    Bxl�p0  �          @�=q>�{��(����H�C���>�{?�p����\��B�{                                    Bxl�~�  �          @��R?E�����  �C��?E�?������aHB�\                                    Bxlڍ|  �          @�\)?�Q쿜(�����.C��?�Q�?���p��\B2�
                                    Bxlڜ"  �          @�
=?�\)������(���C��f?�\)?\���aHBV{                                    Bxlڪ�  �          @�{>�ff��=q����8RC�H�>�ff?�
=����8RB��f                                    Bxlڹn  �          @�ff>�  ��G���ff��C���>�  ?�������3B�\)                                    Bxl��  �          @��?������i���Sp�C��?�녽�\)�����C���                                    Bxl�ֺ  �          @�Q�@7
=�>�R�����33C��@7
=����`  �@33C��
                                    Bxl��`  �          @�ff@I���$z��z��홚C�c�@I���n{�U�5C�Ǯ                                    Bxl��  T          @���@ff�%�Dz��$�C��@ff����~�R�k  C�q�                                    Bxl��  �          @�(�@���$z��H���+ffC�O\@�;��H�����r��C��                                     Bxl�R  �          @��@���:=q�fff�3C�(�@�ÿ���33��C�{                                    Bxl��  
�          @��@�H�?\)�Z=q�&{C�~�@�H�333��\)�r�C��                                    Bxl�.�  �          @�
=@.{�)���e��-=qC�� @.{���
��{�iffC��                                    Bxl�=D  �          @�z�@5�7
=�Mp��33C�\)@5�333��  �]�C�
                                    Bxl�K�  �          @��\@9���,���G
=�ffC�}q@9����R��=q�W�HC���                                    Bxl�Z�  �          @��\@<(��(���I���(�C��@<(������=q�V��C��q                                    Bxl�i6  �          @��@G��!��xQ��CG�C��\@G��L����(���C�O\                                    Bxl�w�  �          @�@���&ff�z�H�Ez�C��)@�ý�Q���ff
=C�Ǯ                                    Bxlۆ�  �          @�G�@	���7
=�z=q�>Q�C�}q@	�����R���\�{C�ٚ                                    Bxlە(  �          @�=q?�
=�����z��s��C��?�
=?.{��=q=qAʸR                                    Bxlۣ�  �          @�G�<��
�����z�{C��
<��
?���p�{B���                                    Bxl۲t  �          @�  ?^�R���R���RW
C�8R?^�R?����{k�Bqff                                    Bxl��  �          @�z�>.{�fff���� =qC�S3>.{@z���{p�B���                                    Bxl���  �          @�p�������
=���
©CjW
����@!���Q��w�B���                                    Bxl��f  �          @�ff�@  �����
¡u�CS�Ϳ@  @\)�����u�RB�\)                                    Bxl��  �          @�ff���;��
���\��CD@ ����@%��p��k�
B��                                    Bxl���  �          @��R?�>L�����8R@��?�@9����ff�K
=Bf�                                    Bxl�
X  �          @�ff?Tz��\��z��C�\?Tz�@z�����w�\B��q                                    Bxl��  �          @�����������C�b���?�
=��\)z�B�Ǯ                                    Bxl�'�  �          @�p�<��5��33¤��C��<�@G�����Q�B��q                                    Bxl�6J  �          @�z�>�>����Q�¥p�B0�>�@N{��{�P�RB���                                    Bxl�D�  �          @�zᾞ�R��z���33z�C��׾��R?����ffaHB�                                    Bxl�S�  �          @�(��Ǯ��Q�����3C~�=�Ǯ?�ff��(�\)B�aH                                    Bxl�b<  �          @�zἣ�
��\)������C������
?������(�B���                                    Bxl�p�  �          @��
�L�ͿL����  ¢�C���L��@�����k�B��H                                    Bxl��  �          @�33>k��\(���� �qC�o\>k�@��z��B��                                     Bxl܎.  �          @��\@   ���
�����s�C�L�@   ?�33����~��A�                                      Bxlܜ�  �          @��@!G���  ���
�Z{C��H@!G�?p�������gQ�A��
                                    Bxlܫz  �          @��
?\������Q�L�C�q�?\?��
������BFG�                                    Bxlܺ   �          @��H?�G��z�H��{�C�t{?�G�?�����ff�}(�BI=q                                    Bxl���  �          @��?�
=��=q��  C�Y�?�
=?�  ��  ��B!�\                                    Bxl��l  �          @�G�?��H��\)��=qQ�C��?��H?�(���{ǮB=q                                    Bxl��  �          @���?ٙ������(���C��)?ٙ�?����ff{B�                                    Bxl���  �          @��R?�zῃ�
���H�C��?�z�?�����33�{G�B@��                                    Bxl�^  W          @��R?�p��aG����u�C��\?�p�@   ��G��tG�BD��                                    Bxl�  
�          @�
=?���33���\�C��)?�@#�
��{�l{B��q                                    Bxl� �  �          @��?�33��=q��\)�C�b�?�33@-p������iQ�B�                                    Bxl�/P  "          @���?Q녾�  ���\¢�C���?Q�@1����H�lffB��                                    Bxl�=�  "          @��?L��=L����33£�@g�?L��@B�\��ff�`B��                                    Bxl�L�  
�          @�>��R?   ���¨Bh33>��R@[����R�Q�\B���                                    Bxl�[B  
Z          @�>\)������±
=C�&f>\)@@  ��G��gp�B��\                                    Bxl�i�  
�          @�G����
?L����p�£33B�\)���
@e������D��B�                                    Bxl�x�  �          @�����\@�������B�.��\@�
=�P  �ffB�                                    Bxl݇4  
�          @��\�#�
@�\���R=qB��{�#�
@����S�
��HB�B�                                    Bxlݕ�  
�          @��>�33?�Q����\�B�ff>�33@����e���B��{                                    Bxlݤ�  "          @��?fff?�33�����
Bu�\?fff@�
=�e���\B��                                    Bxlݳ&  �          @�(�?Y��?��
����\B��?Y��@�=q�`  ��
B��)                                    Bxl���  "          @��\?G�?����{=qB��)?G�@��
�XQ��33B��q                                    Bxl��r  �          @�(�?Y��@ ����{aHB�33?Y��@�\)�S�
��
B��                                    Bxl��  T          @��
?��\@=q���R�x�HB�.?��\@�{�9����G�B��                                    Bxl���  �          @�>�p�>�����§�BM��>�p�@O\)��\)�Q�\B���                                    Bxl��d  T          @�ff?���?�(���p�Br��?���@����G
=��B�.                                    Bxl�
  �          @�
=?У�@���=q�v��BM33?У�@�G��>�R�{B��                                    Bxl��  T          @�{?��?�z���p��{B<ff?��@����QG��z�B�.                                    Bxl�(V  �          @�(�?�{?����z�Q�BA��?�{@z�H�S�
�(�B���                                    Bxl�6�  "          @��?��\?��H���HL�Bl��?��\@����J�H�\)B��                                    Bxl�E�  �          @�p�?G�@33���B�B�?G�@��
�Dz��
z�B��3                                    Bxl�TH  �          @�  ?xQ�@=q���H�v�B�\?xQ�@�(��333����B�=q                                    Bxl�b�  "          @�  ?�G�@	�����R�)B��?�G�@�
=�B�\�p�B�u�                                    Bxl�q�  "          @��\?�z�@�R��\)�|�By�H?�z�@����@���B�L�                                    Bxlހ:  �          @���?���@(������r{Bz�?���@�(��0����B�                                    Bxlގ�  T          @�{?���@
=q���\�{��Bs�?���@���;����B��                                     Bxlޝ�  �          @�p�?���@�����
Bx��?���@��H�C�
�	�B�8R                                    Bxlެ,  �          @�z�?L��@
=��(�u�B�ff?L��@�z��@  �B�W
                                    Bxl޺�  "          @��?p��@�������t�B��f?p��@�Q��)����p�B�Q�                                    Bxl��x  "          @�G�?E�@�R��33�r33B��q?E�@���#�
��(�B��                                    Bxl��  T          @�Q�?u?�Q����z�Bq��?u@����Mp��=qB�.                                    Bxl���  �          @�33=�G�@33����qB��=�G�@�33�C�
�z�B�(�                                    Bxl��j  T          @��
?B�\@#�
�����p�B��H?B�\@����$z���RB�ff                                    Bxl�  T          @�33?���@#33���\�kz�B��?���@�33� ����  B��
                                    Bxl��  "          @�(�?h��@)����=q�i��B�
=?h��@������B���                                    Bxl�!\  
�          @�p�?}p�?�Q���G��{BnQ�?}p�@~�R�I����B�\                                    Bxl�0  �          @���?Q�?s33��
=�BD��?Q�@e��z=q�:�HB�33                                    Bxl�>�  
�          @�G�?��
@33��G��q�
BDG�?��
@�G��=p��
=B���                                    Bxl�MN  
�          @�ff?���?�(���  {BQ�\?���@��
�S�
�B�\)                                    Bxl�[�  T          @�=q?��R?��
�����BH{?��R@�ff�U�G�B���                                    Bxl�j�  �          @��R?��?�z���Q��BHG�?��@��\�Vff�z�B�B�                                    