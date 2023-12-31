CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230724000000_e20230724235959_p20230725021615_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-07-25T02:16:15.299Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-07-24T00:00:00.000Z   time_coverage_end         2023-07-24T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�S!�  �          @��@���@#33�   ���\A��
@���@��Q���\)A�R                                    Bx�S0&  �          @��@�@-p����H��=qB��@�@%�
=���A��R                                    Bx�S>�  
(          @�Q�@|(�@.{�33��p�B
�@|(�@&ff�����B                                    Bx�SMr  T          @��
@|(�@!��!���
=B{@|(�@���*�H���A��                                    Bx�S\  
�          @�\)@�z�?�(�����ˮA�=q@�z�?�����ә�A�Q�                                    Bx�Sj�  �          @�=q@���?�\)������ATQ�@���?��\��Q�����A@z�                                    Bx�Syd  T          @���@|(�@_\)���H��{B&  @|(�@X�ÿ�z���33B"��                                    Bx�S�
  �          @���@i��@p�׿���Q�B7�@i��@i���G�����B4�                                    Bx�S��  T          @���@}p�@HQ��
�H��{B=q@}p�@@  �ff��(�B�\                                    Bx�S�V  "          @�\)@�33@]p���  �)��B �\@�33@\(���ff����B�                                    Bx�S��  
�          @�ff@w
=@N�R?
=@�\)B�\@w
=@P��>��@�33B �\                                    Bx�S¢  
�          @��\@�(�@ ��?�33ADz�A�  @�(�@%�?�  A*�HA�R                                    Bx�S�H  �          @�G�@�@-p�?p��AG�A�z�@�@0��?G�A=qA�{                                    Bx�S��  "          @�G�@c�
@@  @�\A��B p�@c�
@HQ�@
=A���B%33                                    Bx�S�  
�          @��@i��@B�\@�A�B{@i��@J�H@�A�=qB#��                                    Bx�S�:  �          @��@,(�@;�@xQ�B1G�B>(�@,(�@J=q@l��B'{BF�                                    Bx�T�  �          @���?W
=@>�R@���Ba��B���?W
=@P  @�
=BTp�B��                                    Bx�T�  �          @�=q��p�@�@��Bv��B�Ǯ��p�@$z�@�p�Bjp�B�aH                                    Bx�T),  �          @���?�?�p�@��RB�ffB���?�?��@��
B��)B�\                                    Bx�T7�  �          @�p�@-p�����@��Bc�C�5�@-p��L��@���Bhp�C�Ǯ                                    Bx�TFx  
Z          @��R��\?B�\@���B��C���\?��@��RB��C\                                    Bx�TU  T          @��Ϳ��<��
@��
B��C3\)���>���@��B��
C&�                                     Bx�Tc�  
�          @�p�?�����@��
B��C��?���\)@�(�B���C��)                                    Bx�Trj  
�          @�  ?��Ϳ�33@���B�  C���?��ͿW
=@��RB��3C�J=                                    Bx�T�  
�          @�Q�@Tz���@8��B�\C���@Tz��ff@AG�BQ�C��q                                    Bx�T��  
�          @��\@g
=�fff?W
=AffC���@g
=�b�\?���A>ffC��\                                    Bx�T�\  T          @�{@J=q�z=q@I��B =qC�y�@J=q�mp�@X��B=qC�:�                                    Bx�T�  �          @��H@@���XQ�@hQ�B�HC�ٚ@@���I��@uB%p�C��                                     Bx�T��  "          @��
@L(��~�R@5�A�z�C�]q@L(��r�\@E�A���C�
=                                    Bx�T�N  T          @�ff@Mp��|��@B�\A��\C��3@Mp��p  @Q�BffC�O\                                    Bx�T��  
�          @�p�@'��p��@l��B�RC�h�@'��aG�@|(�B'�C�P�                                    Bx�T�  T          @���@<(��W
=@u�B"ffC��q@<(��G
=@���B-=qC���                                    Bx�T�@  "          @�\)�
=q��=q@���B�L�CsJ=�
=q�8Q�@��\B��{Ci{                                    Bx�U�  
�          @���>L�Ϳ.{@��RB��HC�` >L�;��R@�  B���C���                                    Bx�U�  
�          @��
?\)��(�@��B�� C�0�?\)��\)@��\B�p�C�g�                                    Bx�U"2  "          @�Q�?��R�޸R@�
=B��)C���?��R��Q�@�=qB��qC�]q                                    Bx�U0�  
�          @�p�?��׿��@���Bo�\C��?��׿�\)@���B{\)C�,�                                    Bx�U?~  �          @^�R?fff��33@9��Bd(�C�Y�?fff���H@@��Bq33C���                                    Bx�UN$  
�          @�p�?��H���@`��Be�RC�K�?��H��33@hQ�Br=qC�3                                    Bx�U\�  "          @�\)@���7
=@qG�B3�C�AH@���'
=@|��B?G�C��
                                    Bx�Ukp  �          @�=q@Y��� ��@^{B�\C�Ǯ@Y���G�@hQ�B&\)C��                                    Bx�Uz  �          @���@��\�
=q@AG�B��C�@��\����@J=qB�\C�7
                                    Bx�U��  �          @���@���p�@XQ�B�
C��\@����R@b�\Bp�C��                                    Bx�U�b  �          @�Q�@vff�'
=@tz�B��C��@vff�ff@\)B&
=C�L�                                    Bx�U�  �          @��R@Z=q�XQ�@`��B�RC��H@Z=q�H��@o\)B�C��R                                    Bx�U��  T          @���@HQ��vff@Z=qB	�HC���@HQ��g
=@j=qB�C�w
                                    Bx�U�T  
�          @�=q@[��dz�@`��BG�C���@[��Tz�@p  Bz�C��)                                    Bx�U��  �          @�(�@XQ��p��@\(�B	{C���@XQ��`��@l(�B�RC��{                                    Bx�U�  �          @��
@Q��`��@q�B��C���@Q��O\)@���B$p�C���                                    Bx�U�F  T          @�(�@Z=q�U�@uB�\C�ٚ@Z=q�C33@�=qB&�C�R                                    Bx�U��  �          @���@mp��7
=@\)B!�C�8R@mp��$z�@�B+33C��=                                    Bx�V�  �          @�z�@w��\��@QG�B
=C�  @w��Mp�@`��B��C�%                                    Bx�V8  T          @��
@�  �Q�@P  B ��C�N@�  �C33@^�RB
�RC�]q                                    Bx�V)�  "          @��H@�=q�J�H@Mp�A�p�C�f@�=q�;�@[�B	�\C�R                                    Bx�V8�  
�          @��@�p��QG�@G
=A�p�C��=@�p��B�\@UB�C��                                    Bx�VG*  "          @�\)@\)�g�@FffA�p�C��=@\)�XQ�@VffB��C��)                                    Bx�VU�  T          @ə�@y���|(�@>{A�C�Q�@y���n{@P  A��HC�*=                                    Bx�Vdv  
�          @ə�@��dz�@B�\A�C�Ǯ@��U@R�\A���C��R                                    Bx�Vs  
�          @ȣ�@�(��Mp�@,��A�
=C���@�(��@  @;�A�C��=                                    Bx�V��  
�          @˅@��?\)@z�A��HC�H@��5�@�\A�p�C��\                                    Bx�V�h  
�          @�(�@��H�E@G�A���C�XR@��H�:�H@\)A�{C�{                                    Bx�V�  "          @�z�@����9��@(��A���C��@����,(�@6ffAӅC���                                    Bx�V��  
�          @˅@��@  A��HC�T{@��
�H@�HA�{C��                                    Bx�V�Z  �          @�(�@�=q��R?�{A��C�H@�=q��@33A���C��                                    Bx�V�   
Z          @�33@�=q�>{@{A��
C��\@�=q�2�\@(�A�
=C���                                    Bx�V٦  4          @�33@�(��e@\)A�33C��@�(��Z=q@   A�=qC���                                    Bx�V�L  
H          @��@�G��Q�?���A��\C�w
@�G��HQ�@�
A�p�C��                                    Bx�V��  
l          @��@��
�e�?�p�A33C���@��
�\(�@   A�Q�C�`                                     Bx�W�  
H          @�33@�  �mp�?��HA�=qC�f@�  �b�\@\)A��
C��                                    Bx�W>  T          @�33@�{�`  ?���A�{C�U�@�{�U@�A�ffC��                                    Bx�W"�  
�          @˅@��Tz�?ǮAdQ�C��@��L(�?���A�G�C�+�                                    Bx�W1�  
�          @��H@��\�u�?�33ALz�C��f@��\�mp�?ٙ�Ax��C�9�                                    Bx�W@0  	�          @ʏ\@�(��h��?��AqG�C���@�(��`  ?�A�  C�1�                                    Bx�WN�  
�          @�33@�{�p��?��A>�HC�]q@�{�h��?�{Aj�\C�˅                                    Bx�W]|  
z          @ʏ\@�=q�w
=?��ADz�C���@�=q�o\)?�33Aq�C�R                                    Bx�Wl"  
(          @��@�
=�z=q?�z�AO�C�4{@�
=�q�?�(�A}�C��=                                    Bx�Wz�  T          @ʏ\@~{��p�?���AK\)C�1�@~{��G�?�  A�C���                                    Bx�W�n  
�          @ə�@�G���Q�?!G�@�  C�f@�G��|(�?s33A�C�AH                                    Bx�W�  
(          @�
=@����w
=?k�AQ�C���@����qG�?�p�A7
=C�޸                                    Bx�W��  
�          @ə�@����vff?L��@�33C��=@����qG�?�\)A#�
C�33                                    Bx�W�`  
:          @�=q@�  �mp�?J=q@�\)C��{@�  �h��?��A z�C���                                    Bx�W�  
�          @��@�p���(�?W
=@���C�B�@�p�����?�
=A,z�C���                                    Bx�WҬ  
�          @ʏ\@�  ���
>L��?��C��=@�  ���H?\)@��C�                                    Bx�W�R  	�          @ȣ�@������=�?��C���@����  ?   @�=qC��q                                    Bx�W��  "          @�ff@�����>8Q�?�z�C�+�@����(�?��@��C�E                                    Bx�W��  "          @�ff@�p��p��>u@�C�Q�@�p��n�R?��@�(�C�p�                                    Bx�XD  
�          @�
=@����g
==�G�?��
C�5�@����e>�
=@w�C�K�                                    Bx�X�  T          @��H@�  �s�
�k��
�HC���@�  �tz�=���?h��C���                                    Bx�X*�  T          @��H@��H�j�H��{�K�C�h�@��H�l(����
�B�\C�Y�                                    Bx�X96  
�          @�=q@�����  �z���Q�C�B�@������þk��p�C�"�                                    Bx�XG�  �          @\@�\)���\���H��(�C�+�@�\)��33��G���ffC��                                    Bx�XV�  "          @�(�@�����(����
��RC��)@�����ff�#�
��Q�C�~�                                    Bx�Xe(  
�          @�z�@�G��\)�����)G�C�Y�@�G���=q�E���\)C�                                    Bx�Xs�  
�          @Å@�p��n�R���R�<Q�C��\@�p��u��h���	�C�T{                                    Bx�X�t  T          @Å@��H��=q����K\)C�g�@��H��p��xQ��33C��                                    Bx�X�  T          @�=q@�33��������pz�C�K�@�33��G���Q��5��C��                                     Bx�X��  T          @���@z=q��(�����H��C��q@z=q��\)�h���
�HC���                                    Bx�X�f  
�          @���@xQ����������C�j=@xQ����H��G��hz�C��f                                    Bx�X�  
�          @�z�@z=q�����\��{C��@z=q������z����C�                                      Bx�X˲  
�          @��H@w��s33�/\)��ffC��q@w���G��Q����\C��H                                    Bx�X�X  "          @Å@~�R�p���-p�����C�P�@~�R�\)�ff��p�C�t{                                    Bx�X��  "          @���@��\�Fff�X����HC�]q@��\�Y���E��\)C�{                                    Bx�X��  "          @�
=@s�
�/\)�����"33C�#�@s�
�Fff�p  �=qC�n                                    Bx�YJ  "          @�z�@J=q��
��{�H�C���@J=q�.�R���R�;\)C��
                                    Bx�Y�  �          @Å@i���0�������%p�C�q�@i���HQ��p  ��C��{                                    Bx�Y#�  �          @���@@���;������6�\C���@@���U���  �'  C��                                    Bx�Y2<  "          @��H@>�R�1������@�C�|)@>�R�L(�����0�RC�}q                                    Bx�Y@�  "          @�
=@<(��
=q���R�P�HC�ٚ@<(��&ff��\)�C(�C�E                                    Bx�YO�  �          @�\)@E��?\)�c33�ffC��@E��S�
�P  ��C�q�                                    Bx�Y^.  
Z          @�  @Y����(��333��  C�h�@Y�����k��p�C�J=                                    Bx�Yl�  
:          @���@J�H��(��Q��   C��@J�H����{�UC�^�                                    Bx�Y{z  
�          @�
=@HQ���33�%��ͅC��q@HQ����\�������C�+�                                    Bx�Y�   �          @�@A������6ff��\)C�ff@A�����(���
=C��                                    Bx�Y��  
�          @�{@7��<����
=�7  C�3@7��Vff�z=q�&Q�C�B�                                    Bx�Y�l  
�          @�{@Q��N�R�n�R�  C���@Q��e��X����C�@                                     Bx�Y�  
Z          @�(�@7
=�\���o\)�ffC��3@7
=�s33�XQ��z�C�w
                                    Bx�Yĸ  �          @��\@Q��|(��[����C���@Q������AG�����C��
                                    Bx�Y�^  T          @�33@P  ����'��ԏ\C�'�@P  ��������  C�]q                                    Bx�Y�  
�          @��
@R�\���
���H�g\)C���@R�\��  �}p���C�`                                     Bx�Y�  �          @��
@fff����
=q���
C��@fff��Q�޸R���\C�R                                    Bx�Y�P  �          @��@����R�<�����HC���@�����   ��C���                                    Bx�Z�  T          @���?�=q�mp���G��4(�C���?�=q����h���\)C��H                                    Bx�Z�  "          @��@1G��j=q�`  �=qC���@1G���  �Fff�{C�U�                                    Bx�Z+B  "          @�
=?�
=����E���C�g�?�
=���R�'
=�ܸRC��                                    Bx�Z9�  �          @�(�@����  ��  ��{C��H@�������(��AC�k�                                    Bx�ZH�  "          @��@
=q��=q�   �ʣ�C�\@
=q������p�����C��\                                    Bx�ZW4  "          @�  @G��j�H�Y�����C��q@G���Q��@  �33C�                                    Bx�Ze�  �          @�  @�H�j=q�i���{C���@�H�����O\)�
�C���                                    Bx�Zt�  �          @�ff@H�ÿ�Q�����Iz�C���@H���������<��C��                                     Bx�Z�&  �          @�33@����Q���33�3Q�C�y�@���L����G��/�HC��                                     Bx�Z��  "          @�\)@���>�(��z�H�&Q�@��\@���<#�
�|(��'�=�Q�                                    Bx�Z�r  "          @��H@�=q?���tz��'  A��@�=q?k��|���-��AK33                                    Bx�Z�  �          @�p�@W�?�G��w��<Q�A�{@W�?Tz��~�R�C�RA^ff                                    Bx�Z��  "          @��@  >�z���(���@�@  �aG���(�
=C�B�                                    Bx�Z�d  T          @��
@^{��녿s33�c�C��@^{���R�J=q�<��C�`                                     Bx�Z�
  "          @���@.{����|���\C���@.{�W
=�xQ��W{C�`                                     Bx�Z�  �          @�G�?�33������  �t�
C�K�?�33������=q�d��C�.                                    Bx�Z�V  
�          @���?�(�@�\��=q�\z�B7�R?�(�?��������m�RB=q                                    Bx�[�  �          @�
=@��@N{�]p��'Q�Baz�@��@3�
�r�\�<=qBR�H                                    Bx�[�  T          @��?�z�@Vff�n�R�6�B�B�?�z�@:=q���H�NffB�33                                    Bx�[$H  "          @�(�?�z�@G
=�Z=q�,
=Bi�?�z�@-p��o\)�A�B[
=                                    Bx�[2�  T          @�{��  @�(��7����BՔ{��  @qG��Tz��z�B؅                                    Bx�[A�  T          @���?�  @,(��p  �N�HB��?�  @\)��G��f33Bsff                                    Bx�[P:  f          @��@S�
�Ǯ�qG��B
=C��)@S�
�Q��l���=G�C�                                    Bx�[^�  �          @���@z=q>�p��y���3
=@��@z=q���
�z=q�4  C�b�                                    Bx�[m�  T          @��@�
=��p��^{�33C�T{@�
=��\)�Q��ffC��                                    Bx�[|,  �          @��@�p��&ff�l(��$�\C���@�p������e�(�C��\                                    Bx�[��  �          @�@z=q>aG��e��)@K�@z=q�L���e��)�
C���                                    Bx�[�x  �          @���@u?�\�i���-=q@�(�@u=��
�l(��/Q�?�p�                                    Bx�[�  t          @��R@��ÿ=p��e��$p�C���@��ÿ�z��^{�=qC��
                                    Bx�[��  "          @��@����.{�\)��z�C�c�@����>{��33���C�:�                                    Bx�[�j  �          @���@�
=� ���:�H��C�=q@�
=�ff�*=q��\C�t{                                    Bx�[�  �          @��H@x���H���������C��@x���Y���   ��\)C�j=                                    Bx�[�  �          @��@��H����l(�� \)C�
@��H���
�e�G�C�J=                                    Bx�[�\  �          @�=q@��H�c�
�o\)� z�C�0�@��H����fff��\C�p�                                    Bx�\   
�          @��@��ÿL���C�
�=qC�Ff@��ÿ�z��<����G�C�*=                                    Bx�\�  "          @�\)@�G����Z�H�\)C�  @�G��^�R�U�Q�C���                                    Bx�\N  �          @��
@�Q�!G��QG���C�@�Q쿃�
�J�H�	��C��{                                    Bx�\+�  
�          @��\@��
����
=���RC��@��
��녿����
C��                                    Bx�\:�  
l          @��@��H�9�����H�P��C�"�@��H�AG��W
=�z�C��\                                    Bx�\I@  
H          @�\)@�G��a녿����HC�n@�G��dz����{C�AH                                    Bx�\W�  
�          @�  @���k�>L��@33C��@���hQ�?&ff@ۅC�q                                    Bx�\f�  "          @��R@��G�=��
?fffC���@��E>��@��RC��f                                    Bx�\u2  "          @�33@{��HQ�?\(�A(�C���@{��@  ?�  Ac�C�W
                                    Bx�\��  
�          @���@l���^{?�  A]p�C�l�@l���Q�?�
=A�{C�5�                                    Bx�\�~  T          @�@~�R�`  ?�ffA3�C�Q�@~�R�U?��RA�  C���                                    Bx�\�$  
Z          @�  @}p��n{?�G�A�33C�aH@}p��^{@{A���C�c�                                    Bx�\��  �          @�\)@|���C�
@\)A�
=C��@|���0  @'
=A�(�C��\                                    Bx�\�p  
(          @�33@u��   @.{A��C�k�@u��Q�@AG�B
Q�C�w
                                    Bx�\�  T          @�{@qG��z�@:=qBG�C���@qG���
=@J=qB��C��                                    Bx�\ۼ  
�          @��@���L��@#�
A���C�� @���5@=p�A���C��                                    Bx�\�b  T          @�ff@l(��a�@
=AƏ\C�'�@l(��L(�@3�
A�33C���                                    Bx�\�  4          @��
@r�\�l��@"�\A��C��)@r�\�U�@@��A�Q�C�Y�                                    Bx�]�  
�          @�z�@J=q��=q@,��A��HC��=@J=q�k�@Mp�B
=C�W
                                    Bx�]T  
�          @��@QG�����@z�A��
C��q@QG��|��@7�A�C���                                    Bx�]$�  
�          @��@^�R���\@�A�33C�33@^�R�p��@-p�A�33C�c�                                    Bx�]3�  T          @���@�G��[�@"�\A��HC��
@�G��C�
@>�RA�C�q�                                    Bx�]BF  
�          @�33@u��o\)@ffA�\)C�� @u��X��@5�A�C�G�                                    Bx�]P�  �          @���@��\�J�H@��A�(�C���@��\�6ff@#33A�C�P�                                    Bx�]_�  �          @�33@����U@
=qA��
C��@����@��@&ffA��C�s3                                    Bx�]n8  "          @��@u��z=q@p�A��
C�:�@u��dz�@.{A�C��f                                    Bx�]|�  �          @��
@^{����?�p�A�(�C��=@^{�}p�@#33A�z�C��)                                    Bx�]��  �          @�=q@l(��~{@�
A�Q�C�s3@l(��i��@%A��C���                                    Bx�]�*  
�          @��@r�\����?�\)A���C���@r�\�p  @=qA£�C���                                    Bx�]��  
�          @�=q@y���tz�?�A�
=C���@y���aG�@�A���C��q                                    Bx�]�v  
Z          @��@j�H���H?�ffAs\)C��@j�H���H@��A�=qC���                                    Bx�]�  
�          @�z�@|(����?�z�A7�C��R@|(��}p�?޸RA�=qC�p�                                    Bx�]��  "          @��@S�
��?��A3�
C��q@S�
��\)?��A��\C�<)                                    Bx�]�h  T          @�z�@_\)��p�?=p�@�C�b�@_\)����?�33A[�
C��
                                    Bx�]�  "          @�=q@@������?�\)AZffC�q@@������@�\A�=qC���                                    Bx�^ �  "          @��
@333��\)?���ATQ�C���@333���@33A�33C�K�                                    Bx�^Z  
�          @��R@Tz���33?�A�G�C��@Tz�����@��A�Q�C���                                    Bx�^   �          @�\)@qG����
?У�A|z�C�c�@qG����H@\)A��
C�U�                                    Bx�^,�  �          @�
=@b�\��Q�?���Axz�C��@b�\���@�RA�(�C���                                    Bx�^;L  �          @�G�@Z=q���\@=qA�33C��@Z=q�|(�@AG�A��HC�u�                                    Bx�^I�  
�          @�33@Vff��z�@$z�A��
C���@Vff�}p�@L(�A�Q�C�)                                    Bx�^X�  
Z          @Å@W
=��Q�@Q�A��C�Y�@W
=���@AG�A���C���                                    Bx�^g>  	�          @\@Mp����@"�\AŅC��f@Mp����@J�HA��
C�"�                                    Bx�^u�  
�          @\@?\)���R?��RA�G�C�8R@?\)���@*�HA�(�C�:�                                    Bx�^��  �          @��@C33���?�(�A���C���@C33��33@(�A�z�C��f                                    Bx�^�0  	�          @���@A���G�@L��BG�C�h�@A��aG�@p��B�C�Y�                                    Bx�^��  �          @���@8Q���  @H��A��C�@8Q��n�R@p  B�
C��{                                    Bx�^�|  �          @�33@Mp���G�@!G�A��C��@Mp���33@K�A���C�                                    Bx�^�"  �          @�z�@]p�����@*=qA��
C�j=@]p��vff@Q�Bz�C��)                                    Bx�^��  "          @�z�@Q����?���A"=qC��@Q�����?�A�G�C�>�                                    Bx�^�n  
(          @���@e���
=?���A�p�C�]q@e���(�@ ��AŅC�w
                                    Bx�^�  �          @���@Q���\)@&ffA���C�q@Q�����@P��B p�C��
                                    Bx�^��  
�          @�p�@U��G�@EA�\C�˅@U�`��@k�B�
C��f                                    Bx�_`  "          @�@\(��|��@H��A�33C���@\(��Z�H@mp�B��C���                                    Bx�_  T          @���@\�����@>{A�C��@\���fff@dz�B�\C��f                                    Bx�_%�  
�          @��@`  �p��@L��A���C�t{@`  �Mp�@p  B(�C��3                                    Bx�_4R  
(          @Å@aG��Q�@o\)BG�C���@aG��*=q@��RB.�C�~�                                    Bx�_B�  "          @Å@_\)�K�@w�BQ�C��R@_\)�!�@�=qB4=qC�
=                                    Bx�_Q�  
�          @��@w
=�N�R@\��B	�HC��@w
=�(��@z�HB�C��=                                    Bx�_`D  "          @�
=@fff�Q�@Z�HB33C���@fff�,��@x��B#�\C��3                                    Bx�_n�  
�          @�p�@;����@�{B?�C�Z�@;��ٙ�@���BT�C��3                                    Bx�_}�  T          @��
@Q��G�@��HBl
=C�.@Q쿀  @�=qB��C�`                                     Bx�_�6  
(          @�{�aG�>���@�
=B��C�H�aG�?�\)@�33B�33C 5�                                    Bx�_��  
Z          @�{�}p�?�R@�Q�B���C  �}p�?�@��HB��qB��q                                    Bx�_��  "          @�
=��?.{@��RB�.C!�f��?�
=@���Bs��Ch�                                    Bx�_�(  �          @��H�xQ쾽p�@�\)B��=CH녿xQ�>��H@�
=B��=Ch�                                    Bx�_��  "          @��H�}p��\)@�
=B�G�C;�{�}p�?8Q�@�B�\)C{                                    Bx�_�t  T          @��}p��\@��B�ffCI\�}p�>�@��B��qC33                                    Bx�_�  
�          @�z�aG����@��B��=CxB��aG�>�z�@��\B��)B�aH                                    Bx�_��            @����#�
���@�Q�B�
=C�Uü#�
>�@�Q�B��fB�\)                                    Bx�`f  
�          @��H?�p��aG�@�p�B�p�C���?�p�����@�  B�G�C�&f                                    Bx�`  
�          @�z�?�
=��{@�33B�ǮC���?�
=��@�Q�B�8RC��)                                    Bx�`�  	�          @���?�  ���R@���B��C���?�  �5@��RB�#�C�c�                                    Bx�`-X  �          @��R@���  @��\B_33C�@���  @�=qBr�C��3                                    Bx�`;�  �          @�Q�@(�ÿ5@�p�Bi�C�y�@(��=L��@�\)Bm�?��\                                    Bx�`J�  �          @�33@mp��u@r�\B6\)C�+�@mp�>��@q�B5p�@ə�                                    Bx�`YJ  "          @���@�ff?   @QG�B
=@��@�ff?�ff@H��B�A`��                                    Bx�`g�  
�          @��?��@33@���BX�\BP�?��@=p�@e�B8
=Bkff                                    Bx�`v�  �          @�=q>�
=?��R@�G�B��qB�L�>�
=@   @�Q�B���B�{                                    Bx�`�<  "          @�33@G�����@j�HBT
=C��=@G��k�@x��Bg  C��                                    Bx�`��  
�          @��@;���@VffB&p�C���@;���G�@mp�B=\)C���                                    Bx�`��  
l          @~�R?�z�?��\@H��B]��A��?�z�?�ff@:�HBG�RB=q                                    Bx�`�.  
�          @���?�z�?޸R@Y��BR�RB)G�?�z�@�
@B�\B5�\BIQ�                                    Bx�`��  "          @�ff?�녿Y��@p��B�.C�:�?�녾8Q�@vffB�\C�J=                                    Bx�`�z  
�          @�Q�?녿�p�@��RB�B�C�s3?녾��@��B��3C�4{                                    Bx�`�   �          @~{=��
�u@|(�B���C�%=��
>�@z�HB�u�B�\)                                    Bx�`��  	�          @���>.{���@w�B���C��=>.{��{@\)B�L�C�#�                                    Bx�`�l  �          @�?���
=@g�BtQ�C��q?���  @y��B��=C���                                    Bx�a	  �          @��\@J=q���@/\)B  C��@J=q��@8Q�B'�HC�7
                                    Bx�a�  �          @��@W��&ff@z�B=qC���@W��aG�@��B�C�                                      Bx�a&^  "          @���@r�\��ff?��A�C���@r�\��G�?�
=A��C�,�                                    Bx�a5  "          @���@|�Ϳ�@
=A���C�>�@|�ͽ�Q�@�HA�\)C�^�                                    Bx�aC�  
          @�
=@&ff�!�@L(�B%ffC��\@&ff��z�@fffB@p�C��f                                    Bx�aRP  "          @���>�(���\)@L��B�C��R>�(��u�@}p�B7\)C�/\                                    Bx�a`�  t          @�33?�G���33@'�A���C��)?�G���G�@]p�B�\C���                                    Bx�ao�  "          @��?c�
���@Q�A�G�C��?c�
��
=@P��B�C���                                    Bx�a~B  "          @��?L�����R@��A�C��R?L����@Tz�B�C�#�                                    Bx�a��  �          @�z�?ٙ���z�@.�RA�33C�{?ٙ����@b�\B33C�b�                                    Bx�a��  �          @��
?�����R@*�HA���C��?����(�@`  BC�1�                                    Bx�a�4  "          @�=q?�����ff@"�\A�z�C�ff?�������@W�B�\C���                                    Bx�a��  �          @��?��
���R@.{A�z�C���?��
���
@c33B�\C��f                                    Bx�aǀ  �          @�z�?��R��  @0��A��HC�Q�?��R��z�@fffB
=C�S3                                    Bx�a�&  T          @�?�33��ff@>�RA��HC�ٚ?�33���@s�
B(ffC��                                    Bx�a��  
�          @�\)?����G�@P  B	��C��?���vff@�G�B3�\C�G�                                    Bx�a�r  "          @�\)?��R��=q@Mp�B��C��{?��R�xQ�@���B2�C���                                    Bx�b  T          @�ff?�(�����@S33Bp�C�@ ?�(��l��@�=qB6��C�޸                                    Bx�b�  �          @�Q�?�����\)@W
=BC�Z�?����p��@���B8�\C�ٚ                                    Bx�bd  �          @���?�������@S33B�C�z�?����s33@�33B5��C��{                                    Bx�b.
  
�          @���?:�H����@eB33C��)?:�H�`��@�33BIC��f                                    Bx�b<�  T          @�p�?^�R��(�@_\)B
=C��)?^�R�g�@�Q�BC�C��)                                    Bx�bKV  "          @���?G����\@c33B�C��?G��c�
@�=qBGQ�C�.                                    Bx�bY�            @��>#�
����@^{B
=C��>#�
�h��@�Q�BE�
C�K�                                    Bx�bh�  
�          @�p��.{���R@^�RB�HC��\�.{�l��@�G�BD�
C��
                                    Bx�bwH  
�          @�{�.{���R@]p�B(�C��Ϳ.{�l(�@���BB�C���                                    Bx�b��  
�          @��#�
��G�@j=qB �\C����#�
�_\)@�BM=qC�                                    Bx�b��  �          @�z��R���@hQ�B!=qC��\��R�\(�@���BN�C�ٚ                                    Bx�b�:  T          @���>#�
���@W
=Bz�C��>#�
�c�
@�z�BE{C�J=                                    Bx�b��  �          @��\?z���z�@W�B�C��)?z��hQ�@�p�BBp�C��                                     Bx�b��  �          @��R<��
��
=@u�B(C�%<��
�XQ�@��HBV�RC�.                                    Bx�b�,  
�          @�>�Q��~�R@\)B3�C��{>�Q��Fff@��RBa��C�N                                    Bx�b��  �          @��R?�����H@z=qB-z�C�0�?���N{@���B[�C�K�                                    Bx�b�x  �          @�ff?0���y��@���B6C��?0���@  @�  BdG�C�xR                                    Bx�b�  "          @�  ?��
�{�@���B3�RC�P�?��
�A�@�Q�B`p�C�Y�                                    Bx�c	�  T          @��?p������@l��B z�C�(�?p���\��@��BM��C��R                                    Bx�cj  �          @���?������@Y��BC�W
?���`  @�ffB;ffC���                                    Bx�c'  T          @�33?��R��
=@aG�B  C�� ?��R�Z�H@��B=C�                                      Bx�c5�  f          @�Q�>��p��@��\B<��C�y�>��5@���BkffC��{                                    Bx�cD\  
          @��
>�p��r�\@~�RB9  C��3>�p��8��@�p�Bh(�C���                                    Bx�cS  
�          @���?   �b�\@�G�BA��C�  ?   �(Q�@�BpC�Y�                                    Bx�ca�  T          @���@c�
�|(�?\A�
C�
=@c�
�b�\@33A�G�C���                                    Bx�cpN  "          @�Q�@���hQ�>���@�
=C�G�@���^{?�33A@��C��                                    Bx�c~�  �          @�=q@�(��(���G��'33C��R@�(��%����p�C�                                      Bx�c��  
Z          @�z�@�ff����'
=���C�^�@�ff�	���p�����C��{                                    Bx�c�@  
�          @�@����þ�ff��  C���@����=u?5C��f                                    Bx�c��  �          @�Q�?��R��G�@L(�B  C�s3?��R�aG�@���B3G�C��q                                    Bx�c��  �          @�@L(�����?�A��C�Q�@L(��s�
@*�HA�ffC��R                                    Bx�c�2  
�          @��R@9����z�?�\)A��C��)@9����ff@$z�A��
C�Q�                                    Bx�c��  �          @�ff@Q���\)?�p�AHQ�C��@Q����
@
=qA�Q�C�B�                                    Bx�c�~  
�          @�\)@j=q����?5@�C�H�@j=q����?���A�p�C�#�                                    Bx�c�$  4          @�Q�@�����  >�  @!�C���@����vff?���A2ffC�4{                                    Bx�d�  �          @��@��\�J=q�h���33C�b�@��\�R�\�W
=�33C���                                    Bx�dp  
�          @��
@N{��ff?�ffA�Q�C��R@N{�n{@*=qA㙚C�q�                                    Bx�d   
�          @���?������@(�Aҏ\C�J=?���z=q@W
=B�HC��                                    Bx�d.�  
�          @�33>�p���=q@H��B	�\C�XR>�p��q�@���B;ffC��{                                    Bx�d=b  �          @��H������@^{B(�C�������U�@���BKz�C|0�                                    Bx�dL  �          @����Q����@VffB��Cw�H��Q��Q�@��BA��Cr��                                    Bx�dZ�  �          @���@33���@z�A�C�&f@33��p�@G
=BG�C�n                                    Bx�diT  T          @�=q@
�H��(�@��A�\)C�  @
�H����@Q�B	�C��                                     Bx�dw�  �          @���?����33@333A��HC���?�����
@s33B%Q�C��                                    Bx�d��  �          @�G�@�\��{@   A��C�4{@�\��(�@B�\B C�z�                                    Bx�d�F  �          @��?�p���  @R�\BC���?�p��j=q@�ffB=C�S3                                    Bx�d��  �          @�{?0�����H@x��B,�\C��?0���G�@�ffB_
=C�@                                     Bx�d��  "          @�ff?@  ��p�@s�
B'�RC��?@  �N{@�z�BZ=qC���                                    Bx�d�8  �          @���?Tz����H@k�B{C�n?Tz��Y��@���BQ��C�޸                                    Bx�d��  �          @��?������@{�B*�C��?����Dz�@��B[�C��                                     Bx�dބ  �          @���?333��@w�B)ffC���?333�L��@��RB\�C�.                                    Bx�d�*  �          @��>�=q�|��@�\)B;�RC��)>�=q�8��@���BoC���                                    Bx�d��  "          @�G�>�\)���@q�B#��C��\>�\)�X��@�p�BW�HC�T{                                    Bx�e
v  �          @�Q�?���fff@��
B>z�C�s3?���$z�@��\Bo��C���                                    Bx�e  �          @�\)����G�@���Bb�
C��ᾅ�����@�33B���C�.                                    Bx�e'�  4          @�=q���c�
@�
=BM�HC��q���(�@�p�B�\)C���                                    Bx�e6h  B          @��@:�H���@���BPG�C�� @:�H���\@�Q�BiQ�C�b�                                    Bx�eE  
�          @���@(��� ��@���BY�C�XR@(�ÿc�
@��BsG�C��3                                    Bx�eS�  �          @�
=@y����=q@p  B)33C��@y����(�@|��B5  C��{                                    Bx�ebZ  
�          @��@���>�=q@XQ�B�@XQ�@���?��@N{B
p�AN�\                                    Bx�eq   �          @�ff@�
=�\@:=qA���C��R@�
=�L��@L(�BQ�C��f                                    Bx�e�  �          @�{@�G���  @333A�=qC��@�G����
@=p�A���C��                                    Bx�e�L  "          @��@��p�?���AeG�C��3@�����?�{A���C���                                    Bx�e��  �          @�z�@��Ϳ�z�?��
A|��C�{@��Ϳ�  ?�\)A��C�(�                                    Bx�e��  �          @�@��Ϳ��?!G�@�Q�C�f@��Ϳ�33?n{AG�C��
                                    Bx�e�>  
H          @���?�=q���@ffB!z�C��3?�=q����@\)BHffC�Ф                                    Bx�e��  �          @��ÿ�  �7�@��Bc�Cv����  ��33@�33B�Ch��                                    Bx�e׊  �          @����\)�G
=@�
=Bh  C�T{��\)��@�=qB�ffC��q                                    Bx�e�0  T          @�33��G��N{@�(�Bb\)C����G���(�@�Q�B���C�aH                                    Bx�e��  T          @�����Z=q@��BZp�C�����
�H@�\)B�ǮC�W
                                    Bx�f|  �          @��>���}p�@��HB=��C�c�>���3�
@��BtffC�                                    Bx�f"  �          @�����j�H@��HBLffC������p�@��HB��=C��                                    Bx�f �  �          @��׼#�
�J=q@��Ba33C���#�
��
=@��B�k�C���                                    Bx�f/n  
�          @�p���p��!�@��B{��C��;�p���  @�ffB��C}^�                                    Bx�f>  �          @�{��z��XQ�@�=qBU��C��;�z���@�Q�B���C�4{                                    Bx�fL�  �          @�ff�Ǯ�s33@�ffB?  C�;Ǯ�*�H@�  Bv�RC�Ф                                    Bx�f[`  T          @�(�=���\)@mp�B$�C�Ǯ=��L��@�(�B]33C��                                    Bx�fj  T          @�z�>u��{@mp�B%�C��H>u�J=q@��
B^=qC�(�                                    Bx�fx�  �          @�?   �fff@�G�BF�C��3?   �(�@�G�B~��C���                                    Bx�f�R  �          @�\)?��
�|��@s33B)�C���?��
�9��@��B_�C��                                     Bx�f��  �          @�ff?k��w�@n�RB,�C���?k��5�@��Bcp�C�H                                    Bx�f��  �          @���?(���g�@�{BB�C�.?(���{@��RBz�C�}q                                    Bx�f�D  T          @�Q�<#�
�aG�@�BG��C�<#�
�Q�@�B�u�C�{                                    Bx�f��  �          @����\)�Tz�@�=qBQ�RC�]q��\)�	��@�Q�B��{C�                                    Bx�fА  �          @��
�u�5@���Bm
=C�^��u�Ǯ@��\B�L�C���                                    Bx�f�6  �          @�ff>\�\(�@�  BQ�HC�(�>\�p�@��RB��=C��                                    Bx�f��  �          @��Ϳ��g�@�G�BFffC������@��Bp�C��                                     Bx�f��  �          @��?�G��=p�@��BR�HC���?�G���\@��RB�ǮC�AH                                    Bx�g(  
�          @��
@/\)���@��\BJ�C�E@/\)���@�Q�Bh��C��                                    Bx�g�  �          @�p�?#�
���
@u�B*=qC�ff?#�
�AG�@��Bc�RC��{                                    Bx�g(t  �          @�p�=���  @n�RB$�HC��=��J�H@�B_p�C�{                                    Bx�g7  T          @��=�\)��@q�B(�C���=�\)�E�@�
=BcffC���                                    Bx�gE�  �          @�\)?.{�{�@��B7z�C��\?.{�0  @�
=Bq�C��R                                    Bx�gTf  �          @�
=?���X��@�BL��C���?���	��@���B�C�                                    Bx�gc  "          @���>\�G
=@�z�B_��C���>\����@���B�#�C��                                    Bx�gq�  T          @�  ?�Q��^{@~{B={C�z�?�Q��@�\)Bs�C���                                    Bx�g�X  �          @�z�@��� ��@ffA�\)C�@����=q@2�\A��C��3                                    Bx�g��  T          @�@W��Fff@7
=B 
=C��
@W��  @c�
B%(�C�)                                    Bx�g��  
�          @��@QG��Z=q@&ffA�
=C��@QG��'�@Y��B(�C���                                    Bx�g�J  "          @�Q�@XQ��|��?�@��\C�E@XQ��k�?��A��RC�H�                                    Bx�g��  
�          @�G�@7
=�e@{AΣ�C�Ff@7
=�8��@E�Bz�C�Z�                                    Bx�gɖ  
�          @�z�@0���8��@z�HB1�
C���@0�׿�\@�G�BX�C��                                    Bx�g�<  �          @�=q@(�ÿ���@��HBb��C��R@(�þW
=@��Btp�C�                                    Bx�g��  "          @��
@.{?���@�ffBh\)A�p�@.{@��@�Q�BI  B��                                    Bx�g��  "          @���@*�H?�z�@���BiffAޏ\@*�H@(��@��BD�HB2�H                                    Bx�h.  "          @���?��@aG�@z=qB4��B���?��@�p�@6ffA�33B�B�                                    Bx�h�  �          @�(�@�
@\)@��BP(�B<�@�
@c33@c33B�Bc�R                                    Bx�h!z  �          @�\)@�?(��@���B��fAw�@�?��R@�G�Bdp�B 33                                    Bx�h0   
�          @��R@6ff�8Q�@��Bk33C��q@6ff?
=@��
Bl��A:ff                                    Bx�h>�  
�          @�\)@���ff@�\)Bv{C���@�=�\)@�z�B��?�                                      Bx�hMl  
�          @���@"�\���
@��Bd�\C���@"�\��G�@��B|C�#�                                    Bx�h\  �          @�  @7���@��BZ{C�޸@7���p�@�{Bn�C�U�                                    Bx�hj�  T          @���@.{���@�p�BlQ�C��{@.{>W
=@���Bv�\@�Q�                                    Bx�hy^  "          @�z�@����H@�Q�BlC���@���@�\)B��C�w
                                    Bx�h�  
Z          @�z�@���@��BU�HC���@녿���@�\)B{��C��3                                    Bx�h��  
�          @�  @(��U@��HB6�
C��{@(���@��HBgz�C��{                                    Bx�h�P  
�          @���?�G�����@j=qB�RC���?�G��@  @�z�BXz�C�P�                                    Bx�h��  �          @�G�?��H���R@s33B$  C��?��H�AG�@���B_��C��                                    Bx�h  �          @��H?�ff���@VffB�C�\)?�ff�e�@��BI(�C�1�                                    Bx�h�B  T          @�G�@�\�J=q@�ffB@��C�q�@�\��33@���Bq�C���                                    Bx�h��  
�          @���@(���@��Bl{C�Z�@(���@�ffB��C�@                                     Bx�h�  T          @���?�{�\)@���Bzz�C���?�{�E�@�
=B�33C�0�                                    Bx�h�4  "          @��R?xQ��S33@���BS�C�,�?xQ��
=@���B�Q�C�N                                    Bx�i�  T          @���?���mp�@���B933C�E?����@�Q�Bs=qC��                                    Bx�i�  
�          @�Q�?�  �J=q@��RBK�HC�~�?�  ��@���B�
C�f                                    Bx�i)&  
Z          @��R?���:=q@�
=B_Q�C�Z�?����  @��HB���C��\                                    Bx�i7�  
�          @��?�33�B�\@��BSG�C�:�?�33��z�@�
=B�C�ff                                    Bx�iFr  "          @��?�G��xQ�@u�B'�
C�33?�G��*=q@�  B`z�C��R                                    Bx�iU  	�          @���?�Q��g�@���B8z�C�z�?�Q��z�@�  Bpz�C���                                    Bx�ic�  
�          @���?��H�a�@�=qBA�C�AH?��H�(�@�(�B{\)C��                                    Bx�ird  
�          @�Q�?Ǯ�a�@�Q�B?p�C��?Ǯ���@��\BxG�C��=                                    Bx�i�
  f          @�Q�?�(��dz�@�z�B9{C���?�(��G�@�\)Bq  C��                                    Bx�i��  t          @���?����[�@�ffB;�C���?����
=@��Bp(�C�aH                                    Bx�i�V  
l          @��?�Q��h��@�Q�B0C��?�Q���@��
Bg�C���                                    Bx�i��  
H          @�{@$z��l(�@U�B�C�u�@$z��&ff@�\)BD�RC�\)                                    Bx�i��  "          @�@2�\�g�@P��B(�C���@2�\�#33@���B>�
C��f                                    Bx�i�H  
Z          @�ff@C33�g
=@B�\B{C��@C33�&ff@|(�B2
=C�                                    Bx�i��  �          @�  @G�����@�33BW�HC�s3@G�>��@�  Bb�@0��                                    Bx�i�  
Z          @�Q�?�z��2�\@�=qBI{C�&f?�z�\@�ffBz=qC��{                                    Bx�i�:  T          @�z�?���r�\@�ffB6�C���?����@��Bp��C���                                    Bx�j�  T          @��
?�p��^{@���BBffC�AH?�p��z�@�ffBzp�C���                                    Bx�j�  �          @��H?����#33@��\Bq�RC��3?��Ϳ}p�@�33B�C��=                                    Bx�j",  
�          @���?�
=���
@�33B���C���?�
=�.{@�(�B��\C�0�                                    Bx�j0�  �          @��@��	��@�(�Bc�C��
@��(��@���B�B�C�:�                                    Bx�j?x  	�          @���@z���@�(�Bp��C���@z����@�
=B�
=C�}q                                    Bx�jN  4          @��@
�H���@�
=Ba��C��3@
�H�L��@�p�B��qC��f                                    Bx�j\�  B          @�G�?�z�333@�(�B�z�C��?�z�?aG�@��B�A�=q                                    Bx�jkj  �          @�ff?˅>�{@��RB�G�AC33?˅?�@�(�B~{BD��                                    Bx�jz  �          @�\)@@�׿��@�\)BN�C���@@�׿   @��\Bf�RC�AH                                    Bx�j��  "          @�\)@\)����@���Bv  C�\@\)>��@��
B{Az�                                    Bx�j�\  �          @�@*�H��Q�@�{B`�C���@*�H��  @�\)Bv�HC�O\                                    Bx�j�  �          @��?�
=�/\)@��\B[C���?�
=���
@�B��C�O\                                    Bx�j��  
�          @��
?�\)�,��@���BX�C�e?�\)��  @��
B�z�C�"�                                    Bx�j�N  �          @�33?���1�@�p�Bb�C��?�����
@���B���C��\                                    Bx�j��  �          @���>���U�@�(�BS
=C�G�>�����@��B��RC�>�                                    Bx�j��  "          @�Q�G��l��@~{B9z�C���G���@�z�B{�HC{��                                    Bx�j�@  
(          @�G�?�=q�)��@�(�BV�C�` ?�=q��p�@��RB�  C�                                    Bx�j��  T          @�33@��
@  ���
�uAƏ\@��
@�ÿ0����(�A�(�                                    Bx�k�  "          @��@�ff@;���\)�7�B 
=@�ff@{��Q���
=A��                                    Bx�k2  T          @�@��H@5��O\)��A���@��H@p���\)����A��
                                    Bx�k)�  T          @��R@�
=@U���������B33@�
=@(Q��0�����A��                                    Bx�k8~  
�          @�p�@�G�@<(��ٙ����B�
@�G�@33�   ��\)A�
=                                    Bx�kG$  T          @��@�  ?�ff�p���
=A�\)@�  ?�����{�a�AMp�                                    Bx�kU�  �          @��@��׿�z������RC�
=@��׿�33>W
=@Q�C�
                                    Bx�kdp  �          @�\)@��
��  ���\�N�RC�q�@��
���ÿTz��
=C���                                    Bx�ks  �          @�\)@�ff��z��{��G�C�#�@�ff�zῃ�
�'\)C�&f                                    Bx�k��  �          @�{@�\)��׿\�x��C���@�\)�'
=�E����C�,�                                    Bx�k�b  T          @�(�@z=q�!G��G
=��C���@z=q�Vff�����p�C��
                                    Bx�k�  T          @��@`���#33�G��  C�  @`���XQ��������C�                                    Bx�k��  �          @�  @\)�8Q��x���6=qC�h�@\)�{��5���ffC�1�                                    Bx�k�T  �          @��H@#33�a��XQ��G�C��
@#33��z��	����p�C�)                                    Bx�k��  �          @�@`���H���H����
C��@`���|����\����C��R                                    Bx�k٠  
�          @���@��\�C33�z����C��H@��\�g
=���
�S\)C�AH                                    Bx�k�F  "          @�=q@~{�8Q��%�ޏ\C��@~{�a녿�=q����C�/\                                    Bx�k��  
�          @��\@w��8���1����C���@w��fff��G����C���                                    Bx�l�  
�          @�\)@���9����R����C�'�@���\�Ϳ�p��H  C�Ф                                    Bx�l8  �          @��@�G��#�
?aG�A��C��@�G��
=q?�\)A���C�b�                                    Bx�l"�  �          @��\@��H�'
=?��A%G�C�g�@��H�
=q?�ffA�{C�y�                                    Bx�l1�  �          @�\)@����1�?У�A}�C�� @����	��@=qA�Q�C�o\                                    Bx�l@*  f          @��R@�(��%?˅Aw33C��)@�(���(�@�
A�z�C�}q                                    Bx�lN�            @�{@��H�'�?�=qAw\)C�g�@��H�   @z�A���C�K�                                    Bx�l]v  �          @�p�@�=q�)��?��RAiC�5�@�=q��
@\)A�{C��R                                    Bx�ll  �          @��
@��R�@  ?��A�Q�C���@��R�33@(��A��
C��q                                    Bx�lz�  �          @�=q@��H�C33?�{A���C�4{@��H�z�@.{A�33C��q                                    Bx�l�h  �          @���@�p��C�
?���AY��C�Z�@�p��\)@\)A���C���                                    Bx�l�  
          @��\@��R�=p�?�Q�A��C��@��R��\@!�A�ffC�\                                    Bx�l��  
�          @�  @�Q��<(�?�(�A�ffC�p�@�Q���@2�\A��HC�)                                    Bx�l�Z  
�          @���@|(��Mp�@�A̸RC�k�@|(���@Tz�BQ�C��{                                    Bx�l�   4          @�@XQ��HQ�@W
=B�C���@XQ��z�@�B<G�C�=q                                    Bx�lҦ  B          @��@x���7
=@>�RA�G�C�Ф@x�ÿ��
@n�RB$Q�C��{                                   Bx�l�L  T          @�=q@�33�=p�@(Q�A�C�3@�33��p�@[�B��C��                                   Bx�l��  T          @��@o\)�L(�@J�HB33C���@o\)�G�@�Q�B-Q�C���                                    Bx�l��  �          @�(�@y���K�@AG�A�C�h�@y����
@w�B%{C��                                    Bx�m>  �          @��H@u��N{@=p�A��C��q@u��
=@u�B$�
C��3                                    Bx�m�  �          @���@|(��L��@0  A㙚C�l�@|(��
=q@hQ�B��C���                                    Bx�m*�  	�          @�Q�@�Q��:=q@9��A�z�C��@�Q��=q@j�HB{C��R                                    Bx�m90  
�          @���@��
�0��@:�HA�  C�q@��
��
=@i��BG�C��                                    Bx�mG�  
Z          @���@�Q��1G�@ffA��HC�:�@�Q����@7�A�Q�C�J=                                    Bx�mV|  �          @�z�@�G��0  @QG�B�C��@�G��Ǯ@~�RB*=qC�t{                                    Bx�me"  	`          @��@{��E@R�\B=qC��@{���\)@�33B-�C�Ff                                    Bx�ms�  �          @��R@z�H�R�\@C33A�{C���@z�H���@|(�B%�RC���                                    Bx�m�n  "          @��@���7�@@  A���C�(�@�녿�  @qG�B
=C��\                                    Bx�m�  T          @��R@}p��   @fffB=qC��q@}p����H@��RB6{C��f                                    Bx�m��  �          @�@�  �%@333A�z�C��@�  ��ff@^�RB�
C�}q                                    Bx�m�`  �          @��H@��׿�(�@6ffA��\C���@��׾k�@EB�C���                                    Bx�m�  �          @�Q�@����(�@-p�A��C���@����\)@=p�B��C�G�                                    Bx�mˬ  f          @�(�@@  ��{?У�A���C�@@  �j�H@>�RB ��C��H                                    Bx�m�R  �          @��@!G���ff?�33A�p�C�)@!G��z=q@E�B=qC�h�                                    Bx�m��  
�          @�G�?������?���A333C�@ ?������@.{A�{C�1�                                    Bx�m��  
l          @�ff?����\)?��@���C�o\?����Q�@{A��HC�                                      Bx�nD            @�\)?�ff����?(�@��HC�33?�ff����@�A��RC�                                    Bx�n�  
�          @�p�?�33��?�33A<��C��?�33����@1�A�\C�k�                                    Bx�n#�  �          @�=q@
=��
=?Tz�A	p�C���@
=��@=qA͙�C��                                    Bx�n26  �          @���@*=q����?J=qA
=C��@*=q��  @�
A�C��                                    Bx�n@�  �          @��
@Q����\?��A+�C�b�@Q���ff@)��A�  C��=                                    Bx�nO�  �          @��H?�ff���R?Y��A�
C���?�ff��z�@ ��A�Q�C���                                    Bx�n^(  T          @�(�?��R���\?333@�ffC��=?��R����@=qA�\)C���                                    Bx�nl�  "          @���@/\)��(���\��{C��@/\)����?���A9��C��=                                    Bx�n{t  �          @�
=@Z=q��G���G��W33C�\@Z=q��\)>\)?�  C�j=                                    Bx�n�  
�          @�p�@&ff�e��O\)�%p�C�@&ff�h��>�
=@�z�C��f                                    Bx�n��  �          @��
@Z�H��Q쾣�
�^�RC�9�@Z�H�x��?��
A3�C���                                    Bx�n�f  �          @�
=@?\)��
=>u@%C�� @?\)��z�?��HA�z�C��=                                    Bx�n�  �          @��@,�����H�L���z�C��3@,����(�?�Amp�C��                                    Bx�nĲ  �          @��?����
=>�Q�@g
=C�S3?����G�@
=qA���C���                                    Bx�n�X            @���?��\���
>�@�\)C�` ?��\��z�@z�A���C��3                                    Bx�n��  
�          @��>k���?��A-��C�1�>k�����@2�\A�RC�\)                                    Bx�n�  �          @�
=@�\����?���A6�RC�� @�\��p�@'�A�RC�ff                                    Bx�n�J  T          @��
?�{��33>aG�@��C�e?�{��
=@   A��C���                                    Bx�o�  
�          @��?�33��G�>�  @(Q�C�b�?�33����@ ��A�Q�C���                                    Bx�o�  �          @��?ٙ���=q�B�\����C�� ?ٙ���=q?�{A�C�Ff                                    Bx�o+<  �          @��R@$z����
?�A��
C��@$z��l��@W
=B�HC�c�                                    Bx�o9�  "          @���@8Q���p�?�(�A���C�Ф@8Q��tz�@L��B{C��H                                    Bx�oH�  "          @�  @W
=���R?���AYC�y�@W
=�p  @1�A�G�C��\                                    Bx�oW.  �          @��@\(����\?��
Ax  C�<)@\(��c33@9��A�p�C�                                    Bx�oe�  �          @�\)@c33��  ?�Af{C���@c33�aG�@1�A�=qC��)                                    Bx�otz  �          @�
=@O\)��
=?��AZffC��)@O\)�p  @1�A��C�l�                                    Bx�o�   �          @�  @!G����?(�@\C�%@!G���33@33A���C�`                                     Bx�o��  �          @���?���������
�k�C�!H?�����?�A�ffC���                                    Bx�o�l  �          @��?�����{�B�\��C��3?�����33?�(�A@��C��                                    Bx�o�  "          @�?W
=��  �(����{C�.?W
=��(�?��AT(�C�E                                    Bx�o��  
�          @�
=?�p���녿�����C��{?�p�����?�(�AdQ�C�                                      Bx�o�^  �          @�p�?�{���u��
C���?�{��p�?�(�A���C�                                    Bx�o�  �          @�@1G�����?�  AL��C�{@1G����@3�
A�{C�.                                    Bx�o�  
�          @���@����^{?�ffAz�RC��\@����.�R@(��A�p�C��\                                    Bx�o�P  T          @��@���a�@Q�A��RC�~�@���%@Mp�B��C��H                                    Bx�p�  T          @��H@��Ϳ�@)��A�G�C��)@��Ϳ&ff@Dz�A�z�C�33                                    Bx�p�  T          @�{@����'�@{A�
=C��3@�����Q�@?\)A�G�C�H�                                    Bx�p$B  �          @�G�@333���H?�
=A>�HC��@333��z�@1G�A�ffC��                                    Bx�p2�  "          @���@33��Q�?�G�AEC���@33��Q�@?\)A�
=C��                                     Bx�pA�  T          @�{@;���Q�?�p�A?
=C�*=@;�����@8Q�A�ffC�4{                                    Bx�pP4  
�          @���@G����?���AY��C�j=@G���Q�@HQ�BG�C��                                    Bx�p^�  T          @��?�z���ff?fffAp�C�AH?�z�����@.�RA���C�/\                                    Bx�pm�            @�@n�R��z�>���@>{C�*=@n�R����?�ffA���C�p�                                    Bx�p|&  B          @�G�@����@  �z�H�ffC��{@����I��=�?�z�C�K�                                    Bx�p��  �          @�=q?���33?�p�A<z�C�  ?���=q@FffA���C�xR                                    Bx�p�r  
�          @�  ?�{��33?��A!G�C��\?�{���@C�
A���C�                                      Bx�p�  
�          @�ff?�=q��(�?333@�
=C���?�=q����@,��A���C���                                    Bx�p��  !          @��?������H>�\)@1�C��?�����z�@{A��HC��                                    Bx�p�d  "          @��@33��
=���
�B�\C�C�@33��z�?��A��C��                                     Bx�p�
  T          @�?�������#�
���C���?����ff?�A�=qC�~�                                    Bx�p�  "          @��@���ff�W
=�
�HC�3@���?p��A�
C��                                    Bx�p�V  "          @��H?�
=��33�\(��  C�XR?�
=���\?z�HA#33C�b�                                    Bx�p��  �          @���?��R��p���
=�g33C���?��R���H>�@���C�8R                                    Bx�q�  �          @�  ?�=q���R�)����C��f?�=q���H�Tz����C��                                    Bx�qH  �          @�(�@����?W
=@�{C��@���  @1�A��C�C�                                    Bx�q+�  "          @Ǯ@����?L��@�(�C�
=@�����@2�\A���C�0�                                    Bx�q:�  �          @�Q�?��H���H?��A�C�@ ?��H��33@A�A�{C�z�                                    Bx�qI:  �          @�Q�?�p����?uA��C�S3?�p���z�@=p�A�  C���                                    Bx�qW�  �          @��@ ����p�?333@�(�C�b�@ ����G�@/\)A��HC�g�                                    Bx�qf�  "          @��H@<(���33?ٙ�AyG�C�e@<(�����@^�RB\)C�޸                                    Bx�qu,  �          @��@@����z�?��RA5C��
@@����33@Dz�A�33C��
                                    Bx�q��  T          @��
@I����{?��A=qC��@I����
=@9��A�z�C��                                     Bx�q�x  �          @Ϯ@W
=��p�?���A+
=C��@W
=��z�@C33A�{C���                                    Bx�q�  �          @�=q@Z=q��{?�33AEG�C��@Z=q���\@O\)A�\C�N                                    Bx�q��  T          @�  @�=q��\)?k�A�RC��)@�=q���\@(Q�A�\)C���                                    Bx�q�j  T          @�(�@�ff��Q�?&ff@��
C��{@�ff��
=@�
A�\)C�o\                                    Bx�q�  
�          @�p�@������R?=p�@��HC�  @�����z�@�A�\)C���                                    Bx�q۶  �          @�33@�����=q?�=qA:=qC��@�������@=p�A�p�C�s3                                    Bx�q�\  �          @���@�\)���
@�\A��C�=q@�\)�e@c33Bp�C��R                                    Bx�q�  T          @��H@����z=q@UA�33C�@����p�@�Q�B.ffC�b�                                    Bx�r�  
Z          @��
@z=q�e@}p�B��C���@z=q��
=@�\)BC
=C��R                                    Bx�rN  �          @��H@�ff�R�\@uB��C��{@�ff�ٙ�@�Q�B9��C���                                    Bx�r$�  "          @��H@����U@~{BffC�AH@�����Q�@���B@��C���                                    Bx�r3�  "          @�=q@}p��?\)@���B"��C�}q@}p���G�@��BJQ�C�1�                                    Bx�rB@  "          @Ϯ@~{�&ff@�{B,\)C�e@~{�Tz�@��\BMz�C��                                    Bx�rP�  �          @�
=@|(��.{@�ffB%{C���@|(����
@���BI  C���                                    Bx�r_�  
�          @�p�@w
=�J�H@z�HB\)C�Ff@w
=��ff@���BDQ�C�
=                                    Bx�rn2  
�          @�33@Z=q����@`��B\)C��@Z=q�   @�\)B@�C��f                                    Bx�r|�  T          @�=q@p  �aG�@c�
B
C�j=@p  �   @��HB<��C���                                    Bx�r�~  	�          @��@e��?\)@��B(��C�@e���G�@���BS�HC�L�                                    Bx�r�$  T          @ʏ\@�=q�E�@l(�BG�C�k�@�=q���@���B9�C���                                    Bx�r��  �          @ə�@hQ��^�R@mp�B��C�  @hQ��33@�
=BC��C�/\                                    Bx�r�p  �          @ȣ�@�z��N�R@\(�B��C��@�z��\@��
B0�C�p�                                    Bx�r�  �          @��
@z�H�   @���B*=qC��q@z�H�G�@�z�BJ��C�Z�                                    Bx�rԼ  
�          @�@�  ��p�@�33B7�C�Ф@�  �8Q�@�Q�BMz�C��R                                    Bx�r�b  "          @�p�@�33����@��B5�RC���@�33��\)@�p�BH�C���                                    Bx�r�  
�          @�(�@r�\��G�@���BC�C���@r�\=���@��BU�
?���                                    Bx�s �  �          @Ǯ@^�R��{@��BJ�RC��
@^�R<��
@��B`
=>�p�                                    Bx�sT  
�          @�\)@�\)�7�@^{B��C��f@�\)��z�@���B/=qC��H                                    Bx�s�  
Z          @�Q�@\(��=p�@�(�B^Q�C��f@\(�?�
=@���BY{A�G�                                    Bx�s,�  �          @�Q�@HQ쿌��@���Bg�
C�Ff@HQ�?fff@��\Bjz�A��H                                    Bx�s;F  T          @�Q�@A녿���@���Bg{C��@A�?#�
@�p�Bp��A?�
                                    Bx�sI�  T          @�Q�@Q녿��@�Ba��C��@Q�?fff@��RBc��Av�R                                    Bx�sX�  �          @Ǯ@W
=���\@�33B]�C��@W
=?h��@��
B_G�As33                                    Bx�sg8            @�  @R�\�8Q�@�G�Bhp�C�h�@R�\?�\@��BT�
A��                                    Bx�su�            @�Q�@J=q>�Q�@��Bm\)@�  @J=q@33@��BL��B(�                                    Bx�s��  T          @�\)@Tz��(�@��Be�C�L�@Tz�?\@���BX(�Aģ�                                    Bx�s�*  �          @�
=@�녿�G�@�B6C���@��>k�@�p�BD
=@K�                                    Bx�s��  
�          @�
=@\)��z�@�{B9�C�:�@\)>��R@���BE(�@�                                      Bx�s�v  �          @�ff@_\)��33@�33BN{C�]q@_\)>�  @��
B^��@��
                                    Bx�s�  
Z          @�ff@qG���z�@�(�BA�C��@qG�>#�
@�p�BR{@��                                    Bx�s��  �          @�@�  ���R@��RB9�C���@�  >��@�ffBF�@j�H                                    Bx�s�h  T          @�{@}p���{@���B=z�C���@}p�>��@�\)BG��@�{                                    Bx�s�  "          @�z�@�Q��(Q�@`��B=qC�"�@�Q쿓33@��B/\)C�n                                    Bx�s��  �          @��@��
���@~�RB!C��3@��
�z�@�Q�B==qC��                                    Bx�tZ  T          @�z�@z=q�1G�@q�B
=C�Q�@z=q��
=@�G�B@
=C��{                                    Bx�t   
�          @�33@����'�@n{B��C��\@������@�p�B:ffC��                                    Bx�t%�  �          @��
@�  � ��@g
=B�C��{@�  ��  @���B1C�aH                                    Bx�t4L  
�          @�(�@�Q��$z�@`��BC�s3@�Q쿋�@��RB/  C��3                                    Bx�tB�  �          @��
@��R��\@z=qB�C��@��R��G�@�(�B8  C��q                                    Bx�tQ�  �          @Å@�  �G�@p��B  C��\@�  �5@��HB4��C�9�                                    Bx�t`>            @��H@�����@\)B#�C�%@�녿   @�  B>�C�w
                                    Bx�tn�            @��@����Q�@tz�B��C��)@��ÿJ=q@�B<�\C�s3                                    Bx�t}�  T          @��@�\)�.�R@S�
B�C��{@�\)���@��\B*�C�h�                                    Bx�t�0  
�          @���@���(��@QG�B�C�@ @�녿��R@���B'�C�                                      Bx�t��  T          @�Q�@����<��@?\)A�C��=@��׿У�@xQ�B!=qC��\                                    Bx�t�|  "          @���@�\)�/\)@P  B�C���@�\)���@���B)�C�8R                                    Bx�t�"  T          @��@��\��
=@z�HB#��C�]q@��\����@�33B;  C��\                                    Bx�t��  "          @��@mp���
=@�ffB@�HC�t{@mp�>���@�p�BM�@��\                                    Bx�t�n  "          @��@tz��g
=@.�RA�C�P�@tz���@xQ�B#C�O\                                    Bx�t�  
�          @�{@�G���G�@uB"��C�Ǯ@�G�>��@�G�B,��@^�R                                    Bx�t�  
�          @�\)@����G�@�z�B0{C�&f@��?��@�
=B4\)A(�                                    Bx�u`  �          @�{@l�Ϳ��@��BA=qC��@l��>��@��HBL  @ʏ\                                    Bx�u  
�          @�{@Z�H���
@�Q�BI=qC��@Z�H>�=q@�Q�BX�@�{                                    Bx�u�  �          @�p�@~{�޸R@p��B#��C�4{@~{�L��@�z�B8�C���                                    Bx�u-R  T          @�{@����7�@z�A��
C�+�@��Ϳ��@N{Bz�C�w
                                    Bx�u;�  T          @��R@�G��!�@O\)B=qC�@�G�����@|��B&�RC��H                                    Bx�uJ�  �          @�Q�@�Q��/\)@3�
A�(�C�\)@�Q쿾�R@hQ�B�\C�޸                                    Bx�uYD  T          @��@�
=�4z�@ffA�
=C���@�
=��p�@O\)B��C���                                    Bx�ug�  �          @�(�@�  �'�@b�\BQ�C�#�@�  ��{@���B0�RC��{                                    Bx�uv�  
�          @��H@��
�0  @N�RB Q�C���@��
��=q@���B%��C��f                                    Bx�u�6  "          @�=q@���ff@eB��C���@�녿Tz�@��RB/(�C��=                                    Bx�u��  �          @���@���S33?���Ax(�C��R@�����@.{A�=qC�Q�                                    Bx�u��  	�          @�Q�@���b�\?��
A�Q�C��f@���'
=@>�RA��C��                                    Bx�u�(  
�          @�Q�@����Tz�@�A�Q�C�ٚ@����  @QG�B��C��                                     Bx�u��  �          @��@����
?�z�A333C��3@��XQ�@(Q�A�z�C���                                    Bx�u�t  �          @�ff@y����?.{@�33C���@y���vff@33A�Q�C���                                    Bx�u�  
�          @��@������>aG�@C�.@�����Q�?�{A�p�C���                                    Bx�u��  
�          @���@����{=�?�z�C�8R@�����?��A�(�C�y�                                    Bx�u�f  
�          @��@��H���
���Ϳp��C��@��H���\?ǮAqG�C��f                                    Bx�v	  
Z          @�G�@y�����H>8Q�?�  C�/\@y����?�33A�33C���                                    Bx�v�  �          @��@�33���R=L��>��C�O\@�33���?�p�A��RC�y�                                    Bx�v&X  
�          @��@�=q��
=�����s�
C��
@�=q���?�(�A:=qC�e                                    Bx�v4�  	�          @���@����>�(�@��HC�@��u�@G�A�\)C��H                                    Bx�vC�  
�          @���@��\��{>��R@=p�C�Ff@��\�\)?�(�A�(�C�˅                                    Bx�vRJ  "          @���@W���\)?��@�Q�C�f@W�����@��A��C��{                                    Bx�v`�  
�          @���@Vff��{?8Q�@�(�C�{@Vff����@"�\AǅC��
                                    Bx�vo�  
�          @Å@R�\���?aG�A��C��f@R�\��33@.�RA���C��\                                    Bx�v~<  T          @�z�@Mp���?   @�\)C���@Mp���33@�A�G�C�q�                                    Bx�v��  �          @��H@8�����\���H��33C�=q@8�����?���Ao
=C��)                                    Bx�v��  T          @��H@P  ��=q>\@i��C�Z�@P  ��G�@�A�G�C��                                    Bx�v�.  
�          @�=q@J�H��=q?n{A{C�@J�H���H@2�\A�\)C�R                                    Bx�v��  �          @\@>�R��{?fffA	G�C��3@>�R���R@333A���C��                                    Bx�v�z  "          @\@=p���Q�?^�RA��C�K�@=p����@-p�A�\)C�E                                    Bx�v�   T          @�33@&ff��(�?�{A���C�o\@&ff��G�@j�HB�\C�b�                                    Bx�v��  �          @��
@'
=���@�RA���C���@'
=�fff@��B,�C���                                    Bx�v�l  �          @��H@N�R����@=qA�G�C�� @N�R�Q�@|��B"��C�N                                    Bx�w  �          @Å@������@=p�A��C�Z�@���S33@���BC��C���                                    Bx�w�  "          @Å@�H���H@�\A�ffC��@�H�W�@w
=B+�RC���                                    Bx�w^  �          @�  @�z��Y���   ��
=C��@�z��U�?Tz�A(�C�'�                                    Bx�w.  �          @�=q@/\)��p�?aG�A�
C���@/\)�aG�@��A�(�C��3                                    Bx�w<�  �          @�ff@��R�~�R�aG��
�RC�K�@��R��Q�?333@�33C�*=                                    Bx�wKP  "          @�  @fff��\)�&ff�ȣ�C��@fff���
?���A8��C���                                    Bx�wY�  �          @��@w
=���׿����\)C�>�@w
=����?�A4��C���                                    Bx�wh�  T          @��
@u���p��(����(�C�N@u���33?z�HA=qC��f                                    Bx�wwB  T          @���@����%����\)C�@����S33�����8z�C��                                    Bx�w��  �          @��@��R��z��9����33C�~�@��R�7����H���HC���                                    Bx�w��  
�          @\@��\����J=q����C�O\@��\�<�������  C�Ff                                    Bx�w�4  T          @��@�Q�ٙ��g���RC��@�Q��<���,�����HC�e                                    Bx�w��  T          @���@�ff������4�C�h�@�ff�
=�n{�Q�C��                                    Bx�w��  �          @���@vff�
=q����BQ�C���@vff���w��$�\C�9�                                    Bx�w�&  �          @��@�  �p��z�H�"�\C��@�  �a��1���{C�AH                                    Bx�w��  f          @�=q@G
=?�33�����Y�RA�Q�@G
=����{�e�HC�4{                                    Bx�w�r  �          @�p�@i��?Q���=q�PAK�@i����=q�����M�C��H                                    Bx�w�  �          @�z�@aG�?�ff���\�P��A�{@aG��#�
���R�XC��\                                    Bx�x	�  �          @�33@q�?\)��  �L�RA\)@q녿�ff��33�D(�C��=                                    Bx�xd  
�          @���@r�\?@  �����L  A2�\@r�\������ff�G�C��\                                    Bx�x'
  T          @���@|��>�G����F�R@ə�@|�Ϳ������<{C�S3                                    Bx�x5�  �          @Å@����{��(��,��C���@���&ff�X����HC�<)                                    Bx�xDV  �          @���@�����n{��\C�~�@���a��$z����HC���                                    Bx�xR�  �          @���@��ÿ�=q��=q�,��C�S3@����@���J=q� ffC��)                                    Bx�xa�  �          @���@}p������Q��A��C��R@}p����|(��#G�C��                                    Bx�xpH  �          @Å@`  ?����33�T{A�Q�@`  �fff��(��U�HC��3                                    Bx�x~�  �          @�(�@333?u���H�j(�A��@333�u���H�j�C��H                                    Bx�x��  �          @�(�@e����������C��3@e���׾.{���HC�<)                                    Bx�x�:  �          @��@W
=��33��
=���C�R@W
=��z�>W
=?��RC�AH                                    Bx�x��  �          @���@l(���z�k����C�C�@l(���z�?n{A�\C�E                                    Bx�x��  �          @���@�{�tz��33��\)C��3@�{��녾�(���(�C�{                                    Bx�x�,  T          @�Q�@y���'��U��p�C�3@y���l(�����
=C�G�                                    Bx�x��  T          @�  @xQ��ff�o\)�$�C��@xQ��Fff�1G�����C���                                    Bx�x�x  �          @��
@g
=�:�H����JffC�L�@g
=����w��&�C�>�                                    Bx�x�  �          @�Q�@Vff?W
=�����[G�A`(�@Vff�������H�WC�޸                                    Bx�y�  �          @�z�@e�#�
����O(�C��3@e���
��{�9G�C�˅                                    Bx�yj  �          @�ff@]p����R�����W��C�p�@]p�����G��:�C�C�                                    Bx�y   �          @�ff@��.{��ff�C��
@��\)��
=�c�C�f                                    Bx�y.�  T          @��R@  <#�
��\)ff>aG�@  �����k{C��                                    Bx�y=\  T          @��
@3�
�O\)���\�pp�C��\@3�
�,����(��A�C��                                    Bx�yL  �          @�\)@0�׿�{���\�i��C���@0���8Q���G��5\)C��)                                    Bx�yZ�  �          @�?�  <#�
��(�� >Ǯ?�  �	�����R�w�C���                                    Bx�yiN  �          @���@<(�?Ǯ�����a�
Aߙ�@<(�������qffC�Q�                                    Bx�yw�  �          @���@7
=?������\�g�
Aͅ@7
=�.{��
=�rQ�C�^�                                    Bx�y��  �          @���@AG�?�ff��G��c��A��H@AG��:�H����k�
C�0�                                    Bx�y�@  T          @�  @A�?���(��Zz�A��H@A녾��������mQ�C���                                    Bx�y��  �          @��@4z�?�33��  �c=qA�33@4z�Ǯ��  �v�C�3                                    Bx�y��  �          @���@%?޸R��33�e  B�
@%��  �����|�
C�:�                                    Bx�y�2  �          @�{@�{>��
�Tz��\)@�(�@�{�z�H�L(��Q�C�l�                                    Bx�y��  �          @�(�@���>���B�\�
(�@��
@��ͿG��>{�\)C���                                    Bx�y�~  �          @�G�@X��?�ff�u�:�\A�{@X�þ�=q�����G�C��3                                    Bx�y�$  �          @�Q�@&ff?Tz���  �^�HA��@&ff�E������_�HC��H                                    Bx�y��  �          @�{@�G���  �(����(�C�Q�@�G��
=q���H��=qC�B�                                    Bx�z
p  �          @��R@�z��G���{��{C�q�@�z��ff����A�C�Z�                                    Bx�z  �          @�=q@��R���
����>�\C�s3@��R��=q��(���33C��H                                    Bx�z'�  �          @�33@��Ϳ+���z���33C��=@��Ϳ��Ϳ�G����\C��                                    Bx�z6b  �          @�33@�  �333�������
C��\@�  ��G����H�]�C���                                    Bx�zE  
�          @��R@�G�>���:=q�ff@��@�G��=p��5�
��C�Ф                                    Bx�zS�  
�          @���?�@+���G��O�HBhz�?�?n{��Q�G�A���                                    Bx�zbT  T          @�  @@.{�l���7�\BE
=@?�����R�o�A�                                      Bx�zp�  �          @���@�=q>.{�U�z�@p�@�=q��\)�I���{C�O\                                    Bx�z�  �          @��H@���˅�����p�C���@���	������5p�C�˅                                    Bx�z�F  �          @�ff@�z����z����\C��@�z��*=q�u�#\)C�R                                    Bx�z��  �          @�33@���333��ff��\)C�<)@���K���Q��mp�C��\                                    Bx�z��  T          @��
@��Ϳ�Q������C�9�@���� �׿�  �$  C�q�                                    Bx�z�8  
�          @�Q�@�(���p�����33C�f@�(����R� ����Q�C��f                                    Bx�z��  5          @�Q�@�{�8Q������(�C�f@�{���
�G����C�c�                                    Bx�zׄ            @�{@��׿�=q��R���
C�8R@��׿��������C��                                     Bx�z�*  �          @�ff@��?�Q��X���=qA�G�@��=#�
�k���\>�                                    Bx�z��  �          @��@���?��H�U�
��A�@���>����o\)�(�@s33                                    Bx�{v  �          @�
=@���?��g
=���A�(�@���>Ǯ��=q�-�@�                                    Bx�{  �          @�ff@��?�=q�P���\)A��R@��>�G��mp���@�Q�                                    Bx�{ �  �          @�ff@��H@*=q�W��
G�B=q@��H?���(��0z�A�
                                    Bx�{/h  �          @�ff@��@O\)�,(���ffB�@��?��H�n�R�
=A��
                                    Bx�{>  �          @�Q�@��
@G��;���z�B�@��
?�G��y���$=qA�p�                                    Bx�{L�  �          @\@xQ�@U�L�����B#{@xQ�?�\)��\)�2�A͙�                                    Bx�{[Z  �          @�z�@�{@
=�{���HAՅ@�{?�=q�<�����A}                                    Bx�{j   �          @��
@�{@c�
�p���p�B��@�{@(��Y���G�A�Q�                                    Bx�{x�  �          @��H@��@h�ÿٙ���Q�B\)@��@-p��=p���A��
                                    Bx�{�L  �          @�p�@�=q@U���  �ffB�@�=q@,(������A߮                                    Bx�{��  �          @�33@�Q�@(��8Q����A��H@�Q�?�p������t  A��                                    Bx�{��  �          @Å@�@
�H��Q�aG�A�\)@�?�(��k��
�\A�
=                                    Bx�{�>  T          @���@��\?�\?
=q@��A���@��\?��8Q���A��                                    Bx�{��  �          @�p�@��R?�33?0��@θRAS
=@��R?Ǯ=�G�?�  Aj{                                    Bx�{Њ  �          @��@�\)?8Q�?�(�A7�
@�33@�\)?���?\(�A Q�A+\)                                    Bx�{�0  �          @Å@��>aG�?У�Ax(�@(�@��?Tz�?�AW�A ��                                    Bx�{��  �          @�p�@��\�#�
?��HA]p�C�1�@��\��@z�A�G�C���                                    Bx�{�|  �          @�ff@�zὸQ�@  A��C�� @�z�?E�@
=A��
A{                                    Bx�|"  T          @Ǯ@�z�?��@5�A��
A_33@�z�@�
@ffA��
A�                                      Bx�|�  T          @�\)@�?xQ�@7�A�{A!�@�?��R@�\A��\A��                                    Bx�|(n  T          @�\)@�p�=�G�@(�A��?���@�p�?��
@p�A��A$                                      Bx�|7  
�          @�ff@��?p��@0��A�\)A
=@��?�@��A���A�                                    Bx�|E�  �          @ƸR@��\?�G�@J=qA�Q�A��R@��\@&ff@A���A���                                    Bx�|T`  �          @��@��
@8Q�@-p�A��Bp�@��
@l(�?�z�A]B ��                                    Bx�|c  
�          @�p�@@  @��R@6ffA��HBZ33@@  @�\)?��
A!G�Bk                                    Bx�|q�  �          @�(�@7
=@�{@7
=A�33B^��@7
=@��R?�ffA&ffBp
=                                    Bx�|�R  �          @�\)@'
=@�33@>�RA��Be��@'
=@�?���AABx33                                    Bx�|��  �          @���?�z�@��R@@��A���B���?�z�@���?��A0z�B�#�                                    Bx�|��  �          @��H@\��@Fff@B�\B�B'�\@\��@���?��A��BE\)                                    Bx�|�D  T          @�\)@tz�@ ��@\(�B
=B(�@tz�@g�@�RA��
B.�                                    Bx�|��  T          @�G�@p  @33@FffBA�\)@p  @Dz�@A�ffBQ�                                    Bx�|ɐ  �          @���@��@
=@(��A�=qA�\)@��@<��?�33A�G�B
=                                    Bx�|�6  �          @��H@��\?�33@y��B*z�A|��@��\@#33@J�HB��A��
                                    Bx�|��  �          @��@w�?z�@���B>�AQ�@w�@
=q@p��B!
=A�                                    Bx�|��  �          @��
@��R��@���B/
=C���@��R?��@|(�B(�AhQ�                                    Bx�}(  �          @��H@u��I��@(�A��C�Ff@u����R@]p�BC�C�                                    Bx�}�  �          @��@��333@s�
BffC��@���33@��\B:G�C�N                                    Bx�}!t  �          @��H@���.�R@p  Bp�C��@�녿�{@�Q�B5�\C���                                    Bx�}0  �          @��H@����'�@tz�B�RC�Y�@����z�H@���B6�HC���                                    Bx�}>�  �          @���@�\)�   @uB  C��)@�\)�\(�@�  B8��C�B�                                    Bx�}Mf  �          @���@�(��33@�\)B*{C��@�(�����@�{BB33C���                                    Bx�}\  �          @���@�{�ff@��B&z�C���@�{��p�@���B?\)C�|)                                    Bx�}j�  �          @��@�33�{@~{B�\C�~�@�33���@�Q�B7ffC�`                                     Bx�}yX  �          @��@��H�&ff@Z�HBC�:�@��H��\)@���B%�C�&f                                    Bx�}��  �          @��@�� ��@Z�HB
=C�� @����@��B"�C��)                                    Bx�}��  �          @��@�\)�(Q�@e�B
z�C�Ǯ@�\)���@��B,p�C�.                                    Bx�}�J  �          @���@���6ff@N�RA�C�  @�녿�@��B!�C�S3                                    Bx�}��  �          @��H@�33�*�H@R�\BG�C�:�@�33��p�@���B'�C�3                                    Bx�}  �          @�\)@~�R�i��@$z�A˅C���@~�R��H@p��B��C�W
                                    Bx�}�<  �          @�{@�G��W�@(Q�A�z�C�
@�G��Q�@mp�B�C��                                    Bx�}��  �          @���@x���5�@*=qA�C�  @x�ÿ�\)@a�BC��{                                    Bx�}�  �          @���@HQ��h��@�A��HC�Y�@HQ��   @aG�B%\)C��3                                    Bx�}�.  �          @���@�G��w
=?��RAo\)C�&f@�G��AG�@5�A�\C��q                                    Bx�~�  �          @��@{��s�
?�AAG�C��\@{��Fff@!G�Aՙ�C��                                    Bx�~z  �          @�=q@xQ��u�?��A/33C��{@xQ��I��@=qA�{C�u�                                    Bx�~)   �          @���@q���Q�=��
?L��C���@q��l(�?���A���C��q                                    Bx�~7�  �          @�\)@p����33�k����C�G�@p���xQ�?���A[\)C��                                    Bx�~Fl  �          @��H@G��aG�@r�\BG�C�Ǯ@G�����@��\BTQ�C��                                    Bx�~U  �          @��@A��l��@k�B33C���@A���
@�G�BRffC��\                                    Bx�~c�  �          @�
=@S�
�o\)@N�RB�C��H@S�
���@�z�B>�\C���                                    Bx�~r^  �          @��R@h���l��@8Q�A�\C�B�@h���ff@���B,p�C��=                                    Bx�~�  �          @�\)@e��(�?��A��RC��f@e�W
=@XQ�B	�HC�j=                                    Bx�~��  
(          @��R@dz����\?���A0��C��{@dz��u�@0��Aޏ\C�~�                                    Bx�~�P  �          @�(�@8����z�@z�A��
C���@8���a�@i��B�\C���                                    Bx�~��  �          @�G�@(Q����H@�A�z�C�<)@(Q��e@��B)��C�R                                    Bx�~��  �          @\@����@,��A��
C���@�b�\@��\B6ffC��{                                    Bx�~�B  �          @�
=@1G���p�@ffA�G�C�S3@1G��]p�@z�HB%�C�Q�                                    Bx�~��  �          @���@!�����@0  A�G�C�@ @!��S33@���B7Q�C��R                                    Bx�~�  �          @�
=@G
=���R@z�A��\C�l�@G
=�Q�@s�
B p�C��
                                    Bx�~�4  �          @���@n{��{?��An�\C��@n{�c33@E�A�33C�33                                    Bx��  T          @�\)@fff��ff?��A�{C�z�@fff�aG�@J�HB �C�˅                                    Bx��  �          @�  @w���p�?��A$(�C��R@w��mp�@'�AиRC�R                                    Bx�"&  �          @�Q�@xQ����R?��A"�RC���@xQ��p  @(��A�=qC�H                                    Bx�0�  T          @�\)@j�H��z�?W
=A��C�*=@j�H��  @   A��
C�C�                                    Bx�?r  �          @��@�Q��P��@ ��A�z�C�G�@�Q��ff@b�\B��C��)                                    Bx�N  
�          @�G�@w
=�u�@#�
A�{C���@w
=�'
=@s33BC��
                                    Bx�\�  �          @��@^�R����@*�HA��C�p�@^�R�0  @~{B'�C��{                                    Bx�kd  �          @���@j=q���H@�\A�  C��@j=q�R�\@`  B��C�                                    Bx�z
  �          @��H@qG����?�
=A��C�l�@qG��U@Y��B  C�8R                                    Bx���  �          @��
@�\)�|(�@G�A��
C���@�\)�:�H@U�B�\C���                                    Bx��V  �          @�33@�G���\)?�p�A���C�ٚ@�G��S33@J=qA�C�e                                    Bx���  �          @\@u����?��
A�z�C��f@u��Y��@P��B�HC�4{                                    Bx���  �          @\@�Q��hQ�@�
A�C��@�Q��!�@_\)B��C���                                    Bx��H  T          @���@���J=q@A�
=C���@����@UB��C��H                                    Bx���  �          @�=q@�{�e�@�A�{C��\@�{�$z�@P  BG�C��R                                    Bx���  �          @��H@�=q�_\)@   A�33C�T{@�=q� ��@HQ�A�{C���                                    Bx��:  
�          @Å@���n{?�z�A���C���@���0��@H��A��HC��
                                    Bx���  �          @Å@��R�n�R?�\)A���C�@��R�2�\@G
=A���C��R                                    Bx���  �          @���@�G��q�?�z�A{
=C��@�G��:=q@;�A��C��f                                    Bx��,  �          @��@����{�?�33AR�RC��=@����H��@/\)A�G�C���                                    Bx��)�  �          @�p�@���k�?�\)A(  C�H�@���AG�@��A�ffC���                                    Bx��8x  �          @��@�p��xQ�?�33A���C��=@�p��AG�@<��A�ffC�{                                    Bx��G  T          @�z�@.{�1�@���BE�\C�:�@.{�n{@�\)Bt�RC���                                    Bx��U�  �          @��H@:=q��@�33BR
=C���@:=q����@��
BuQ�C��                                    Bx��dj  �          @�G�@����tz�?k�AG�C���@����N�R@  A�  C�AH                                    Bx��s  �          @�=q@�ff�`��?���AT(�C���@�ff�1G�@#�
A�33C�                                    Bx����  �          @�=q@����AG�?���A]p�C��=@�����\@��A��\C��=                                    Bx���\  �          @�G�@�p��n{?c�
Az�C��q@�p��H��@�A�p�C��                                    Bx���  T          @�  @��H�S�
?��A(��C�˅@��H�,(�@{A���C�q�                                    Bx����  T          @�ff@��
���\@G�A���C���@��
��@\)A��HC�K�                                    Bx���N  T          @�(�@��׿h��@��A��C�R@���<��
@$z�Aϙ�>.{                                    Bx����  �          @��@7
=@u�@9��A�  BT�R@7
=@��
?���A[\)Bh�                                    Bx��ٚ  �          @�{@<��@s33@<(�A�p�BP��@<��@��?�\)Aap�Bez�                                    Bx���@  �          @�=q@��׿�@#�
A���C��f@���>Ǯ@%A��@��R                                    Bx����            @�(�@������@\)A�G�C�Y�@��?(�@(�A�(�@�{                                    Bx���  
�          @��@��
��Q�@(�A��C���@��
?333@�A��
@�ff                                    Bx��2  
�          @���@���>B�\@%AՅ@�
@���?���@ffA�z�AAp�                                    Bx��"�  T          @��@��;B�\@`��B
=C��f@���?�=q@UB�\A\��                                    Bx��1~  �          @��R@��H�u@i��B�
C�q�@��H?��@_\)B�A`Q�                                    Bx��@$  �          @�Q�@��ͼ��
@>�RA���C�� @���?�ff@333A�
=A@��                                    Bx��N�  T          @���@��;��@�A��RC��3@���>�{@33A�@h��                                    Bx��]p  �          @��H@��R��G�@!�AθRC��)@��R?   @!G�Aͮ@�ff                                    Bx��l  T          @�
=@�33�aG�@=p�A�ffC���@�33?Y��@5A�A�R                                    Bx��z�  T          @�=q@�  ��{@,��A���C��H@�  ��\)@:�HA�z�C��
                                    Bx���b  T          @��
@��H?z�@R�\B�@��@��H?��H@7�A���A��                                    Bx���  T          @�z�@�\)?��@8��BffA\z�@�\)@   @A�Q�A�=q                                    Bx����  �          @�=q@]p�@   @`��B��B\)@]p�@e@��A��B7�                                    Bx���T  �          @��\@���?�\)@VffB�A���@���@-p�@#�
AݮB                                      Bx����  �          @��@~�R?�33@c�
B�RA�(�@~�R@B�\@*=qA�Bp�                                    Bx��Ҡ  �          @�p�@�ff?�@>�RA�z�A�p�@�ff@0  @	��A�z�A��
                                    Bx���F  �          @�{@�G�?��@Tz�BG�A�  @�G�@'
=@#�
A�=qA��H                                    Bx����  �          @�{@���?:�H?�
=A�G�@�(�@���?�\)?�ffA}��Ag�
                                    Bx����  �          @�p�@�ff<�?�p�AL  >���@�ff>�?��A<��@��
                                    Bx��8  �          @��H@�=q��z�?5@��
C�!H@�=q�Tz�?���A,z�C���                                    Bx���  �          @��H@�
=�G�>.{?ٙ�C�� @�
=��=q?^�RA	p�C�                                    Bx��*�  T          @��@�33� ��?xQ�AC��{@�33���?�{A��C���                                    Bx��9*  �          @��@�(���\@	��A�z�C��H@�(��fff@(Q�A�{C��                                    Bx��G�  �          @��@��׿��@2�\A��
C��q@��׽L��@?\)A�ffC���                                    Bx��Vv  �          @�33@�\)�5@9��A�C��
@�\)>�33@=p�A���@���                                    Bx��e  �          @�  @��R����@�A��C��)@��R�\)@�A���C��3                                    Bx��s�  �          @�
=@�G���Q�?�ffA���C��@�G����H@ffA���C�Z�                                    Bx���h  �          @��@��ÿG�@ffA��
C���@���    @\)A��<��
                                    Bx���  �          @���@��H��@Q�A�33C��@��H>��R@	��A�p�@Vff                                    Bx����  �          @��@�������@	��A�(�C��3@���>�p�@
=qA��R@\)                                    Bx���Z  �          @��R@���.{?��A�{C�{@��>��H?���A�33@��R                                    Bx���   
�          @�ff@����\?�{A��\C�H�@��>.{?�A�?��                                    Bx��˦  "          @�
=@�z�z�?��HA�  C��=@�z�=L��?�A�(�>�                                    Bx���L  "          @�Q�@�\)��G�?�
=Ae�C��=@�\)��ff?�Q�A�z�C���                                    Bx����  
Z          @���@�zῌ��?�Q�A��\C�>�@�z��ff?�(�A��C��                                     Bx����  T          @�{@��\��\)?��A�(�C��=@��\���@z�A�(�C��3                                    Bx��>  "          @��H@��
�u?���A���C�g�@��
��Q�?�=qA�ffC��=                                    Bx���  T          @�@�(���?333@��
C��R@�(�����?��A<  C�=q                                    Bx��#�  �          @��H@�\)�p��>W
=@	��C�"�@�\)�O\)?�@��HC��=                                    Bx��20  
�          @��
@�Q�c�
>Ǯ@���C�l�@�Q�333?+�@��C�b�                                    Bx��@�  
�          @��
@�p���ff?���A�C���@�p����@
=Aȣ�C��                                    Bx��O|  
(          @�Q�@�z��&ff?���A5C�` @�z���?��A�\)C��                                    Bx��^"  
�          @��
@�
=�B�\��G����C��@�
=�W
=�#�
���C��                                    Bx��l�  T          @�ff@]p���Q�@C�
B  C��3@]p����@]p�B2�C�|)                                    Bx��{n  T          @�z�@��þ��H���
��\)C�R@��ÿz�H���\�j{C�7
                                    Bx���  
�          @�
=@�z���
�Q��ffC�Ǯ@�z��C33������C��\                                    Bx����  T          @�
=@�33��33�7
=��RC�'�@�33�!G��Q�����C��f                                    Bx���`  
�          @�Q�@�33�c�
�\)��p�C��@�33��
=��\��z�C��H                                    Bx���  �          @�  @z=q��H@z�A��C��@z=q��=q@0��B�C�H                                    Bx��Ĭ  T          @���@�33�:=q@��A���C��@�33��p�@G
=BG�C�                                    Bx���R  T          @��@�Q�����
�S�
C�H�@�Q��������C���                                    Bx����  "          @���@�  ���\<�>��RC�� @�  �s33>�p�@s�
C��                                    Bx���  "          @��@�z��ff?�p�A��RC��@�zῃ�
@{Aҏ\C�{                                    Bx���D  �          @��@c33�U�@EBp�C�k�@c33�ff@���B1�C���                                    Bx���  �          @�z�@hQ��l(�@��A�ffC�=q@hQ��*=q@b�\B�C��                                     Bx���  �          @��H@`  �~{@�A���C���@`  �<��@c�
B\)C��
                                    Bx��+6  
�          @��\@3�
��@7�A�p�C��R@3�
�>�R@�33B4�C���                                    Bx��9�  
�          @���@(���ff@C33A�C��H@(��Z=q@�{B>Q�C�Y�                                    Bx��H�  T          @�
=?�Q����@Tz�B33C���?�Q��R�\@�BN��C���                                    Bx��W(  �          @�\)?�ff��=q@I��A���C��?�ff�`  @�=qBH33C��R                                    Bx��e�  s          @�Q�@G�����@ ��A��C��=@G���{@g
=B�\C�@                                     Bx��tt  �          @���@dz��r�\?�\)Alz�C���@dz��G�@#33A�ffC�h�                                    Bx���  �          @�@�
=�]p�?ǮAv�\C�"�@�
=�0  @'
=A��
C�4{                                    Bx����  	�          @�p�@u��|��?�\)A_33C��@u��Q�@&ffA��C��
                                    Bx���f  
�          @��@����g�?�G�A�\C��f@����E�@	��A��C��R                                    Bx���  
�          @�z�@���ff@8Q�A��C���@��P��@�B<Q�C���                                    Bx����  �          @�{?
=����@���B,��C��=?
=�,��@�\)Bx��C�!H                                    Bx���X  "          @���?�������@S�
B(�C��?����Mp�@��
BMG�C��{                                    Bx����  �          @�p�@g
=����@$z�A�G�C��@g
=�L(�@tz�B�C�@                                     Bx���  "          @��
@Dz����@6ffA֏\C�u�@Dz��b�\@��B(�C�o\                                    Bx���J  
�          @�(�@L����{@8��A��
C�&f@L���_\)@�Q�B'��C�AH                                    Bx���  T          @�@K���=q@P  A�=qC�l�@K��P��@�=qB4(�C�%                                    Bx���  �          @�@S�
��(�@@��A޸RC��H@S�
�Y��@�33B*  C��                                    Bx��$<  
�          @��
@S33���@Dz�A�
=C�(�@S33�P  @��
B-33C���                                    Bx��2�  T          @�ff@XQ����@J�HA�C�� @XQ��N{@��RB.�RC�7
                                    Bx��A�  T          @�ff@L(�����@Q�A�  C��f@L(��P  @�=qB4p�C�AH                                    Bx��P.  
�          @���@>{����@S�
A�\)C��@>{�N{@��HB9p�C�^�                                    Bx��^�  T          @���@����@&ffA�{C��=@��@��@qG�B�RC�"�                                    Bx��mz  
�          @���@aG�����@[�BC���@aG��-p�@�G�B6��C�8R                                    Bx��|   
�          @�ff@QG��vff@�  B\)C�%@QG���@���BL�HC��                                    Bx����  T          @�{@s�
��G�@8Q�AՅC�Ф@s�
�H��@��B
=C�C�                                    Bx���l  
�          @�ff@\)���@9��A�\)C��H@\)�AG�@�=qB�RC�p�                                    Bx���  
Z          @�\)@����G�@AG�A�ffC��3@���7
=@���B�C�g�                                    Bx����  T          @θR@����\)@G
=A�  C���@����2�\@�
=B#
=C��
                                    Bx���^  
�          @�
=@Tz��|(�@w�B�RC��@Tz�� ��@�p�BG\)C�w
                                    Bx���  	.          @�Q�@XQ���Q�@s�
B�HC�{@XQ��%@�(�BC�C�E                                    Bx���  
�          @�  @s33���@Q�A���C�c�@s33�7
=@��B+p�C���                                    Bx���P  
�          @У�@�����  @�HA��\C���@����`  @n{B\)C��f                                    Bx����  �          @Ϯ@�  ��  @2�\A̸RC��H@�  �I��@\)Bp�C��                                    Bx���  
(          @�
=@��
��@	��A���C�q@��
�R�\@W
=A���C�~�                                    Bx��B  �          @�Q�@����y��@=p�A�z�C��
@����1G�@�G�B\)C���                                    Bx��+�  	�          @�\)@�Q����@��A��C���@�Q��J�H@W�A�\)C�xR                                    Bx��:�  �          @���@�ff���\@   A���C��f@�ff�^�R@P  A�G�C��R                                    Bx��I4  T          @���@��
���\@�A�ffC���@��
�[�@[�A�p�C��                                    Bx��W�  
�          @�G�@������H@��A�z�C���@����]p�@XQ�A�G�C��=                                    Bx��f�  "          @���@�����@��A�G�C��@���^{@X��A��\C��q                                    Bx��u&  
�          @���@��\����@�\A���C���@��\�g
=@fffB�C�9�                                    Bx����  T          @�=q@������@
�HA��RC��3@���g�@^{A��C��)                                    Bx���r  T          @љ�@��R����@	��A���C�t{@��R�i��@\(�A��C���                                    Bx���  T          @љ�@���=q?�p�AQ�C��H@��i��@0  AƸRC�H                                    Bx����  	�          @��@�ff��=q?�  AS
=C��{@�ff�i��@0  AƸRC��                                    Bx���d  "          @�=q@�p���ff?�z�A"ffC��q@�p��h��@��A���C��)                                    Bx���
  �          @��@����33?���A\��C�|)@���j=q@5�A�  C���                                    Bx��۰  �          @�G�@�(�����?�p�Av=qC��H@�(��b�\@=p�AׅC�K�                                    Bx���V  �          @��
@�\)��z�?�z�AD��C��\@�\)�p  @+�A���C��=                                    Bx����  �          @�=q@�G���Q�?���AK�
C�.@�G��g
=@+�A�z�C�|)                                    Bx���  
(          @��@�ff��=q?�  AS�C���@�ff�j=q@0  A�  C��                                    Bx��H  "          @ҏ\@����ff?��AX��C�j=@���b�\@/\)Ař�C��3                                    Bx��$�  �          @�Q�@����ff?��HA+�C�l�@�����@%�A�ffC�W
                                    Bx��3�  �          @��@5���녾\)���C�xR@5����H?�ffAe��C��\                                    Bx��B:  �          @�ff@-p���ff�8Q���Q�C�7
@-p����?uA�C�J=                                    Bx��P�  T          @ƸR@�=q��=�\)?#�
C��@�=q��{?�(�A\  C�C�                                    Bx��_�  f          @ƸR@�����ff>�ff@��RC���@������?�\A���C�
                                    Bx��n,  �          @�ff@�ff���þ����mp�C�e@�ff��ff?z�HAG�C��                                    Bx��|�  �          @�{@�33���H>���@2�\C��@�33����?�=qAn�\C���                                    Bx���x  �          @��H@�(���G������6�RC��R@�(���ff>�?��RC�o\                                    Bx���  �          @\@w
=���R����L��C�p�@w
=���=L��>�ffC��{                                    Bx����  �          @��H@q����^�R�
=C�w
@q���
=?\)@�=qC�T{                                    Bx���j  �          @�33@n�R��Q�J=q��{C��@n�R����?(��@�C���                                    Bx���  T          @��H@j=q��ff��33�/
=C���@j=q���\>�z�@.�RC���                                    Bx��Զ  T          @\@aG����׿���K33C�0�@aG���ff>�?��RC��\                                    Bx���\  �          @�G�@^�R����=q�r�\C�O\@^�R������  C��{                                    Bx���  �          @��@hQ���ff�n{�{C�Ф@hQ�����>��H@�
=C��                                     Bx�� �  �          @\@aG�����=q�!�C���@aG���G�?�Q�A4��C�&f                                    Bx��N  �          @\@N{�����J=q��\)C�C�@N{���?8Q�@�z�C�=q                                    Bx���  �          @�=q@|(����ÿ�  ���C�K�@|(������
=�~�RC�B�                                    Bx��,�  �          @��H@�(���������o\)C�^�@�(���{��z��-p�C�t{                                    Bx��;@  �          @�33@����G���G��d��C�k�@��������=q�!�C���                                    Bx��I�  �          @���@�33�tz�ٙ����RC�]q@�33��z����C�0�                                    Bx��X�  �          @���@�Q���녿�p��=�C�.@�Q���  �#�
�uC���                                    Bx��g2  �          @��@����G���=q�K�C�l�@����  ��G�����C���                                    Bx��u�  �          @�=q@���w
=��
=���RC�>�@�����   ��
=C�)                                    Bx���~  �          @�G�@�G��~�R����  C���@�G�����333�׮C�h�                                    Bx���$  
�          @�G�@�ff�:=q�,(���ffC�g�@�ff�c�
�޸R���C��                                    Bx����  �          @��@����z��`  �  C���@����N{�,(���C��                                     Bx���p  �          @�  @�  �=q�Fff��=qC��=@�  �L(���\��{C�S3                                    Bx���  �          @���@�G���\�Vff��RC��f@�G��:=q�(Q���{C��                                    Bx��ͼ  �          @�  @�\)�]p���
=���HC�&f@�\)�w��^�R�ffC��R                                    Bx���b  �          @ƸR@A����R@
=qA��C���@A�����@^�RBG�C��                                    Bx���  �          @�\)@`  ��Q�?��
A7�C���@`  ��{@-p�A��C�Z�                                    Bx����  �          @�  @U���\)?xQ�A
=qC���@U����@�RA��C�޸                                    Bx��T  �          @θR@QG����?n{A��C�j=@QG���Q�@�A��C��R                                    Bx���  �          @�  @U���R?��
AC��{@U���R@ ��A�z�C��R                                    Bx��%�  �          @���@:=q��Q�@z�A��\C�s3@:=q���@l(�B�HC���                                    Bx��4F  �          @���?�Q���  @���B�RC�R?�Q��\(�@�{BS�C���                                    Bx��B�  �          @�=q@@����Q�?�  A��\C��f@@����=q@H��A��C��3                                    Bx��Q�  T          @Ӆ@l(���{>�@�z�C�R@l(����H?��HA���C���                                    Bx��`8  �          @��@|(���=q?�@�C�AH@|(���ff@G�A�p�C�=q                                    Bx��n�  �          @�p�@��R���
?.{@��C��R@��R���@�
A�Q�C��3                                    Bx��}�  �          @�p�@�=q��ff?k�@��RC�@�=q��  @z�A���C�N                                    Bx���*  �          @�{@�ff���
?fff@�
=C���@�ff��@G�A��\C���                                    Bx����  �          @�z�@����p�?fff@��\C�R@����
=@�A��HC�Z�                                    Bx���v  �          @��H@y������?�ffA6=qC���@y����33@*=qA�=qC�                                      Bx���  �          @���@i����\)?��
A5G�C�s3@i����{@)��A�z�C��{                                    Bx����  �          @�G�@X����  ?�33A��C�j=@X����G�@P  A�RC�]q                                    Bx���h  �          @У�@A����@!�A��C�4{@A�����@u�B��C��H                                    Bx���  �          @�\)@>�R����@�A�p�C���@>�R����@a�BQ�C��\                                    Bx���  �          @�z�@A���p�?У�Ac�
C��@A�����@FffAޣ�C��=                                    Bx��Z  �          @Ӆ@*�H��(�?���A@Q�C�9�@*�H��G�@9��AЏ\C�c�                                    Bx��   T          @�(�@4z���(�?�
=A$  C���@4z���33@-p�A���C��                                    Bx���  �          @Ӆ@=p����?�R@�33C�g�@=p����@
=qA���C�/\                                    Bx��-L  �          @�=q@'
=���
?���A9G�C�H@'
=���@5�A�z�C�)                                    Bx��;�  �          @���@5����?�ffA  C�R@5��G�@"�\A���C�q                                    Bx��J�  �          @�(�@5���p�?G�@ᙚC�B�@5���Q�@  A��C�"�                                    Bx��Y>  �          @���@%����?5@˅C�
=@%���@p�A�  C��                                    Bx��g�  �          @�z�@#33���\>�@���C��{@#33��  ?�p�A�z�C�s3                                    Bx��v�  �          @�{@����R?.{@��
C�|)@���=q@{A�(�C�&f                                    Bx���0  �          @��@e���z�?�(�A}�C��@e�����@<(�Aޣ�C��q                                    Bx����  �          @θR@#�
����@C�
A�Q�C��@#�
��G�@���B&��C�+�                                    Bx���|  �          @љ�@R�\����?��Al  C��
@R�\��@=p�AڸRC��3                                    Bx���"  �          @�
=@b�\��녾�{�EC��{@b�\��
=?��AffC��                                    Bx����  �          @�G�@4z���
=@�\A���C��H@4z���Q�@XQ�A��C�Q�                                    Bx���n  �          @���@#�
��@XQ�A���C���@#�
�x��@�G�B1  C��                                    Bx���  �          @��
@4z����\@��A���C�j=@4z���33@`  A��
C�q                                    Bx���  �          @���@ff��{@ ��A�G�C�4{@ff���@xQ�B�C���                                    Bx���`  �          @�z�@HQ���  ?p��A�C�G�@HQ���=q@Q�A���C�=q                                    Bx��	  �          @��
@@  ��Q�?��RA,��C��@@  ��Q�@*�HA�=qC��                                     Bx���  �          @�(�@<����33?s33Az�C�aH@<�����@=qA���C�H�                                    Bx��&R  �          @Ӆ@2�\��?G�@�Q�C���@2�\��G�@��A��C�`                                     Bx��4�  �          @�(�@QG���\)>8Q�?�=qC�޸@QG���  ?�\)Ac�C�b�                                    Bx��C�  �          @Ӆ@.�R���\?��HAL��C��f@.�R����@8Q�A��C���                                    Bx��RD  �          @���@:�H����?n{A�C�.@:�H��\)@Q�A�{C��                                    Bx��`�  �          @��@7���\)?�@���C�Ф@7����@ ��A�z�C�q�                                    Bx��o�  �          @�z�@6ff��\)>��H@��C�� @6ff��p�?�(�A�C�Z�                                    Bx��~6  �          @Ӆ@@����(�>���@5C���@@�����?��
AyC�R                                    Bx����  �          @�z�@Fff���?E�@���C��@Fff��@�A��
C��R                                    Bx����  �          @�(�@O\)����=�?�=qC���@O\)���?�ffAX(�C�                                      Bx���(  �          @�{@W
=���R?c�
@�(�C�B�@W
=���@G�A��\C�+�                                    Bx����  �          @�ff@L�����?��@�C�W
@L����G�?�(�A��RC��                                    Bx���t  �          @�@Y�����>.{?��RC�P�@Y������?���AZ=qC��                                    Bx���  �          @��H@XQ�����<#�
=���C�s3@XQ���\)?��AC
=C�ٚ                                    Bx����  �          @�33@P����ff�(�����C�޸@P����?L��@�{C��                                    Bx���f  �          @�33@QG����Ϳp���
=C��@QG����R>�@�  C��f                                    Bx��  �          @ָR@QG���{��33�@(�C���@QG����<��
>.{C���                                    Bx���  �          @�ff@H����\)���R�MG�C�Y�@H����p��u�
=qC���                                    Bx��X  �          @�=q@6ff���ÿ�
=�&ffC�!H@6ff����>u@
=C��                                    Bx��-�  �          @�G�@9����
=����6ffC�l�@9�����=�G�?�  C�#�                                    Bx��<�  �          @���@7����R����=�C�U�@7����=u?\)C��                                    Bx��KJ  �          @��
@>{��p���
=�j�HC��R@>{���;�\)��C�]q                                    Bx��Y�  �          @�33@E����H��Q��mp�C�k�@E����\���
�0  C���                                    Bx��h�  �          @��
@c33��  ������C���@c33����
=�$��C���                                    Bx��w<  �          @��H@P����G��-p��\C�y�@P�����ÿ����K�C�C�                                    Bx����  �          @�G�@
=��
=�(����
=C��=@
=��{���
�5�C���                                    Bx����  �          @ҏ\@����H�!����C�|)@����׿�33�!��C���                                    Bx���.  �          @��@/\)���R�=q��ffC�Q�@/\)��(������=qC��H                                    Bx����  �          @���@>{��G�>u@ffC���@>{���\?�=qA`z�C��                                    Bx���z  �          @���@(����{��=q�=qC�H@(�����H?�{A�C�.                                    Bx���   �          @У�@,(����Ϳ����
C�Ff@,(����?Y��@���C�XR                                    Bx����  T          @У�@9�����
�����O33C���@9������������C�C�                                    Bx���l  �          @���@@  ��������$Q�C�+�@@  ��p�>#�
?�Q�C��                                    Bx���  �          @��@)����?   @�z�C�f@)�����?�A�\)C���                                    Bx��	�  �          @�=q@dz����\?�{AEG�C���@dz���z�@�RA�C�Ǯ                                    Bx��^  �          @��H@��H�o\)@O\)A�(�C��q@��H�:=q@�Q�B
=C�
                                    Bx��'  �          @ҏ\@`����
=?У�Ai�C��
@`�����R@1G�A�(�C�W
                                    Bx��5�  �          @�=q@5���
?5@�p�C��@5���@ ��A���C��=                                    Bx��DP  T          @ҏ\@7
=��
=?�  AR�HC�O\@7
=��\)@/\)Ař�C�XR                                    Bx��R�  �          @љ�@B�\��p�?�{A��C��@B�\����@ffA��HC���                                    Bx��a�  �          @�(�@���=q?���A&�HC�� @�����@!G�A�\)C�4{                                    Bx��pB  �          @ָR@p���{?�=qA7�C���@p����@+�A�(�C��f                                    Bx��~�  T          @�z�?����=q?�A��C���?����Q�@H��A�\C�k�                                    Bx����  T          @��
?��
��Q�@z�A�z�C�C�?��
����@VffA�  C�"�                                    Bx���4  �          @�(�@��  @%A�G�C���@��G�@r�\B{C�AH                                    Bx����  �          @ָR?˅��\)@AG�AׅC��)?˅��@��RBQ�C��                                    Bx����  �          @�p�@{��=q@l(�B��C��{@{��z�@��RB4G�C��                                    Bx���&  �          @�z�@���=q@�  BC�ff@��u�@��RBCG�C��\                                    Bx����  �          @�?�����33@@  A�C�W
?�����=q@��HB �\C���                                    Bx���r  �          @�{?=p��ʏ\?��@�33C�Q�?=p�����?���A�  C�xR                                    Bx���  �          @�(�?�
=��G��#�
�ǮC���?�
=��p�?�33A5��C�Ǯ                                    Bx���  �          @�
=@Fff��ff@
=qA��RC�h�@Fff���
@N{A��C��3                                    Bx��d  �          @���@C�
���@0  AǙ�C��R@C�
��33@qG�B=qC���                                    Bx�� 
  �          @У�@A����
@(��A���C�J=@A���{@j�HB	ffC�"�                                    Bx��.�  �          @θR@<����
=@6ffA�=qC�]q@<����  @uBp�C�ff                                    Bx��=V  �          @�p�@C33���@'�A�\)C���@C33��=q@g�B	�\C���                                    Bx��K�  
�          @�{@L�����@7
=AԸRC��@L������@s33B�C�>�                                    Bx��Z�  T          @�  @S33��33@1G�A�(�C��@S33���@n{B=qC�1�                                    Bx��iH  �          @�Q�@Mp���\)@.�RA�C�ff@Mp�����@mp�B
��C�^�                                    Bx��w�  �          @�@O\)��
=?�=qA�ffC���@O\)��
=@7�A�
=C�@                                     Bx����  �          @��@?\)��33@�\A��C�/\@?\)��Q�@S33A���C��H                                    Bx���:  �          @���@W
=����@�RA�\)C�� @W
=��{@Z�HB��C�c�                                    Bx����  �          @θR@w
=��
=?�p�A1��C���@w
=���@�RA���C���                                    Bx����  �          @�=q@e���(�@{A��
C�q@e�����@Z�HA��\C���                                    Bx���,  �          @��
@j=q��{?�\Ayp�C���@j=q��
=@333A���C���                                    Bx����  �          @�z�@Y�����?�(�A��HC�S3@Y������@@��A���C���                                    Bx���x  �          @�(�@Tz���G�@�RA�p�C�\@Tz���\)@P��A뙚C���                                    Bx���  �          @�z�@qG����R@��A�p�C��R@qG���p�@J=qA�=qC�9�                                    Bx����  T          @�z�@�p���z�?��Adz�C�:�@�p����R@%A��C���                                    Bx��
j  �          @Ӆ@[���  ?�A#�
C���@[����@  A��
C���                                    Bx��  �          @��@e���Q�?�  A
�\C�~�@e���ff@�A�  C�C�                                    Bx��'�  �          @���@~{���?�\)Ab�\C��@~{��z�@&ffA�z�C�G�                                    Bx��6\  �          @��
@tz���
=?��AAC��@tz����H@��A��
C�                                      Bx��E  T          @љ�@XQ���G�?W
=@�33C��3@XQ�����?�z�A�{C�Z�                                    Bx��S�  �          @Ӆ@��H��?�z�Ah��C�k�@��H��Q�@"�\A��C�                                    Bx��bN  �          @У�@n�R����?aG�@�Q�C���@n�R��  ?�33A�G�C�\)                                    Bx��p�  �          @��
@�33��{?Q�@��HC��@�33��?�\Aw\)C�o\                                    Bx���  �          @�ff@��
��
=?xQ�AC���@��
��{?�A�p�C�~�                                    Bx���@  �          @�p�@u���>�ff@xQ�C�� @u����?��RAO\)C�)                                    Bx����  �          @�ff@qG���ff=�?��C�]q@qG���=q?�A"�\C��                                    Bx����  T          @��@AG���=q?�33A���C�7
@AG���33@<(�A���C�K�                                    Bx���2  T          @�33@�
����@�
A���C���@�
���@W�A�\)C��R                                    Bx����  �          @��@^�R����?���A733C�q@^�R��p�@ffA�G�C���                                    Bx���~  �          @��@}p���(�?�{A>{C��3@}p�����@�
A�p�C���                                    Bx���$  �          @�z�@w
=��z�?�=qA]�C�u�@w
=��  @!�A�G�C��\                                    Bx����  �          @Ӆ@b�\����?�(�Ar�RC���@b�\���@,(�A��\C��                                    Bx��p  �          @�33@2�\��Q�@G�A��C�o\@2�\���@P��A�p�C���                                    Bx��  �          @��@0������?�{Af{C��@0����  @(��A��C��)                                    Bx�� �  �          @�33@�����@�A��C�� @����\)@Mp�A���C��                                    Bx��/b  �          @�(�@Q���ff@z�A��\C�XR@Q���p�@U�A���C�^�                                    Bx��>  �          @��@	����=q@��A��RC�  @	������@[�A�{C��                                    Bx��L�  �          @��@:=q���?��\A1G�C�S3@:=q��\)@z�A��C��                                    Bx��[T  �          @��
@G���=q?�\)Ac�
C���@G���@'
=A��HC���                                    Bx��i�  T          @ָR@�����p�?��A2�RC���@������H@p�A���C��f                                    Bx��x�  �          @��
@S33���@	��A���C��{@S33���\@Dz�AݮC�0�                                    Bx���F  �          @�p�@k���Q�?�
=Ak\)C�xR@k����
@'
=A�
=C��                                    Bx����  �          @�@XQ���=q@(��A�\)C���@XQ���Q�@`��A��C�g�                                    Bx����  �          @�
=@1����@N�RA�RC�#�@1���  @�33B�C��
                                    Bx���8  �          @�
=@*=q����@x��B��C��\@*=q��  @�B0�C�˅                                    Bx����  �          @�z�@�����?�A@Q�C��@�����@�A��
C��R                                    Bx��Є  �          @�(�@�z���{?�Ab{C�K�@�z����@#�
A��
C�Z�                                    Bx���*  �          @�=q@i���z�H@�
=B�HC�y�@i���G�@��HB5  C��H                                    Bx����  �          @�z�@�p��|��@k�B��C�Ff@�p��O\)@�=qB(�C��                                    Bx���v  �          @���@~�R��G�@Q�A�=qC���@~�R�y��@���BffC�Ǯ                                    Bx��  
�          @���@�����G�@   A���C�"�@������@6ffA�
=C�b�                                    Bx���  �          @��@�33��Q�?�33A
=C�t{@�33���@.�RA��RC��f                                    Bx��(h  �          @�(�@s�
��33?�
=AffC�R@s�
��=q@Q�A�ffC���                                    Bx��7  �          @�ff@~{��=q?z�HA�\C���@~{��=q?�A�33C�`                                     Bx��E�  �          @�Q�@��H��(�?E�@���C��@��H���?�(�Ac
=C��=                                    Bx��TZ  �          @߮@��R��
=?�(�A ��C�˅@��R��@��A�C���                                    Bx��c   T          @���@��H��{@33A�(�C�B�@��H�q�@.�RA�Q�C��                                    Bx��q�  �          @���@�����\)@Q�A�{C��\@����s�
@4z�A���C�e                                    Bx���L  �          @�@�p���33@(�A�\)C�C�@�p��z�H@8��A�33C���                                    Bx����  T          @�z�@��
��{?�(�A#33C�\@��
��p�?�p�A�(�C��                                    Bx����  T          @�33@��
��  ?E�@�C��H@��
����?�ffAP��C�y�                                    Bx���>  T          @��
@�ff��\)?\)@���C�'�@�ff��=q?�=qA3
=C���                                    Bx����  �          @�33@�  ��z�>���@Y��C�3@�  ��Q�?���A"�RC�y�                                    Bx��Ɋ  T          @�(�@�=q���
>Ǯ@Q�C�W
@�=q���?�Q�A\)C���                                    Bx���0  �          @��
@�ff��
=>�{@8Q�C�9�@�ff���H?�Q�A   C���                                    Bx����  
�          @ۅ@�33���\�����C�
@�33��Q�?L��@�p�C�C�                                    Bx���|  �          @ڏ\@���p�>�@���C�Ǯ@�����?��\A,Q�C�33                                    Bx��"  �          @�G�@����
?(��@��HC��@���ff?�Q�AC�C�s3                                    Bx���  �          @ٙ�@��H��z�?^�R@�(�C��
@��H��{?��A`��C�1�                                    Bx��!n  �          @��@�������?(��@���C��@������?���AD��C��\                                    Bx��0  T          @�G�@��H���>���@3�
C�J=@��H��(�?�\)A(�C��H                                    Bx��>�  �          @�\)@�
=��p�?�@�C�!H@�
=����?��A2=qC��                                    Bx��M`  �          @�p�@�(����=#�
>�{C�,�@�(���?J=q@��C�`                                     Bx��\  �          @�@�Q���(�>�?���C��R@�Q�����?n{A z�C�3                                    Bx��j�  �          @�{@�����33>�(�@j=qC�\@�����
=?�(�A(  C�k�                                    Bx��yR  �          @��@hQ����=���?Y��C���@hQ���\)?uAG�C��                                    Bx����  �          @���@u����\?\(�@�ffC��
@u���z�?�Aj=qC�Z�                                    Bx����  �          @�z�@~{��Q�?5@�p�C���@~{���H?\AS33C��q                                    Bx���D  �          @��@q�����>�(�@n�RC�y�@q�����?�  A.�HC��                                    Bx����  �          @�p�@dz���{?�33AB{C���@dz����@p�A�G�C�U�                                    Bx��  �          @���@��\���׾L�Ϳ޸RC�aH@��\���?\)@��
C�u�                                    Bx���6  �          @�{@P  ��p�?���Ac
=C�w
@P  ���
@��A��RC�33                                    Bx����  �          @�{@qG���ff=L��>�
=C�S3@qG���(�?^�R@��C�~�                                    Bx���  �          @�
=@aG���p���p��J=qC��f@aG����>�@���C��                                    Bx���(  �          @׮@y�����
?�G�A33C��R@y�����?�ffAyp�C��                                    Bx���  �          @�
=@y����{>�G�@p  C�˅@y����=q?��RA*�\C�q                                    Bx��t  �          @�\)@l(���33��p��L��C��{@l(����H>�@x��C��
                                    Bx��)  �          @�{@j=q��녿����{C���@j=q���\>�\)@�C��                                     Bx��7�  
�          @�  @|����{��R��G�C��q@|����
=>8Q�?\C��                                    Bx��Ff  T          @�ff@������\�B�\����C��@�����(��#�
��C��
                                    Bx��U  �          @ָR@���Q�=�G�?k�C��)@���{?Tz�@�z�C��                                    Bx��c�  �          @ָR@�z���G�>�?��C��H@�z���
=?Y��@�=qC���                                    Bx��rX  �          @�\)@�33��G��(����RC���@�33��=q>��?���C��3                                    Bx����  �          @׮@}p���{���z�HC�@}p���ff>��
@.{C���                                    Bx����  �          @�Q�@����������=qC�aH@������?0��@��C���                                    Bx���J  T          @أ�@��H���
�����
=C�,�@��H����=�G�?h��C�{                                    Bx����  �          @���@�������   ��
=C��)@�����(�>B�\?��C�˅                                    Bx����  �          @�Q�@�p�������ff�s33C�"�@�p���=q>#�
?�{C��                                    Bx���<  �          @���@�33���R��z��@  C�8R@�33���
�O\)���HC��
                                    Bx����  T          @�G�@��������G��q�C��3@����=q>#�
?��C���                                    Bx���  �          @�Q�@�(���z῾�R�MG�C�|)@�(������W
=��p�C���                                    Bx���.  T          @�G�@��
��Q쿜(��'
=C�3@��
��(��\)���C���                                    Bx���  �          @��@�(���녿�33�\)C���@�(���������
C��R                                    Bx��z  �          @�=q@�\)����}p����C�y�@�\)���\�����X��C�(�                                    Bx��"   �          @�33@�z���33��ff�G�C��H@�z���ff��ff�qG�C��                                    Bx��0�  T          @��H@�=q�����\)�8��C���@�=q��Q�E���ffC�.                                    Bx��?l  �          @��H@���������z�C���@����Q���
�O\)C��                                    Bx��N  T          @��
@����~{�p���{C�0�@�����\)��z��_�
C�N                                    Bx��\�  T          @ڏ\@�=q�}p��0����
=C�K�@�=q��G��p���\)C�(�                                    Bx��k^  �          @ڏ\@�
=�~�R��H���RC���@�
=���׿����  C���                                    Bx��z  �          @��H@�G������\)��p�C���@�G����ÿٙ��f{C��{                                    Bx����  �          @�33@���u�>�R�θRC��\@����ff�p���=qC�o\                                    Bx���P  �          @�=q@�=q��  �>{�Џ\C�e@�=q��33��H��=qC�1�                                    Bx����  �          @��
@<(��=p���(��Kz�C�q�@<(��e��ff�5ffC��=                                    Bx����  
�          @љ�?��ÿ�G�������C��?����!G������v�
C��                                    Bx���B  �          @ҏ\@{�������H�qC��q@{��\)�����G�C���                                    Bx����  �          @�G�?�=q��33���L�C��?�=q��
=���R�C��f                                    Bx����  �          @˅?�{��ff��(���C��=?�{����p�.C�9�                                    Bx���4  �          @˅?�
=����G��r�HC�33?�
=�G
=��{�[�C��f                                    Bx����  �          @��?�\)�	�����
�|Q�C��H?�\)�5����eQ�C���                                    Bx���  �          @�(�?��Ϳ���(�� C�o\?��Ϳ�
=���qC�p�                                    Bx��&  �          @��H?��H<��ƸR�
?�Q�?��H�=p����  C�T{                                    Bx��)�  �          @�(�?�Q���������C��{?�Q쿐������u�C�4{                                    Bx��8r  �          @���@U��   ��=q�8(�C��3@U��@���~�R�&33C���                                    Bx��G  �          @���@O\)�qG��c�
�#\)C�S3@O\)�vff��(���33C�f                                    Bx��U�  �          @˅@AG�����?У�Ao�C���@AG����@��A�{C�n                                    Bx��dd  �          @Ϯ@n{���?�\A~�\C�U�@n{���@
=A�p�C��                                    Bx��s
  
�          @��@w���(�?�z�Ao�C�5�@w���z�@\)A��RC��                                    Bx����  �          @��
@\����Q�@#�
A��RC�Ff@\�����@H��A�C�<)                                    Bx���V  �          @��@;���\)@6ffA�p�C���@;���33@\��A��RC���                                    Bx����  �          @�@5��{@E�A�(�C�U�@5��G�@k�BC�S3                                    Bx����  �          @�@4z�����@K�A�p�C�Y�@4z����@p��B	=qC�`                                     Bx���H  �          @ָR@'�����@Mp�A�
=C�4{@'���33@s�
B
p�C�(�                                    Bx����  �          @�  @Dz���p�@C33A�\)C�Z�@Dz�����@h��Bz�C�]q                                    Bx��ٔ  �          @���@<�����@@  A�ffC���@<����p�@fffB G�C�q�                                    Bx���:  �          @���@#�
���\@5�A�  C�W
@#�
��
=@]p�A��C��                                    Bx����  �          @�
=@=p���  @l��B�C��
@=p���G�@�
=B��C�K�                                    Bx���  �          @�\)@G����H@EAڏ\C���@G����R@i��Bp�C���                                    Bx��,  T          @�\)@I�����@^{A��\C���@I����z�@\)B��C���                                    Bx��"�  
(          @�
=@XQ���{@\(�A�{C���@XQ�����@|��BC�.                                    Bx��1x  �          @�ff@?\)��G�@eB�RC���@?\)���@�33B\)C�4{                                    Bx��@  �          @�
=@8Q���z�@w
=B33C���@8Q���p�@��B"�RC�N                                    Bx��N�  �          @�\)@&ff���@�p�B��C���@&ff��=q@��B/�
C�G�                                    Bx��]j  �          @Ӆ@�����@\)B�HC��@���@�\)B+(�C�R                                    Bx��l  �          @�Q�@(����p�@QG�A���C�)@(������@r�\B�C�q                                    Bx��z�  �          @���@Dz����@7�A��C��3@Dz���z�@Y��A��C��H                                    Bx���\  �          @У�@P����p�@1G�A�33C�@P�����H@R�\A�z�C���                                    Bx���  �          @��@_\)����@%A�z�C��
@_\)���H@G
=A��C��)                                    Bx����  �          @�Q�@}p���G�?�=qA�G�C��@}p���=q@A��C�q�                                    Bx���N  �          @Ϯ@�����{?��
A6�RC���@�������?�  A{33C�B�                                    Bx����  �          @Ϯ@������?E�@ڏ\C��3@������?��HA,(�C�/\                                    Bx��Қ  
�          @�@�Q���  >B�\?�p�C��R@�Q��|��?�R@���C��                                     Bx���@  �          @�  @����c33�p���
�\C�c�@����hQ�\)��z�C�
                                    Bx����  �          @�\)@�����Ϳ�����z�C��\@����(Q��=q�l  C�f                                    Bx����  �          @�{@�\)�\)�����N�HC�@�\)�(Q쿎{�&=qC�33                                    Bx��2  �          @ƸR@���z��=q�mG�C�� @���{��{�K33C�1�                                    Bx���  �          @�G�@��H��
�   ����C�޸@��H��׿��
���C���                                    Bx��*~  �          @ʏ\@��
���������ffC�j=@��
�
=q��
=��(�C�xR                                    Bx��9$  �          @ə�@�p��*�H��  ����C��@�p��5���H�W\)C�1�                                    Bx��G�  �          @�{@�p��*�H��  �;�C��@�p��1녿z�H���C�g�                                    Bx��Vp  �          @�=q@�(��-p���R���C���@�(��0�׾�33�Q�C�o\                                    Bx��e  �          @�@�\)����R�\C���@�\)�Q�����Q�C�^�                                    Bx��s�  �          @�z�@�ff��p���H�ÅC�q@�ff�p��p�����C���                                    Bx���b  T          @���@���xQ��tz��'Q�C�s3@������l���!33C��                                    Bx���  T          @�=q@��
�\)�s33�#
=C�S3@��
�p���n{�(�C��3                                    Bx����  T          @�p�@��׿���[���C��R@��׿aG��W
=�
��C��                                     Bx���T  �          @�33@�33�#�
����4G�C���@�33������H�3(�C�#�                                    Bx����  �          @�ff@�G��Ǯ�Z�H�
=C�n@�G�����P  �G�C�z�                                    Bx��ˠ  �          @�  @[��HQ��P���p�C�Ф@[��[��<(���(�C��                                     Bx���F  �          @��H@c33��\)���G��C�@ @c33��������@
=C�E                                    Bx����  �          @�(�@��
��=q��33�4ffC�
=@��
�޸R��ff�,�RC���                                    Bx����  �          @�(�@�������p  �=qC�K�@����p��g��p�C�P�                                    Bx��8  T          @ȣ�@�Q쿡G��w��C��3@�Q��\)�n�R�p�C���                                    Bx���  �          @�G�@�녿�  �vff�{C���@�녿�{�n{��C��f                                    Bx��#�  �          @�
=@��H��ff�e���C��@��H����Z=q�Q�C�aH                                    Bx��2*  �          @�@�\)�����`���
�RC�|)@�\)����Tz��p�C��
                                    Bx��@�  �          @�ff@�
=��33�u��(�C�xR@�
=�  �h���33C���                                    Bx��Ov  �          @ƸR@�G�����r�\�ffC���@�G���R�fff��C��\                                    Bx��^  
�          @�ff@�33�˅�_\)�	z�C���@�33��33�U��\)C�H�                                    Bx��l�  �          @�@�Q�ٙ���G��%ffC�  @�Q���
�w���C�
                                    Bx��{h  �          @�
=@�녾��
���C�
C�� @�녿=p���(��A�C�Ф                                    Bx���  �          @�
=@p  �\)�����R�C��3@p  �z����
�P(�C��)                                    Bx����  �          @Ǯ@Vff<���
=�e33?�@Vff��G���ff�c��C�E                                    Bx���Z  �          @ȣ�@J=q?������gG�A�Q�@J=q?(���z��l��A/33                                    Bx���   �          @�33@Y��?�Q�����X�RA��@Y��?z�H��
=�_�\A���                                    Bx��Ħ  �          @ȣ�@U�?�{���_�A��H@U�?#�
��Q��d�A.�H                                    Bx���L  �          @Ǯ@P��?�33���a  A�\)@P��?0����Q��fz�A?33                                    Bx����  �          @Ǯ@   ?J=q���=qA��R@   >�z����RaH@�
=                                    Bx���  �          @�
=@.�R?s33��  �y(�A��@.�R>������~(�A�                                    Bx���>  �          @�\)@*=q?.{���H�ffAd(�@*=q>B�\��(�L�@��\                                    Bx���  
�          @�  @C33?+���(��oz�AEG�@C33>L������rG�@qG�                                    Bx���  �          @�\)@'�?fff����~  A��
@'�>�
=���ffAff                                    Bx��+0  �          @�Q�@L(�>aG����
�l��@���@L(��k����
�l�C��f                                    Bx��9�  �          @�G�@E�>�����R�r\)@5�@E����R���R�q�
C�)                                    Bx��H|  �          @ʏ\@C�
�������tG�C��@C�
�����  �q��C�xR                                    Bx��W"  �          @˅@4z�>Ǯ�����|��@�@4zὣ�
����~  C�+�                                    Bx��e�  T          @�z�@�����\)���J�C��@���������I=qC��                                     Bx��tn  �          @ʏ\@C�
�u��\)�s=qC�o\@C�
��\���R�qz�C�:�                                    Bx���  �          @�  @E>#�
���qp�@:=q@E��z����q
=C�L�                                    Bx����  �          @ʏ\@^{>��������bz�@��@^{��G���G��c
=C�R                                    Bx���`  �          @ə�@w
=>��H��{�O  @�{@w
==��
���R�P\)?�(�                                    Bx���  �          @��@��\>����G��E\)@��
@��\=��
����F��?�                                    Bx����  T          @��
@{�?k����J��AR�H@{�?�����N  @��                                    Bx���R  �          @��
@~{?�������F��Aw�
@~{?5���J�HA!p�                                    Bx����  �          @�z�@y��?z���Q��OQ�A�\@y��>.{��G��Q{@#33                                    Bx���  �          @�z�@w
=?\(���  �N�RAI��@w
=>������Q��@�=q                                    Bx���D  �          @��
@(��@�R�����[
=B �@(��?�����f{B(�                                    Bx���  �          @ȣ�?�p�@s33���R�Dp�B�p�?�p�@Z�H����U(�B�(�                                    Bx���  �          @�Q�?�ff@�(���=q�/G�B�?�ff@��������@�B�                                    Bx��$6  �          @��?���@R�\��G��T��Br�
?���@8�������c��Be�                                    Bx��2�  �          @��@P  >�=q��{�V�R@��\@P  ��\)��ff�WG�C�aH                                    Bx��A�  �          @�{@~{�B�\���H�<{C��)@~{��������8{C�U�                                    Bx��P(  �          @�Q�@s33�����
�I�C��@s33�aG���=q�E�C�z�                                    Bx��^�            @��@b�\�J=q�w��;G�C��R@b�\��=q�s33�6��C��H                                    Bx��mt  �          @��@Z=q��G��^{�.�HC���@Z=q�\�W��(Q�C���                                    Bx��|  �          @��@Z=q�5��=q�L�C�3@Z=q�����Q��H\)C���                                    Bx����  �          @��\@r�\�Q�����@  C�� @r�\�����\)�;�C���                                    Bx���f  T          @�33@��ÿ�����  �1��C�Z�@��ÿ�����
�+
=C�y�                                    Bx���  �          @���@�z῔z���(��){C��3@�zῼ(������$(�C��H                                    Bx����  �          @�\)@��H��G��QG���
=C�@��H�޸R�I�����C���                                    Bx���X  �          @ƸR@�33���
�[��ffC��@�33���R�\� 33C���                                    Bx����  �          @�  @��
�33�I����\C�XR@��
�!��>{�㙚C�G�                                    Bx���  �          @�{@�z��  �@����\)C���@�z��p��5����HC���                                    Bx���J  T          @Å@�\)�(��C33��\C��@�\)�=q�8Q���  C��                                     Bx����  �          @�33@�G��0  �QG����C���@�G��>�R�C�
����C��q                                    Bx���  
Z          @�G�@��
�E�.�R��(�C�c�@��
�Q��   �ĸRC���                                    Bx��<  
�          @�33@�=q�\(���33��{C���@�=q�dz��33��\)C�<)                                    Bx��+�  �          @�33@�  �E��\��(�C��f@�  �N�R������p�C�/\                                    Bx��:�  �          @�p�@�Q��*=q�33���HC�k�@�Q��3�
�ff����C���                                    Bx��I.  �          @��
@�=q�p��   ��=qC���@�=q������Q�C��{                                    Bx��W�  T          @��R@��׿�33�*=q��ffC�y�@��׿�=q�"�\��\)C��H                                    Bx��fz  T          @�{@��H��{��Q�C�� @��H�����
�£�C��f                                    Bx��u   �          @�Q�@���HQ��Q���  C�T{@���P  ��(���\)C���                                    Bx����  �          @���@����\(���Q����\C��q@����c33��Q��h��C�.                                    Bx���l  T          @�{@w
=�x�ÿ����mC�` @w
=�~�R��
=�@z�C�
=                                    Bx���  �          @���@|(��U���(���p�C��f@|(��]p��޸R����C�]q                                    Bx����  �          @��@ ����33?z�@�z�C��@ ������?\(�A#�C��                                    Bx���^  �          @�33@
=���\?�ffAc33C���@
=��\)?�{A�33C��                                    Bx���  �          @�\)@N{���
>�{@g
=C�1�@N{���H?#�
@�\)C�L�                                    Bx��۪  �          @�@X����z�:�H��  C���@X����p���ff��Q�C���                                    Bx���P  �          @�(�@:=q����!G���G�C�� @:=q��{��33�xQ�C�e                                    Bx����  �          @��@\���R�\��33�Yp�C�4{@\���W
=�n{�/
=C��                                    Bx���  �          @��R@p  �=p������w�C�ٚ@p  �B�\��\)�Q��C�xR                                    Bx��B  �          @�
=@\)�\)��\)���\C�@\)�%���H��=qC��                                     Bx��$�  �          @��\@��Ϳ�(��У���\)C��R@��Ϳ�=q��G����C�q                                    Bx��3�  �          @�  @��\�B�\��
��C���@��\��{��\��C���                                    Bx��B4  �          @��@n�R�����8Q����C��R@n�R�z��5��
C���                                    Bx��P�  �          @��@mp�>aG��3�
��\@X��@mp�=#�
�4z��{?�                                    Bx��_�  �          @�G�@�33��z῁G��M�C��R@�33��(��fff�6�\C���                                    Bx��n&  �          @�(�@���z��
=����C��H@���
�H������C��\                                    Bx��|�  T          @���@�\)�޸R������C���@�\)��\)�   ��\)C��                                    Bx���r  
�          @�(�@�  ��G���33����C�T{@�  ��{���
���
C��                                    Bx���  
�          @��R@�����ٙ����C�b�@����������33C��
                                    Bx����  �          @�ff@�ff�G���\)�<(�C��H@�ff�HQ콣�
�Y��C�t{                                    Bx���d  �          @�Q�@�\)�`��>L��@�C�#�@�\)�_\)>�
=@�C�8R                                    Bx���
  �          @�G�@����`  =�\)?=p�C�XR@����_\)>�z�@AG�C�c�                                    Bx��԰  �          @��@�ff�g�>���@\��C�� @�ff�e?\)@�G�C��)                                    Bx���V  �          @��@�Q��c�
?=p�@��C��@�Q��`��?uAp�C�Ff                                    Bx����  �          @���@����3�
?Q�A	C�j=@����0  ?}p�A&�HC���                                    Bx�� �  �          @�G�@�p���{�����C���@�p���\)    �L��C���                                    Bx��H  �          @�=q@�
=��>�{@n{C��q@�
=��>�@�C��q                                    Bx���  �          @��@�
=���R?�@���C�O\@�
=���H?#�
@��HC��                                     Bx��,�  �          @���@�ff�   ?�@��C�c�@�ff��?��@�
=C��3                                    Bx��;:  �          @��@��R��(���G���z�C��H@��R��ff��
=���C���                                    Bx��I�  T          @��@��׾������B�RC��
@��׾��ÿ�\)�?�C�8R                                    Bx��X�  �          @�G�@�ff��p���ff���C��
@�ff�����
��G�C�8R                                    Bx��g,  �          @�@��\��z��
=��ffC��@��\���
��\��G�C�33                                    Bx��u�  �          @�=q@�z�Q��33��RC�c�@�z�u�  ����C�|)                                    Bx���x  �          @�\)@xQ�<��
�E��>k�@xQ�.{�E���
C��H                                    Bx���  �          @��H@�{�J�H�n{� (�C�t{@�{�N{�=p���{C�=q                                    Bx����  �          @�{@�  �>�R�����xQ�C��H@�  �C�
���H�W�C�Ff                                    Bx���j  �          @�G�@mp����þ����\)C�XR@mp�����=��
?aG�C�W
                                    Bx���  �          @�=q@!�����?�(�A�ffC�N@!���G�?��RA�  C��
                                    Bx��Ͷ  �          @��@'�����?��HA�{C��R@'���p�@�RA�\)C�O\                                    Bx���\  �          @��H@C�
����?���A0��C���@C�
��\)?��A]G�C�/\                                    Bx���  T          @�33@N�R��G�>��@���C���@N�R��  ?:�H@��C�Ф                                    Bx����  
�          @��@5����R?.{@�33C�~�@5����?uA33C��                                     Bx��N  �          @�=q@/\)���\>\)?�(�C��=@/\)���>�
=@�33C��{                                    Bx���  �          @��@3�
���H��
=��{C��)@3�
��33�����{C���                                    Bx��%�  �          @�Q�@\)��녿Tz����C�Q�@\)��33����ffC�7
                                    Bx��4@  �          @�ff@'
=��������G�C�ff@'
=��Q����z�C�\)                                    Bx��B�  �          @�Q�@G����H���
�8Q�C�c�@G����\>��@%�C�h�                                    Bx��Q�  �          @�
=@K���  >\)?�{C��@K����>��@��\C��                                    Bx��`2  �          @�Q�@N�R��  >�33@`  C�R@N�R��\)?�R@�ffC�+�                                    Bx��n�  �          @�p�@U����?!G�@��
C�9�@U��\)?aG�A�
C�Z�                                    Bx��}~  �          @�{@l(���  ?Q�A��C�z�@l(���ff?��A+�
C���                                    Bx���$  �          @�p�@\)�vff?��HAD��C��@\)�qG�?�Ahz�C�J=                                    Bx����  �          @�@b�\���
?:�H@��C���@b�\��=q?z�HA�\C���                                    Bx���p  �          @�@S33��Q�?��
A&�\C�3@S33��ff?��
AP  C�Ff                                    Bx���  T          @���@b�\��?�G�A!C�H�@b�\���
?�G�AIC�|)                                    Bx��Ƽ  �          @��@Vff���?fffA33C��{@Vff��33?�z�A8��C�                                      Bx���b  �          @��
@Tz����R?:�H@��C�U�@Tz����?z�HA!G�C�|)                                    Bx���  �          @��
@\�����?E�@�C�+�@\�����?��\A&�HC�T{                                    Bx���  �          @���@q����\>�G�@�C��\@q�����?.{@�=qC��=                                    Bx��T  �          @�@n{��Q�>�ff@�33C��@n{��\)?0��@�\)C���                                    Bx���  �          @���@a���(�?n{A��C���@a���=q?�
=A8��C�ٚ                                    Bx���  �          @�@<������@�
A��RC�z�@<�����@#�
A��
C�޸                                    Bx��-F  �          @��@3�
���@�RA���C���@3�
��\)@�RA�
=C�f                                    Bx��;�  �          @�(�@1���G�?�
=A��
C��@1���@(�A��C�b�                                    Bx��J�  �          @��@HQ�����@��A�
=C���@HQ���z�@{A�=qC�T{                                    Bx��Y8  T          @�(�@Q���G�@��A�z�C���@Q����@=qA�\)C��)                                    Bx��g�  T          @�G�@A���G�@{A��C���@A����@\)A���C��
                                    Bx��v�  �          @��@<(����R@Q�A��C�Y�@<(����H@��A�G�C���                                    Bx���*  �          @�{@2�\���H@ffA�
=C�XR@2�\��
=@�A���C���                                    Bx����  �          @ȣ�@.{���R@
=qA�33C��=@.{���\@(�A�33C�{                                    Bx���v  �          @�
=@5����@ ��A�
=C�k�@5����@�\A��RC��3                                    Bx���  �          @�G�@:�H��{@   A�Q�C��@:�H��=q@G�A��
C��{                                    Bx����  �          @��@>�R��@ ��A�ffC���@>�R��=q@�A�C�=q                                    Bx���h  �          @�33@>�R��Q�?�Q�A�{C��f@>�R����@{A��C�
=                                    Bx���  �          @�z�@b�\��33?��
A^=qC�` @b�\��Q�?�A�\)C��q                                    Bx���  �          @��H@Vff��(�?ǮAe�C��\@Vff��G�?�A��C�˅                                    Bx���Z  �          @ʏ\@QG���ff?\A^�\C��@QG����?�ffA�{C�K�                                    Bx��	   T          @��H@\�����?�Q�AR=qC�f@\������?��HAz�HC�>�                                    Bx���  �          @�(�@8�����?���Am��C��R@8����=q?�A�(�C�Ф                                    Bx��&L  T          @ə�@8Q����?�\A��HC�B�@8Q����R@33A�ffC�~�                                    Bx��4�  �          @ə�@@  ��p�?�p�A���C�@@  ���@��A�{C�U�                                    Bx��C�  �          @�=q@E���z�?�=qA��RC��@E���G�@A�G�C�b�                                    Bx��R>  �          @��H@^�R���?��A)G�C�޸@^�R��?�ffAO�
C��                                    Bx��`�  �          @�@_\)����?0��@�\C��H@_\)�~�R?fffA{C��f                                    Bx��o�  �          @��\@o\)��ff?�  Ao
=C�ٚ@o\)���?�(�A���C�#�                                    Bx��~0  �          @�  @u����
?�
=A>ffC�q�@u�����?�33AaC��\                                    Bx����  �          @��@|���z=q?��A)C��@|���vff?�  AK�
C�޸                                    Bx���|  �          @���@u���33>�ff@�G�C���@u���=q?(��@�  C���                                    Bx���"  �          @�Q�@j=q����?Y��A�C�\@j=q��  ?��A1C�=q                                    Bx����  �          @���@\���c�
?uA0  C�\@\���`  ?�33AR�RC�H�                                    Bx���n  �          @�\)@Z=q�z�H>�@���C��H@Z=q�x��?(��@陚C��)                                    Bx���  �          @��@fff�x��?L��A	p�C�j=@fff�u?�G�A,��C��
                                    Bx���  
�          @��@y���e@��A�z�C���@y���^{@��A���C�/\                                    Bx���`  �          @���@w
=�\��?�
=A�\)C��@w
=�U@
=A�\)C���                                    Bx��  �          @��R@��H�2�\@33AǙ�C��R@��H�*=q@��A�
=C�t{                                    Bx���  �          @�{@^{�C33@�
A���C�Y�@^{�:�H@{A�RC��3                                    Bx��R  �          @�33@L���`��@�HA��C�*=@L���XQ�@&ffA陚C��
                                    Bx��-�  �          @���@x���.�R@1�A�RC�xR@x���%�@:�HB (�C�9�                                    Bx��<�  �          @��
@s33�@9��B{C�/\@s33��@AG�B
=qC�                                    Bx��KD  �          @���@z=q����@!G�A��
C�~�@z=q��Q�@'
=A�C�Q�                                    Bx��Y�  �          @��\@}p��&ff?
=q@�ffC�W
@}p��$z�?+�A��C��H                                    Bx��h�  �          @�33@����
�H?���A��\C���@����ff?�
=A��
C�AH                                    Bx��w6  �          @�ff@���z�?���A�=qC���@���   ?��RA��RC�c�                                    Bx����  �          @�Q�@��Ϳ�  >��@���C���@��Ϳ�p�?\)@θRC���                                    Bx����  �          @�  @�\)��p�?��@�ffC�S3@�\)��Q�?.{@��HC���                                    Bx���(  �          @�Q�@�
=�Ǯ?�\@�(�C���@�
=���
?
=@�=qC��                                    Bx����  �          @�=q@�33���>�\)@L��C�  @�33��\)>�33@���C�R                                    Bx���t  �          @�ff@�(���G�>L��@\)C��@�(���  >���@P��C�!H                                    Bx���  �          @�Q�@�
=�ٙ�>8Q�?��HC��3@�
=��Q�>�=q@;�C���                                    Bx����  �          @�ff@�  ��z�����
C�  @�  ��z�=L��?�\C�!H                                    Bx���f  �          @��
@�\)�\����@z�C���@�\)��(�����<��C���                                    Bx���  �          @�=q@�(��Y���h���%��C�
=@�(��fff�\(����C��f                                    Bx��	�  �          @�{@�=q�J=q��\)��Q�C�  @�=q�^�R��=q��  C��)                                    Bx��X  �          @�\)@�\)�\)�����yC�*=@�\)�W
=����x  C��q                                    Bx��&�  	�          @���@�ff��Ϳz����HC��3@�ff�z����ǮC���                                    Bx��5�  �          @�\)@��;��
�J=q�{C�  @��;�Q�E���RC��H                                    Bx��DJ  "          @�ff@��
��z�O\)��C�Ff@��
��{�L���z�C�f                                    Bx��R�  �          @���@�p��#�
�޸R��(�C���@�p����Ϳ޸R���
C�Q�                                    Bx��a�  T          @��\@���?z�H�333�Q�ABff@���?p�׿@  ��\A;
=                                    Bx��p<  �          @�@�ff?��R>�
=@�Ay�@�ff?�G�>�33@�33A|��                                    Bx��~�  
�          @���@��?�Q�>�?�=qA��
@��?�Q�<�>��
A�=q                                    Bx����  
Z          @��@��?�=q�8Q��
�HA��@��?��þ���L(�A�                                      Bx���.  �          @�{@e�@�
���
���A�@e�@�\��(���{A�                                    Bx����  T          @�Q�@�녾aG��   ����C��f@�녾��ÿ�p���33C���                                    Bx���z  �          @�{@p�׾L���'
=�
=C�q�@p�׾�{�&ff�
�C�b�                                    Bx���   �          @�\)@S33��=q�U��4z�C�� @S33��ff�S�
�3�C�R                                    Bx����  @          @��@j=q���R�Vff�)��C��)@j=q���H�U��(G�C�7
                                    Bx���l  
�          @�\)@r�\�.{�Mp�� �HC��@r�\��33�L(�� 
=C�aH                                    Bx���  
(          @�(�@z�H�L���5��33C��=@z�H��33�4z��Q�C�n                                    Bx���  �          @��\@�33�u��R��p�C���@�33�B�\��R���RC��=                                    Bx��^  �          @��\@���>8Q�#�
���H@
=@���>.{�.{��@                                       Bx��   
Z          @��
@�  ?�R�����ff@�{@�  ?����ff���H@��                                    Bx��.�  
Z          @���@��
?��ü��
�B�\A�ff@��
?��ý��Ϳ�Q�A�(�                                    Bx��=P  T          @�G�@���?n{��\)�Z�HA;
=@���?^�R���d��A.�R                                    Bx��K�  h          @��H@�
=?��׿
=��A�\)@�
=?��Ϳ(�����A�(�                                    Bx��Z�  
x          @��@�G�=��Ϳ�p���(�?��\@�G�<��
��p���z�>u                                    Bx��iB  
�          @���@�
==�G���z����H?��@�
=<��
��z���G�>aG�                                    Bx��w�  	�          @���@�z�>.{�����Z{@@�z�=�G������[�?���                                    Bx����  
�          @�G�@���?}p�����^{AE�@���?k������hz�A8��                                    Bx���4  �          @�(�@��>�G��aG��7�
@��H@��>Ǯ�fff�<��@�{                                    Bx����  "          @��H@6ff�����:=q� \)C��@6ff�
=�333�  C��                                     Bx����  T          @���@e��p��
=q��\)C���@e�˅����\)C�                                    Bx���&  
Z          @��@��Ϳ+���ff�T(�C���@��Ϳ:�H��G��L(�C�K�                                    Bx����  
�          @��@�
=��p��(���{C�� @�
=���Ϳz����C�j=                                    Bx���r  
�          @��@�Q쾽p����
���C��\@�Q�\�����uC��3                                    Bx���  �          @��@�33��{���R�vffC��\@�33��
=���i�C�e                                    Bx����  
�          @�p�@�p���p���33��ffC���@�p����ÿ�=q���HC�:�                                    Bx��
d  �          @�\)@��R��{��33��C�
=@��R������������C�z�                                    Bx��
  "          @��H@�33�n{��ff��\)C���@�33���
��  ��\)C���                                    Bx��'�  6          @���@~�R������H��\)C�޸@~�R��z�У�����C�B�                                    Bx��6V  
�          @��@�33��Q��  ��=qC�H@�33�G������p�C��                                    Bx��D�  
�          @���@�{�.�R�W
=��RC�33@�{�1G��.{��ffC�                                      Bx��S�  @          @��H@�G��1G���(����
C�J=@�G��2�\��=q�9��C�33                                    Bx��bH  
�          @�  @��R�.�R=��
?Y��C�B�@��R�.{>k�@$z�C�N                                    Bx��p�  
�          @�Q�@���'�=#�
>�
=C��@���'
=>B�\@�
C�q                                    Bx���  �          @��
@���*�H>8Q�?�
=C���@���*=q>���@e�C�f                                    Bx���:  �          @�=q@���*=q��G���  C�.@���*=q=#�
?   C�,�                                    Bx����  
<          @��H@��H�)��<��
>B�\C�N@��H�(��>.{?�
=C�U�                                    Bx����  
�          @��
@��R�1녾�p�����C�G�@��R�333�L���33C�33                                    Bx���,  @          @��@���
=q�:�H�=qC�o\@����Ϳ���ָRC�=q                                    Bx����  �          @��@�ff�E����H��33C��q@�ff�Fff���R�VffC�                                    Bx���x  �          @�Q�@�(��O\)��p���=qC��3@�(��P  �8Q���RC��H                                    Bx���  
�          @���@�G��W
=��{�n�RC��@�G��XQ�\)����C�\                                    Bx����  T          @��R@�=q�L�;����ffC��f@�=q�N�R��\)�EC���                                    Bx��j  
(          @���@�G��&ff�\)��(�C��@�G��(Q�����p�C��\                                    Bx��  
�          @��@�zΌ����z��MG�C���@�z��G������=p�C�g�                                    Bx�� �  
�          @�\)@��Ϳ˅�aG���\C���@��Ϳ�녿G��	G�C���                                    Bx��/\  
�          @�\)@�녿�����G�C�q@�녿�����ff��Q�C��{                                    Bx��>  T          @�\)@�p��z�L���(�C���@�p��z�u�&ffC�Ǯ                                    Bx��L�  �          @�\)@�Q���þ����
=C��@�Q��	�����
���C�޸                                    Bx��[N  �          @��@�{�*�H?G�A	��C�� @�{�'�?p��A&{C��                                     Bx��i�  �          @��\@�z��5�?��A7�
C���@�z��0��?�p�AUC��                                    Bx��x�  �          @�@�Q��7�?k�A�C��=@�Q��3�
?��A9p�C��                                    