CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230120000000_e20230120235959_p20230121021004_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-21T02:10:04.505Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-20T00:00:00.000Z   time_coverage_end         2023-01-20T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data           records_fill         �   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx\Ǚ�  �          A"�\�8��A��z���B�.�8��AG���
=�ffBՙ�                                    Bx\Ǩf  T          A!����A	�^�R���B�\����Az���8Q�B���                                    Bx\Ƿ  T          A"=q����A
�\�8Q���=qB��)����A?��@E�B�=                                    Bx\�Ų  "          A"�H��
=A��'
=�mB��H��
=A\)?aG�@��RB�B�                                    Bx\��X  
�          A#
=�z=qA���N�R��(�B��)�z=qA�>�\)?�ffB�B�                                    Bx\���  �          A$  �l��A
ff������=qB�Q��l��A�׿���C33B�k�                                    Bx\��  �          A$���j�HA�������\)B���j�HA�����:=qB���                                    Bx\� J  
�          A$z��j�HA�\�c�
��  B�3�j�HA��<#�
=#�
B���                                    Bx\��  �          A$���qG�Ap��mp����HB�8R�qG�AG��.{�n{B���                                    Bx\��  
Z          A$z�����Az��`�����
B�.����A\)    ���
B�
=                                    Bx\�,<  �          A$Q����A
�H�W�����B������A��=��
>��B�                                    Bx\�:�  �          A#�
��A
�\�Mp���{B�.��A�>k�?��\B�B�                                    Bx\�I�  "          A#�
��
=A���Fff���HB�L���
=A��>\@Q�B�                                    Bx\�X.  T          A#���
=Ap��8����\)B���
=Az�?
=@U�B��H                                    Bx\�f�  �          A#���Q�Az��A���=qB�����Q�Az�>�(�@=qB�\)                                    Bx\�uz  "          A#���p�A\)�)���p(�B�aH��p�A�?B�\@�Q�B�u�                                    Bx\Ȅ   �          A!����A	�����[�B����Aff?u@�\)B�p�                                    Bx\Ȓ�  "          A ����z�A�Ϳ�(��5G�B���z�A
�H?�=q@��HB���                                    Bx\ȡl  
�          A (����HA��Ǯ�33B�{���HA
=?�
=A�RB�8R                                    Bx\Ȱ  "          A\)����A녿��H��  B�W
����A�?��HA5B�=q                                    Bx\Ⱦ�  T          A�H��33Aff�������B���33A��?�\)A.{B�(�                                    Bx\��^  �          A=q���A����  ��  B�ff���A�?�A3
=B�33                                    Bx\��  �          A=q��  A�H���R��{B�W
��  A��?��HA6�HB�.                                    Bx\��  �          A����Az��
=���B�33����A��?�=qA�
B�{                                    Bx\��P  T          Aff����A	G��\)�Q��B�L�����A��?���@�\)B��                                    Bx\��  �          AG����RA�ÿ����*�RB�{���RA
{?��HA��B�3                                    Bx\��  T          A����\)A	녿�p���RB��f��\)A\)@�A?33B�q                                    Bx\�%B  "          A  ��\)A�
���W�B�8R��\)A@%Av�RB��                                     Bx\�3�  	�          A�
��33A녾�{�G�B��q��33@�@*�HA~=qB��R                                    Bx\�B�  "          A����H@�{��=q��\)B�B����H@�@*=qA~ffB�u�                                    Bx\�Q4  
�          A  ��  A z὏\)�ǮB�����  @�@8��A�  B��=                                    Bx\�_�  "          AQ���  @�p��޸R�$��B��3��  @���?�Q�@���B���                                    Bx\�n�  �          Az���(�@�G���p��#�B��q��(�@�z�?��R@��B��                                    Bx\�}&  T          A{��z�@��������33B�ff��z�@�?�A(�B���                                    Bx\ɋ�  	�          Aff����@��Ϳ�{��{B�W
����@�33?�\)A\)B��                                    Bx\ɚr  �          A (���p�@�=q@{AL��C33��p�@�\)@��HA���C��                                    Bx\ɩ  
�          A!���33@��H?��A,��C����33@��
@���A��C�H                                    Bx\ɷ�  "          A (���z�@�  ?�\A#33Ck���z�@ʏ\@�z�A�(�C=q                                    Bx\��d  �          A�H��
=@陚?��A.ffC����
=@Å@��A�
=C��                                    Bx\��
  T          A�H��(�@��?��AQ�C����(�@ə�@�\)A�{C\)                                    Bx\��  T          A�\��@��
?ǮAG�C���@�G�@�z�A��
C��                                    Bx\��V  
Z          Aff��\)@�?�\)@��RC^���\)@�33@}p�A��\C��                                    Bx\� �  �          A{�θR@�?���@��CB��θR@�(�@z=qA�Q�C\)                                    Bx\��  �          A=q���@�p�?��@�C�
���@�@z�HA���C��                                    Bx\�H  T          A�R��Q�@��
?��\@�(�Cp���Q�@��@w�A��Cu�                                    Bx\�,�  "          A�R��  @�z�?��R@�CW
��  @�{@uA�=qCL�                                    Bx\�;�  �          A{��ff@�(�?�p�@�(�C!H��ff@�@tz�A�(�C\                                    Bx\�J:  T          A{�ʏ\@��?}p�@�\)C��ʏ\@���@i��A�p�C��                                    Bx\�X�  �          A����ff@�33?aG�@��C5���ff@�Q�@dz�A�=qC��                                    Bx\�g�  T          Ap��ȣ�@�Q�?��@�ffCٚ�ȣ�@Ӆ@mp�A�\)Cu�                                    Bx\�v,  
�          AG���33@�?�=q@��C����33@���@l��A��RC+�                                    Bx\ʄ�  �          AG����@�?�@ڏ\C����@�{@p  A��C��                                    Bx\ʓx  �          AG���Q�@��?W
=@�z�C���Q�@ָR@`  A�
=C�                                    Bx\ʢ  
�          Ap�����@��?��@^{C������@ڏ\@R�\A�  C�
                                    Bx\ʰ�  "          AG���G�@�G�>�
=@(�C�)��G�@�(�@G
=A���Cu�                                    Bx\ʿj  �          A{����@��
>�Q�@ffCu�����@�\)@EA��C��                                    Bx\��  �          A�Ǯ@��
>�\)?�33CY��Ǯ@�  @@��A�=qC�R                                    Bx\�ܶ  �          A��\)@�z��G��#�
C.��\)@�p�@)��Ayp�C�q                                    Bx\��\  �          A{�ƸR@�{���+�C ���ƸR@�
=@*=qAyC�3                                    Bx\��  �          Ap���  @������
���B�.��  @陚@.�RA���Ch�                                    Bx\��  �          A���(�@�����Ϳ�C ����(�@�{@*=qA{�
Cs3                                    Bx\�N  �          AG���z�@�p����B�\C �f��z�@�R@(Q�Ax��Ch�                                    Bx\�%�  
�          A���ҏ\@�33?�{A   C���ҏ\@�z�@tz�A��
C�R                                    Bx\�4�  T          A(��ҏ\@�33?�\)@�(�C�
�ҏ\@�
=@eA��C��                                    Bx\�C@  �          A  ��33@�33?Y��@�  C����33@��@Z�HA�z�C�                                    Bx\�Q�  �          A����H@��H?��@b�\C�=���H@�z�@L(�A�33C�f                                    Bx\�`�  �          A
=��Q�@��
?�R@j=qC\)��Q�@��@Mp�A���C=q                                    Bx\�o2  
�          A�
���@��H?��@P  C �)���@�z�@N{A���CT{                                    Bx\�}�  �          A������@����
=q�K�B�=q����@�\)@�AV�HC G�                                    Bx\ˌ~  �          AG���\)@��þu���B����\)@�33@#33Ap��C�                                    Bx\˛$  �          Ap���p�@��H>#�
?k�B����p�@��@<(�A�33C(�                                    Bx\˩�  �          A\)��p�@����ff�&ffB�\��p�@�(�@�HAa�B���                                    Bx\˸p  �          A (�����A33���0��B�����Ap�?�\)@�B���                                    Bx\��  �          A ����ffA�����Q��B�8R��ffA��?O\)@�z�B�{                                    Bx\�ռ  T          A z����A�\���=qB�z����A{?���A�B���                                    Bx\��b  T          A���(�A��z��W�B����(�A
ff?L��@��\B��f                                    Bx\��  "          Aff��Q�A
=�L������B�����Q�A�ͽ�Q��B�L�                                    Bx\��  T          A�H�|(�A��l����33B�q�|(�Ap����@��B��f                                    Bx\�T  �          A (��^�RA=q��(��Ù�B�
=�^�RA�ÿfff��B�                                      Bx\��  T          A ���QG�A33��Q����B�z��QG�A�Ǯ�33Bڊ=                                    Bx\�-�  �          A   �Tz�A����\��=qB��f�Tz�A�������ffB�33                                    Bx\�<F  �          A z��FffA33��33��B�aH�FffA=q��z��  B؅                                    Bx\�J�  �          A (��+�@�z���G��ffB�p��+�Ap��=q�^�RB���                                    Bx\�Y�  �          A��@�(�����	Q�B�\�A���\)�g�B�{                                    Bx\�h8  �          A\)�#�
@�p���G��ffB���#�
A��.�R�~�RB�\                                    Bx\�v�  �          A�\�)��@�33��G���B�ff�)��A=q�0����33B�B�                                    Bx\̅�  
�          A��p��@�(���G���  B�
=�p��A�ÿ�p���
B��                                    Bx\̔*  �          A!G���ffA (��[����
B�k���ffA\)����z�B�B�                                    Bx\̢�  �          A"=q���R@���@  ���B��3���RA�׼#�
���
B�=q                                    Bx\̱v  T          A"{��33@����=p���{B����33A
=���
��G�B�.                                    Bx\��  "          A"{��Q�@�33�4z���
=B����Q�Ap�=���?\)B�=q                                    Bx\���  �          A"�R���@�33���]�B�����A33>�@%�B�B�                                    Bx\��h  T          A"�\��\)@�p���� ��C33��\)@�ff?�  @�C
                                    Bx\��  
�          A"=q��G�@�33�u���C����G�@�  ?���A�C+�                                    Bx\���  T          A!���@�{>���?�{C���@�{@p�Ab{CQ�                                    Bx\�	Z  T          A!G���ff@Ϯ>�=q?��
C�)��ff@�  @   Af=qC{                                    Bx\�   
�          A!���
@ʏ\?��@HQ�C8R���
@��@-p�Axz�C��                                    Bx\�&�  �          A ����@�z�?O\)@��
CE��@�\)@8Q�A�33C^�                                    Bx\�5L  
�          A"{���R@�
=?E�@��\C����R@�=q@7�A�p�C                                    Bx\�C�  T          A!����@˅>��?�p�C
=���@�(�@(�A_33C:�                                    Bx\�R�  T          A"=q���@�=q?�@:�HCY����@�Q�@*=qAs�C�                                    Bx\�a>  T          A"=q���R@�  >��@(��C�
���R@��R@%Al��CW
                                    Bx\�o�  T          A"ff��33@�>��R?�p�C�f��33@�{@   Ad  C��                                    Bx\�~�  �          A"{���R@��?
=q@C�
C�)���R@�
=@0��A|��C:�                                    Bx\͍0  
�          A!����@�33>��H@0��C����@���@(Q�Aqp�C�
                                    Bx\͛�  b          A!�����@�=q>�Q�@�
CW
����@��@ ��AeG�C��                                    Bx\ͪ|  �          A�\��@�Q�>Ǯ@  C�\��@�  @ ��Aj=qC+�                                    Bx\͹"  "          A�
���@�
=?�@=p�C}q���@�p�@'
=Ar{C
=                                    Bx\���  "          A ������@�{?�\@;�C�3����@�z�@%Ao33Cz�                                    Bx\��n  
�          A����\@���>��@+�Cٚ���\@��
@!�Ak�CO\                                    Bx\��  "          A"ff����@�\)>��@33CG�����@�ff@%AlQ�C�H                                    Bx\��  �          A"{����@�{>�Q�@ ��Cs3����@�{@!G�Af�HC�R                                    Bx\�`  "          A"{��ff@��>���@  C����ff@�G�@&ffAn{C��                                    Bx\�  
�          A"�\��=q@��?\(�@�z�C�f��=q@��@?\)A��HC�R                                    Bx\��  
�          A"�H��33@���?�ff@���C����33@�p�@I��A�(�C(�                                    Bx\�.R  �          A"�R���\@���?��@�C�����\@��@L(�A�ffC#�                                    Bx\�<�  T          A"�R����@��?���@���Cc�����@��\@H��A���C�                                    Bx\�K�  �          A"�R���@ə�?�=q@\Cz����@�=q@H��A��C޸                                    Bx\�ZD  
�          A"�R����@�G�?�{@�G�Cp�����@���@J�HA�G�C��                                    Bx\�h�  �          A"�R����@ȣ�?��\@��C�\����@�\)@S33A��C@                                     Bx\�w�  �          A"�R��p�@�  ?�G�@�33C����p�@�
=@Q�A���CaH                                    Bx\Ά6  "          A!���(�@�\)?�Q�@���C����(�@�
=@Mp�A�{C:�                                    Bx\Δ�  
�          A"�H��  @љ�?n{@�
=C�\��  @��@C�
A�  C�f                                    Bx\Σ�  �          A"�R��{@�33?W
=@��CY���{@�{@@  A��CL�                                    Bx\β(  
�          A"=q���R@��
?��\@�\)Cz����R@��H@O\)A�p�C(�                                    Bx\���  �          A (�����@�p�?�=qA�C.����@���@]p�A��
Cff                                    Bx\��t  �          A�\��z�@��?�p�A	p�C�f��z�@�\)@U�A���C�R                                    Bx\��  �          A���33@�\)?�\)@�C����33@�{@Q�A��
C��                                    Bx\���  �          A Q����H@��
?�Q�@ڏ\C
=���H@�(�@I��A��RC�\                                    Bx\��f  �          Aff��
=@Å?���@���C����
=@�z�@EA��C
                                    Bx\�
  "          A���Q�@��?��\@���C���Q�@�(�@>{A��\CaH                                    Bx\��  �          A�����
@��H?�  @��
Cc����
@��@=p�A�ffC�)                                    Bx\�'X  �          A�\���@�=q?��
@�C�����@�=q@HQ�A�33CW
                                    Bx\�5�  �          A   �  @�
=@ffAB{C���  @��@l��A��C��                                    Bx\�D�  
�          A����@�{@�RAh��C�{���@g�@xQ�A�\)C�                                     Bx\�SJ  "          A(���@��@ffAG
=C�3��@vff@c�
A�G�C��                                    Bx\�a�  
�          A�R���@�=q?�p�A<z�C����@j�H@W
=A���C�                                    Bx\�p�  
�          Az��\)@��@AJffC��\)@[�@Y��A��CL�                                    Bx\�<  �          A���p�@u�@$z�Az�\CO\�p�@1�@k�A��C!�=                                    Bx\ύ�  
�          A(��\)@QG�@/\)A�  CG��\)@��@j=qA�ffC%                                    Bx\Ϝ�  �          A���ff@@��@P��A�ffC�R�ff?��@��AυC'�H                                    Bx\ϫ.  "          Az��Q�?�@j=qA�\)C'�H�Q�?�@���AиRC0^�                                    Bx\Ϲ�  "          A=q��\)?\@�z�A�Q�C)=q��\)=�Q�@��A�C3\)                                    Bx\��z  T          A����(�@Q�@�(�A�p�C$ff��(�?��@��\B{C/��                                    Bx\��   
�          A  ���\?��R@�A�\)C)33���\=u@�{A�Q�C3�=                                    Bx\���  �          A������?�@��A�33C&#�����>�ff@�(�A��C0�3                                    Bx\��l  �          A����(�@\)@��A�33C#���(�?�R@�(�B
�C/5�                                    Bx\�  �          AG���\@�@���A��
C"�H��\?Tz�@�p�Bz�C-�q                                    Bx\��  "          A(����>�@�  B{C0:���׿�z�@�33B
=C?W
                                    Bx\� ^  �          A�����>u@��\B"(�C1����녿��@�33B�\CA�                                    Bx\�/  T          Aff���H@$z�@@  A��C!�
���H?�p�@j=qA�G�C)Q�                                    Bx\�=�  "          AG���p�@0��@.{A��C �\��p�?޸R@]p�A��HC'�
                                    Bx\�LP  T          A��ff@33@J�HA�C#����ff?�@o\)AǙ�C+��                                    Bx\�Z�  �          A����33@
=q@W
=A���C$����33?z�H@xQ�A��HC,��                                    Bx\�i�  T          A������@@��@vffA�z�C������?У�@��A��C'\)                                    Bx\�xB  �          A���(�?�=q@�B�\C%�)��(�=�Q�@��Bp�C3B�                                    Bx\І�  �          A���׿�ff@�  B${C@}q����\��@�{B=qCN33                                    Bx\Е�  
�          A���  ���R@��
B${CDu���  �u�@�B�HCQ�=                                    Bx\Ф4  �          A
=���Ϳ�
=@�
=B@ffC>�{�����S�
@�  B'G�CPW
                                    Bx\в�  
(          A�
�ə���
=@�{B<33C@���ə��c33@�z�B!Q�CQ^�                                    Bx\���  �          A����G��   @���B8��CE����G����@�p�B=qCTٚ                                    Bx\��&  �          A�����@޸RB:G�CH������z�@�(�B\)CWxR                                    Bx\���  "          A��ʏ\�p�@�ffB3��CG8R�ʏ\��{@�p�B{CU�                                    Bx\��r  
�          A���ff�2�\@�p�B5��CM&f��ff���@��RB�RCZ�                                    Bx\��  T          A�R���\�L(�@��B4Q�CP����\���@��\B
33C]8R                                    Bx\�
�  "          A\)��
=�z=q@�p�B3{CW�{��
=����@��
B�Cb��                                    Bx\�d  "          A�����h��@���B0z�CT)�������@�{B��C_�
                                    Bx\�(
  �          A����Q��h��@�33B3�HCUc���Q���  @���B��C`�                                    Bx\�6�  
�          A���=q��{@�p�B5��C[�
��=q����@�G�BffCf
=                                    Bx\�EV  
Z          A����\�l(�@�p�B;ffCV�����\��z�@�{B{Cb��                                    Bx\�S�  �          A�R��{�0  @��BD��CM����{���
@ÅBC\��                                    Bx\�b�  "          A�R��33�!�@�
=BJQ�CLO\��33��ff@ʏ\B#��C\
=                                    Bx\�qH  T          A�H��ff�%@��
BF�RCLp���ff��\)@�
=B 33C[��                                    Bx\��  
�          A�H��G���H@��
BFG�CJ����G���=q@ȣ�B!�CZB�                                    Bx\ю�  �          A\)��(��,��@�BG�HCM�)��(����H@�  B ffC\�3                                    Bx\ѝ:  
�          A�
����\)@�RB>��CH�������H@�BffCW!H                                    Bx\ѫ�  �          A�
��=q��
=@�ffB>��CBٚ��=q�r�\@��HB"�CR�                                    Bx\Ѻ�  
�          A�
��(�����@��BF�CBaH��(��p  @ҏ\B*{CSu�                                    Bx\��,  T          A�H��녿�\)@��BD(�CE#������  @���B%�CUh�                                    Bx\���  "          A���33��\@�BD�CD33��33�z�H@�\)B'=qCT�R                                    Bx\��x  �          A  �����˅@�\)BHffCB� �����q�@���B,  CS�q                                    Bx\��  �          A�
��zῦff@�z�BN�C@s3��z��b�\@���B4ffCS
=                                    Bx\��  �          A���=q����@�
=BR�C>n��=q�U@�G�B9z�CQ޸                                    Bx\�j  �          A  ��{�.{@�z�BX{C:�
��{�AG�@�=qBB�CO�3                                    Bx\�!  �          A���\)��Q�A z�B^p�C7�=��\)�0  @�BK��CN��                                    Bx\�/�  �          A\)��=q�.{ABb�
C5�)��=q�%@�BQ��CM�                                    Bx\�>\  �          A���p��Q�@�BM��C;޸��p��@��@��B8�\CO                                      Bx\�M  �          A{�˅��Q�@�\B=z�C@�q�˅�^�R@ʏ\B$�\CP�                                    Bx\�[�  �          A�\��(�����@�BBG�CD�\��(��x��@��
B%33CTff                                    Bx\�jN  �          A�R��ff��Q�@�BH��CCٚ��ff�tz�@�33B,(�CT��                                    Bx\�x�  �          A\)�\��=q@���BF�CB�=�\�l(�@�33B+z�CSG�                                    Bx\҇�  �          A���{���H@�BDCAE��{�c�
@�33B+
=CQ�3                                    Bx\Җ@  �          A=q���H��ff@�33BF�C@
���H�Z=q@�z�B.\)CQ@                                     Bx\Ҥ�  �          A
=�������@��HBEp�C@�����Z�H@�(�B-
=CQ�                                    Bx\ҳ�  �          A��������@�(�BI33C@�����X��@�B0�\CQ�                                    Bx\��2  �          A=q�\��=q@�33BF��C@\)�\�Z�H@�z�B.Q�CQ^�                                    Bx\���  �          A(������ff@�RBE�CB}q����e�@�ffB*�CR��                                    Bx\��~  �          A�\��G����@�BH��CC����G��j�H@�ffB,�HCTaH                                    Bx\��$  �          A{��녿�ff@��
BG��C@)����XQ�@�B/�\CQ.                                    Bx\���  �          A����  ����@�=qBF�RCB�R��  �i��@�G�B+��CSY�                                    Bx\�p  �          Az�������z�@�
=BE33CA
�����[�@�Q�B,\)CQ�{                                    Bx\�  �          A�����H��
=@�  BE�
C>�R���H�N{@�33B/33CO�H                                    Bx\�(�  T          A���\)���
@�33BH�C@
��\)�U@�p�B0�CQ.                                    Bx\�7b  �          A��G���  @�p�BA�C=���G��AG�@ҏ\B-Q�CM��                                    Bx\�F  �          A=q��=q�s33@�ffBA��C<����=q�>{@�(�B-��CM0�                                    Bx\�T�  �          A{��33��G�@�
=BB��C4�q��33�
�H@�z�B7  CF�
                                    Bx\�cT  �          A������@�=qBF�HC9@ ���(Q�@ۅB6ffCK\                                    Bx\�q�  "          A����  ����@�BM�C7!H��  �(�@��HB>\)CJ+�                                    Bx\Ӏ�  �          Az���  ��@�p�BC\)C8ff��  �\)@׮B433CI                                    Bx\ӏF  �          A����G���@��BB��C8\)��G��\)@׮B3��CI��                                    Bx\ӝ�  �          A���p��333@ٙ�B5��C:���p��%@ʏ\B%��CI@                                     Bx\Ӭ�  �          A(���녽�G�@ڏ\B8��C4�R�����@���B.{CE.                                    Bx\ӻ8  �          A�
��>�z�@�\)B4�RC1������{@�G�B.Q�CA��                                    Bx\���  �          A���ff�W
=@θRB/�HC5�\��ff� ��@���B%=qCD�q                                    Bx\�؄  �          A����녽u@ƸRB-�RC4� ��녿��@�{B$�CCB�                                    Bx\��*  �          A�H��p����
@�ffB/�C6�
��p���@��B$G�CE�{                                    Bx\���  �          A�H��=q��Q�@��B3��C78R��=q�@��RB'ffCFW
                                    Bx\�v  T          A�\��\)�u@ÅB-G�C6
��\)��@��B"�RCD�                                    Bx\�  �          A{���;k�@��B/�C6\���Ϳ�@�33B$�HCD��                                    Bx\�!�  �          Aff��(��aG�@�ffB0�\C6���(���
=@���B%��CD�=                                    Bx\�0h  �          A{���þW
=@ȣ�B3��C5�����ÿ�
=@�
=B(�CE{                                    Bx\�?  �          AG���z��@�33B.�C50���z��ff@��\B%  CC��                                    Bx\�M�  �          Ap���\)��Q�@�Q�B4��C4�
��\)��ff@�  B+{CD!H                                    Bx\�\Z  �          A���(�=���@��HB7��C3
=��(��У�@�(�B0  CB�f                                    Bx\�k   �          A����\)>���@���B+�C1Q���\)����@�(�B&p�C?�                                     Bx\�y�  �          A���G�>�ff@��
B=qC033��G����@���BG�C=#�                                    Bx\ԈL  �          A���أ�>\@��\BC0��أ׿��@�
=B=qC=�
                                    Bx\Ԗ�  �          AG�����>�  @�=qB-��C1�q���Ϳ�33@�p�B(ffC@T{                                    Bx\ԥ�  �          AG����þL��@�B1��C5�����ÿ�{@�z�B'�HCDs3                                    Bx\Դ>  �          A33��ff���@��
B7  C6c���ff���H@�=qB,
=CE��                                    Bx\���  �          A(����>�  @��B,��C1Ǯ��녿�z�@�Q�B'\)C@�                                    Bx\�ъ  �          A��ָR>���@�{B%�HC1&f�ָR��G�@��B!��C>�                                    Bx\��0  T          A
=��  ?�
=@���BQ�C(u���  �#�
@��Bz�C4J=                                    Bx\���  �          A�H��?�z�@�33B�C&�{��>�  @��B�C1�q                                    Bx\��|  �          A33��?��@�33Bp�C/���fff@��B�C;n                                    Bx\�"  �          A�R��
=�L��@�(�B-��C4u���
=��@�z�B%�RCBxR                                    Bx\��  �          A\)��=q�J=q@���B�RC:\)��=q��@�  B�CE&f                                    Bx\�)n  �          A�R����?
=q@�G�B*��C/:����ÿ��@�
=B(33C=:�                                    Bx\�8  �          A��θR��=q@ƸRB/\)C6ff�θR���@�p�B%\)CDL�                                    Bx\�F�  �          A
=����=�Q�@˅B5z�C333���ÿ�=q@��B.\)CB)                                    Bx\�U`  �          A33����?˅@�33BG�C#�3���þ�  @���BO�C6�
                                    Bx\�d  T          A�
��?�ff@�(�B>
=C'�������@׮BB(�C8��                                    Bx\�r�  �          A�
��?�@�p�B6p�C)J=���
=q@�Q�B9z�C8��                                    Bx\ՁR  �          Az����?B�\@ə�B1G�C-5�����c�
@���B0�C;�                                    Bx\Տ�  �          A������?B�\@�
=B?�\C,ٚ���׿�  @�B>Q�C=n                                    Bx\՞�  �          A����{?#�
@�G�BB��C-޸��{����@�
=B@(�C>�                                     Bx\խD  �          Az��ƸR?!G�@�Q�B8�HC.:��ƸR���@�ffB6��C=��                                    Bx\ջ�  �          A����
=?
=@��B0��C.����
=���@�  B.�RC=+�                                    Bx\�ʐ  �          A����33>�@�B>(�C/u���33��p�@��HB:�\C?s3                                    Bx\��6  �          A(���{>�=q@�G�B:(�C1���{��33@�z�B4�
C@                                    Bx\���  �          A\)��{>k�@�
=B9
=C1޸��{��@�=qB3p�C@��                                    Bx\���  �          A�
��z�>B�\@�=qB;��C2:���zῼ(�@���B5�\CA�                                     Bx\�(  �          Ap�����?(�@�B3�HC.�����Ϳ��
@��
B2  C=�                                    Bx\��  T          A{���>���@ϮB5�C0c�������R@�(�B1ffC>�                                    Bx\�"t  �          A�R��ff>�ff@�  B4��C0���ff��Q�@���B1G�C>u�                                    Bx\�1  �          A�\��=q?c�
@��B.Q�C,G���=q�5@ʏ\B/{C:.                                    Bx\�?�  �          A����
=?�G�@�=qB0  C+���
=�
=@��
B1�HC9=q                                    Bx\�Nf  �          A����
=?5@�G�B9  C-����
=�p��@�Q�B7��C<�\                                    Bx\�]  T          A����ff?��R@��B�RC'����ff=�Q�@�{B(�C3E                                    Bx\�k�  T          A���=q>��@�
=B-ffC/���=q��=q@�z�B*�RC=W
                                    Bx\�zX  �          A�У׽#�
@�33B0��C4O\�У׿˅@���B)�CA�q                                    Bx\ֈ�  
�          A�љ�?E�@���B.G�C-J=�љ��J=q@���B.33C:�H                                    Bx\֗�  �          AG��љ�?���@�p�B+=qC)���љ���33@���B.�HC7�                                    Bx\֦J  "          AG��Ϯ?�
=@�p�B+p�C'���Ϯ��G�@ʏ\B1�C5                                      Bx\ִ�  �          A�����?��H@��B(�C'}q��녽u@�\)B.
=C4�                                     Bx\�Ö  �          A�����?���@��B(p�C()��녾�@ƸRB-��C5
                                    Bx\��<  "          A����?��@�
=B$�HC(�����.{@ÅB)��C5n                                    Bx\���  �          A�����
?���@�\)B%Q�C&�H���
=�Q�@�B,=qC3:�                                    Bx\��  T          A����\)?��H@��
B!�HC'����\)<#�
@���B'�C3�H                                    Bx\��.  �          A������?���@��\B \)C'�f����<��
@�Q�B&G�C3ٚ                                    Bx\��  �          A�����?��R@�ffBC'�\����=�Q�@�z�B"  C3@                                     Bx\�z  �          A{��  ?��@��RB�HC)h���  ���
@��B�C4�                                    Bx\�*   �          A�R�߮?��@���B  C(�q�߮�#�
@�{B!\)C4�                                    Bx\�8�  �          A�R��ff?��H@�=qBQ�C(+���ff=#�
@��B#33C3��                                    Bx\�Gl  T          A���  ?��@�{B"p�C'!H��  =���@�(�B)  C3)                                    Bx\�V  T          A��{?��
@�\)B$\)C'(���{=��
@�p�B*��C3L�                                    Bx\�d�  
�          A��
=?���@�ffB#  C&޸��
=>\)@���B)��C2�{                                    Bx\�s^  �          A�أ�?�
=@�33B��C&�أ�>�\)@\B'�C1�f                                    Bx\ׂ  "          A����{?��H@���B��C$5���{?!G�@��\B��C.��                                    Bx\א�  
�          A����33?��R@���B��C#�=��33?&ff@��B!C.��                                    Bx\ןP  
�          A�����@ ��@�G�B�C#33���?�R@��HB)p�C.�f                                    Bx\׭�  �          A���أ�@*=q@�33BQ�C���أ�?�{@�=qB p�C(�H                                    Bx\׼�  �          AQ���  @&ff@�(�B�\C���  ?�ff@��\B!=qC))                                    Bx\��B  G          AQ���@�@�G�B�RC!����?�G�@�p�B�C+�                                    Bx\���  �          AQ���p�@�@�B��C"���p�?�33@��\B33C*��                                    Bx\��  "          A����=q?�ff@��
BffC%�3��=q?
=q@�(�B��C/�H                                    Bx\��4  �          AG���=q?�33@�{B�C$s3��=q?�@�
=B$\)C/B�                                    Bx\��  T          A����
@G�@�{B��C!�R���
?}p�@��B��C+��                                    Bx\��  �          A���(�?�
=@�=qBC$Y���(�?!G�@��B!33C.�                                    Bx\�#&  
�          A��{?��@�G�B��C%�q��{?�@���B\)C/xR                                    Bx\�1�  �          A����p�?�@�Q�B\)C%���p�?333@��B��C.p�                                    Bx\�@r  �          Ap���G�@
=q@�=qBz�C"���G�?n{@�p�Bp�C,��                                    Bx\�O  T          A����(�@Q�@�ffB(�C ���(�?�\)@��HBC*�                                    Bx\�]�  
�          Ap��ۅ@3�
@�  BQ�C�R�ۅ?�=q@��B�
C&�q                                    Bx\�ld  
�          A���{@;�@�(�B	  C���{?޸R@���B(�C%�                                    Bx\�{
  "          A�H��33@P  @�p�B	
=C�f��33@�\@�  B33C#n                                    Bx\؉�  "          A�\��\)@o\)@���B33C����\)@   @�\)B#
=C�f                                    Bx\ؘV  "          Aff���@g�@�  B(�C)���@��@��B!
=C��                                    Bx\ئ�  "          A�R��@qG�@���B
=C�=��@%@�\)B�\C�                                    Bx\ص�  �          A�H��{@^{@�
=B
C����{@G�@�33BffC!J=                                    Bx\��H  "          A�H��\)@aG�@���B�Cff��\)@�@���BG�C ��                                    Bx\���  �          A�R�أ�@h��@�  B(�C�أ�@�R@��B�Cٚ                                    Bx\��  �          A=q��{@N{@�  BC!H��{@�@�=qBQ�C#W
                                    Bx\��:  
�          A{�ۅ@R�\@��B�Cc��ۅ@��@�z�B�\C"�                                    Bx\���  �          A��ҏ\@Tz�@�  B�HC5��ҏ\@��@��HB ��C"�                                    Bx\��  �          AG���33@W�@�
=BC���33@��@�=qB�C!�
                                    Bx\�,  T          A����  @5@�G�B33C(���  ?�@�Q�B�\C&�                                    Bx\�*�  �          AG����@0  @��HB(�C�����?�=q@���B��C&��                                    Bx\�9x  �          Ap���@I��@�ffB(�C����@33@�  B{C#�=                                    Bx\�H  
�          Ap����@@  @��HBG�C����?��@�33B=qC$�=                                    Bx\�V�  "          A�\��Q�@��\@�  A�33C�=��Q�@G
=@�  Bp�C
=                                    Bx\�ej  T          A�\���H@z�H@���A�{C{���H@=p�@�\)B�
CaH                                    Bx\�t  "          A=q���
@�\)@��A�
=Ch����
@P  @��HB(�C�                                    Bx\ق�  �          A�\�ָR@�(�@�A�C�)�ָR@X��@��BffC=q                                    Bx\ّ\  �          A��׮@���@�(�A�Cs3�׮@Tz�@�p�B

=CǮ                                    Bx\٠  
�          Ap���\)@�33@�G�A�C�)��\)@Fff@���B�CE                                    Bx\ٮ�  �          A����z�@s33@��A�RC#���z�@333@�\)Bz�C�f                                    Bx\ٽN  T          A���ۅ@qG�@�z�A�G�C0��ۅ@0��@���B��C�                                    Bx\���  !          A��ۅ@qG�@�=qA�=qC5��ۅ@1�@��B{C��                                    Bx\�ښ  
�          AG����@q�@���A��CT{���@333@�{BffC�3                                    Bx\��@  
N          A����z�@r�\@��A��C33��z�@4z�@���B
�C�q                                    Bx\���  
�          A�����@o\)@�G�A�=qC������@1G�@�{B��C#�                                    Bx\��  "          Az���=q@z=q@�{A�  C+���=q@<��@�(�B
z�C��                                    Bx\�2  "          AQ����H@��
@�A�
=C����H@N{@��B��C�\                                    Bx\�#�  T          AQ���z�@�Q�@�\)A��
C���z�@G
=@�{B�C�R                                    Bx\�2~  T          A(���ff@s33@�G�A�\)CT{��ff@8��@�ffB=qCk�                                    Bx\�A$  H          A����(�@�=q@�{A�p�Ch���(�@K�@���B��C:�                                    Bx\�O�  `          AQ���z�@n{@��A���C����z�@1�@�(�B
p�C�                                    Bx\�^p  �          A(���\)@n{@��\A�C�3��\)@3�
@��RB�RC
=                                    Bx\�m  	�          AQ��߮@u�@�
=A�
=CO\�߮@<��@��
B��C&f                                    Bx\�{�  �          A(���@g
=@�ffA�z�C���@/\)@�=qB=qC��                                    Bx\ڊb  
�          Az����
@l(�@��A�p�C�����
@5�@�G�B 
=CQ�                                    Bx\ڙ  �          A�
��33@e�@�A��CB���33@.{@�G�B �C�                                    Bx\ڧ�  
�          A�H����@C�
@�(�A���C������@�@�(�Bz�C#�                                    Bx\ڶT  �          A{���
@1�@�  A�(�C�����
?��@�{B�\C%33                                    Bx\���  �          A{����@9��@��\A��RC�
����?��R@�G�B
�C$33                                    Bx\�Ӡ  T          Ap��ᙚ@7�@��A�  C޸�ᙚ?�p�@�{B  C$O\                                    Bx\��F  �          A����  @9��@��\A�33C}q��  @   @�G�B
�RC$�                                    Bx\���  
�          A����z�@G�@��HA�(�C����z�@{@��HBz�C"(�                                    Bx\���  T          A����@O\)@��
A��C�
���@�@�z�B  C!)                                    Bx\�8  "          A{���@_\)@��A��CW
���@&ff@��
B�C��                                    Bx\��  �          A33��  @�  @/\)A���C����  @���@l��A�p�C�q                                    Bx\�+�  "          A�H�ڏ\@��R@P  A�G�Ck��ڏ\@�G�@���A�z�Cff                                    Bx\�:*  �          A\)��\)@��R@L��A��C�
��\)@���@���A�
=Ck�                                    Bx\�H�  T          A\)�ٙ�@�=q@dz�A�(�C
�ٙ�@w
=@��\A�  Cu�                                    Bx\�Wv  "          A���
=@��@[�A�  C5���
=@s33@�A�Q�Cff                                    Bx\�f  
�          A���  @��@W
=A��
CT{��  @s�
@�33A��Cff                                    Bx\�t�  "          A�H��=q@�=q@U�A�
=C�
��=q@j=q@�G�A�\)C�f                                    Bx\ۃh  �          A�\���@�ff@O\)A��RC�)���@c�
@{�Aљ�C��                                    Bx\ے  �          A�\��@�  @\(�A�p�C����@U�@��HA�z�C�                                    Bx\۠�  �          Aff��33@~�R@g
=A�33C�q��33@Q�@�  A�{C.                                    Bx\ۯZ  �          A���@���@j�HA�p�C���@\��@��HA�(�C��                                    Bx\۾   �          A{��@�z�@l��A���C#���@\(�@��A�G�C�H                                    Bx\�̦  T          A���H@�{@p  A�(�C}q���H@^�R@�p�A�33C�                                    Bx\��L  �          A�H�޸R@��@n�RAŮC!H�޸R@\��@�z�A��
C��                                    Bx\���  �          A�\��p�@�@n{A��
C����p�@^{@�z�A�(�CW
                                    Bx\���  �          A{��33@���@j�HA��
C���33@e�@�33A���Ck�                                    Bx\�>  �          Aff��{@��@vffA�p�C����{@l(�@���A�(�C{                                    Bx\��  T          AQ���(�@g
=@�{A��Cn��(�@333@��B
��C\                                    Bx\�$�  �          A  ��G�@qG�@�p�A��HC���G�@>{@�  BG�C�=                                    Bx\�30  �          Az����H@}p�@��A�33C����H@L(�@�33B
=C+�                                    Bx\�A�  T          Az���\)@�G�@��\A�ffC\��\)@P��@�ffB	{CL�                                    Bx\�P|  �          A���ҏ\@���@�ffA�ffCaH�ҏ\@S33@�=qB�HCc�                                    Bx\�_"  "          Az���
=@\)@��
A�G�C^���
=@Mp�@�
=B
(�C�H                                    Bx\�m�  �          A������@\)@��\A�  C�{����@N{@�BQ�C�                                     Bx\�|n  �          Ap���G�@\)@��A�\)C�
��G�@N{@�Q�B	�C��                                    Bx\܋  
�          A����(�@��H@��A�Q�CW
��(�@W�@�
=B��C�                                    Bx\ܙ�  �          A����G�@|(�@�(�A��HC����G�@K�@�
=B	\)C{                                    Bx\ܨ`  
�          A����  @\)@��A�{Cp���  @N�R@��B
(�C�
                                    Bx\ܷ  
�          Ap����H@���@�Q�A��HCk����H@S�
@��B�CQ�                                    Bx\�Ŭ  �          A����
=@�G�@��A�G�C  ��
=@U�@�ffB ffC��                                    Bx\��R  
�          A���ڏ\@��@uA��
C  �ڏ\@\(�@�ffA�\)CE                                    Bx\���  �          A��ڏ\@�{@r�\Aʏ\C}q�ڏ\@a�@��A�z�C�f                                    Bx\��  "          AG���ff@y��@�\)A�RC����ff@K�@�G�B\)C�)                                    Bx\� D  �          A��\)@��H@c�
A��C����\)@^�R@��A�z�C��                                    Bx\��  T          A{���@�z�@(Q�A��C�H���@z�H@Q�A�G�C��                                    Bx\��  �          A=q�陚@�=q@
�HAb{C��陚@�@6ffA�  C33                                    Bx\�,6  T          A{���
@�ff@��A^�HC�
���
@�=q@333A�33C{                                    Bx\�:�  
�          A����@�p�@33AV=qC0����@���@-p�A�ffCW
                                    Bx\�I�  T          Ap���ff@���@ffA\(�C33��ff@x��@.�RA�  Ch�                                    Bx\�X(  �          Ap���{@���@��A_�
C+���{@x��@0��A�Cff                                    Bx\�f�  �          AG���@�  @��A`z�CE��@w�@0��A�C}q                                    Bx\�ut  �          A����@�=q@(�Ae�C����@|(�@4z�A�
=C�)                                    Bx\݄  
�          A�����@�Q�@(�A���C� ���@u@C33A�z�C@                                     Bx\ݒ�  �          Az���33@�ff@
=AyG�C@ ��33@s33@=p�A���C��                                    Bx\ݡf  �          A�
��@�ff@�AV�RCk���@�33@*�HA�Q�Cz�                                    Bx\ݰ  
�          A��ᙚ@��H@\)An�\C��ᙚ@��R@9��A��C+�                                    Bx\ݾ�  �          A�
��\)@�ff@�
AZ{C\)��\)@�33@,(�A���Cn                                    Bx\��X  
�          A  ��R@�{@��Ai�CaH��R@�=q@5�A���C�\                                    Bx\���  �          A�
��@�
=@=qA�  C�{��@�=q@B�\A��RC33                                    Bx\��  �          A����@��R@   A�\)C�����@���@J=qA��C!H                                    Bx\��J  
�          A\)����@�  @\)A��HCW
����@�33@G�A��C                                    Bx\��  "          A33�޸R@�=q@��A��\C�R�޸R@�@A�A�C�                                    Bx\��  "          A33��{@�G�@@  A�CJ=��{@s�
@eA���C8R                                    Bx\�%<  �          A
=�ٙ�@��@?\)A�
=C�=�ٙ�@���@fffA��
Cc�                                    Bx\�3�  
�          A�����@�z�@#33A�Q�CǮ����@�\)@P  A��C
                                    Bx\�B�  
�          A�����@�Q�@p�Aj=qCG�����@�z�@=p�A���C8R                                    Bx\�Q.  T          A�
��G�@��@-p�A��RC����G�@\)@S33A��
Cu�                                    Bx\�_�  �          A  ��G�@��\@7�A�33Cff��G�@x��@\��A���C
                                    Bx\�nz  �          A�
�޸R@�Q�@.�RA�C��޸R@��H@U�A�p�C��                                    Bx\�}   
�          A�����H@�=q@G
=A��CJ=���H@��H@mp�AǮC�                                    Bx\ދ�  �          AQ����H@C�
@�\)A�33C޸���H@�H@��B\)C �=                                    Bx\ޚl  �          A  ��  @l��@z=qAׅCJ=��  @HQ�@�(�A��C#�                                    Bx\ީ  �          A\)��(�@}p�@b�\A�ffC���(�@\(�@���A�{Ck�                                    Bx\޷�  
�          A\)��@c�
@x��A�ffC�{��@@  @��HA�
=C�)                                    Bx\��^  
�          Az���G�@^�R@�Q�A�p�C�)��G�@8Q�@�{B�HC�                                    Bx\��  
�          A33���@*=q@���B�C�����?��H@��BffC#W
                                    Bx\��  
�          A
=q��Q�@-p�@��B33C�{��Q�?��R@��B ��C"aH                                    Bx\��P  �          A
ff��\)@tz�@y��A�=qCk���\)@QG�@��
A��C�                                    Bx\� �  
�          A
�\��p�@��@AG�A�G�CQ���p�@�@fffAģ�C�3                                    Bx\��  T          A
�H��33@�33@
=A|��CW
��33@��@@��A�z�CO\                                    Bx\�B  �          A
�R�љ�@�z�?�A@��C���љ�@��@   A��C                                    Bx\�,�  T          A
�R��\)@�\)?�  A:�RC�q��\)@��R@��A�\)C0�                                    Bx\�;�  �          A33��  @�\)?�A@Q�C޸��  @�ff@   A��
CY�                                    Bx\�J4  T          A33��p�@��R?��RAT(�C  ��p�@��@)��A�C��                                    Bx\�X�  T          A\)���
@�ff@�AhQ�C�����
@��
@5�A�C�f                                    Bx\�g�  �          A\)��{@��H@�Ar�\C�3��{@�Q�@:=qA�  C��                                    Bx\�v&  �          A�
���@���?�  A8��C  ���@�Q�@�A|(�Cs3                                    Bx\߄�  �          A�����@��\?�ffA	G�C������@�(�?��HAO�
C��                                    Bx\ߓr  T          A33��
=@��?�{@��HC&f��
=@�\)?��
A<��C�                                    Bx\ߢ  T          A��ڏ\@��\?z�H@�p�C
=�ڏ\@�p�?У�A,Q�C�f                                    Bx\߰�  �          A�
���
@��\?^�R@��RC+����
@�{?\A z�C�3                                    Bx\߿d  �          A�����
@��>\)?k�CW
���
@�?Q�@�=qC��                                    Bx\��
  `          A������@�>�  ?��C�{����@��?h��@�
=C33                                    Bx\�ܰ  
�          A����
=@��
    �#�
Cc���
=@��\?&ff@�  C�
                                    Bx\��V  
Z          A����Q�@��\�\)�n{C���Q�@���?   @QG�C�H                                    Bx\���  �          AQ��߮@��ÿ�\�W
=C�3�߮@���=�?O\)C�{                                    Bx\��  T          A  ��  @�\)��p���C:���  @��>��?�Q�C33                                    Bx\�H  �          A  ���@�ff�������C�����@�ff>�z�?�
=C�                                     Bx\�%�  {          A\)���@�(�>\@ ��CY����@���?s33@˅CǮ                                    Bx\�4�  �          A�����@��
@A�A��HC�f����@~�R@aG�A��HC                                      Bx\�C:  �          A�
��=q@�G�@XQ�A�{C�{��=q@w�@w
=AѮCu�                                    Bx\�Q�  �          A
�H��ff@���@0��A��\C����ff@���@P��A�ffC                                    Bx\�`�  �          A
�R��{@���@;�A�C@ ��{@{�@Y��A��HC}q                                    Bx\�o,  �          A
�H��{@�{@EA��HC�f��{@s33@c33A�G�CG�                                    Bx\�}�  T          A
�\�޸R@�  @9��A���C���޸R@x��@W�A�p�CǮ                                    Bx\��x  �          A
�\���@\)@W�A�G�C�����@e�@s�
A�z�C�
                                    Bx\��  T          A
ff����@~{@eA�z�C������@b�\@���AݮCxR                                    Bx\��  �          A	��ff@w
=@r�\A���C
=��ff@Z=q@�ffA�C                                    Bx\�j  T          A	����p�@dz�@��HA��C�
��p�@E@��RA�{C#�                                    Bx\��  �          A
{��ff@e@�(�A�(�C����ff@Fff@�  A�
=C�                                    Bx\�ն  �          A	��@?\)@���A��C�H��@{@��\B�\C��                                    Bx\��\  �          A33�أ�@B�\@���A�p�C�\�أ�@!G�@��HBQ�C�\                                    Bx\��  �          A�
����@=p�@�\)A�ffC�����@��@���BffC k�                                    Bx\��  �          A(���z�@QG�@���A�RC�H��z�@1�@��A���C�                                    Bx\�N  T          A����\)@��H@Z�HA�G�C�H��\)@l��@uAυC�                                    Bx\��  �          A����33@n{@eA��CT{��33@S�
@~{A��C�R                                    Bx\�-�  �          A����\@�p�@C33A��HC� ��\@tz�@^�RA���C��                                    Bx\�<@  �          A����@��@C33A��
C�
���@��R@aG�A�  C��                                    Bx\�J�  �          AG���Q�@��
@EA�(�C\��Q�@���@b�\A�
=C+�                                    Bx\�Y�  �          A��Q�@�z�@333A�\)C� ��Q�@�=q@QG�A�\)C\)                                    Bx\�h2  T          AG���  @��@5�A��C����  @��@R�\A�33C�                                    Bx\�v�  �          A���ᙚ@�@?\)A�=qC�H�ᙚ@��H@\(�A��HC޸                                    Bx\�~  �          A��{@|(�@UA��CB���{@e�@n�RA��C��                                    Bx\�$  "          Aff��R@�p�@1G�A�33CxR��R@��@Mp�A�33CJ=                                    Bx\��  "          A�R���@��
@/\)A�33C�q���@�=q@J�HA���C�                                    Bx\�p  �          Aff��ff@�=q@#�
A��C�{��ff@�G�@@��A�{C8R                                    Bx\��  "          A�R��
=@�
=@-p�A��
C5���
=@�@I��A���C�                                    Bx\�μ  �          A=q��p�@���@'�A�G�C�H��p�@�Q�@Dz�A�p�CJ=                                    Bx\��b  T          A=q��z�@��\@A�A���CǮ��z�@�Q�@\��A��RC��                                    Bx\��  	�          A
=��Q�@�G�@0  A���C����Q�@�  @Mp�A��RCQ�                                    Bx\���  �          A\)�ٙ�@��@-p�A���C�=�ٙ�@��
@Mp�A��
Cc�                                    Bx\�	T  �          Aff��G�@�z�@2�\A��\C����G�@��H@N�RA��RCT{                                    Bx\��  �          A���\@��
@*�HA�ffC�)��\@��H@G
=A�Q�C}q                                    Bx\�&�  )          A�\�߮@��H@*=qA��HCW
�߮@��@G
=A��C�                                   Bx\�5F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�C�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�R�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�a8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�o�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�~�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�.L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�<�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�K�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�Z>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�h�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�w�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�'R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�5�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�D�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�SD              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�a�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�p�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�6              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\䜂              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�(              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��t              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��f              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\� X              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�.�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�=�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�LJ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�Z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�i�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\�x<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\��   d          Aff��
=@�  �B�\���RCB���
=@�  >�=q?�(�CE                                    Bx\啈  �          A��z�@���8Q����B���z�@�{�L�Ϳ���B�z�                                    Bx\�.  
�          Aff����@�G���
=�.�RB� ����@���=���?�RB�p�                                    Bx\��  
�          A  ���\@�\)�Q���\)B�(����\A (���z��B��                                    Bx\��z  
�          A�����@��H�B�\��z�B�k�����@�(���  ����B�8R                                    Bx\��   	#          A����R@��Tz����\B�z����R@�
=��{���B�33                                    Bx\���  �          A�����@�ff�G���  B��H����@�\)��\)��\B��                                    Bx\��l  	�          A33�tz�A �ͿE���B����tz�AG���  ����B枸                                    Bx\��  
(          Aff�^{Aff�^�R��(�B�{�^{A
=��33��RB��H                                    Bx\�
�  "          A�\�j=qA �Ϳ:�H���RB���j=qAG��W
=��\)B�                                    Bx\�^  �          A��S�
A(��xQ���ffB߮�S�
A�;�G��3�
B�z�                                    Bx\�(  "          A��aG�A������%�B�ff�aG�A�=�?@  B�\)                                    Bx\�6�  "          A�R�Q�Az�u���B�G��Q�AQ�>�@<(�B�W
                                    Bx\�EP  	�          A�R�R�\Azᾊ=q��G�B�\)�R�\Az�>u?���B�\)                                    Bx\�S�  �          A=q����@��H���FffB�������@�33<#�
=L��B�R                                    Bx\�b�  �          Aff��@�p��J=q��33B�\��@�ff���
��B��
                                    Bx\�qB  T          A�\��Q�@�  ���h��B���Q�@�׽��Ϳ+�B���                                    Bx\��  
�          A=q��\)@�\)�=p�����B�����\)@�Q쾔z��B�p�                                    Bx\掎  �          A=q���@�\)�#�
���B����@�  �8Q쿕B���                                    Bx\�4  �          A��p�@�z��G��7�B����p�@�\)��=q�	�B�p�                                    Bx\��  �          A�H��33@�p����
�=qB��H��33@��������HB�G�                                    Bx\満  �          A�H����@�33��{����B�=q����@��Ϳ.{����B���                                    Bx\��&  �          A{��Q�@�G��5��=qB�����Q�@�=q��=q��\B�                                    Bx\���  �          A�R���@��O\)��B�z����@��;\�{B�33                                    Bx\��r  T          A�����@�33��33��p�C ٚ����@���=p�����C ��                                    Bx\��  �          A�����@ٙ���  ��  CG����@�33����z=qC
                                    Bx\��  �          A����@��ÿ�R����C޸���@ٙ��k���(�C�                                    Bx\�d  �          A���z�@�  �333����B�33��z�@��þ�����z�B���                                    Bx\�!
  �          Ap����@�ff�(������B��q���@�\)����ٙ�B��                                    Bx\�/�  �          A����
@�{�fff���B�u����
@�\)�   �Q�B�#�                                    Bx\�>V  �          Ap���{@�p��^�R��z�C 0���{@޸R���E�C �                                    Bx\�L�  �          A����\@��
�.{����C
���\@�zᾔz���C ��                                    Bx\�[�  T          A��\)@�녿�z����B��
��\)@��=p����B�u�                                    Bx\�jH  �          A  �QG�A����H�r�\B��QG�A�R�����B�HB�u�                                    Bx\�x�  �          A���k�A�
�
=q�W
=B���k�Ap��ٙ��(z�B�{                                    Bx\燔  �          A��fffA�\����]p�B㞸�fffAQ�޸R�/
=B�\                                    Bx\�:  �          AG��K�A=q����up�Bޣ��K�A(������F�\B�{                                    Bx\��  �          AG��'�A��/\)���RB�W
�'�A���iB���                                    Bx\糆  
�          Az��L(�@���-p���{Bߔ{�L(�A{���iB��                                    Bx\��,  �          A���`  Aff���N=qB�u��`  A  �˅� ��B���                                    Bx\���  �          Ap��fffA�\���H�-G�B���fffA�
���\� (�B�=q                                    Bx\��x  �          Ap��|(�AG���ff��
B�\�|(�A=q�^�R��\)B�R                                    Bx\��  �          Az��uA��������B�R�uAff�&ff���B�z�                                    Bx\���  T          A���xQ�A �ÿ�ff�z�B�k��xQ�A녿^�R����B��                                    Bx\�j  �          A���p  A녿�{�
�\B���p  A�H�n{��p�B�Q�                                    Bx\�  �          A��k�A   ��z��+
=B�p��k�AG����R���RB�
=                                    Bx\�(�  �          Az��~�R@��R�\��RB�(��~�RA zῌ����  B�Ǯ                                    Bx\�7\  T          A���  @����Q���
B����  @�\)���
���HB�L�                                    Bx\�F  T          A�\����@��ÿ\�B�����@�33��\)��BꞸ                                    Bx\�T�  �          A�H��Q�@�\)����RB��R��Q�@�G����
���HB�Q�                                    Bx\�cN  �          A=q���@���Q��.�RB��f���@�=q��ff��\B�ff                                    Bx\�q�  �          A���ff@���\�ffB�k���ff@�\)�������HB�                                      Bx\耚  �          A=q��(�@�녿˅�%��B�#���(�@�(����H���HB�                                    Bx\�@  �          A{���
@�\��  ��B�����
@�z῏\)��
=B�z�                                    Bx\��  T          A=q��Q�@�ff�����Q�B��)��Q�@�  �n{��Q�B�                                    Bx\謌  �          A���  @����� (�B�  ��  @�녿����B�=                                    Bx\�2  �          A�����@�(���\)�(  B�ff����@�ff��  �p�B��f                                    Bx\���  �          A����@�(��Ǯ�"�\B����@�{������Q�B�
=                                    Bx\��~  �          A=q��ff@�p���ff��\B��R��ff@�
=�n{����B�W
                                    Bx\��$  �          A�����@�\�k����B�G����@��
�
=q�\��B�                                    Bx\���  �          Az���ff@�p�����{B�\)��ff@�\)���
�љ�B��                                    Bx\�p  T          A�R���
@�  ��=q�
=B�����
@�녿������B�33                                    Bx\�  �          A(����A Q��
=�'33B�z����Ap����� (�B�
=                                    Bx\�!�  �          A33�p  AG��G��d��B��)�p  A�H����<��B�W
                                    Bx\�0b  T          A���w�@�
=�G��Mp�B����w�A �ÿ���%�B�Q�                                    Bx\�?  �          A
=��  @���Q��(�B�8R��  @�p���=q���B���                                    Bx\�M�  �          A33���\@�\)��33�>�HB�W
���\@�녿��
�p�B���                                    Bx\�\T  "          A�H��p�@�
=�%���z�B�\��p�@�\�{�`(�B�G�                                    Bx\�j�  
�          A����@�z��0  ��z�B�����@�Q�����p��B�=q                                    Bx\�y�  T          A  ���@��
�0  ��=qB��q���@�����pz�B��H                                    Bx\�F  �          A(�����@�z��(������B������@�Q���\�e��B�L�                                    Bx\��  �          A��
=@�33�%���B�aH��
=@�R�\)�c�B���                                    Bx\饒  T          AQ����@�
=��Pz�B�33���@�녿޸R�-G�B��\                                    Bx\�8  T          A33���R@�z��&ff���B��H���R@�  ����j�HB�\                                    Bx\���  �          Aff��ff@�(����
�:=qB�.��ff@�ff���H���B��{                                    Bx\�ф  T          A  ����@��ÿ:�H����C������@�녾�(��5�Cc�                                    Bx\��*  �          AQ����H@У׿   �P��C޸���H@��þL�Ϳ���C��                                    Bx\���  
�          A
{����@�����{����C����@��H�W
=����C�\                                    Bx\��v  �          A��\@����{����C�H�\@�ff�Tz���Cn                                    Bx\�  �          A���{@�=q�Q����
C@ ��{@�33���^�RC)                                    Bx\��  
�          A\)���R@��ÿY�����
C�����R@�=q���p  C\)                                    Bx\�)h  
�          A
=��\)@�
=�u��33C�H��\)@�Q�.{���C��                                    Bx\�8  
�          A
{��z�@�
=��ff��  Cp���z�@�Q�E���z�C@                                     Bx\�F�  	`          A
=q��
=@˅��  ��C����
=@���xQ���p�C                                    Bx\�UZ  �          A
�\��(�@�{�����RC0���(�@Ϯ�����  C��                                    Bx\�d   T          A
=q����@��ÿ�����\)C0�����@ҏ\�h����G�C�q                                    Bx\�r�  �          A\)��Q�@ҏ\�������C���Q�@�z῔z����C��                                    Bx\�L  �          AQ���33@�
=��(��5p�C ����33@��ÿ��C B�                                    Bx\��  �          A���33@��׽��Ϳ!G�C���33@���>.{?�{C��                                    Bx\Ꞙ  �          A�����
@�{�&ff���B�z����
@޸R��33�  B�L�                                    Bx\�>  �          AQ���(�@Ǯ���R�33Cu���(�@�  �#�
���Ck�                                    Bx\��  T          A(���{@��
>Ǯ@%C�R��{@�33?#�
@�{C{                                    Bx\�ʊ  "          A�H���@L(�@<(�A��C� ���@C�
@Dz�A�
=C��                                    Bx\��0  "          A
=���H@q�@4z�A��HC����H@j=q@>�RA���CG�                                    Bx\���  T          A�R��G�@\)@%�A�Q�C���G�@xQ�@0  A��C�=                                    Bx\��|  �          A�\��  @��@{A���CL���  @�  @)��A�Q�C�3                                    Bx\�"  �          A�H��@��
@�Aqp�C�=��@���@ ��A���C#�                                    Bx\��  �          A����Q�@�p�@��Am�C�f��Q�@�=q@(�A�
=C=q                                    Bx\�"n  �          A	��ۅ@���?��A@  C}q�ۅ@�=q@   AV�RC�3                                    Bx\�1  �          A	���G�@�=q?�=qA)G�C:���G�@�  ?�ffA@��C�H                                    Bx\�?�  �          A	���  @�{?���AG33C����  @�33@33A^{CE                                    Bx\�N`  �          A33�׮@��@z�Ab�\CY��׮@�=q@��Axz�C�H                                    Bx\�]  T          A�����
@�  @
=qAq�C�����
@��@ffA��
C^�                                    Bx\�k�  �          A����z�@�G�?���AR=qC����z�@�ff@
=qAi��CG�                                    Bx\�zR  �          A���Ӆ@���?���AS\)C���Ӆ@�
=@
=qAj�RC{                                    Bx\��  T          A	��p�@�=q?�p�AU�C� ��p�@��@��Alz�C:�                                    Bx\뗞  �          A
�\��G�@��?�AC�
CQ���G�@�
=@33AZ�HC�                                    Bx\�D  �          A
�R��(�@��
@G�AW�C����(�@���@{AmC5�                                    Bx\��  �          A
=��{@���?��
A=C�=��{@�=q?��RAS�
C:�                                    Bx\�Ð  "          A���=q@��?�p�AR{Cs3��=q@�
=@(�Ah��C�                                    Bx\��6  �          A33����@�z�?���AO\)CxR����@��@
�HAg�C�                                    Bx\���  "          A�
����@��R?�\)A*�RC
=q����@�z�?���AD(�C
��                                    Bx\��  �          A(���z�@�Q�?�ffA=p�C����z�@�@33AX(�C                                    Bx\��(  �          A�
����@���?�=qAA�C�����@�ff@A\��C��                                    Bx\��  �          A=q��33@��@��Al��C���33@�z�@ ��A��C�q                                    Bx\�t  �          A�
��  @��H@	��A]��C�3��  @�  @��Ax(�C&f                                    Bx\�*  �          A33��(�@�33@{Af�\C�{��(�@�  @\)A�G�CE                                    Bx\�8�  �          A�
���H@��
@5A�C�����H@�  @C�
A�Cs3                                    Bx\�Gf  
�          A�\��=q@��H?���A�RCk���=q@���?�33A+33C��                                    Bx\�V  �          A���{@��?�z�@��C 5���{@ۅ?���A=qC n                                    Bx\�d�  �          A
=��=q@أ�?�A:�RCp���=q@�@AX  CǮ                                    Bx\�sX  �          A���G�@��?��AEG�C��G�@�=q@
�HAbffC�                                    Bx\��  T          AQ���  @�p�?�(�A�C ����  @�33?�G�A5�C Ǯ                                    Bx\쐤  �          A�����@�Q�?���@��B��
���@�R?�
=A=qC !H                                    Bx\�J  �          AQ����R@��
?��
AQ�C �R���R@ᙚ?�=qA6{C �q                                    Bx\��  
�          Az���33@�
=?\(�@��
B�8R��33@�?�@陚B��=                                    Bx\켖  �          A�H����@��@U�A�{Cc�����@ȣ�@eA�=qC
=                                    Bx\��<  "          A�H����@���@J�HA��C
=����@���@[�A��C�                                    Bx\���  T          A�\��(�@���@\��A�p�B��\��(�@�(�@n{AƏ\B���                                    Bx\��  �          A	���G�@�p�@��
B(�C}q��G�@�
=@��HB	=qC�=                                    Bx\��.  �          A�����@�@{�A�=qC���@���@�p�A�z�C�
                                    Bx\��  �          AG�����@��
@z=qA��HCn����@�ff@��A�
=C8R                                    Bx\�z  T          A�\����@���?��HA@Q�B�{����@�ff@G�A_\)B��3                                    Bx\�#   �          A33��\)@�\)?�
=A
�HB�(���\)@��?�  A*ffB���                                    Bx\�1�  "          A  ��ff@�G�?@  @���B�33��ff@�  ?��@�p�B�z�                                    Bx\�@l  T          A����33@�
=?�=qA z�C ��33@���?�\)A>=qC�                                    Bx\�O  �          A����(�@���?�
=A&{B���(�@�\?�p�AD��B���                                    Bx\�]�  �          A����H@��H?8Q�@��B�Ǯ���H@�?��
@�G�B�
=                                    Bx\�l^  �          A�
���R@�p�?�R@o\)B������R@�z�?s33@�\)B��)                                    Bx\�{  �          A  ��
=@�p�>�?L��B���
=@��>�@2�\B��
                                    Bx\퉪  T          A����H@�Q콸Q�
=qB�L����H@�Q�>k�?���B�Q�                                    Bx\�P  �          Aff����@�R���Q�B�������@�R>8Q�?��B���                                    Bx\���  �          A����@�׽��=p�B��
���@��>L��?�  B��)                                    Bx\���  �          A�
����@���=q���HB�B�����@�=L��>���B�=q                                    Bx\��B  �          A�����@�׾8Q쿓33B�(�����@��>\)?^�RB�(�                                    Bx\���  T          Ap���
=@��ͿE����
B�ff��
=@����9��B�33                                    Bx\��  �          A�R��(�@陚�������B�Ǯ��(�@�33��G����HB�ff                                    Bx\��4  �          Aff��G�@�Q����p  B�����G�@�33��
�P(�B��                                    Bx\���  �          A����z�@�  ������(�B�����z�@�Q�<#�
=#�
B��                                    Bx\��  T          A���@��>�@AG�B�\��@�(�?B�\@�Q�B�B�                                    Bx\�&  
�          A
=��=q@���?�\A>�\C����=q@�ff@�A[�C�                                    Bx\�*�  
�          A���H@�R>k�?��RB�z����H@�{?
=q@^�RB���                                    Bx\�9r  "          A����=q@߮�L�Ϳ��B��\��=q@߮=���?&ffB��=                                    Bx\�H  
�          A
ff���@�33���s33B�u����@��
��=q��B�Q�                                    Bx\�V�  T          A\)��@�\)?��@|(�C ����@�ff?aG�@��HC�                                    Bx\�ed  �          A
=��  @�=q?�(�AL��B��  @�\)@�\An�RB�33                                    Bx\�t
  �          A
=���\@�?�AG\)B������\@��@�RAh��B�p�                                    Bx\  �          A�\�z�H@��;�����
=B����z�H@���=#�
>�\)B�Ǯ                                    Bx\�V  T          A���^{A Q�>���@(��B��
�^{@��?@  @��B���                                    Bx\��  �          A=q�FffA�H?333@��B�z��FffA=q?�ff@��
Bݣ�                                    Bx\  T          A�R�(Q�AQ�@��AXz�B�W
�(Q�A�\@ ��A~ffBָR                                    Bx\�H  
Z          A���HA	�@�An=qBӊ=��HA\)@/\)A�Q�B��                                    Bx\���  �          A���\)A	G�@#�
A�{B�aH�\)A\)@;�A�33B���                                    Bx\�ڔ  T          A��Q�A
=@j�HA�(�B�k��Q�A Q�@���A�\)B�\                                    Bx\��:  �          Ap���A�@��A���BӅ��@�(�@��RA�=qB�8R                                    Bx\���  
�          A�`  @�
=@z�AV=qB�z��`  @��
@�HA{
=B�                                      Bx\��  �          A���{�A ��?�z�@��
B����{�A   ?�G�A=qB�L�                                    Bx\�,  
�          A��~�RAG�?fff@�ffB�ff�~�RA z�?�  @��RB�                                    Bx\�#�  T          A(���(�@��?�
=A��B�(���(�@�p�?�\A8Q�B왚                                    Bx\�2x  T          Az�����@�Q쿣�
�ffC B�����@�녿xQ�����C \                                    Bx\�A  �          A����R@�Q��X����{C�����R@���HQ���C@                                     Bx\�O�  T          A=q���@�
=�E����C �����@�33�2�\���RC k�                                    Bx\�^j  T          A����H@�(��vff��
=C�\���H@����e��Q�C�R                                    Bx\�m  �          A33���
@{�����{CJ=���
@��������C�                                    Bx\�{�  
�          A�H��=q@dz���ff��C����=q@s33�����	Q�C�q                                    Bx\�\  �          A���Ǯ@N{����
C�3�Ǯ@^�R�������C��                                    Bx\�  
�          A�\�^{A�H?#�
@�(�B���^{Aff?�G�@У�B��                                    Bx\裡  �          A33�_\)A�<#�
=#�
B����_\)A�>\@(�B�                                      Bx\�N  "          A\)�k�A��W
=����B��k�A������AG�B��)                                    Bx\���  �          A
=�n�RAG��=p�����B垸�n�RA����p����B�z�                                    Bx\�Ӛ  
(          A�����@�
=������ffB�����A Q�B�\��33B�                                    Bx\��@  �          AG����@�Q�����t��B�=q���@�����O
=B��                                    Bx\���  T          A�g
=@���3�
��z�B�{�g
=@�{�(��z=qB��f                                    Bx\���  
�          A�R�aG�A ���(Q���\)B�=q�aG�A�\�  �c�B��                                    Bx\�2  T          AQ���{@�zΐ33���B�����{@��G���ffB�3                                    Bx\��  
�          A���
=@�G���G���z�B�R��
=@�=q�#�
���B�z�                                    Bx\�+~  �          A�R��z�@�=q��(��-p�B��z�@��Ϳ�����B�(�                                    Bx\�:$  �          A  ���@��[���
=B�ff���@�\�E���(�B�ff                                    Bx\�H�  "          A�����H@��������B�=q���H@ۅ���\�ݙ�B��\                                    Bx\�Wp  �          A���~�R@�����{B���~�R@�����
��RB�                                    Bx\�f  �          A  �y��@�Q������B����y��@У�����
ffB�q                                    Bx\�t�  "          Ap��N{@�p������8
=B�z��N{@�\)�ȣ��.�B�q                                    Bx\��b  "          A\)�E�@�33��\�SB���E�@�ff��33�Jz�B��)                                    Bx\�  �          A���>�R@xQ����e(�B�#��>�R@������\33B�8R                                    Bx\�  �          A
=�=q@dz�����tz�B����=q@|�����
�k{B�Ǯ                                    Bx\�T  �          A\)�O\)@}p����z�C8R�O\)@�=q�����
=C u�                                    Bx\��  �          A
�\���@��
@*�HA�  C�H���@�\)@>{A��RC��                                    Bx\�̠  �          A���{@�=q@%A�(�C(���{@�@8Q�A��\C��                                    Bx\��F  
�          A	���
=@�(�?ٙ�A733Cn��
=@�G�?��RAV�HC޸                                    Bx\���  T          A
=q��@���@�A_�
C(���@�G�@Q�A�
C�3                                    Bx\���  �          A33��ff@�\)@(Q�A�33C
��ff@��H@:=qA���C�=                                    Bx\�8  T          A��ȣ�@�33@6ffA��\C���ȣ�@�ff@G
=A�p�CW
                                    Bx\��  �          A(����@�
=@a�A�(�C�����@���@s33A�C	�R                                    Bx\�$�  T          AQ���  @�p�@z�Aa�C����  @��@Q�A��C0�                                    Bx\�3*  "          A	G�����@���?�AF=qCE����@�@
=qAiG�C�q                                    Bx\�A�  �          A
�\���H@�{@8Q�A�\)CE���H@�G�@K�A�Q�C�                                    Bx\�Pv  �          A������@�G�@+�A��C�)����@�z�@?\)A���C�\                                    Bx\�_  T          A
=����@�\)@FffA���C.����@�=q@X��A�\)C�                                    Bx\�m�  "          A(���p�@��H@
�HAd��C�
��p�@ƸR@\)A�{Cc�                                    Bx\�|h  T          AQ���\)@Ǯ@"�\A���CǮ��\)@�33@7
=A�(�Cn                                    Bx\�  �          A����=q@�p�@)��A�{C����=q@���@>{A��C33                                    Bx\�  T          A�H����@�z�?�(�A2=qC �����@�G�@�AX(�C ��                                    Bx\�Z  T          A�\���@�33?�  A�B�{���@�Q�?��AB�RB�                                    Bx\�   �          A����@�  ?�33@��HB��)���@�?��
A�B�ff                                    Bx\�Ŧ  "          A\)���H@�G�?��\@�=qB��f���H@�
=?�z�AG�B�aH                                    Bx\��L  T          A
=��@�?�z�@�{B��\��@�?��A
=B��                                    Bx\���  �          A{��
=@߮?˅A$��B�� ��
=@���?��HAL��B�B�                                    Bx\��  �          AG����\@�(�?���@�33B��f���\@��?�G�A{B�u�                                    Bx\� >  �          A����@�33?�\)@��B��=���@���?�  A��B��                                    Bx\��  
�          A�����R@�z�?�  @ϮC aH���R@ڏ\?���A33C �f                                    Bx\��  �          A{��\)@�z�?�33@�C }q��\)@�=q?��
AffC �=                                    Bx\�,0  T          A
=��33@�\)?��@�ffC ����33@��?��HA�HC ��                                    Bx\�:�  �          A\)����@Ǯ?���A�C������@���?�ffAD  C{                                    Bx\�I|  �          A
=���@�(�?�z�@�p�CT{���@ə�?\A%C��                                    Bx\�X"  �          A�\��=q@�ff?c�
@�33C����=q@�z�?�G�A	p�CǮ                                    Bx\�f�  T          A����@�33?���@�C �
���@���?��HA#33C(�                                    Bx\�un  
�          AQ���@�\)�#�
���C8R��@�
=>���@(�CB�                                    Bx\�  �          A�����@���I�����
C������@�\)�1G����
C�q                                    Bx\�  T          A
=��p�@���r�\��\)C 8R��p�@أ��Z=q���\B��R                                    Bx\�`  T          A���(�@׮�-p���G�C �
��(�@�z���
�l��C                                       Bx\�  "          A�\��G�@�  ��  ���\C�{��G�@�녿Y����33C�{                                    Bx\�  �          A=q��@ۅ���R�I�C����@޸R�˅� z�C0�                                    Bx\��R  
�          A{���@ҏ\�G���{C�����@�Q��.�R��G�C
                                    Bx\���  
�          A=q��ff@����A���G�C\)��ff@�=q�(Q���=qC ��                                    Bx\��  �          A�H��\)@�=q�0  ��p�C ����\)@�
=��l(�C 0�                                    Bx\��D  �          A33���@���{�_
=CQ����@��ÿ�ff�4��C �
                                    Bx\��  �          A����R@أ��!��~�RC&f���R@�����Tz�C�{                                    Bx\��  �          AQ����@�=q�\)�yC����@޸R���O33C��                                    Bx\�%6  �          Az����@�\)����o�CB����@ۅ��p��E��C��                                    Bx\�3�  �          Az����@��%��p�C�����@�=q�
�H�X��C�f                                    Bx\�B�  T          A������@�
=�   �y�CQ�����@ۅ���O�C��                                    Bx\�Q(  "          A(���z�@�{� ���{\)CT{��z�@��H��P��C��                                    Bx\�_�  �          Aff���
@���33�h��C�����
@�G���33�?�C)                                    Bx\�nt  "          A{�ƸR@�  �=q�tQ�C�\�ƸR@�z�� ���K\)C.                                    Bx\�}  T          A=q����@�=q�)�����C������@�
=�  �dQ�C��                                    Bx\��  �          Aff�\@ȣ��.�R��
=C)�\@����l(�Cc�                                    Bx\�f  �          A\)����@����<����G�C�����@�\)�#�
����CB�                                    Bx\�  "          A�
��{@��R�Z�H��C
��{@���A���G�C�                                    Bx\�  �          A����Q�@�33�g���G�C�
��Q�@�=q�N{��{C�{                                    Bx\��X  �          A33���@��R�|(����
C^����@�ff�dz���33C.                                    Bx\���  �          A�H���@�p��}p��̸RC����@���c33��=qC�{                                    Bx\��  T          A
=���@����~{�ͮC� ���@У��c�
���RCh�                                    Bx\��J  �          A�
��33@�z��c33���\C �H��33@ۅ�Fff����B��                                    Bx\� �  �          A(���ff@˅��  ��\)B��
��ff@�z���=q��G�B�k�                                    Bx\��  �          A33���@�ff�P����=qCL����@����4z���33Ck�                                    Bx\�<  
�          AQ����@�\)�HQ���ffC޸���@��,(���p�C�                                    Bx\�,�  �          A�H���@�\)�j=q���\C�R���@θR�N�R���C�                                    Bx\�;�  "          A33���@���������
=C Q����@θR��33��z�B��                                    Bx\�J.  �          A{���H@������\����C �����H@˅�����
=B��=                                    Bx\�X�  �          A\)��{@�33�����{B����{@�{���
���B�8R                                    Bx\�gz  �          A���G�@�G����\��C k���G�@��
�����ffB��
                                    Bx\�v   �          A\)����@\����33C +�����@��������33B�k�                                    Bx\��  �          A������@�����ff�
=B�#�����@�33������Q�B�8R                                    Bx\��l  �          A{����@Å��G��Q�B�����@�{��33��\B��                                    Bx\��  �          A���{@������
���B�����{@���ff�\)B�=q                                    Bx\���  �          A�����
@�����33��\B�  ���
@�p����p�B��                                    Bx\��^  �          A����  @�z���{���C ����  @��������{B��\                                    Bx\��  �          A=q���
@���������C�����
@�  ��(����C
=                                    Bx\�ܪ  �          A�\��ff@�G����
����C����ff@\�l(���33C33                                    Bx\��P  �          Aff��  @�33�z�H��{C���  @˅�^{��33C�{                                    Bx\���  �          A{���\@�z���z���Q�C���\@ƸR��{����CB�                                    Bx\��  T          A�����R@��������B��=���R@�
=����� Q�B�8R                                    Bx\�B  �          Aff��(�@��������\)CL���(�@�G����H��B��H                                    Bx\�%�  4          A{���\@�G�������B�p����\@�z���33���B�G�                                    Bx\�4�  �          A�����
@�
=��
=��
=C�����
@�G���������C.                                    Bx\�C4  �          A(����@�  ��33�癚C�q���@���y����(�C�                                     Bx\�Q�  �          A�����@���������{C�H���@��H������33C +�                                    Bx\�`�  �          A
=���@�����
��{B������@�  ���
�ծB�=q                                    Bx\�o&  �          A33��Q�@����ff�3(�C ���Q�@��H��G��%33B�W
                                    Bx\�}�  �          A���Q�@�������z�C(���Q�@�=q����
=C�                                    Bx\��r  T          A�
�X��@����+G�B���X��@��
�����B�                                      Bx\��  �          A	��X��@�G��ə��<
=B��{�X��@�������-Q�B��{                                    Bx\���  �          Ap��q�@��\�Ϯ�>  C���q�@�=q�Å�/��B�aH                                    Bx\��d  T          AQ���(�@�=q����(�RCٚ��(�@������{C�                                    Bx\��
  �          A���z�@�(���
=�#(�C
  ��z�@��\���\��RCO\                                    Bx\�հ  �          A�����R@�������G�C
����R@�=q��z���
C�                                     Bx\��V  �          A{���@��R�����ffC�R���@��
���{C��                                    Bx\���  �          A���
=@�33�����z�C����
=@�Q����Q�Ck�                                    Bx\��  �          A�R��{@�����\)��
C�q��{@��\���H��
C
B�                                    Bx\�H  �          A���@�33��{�Q�C
+����@����������C                                    Bx\��  �          A��33@u������.C\)��33@��\��ff�#ffC��                                    Bx\�-�  �          A����p�@�33��G��G�C8R��p�@�������C
aH                                    Bx\�<:  
�          A���(�@��\�\�(�RC�\��(�@�=q��ff��CǮ                                    Bx\�J�  �          A(����
@�{�Å�*��C�����
@�{����z�C	��                                    Bx\�Y�  �          A33��G�@\)��
=�0(�C�)��G�@�  ����#��C
@                                     Bx\�h,  �          A�\��(�@��������3(�C����(�@�����p��&z�C	�                                    Bx\�v�  �          A{��z�@�Q���{�9�
C�f��z�@�G�����+��C�\                                    Bx\��x  �          A=q��\)@��\��Q��<G�CJ=��\)@��
���
�-�C��                                    Bx\��  �          A���@����Ϯ�<Q�C�{���@��H��33�-��C5�                                    Bx\���  �          Aff��z�@~�R��z��A{C	����z�@��������3\)C{                                    Bx\��j  T          AG�����@�ff�љ��?�C�)����@�  ��p��1ffC�                                    Bx\��  �          A����(�@o\)�����D�\C� ��(�@����ə��7{C�                                     Bx\�ζ  �          A=q��\)@a���z��Mp�C&f��\)@��
�љ��@  C�                                    Bx\��\  �          A
=��z�@�����e��CE��z�@3�
���H�[=qC�{                                    Bx\��  �          A�����H@i�����H�M
=C
G����H@���Ϯ�?  C                                      Bx\���  �          A  ����@:�H�ᙚ�Pz�C������@a��أ��D�HC�                                    Bx\�	N  �          A�����@[������KC����@�����ff�>C
c�                                    Bx\��  �          Ap����@mp����H�L��CG����@��\��
=�>�C�)                                    Bx\�&�  �          Ap����@o\)���
�N
=C
�)���@��
��  �?C.                                    Bx\�5@  �          AG����\@aG��߮�I�C� ���\@�z���z��<\)C	�                                    Bx\�C�  �          Ap�����@^{�߮�I33CE����@�33�����<  C
��                                    Bx\�R�  �          A����\@�p���(��,Q�C	�����\@�
=��ff���C:�                                    Bx\�a2  �          A(���\)@�����H�*p�C
�=��\)@������  Cp�                                    Bx\�o�  �          A
=��
=@�(���Q��(��C
����
=@�p���=q�=qCB�                                    Bx\�~~  �          A�R��Q�@�(���ff�'�C
����Q�@�p���Q���\C��                                    Bx\��$  �          A  ���@�����\)�&�RC5����@��H�����p�C�{                                    Bx\���  �          A���
=@��H�\�*C
�{��
=@��������  Cff                                    Bx\��p  �          Ap����@��H����/CaH���@�p���z��!��C	�H                                    Bx\��  �          A(����\@��\��=q�)p�C�����\@�z������C:�                                    Bx\�Ǽ  T          A����Q�@�  ��Q��.��C�)��Q�@��\�����\)C#�                                    Bx\��b  �          A����p�@���=q�1�C�H��p�@�����z��"��C޸                                    Bx\��  �          A����=q@�Q��Ǯ�-G�C����=q@�33�����\)C@                                     Bx\��  �          AG���(�@��\��p��"�
C
8R��(�@�z���ff�z�C�3                                    Bx\�T  �          A����(�@�G�����#
=C
� ��(�@�33�����C0�                                    Bx\��  �          A�
���@�������)��C�����@��
���\���C:�                                    Bx\��  �          A���G�@�  ��{�(�C��G�@�=q��
=�p�C\)                                    Bx\�.F  �          A�H���R@��\��=q�#
=C	J=���R@�z����H�  C                                      Bx\�<�  �          A�H��\)@�����&=qC
c���\)@�  ��{�p�C�                                    Bx\�K�  �          A�R��p�@��R��ff�'�C	�\��p�@�����
=�z�CT{                                    Bx\�Z8  �          A�\���R@����z��%\)C	�f���R@�������=qCu�                                    Bx\�h�  �          A�H���H@�
=����"z�C
�3���H@�G����\��CJ=                                    Bx\�w�  �          A{����@�(����
�%p�C
�H����@��R��z��p�CW
                                    Bx\��*  T          A���G�@������%ffC#���G�@�{��(��ffC�{                                    Bx\���  �          A�R����@��H��ff�'�\C)����@�{��
=�p�Cu�                                    Bx\��v  �          A\)��p�@�\)��G��!�C{��p�@��������C��                                    Bx\��  �          A�H���H@�������� C
:����H@�(���Q��=qC�\                                    Bx\���  �          A�R���@�Q�������
C	����@����Q���HC�H                                    Bx\��h  �          A�R��{@�z���33�p�C����{@������Q�C��                                    Bx\��  �          A�\���@������H��C}q���@�{��G��=qC�                                     Bx\��  �          A=q���@�p��������C	5����@�{��33��\)CQ�                                    Bx\��Z  �          Aff����@�
=��z���C������@�  ���\����C��                                    Bx\�
   �          A�R����@�������HC�����@��H�����(�C�H                                    Bx\��  �          A�H��Q�@��H����z�C����Q�@�z������G�C��                                    Bx\�'L  �          Aff���@�p���p��ffC�q���@����\����C�                                    Bx\�5�  �          A=q���@�ff��(��{C5����@�\)������33Cn                                    Bx\�D�  �          Ap���=q@�{��G��33C^���=q@��R��{��G�C��                                    Bx\�S>  �          AG����\@�\)����	�C+����\@�  ��(���  Cu�                                    Bx\�a�  �          A{����@�Q����\�C�3����@�G���
=��C�3                                    Bx\�p�  �          A33��{@�������CaH��{@��\���
���
C�                                    Bx\�0  �          A�\����@�����H���C�H����@�\)�~{�ә�C)                                    Bx\���  �          Aff���H@����\��z�C�����H@���|(��ҏ\C�                                    Bx\��|  �          A�H���@�Q���
=��C�����@�ff�c�
��Q�CJ=                                    Bx\��"  �          A�R��@��
������RC���@�=q�hQ���=qCǮ                                    Bx\���  �          Aff��@�Q���p����C�\��@�\)�p���Ǚ�C8R                                    Bx\��n  �          A�����@���
=���
C�f���@�z��c�
��  C^�                                    Bx\��  �          A����p�@���������C{��p�@�  �n�R��  C��                                    Bx\��  �          A	���z�@�����H�"  C{��z�@�\)�����\C ��                                    Bx\��`  �          Az���  @���G����CY���  @������33C+�                                    Bx\�  
�          A33��G�@�(����G�C����G�@��R������C�q                                    Bx\��  �          A�
����@���������C������@��
��Q��\)C&f                                    Bx\� R  �          A	��Q�@��H��Q��G�C�R��Q�@��R�����
�C�=                                    Bx\�.�  �          A�����@����p��=qC
=���@�Q��~�R�ݙ�B�                                    Bx\�=�  �          A
=q���H@�ff��{���C����H@�=q���s�
Ck�                                    Bx\�LD  �          A\)��\)@��Ϳ������C�\��\)@У׿
=�y��CG�                                    Bx\�Z�  T          AG���(�@�p�@
=qAf�\B�#���(�@˅@=p�A��C ��                                    Bx\�i�  T          A=q���@˅@��HA�=qB�ff���@�G�@��\B�
B��{                                    Bx\�x6  �          A
=�X��@\@�33B(�B�W
�X��@�33@ə�B3ffB���                                    Bx\���  �          A\)�|��@�
=@�  B  B��|��@���@�ffB((�B���                                    Bx\���  �          A33�~{@���@�ffB�B���~{@���@�33B4�
C ��                                    Bx\��(  �          A����\@���@�z�B$�B�#����\@�(�@У�B:G�Ch�                                    Bx\���  �          A���ff@��\@�
=B��C ����ff@�33@�(�B+�
C�\                                    Bx\��t  �          A\)�{�@�{@��HB\)B�  �{�@�
=@�G�B+  B��                                    Bx\��  �          A\)�hQ�@��
@�
=B'
=B��hQ�@�=q@�(�B>�HB�                                      Bx\���  �          A�
�J=q@��H@��RB�\B����J=q@��H@�
=B0Q�B��H                                    Bx\��f  �          A  �8��@���@��B�B����8��@�p�@��
B+�\B�                                    Bx\��  �          AQ����@ڏ\@��\B\)Bڮ���@��H@��B,��B���                                    Bx\�
�  T          A�
�G�@���@�p�B�B�8R�G�@�ff@���B!33Bڽq                                    Bx\�X  �          A�H��p�@���@c�
A�  B�=q��p�@�Q�@��A�G�B�z�                                    Bx\�'�  �          A�H��  @���@2�\A��C O\��  @�\)@h��A�Q�C!H                                    Bx\�6�  �          A���\)@��
@-p�A�z�B�8R��\)@ָR@g
=A�{B�k�                                    Bx\�EJ  �          A�H����@�\)@)��A��
B������@�=q@b�\A���B�ff                                    Bx\�S�  �          A=q��=q@�G�@*�HA��B�� ��=q@�(�@b�\A�C z�                                    Bx\�b�  �          Ap�����@�Q�@J=qA�{B�������@���@�  AׅC�                                    Bx\�q<  �          A��=q@��@$z�A���B�\)��=q@���@\��A�\)C \)                                    Bx\��  �          A{��Q�@޸R@�\Ao33B��=��Q�@��H@L��A�
=B��                                    Bx\���  �          A��33@���?��
A8��B��H��33@�\)@,��A�(�B�L�                                    Bx\��.  �          A=q��z�@ᙚ?�A-p�B�8R��z�@�Q�@'
=A��\B��=                                    Bx\���  �          A{��(�@�@�\AT��B����(�@أ�@>�RA�p�B��{                                    Bx\��z  T          Az���33@�G�?�(�A�\C =q��33@���@Q�A|z�CY�                                    Bx\��   T          A���  @�\)?��H@�(�B����  @׮@	��Ab=qB���                                    Bx\���  T          A	G����\@�z�c�
��ffC �q���\@�ff<�>k�C }q                                    Bx\��l  �          A	G���Q�@��ÿ�Q��P��C:���Q�@Ϯ��ff���CO\                                    Bx\��  �          A���33@�
=�p�����C���33@�{�@  ��=qC33                                    Bx\��  �          Az����R@���^�R���HC�����R@�33�,(���C��                                    Bx\�^  �          A(���
=@�(��P����z�C&f��
=@����(���p�C.                                    Bx\�!  �          A����
@�G��5��C����
@�(����H�O\)C\)                                    Bx\�/�  �          A
�H����@ȣ׿Ǯ�%�C������@��(�����HC��                                    Bx\�>P  �          A����R@�Q�n{��33CG����R@ҏ\<#�
=uC�q                                    Bx\�L�  �          A�
��=q@�Q쾳33�z�CǮ��=q@Ϯ?z�@s�
Cٚ                                    Bx\�[�  �          A
=��Q�@�
=?E�@�=qC��Q�@�G�?�33A/�C�
                                    Bx\�jB  �          A
�R��p�@ə�?�@b�\C0���p�@��?�Q�A��C�
                                    Bx\�x�  �          A	��z�@˅?��RA��C����z�@��
@	��Ag�
C�3                                    Bx\���  �          Aff���\@���?�=qA�C8R���\@���@p�As\)Cp�                                    Bx\��4  �          A
=���@�G�?�{A0��C����@��@�RA���C�
                                    Bx\���  �          A\)��ff@��H?��
A\)C{��ff@��H@
=qAm�CJ=                                    Bx\���  �          A
=���@Ǯ=#�
>�\)C:����@��?u@�G�C��                                    Bx\��&  �          A=q��  @ə����i��C)��  @��>���@1G�C�                                    Bx\���  �          Ap����@�\)?�=qAJ�HC
=���@���@*=qA��C�                                    Bx\��r  �          A����(�@��?�ALQ�C^���(�@���@*=qA��HC
&f                                    Bx\��  �          AQ���z�@���?��A7\)C�{��z�@��R@!G�A��\CY�                                    Bx\���  �          A����\)@���?��A�C�=��\)@�  @G�A~�RC\                                    Bx\�d  �          A���@���>W
=?���B�Ǯ���@�p�?�p�Az�B���                                    Bx\�
  �          A��ff@��@��As�
C&f��ff@�Q�@G
=A���C{                                    Bx\�(�  �          A�\��@�  @�\A`z�B�aH��@Å@@��A���C �f                                    Bx\�7V  �          A=q����@�G�@=qA�C(�����@�33@S�
A�{CQ�                                    Bx\�E�  �          A���
@�@=qA��
C+����
@��@R�\A��CaH                                    Bx\�T�  �          A{���\@\@
�HAp(�C:����\@�p�@E�A��HC8R                                    Bx\�cH  �          A
=��  @��@��A�
=Cz���  @��R@S33A�=qC	�
                                    Bx\�q�  �          A�H��@��R@{A�(�C	\)��@�Q�@R�\A�C��                                    Bx\���  �          A�H��(�@��@!G�A�G�C�q��(�@���@VffA�\)C�                                     Bx\��:  �          A���=q@�=q?�AR{C	u���=q@�ff@1G�A��Cn                                    Bx\���  �          A
=���@�G�@
=qAmCǮ���@��
@C33A��C�f                                    Bx\���  �          A\)���@��H?˅A-��C^����@�Q�@   A�p�C	                                    Bx\��,  �          A����G�@��R?�
=AQp�C+���G�@��\@7
=A��\C�                                    Bx\���  �          Az���ff@��
?��HA9��Cc���ff@���@(Q�A�C	&f                                    Bx\��x  �          A(���G�@��?�\)A  C
� ��G�@���@  At��C�                                    Bx\��  �          Az���{@��?�  A"�RC!H��{@��H@(�A��C�R                                    Bx\���  �          A����@��R?��HA33C����@�z�@=qA�(�C{                                    Bx\�j  �          A{��Q�@�(�?�ffA*�HCp���Q�@���@\)A�=qC�                                    Bx\�  �          A�\��G�@�(�?���A,��C����G�@�G�@ ��A�G�CJ=                                    Bx\�!�  �          A�R��  @�  ?��A�HC:���  @��R@�RAv{C	�R                                    Bx\�0\  "          AQ���
=@�33?��AR�HC�\��
=@�ff@7
=A�ffCu�                                    Bx\�?  �          A�H��z�@�33?��
AIG�C(���z�@�
=@1G�A�=qC��                                    Bx\�M�  �          A
=��@�=q?��@��CO\��@��?��RA`��C�{                                    Bx\�\N  �          Aff��p�@���?0��@��HC�\��p�@��?���A5C	��                                    Bx\�j�  �          A=q��p�@���?0��@�(�C�3��p�@��\?�Q�A@(�C�                                    Bx\�y�  �          A���=q@�
=?E�@��
C����=q@�  ?�ffAJ�HC�\                                    Bx\��@  �          A����\)@�ff?�R@�=qCxR��\)@�Q�?�z�A8��CaH                                    Bx\���  �          A�����@ƸR?p��@�33Cs3����@��R?�p�A_�C�)                                    Bx\���  �          A
=���@�Q�?L��@��Ch����@�G�?�AL��C�                                     Bx\��2  �          A�H��\)@�Q�?Y��@�ffC\)��\)@���?�{AR�RC}q                                    Bx\���  �          A\)��(�@ȣ�?��A��CB���(�@��R@ffA�
=C��                                    Bx\��~  �          A�H��  @�?��@�CY���  @�z�@��Ar�HC��                                    Bx\��$  �          A{��G�@\?�  @��C  ��G�@�=q@�\AhQ�CB�                                    Bx\���  �          A{��=q@��?�G�@��CJ=��=q@�G�@33AiC�{                                    Bx\��p  �          Aff��p�@�p�?��\A�C���p�@��@�A���Ch�                                    Bx\�  �          A������@���?�@p  C�)����@��H?�=qA3�C                                    Bx\��  �          A33��\)@���@!G�A�G�B����\)@��@e�AՅB�Ǯ                                    Bx\�)b  �          A�\��
=@�33@7
=A�Q�B�=q��
=@��@{�A�
=B���                                    Bx\�8  �          A (���=q@\?��@��HC����=q@�G�@	��AyC.                                    Bx\�F�  �          A ������@���������C
�=����@��?J=q@���C
                                    Bx\�UT  �          Aff���
@�=q�aG��˅C� ���
@���?E�@��C�                                     Bx\�c�  �          A�R��33@�����j�HC���33@��?   @a�C�                                    Bx\�r�  �          A�H��@�������1�C����@��H?�R@�33C�3                                    Bx\��F  �          A=q���@�(�>�{@��C(����@�
=?���A�C��                                    Bx\���  �          A=q���
@�\)?��HA((�C�R���
@�(�@�A��C	�H                                    Bx\���  �          A�����@��@aG�A�z�Ch�����@�p�@�ffB�RC��                                    Bx\��8  �          A���  @��\@U�A�p�CaH��  @�(�@���B ��C8R                                    Bx\���  �          A����\@���@�A��RC:����\@���@X��A�
=C��                                    Bx\�ʄ  T          A����33@���@p�A�p�Cn��33@�\)@^�RA�CL�                                    Bx\��*  �          Ap�����@�(�@.{A�
=C�����@���@j�HA�p�C
xR                                    Bx\���  �          Ap�����@�33@   A�CT{����@�G�@_\)Ȁ\CY�                                    Bx\��v  �          Ap���=q@�@/\)A�{C�q��=q@�=q@p  Aܣ�C��                                    Bx]   �          A�����\@�33@5A�  C�=���\@��R@xQ�A�
=C��                                    Bx] �  �          Ap����@�ff@0  A���C �\���@��\@s�
A�
=C��                                    Bx] "h  �          Ap����
@��
@ ��A��B�
=���
@�G�@g�A�G�C�\                                    Bx] 1  �          Ap����H@�
=@�A�  B��)���H@�p�@]p�A�
=C�                                    Bx] ?�  �          A ����ff@�G�@�
Amp�C����ff@���@G�A���Cs3                                    Bx] NZ  �          A ����ff@���@ffAr{C����ff@���@G
=A�(�C	\)                                    Bx] ]   �          A �����@�ff@ffA�{C	ff���@���@S33A�{C��                                    Bx] k�  �          A ����@�{@Q�A���C�q��@^�R@���A��HC�\                                    Bx] zL  �          A Q����@p  @�=qA�C&f���@9��@��RB�C}q                                    Bx] ��  �          @������@���@4z�A�  C(�����@z�H@i��A߅CB�                                    Bx] ��  �          A (����@�  @5�A���B������@��H@\)A��B��R                                    Bx] �>  �          A Q���
=@��@6ffA���C���
=@�z�@z�HA�RC��                                    Bx] ��  �          A Q���33@�
=@G�A�
=B�.��33@�  @�\)A�Q�C�                                    Bx] Ê  �          A{��=q@��R@J=qA�
=C.��=q@�\)@��RA�Q�C�                                    Bx] �0  �          Aff��Q�@��
@H��A�\)C����Q�@�z�@�{A��C��                                    Bx] ��  �          A{���@��
@4z�A�  C�����@��R@w�A�\)C#�                                    Bx] �|  �          A ����ff@�\)@H��A�(�C)��ff@�  @��A��C33                                    Bx] �"  �          A z�����@���@g
=A�Q�C&f����@�\)@��HB�C��                                    Bx]�  �          A �����
@�G�@e�AӮC�H���
@�
=@��B
z�C	s3                                    Bx]n  �          A Q����@�ff@{�A��C�f���@��@�z�B\)C	G�                                    Bx]*  
�          A Q����\@�
=@l��AۮC�����\@��
@�p�B�C	�H                                    Bx]8�  �          A   ���H@���@l��A�G�C8R���H@�G�@��B(�C
ff                                    Bx]G`  �          A�\����@�?�=qA2ffB�L�����@�\)@6ffA�G�C ��                                    Bx]V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]d�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]sR             A{���@Ϯ?��HA�C � ���@Å@!G�A��C8R                                   Bx]��  �          AG�����@�{?�{A2�HB�������@�
=@9��A��C�                                   Bx]��  �          A������@�z�?�AT��C &f����@��@L��A�{C��                                   Bx]�D  �          A�����@�33@ ��A`z�B�u����@���@Q�A�z�CE                                    Bx]��  �          A����@ə�?�Q�AZffB��
���@�Q�@L��A��Cn                                    Bx]��  �          A�
��=q@θR?�Q�A=G�B�u���=q@�
=@@  A���C �                                    Bx]�6  �          A�R����@��
@!G�A��
C )����@��R@o\)A�{CT{                                    Bx]��  �          A���\@���@1G�A�  CT{���\@�@|��A���C��                                    Bx]�  �          A ����(�@�p�@Q�A�{CxR��(�@�G�@dz�A�\)C��                                    Bx]�(  �          @�
=��  @�=q@Q�Ax  C����  @��@S�
A�Q�C��                                    Bx]�  �          @���Q�@��@   A���C���Q�@��R@hQ�A��
CQ�                                    Bx]t  �          A ����(�@��@G
=A��C����(�@��\@�\)A�G�CL�                                    Bx]#  "          AG���@��R@U�A���C���@��
@��B�\C޸                                    Bx]1�  �          Ap����H@���@|(�A�
=C0����H@�{@�ffB�
C{                                    Bx]@f  �          AG�����@�\)@vffA�{C�����@���@�z�B�C
:�                                    Bx]O  �          @������H@��@r�\A��C�����H@�@���B�C	��                                    Bx]]�  �          @�����{@�=q@��A�z�C ����{@��@fffA�ffCE                                    Bx]lX  �          A
=���H@�33?5@�\)C �R���H@���@z�Aj�\C�                                    Bx]z�  �          A=q���@��Ϳ.{���B�  ���@�(�?W
=@��B�#�                                    Bx]��  �          A������@�
=�.{��B��)����@ָR?O\)@��\B���                                    Bx]�J  T          A���G�@Ӆ>�G�@B�\B�����G�@�33?�\)APQ�C k�                                    Bx]��  �          A���Q�@�Q�?\)@x��B�33��Q�@�
=?�(�A]C �{                                    Bx]��  �          A�\���H@��
���R�
=qC �)���H@�G�?���@�Q�C                                      Bx]�<  �          A=q���@˅���H�^�RC �=���@��?k�@ϮC �                                     Bx]��  �          A����
@�ff�@  ����B�{���
@�ff?333@�{B�
=                                    Bx]�  �          A Q����\@У׿Q����B�����\@�G�?#�
@��
B��                                    Bx]�.  �          A ����p�@θR�}p���=qB��3��p�@У�>�@X��B�8R                                    Bx]��  �          A{��p�@��ÿ��K
=B�����p�@��þ�����\B��                                    Bx]z  �          A  ���@ָR�G��\z�B�L����@߮����J=qB�\                                    Bx]   �          A�
��(�@��Ϳٙ��8��B��q��(�@����Ϳ333B�33                                    Bx]*�  �          A  ����@�\)����B�W
����@�33>��?��B�k�                                    Bx]9l  �          A�����@ҏ\��Q��8��B�������@�G��#�
����B�\                                    Bx]H  �          A(���p�@��Ϳ�\�?�B�(���p�@��
������
B��                                     Bx]V�  T          A������@�{��(����B�aH����@��H>��?��\B�=q                                    Bx]e^  �          A	p���p�@�\)���Z�\B�B���p�@�׾�G��:=qB�                                    Bx]t  �          A����p�@����	���h��B���p�@޸R�\)�r�\B��                                     Bx]��  �          Az���=q@љ��"�\���RB�����=q@޸R�xQ���Q�B�ff                                    Bx]�P  �          A
=q��{@�=q��
=�2�HB����{@�Q�    ���
B���                                    Bx]��  �          A\)��33@�z��
=�M�B��\��33@��;k���G�B�q                                    Bx]��  �          A33��{@������c�B���{@�R��
=�0��B�                                    Bx]�B  T          A
=���@�G����^=qB�k����@�=q�������B�=                                    Bx]��  �          A
�R���@����D  B������@����G��@  B�\)                                    Bx]ڎ  �          A
�\��@���p��B�q��@�=q>���?�(�B��
                                    Bx]�4  �          A(���Q�@��
���p�B�.��Q�@�>�33@
=B�\)                                    Bx]��  �          Az����R@����Q��6ffB������R@��#�
���
B�\                                    Bx]�  �          A\)��z�@��
��Q��7�
B�{��z�@�\���
��G�B��                                    Bx]&  �          A	���@߮��
=�4  B�G����@�<��
>��B���                                    Bx]#�  �          A
�R���
@��ÿ�(���B�u����
@�p�>�  ?�\)B�W
                                    Bx]2r  �          A	�����@�33�����)p�B�����@��>#�
?�=qB���                                    Bx]A  �          A����33@�33��Q�� (�B�����33@�?
=q@hQ�B�=q                                    Bx]O�  �          A	�����@�R�W
=���B�z�����@�ff?h��@\B�=                                    Bx]^d  �          A�
��z�@�Q��\�]p�B�L���z�@�p�?�  A�HB��                                    Bx]m
  �          A�
��G�@�������\B���G�@�?��HA�RB�                                     Bx]{�  �          A(��k�@�p���  �ٙ�B�=q�k�@�Q�?�{A.ffB�8R                                    Bx]�V  �          A�\�s�
@陚�Q���
=B�{�s�
@�G�?xQ�@�\)B�.                                    Bx]��  �          A�\��
=@�=q��\�`��B�z���
=@����
�p�B�G�                                    Bx]��  �          A
=���@�(��ff�f�\B�Q����@�{��Q��(�B��                                    Bx]�H  �          A���33@߮���Lz�B�q��33@�\)��\)��\B�{                                    Bx]��  �          A��q�@��ÿ�����B����q�@�(�?�\@a�B�.                                    Bx]Ӕ  T          A��mp�@���\)���\B�B��mp�@�G�?333@��B��                                    Bx]�:  �          A���l��@��Ϳ
=���HB���l��@�=q?�  A
�RB��                                    Bx]��  �          A��qG�@��c�
��ffB�  �qG�@�\)?n{@�Q�B�                                    Bx]��  �          A ���z=q@�{����ffB��)�z=q@�Q�?(�@�(�B�ff                                    Bx],  �          A(����
@�z��  �"�RB��H���
@�G�>��
@
=qB���                                    Bx]�  �          A
�H��(�@�\���LQ�B�\)��(�@�\��Q�(�B�                                    Bx]+x  �          A
�R���@�����H��B�
=���@�p����L��B�Q�                                    Bx]:  �          A
{���
@�p�� ���X  B�R���
@�{������
B���                                    Bx]H�  �          A  ��Q�@�R�33�X��B�{��Q�@��.{��\)B��                                    Bx]Wj  �          A	����@��	���g33B�#����@���=q��B�
=                                    Bx]f  �          A�\�]p�@�Q�#�
��=qB���]p�@�G�?�ffAN�\B�\)                                    Bx]t�  �          A��Q�@�G�@�G�B�\B��쿸Q�@�p�@���BA�B�\                                    Bx]�\  �          A{��z�@�Q�@�Q�A���B˅��z�@�ff@��HB8(�B��                                    Bx]�  T          A
=��z�@�=q@FffA�  B�LͿ�z�@�G�@��B�
B��
                                    Bx]��  �          A���@�@B�\A��
B�����@��H@�=qB�B�z�                                    Bx]�N  �          A\)��
@��@z�A�p�B����
@�p�@�p�A�Q�B�#�                                    Bx]��  �          A��<(�@��?�{A��B��<(�@�p�@O\)A��RB��                                    Bx]̚  �          A\)�C�
@�G�?\(�@���B���C�
@�=q@1G�A�Q�B�                                    Bx]�@  �          A
=�?\)@�ff?��A��B߽q�?\)@ۅ@L(�A��HB��                                    Bx]��  �          A\)�E@�?�A�B��
�E@�@Dz�A�(�B�
=                                    Bx]��  �          A
=�B�\@�Q�?�z�A{B�
=�B�\@�ff@C�
A�  B�33                                    Bx]2  �          A\)�P  @�33?�Q�A"�RB��P  @�
=@S33A���B�{                                    Bx]�  �          A33��33@�=q>��@9��B�L���33@�
=@\)A~ffB��)                                    Bx]$~  �          A��Z=q@�p�?�  @�
=B�k��Z=q@���@9��A�G�B��                                    Bx]3$  �          AQ��`��@�33?�\)Az�B�
=�`��@׮@P  A��B�\                                    Bx]A�  �          A���QG�@��?���@�(�B��
�QG�@�Q�@@��A�=qB�\                                    Bx]Pp  �          A���`��@�  ?�G�@�{B�8R�`��@�
=@<��A��B�                                    Bx]_  �          Ap��QG�@�33?��@�{B���QG�@ᙚ@@��A�p�B���                                    Bx]m�  �          Az��S33@��?@  @��RB�(��S33@��H@.�RA�B��                                    Bx]|b  �          A(��O\)@�=q?+�@��B�\)�O\)@��
@*=qA�  B��                                    Bx]�  �          A��HQ�@��H>�Q�@�RB��f�HQ�@�
=@
=A���B��                                    Bx]��  �          A�
�AG�@�p�>#�
?�\)B�
=�AG�@�33@��Aw�B�q                                    Bx]�T  �          A��B�\@�z�>8Q�?�p�B�aH�B�\@��@p�Ay��B��                                    Bx]��  �          A33�AG�@��
>#�
?���B�L��AG�@�G�@��Ax��B�
=                                    Bx]Š  �          A\)�=p�@��ͼ#�
�uB�W
�=p�@��
@33Ag33B���                                    Bx]�F  �          Ap��S�
@�(�=u>���B����S�
@�\@
=Ak33B��                                    Bx]��  �          A{�}p�@���?��@�33B��}p�@�ff@Dz�A�  B�{                                    Bx]�  �          Aff�U@�p�?(��@���B���U@�R@-p�A��HB�Ǯ                                    Bx] 8  �          A��XQ�@�\?aG�@\B�
=�XQ�@ᙚ@:=qA��B�.                                    Bx]�  �          A{�K�@�{?Y��@��HB��K�@��@:=qA��HB��                                    Bx]�  �          A�R�Dz�@��H=#�
>uB����Dz�@���@
�HAo33B�aH                                    Bx],*  �          A���AG�@��׾\�'�Bޅ�AG�@�\?�G�ABffB�z�                                    Bx]:�  �          A��*=q@�33�&ff��\)B�k��*=q@�\)?��
A)G�B���                                    Bx]Iv  �          A=q�$z�@��J=q��(�B���$z�@��H?�Az�B�L�                                    Bx]X  �          A���A (��������B͊=��A�?fff@�\)B�W
                                    Bx]f�  �          A=q��A=q��{��B�.��A{?�
=AG�B�33                                    Bx]uh  �          A
=����@ҏ\?s33@���B�������@��@.�RA��C @                                     Bx]�  �          A\)��  @��
>�G�@FffB����  @ָR@��A�ffB�                                    Bx]��  �          A��{�@�\)���
�\)B���{�@�ff?��RA_�B�                                    Bx]�Z  T          A\)�n{@陚�\�*=qB��n{@��
?�A;�B�33                                    Bx]�   �          A��c�
@��p����G�B�\�c�
@�\?�33A�B�q                                    Bx]��  �          AG��n{@�33��p���
B���n{@���?aG�@�G�B�W
                                    Bx]�L  �          AG��fff@�(���z��(�B�  �fff@�\)?5@��B�ff                                    Bx]��  �          A��e@�(���
=�8��B���e@�>�@J�HB��H                                    Bx]�  �          A(��W�@��
�Ǯ�-B�#��W�@�Q�?z�@���B�W
                                    Bx]�>  �          A=q�aG�@�  ���x��B�k��aG�@�\���Tz�B�Q�                                    Bx]�  �          A�R�S33@��H�(����\B��)�S33@�������B�                                     Bx]�  �          A{�S�
@�{�-p���
=B�k��S�
@���\)�tz�B�3                                    Bx]%0  �          AG��N�R@���*=q���B�\�N�R@�(��
=��B�8R                                    Bx]3�  �          A��X��@�p��#33����B�.�X��@����W
=B�p�                                    Bx]B|  �          AQ��[�@��{��
=B�u��[�@�׾�{�B���                                    Bx]Q"  �          A��g�@����G���ffB�u��g�@�(��.{��B�.                                    Bx]_�  �          AQ��hQ�@�����v�RB���hQ�@�{��\)���HB�                                      Bx]nn  �          A���]p�@�ff��
���B�L��]p�@�녾����G�B�#�                                    Bx]}  �          A��l(�@�33���~�RB���l(�@�ff�\)�xQ�B��                                    Bx]��  �          A����p�@��Ϳ�33�T��B�G���p�@���>.{?�z�B�p�                                    Bx]�`  �          A�H�K�@ҏ\�i����z�B���K�@�33�˅�4  B���                                    Bx]�  �          Az��;�@�
=���\��RB�k��;�@����%����B���                                    Bx]��  �          AQ��7�@��H��ff�Q�B���7�@��=q��  Bފ=                                    Bx]�R  �          AG��7�@�{��p�� ffB����7�@�ff�
=����B��                                    Bx]��  �          A{�1�@�\)�������B� �1�@�Q������Bܨ�                                    Bx]�  T          A���333@�  ��� 33B��333@�Q����{B���                                    Bx]�D  �          A���1�@�=q������G�B��
�1�@����
=q�p��B�z�                                    Bx]	 �  �          A���G
=@�33�|(���33B�p��G
=@�R��=q�LQ�B�33                                    Bx]	�  �          A��0��@��|(���z�B����0��@�G���ff�IB��                                    Bx]	6  �          A�
�5@����n�R�י�B�p��5@�\�����0  B�{                                    Bx]	,�  �          Az��/\)@ڏ\�n{��ffB߳3�/\)@�(�����+�Bۅ                                    Bx]	;�  �          AQ��8Q�@��^�R����B�#��8Q�@��Ϳ��\���B�G�                                    Bx]	J(  �          A  �H��@�  �AG���{B�\)�H��@�\�O\)����B�                                    Bx]	X�  �          A�
�QG�@����333����B��)�QG�@�G��z���G�B��
                                    Bx]	gt  �          A��S�
@�\�!G����RB�{�S�
@�Q쾔z��33B�=                                    Bx]	v  �          Az��J�H@�Q��C33���HB���J�H@�33�Q���p�B�G�                                    Bx]	��  �          Az��U@Ϯ�w���G�B�k��U@�33��  �C�B��
                                    Bx]	�f  T          AG��U�@��H�u���B��U�@���Q��;�B�L�                                    Bx]	�  �          Ap��U@���z=q�߮B���U@�{��G��B�HB�aH                                    Bx]	��  �          AG��N�R@ҏ\�|(����B�=q�N�R@�R���
�E�B�Ǯ                                    Bx]	�X  �          A=q�8Q�@����p���  B���8Q�@���z��m��B�=q                                    Bx]	��  �          A�
�-p�@��H��  ���B�=q�-p�@����Q���Q�B�.                                    Bx]	ܤ  �          A��8��@ָR�o\)�أ�B➸�8��@��ÿ��
�,(�B�
=                                    Bx]	�J  
�          A�
�<��@ٙ��dz����
B��f�<��@�=q����ffBޞ�                                    Bx]	��  T          A�H�<(�@����z�H��B�p��<(�@����  �FffB�=q                                    Bx]
�  �          A
=�0  @У����\��B����0  @�ff��33�W�B܏\                                    Bx]
<  �          A33�#33@Ӆ��z���=qB�8R�#33@�녿�
=�YG�B�Q�                                    Bx]
%�  �          A��z�@��H�������Bը��z�@�  ���
�H  B��H                                    Bx]
4�  �          A���@�������Q�Bѽq��@�  ��\)�T  B�B�                                    Bx]
C.  �          AQ��\)@���>{��G�B����\)@��5����B�{                                    Bx]
Q�  �          A	�_\)@�
=�G
=��\)B瞸�_\)@��\�=p���p�B��                                    Bx]
`z  �          A
=q�j�H@����J=q��  B�W
�j�H@��׿O\)��z�B�                                    Bx]
o   �          A
�\�x��@��
�AG����B�=q�x��@��R�+���
=B�=                                    Bx]
}�  �          A��|��@�ff�:=q���B�=�|��@���
=q�c�
B�{                                    Bx]
�l  �          A
ff�xQ�@��,(���
=B�aH�xQ�@�
=���
�
=B�aH                                    Bx]
�  �          A����Q�@�=q�9����z�B�����Q�@�z�#�
��G�B���                                    Bx]
��  T          A������@�=q�:=q���B�(�����@�(��&ff���B��                                    Bx]
�^  T          AQ����@ڏ\�4z�����B�q���@��
�\)�p  B��f                                    Bx]
�  �          AQ�����@��H�.{���RB�G�����@��H��ff�A�B��                                    Bx]
ժ  �          A����Q�@�z����|��B�\)��Q�@�׽u��
=B�\                                    Bx]
�P  �          A	�����@�=q��
=�N�\B�p�����@�\>���@G�B�k�                                    Bx]
��  �          A	�����@�=q�˅�*�RB�L����@�\)?!G�@�p�B�{                                    Bx]�  �          A
=q���
@ٙ�������RB������
@�p�?B�\@���B�\                                    Bx]B  �          A\)����@أ׿�\)�z�C \����@ۅ?Tz�@�
=B�k�                                    Bx]�  �          A33���H@�33��\)�FffC �3���H@ۅ>��R@G�B���                                    Bx]-�  �          A���G�@�p��(Q���(�C����G�@�p�����E�B��)                                    Bx]<4  �          A
=���
@�=q�HQ����HC}q���
@�  ��ff��{C ��                                    Bx]J�  T          A(���33@�  �B�\����C  ��33@���}p���
=C�                                    Bx]Y�  T          A	p���33@����S�
���C5���33@أ׿�p���B���                                    Bx]h&  �          A(�����@�=q�O\)���C33����@љ���(��(�C �H                                    Bx]v�  �          A�R��\)@���A���
=C����\)@У׿�G�����C ��                                    Bx]�r  �          A�\����@\�\�����B��
����@ۅ��=q�B�G�                                    Bx]�  �          A�R���\@�
=�@����G�B������\@�33�\(���z�B�Q�                                    Bx]��  �          A\)���@�(��G
=���\B��q���@�G��h�����B�u�                                    Bx]�d  �          A�����@�33�P  ���B�{����@ᙚ��ff��(�B�u�                                    Bx]�
  �          A	p���p�@�������p�C �f��p�@��H�W
=����B�33                                    Bx]ΰ  �          A
{����@�33����J�HC����@��
>�  ?��C�{                                    Bx]�V  �          A
�\����@�33��H����C�{����@љ����
���C�=                                    Bx]��  �          A���=q@�=q�
=q�f{CY���=q@��#�
��C�=                                    Bx]��  �          Ap����H@�
=�p���G�B��{���H@�zὣ�
���HB�u�                                    Bx]	H  �          A����@�  ��w�C޸��@��ͽ��@  C (�                                    Bx]�  T          A�����R@�z��*�H���C����R@�p��
=q�aG�C�\                                    Bx]&�  �          A�����@��H�<(���=qC�H���@�
=�L����\)C�)                                    Bx]5:  �          A
{��(�@�=q��H���\C�
��(�@��þ��
��C�                                    Bx]C�  �          A
�H��Q�@�G��
=�|��C����Q�@�\)��=q��G�C�H                                    Bx]R�  �          A����@\�7
=��C\��@��8Q���ffCaH                                    Bx]a,  �          AQ���ff@��
�(����  C�q��ff@�z���H�O\)C��                                    Bx]o�  �          A\)��  @���0����=qC8R��  @Ϯ�.{��
=C��                                    Bx]~x  �          A  ���
@���*=q���C�����
@θR���p��CG�                                    Bx]�  �          A���{@����
=���C)��{@θR��  ��(�C)                                    Bx]��  �          A���@ʏ\�	���k\)CO\��@�=L��>�33B��{                                    Bx]�j  �          A�
���H@���
=��G�B��)���H@�
=��\)��B��\                                    Bx]�  �          A����H@�{�33�`z�C G����H@�Q�>B�\?��\B��                                    Bx]Ƕ  T          A�����@���G��$��C�)����@ʏ\?�R@�{C+�                                    Bx]�\  T          Az���z�@���E���G�C����z�@љ��xQ���G�CxR                                    Bx]�  �          AQ���G�@�\)��R���HCQ���G�@�ff��=q��C J=                                    Bx]�  &          AQ����R@�ff�'
=���HC����R@�
=����.�RB���                                    Bx]N  �          A
�\��z�@����p���Q�C����z�@׮�k����C �f                                    Bx]�  "          A�
����@��
����G33C5�����@��
>�
=@1G�C +�                                    Bx]�  
�          A	���
=@�Q����t��C#���
=@���u���Cc�                                    Bx].@  T          A	�����@�=q���H�R=qC+�����@Ӆ>��?�p�C��                                    Bx]<�  �          A	����@��H��ff�@z�CG����@�=q>�
=@3�
C=q                                    Bx]K�  T          A�R��
=@��H��(��X��C�f��
=@���>8Q�?���C}q                                    Bx]Z2  
�          A=q���\@�Q������C)���\@�  �\�%�C�=                                    Bx]h�  
�          A(���G�@�\)��ff�C
=C� ��G�@�  >k�?��
C+�                                    Bx]w~  �          A	��Q�@�(���{�G�C����Q�@��>aG�?�p�CJ=                                    Bx]�$  T          A�����@����
=�fffC������@�p����aG�C33                                    Bx]��  
Z          A
=��  @�33���R�YC�=��  @�p�=���?+�C�q                                    Bx]�p  �          A�R��\)@�(����G33C=q��\)@���>�\)?��C�R                                    Bx]�  "          A=q����@����\�`��C������@�ff=u>�
=CY�                                    Bx]��  �          A{��z�@�G��p��tz�C=q��z�@�{���Q�CO\                                    Bx]�b  
�          A{��z�@��\�ff�g�C���z�@�ff    <�CQ�                                    Bx]�  
�          A���@�z��{�MC&f���@��>���@G�C�                                    Bx]�  "          Aff���H@�
=���J�\C�����H@�
=>\@%C�                                     Bx]�T  �          A{����@ƸR�����W33CT{����@�Q�>�\)?�z�C�                                    Bx]	�  "          A����\)@�z��(��s�
C���\)@�G��u�\C{                                    Bx]�  
�          A=q��
=@��R�\)�w�C���
=@�(��#�
����C�                                    Bx]'F  "          Aff���\@�{���_�
C����\@�G�=#�
>uC��                                    Bx]5�  	�          Aff��@�G����j�RC�R��@���=�\)>�ffCO\                                    Bx]D�  
�          A����
@�G��G���CB����
@�ff�u��
=C k�                                    Bx]S8  	�          A\)��z�@�{�:�H���RC���z�@�33�J=q��G�C ��                                    Bx]a�  X          A�
��Q�@�ff�1���C����Q�@�녿(�����C��                                    Bx]p�  
Z          AQ����@�����S33CW
���@�
=>�=q?��C                                      Bx]*  T          A����
@�����j=qC}q���
@�
==�\)>��C�\                                    Bx]��  �          A�\��\)@�{�%��=qC�)��\)@�\)����Tz�C�                                    Bx]�v  
�          A�
��  @�\)�+���(�C}q��  @ə��
=q�p  C�=                                    Bx]�  �          Az���
=@��
�%��{C�f��
=@������3�
C0�                                    Bx]��  "          A�����@�
=�*�H���C������@�G����n{C�3                                    Bx]�h  �          A�H����@���4z���33C������@�{�8Q�����C��                                    Bx]�  "          A
=��=q@��
�$z���z�C���=q@��;�p��'
=C \)                                    Bx]�  �          Ap���\)@��\���\��ffC	G���\)@����Q���
=C�                                    Bx]�Z  "          Ap����
@�G���G����C����
@���%����C0�                                    Bx]   T          A�H��(�@��R��Q���\)C	s3��(�@����ff���C�                                    Bx]�  "          A����@��
������
C	k�����@�ff�0  ��G�C5�                                    Bx] L  �          Az���  @Q���(��?  C����  @��R��33�{Cٚ                                    Bx].�  
�          A����@s33����%{C�����@�ff�p  ��G�C��                                    Bx]=�  �          A33���@�G����R�  C
�H���@�  �I�����RC��                                    Bx]L>  �          A����@�z��j=q��p�C�
����@ʏ\����0Q�B�(�                                    Bx]Z�  T          A�R��(�@�p�������(�C^���(�@�  ���H�^=qB���                                    Bx]i�  �          A=q���@�(��j=q��ffC{���@�=q����/33B�{                                    Bx]x0  "          A{��ff@�p���z��
=C
h���ff@��R�U�¸RCu�                                    Bx]��  "          A33��z�@��H���\�+�\C	���z�@�G��q���  B�ff                                    Bx]�|  T          A33�qG�@�  ����(�\C k��qG�@�(��]p��ȏ\B�(�                                    Bx]�"  �          A�H�l(�@�=q���0{C ��l(�@����mp��׮B��f                                    Bx]��  
�          A�\�mp�@�����H�7�C�3�mp�@�33�|����B��                                    Bx]�n  T          A�����@�  ��z����CO\����@�{�?\)��Q�B��                                    Bx]�  
�          A{�vff@��\��Q����B�G��vff@У��8�����B�{                                    Bx]޺  �          A��aG�@�  ������
B�Ǯ�aG�@��5����B랸                                    Bx]�`  
�          A�H�^{@�G���33�G�B���^{@�z��$z���p�B�aH                                    Bx]�  �          Aff��=q@��
���
�
�C(���=q@�{�%����B��
                                    Bx]
�  �          A=q��Q�@�z�����ffC.��Q�@�z��Q���z�B��=                                    Bx]R  "          A�����@���^{�ʸRC����@�\)����.�HC                                    Bx]'�  
�          Aff��p�@�=q������33C	  ��p�@��33�iG�C�                                    Bx]6�  
Z          Aff���@����z���C�)���@��333��G�C��                                    Bx]ED  �          A �����H@�ff�n�R���C
�����H@�
=���P  C
=                                    Bx]S�  
�          A�\���@vff��z��G�C�f���@��H�L(���{C��                                    Bx]b�  T          A33��G�@:�H��\)�;ffC����G�@����G���C��                                    Bx]q6  "          A�\��ff@Q������+�RC}q��ff@�G��\)��C��                                    Bx]�  �          Aff���@E�����!�CW
���@����w
=���C
��                                    Bx]��  �          A33��\)@&ff��  �(C����\)@�z���ff��33CJ=                                    Bx]�(  �          A����  @\����33�33C�3��  @����aG���ffC#�                                    Bx]��  �          A=q����@hQ���(��C������@�\)�_\)�˅C��                                    Bx]�t  T          A33����@^�R��
=�\)C�)����@��
�hQ���(�C�
                                    Bx]�  �          A�
��=q@e������(�C���=q@�{�aG����HC�3                                    Bx]��  "          A\)��Q�@n�R��=q��C����Q�@����X����G�C��                                    Bx]�f  �          A�\���R@q�����33C����R@��\�S33����C^�                                    Bx]�  �          A=q����@i����ff�(\)C� ����@��
�qG����HC��                                    Bx]�  "          A=q���@w���p��G�C�q���@�\)�[��ǮC�H                                    Bx]X  �          A����z�@����>{����Cc���z�@�  �B�\���C \                                    Bx] �  �          AG����@��
�33��z�C����@ə����L��C �                                     Bx]/�  �          A�����@���>�R���C������@��ÿ\(���z�CW
                                    Bx]>J  "          A{���\@�33�W
=��p�C
�����\@�������  C.                                    Bx]L�  
�          A ����Q�@�{�~�R��G�C����Q�@�33�(��~�HC�
                                    Bx][�  �          @�����H@<(������'�\C�R���H@����u��33C	�\                                    Bx]j<  T          @��H���@+���=q�@(�C&f���@����ff�
�\C=q                                    Bx]x�  T          @����@]p��x���Q�C�=���@�z������CY�                                    Bx]��  T          @��
����@o\)�%��
=C����@�p������C	B�                                    Bx]�.  �          @�\)���@S33�]p���  C�����@��\����\)C	�                                    Bx]��  T          @�{���@q녿�{���
C�=���@�ff��Q��E�C�H                                    Bx]�z  
�          @�  ��=q@~{���R����C����=q@�p�����`  C
�                                    Bx]�   �          @�
=����@qG��(�����HCxR����@�
=��=q�Q�C	�f                                    Bx]��  	�          @�Q���
=@q�����(�C���
=@�Q��\���C�                                     Bx]�l  
�          @ٙ���p�@w������HC(���p�@��
����p�C��                                    Bx]�  T          @أ����H@~�R�   ����C����H@�����]p�C
��                                    Bx]��   `          @�  ����@s33��ff�x(�CL�����@�{��z��{C�\                                   Bx]^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx](�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]7P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]E�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]T�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]cB              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]q�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�r              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]d              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]!�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]0V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]>�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]M�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]\H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]j�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�x              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�j              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx])\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]F�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]UN              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]c�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]r�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�p              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]"b              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]1              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]NT              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]\�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]k�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]zF              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]Ä              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx]h  a          A ����p��u�����]G�C@L���p�?˅����W(�C B�                                    Bx]*  c          A��_\)�����Å�C\)Ce=q�_\)��
=��{�y\)CJ@                                     Bx]8�  �          A�
��  ��33���R�0{Cc^���  ��{���f�CL��                                    Bx]GZ  	�          A
=��  �h������;C\����  �������fz�CBu�                                    Bx]V   "          A����
�G��ƸR�D��CW�{���
�����fffC:��                                    Bx]d�  "          Az��z=q��\)�����i\)CM��z=q?p����R�s\)C&�                                     Bx]sL  �          A������33��p��^ffCM�)����?:�H���kz�C*G�                                    Bx]��  "          A{����+���=q�Z
=CUn���=�G���\�t  C2p�                                    Bx]��  T          A����{����G��Z��CJ����{?aG���\)�dG�C(��                                    Bx]�>  �          AG���(���
=����_G�CI  ��(�?��������eQ�C&J=                                    Bx]��  
�          A�������\�����P�CK�����?#�
���\�RC,W
                                    Bx]��  T          A����Ϳ�����{�E�
CH�=����?#�
��ff�Q  C,�                                    Bx]�0  
�          AQ����R������(��B(�CB0����R?����p��D
=C(+�                                    Bx]��  �          A�\��p���{��
=���HC6�
��p�?���������C)xR                                    Bx]�|  �          Az���\�(���C33��
=C9���\?���E���ffC/�H                                    Bx]�"  "          @���ۅ?�z��5���{C(aH�ۅ@Q�����  C �)                                    Bx]�  
�          @�(���(���ff����!�C>)��(�?s33���\�!�
C*�=                                    Bx]n  �          @�p���녿�����BG�CBٚ���?�G�����E(�C(��                                    Bx]#  �          @�{��ff��G����R�0{CCp���ff?.{��(��6�RC,ٚ                                    Bx]1�  T          A=q��Q���H����-ffCF�\��Q�>�p�����9�\C0W
                                    Bx]@`  �          A Q���녿����33���C@h����>������\)C/�=                                    Bx]O  �          A z��ڏ\�\(���(����C;33�ڏ\?E��������HC-��                                    Bx]]�  �          A ���љ���������RC?G��љ�>����=q�z�C/��                                    Bx]lR  �          A����z�H��Q��*
=C=W
��?�Q����R�(ffC(�H                                    Bx]z�  �          A ������G���(��2�C5)��@�\��  �$
=C B�                                    Bx]��  "          @�����(�?\������C%���(�@N{�u��ffCL�                                    Bx]�D  "          @����ff>��H��G��$C.޸��ff@G�������C\)                                    Bx]��  
�          @�z���ff�\�����5��C8����ff?�(���=q�+\)C"�f                                    Bx]��  
�          @�33��33�!G���(���CP�3��33�(����Q��<��C<33                                    Bx]�6  T          @�=q��녿J=q��\)�X�C?����?�����33�Q��C c�                                    Bx]��  
�          @�\)���ÿO\)���
�@�C>
����?�����G��<G�C%+�                                    Bx]�  "          @����R�����
�@
=CB�����R?\(����C{C)�                                    Bx]�(  �          @�p����� ����z��5  CL}q���    ��G��H��C4                                      Bx]��  "          @�{���R����z��(�CJ33���R�W
=���\�0�C6^�                                    Bx]t  "          @��
��
=������Q��0z�CC�q��
=?\)��p��7�C-=q                                    Bx]  �          @�(����R�����aG���
=C^���R�Q������-�CP�                                    Bx]*�  �          @�\��G���p�����o�
Cc:���G���=q�mp�����C\^�                                    Bx]9f  �          @陚�������R��
���
Ca�������mp������  CY��                                    Bx]H  �          @���r�\��ff�|(����Ce�)�r�\�'����\�D�CV�3                                    Bx]V�  T          @���P  ��33���ffCj�q�P  �*=q����TC[T{                                    Bx]eX  �          @�����G����׿�z���C`����G���{�E���  C[�R                                    Bx]s�  �          @�z���\)��z��Q��l(�Cf���\)��\)�|(����C_p�                                    Bx]��  
�          @���Z=q���R�333��=qCm���Z=q�~�R����&��Ceh�                                    Bx]�J  �          @��H�xQ���
=��Q��Up�Cl�R�xQ���(��u���z�Cg��                                    Bx]��  �          @�\�C�
��
=��=q�-p�Cs�C�
��  �e��\Co��                                    Bx]��  
�          @�G��N�R����\)�(�CpW
�N�R�����=q��G�Cm�)                                    Bx]�<  
�          @ۅ�l����\)>��@p�Cm0��l�����Ϳ�
=���Ck�H                                    Bx]��  �          @�{�c33��Q�?xQ�A�\CnT{�c33��ff����-�Cn�                                    Bx]ڈ  	�          @ڏ\�c�
���?�z�Ac�
Cm
=�c�
���R������Cn\                                    Bx]�.  "          @����\(�����@0  A�(�Cm��\(���ff?�@�Co�R                                    Bx]��  T          @���R�\�U@��HB7�Cac��R�\��@Mp�A�p�Cl8R                                    Bx]z  
(          @أ��aG��x��@�z�B33Cc�f�aG���p�@
=A��Ck�=                                    Bx]   �          @�p��XQ��U�0  ��{C`��XQ��33�s�
�/�RCS@                                     Bx]#�  �          @�녿��
=L����{B�C2=q���
@+������}=qB�\                                    Bx]2l  
�          @�=q��\�(���z�8RCB�q��\@�����H�qC�q                                    Bx]A  �          @�G���33���H��\)�{CBT{��33@����33�C ��                                    Bx]O�  T          @��
��\)�@  ��z��CL���\)@����B�B�                                    Bx]^^  T          @�p��}p��
=��¢��CR���}p�@z���Rp�B�Q�                                    Bx]m  T          @�����XQ��%��Cl�����	���j�H�HQ�C`Q�                                    Bx]{�  �          @�  �^{��?��HARffCl0��^{���H�(���  Cm                                    Bx]�P  
�          @�p��.�R����@�
A���Cu!H�.�R�ʏ\��Q�@  Cv��                                    Bx]��  �          @�z��\����
=?�ffAP��Cn���\����(��:�H��z�Co�)                                    Bx]��  T          @�(��r�\����?�A?33Ck���r�\����G�����Cl33                                    Bx]�B  �          @޸R�}p����?�Q�ABffCi���}p���  �333��z�CjB�                                    Bx]��  �          @�p��������?��@�(�Ck�������G���\)�T��Cj�                                    Bx]ӎ  
�          @����R��33@��A��Cg�\���R��Q�#�
��33Ciٚ                                    Bx]�4  T          @���ff��(�?��A	G�Ci@ ��ff�����33���Ci+�                                    Bx]��  T          @�  ������H?8Q�@�  CfǮ�����
=�����3\)Cf!H                                    