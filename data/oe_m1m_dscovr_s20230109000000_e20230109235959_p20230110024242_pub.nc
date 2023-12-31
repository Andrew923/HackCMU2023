CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230109000000_e20230109235959_p20230110024242_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-10T02:42:42.920Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-09T00:00:00.000Z   time_coverage_end         2023-01-09T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill         �   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxY=:�  T          @���@�H�U�����ĸRC��q@�H�&ff�333�33C�s3                                    BxY=I&  �          @��@G��J=q��
=���HC�ٚ@G����333� G�C���                                    BxY=W�  T          @�z�?�p��AG�� �����C��=?�p�����5��-
=C��)                                    BxY=fr  
�          @���@
=�<������RC���@
=�
=�G
=�8�HC��                                    BxY=u  
w          @�G�@��Tz��,�����C��\@���e��@��C�}q                                    BxY=��  7          @���@��@���W
=�)�C���@��������
�^p�C�o\                                    BxY=�d  T          @�  ?���#33�x���MffC�}q?����  ��\)�C�t{                                    BxY=�
  T          @��
@
�H����G
=�/��C��@
�H��=q�l(��]��C�.                                    BxY=��  T          @�(�@\)��  �R�\�DC��q@\)�+��j�H�e�HC��                                     BxY=�V  
�          @�z�@,�Ϳ���p���33C�#�@,�Ϳ�(������
C��
                                    BxY=��  
�          @�z�@j�H�޸R@{A�G�C�N@j�H��?˅A���C��=                                    BxY=ۢ  
Z          @�{@[���33@  A�z�C��@[����?��A���C�T{                                    BxY=�H  �          @���@E��33@{B��C��{@E�z�?�{A��
C��\                                    BxY=��  T          @�{@J=q���@(�A���C���@J=q�Q�?�\)A��RC��)                                    BxY>�  
Z          @~�R?��ÿ
=@g
=B�=qC��3?��ÿ�33@P��Bd(�C�Y�                                    BxY>:  "          @z=q>aG��L��@qG�B�C�3>aG���
=@e�B��C�e                                    BxY>$�  
(          @�Q�?�
=��=q@n{B���C��H?�
=����@]p�By=qC�S3                                    BxY>3�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY>B,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY>P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY>_x  O          @���?�ff�Y��@�Q�B���C�aH?�ff�@s�
Bd��C���                                   BxY>n  �          @���?�33=��
@�z�B��\@|(�?�33���R@��RB�8RC�j=                                   BxY>|�  
�          @�p�?�{��z�@~�RB��)C��?�{��(�@mp�B~ffC���                                   BxY>�j  
�          @`  ?�����=q?��A���C�O\?�����?�AIp�C�Q�                                   BxY>�  "          @{�?���(��>�R�E�
C�}q?����Q��`  �|p�C���                                    BxY>��  T          @��?0���ff�l(��l
=C�"�?0�׿fff����B�C��H                                    BxY>�\  T          @�33?�z��Z=q@��A�G�C�Z�?�z��z=q?�=qAm�C�:�                                    BxY>�  �          @�  ?�z��Mp�@9��B!G�C��R?�z��{�?�A�ffC�AH                                    BxY>Ԩ  i          @���?�{�Dz�@(Q�B��C��3?�{�mp�?˅A��C�Ff                                    BxY>�N  
�          @�(�@,(���@A�B&�
C�,�@,(��8Q�@G�AC��H                                    BxY>��  
�          @���@<(��G�@2�\B�HC�.@<(��@  ?��RA�
=C�9�                                    BxY? �  q          @��H?�=q�@^�RBF�C��?�=q�QG�@(Q�B(�C��H                                    BxY?@  
          @�{?�z��(�@W�BH\)C��?�z��U�@\)B
=qC���                                    BxY?�  
�          @�(�?����*�H@\)BQ�C�
?����Mp�?��A�G�C���                                    BxY?,�  
�          @�p�@	���q녿L���&�HC���@	���W������{C�G�                                    BxY?;2  �          @�33@8���?\)�}p��U�C��@8���#�
��=q���HC�9�                                    BxY?I�  T          @�z�@1��G���  ����C�޸@1��&ff����RC�o\                                    BxY?X~  
w          @�G�@33�   ��\�G�C��{@33���H�;��>�\C��                                    BxY?g$  	`          @�\)?����ff��=q��C���?��>�
=��p�£#�B                                    BxY?u�  
�          @�G�?�R��z�����L�C��R?�R�#�
���¥{C�]q                                    BxY?�p  
�          @�z�?:�H��33���
B�C��H?:�H<����\¢��@ff                                    BxY?�  �          @�ff?
=q��\)��G��HC��H?
=q>�{����¥#�B�R                                    BxY?��  �          @��?��Ϳ�=q�����~
=C���?��;���(��fC��                                    BxY?�b  
�          @�G�?h�ÿ�������)C��?h�þ�=q��{�C��q                                    BxY?�  
(          @�  ?B�\�����Q��C�N?B�\>.{��¢  AI                                    BxY?ͮ  
�          @��?�{��Q���G��y�
C�� ?�{�
=��p���C��                                    BxY?�T            @���?�����
����qC�)?���aG���  \)C��                                    BxY?��  "          @�\)?����p���ff33C�4{?��>W
=���G�A4z�                                    BxY?��  T          @�=q?��\���
��Q�p�C�T{?��\>�{����
A�\)                                    BxY@F  
�          @�  ?��R���R��=q��C�޸?��R�����=q� C�l�                                    BxY@�  "          @�=q?�zῆff��G��C��{?�z�>�  �����AAG�                                    BxY@%�  T          @�\)?���\(�����z�C�5�?��?z����HQ�A�                                    BxY@48  "          @���?�z�^�R��=q�\C��=?�z�?z����=qAә�                                    BxY@B�  "          @���?���p����Q��)C���?��>��H���\
=A���                                    BxY@Q�  T          @���?��H�xQ���z�{C�+�?��H>�
=��
=#�A]�                                    BxY@`*  
�          @�G�@�R������  �}�C�޸@�R?k���p��u�A�{                                    BxY@n�  S          @���@
=��
=���uG�C�ff@
=>#�
���\Q�@�                                      BxY@}v  "          @��@   �8Q����Hp�C��@   ?&ff��33  A�                                      BxY@�  �          @��@{�5��  �zz�C�+�@{?!G���Q��{��A|��                                    BxY@��  "          @��@�H�&ff�����q�C�}q@�H?&ff�����qffAr�\                                    BxY@�h  
�          @�Q�@ff�B�\��
=�}��C�R@ff?���  W
Ar�\                                    BxY@�  "          @���?�Q쾽p���(��)C��?�Q�?z�H��G��)A��                                    BxY@ƴ  "          @�ff>�(�?�������W
B�� >�(�@(������o��B��                                    BxY@�Z  
�          @�
=?W
=?����  .BM�H?W
=@Q�����l��B�\                                    BxY@�   	.          @�\)?p��?G���=q� B\)?p��@����z��w�HB��{                                    BxY@�  T          @��R?�\)>Ǯ���RG�A�=q?�\)?�p������}=qBN�                                    BxYAL  
Z          @��?�p�?���
=p�A�(�?�p�?�{���
�{�Bb��                                    BxYA�  �          @���?��
?   ��ffW
A���?��
?�������{�B[�\                                    BxYA�  
�          @���?xQ�>������B�A�  ?xQ�?�33���
�3Bn�                                    BxYA->  "          @��
?�
=��\)����� C��f?�
=?xQ���ffB�B�                                    BxYA;�  
�          @�p�?�Q쿜(��z�H���C��?�Q�c�
������C���                                    BxYAJ�  T          @��H@z��S�
@�A��C���@z��r�\?�33Ag�C��)                                    BxYAY0  �          @�G�@���2�\@>{B�C�U�@���aG�@33A�{C�q                                    BxYAg�  �          @�p�?�33�Mp�@&ffB�C�XR?�33�tz�?�=qA���C�AH                                    BxYAv|  �          @�z�?�ff����?�A���C��f?�ff���H>�ff@��C��
                                    BxYA�"  T          @�G�@�\�Vff@{A�
=C�4{@�\�z=q?�A�33C�/\                                    BxYA��  �          @�z�@Q��_\)@,(�BffC���@Q����?˅A��C���                                    BxYA�n  �          @��@
=�J�H@AG�B��C�޸@
=�x��@   A�(�C�B�                                    BxYA�  
(          @��H@   �G
=@HQ�B �C�^�@   �w�@�Aϙ�C���                                    BxYA��  T          @���?�z��8Q�@dz�B@ffC�\?�z��qG�@'
=B��C�B�                                    BxYA�`  "          @��R?���o\)@A�p�C�+�?����  ?�Q�An�HC��                                    BxYA�  �          @�{?�����?ٙ�A���C�?����\)>�@��C�/\                                    BxYA�  T          @��?�{�<(�@L��B*z�C�*=?�{�n{@\)A�RC�G�                                    BxYA�R  "          @�33@G��`  @,��B�RC�H@G����
?���A�  C�{                                    BxYB�  
�          @���?��H�[�@��A�p�C�J=?��H�x��?��AfffC��                                    BxYB�  "          @���?��
�U�?�(�A܏\C�{?��
�p  ?uAP(�C��
                                    BxYB&D  
�          @��?��R�l(�?��
A�  C�+�?��R��G�?+�Az�C�q                                    BxYB4�  T          @�  @�
�\(�?��HA�p�C��\@�
�q�?.{A�C��{                                    BxYBC�  
�          @�(�?�����
?Y��A*�HC�Q�?����{�������HC�                                      BxYBR6  
�          @��H@G����\��������C�+�@G��s33�\��C���                                    BxYB`�  T          @��
?�\)�I���<���%�C��\?�\)����mp��az�C��H                                    BxYBo�  �          @��?5�����=q�q�\C��)?5�z�H�����C���                                    BxYB~(  T          @�\)?aG���  ����qC���?aG�<��
���H.?��                                    BxYB��  �          @��R?���(���8���5C��
?���޸R�`���o�C��q                                    BxYB�t  �          @��R?��qG��Q�����C�J=?��>{�U��5�C�                                    BxYB�  �          @�\)?�G���(��z���C��?�G��Y���HQ��%��C�N                                    BxYB��  �          @��?������:�H��
C�C�?�����녿�
=��=qC���                                    BxYB�f  �          @�G�?�(�����<#�
=�G�C�Y�?�(����
���e�C��f                                    BxYB�  �          @�(�@����ff?Q�A33C�"�@�����׾�p���z�C���                                    BxYB�  �          @��?�p����>�Q�@��\C��f?�p���p��aG��(z�C��                                    BxYB�X  "          @��@�
���\?0��A33C��{@�
�������C��                                    BxYC�  �          @��?�{����>�(�@��
C�}q?�{��33�J=q��
C��)                                    BxYC�  T          @�  ?�(����\=u?E�C�c�?�(�����\)�c�C��3                                    BxYCJ  �          @���?�\��\)���ǮC��?�\���������C�T{                                    BxYC-�  �          @�  ?^�R��=q�&ff���C�e?^�R��
=��\)��33C��3                                    BxYC<�  
(          @���>���J�H@e�BAC�Q�>������@%B�HC��3                                    BxYCK<  "          @�
=?��AG�@e�BE��C��?��xQ�@(Q�B\)C��R                                    BxYCY�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYCh�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYCw.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYC��  H          @��=����=p�@r�\BP(�C��{=����w�@7
=B�C���                                   BxYC�z  
Z          @�=q?=p���@�p�Bp��C���?=p��Tz�@Z=qB4\)C�@                                    BxYC�   T          @�=q>�{�#33@��Bf�RC��>�{�c33@N�RB(�HC��q                                   BxYC��  
�          @���>u�7
=@r�\BSffC�` >u�qG�@8Q�Bp�C��                                    BxYC�l  
�          @���>.{�Mp�@`��B=�HC�~�>.{����@!G�A��
C�0�                                    BxYC�  "          @�33>�p��@�{BrG�C�}q>�p��W�@Z�HB4�HC�                                      BxYCݸ  T          @��>��R�ff@�Bq��C�>��R�XQ�@Y��B4(�C��                                     BxYC�^  
�          @��\>�
=�,(�@\)B^�C�g�>�
=�i��@HQ�B!�\C�AH                                    BxYC�  �          @�33>��
�A�@qG�BL=qC�>��
�z�H@5�B��C�]q                                    BxYD	�  �          @�G�>�
=�O\)@b�\B={C���>�
=���@#33B 
=C��                                    BxYDP  T          @�\)������@�A�33C������  ?�p�Ar�RC��\                                    BxYD&�  �          @�
=�}p����R?(��A(�C���}p���\)�
=q��C��H                                    BxYD5�  "          @�  ��  ���
>��R@q�C�����  ��G��k��2=qC��                                    BxYDDB  "          @��׿\��(�?��AMG�Cz녿\��  ������C{h�                                    BxYDR�  T          @�
=������?�{A��RC|�Ϳ����ff?8Q�Az�C}��                                    BxYDa�  T          @���0�����R?�33A�
=C�W
�0�����>��H@�z�C��H                                    BxYDp4  
�          @����
���H?�p�A��C�)���
��
=?W
=A&�\C��H                                    BxYD~�  	�          @���������?��HA���CG�������?W
=A'\)C�<)                                    BxYD��  "          @�(��k�����?�G�A�ffC���k�����>�33@�G�C�@                                     BxYD�&  
�          @��
�W
=��G�?\A�C�|)�W
=��G�>�Q�@�C�Ǯ                                    BxYD��  T          @�\)�����&ff@h��BM
=Cs\�����]p�@5B
=Cy\                                    BxYD�r  �          @�{�˅�2�\@W
=B9�Cpc׿˅�dz�@!G�B�Cv�                                    BxYD�  "          @�{��{�Dz�@Mp�B/  Cv0���{�s33@�
A�RCzY�                                    BxYD־  �          @�ff��{�W�@C�
B#(�C{�Ϳ�{����@A��C~�                                    BxYD�d  �          @�ff��33�~�R@p�A��\C}�{��33���?�{A[\)CY�                                    BxYD�
  �          @��
��p��z=q@�A׮C|����p����?�G�AK�C~{                                    BxYE�  "          @���c�
�mp�@*=qBz�C�Ff�c�
����?�\)A�33C�                                      BxYEV  "          @������Z=q@4z�B�HCx���������?�{A���C{�                                    BxYE�  "          @���   ���?��HA��\C����   ����?��@���C��                                    BxYE.�  "          @�G��
=q��(�?�A�\)C�|)�
=q��ff?+�A ��C���                                    BxYE=H  �          @�
=�!G����R?��RA�Q�C��R�!G���=q?\(�A(  C�                                    BxYEK�  T          @���
=�z�H@%B�\C��)�
=��{?�G�A�
=C�<)                                    BxYEZ�  
(          @�
=>���E@W
=B=(�C�k�>���u@{B�RC�%                                    BxYEi:  
Z          @��R����G�@��Bs��C��׾���N�R@Y��B9�\C���                                    BxYEw�  "          @��R��=q�1G�@qG�BV\)C�:ᾊ=q�hQ�@<��BQ�C��                                    BxYE��  
Z          @�
=?&ff�7
=@a�BI\)C�ff?&ff�i��@-p�BffC�
=                                    BxYE�,  
�          @��>.{��R@��Bw�C�&f>.{�L��@^{B=G�C��                                     BxYE��  	�          @�p���  ��33@��B���C�B���  �8Q�@c�
BK��C��f                                    BxYE�x  �          @��R����@�B{ffC��R����Dz�@a�BB�C��                                    BxYE�  "          @�{��G��@�G�BoffC��f��G��QG�@U�B5�C��                                    BxYE��  �          @�
=�����8��@i��BMC��q�����mp�@4z�B��C�t{                                    BxYE�j  �          @���#�
�P��@J=qB.  C�p��#�
�|(�@  A�G�C�aH                                    BxYE�  
�          @�=q���`  @6ffB��C�˅�����H?�33A�(�C�ff                                    BxYE��  
(          @�G����R�j=q@$z�B�RC��R���R��?˅A��C���                                    BxYF
\  "          @�����X��@Dz�B'��C�+�����G�@��A�ffC�Ǯ                                    BxYF  �          @�z����R@n{BaG�C�����S�
@@  B(��C�                                    BxYF'�  �          @�=q�.{�.�R@h��BT33C�.�.{�b�\@7
=BC��R                                    BxYF6N  5          @�\)���H�@eBW(�Cr�R���H�H��@:=qB#�\Cy�                                    BxYFD�  T          @�=q��R���@�{B���Cy\)��R�'
=@j=qBWG�C�W
                                    BxYFS�  �          @��R>.{��{@��B���C��\>.{�(Q�@xQ�B_Q�C���                                    BxYFb@  
�          @�\)>k����@���Bw�\C�  >k��H��@_\)B?��C�)                                    BxYFp�  T          @�ff>�{�5@n{BQ�HC�t{>�{�i��@;�B\)C���                                    BxYF�  "          @��
?��
��Q�@���B���C�O\?��
��=q@vffB���C��                                    BxYF�2  
�          @��?��>��R@�ffB���A��?���Tz�@�(�B��C��                                    BxYF��  
�          @��?���>��H@�p�B�#�A�  ?��Ϳ#�
@���B�k�C�Ф                                    BxYF�~  
�          @�p�?��׿���@�  B�\)C��=?�����@w�Bj�C�>�                                    BxYF�$  
m          @�{?Tz��5@g�BJC�'�?Tz��g�@5Bp�C�w
                                    BxYF��  �          @�
=?:�H�(�@��Br�C�9�?:�H�Fff@^{B=C��                                     BxYF�p  T          @��
?xQ����@uB`  C��?xQ��O\)@J=qB,=qC�W
                                    BxYF�  	�          @�p�?B�\�8��@e�BH��C�\)?B�\�i��@2�\B\)C�޸                                    BxYF��  �          @�
=?k����@x��B`33C�K�?k��S33@L��B,\)C���                                    BxYGb  
�          @��?G�����@��
B�L�C�O\?G��
=@�Bl�
C�                                      BxYG  
�          @��?@  ��  @�  B��C�XR?@  � ��@���Bc��C�Z�                                    BxYG �  "          @�=�\)��z�@��
B�W
C�8R=�\)�(��@w
=B^�C��                                    BxYG/T  T          @��H    �G�@��B�8RC�%    ��@�B�G�C�\                                    BxYG=�  �          @��>�ff��G�@�Q�B�� C�AH>�ff�p�@r�\Bb��C�'�                                    BxYGL�  �          @�G�?z�H�Z�H@33B��C�?z�H�w
=?���A��C�%                                    BxYG[F  �          @���?����@w�B�C��?���z�@aG�B_�
C�K�                                    BxYGi�  T          @��R?O\)�.{@��B�\)C��\?O\)��z�@�G�B��HC��                                    BxYGx�  �          @�33>��R���
@�B�C�o\>��R���H@u�Bz(�C���                                    BxYG�8  
�          @��
?�����@o\)Br�HC���?���&ff@N�RBC�C�U�                                    BxYG��  �          @�z�?�33�&ff@1�B#=qC�!H?�33�J=q@
=A�ffC���                                    BxYG��  �          @��@�ff@7
=B(�HC��@�,��@33B33C�}q                                    BxYG�*  "          @��H?�(���Q�@~{B���C�Ф?�(���
@fffBaz�C�G�                                    BxYG��  �          @�?��
��Q�@a�BUp�C�E?��
�,(�@@  B,
=C���                                    BxYG�v  �          @�(�?\(���p�@�  B��3C�f?\(��ff@c�
B[��C��                                    BxYG�  
�          @�p�?.{����@|(�B~�RC�J=?.{�*=q@[�BM
=C�5�                                    BxYG��  
�          @�ff?fff��@s�
Bn��C��R?fff�5@P��B>ffC��3                                    BxYG�h  "          @�ff?333�<(�@P  B<Q�C��q?333�e@ ��B	��C���                                    BxYH  T          @�
=?&ff�2�\@XQ�BF�C��R?&ff�^{@*�HBz�C�T{                                    BxYH�  T          @��R?���z�@n{Bd��C�>�?���E@G
=B2z�C�~�                                    BxYH(Z  �          @�p�    �%�@aG�BW
=C�H    �R�\@7
=B#�
C�                                      BxYH7   �          @�
=?�{�˅@z�HBwp�C�G�?�{��H@^{BMffC��3                                    BxYHE�  �          @���?p�׿�(�@~{Bt�HC�?p���2�\@\(�BE��C�K�                                    BxYHTL  �          @��\?c�
��@~�RBsp�C��{?c�
�6ff@\(�BC�
C���                                    BxYHb�  �          @�33>�
=?O\)@�
=B��HB{�
>�
=�8Q�@�G�B��{C�q                                    BxYHq�  �          @�(�>��=�Q�@��HB�33A�\)>���h��@��B�=qC��)                                    BxYH�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYH��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYH��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYH�0  y          @��?.{��@��
B�B�C��?.{��ff@�z�B���C��)                                   BxYH��  �          @��>Ǯ�(��@�B�� C�Z�>Ǯ���@�B�=qC���                                   BxYH�|  T          @�ff>�׿E�@��HB���C���>�׿�p�@�=qB��HC���                                   BxYH�"  �          @�ff>�G���{@���B�p�C���>�G��33@�B}��C��                                    BxYH��  �          @�ff?\)�&ff@��B��C�z�?\)��{@��
B�Q�C���                                    BxYH�n  T          @�
=>��H�0��@�(�B�Q�C���>��H��33@�(�B�C�:�                                    BxYI  �          @��?���@  @���B���C���?���ٙ�@�G�B��C�Q�                                    BxYI�  �          @���?��H�@  @�G�B�.C�(�?��H��Q�@�G�B�B�C�ٚ                                    BxYI!`  �          @�Q�?k��O\)@��HB�8RC�` ?k���  @�=qB��)C��
                                    BxYI0  �          @�p�?(��?�Q�@�Q�B���B�k�?(��?E�@���B���BE                                      BxYI>�  T          @��;�{?���@�p�B���B�Q쾮{>�p�@�33B�� C+�                                    BxYIMR  �          @�  >���?z�@�B�(�Bxz�>������@�ffB���C�^�                                    BxYI[�  "          @�33>�>�p�@���B���B{>��!G�@���B�z�C��                                    BxYIj�  T          @��>�=q�Ǯ@���B��3C�h�>�=q��{@�33B�W
C��R                                    BxYIyD  "          @�=q>\��\)@���B��)C���>\���@�B�{C���                                    BxYI��  "          @�33>���33@���B��
C��>�����@��B�{C���                                    BxYI��  T          @�33>Ǯ��  @��B���C��>Ǯ�
�H@���B{p�C�R                                    BxYI�6  �          @�(�>�p���p�@�{B�� C�\)>�p��	��@�33B}G�C�ٚ                                    BxYI��  T          @���>���	��@���B}
=C���>���>{@p  BNp�C�q�                                    BxYI  T          @��=�\)��@�p�B��{C�
==�\)�.{@l��BV�C��)                                    BxYI�(  T          @�=q��녿�ff@��B���C33�����@uBg�C��                                    BxYI��  T          @��H<��G�@�\)B�=qC�H<���z�@�\)B�(�C�xR                                    BxYI�t  �          @�z�#�
�u@�(�B��C��)�#�
��z�@��B�  C�{                                    BxYI�  "          @��
=u?#�
@���B���B���=u��=q@��B�C���                                    BxYJ�  �          @����.{?��@�\)B�#�B��H�.{>k�@��B�=qC!                                      BxYJf  �          @�{=�G�=�Q�@�p�B�aHB\)=�G��Tz�@��HB�W
C��                                    BxYJ)  T          @�
=>�  ?=p�@���B��)B�Q�>�  �B�\@��RB�\)C��)                                    BxYJ7�  "          @���=u?�{@�Q�B�Q�B�u�=u>W
=@�z�B��B�Ǯ                                    BxYJFX  T          @�=q<#�
?��
@�{B���B�.<#�
>\)@��B�p�B�\                                    BxYJT�  "          @��\�n{?�@�z�B�z�B�\�n{?Tz�@�(�B�=qC	�q                                    BxYJc�  �          @�(���G�?�z�@�33B�  C.��G�>���@��B��C#��                                    BxYJrJ  T          @��ÿu?Q�@��\B�u�CQ�u��@��B�u�C5��                                    BxYJ��  �          @�  ��?W
=@�z�B��B����#�
@�
=B��)C5��                                    BxYJ��  "          @�z�?:�H?Tz�@�  B�Q�BB��?:�H�#�
@�=qB�ǮC�W
                                    BxYJ�<  �          @�{?\�aG�@�Q�B�{C��?\��
=@�Q�Br(�C�
                                    BxYJ��  "          @��@   ��p�@y��Be��C���@   ���@c33BHQ�C�/\                                    BxYJ��  T          @�z�?��&ff@�B���C�#�?���@~�RBo�
C��                                     BxYJ�.  T          @�p�@G���@w�Bb�C�Z�@G����@eBJp�C�0�                                    BxYJ��  �          @���@  ���@u�B^��C��@  ��@aG�BD�HC�                                      BxYJ�z  �          @��@����@o\)B\�
C��)@��޸R@^�RBG{C��f                                    BxYJ�   "          @���@C�
���@hQ�BG�C���@C�
�Tz�@b�\B@C�o\                                    BxYK�  �          @�33@`��?��H@@��B��A���@`��?aG�@N�RB'�Aa�                                    BxYKl  �          @���@i��?��
@.{B{A��@i��?�  @@  B\)A��                                    BxYK"  �          @�33@Vff?�G�@Tz�B.33A��H@Vff>Ǯ@\��B6�H@Ӆ                                    BxYK0�  �          @�{@aG�?��H@I��BffA�{@aG�?\(�@W
=B+�\A[33                                    BxYK?^  �          @���@_\)?k�@S�
B)�Al��@_\)>��R@Z�HB1(�@�Q�                                    BxYKN  �          @��\@;�?���@Z�HB6��A�@;�?u@j=qBG��A��\                                    BxYK\�  �          @�z�@7�?��@`��B8�A�@7�?���@qG�BJ�A��                                    BxYKkP  
�          @��@?\)?�(�@fffB<��A�\)@?\)?L��@s�
BKp�Ao�                                    BxYKy�  T          @��\@7
=?�33@mp�BH�HA�p�@7
=>��@vffBT{A                                    BxYK��  T          @�33@�����@y��Bj�RC���@��L��@tz�Bc�C��R                                    BxYK�B  "          @�33@��>��@���Br�A@Q�@�;�  @�G�Bt�C�                                    BxYK��  �          @��@�\=u@s�
Bl�?\@�\�(�@p��BgG�C��H                                    BxYK��  �          @��
@@  ?�\)@^{B5�A��@@  ?}p�@l��BF
=A�                                    BxYK�4  T          @�33@:�H?��@`  B933A���@:�H?�  @n�RBI��A�                                      BxYK��  �          @�(�@.�R?��@s33BLA�@.�R?333@\)B[
=Ag�                                    BxYK��  �          @��H@��?�z�@���Bb�RA�\)@��?.{@�\)Bs\)A��H                                    BxYK�&  �          @�(�@��?�33@�(�Bh��B��@��?+�@��Bz{A�                                      BxYK��  
�          @��@33?�\)@��Bu�A�\)@33>�p�@�{B���A"{                                    BxYLr  
�          @���@��?�@�Bj�RA��H@��>�G�@��Bw��A/\)                                    BxYL  T          @�(�?�\)?�G�@�z�B�W
A��
?�\)>�  @�  B��@�{                                    BxYL)�  
�          @�z�?��R>�(�@�{B��AC�?��R����@�ffB�.C�Z�                                    BxYL8d  T          @���@z�>�=q@�B�k�@��@z��@��B�u�C�|)                                    BxYLG
  �          @�(�?�33>��@��RB�z�@��H?�33���H@�{B�W
C���                                    BxYLU�  T          @��H?�p�>\@���B���A-?�p���Q�@���B��HC��R                                    BxYLdV  
�          @�=q@�>B�\@�z�Bn�@�\)@��   @��
Bk�RC�/\                                    BxYLr�  T          @��H@p��L��@���Bz�
C�g�@p��h��@��RBq��C���                                    BxYL��  �          @�=q@�
=q@���B}�C��q@���R@��Bn��C��H                                    BxYL�H  �          @���?��W
=@�G�B��C�.?����
@��HBl(�C��
                                    BxYL��  �          @���@���Q�@�p�BnG�C��@����
@��Bd  C�w
                                    BxYL��  T          @���@,(�=�\)@��HBb��?\@,(��
=@���B_
=C��=                                    BxYL�:  
�          @��@!논#�
@�z�Bjz�C��@!녿.{@��HBez�C�|)                                    BxYL��  �          @�Q�?��H���
@���B�k�C��)?��H��(�@�z�Bs\)C��                                    BxYLن  �          @��?Y�����H@��B�8RC�=q?Y����@~�RBf�\C���                                    BxYL�,  �          @��?�  ��=q@�By�
C��R?�  ��R@xQ�B](�C��R                                    BxYL��  �          @�\)?�{����@�{B~Q�C�XR?�{���H@{�Bdp�C���                                    BxYMx  T          @�\)?�  �k�@���B�aHC��?�  ��=q@��\Bo��C��                                    BxYM  T          @��?E���@y��Bq\)C�#�?E��*�H@a�BO�C��                                    BxYM"�  �          @�(�?�������@�  Bs
=C�P�?������H@q�B^33C�h�                                    BxYM1j  �          @���?Y����  @��B��C�f?Y����33@\)Bz{C��                                    BxYM@  
�          @���?5��
=@�
=B���C�:�?5��
@|��Btp�C���                                    BxYMN�  �          @�ff?�\)>aG�@�  B�(�@�G�?�\)��(�@��B�33C���                                    BxYM]\  �          @�  ?�
=�8Q�@�B�W
C��=?�
=�Y��@�33B�=qC�~�                                    BxYMl  "          @�
=?��>��@�z�B���@��?�녿�@��B��fC�:�                                    BxYMz�  T          @�Q�?��H<�@�B�?�=q?��H�#�
@�z�B���C���                                    BxYM�N  T          @�=q?z�H��R@�33B��{C��q?z�H��ff@�ffB��3C��                                    BxYM��  �          @��?�G����@�
=B���C���?�G���  @��\B�aHC�aH                                    BxYM��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYM�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYM��   v          @��?�z�?�
=@���B�B6G�?�z�?�\@�z�B��=A��R                                   BxYMҌ  �          @�33@�?5@��\B~\)A�(�@�=�\)@�z�B�Q�?�(�                                   BxYM�2  T          @��?���>�  @�
=B�Q�@���?��þ���@�ffB���C�                                   BxYM��  T          @��?�z�    @�Q�B�\C��\?�z�(��@�
=B�B�C��=                                   BxYM�~  �          @�
=?�=���@��B�Q�@;�?��\)@���B�aHC��                                   BxYN$  �          @�p�@   �   @��RB�z�C���@   ����@�33Bx��C�>�                                    BxYN�  �          @�z�@z�:�H@��B}=qC�P�@zΎ�@�
=Bn�RC���                                    BxYN*p  �          @��H@����\@�ffBp�\C�p�@�����@�Q�B_�HC��{                                    BxYN9  �          @��R@{����@s�
B[Q�C�O\@{�˅@hQ�BL
=C��
                                    BxYNG�  
�          @�{?�z�&ff@��RB��C��q?�z῝p�@��\Bs��C��
                                    BxYNVb  "          @�Q�@(��k�@|(�Bb  C���@(����H@q�BT{C��
                                    BxYNe  T          @��\@$z���
@g�BD�C���@$z����@UB0�C�W
                                    BxYNs�  "          @��H@G���R@7�BG�C�<)@G��&ff@"�\B   C�R                                    BxYN�T  T          @�(�@>{�(Q�@5�B�C�7
@>{�?\)@��A��C�`                                     BxYN��  �          @��\@(Q��2�\@:=qB�RC��\@(Q��J=q@ ��A��HC���                                    BxYN��  
(          @�  @*�H�1G�@1�B=qC��@*�H�G�@Q�A��\C�>�                                    BxYN�F  
�          @���@'
=�AG�@�A��
C�p�@'
=�S�
?���A�ffC�&f                                    BxYN��  
�          @��\@���J=q@p�A��HC��\@���[�?�\A�p�C�u�                                    BxYN˒  
�          @��H@5��p�@)��B�HC���@5��2�\@33A�z�C���                                    BxYN�8  "          @��
@,(��0��@#33B�C�
@,(��Dz�@
�HA�C���                                    BxYN��  �          @�@5��=q@a�BE(�C��{@5���
@W
=B8��C���                                    BxYN��  
�          @�p�@{��xQ�@
�HA�C�@{���  @G�A�G�C�5�                                    BxYO*  "          @�{@N{�\(�@N{B0{C��@N{���\@EB&�
C�H�                                    BxYO�  T          @�@����H@h��BP\)C�w
@���@[�B?�\C�ٚ                                    BxYO#v  �          @�{@;��@  @b�\BE�C�˅@;�����@Z�HB<�C��                                    BxYO2  �          @���@�  ��{>�z�@X��C��@�  ���=L��?��C��f                                    BxYO@�  �          @��R@��
��������l(�C��=@��
��p��\)����C�5�                                    BxYOOh  �          @�Q�@�ff������C���@�ff���Ϳ:�H��C�3                                    BxYO^  �          @��@��Ϳ��ÿk���C��=@��Ϳٙ�����F=qC�t{                                    BxYOl�  T          @��H@��׿��R����hz�C��=@��׿�������G�C���                                    BxYO{Z  �          @��@�녿�녿�\)�mG�C�J=@�녿��H�˅��ffC�=q                                    BxYO�   	�          @���@���ٙ���z���p�C�� @�����H�ff���C�:�                                    BxYO��  �          @���@�p��\(���ff��
=C�'�@�p��L�Ϳ����\C��                                    BxYO�L  �          @���?��׿��@��B�p�C�f?��׿��@��RB��C�aH                                    BxYO��  
�          @�z�?��H���@��
B�B�C���?��H���@���B�ǮC��q                                    BxYOĘ  T          @���@E�����@N{B8  C�Q�@E��G�@I��B2�C��                                    BxYO�>  T          @�33@�
��
=@j=qBe�C�ٚ@�
�Y��@eB^ffC��                                    BxYO��  "          @�\)>�����@�G�B���C�^�>���\)@�(�B��C��                                    BxYO��  �          @�ff>��Ϳ�p�@�p�B���C���>��Ϳ�p�@�
=B���C���                                    BxYO�0  "          @�ff?����@�33B�ǮC�޸?��Q�@�(�Bx{C��                                    BxYP�  �          @���?O\)�ٙ�@��B�.C���?O\)�(�@�z�Br�C�(�                                    BxYP|  �          @�=q?������@��B(�C���?����@�=qBg��C���                                    BxYP+"  "          @�
=?�=q���H@�33B��C�8R?�=q��Q�@���Bw�C��f                                    BxYP9�  
�          @�(�?�{��(�@���Bx=qC��{?�{�(�@���BaQ�C�AH                                    BxYPHn  
Z          @��?�33��@{�BU  C��?�33�6ff@hQ�B?\)C��                                    BxYPW  T          @���@�\��@xQ�BQC���@�\�%@g�B>�\C�q                                    BxYPe�  �          @�(�?�p��'�@fffB>z�C��
?�p��@  @R�\B)��C���                                    BxYPt`  "          @�=q?��H�
�H@�G�Bd33C��{?��H�&ff@q�BN�C���                                    BxYP�  �          @�  ?��H�7
=@dz�BD  C�q�?��H�N�R@O\)B,��C�@                                     BxYP��  
�          @�z�?�{�J�H@G
=B(
=C���?�{�_\)@0  BG�C���                                    BxYP�R  
H          @�33?��H�`  @p�B�C��q?��H�o\)@z�A�  C�@                                     BxYP��  
�          @���?�
=�~�R?�
=A��C�Ff?�
=��z�?�  A�33C��)                                    BxYP��  T          @�Q�?����g�@33Aۙ�C���?����tz�?�z�A�G�C�(�                                    BxYP�D  "          @��?���u�?���A�z�C�J=?���\)?�A���C��{                                    BxYP��  T          @���?J=q�y��@Aݙ�C���?J=q���H?�
=A��C�z�                                    BxYP�  
�          @�G�?}p��#33@g
=BSp�C���?}p��:=q@U�B=\)C�]q                                    BxYP�6  T          @��?Y��� ��@k�BXC�aH?Y���7�@Z=qBB�\C�E                                    BxYQ�  �          @��
?�
=�S�
@=p�B (�C���?�
=�e@'
=B
G�C��                                    BxYQ�  
�          @��
?�=q�z�@p��B_�C�z�?�=q�+�@`��BJ
=C���                                    BxYQ$(  "          @�z�?��\�:�H@UB9�C��q?��\�N�R@B�\B$��C���                                    BxYQ2�  �          @��?�  �^{@5�B  C��=?�  �n�R@{B �C�C�                                    BxYQAt  �          @�G�?^�R�)��@c33BO��C�%?^�R�>�R@Q�B:33C�.                                    BxYQP  T          @��
?O\)�p�@u�Bi��C��?O\)�$z�@fffBT�C��f                                    BxYQ^�  �          @��?�ff�"�\@fffBRC�5�?�ff�7�@UB>33C��                                    BxYQmf  "          @�p�?�G��S33@EB'��C��H?�G��e�@1G�B��C��                                     BxYQ|  
z          @�p�?��H�tz�@
=A�  C�Ǯ?��H����?��RAʏ\C�Z�                                    BxYQ��  ,          @�(�?���x��@�A�33C��
?����=q?�z�A���C�t{                                    BxYQ�X  �          @�z�?�{�\)?���A�Q�C��H?�{��33?xQ�AC33C�7
                                    BxYQ��  �          @�p�?˅����?&ffAp�C�5�?˅���\>�  @HQ�C��                                    BxYQ��   �          @�33?�(���  �u�:�HC�H?�(���\)��ff���\C�\                                   BxYQ�J   d          @�{@!����@8��B#ffC�z�@!����@+�BQ�C���                                    BxYQ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYQ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYQ�<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYQ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYR�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYR.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYR+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYR:z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYRI               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYRW�   �          @��?�G���ff�^�R�,Q�C�XR?�G�������H�r=qC��                                    BxYRfl  	�          @�=q@+��b�\������  C��R@+��XQ��ff�ϮC�=q                                    BxYRu            @��
@%��c33��p���  C��@%��XQ�������
C���                                    BxYR��  
�          @��H@ ����G����H��  C�9�@ ���x�������HC���                                    BxYR�^  
�          @�ff@0���U��ff��  C���@0���HQ��&ff��C��3                                    BxYR�  �          @�{@@���0���1G���C��q@@���!��>�R���C���                                    BxYR��  
�          @��@G��.�R�{��=qC�h�@G��!��*�H��C�}q                                    BxYR�P  �          @�p�@W����=p����C��=@W���Q��E�� 
=C�l�                                    BxYR��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYRۜ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYR�B   C          @�(�@�Ϳ�p��mp��F�C���@�Ϳٙ��vff�Q  C���                                   BxYR��  �          @��H@#�
�G��X���8p�C���@#�
��G��a��B��C��R                                   BxYS�  
�          @���@;���{�Dz��*(�C���@;������K��1��C�ff                                   BxYS4  
�          @�Q�@*�H�S�
���иRC�n@*�H�I���z���z�C�!H                                   BxYS$�  T          @���@5�E������RC�]q@5�:=q��R���RC�/\                                   BxYS3�  T          @��
@N�R�E���
=��33C�1�@N�R�;��	����p�C��                                    BxYSB&  "          @���@�Q��$zῌ���O�C��3@�Q���R���
�r=qC�!H                                    BxYSP�  
�          @��
@n�R�3�
����w�C�}q@n�R�-p���p�����C��R                                    BxYS_r  �          @�33@W
=��
�����C���@W
=����&ff���C�Ǯ                                    BxYSn  �          @��@c33��=q����C���@c33���#33�33C�Ǯ                                    BxYS|�  �          @�33@_\)�7����
��C�J=@_\)�/\)��(���33C���                                    BxYS�d  
Z          @��H@hQ��:=q��33��{C���@hQ��3�
�˅��
=C�                                      BxYS�
  "          @�G�@n{�<�Ϳ��˅C��=@n{�:=q�8Q��\)C��)                                    BxYS��  �          @��@hQ��>�R����ffC�S3@hQ��;��E���C���                                    BxYS�V  �          @�G�@a��6ff��Q����HC���@a��0  ��\)��G�C��                                    BxYS��  �          @�\)@g��33��(���\)C�@g��
�H�
=�ԸRC���                                    BxYSԢ  T          @��H@���=q�����z�C��=@�����5�\)C���                                    BxYS�H  
�          @���@���Q쿆ff�Mp�C���@���\)���e�C���                                    BxYS��  �          @�  @Y���   ��(�����C��
@Y����ÿ�\)��p�C�t{                                    BxYT �  
�          @��\@?\)�H��?s33AF�HC��@?\)�L(�?B�\A=qC���                                    BxYT:  �          @�{@A��5���Q���p�C���@A��.{��{��{C��                                    BxYT�  �          @��@G
=�C�
��(���ffC���@G
=�;��	����G�C�Y�                                    BxYT,�  "          @���@H���U��p��p��C�� @H���P�׿�
=���
C��
                                    BxYT;,  "          @�Q�@�z���\�#�
��\)C���@�z��녽����C��                                    BxYTI�  
�          @�\)@z�H� ��>Ǯ@�Q�C��=@z�H�!�>u@>�RC���                                    BxYTXx  
�          @��R@��׾��?���A���C���@��׿��?��A�\)C�q                                    BxYTg  �          @��@H��?&ff@N{B4�\A9�@H��>�@P  B6��A�\                                    BxYTu�  
�          @�
=@u?
=q@*=qB	��@�{@u>Ǯ@,(�Bz�@���                                    BxYT�j  
�          @���@�p��W
=?�=qA�C��\@�p����R?���A�(�C��H                                    BxYT�  T          @�
=@�ff�\?�(�A���C��3@�ff��?���A�G�C�3                                    BxYT��  
(          @�33@�?�=q@FffB:�RB��@�?�z�@L(�BB{B33                                    BxYT�\  
�          @���@I��?˅@5�BAծ@I��?�Q�@:=qB��A��                                    BxYT�  
�          @��\@,(�@��@A�B%�\B@,(�?�p�@H��B,B33                                    BxYTͨ  "          @�  @,(�@�\@J�HB'��B!�
@,(�@Q�@Q�B/
=Bz�                                    BxYT�N  T          @���@@  @	��@A�B�B�R@@  @   @H��B$=qB�                                    BxYT��  �          @��
@'
=@0  @W
=B&ffB:  @'
=@%�@_\)B.�B2�                                    BxYT��  �          @�{@1G�@j�H@6ffA�z�BS�@1G�@aG�@A�B�BOG�                                    BxYU@  
�          @��@#�
@QG�@j=qB%Q�BO@#�
@E@s�
B-�BIp�                                    BxYU�  �          @��@'
=@:=q@|(�B4�HB@{@'
=@-p�@�=qB<�
B8Q�                                    BxYU%�  �          @��@,(�@<(�@w�B0��B>G�@,(�@0��@�Q�B8ffB6�H                                    BxYU42  
�          @��
@P��@ff@x��B033B  @P��@
�H@\)B6=qBQ�                                    BxYUB�  
�          @�Q�@g
=@  @w
=B(��B {@g
=@�@}p�B.(�A                                    BxYUQ~  
�          @���@X��@!�@|(�B+��B�@X��@ff@�G�B1��B
��                                    BxYU`$  T          @�G�@a�@�@|��B,=qB��@a�@	��@���B1��A�G�                                    BxYUn�  T          @��@��?�p�@h��B=qA���@��?���@mp�B!��A�G�                                    BxYU}p  "          @�z�@u�?�  @n{B%A�=q@u�?˅@r�\B)�A�(�                                    BxYU�  �          @��@^�R?��H@VffB)\)A���@^�R?��@Y��B,\)A�{                                    BxYU��  
�          @��\@L(�?��@n{B<��A�Q�@L(�?���@qG�B@Q�A��H                                    BxYU�b  T          @��@L(�?���@~{B<��A��@L(�?�33@�G�BA=qA�
=                                    BxYU�  "          @�  @!�@�@\)BF  B(�@!�@�@��\BL  B��                                    BxYUƮ  �          @���@���L��@2�\B�\C��f@���#�
@2�\BQ�C���                                    BxYU�T  �          @�ff@�{>aG�@'
=A���@=p�@�{=�@'
=A�\)?�=q                                    BxYU��  "          @�z�@���>��H@,(�Bff@�p�@���>Ǯ@-p�B\)@�
=                                    BxYU�  �          @�z�@�z�?xQ�@=qA�z�AQ�@�z�?^�R@��A�(�A>�R                                    BxYVF  �          @�=q@��R��@ffA�33C��@��R��@A��HC�/\                                    BxYV�  
�          @�ff@���>��H@ffA�33@�=q@���>��@
=A���@�{                                    BxYV�  T          @�p�@\��?�(�@qG�B1��A�Q�@\��?˅@u�B5(�A�                                    BxYV-8  
�          @���@S33?�G�@~�RB;{A�
=@S33?�\)@�G�B>�RAѮ                                    BxYV;�  T          @���@33@%�@��BU{BN{@33@�H@�  B[(�BG=q                                    BxYVJ�  T          @�=q@   @
=@��Bi�B:Q�@   ?���@��BoffB1G�                                    BxYVY*  
�          @��@5�?�@�Q�BU\)B{@5�?�z�@�=qBYG�A�\                                   BxYVg�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYVvv              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYV�h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYV�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYV�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYV�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW&>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW4�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYWC�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYWR0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW`�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYWo|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW~"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW�n              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW�`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYW�R              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYXD              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX-�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX<�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYXK6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYXY�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYXh�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYXw(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX�t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX�f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYXݲ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYY	�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYYJ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYY&�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYY5�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYYD<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYYR�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYYa�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYYp.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYY~�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYY�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYY�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYY�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYY�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYYָ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYY�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYY�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZP              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZ.�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZ=B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZK�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZi4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZw�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZ�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZ�r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZϾ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZ�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYZ�
  o          @�33@��
�333>�G�@�=qC��@��
�333>��@��\C��=                                    BxYZ��  	�          @��\@��H�333>�(�@�ffC���@��H�333>�ff@�\)C���                                    BxY[
V  �          @�p�@����@��>���@��\C���@����@  >�G�@���C���                                    BxY[�  �          @�z�@��+�>�@��RC���@��+�>��H@�Q�C��H                                    BxY['�  
�          @�ff@�z��(�>�p�@l(�C�J=@�z��(�>Ǯ@~{C�O\                                    BxY[6H  
�          @�\)@�ff�33��Q��A�C�B�@�ff��
���=�C�1�                                    BxY[D�  2          @\@�
=�{�fff�	G�C�ٚ@�
=��R�aG����C��                                    BxY[S�  "          @�@�z��1G��h���=qC�ٚ@�z��1녿aG����C���                                    BxY[b:  
�          @��@���,��<��
>#�
C�p�@���,��=L��?�\C�q�                                    BxY[p�  T          @�{@����)��>�?��C�+�@����(��>.{?�z�C�,�                                    BxY[�  "          @�Q�@����/\)>\@qG�C��
@����.�R>��@�z�C��q                                    BxY[�,  T          @��@��R�&ff���
�Tz�C�(�@��R�&ff��z��<��C�#�                                    BxY[��  
�          @�=q@��%�=�Q�?uC�33@��$z�>�?��C�5�                                    BxY[�x  
Z          @��
@�Q��#33=�Q�?s33C��@�Q��#33>�?�=qC���                                    BxY[�  �          @�ff@��\�%���
�L��C�� @��\�%<��
>B�\C��                                     BxY[��  
8          @��@��R�����z�C���@��R�ff��(�����C��)                                    BxY[�j  
�          @�(�@�  �#�
������C�s3@�  �$z��
=���C�k�                                    BxY[�  
�          @���@���5������Y��C���@���5��\)�:�HC���                                    BxY[��  �          @��\@����5���G����C��=@����5������=qC���                                    BxY\\  
Z          @�=q@���3�
���
�^�RC�޸@���3�
�#�
�\C��q                                    BxY\  
(          @�(�@�G���H��\��\)C�/\@�G���������C�%                                    BxY\ �  T          @�(�@�=q�녿\(��(�C��=@�=q��\�Q����C�ٚ                                    BxY\/N  "          @�Q�@�Q��N�R��p��hQ�C��f@�Q��O\)���R�C33C��                                     BxY\=�  �          @�(�@�z��G����k�C��\@�z��33�����d��C�o\                                    BxY\L�  "          @�Q�@��ÿ�
=��
=��=qC��@��ÿ��H�����
=C��R                                    BxY\[@  "          @���@������Q�����C���@�������33C�u�                                    BxY\i�  
�          @��\@���L�Ϳ�33�?\)C�H@���N{����4��C��=                                    BxY\x�  T          @���@�z��]p����
�U��C�
=@�z��_\)���H�J{C��                                    BxY\�2  �          @���@�ff�S33��z��l(�C��@�ff�U�����`��C��=                                    BxY\��  �          @���@��H�XQ��{����C�8R@��H�Z=q������RC�
                                    BxY\�~  �          @��\@��\�.{�������C���@��\�0�������C���                                    BxY\�$  T          @���@��\�0  ������(�C���@��\�2�\�����
=C��)                                    BxY\��  
Z          @���@��G���{���C��@���
����G�C��                                     BxY\�p  "          @�G�@�\)�ff�  ��(�C�,�@�\)���������C��\                                    BxY\�  �          @�Q�@�=q�#�
�
=��Q�C�ff@�=q�'
=��
��\)C�'�                                    BxY\��  �          @�z�@����	���������C���@�����������ffC�c�                                    BxY\�b  �          @�(�@�G��\)�ff�υC�7
@�G���\�33�ʸRC���                                    BxY]  �          @�p�@�
=���H�333��=qC���@�
=�G��0  ��C�7
                                    BxY]�  T          @�33@�G���\�'����C�Ф@�G���=q�%����
C�y�                                    BxY](T  �          @��\@���p����  C��3@�����\����C���                                    BxY]6�  �          @��@��
�C33����{C��q@��
�E�˅����C���                                    BxY]E�  	�          @���@����8Q�������C�/\@����;��z����RC��3                                    BxY]TF  �          @��
@l���9���$z����
C���@l���=p��   ����C���                                    BxY]b�  �          @�
=@��R��녾�z��S33C��@��R��녾�  �1�C��                                    BxY]q�  �          @�z�@��Ϳ�33>\@�Q�C�W
@��Ϳ��>�
=@�=qC�e                                    BxY]�8  
i          @��\@����ͽ�\)�Y��C��
@����ͽu�.{C��{                                    BxY]��  �          @���@mp�>�\)@�G�B=��@��H@mp�>\@���B={@���                                    BxY]��  
�          @Ǯ@S�
?��R@�z�B]�A��@S�
?�\)@��B[��A���                                    BxY]�*  �          @���@W
=?��@�{B_  A�\)@W
=?�(�@��B]�A��                                    BxY]��  �          @��H@n�R?z�H@���BR�Ak�@n�R?�{@�  BP�A���                                    BxY]�v  
�          @�(�@w
=?Y��@�  BN�AF=q@w
=?z�H@�\)BMG�Ac�                                    BxY]�  c          @�p�@���?s33@�(�BFG�AR�\@���?�=q@��BD�
An{                                    BxY]��  
Z          @�{@���?:�H@�33BD
=A Q�@���?\(�@��\BB�HA;�
                                    BxY]�h  T          @�{@�p�?   @��B933@�ff@�p�?!G�@�z�B8p�A ��                                    BxY^  �          @���@�=q>.{@�
=B1p�@�@�=q>�z�@�
=B1�@fff                                    BxY^�  �          @˅@���?\(�@��HB9�A5p�@���?z�H@�=qB8��AO33                                    BxY^!Z  
�          @���@�G�?!G�@��RB=\)A��@�G�?B�\@�{B<Q�A�
                                    BxY^0   "          @��H@��R?�@�\)B3�@��@��R?(��@��RB2�
A{                                    BxY^>�  
�          @��
@�p��8Q�@���BCG�C���@�p��#�
@���BCp�C��{                                    BxY^ML  
�          @��H@�p���  @�  B5�
C�e@�p���@�Q�B6�C�8R                                    BxY^[�  
�          @�=q@��þ#�
@�z�B={C��@��ü�@�z�B=33C��                                    BxY^j�  �          @���@��}p�@�ffB%  C��@��\(�@�
=B&G�C���                                    BxY^y>  T          @�(�@�G���33@c33B�C�� @�G����@eB�RC�q�                                    BxY^��  �          @���@�z῞�R@uB��C��H@�zῐ��@xQ�B=qC�'�                                    BxY^��  "          @Å@�(���(�@r�\Bz�C���@�(���{@u�B�C�J=                                    BxY^�0  T          @�=q@����c�
@x��B�HC�u�@����G�@z�HB!�C�(�                                    BxY^��  
�          @�\)@���   @��HB%��C�� @���\@�33B&=qC��                                     BxY^�|  �          @�(�@�(���@I��A��C��
@�(���p�@Mp�A�(�C�l�                                    BxY^�"  �          @�{@�z���@A�A�\)C��H@�z���@FffA�
=C�\                                    BxY^��  �          @θR@�\)�{@8Q�A�  C�XR@�\)�Q�@=p�A��
C���                                    BxY^�n  
�          @���@�{�$z�@+�A��HC���@�{��R@0��A���C�4{                                    BxY^�  �          @�=q@��ÿ�=q@hQ�B\)C�ٚ@��ÿ�(�@k�B�C�u�                                    BxY_�  "          @�=q@������@c33B	�HC��@����z�H@e�BG�C��f                                    BxY_`  �          @ȣ�@����z�@qG�B(�C�H�@�����@s�
BC���                                    BxY_)  �          @�
=@��׿u@p��BC�U�@��׿W
=@r�\B{C�                                    BxY_7�  �          @Ǯ@���W
=@�  B �\C���@�����
@�Q�B ��C�}q                                    BxY_FR  
Z          @�Q�@��Y��@l��B{C��@��=p�@n�RB=qC���                                    BxY_T�  �          @��
@�{<�@\)B��>���@�{>#�
@~�RBp�?�{                                    BxY_c�  
�          @�33@��>�33@�  BG�@�33@��>�@\)B�@�=q                                    BxY_rD  T          @�33@�=q>��R@r�\B�
@^{@�=q>�(�@q�BQ�@��\                                    BxY_��  �          @�Q�@�33?\(�@\��B�
Ap�@�33?xQ�@Z�HB�A,��                                    BxY_��  �          @�{@�\)?u@\��B�HA/�@�\)?�=q@Z=qBffAC33                                    BxY_�6  �          @Å@��׿&ff@@  A�p�C�y�@��׿��@AG�A��C�                                      BxY_��  T          @��@���O\)@0��A�C���@���8Q�@1�A��
C�1�                                    BxY_��  "          @�@�ff�(Q�?�z�A��\C��@�ff�#�
@   A��C�S3                                    BxY_�(  
�          @ƸR@�p���Q�?�  A`��C���@�p��}p�?У�AuG�C��                                    BxY_��  �          @��@�G��_\)@ ��A�=qC���@�G��Z�H@Q�A�
=C�=q                                    BxY_�t  T          @�(�@���(Q�@��A��C��\@���#33@!�A��
C��                                    BxY_�  
(          @��@�  �5@�A�\)C�Ǯ@�  �0��@�A��\C�)                                    BxY`�  "          @�(�@�
=�g�@\)A�
=C�>�@�
=�c33@
=A�Q�C��=                                    BxY`f  "          @ə�@�\)�xQ�@�RA�33C��@�\)�s33@
=A�33C��
                                    BxY`"  "          @�{@��
�u@
=A��C�XR@��
�qG�@\)A��C���                                    BxY`0�  �          @�
=@��\�e�?�G�Aap�C���@��\�a�?У�As�
C���                                    BxY`?X  "          @�\)@��\�q�@.{AУ�C��R@��\�l(�@6ffA��C��{                                    BxY`M�  "          @��@����tz�@z�A���C�W
@����o\)@��A���C��f                                    BxY`\�  
�          @�{@�ff�S�
@��A���C���@�ff�O\)@  A�{C�H                                    BxY`kJ  
Z          @�  @����7
=@  A�C�:�@����1�@ffA�\)C���                                    BxY`y�  
�          @��
@��H�   @(�A�
=C��3@��H��H@�A��
C�L�                                    BxY`��  
�          @�{@����R?޸RA���C��{@���
�H?�A�z�C�:�                                    BxY`�<  T          @�
=@�G��,��>��H@���C�  @�G��+�?z�@���C�3                                    BxY`��  �          @�  @���/\)�aG����C�}q@���/\)�����RC�u�                                    BxY`��  T          @Ǯ@��\�h��@	��A�\)C��H@��\�dz�@�A�33C��                                    BxY`�.  "          @ȣ�@��vff@��A�Q�C�~�@��qG�@�A��RC���                                    BxY`��  
�          @���@�{�]p�@A��\C�f@�{�X��@��A�=qC�S3                                    BxY`�z  c          @��H?0����
=@dz�B�C�ff?0�����H@n{B!  C��f                                    BxY`�   
�          @�z�?���qG�@hQ�B(33C�  ?���h��@p��B/�C�s3                                    BxY`��  "          @�{@-p��{�@XQ�B33C�K�@-p��s�
@`��BC��
                                    BxYal  �          @��R@L���u@J=qB{C��\@L���n{@R�\B33C�Z�                                    BxYa  �          @Å@fff�s�
@G
=A�p�C��\@fff�l��@O\)B �C�)                                    BxYa)�  T          @��
@�\)�g
=@#33A��C���@�\)�aG�@+�A�p�C�!H                                    BxYa8^  �          @�33@mp�����@)��AΏ\C�^�@mp��z�H@2�\A�Q�C��
                                    BxYaG  T          @�=q@h���|(�@5�A�{C�]q@h���u@=p�A��
C��q                                    BxYaU�  w          @��@N�R��\)@7�A�ffC��
@N�R���
@@��A�33C�\                                    BxYadP  
�          @��@>{����@EA�p�C�l�@>{���@O\)BQ�C�Ǯ                                    BxYar�  "          @��H@C�
�z�H@`��BC���@C�
�r�\@h��B�C�t{                                    BxYa��  T          @�{@S�
����@K�A�p�C�K�@S�
��G�@U�B  C��\                                    BxYa�B  
�          @��
@g
=�g�@W
=B
=C�w
@g
=�`  @^�RB
�RC��3                                    BxYa��  �          @�@r�\�j�H@I��A��HC��@r�\�c�
@Q�B
=C�aH                                    BxYa��  
�          @�
=@u��z=q@7�A�p�C�5�@u��s�
@@��A���C���                                    BxYa�4  �          @�\)@g
=���R@333A�ffC�O\@g
=���@<��A��C���                                    BxYa��  �          @�  @j�H��  @EA���C�Ff@j�H�x��@N�RA��HC��                                    BxYaـ  �          @ȣ�@qG����H@5AظRC�P�@qG��\)@?\)A��C��                                    BxYa�&  T          @�Q�@tz���@*�HA�{C�/\@tz����H@4z�A�  C��                                    BxYa��  
Z          @�  @������@�RA�G�C�b�@����~{@'�A���C���                                    BxYbr  �          @�  @�G���{@\)A�z�C���@�G����@��A�(�C�=q                                    BxYb  �          @�Q�@~�R��\)?���Aj=qC��\@~�R��?�(�A�G�C���                                    BxYb"�  
�          @�
=@�Q��i��@#�
A�  C���@�Q��c�
@,(�A�z�C�
                                    BxYb1d  "          @�  @�\)�>�R@J�HA�C�/\@�\)�7�@QG�A��\C���                                    BxYb@
  
�          @���@tz��p��?�=qA��\C���@tz��l(�?��HA��C��)                                    BxYbN�  T          @��@�33���H?�z�A�(�C���@�33����?�A���C���                                    BxYb]V  �          @�=q@������?�33A0��C���@����\)?�ffAHz�C��=                                    BxYbk�  T          @�G�@a���33�L�Ϳ�{C�H@a�������
�aG�C���                                    BxYbz�  T          @�=q@_\)����#�
���C��
@_\)���<#�
=�G�C���                                    BxYb�H  
�          @�G�@`������J=q��C��=@`����z��R���HC��R                                    BxYb��  
�          @���@^{���ÿ��2�RC�  @^{��=q��  �z�C���                                    BxYb��  w          @�
=@����p��(���G�C�"�@������33���C��H                                    BxYb�:  �          @�{@r�\��\)���H���C��@r�\������ff���C���                                    BxYb��  
Y          @ƸR@w����\�	�����RC��H@w�����   ���\C���                                    BxYb҆  T          @�z�@G������B�\��{C��q@G������8Q�����C��=                                    BxYb�,  �          @�\)@O\)�o\)�U���\C�q�@O\)�w
=�L(��p�C��                                    BxYb��  "          @�  ?����
=�E��\)C���?���������ٙ�C��\                                    BxYb�x  "          @��>����
?�Al��C���>����?�{A�Q�C���                                    BxYc  x          @��?W
=��p�?�ffA-p�C�g�?W
=��(�?��RAMG�C�p�                                    BxYc�  �          @��?aG���ff?˅A�(�C��?aG���z�?�\A��C��)                                    BxYc*j  "          @�ff?��R��(�?�@���C���?��R���?333@�RC�˅                                    BxYc9  
(          @�Q������{@O\)B

=C���������\@Y��B  C���                                    BxYcG�  �          @�=q�.{��z�@\��B�C��
�.{��Q�@g
=BffC���                                    BxYcV\  �          @����&ff��G�@b�\B��C��&ff���@l��B�C��3                                    BxYce  
�          @�\)�������
@=p�A�(�C�h�������Q�@G�B
=C�e                                    BxYcs�  x          @��H�����R@o\)BQ�C�������\@x��B'33C��
                                    BxYc�N  �          @��=u��33@_\)B��C�` =u��\)@h��B�C�b�                                    BxYc��  �          @�=q>8Q����H@@��A���C��>8Q����@K�B\)C�3                                    BxYc��  �          @���?ٙ����\@*�HA�=qC���?ٙ����@5�A�
=C�ٚ                                    BxYc�@  
�          @�
=@'
=����@p�A��HC�Y�@'
=��@'�AУ�C��3                                    BxYc��  
�          @�p�?5��{?n{A
=C���?5����?�\)A2{C��3                                    BxYcˌ  �          @��\@G���  ��Q�c�
C���@G���  =�Q�?fffC���                                    BxYc�2  
�          @�@�=q����?+�@���C���@�=q����?O\)A Q�C���                                    BxYc��  
�          @�  @�ff�aG�?&ff@���C���@�ff�_\)?E�@�{C��3                                    BxYc�~  	�          @���@�(���Q�>u@�
C��H@�(���  >�p�@g
=C�˅                                    BxYd$  
h          @���@�ff�~�R?���AL��C�J=@�ff�{�?���Ab{C�u�                                    BxYd�  
�          @�G�@}p��c�
@1G�AۅC��@}p��^{@8��A�C�ff                                    BxYd#p  T          @���@x���Tz�@H��A��\C��H@x���Mp�@P  B=qC�8R                                    BxYd2  
L          @���@��
�`  @#�
A���C��3@��
�Z=q@*�HAԣ�C�.                                    BxYd@�  &          @�Q�@k�����@!�A���C�+�@k��|��@*=qA��C�z�                                    BxYdOb              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYd^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYdl�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYd{T              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYd��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYd��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYd�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYd��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYdĒ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYd�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYd��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYd��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYd�*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYev              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe+              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe9�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYeHh              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYeW              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYee�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYetZ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYe�0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf$"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf2�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYfAn              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYfP              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYfm`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf�R              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf�D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf�6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYf��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYg�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYg(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYg+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYg:t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYgI              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYgW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYgff              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYgu              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYg��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYg�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYg��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYg��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYg�J  4          @�\)@�\)�K��*�H�¸RC���@�\)�N�R�'
=��C���                                    BxYg��  T          @�Q�@��H�J=q�!G���Q�C�3@��H�Mp��p����C��                                     BxYgۖ  �          @Ϯ@���E��0  ����C�'�@���H���,(���=qC��                                    BxYg�<  "          @�p�@��
�/\)�.{����C��@��
�2�\�*�H�ģ�C��
                                    BxYg��  "          @��@�G��-p��8Q���\)C���@�G��0���4z���33C���                                    BxYh�  �          @�z�@�33�?\)�8����G�C�&f@�33�C33�5����C���                                    BxYh.  �          @�(�@��׿�
=�_\)��C��)@��׿�  �]p��33C�j=                                    BxYh$�  �          @˅@���ff�\(���C�o\@���
=q�Y���Q�C�"�                                    BxYh3z  �          @�=q@�p��Q��P  ���C�O\@�p����Mp���z�C��                                    BxYhB   �          @�G�@�녿�
=�k��ffC�Y�@�녿�  �i���{C�                                    BxYhP�  T          @�(�@��ÿ�33�~�R�ffC��3@��ÿ�(��|���G�C�t{                                    BxYh_l  �          @��H@��Ϳ�\)��G��33C�Ǯ@��Ϳ�Q������{C�g�                                    BxYhn  �          @�Q�@��H��{��{�(��C�Ф@��H��
=����'Q�C�l�                                    BxYh|�  !          @�  @�{�������'�C��)@�{��33��z��&��C�9�                                    BxYh�^  �          @���@�p���=q����!  C�u�@�p���33��G�� 33C�R                                    BxYh�  T          @ȣ�@��ÿ(������9\)C��R@��ÿ.{�����8�
C��                                    BxYh��  "          @�ff@��
��������@z�C���@��
��p���33�@33C�o\                                    BxYh�P  T          @��
@��H��Q���  �>��C�|)@��H��
=��  �>G�C��                                    BxYh��  �          @�
=@�
=����w��=qC��\@�
=��Q��u���C�C�                                    BxYhԜ  
�          @�Q�@x���X���AG���\C�y�@x���[��>{��ffC�N                                    BxYh�B            @��@\(�����ff��G�C�L�@\(����\�33��z�C�5�                                   BxYh��  G          @��@c33�z�H�'
=��z�C�{@c33�|���#�
��  C��{                                    BxYi �  �          @�z�@U���{����'�C��\@U���ff��  �{C��f                                    BxYi4  	.          @��H@Vff��p���  ���C���@Vff��ff��Q���33C�z�                                    BxYi�   %          @���@�(��Z�H�(��ÅC�1�@�(��\�������  C�3                                    BxYi,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxYi;&  Y          @��
@��R���E��ffC���@��R�����C�
���C���                                    BxYiI�  "          @��
@�{�޸R�-p����HC���@�{��\�,(���\)C���                                    BxYiXr  T          @��@�\)�����C�
��C�Ǯ@�\)����B�\��{C���                                    BxYig  �          @�=q@����E�(���33C�u�@����G
=�
=q���\C�]q                                    BxYiu�  T          @���@��\�.�R�{���\C��R@��\�0  �(���Q�C�~�                                    BxYi�d  T          @�=q@����5��  ���HC��@����6ff��R����C��                                    BxYi�
  �          @��
@������p���{C��@����������Q�C��=                                    BxYi��  "          @�p�@����p��#33���C���@�����R�!���=qC���                                    BxYi�V  
�          @�
=@�ff�(���{���
C��3@�ff�p�����ffC��H                                    BxYi��  /          @��
@�z��.{��\��ffC���@�z��/\)�G�����C��R                                    BxYi͢  y          @��@�
=����H�?�C��@�
=�ff�����<��C��H                                    BxYi�H            @�=q@���&ff��Q��<z�C�c�@���'
=���9��C�Z�                                    BxYi��  T          @�=q@����333�c�
�z�C�n@����333�^�R�	p�C�g�                                    BxYi��  �          @�G�@��R�333�W
=�C�E@��R�3�
�Q��
=C�>�                                    BxYj:  �          @�\)@�p��:=q���R�K�C�f@�p��:�H��p��H��C��q                                    BxYj�  �          @��@�
=�(�ÿ���*�RC�  @�
=�)����ff�(Q�C���                                    BxYj%�  �          @�p�@�\)�.{����(�C�Z�@�\)�.�R�
=����C�N                                    BxYj4,  �          @�p�@�z��3�
��33��33C�k�@�z��3�
�����(�C�c�                                    BxYjB�  �          @�
=@�ff�;���
=�g�C�@�ff�;����e��C��                                    BxYjQx  �          @�\)@�{�8�ÿ\�w�C�(�@�{�9����G��u�C�"�                                    BxYj`  �          @���@���1녿��H���\C��
@���2�\�������
C��\                                    BxYjn�  �          @��\@����0���B�\���HC�Ф@����1G��A���(�C��f                                    BxYj}j  �          @���@�=q�I����(��m��C��3@�=q�J=q���H�lQ�C��\                                    BxYj�  �          @�ff@���B�\����  C��\@���C33�����C���                                    BxYj��  �          @��R@���>{�Q��ŮC�}q@���>{�Q���G�C�y�                                    BxYj�\  �          @���@�=q�333�#33���C���@�=q�333�#33���HC���                                    BxYj�  �          @���@����R�Tz��p�C��@����R�Tz��\)C��                                    BxYjƨ  �          @���@�33�s�
�u�(�C���@�33�s�
�u�(�C���                                    BxYj�N  �          @�(�@����w���\)�3�
C�&f@����w���\)�4z�C�&f                                    BxYj��  �          @�
=@�G��l(��&ff�ҏ\C���@�G��l(��&ff�ӅC��                                    BxYj�  �          @���@�  �z=q�����]��C��3@�  �y�������^ffC��{                                    BxYk@  �          @���@�=q�QG�����Q�C�7
@�=q�P�׿���R�HC�8R                                    BxYk�  �          @���@����G
=�u��
C��\@����Fff�u���C���                                    BxYk�  �          @�������J=q@0��BffCjc�����J=q@0  B�Cjs3                                    BxYk-2  �          @�\)�z��Vff@ffA�=qCk=q�z��Vff@A��CkJ=                                    BxYk;�  �          @�G���Q����
@   AΏ\C�y���Q���(�?��RA�
=C�z�                                    BxYkJ~  �          @���@����  ?fffA(  C���@����Q�?c�
A$��C��                                    BxYkY$  �          @���@H���n{?�=qAHz�C��@H���n�R?��AEG�C��                                    BxYkg�  �          @�{@p���Y���fff�!p�C��@p���Y���k��$z�C���                                    BxYkvp  T          @�ff@u�W��c�
�p�C�^�@u�W��h��� ��C�c�                                    BxYk�  �          @�=q@����<(��Y����C�Ф@����;��\(���\C���                                    BxYk��  �          @��\@g��k�����G
=C�Ff@g��j�H��z��K33C�L�                                    BxYk�b  T          @�  @?\)�����  �2�\C�H@?\)��33���
�7�C�f                                    BxYk�  �          @��@aG��Z=q�����qC��@aG��Y�������vffC��
                                    BxYk��  �          @�G�@b�\�Q녿�Q�����C���@b�\�P�׿��H��\)C��\                                    BxYk�T  �          @��H@8���z=q�
=q���C�9�@8���y���(���z�C�H�                                    BxYk��  �          @�(�@/\)�a��<�����C���@/\)�`  �>�R�33C�f                                    BxYk�  �          @��
@5��Vff�C33�G�C�)@5��Tz��E���HC�:�                                    BxYk�F  �          @�=q@C33�P  �1G���z�C��{@C33�N�R�333� �
C���                                    BxYl�  �          @���@���(���������C�>�@����H��\)���C�W
                                    BxYl�  T          @�=q@�������
=����C�AH@����=q���H��\)C�\)                                    BxYl&8  
�          @�Q�@���2�\��G����C��@���1G������  C�                                      BxYl4�  �          @���@w
=�?\)���
��  C�"�@w
=�>{����G�C�<)                                    BxYlC�  �          @�Q�@`���Dz��:�H� =qC�l�@`���A��=p��{C��
                                    BxYlR*  �          @�=q@_\)�K��<�����C���@_\)�I���?\)��C��q                                    BxYl`�  �          @��@a��A��8�����RC���@a��?\)�;��Q�C��                                     BxYlov  �          @���@�=q�9����\���C�Ff@�=q�7����ȸRC�l�                                    BxYl~  �          @���@�(��ff�����{C���@�(���
�����C�\                                    BxYl��  �          @���@���	���,����\C�� @���
=�/\)��C��
                                    BxYl�h  �          @�{@N{�^�R�+���(�C�aH@N{�\(��/\)��33C���                                    BxYl�  �          @��@E��j�H�
=��  C��@E��hQ���H�ׅC�.                                    BxYl��  �          @�{@5����׿}p��6�\C��3@5���  ����B�RC���                                    BxYl�Z  �          @�33@�ff�8�ÿ��R����C���@�ff�7��������C��
                                    BxYl�   T          @��\@�Q��z�O\)��C�c�@�Q���
�Y����C�u�                                    BxYl�  �          @�{@����
�H��(��QG�C�Ф@����	����G��X  C���                                    BxYl�L  �          @�{@��ÿ˅���R��{C��@��ÿǮ�\���RC��{                                    BxYm�  �          @�
=@��H�����"�\��=qC�7
@��H��z��#�
����C�}q                                    BxYm�  �          @�@��\�����\��G�C��R@��\�ff����C��                                    BxYm>  �          @�  @\)�Q녿����C�Ff@\)�O\)��Q�����C�p�                                    BxYm-�  �          @�{@\(��g����33C���@\(��e��
=q��  C��=                                    BxYm<�  �          @�  @����=q�{����C��q@������G���33C�ٚ                                    BxYmK0  �          @�(�@�
���\�u�0��C��f@�
���\�.{��ffC���                                    BxYmY�  �          @�@,(���G��s33�#33C�J=@,(����׿�ff�4��C�Y�                                    BxYmh|  �          @��
@xQ��k��\����C�C�@xQ��j�H����p�C�L�                                    BxYmw"  �          @�(�@R�\���þu�(��C���@R�\���׾����j=qC��                                    BxYm��  �          @�(�@J�H����p���$z�C��3@J�H���\����5��C��f                                    BxYm�n  
�          @�ff@Fff�w��{��
=C�T{@Fff�tz���
�ɮC���                                    BxYm�  T          @�p�@s33�i������C33C�R@s33�g���p��R�HC�5�                                    BxYm��  �          @�p�@[��[��^�R�#33C�|)@[��Z=q�u�3�C��{                                    BxYm�`  �          @���@#�
�\)@#33A��
C�\)@#�
����@(�A��
C�'�                                    BxYm�  �          @���@���h��@B�\B�HC�>�@���n{@<(�B	��C���                                    BxYmݬ  �          @���@<���u?�\)A�C���@<���xQ�?�G�A��C��q                                    BxYm�R  �          @��@�  �N�R��=q����C���@�  �K������\C���                                    BxYm��  �          @��@�Q��2�\�'��ᙚC�� @�Q��-p��,�����HC��)                                    BxYn	�  �          @�ff@c�
�]p������L��C���@c�
�[���p��_
=C��                                    BxYnD  �          @�33@e�%�Tz��33C�{@e�   �X���{C���                                    BxYn&�  �          @��@��\�p��:=q��G�C�w
@��\���>�R��C���                                    BxYn5�  �          @��\@z�H��
�XQ���C�,�@z�H�����\����C��f                                    BxYnD6  �          @��\@z=q�ٙ��e�� 
=C�Ff@z=q��=q�hQ��"��C��R                                    BxYnR�  T          @�33@q녿��R�`  �p�C�
@q녿���dz�� �RC�                                    BxYna�  �          @��\@?\)�Z=q�N�R�C�� @?\)�S�
�U�=qC�3                                    BxYnp(  �          @��\@U��AG��S�
�p�C��@U��:=q�Z=q�ffC�s3                                    BxYn~�  �          @���@\(��9���L(��ffC��\@\(��2�\�R�\�G�C�xR                                    BxYn�t  �          @�33@n�R�.{�J�H�
{C��\@n�R�'
=�P����\C�}q                                    BxYn�  �          @��@�p��33�Q���  C�>�@�p��{�p���33C���                                    BxYn��  �          @��R@;��7��e�$��C��@;��/\)�l(��*\)C�k�                                    BxYn�f  �          @�@Dz��5�b�\�!(�C��@Dz��-p��h���&��C�L�                                    BxYn�  �          @���@o\)��G��L����C�j=@o\)���þ�33�fffC�s3                                    BxYnֲ  �          @��@|���c�
��G���C���@|���b�\�����HC��                                    BxYn�X  �          @�\)@���(��?W
=AQ�C��@���*�H?=p�@�ffC���                                    BxYn��  �          @�
=@z�H�mp������
C�G�@z�H�l(�����˅C�\)                                    BxYo�  T          @��R@r�\�hQ��G�����C��@r�\�dz��33����C�W
                                    BxYoJ  T          @�{@e�O\)����ҏ\C��@e�I���!G���z�C�\)                                    BxYo�  �          @�z�@,����33�333��
=C�=q@,����녿aG���HC�S3                                    BxYo.�  �          @�(�@(Q���  ����4��C�'�@(Q���ff��(��T��C�G�                                    BxYo=<  �          @�(�@>{�G��Dz����C��=@>{�?\)�L(���C�e                                    BxYoK�  �          @��@{�;��n�R�1  C��@{�1G��vff�833C���                                    BxYoZ�  �          @��?�(��Y���p  �2G�C�g�?�(��O\)�x���:�
C���                                    BxYoi.  �          @��
?У��`  �l���/�C�w
?У��U�vff�7��C��)                                    BxYow�  �          @��H?�Q��;����\�I\)C�?�Q��0  ��ff�Q�C�Ф                                    BxYo�z  �          @�=q?���,�����a�
C�l�?��� ����G��k{C�8R                                    BxYo�   �          @�33?�z��=p����
�J�C���?�z��1G�����R��C�s3                                    BxYo��  T          @��@&ff�S�
�Vff���C��@&ff�J=q�_\)�"
=C��H                                    BxYo�l  T          @�
=@L���tz��{���C��R@L���mp�������C�]q                                    BxYo�  T          @��\@J=q�u���ff���C���@J=q�p  ��p���p�C��                                    BxYoϸ  
�          @���@<�����
=�Q�?��\C��@<�����
��G����HC��                                    BxYo�^  �          @�G�@N�R�z�H��{�o�C��R@N�R�vff������\C��R                                    BxYo�  �          @�33@>�R���\��G����C�@>�R�\)���H��p�C�\)                                    BxYo��  �          @���@AG���G���33��\)C�` @AG��|(��ff��G�C��
                                    BxYp
P  �          @�ff@S33�q�����z�C��\@S33�k��{��p�C���                                    BxYp�  �          @�(�@7��l(��\)��(�C���@7��c�
�*�H��(�C�q�                                    BxYp'�  �          @�(�@
=�u��8����C���@
=�k��E��C�N                                    BxYp6B  �          @���@R�\�vff�(���p�C�AH@R�\�n�R�����G�C���                                    BxYpD�  �          @�G�@XQ��vff����Q�C�� @XQ��o\)�G���(�C��                                    BxYpS�  �          @��@g��w���ff��(�C���@g��qG��   ���C���                                    BxYpb4  �          @��H@Fff��{��p����\C�:�@Fff���\�����{C���                                    BxYpp�  �          @��@j�H�x�ÿ��H��{C���@j�H�s33�����C�f                                    BxYp�  �          @��H@L(����R���H��33C��\@L(������
=��
=C���                                    BxYp�&  T          @���@b�\�}p�����j=qC��)@b�\�xQ�������C�(�                                    BxYp��  �          @��R@����i���Tz��p�C���@����fff���
�&ffC��                                    BxYp�r  �          @�\)@�z��fff�@  ��
=C�U�@�z��c33�s33�Q�C��f                                    BxYp�  �          @�{@����p�׿��\�$��C��H@����l�Ϳ�p��G�C�                                      BxYpȾ  �          @�ff@�Q��S�
�}p�� z�C���@�Q��P  ��
=�?�C�q                                    BxYp�d  �          @�{@����>{�O\)�33C��@����:�H�z�H�\)C�C�                                    BxYp�
  �          @��\@l���`  ��
��ffC�G�@l���XQ��G�����C��\                                    BxYp��  �          @��
@U��tz��G����RC��@U��k��\)��33C�
                                    BxYqV  �          @�{@�33�\(����H�Ep�C��)@�33�W
=���g\)C�/\                                    BxYq�  �          @���@h���z�H�����c33C�n@h���u��˅���C�                                    BxYq �  �          @���@e��y����G��UG�C�G�@e��tz῾�R�~�RC��
                                    BxYq/H  �          @���@s33�fff��G���(�C�Ff@s33�_\)��p����C��q                                    BxYq=�  �          @���@y���Q�����ffC���@y���H���z��ƸRC��
                                    BxYqL�  �          @���@z�H�Vff��
=��(�C��H@z�H�N{������HC�O\                                    BxYq[:  �          @�Q�@vff�w
=���R�S33C�t{@vff�u�\)���C��\                                    BxYqi�  �          @���@r�\�{��W
=�p�C�  @r�\�z=q�����(�C�{                                    BxYqx�  �          @�33@vff�\)�E����C��q@vff�{����
�)��C�33                                    BxYq�,  �          @��@w
=�S33���ŅC��)@w
=�H���#33����C�p�                                    BxYq��  �          @��H@h���Vff�%����C��3@h���J�H�333��G�C�}q                                    BxYq�x  �          @�G�@e�{���
=�o�
C�5�@e�u���Q����\C���                                    BxYq�  �          @���@h���w
=��ff��ffC��@h���p  ������C�3                                    BxYq��  �          @��@l(��dz�� ����G�C��)@l(��Z�H�  ����C���                                    BxYq�j  �          @��@	���33��  �gG�C���@	��������z��rz�C��f                                    BxYq�  �          @�z�@������[��C�  @����
=�g�C�/\                                    BxYq��  �          @��H@'��333����C��C��=@'���R��33�Oz�C�E                                    BxYq�\  �          @�  @��dz��\)�0�HC���@��QG�����>��C��                                    BxYr  �          @��?�p��$z�����`
=C�Ǯ?�p��{����lC��)                                    BxYr�  �          @�=q@z������ff�h�C�:�@z�����33�t�RC��H                                    BxYr(N  �          @�ff@���\)��  �w��C��q@���p����
33C�B�                                    BxYr6�  �          @�33@��
=q��z��d{C��@������G��o{C�0�                                    BxYrE�  �          @�=q@  �!G���ff�Y�C��@  �
=q���
�ez�C�R                                    BxYrT@  �          @�p�@������(��^Q�C��\@�Ϳ�33��G��i�\C��                                    BxYrb�  �          @��@,���
�H��ff�Vp�C��H@,�Ϳ�ff��33�`C�"�                                    BxYrq�  �          @�z�@+���
=��\)�fffC���@+�������H�n��C�%                                    BxYr�2  �          @�{@p���(���{�tffC���@p������G��|z�C�]q                                    BxYr��  �          @��\@\)>B�\���933@*=q@\)?�\�����7��@陚                                    BxYr�~  H          @�33@p����\�����z�C���@p��(����R�C��                                    BxYr�$  
4          @��H@�.{���H� C�޸@>k����HaH@���                                    BxYr��  �          @�  @U?�  ��(��AA�p�@U?����\)�9�\A�\)                                    BxYr�p  �          @�ff@tz�@0���N{�	�Bp�@tz�@@���?\)����B
=                                    BxYr�  �          @�p�@o\)@L(��0����Q�B!�@o\)@Y���\)�љ�B){                                    BxYr�  �          @��
@[�@W��8����Q�B2{@[�@fff�'
=��p�B9�                                    BxYr�b  �          @��
@|(�@G������\)B�\@|(�@S�
����G�B {                                    BxYs  �          @���@��?��mp��z�A�=q@��@
=�c33��A�{                                    BxYs�  �          @�z�@�p�?�{�{��'=qA���@�p�?�Q��s33� ��A�=q                                    BxYs!T  
�          @�{@���?�Q���  �)(�A���@���?�\�w
=�"�A���                                    BxYs/�  �          @�\)@���?���z�H�$�A�z�@���?��r�\���A��                                    BxYs>�  T          @���@�G�?����~{�%
=A�\)@�G�?���u��
=A���                                    BxYsMF  �          @�Q�@���?L����  �'{A#�
@���?���z=q�"��AhQ�                                    BxYs[�  
�          @�G�@��
?˅�s�
�=qA�p�@��
?�z��j=q�A�
=                                    BxYsj�  T          @�z�@�G�?fff��  �#p�A4  @�G�?�  �z=q�Aw
=                                    BxYsy8  �          @���@�=q?�{�c33�
=A���@�=q?��Y���	�A�{                                    BxYs��  �          @��H@���?�G��h���\)As�@���?˅�`���G�A���                                    BxYs��  `          @�(�@�Q�>�\)�����7G�@s33@�Q�?.{��33�4��A�                                    BxYs�*  �          @�=q@5��xQ���p��j��C��f@5�������pG�C���                                    BxYs��  �          @���@?\)�u��(��e�C�!H@?\)�   ��ff�j�C�8R                                    BxYs�v  �          @���@mp��:�H����K�C�s3@mp���z���ff�N�\C��q                                    BxYs�  �          @�G�@�33���H�����;\)C�� @�33��\)���<��C�~�                                    BxYs��  T          @�33@u�c�
�����FC�s3@u��ff���R�Jz�C���                                    BxYs�h  
�          @ȣ�@�����ff����@\)C��)@����������D�RC���                                    BxYs�  `          @��@�(���������933C���@�(��p����p��?�C���                                    BxYt�  �          @�(�@!��-p���33�R\)C�}q@!��\)���H�a��C�C�                                    BxYtZ  T          @��
@'���
����\{C�H�@'�������{�i�
C���                                    BxYt)   �          @�=q@I���(���33�DffC�
@I����p���=q�QG�C��                                    BxYt7�  �          @�@P���%�����>�C���@P���Q���G��KQ�C�j=                                    BxYtFL  �          @�p�@QG��\)���H�@�\C�Y�@QG�����=q�Mp�C��                                    BxYtT�  �          @���@5��Q���G��Q(�C��{@5������  �^�C��                                    BxYtc�  �          @���@p  �����  �.��C���@p  ���H���R�:(�C�:�                                    BxYtr>  �          @�=q@dz�� ����ff�=��C�H�@dz��ff��(��G��C�=q                                    BxYt��  �          @��H@qG��޸R���R�<(�C���@qG����
����D�
C��f                                    BxYt��  �          @��
@��Ϳ�������+�C�1�@��Ϳ��H��G��3(�C��q                                    BxYt�0  �          @���@33��\)�=q����C�,�@33��ff�:=q��z�C�                                    BxYt��  �          @�(�?������   ����C���?�����ff�!���ffC�C�                                    BxYt�|  �          @��R@^{�G��j=q�\)C��@^{�.{�}p��'�
C���                                    BxYt�"  �          @�  @Vff�U��l���p�C��R@Vff�:�H�����(�C�w
                                    BxYt��  �          @��@I���$z���\)�8��C�aH@I���
=��\)�G(�C�R                                    BxYt�n  �          @��
@>{�5�����5G�C�"�@>{�Q����Ez�C���                                    BxYt�  �          @��?�G��fff�o\)�4
=C��q?�G��K�����K��C�ٚ                                    BxYu�  �          @��R�����\�N�R�
  CwLͿ��|���j�H� Cuff                                    BxYu`  �          @�ff���H��
=�E���
Cy!H���H����b�\�{Cwz�                                    BxYu"  �          @��������
�B�\��RCw^�������Q��`  �Cu�{                                    BxYu0�  �          @�p�������H�=p���\)C}.�������\����
C{�)                                    BxYu?R  �          @�{���\���R�W���C������\����vff�*G�C�f                                    BxYuM�  �          @���aG������XQ���\C����aG���z��w��)�C��)                                    BxYu\�  �          @��R���R�����Mp��	�C~�����R��z��l���"Q�C}L�                                    BxYukD  �          @��R�������S33�33C�/\������H�r�\�'ffC
                                    BxYuy�  �          @�Q�����Q��^�R�{C�l;�����\�~{�0�C�(�                                    BxYu��  �          @�Q�?��
�z�H��  �0\)C��?��
�\(���p��I33C�8R                                    BxYu�6  �          @�\)    ��ff�P  �
C��    ��G��qG��%=qC��                                    BxYu��  �          @�G�?�����=q�z�H�*��C�4{?����e����D33C�<)                                    BxYu��  �          @�ff�
=����QG��z�C�]q�
=��ff�q��'
=C��                                    BxYu�(  �          @��>�z����H�mp��"
=C��>�z��w���ff�<�C�"�                                    BxYu��  �          @�p�?Y������S�
��
C�]q?Y������tz��*p�C��                                    BxYu�t  �          @���?k����U��
C��
?k���  �vff�,�C�q�                                    BxYu�  �          @��?��\��G��a���
C��\?��\�u�����5z�C�p�                                    BxYu��  �          @���?^�R��(��\(���C�� ?^�R�{��|���1�RC�>�                                    BxYvf  �          @���?��
�����P  �G�C�g�?��
���H�q��'Q�C��                                    BxYv  �          @���?ٙ���z���R��\)C�#�?ٙ�����&ff��Q�C��                                     BxYv)�  �          @�
=?�Q���
=�p���p�C��q?�Q���(��C33�
=C�5�                                    BxYv8X  �          @�33?@  ���Ϳ��B=qC���?@  ���R��=q���C��                                    BxYvF�  �          @�z�?�����\)��{���RC�
?�������\)��C�j=                                    BxYvU�  �          @�z�?�=q�������Tz�C�e?�=q��G���Q�����C���                                    BxYvdJ  �          @�(�?�����Q��G��|  C��R?��������
=q��ffC���                                    BxYvr�  �          @�������H��H�ʏ\C��׿������B�\�
=C��=                                    BxYv��  �          @�{=u��
=�-p�����C�S3=u���H�Tz����C�Y�                                    BxYv�<  �          @��R���
��Q��H����
C�~����
��=q�n{�#�C�p�                                    BxYv��  �          @��R����
=�XQ����C�����  �{��1{C���                                    BxYv��  �          @�33=����R����
=C��3=���33�C33��C��                                     BxYv�.  �          @��>�\)���H����i��C�|)>�\)���
�z���=qC���                                    BxYv��  �          @��R�.{�����*=q�߮C���.{��(��R�\�C��3                                    BxYv�z  �          @��=L�������H�����C�S3=L����33�o\)�"�RC�\)                                    BxYv�   �          @�\)>\���
�U�
=C�T{>\��z��z�H�-33C���                                    BxYv��  �          @�\)?�G�����Z=q�G�C�^�?�G��\)�~�R�/�HC��                                    BxYwl  �          @�Q�?p�������|���.33C��H?p���_\)���R�K�C���                                    BxYw  �          @���?��R�y���\)�/  C�xR?��R�U���\)�Kp�C��                                    BxYw"�  T          @��?\��=q������C�Z�?\��Q��'
=���HC��)                                    BxYw1^  �          @�\)?�{��G��!G��иRC���?�{����H���33C�k�                                    BxYw@  �          @�\)?�����\�J=q�(�C�%?������p���$
=C�                                    BxYwN�  �          @�Q�>\�xQ�����=z�C��\>\�QG���\)�\C�U�                                    BxYw]P  �          @�Q�?�  �����1G���z�C���?�  ���
�Y���G�C�|)                                    BxYwk�  �          @�Q�?�Q����\�U��Q�C���?�Q��u��y���0�
C��                                     BxYwz�  �          @��?�33�����.�R��C�� ?�33��
=�XQ���HC�:�                                    BxYw�B  �          @���?ٙ�������R���C�K�?ٙ���{�:=q��Q�C��{                                    BxYw��  �          @�33@�����C33����C�e@�����i����
C��                                    BxYw��  �          @��H?��R��{�aG���C���?��R�i�����\�1��C�H�                                    BxYw�4  �          @�33?�33�s�
�����-ffC�>�?�33�L����Q��J{C�U�                                    BxYw��  �          @��\?�p������p  ���C�)?�p��\����G��<��C���                                    BxYwҀ  �          @�Q�@�H���
��H��\)C�Ф@�H��
=�C�
� ��C���                                    BxYw�&  �          @�Q�@���R�ff��ffC�9�@��=q�@  ��33C�9�                                    BxYw��  �          @���@ff��Q��/\)��\C��@ff��=q�W��z�C�f                                    BxYw�r  �          @��@z������@  ����C�H�@z������g���C��                                    BxYx  �          @�=q@	����G��R�\�	��C�J=@	���qG��xQ��'Q�C�ٚ                                    BxYx�  �          @�33?�\)�}p��s33�#��C���?�\)�Vff���H�B  C��R                                    BxYx*d  �          @�(�?����j�H��(��@��C�N?����>�R���
�`(�C�k�                                    BxYx9
  T          @��
?�
=�j=q���D  C��?�
=�>{��p��d(�C��\                                    BxYxG�  �          @��?޸R���
�p��� Q�C�t{?޸R�`�����\�?�C�/\                                    BxYxVV  �          @�z�?�\)�\)�y���&{C���?�\)�W
=���R�DC��\                                    BxYxd�  �          @�z�@�
�o\)��=q�.p�C�q�@�
�E����\�L
=C���                                    BxYxs�  �          @�(�@
�H�^{��\)�7�C��@
�H�2�\��ff�T33C��3                                    BxYx�H  �          @���@(��fff��(��1�
C���@(��;���(��N�HC�b�                                    BxYx��  �          @��?�{�{������.{C�  ?�{�QG����\�N�C�R                                    BxYx��  �          @�(�?��R�~{�vff�#�C�K�?��R�U�����B�C�g�                                    BxYx�:  �          @��?J=q�n�R�����DffC��
?J=q�@����p��f�HC�W
                                    BxYx��  �          @�33?�������j=q�G�C�l�?���\(�����=�C�P�                                    BxYxˆ  �          @�(�@�ff�J�H��p��j=qC�H@�ff�9��������  C�*=                                    BxYx�,  �          @���@�
=����{��{C�@�
=��=q�"�\���C���                                    BxYx��  
�          @��@��׿��������C��3@��׿�Q��)����p�C�H                                    BxYx�x  �          @��@7��vff�I����C�W
@7��S33�mp��!ffC�y�                                    BxYy  �          @�33@K��mp��J=q���C�H�@K��J=q�mp��\)C��\                                    BxYy�  �          @�{@��
��P���Q�C��@��
����e���C��)                                    BxYy#j  �          @��@��R��c�
��C�Ф@��R���R�vff�#{C�>�                                    BxYy2  �          @���@{�����i�����C�Y�@{���\�~�R�*��C���                                    BxYy@�  �          @��H@\����
�z=q�-
=C�{@\�Ϳ�33��\)�?�C�4{                                    BxYyO\  �          @�Q�@g
=�p��e��!
=C�E@g
=�����x���2ffC�                                    BxYy^  T          @���?�33���\�'��ә�C��R?�33���\�Z=q��RC��H                                    BxYyl�  �          @��
?�����33�ٙ���G�C�W
?�����  �#33���C���                                    BxYy{N  �          @�33?����ff�������C�` ?�����H�*=q����C��                                     BxYy��  �          @��H?�����  ����{C���?������\�=p�����C�Y�                                    BxYy��  �          @�33?�ff���
��(����\C�?�ff��Q��%��хC��R                                    BxYy�@  �          @��
@��Q쿺�H�g�
C�ٚ@��ff�z����RC�y�                                    BxYy��  �          @��
?�Q���{���
�"=qC���?�Q���{��
=��G�C�@                                     BxYyČ  �          @��@ ����p��h���  C�/\@ ����{����(�C���                                    BxYy�2  �          @��@�R��G�������
C��3@�R��(������U�C��                                    BxYy��  �          @�z�@   ��녽�\)�333C��R@   ��
=�xQ��  C���                                    BxYy�~  �          @�(�?��������'
=��p�C�]q?�����  �Z�H�ffC�*=                                    BxYy�$  �          @�{?�ff��  �Z�H��
C���?�ff�vff��(��0�HC���                                    BxYz�  �          @�ff@����
=�O\)��C��@���vff�|���&��C��                                     BxYzp  �          @�
=@2�\�����C�
���C���@2�\�l(��p  ���C��=                                    BxYz+  �          @��R@Dz���Q��2�\�߅C�� @Dz��n{�^�R�G�C���                                    BxYz9�  �          @�{@7����������C���@7���(��I��� =qC�`                                     BxYzHb  �          @�
=@=p���  �33���C��
@=p���=q�6ff��=qC�4{                                    BxYzW  �          @���@Dz���  ��
=��ffC�s3@Dz���(���R��33C���                                    BxYze�  �          @�33@Vff��z῕�8��C��=@Vff������H��G�C�                                    BxYztT  �          @��\@K���zΌ���g�C�8R@K�����  ���C�5�                                    BxYz��  �          @���@U����H�u�{C��)@U���33��G���ffC���                                    BxYz��  �          @�G�@fff��p��fff��C��\@fff��{������C�O\                                    BxYz�F  �          @���@Q����������]��C���@Q���\)�
�H����C���                                    BxYz��  �          @���@e��
=��{��G�C�.@e�w������C�n                                    BxYz��  �          @���@Z�H��  ����p�C�j=@Z�H�u�*=q�ڣ�C��q                                    BxYz�8  �          @�  @?\)��\)����ʏ\C��@?\)�n{�J�H�z�C�k�                                    BxYz��  �          @��R@c�
���c�
�
=C�8R@c�
�|�ͿУ���Q�C�f                                    BxYz�  �          @���@Y����p��\�s\)C��=@Y����=q��\��
=C��                                    BxYz�*  �          @���@Y�������ff�y�C��q@Y����  ��
��=qC�.                                    BxY{�  �          @�  @�\)�fff�.{��p�C��)@�\)�`  �Tz����C���                                    BxY{v  �          @�Q�@��U=�G�?�ffC�:�@��S�
��\���HC�b�                                    BxY{$  �          @��
@��H�-p�?�  A�C�H@��H�5>�@�\)C�h�                                    BxY{2�  �          @�33@��H�[�?k�A�C��)@��H�b�\>��@'
=C�,�                                    BxY{Ah  �          @���@�(��g
=?�Q�A<��C�B�@�(��p��>��H@�z�C���                                    BxY{P  �          @��H@����U?˅A}p�C��@����e�?p��A��C�Ф                                    BxY{^�  �          @��@�G��S33?�A�ffC���@�G��c33?��\A z�C���                                    BxY{mZ  �          @�
=@�{�5�?(�@�z�C���@�{�8��=�\)?(��C�s3                                    BxY{|   �          @�@�=q�'
=������C��3@�=q�#�
�����HC�.                                    BxY{��  �          @���@�p��.{?L��@���C�&f@�p��4z�>�=q@'
=C���                                    BxY{�L  �          @�@��\�;�?8Q�@�G�C��@��\�@��>��?�  C���                                    BxY{��  �          @��R@���Fff?�@��\C���@���H�ý��
�Tz�C��                                    BxY{��  �          @��@��
�.{?#�
@���C���@��
�2�\=���?z�HC�H�                                    BxY{�>  �          @���@��H�.�R>��@�  C�y�@��H�1G��u�z�C�N                                    BxY{��  �          @��R@��� ��?}p�A��C�\)@���)��>��H@�p�C��)                                    BxY{�  �          @��@�z���?h��A\)C�� @�z��#�
>�
=@���C�O\                                    BxY{�0  �          @�G�@�  �/\)?��A.�RC�33@�  �:=q?
=@�33C�z�                                    BxY{��  �          @���@�(��!G�?��A!G�C�p�@�(��+�?
=q@�C�                                    BxY||  �          @���@��(�?n{Ap�C��=@��$z�>�(�@��C�T{                                    BxY|"  �          @�G�@�\)�G�?�
=A3�C���@�\)�p�?5@�\)C���                                    BxY|+�  �          @���@����\)?�  A�C���@������?
=q@�=qC�O\                                    BxY|:n  T          @���@����\)?}p�A
=C���@������?�@���C�O\                                    BxY|I  �          @�G�@��R��\?�Q�A5G�C���@��R��R?5@׮C��                                    BxY|W�  �          @��@��H���?��Al  C��3@��H�)��?��A��C�Ф                                    BxY|f`  �          @�G�@��H�=q?��HA`(�C��{@��H�*=q?s33A��C���                                    BxY|u  �          @�{@�\)�
=q?��HA�(�C��@�\)�p�?��RAAp�C�g�                                    BxY|��  �          @�z�@�Q��\)@
=A�=qC�p�@�Q�� ��?�  A��C���                                    BxY|�R  �          @�
=@�=q�Q�?�Q�A�G�C�)@�=q��H?�(�A=p�C�                                    BxY|��  �          @�@��\�Q�?�{A���C�u�@��\�-p�?��AR{C��)                                    BxY|��  �          @���@��H���@
=A�=qC�^�@��H�1�?�=qAs\)C��\                                    BxY|�D  �          @���@��=q?�A�=qC��@��0  ?���AT��C�                                    BxY|��  �          @�G�@�
=�4z�?��\A  C��\@�
=�>{>�
=@���C�/\                                    BxY|ې  �          @��@����"�\?ǮAo33C�#�@����3�
?�G�Ap�C��q                                    BxY|�6  �          @�G�@��R�'
=?�G�Ag�
C��\@��R�7�?p��A\)C��
                                    BxY|��  �          @�G�@��
��R@�
A�p�C��@��
�6ff?�G�Af�HC�s3                                    BxY}�  �          @��@�Q��#33?�G�Ag�
C��@�Q��3�
?s33A��C��                                    BxY}(  �          @�  @��#�
?���As�C��)@��5�?�G�A
=C��                                    BxY}$�  �          @�
=@����(Q�>�G�@���C�Ф@����*=q����{C��=                                    BxY}3t  �          @��@�{�p�>.{?�{C��R@�{�(���33�VffC���                                    BxY}B  �          @���@�ff�$z�=L��>�G�C�aH@�ff�!G����H��ffC��{                                    BxY}P�  �          @�Q�@���%?�G�A���C���@���:=q?�
=A5p�C�0�                                    BxY}_f  �          @���@�G��/\)?�A��
C��q@�G��C�
?�Q�A6ffC�aH                                    BxY}n  �          @�  @�\)�=p�?�Q�A_\)C��H@�\)�L��?G�@���C��                                    BxY}|�  �          @���@����8Q�?+�@�(�C���@����=p�=#�
>�
=C�U�                                    BxY}�X  �          @�G�@�\)�N�R��G���ffC���@�\)�HQ�J=q��p�C��                                    BxY}��  �          @�  @���X�ÿ�G��z�C�b�@���Fff���H���C��
                                    BxY}��  �          @��H@�{�`�׿!G����RC�Q�@�{�R�\��\)�O�C�+�                                    BxY}�J  �          @�z�@����J�H���
��HC�7
@����7���Q��\)C�p�                                    BxY}��  �          @��@����8��>��H@��C��@����;��\)���C��                                    BxY}Ԗ  �          @�  @���Dz�?޸RA��C��{@���XQ�?��A   C�u�                                    BxY}�<  �          @�\)@����>�R?��A#�C���@����H��>Ǯ@n�RC��                                    BxY}��  �          @�ff@��\�9��?���A�C���@��\�N�R?�
=A7�C��                                    BxY~ �  �          @��@�Q��7
=?�p�Af=qC�"�@�Q��G�?Q�@�p�C��                                    BxY~.  �          @���@�(��p�@5�A��C��@�(��AG�@{A�33C�y�                                    BxY~�  �          @�Q�@�33�4z�@�A�G�C��@�33�N{?���A_\)C�=q                                    BxY~,z  �          @�  @���Fff?��RAeC�޸@���Vff?E�@�(�C�ٚ                                    BxY~;   �          @�=q@��
�*�H?�\A�{C�8R@��
�@  ?��A-G�C���                                    BxY~I�  �          @��R@�z῕@FffA��C�H�@�z����@1G�Aߙ�C��                                    BxY~Xl  �          @�  @�{��p�@HQ�A�ffC�f@�{���@1�Aݙ�C���                                    BxY~g  
�          @Å@�  ��
@(Q�A�ffC���@�  �5@�
A�\)C�:�                                    BxY~u�  �          @�(�@����33@&ffA�33C���@����4z�@G�A�Q�C�XR                                    BxY~�^  �          @��@����
@#�
A�{C�ٚ@���4z�?�p�A��C���                                    BxY~�  �          @���@���@p�A���C�n@��7
=?�{As
=C���                                    BxY~��  �          @���@��R���@
=qA�ffC�j=@��R�7�?�ffAj{C���                                    BxY~�P  �          @�ff@�p���Q�@��A�z�C�Y�@�p����@   A�G�C��R                                    BxY~��  �          @�
=@�z��Q�@$z�A�C�K�@�z���R@�A��C��f                                    BxY~͜  �          @�{@��\��
@�A��
C�q�@��\�"�\?�A�=qC�B�                                    BxY~�B  �          @�
=@����\@(�A��C��
@���p�?���A�=qC��                                    BxY~��  �          @ƸR@��
��=q?�A��
C��R@��
�{?���AX  C�7
                                    BxY~��  �          @�p�@��
��
=@!G�A��C�S3@��
�p�@z�A�
=C���                                    BxY4  �          @�ff@��ÿ�@
=A��C��@����  ?��Av{C��3                                    BxY�  T          @�ff@�33��z�?��A���C���@�33��\?��AN�\C��q                                    BxY%�  �          @�{@�ff��R?��A��C��H@�ff�&ff?���AEG�C�<)                                    BxY4&  �          @�@����{@Q�A���C��H@���33?�33Aw33C��q                                    BxYB�  �          @�z�@�z��(�@p�A�z�C���@�z���H?�Q�A�
C���                                    BxYQr  �          @�ff@�  ��z�@
=A�33C�j=@�  �ff?�{Ap��C�o\                                    BxY`  �          @�ff@����33@
�HA�p�C�o\@���
=?�z�Ax��C�c�                                    BxYn�  �          @�ff@�
=��\@ffA�(�C���@�
=�{?ǮAi��C��3                                    BxY}d  �          @�\)@�ff�@
�HA���C��f@�ff�"�\?�\)Aq�C��f                                    BxY�
  �          @���@������@�RA�z�C��H@����.{?У�Aq��C��                                     BxY��  �          @�Q�@�  ��z�@�\A��C�j=@�  ���?�\A�C�:�                                    BxY�V  �          @�  @�\)��H?�\A���C��@�\)�1G�?�33A)p�C��{                                    BxY��  �          @�\)@�\)�4z�?��A�G�C���@�\)�L(�?�A-p�C�W
                                    BxYƢ  �          @�  @�=q�(��?�\)A���C��3@�=q�@  ?�Q�A0(�C�G�                                    BxY�H  �          @ƸR@������@ ��A�\)C�5�@����ٙ�@
�HA�{C�z�                                    BxY��  �          @Ǯ@��ÿ�33@ffA��C���@����
�H?��A�p�C�L�                                    BxY�  �          @�p�@���G�@(�A��C�ff@���.�R?�=qAmG�C�c�                                    BxY�:  �          @�@�
=�ff@33A�33C��f@�
=�5�?�A{�
C��f                                    BxY��  �          @���@�{�  @&ffA�33C��@�{�333?��RA�z�C�33                                    BxY��  �          @��@�G���(�>\)?��C��@�G���G��\(����C�Z�                                    BxY�-,  �          @��@��H�i���333���
C�� @��H�W��Ǯ�mG�C��3                                    BxY�;�  �          @�ff@��U������[�
C���@��8Q��{���
C��{                                    BxY�Jx  �          @�@��H�i�������C��q@��H�G
=�(����C��{                                    BxY�Y  �          @\@Y�����R��
=�]��C���@Y����
=�!���Q�C�p�                                    BxY�g�  �          @Å@g�����p��a�C���@g����%����C�q�                                    BxY�vj  �          @�p�@k����ÿ�(���G�C��=@k��z=q�AG���z�C��q                                    BxY��  �          @�=q@,����  �%���=qC�Ф@,���~{�j=q�G�C��                                    BxY���  �          @���@J=q��p��\(��\)C���@J=q�L������0p�C�U�                                    BxY��\  �          @Ǯ@X����z��L(���ffC��f@X���N�R����%{C�/\                                    BxY��  �          @�  @Z�H�u��X�����C��H@Z�H�8Q���\)�-�RC��
                                    BxY���  �          @���@A��{��s�
��C��@A��7�����@�RC�E                                    BxY��N  �          @�\)@Dz��z=q�l�����C�{@Dz��7������=33C�u�                                    BxY���  �          @�Q�@E��vff�s33�z�C�P�@E��2�\��z��@�C��                                    BxY��  �          @˅@U�fff��G����C�o\@U��R����Dz�C���                                    BxY��@  �          @�33@_\)�i���u�  C���@_\)�%���(��;\)C�                                    BxY��  �          @��H@aG��p  �mp��ffC��q@aG��,�������6ffC�B�                                    BxY��  �          @��H@^{�p  �p  �=qC�h�@^{�,(���=q�8��C��                                    BxY�&2  T          @��@Z=q�j�H�u�{C�t{@Z=q�%��z��=G�C�g�                                    BxY�4�  �          @��H@_\)�`���|(��
=C�q�@_\)�����{�?��C���                                    BxY�C~  �          @��H@o\)�a��l(���RC�Q�@o\)�\)���R�3G�C�.                                    BxY�R$  �          @�Q�@a���z��E����
C�@ @a��N�R����� �C�˅                                    BxY�`�  
�          @ȣ�@k��h���b�\�	G�C��@k��(Q����H�/=qC�7
                                    BxY�op  �          @ə�@o\)�Z=q�p���G�C���@o\)�ff��  �6{C��                                    BxY�~  �          @�G�@Y���\(���Q��C�N@Y����
��Q��D�RC��                                    BxY���  	�          @�=q@����mp��=p���C�{@����5�s33�z�C��                                    BxY��b  �          @��H@����k���R���C�xR@����:�H�U����C��{                                    BxY��  "          @��@�Q��I���(���
=C���@�Q���H�J=q��RC��                                    BxY���  �          @��@���S33�	����ffC�xR@���(���:�H�؏\C�>�                                    BxY��T  K          @�\)@���L����\����C�G�@���%��2�\��=qC��                                    BxY���  	�          @ƸR@�  ���\��(�����C�8R@�  ��=q���
�h��C�
                                    BxY��  T          @�@���N{���R�_33C�Ǯ@���.�R�G���=qC�ٚ                                    BxY��F  
�          @�z�@�ff�4z�Ǯ�lQ�C��H@�ff�z��\)��p�C���                                    BxY��  
(          @Å@�
=�<(���(��8z�C�S3@�
=�!G���Q���  C�q                                    BxY��  �          @ə�@��R�:=q��\)�$��C��)@��R�!G�����{C��H                                    BxY�8  	�          @�(�@��H�N{?�\)A�ffC�33@��H�e?xQ�AC��R                                    BxY�-�  
Z          @�{@�  �/\)>�=q@   C��H@�  �.{��ff��\)C��)                                    BxY�<�  
�          @�  @�=q�p���
=�u�C��R@�=q�녿z�H�z�C�ff                                    BxY�K*  �          @�  @�����  ��Q���33C��@����333�����p�C���                                    BxY�Y�  T          @�p�@�=q��Q쿳33�R�RC��@�=q���\��p����\C��                                    BxY�hv  
�          @ȣ�@\��
=�s33�(�C���@\�c�
��p��5�C��R                                    BxY�w  
�          @���@�33��p��O\)��ffC�N@�33�xQ쿎{�"�RC�y�                                    BxY���  
Z          @˅@Å��(��z�H���C�9�@Å��zῪ=q�AG�C��)                                    BxY��h  �          @�33@�ff������
�8��C�#�@�ff��z�&ff���C��R                                    BxY��  
�          @��
@�\)���
=���?c�
C�/\@�\)��G��u�(�C�E                                    BxY���  
�          @�p�@�=q��=q>�{@C�
C�%@�=q����=#�
>���C��=                                    BxY��Z  
�          @��@�=q�u>Ǯ@^{C���@�=q���
=�?��C�^�                                    BxY��   
�          @�z�@��
��33<#�
=��
C�s3@��
��{���
�@  C�~�                                    BxY�ݦ  �          @�@�=q�:�H?W
=@�\)C���@�=q�k�?(�@�
=C���                                    BxY��L  
�          @�33@��׿���?Y��@�\)C�t{@���� ��>���@>�RC�                                    BxY���  T          @�\)@����?�\)A%��C�ff@���33?
=@�
=C�^�                                    BxY�	�  �          @Ǯ@�  ���\?�\)A%C��@�  �\?=p�@�=qC��                                    BxY�>  �          @Ǯ@�Q쿏\)?�G�A:{C���@�Q쿴z�?h��A�\C�aH                                    BxY�&�  �          @��@��\�^{���
�B�\C��3@��\�Tz῁G��\)C�g�                                    BxY�5�  �          @���@�Q��J=q>.{?�{C��@�Q��E�.{��z�C���                                    BxY�D0  T          @�@�33�E�=��
?G�C��@�33�?\)�@  ��
=C�g�                                    BxY�R�  �          @�@�(��?\)>��R@8Q�C�s3@�(��>{��\��  C���                                    BxY�a|  
(          @�G�@��
���R?�33A���C�=q@��
��?��A>�HC�K�                                    BxY�p"  T          @ȣ�@����У�@Q�A���C��@����{?���A��C�q                                    BxY�~�  T          @�
=@���,��?0��@��C���@���1녽u�
=C�`                                     BxY��n  �          @�(�@�Q��aG��8Q��Q�C�q�@�Q��U�����*=qC�'�                                    BxY��  "          @\@�\)�z�H��
=�4  C�` @�\)�\���p����RC�+�                                    BxY���  �          @�=q@�
=�|�Ϳ�\)�)p�C�L�@�
=�_\)�	�����C��                                    BxY��`  
�          @��H@�(��p  ���R�;�C�}q@�(��QG��p����RC�`                                     BxY��  �          @�z�@�=q�u�����P(�C���@�=q�Tz��Q����RC��                                    BxY�֬  "          @�p�@�\)���Ϳ��|(�C��=@�\)�a��0  ���
C��                                    BxY��R  �          @�ff@8Q���=q�3�
���C�q@8Q��g��}p��"(�C�>�                                    BxY���  �          @�\)?�����\�����C���?��k���z��@��C���                                    BxY��  
Z          @�  @33��=q�Z=q�33C�|)@33�l������=�HC�xR                                    BxY�D  "          @�Q�@
=��\)�U�33C�@ @
=�g���Q��9  C��f                                    BxY��  
�          @�Q�@"�\�����J=q��RC�H@"�\�mp���33�0p�C�/\                                    BxY�.�  �          @�Q�@!G�����;���G�C��R@!G��z=q��p��'p�C�j=                                    BxY�=6  
�          @Ǯ@p���ff�E�홚C��@p��y�����H�0ffC��=                                    BxY�K�  �          @�p�@  ����33��33C�aH@  ���\�]p��ffC�3                                    BxY�Z�  "          @�(�@$z����\�33��ffC�l�@$z�����XQ���C�j=                                    BxY�i(  
�          @�z�@L������{��\)C�H@L���|(��\�����C���                                    BxY�w�  
Q          @�33@Fff��33������  C�B�@Fff���
�N�R� ffC�xR                                    BxY��t  
�          @�ff@Dz����
�0  ����C��3@Dz��j�H�{����C��q                                    BxY��  �          @���@�Q���
=��\)��RC���@�Q����ÿ��R�?�C�H�                                    BxY���  
�          @\@��\��G�>B�\?�\C��)@��\��p����\�ffC��                                    BxY��f  
�          @��H@������?
=@��
C���@��������R���
C��3                                    BxY��  T          @�z�@�����?�\)A(��C��{@����  ������C�`                                     BxY�ϲ  
Z          @�{@���R�\@�\A��
C�9�@���s33?��\A?�C�<)                                    BxY��X  
�          @�Q�@G�>�Q�@���B���A(�@G����H@��B�
=C���                                    BxY���  �          @�G�@6ff��  @���Bv�C���@6ff��p�@�Q�BaffC�XR                                    BxY���  	�          @ƸR@p  �Ǯ@�  B?�HC���@p  �1G�@xQ�B{C���                                    BxY�
J  "          @��
@�{�&ff@Z�HBG�C���@�{�]p�@#33A���C���                                    BxY��  	�          @�z�@����G�@��B.�RC�s3@���,��@s33B�C�!H                                    BxY�'�  
�          @�p�@fff>�@��BS(�@ ��@fff��p�@��\BIQ�C���                                    BxY�6<  
�          @�z�@c33�W
=@�BX��C�O\@c33����@��BHz�C��q                                    BxY�D�  
�          @ʏ\@�\)�{�?�p�A���C�` @�\)���\?G�@��
C��q                                    BxY�S�  "          @�=q@�ff����?�Q�AnffC�ٚ@�ff���\>���@`  C��q                                    BxY�b.  K          @�
=@����;�@>�RA�
=C���@����j=q@G�A���C���                                    BxY�p�  �          @�  @�  �ff@�p�B {C�z�@�  �Mp�@Y��A��\C�B�                                    BxY�z  �          @�\)@�33��p�@�z�B*(�C��H@�33�I��@hQ�BQ�C��                                    BxY��   �          @�ff@����
=@��RB#
=C�/\@����N�R@[�B z�C�޸                                    BxY���  
Z          @�z�@����+�@`  BG�C��)@����dz�@%A���C��=                                    BxY��l  �          @�
=@�{�8Q�@Z�HA�33C�:�@�{�o\)@��A��C���                                    BxY��  �          @�=q@����<(�@l��B	G�C���@����xQ�@,��A�=qC��=                                    BxY�ȸ  
�          @�
=@��
�@��@XQ�A�(�C�o\@��
�w
=@�A��C��                                    BxY��^  
c          @�Q�@��H�HQ�@X��A�\)C���@��H�~�R@ffA�
=C��                                    BxY��  "          @љ�@�=q�\(�@1G�A���C�@ @�=q���H?�z�Ak�C��{                                    BxY���  K          @�p�@�z��P  @J=qA��
C�1�@�z�����@�A�\)C�Ff                                    BxY�P  r          @љ�@�z��J�H@W�A��C�Ф@�z���Q�@33A���C��3                                    BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY� �              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�/B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�=�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�L�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�[4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�i�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�x�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��V              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�
�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�(H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�6�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�E�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�T:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�b�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�q�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��x              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�!N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�/�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�>�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�M@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�j�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�y2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�߼              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�7�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�FF              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�T�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�c�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�r8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�"               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�0�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�?L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�M�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�\�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�k>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��n              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�`  �          @�33@   �S�
@|��B.G�C���@   ���
@.{A�
=C��)                                    BxY�  �          @���@#�
�e�@qG�B"G�C��f@#�
���\@{A�C��)                                    BxY�)�  
�          @���@�
�QG�@�{B933C��@�
���@=p�A�=qC��
                                    BxY�8R  
�          @��\@ff�1G�@�p�BUffC��{@ff���H@e�B�C���                                    BxY�F�  �          @�G�@��2�\@��
BT33C�\)@����H@a�B=qC�z�                                    BxY�U�  
�          @�=q@   �8��@�z�BS�\C�N@   ��{@`��BffC���                                    BxY�dD  
0          @���?���+�@��\Ba=qC��?������@p��B!Q�C���                                    BxY�r�  �          @�{?��
�
=q@���By{C���?��
�g�@��B933C�z�                                    BxY���  
�          @���?Y�����
@��B�  C���?Y���E@�  BZ33C���                                    BxY��6  T          @��?��Ϳ�
=@���B�aHC���?����J=q@��BO  C��{                                    BxY���  �          @�Q�?
=��
=@���B�aHC�0�?
=�@  @�=qB`C���                                    BxY���  �          @��\?����R@��HB�C��R?��E�@��B_\)C�Ǯ                                    BxY��(  "          @��H>k�����@��B���C��q>k��=p�@�
=Bg�C�9�                                    BxY���  "          @�Q��녽��
@�\)B��fC>���녿��H@�ffB��C�B�                                    BxY��t  
�          @�z�.{��@��\B�u�C>ٚ�.{���H@���B��CxJ=                                    BxY��  T          @�  ���H�Ǯ@�{B�aHCZz���H���R@��B��C�#�                                    BxY���  "          @�녾�\)��@���B�.Cq�\��\)���@��HB��C�G�                                    BxY�f  
Z          @����\)�fff@��\B���C��=��\)�!G�@���Bx�C�AH                                    BxY�  "          @�=q�u��@���B�.Cwn�u�G�@�=qB��C�H                                    BxY�"�  �          @�=q���u@�\)B�#�C�"����'�@��Bwz�C��\                                    BxY�1X            @�Q�>aG���  @��HB�B�C��>aG��6ff@�Bj33C�@                                     BxY�?�  	�          @�p�?��R�p�@���BhC���?��R�fff@w
=B,�
C�t{                                    BxY�N�  �          @�  ?E����@�{B���C�p�?E��qG�@�\)B>�
C��=                                    BxY�]J  
�          @�
=?��\�33@��\Bz�RC���?��\�qG�@��B7�RC�W
                                    BxY�k�  \          @�=q?�{�?\)@��RBX�C�  ?�{��=q@b�\B
=C�9�                                    BxY�z�  
�          @��?�{�{@s33BX�C��=?�{�S33@:=qBC���                                    BxY��<  ~          @�G�@  �
=�@  �MC���@  � �׿����̣�C�#�                                    BxY���  �          @��@*�H�
�H��  �RG�C�� @*�H�&ff��ff�s��C�(�                                    BxY���  �          @��@N{�@����  �6p�C�q�@N{������
=�_�C���                                    BxY��.  
�          @Ǯ@R�\�7���ff�6=qC�l�@R�\�����(��]=qC��                                    BxY���  �          @���@Q��9����G��1�C�AH@Q녿�z���\)�Yz�C�Y�                                    BxY��z  
�          @��@Z=q�>{��G��'Q�C�z�@Z=q��ff�����Op�C��                                     BxY��   T          @���@?\)�>{���
�1z�C��
@?\)���
��33�]
=C�o\                                    BxY���  L          @��@4z��L���:�H�	�C��3@4z����qG��;�\C��                                     BxY��l  *          @�(�@5�b�\�G����HC�^�@5�)���QG�� =qC�|)                                    BxY�  
Z          @��
@Dz��5��Vff��
C���@Dz�����=q�E�C��\                                    BxY��  "          @���@K��1G������6  C�u�@K������ff�]  C���                                    BxY�*^  
�          @��@W
=�I������%ffC�j=@W
=��p����H�P{C�h�                                    BxY�9  �          @˅@`  �[���G��C���@`  �   ����J\)C�%                                    BxY�G�  
�          @Ϯ@j=q�Vff��� p�C��H@j=q��������J�C�o\                                    BxY�VP  
�          @�z�@X���8���xQ��$ffC��@X�ÿ��
���H�L{C��)                                    BxY�d�  T          @�{@U��.�R���
�/33C�U�@U���ff�����T�RC�W
                                    BxY�s�  
�          @��
@X���^{�P���C�0�@X���G����6�
C��                                    BxY��B  �          @��R@Z�H�HQ��hQ��\)C�@Z�H�����C(�C��                                     BxY���  "          @ƸR@n{�S�
�mp���\C�*=@n{���R����<��C��{                                    BxY���  "          @�ff@|(��X���U���C��f@|(�����
=�,��C���                                    BxY��4  
�          @�33@p  �HQ��h�����C�R@p  ��=q���:C���                                    BxY���  �          @�z�@hQ��Q���33��HC��
@hQ��=q��p��I�RC��                                     BxY�ˀ  M          @˅@Q��J=q���0��C��@Q녿�{��{�[��C���                                    BxY��&  
�          @�z�@9���+����
�?\)C��@9����
=��\)�gQ�C��                                    BxY���  
�          @�  @[��dz�����  C��f@[��*=q�U��C��                                    BxY��r  �          @�z�@A��U?!G�A   C�  @A��W���G����C���                                    BxY�  T          @���@Tz���{<#�
>\)C�` @Tz���
=�����e�C�3                                    BxY��  �          @���@Y���}p�?ǮA���C�U�@Y����  >u@!�C�XR                                    BxY�#d  �          @���@{���R@[�B�
C�޸@{��[�@\)A��C�q�                                    BxY�2
  
�          @��R@N�R�g
=?�z�A��C���@N�R�w�>W
=@��C��                                    BxY�@�  �          @�G�@P����=q?O\)AG�C�Z�@P�����
�   ��ffC�/\                                    BxY�OV  
Z          @�Q�@Fff����?��AtQ�C���@Fff��  =u?+�C��                                    BxY�]�  
�          @��@G
=��(�?���AQ�C�~�@G
=���������{C��3                                    BxY�l�  T          @�Q�@�\����@�\A���C���@�\����?Y��AQ�C���                                    BxY�{H  "          @��@��G�@z�A�33C�S3@��
=?�R@׮C�4{                                    BxY���  "          @��@7�����?�p�A��
C���@7���
=?(�@ӅC�Y�                                    BxY���  �          @�ff@Y���q�?���A��
C���@Y����ff?+�@��HC�~�                                    BxY��:  �          @�{@Vff�y��?��A�p�C�XR@Vff��Q�>�@��HC�R                                    BxY���  
�          @�\)@Mp���Q�?�A��C�Q�@Mp���z�?   @�\)C�
                                    BxY�Ć  M          @��@N�R��33?�A��C��@N�R��
=>��@��C��=                                    BxY��,  �          @�(�@_\)����>��@��HC�q�@_\)�~{�L���
ffC��H                                    BxY���  T          @�
=@���n�R�
=q��{C��f@���7��Mp����C���                                    BxY��x  "          @���@|(��}p���\��(�C�p�@|(��C33�Y���	p�C�"�                                    BxY��  �          @�(�@�{�s�
��ff���\C��@�{�C�
�8����C��{                                    BxY��  "          @Ǯ@����s33��G��:�HC��)@����N{�Q���C�                                      BxY�j  �          @��
@|���~�R�������C�b�@|���Mp��=p���C�j=                                    BxY�+  �          @�@qG��z�H�\�yp�C���@qG��P  �*=q����C���                                    BxY�9�  �          @��@����׾�
=���C�3@��l(���33���C�L�                                    BxY�H\  �          @���@����\)��
=��z�C���@����j=q������
C��=                                    BxY�W  �          @�
=@����p��>L��?�(�C�j=@����hQ�z�H�=qC��                                    BxY�e�  
�          @�=q@�33�p  ?��@���C��)@�33�o\)�#�
���HC���                                    BxY�tN  T          @�G�@�G��S33@{A�z�C��R@�G��u�?�=qA%G�C��                                    BxY���  
�          @�33@���&ff@P  B�\C��H@���^�R@�A���C��                                     BxY���  �          @�Q�@�������s33�C�Z�@����vff�  ��z�C�"�                                    