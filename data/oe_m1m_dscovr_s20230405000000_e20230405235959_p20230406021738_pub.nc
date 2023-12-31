CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230405000000_e20230405235959_p20230406021738_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-06T02:17:38.438Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-05T00:00:00.000Z   time_coverage_end         2023-04-05T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxt�i   "          @����ff�.{?�\)Au�CT���ff�A녾���8Q�CW�
                                    Bxt�w�  �          @�ff�������?s33A'�
CO������!녾�ff��
=CQJ=                                    Bxt�L  �          @��
����4z�?c�
A ��CV!H����7��8Q��{CV�{                                    Bxt��  
�          @��R��Q��,��?���A�\)CU���Q��QG�>#�
?�G�C[&f                                    Bxt룘  �          @�\)��p��
=@�
A��CQxR��p��E�?�@�CX}q                                    Bxt�>  
�          @�p�����*=q?�  A`��CT
����:�H�����h��CV�\                                    Bxt���  
�          @�G�����#�
?��RAfffCS�H����5���\)�N{CV��                                    Bxt�ϊ  T          @�Q��G���z�?n{A"ffCs���G���{������Cr�                                    Bxt��0  
�          @��
�=p����R?�(�A~ffCj��=p���33�s33�$z�Ck��                                    Bxt���            @���L(��r�\?�A��Ce���L(���ff��
=��Q�ChǮ                                    Bxt��|  
�          @��
�<���{�?�(�A��HCi(��<�����
�O\)���Cjk�                                    Bxt�
"  �          @�33�����?�G�A��Ct�������H�B�\�	�Cv+�                                    Bxt��  �          @�����(���=q?��\Aj=qCyT{��(���녿�ff�n�HCyL�                                    Bxt�'n  "          @�z���}p�?���A�(�Cr+����z�����G�Ct�\                                    Bxt�6  �          @�Q����s33?��A�{Cp������
=��p���\)Cs\)                                    Bxt�D�  
�          @��H���
�j=q@
=A�ffC�}q���
��
=������C��3                                    Bxt�S`  
�          @��?�G�����?�\)A�p�C�w
?�G���33�k��0��C�1�                                    Bxt�b  �          @�ff?fff�p  @'�B�C��f?fff��=q>�\)@^{C��R                                    Bxt�p�  �          @��?s33��p�?�  A�33C�3?s33���O\)�ffC���                                    Bxt�R  �          @�  ?(������?�
=A��
C�e?(������
=��C��                                    Bxt��  "          @�z�?����y��@G�A�G�C�4{?�����zᾞ�R�s33C�*=                                    Bxt윞  �          @���?��
�dz�@\)A�(�C��?��
���R=u?=p�C��                                    Bxt�D  �          @�\)?�\�e�?�(�A�Q�C�,�?�\���\�8Q��33C��)                                    Bxt��  �          @��R?�{�_\)?�
=A���C�e?�{�vff�\��
=C�Y�                                    Bxt�Ȑ  �          @�z�?��H�tz�?�G�A��C�w
?��H�����333��C��                                    Bxt��6  
�          @�ff?ٙ��B�\@
=A��C���?ٙ��l(�>aG�@J=qC�XR                                    Bxt���  T          @�p�?�
=�:=q@�B�\C��?�
=�p  ?�@�33C�R                                    Bxt��  
�          @�z�@%����@.�RB.C��@%�(�?ٙ�Ạ�C�N                                    Bxt�(  "          @���@?\)?�@��B33A"=q@?\)�J=q@z�B�HC��q                                    Bxt��  
�          @��H@l(�>Ǯ?���A�=q@�=q@l(����?�Q�A��C�˅                                    Bxt� t  
�          @�=q@L(���?ٙ�A�p�C�7
@L(���?�@��
C�W
                                    Bxt�/  T          @|��@:�H�	��?�=qA�ffC��{@:�H�*=q>u@\��C���                                    Bxt�=�  
�          @|��@333�*�H>���@���C�/\@333�\)����xz�C�,�                                    Bxt�Lf  �          @�
=@z��W
=?�{A�Q�C��\@z��e��z���C��                                    Bxt�[  "          @���@*�H�7�?�33A�
=C�t{@*�H�C33�   ��G�C��{                                    Bxt�i�  T          @�ff?�(��\��?�=qA��C���?�(��y���W
=�@  C���                                    Bxt�xX  "          @���@+��7�>�(�@���C���@+��+���{���C��                                     Bxt��  
�          @���@<���p�?G�A7�C��@<��� �׿(��{C��=                                    Bxt핤  
�          @�@r�\�=p����H��(�C�� @r�\>��У����\?�33                                    Bxt��J  
�          @��R@h�þk���(����C�+�@h��?0�׿˅��(�A+\)                                    Bxt���  T          @��@HQ�?Ǯ�   �p�A�p�@HQ�@'������p�B�                                    Bxt���  
�          @��@C33?�������A��@C33@�������(�B�                                    Bxt��<  T          @���@H��?�(���(���\A�(�@H��@G��n{�W�
Bff                                    Bxt���  T          @r�\?�?n{�3�
�Bi33?�@  ��
=�(�B�Q�                                    Bxt��  �          @��H@L�;8Q��z���C�k�@L��?�������
A��\                                    Bxt��.  
�          @�\)@�=q    �aG��C\)<#�
@�=q>�ff�B�\�'33@˅                                    Bxt�
�  "          @�(�@r�\�W
=��\)���RC�o\@r�\?&ff���R���
A��                                    Bxt�z  "          @���@o\)=��Ϳ�
=���
?��
@o\)?s33�������AdQ�                                    Bxt�(   T          @���@vff>\��ff���@��
@vff?�  ������
=A���                                    Bxt�6�  
�          @��@~�R?�G����
��p�Ac�
@~�R?�33�J=q�&�HA�ff                                    Bxt�El  �          @�{@w��W
=�z�H�^�HC��q@w��aG����
���C�c�                                    Bxt�T  
�          @w�@R�\���ÿ����C��@R�\��
=�����  C�Z�                                    Bxt�b�  
�          @y��@a녾��˅��G�C���@a�?333��
=����A4z�                                    Bxt�q^  T          @~{@N{>W
=�  ��@j=q@N{?�=q��=q��G�A��                                    Bxt�  T          @���@j�H>����
�ͅ@�(�@j�H?��ÿ����  A�=q                                    Bxt  
�          @���@\�;�����Q���  C���@\��?:�H������Q�A@Q�                                    Bxt�P  
�          @��@n{���������C��@n{>�  ���H��\)@s�
                                    Bxt��  �          @�z�@z=q��z῔z���C�޸@z=q>�{��33��{@�                                      Bxt  T          @��\@_\)=�G���p����?��@_\)?�{�������A��H                                    Bxt��B  �          @{�@`�׿p�׿�
=����C���@`�׽��
���H��G�C�aH                                    Bxt���  T          @~{@c33��\)���
���C�8R@c33���R�����C�xR                                    Bxt��  
�          @~�R@i�����Ϳ����=qC��)@i���Ǯ��  ��=qC��
                                    Bxt��4  T          @���@`�׿���=��
?�(�C�S3@`�׿�{�Y���G\)C��
                                    Bxt��  "          @~{@Z�H��Q�>���@��C�1�@Z�H���333�$  C��
                                    Bxt��  T          @���@c�
��  �L���6ffC���@c�
��zῇ��w�
C�9�                                    Bxt�!&  "          @�33@p�׿���{�|  C�Q�@p�׾�ff��ff���\C��)                                    Bxt�/�  
�          @��@s33�Q녿�����Q�C��@s33    ������
=C��)                                    Bxt�>r  
�          @��@k������=q�иRC�` @k�>�녿����
=@�(�                                    Bxt�M  "          @�ff@mp��B�\������HC�<)@mp�>����
=��p�@\)                                    Bxt�[�  �          @�
=@u�B�\��=q��  C�j=@u>����  �Ù�@{                                    Bxt�jd  �          @�ff@c33���ÿ�  ��Q�C�Ф@c33��  �����C��q                                    Bxt�y
  �          @�\)@r�\���
���
��ffC�b�@r�\��G�����p�C�#�                                    Bxt  
�          @�33@}p���(��У����C��\@}p�>���\)���H@��H                                    Bxt�V  �          @�=q@}p������Q�����C�%@}p�?#�
�˅���A�                                    Bxt��  T          @�Q�@xQ�=L�Ϳ޸R��G�?0��@xQ�?k���p����RAU��                                    Bxtﳢ  T          @�z�@|(��L�Ϳ����ə�C�� @|(�?c�
��\)��p�AL(�                                    Bxt��H  
�          @���@{���33������ffC�t{@{�?   �Ǯ���@��                                    Bxt���  �          @�z�@}p�>�׿�ff�¸R@ڏ\@}p�?��ÿ�����=qA�p�                                    Bxt�ߔ  T          @�(�@mp�?333��癚A*=q@mp�?�33������G�A�Q�                                    Bxt��:  �          @��@k�?J=q�������AA@k�?����G�����A�                                    Bxt���  "          @��R@tz�?.{�
�H��  A Q�@tz�?�
=�����A���                                    Bxt��  �          @�ff@tz�?0���
=��ffA#\)@tz�?�z´p���ffA���                                    Bxt�,  "          @q�@Mp��Y�����H��ffC�� @Mp�>���z���p�@                                    Bxt�(�  �          @�{@b�\�z�H�����z�C�AH@b�\=����Q���z�?�Q�                                    Bxt�7x  T          @�ff@u�\)�ff��p�C��)@u?��ff��
=A\)                                    Bxt�F  
(          @�z�@u>\)�%��H@@u?��
�H��G�A�ff                                    Bxt�T�  �          @���@�
@(��C�
�/�\B.(�@�
@[���ff��ffB`\)                                    Bxt�cj  "          @�Q�@&ff�HQ�?O\)A3�C��R@&ff�G
=�^�R�Ap�C��                                    Bxt�r  
(          @�z�@�c33?��
A��C�@ @�u��ff��z�C�H�                                    Bxt���  
�          @�ff@z��a�?�
=A�z�C�1�@z��y�����R��G�C��q                                    Bxt��\  
�          @�Q�?��fff@�
A�33C�p�?����ͼ��
�aG�C��
                                    Bxt�  
�          @�\)@��c�
?�Q�Ax��C��{@��k��@  ���C�*=                                    Bxt�  T          @�  @N�R�HQ�#�
��G�C���@N�R�-p�������33C��                                    Bxt�N  �          @��R@L���C�
��\)�Y��C�(�@L���'���=q��p�C�]q                                    Bxt���  
�          @�
=@g
=���������=qC���@g
=�E��  ��G�C��
                                    Bxt�ؚ  
�          @��@c�
?����Fff��RA��R@c�
@!G��Q���B33                                    Bxt��@  �          @��@^�R?���E�ffA�{@^�R@,��� ����ffB��                                    Bxt���  
�          @�G�@e?��\�>�R�\)A
=@e@��z���{B                                    Bxt��  "          @�=q@P  >����aG��<z�@��@P  @�\�8����B Q�                                    Bxt�2  �          @��H@^�R?�Q��C�
�(�A�\)@^�R@#�
�z��̸RB(�                                    Bxt�!�  
�          @�\)@{�>Ǯ�!��p�@���@{�?˅�   ��(�A�=q                                    Bxt�0~  v          @�Q�@~�R=��%���H?޸R@~�R?��������A��                                    Bxt�?$             @�(�@��#�
�   ��
=C��q@�?����(����
A��\                                    Bxt�M�  	�          @��
@��\���
�����C��)@��\?����z���33A\(�                                    Bxt�\p  T          @���@�  ��R�ff��{C��)@�  ?   �Q��Ӆ@�{                                    Bxt�k  �          @��
@|(���33���R��  C��@|(��333�
=�ޏ\C���                                    Bxt�y�  �          @��@mp��=q���R�z�RC�y�@mp����R�G�����C�{                                    Bxt�b  "          @�@dz��:=q�=p���\C�e@dz��
�H����  C�]q                                    Bxt�  "          @�
=@Mp��Dzᾮ{��C�"�@Mp�� �׿���  C�f                                    Bxt�  �          @�p�@?\)�O\)��
=��{C�\)@?\)�'
=���H��{C�o\                                    Bxt�T  "          @�{@j=q�!G���33��=qC�"�@j=q>Ǯ���H��(�@�Q�                                    Bxt���  �          @���@n{��z�Ǯ���RC��
@n{����33��G�C�j=                                    Bxt�Ѡ  �          @��
@H����
��ff�pz�C��@H�ÿ�  �33���
C�8R                                    Bxt��F  "          @y��?���G�?��A�33C��q?���O\)����(�C�9�                                    Bxt���  
�          @xQ�=L���3�
?�p�A��HC��\=L���C33���R��\)C���                                    Bxt���  T          @���}p��Tz�@+�B�HC}O\�}p���ff?5A��C�XR                                    Bxt�8  �          @���!G��a�@�A�z�C����!G�����>8Q�@��C���                                    Bxt��  
�          @���?��R�p��?Tz�A8��C�˅?��R�l(���\)�{
=C���                                    Bxt�)�  �          @��R?޸R�q�?B�\A%��C�\)?޸R�j�H��
=���
C��\                                    Bxt�8*  �          @�{?�(��x��>�G�@���C�S3?�(��g
=��G���C��                                    Bxt�F�  �          @��R@*=q�.{����	�C��@*=q>��
��p����@�                                    Bxt�Uv  T          @��@R�\?�\)�AG��$(�A�  @R�\@���ff�ظRB��                                    Bxt�d  T          @�@QG�?k��K��,(�Az{@QG�@ff����Q�Bp�                                    Bxt�r�  �          @��@%�@�\�O\)�,�RB&33@%�@dz��(���=qBX��                                    Bxt�h  �          @��R@+�@=q�A�� Q�B'��@+�@e���p����RBT�                                    Bxt�  "          @�{@,(�@\)�Fff�&33B�H@,(�@]p��У����HBPz�                                    Bxt�  T          @���@#33@�QG��3\)BQ�@#33@Z=q������{BU
=                                    Bxt�Z  "          @��@,��?����L(��.�HB�@,��@P  ��������BH�                                    Bxt�   T          @��@(��?�
=�P���3z�Bp�@(��@P�׿���ffBL                                      Bxt�ʦ  �          @�(�@;�?�p��L(��0�HA�
=@;�@5����  B0��                                    Bxt��L  �          @�=q@@  ?n{�Tz��:G�A�  @@  @�H����
=Bz�                                    Bxt���  "          @��@5?��
�U��;��A��
@5@.{�33��Q�B/G�                                    Bxt���  	�          @�\)@(Q�O\)�aG��P  C�p�@(Q�?}p��^�R�L=qA���                                    Bxt�>  
�          @�(�@8Q쿀  �HQ��6�
C�n@8Q�?!G��N{�>�AD��                                    Bxt��  
�          @��R@P  �.{�X���8�C�w
@P  ?�p��C33�"�A�\)                                    Bxt�"�  �          @���@S33��  �^�R�9��C��
@S33?����J�H�%ffA�=q                                    Bxt�10  "          @���@U�����X���5�C�w
@U�?�G��J�H�&�A�Q�                                    Bxt�?�  T          @��\@G�����S33�8�C��=@G�?���K��0G�A��                                    Bxt�N|  "          @�{@Q녾�{�S�
�4\)C���@Q�?��
�Dz��$Q�A���                                    Bxt�]"  �          @��@Fff���P  �7ffC��@Fff?�ff�HQ��.�A�                                      Bxt�k�  
�          @�@8��?�z��@  �,�A�@8��@*�H���H��  B+                                      Bxt�zn  
�          @�{@HQ�?=p��AG��,�
AU��@HQ�@ff�33��G�B33                                    Bxt�  
�          @��@1G���{�J�H�7{C��=@1G�>��
�\(��K@У�                                    Bxt�  �          @�(�@�\���'�� Q�C��q@�\�(��S�
�Y�
C���                                    Bxt�`  �          @��?��Vff�33���C���?������g��k\)C�4{                                    Bxt�  T          @���?����AG��4z��!��C��3?��ÿ����z�H��C��                                    Bxt�ì  T          @�33?�\)����5�/�
C��)?�\)�O\)�j=q�~��C��
                                    Bxt��R  �          @��@(Q���
�����HC��@(Q�&ff�HQ��DQ�C�
                                    Bxt���  �          @�G�@'��'
=�У�����C��\@'��\�*�H�%C��                                    Bxt��  �          @�Q�?�33�C33��G�����C��)?�33�����?\)�A��C��                                    Bxt��D  
�          @�G�?z��hQ�xQ��k�C���?z��/\)�$z��*{C���                                    Bxt��  �          @��þ����u�^�R�K�C�
=�����>{�%��#  C�.                                    Bxt��  
�          @�=q���y���!G��=qC�"����H������33C���                                    Bxt�*6  "          @���?s33�w������o�C��?s33�:=q�1G��(\)C�
=                                    Bxt�8�  �          @��?   �tz�u�`  C��
?   �:�H�)���'G�C�ٚ                                    Bxt�G�  �          @���?.{�n�R��������C�/\?.{�*=q�<���<Q�C�5�                                    Bxt�V(  
�          @��?L���z�H��G��ǮC�˅?L���P  ��R��C��                                    Bxt�d�  �          @��H@I���333�7��&C���@I��?:�H�7
=�&33AQ�                                    Bxt�st  "          @�ff@333��  �/\)�#p�C��@333���G��@�C��                                    Bxt�  
�          @���@>�R��{�7��(=qC��3@>�R>�p��C33�5�
@���                                    Bxt���  �          @�ff@1G���  �9���.�C��3@1G�>�  �I���B(�@�z�                                    Bxt��f  �          @��H��׿��?G�A��Czs3��׿\=��
@333C}                                    Bxt��  "          @��\�w
=��=q@w�B/�HCC���w
=�0��@:�HA�z�CW�\                                    Bxt���  S          @�z������Y��@tz�B*��C?�{�����"�\@?\)A�CT                                    Bxt��X  
�          @��\��\)�!G�@e�B�C<����\)��R@7�A��CO�                                     Bxt���  �          @��
���R�.{@UB{C<�R���R�
=q@(��A�Q�CMٚ                                    Bxt��  �          @����qG���
=@W�B&Q�C:^��qG���Q�@1�B33CO33                                    Bxt��J  
�          @��
�`  �!G�@QG�B*33C>B��`  ��@&ffB(�CR��                                    Bxt��  �          @�=q�[��W
=@fffB6{CA� �[��=q@333B��CW)                                    Bxt��  �          @���W
=��ff@l(�B9G�CEB��W
=�(��@1�B\)CZ{                                    Bxt�#<  �          @���c�
��@k�B1�HCF&f�c�
�/\)@.{A���CY�\                                    Bxt�1�  �          @�(��hQ�L��@\(�B+=qC@ff�hQ��33@+�A��CTY�                                    Bxt�@�  
�          @���{��0��@>�RB�
C=�f�{���p�@z�A�z�CN                                    Bxt�O.  �          @���  ��G�@0��A���CC�)��  ���?�\)A�33CO�R                                    Bxt�]�  
�          @����������@'
=A���CIxR�����2�\?��RAy�CS��                                    Bxt�lz  
�          @����33��{@%�A�RCK)��33�5?�
=Aw
=CU
                                    Bxt�{   "          @�=q������@0��A�(�CO�����I��?�p�A��CZ\                                    Bxt���  "          @�33����<��@7�A���CWG�����{�?�Q�A<z�C_^�                                    Bxt��l  T          @��H�tz��C33@,��A�33CZ���tz��|(�?�  A%G�Ca�H                                    Bxt��  �          @��R�N�R�0  @Z=qB{C\ff�N�R��  ?�\A��
Cg)                                    Bxt���  T          @�z��^�R�#33@Mp�B�CX+��^�R�o\)?�Q�A��\Cb�q                                    Bxt��^  T          @����h�ÿ�{@S33B��CO)�h���J=q@�A��HC]�                                    Bxt��  �          @�G��q��\��@Tz�B�C^k��q����H?���AP��Cf�                                    Bxt��  
Z          @�  �r�\�j�H@UBffC`��r�\��G�?�ffA@z�Cg��                                    Bxt��P  �          @��j�H�p  @P��A��\Ca���j�H��=q?�Q�A2{Ch��                                    Bxt���  "          @�  �c�
����@K�A��RCd���c�
��G�?�  A
=Cj                                    Bxt��  �          @Ǯ�X����ff@G
=A�Cg��X������?\(�@�
=Cl�                                    Bxt�B  
Z          @Ǯ�c�
��
=@:�HA�
=Ce�
�c�
���\?0��@ʏ\Ck�                                    Bxt�*�  "          @Ǯ�^�R����@;�A��
Cf���^�R��(�?+�@ƸRCkٚ                                    Bxt�9�  �          @�G��\(���z�@{A�\)Ci}q�\(���Q�=���?s33Cl�{                                    Bxt�H4  T          @��=p���{@.{AиRCmǮ�=p����>���@3�
Cq\)                                    Bxt�V�  
�          @�G��E��{@,(�A�33Ck+��E��p�>Ǯ@n{Co(�                                    Bxt�e�  
�          @�(��@  ���@?\)A�Ck���@  ��G�?+�@ʏ\Cpp�                                    Bxt�t&  �          @����2�\��ff@<��A�Q�Cm�H�2�\����?!G�@�\)Cr5�                                    Bxt���  "          @�=q�<(����@<��A�RClW
�<(�����?&ff@���Cpٚ                                    Bxt��r  �          @�33�A���33@'�A�p�Cl���A�����>�=q@%�Cp+�                                    Bxt��  c          @�z��I������@p�A�33Ck�f�I����Q�=�G�?�ffCo�                                    Bxt���  �          @����7�����@6ffA��Co��7���=q>���@g�Cr�R                                    Bxt��d  T          @�G��!���ff@:�HA��
Cr��!����>���@h��Cv33                                    Bxt��
  �          @ȣ��p����@8Q�A���Cs�=�p���  >�{@I��Cv�)                                    Bxt�ڰ  T          @ʏ\��
��
=@:=qAۅCx}q��
��
=>�\)@!G�Cz�q                                    Bxt��V  �          @˅�(����H@8��AظRCtW
�(����H>��
@7�CwJ=                                    Bxt���  �          @�ff�3�
���@"�\A���Cp+��3�
����=�Q�?Y��Cs�                                    Bxt��  
�          @���Dz����@)��A���ClL��Dz����>���@Dz�Co�                                    Bxt�H  "          @�(��Fff��33@$z�A��Cl\�Fff��Q�>��@p�Co�=                                    Bxt�#�  
�          @�Q��K����@.{A�{Ck�H�K���(�>\@\(�CoaH                                    Bxt�2�  "          @Ǯ�AG����
?�33A��Coz��AG����\)��p�Cp�3                                    Bxt�A:  �          @���0  ��=q@*=qA�33CpJ=�0  ��  >��@=qCsp�                                    Bxt�O�  �          @���5�����@'
=AȸRCou��5����R>aG�@�
Cr�{                                    Bxt�^�  "          @�=q�1G���Q�@!�A�G�Co�1G���z�>.{?�{Cr�                                     Bxt�m,  �          @ə���(���Q�@p�A���CzY���(���p���G�����C{�)                                    Bxt�{�  �          @˅�
=���@ ��A���Cy���
=���R�0����ffCz�                                     Bxt��x  �          @�  �33��(�?�\A��Cz
=�33���H�aG����Cz�3                                    Bxt��  
�          @�\)����z�?��HA33Cz:�����=q�n{�	�Cz��                                    Bxt���  
�          @���Q�����?�=qAm��C{{��Q����׿�ff���C{s3                                    Bxt��j  "          @ȣ�����
=?���AV=qCy�������ÿ����0  Cy�H                                    Bxt��  �          @\� ����  ?���Atz�Cy�3� �����Ϳu�=qCzn                                    Bxt�Ӷ  
�          @Ǯ��R���R?�{A$Q�Cx����R���
���R�]��Cx^�                                    Bxt��\  "          @�  ��
��
=?�  A
=Cx���
���\�˅�lQ�Cw��                                    Bxt��  �          @Å�Q���(�?�ffA{CyE�Q����ÿ�G��dQ�Cx��                                    Bxt���  
�          @\�G�����?k�A\)CzQ��G�����У��x��CyǮ                                    Bxt�N  
�          @��׿����p�?.{@�Q�C{� ������޸R��33Cz�3                                    Bxt��  T          @�33��Q����?   @��C|LͿ�Q���  �������\C{G�                                    Bxt�+�  
�          @����
=���?5@�p�C~Y���
=���������{C}�                                    Bxt�:@  
�          @�G�������ff?uA&�RC|h�������(����
�`  C|(�                                    Bxt�H�  
�          @����\���\?�\)A���Cw���\��33������RCx�3                                    Bxt�W�  
�          @�녿�\)�mp�@33A�ffCvff��\)��\)>�  @HQ�Cy�                                    Bxt�f2  "          @��
���z=q?�  A�{Cz����=q�������Cz�q                                    Bxt�t�  �          @�G��B�\�w�?�
=A�(�C�w
�B�\��
=����   C���                                    Bxt��~  
�          @�=q��
=�u?��\A�G�Cy�=��
=��Q�   ��  CzW
                                    Bxt��$  
�          @�ff���
��(�?�G�A��C�q���
���׿!G��\)C�7
                                    Bxt���  T          @�������l(�?
=q@�C}h������c�
������RC|�)                                    Bxt��p  "          @�33�(Q��AG�?@  A)�Ce��(Q��C�
�(��
ffCeQ�                                    Bxt��  �          @�=q�8Q��
=?^�RAS
=C[Q��8Q�� �׾aG��P��C]�                                    Bxt�̼  �          @���Z=q�5?���A��RC[�=�Z=q�P  >u@:=qC_��                                    Bxt��b  �          @�z��j�H�#33?�Q�A�CV�\�j�H�HQ�?+�@�p�C\��                                    Bxt��  "          @����g��333?���A��\CY���g��E<#�
=��
C\s3                                    Bxt���  �          @���U��C33��p���(�C^p��U��&ff��\)��  CZ�                                    Bxt�T  	�          @�33�o\)�{?��A�=qCUk��o\)�?\)?z�@��
CZ�                                    Bxt��  "          @�Q��@  �Q�?ٙ�A�
=CZu��@  �8Q�?�@�Q�C_��                                    Bxt�$�  �          @�{�-p��<(�?�33A�{CcJ=�-p��I���L���0��Ce@                                     Bxt�3F  
Z          @�z��@  ��(�@G�B(�CUE�@  �/\)?�p�A�Q�C^h�                                    Bxt�A�  T          @��?�����
��=q��C�e?���Z�H�g
=�8�\C���                                    Bxt�P�  �          @�
==���
=��G���p�C��H=��i���Vff�*{C���                                    Bxt�_8  "          @��þk����ÿ�p���G�C����k��n{�Vff�'C�@                                     Bxt�m�  
�          @��R�   ��=q��ff�B�HC��   �|(��>{�=qC�ff                                    Bxt�|�  �          @�  �\)��z�Y���%��C��Ϳ\)�w
=�.{��C���                                    Bxt��*  
�          @�
=�O\)��Q쿆ff�MC��׿O\)�j=q�5�ffC��q                                    Bxt���  
�          @�p��(�����׾Ǯ����C�Ф�(���|(��\)��(�C�7
                                    Bxt��v  
�          @�(������33>���@���C�s3�����녿�\)��C�H�                                    Bxt��  
�          @�������H>�G�@��
C�p��������G���(�C�@                                     Bxt���  
�          @�z�����H>�33@���C�s3����=q�˅���\C�:�                                    Bxt��h  �          @�
=�(�����>�(�@�=qC���(�����
=���C���                                    Bxt��  
�          @�Q�G�����?\)@�C���G���\)�����C��H                                    Bxt��  "          @�z�(���\)?�Q�A�Q�C����(������W
=�*=qC�0�                                    Bxt� Z  
�          @�  �L���G�@\��Bb�C�g��L���_\)@p�B(�C��q                                    Bxt�   �          @�{��Q���@l(�B{ffC��3��Q��P  @%B{C�>�                                    Bxt��  �          @��׿n{�*�H@I��B@z�Cz޸�n{�n�R?�\A�(�C��                                    Bxt�,L  T          @�(��fff�A�@@��B.(�C}p��fff��  ?��RA�=qC��                                    Bxt�:�  
Z          @����
�Tz�@+�BQ�C|�쿃�
��(�?�=qAc\)C��                                    Bxt�I�  "          @�z�J=q�W�@(Q�B�
C�ff�J=q����?��\AXz�C��)                                    Bxt�X>  �          @�p������)��@\��BP�C��������u�@z�A�C�H                                    Bxt�f�  T          @�33�Y�����@]p�BTG�Cz�{�Y���i��@	��A��HC�l�                                    Bxt�u�  
�          @���
=q�@  @8��B.=qC��\�
=q�z�H?�A�(�C�)                                    Bxt��0  T          @�����G��b�\@ffB\)C�l;�G���{?333A�C��R                                    Bxt���  T          @��>����\(�@�B	�RC���>������?E�A(z�C�H�                                    Bxt��|  T          @�(�>�p��\)?�p�A���C���>�p���33�#�
��C�o\                                    Bxt��"  "          @��?��\��  ?�G�A�
=C�  ?��\���׾u�FffC���                                    Bxt���  "          @�\)?������?���A�{C�]q?�����\�.{���C�޸                                    Bxt��n  
�          @���\)�w�?O\)A2=qCzxR��\)�w��Q��5��Czs3                                    Bxt��  �          @����J=q�s�
�\)�˅CfL��J=q�[���z����HCcQ�                                    Bxt��  �          @�����\��{<��
>�=qCuL���\���\��\���CsaH                                    Bxt��`  
�          @�����R��
=        Cu����R��33��ff��Q�Ct�                                    Bxt�  
Z          @�
=�����  ��Q���33Co����aG���
=����Ck��                                    Bxt��  
Z          @���*=q�n�R>�?��Cj���*=q�^{������Ch��                                    Bxt�%R  �          @����{�s33<��
>��RCl�3�{�_\)��  ��Q�Cj�3                                    Bxt�3�  �          @����?\)�W�>.{@��Cdff�?\)�J=q��Q��t(�Cb��                                    Bxt�B�  T          @�z���s�
>��@[�Cq=q��fff���
����Co�{                                    Bxt�QD  T          @�ff��\)��33>��
@�Cxz��\)�x�ÿ�����HCwn                                    Bxt�_�  "          @��c�
���>���@�  C���c�
��Q쿴z���G�C��                                    Bxt�n�  "          @���#�
���
�Q��)p�C��
�#�
�l(��\)��HC���                                    Bxt�}6  "          @�ff>�p���녿z�H�K�C�k�>�p��dz��&ff��
C��                                    Bxt���  "          @���������
=��\)C�K������QG��>{�(�
C�R                                    Bxt���  �          @�\)?���j=q��Q����C���?���,���?\)�6��C�k�                                    Bxt��(  �          @�?�Q��a녿G��.�\C�c�?�Q��<(��
=��
=C���                                    Bxt���  �          @�=q@{�Y����\)�p��C��{@{�Fff��33��G�C�Ф                                    Bxt��t  �          @�Q�?�
=�dz῏\)�w
=C�5�?�
=�5���33C��                                    Bxt��  T          @���?����_\)��Q����\C��3?����)���,���z�C�+�                                    Bxt���  T          @��
?����Z�H�^�R�F=qC�ٚ?����3�
������C�e                                    Bxt��f  �          @���@ ���I�����R��  C�<)@ ���z��&ff�!ffC�o\                                    Bxt�  T          @�  @
=q�
=�z��z�C�B�@
=q���\�C33�JQ�C��\                                    Bxt��  T          @���@.�R���z��
�C�t{@.�R�^�R�8Q��4�\C�5�                                    Bxt�X  T          @x��@?\)��ff����(�C��@?\)?
=q����A"�\                                    Bxt�,�  
�          @{�@6ff���(Q��(�C�` @6ff?   �(���)��A�R                                    Bxt�;�  �          @l��?�Q�O\)�0����\)C�7
?�Q���s33���HC�R                                    Bxt�JJ  "          @u�p�׿�z�@H��B_�\Cs޸�p���>�R@
=qB
��C|��                                    Bxt�X�  �          @{��s33�33@L��B[�Cu=q�s33�HQ�@
=qBC}+�                                    Bxt�g�  �          @��
��z���
@P��BTffCZ�q��z��*�H@=qBffCjp�                                    Bxt�v<  �          @�\)��ÿ�p�@S�
BO
=CV�����(��@�RB\)Cf�3                                    Bxt���  �          @�����ff��33@^�RB[��C^���ff�7
=@%�B(�Cm�\                                    Bxt���  
�          @�G�����H@3�
B%�\CeG����S�
?�A��RCmٚ                                    Bxt��.  
�          @�
=�z���
@0��B&��Cd��z��L(�?�A�
=Cm                                      Bxt���  T          @��Ϳ�=q�G�?�(�A�z�Co���=q�h��?!G�A�
Cs@                                     Bxt��z  �          @~�R��33�W
=?���A�z�Cs�f��33�c�
�.{� ��Cu5�                                    Bxt��   "          @k�>�p��=p�?xQ�A���C��\>�p��Fff�k�����C�e                                    Bxt���  
�          @}p���33�:=q?��HA���Cl�{��33�O\)>�  @qG�Co�f                                    Bxt��l  T          @w
=��{�QG�?���A�
=Cwc׿�{�[���  �uCx\)                                    Bxt��  �          @hQ쿨���E?�  A�G�Cv�������U����
��  Cxn                                    Bxt��  
�          @l(���\���R?�p�B��C\����\�%?�=qA�33Cd��                                    Bxt�^  "          @|(����H�h��?��RA���C�#׾��H�u�aG��Tz�C�W
                                    Bxt�&  T          @��\��  �_\)?���A�=qCzJ=��  �n�R���Ϳ�Q�C{u�                                    Bxt�4�  �          @}p��
�H�>{?�A���Ci޸�
�H�L�ͽ#�
���Ck�\                                    Bxt�CP  
�          @�\)����e�?�  A���Cv�q����xQ�=L��?�RCxaH                                    Bxt�Q�  "          @�녽��
�z�@I��BV��C��ὣ�
�U�@33A���C�J=                                    Bxt�`�  �          @�>��
�]p�@��BG�C��>��
����?=p�A&=qC�E                                    Bxt�oB  T          @�
=?���|��?O\)A3
=C��?���}p��:�H� ��C�}q                                    Bxt�}�  T          @�?�p��q�?p��AO�C���?�p��w
=�
=q��C��                                    Bxt���  T          @�?���w�?��A�\)C�h�?�����H�L���.�RC�,�                                    Bxt��4  
�          @���?   �O\)@33Bp�C�o\?   �w
=?n{AW�C���                                    Bxt���  �          @���>L���l��?�ffA�C��=>L����Q�=�\)?p��C�k�                                    Bxt���  T          @��?�G��N{@�
B	p�C��)?�G��vff?s33AV�RC�aH                                    Bxt��&  T          @�(�?��R�L��@��B33C���?��R�s�
?h��AL��C��                                    Bxt���  "          @�?����W�@�\A���C��?����xQ�?&ffA�
C��\                                    Bxt��r  �          @�{?�G��p��?��A��C�H�?�G��\)�\)��C���                                    Bxt��  T          @�?��e?z�HAW�C�{?��l�;����C���                                    Bxt��  "          @��?У��r�\=�G�?�G�C���?У��dzῥ����C�J=                                    Bxt�d  �          @��@,(��6ff��{���C���@,(��	����
�z�C���                                    Bxt�
  �          @�  @B�\�   �Ǯ����C�J=@B�\��G�����
C�                                      Bxt�-�  
�          @�=q@P�׿�p�����  C�c�@P�׿����'
=��C���                                    Bxt�<V  �          @��@J=q��p��z��G�C���@J=q�B�\�2�\�"C�33                                    Bxt�J�  
�          @�  @i����Q쿪=q���C���@i�����������G�C��=                                    Bxt�Y�  �          @�G�@s�
��(���G����C�p�@s�
�h�ÿ��H���\C�B�                                    Bxt�hH  �          @�Q�@n�R��zῆff�g33C�H@n�R��z��=q��ffC�T{                                    Bxt�v�  "          @��@n�R��׿!G��	p�C��
@n�R���
���\��{C��)                                    Bxt���  T          @��@��\����\����C��
@��\�Tz�:�H�33C�9�                                    Bxt��:  T          @�\)@S�
��ÿG��-�C��@S�
�����˅��z�C�                                    Bxt���  �          @��@Q��\)�W
=�7�
C�l�@Q��G���
=��=qC�5�                                    Bxt���  T          @�Q�@n{��z�@  �#�C�q�@n{��G��������C���                                    Bxt��,  T          @�\)@h�ÿ����8Q��{C���@h�ÿǮ��\)����C�^�                                    Bxt���  
�          @�{@p  �޸R�   ��33C��=@p  ��������rffC�p�                                    Bxt��x  T          @��R@s�
�ٙ���ff��ffC�  @s�
�����
�c�C���                                    Bxt��  
Z          @�@w
=���\�z�H�XQ�C��@w
=�Tz`\)��z�C��                                    Bxt���  
Z          @�p�@w
=��\)��=q�q��C��@w
=�(�ÿ�z���z�C�.                                    Bxt�	j  
�          @�ff@y����33����h��C�Ǯ@y���333��33��G�C���                                    Bxt�  T          @��@�Q쿆ff�Y���9p�C���@�Q�.{����{C�4{                                    Bxt�&�            @�\)@��׿}p��Y���:=qC�{@��׿�R��33�|Q�C��
                                    Bxt�5\  
�          @�  @�녿L�Ϳk��Ip�C�p�@�녾�
=��33�{33C�                                    Bxt�D  F          @��@{���
=��Q����\C��\@{�=���  ���?�
=                                    Bxt�R�  
�          @���@Mp��(�������C��q@Mp�=��
������p�?�
=                                    Bxt�aN  �          @�p�?����
=��\�׮C��=?���qG����H�ՙ�C��{                                    Bxt�o�  
�          @�
=@\)�g
=��p����C��@\)�<(���H��HC���                                    Bxt�~�  �          @��@��W
=���R���C��@��-p���C�\                                    Bxt��@  T          @�p�@Q��E���z����\C���@Q���R�
�H� �RC���                                    Bxt���  T          @�{?s33���ý�G���(�C��)?s33�o\)��G�����C��                                    Bxt���  "          @��R?�G��n{�������\C��f?�G��Vff��
=���C��q                                    Bxt��2  �          @��?�=q�g���(���C�N?�=q�333�8Q��/�C��\                                    Bxt���  "          @�Q�?k��tz�������
C�?k��AG��5��'z�C�q�                                    Bxt��~  �          @�\)?�p��w��8Q��{C�t{?�p��XQ��G��噚C�Ф                                    Bxt��$  T          @�(�?��s�
��Q����C�9�?��\�Ϳ�33��
=C�/\                                    Bxt���  �          @�p�?Ǯ�r�\��ff����C�0�?Ǯ�Y����p�����C�T{                                    Bxu p  
�          @�{@���X�ÿ0����\C��3@���<(�������33C��q                                    Bxu   	�          @�(�@
=q�S�
�����ffC���@
=q�-p��p���
C�@                                     Bxu �  
�          @��R?����U���p���p�C���?�������@  �<�HC�#�                                    Bxu .b  �          @���@=p��,(����R��G�C�޸@=p��G��z��C�޸                                    Bxu =  �          @�(�@z=q��G�����{C�{@z=q�333��G���G�C���                                    Bxu K�  "          @���@^{�	�����R���RC�
@^{���Ϳ�33��G�C��q                                    Bxu ZT  �          @�{@^�R��=q��Q�����C��@^�R��p���p���C�>�                                    Bxu h�  "          @�ff@b�\��{������RC��H@b�\�}p��   ��{C�,�                                    Bxu w�  
�          @��@y�����H������=qC�h�@y��=�\)��33���H?��                                    Bxu �F  T          @�ff@n�R��R��=q�Ώ\C�G�@n�R=#�
��
=��
=?�                                    Bxu ��  T          @�G�@z�H��  ��
=��  C�ٚ@z�H��ff��
=���C���                                    Bxu ��  
�          @�ff@s33�p�׿�G���
=C��@s33��p��޸R��  C�9�                                    Bxu �8  T          @�G�@�����z΅{��Q�C���@���>8Q쿰����z�@%�                                    Bxu ��  �          @�  @p  ��\���H��=qC�
@p  =�Q���
��33?���                                    Bxu τ  �          @�z�@L(����
�33�(�C��@L(���(��&ff���C�                                      Bxu �*  "          @�z�@S33�k�����Q�C�:�@S33��G����ffC�                                    Bxu ��  �          @�@S�
�u�p��\)C��{@S�
?������
=A$��                                    Bxu �v  T          @���@`  ?:�H�33���A<��@`  ?��H��
=���A��H                                    Bxu
  
�          @�Q�@c33?�\)���θRA�33@c33?�zΰ�
���HA�(�                                    Bxu�  �          @�ff@N�R?�����ff�˅A���@N�R@(�����g\)B
=                                    Bxu'h  �          @��\@Fff?��H��p���A��@Fff@G���ff���B�R                                    Bxu6  �          @���@Fff?�����\A�33@Fff@z�˅����B                                    BxuD�  T          @���@>{?�33�
=�A�G�@>{@ff��G����HB�                                    BxuSZ  �          @��@2�\?���,���$\)A���@2�\@	���ff���RB�R                                    Bxub   "          @�@Q�?�Q��33�p�A�{@Q�?�׿�\���A��                                    Bxup�  "          @�@Mp�?�=q�(���HA�ff@Mp�?��ÿ�Q���Q�A�Q�                                    BxuL            @�{@P��?�z��
=�	�A��H@P��?�\)����33A���                                    Bxu��  
�          @�{@Z=q?xQ��p����A~�R@Z=q?�녿�\��  Aͮ                                    Bxu��  �          @�p�@p��?!G������A�
@p��?�녿�\)��=qA��                                    Bxu�>  �          @J�H@�H��p��E��s
=C�>�@�H��33���H���HC�J=                                    Bxu��  �          @�=q�J=q��ff@   BffCN\�J=q��?�{A�CW�                                     BxuȊ  �          @�G���H�$z�?��A�G�Cb����H�C33?s33A[�
Cgz�                                    Bxu�0  �          @�
=�/\)�*=q?޸RA�{C`.�/\)�E?G�A+�CdaH                                    Bxu��  T          @��H�B�\���?�(�AۮCX�
�B�\�1G�?���Au��C^^�                                    Bxu�|  T          @l�Ϳ�p���
?���A�\)CeW
��p��*=q?�RA)�Ci8R                                    Bxu"  �          @g
=���
�A�?O\)AN�RCo����
�HQ�aG��\(�Cp\)                                    Bxu�  �          @Vff��Q��1G�?8Q�AG�Cn����Q��7
=�W
=�mp�Coh�                                    Bxu n  �          @N�R����6ff?\)A!Ct�����8Q�Ǯ��p�Ct�q                                    Bxu/  �          @I�������5�?5AP��CxO\�����:=q�u��p�Cx�{                                    Bxu=�  �          @p��(���33>�Q�AG�C|#׿(����
���R�  C|5�                                    BxuL`  �          @5�J=q�(Q�=��
?���C}G��J=q�!G��E���{C|��                                    Bxu[  T          @AG���p��-p��#�
�J=qCu�׿�p��#�
�h�����CtW
                                    Bxui�  T          @h���   �$z�>�@�Ca޸�   �%�\���Cb                                    BxuxR  �          @vff�(���/\)?   @�  Cb��(���0  ������{Cb:�                                    Bxu��  T          @~�R�!��@��>��
@���Ce���!��<�Ϳ&ff���Ceu�                                    Bxu��  �          @���0���=p�>�33@��RCb���0���:�H�(��	�Cb�{                                    Bxu�D  �          @w
=�&ff�6ff=�Q�?���Cc���&ff�.�R�Q��C�
Cbn                                    Bxu��  �          @g
=�,(����=L��?Tz�C]���,(���\�5�4��C\W
                                    Bxu��  T          @[��*�H�������CY�q�*�H��33�fff�uCWh�                                    Bxu�6  �          @\������z�=p��I�C_�q������H��
=�ƣ�C[O\                                    Bxu��  �          @\(���33�����\)Cc����  �33�C[�f                                    Bxu�  �          @Q녿����p��fff���\Cg�������33�У���p�CbxR                                    Bxu�(  0          @P�׿�G��{�ٙ��Ck�Ϳ�G�������
�:�
Cb8R                                    Bxu
�  T          @l(��
=q�(�ÿn{�r�HCf��
=q�{�ٙ���z�Ca�q                                    Bxut  
�          @r�\�
=�5��z���p�Cic��
=�����R�(�Cc��                                    Bxu(  T          @s�
�4z���Ϳ.{�&=qC\�R�4z��
=��z����RCXٚ                                    Bxu6�  
�          @l���AG��33�u�}p�CV33�AG���
=�333�0(�CT��                                    BxuEf  
�          @x���e��z�>�33@��CIk��e�������Ϳ�p�CI��                                    BxuT  �          @�Q��s�
��\)>��@�33CDY��s�
��Q�=#�
?
=CET{                                    Bxub�  "          @��
�q녿\?�@��HCI�{�q녿���<�>�G�CJ�R                                    BxuqX  �          @���j=q��  ?
=q@��HCM���j=q��=q���
��{CN�\                                    Bxu�  �          @���U�G�>\)?�
=CV#��U��Ϳ��G�CUT{                                    Bxu��  �          @�G��c33��=��
?�{CVff�c33��+���RCUaH                                    Bxu�J  �          @���g��=q>u@L��CU���g�����\��ffCU=q                                    Bxu��  
�          @�=q�aG���>�
=@�z�CV�f�aG���;�{��=qCV�=                                    Bxu��  
�          @L(�@�׿��H��(���(�C���@�׿333��\���C�c�                                    Bxu�<  T          @P��@;�������\��\)C�1�@;��B�\�����z�C�+�                                    Bxu��  T          @^{@J=q�8Q쿱����C�c�@J=q>�  �������@��H                                    Bxu�  �          @:�H@*�H=��
�����33?ٙ�@*�H>��Ϳ}p����\A�                                    Bxu�.  �          @L(�@2�\?�ff�aG���(�A�@2�\?�G���ff� ��A��H                                    Bxu�  �          @z=q@C�
@G��fff�U�Bz�@C�
@(��.{�!G�BQ�                                    Bxuz  �          @}p�@J=q@p��O\)�=�B@J=q@ff��G����B�                                    Bxu!   �          @�  @i��?��\��  �i�A��@i��?�G��z����A��
                                    Bxu/�  T          @��@w�?��\����p��A���@w�?���(�����A�G�                                    Bxu>l  "          @��
@�Q�?�33����[�A�{@�Q�?�33����
=A��R                                    BxuM  
�          @���@h��?�ff�333�!��A���@h��?�Q�W
=�=p�A�\)                                    Bxu[�  
�          @���@e?�p����H��G�A�=q@e?�ff=L��?333A�33                                    Bxuj^  
�          @��@w�@��B�\�ffB Q�@w�@�?��@ۅA��                                    Bxuy  
�          @��@s33@,(��#�
����Bz�@s33@$z�?L��Ap�B�\                                    Bxu��  �          @�p�@z=q@#�
    �#�
B��@z=q@(�?B�\AA��                                    Bxu�P  T          @��@n�R@&ff����33B\)@n�R@!G�?&ffA�HB{                                    Bxu��  �          @�Q�@k�@&ff���Ϳ�=qB(�@k�@!G�?+�A	�B	�\                                    Bxu��  �          @��@u�@=���?�(�A���@u�@p�?J=qA!�A��
                                    Bxu�B  �          @�  @}p�@Q�=���?��
A��
@}p�@   ?8Q�AQ�A�z�                                    Bxu��  "          @�=q@�G�@>���@��
A�z�@�G�?�33?n{A<  A�33                                    Bxuߎ  �          @�=q@�  @33?5AQ�A�33@�  ?�  ?��A�G�A�
=                                    Bxu�4  �          @���@�  ?�  ?�  A�=qA���@�  ?�ff?��HA�G�A�ff                                    Bxu��  T          @�G�@�33?�z�>��@O\)A�p�@�33?�  ?L��A#�A���                                    Bxu�  �          @���@��
?�{?�\@�\)A£�@��
?У�?��AR�\A���                                    Bxu&  �          @��
@��\@�\?5AA���@��\?�  ?��
A�ffA�G�                                    Bxu(�  T          @�=q@|(�@(�?G�AQ�A�\@|(�?�\)?���A��\A���                                    Bxu7r  
�          @�ff@q�@{?G�A!��A�@q�?�33?��A��A�p�                                    BxuF  T          @�
=@u�@
=?Y��A/
=A���@u�?��
?�A���A���                                    BxuT�  "          @���@~�R?�Q�?�Aw�A��R@~�R?��?��A�  Ai�                                    Bxucd  �          @��@w
=?�=q?���A��A��@w
=?aG�?�A��AM�                                    Bxur
  
�          @�ff@w
=?���?���A���A�33@w
=?�=q?�Ař�A{33                                    Bxu��  
�          @�
=@k�?�z�?�33A��A��@k�?�{@Q�A�ffA��
                                    Bxu�V  �          @���@b�\?�
=?�(�A��A�z�@b�\?���@��A�A��H                                    Bxu��  
�          @c33@�?��H?��A�B \)@�?�z�@Q�B{A�G�                                    Bxu��  "          @h��@%�?�ff?���A��B�@%�?���@��B�A��                                    Bxu�H  "          @u@@��?���?�
=A�Aď\@@��?E�@\)B=qAdz�                                    Bxu��  �          @S33@G�?���?�BQ�A��\@G�?@  @�RB,=qA��                                    Bxuؔ  
�          @7
=@
=?��?�{B��A�33@
=?!G�?�\)B!\)A���                                    Bxu�:  
�          @5@{?z�H?��HA��A�Q�@{?�?�B�ATQ�                                    Bxu��  
�          @E�@)��?8Q�?�
=A�z�Aq��@)��>�33?��Aԣ�@�Q�                                    Bxu�  T          @Mp�@333��  ?�Q�A�=qC�4{@333���?\(�Ax(�C��)                                    Bxu,  �          @O\)@333�fff?��A�ffC�q@333��p�?�  A�\)C��                                    Bxu!�  "          @N{@&ff�O\)?�
=A��RC�W
@&ff��  ?���A�{C�&f                                    Bxu0x  "          @\��@�R�k�@p�B5�
C�� @�R��=q@ffBC�Y�                                    Bxu?  T          @b�\@"�\�B�\@�
B${C���@"�\����@   B
��C��q                                    BxuM�  T          @l(�@/\)��p�@Q�B�RC��=@/\)���
?ٙ�A�Q�C�|)                                    Bxu\j  �          @n{@!G���p�@�HB#�C���@!G���\)?�p�B ��C��f                                    Bxuk  "          @u�@5��u@Q�B��C���@5��˅@ ��A�{C�]q                                    Bxuy�  T          @y��@>�R�aG�@�Bz�C��=@>�R���R?��RA���C���                                    Bxu�\  �          @\)@9���Q�@'
=B#�C��@9����G�@�BG�C�5�                                    Bxu�  
�          @~�R@5�Q�@*�HB(�C��q@5���
@�Bp�C�޸                                    Bxu��  �          @mp�@(�ÿL��@p�B'=qC��@(�ÿ��H@��B��C���                                    Bxu�N  �          @}p�@��?��RA�G�C�AH@� ��?xQ�A��C���                                    Bxu��  
�          @�@   �P  ?�@��RC�˅@   �Q녾�33���
C��=                                    Bxuњ  �          @��R@\)�e�?W
=A-p�C�b�@\)�k��B�\���C��                                    Bxu�@  
Z          @��@
�H�s�
>��R@���C���@
�H�p  �B�\��\C�H                                    Bxu��  �          @���@{�e?z�@���C�H�@{�g
=��(����HC�4{                                    Bxu��  �          @�Q�@#33�W�>�p�@�Q�C��@#33�U������C���                                    Bxu2  �          @��@&ff�Tz�>\@��HC��@&ff�S33�����  C�)                                    Bxu�  T          @u@�R�:=q>�@ۅC�0�@�R�;���Q����HC��                                    Bxu)~  "          @l(�?��R�Fff=L��?W
=C�\)?��R�?\)�L���H(�C��\                                    Bxu8$  
�          @g�@�*�H����G�C��f@�#�
�B�\�D��C�>�                                    BxuF�  T          @x��@�\�E��\)��\C�N@�\�:�H�xQ��j�HC��                                    BxuUp  "          @e?�p��S�
�B�\�J=qC�.?�p��HQ쿋����C��{                                    Bxud  	�          @k�?�{�P  �Ǯ���C�0�?�{�AG���G����C��                                    Bxur�  
Z          @7
=?h���'���G��(�C���?h������������HC�b�                                    Bxu�b  T          @<��?���'�>�G�A�C�AH?���(�þ�z���Q�C�,�                                    Bxu�  
�          @C�
@	����  ����� ��C��3@	���G������C�
                                    Bxu��  �          @XQ�@9�����\��Q����RC�(�@9���fff���R��
=C�\)                                    Bxu�T  "          @k�@E�� �׽�Q쿬��C�g�@E���z�!G���C�                                    Bxu��  �          @Z�H@Dz῱�>�z�@�{C��q@DzῴzὸQ��=qC���                                    Bxuʠ  �          @J�H@:=q���\>��@��C�1�@:=q������Ϳ�p�C��                                    Bxu�F  �          @Dz�@?\)�\)>�{@�(�C��
@?\)�!G�>8Q�@U�C�\                                    Bxu��  �          @N{@Dz��?=p�AV�RC�9�@Dz�333?z�A&{C���                                    Bxu��  "          @\(�@Tz���?O\)AYC�w
@Tz�(�?+�A3�C���                                    Bxu8  �          @s33@l�Ϳ�?(�AQ�C��R@l�Ϳ8Q�>�G�@��HC��f                                    Bxu�  �          @w
=@qG���?#�
Az�C�]q@qG��!G�>��H@�  C�9�                                    Bxu"�  "          @�G�@|(����?@  A,Q�C���@|(��&ff?��A	G�C�N                                    Bxu1*  T          @��@�G���?Y��A<��C��q@�G��0��?0��AG�C�(�                                    Bxu?�  
�          @��R@���
=?8Q�AC��@���B�\?
=q@�\C��q                                    BxuNv  "          @�(�@�Q�#�
?8Q�A ��C�~�@�Q�L��?�@��C�T{                                    Bxu]  "          @��@\)�
=?E�A,��C���@\)�E�?z�A�HC���                                    Bxuk�  T          @{�@o\)���>�=q@}p�C�� @o\)��
=�#�
�(�C�H�                                    Bxuzh  
�          @k�@a녿�G�>u@n�RC�@a녿���#�
���C�Ф                                    Bxu�  "          @~�R@p�׿����Q쿢�\C�k�@p�׿��R��(���{C��                                    Bxu��  
�          @���@s�
��ff<#�
=�C��q@s�
��  �������C���                                    Bxu�Z  
(          @���@tz῞�R���
��C�f@tz῕��������C�xR                                    Bxu�   �          @�Q�@j=q��=q������\)C�Q�@j=q��Q�:�H�(��C�E                                    Bxuæ  T          @��@s�
��\)���Ϳ�
=C��@s�
��ff���ϮC��{                                    Bxu�L  �          @���@q녿�
=���
��33C��q@q녿��׾������C��q                                    Bxu��  �          @z�H@i����>\)@33C�W
@i����z�u�]p�C�l�                                    Bxu�  
�          @~{@tzῆff�L�Ϳ5C�Z�@tz῀  ���
��(�C���                                    Bxu�>  �          @~{@q녿�
=�u�Z=qC�O\@q녿�=q�
=q����C�                                    Bxu	�  �          @���@o\)��������RC�]q@o\)���\�c�
�JffC��H                                    Bxu	�  �          @\)@tzῇ��L�ͿG�C�@ @tz῁G��������C���                                    Bxu	*0  
�          @�  @u���\����=qC���@u�s33�Ǯ��=qC�                                    Bxu	8�  "          @���@w
=�����=q�uC�  @w
=�z�H�
=q��  C��q                                    Bxu	G|  
Z          @���@vff���aG��L(�C��=@vff���ÿ�����C�5�                                    Bxu	V"  �          @�G�@w
=���׾L���8��C�ٚ@w
=������H��ffC�y�                                    Bxu	d�  �          @\)@q녿�ff�5�$z�C�4{@q녿Tz�u�^�RC��                                    Bxu	sn  
�          @|��@o\)��p����
��C���@o\)���Ǯ��  C�Q�                                    Bxu	�  
�          @~�R@r�\��G����� ��C���@r�\�Tz�J=q�8��C��{                                    Bxu	��  
�          @}p�@q녿}p��
=�	p�C���@q녿O\)�Tz��@��C��
                                    Bxu	�`  �          @|��@p�׿h�ÿL���<  C�7
@p�׿+���G��l��C��                                    Bxu	�  �          @|(�@n�R�aG��k��X  C�U�@n�R��R��{���C�J=                                    Bxu	��  
�          @}p�@s33�W
=�B�\�1��C���@s33�(��s33�^{C�k�                                    Bxu	�R  
M          @~�R@u��&ff�\(��H(�C�0�@u���녿�  �h��C���                                    Bxu	��  
g          @}p�@s33�5�^�R�J=qC��3@s33��׿��\�n�\C��                                     Bxu	�  
(          @|��@k��@  ������C�9�@k���
=������z�C���                                    Bxu	�D  �          @\)@qG��@  ���\�nffC�Z�@qG���׿�
=��(�C�w
                                    Bxu
�  T          @�Q�@r�\�:�H����vffC���@r�\��G����H��33C��
                                    Bxu
�  
�          @~�R@mp��Q녿�����C��=@mp����H�����\)C�4{                                    Bxu
#6  �          @���@s33�����H��  C��@s33�\)���
��p�C��                                    Bxu
1�  �          @~�R@u=�Q쿃�
�p(�?��@u>�{�xQ��c
=@��
                                    Bxu
@�            @z=q@p��>B�\���\�s
=@5@p��>�G��p���_�@���                                    Bxu
O(  /          @{�@qG��W
=��ff�x��C�` @qG�=u�����}�?aG�                                    Bxu
]�  "          @|��@n{�����33��{C�l�@n{��zῢ�\��  C��f                                    Bxu
lt  T          @u@e�#�
��G���=qC�� @e>#�
��  ��(�@'�                                    Bxu
{  "          @fff@$z���ÿ5�6=qC���@$z��Q쿣�
��{C�*=                                    Bxu
��  �          @_\)@��#33�z��=qC��)@���������=qC�0�                                    Bxu
�f  T          @Z=q@��(Q�333�=�C��@�����=q���HC�AH                                    Bxu
�  �          @L��?�(��*�H��������C�J=?�(��33��G��Q�C��R                                    Bxu
��  
          @:�H>�
=�6ff>���@��C�*=>�
=�5���
=��C�0�                                    Bxu
�X  �          @:�H�W
=�9������=p�C��;W
=�1G��^�R����C��3                                    Bxu
��  
�          @>{���:=q�#�
��C��ÿ��4z�:�H�e�C���                                    Bxu
�  "          @<�;W
=�:�H>�\)@�  C�箾W
=�9����ff��C���                                    Bxu
�J  T          @=p����
�;�>���@���C�ٚ���
�:�H��G��	p�C�ٚ                                    Bxu
��  "          @<�;��8��>Ǯ@�\)C��\���9����{���
C���                                    Bxu�  T          @>�R<��
�7
=?Q�A�{C�4{<��
�>{=�Q�?�z�C�33                                    Bxu<  "          @AG�>���<��<�?�C���>���7
=�333�Z{C��                                    Bxu*�  
�          @@��?�\�<�ͽ��Ϳ�
=C���?�\�5��Tz���=qC��                                    Bxu9�  T          @@  >���>{��\)����C�u�>���7
=�L���z{C���                                    BxuH.  
�          @C33>.{�AG���{���HC���>.{�5��������C��R                                    BxuV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxuez              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxut               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxu��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxu�l  s          @0��?W
=��{�ff�mG�C��q?W
=�   �#33�C���                                    Bxu�  "          @XQ�?��\�Q��ff�3��C���?��\���R�3�
�e  C��                                    Bxu��  T          @N�R?
=q�333��=q�ȏ\C�p�?
=q��ÿ�p��C�Y�                                    Bxu�^  �          @U?8Q��!G���\���C��R?8Q��Q��%�Mz�C�'�                                    Bxu�  
          @S�
>��H�%����R�ffC�g�>��H�G��#�
�K�C��q                                    Bxuڪ  
�          @3�
���H�   ��ff�%
=C�%���H��  ��R�[ffC{�R                                    Bxu�P  
�          @H�ÿ\(��z��\)��Cy���\(�����Q��H�RCth�                                    Bxu��  �          @)���@  �33�n{��ffC{�Ϳ@  �   ���H�	\)CyT{                                    Bxu�  
�          @J�H�
=q�3�
���
��G�C��ÿ
=q�=q��
=�C��R                                    BxuB  �          @G
=�(���	�����-  C}  �(�ÿ����$z��b  CwO\                                    Bxu#�  �          @L(��E���������C{�f�E�����p��Np�Cv��                                    Bxu2�  
          @`�׿����8Q�˅��  Cwc׿����=q�\)��Cs��                                    BxuA4  /          @G
=��Q��Q����1z�C�%��Q�Ǯ�$z��hp�C�g�                                    BxuO�  "          @@  ?
=q��Q���F��C���?
=q�������zC��\                                    Bxu^�  "          @AG�@*�H�Q녿�����p�C�l�@*�H����G����C�U�                                    Bxum&  
          @9��@��
=��=q�\)C�U�@�B�\������
C�                                    Bxu{�  
�          @?\)@zῥ���\)��z�C�q�@z�k�����Q�C�'�                                    Bxu�r  "          @@  @
�H��Q쿺�H��z�C�5�@
�H�����G��  C�!H                                    Bxu�  
�          @A�@��޸R��������C�
@�����������C��f                                    Bxu��  "          @o\)@���
�{��RC�E@���(��)���4C���                                    Bxu�d  �          @s33@�����
��C�G�@����H�+��4��C��                                    Bxu�
  T          @vff@#33�G���\� �\C���@#33��(��{� 
=C��
                                    BxuӰ  T          @a�@����
�������\C��@����
�{��
C�q                                    Bxu�V  "          @Vff@���33��G��؏\C�"�@�ÿУ׿��H�ffC�e                                    Bxu��  "          @U�?�(��\)���
����C�S3?�(��
=���(�C��R                                    Bxu��  T          @Y��?���G
=�8Q��K�C�o\?���6ff��Q���Q�C��                                    BxuH  �          @l(�>���c�
��R���C�J=>���S�
������=qC���                                    Bxu�  �          @l��?u�X�ÿ�\)��33C���?u�AG���33����C�ٚ                                    Bxu+�  T          @qG�>�=q�^{�����C�7
>�=q�B�\����
C��f                                    Bxu::  �          @h�þB�\�/\)�����C�H�B�\����2�\�Q��C�s3                                    BxuH�  T          @p  >�ff�H�ÿ��H���C�
>�ff�%��)���533C��{                                    BxuW�  "          @W��B�\�������==qC����B�\��{�:�H�t
=C���                                    Bxuf,  
�          @[�?�{�&ff�����
C���?�{������8�C�
=                                    Bxut�  G          @N{=��
��Q�� ���Q
=C�1�=��
����9���C�˅                                    Bxu�x  
g          @P��>L�Ϳ�
=�0���i��C�b�>L�Ϳz�H�E��C���                                    Bxu�  "          @XQ�>u��\)�1G��_(�C��R>u��z��H����C���                                    Bxu��  "          @]p�>�\)�  �'
=�C�
C��>�\)�����C�
�z{C�f                                    Bxu�j  �          @]p�>�{����%��D�
C�\)>�{���
�A��z�RC�<)                                    Bxu�  
�          @]p��\(������;��j��CqJ=�\(��Tz��N{W
C`�                                    Bxu̶  
�          @_\)���Ϳ�=q�N�R�fCy�)���;�\)�X��£��CW!H                                    Bxu�\  
�          @N�R?��R�6ff�u��C���?��R�0  �:�H�Z�RC�
                                    Bxu�  "          @L��?\�'
=�z�H��G�C�{?\��
�Ǯ��Q�C��=                                    Bxu��  �          @Tz�?�z��7��G��\  C��?�z��&ff��Q���
=C�<)                                    BxuN  �          @N�R?�{�9���L���g�C�o\?�{�(Q쿺�H��
=C�g�                                    Bxu�  �          @L��?}p��8Q�
=q�!�C�xR?}p��*�H��������C�'�                                    Bxu$�  �          @K�?���*�H��Q����\C�{?���z���
�=qC��3                                    Bxu3@  y          @U�?k��{��\�.�\C�E?k���\)�0  �_�\C��{                                    BxuA�  
�          @U�?fff��Q���R�D�C�j=?fff����7
=�s��C�C�                                    BxuP�  "          @]p�?�33�<�Ϳ(��$��C���?�33�.{�����ffC��q                                    Bxu_2  
�          @_\)?�=q�+����
����C�7
?�=q���
=�+
=C��                                    Bxum�  "          @`��?�ff�=p����R����C�� ?�ff�!������C�AH                                    Bxu|~  T          @g�?���G���  ���C�g�?���/\)�����z�C��=                                    Bxu�$  �          @b�\?��
�5���33���C�
?��
��R���
���
C��3                                    Bxu��  
�          @c�
?����	������+
=C���?������
�5�U�C��R                                    Bxu�p  �          @b�\?����
=�5��Z�C�~�?���xQ��H��8RC�+�                                    Bxu�  T          @`��>��R��z��8Q��`z�C���>��R��
=�P  �RC�e                                    Bxuż  "          @c�
����(��Tz�\)C�.������aG�¦
=C}��                                    Bxu�b  "          @e>\=L���aG�§�A�>\?aG��Z=qǮB�z�                                    Bxu�  "          @`  ?�\)?��J=q�Aٮ?�\)?�ff�<(��o33BE
=                                    Bxu�  �          @e?�G�?��R�C�
�n
=B_G�?�G�@
�H�(Q��>�B��)                                    Bxu T  T          @g�?\(�?���C33�k=qBy\)?\(�@33�%�9�
B�#�                                    Bxu�  
�          @o\)?�Q쿽p��3�
�C�HC�S3?�Q�J=q�E��^��C���                                    Bxu�  �          @x��?��
��p��G��Uz�C�{?��
�8Q��XQ��q��C�f                                    Bxu,F  
�          @~{?��Ϳ�  �G
=�Rp�C�5�?��Ϳz�H�[��u(�C�AH                                    Bxu:�  
�          @�  ?���Z�H��ff��
=C��)?���=p��33�33C�33                                    BxuI�  "          @y��?����J�H��\���C��f?����*�H�p��=qC���                                    BxuX8  �          @�  ?��\�C�
�{�=qC�P�?��\�p��7��8�C��{                                    Bxuf�  
�          @��H?�33�4z��%��RC�9�?�33����J�H�L�C���                                    Bxuu�  �          @�=q?�Q��*=q�#�
�G�C�1�?�Q���R�G
=�GC�!H                                    Bxu�*  �          @��
?��
=q�E�B
=C��?������`  �h��C�7
                                    Bxu��  �          @��H?��H����G��HC�'�?��H��{�a��q�
C���                                    Bxu�v  �          @���@�
�@�׿�Q���(�C�33@�
��R�%�
=C��H                                    Bxu�  
�          @���@
�H�<(�������=qC�<)@
�H�=q�%����C��                                    Bxu��  �          @�  ?�\)���1��1�
C���?�\)���L(��V��C�n                                    Bxu�h  "          @�33@   ��  �E��D�HC�l�@   �}p��Y���c  C��H                                    Bxu�  �          @��?�ff��(��K��OC�8R?�ff�n{�_\)�oG�C�XR                                    Bxu�  "          @��\?�����7��3Q�C�Z�?�녿���U�\�RC��H                                    Bxu�Z  �          @�z�?ٙ��L(��   ��C��?ٙ��(���,(��"�C�j=                                    Bxu   
�          @�33?�
=�Q녿�ff��ffC���?�
=�1G�� ���{C��
                                    Bxu�  "          @��?���N{������ffC�z�?���+��(��� {C��{                                    Bxu%L  
(          @���?��E�p�� ��C�8R?��\)�7
=�.�\C���                                    Bxu3�  �          @�
=?����@����
�  C���?�������<(��1p�C��)                                    BxuB�  T          @�{@ff�$z���\�p�C�33@ff��(��4z��*�C���                                    BxuQ>  	.          @�{@33�+��p�� =qC�H�@33�ff�1G��&�
C�˅                                    Bxu_�  T          @�\)?�Q��:=q�����C�ٚ?�Q����>�R�3p�C�/\                                    Bxun�  T          @�
=?��Dz�������RC��q?��   �333�&ffC���                                    Bxu}0  �          @�{?����K�� ����  C��=?����(Q��,(�� Q�C�`                                     Bxu��  
Z          @���?�(��QG���\)���C���?�(��3�
����RC��H                                    Bxu�|  �          @��@Q��K��У�����C���@Q��.{�z����C��                                    Bxu�"  
�          @��R@G��O\)��z�����C��=@G��4z������C�k�                                    Bxu��  T          @��@=q�O\)��  �]�C�XR@=q�:�H��(���G�C���                                    Bxu�n  "          @���@=q�K�����t(�C��
@=q�5����ͮC�(�                                    Bxu�  T          @��@��I��������(�C�5�@��,����\���C�c�                                    Bxu�  �          @���@=q�C�
��G���(�C��@=q�$z��=q�	C���                                    Bxu�`  �          @��@��HQ쿾�R����C��@��,���
�H���C��{                                    Bxu            @���@��HQ�У����C��@��*�H��
���C�(�                                    Bxu�  `          @�(�@���P�׿�
=��=qC��@���2�\����{C�Q�                                    BxuR  T          @�(�@  �O\)��33���C�b�@  �-p��&ff���C��{                                    Bxu,�  
�          @���@
=�K���33�ͅC�O\@
=�)���%�Q�C��R                                    Bxu;�  
�          @�p�@Q��HQ�� �����HC���@Q��%��+���C�W
                                    BxuJD  T          @�p�@�E��Q�����C��H@�   �1��\)C���                                    BxuX�  "          @��@��B�\��R��C�j=@��(��8Q��"�C�~�                                    Bxug�  �          @���@��@  ���33C��@����?\)�+��C�S3                                    Bxuv6  "          @�p�@�A��(��33C�U�@�Q��Dz��0z�C��=                                    Bxu��  �          @�z�@�E��
��{C��R@�!��.{��C�h�                                    Bxu��  �          @�z�@G��Fff�
=��ffC�R@G��!��1G���
C��3                                    Bxu�(  T          @���@G��G�����Q�C�  @G��"�\�1���C��R                                    Bxu��  �          @��R@
=�L��� ���֏\C�,�@
=�)���,�����C���                                    Bxu�t  �          @�  @��HQ������C��@��#33�3�
�ffC���                                    Bxu�  T          @�\)@�H���
=q���C�^�@�#33�5���HC�C�                                    Bxu��  �          @�{@{�C�
�33��(�C�j=@{� ���,���(�C�C�                                    Bxu�f  �          @�@(���9����
���C�1�@(����+����C�=q                                    Bxu�  �          @�ff@5��)������\C�e@5����.�R���C��{                                    Bxu�  
�          @�@/\)� �������\C�@/\)����:=q�%
=C��)                                    BxuX  	�          @�(�@$z��)���ff��C��@$z����9���&�C��                                     Bxu%�  
          @�(�@!G��+��
=��C���@!G���
�:�H�'��C�XR                                    Bxu4�  �          @���@Q��*�H�#33�{C���@Q�� ���Fff�3��C�޸                                    BxuCJ  �          @�z�@(���=q�#33�  C���@(�ÿ�  �A��/(�C�1�                                    BxuQ�  �          @�z�@'
=�{�/\)���C��{@'
=�\�K��9C�޸                                    Bxu`�  �          @�z�@!����:�H�&C�Q�@!녿����Tz��D�RC��                                    Bxuo<  �          @���@��=q�5�!
=C�  @���Q��Tz��DffC��                                    Bxu}�  �          @�p�@���H�:=q�$�RC���@녿�
=�X���H�\C�                                    Bxu��  �          @�{@	���G��J=q�4��C��q@	����p��fff�X
=C��f                                    Bxu�.  �          @�p�@��   �H���5�RC��\@����H�aG��S�RC�q�                                    Bxu��  T          @�
=@'��Ǯ�P���;�C���@'��B�\�a��Q�C���                                    Bxu�z  T          @��R@#33�Ǯ�S33�?z�C�E@#33�@  �e��U�\C���                                    Bxu�   �          @�ff?�  �S�
�
=q����C�N?�  �.{�7��/
=C�L�                                    Bxu��  �          @��R?u�R�\��
��C�/\?u�*=q�@���;33C���                                    Bxu�l  �          @���?��p�׿��R��p�C��)?��L(��3�
�#�
C��                                    Bxu�  �          @��
>��H��G��˅��G�C�|)>��H�c33�\)�
�C��3                                    Bxu�  �          @�{?=p����\���
��{C�"�?=p��g
=�(���HC���                                    Bxu^  �          @�ff?E���ff��=q�c�
C�/\?E��u���ۙ�C���                                    Bxu  �          @���?h�����Ϳ��
��(�C�5�?h���k������C��R                                    Bxu-�  �          @�?�����������  C�j=?����\(��=p��C��H                                    Bxu<P  �          @�=q?���l(��1��
=C�  ?���<(��c�
�>��C�E                                    BxuJ�  �          @��H?����Z�H�H���"  C�˅?����%�vff�S�C��f                                    BxuY�  �          @���?�ff�5��`  �=z�C�Y�?�ff��Q���=q�j�\C�W
                                    BxuhB  �          @�  ?�\)��
�z=q�^  C�W
?�\)��=q���Hk�C��                                     Bxuv�  T          @���?�����~{�`�C��f?����������HC�%                                    Bxu��  �          @�=q?�=q�!��p���NffC��q?�=q��=q��  �y  C�|)                                    Bxu�4  �          @��H?�G�����z=q�W(�C��?�G����
���\�}=qC���                                    Bxu��  �          @���?�  �5�Y���6
=C��
?�  ��(��~{�a��C��R                                    Bxu��  �          @�Q�?Ǯ�)���p���J�
C�H�?Ǯ��Q������v�\C�XR                                    Bxu�&  �          @��R?��H�b�\�a��*��C�8R?��H�'���Q��[z�C���                                    Bxu��  �          @�ff?Ǯ�\���e�-C�.?Ǯ�!G������]��C��                                    Bxu�r  �          @�=q?����.�R�r�\�A�C��?�����\��=q�j�C���                                    Bxu�  �          @�G�?�33�)���u�F�RC�ٚ?�33������o=qC�U�                                    Bxu��  �          @��H?�z��.�R�vff�D�\C�u�?�z��  ��(��m�RC���                                    Bxu	d  �          @��
?���5�u��A33C��)?�׿�\)��z��kz�C���                                    Bxu
  �          @�Q�?Ǯ�N{�C�
�"33C��3?Ǯ�=q�n�R�Q��C�z�                                    Bxu&�  �          @�=q?B�\�N�R����{C��f?B�\�.{� ���&�C���                                    Bxu5V  �          @�ff�L�����?��
Ac�C���L����{=L��?#�
C��                                    BxuC�  �          @�
=�k���G�?�Q�A���C�]q�k����R>L��@0��C�n                                    BxuR�  �          @�(��L���}p�?�
=A�(�C��H�L����(�>W
=@9��C��                                    BxuaH  �          @�=q�#�
�p  ?�=qA�\)C��R�#�
����?(��A��C��q                                    Bxuo�  �          @���=����dz�?�=qAمC���=����y��?p��AX��C��                                     Bxu~�  �          @�  >\�`  ?��A�=qC�
>\�vff?�G�Ak\)C��\                                    Bxu�:  �          @u�?�33��(�@;�BT  C�q?�33��@   B+�
C��=                                    Bxu��  �          @s33@Q쿠  @)��B2�C�'�@Q��@�BC�%                                    Bxu��  �          @e@ �׿�@  Bp�C�� @ �׿�?�z�Bz�C�/\                                    Bxu�,  T          @U?xQ��ff@p�B.ffC�l�?xQ��#�
?�33A�C�e                                    Bxu��  �          @dz��
=�333?�33A�\)Co
=��
=�C33?0��A4z�Cq+�                                    Bxu�x  �          @w���z��7
=?�  A�ffClB���z��L(�?��
Aw33Co�                                    Bxu�  
�          @{����;�?��
A�33Cl�����P��?��
As�Coz�                                    Bxu��  T          @y�����1G�?ٙ�A�Ch�����Fff?}p�AmG�Ck�\                                    Bxuj  �          @c33�Q���R?�\)A���C_0��Q��   ?L��AR{Cb^�                                    Bxu  �          @hQ�(�ÿ���@Q�B��Co�
�(�ÿ��@:�HB^�Cz��                                    Bxu�  �          @e���=q��z�@.{BEz�Ck!H��=q�   @�B\)Cq��                                    Bxu.\  �          @l(����׿�z�@?\)BY
=Cf@ �����z�@ ��B+�
CoO\                                    Bxu=  �          @~�R��ff��Q�@EBL�RCguÿ�ff�(Q�@"�\B�Cou�                                    BxuK�  �          @�녿���� ��@P  BU=qCl:Ῥ���.�R@*�HB%G�Cs�                                     BxuZN  �          @�Q�ٙ��33@?\)BA=qCfQ�ٙ��-p�@�HBQ�Cm�
                                    Bxuh�  �          @}p���G���@G
=BOG�Cg�\��G��'
=@#�
B!Q�Co��                                    Bxuw�  �          @y������G�@I��BW�\Cq�����.{@$z�B%Q�Cx#�                                    Bxu�@  �          @|�ͿJ=q�z�@QG�B_��Cy33�J=q�333@+�B*��C~E                                    Bxu��  �          @z�H�Y��� ��@P  B`33Cw��Y���.�R@*�HB,
=C|��                                    Bxu��  �          @}p��z��33@U�Be��C~G��z��2�\@/\)B/��C�'�                                    Bxu�2  �          @u����H�(�@8Q�BD\)Cq
=���H�3�
@G�B\)Cv�q                                    Bxu��  �          @z=q?��\���
@FffB[Q�C�� ?��\�{@%B+��C���                                    Bxu�~  �          @w
=?�=q�ff@AG�BP{C��?�=q�0��@�B=qC���                                    Bxu�$  �          @�Q�?���   @Mp�BU��C�s3?���-p�@(Q�B%  C��q                                    Bxu��  �          @�=q?��R��\@S�
BY{C���?��R�1�@.{B'�C��                                    Bxu�p  �          @~�R?\)� ��@A�BFffC�P�?\)�J�H@�B��C��                                    Bxu
  �          @|(�?���/\)@0��B2�C��
?���U�@G�A�
=C���                                    Bxu�  �          @�  ?B�\�333@/\)B-G�C���?B�\�XQ�?�p�A��C�\)                                    Bxu'b  �          @�ff?
=q�;�@:=qB1  C�8R?
=q�c33@
=A�C�T{                                    Bxu6  �          @�33���QG�@��B33C��3���qG�?�=qA�Q�C��q                                    BxuD�  �          @��H�.{�P��@(�B33C�~��.{�p  ?�=qA�=qC���                                    BxuST  �          @���#�
�I��@'
=B�\C����#�
�l(�?�\A��HC�                                    Bxua�  �          @��?��r�\?�G�A��C��
?��~�R>���@���C��f                                    Bxup�  �          @\)?��j�H?�(�A���C�\?��z�H?�@�p�C��                                    BxuF  �          @��H>���n�R?�\)A�Q�C�%>������?(��AQ�C���                                    Bxu��  �          @�(�?(��r�\?��RA�  C���?(���G�?�@陚C�J=                                    Bxu��  �          @��?
=q�r�\?�p�A���C�
?
=q��G�?�\@�C��R                                    Bxu�8  �          @��H?8Q��l(�?�
=A�33C��=?8Q��{�>��H@�  C�5�                                    Bxu��  �          @��H=#�
�}p�?}p�Aap�C�N=#�
���\    ��\)C�K�                                    BxuȄ  T          @��=��x��?��A�Q�C��)=���33>��R@��
C���                                    Bxu�*  �          @��R>���~�R?��A�z�C�� >����{>��
@�z�C�Ǯ                                    Bxu��  �          @�(�?0���~�R?@  A'\)C��{?0����G���  �b�\C��                                    Bxu�v  �          @��?����|�ͼ����C���?����tz῀  �aG�C�<)                                    Bxu  �          @�p�?�ff���׾���eC�K�?�ff�u����R����C���                                    Bxu�  T          @w�?5�mp��J=q�=G�C�g�?5�Y����
=�Ώ\C��H                                    Bxu h  �          @\)>���w�>�?�\)C�R>���q녿Tz��FffC��                                    Bxu/  �          @�33?.{��Q����\)C��\?.{�p�׿��R���\C�!H                                    Bxu=�  �          @���?!G��z�H�J=q�4(�C���?!G��fff�޸R��
=C��
                                    BxuLZ  �          @|��>��q녿�  �k�C�u�>��Z�H��z���C��3                                    Bxu[   �          @z�H>L���n�R��
=��z�C��H>L���Tz��z����C���                                    Bxui�  �          @p  >L���S�
�ٙ����C�� >L���2�\�{�%�C�{                                    BxuxL  �          @p  =��[����R��\)C���=��<����\�33C�#�                                    Bxu��  T          @p  �����c�
�����z�C�(������J=q���R� �
C�                                    Bxu��  �          @n�R�B�\�e�u�\(�C�ͿB�\�^{�s33�o�C���                                    Bxu�>  �          @k�<#�
�dz�>�G�@�
=C��<#�
�dz�����
=C��                                    Bxu��  �          @`  �Tz��K���(����Cff�Tz��>{��(����C~k�                                    Bxu��  �          @`  �����Fff���
���
Cx�����/\)��\����CvJ=                                    Bxu�0  �          @_\)��{�C�
�������Cz!H��{�+���\)��Cw��                                    Bxu��  �          @j=q�G��0�����{C~T{�G��Q��3�
�Lp�Cy�R                                    Bxu�|  �          @U��p���z��
�H�#�RCx
=�p�׿ٙ��,(��Xp�Cq0�                                    Bxu�"  �          @S�
�   ��R����9�C��H�   ��ff�8Q��qC|
=                                    Bxu
�  �          @y����
=�   �'��)33CpB���
=��G��J�H�Y�\Cf�                                    Bxun  �          @[���
=��
����RCnB���
=���H�%�EG�Cf!H                                    Bxu(  �          @W
=��z���
��(����CjT{��z��ff�\)�*�CcaH                                    Bxu6�  �          @\�Ϳ�p���\�����Q�Ce#׿�p���ff�
�H�=qC^@                                     BxuE`  �          @c33��Q��(���Q��{CkB���Q��{�   �3=qCc�\                                    BxuT  T          @g
=������  ��Ck�\����z���
�)z�Cdٚ                                    Bxub�  �          @fff��33�Q���
��Co� ��33�������4(�Chٚ                                    BxuqR  �          @g
=�L��� ���8Q��T33Cx=q�L�Ϳ��H�S33aHCl��                                    Bxu�  T          @l(���Q��ff�=p��W�Cl� ��Q�}p��Tz�{C[��                                    Bxu��  �          @�G��'
=�3�
���H����Cc#��'
=�=q�����\)C^�q                                    Bxu�D  �          @�33���J�H�^�R�Dz�Ch�\���5��33���\Ce��                                    Bxu��  �          @����{�P  �=p��)Ck�)�{�<�Ϳ����\)Ci                                    Bxu��  �          @�Q�� ���Y�����  Cop�� ���H�ÿ�
=��G�Cmh�                                    Bxu�6  �          @w
=���@  �#�
�Ch8R���/\)�����\)Ce�H                                    Bxu��  �          @j�H��\�7
=����RCgc���\�(�ÿ�����p�Ce&f                                    Bxu�  �          @W
=����(�=#�
?E�Cc5�������z��!�Cb^�                                    Bxu�(  �          @Q���\��
�#�
�333Ca8R��\�(��@  �T��C_��                                    Bxu�  T          @R�\�����R������C_\������\(��s�C]�                                    Bxut  T          @Y���333��=q���Q�CU=q�333��녿xQ����CRY�                                    Bxu!  �          @W
=�=p���33�333�@Q�CMY��=p��������=qCI��                                    Bxu/�  �          @S�
�:=q��z��R�.�\CM�{�:=q�����xQ����\CJff                                    Bxu>f  T          @P  �4zῬ�ͿQ��l(�CM��4zΉ���33��  CI�                                    BxuM  �          @P  �\)��=q��G����C[.�\)��33�fff���\CX^�                                    Bxu[�  �          @N{�W
=�>{>���@�33C~0��W
=�=p�����RC~)                                    BxujX  �          @H�ÿ����
=����Cfc׿�����ÿ�{��z�Cc��                                    Bxux�  �          @:=q��(���
=?&ffAS�
C`����(���=�@p�Ca�q                                    Bxu��  �          @7���\�ff>���@���Ce����\�
=�k����Cf�                                    Bxu�J  T          @J�H�AG���p��E��bffC;��AG��#�
�W
=�w�
C7                                      Bxu��  �          @J�H�0�׿.{��33��z�CA���0�׾�Q쿥���33C;k�                                    Bxu��  �          @6ff��\����Q���
=Ca��\��Q��\)��
C[&f                                    Bxu�<  �          @;���ff��{��{��\)Cb\��ff��Q��ff�{CZ�q                                    Bxu��  �          @:�H��p���������=qCb=q��p���{��\)�!p�CZ33                                    Bxu߈  �          @:�H��{��=q���R��  C`���{��Q��
=�z�CY��                                    Bxu�.  �          @S�
�33�s33�(��<�
CM  �33��\)�'
=�N\)C;��                                    Bxu��  �          @N{��Q�}p��#33�QCRG���Q쾏\)�.{�g
=C=Q�                                    Bxuz  �          @S�
��G���
=�"�\�Hz�CU�ÿ�G����0���b(�CB��                                    Bxu   �          @X�ÿ�  �����)���PG�CT!H��  ��Q��6ff�g��C?xR                                    Bxu(�  �          @[��	����=q�(��5ffCN���	���Ǯ�(���I\)C>c�                                    Bxu7l  �          @Q��=q���\��G���\CO�f�=q�@  ��\�ffCE=q                                    BxuF  �          @QG��'���Q���
��  CLff�'��=p�����p�CC�3                                    BxuT�  �          @L���#33�����\)��=qCJ5��#33�녿���
C@��                                    Bxuc^  �          @L���33��ff���\���CY���33���H���H���
CTW
                                    Bxur  �          @H�ÿ�
=�
=�5�R�RCf����
=�������ǅCc#�                                    Bxu��  �          @J=q����
�����G�Ch
=������(��33Cb�\                                    Bxu�P  �          @^�R��\)�-p���  ����Cw����\)�����H�4{Crs3                                    Bxu��  �          @\�Ϳ���*�H�����{Cw�
����z��   �;��CrJ=                                    Bxu��  �          @^�R�O\)�{��\�%Q�C{ٚ�O\)��G��7��_�
CuL�                                    Bxu�B  �          @\(�����G��\)�:z�C������G��@���v=qCx:�                                    Bxu��  �          @_\)���{����-(�C����녿�p��>{�i�HC{��                                    Bxu؎  �          @aG�>\�I��������C�t{>\�)���{��C��                                    Bxu�4  �          @^{�#�
�G
=��33���
C��H�#�
�'
=������C�8R                                    Bxu��  �          @Z�H�����,(��(���[
=C��������=q��\)���C��)                                    Bxu�  �          @[���p��1G���
=�
33C�7
��p�����'
=�H�
C��                                    Bxu&  �          @U�u�9����{�癚C��=�u�ff��3  C�                                      Bxu!�  �          @U������<(���z���{C�q�����(��
=q�%=qC���                                    Bxu0r  �          @XQ�L���P  ������33C�=q�L���A녿��H���C�q                                    Bxu?  �          @S33>����{��
�{C��3>��Ϳ��*=q�\��C�4{                                    BxuM�  �          @]p�?�  �(����ffC�&f?�  ��  �1G��X
=C��=                                    Bxu\d  �          @[�?�ff��G��&
=C�?�ff��\)�5��^��C�h�                                    Bxuk
  �          @a�?����(��33�"�C��q?��ÿ��H�8Q��[�
C�
=                                    Bxuy�  �          @c�
?��R���  �=qC���?��R�����333�P��C���                                    Bxu�V  �          @Z=q?�\)��� ���z�C�:�?�\)��
=�%��G��C��
                                    Bxu��  �          @<(�����
=�\)�i�Cy�����.{�2�\z�Cg��                                    Bxu��  �          @6ff=�Q���
�
=�d(�C��)=�Q�O\)�,��L�C��                                    Bxu�H  �          @\)>�G����׿�p��W(�C���>�G��E���\ffC��                                    Bxu��  �          @5?fff�����   �2�RC�q?fff��Q����jQ�C���                                    Bxuє  �          @:=q?�\)��\)��:�C�E?�\)�xQ��p��k�C�z�                                    Bxu�:  �          @@  ?��Ϳ˅�Q��6�HC�#�?��Ϳn{�\)�c�C���                                    Bxu��  �          @G����'
=�L�Ϳ�  C�\)���\)�G����
C��                                    Bxu��  �          @>{��  �
=?�p�A�{Cr��  �'
=?�A"�\CtY�                                    Bxu,  �          @<(��Y���.�R>�ffA�
C|���Y���0  ������z�C|�R                                    Bxu�  �          @@  ���:=q�u��=qC�uÿ��1G��^�R��G�C�1�                                    Bxu)x  �          @@  ��\)�'��.{�V=qCv�ÿ�\)�z῱���
=Ct\)                                    Bxu8  �          @H�ÿ����-p����\��33Cw\)������
�޸R��Cs�                                    BxuF�  �          @8Q쿂�\��ÿǮ�=qCts3���\��{�ff�?�Cm�\                                    BxuUj  �          @9����(�� �׾����Cs�R��(��G�����(�Cq��                                    Bxud  �          @W
=�ٙ��'
=��Q���  Cl�R�ٙ��
=q����
  Cg�)                                    Bxur�  �          @aG���G��@  ��{��
=CwG���G��#�
���  Csٚ                                    Bxu�\  �          @Z=q�E��7
=��z���ffC~�׿E����
�H�%�HC{��                                    Bxu�  �          @W
=�(���0  ��ff�33C�H��(�����!G��Bz�C|�                                    Bxu��  �          @dz῁G��U����
C}!H��G��A녿�  ���C{�\                                    Bxu�N  �          @xQ쿋��fff�p���`z�C}�����K���Q���C{                                    Bxu��  �          @xQ쿳33�^{���\�tz�Cw𤿳33�A녿�p���33Cu�                                    Bxuʚ  �          @n{��=q�?\)=�Q�?�G�Cn����=q�8�ÿE��L(�Cm�R                                    Bxu�@  �          @z=q����J�H����G�Cj������?\)����}��Ch��                                    Bxu��  �          @����8���.�R�J=q�3\)C_ff�8����ÿ�ff��G�C[�\                                    Bxu��  T          @�z��@  �   �����{C[�\�@  � �׿�p���CU޸                                    Bxu2  �          @����>{� �׿ٙ���33C\��>{��33�
=���CT�)                                    Bxu�  �          @�G��=p��=q���ՅC[!H�=p���  �"�\�CR�                                    Bxu"~  �          @���1G���H�ff���C]8R�1G��ٙ��.{���CS�f                                    Bxu1$  �          @�����H��R��\��Ca����H��\�*�H�&�RCX.                                    Bxu?�  �          @tz��z���
�H�
�HCd� �z��{�0���9�
CY�                                     BxuNp  �          @u������H��\��Ch��녿���9���D��C\�q                                    Bxu]  �          @x��������
=�\)Cb����Ϳ�{�,���2�CX�                                    Bxuk�  �          @u����R�ff���C`u��녿\�*�H�133CU�f                                    Bxuzb  �          @tz��{�{�
=q�
ffCa  �{��p��.{�6\)CU�R                                    Bxu�  �          @mp��  ��\�
=�(�C^G��  ��=q�'��4ffCR�{                                    Bxu��  T          @l(��z��z���R�\)CdW
�z�У��%�2  CZB�                                    Bxu�T  �          @o\)�
=q�\)���H��p�Ce��
=q��\)�Q���C\�3                                    Bxu��  �          @tz�����ÿ���=qCa�q���޸R�(�� �CX�\                                    Bxuà  �          @p����R�Q쿺�H���C_����R���ff���CX��                                    Bxu�F  �          @mp���(��	���p����Cc����(���33�0  �C{CW^�                                    Bxu��  �          @qG�����,���
�H�(�Cr�q��녿�
=�8Q��I��Cj0�                                    Bxu�  �          @w
=��33�(��>{�Hp�CrG���33�����_\)�HCbB�                                    Bxu�8  �          @xQ쿵��\�1��7p�Cn5ÿ������U�n=qC_�
                                    Bxu�  �          @w
=��
=�$z��'
=�*�CuJ=��
=���P���g�Cj�f                                    Bxu�  �          @\(��L�Ϳ���'��\z�Ct��L�ͿG��?\).C`J=                                    Bxu**  �          @mp����7��   �p�Cw�����Q��1��C(�CqY�                                    Bxu8�  �          @p  �p���\)�   �,�\CyY��p�׿�\)�HQ��lz�Co�f                                    BxuGv  T          @u�
=q�W���(����C�o\�
=q�1G���H�"z�C�w
                                    BxuV  T          @z=q���
�I���
=q�	��C�C׽��
��AG��P��C��                                    Bxud�  �          @{������Fff�ff�\)C��{������R�K��Z�C�˅                                    Bxush  �          @w
=>�(��U������(�C��\>�(��&ff�3�
�;Q�C���                                    Bxu�  �          @|��>�33�S�
��
���C��>�33�!��>�R�E�\C���                                    Bxu��  �          @�Q�����P  ���G�C�ý�������Mp��Uz�C��                                     Bxu�Z  �          @{�>�  �9���)���)Q�C�p�>�  ��Q��Z=q�p\)C���                                    Bxu�   �          @z=q��\)�QG��   ���C�����\)�\)�:=q�D��C��q                                    Bxu��  �          @�  ��(��fff��
=����C�����(��:=q�-p��*�C�                                    Bxu�L  �          @��H��  �]p������
=C�����  �(Q��Fff�F{C�AH                                    Bxu��  T          @�{����^{�����C�������&ff�N{�I�C��                                    Bxu�  �          @�z�Tz��\(��������C�4{�Tz��'
=�Fff�B
=C|T{                                    Bxu�>  T          @�z�0���e��z��܏\C��=�0���4z��;��4�HC��                                    Bxu�  T          @�  �(��\�Ϳ�33���C��q�(��,(��8Q��9  C��)                                    Bxu�  �          @�  �J=q�]p���=q��{C��\�J=q�.{�3�
�3Q�C}�                                    Bxu#0  �          @�(��s33�h�ÿ���\)Cp��s33�<(��.{�%�\C|(�                                    Bxu1�  T          @�{�0���e����ffC����0���1G��C33�;\)C�R                                    Bxu@|  �          @��+��l(���=q��p�C�� �+��<(��8���/G�C��R                                    BxuO"  �          @�(��L���xQ쿈���q�C�'��L���Vff��R��C�B�                                    Bxu]�  �          @z�H����n{�������C�\)����L(�����	p�C���                                    Bxuln  �          @u��u�o\)�E��;
=C�,;u�S33��z���{C��\                                    Bxu{  �          @}p��333�n{��=q��C����333�L(�����
=C��=                                    Bxu��  �          @z�H��ff�s33�G��9�C��H��ff�Vff��Q����HC�0�                                    Bxu�`  �          @�����
�qG���{�}��C~���
�N�R�  �Q�C|T{                                    Bxu�  �          @��
�����l�Ϳ����
Cy𤿬���H�������Cv��                                    Bxu��  �          @tzῴz��XQ�W
=�O�CwW
��z��<(������G�CtT{                                    Bxu�R  T          @vff�����׿Ǯ�ʏ\C_:�����������=qCVaH                                    Bxu��  �          @qG��B�\��(���Q���Q�CM�\�B�\�\(��z���
CC�3                                    Bxu�  �          @fff�p����ÿ�{��(�CXz��p�������#z�CL��                                    Bxu�D  �          @c�
�5��Ǯ�����
CP�f�5����\����Q�CG�f                                    Bxu��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxu �   o          @�\)���H���
�O\)�1C�����H�h���ff���
C�(�                                    Bxu 6  �          @w��
=q�p  �=p��/�
C�箿
=q�Tz��z����C�^�                                    Bxu *�  �          @l�;����fff�E��@��C��׾����J=q�����z�C�P�                                    Bxu 9�  �          @�����
��녾����
=C�q콣�
�mp��ٙ�����C�c�                                    Bxu H(  �          @��H�\)�y���O\)�9�C��=�\)�[�����\C�\)                                    Bxu V�  �          @�=q?���~�R�
=q����C�E?���e����
=C���                                    Bxu et  �          @��H?8Q��~�R�L���1�C�)?8Q��l�Ϳ�p�����C�}q                                    Bxu t  �          @�z�>���=q��  �Z�HC�Z�>��qG��Ǯ���\C���                                    Bxu ��  
�          @�ff��=q���ͼ��
��z�C�#׾�=q�z�H�������\C��                                    Bxu �f  �          @�녿5�{��\)��
=C��q�5�a녿������C�L�                                    Bxu �  �          @��
���\�{��z��C�����\�aG���=q�ԸRC}�                                    Bxu ��  �          @�\)��33���Ϳ0����RC��þ�33�l(��G���=qC�H�                                    Bxu �X  �          @������~�R��(��ÅC������g���(���p�C�0�                                    Bxu ��  �          @�33�+��x�ÿ���q�C�  �+��Tz���\�
=C�O\                                    Bxu ڤ  �          @�G����b�\��=q���C�uÿ��/\)�8���7��C�(�                                    Bxu �J  �          @��׾�z��j=q��z���ffC��R��z��?\)�#33�!=qC�4{                                    Bxu ��  �          @����0���p  ������33C��Ϳ0���Fff�\)���C��q                                    Bxu!�  �          @\)�\�L(����
�ՅCt�=�\��H�/\)�/{Cm�)                                    Bxu!<  �          @y����z��HQ쿗
=��ffCn�)��z��#�
�
=q���CiE                                    Bxu!#�  �          @~{��\�P�׿�����HCq� ��\�'
=����Q�Ck�{                                    Bxu!2�  �          @��
��ff�U��������RCq����ff�'
=�&ff�33CkQ�                                    Bxu!A.  �          @��\����Z�H���
��33Cy�����(Q��4z��/z�Csu�                                    Bxu!O�  �          @}p����R�HQ��z�����CxaH���R�  �@  �E�RCq(�                                    Bxu!^z  �          @l�Ϳ\)�(Q�� ���+�C��{�\)����P  �w�HCz�R                                    Bxu!m   �          @l�ͽ�\)�0����H�%{C�O\��\)���
�Mp��s��C��                                    Bxu!{�  �          @k���
=�.{���%��C��{��
=�޸R�L���s�C�+�                                    Bxu!�l  �          @j=q=�Q��@  ��\�	�C�ٚ=�Q��Q��<(��X(�C�1�                                    Bxu!�  �          @j=q>���<���ff�{C���>���z��>{�[�C���                                    Bxu!��  �          @tz���HQ��	���	��C��=���p��E�YG�C�w
                                    Bxu!�^  �          @u��ff�N�R� ����\)C�׾�ff�ff�?\)�M33C��                                    Bxu!�  �          @o\)���
�E������C��=���
��\�/\)�H�C��
                                    Bxu!Ӫ  �          @p��?����P�׿�Q���{C��?����$z��{�&{C�C�                                    Bxu!�P  �          @l��?&ff�^{��ff��=qC�L�?&ff�9���
�H��\C�N                                    Bxu!��  �          @n{>u�\(���=q��z�C���>u�1G����$33C�s3                                    Bxu!��  
�          @h�ÿk��@  ��ff��(�C|�ÿk�����R�5  Cx
                                    Bxu"B  �          @j=q����7���
=���C��\������333�T(�C~�\                                    Bxu"�  �          @s33?n{�Q녿8Q��>�RC��\?n{�5������
C�q                                    Bxu"+�  �          @z=q?�
=���4z��?Q�C���?�
=���\�XQ��z�C�AH                                    Bxu":4  �          @w�?�(���
=�B�\�Nz�C��H?�(����[��y33C��                                    Bxu"H�  �          @j�H��\)��Q��B�\�e  C����\)�B�\�a��Cz                                      Bxu"W�  �          @q녾���W
=�Tz�33C|�{���>�Q��Z=q¥{B��=                                    Bxu"f&  �          @|(��k��w���\��Q�C�O\�k��\�Ϳ�=q�޸RC��                                    Bxu"t�  �          @e��Ǯ�`�׾�����C�Ф�Ǯ�HQ��
=��ffC�k�                                    Bxu"�r  �          @b�\�B�\�`  �\��
=C�l;B�\�I����=q���C�>�                                    Bxu"�  �          @n�R<��
�Q녿˅�Σ�C�,�<��
�!G��(���9G�C�9�                                    Bxu"��  �          @n{>���Q��1G��C  C���>�׿�G��[�(�C�B�                                    Bxu"�d  �          @u����
�AG�����ޏ\Cs#׿��
����/\)�6Q�Ck.                                    Bxu"�
  �          @u��  �4z��p���Cv���  ��{�C�
�W\)Cl!H                                    Bxu"̰  �          @��H��p��G
=��  �ʣ�Cmk���p���\�.�R�'��Ce�                                    Bxu"�V  �          @������R�@�׿�\��p�Cl�)���R�(��.{�*��Cc��                                    Bxu"��  �          @����(Q��0�׿����
Cbn�(Q�����   ��=qC[��                                    Bxu"��  �          @z�H�(��%���33����Ce���(���p��,(��/��CZT{                                    Bxu#H  �          @|(����
�:=q��Q���G�Cnz���
���6ff�:(�Cd��                                    Bxu#�  �          @��׿����>{������Cl���������1G��/�RCcT{                                    Bxu#$�  �          @��H��
�:�H���R��\)Cj����
� ���9���4�C`h�                                    Bxu#3:  �          @����	���@�׿У���(�Cj��	����R�&ff�   Cb�                                    Bxu#A�  �          @��\����C33�������RCip�������(��Ca�)                                    Bxu#P�  �          @�
=����N{�
�H��Q�Cs\�����R�L(��D=qCi�                                    Bxu#_,  �          @��׿�ff�J�H�����
Cs����ff�ff�XQ��QQ�Ciz�                                    Bxu#m�  �          @���Ǯ�J�H���Cs�Ǯ�
=�Tz��N�Ci��                                    Bxu#|x  
�          @�ff��p��3�
�#33��RCn\)��p��ٙ��X���Y��C`�                                     Bxu#�  �          @�녿��� ���#�
��
Cj��������S33�]33CZT{                                    Bxu#��  �          @u��z��
=q�(���Cb=q�z῜(��4z��FQ�CR�\                                    Bxu#�j  �          @p���!G��z��\)��(�C[\)�!G���  �\)�%�CNc�                                    Bxu#�  �          @tz��=q��R�p��z�Ci� ��=q��  �=p��M
=C[ff                                    Bxu#Ŷ  T          @p�׿����\��33Cf�=��׿��
�@���S�RCV.                                    Bxu#�\  �          @qG��n{�1G��Q��  C{}q�n{���H�N�R�k�Cq��                                    Bxu#�  �          @tz�\)�?\)�z��(�C��{�\)��Q��P  �hC}�f                                    Bxu#�  �          @�33�h���Mp��Q��
=C~(��h���
=�X���_=qCv��                                    Bxu$ N  �          @u���
=�.{�  �  CrY���
=���H�Fff�Y33Cf)                                    Bxu$�  �          @�Q���\��R�ff���Cf�
��\��Q��Fff�L��CWJ=                                    Bxu$�  �          @��׿�=q�6ff�&ff�(�Cm8R��=q��Q��^{�Y�C^�f                                    Bxu$,@  T          @����\)�%�\)�(�Ce!H�\)���R�QG��JG�CU�
                                    Bxu$:�  �          @�33����L(���H�	\)CtJ=����z��[��T(�Cin                                    Bxu$I�  �          @�(���G��^{����(�Cs!H��G��{�J�H�8�Cj�
                                    Bxu$X2  �          @�����H�l(�� ���ۅC{�{���H�+��O\)�?(�Cu�                                    Bxu$f�  �          @��R=�Q������(���\)C���=�Q��K��G��1�HC��3                                    Bxu$u~  �          @��=�\)��=q��
=���C�y�=�\)�I���Dz��133C��q                                    Bxu$�$  �          @�z��G���녿\��C��\��G��L���;��(�
C�q                                    Bxu$��  �          @��Ϳ333��{��{�jffC�G��333�^{�&ff���C�S3                                    Bxu$�p  �          @�(��u��(���\)���HC�XR�u�S�
�4z��!(�C��                                    Bxu$�  �          @�Q�?
=�j�H��33���RC���?
=�8Q��*�H�){C���                                    Bxu$��  �          @�G�?:�H�hQ�У�����C���?:�H�0  �7��4�
C�o\                                    Bxu$�b  �          @��H?s33�j�H�\��=qC�Ff?s33�5��2�\�,  C�N                                    Bxu$�  
�          @�ff?���p  ��ff��
=C��)?���?\)�'��(�C��{                                    Bxu$�  �          @���?����^{��=q��G�C�aH?����-p��"�\��C�Ff                                    Bxu$�T  �          @�(���\)�j=q�z�H�n�HC�lͽ�\)�AG���\���C�N                                    Bxu%�  �          @��\�\(����
�aG��<��C���\(��^�R�Q���C�3                                    Bxu%�  �          @�=q��  ��
=�Tz��1�C�K���  �e����C���                                    Bxu%%F  �          @�녾�  ��  �����
C�N��  �n�R�
=��p�C��                                    Bxu%3�  �          @��R��Q�����>���@��\C�~���Q��|�Ϳ�ff��C�^�                                    Bxu%B�  �          @�
=?
=q���׿c�
�G
=C�Ф?
=q�W��
=�
�\C���                                    Bxu%Q8  �          @|�;���hQ쿂�\�z{C�Ф����=p��z��Q�C���                                    Bxu%_�  �          @hQ�n{�2�\?O\)Aw\)C{���n{�8�þ��
���RC|�                                    Bxu%n�  �          @u������W
=    �#�
C{ff�����Fff������HCy�R                                    Bxu%}*  �          @��\��
=�fff����RCu��
=�Fff��z���p�Cq��                                    Bxu%��  �          @u���(��^{�Ǯ���HCw���(��C�
��Q���Q�Ct\)                                    Bxu%�v  T          @k�����-p��:�H�>�RCg�3����{��(�����Cb�                                    Bxu%�  �          @tz��Q��1녾���  Chp��Q��Q��  �ȸRCd#�                                    Bxu%��  �          @l(���
=�P  �Ǯ���
Cr����
=�6ff��\)���HCou�                                    Bxu%�h  �          @j�H��Q��N{��G���CrJ=��Q��333��z����HCn�
                                    Bxu%�  �          @i����Q��AG��
=��CmE��Q��#33��(���Ch                                    Bxu%�  �          @w���\�X�ÿ#�
��Crp���\�7���z���CnaH                                    Bxu%�Z  �          @h�ÿ����AG���z�����Cu�ÿ����\)��R�.{Cn�H                                    Bxu&   �          @c�
�#�
�Mp���(���p�C�b��#�
��R���+p�C�{                                    Bxu&�  �          @x��?�G��mp��   ��(�C���?�G��N{��z����C��)                                    Bxu&L  �          @�\)?���}p�������p�C��{?���_\)����Q�C��f                                    Bxu&,�  �          @�
=?�p��z�H�\)��C�Z�?�p��c33����p�C�T{                                    Bxu&;�  �          @�p�?h����G���Q쿘Q�C�O\?h���k������C��f                                    Bxu&J>  �          @��R?�  �\)������ffC��?�  �`�׿�Q���C�Ǯ                                    Bxu&X�  
�          @��R?����}p�����\)C�8R?����Z=q������C���                                    Bxu&g�  T          @�?����vff�0����C�Q�?����QG��	����ffC���                                    Bxu&v0  �          @�?����xQ�8Q���HC�@ ?����_\)�ٙ���33C�Ff                                    Bxu&��  �          @�Q�?W
=�r�\?B�\A2{C�8R?W
=�q녿L���;
=C�<)                                    Bxu&�|  �          @~�R?E��xQ�>W
=@?\)C��H?E��h�ÿ������C��q                                    Bxu&�"  �          @�Q�?Q��x�þ�z���ffC���?Q��\�Ϳ��أ�C���                                    Bxu&��  �          @s�
>���c33���\���RC�T{>���0  �%��+z�C�H�                                    Bxu&�n  �          @s�
>��R�i���0���+
=C�q�>��R�Dz���p�C��                                    Bxu&�  �          @q녿(��k��8Q��,(�C�H��(��S33�У���33C�                                    Bxu&ܺ  �          @n{?B�\�c33�O\)�Hz�C��?B�\�;��
=q�
=C�N                                    Bxu&�`  �          @q�?O\)�g
=�.{�'
=C�T{?O\)�A��z���C�z�                                    Bxu&�  �          @p  ?!G��g
=�E��=C���?!G��?\)�	����C��R                                    Bxu'�  
�          @|��?��H�`  ��z�����C���?��H�/\)�{�  C��{                                    Bxu'R  
�          @�{?�=q�w��L���0��C�|)?�=q�N{��\�Q�C�9�                                    Bxu'%�  T          @���?�ff�mp�����o\)C�\)?�ff�R�\�޸R��ffC��                                    Bxu'4�  �          @���?����]p��
=q��Q�C���?����<�Ϳ�33��C��q                                    Bxu'CD  �          @�Q�@���W
=���
���HC�AH@���<�Ϳ�33��p�C�                                    Bxu'Q�  �          @���@,���<(���
=��ffC�K�@,���!G��������\C���                                    Bxu'`�  �          @�  @)���>�R�W
=�<��C��{@)���(�ÿ�33��  C���                                    Bxu'o6  �          @}p�@�R�Dz�u�`��C�}q@�R�-p���(���z�C�G�                                    Bxu'}�  �          @z�H?�{�c33�#�
�(��C�1�?�{�N{���R���C�C�                                    Bxu'��  �          @u�?ٙ��Z�H<#�
>��C�>�?ٙ��HQ쿱���=qC�L�                                    Bxu'�(  �          @z=q?�=q�p  ��G����C�f?�=q�XQ�У���\)C���                                    Bxu'��  �          @w�?�
=�k���\)��ffC��f?�
=�U�������  C��                                    Bxu'�t  �          @w
=?��
�g��aG��Q�C��q?��
�N{��Q����C�޸                                    Bxu'�  T          @s33?\�]p���=q��33C��3?\�C33����33C�<)                                    Bxu'��  T          @n�R@��HQ�#�
���C�u�@��2�\��
=��=qC�H                                    Bxu'�f  �          @n�R?����N�R>8Q�@-p�C��{?����@�׿���33C��\                                    Bxu'�  �          @e�?�  �U�>��
@��C�P�?�  �J=q��=q��p�C��                                    Bxu(�  �          @p  ?���e���������C�E?���HQ��  �޸RC�\)                                    Bxu(X  �          @|��?p���p�׿0��� z�C��q?p���H�������C�N                                    Bxu(�  �          @}p�?L���n�R�u�c33C�
=?L���?\)��H�=qC�w
                                    Bxu(-�  �          @}p�?   �qG���������C���?   �>{�#�
�!z�C�Ǯ                                    Bxu(<J  �          @
�H�(�����u��
Cts3�(��E���z��\{Cgff                                    Bxu(J�  �          ?�����  �#�
�333��  C4���  >�\)�#�
��ffC)u�                                    Bxu(Y�  �          ?��>��Ϳ�z�>�  A�C��f>��Ϳ�{����|  C���                                    Bxu(h<  �          @�\?�녿�?@  A��HC��?�녿�\)>B�\@���C��H                                    Bxu(v�  �          @   ?��ÿǮ=#�
?��C�%?��ÿ�Q��R��C�Ff                                    Bxu(��  �          @4zῥ�����������Cl�������ff���4�Ca0�                                    Bxu(�.  �          @AG������������CVW
���!G���-��CDu�                                    Bxu(��  �          @?\)�:=q��33��\��RC:Ǯ�:=q���Ϳ(��=p�C5�                                    Bxu(�z  �          @k��H�ÿ&ff��z���33C?���H��=�G����
��ffC1�q                                    Bxu(�   �          @mp��W���Q��G�����C:��W�>��R�\��{C.�3                                    Bxu(��  �          @i���Y�������H��
=C;��Y��=�Q쿥����C2h�                                    Bxu(�l  �          @c33�Z�H>.{�fff�k�C1{�Z�H?
=q�=p��Ap�C+
=                                    Bxu(�  �          @dz��Z�H?c�
�����C%s3�Z�H?�G��u�p��C#�
                                    Bxu(��  �          @dz��U�?�p���33��C� �U�?�G�>L��@P  C=q                                    Bxu)	^  �          @\(��P��?u��ff��  C#���P��?��    =#�
C"                                    Bxu)  �          @_\)�Mp�?�녿G��O\)C }q�Mp�?�{�k��l(�C                                      Bxu)&�  �          @U�AG�?�  �p����=qC!��AG�?�ff��(���{C�                                    Bxu)5P  �          @N{�1G�?�녿�����p�C�=�1G�?�G���\�=qCxR                                    Bxu)C�  �          @Fff�p�?��Ϳ�
=����C5��p�@33�(��6{C	(�                                    Bxu)R�  �          @L(��{?�\)���R����C{�{?�\)�B�\�]G�C�
                                    Bxu)aB  �          @\(��7
=?��Ϳ����=qC���7
=?�ff�#�
�*�\C޸                                    Bxu)o�  �          @Q��,��?8Q쿋����RC%
=�,��?�\)�.{�PQ�CxR                                    Bxu)~�  �          @'�=L���������ָRC���=L�Ϳ��R��
=�Q=qC��                                    Bxu)�4  �          @	��?Y����33���H(�C��q?Y���\���R��RC���                                    Bxu)��  �          @S�
?����>{�#�
�uC��?����*�H�����=qC�G�                                    Bxu)��  �          @e?�(��P  ���R��z�C�0�?�(��333��
=��C��                                     Bxu)�&  �          @j=q?�  �N{�Tz��S�C�u�?�  �"�\�	����C�<)                                    Bxu)��  �          @e�?�33�Mp��333�4Q�C��H?�33�%�G��	�
C�(�                                    Bxu)�r  �          @`��?����@�׿8Q��?33C��?���������H��\C��                                     Bxu)�  �          @i��?У��I���J=q�I��C���?У��\)�����C��)                                    Bxu)�  �          @�Q�?��H�`�׿�  ���C�H�?��H�'��*=q�&�C���                                    Bxu*d  �          @�p�?�p��l(���  ��\)C��?�p��1��.�R�#C�
=                                    Bxu*
  �          @��\?��
�hQ��  ��p�C��\?��
�'
=�;��9  C�Ǯ                                    Bxu*�  �          @�=q?z�H�n{�������C�k�?z�H�1G��3�
�.C��=                                    Bxu*.V  �          @���?�  �c�
�\(��L��C���?�  �5����
=C��H                                    Bxu*<�  �          @y��@
=�N�R�#�
�.{C��3@
=�9����
=��p�C�f                                    Bxu*K�  �          @}p�?�\)�Tz�?z�HAfffC��=?�\)�Z�H�z��33C�T{                                    Bxu*ZH  �          @|(�?�Q��I��?���A�p�C��
?�Q��W
=���R����C�
=                                    Bxu*h�  �          @�  @p��HQ�?�\)A�(�C���@p��S33��p����HC��                                    Bxu*w�  �          @u@ ���B�\?n{AeC��)@ ���H�ÿ   ��p�C�O\                                    Bxu*�:  �          @h��@���?�A�(�C�~�@��.{>�  @�G�C�Q�                                    Bxu*��  �          @l(�@
�H�333?�  A}p�C�޸@
�H�<�;�{��\)C�                                      Bxu*��  �          @�p�@��Z=q��\)�xQ�C�ff@��;���G���=qC�xR                                    Bxu*�,  �          @�@!G��g�����C�l�@!G��L(���(���p�C�*=                                    Bxu*��  �          @���@5�^�R���R�x��C��q@5�?\)������z�C���                                    Bxu*�x  �          @�=q@AG��Vff�8Q��G�C��@AG��+��Q�����C�33                                    Bxu*�  �          @~�R@ff�J=q��G��ʏ\C�P�@ff�(�ÿ�ff��C���                                    Bxu*��  �          @���@1G��S33�5��RC��@1G��(Q��
=����C�<)                                    Bxu*�j  �          @�(�@!��H�ÿQ��8��C�g�@!���������z�C���                                    Bxu+
  �          @vff@��5�����
=C��
@�� ����
�\)C��3                                    Bxu+�  �          @w�@ff�5��ff�~�HC��=@ff�z��p���C�K�                                    Bxu+'\  �          @\)@
=���z���C�o\@
=��Q��:=q�>��C��{                                    Bxu+6  �          @}p�@(���ff�!��$�C��f@(���  �=p��Ip�C��                                    Bxu+D�  �          @x��@ff��=q�+��/G�C�z�@ff��(��Mp��`��C�(�                                    Bxu+SN  �          @��?���Q����� \)C���?�׿Tz��E�a��C�'�                                    Bxu+a�  �          @~�R?�33�U��������C�S3?�33�G��<(��?G�C�Ф                                    Bxu+p�  �          @�  ?�{�J�H��ff��
=C�|)?�{�G��B�\�Fz�C�S3                                    Bxu+@  �          @���?�z��>�R��33��C�]q?�z��ff�B�\�D�\C�`                                     Bxu+��  �          @���@z��0���Q����C�p�@z��  �I���K�RC��                                    Bxu+��  �          @���@(��-p��   ��33C��H@(���G��@���B�C��q                                    Bxu+�2  �          @���@L(��ٙ��G��陚C���@L(�����$z���C���                                    Bxu+��  �          @�G�@X�ÿٙ�� �����C���@X�ÿ���#�
�ffC��R                                    Bxu+�~  �          @��@c33��Q��p���(�C�C�@c33�8Q���
�(�C�C�                                    Bxu+�$  �          @���@e��p���\�Ù�C�˅@e���  ��(�C���                                    Bxu+��  �          @��
@aG�����������C��)@aG��
=q�z���C���                                    Bxu+�p  T          @~�R@Tz��  ��=q��\)C�޸@Tz�(�����C��                                    Bxu,  T          @�Q�@Z�H��
=��
=�ƣ�C�y�@Z�H��z������RC���                                    Bxu,�  �          @�Q�@X�ÿ�ff������Q�C�\)@X�þW
=��z���z�C�C�                                    Bxu, b  T          @\)?����S�
��=q�
=C��{?��������R��RC���                                    Bxu,/  �          @�z�?��\(���Q���33C��)?�����7
=�.�C��                                    Bxu,=�  �          @�p�?���aG���\)��C��q?���Q��C�
�?��C�{                                    Bxu,LT  �          @�{?�ff�n{��  ��G�C���?�ff�'��B�\�<z�C��                                    Bxu,Z�  �          @��R?��Z=q�����=qC��?�����J=q�C33C���                                    Bxu,i�  �          @��?����P  ��p��ޣ�C��3?��Ϳ�p��P  �H��C���                                    Bxu,xF  �          @���@!��AG��˅��p�C��@!녿����333�%
=C�'�                                    Bxu,��  �          @�\)@��Q녿У���ffC�5�@��
=q�=p��2��C�H                                    Bxu,��  �          @�G�?�{�U�������\C��?�{���H�\(��V(�C��                                    Bxu,�8  �          @�
=?�p��G
=���

=C���?�p���z��b�\�gQ�C���                                    Bxu,��  �          @��@(���(Q��������C���@(�ÿ�p��5��,p�C�W
                                    Bxu,��  �          @�p�?�=q�Mp������HC��?�=q��{�W
=�X�C��                                    Bxu,�*  �          @�?��J=q�'��33C��?���=q�r�\�=C��                                    Bxu,��  T          @�{>��HQ��+��!{C�,�>����
�u���C�`                                     Bxu,�v  �          @�ff@
�H�0�׿���ᙚC�3@
�H�����<���?(�C��                                    Bxu,�  �          @�G�@o\)�����@  �"ffC�AH@o\)��\)������z�C��\                                    Bxu-
�  �          @���@p�׿�p������C�q@p�׿����=q���HC��                                     Bxu-h  �          @��H@c�
��H�����\)C��@c�
��׿�{����C�
=                                    Bxu-(  �          @�33@g
=��\�E��#�C��\@g
=����  ��(�C��H                                    Bxu-6�  �          @��@j�H��ͿE��#�C���@j�H�˅���H���C�Q�                                    Bxu-EZ  �          @�(�@mp��p��@  �p�C��{@mp���{�ٙ���ffC�K�                                    Bxu-T   �          @�z�@h���
=�.{��RC���@h�ÿ�G����H��
=C�3                                    Bxu-b�  �          @���@dz�� �׾��H��{C�u�@dz��p���{��=qC�y�                                    Bxu-qL  �          @�p�@XQ��4zᾮ{���
C��@XQ�����\)���\C���                                    Bxu-�  �          @��@\(��.�R�B�\�p�C��f@\(��zΌ����=qC���                                    Bxu-��  �          @�{@J=q�Fff�aG��2�\C�Ǯ@J=q�(Q��33���HC��                                    Bxu-�>  
�          @�
=@333�]p��8Q��Q�C�~�@333�>{��ff��  C���                                    Bxu-��  �          @��R@%�a녿J=q�#\)C�!H@%�/\)�
=� =qC���                                    Bxu-��  �          @�p�@!��[���33�qG�C�33@!���R�(Q��z�C�Ǯ                                    Bxu-�0  �          @�  @���i���p���B�HC��H@���1G��#�
��RC�k�                                    Bxu-��  �          @�Q�@�
�l(������h��C���@�
�.{�/\)�ffC�*=                                    Bxu-�|  �          @�Q�@p��c�
��(��z�HC�Z�@p��#�
�0���G�C���                                    Bxu-�"  �          @���@  �g
=��(���  C��
@  ��R�@���'z�C�R                                    Bxu.�  �          @�G�@�c�
���
��=qC���@����B�\�(��C�&f                                    Bxu.n  �          @�33@Q��q녿�  ��{C��@Q��'��G
=�*�HC���                                    Bxu.!  �          @�{@33�~�R��33����C���@33�5��G��'  C��3                                    Bxu./�  �          @��R?�����  ���R��z�C��q?����3�
�N{�-=qC�ff                                    Bxu.>`  �          @��\?Ǯ���>�=q@^�RC�{?Ǯ�w
=���
��p�C��)                                    Bxu.M  �          @�=q?���zᾳ33��
=C�H?��^{��\��C�                                      Bxu.[�  �          @���?�  ��ff�Q��&ffC��
?�  �S33�.�R��
C�>�                                    Bxu.jR  �          @��?��
�����=q�W�C�'�?��
�l���z���\)C���                                    Bxu.x�  �          @��?����33�8Q����C��?���^{�.{���C��R                                    Bxu.��  �          @�=q?�����ÿB�\�G�C�z�?���Y���.{�  C���                                    Bxu.�D  �          @�33?s33��(����\�MG�C��?s33�Vff�@  �"�
C��                                    Bxu.��  �          @��\?�������Tz��'�C��?����[��5���C��                                    Bxu.��  �          @�33?ٙ���\)��\�˅C���?ٙ��^{��R��C��                                    Bxu.�6  �          @�33@.{�`  ��p��xz�C��@.{�{�0����C�޸                                    Bxu.��  �          @��
@,(��i���\(��*�HC�*=@,(��2�\� �����C��
                                    Bxu.߂  �          @�33@@���Z=q�5�C��
@@���(��������C�XR                                    Bxu.�(  �          @�(�@Y���B�\�J=q�G�C�  @Y�����
=q��Q�C��                                    Bxu.��  �          @�33@XQ��C33�#�
� ��C��R@XQ��
=��\�ҏ\C���                                    Bxu/t  �          @�33@[��;��^�R�/�C��q@[����������C�H                                    Bxu/  �          @�(�@/\)�^�R�s33�@��C��@/\)�%� ����C�G�                                    Bxu/(�  T          @��?�p���=q�\(��+�C���?�p��I���.{�  C��                                    Bxu/7f  �          @�?�Q���ff�.{�p�C�h�?�Q��U�(Q���C�{                                    Bxu/F  �          @���@33���\�c�
�0��C�T{@33�HQ��0�����C���                                    Bxu/T�  �          @�ff@�
�|�Ϳ���P  C�!H@�
�<(��5��C�                                    Bxu/cX  �          @�p�@�
�w����H�qG�C�g�@�
�2�\�<(��p�C�˅                                    Bxu/q�  �          @��R@33��z�^�R�*{C�&f@33�L���1���C�Q�                                    Bxu/��  �          @��
@p��n{��Q��n�HC��
@p��*�H�6ff��C�O\                                    Bxu/�J  �          @��H@4z��W���  �}p�C��{@4z���.�R���C�0�                                    Bxu/��  �          @��\@;��S33���H�uG�C���@;���\�*�H��\C�                                    Bxu/��  �          @��@6ff�Z=q��Q��o�C���@6ff����-p��  C��)                                    Bxu/�<  �          @�(�@0���`�׿�Q��o\)C��@0����R�0�����C�f                                    Bxu/��  �          @�z�@G��h�ÿ}p��O�
C��3@G��,(��(�����C�                                    Bxu/؈  �          @�
=�����
�z���p�C��;��S�
�!G��(�C��q                                    Bxu/�.  �          @��þ���33�Y���;�C�q���J=q�0���$Q�C�ٚ                                    Bxu/��  �          @���=����Y���7�
C�Ф=��N{�2�\�#�C�                                    Bxu0z  �          @��\�W
=�vff�E��5C�y��W
=�>{�#�
�"��C�f                                    Bxu0   �          @w�����e���33���Cy𤿧��<���z����Cv\                                    Bxu0!�  �          @p�����A녾aG��Z=qCj=q���!녿�
=��(�Ce5�                                    Bxu00l  �          @tz����C�
��R���CjxR���
=��\���Cc33                                    Bxu0?  �          @p  �   �.�R��\��33CcxR�   ���������C\L�                                    Bxu0M�  �          @r�\�
=�3�
�aG��W
=Cf  �
=� ���	���
�C\u�                                    Bxu0\^  �          @s�
��@  ��G��w�Ck)��
=���  Ca:�                                    Bxu0k  �          @n{�#33�*�H��\��CbQ��#33�z��\��Q�C[                                    Bxu0y�  T          @n�R�{�%��Q��M�CbB��{�����R�G�CX��                                    Bxu0�P  �          @n{�33�?\)�Q��L��Ck���33�(��(���RCb�{                                    Bxu0��  �          @k�����;������G�CmJ=��׿�p��(��'Q�Cbp�                                    Bxu0��  �          @qG���33�4z`\)��\)Ck�3��33��\�%�3p�C^޸                                    Bxu0�B  �          @n�R�.�R��Ϳ��R���
CZ���.�R��ff�
�H��RCMc�                                    Bxu0��  �          @i��>#�
�*�H��H�(��C��>#�
��{�Z�H��C�7
                                    Bxu0ю  �          @k�?(���L(��Tz��c\)C��?(���
=�33�,��C��                                    Bxu0�4  �          @s33?����R�\>\@���C�z�?����B�\������
C�q�                                    Bxu0��  �          @mp�@.�R��;�Q����C��@.�R��Q���
��ffC�J=                                    Bxu0��  �          @qG�@?\)�G�������z�C�g�@?\)���ÿ�33���C�^�                                    Bxu1&  �          @j=q@>�R��ýu�}p�C�.@>�R���ÿ�\)��Q�C�K�                                    Bxu1�  �          @q�@E�
�H    �L��C�t{@E��׿����C�XR                                    Bxu1)r  T          @w
=@Fff�  >Ǯ@�33C��R@Fff�	���J=q�<z�C��q                                    Bxu18  �          @p��@HQ���R����{�C��)@HQ���Ϳ�(�����C�t{                                    Bxu1F�  �          @hQ�@N{������
���C��)@N{��\)����י�C���                                    Bxu1Ud  �          @g�@G
=��{��  ����C�,�@G
=�\)��G�����C��                                    Bxu1d
  �          @j�H@N�R���R�^�R�Z�RC�� @N�R�\(���  ����C���                                    Bxu1r�  �          @j�H@Mp���
=������C�  @Mp��8Q��\)��33C��
                                    Bxu1�V  �          @hQ�@=p���{��������C�� @=p��:�H�����(�C�{                                    Bxu1��  �          @k�@L(���33�@  �<��C�J=@L(���ff��p���Q�C��)                                    Bxu1��  �          @q�@Q녿��
������
C���@Q녿�ff��ff��\)C�(�                                    Bxu1�H  �          @n{@W��\���
��(�C�ٚ@W����\�Tz��PQ�C���                                    Bxu1��  �          @s33@]p���ff�#�
�
=C���@]p���G��h���_
=C�                                      Bxu1ʔ  �          @s33@S33��ff����{C��@S33�����{��  C�N                                    Bxu1�:  �          @s�
@_\)��{�(��\)C�XR@_\)�^�R���H����C�H                                    Bxu1��  �          @{�@q녿fff���G�C�P�@q녾��H�p���^�\C�P�                                    Bxu1��  �          @xQ�@_\)��33�G��;33C�3@_\)�Q녿�����=qC�l�                                    Bxu2,  �          @w�@i������{��z�C�R@i���W
=�c�
�UC���                                    Bxu2�  �          @xQ�@u��\)��G��ٙ�C�Ф@u���(��\��{C��                                    Bxu2"x  �          @x��@xQ�<�=��
?�z�>�p�@xQ켣�
=��
?�Q�C��q                                    Bxu21  �          @y��@x��=u����
=?Tz�@x��=u    =u?k�                                    Bxu2?�  �          @r�\@q녾W
=����G�C�c�@q녽��W
=�I��C�                                    Bxu2Nj  �          @p��@o\)��z�k��^�RC���@o\)����{���RC�                                      Bxu2]  �          @n�R@mp������W
=�N{C���@mp����������z�C��                                    Bxu2k�  �          @o\)@l��>B�\�������@:�H@l��>�p���  �u�@�z�                                    Bxu2z\  �          @s33@j�H?z�E��<Q�A�
@j�H?c�
��p���(�AZ�H                                    Bxu2�  �          @j�H@(��^�R��  ��Q�C�/\@(���G��\��33C���                                    Bxu2��  �          @r�\=u�hQ쿇���z�C�xR=u�%��0���;p�C���                                    Bxu2�N  T          @y��>�Q��l(���
=��G�C��>�Q��$z��8���@(�C�H                                    Bxu2��  �          @��
��Q��xQ��G���z�C|�R��Q��G��ff��Cy#�                                    Bxu2Ú  �          @��
����vff����hQ�Cz#׿���K���� �RCvaH                                    Bxu2�@  �          @|�Ϳ����l(��������Cy�R�����@���
�H�G�Cu�f                                    Bxu2��  �          @vff����a녿#�
�33Cy������-p��
=�33Ct@                                     Bxu2�  �          @e��%���ff>aG�@|��CV�
�%���z�:�H�V=qCT��                                    Bxu2�2  �          @\(��Tz�B�\?Y��Ae�C7L��Tz���?!G�A)C>5�                                    Bxu3�  �          @dz��^{?.{>\@���C(�)�^{>Ǯ?.{A/�C-��                                    Bxu3~  �          @g��e�>������HC,ff�e�>��H=�G�?�ffC,:�                                    Bxu3*$  �          @h���aG�>��ͿB�\�BffC-�
�aG�?:�H��ff��C(G�                                    Bxu38�  �          @e��W
=��z�>Ǯ@�{CG\�W
=��Q쾅���
=CG��                                    Bxu3Gp  �          @g��[���{���ͿУ�CE�H�[��fff�(���'\)CB��                                    Bxu3V  �          @^�R�Tz�Q녿���
CA���Tz��(��^�R�i�C;k�                                    Bxu3d�  T          @Y���P  �z�H�#�
��G�CD���P  �W
=��\�
{CB�                                     Bxu3sb  �          @Y���R�\�0�׾�(����C?���R�\��p��8Q��Dz�C:k�                                    Bxu3�  �          @N�R�L(�=��B�\�XQ�C1���L(�>L�ͽ��Ϳ���C0u�                                    Bxu3��  �          @Tz��C33=��aG���
=C1��C33?\)�0���I�C)��                                    Bxu3�T  �          @XQ��I���B�\��Q����HC7s3�I��>�ff��{����C+�                                    Bxu3��  �          @Fff�'
=�����u��Q�CL�'
=�
=q��Q����C?��                                    Bxu3��  �          @=p��녿�(����\��z�CT���녿=p��У����CE�q                                    Bxu3�F  �          @Fff� �׿�Q쾽p����
CU��� �׿�G���Q����RCN��                                    Bxu3��  �          @9���,�Ϳ��?�\A'\)C@���,�ͿG�>�@%CD�                                    Bxu3�  �          @^{�Mp�����?+�A3�CGQ��Mp����    ��CJ:�                                    Bxu3�8  T          @]p��A녿��\?W
=Ae��CJ���A녿\=�\)?�{CN�f                                    Bxu4�  T          @[������=q���p�CYh������ff����؏\CPz�                                    Bxu4�  
�          @@���Q쿥�>�
=AffCP���Q쿪=q������(�CQ&f                                    Bxu4#*  T          @J�H�����(��\��HC��쾨�ÿ�
=�����!Q�C�k�                                    Bxu41�  �          @A�@  ��G��s33���HC��)@  ���\��(��z�C��
                                    Bxu4@v  �          @L��@=q��33�\)�%G�C��@=q������H��\)C�~�                                    Bxu4O  �          @Tz�?L���H�þ�����Q�C�&f?L���!G���33�
=C�Ф                                    Bxu4]�  �          @aG��aG��Z�H��p��ƸRC�!H�aG��/\)��(�C��=                                    Bxu4lh  �          @tz�:�H�l(�������C�` �:�H�7��Q����C��                                    Bxu4{  �          @l��=�G��Tz�?O\)A[�C���=�G��P�׿��\���C�                                      Bxu4��  �          @g
=���`  >k�@p��C�{���G���\)��
=C���                                    Bxu4�Z  �          @o\)    �n{����
=C�H    �J�H�������C�H                                    Bxu4�   �          @`��>���\(����R��C�:�>���1��33��C��                                    Bxu4��  �          @w���{�g
=�aG��S33CyT{��{�>{�z��
=Cu^�                                    Bxu4�L  �          @z=q��  �a�>�z�@��HCv��  �J�H��=q��ffCt��                                    Bxu4��  
�          @y�������fff>���@�Q�Cyk������Q녿\����Cw�f                                    Bxu4�  S          @x��>��u>Ǯ@���C���>��^�R��z����C�\                                    Bxu4�>  �          @xQ�?+��p  >��@\C��?+��Z�H������C���                                    Bxu4��  �          @z�H?E��u�>���@�p�C��?E��Z�H��p���  C�S3                                    Bxu5�  �          @~�R?�
=�p��>�@ӅC���?�
=�\(��Ǯ���C�u�                                    Bxu50  �          @|��?���n{=�?�  C��R?���O\)��=q��C��                                    Bxu5*�  �          @�Q�?�Q��l��>W
=@?\)C���?�Q��QG���  ����C��f                                    Bxu59|  �          @�Q�?s33�vff>�
=@��HC��?s33�`  ����¸RC���                                    Bxu5H"  �          @}p�?E��tz�?+�AQ�C���?E��g
=����{C�                                    Bxu5V�  �          @z=q���
�xQ�=�G�?�33C�b����
�W���
=��{C�K�                                    Bxu5en  �          @z�H�#�
�qG�?+�A"{C��#�
�dz῱���{C���                                    Bxu5t  �          @��׾u�w�?J=qA8  C�@ �u�mp������
=C�,�                                    Bxu5��  �          @{����r�\>u@g�C�j=���W
=��\�܏\C��
                                    Bxu5�`  �          @�G��@  �w��#�
�
=C��H�@  �Q���
��G�C���                                    Bxu5�  �          @}p�>Ǯ�p��?�=qA�{C��3>Ǯ�qG�����t��C��                                    Bxu5��  �          @~�R?�(��5?�{A�z�C�` ?�(��HQ쾣�
��p�C��                                    Bxu5�R  �          @�33@(��P��?Q�A<��C��@(��Mp���G��h(�C�'�                                    Bxu5��  �          @u�?�Q��XQ�>��
@�z�C�AH?�Q��C33���R��G�C�w
                                    Bxu5ڞ  �          @n{@  �0  >u@{�C���@  �{��p����C�,�                                    Bxu5�D  �          @]p�?�33��?�G�A�p�C��q?�33�1G�=�\)?�
=C�5�                                    Bxu5��  �          @p  @	���&ff?�  A�ffC�ٚ@	���7
=��z���ffC�y�                                    Bxu6�  �          @~{@.�R�1G�?8Q�A'
=C�H�@.�R�/\)�W
=�D(�C�w
                                    Bxu66  �          @z�H@$z��4z�?\(�AJ=qC�7
@$z��6ff�@  �/�
C�\                                    Bxu6#�  �          @l(�@'����?z�HA�=qC���@'��Q쾞�R��=qC��3                                    Bxu62�  �          @|��@hQ쿵?��A�C�Y�@hQ�\�k��W�C��                                    Bxu6A(  �          @x��@dz´p�>�Q�@���C���@dzΌ��������C��\                                    Bxu6O�  T          @j�H@L(���\>�(�@���C�w
@L(���p����{C���                                    Bxu6^t  �          @p��=#�
�_\)?���A��
C�Z�=#�
�fff�O\)�JffC�XR                                    Bxu6m  �          @|(����W�?�Q�A�(�Cw!H���g��
=q����Cx�
                                    Bxu6{�  �          @~�R��Q��e?h��AX��C{���Q��b�\��{��z�C{h�                                    Bxu6�f  �          @|�;�\)�x��?��A ��C��=��\)�fff������(�C��                                     Bxu6�  �          @��ÿQ��q�?���Aw�
C��H�Q��q녿����x  C��H                                    Bxu6��  �          @�G��n{�y��>�G�@�
=C�H��n{�c33��
=��z�CJ=                                    Bxu6�X  �          @{�>\)�u>�?��HC��>\)�U��z����C�0�                                    Bxu6��  �          @}p�?L���s33?\)A  C��3?L���a녿\��  C�c�                                    Bxu6Ӥ  �          @s33?L���i��>�{@�
=C�1�?L���R�\�У���C��)                                    Bxu6�J  �          @��?�z��w
=>k�@U�C�U�?�z��Z=q��=q��C�`                                     Bxu6��  T          @}p�?�{�o\)>�(�@ə�C�E?�{�Z=q�������\C�f                                    Bxu6��  T          @~�R?��j�H<�>��HC���?��H�ÿ�33��G�C�"�                                    Bxu7<  �          @���?���r�\>aG�@P  C�q?���U�����C���                                    Bxu7�  �          @{�>��
�w
=?�\@�Q�C�]q>��
�c33�������C���                                    Bxu7+�  �          @tz�k��h��>�33@���C�Ϳk��Q녿�\)�˅C~L�                                    Bxu7:.  �          @o\)>�\)�]p�>u@~{C�N>�\)�Dz��\)�ݙ�C��R                                    Bxu7H�  �          @s�
@_\)���?   @���C���@_\)��녽���
=C��\                                    Bxu7Wz  �          @�=q@�  =���?
=A�R?�z�@�  �k�?\)@�z�C�]q                                    Bxu7f   �          @��@�녾�p�>�G�@�ffC�j=@�녿��>8Q�@#�
C�+�                                    Bxu7t�  �          @��@z�H�xQ�>�G�@�
=C�f@z�H����\)���HC�o\                                    Bxu7�l  �          @��@|�ͿQ�?E�A-p�C�%@|�Ϳ���>aG�@H��C�33                                    Bxu7�  �          @~�R@n�R�aG�?�ffAv=qC�W
@n�R���>�(�@�
=C�XR                                    Bxu7��  �          @�=q@p  ���?���A�G�C�޸@p  ���
>\@�C��                                    Bxu7�^  
�          @���@l(��\(�?�\)A��RC�k�@l(����H?5A!��C�>�                                    Bxu7�  �          @u@Tz῔z�?��RA�=qC�U�@Tz���
?(��A{C���                                    Bxu7̪  �          @r�\@1G���=q?�=qA�RC�H�@1G�� ��?
=A�HC��q                                    Bxu7�P  �          @w
=@<�Ϳ�  ?�
=A��C���@<����?�\@�(�C���                                    Bxu7��  �          @xQ�@N{��(�?�\A�\)C�� @N{��(�?\(�AL��C�Ff                                    Bxu7��  �          @~�R@1G��p�?���A�C�<)@1G��3�
����\C�G�                                    Bxu8B  �          @x��@B�\���H?�A�p�C���@B�\�=q>.{@#�
C��\                                    Bxu8�  �          @N�R@�R��
?�ffA�{C���@�R�33�L���dz�C�                                    Bxu8$�  �          @E>��
�5�?�  A��RC�7
>��
�;��&ff�EC��                                    Bxu834  �          @K�>�\)�(��?ٙ�B�C���>�\)�H��=L��?aG�C��f                                    Bxu8A�  �          @QG��c�
�333?�=qA���C|c׿c�
�E���33��  C}�f                                    Bxu8P�  �          @^�R>���Dz�?��HA�  C���>���XQ�\����C�}q                                    Bxu8_&  �          @e�@0  ��33?���A���C��f@0  ��
=�?�\)C��3                                    Bxu8m�  �          @e�@'
=�ff?�A�HC��R@'
=�33�B�\�EC�O\                                    Bxu8|r  �          @j�H?�G��E?5A4Q�C��\?�G��@  ���
��ffC�0�                                    Bxu8�  �          @�  ?�Q��S�
?��A�C�� ?�Q��s�
����
=qC���                                    Bxu8��  �          @�Q�?���`  ?�
=A�p�C�Ff?���n{�#�
��\C��3                                    Bxu8�d  �          @�Q�?���N�R?��HA�
=C��q?���q녽L�Ϳ+�C�h�                                    Bxu8�
  �          @�{@��Dz�?��HA��
C��3@��h��<�>�C���                                    Bxu8Ű  T          @�{@���Dz�?�\)A�(�C�y�@���e�#�
��C�k�                                    Bxu8�V  �          @|��?�Q��7�?�{A�\)C��?�Q��Z�H=u?Q�C�˅                                    Bxu8��  T          @n�R?�{�7�?�(�Aܣ�C�� ?�{�Vff�u�n{C���                                    Bxu8�  �          @]p��@  �˅�L�ͿQ�CP�@  ��=q�aG��qCK��                                    Bxu9 H  �          @i���HQ�˅��ff��33CNٚ�HQ쿎{��(����CGs3                                    Bxu9�  �          @c�
�#�
�ff<�?   C^� �#�
� �׿�(����CZ
                                    Bxu9�  �          @_\)��\�
=?\AиRC^����\�%>8Q�@<(�Cd�\                                    Bxu9,:  �          @aG��*�H�޸R?�z�A�
=CU#��*�H�{>�=q@���C[��                                    Bxu9:�  �          @Z�H�,(�����?k�A|(�CV��,(���
�L���UCYh�                                    Bxu9I�  T          @J=q��R��33?��HA���CQc���R��=q>�z�@��
CXaH                                    Bxu9X,  �          @G��   �޸R>u@�=qCV���   ��{�8Q��XQ�CT��                                    Bxu9f�  �          @H���\)��=q>��R@�ffCXW
�\)���H�5�R�RCV��                                    Bxu9ux  �          @Z�H�-p���=q?fffAt��CV{�-p��녾W
=�c�
CX�{                                    Bxu9�  �          @U�3�
���
?n{A��
CP���3�
������
���
CTz�                                    Bxu9��  �          @X���1G����\?���A�G�CL�H�1G���?   A	CU(�                                    Bxu9�j  T          @U�1G���Q�?p��A�G�COff�1G����H<�?\)CS��                                    Bxu9�  �          @R�\�   ��ff?333ALz�CWǮ�   ��33��Q����CY8R                                    Bxu9��  �          @XQ�?�p�����@$z�BF  C���?�p��9��?�(�A���C�s3                                    Bxu9�\  T          @\(�?O\)�'
=?�B�
C��)?O\)�K�>8Q�@G�C�q                                    Bxu9�  
�          @L��=�\)���?�(�B=qC�Ǯ=�\)�HQ�>\@�33C��)                                    Bxu9�  �          @c33��Q���R@B-z�C�  ��Q��U?+�A6�RC�AH                                    Bxu9�N  �          @x�ÿ���L(�?�ffA�Cp������P�׿L���B=qCq33                                    Bxu:�  T          @p  ��p��G����Ϳ�=qCm�)��p��&ff�޸R��ffCh�f                                    Bxu:�  �          @fff�	���9��>���@�=qCik��	���'����\��
=Cf�)                                    Bxu:%@  �          @\(��	���#33�#�
�W
=Ce�H�	���
=q��{����Ca�                                    Bxu:3�  �          @g����H�>�R?�AG�Cl�����H�5�������
=CkJ=                                    Bxu:B�  �          @[����2�\?8Q�AC�CmͿ��0  �^�R�k�
Cl�f                                    Bxu:Q2  �          @mp��9�����?!G�AffCY!H�9����Ϳ�R��CY+�                                    Bxu:_�  �          @j=q�=p��   ?#�
A#�CV��=p���\��\�p�CV��                                    Bxu:n~  �          @`���(Q��  =���?�p�C\�=�(Q���H������z�CX�                                     Bxu:}$  �          @N�R�33�(��#�
���C_�
�33���Ϳ���ffCZ�)                                   Bxu:��  �          @N�R�"�\��(�?�33A��CM�3�"�\���
?
=qA�CV��                                   Bxu:�p  �          @aG��7���{?��A�(�CI.�7���G�?5A?�
CSu�                                    Bxu:�  T          @`���$zῑ�@�\B  CK��$z��33?���A�CZ�\                                    Bxu:��  �          @XQ��33�h��@�B*CI���33��p�?��HA�
=C\�                                    Bxu:�b  �          @C�
�˅�Y��@(�BU�
CP#׿˅� ��?�\)B �HCg��                                    Bxu:�  �          @]p��
=��
?��A�{Cc���
=�*�H���
����Cg��                                    Bxu:�  �          @N�R�\(��#33?�
=A�RC{k��\(��:�H�����C}��                                    Bxu:�T  �          @XQ쿀  �z�?��B�Cv����  �>{>�p�@ٙ�C{L�                                    Bxu; �  �          @Y�����A�?�Q�A�z�C�)���U��\��
=C���                                    Bxu;�  �          @S�
���C33?�A�p�C�3���Mp���R�-G�C�O\                                    Bxu;F  �          @N�R�G��7
=?�Q�A�(�C~�q�G��C�
��\�ffC��                                    Bxu;,�  �          @L�ͿG��3�
?�ffA��
C~k��G��Dz�\��p�C��                                    Bxu;;�  �          @Vff?z��!G�?�G�A�z�C�s3?z��*�H���=qC�R                                    Bxu;J8  �          @j�H@2�\�>���@�  C��@2�\��ÿz�H�x��C�9�                                    Bxu;X�  �          @aG������L��?��RA���C�#׽����W��#�
�,z�C�/\                                    Bxu;g�  T          @O\)@\)��?L��Adz�C�%@\)��
��{���RC�7
                                    Bxu;v*  �          @W�@*=q�\?���A�p�C�q@*=q�G�>�Q�@�
=C�g�                                    Bxu;��  �          @Mp�@4zῑ�?fffA�\)C��q@4zῸQ�>#�
@7
=C�u�                                    Bxu;�v  �          @L��@G��
=>8Q�@Q�C��q@G�����#�
�7�C��                                    Bxu;�  �          @l��@Z=q����>aG�@\(�C�.@Z=q��G����  C��3                                    Bxu;��  �          @QG�@*=q����>B�\@VffC��@*=q��33�L���e�C�                                      Bxu;�h  �          @s33@��5�>��@ʏ\C��H@��'���33���C���                                    Bxu;�  �          @|(�@`  ��G�>L��@<(�C��f@`  ���ͿB�\�2ffC��{                                    Bxu;ܴ  �          @�Q�@U���R��\)�p��C�R@U���{��p���\)C�h�                                    Bxu;�Z  �          @�Q�@2�\�%������C��@2�\��Q��G���G�C��)                                    Bxu;�   �          @}p�@J�H�\)������C�Y�@J�H�˅�ٙ���ffC��\                                    Bxu<�  �          @�z�@Z�H��>L��@,��C��)@Z�H������n=qC�S3                                    Bxu<L  �          @��@[���>8Q�@�C���@[��zῊ=q�p��C�xR                                    Bxu<%�  T          @���@g�� ��=��
?�33C�u�@g���  �}p��]p�C��                                    Bxu<4�  �          @���@l(���\)=#�
?��C���@l(���{�s33�T��C�9�                                    Bxu<C>  �          @�
=@j�H��<��
>aG�C�=q@j�H���
��=q�m�C��                                    Bxu<Q�  �          @���@^�R�\)������C�� @^�R���ÿ�����C�0�                                    Bxu<`�  �          @���@Z�H���L�Ϳ333C��@Z�H�������\����C�&f                                    Bxu<o0  �          @���@b�\�Q�>�Q�@��C�y�@b�\�   �Q��5��C�B�                                    Bxu<}�  �          @�z�@n{��p�?��@��C���@n{��G���G����
C�XR                                    Bxu<�|  �          @��
@j�H�ٙ�?G�A.�\C��)@j�H���;k��Mp�C���                                    Bxu<�"  �          @�=q@j=q�˅?J=qA2�RC�@ @j=q��\�.{��C��                                    Bxu<��  �          @��@h�ÿ�z�?&ffA(�C�@h�ÿ�  ���
��Q�C�%                                    Bxu<�n  �          @��\@a녿��H?�A ��C�� @a녿��H�����{C�t{                                    Bxu<�  	�          @��
@a��z�?�\@�33C���@a��녿(����RC�
=                                    Bxu<պ  
�          @��@c33�p�?#�
A
=C�\@c33�p���R�ffC�f                                    Bxu<�`  "          @�{@]p���?(��A�HC���@]p���Ϳz��G�C�˅                                    Bxu<�  
�          @��@k���
=�(����
C��
@k������Q���
=C��=                                    Bxu=�  
(          @��
@u���(��������C�}q@u���=q����m��C�#�                                    Bxu=R  T          @��
@s�
�\���
���C�R@s�
���׿����qG�C��)                                    Bxu=�  �          @��@w
=��녾L���1G�C�3@w
=��=q�fff�J�RC�,�                                    Bxu=-�  T          @�z�@�  ���
�u�W�C�@�  �@  �@  �&�HC��\                                    Bxu=<D  T          @��@�Q�Q�=�G�?�p�C�1�@�Q�B�\��{����C���                                    Bxu=J�  �          @��@s�
��G�>.{@�
C�/\@s�
��\)�&ff�ffC�R                                   Bxu=Y�  �          @��R@xQ��{�#�
���C��{@xQ쿯\)�\(��<(�C�J=                                   Bxu=h6  "          @���@x�ÿ�33>L��@0  C�"�@x�ÿ���\)��=qC�Ф                                    Bxu=v�  T          @��R@~{��G���33����C�/\@~{�c�
�u�S�
C��=                                    Bxu=��  L          @�\)@��H��ff�L�Ϳ!G�C�˅@��H�^�R�
=� ��C��q                                    Bxu=�(  
�          @��R@�33�u>\)?��HC�j=@�33�c�
�\��C��)                                    Bxu=��  ~          @�ff@�=q�z�H>k�@FffC�5�@�=q�u���
���C�c�                                    Bxu=�t  �          @�ff@�=q�z�H>k�@FffC�B�@�=q�s33���
��33C�n                                    Bxu=�  	�          @��
@�녿   �����=qC�}q@�녾8Q�(���Q�C��q                                    