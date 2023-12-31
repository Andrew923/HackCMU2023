CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230228000000_e20230228235959_p20230301021700_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-01T02:17:00.615Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-28T00:00:00.000Z   time_coverage_end         2023-02-28T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxiU   
(          AQ�A
=@�������1G�A�z�A
=@�p�?5@���A�                                    BxiU(�  
�          A\)A	�@4z���'33A��HA	�@Q�<�>L��A��\                                    BxiU7L  "          A�A��?�p��<(����HA7
=A��@@  ��\)�!A��R                                    BxiUE�  �          A��A�@  �8Q����Ar�\A�@Z�H�����HA��R                                    BxiUT�  
�          AffA33@7��5��
=A�z�A33@{��s33��{A��H                                    BxiUc>  "          Az�Az�@�׿�(���{Ag
=Az�@$z�=�?G�A�z�                                    BxiUq�  
Z          A��A�@W�����p�A�p�A�@k�>�
=@(Q�A�                                    BxiU��  T          A�
@޸R@��p��A��
@޸R@��H�c33��z�B�                                    BxiU�0  �          Az�@�
=@�����'33A�@�
=@�G�������B(�                                    BxiU��  �          A(�@�33@<(������HA��
@�33@�\)�X������B�
                                    BxiU�|  �          A��@�Q�@(Q����
�$ffA�  @�Q�@�p��xQ���
=B{                                    BxiU�"  �          A��@�G�?�
=�����7��AL��@�G�@�����R�ffB33                                    BxiU��  
�          A@�  @?\)����� �A�\)@�  @��g���{B$ff                                    BxiU�n  
�          A�@�p�@�
���H��A�p�@�p�@���s�
��Q�B33                                    BxiU�  �          A�H@�33@W
=��G���
A��@�33@���@  ���RB��                                    BxiU��  �          Aff@׮@u��G��	=qA홚@׮@����#33�|  B'33                                    BxiV`  �          A(�@�(�@�\)��z���p�A��@�(�@�z��=q�33BG�                                    BxiV  
�          A@�\@��\�I����33B
=@�\@�=L��>��
B ��                                    BxiV!�  
Z          Az�@陚@����L(����B�H@陚@�녽�\)��ffB��                                    BxiV0R  �          Az�@���@`����(���  A�  @���@�z��(��'\)B�                                    BxiV>�  T          A(�@��@(Q����
����A�@��@�(��&ff�~�HA�                                      BxiVM�  �          AQ�@���@+���\)��\)A�z�@���@���*�H���HA���                                    BxiV\D  �          A
=@�33?�ff��{��HA
=@�33@r�\�q���(�A��                                    BxiVj�  
�          AQ�@�
=�����
=���C���@�
=?Y����p��)G�@�p�                                    BxiVy�  "          AG�@�=q?��
�Ϯ�0�A0��@�=q@�{�����  BG�                                    BxiV�6  �          A�@�p�?����Q��&�A+
=@�p�@��H��{���B ff                                    BxiV��  �          AQ�@�33?����ʏ\�)�A=q@�33@�ff�����p�A�(�                                    BxiV��            A�@�?fff��p���@���@�@l(�������  A�                                      BxiV�(  
�          Ap�A�@���N�R��=qAb=qA�@fff��{��A�
=                                    BxiV��  �          A  A33?��
�����@�
=A33@P  �Z�H��A�z�                                    BxiV�t  �          Aff@�R?
=q�������@��@�R@c�
������p�A�{                                    BxiV�  "          A\)@�ff?����33�	
=A5G�@�ff@�=q�s33���\A��H                                    BxiV��  "          A(�@��?k����
��\@�{@��@k���33���A�z�                                    BxiV�f  �          A��A�H@8Q����R��G�A�p�A�H@�z�����L��A�z�                                    BxiW  �          A�H@�@AG���z���A��@�@���E���z�B(�                                    BxiW�  T          A�\@���?�p������Aw�
@���@�������ҸRB�
                                    BxiW)X  �          Aff@�{@����\)��A��@�{@�  �~�R��\)BG�                                    BxiW7�  �          A�\@���@p����(���A��@���@�\)�,(���  B!�\                                    BxiWF�  T          A�@�(�>��R��  �-
=@+�@�(�@]p����R�\)Aܣ�                                    BxiWUJ  T          A33@�p�������H�*�C�^�@�p�@AG����
��A��R                                    BxiWc�  "          A@�p�@B�\�������A��@�p�@��\�[���33Bz�                                    BxiWr�  "          A�H@�@G���p���RA��
@�@�ff�_\)��=qB�H                                    BxiW�<  
�          A�
@�=q?����\��A&�R@�=q@�Q���G���\)A�                                    BxiW��  "          A�@�
=@\����Q��
=qȀ\@�
=@����>{����B�
                                    BxiW��  T          A�@�(�@$z��\�\)A�  @�(�@���z=q��B�R                                    BxiW�.            A��@���?G���{�W��A z�@���@��H�ƸR�#{B�                                    BxiW��  �          A�R@���ff�����]ffC��f@�@\���ָR�<p�B�                                    BxiW�z  J          A\)@�p�������H�Cp�C�u�@�p�@H����(��*��A��                                    BxiW�   r          AQ�@�33>�����G��mff@�  @�33@�=q��  �7p�B,�                                    BxiW��  T          A�@j=q@p��\)�yB��@j=q@�p�������Bq=q                                    BxiW�l  �          A�@�(�@)��� z��e�HA�
=@�(�@��
����  B]��                                    BxiX  �          Ap�@�z�@�H��{�V=qA�G�@�z�@�������	\)BEz�                                    BxiX�  T          A��@�G�@����
�]p�A�p�@�G�@�  ������BB��                                    BxiX"^  "          A��@��?������R�HA'
=@��@������{B��                                    BxiX1  �          A��@���?����\)�X�@�@���@�����(��'��B��                                    BxiX?�  �          AG�@�{>�(������Z
=@�\)@�{@�  ���
�*�B
=                                    BxiXNP  �          A��@��\@0  ���RG�A�G�@��\@�ff����{BJ��                                    BxiX\�  T          AQ�@�Q�@(����33�U=qAޏ\@�Q�@�(����R�ffBK
=                                    BxiXk�  T          A�@��H@  ��G��^G�A�@��H@�z���=q���BJ\)                                    BxiXzB  �          A�
@�(�?�p����H�K��A��H@�(�@�p����\�	�B/                                    BxiX��  
�          A�@���?���=q�T��A�33@���@��������B1(�                                    BxiX��  T          A(�@�{@���=q�S�Aȏ\@�{@�p��������BC�                                    BxiX�4  
�          A(�@��@Mp������[�
B=q@��@�ff���H�z�Bb��                                    BxiX��  
(          A33@���@0  ��Q��8
=A��
@���@�p���\)�޸RB4Q�                                    BxiXÀ  
�          A
=@���?����=q�(�\A��@���@��H����ffA�
=                                    BxiX�&  
�          A�H@љ�?�\�У��/z�Aqp�@љ�@�=q������
BQ�                                    BxiX��  �          Aff@�Q�?fff��ff�z�@�33@�Q�@QG��tz���G�A���                                    BxiX�r  
�          A��@У�@4z��˅�'G�A��H@У�@�z���=q��(�B#\)                                    BxiX�  �          Az�@���@Z=q���I�
B�@���@����z����BS                                    BxiY�  
�          Aff@���@,(����
�4p�A��R@���@������޸RB,�                                    BxiYd  
Z          A��@��@#�
� z��l�B�@��@�Q�������\Bd�                                    BxiY*
  �          AG�@��\?�ff����q�A�{@��\@�p���
=�'�\BM�\                                    BxiY8�  
�          A��@�  ?aG����wz�A:�R@�  @��H�ҏ\�6ffBB�
                                    BxiYGV  T          AG�@���>�����[��@��@���@�ff��G��+�B��                                    BxiYU�  "          A=q@��=�Q���G��:��?E�@��@^�R���\��A�=q                                    BxiYd�  J          A
=A33@ �������ә�A��A33@�����k�A�                                    BxiYsH  T          A��@�33@�  �%��t��B{@�33@��\>���@�B�\                                    BxiY��  �          Ap�@�z�@��H�s33����B p�@�z�@׮���:�HB5                                    BxiY��  
Z          A33@ٙ�@'
=��Q��"A�=q@ٙ�@����������B��                                    BxiY�:  "          A��@��?�(���33�0�Ajff@��@�G������
=Bz�                                    BxiY��  �          A  @���?����
=�7Q�A7
=@���@�\)��p��  B{                                    BxiY��  
F          Az�@X��?��
����=A��@X��@�ff����.Q�Bm
=                                    BxiY�,  �          A�R@XQ�@#33�p��|ffB�@XQ�@��
��ff�(�BxQ�                                    BxiY��  �          A@���@Q��Ӆ�5ffA���@���@�z����H���B=z�                                    BxiY�x  �          A\)@��ÿ!G�����J33C��=@���@>�R���2��A�Q�                                    BxiY�  |          A=q@��
@�������p�A��@��
@�p��u�����B�                                    BxiZ�  
�          A��@�Q�@}p������(�A�ff@�Q�@�p����4��B�                                    BxiZj  �          A
=@�p��G���{�*G�C��@�p�?�{���2{A1                                    BxiZ#  �          A�H@�p��\(���z��:�C�XR@�p�@3�
���
�)�
A���                                    BxiZ1�  �          A�H@�\��\)����'ffC��R@�\?�����
�)\)APQ�                                    BxiZ@\  �          A�\@���
=q�����C���@��?�����
=�G�AG�                                    BxiZO  �          A{@�������\)��RC�9�@���@#33��{�  A���                                    BxiZ]�  �          A\)@�(�������8��C�� @�(�@=p���G��#��A�Q�                                    BxiZlN  �          A=q@�녿aG���{�_ffC�G�@��@J�H��\�GffA�z�                                    BxiZz�  �          AQ�@�z�u���H�C�HC���@�z�@e�����$A�Q�                                    BxiZ��  "          AQ�@�=q���H�����L=qC���@�=q@S33��G��2
=A�=q                                    BxiZ�@  "          A��@�G���
=��p��Y�C��)@�G�@8����
=�HG�A�Q�                                    BxiZ��  |          A�
Az�@��������(�A{33Az�@�ff�!��n�RAѮ                                    BxiZ��  T          A�A�?��H�vff����A,��A�@X��� ���j=qA���                                    BxiZ�2  �          A33A@b�\����%��A�A@}p�>B�\?�{A��
                                    BxiZ��  
�          AffA
�R@���=p�����A�  A
�R@�p�?�z�A�\A�                                      BxiZ�~  T          A�\A�
@�
=>�G�@#�
BQ�A�
@�G�@E�A�
=A���                                    BxiZ�$  
Z          A�@�
=@�Q�>�33@z�B��@�
=@��@QG�A��B�                                    BxiZ��  "          A(�A@��=�?5B��A@��@/\)A��A���                                    Bxi[p  �          A�
A	@�  =u>���A���A	@x��@��AW\)A¸R                                    Bxi[  �          A�A
�\@��ͽ�G��#�
A�G�A
�\@xQ�@z�ADz�A�33                                    Bxi[*�  !          A  A	G�@�33=L��>�=qA�A	G�@~�R@33AZffA�G�                                    Bxi[9b  
(          A�H@�=q@��
��Q��=p�B\)@�=q@�(�?p��@�p�B�\                                    Bxi[H  "          Aff@���@���u���p�B (�@���@���&ff�\)B6G�                                    Bxi[V�  
Z          A�@��@�
=���
�,Q�B�R@��@��?�  @���Bff                                    Bxi[eT  "          A  @�=q@����K���ffB
��@�=q@��\�\�Q�B��                                    Bxi[s�  �          A�R@�Q�@�z���G��Q�B
=@�Q�@�G��HQ���
=B8                                    Bxi[��  T          A�H@ۅ@���z��
�Bff@ۅ@��
�0�����B/�                                    Bxi[�F  �          Aff@��@�����(���(�B\)@��@���p��O
=B+�\                                    Bxi[��  
�          A�\@�{@o\)��{�ffA�  @�{@�p��:�H��\)B�
                                    Bxi[��  "          A(�@�=q>�z���\�M  @,(�@�=q@|(���\)�'G�B�H                                    Bxi[�8  
          A33@�ff�����
=�^  C�  @�ff@Y����\)�A�A��                                    Bxi[��  "          A33@�(�?�=q��=q�D��Ag�@�(�@�{���
�33B�\                                    Bxi[ڄ  |          A�@�=q@)����ff�!�A�=q@�=q@��\��(���  B(�                                    Bxi[�*  �          AQ�@�@������ �B"\)@�@���I�����\BP                                    Bxi[��  
�          A\)@�ff@����  �{BA@�ff@�p����O�
Bb�
                                    Bxi\v  �          A\)@�\)@�33�����(  BHQ�@�\)@�\)�9����{Bo��                                    Bxi\  �          A�@���@�  ��{�=�B3p�@���@陚�x�����HBh��                                    Bxi\#�  
�          A  @���@��
��(�����A�  @���@��R���d��B'z�                                    Bxi\2h  �          AG�@���@�����H��B:��@���@�
=��33�	G�BU�H                                    Bxi\A  "          A��@�Q�@����{���B�@�Q�@�{�Fff��  B@=q                                    Bxi\O�            A�@��
@e����JffB=q@��
@�{��
=���BW�R                                    Bxi\^Z  
�          A��@��@j�H��33�U\)B!�@��@�(���z�����Bg��                                    Bxi\m   T          A��@xQ�@
=�Q��t�
A��@xQ�@����Å�!z�Be�                                    Bxi\{�  T          A{@���@=q�z��m\)A�@���@�33��33�33B[��                                    Bxi\�L  T          A=q@��@K�� ���d��B  @��@����G��(�Biff                                    Bxi\��  �          A{@��\?�{�  �y{Ae�@��\@������8��BE��                                    Bxi\��  
�          A@^�R?��
���qAظR@^�R@�  ����5ffBk=q                                    Bxi\�>  
�          Az�@���@ff�����\�A�\)@���@�G���{��BG{                                    Bxi\��  �          A��@��@;���(��`p�B�@��@�33��Q��
=B\�                                    Bxi\ӊ  �          Aff@X��@p����p��[�B?��@X��@�\)��  ����B}
=                                    Bxi\�0  T          AQ�@��
?����Ϯ�-
=A�33@��
@��������  B�                                    Bxi\��  J          A ��AQ�>�\)��\)�  ?�Q�AQ�@7
=����
=A��R                                    Bxi\�|            A�\@�ff@E���p��6ffA�33@�ff@����}p���  B=�                                    Bxi]"  
�          A��@+�@����θR�9��B�B�@+�@�{�C33���B��                                    Bxi]�  "          A{@���@	����z��e\)A�z�@���@�����ff�p�BM��                                    Bxi]+n  
�          A�\@���@������j=qAυ@���@�����33��RBQ�                                    Bxi]:  �          A��@�p�@)�����R�`�A�p�@�p�@�  �����G�BV\)                                    Bxi]H�  	�          A��@�
=@L(���\�P�RB�@�
=@�����{� ��BTp�                                    Bxi]W`  �          A�@�=q@o\)�ƸR�.�RBp�@�=q@Ǯ�l����{BF(�                                    Bxi]f  "          A��@�  @Tz���{�A�Q�@�  @��
�\(���z�B#Q�                                    Bxi]t�  
�          A�@��@G���33�G{A��H@��@�z������  B(�                                    Bxi]�R  
�          A(�@�@
=���
�(z�A�p�@�@�  ��z���=qB�                                    Bxi]��            Ap�@�(�?��������  Al��@�(�@����g����A���                                    Bxi]��  �          A�\@�=q�
=q�����!C�Ф@�=q@Q����H�G�A��\                                    Bxi]�D  T          A{@���\�ڏ\�;{C�Q�@��@8����ff�%�A�                                      Bxi]��  �          A�\@��ͿW
=����:33C�E@���@{����,(�A���                                    Bxi]̐  �          A=q@�
=>��H����6��@�=q@�
=@c�
������\A�R                                    Bxi]�6  �          A�H@ָR?+���
=�/G�@��R@ָR@hQ�������A�p�                                    Bxi]��  �          A�@�Q���\��p��	C�(�@�Q�?!G��������@�{                                    Bxi]��  �          A�@�
=�5�����C�Ff@�
=?�Q����p�AiG�                                    Bxi^(  �          A\)@�Q�@AG���
=���RA��@�Q�@���>�R���B
=                                    Bxi^�  �          A�
@׮@Dz���p����A�  @׮@��
�dz���  B{                                    Bxi^$t  T          A��@�33?�(����
���Ao�@�33@��
�����ʸRA���                                    Bxi^3  T          Az�@��
@�����G���B�@��
@�ff�(��uG�BC{                                    Bxi^A�  �          A�
@��
@Q�������Q�A��\@��
@��R�8����p�A�G�                                    Bxi^Pf  @          A�@���@�\)�����{B2p�@���@�
=� ���O33BR��                                    Bxi^_            A{@޸R@����z����A�  @޸R@����5���33A�G�                                    Bxi^m�  �          A�@�33@^�R��z��=qA�p�@�33@�ff�XQ���(�B(�                                    Bxi^|X  
�          A��@�p�@g������p�A�\)@�p�@�Q��j�H��G�B,z�                                    Bxi^��  �          Az�@��@�(�����5�\Bp�@��@�ff�y����=qBU\)                                    Bxi^��  �          A33@�{@�������((�B=q@�{@Ϯ�c33��BH{                                    Bxi^�J  T          A�
@��@���(��0ffBOz�@��@�G��P������Bv�R                                    Bxi^��  �          A(�@��
@I�������GB\)@��
@�\)�����G�BK=q                                    Bxi^Ŗ  
�          A��@�
=@�  �ٙ��Bp�B'\)@�
=@�{��{��{Ba
=                                    Bxi^�<  �          A�@j�H@�����33�+G�Bf\)@j�HA ���=p���{B�                                    Bxi^��  T          A��@�ff�'
=��R�PC�R@�ff?�  ��33�a��A7�                                    Bxi^�  |          A��@�Q쿎{��
=�Q�C�XR@�Q�?�33���H���AZ�R                                    Bxi_ .  �          A�AG����R��������C��AG�?���Q���(�@n�R                                    Bxi_�  �          A\)A
�\?�33�_\)��(�@��A
�\@'��$z��}�A���                                    Bxi_z  T          A  @�  @�  >�z�?�G�Bz�@�  @���@*=qA�=qA�R                                    Bxi_,   �          A   AQ�@�z������
A�
=AQ�@�Q�@   A9G�A�p�                                    Bxi_:�  �          A (�A{@�33�\�(�A���A{@�=q?!G�@hQ�Aٮ                                    Bxi_Il  T          A��A�H@��������ffA�33A�H@��
?��
@���A�                                      Bxi_X  "          AQ�A��@��Ϳ˅�ffA�  A��@���?
=@^�RA�{                                    Bxi_f�  �          A��A
=@��R�����/\)Aޣ�A
=@��>Ǯ@G�A�\)                                    Bxi_u^  �          A{A�
@w
=���d(�A��HA�
@��������A��                                    Bxi_�  �          A��A�H@J=q��H�eA�  A�H@z�H�8Q���\)A���                                    Bxi_��  �          AG�A�@Z=q�"�\�o�A�{A�@�{�8Q���{A�Q�                                    Bxi_�P  T          AQ�A(�@E�����8��A�p�A(�@h�þ��ÿ��HA��
                                    Bxi_��  �          A\)A�H@��H����ffA�33A�H@���?���AQ�A�33                                    Bxi_��  "          A�HA��@q녿333��=qA���A��@l(�?��@�=qA���                                    Bxi_�B  �          AQ�A��@�=q�H����  A�z�A��@����xQ���p�A�{                                    Bxi_��  T          A�@�p�@��\�fff��(�A�G�@�p�@����G���
=B
�                                    Bxi_�  T          A�H@��\@��
�x�����
A�=q@��\@�{��{���B
=                                    Bxi_�4  �          A{@�z�@�33�����\)A��@�z�@�33��
=�9G�B\)                                    Bxi`�  "          A��@�@�=q������
B�H@�@�33���H�<��B"z�                                    Bxi`�  T          A��@�Q�@|(����
��{A�@�Q�@�ff�%��y�B��                                    Bxi`%&  
H          A(�A�\@l���;�����A�  A�\@�(��u��
=A�ff                                    Bxi`3�  �          A��A��@_\)�#33�p  A�{A��@�Q�8Q���p�Aͮ                                    Bxi`Br  �          AQ�Az�@h�����H��A�=qAz�@�ff������Ạ�                                    Bxi`Q  �          AQ�A�
@'
=�(���{�
A��A�
@`�׿�Q��ᙚA��R                                    Bxi`_�  T          A33A33@7
=�(��h��A��
A33@i���fff��33A�p�                                    Bxi`nd  
�          AffA�H@.�R���d  A��
A�H@`  �h�����A��                                    Bxi`}
  
�          Ap�@�ff����(���C���@�ff@�\���
�Q�A���                                    Bxi`��  �          Az�@�����R��
=�6  C�&f@��?�33���
�2\)Aw33                                    Bxi`�V  
�          A(�@�녿�z����+{C��f@��?˅���\�'�Abff                                    Bxi`��  
�          Aff@���?(������  @�@���@8����\)��G�A��R                                    Bxi`��  "          A��A\)?�{�~�R��=qA,(�A\)@L���6ff��  A�                                    Bxi`�H  �          A�A
�\?�Q���Q���{A1G�A
�\@Y���Dz���{A���                                    Bxi`��  �          AA��@5�^�R��p�A��A��@��\��\)�/33A�\)                                    Bxi`�  
�          A=q@��
@%�z�H��Q�A��@��
@������o�
A�{                                    Bxi`�:  @          A�
@��Ap�@A^�HBl
=@��@�\)@��B�BP=q                                    Bxia �  T          A�H@�(�@�p�=#�
>uBiz�@�(�@��H@aG�A�B]��                                    Bxia�  �          AQ�@��@�  ?�33A�
BU�H@��@�33@��A�z�B?�                                    Bxia,  T          A33@���@�R@	��AM�BQ  @���@���@�ffB{B3�R                                    Bxia,�  �          A�@���A�?z�@W�Bb@���@�(�@��
A��
BS�                                    Bxia;x  �          Ap�@�z�Aff>�
=@�RBb�@�z�@�z�@|(�A�Q�BS�H                                    BxiaJ  T          AQ�@��H@�=q�E��(�B=z�@��H@�녾��Q�BJ�                                    BxiaX�  �          Ap�@��R@��ÿz��j�HBB�@��R@�33@(�Ay�B;G�                                    Bxiagj  
�          A�@�  @�(����H�N�RB[�
@�  @�(�@*=qA�=qBS��                                    Bxiav  
�          A
=q@�(�@��Tz�����Bd@�(�@ڏ\@Q�A���B_p�                                    Bxia��  �          A�@��@�����R�!G�BG=q@��@�p�?�Q�A�BGz�                                    Bxia�\  �          A@��H@�{�
=���BXp�@��H@�G�@�A��HBQ�\                                    Bxia�  �          A�@���@�33��
=�#
=B2��@���@��?��Ap�B4Q�                                    Bxia��  �          @��@���@�G������\B.ff@���@�33?�=q@�
=B/�                                    Bxia�N  T          A�
@�Q�@��\�>�R���B p�@�Q�@�
=�n{��
=B�                                    Bxia��  �          A�@�\)@�\)����ffB�@�\)@��H���
��RB(�                                    Bxiaܚ  
�          @�
=@��R@'����\�
p�A�p�@��R@�\)�1���
=BG�                                    Bxia�@  T          A�@ٙ�@,����=q��z�A���@ٙ�@�� �����A���                                    Bxia��  
�          A\)@��@5��{��\A�\)@��@��
�#�
��33A�                                    Bxib�  T          A��@�@`��������\)AָR@�@�������b{B��                                    Bxib2  �          AQ�@��@H���`  ���A��@��@��H���>�HA���                                    Bxib%�  �          @�\)@�Q�@��R��
��  B�@�Q�@�G���z��B��                                    Bxib4~  "          A{@�G�@�
=��(��<��A�\)@�G�@���>�  ?ٙ�B\)                                    BxibC$  �          A�R@�@Z�H�����2=qA˙�@�@p��=#�
>�\)A�
=                                    BxibQ�  
�          @��@�=q?޸R��z��k
=A\��@�=q@Q�}p���G�A��                                    Bxib`p  
�          A{Aff@ ����R�dQ�A�G�Aff@N{�s33����A��                                    Bxibo  
�          A�@��@+��%�����A�
=@��@aG��������A�\)                                    Bxib}�  
�          A�\@��H?�
=��z��Q�AN{@��H@z=q��z����HA�ff                                    Bxib�b  �          A@�\?����H��AT��@�\@\)���\��A�G�                                    Bxib�  
�          A@�ff?�
=�Ϯ�0�
Ai@�ff@�����p��  B�H                                    Bxib��  �          A�R@��?�
=��{�5{Ajff@��@��H����
�RBff                                    Bxib�T  
�          Az�@���?��H��
=�m�An�\@���@��R��
=�:{B1��                                    Bxib��  �          A@��
?��R����\�A�  @��
@��\��=q�*�\B'33                                    Bxibՠ  
�          A=q@��@���Q��0\)A��
@��@����R���\B�
                                    Bxib�F  �          A��@љ�@��\��=q���
B@љ�@��H��n�\B'                                      Bxib��  T          AG�@�=q@����������A�ff@�=q@����'
=��Q�B"                                    Bxic�  
�          A�\@�p�@vff�c�
���HA��@�p�@����\)�(Q�B=q                                    Bxic8  "          A	G�@ָR@��R�\)��G�B\)@ָR@�=q�������RB�R                                    Bxic�  �          A��@�
=@�ff�.�R����A�p�@�
=@�ff�:�H��  B	��                                    Bxic-�  
�          A(�A33@��
�:�H���A�z�A33@�{�Tz����BQ�                                    Bxic<*  
�          A�A�@�(��ff�`Q�A��A�@�\)��
=� ��A陚                                    BxicJ�  T          A�A	@�p������*�RA�  A	@���>��?��RA��                                    BxicYv  T          A"{A��@�������p�A�{A��@��H?fff@��B �                                    Bxich  
�          A$  Ap�@����33���\A�\)Ap�@��
?J=q@�A�z�                                    Bxicv�  
�          A#�
A(�@�33��(��z�A�A(�@�  ?B�\@���A��                                    Bxic�h  "          A$��A��@��׿�{��B=qA��@�{?O\)@�Q�Bz�                                    Bxic�  �          A#�
A��@��H��
=��A�A��@��?(��@n{B
=                                    Bxic��  �          A#�A�@����=q�%G�B(�A�@�?��@EB\)                                    Bxic�Z  �          A"�\Az�@�=q��{�)�B��Az�@�33?
=@U�B�                                    Bxic�   �          A!�Az�@��ÿ����(z�B  Az�@���?z�@Q�B�                                    BxicΦ            A!A@�����\� ��B33A@�Q�?B�\@���B(�                                    Bxic�L  
k          A#33A��@��R��33�,z�AǅA��@��
���
��Q�A�=q                                    Bxic��  �          A"�HAQ�@xQ���
�
�RA���AQ�@�p�>8Q�?��
A��                                    Bxic��  
�          A!�Ap�@��
������\)A���Ap�@���?#�
@j�HA��H                                    Bxid	>  
�          A#33A�@�(�������RA�(�A�@�=q?\)@G�A�                                    Bxid�  
�          A$��Ap�@��H���R��A�=qAp�@���?   @2�\Aޏ\                                    Bxid&�  
�          A&�\A
=@�p�������{AׅA
=@�=q?(��@i��A�33                                    Bxid50  "          A"ffA	@���0  �{�A�\)A	@��\�&ff�j�HA�                                      BxidC�  "          A$(�A
=@����\�N=qAՅA
=@�G���\)�ǮA��                                    BxidR|  
�          A%p�A�\@��׿����ffA�  A�\@���>�Q�?�p�A�ff                                    Bxida"  �          A$��A@|�Ϳ�\)��A��A@���=�G�?
=A�(�                                    Bxido�  T          A&�RA�@e�У��(�A��A�@|(��L�;�=qA�                                      Bxid~n  T          A'\)A�
@Vff��ff�
=A��
A�
@r�\���R��Q�A�=q                                    Bxid�  T          A'
=A�@Z�H��G��p�A��RA�@o\)���
����A�                                      Bxid��  T          A#�
Aff@AG���{�z�A��HAff@Z=q��=q���RA���                                    Bxid�`  T          A#�
A�\@7
=�����#�A�  A�\@Vff���;�A���                                    Bxid�  T          A$��A�
@333��  ���A�z�A�
@QG����H�.�RA�z�                                    BxidǬ  �          A#�
A\)@1녿ٙ���A\)A\)@N{���#�
A�
=                                    Bxid�R  �          A'\)A{@,(���
�6=qAs�A{@R�\�Q�����A�G�                                    Bxid��  
�          A*�\A#�
@%���R� Q�Ac\)A#�
@>{��{����A�p�                                    Bxid�  
�          A+\)A$Q�@*=q�������Ah(�A$Q�@AG���\)�\A���                                    BxieD  T          A)G�A!��@=p��aG���Q�A��\A!��@C�
>�Q�?�(�A���                                    Bxie�  
�          A"�HA@�Q�?�
=A�Bz�A@�(�@s�
A��A�                                    Bxie�  "          A"�H@��@�p�@�
AR=qB)z�@��@�
=@�  A���B�                                    Bxie.6  �          A!@�@�@'
=Ao\)B,�H@�@��
@���A��B                                      Bxie<�  �          A#
=@ٙ�@�  @�AEB;G�@ٙ�@��@�33A㙚B"{                                    BxieK�  T          A ��@�33@��@ ��A8��BGG�@�33@�(�@�G�A�(�B/��                                    BxieZ(  "          A ��@�  @�R?��RA	G�BC�@�  @���@�G�A��B0
=                                    Bxieh�  �          A ��@�{@���?p��@��B
=@�{@���@W
=A�ffB                                    Bxiewt  �          A ��A��@�=q?G�@�\)B�
A��@��@?\)A�
=A��
                                    Bxie�  �          A ��A   @��>�Q�@�B��A   @�{@,(�Ax��B�                                    Bxie��  �          A"�\Aff@��>8Q�?��\Bz�Aff@�Q�@!G�AeBG�                                    Bxie�f  T          A"ffA��@�Q�?L��@�G�B{A��@�  @>{A��\A�z�                                    Bxie�  �          A#
=A ��@��
?��@�p�B��A ��@�ff@Z�HA��HBQ�                                    Bxie��  "          A"�H@���@��H?���A=qBG�@���@��@y��A�=qBff                                    Bxie�X  �          A�H@�33@��@��AH(�B�@�33@�(�@�  A�Q�A���                                    Bxie��  T          A (�@��@��@a�A�(�Bz�@��@�  @�  B
=A��                                    Bxie�  
�          A(�@�G�@�(�@C�
A�=qB1=q@�G�@��R@���B�HB33                                    Bxie�J  �          A�\@�  @�
=?�G�A,��B{@�  @�33@tz�A��B�                                    Bxif	�  T          A�
@�\@�z�?�
=A
�\B#��@�\@��
@j�HA�(�B�                                    Bxif�  T          Az�@�=q@ə�?�  @�Q�B&�R@�=q@�{@U�A��B=q                                    Bxif'<  �          AQ�@���@�
=?xQ�@�33B5G�@���@�33@]p�A��HB%G�                                    Bxif5�  T          AQ�@�Q�@�G�?��ABAp�@�Q�@��@z�HA���B/�                                    BxifD�  �          A��@�Q�@��?�(�A�B9{@�Q�@��@|��AîB%�                                    BxifS.  �          A��@Ǯ@���?�33A�BC��@Ǯ@��H@�Q�AŅB133                                    Bxifa�  �          A��@���@��?�Q�A!�BIz�@���@��H@��A��B5Q�                                    Bxifpz            A@�(�@��?���@��BG�@�(�@��@L(�A��B �                                    Bxif   "          A�
Ap�@���?��@��B��Ap�@�33@@  A�  A�33                                    Bxif��  
�          A ��AG�@�33?�R@b�\B�\AG�@�ff@+�Aw�A�                                    Bxif�l  �          A�HA�@�p�?   @9��B�A�@��@&ffAr�HA��                                    Bxif�  
�          A   A��@�p�>�p�@��A�A��@���@�\AS�
A���                                    Bxif��  T          A!p�A(�@�  >��?�  A홚A(�@�G�@�AA�A���                                    Bxif�^  �          A   A33@�>�@*=qB
�HA33@��H@#33AmG�A��H                                    Bxif�  �          A�
@���@�
=>���@�
B��@���@��
@,��A{33Bp�                                    Bxif�  T          A@�\@�\)?z�@Y��B�\@�\@��@7
=A�p�B                                      Bxif�P  �          A Q�A
=@������A���A
=@��>��H@333B                                    Bxig�  "          A!�A��@��׿Tz����B  A��@�{?��
@�\BQ�                                    Bxig�  �          A!��A
�R@��
�@  ���A�ffA
�R@���?���@�33A��H                                    Bxig B  "          A!�A�@��\�0���xQ�A�A�@�
=?��R@��A�                                    Bxig.�  "          A"ffA	p�@�33>�p�@�A�G�A	p�@��H@z�AS�A�
=                                    Bxig=�  T          A"{Ap�@�
=?�R@a�B	�
Ap�@��H@,(�AvffA�33                                    BxigL4  
�          A"{AG�@�  >��?���B
\)AG�@�  @ffAW
=B �
                                    BxigZ�  
�          A"ffA�@���>\@
�HB  A�@�\)@�RAbffB �\                                    Bxigi�  "          A"�\Az�@��?��@XQ�B33Az�@�{@,(�Au�B =q                                    Bxigx&  T          A#
=@�(�@�Q���Ϳ
=B.33@�(�@��@{A`��B&�                                    Bxig��  "          A#\)@陚@�(�?�@@��B-=q@陚@ƸR@@  A�33B!��                                    Bxig�r  
�          A$  A ��@�@��AF�HBz�A ��@�
=@��RA�z�A���                                    Bxig�  T          A&{A=q@��?���A-p�B
�RA=q@�ff@{�A�  A��                                    Bxig��  �          A$z�A=q@�  @��A]p�B{A=q@�
=@�(�A��A�=q                                    Bxig�d  �          A$Q�A
=@�  @*�HAq�A�33A
=@]p�@��RA��A���                                    Bxig�
  	�          A$��A
{@�ff@*=qAo
=A�z�A
{@i��@���A�A�\)                                    Bxigް  T          A#�AG�@�Q�@G
=A�{A�=qAG�@c33@��RA�  A���                                    Bxig�V  
�          A"=qA�@��H@P��A�(�B ffA�@e�@��
A�=qA�(�                                    Bxig��  
Z          A (�A z�@�G�@I��A�
=B p�A z�@dz�@�  A�33A��                                    Bxih
�  �          A�@��R@���@C33A���B�@��R@mp�@�ffA�G�A�                                      BxihH  
(          A�H@���@�Q�@L��A��\Bff@���@p��@��
A�p�A���                                    Bxih'�  �          Ap�@�ff@��\@4z�A��
B
�@�ff@~{@�G�A���A�                                      Bxih6�  "          Ap�@�G�@���@'
=Av�\Bz�@�G�@�ff@��HA��B\)                                    BxihE:  �          A{@���@�  @XQ�A��\B��@���@^{@�{A��HA�=q                                    BxihS�  "          A�@�  @���@mp�A�G�A�(�@�  @J=q@�A��A�                                    Bxihb�  �          A�R@أ�@�{@\��A�p�B$�@أ�@��\@��\B33BG�                                    Bxihq,  �          A
�R@�33@��
?�=q@���BX�@�33@��\@P��A�ffBI�                                    Bxih�  �          @��@I��@�
=�xQ���p�B���@I��@�(�?�(�A5��B�33                                    Bxih�x  "          @�G�@�(�@�ff�L�;�
=B]
=@�(�@��H@33A���BV(�                                    Bxih�  "          @��
@q�@�
=���R�#33Bb�@q�@��R?�p�Aip�B](�                                    Bxih��  �          @�G�@5@����G����B}\)@5@���?�ffA=qB}=q                                    Bxih�j  
�          @߮@z�@��ÿ�33�(  B�Q�@z�@�=q?s33A	B��{                                    Bxih�  T          @���?�\)@��R�!G��\B���?�\)@�  ��ff��{B���                                    Bxih׶  �          @�=q?��@�{��=q���HB�B�?��@���<#�
>\)B�ff                                    Bxih�\  
Z          @�=q@z�@�33�����B�k�@z�@�p�����W
=B��q                                    Bxih�  �          @�z�@
=q@���g���{B�\@
=q@����p��EG�B��q                                    Bxii�  
�          @��?h��@�  �E��\)B�(�?h��@�G������]p�B��                                     BxiiN  
(          @���?��@����2�\���B�B�?��@�  ��\)�@z�B��3                                    Bxii �  T          @ҏ\@6ff@�{�B�\���B^��@6ff@�\)��z��\(�Bp�R                                    Bxii/�  
�          @�z�@�=q@tz�fff��Q�A�33@�=q@x��>��@c33A��                                    Bxii>@  T          A=q@�(�@y����G��G�A�p�@�(�@��H>#�
?�33A�p�                                    BxiiL�  
�          A33A{@1�@   AQp�A���A{?�@5�A��
AT��                                    Bxii[�  T          A=q@�=q@333@�G�B��A�Q�@�=q?O\)@�B#(�@أ�                                    Bxiij2            A
�H@ٙ�@��@5A�ffA��@ٙ�@@��@�(�A�=qA���                                    Bxiix�  "          A\)@�=q@z�H@/\)A���A�33@�=q@0  @z�HAϙ�A���                                    Bxii�~  
�          A\)@�  @u@@��A�G�A��H@�  @%�@�z�A܏\A��                                    Bxii�$  
Z          A33A(�@@��?˅A$  A�ffA(�@�\@!G�A�33Ax(�                                    Bxii��  T          AQ�A{@�?�G�@�(�Ah��A{?��?�G�A*�HA:=q                                    Bxii�p  
Z          Ap�A  @R�\@�AL��A���A  @�@G
=A��Ar�R                                    Bxii�  "          A�A
�R@U@�\A]p�A��RA
�R@�@R�\A���At��                                    Bxiiм  �          Aff@��
@i��@uA�G�A���@��
@Q�@�33A�  Ar�\                                    Bxii�b  T          A{A
�R@
�H@��RA��
A`��A
�R?�R@�{A�
=@��H                                    Bxii�  �          AffA��?L��@���B�@�  A�ÿ��\@��B 
=C�~�                                    Bxii��  "          A\)@��\@{@�Q�BffA|��@��\>��
@�{B  @ff                                    BxijT  
�          AffA�?�@�z�A��A33A녾���@��HB G�C��                                    Bxij�  �          Ap�A�
?�(�@��
A�ffA7
=A�
=�\)@�A��H?                                       Bxij(�  �          A�RA33?�=q@�Q�B (�AI�A33��@��B�C��H                                    Bxij7F  "          A   @��?�p�@�=qB��AEp�@�����R@��B��C��H                                    BxijE�  �          A"{A
�\?��@�=qA�RA�
A
�\���@��RA�(�C�s3                                    BxijT�  "          A!�A�H?\@��B
ffA(z�A�H���@�p�BffC�XR                                    Bxijc8  
�          A#�Aff>�(�@�{A���@.{Aff���@��\A�p�C���                                    Bxijq�  
(          A%A�R?O\)@eA�ff@�=qA�R��
=@j=qA��C�                                    Bxij��  �          A((�A\)?Tz�@w
=A���@��
A\)���H@z�HA�p�C���                                    Bxij�*  �          A+�
A$(�?��@7�Ax��@��
A$(�=L��@C�
A���>�\)                                    Bxij��  
�          A(��@���@��@�B7�B�@���?�G�A	��B[�A��
                                    Bxij�v  T          A)@��@Mp�@�p�B\)A��H@��?�G�@�(�B#�@�G�                                    Bxij�  T          A&�RAz�@e�@p�AC�
A���Az�@*=q@P��A��
AyG�                                    Bxij��  
�          A(z�A��@U?���A�A�ffA��@#�
@3�
Ax(�Aj{                                    Bxij�h  �          A)G�Ap�@j�H?��
@޸RA��
Ap�@C33@=qAR�RA�                                      Bxij�  
�          A)p�A\)@u?��
@�z�A��\A\)@R�\@{AC�
A��
                                    Bxij��  
(          A%G�A=q�#�
@j�HA���C��
A=q����@^{A��C�p�                                    BxikZ  
�          A#�
A�>L��@fffA�
=?��HA��z�H@^{A���C��                                    Bxik   �          A"�HA  >\)@�ffA��?c�
A  ��(�@���A�=qC�=q                                    Bxik!�  T          A!�A
=?&ff@��A\@��A
=�:�H@���A��C��q                                    Bxik0L  �          A!A  ?
=@���A�(�@p  A  �h��@�  AӅC�q                                    Bxik>�  �          A�
A�>aG�@�G�Aأ�?�33A녿�  @��A��
C���                                    BxikM�  
�          A�
A  ?   @��A�@J�HA  �c�
@�A�z�C�,�                                    Bxik\>  �          A!��AQ�?��@��A���@�{AQ쾣�
@��A�C��q                                    Bxikj�  �          A"{A�?�=q@�  A�=q@���A����@�(�A��HC�1�                                    Bxiky�  �          A"�HAz�?xQ�@\��A��R@�G�Az�B�\@e�A���C�o\                                    Bxik�0  �          A"{A�\���
@1�A�ffC��3A�\�fff@(Q�Ar�\C�U�                                    Bxik��  "          A!G�A�
��  ?�\)@�z�C�EA�
���?z�H@�33C�J=                                    Bxik�|  �          A (�A33?�Q�@�Aԏ\A/�A33>B�\@��A�\?��\                                    Bxik�"  
�          A (�A��?�Q�@�  A�{A��A�ýu@�\)A�C�Ф                                    Bxik��  �          A   A?�ff@�
=A�  A8(�A>\@�=qA�{@p�                                    Bxik�n  �          A�HAff@
=@�\)A㙚A{
=Aff?@  @�\)A�ff@��H                                    Bxik�  �          A{Az�@{@�Q�A��HA�\)Az�?�G�@�=qA�ff@׮                                    Bxik�  �          A=qA
�H?�(�@��A��A333A
�H>�  @�
=A�z�?У�                                    Bxik�`  
�          A�Ap�?���@�
=A��\A*ffAp��\)@�{B�C��                                     Bxil  
�          A33A�
�\)@�A�
=C���A�
��G�@��
A���C�'�                                    Bxil�  �          Ap�A����@��A�z�C���A���@�G�A܏\C���                                    Bxil)R  	�          A{A�?5@��RA�z�@�33A녿!G�@�
=A��C���                                    Bxil7�  �          A{A�?�
=@=p�A�p�A=�A�?h��@Z=qA���@�p�                                    BxilF�  �          A{A�?Tz�@Tz�A��@�z�A��u@Z�HA�Q�C�@                                     BxilUD  �          A��A�\���@@��A�p�C�  A�\��  @+�A���C�W
                                    Bxilc�  �          A  A?���?�p�A%�@ָRA?�?�p�A>�H@I��                                    Bxilr�  �          A��AG�@@1G�A���Ag�AG�?���@W
=A���A	�                                    Bxil�6  �          AG�A�
@(��@=p�A�{A��HA�
?���@g�A�33A!��                                    Bxil��  "          AffA�?��\@   A;\)@�A�?
=@�\AW33@c33                                    Bxil��  �          A�HAff?�Q�@0  A��A\)Aff?�@C�
A�{@Mp�                                    Bxil�(  T          A�\A��?�z�@�
A@Q�A=qA��?s33@{Ag\)@�ff                                    Bxil��  "          AffA=q@hQ�=L��>�\)A�\)A=q@\��?�33@׮A�33                                    Bxil�t  �          Ap�A�
@�R?aG�@�AS�A�
?�{?\A
=A0��                                    Bxil�  
�          A!��Aff@��@�A9AH  Aff?�@%Am��A{                                    Bxil��  �          A$(�Az�@(�?���A&�RA`Q�Az�?�\@ ��Ab�\A#�                                    Bxil�f  "          A!��AQ�@0��@G�A8Q�A�\)AQ�@G�@0��A}p�A?�                                    Bxim  T          AffA��@{@(�Ad��AW
=A��?���@@  A�p�A\)                                    Bxim�  T          A�A�?�  @L(�A�G�@��RA�>k�@Z�HA��\?�                                    Bxim"X  T          Az�A=q>�Q�@UA�Q�@\)A=q�(��@R�\A�  C���                                    Bxim0�  
�          A�RA\)>L��?���A5��?�33A\)����?�A2�\C��3                                    Bxim?�  �          A��A���ff?��H@��HC���A���?�R@g
=C��                                    BximNJ  �          AQ�A\)��\)?.{@�Q�C���A\)��G�>k�?�p�C�5�                                    Bxim\�  �          AG�A(�@�R@3�
A�
=A|��A(�?��
@Z�HA�=qAff                                    Bximk�  "          A�A33@HQ�@Mp�A��RA�
=A33@�@�  Aʏ\A^ff                                    Bximz<  "          AG�AG�?���@S�
A���@���AG�=�Q�@_\)A��\?(�                                    Bxim��  T          AG�A=q��\?���A\)C���A=q��?k�@��\C��\                                    Bxim��  T          A��A��   ?k�@��C�W
A��*=q>\)?W
=C�޸                                    Bxim�.  
�          A  A�R�����
��\C��RA�R��\�L����=qC��)                                    Bxim��  T          A��A
=��ff�:�H��
=C���A
=���
��=q����C��                                    Bxim�z  �          A��A33�Y��?^�R@���C�]qA33����?z�@g
=C���                                    Bxim�   
�          A��Ap��#�
?��A
�HC��)Ap����\?��@��
C���                                    Bxim��  �          A\)AG���?�@VffC���AG���G�<#�
=��
C�aH                                    Bxim�l  �          A
=@�{@�  ?0��@���A��@�{@w
=?�33AB�RA�\)                                    Bxim�  
�          AG�@�\)@��ͽ�\)��A�p�@�\)@�\)?�(�@��
A�                                    Bxin�  �          A�@�
=@��
�u�\A�
=@�
=@�ff>�(�@.{A�z�                                    Bxin^  T          Ap�A?��
@Tz�A���A�A>��@c33A��R?�                                    Bxin*  
�          A��A�@G�@�AEG�AH��A�?��@!�Av�RA�R                                    Bxin8�  �          A�AQ�?�p�@ffAg
=A��AQ�?:�H@+�A�  @�(�                                    BxinGP  �          A�AG�@<(�?n{@��A�p�AG�@#33?޸RA,(�A��R                                    BxinU�  �          A�A
�R@C�
>��@<(�A�\)A
�R@2�\?�=qAG�A��R                                    Bxind�  T          Ap�A	�@:�H?8Q�@���A���A	�@%?��
A
=A��\                                    BxinsB  T          AQ�A��@Mp�?�{A�A�z�A��@+�@�RA_
=A�G�                                    Bxin��  T          A�A  @>{?p��@�G�A�{A  @%�?�  A-��A�\)                                    Bxin��  "          Az�A{@K�@��Ab=qA�=qA{@��@Dz�A�\)A�                                    Bxin�4  
�          A��A=q@AG�@�AmA�{A=q@p�@HQ�A���AlQ�                                    Bxin��  
�          Ap�A	@z�@-p�A�Q�AXz�A	?��H@L(�A�G�A (�                                    Bxin��  "          AQ�A�?O\)@�
=A��
@���A녾��H@�Q�A�ffC�Q�                                    Bxin�&  �          A=qA�H?
=q@�z�A�=q@j�HA�H�L��@�33A�=qC�K�                                    Bxin��  
�          A�Az�?.{@��A�  @�G�Az���@�p�Aڣ�C��q                                    Bxin�r  �          AA\)?\(�@��RA݅@�=qA\)��G�@���A��HC���                                    Bxin�  
�          A�\A
ff?E�@�
=A�G�@��\A
ff��ff@�Q�AѮC��                                     Bxio�  "          AG�A��?�
=@W
=A�ffA,��A��?(��@l(�A�@��                                    Bxiod  �          AAp�?�p�@S33A�
=A0��Ap�?:�H@j=qA��@��R                                    Bxio#
  
Z          A�HA33@�@0��A��HAl��A33?�  @Tz�A��\A�
                                    Bxio1�  T          A�A�
@�@B�\A��AK\)A�
?���@_\)A�Q�@�ff                                    Bxio@V  
�          A�A=q@�\@:=qA�p�A`��A=q?���@\(�A�
=A	p�                                    BxioN�  �          A�
Az�@0  @Q�Ab=qA�\)Az�?�(�@C33A�{AEG�                                    Bxio]�  "          AffAff@\(�?��@���A�G�Aff@<(�@{AT��A�ff                                    BxiolH  "          AQ�A
=@[�?�{A0  A�  A
=@1�@/\)A���A�(�                                    Bxioz�  
Z          A��A  @Z=q?��HA ��A��A  @333@%Aup�A�Q�                                    Bxio��  �          A=qA��@Q�@��AMA�p�A��@#33@AG�A�z�A{33                                    Bxio�:  T          A
=A�@W�@  AQ�A��HA�@'�@EA���A���                                    Bxio��  �          A   A{@S�
@Q�A\(�A�\)A{@!G�@L��A�33Aw\)                                    Bxio��  �          A ��Aff@P��@%An�RA�
=Aff@�H@XQ�A�p�Amp�                                    Bxio�,  T          A�A�H@A�?�33A
=A�Q�A�H@p�@�HAd��ApQ�                                    Bxio��  �          A
=A��@:�H?��A  A�{A��@ff@Q�Ac33Ah                                      Bxio�x  �          A�RA��@<(�?У�A33A��
A��@Q�@�Ac33Ak�                                    Bxio�  �          AA��@2�\?���A�HA��A��@  @G�A[\)A_�
                                    Bxio��  "          A33A�@Mp�?�33@ڏ\A�A�@1G�?�p�A<��A��                                    Bxipj  "          AG�A��@mp�?L��@�A�z�A��@Vff?��
A'33A��R                                    Bxip  �          AA��@s�
?s33@�G�A��HA��@Z=q?���A6�RA�\)                                    Bxip*�  "          A�A�R@n{?
=@^�RA�
=A�R@Z�H?���A��A�                                    Bxip9\  
(          A\)@�  ?�=q��33�7��A?�
@�  @C�
����"�
A��H                                    BxipH  
�          A�\@�\)@���{�-�\A�  @�\)@o\)��z���A                                    BxipV�  �          A��@��@\)����O�A�Q�@��@���ʏ\�,��B �H                                    BxipeN  �          A�@�z�?fff��z��M\)A��@�z�@2�\��z��:{A�ff                                    Bxips�  
�          A�@�\)���
���X�
C�XR@�\)?޸R��p��P�A���                                    Bxip��  
�          A��@�?����
=�9�An=q@�@X����G��"G�A噚                                    Bxip�@  "          A  @�Q�?\��{�4�AS
=@�Q�@P����G��A��H                                    Bxip��  �          A{@�{?У���{�(\)AS\)@�{@R�\�������A��H                                    Bxip��  |          A��A�\@
=��G���33A��HA�\@c33�\)����A�                                      Bxip�2  "          A�A�
@I�����H���A�G�A�
@�  �e����Aڏ\                                    Bxip��  �          AffA\)@�  �tz�����A��A\)@��
�'��u��A��                                    Bxip�~  
�          A�A�@���L����  A�33A�@�����R�7�
A�R                                    Bxip�$  "          A�A��@��
�#�
�nffA�\)A��@��
��
=�ٙ�B                                    Bxip��  �          A�R@�{@`  ������  A�Q�@�{@�  �L������A�\                                    Bxiqp  
�          A\)@��@�\���\�G�Ae��@��@\(����H��RA�ff                                    Bxiq  "          A
=@�\)@'�����{A�G�@�\)@z�H����A�\)                                    Bxiq#�  �          A
=@�\)@'�����33A�p�@�\)@{���{��  A�p�                                    Bxiq2b  �          A
=AG�@1����\���A��AG�@�����p��ƣ�A�p�                                    BxiqA  T          A�\@�  @W����
���A�  @�  @��H�����  A�33                                    BxiqO�  �          A\)@�
=@B�\�����HA��@�
=@��
��  ��G�A��                                    Bxiq^T  
�          A
=@�  @K����\��A���@�  @��R��=q��(�A�                                    Bxiql�  �          A\)A�@O\)��z����
A�{A�@��H�h�����\A���                                    Bxiq{�  �          A�\@���@�����\���
BG�@���@��
�9������B��                                    Bxiq�F  "          A{@�33@�{��ff��=qBG�@�33@���P�����RB ��                                    Bxiq��  T          A�H@���@|(������(�A��H@���@����c�
��  B��                                    Bxiq��  "          A�@�
=@��
����=qA�  @�
=@������\B\)                                    Bxiq�8  "          A�\@��
@o\)��{��Q�A�\)@��
@��\�dz�����A�                                      Bxiq��  �          A�RA��@r�\�w
=���
A�z�A��@����0����p�A�\                                    Bxiqӄ  �          A�A�@k��XQ���
=A��
A�@�p��z��Z�RAܸR                                    Bxiq�*  T          AG�A�\@/\)�����  A�Q�A�\@p  �]p�����A�z�                                    Bxiq��  	�          A{A�
@mp��\)���RA�{A�
@���:=q����A��                                    Bxiq�v  "          Ap�A�
@l���z�H���A��A�
@��\�5��  A�\                                    Bxir  
�          A  @�@b�\��33���HA�ff@�@����S33��
=A�                                      Bxir�  T          A��Aff@\������Q�A��Aff@����J=q����A��                                    Bxir+h  	�          A�A ��@p�����R����Aȏ\A ��@��R�G
=���A��H                                    Bxir:  "          AQ�@�z�@|(���ff�˅A�ff@�z�@�(��C�
���
A��
                                    BxirH�  �          A�R@�Q�@����Q���(�A�@�Q�@��
�Tz���z�B	�                                    BxirWZ  �          A{@��@�\)���H����B@��@��@�����B��                                    Bxirf   �          A��@��@���������  A��H@��@��R�C�
��(�B�
                                    Bxirt�  �          A�RA z�@vff�����ffA��HA z�@����C�
���HA��                                    Bxir�L  T          A Q�A��@*�H�W
=��
=A�p�A��@[��%��o33A�{                                    Bxir��  �          A (�Ap�@z��.{�|��A_\)Ap�@;��33�=��A�p�                                    Bxir��  
�          A ��A�R@H���(Q��r�RA�p�A�R@l�Ϳ�  � ��A�                                      Bxir�>  "          A z�A(�@c�
�&ff�pz�A��RA(�@�33��\)��A��                                    Bxir��  �          A!A��@O\)�33�;33A��HA��@i�������A��\                                    Bxir̊  �          A ��AG�@u���p��5A��HAG�@�ff�u��\)Aƣ�                                    Bxir�0  �          A z�A�@h��� ���9p�A��
A�@��׿����
=A�ff                                    Bxir��  �          A ��Aff@������R�\A�=qAff@��R���H��A���                                    Bxir�|  �          A!G�A��@��H�,(��w\)A�
=A��@�(���=q���Aݙ�                                    Bxis"  �          A!�A�\@�{�,(��v=qA�G�A�\@�ff��\)���B                                    Bxis�  �          A!�A�H@�\)�{�b�\A���A�H@�{��z����HBQ�                                    Bxis$n  
�          AffA(�@��\�!G��k�A��A(�@������R��RB�
                                    Bxis3  �          AG�AG�@���=p�����A��AG�@��H�����+�A�p�                                    BxisA�  T          A�\A��@�Q���R�h(�A��HA��@�\)��(���B �
                                    BxisP`  "          A
=A�R@���.{�}�A�\)A�R@��H��G��  A�
=                                    Bxis_  �          A z�A	@��\�G�����AʸRA	@�
=�G��9A��
                                    Bxism�  "          A\)A��@e�a����RA��
A��@�33�"�\�mp�A�G�                                    Bxis|R  T          A�A�R@|(��333��(�A���A�R@�Q��  �&{A�p�                                    Bxis��  
�          A�A�
@\)�"�\�rffAɅA�
@�  ���R���A��                                    Bxis��  T          AA�@x�ÿW
=��p�A��A�@~�R>�?J=qA�G�                                    Bxis�D  
�          A!��A
�\@��@EA���A�\)A
�\@O\)@�  A�G�A��                                    Bxis��  "          A
=@��H@�p�@��\A���A��H@��H@a�@�33A���A�                                    BxisŐ  T          A33@�G�@��@�G�A�33A�@�G�@L��@�
=A�Q�A���                                    Bxis�6  "          A#\)@ۅ@z�H@�Bz�A��
@ۅ@\)@�{B3z�A��\                                    Bxis��  
�          A ��@�ff@�33@��
B�\B'��@�ff@mp�@�\)B7��A���                                    Bxis�  �          A!�@��H@�33@�B  B*  @��H@l(�@陚B:Q�B33                                    Bxit (  �          Aff@�Q�@�(�@�z�A���B=q@�Q�@~{@�G�B�A��                                    Bxit�            Az�@��@�p�@�  B(��B8�@��@\��@��BM�B
�\                                    Bxitt  �          A�@�
=@��@�B<ffBD(�@�
=@C33@���Bbp�BG�                                    Bxit,  "          AQ�@��@�(�@�{B;��B5�H@��@4z�@�33B^A�(�                                    Bxit:�  �          A�@�{@�  @�=qBG�HB<=q@�{@'
=A
=Bk�A�33                                    BxitIf  �          A(�@���@�z�@�  B@
=B0��@���@%�@�33Ba�
A�p�                                    BxitX  
�          A33@��H@P��@�G�BQ�AŮ@��H?��
@���B$�A`��                                    Bxitf�  
�          A%A33@���@x��A��RA�z�A33@^�R@��
A�z�A�\)                                    BxituX  "          A&ffA\)@u@l(�A��
A��
A\)@6ff@�  A���A�33                                    Bxit��  �          A$(�A�@Z=q?�z�@���A��HA�@>{@�AD��A�(�                                    Bxit��  T          A%p�A33@�Q�@Mp�A��A���A33@HQ�@��\A�(�A�                                      Bxit�J  
�          A%A  @\)@L��A��
A�33A  @G
=@���A�ffA���                                    Bxit��  T          A&=qA�R@�Q�@3�
Az�RA��HA�R@N{@k�A��A���                                    Bxit��  �          A%�A�\@o\)@p�AD��A��A�\@G
=@A�A�(�A�Q�                                    Bxit�<  �          A%�AQ�@]p�?��RA1A��AQ�@8��@0  Aw�A�33                                    Bxit��  
(          A#�A��@N�R?�A�RA��A��@/\)@��AXQ�A�                                    Bxit�  �          A#\)A�\@HQ�?k�@���A��A�\@4z�?У�A33A�=q                                    Bxit�.  
�          A#
=A�@E?˅A  A�
=A�@'�@�AN�HAuG�                                    Bxiu�  �          A#�A�\@B�\?��\@��A�A�\@)��?���A0��Au                                    Bxiuz  �          A#33A\)@Dz�>��H@/\)A�z�A\)@7�?���@�Q�A�                                    Bxiu%   
�          A!�A�\@*=q�s33��p�Av=qA�\@3�
������Q�A��                                    Bxiu3�  �          A!p�A�H@(���z�� ��Ab{A�H@,�ͿL����G�AyG�                                    BxiuBl  �          A"ffAz�@p��Ǯ���AK�Az�@ �׿}p����Afff                                    BxiuQ  �          A!A@<�Ϳ&ff�j�HA��\A@AG�=�\)>\A���                                    Bxiu_�  �          A!�A��@��R?(��@n�RA��
A��@�?�
=Ap�A�Q�                                    Bxiun^  
�          A!��A
=@0  <#�
=#�
A}A
=@+�?&ff@mp�Av�H                                    Bxiu}  T          A!p�AQ�@R�\��\�8Q�A���AQ�@Tz�>�\)?�=qA��
                                    Bxiu��  �          A!�A�@�{�aG����A�33A�@���>�?5A��H                                    Bxiu�P  "          A (�A�@�=q������=qA�p�A�@��R���0��A�p�                                    Bxiu��  �          A   A@�=q���@  A���A@��\>�G�@!G�A�
=                                    Bxiu��  
�          A{AG�@���?}p�@�=qAɅAG�@s�
?�
=A5�A��\                                    Bxiu�B  
Z          A��Aff@�  ?��R@�Q�A�Aff@g
=@��AH��A�ff                                    Bxiu��  �          AG�Aff@��
?&ff@s�
Aƣ�Aff@vff?˅A�A�G�                                    Bxiu�  �          Az�A=q@�=q>���?�z�A�
=A=q@x��?�G�@�p�A���                                    Bxiu�4  	�          A�A  @�\)>�G�@%AΣ�A  @���?�z�A�A�\)                                    Bxiv �  
�          A�A	��@��\?��@ÅA�ffA	��@��R@z�AC�A���                                    Bxiv�  T          Az�A	�@�p�?��\@�RA�\)A	�@�Q�@��AUAǮ                                    Bxiv&  	`          A��A(�@��?��HA9G�A�
=A(�@u�@;�A�\)A�                                      Bxiv,�  W          A�A�@�ff@#33Apz�A��A�@o\)@`  A�G�A���                                    Bxiv;r            A�A�
@�(�@p�Ag33A�=qA�
@l(�@Z=qA���A�{                                    BxivJ  %          AA	�@�{@G�AT��AυA	�@c�
@J�HA�Q�A�p�                                    BxivX�            A�A	p�@��@%�Ar{Ȁ\A	p�@Z=q@\��A�A��                                    Bxivgd  	          A��AQ�@���@K�A���A�p�AQ�@S�
@���A�p�A��R                                    Bxivv
  	          AQ�@�ff@���@b�\A�z�A癚@�ff@^{@��RAٮA�z�                                    Bxiv��  �          A  AQ�@��@3�
A�Q�A�=qAQ�@c33@n�RA�  A��                                    Bxiv�V  �          A
=A{@�33@?\)A��A���A{@b�\@z=qA�=qA�ff                                    Bxiv��  %          A33A�
@�\)@8Q�A�{A�\)A�
@]p�@qG�A�
=A�(�                                    Bxiv��  �          A�HA=q@���@A�A�{A�A=q@^{@{�A��A���                                    Bxiv�H  
(          A33A ��@��@a�A��HA��A ��@I��@��A�{A��\                                    Bxiv��  
�          A
=A   @AG�@���A���A��A   ?�z�@��HA�(�AW33                                    Bxivܔ  	�          A��@��@�\@�
=Bp�Ak�@��?J=q@���B�@��H                                    Bxiv�:  u          A
=A@�(�@S�
A�  A�A@P��@�p�A��
A�\)                                    Bxiv��  T          A�@�{@��
@qG�A�=qA��
@�{@h��@�
=A�(�Aʏ\                                    Bxiw�  T          A�@�p�@�p�@��
A�A���@�p�@@��@�ffB�
A��H                                    Bxiw,  T          A�
@��H@��
@g
=A�Q�B�@��H@z�H@��
A��Aڏ\                                    Bxiw%�  
�          A�@��R@��@J=qA�B�R@��R@���@��RA�p�A�(�                                    Bxiw4x  �          A�
@���@���@l��A���A�R@���@e@�z�A㙚A�ff                                    BxiwC  �          A33@��H@�ff@qG�A�G�A�Q�@��H@n�R@��A�=qA�33                                    BxiwQ�  
�          A�R@���@�33@k�A�\)A�  @���@h��@�(�A��A�                                    Bxiw`j  �          A=q@�ff@�Q�@��
A���A�p�@�ff@]p�@�G�A�(�A�33                                    Bxiwo  "          Ap�@�@��@���A��B G�@�@e�@��A��HA�\)                                    Bxiw}�  �          A�RAG�@�G�?�A�A��AG�@�G�@)��A�A��                                    Bxiw�\  
�          A\)A��@��R?�33AQ�A�Q�A��@�
=@*�HA\)A�33                                    Bxiw�  
�          Az�@�z�@�33@G�A���A��\@�z�@���@�(�A�  A�                                      Bxiw��  
Z          A33@��@���@EA�ffA�ff@��@}p�@��\A�33A��                                    Bxiw�N  �          A�A ��@�\)@G
=A�
=A�ffA ��@j=q@�G�A�{A�p�                                    Bxiw��  �          A
=A�@���@J=qA�Q�AЏ\A�@Mp�@\)A�Q�A��\                                    Bxiw՚  �          A�\A   @��@c33A���A�\)A   @J=q@��
Aי�A�ff                                    Bxiw�@  
�          Aff@��
@�
=@VffA���A��H@��
@fff@���A�(�A��H                                    Bxiw��  �          AG�@�G�@�33@G�A�Q�A�z�@�G�@q�@��\AɮA��                                    Bxix�  
�          A�A{@�=q@�AL��A�RA{@~�R@EA���AЏ\                                    Bxix2  �          A��A�H@�?���A
=A�Q�A�H@��R@%�AzffA�                                      Bxix�  T          A�A ��@��@<(�A��
A�p�A ��@h��@vffA�=qA�ff                                    Bxix-~  T          AG�@��\@���@]p�A�G�A�{@��\@Z=q@��\A��A�z�                                    Bxix<$  �          Ap�@�
=@�p�@xQ�A���A�
=@�
=@L��@�
=A뙚A��                                    BxixJ�  T          Ap�A{@�
=@/\)A�33A�G�A{@`  @g
=A���A�(�                                    BxixYp  
�          A�A�@��@5A��Aԣ�A�@W�@l(�A�ffA�z�                                    Bxixh  w          A�A{@��H@HQ�A�A��
A{@R�\@}p�A\A�Q�                                    Bxixv�  
i          Ap�A��@��@@��A�z�AٮA��@X��@w�A�(�A��                                    Bxix�b  T          Ap�A=q@��@=p�A��A�=qA=q@Vff@s33A��RA��R                                    Bxix�  T          AffA (�@�33@_\)A���A�
=A (�@N{@�=qAԣ�A�G�                                    Bxix��  �          A��A��@r�\@>�RA���A�z�A��@A�@p  A�=qA�z�                                    Bxix�T  T          A��A��@q�@<(�A��RA�{A��@A�@l��A�  A���                                    Bxix��  
�          Ap�A�@l��@P��A��HA��A�@8Q�@�  A�G�A���                                    BxixΠ  T          A��A=q@p  @S�
A�{A��
A=q@:�H@��A�33A��
                                    Bxix�F  T          A��A{@n�R@XQ�A��
A�33A{@8Q�@�(�A̸RA�=q                                    Bxix��  
�          Ap�AQ�@]p�@W
=A�z�A�AQ�@(Q�@���A�{A�
=                                    Bxix��  �          A��A@B�\@i��A�ffA�A@	��@�\)A���Af�R                                    Bxiy	8  �          A��@�\)@e�@vffA�{A�G�@�\)@(��@���A�p�A�=q                                    Bxiy�  T          A��Ap�@Mp�@x��A���A�p�Ap�@G�@�Q�A�Az�H                                    Bxiy&�  �          A(�@��R@Z�H@y��A��
A��
@��R@{@��A�A��                                    Bxiy5*  �          A��@�@X��@�(�A̸RA�\)@�@��@���A��A�=q                                    BxiyC�  �          Ap�@�
=@X��@��A��HA�z�@�
=@��@�Q�A�  A��                                    BxiyRv  T          AQ�@�z�@L(�@���A��A��@�z�@
=q@��
A�{At��                                    Bxiya  
�          AQ�@�
=@P  @���A���A�z�@�
=@
=q@�(�B\)Az�\                                    Bxiyo�  T          A��@�p�@5@�{Aޣ�A���@�p�?��
@��RA�\)AJ�R                                    Bxiy~h  
Z          AQ�A\)@�@��\A�Q�Ag
=A\)?���@�{A�{@��                                    Bxiy�  T          A�A Q�@
=q@���A�Aqp�A Q�?���@�p�A�=q@��                                    Bxiy��  �          A\)@��@   @���A���Ac
=@��?c�
@�33B��@�{                                    Bxiy�Z  
Z          A33@�ff@aG�@�(�A���A�Q�@�ff@=q@���BQ�A�G�                                    Bxiy�   
�          A33@�p�@^{@�
=AAȣ�@�p�@@��
B
z�A�(�                                    BxiyǦ  �          A
=@�  @J�H@�Q�A�ffA�33@�  @�\@��HB	��As33                                    Bxiy�L  �          A
=@�@K�@��HA�
=A�\)@�@�@�B=qAu�                                    Bxiy��  T          A33@�ff@:�H@���B Q�A�G�@�ff?޸R@�G�B��ARff                                    Bxiy�  T          A�H@��R@Q�@��\A���A�ff@��R@\)@�ffA�G�A��                                    Bxiz>  �          Aff@�z�@W
=@��\A�\)A�=q@�z�@z�@�
=A�
=A�p�                                    Bxiz�  T          Aff@�ff@W
=@�\)AծA��H@�ff@ff@��A�33A��                                    Bxiz�  �          A�\@�@U@��Aߙ�A�p�@�@�@�G�B�A�p�                                    Bxiz.0  
�          A�\@�ff@c33@���A�\)A��@�ff@{@�{B�
A���                                    Bxiz<�  �          A�\@�Q�@fff@��
A�G�A̸R@�Q�@#33@���B��A�                                    BxizK|  
�          A=q@�(�@j�H@�\)A�Q�AӅ@�(�@%@�B�A���                                    BxizZ"  T          A�R@��@J�H@�(�A�A�  @��@z�@�
=B��At��                                    Bxizh�  "          A�R@�{@P��@��HA��A��@�{@{@��RA�=qA���                                    Bxizwn  
�          A�@�p�@W�@�{AԸRA��@�p�@
=@��HA�z�A���                                    Bxiz�  T          A{@�@Y��@�p�A�\)A�G�@�@��@�=qA��A���                                    Bxiz��  �          A�\@�z�@Y��@���A؏\A��@�z�@�@�p�A��RA�                                      Bxiz�`  "          A
=@�33@I��@�z�A�33A��
@�33@�\@�
=B�Ap��                                    Bxiz�  �          AG�@�{@|��@?\)A��RAم@�{@L(�@r�\A�A�ff                                    Bxiz��  T          A��@�=q@�=q@{Av�HA�\)@�=q@j�H@W
=A��A��                                    Bxiz�R  �          A��@�\)@y��@`��A�Q�A�=q@�\)@A�@���A�(�A�\)                                    Bxiz��  T          Ap�@�Q�@g
=@p  A�\)AǙ�@�Q�@,(�@��RA��
A��                                    Bxiz�  �          A=q@��
@\��@q�A�{A�33@��
@!�@�ffA�Q�A�ff                                    Bxiz�D  �          A\)@���@c33@r�\A���A�G�@���@'�@�\)A�{A��H                                    Bxi{	�  �          A�
@�G�@fff@���A�Q�A�Q�@�G�@'
=@�  A�=qA�ff                                    Bxi{�  �          A
=@�\)@���@��\AͅA��@�\)@AG�@��
A���A��                                    Bxi{'6  �          A�@�Q�@z=q@�\)A�(�A��@�Q�@8Q�@��A�=qA��
                                    Bxi{5�  �          A��A   @Q�@�=qAɮA�z�A   @33@�ffA�\)A�z�                                    Bxi{D�  �          A��@���@U�@�A�33A���@���@G�@��A��
A�Q�                                    Bxi{S(  �          A��@�z�@Vff@�p�A�  A���@�z�@\)@�G�B�\A��R                                    Bxi{a�  �          Ap�@�\@Y��@���A�=qA�p�@�\@��@�{B
  A��R                                    Bxi{pt  �          A��@�\)@vff@��
A�  A�ff@�\)@+�@�33B
=A���                                    Bxi{  "          Ap�@��
@�(�@���A��A�{@��
@A�@�=qB�HA���                                    Bxi{��  "          Aff@�{@��\@��A��HA�G�@�{@P��@��RB��A�
=                                    Bxi{�f  �          A\)@���@w
=@��A�z�A�=q@���@0  @���B�A��\                                    Bxi{�  �          A�@���@aG�@�A��A�33@���@��@��HB(�A��                                    Bxi{��  �          A(�@�p�@L(�@�{A�{A��@�p�@�@���B33Ak\)                                    Bxi{�X  
�          A��@�G�@Q�@�{A�\A��\@�G�@ff@�G�B
  Aq                                    Bxi{��  �          A  @�R@3�
@��B
33A�G�@�R?��
@�\)Bz�A9p�                                    Bxi{�  �          A
=@�ff?�{@�{B
=Ah(�@�ff>�ff@�ffB'(�@dz�                                    Bxi{�J  �          A  @��\@
=q@�z�B\)Aw
=@��\?k�@��B=q@�p�                                    Bxi|�  T          A��@���?�p�@���B
(�Ad��@���?5@�=qB�H@��                                    Bxi|�  T          A��@�p�?�=q@�\)B��APz�@�p�?z�@�  Bz�@�p�                                    Bxi| <  �          AA
=@   @��A��A[�A
=?Q�@�B�@�
=                                    Bxi|.�  �          AffAz�@�@�=qAۮA_\)Az�?��@�p�A�z�@��                                    Bxi|=�  �          A�
A�
@Q�@�\)A��HA���A�
?�(�@�(�B\)Aff                                    Bxi|L.  �          A   @��@0  @��RB�A�(�@��?�z�@�p�BffA%p�                                    Bxi|Z�  
�          A Q�@�{@#33@�G�BffA���@�{?�G�@�
=BffAz�                                    Bxi|iz  "          A ��A�@��@�G�A��
A�ffA�?�(�@�ffB��A��                                    Bxi|x   �          A!��A ��@(�@��HB�A���A ��?��@��B��A ��                                    Bxi|��  
�          A"�RA (�@p�@��B�RAv�\A (�?Y��@�ffBQ�@\                                    Bxi|�l  T          A"�\@�ff@33@��BffAg33@�ff?+�@�G�B
=@��                                    Bxi|�  
�          A"�HA	��@��@�A�Q�Ae�A	��?�ff@�G�A�\)@�{                                    Bxi|��  T          A#�A{@�@��HA��
AR�HA{?�G�@�{A�p�@�\)                                    Bxi|�^  T          A$  A
ff@�@�=qA�G�AXz�A
ff?fff@��A���@�
=                                    Bxi|�  �          A$Q�A�\?�{@��\BAG33A�\?�@��B�@w�                                    Bxi|ު  �          A$(�@���?Ǯ@�33B  A2�H@���=�G�@�G�Bff?L��                                    Bxi|�P  
�          A%p�A��?�
=@��HB	�HAQp�A��?z�@��
B�\@\)                                    Bxi|��  T          A&ff@�ff?�(�@�{Bz�AC�@�ff>�  @��B?�\                                    Bxi}
�  �          A&�\@���?�z�@ۅB%
=AD(�@���=�Q�@��B*�H?0��                                    Bxi}B  �          A%�@�(�?�(�@�\)B(�AEp�@�(�>u@�ffB!�?��H                                    Bxi}'�  
�          A&=q@�=q?\@�p�B��A/�@�=q�#�
@��HB$�RC��
                                    Bxi}6�  
�          A&�\A Q�?���@�
=B�
A$Q�A Q��@�(�BffC��f                                    Bxi}E4  "          A&�RAz�?�Q�@�p�B{A=qAz�<�@ʏ\B��>aG�                                    Bxi}S�  T          A&�HA\)?�z�@�Q�B�A�A\)���
@�p�B  C��=                                    Bxi}b�  �          A'�
A{?��H@ƸRB  A33A{�W
=@�=qB{C�H�                                    Bxi}q&  
�          A((�A�H?��R@�(�B�HAA�H�#�
@�  B(�C�s3                                    Bxi}�  T          A(z�A�?��@�  A�\)@���A�>\@�ffA�z�@�                                    Bxi}�r  T          A)G�A33?���@R�\A���Az�A33?.{@a�A��
@xQ�                                    Bxi}�  �          A*�\A!G�?�z�@Q�A�33@��A!G�>���@\��A��@                                      Bxi}��  "          A)p�A   ?�(�@Q�A�ff@�
=A   >�@^{A��@'
=                                    Bxi}�d  �          A(��A!?���@/\)Apz�@�Q�A!?:�H@>�RA�
=@��
                                    Bxi}�
  �          A)p�A"ff?\@#�
A`  A��A"ff?h��@5Ax��@�(�                                    Bxi}װ  �          A*{A!�?���@>{A��@�\A!�?&ff@L(�A��@i��                                    Bxi}�V  �          A*ffA"=q?�{@<��A���@���A"=q?(��@K�A��@o\)                                    Bxi}��  �          A*ffA"{?���@<(�A�(�A{A"{?@  @L(�A�p�@�\)                                    Bxi~�  "          A*=qA!G�?�=q@H��A�p�@�Q�A!G�?
=@W
=A�G�@U                                    Bxi~H  "          A*ffA!G�?�=q@J�HA���@���A!G�?z�@X��A�z�@Tz�                                    Bxi~ �  �          A*�RA"�\?��H@:�HA}�AffA"�\?B�\@J�HA�Q�@���                                    Bxi~/�  �          A*�HA"ff?�z�@AG�A�p�@�p�A"ff?0��@P��A�=q@z=q                                    Bxi~>:  �          A*ffA"�\?�33@-p�Ak�A\)A"�\?}p�@@��A�\)@��H                                    Bxi~L�  �          A)p�A!p�?��
@)��Ah(�A�
A!p�?���@>�RA�
=@��                                    Bxi~[�  �          A*=qA�?��@\(�A��@�33A�>�@h��A�Q�@.�R                                    Bxi~j,  �          A*ffAff?�33@o\)A�Q�@ӅAff>�=q@y��A��
?�=q                                    Bxi~x�  
�          A*�HA�?��@hQ�A���@�
=A�>B�\@qG�A��?�\)                                    Bxi~�x   d          A*�\A!�?�G�@U�A�@�  A!�>k�@^�RA�Q�?���                                    Bxi~�  f          A+�A(��?c�
?��A�@��A(��>��H?�ffA�H@*�H                                    Bxi~��  "          A*�\A(  ?5?�Q�A{@w
=A(  >��R?�AQ�?�                                    Bxi~�j  �          A*�\A(��?\(�?�(�@љ�@�p�A(��?\)?���@�{@C�
                                    Bxi~�  �          A)�A((�?c�
?�G�@ڏ\@��A((�?z�?�
=@�  @K�                                    Bxi~ж  �          A*ffA'�?�33?�G�Aff@�  A'�?E�?�p�A��@�\)                                    Bxi~�\  �          A'�
A&ff?B�\?��@@  @�ffA&ff?�R?5@xQ�@Z=q                                    Bxi~�  �          A'33A&�R>��
=q�>{@!�A&�R?녾�
=�z�@I��                                    Bxi~��  �          A)G�A)�>��H>.{?p��@(��A)�>�G�>�\)?\@
=                                    BxiN  �          A*�RA*=q>��H?�@C�
@'�A*=q>�{?+�@e�?���                                    Bxi�  T          A)�A)p�?�?&ff@`��@6ffA)p�>�Q�?@  @�=q?�(�                                    Bxi(�  �          A(  A&{=�?�G�Az�?(��A&{�k�?�  A\)C�^�                                    Bxi7@  T          A&�HA%G�>��?�\)@��@%�A%G�>\)?�Q�@�ff?E�                                    BxiE�  �          A%�A$��?.{?Tz�@�=q@p��A$��>��?u@��@'
=                                    BxiT�  �          A&�RA%�?\)?!G�@`  @G
=A%�>���?@  @��
@{                                    Bxic2  "          A'\)A&{?��?5@w�@�ffA&{?Y��?k�@��\@��                                    Bxiq�  �          A&{A%�>�
=?k�@��@�
A%�>B�\?}p�@�  ?���                                    Bxi�~  �          A%p�A$Q�?:�H?Y��@�ff@��\A$Q�?�?}p�@�Q�@8��                                    Bxi�$  �          A%��A$��?Tz�>��@%�@��HA$��?333?#�
@dz�@w�                                    Bxi��  �          A'\)A&�\?}p�>�\)?��@�ffA&�\?fff?   @0  @�ff                                    Bxi�p  T          A)��A(Q�?xQ�?^�R@�\)@���A(Q�?=p�?���@�=q@���                                    Bxi�  �          A*=qA)��?8Q�?z�@I��@w
=A)��?\)?:�H@|��@A�                                    Bxiɼ  �          A(��A(  ?
=q?:�H@~{@:�HA(  >�33?Tz�@�G�?�                                    Bxi�b  �          A&ffA%p�?.{?333@vff@qG�A%p�?   ?W
=@��@1�                                    Bxi�  "          A%�A$  ?:�H?G�@�33@��HA$  ?�?n{@�@>{                                    Bxi��  �          A"�RA!��?W
=?:�H@�z�@�\)A!��?#�
?h��@��
@i��                                    Bxi�T  �          A"{A!�?Q�?��@Z=q@�A!�?(��?G�@��@p��                                    Bxi��  �          A!�A ��?\(�?(�@]p�@�(�A ��?0��?J=q@��@|(�                                    Bxi�!�  �          A ��A (�?�?:�H@�@>{A (�>���?Tz�@�  ?�z�                                    Bxi�0F  �          A�A�\?�?G�@�\)@>�RA�\>��
?aG�@���?�{                                    Bxi�>�  "          A z�A�?��?:�H@�p�@HQ�A�>�Q�?W
=@���@z�                                    Bxi�M�  �          A ��A�
?��?\(�@�@H��A�
>���?xQ�@���?�33                                    Bxi�\8  �          A"�RA!G�?�ff>�33?��H@��RA!G�?p��?z�@S33@��H                                    Bxi�j�  �          A#�
A"=q?���#�
���
@��\A"=q?�G�>k�?��@�{                                    Bxi�y�  �          A%��A$z�?p��=L��>�=q@�ffA$z�?fff>�=q?�p�@��                                    Bxi��*  �          A$��A$  ?aG�>W
=?�
=@�(�A$  ?L��>��@�\@�
=                                    Bxi���            A$��A#�
?:�H>\@ff@��A#�
?�R?
=q@@  @]p�                                    Bxi��v  �          A"�HA!�>��
?��\@�\)?�A!�=�\)?��@��>���                                    Bxi��  �          A"{A ��>Ǯ?�z�@�33@p�A ��=���?�(�@�?
=                                    Bxi���  �          A (�Aff>W
=?��@�{?��RAff����?��@��C��R                                    Bxi��h  �          A��A�R>���?���@�p�@Q�A�R=�?�Q�@�G�?=p�                                    Bxi��  �          Ap�A\)?�R?�Q�@�
=@h��A\)>��
?�ff@�z�?��                                    Bxi��  �          A�A(�=L��?�Q�A��>��RA(���33?�33AC���                                    Bxi��Z  �          A"=qA=q��ff@�A>{C��{A=q�n{?�A.�RC�P�                                    Bxi�   �          A#33A��@  ?�33A+�C��RA���
=?�z�AffC��H                                    Bxi��  �          A"�HA   �Q�?У�A(�C��A   ��Q�?��@��
C��)                                    Bxi�)L  �          A#\)A\)��G�?ǮAG�C���A\)��?�z�@���C��H                                    Bxi�7�  �          A&=qA$(���?�\)@�(�C���A$(���?�ff@�\)C���                                    Bxi�F�  �          A&ffA$Q�<��
?���A ��=�Q�A$Q쾨��?�z�@�=qC�3                                    Bxi�U>  �          A(Q�A&ff>��?�p�AG�?�Q�A&ff����?�  A
=C���                                    Bxi�c�  �          A'�A%p��#�
?���A	�C��\A%p���?�p�A�\C���                                    Bxi�r�  �          A)�A&�H>���?��H@�{?�A&�H��?��RAffC���                                    Bxi��0  �          A&�\A&{?��?��@B�\@AG�A&{>���?+�@k�@��                                    Bxi���  �          A(��A(  ?�\?��@@  @333A(  >�p�?(��@e�@                                       Bxi��|  �          A'�A%��?u?z�H@���@�G�A%��?0��?�
=@�Q�@u                                    Bxi��"  T          A'�A"=q@(�?�
=@�
=AC33A"=q?�?�A\)A$��                                    Bxi���  �          A*ffA$Q�@$z�?�{@�  A`z�A$Q�@{?ٙ�A�HAC33                                    Bxi��n  �          A.ffA'�@,��?��\@ָRAg�A'�@z�?��A�
AG\)                                    Bxi��  �          A*�\A%@?�@���A6�\A%?�Q�?��A"�RA�
                                    Bxi��  �          A'�A"�\@(�?�p�@�  AB�\A"�\?�=q?�(�A�A#
=                                    Bxi��`  �          A)��A&=q?�33?�@��@�{A&=q?��\?�(�A�@��\                                    Bxi�  �          A%p�A#�?L��?�p�@ڏ\@�  A#�>�?��@�
=@,��                                    Bxi��  �          A$Q�A"�R?Q�?��@��
@��HA"�R?�\?�ff@陚@9��                                    Bxi�"R  �          A%p�A$  ?z�?�@Ϯ@O\)A$  >�\)?��\@�\?�ff                                    Bxi�0�  �          A%p�A$Q�>�z�?�33@˅?���A$Q�    ?�
=@��C���                                    Bxi�?�  �          A"�\A!p�>��R?��@�p�?��A!p�=#�
?���@�p�>aG�                                    Bxi�ND  �          A$(�A"�R=�G�?��@�\?!G�A"�R�W
=?�ff@�Q�C�ff                                    Bxi�\�  �          A%G�A$(�>W
=?�{@�z�?�Q�A$(��u?�\)@�  C���                                    Bxi�k�  �          A&�HA%>�  ?��@ȣ�?���A%�#�
?�z�@��C��                                    Bxi�z6  �          A$��A#�
>L��?���@�G�?���A#�
���
?��@�(�C��f                                    Bxi���  �          A%p�A$(�>��H?��@���@0��A$(�>W
=?�
=@�G�?�
=                                    Bxi���  �          A$��A$  >\)?���@�{?B�\A$  ��?���@�{C���                                    Bxi��(  �          A$  A#
=�8Q�?fff@��C�|)A#
=����?Tz�@�z�C��                                     Bxi���  �          A%��A$z�=���?�z�@�ff?�A$z�B�\?�33@�z�C�y�                                    Bxi��t  �          A%��A$Q�Ǯ?��@�33C��fA$Q�+�?�G�@��C�%                                    Bxi��  �          A"=qA Q쿈��?^�R@�C��3A Q쿠  ?z�@QG�C�o\                                    Bxi���  �          A�A녿aG�?�  @�  C�u�A녿���?@  @��C��3                                    Bxi��f  �          A{A�Ϳ��\?#�
@n{C��A�Ϳ�33>�p�@Q�C��=                                    Bxi��  �          A z�A�H��p�>�ff@%C�xRA�H��ff>�?@  C�B�                                    Bxi��  �          A"=qA zΐ33?J=q@�  C��RA zῨ��>�@.�RC�C�                                    Bxi�X  �          A!�A��n{?�ff@�  C�S3A���?J=q@�
=C���                                    Bxi�)�  �          A!A�
�z�H?�
=@�\)C�0�A�
��  ?fff@��C�o\                                    Bxi�8�  �          A"�\A (��\(�?�z�@�\)C���A (���Q�?�33@ϮC���                                    Bxi�GJ  �          A&�\A"�R�
=?��HA.�\C�XRA"�R���?�  A�C��                                    Bxi�U�  �          A)p�A(Q�B�\?aG�@�G�C���A(Q�s33?+�@fffC�h�                                    Bxi�d�  �          A)�A(�׿��?��\@�  C���A(�׿J=q?Y��@��\C�޸                                    Bxi�s<  �          A)p�A  ���H@�(�A��
C�~�A  ��G�@��HAٮC���                                    Bxi���  �          A.=qA
�\��Q�@�33B33C�˅A
�\�G�@ə�B(�C�n                                    Bxi���  �          A/�A	������@��BQ�C���A	����\@�  BQ�C�S3                                    Bxi��.  �          A0  A (���@���B,��C���A (���@�  B%�C��H                                    Bxi���  �          A/�
A z��\@�\)B+��C�1�A z����@�33B!33C��
                                    Bxi��z  T          A/�
A
=�Tz�@߮B�HC�1�A
=�%�@�G�B=qC��H                                    Bxi��   T          A0Q�A{�   @�z�B!�C�H�A{�33@أ�B��C�O\                                    Bxi���  �          A/�Aff�(��@��B'(�C���Aff� ��@���B�
C�q�                                    Bxi��l  �          A0z�A (���R@�B,�
C���A (��!�@�z�B!ffC�<)                                    Bxi��  �          A0��@�녿5@���B3{C�ff@���+�@�\B&�C���                                    Bxi��  �          A0z�@��>\)@�Q�B3=q?�ff@�녿���@�G�B,C�S3                                    Bxi�^  �          A1�@�(�>aG�A (�B9�\?�Q�@�(���@���B3G�C�5�                                    Bxi�#  �          A1�@��=�A�B?�?p��@���   @�\)B8p�C�q�                                    Bxi�1�  �          A0Q�Ap�<�@�
=B*�
>B�\Ap���33@�
=B#��C�c�                                    Bxi�@P  
�          A0Q�A�ý�\)@�B
=C���A�ÿ�{@�BQ�C��                                     Bxi�N�  
�          A0  @��#�
AffB?��C�c�@����@��\B5�C�w
                                    Bxi�]�  
(          A0  @�(����A��BD��C��)@�(��.�R@��B7\)C���                                    Bxi�lB  
(          A0Q�@�(�>.{A	��BMG�?���@�(��z�A��BE(�C��f                                    Bxi�z�  	�          A0��@��þ�=qA=qBV�RC��\@����%�AQ�BJ  C�=q                                    Bxi���  T          A0��@��p��A��B^�C���@��Q�A��BJC�                                    Bxi��4  
(          A1�@׮��(�A
ffBNffC�ٚ@׮�\(�A z�B:�RC�|)                                    Bxi���  
�          A2=q@\��G�AG�BcffC��@\�7
=A=qBS�C�b�                                    Bxi���  �          A333@�=q�W
=A��BP�RC��@�=q� ��A�HBD�C��f                                    Bxi��&  �          A4z�@�33��  A�BC=qC�)@�33�L(�@��B2��C�B�                                    Bxi���  �          A5�@�Q�fffA{BM�C�W
@�Q��L��AG�B<�C��R                                    Bxi��r  
�          A8  @�Q�k�A
{BC  C���@�Q��J=qAp�B3{C��R                                    Bxi��  
�          A8��@��ÿУ�A��BF�RC���@����x��A ��B1�C��\                                    Bxi���  �          A9@����33A��B6ffC�˅@������@���B=qC�AH                                    Bxi�d  �          A9p�@��H���A�B_  C�n@��H��G�A��BCz�C�k�                                    Bxi�
  
�          A8��@{��\)A3�B�\C��{@{���A(z�B��{C�P�                                    Bxi�*�  T          A9@xQ쿫�A,��B�Q�C�w
@xQ����A ��Bq�HC���                                    Bxi�9V  �          A9�@?\)��{A2=qB��C��q@?\)����A%�B}�C�y�                                    Bxi�G�  �          A9p�@X���ffA,��B�=qC��@X����z�A��BgQ�C���                                    Bxi�V�  �          A:ff@����ffA&=qB}\)C��@�������A\)BY�C�k�                                    Bxi�eH  �          A9@r�\�1G�A((�B��)C��=@r�\��\)A=qBZz�C�T{                                    Bxi�s�  �          A;�@����p�A!�BmG�C�(�@�����=qABLQ�C��                                     Bxi���  �          A;�@�z��   A�B\�RC�L�@�z���Q�A�B@��C�b�                                    Bxi��:  �          A:ff@�\)�33A#�
By  C��@�\)��\)A��BVQ�C�=q                                    Bxi���  �          A:�\@���
�HAz�BJ�HC�p�@������AB0(�C��=                                    Bxi���  �          A;
=@�p����A�BY�\C���@�p���Az�B<(�C��{                                    Bxi��,  �          A:{@�\)�I��A{BN�C��@�\)��  @�{B,33C���                                    Bxi���  �          A:�H@�G����RAp�B\�HC���@�G��r�\A=qBFz�C��{                                    Bxi��x  �          A9�@��
��A  Bd�
C�s3@��
����A�
BK�C�Z�                                    Bxi��  �          A9G�@���3�
A
=Bdp�C��3@�����HA��B?��C�33                                    Bxi���  �          A8��@����(��A�B\��C�4{@������
AffB:��C��q                                    Bxi�j  �          A8��@�33�\)Ap�Bi�C�)@�33���HA��BHffC��
                                    Bxi�  �          A9@����RA��BF�RC�J=@��vffA��B133C�1�                                    Bxi�#�  �          A:{@�녿�  A  BC�C�9�@���W
=A=qB2�C�                                    Bxi�2\  G          A9��@�
=���AQ�BDz�C��f@�
=�l��A ��B0(�C��3                                    Bxi�A  
�          A8��@陚��=qA��BH�C��R@陚�mp�AffB3\)C���                                    Bxi�O�  �          A7�
@��!G�A�RBp33C�Ф@��UAB[ffC���                                    Bxi�^N  �          A7�@�G��@  A (�Bs=qC�+�@�G��^�RA�RB\��C��                                    Bxi�l�  �          A6�R@�  �=p�A  BjffC�XR@�  �UA�HBU=qC���                                    Bxi�{�  �          A6�RA
ff�7�@ڏ\B�C�ФA
ff����@���A�C��)                                    Bxi��@  �          A7\)@�녿�33A�B533C�&f@�����\@��B(�C�1�                                    Bxi���  
�          A7
=@�p���\)A=qB=
=C��
@�p��Z=q@�  B*��C��                                    Bxi���  
�          A6ffA���u@�p�B)�C���A���C�
@�33B\)C��                                    Bxi��2  T          A5A{�\@�p�B0�HC��A{�$z�@�  B%\)C�9�                                    Bxi���  	�          A4(�@Ϯ�z�A�HBZ�\C�n@Ϯ�HQ�A
ffBH�HC�!H                                    Bxi��~  
�          A3�@�녿�33A�RB[\)C���@���z�HA�\BBG�C�3                                    Bxi��$  �          A2�H@�=q�fffA��BO�RC�@ @�=q�U�A�HB<�\C���                                    Bxi���  	`          A333@�G���{A��Bb�HC��=@�G��l��A
�\BK  C�C�                                    Bxi��p  	�          A3
=@\��ffA��BbG�C��@\�h��A
�RBK  C��=                                    Bxi�  �          A2ff@�녿��A33Bh�C��R@���k�AQ�BO��C��\                                    Bxi��  	�          A1�@�녿�Q�A\)B`��C�t{@���o\)A  BH{C�%                                    Bxi�+b  �          A2{@�\)��  A33BV=qC���@�\)�o\)A�B>C�                                    Bxi�:  
�          A1��@�(���Q�A�
BXC��
@�(��l(�Az�BA\)C���                                    Bxi�H�  
(          A1�@�
=��{A�\BV�C�#�@�
=�fffA�B@{C�w
                                    Bxi�WT  �          A/�
@�{�c�
Az�B]�C��@�{�[�AffBG�C��                                    Bxi�e�  
�          A/�@ƸR��A\)B[G�C��@ƸR�l(�A(�BCQ�C���                                    Bxi�t�  
(          A/\)@�
=>uA�Bi��@
=@�
=�(�A  B]p�C�q�                                    Bxi��F  �          A.�R@�{���HA{Bcz�C���@�{�E�A	BP�\C�G�                                    Bxi���  
(          A.�H@���\)A\)Bf{C�P�@���1�A��BV33C�P�                                    Bxi���  
�          A/
=@�33=�G�A�Bf��?�ff@�33�#33A�BY�C�4{                                    Bxi��8  �          A.�R@�33�^�RA
=BJ�C�]q@�33�QG�@��\B7��C�<)                                    Bxi���  �          A-�@�G���ff@�\B*
=C�� @�G��u@�  B33C��H                                    Bxi�̄  �          A-G�@߮����Ap�BBffC�%@߮�n�R@�\B+�C��{                                    Bxi��*  �          A-�@�\)��=qA��BB�C��)@�\)�w
=@�G�B)�RC���                                    Bxi���  �          A-G�@����R@�p�B533C�K�@��Z�H@�\)B!(�C��q                                    Bxi��v  �          A,Q�@��H��ffA
=BFz�C���@��H�hQ�@�RB/��C��                                    Bxi�  �          A,��@��׿\@�=qB+  C�s3@����e@љ�B��C��R                                    Bxi��  �          A+33@˅<�A��BU�\>k�@˅��RA
=BH�
C�\)                                    Bxi�$h  �          A*�H@�=q?�  A  Bc��AJ�H@�=q��\)A�
Bb�C�)                                    Bxi�3  
�          A+�@�ff>W
=A�BZ33?�p�@�ff�ffAffBNC��q                                    Bxi�A�  �          A+�
@�33�#�
A�
BI  C�O\@�33�%�@�=qB;�C��\                                    Bxi�PZ  �          A+�@�z��@��B1��C�33@�z��*�H@�G�B$�C�Z�                                    Bxi�_   �          A,(�@�Q�L��@�(�B533C���@�Q��@��@ᙚB$Q�C��                                    Bxi�m�  �          A+33@�p���G�@�z�B!Q�C���@�p��n{@��B
�
C�k�                                    Bxi�|L  �          A*ff@߮�z�@�\)BB��C��)@߮�:�H@�ffB2�C��f                                    Bxi���  
�          A*�\@��H��z�@�(�B/z�C��f@��H�S33@�ffB
=C�@                                     Bxi���  
�          A+\)@�
=����A ��BC33C��H@�
=�Z�H@��
B.{C��                                    Bxi��>  �          A*�\@�(���{A\)BJ{C�.@�(��p  @�B1=qC�@                                     Bxi���  
�          A)�@��R��Q�A	BX��C��@��R��@��RB:z�C�z�                                    Bxi�Ŋ  �          A*ff@ҏ\��33A�
BL  C��@ҏ\�c�
@��B4�C�Ǯ                                    Bxi��0  �          A*{@�  �B�\Az�BV33C���@�  �Q�@��BA  C�*=                                    Bxi���  �          A*�\@�ff�ǮA
{BY
=C�.@�ff�>{ABF��C�5�                                    Bxi��|  
�          A*=q@��\>W
=Az�Bi
=@��@��\� ��A
�HB[33C��                                    Bxi� "  T          A*�R@�33<#�
A�
B\\)=���@�33�'�AG�BM�HC�]q                                    Bxi��  �          A+33@��
���\@���B>C��@��
�W
=@�\)B*33C�aH                                    Bxi�n  �          A+\)@��Ϳ�@�p�B?z�C���@����9��@�z�B/�C���                                    Bxi�,  �          A+�@�z���A�RBG33C��)@�z��@  @��
B5�RC�<)                                    Bxi�:�  �          A+\)@�(����@��RB@\)C�)@�(��7�@�{B0Q�C��                                    Bxi�I`  �          A*�H@�녽�A�BI�\C�y�@���'�@���B;��C�w
                                    Bxi�X  �          A,Q�@�
=�k�@��B?�\C�
@�
=�)��@�G�B1�RC��                                    Bxi�f�  �          A+�
@׮�=p�A�BK=qC��)@׮�P  @�ffB733C�#�                                    Bxi�uR  �          A+
=@�G��Tz�@�
=BAffC���@�G��N�R@��HB-�RC���                                    Bxi���  
�          A*=q@��\A ��BD��C�p�@��5�@�G�B4��C���                                    Bxi���  �          A)�@���\)@�=qB=C�޸@��*�H@�B/\)C���                                    Bxi��D  �          A)�@��ÿh��@�33B8�C�t{@����L��@޸RB$�RC�                                      Bxi���  �          A((�@�Q�0��@�=qB8Q�C�C�@�Q��@  @߮B&�C�Ǯ                                    Bxi���  �          A'�@��
�h��@��
B3�C�w
@��
�I��@�
=B 
=C�o\                                    Bxi��6  �          A(��@�=q�&ff@�G�B6�C�w
@�=q�<��@�
=B%�RC��                                    Bxi���  �          A)�@���@�(�B7C�"�@��4z�@�33B(  C���                                    Bxi��  �          A*=q@��׾��@�  B+��C�@����   @��BQ�C�R                                    Bxi��(  �          A)��A�:�H@θRBz�C��HA�-p�@��B�
C�f                                    Bxi��  �          A(��A   ����@�G�B =qC�3A   �J�H@ÅBz�C�33                                    Bxi�t  
�          A(��@�\)��ff@׮B�C�` @�\)�U@��B
C��                                    Bxi�%  �          A(��AQ쿁G�@�z�B\)C��HAQ��8Q�@�Q�A�{C��=                                    Bxi�3�  �          A(Q�A(��L��@�33A���C�w
A(��(�@��\Aڣ�C�n                                    Bxi�Bf  �          A(Q�A�\���@���A��C���A�\� ��@�Q�A�\)C���                                    Bxi�Q  T          A'�A
=�=p�@�ffA�C��\A
=�G�@��RA��
C�3                                    Bxi�_�  �          A'33A  �.{@���A�(�C��fA  �
=q@�=qA�C�q�                                    Bxi�nX  �          A&�RA�;\)@�z�A�G�C���A�Ϳ�Q�@xQ�A���C��3                                    Bxi�|�  �          A%p�AQ�>��@�G�A�{?n{AQ쿑�@xQ�A�ffC���                                    Bxi���  �          A%G�A{>�
=@_\)A�p�@\)A{�5@\��A��C��H                                    Bxi��J  �          A%�A��>W
=@|(�A�
=?�  A�ÿ�ff@s33A�z�C�ٚ                                    Bxi���  �          A%p�Aff=�G�@��A�
=?&ffAff���\@��
A�C�%                                    Bxi���  �          A$(�Aff���R@�(�A�z�C��Aff��\@�G�AŅC��                                    Bxi��<  �          A#�AQ쾨��@���A��C���AQ��
=@|(�A��C��)                                    Bxi���  �          A$z�A�H�L��@��A�(�C�aHA�H��  @q�A���C�~�                                    Bxi��  �          A#�
A녾#�
@��
A�C���A녿�(�@vffA���C���                                    Bxi��.  �          A#�A\)=�G�@w
=A��?(��A\)���@l(�A�G�C���                                    Bxi� �  T          A"ffA    @z=qA�\)C���A��G�@l��A�33C�(�                                    Bxi�z  �          A!�A�H��\)@`  A��RC���A�H��Q�@R�\A���C�e                                    Bxi�   �          A ��A=q�u@a�A���C�FfA=q��\)@P��A��C��)                                    Bxi�,�  �          A!��A�\��G�@fffA�G�C���A�\��=q@P��A�
=C�<)                                    Bxi�;l  �          AffA�B�\@L(�A���C�l�A��(�@=p�A�33C�L�                                    Bxi�J  �          AG�A33>Ǯ@,(�A~{@Q�A33��\@*�HA|Q�C�xR                                    Bxi�X�  �          AA(�?.{@!G�Alz�@��A(��8Q�@&ffAt��C�xR                                    Bxi�g^  �          A��A�?
=@   Ak�
@c�
A����@#33AqG�C�7
                                    Bxi�v  �          Ap�A�?�@&ffAu�@N{A���33@(Q�Ax(�C��3                                    Bxi���  �          Ap�A  ?@  @(�Ae�@�  A  ��Q�@#33Apz�C��
                                    Bxi��P  �          A��Az�?�@(�AN�\@Z�HAz�8Q�@  AT��C�t{                                    Bxi���  �          Ap�A�H>�  ?�A(�?��RA�H���R?�z�A33C�{                                    Bxi���  �          AQ�Ap��#�
?���A+�C���Ap��(�?��HA!��C�,�                                    Bxi��B  T          A��A�׾��H@	��AJ�\C���A�׿�?�{A/�C��H                                    Bxi���  �          AQ�AG����H?�A-C���AG�����?���AQ�C��                                    Bxi�܎  �          A�A�5@Q�Ad(�C���A���H@ ��A@(�C���                                    Bxi��4  �          A  A�>L��@@  A��?�p�A��O\)@8��A��RC�}q                                    Bxi���  �          A��AQ�?�
=@(�AM@��AQ�>\@p�Ag�@�\                                    Bxi��  �          AAff?L��?�=qA+
=@���Aff>��?��RA:�\?aG�                                    Bxi�&  �          A��A>��?�33A2ff@{A�u?�
=A5G�C�Ff                                    Bxi�%�  �          A��A=q?(��?�{A�@z=qA=q=��
?޸RA#�>�                                    Bxi�4r  �          A  A?��?ǮA�H@R�\A    ?�33A  C��q                                    Bxi�C  �          A��A
=>�
=?�ff@�(�@{A
=�#�
?�{A (�C�޸                                    Bxi�Q�  �          A�A
=>.{>�33@�?��\A
==#�
>Ǯ@33>��                                    Bxi�`d  �          AA�Ϳ^�R�Ǯ�z�C�c�A�Ϳ0�׿(���|(�C��\                                    Bxi�o
  �          A�A���;��L��C�ФA���þ���ǮC�                                    Bxi�}�  T          AAp���\��G��(��C�}qAp���G���\)���C���                                    Bxi��V  �          A�AG�����(��#�
C�o\AG���33�z��^{C��
                                    Bxi���  �          AG�A�þ�33��=q���C���A�þk��\�  C�L�                                    Bxi���  �          A�A��>8Q�=�?@  ?�=qA��>�>8Q�?���?B�\                                    Bxi��H  �          A�
A�>��R=�\)>���?�{A�>�=q>.{?�G�?�{                                    Bxi���  ~          A\)A33?�\=L��>���@G�A33>��>k�?���@4z�                                    Bxi�Ք  x          A(�A�
=�\)>�
=@!G�>���A�
���
>��@   C�                                    Bxi��:  �          A�A\)=#�
>�Q�@�>��A\)���
>�33@��C���                                    Bxi���  �          A33A�R>��R?=p�@�\)?�{A�R<�?L��@�33>L��                                    Bxi��  �          A\)A�H���
?@  @�=qC��HA�H��{?.{@��
C���                                    Bxi�,  �          A
=A�\���
?(�@n�RC��{A�\�k�?�@^{C�O\                                    Bxi��  �          A�RA��z�>�G�@+�C��A��(�>��R?�33C���                                    Bxi�-x  �          A�
A��u�u��33C�FfA��W
=�\)�Tz�C�aH                                    Bxi�<  �          A=qA{<����B�\>B�\A{=�\)��G��#�
>�G�                                    Bxi�J�  �          A
=A�H>L��>��
?�
=?�Q�A�H=�\)>�p�@{>�G�                                    Bxi�Yj  �          A�\A{>\)>B�\?�
=?Y��A{=u>k�?�33>\                                    Bxi�h  �          A�HA����������{C���A��>����\)��=q?˅                                    Bxi�v�  �          A
=A��Q쿑���
=C���A>��
�����ָR?�Q�                                    Bxi��\  �          A\)A�>u�����?�(�A�?���h����Q�@h��                                    Bxi��  �          A��A(�>B�\=#�
>��?���A(�>#�
=�G�?#�
?xQ�                                    Bxi���  �          A�A
=>�(�=���?!G�@'
=A
=>�p�>�  ?�  @{                                    Bxi��N  �          A  A\)?�\�#�
���
@G
=A\)>��H>.{?��
@<(�                                    Bxi���  �          A��A\)?J=q?Y��@�(�@�G�A\)>�G�?�=q@Ϯ@*�H                                    Bxi�Κ  �          Az�A��?�(�?��A�R@��A��?#�
?�p�A'�
@|(�                                    Bxi��@  �          Az�A=q?�\)?���@���@ٙ�A=q?(��?�
=A
{@�                                      Bxi���  �          A(�A�>aG�?0��@�p�?�{A��#�
?:�H@�(�C��                                    Bxi���  �          A
=A�R�W
=>\)?Tz�C�]qA�R�u=L��>���C�B�                                    Bxi�	2  �          A�\A{���;��
��(�C��A{�����ff�.{C�4{                                    Bxi��  �          A�HA�\�Ǯ���5�C���A�\�L�Ϳ��^�RC�ff                                    Bxi�&~  �          A�
A\)��ff�   �B�\C��A\)�u�!G��s�
C�G�                                    Bxi�5$  �          A  A�����333��  C�5�A�<#�
�@  ��G�=L��                                    Bxi�C�  �          A�
A�\    ��=q����C��qA�\>\��G���33@                                    Bxi�Rp  �          A�A�H�!G�����i��C�{A�H��p��J=q��G�C��q                                    Bxi�a  �          AQ�A����H��R�n{C��fA��k��@  ��G�C�K�                                    Bxi�o�  �          A�A33�����
=�c33C��A33��\)�&ff�}p�C���                                    Bxi�~b  �          A�
A�H?.{�0����p�@�z�A�H?c�
�������@��
                                    Bxi��  �          A33A=q>��H�G���\)@@  A=q?=p�����Tz�@�Q�                                    Bxi���  �          A33A�>���=q���H@:�HA�?W
=�W
=���\@��                                    Bxi��T  T          A33A?333�����(�@�G�A?�ff�B�\���
@��                                    Bxi���  �          AG�A  <����H��p�>W
=A  >�׿�\)���@@��                                    Bxi�Ǡ  �          A�
A�\>B�\������=q?�33A�\?
=�z�H��@dz�                                    Bxi��F  �          A�A�>�{��ff��=q@�\A�?J=q�������@���                                    Bxi���  �          A��A\)>k���=q�   ?�\)A\)?333��z���{@�
=                                    Bxi��  �          Az�A\)>B�\������G�?�z�A\)?
=�z�H��z�@e�                                    Bxi�8  �          A\)A�þ��H���{C�u�A��=�\)��  �=q>�(�                                    Bxi��  �          A�\AQ���H���\)C�g�AQ�=�\)��  ��>�                                    Bxi��  �          A�A(��L�Ϳ����
=C���A(�>���ff��R@AG�                                    Bxi�.*  �          AA\)�����Q���HC���A\)���
�Ǯ�,  C���                                    Bxi�<�  H          Ap�A z���H������\C�&fA zῴz��  �Ap�C��                                    Bxi�Kv  `          A�A�R���Ϳ����%��C���A�R�����	���c
=C�(�                                    Bxi�Z  �          A�A(�����Ǯ�*{C�k�A(��\��=q�HQ�C���                                    Bxi�h�  �          A��A�H�������C�t{A�H=�\)��(���>�                                    Bxi�wh  �          A�A\)�h�ÿ��R��\C�
A\)��z��(��.=qC��                                    Bxi��  �          A�A
�\��G����R��HC��)A
�\�(������@��C��)                                    Bxi���  �          AQ�A�Ϳ�p���G�� ��C�5�A�Ϳfff��p��0��C��                                    Bxi��Z  �          Ap�A	녿��ÿ�Q��ffC���A	녿.{��=q�?�C���                                    Bxi��   �          A��A
�\���\��(����C���A
�\���H�\��RC�e                                    Bxi���  �          A�A
ff��������RC�h�A
ff����{�'�
C�C�                                    Bxi��L  �          A��A
ff�xQ쿥��33C��\A
ff��녿Ǯ�#�C��f                                    Bxi���  �          A��A	G�����������C�aHA	G����H��Q��2ffC�aH                                    Bxi��  �          AA
=�{�33�r{C�nA
=��
=�=p�����C���                                    Bxi��>  �          Az�A
=q?L�ͽ�G��333@��A
=q?G�>W
=?�{@�                                    Bxi�	�  T          A�A
ff>B�\��G��8Q�?�p�A
ff>�{�������@�R                                    Bxi��  �          A
ffA\)��33������C�H�A\)�fff�\�!C���                                    Bxi�'0  �          @��R@׮�W
=�Ǯ�;
=C��H@׮� ���.{��C��f                                    Bxi�5�  _          @�=q@�33�S33��\�Q�C�(�@�33���9����(�C�u�                                    Bxi�D|  �          @��H@��*�H���H�-G�C�Ф@���33�����C��3                                    Bxi�S"  �          @�@���
�H��
=�	G�C��\@�׿Ǯ���`��C�&f                                    Bxi�a�  �          A ��@���   �����\C���@�������p��dQ�C��                                    Bxi�pn  �          AA (�����Q���\C��{A (���  ��z��S33C��\                                    Bxi�  �          @��@�z῵��z��  C���@�z�\(���\)�>=qC��                                    Bxi���  �          @�(�@�  �0�׿����C�C�@�  �B�\����(  C�>�                                    Bxi��`  T          AG�@�p��xQ쿗
=��RC�� @�p���G����H�'\)C�g�                                    Bxi��  �          A{A��Q녿������C�&fA��k������,z�C�/\                                    Bxi���  �          A�\A�þ\)��  �	G�C���A��>�33���H���@(�                                    Bxi��R  �          A��A녾������/
=C�` A�>B�\�У��5G�?��                                    Bxi���  �          A�RAz�z΅{���C�  Az�#�
��p��!��C���                                    Bxi��  �          A�A{���Ϳ�
=�U��C���A{?(�ÿ����H��@�z�                                    Bxi��D  �          A��A33��Q쿪=q�
=C���A33>�(���G��\)@AG�                                    Bxi��  �          A ��@��R��zῗ
=�33C��@��R>B�\�����	��?�\)                                    Bxi��  �          @�ff@��H���������
=C��\@��H����  ���C���                                    Bxi� 6  
�          @��@�(��}p��Tz���z�C�P�@�(��zΐ33���C�Ф                                    Bxi�.�  
�          @��@�(���ff�xQ���(�C��@�(��O\)��33�+
=C��)                                    Bxi�=�  �          @�  @�녿�ff��{�Q�C���@�녿B�\���
�<��C��                                    Bxi�L(  �          @�ff@�  ��׿��H�T��C�y�@�  ����z�����C��
                                    Bxi�Z�  �          @�Q�@�
=��녿�  �C��q@�
=��G���ff�]��C�                                    Bxi�it  �          @�G�@�(���Q쿾�R�6�\C�e@�(���Q�������C�AH                                    Bxi�x  �          @�  @��� �׿���=G�C��@�녿��R�p�����C�f                                    Bxi���  �          @�@ᙚ�����$��C���@ᙚ����33�
=C�p�                                    Bxi��f  �          @�Q�@�������i��C�G�@���  �&ff��(�C��f                                    Bxi��  �          @�Q�@�{�$z��\)���RC�~�@�{��(��Dz���p�C�Ф                                    Bxi���  �          @�33@�ff�8���
�H���\C�T{@�ff����H����
=C���                                    Bxi��X  �          @���@��
�(Q��ff�>ffC��=@��
����{��p�C���                                    Bxi���  �          @��@�ff�+���
�~�\C��@�ff�У��<�����C�%                                    Bxi�ޤ  �          @�@�G��p��#�
���\C��@�G���  �N{���C���                                    Bxi��J  �          @�33@Å�;��5���33C�.@Å��ff�qG�����C�޸                                    Bxi���  �          @��H@�\)�*�H�4z�����C�g�@�\)����i����RC��                                    Bxi�
�  �          @�33@Ǯ�,���2�\���RC�Q�@Ǯ����h���홚C��                                    Bxi�<  �          @��
@���
�0����=qC�  @���  �]p���  C��{                                    Bxi�'�  �          @�(�@�ff�z��.�R���C�q@�ff���\�[����
C��H                                    Bxi�6�  �          @陚@�\)�(��<�����\C�K�@�\)����l(���Q�C�B�                                    Bxi�E.  T          @�\@ȣ��\)�C�
�ř�C�.@ȣ׿O\)�mp���
=C�U�                                    Bxi�S�  T          @��@�Q���
��p��7
=C���@�Q쿣�
�(���ffC���                                    Bxi�bz  �          @��\@�Q���þ���\(�C��@�Q��\��ff���C�`                                     Bxi�q   �          @�=q@�z���������C��@�z�Ǯ��{�dQ�C�ٚ                                    Bxi��  �          @�\@�p�����\�8Q�C���@�p���\�����!�C��                                    Bxi��l  �          @�Q�@�33���u��G�C��\@�33�(�������C�j=                                    Bxi��  �          @�  @�\)�%��W
=��ffC�޸@�\)� �׿����a�C���                                    Bxi���  �          @�@��&ff��(��XQ�C�]q@����(�����C��q                                    Bxi��^  �          @�33@Å�<���2�\���RC�  @Å��ff�p  ��  C��f                                    Bxi��  �          @�@�z��O\)�p���  C��@�z�����dz���(�C�4{                                    Bxi�ת  �          @陚@θR�7������j=qC�f@θR��{�7
=���\C���                                    Bxi��P  
�          @陚@�ff�;���G��^�HC�Ǯ@�ff�����3�
����C��R                                    Bxi���  �          @�=q@Ϯ�C33��(��8��C�g�@Ϯ���%��{C���                                    Bxi��  
�          @�G�@�{�G����3�
C��@�{����%����C�W
                                    Bxi�B  �          @�33@��?\)����n�\C��=@���Q��<����  C���                                    Bxi� �  �          @��H@�p��.�R��G��
=C�� @�p����H����G�C��\                                    Bxi�/�  �          @��H@أ��)���s33��C�Q�@أ��G����H�x(�C���                                    Bxi�>4  �          @�@�33�&ff���
��C��H@�33�ff����	�C��                                    Bxi�L�  �          @�z�@�ff��
?}p�@�\)C���@�ff�!G��u��C��                                    Bxi�[�  �          @��
@�(���?�
=A33C���@�(��(��=�Q�?(��C�~�                                    Bxi�j&  �          @�33@ۅ���?�33A0  C�  @ۅ�%�>�Q�@1�C���                                    Bxi�x�  �          @��@ۅ�(�?!G�@��RC�0�@ۅ�\)��ff�`��C�
=                                    Bxi��r  �          @�  @�
=�+�>�
=@VffC�!H@�
=�'
=�=p����\C�aH                                    Bxi��  �          @���@�����?fff@�=qC�O\@���#33�B�\��  C��)                                    Bxi���  �          @��@��H�0��>�z�@C���@��H�(Q�c�
��p�C�!H                                    Bxi��d  �          @��
@�G��1논#�
��G�C�y�@�G��!G���
=�G�C�w
                                    Bxi��
  T          @�@�  �5�������C�/\@�  ��Ϳ�(��>�\C���                                    Bxi�а  �          @��H@˅�?\)�Y�����C�ff@˅�ff�����C��
                                    Bxi��V  �          @�@�\)�3�
�:�H��{C�H�@�\)��R�����qC�z�                                    Bxi���  �          @��@���4z���H�z�HC�Z�@���ff����S�C�"�                                    Bxi���  �          @陚@�z��>�R����P  C��@�z��!녿���O�C��{                                    Bxi�H  �          @�G�@���E�����33C�h�@���#�
���f=qC�W
                                    Bxi��  �          @ٙ�@�33�S33�p����\)C�J=@�33�%�  ��33C��                                    Bxi�(�  �          @��@ȣ��%�����4Q�C���@ȣ׿�\�z���
=C�                                      Bxi�7:  �          @޸R@�z��	����  �h��C��
@�zῙ���   ��{C���                                    Bxi�E�  �          @ᙚ@�\)� �׿���w33C�e@�\)���\�#33��{C���                                    Bxi�T�  �          @���@˅�"�\��\����C�)@˅��
=�;����C���                                    Bxi�c,  �          @�33@�G����33��\)C�ٚ@�G���\)�Dz���p�C��3                                    Bxi�q�  �          @�{@�녿��
�޸R�`z�C���@�녿!G���R��ffC�Y�                                    Bxi��x  
�          @�
=@�����33�t(�C��R@��k��"�\����C��                                    Bxi��  T          @�R@�z��G���33�S�C��{@�zΎ��p���C�N                                    Bxi���  �          @�{@�=q���(��]G�C�4{@�=q��\)�#�
����C�R                                    Bxi��j  �          @�z�@�Q��1�����=qC��@�Q����A�����C���                                    Bxi��  �          @�@����'���33�3�C�\@��ÿ�G������C�p�                                    Bxi�ɶ  �          @�{@��H�"�\����+�C�w
@��H���H�33���C��
                                    Bxi��\  �          @�@�(��p�������C��{@�(��޸R�33��\)C���                                    Bxi��  �          @�@ָR�33���H�|��C���@ָR��{��Q��:�\C�AH                                    Bxi���  �          @��
@ָR�  �+���z�C��)@ָR�޸R��=q�M�C��q                                    Bxi�N  �          @���@�(��
=��  �=qC�.@�(���(����o
=C�                                    Bxi��  �          @ᙚ@�
=��녿n{��\C�!H@�
=��ff��z��Z{C���                                    Bxi�!�  T          @ᙚ@�녿޸R���q�C���@�녿�{���C�W
                                    Bxi�0@  T          @��@�
=��?
=q@�C��{@�
=�z�����N�RC�s3                                    Bxi�>�  �          @�\@��ff?n{@�=qC�K�@���\���
�!G�C���                                    Bxi�M�  �          @�33@ָR�G�>k�?�\)C��@ָR��ÿG��ʏ\C�*=                                    Bxi�\2  �          @�Q�@�  �#33?
=q@��RC�K�@�  �!녿!G���(�C�]q                                    Bxi�j�  �          @�\)@��
�?(��@��C�9�@��
�
�H�����+�C���                                    Bxi�y~  �          @�ff@��Ϳ�\)>��@W�C�(�@��Ϳ�{��ff�p  C�33                                    Bxi��$  �          @��
@���   >�  @�
C���@�녿�33�(����\)C��=                                    Bxi���  �          @ۅ@�녿�>���@   C�(�@�녿���
=q��=qC�e                                    Bxi��p  �          @�  @�ff��{?(�@�\)C��\@�ff��Q쾊=q��\C��)                                    Bxi��  �          @�\)@�  �˅?\)@���C�)@�  ��
=�L�Ϳ�(�C���                                    Bxi�¼  �          @׮@���33�����C��@��� �׿�\)��C�/\                                    Bxi��b  �          @׮@���*=q�ff����C���@�녿����C�
��ffC��)                                    Bxi��  �          @�{@�\)�*�H�3�
���C��@�\)��33�l����
C��                                    Bxi��  �          @�p�@��
���5���
C�b�@��
�333�`�����\C�w
                                    Bxi��T  �          @ָR@��H�6ff�}p��{C�3@��H�L����=q�7�
C��                                    Bxi��  �          @ָR@�(��0���%�����C�o\@�(������b�\�(�C��                                    Bxi��  �          @��@����/\)�����C��@��Ϳ�G��G���z�C�~�                                    Bxi�)F  �          @�p�@��\�������G�C�N@��\��\)�����8�C�G�                                    Bxi�7�  �          @��@����z��{��Q�C�@��=#�
���
�*?�                                    Bxi�F�  �          @�G�@������AG���{C�>�@��ÿ�\�hQ��	�C�9�                                    Bxi�U8  �          @���@��\�8Q��
=�o�
C�Z�@��\�����2�\��p�C��{                                    Bxi�c�  �          @��H@�(��3�
������C��)@�(����<����p�C�                                    Bxi�r�  �          @��H@���   �33���HC�7
@�������<����
=C��                                    Bxi��*  �          @��
@�=q�&ff����Q�C�u�@�=q���
�P  ��\)C��f                                    Bxi���  
�          @�33@�����6ff��
=C��@���c�
�hQ��{C�W
                                    Bxi��v  
�          @Ӆ@�{�!��-p���=qC��f@�{����c�
���C��3                                    Bxi��  �          @�z�@�
=�p��A��ٮC��@�
=����l����RC��R                                    Bxi���  �          @�p�@�=q����c33�ffC�Y�@�=q�
=q��  �Q�C���                                    Bxi��h  �          @�p�@������k��Q�C��\@��=u���H��H?+�                                    Bxi��  �          @��H@��׿��tz��Q�C��=@���=�Q���  � �H?xQ�                                    Bxi��  �          @љ�@�33�
=�`  ��\C��H@�33�������\�ffC�T{                                    Bxi��Z  �          @�{@�ff��ff�q��	��C�y�@�ff=�G����33?��R                                    Bxi�   �          @��
@��\�����`����33C���@��\���|���33C��3                                    Bxi��  �          @љ�@������\����
=C��@����{��G���C��                                    Bxi�"L  �          @�Q�@�{���G���  C��@�{�(���u�(�C�e                                    Bxi�0�  �          @���@�ff�\)�Dz��㙚C�` @�ff�
=�p  �ffC��                                     Bxi�?�  �          @�(�@�ff�'
=�z���(�C���@�ff�����A����HC�.                                    Bxi�N>  T          @�33@�(��p��!����C�Ff@�(��O\)�P����z�C��                                    Bxi�\�  
�          @�  @�p��33������RC�^�@�p�����@���ԣ�C���                                    Bxi�k�  �          @߮@Å��
����p�C���@Å�p���N{��{C��H                                    Bxi�z0  �          @�  @�
=�33������C�� @�
=���
�@����(�C�K�                                    Bxi���  �          @��
@�p��*�H���
�33C�O\@�p���33������\C�q�                                    Bxi��|  �          @���@���Dz�\)���C�
=@���*=q���
�IC���                                    Bxi��"  �          @�@�33�J�H�&ff���HC�,�@�33�!G���\����C���                                    Bxi���  �          @��@�
=�Fff��p��5C��H@�
=�%���  �Z�RC��                                     Bxi��n  �          @�@�G��G�?Tz�@�ffC�Ф@�G��I���333��
=C��{                                    Bxi��  �          @��H@Ǯ�dz�?�=qA'�
C��@Ǯ�q녾���i��C�b�                                    Bxi��  �          @�(�@ƸR�o\)?�{A)�C�y�@ƸR�|(�����G�C��=                                    Bxi��`  �          @��@ə��hQ�?z�@��HC�@ə��_\)��z���\C���                                    Bxi��  �          @��@˅�^{>�@j=qC��3@˅�R�\��Q��\)C�T{                                    Bxi��  
�          @�\@����N{?�33A0z�C��f@����`  ��  ���RC��=                                    Bxi�R  �          @��H@�=q�\(�?��A$��C��q@�=q�i����(��X��C�                                      Bxi�)�  �          @�G�@\�l��?˅AIG�C�T{@\��Q쾞�R���C�Ff                                    Bxi�8�  �          @�33@�
=�Y��?G�@��HC�  @�
=�W��c�
��
=C�9�                                    Bxi�GD  �          @�@�
=�1�?n{@���C�@�
=�8�þ���k�C�Y�                                    Bxi�U�  
�          @��
@���   >�ff@`��C��@����H�E���\)C�U�                                    Bxi�d�  T          @�33@����R�\?�@��C���@����J=q�����=qC�R                                    Bxi�s6  
�          @陚@��\��<�>W
=C���@��C�
��=q�H��C�Ff                                    Bxi���  	.          @��H@��
�]p�?�\)A(�C�@��
�e��!G����C�XR                                    Bxi���  �          @�  @ə��b�\�����C�S3@ə��@�׿���q��C�9�                                    Bxi��(  �          @�Q�@���dz�aG���(�C�B�@���C33��{�n{C��                                    Bxi���  
�          @�\)@\�s�
�O\)�θRC��R@\�?\)�   ��=qC��                                    Bxi��t  �          @�Q�@��
�<��>���@��C��q@��
�0  ��\)�C���                                    Bxi��  
(          @��@��6ff>��@z�C�p�@��(�ÿ�{�(�C�5�                                    Bxi���  �          @��@Ϯ�P��=L��>�Q�C���@Ϯ�9�����R�=��C���                                    Bxi��f  	�          @�=q@љ��O\)>u?�33C��R@љ��>{����%�C���                                    Bxi��  �          @�@�  �h�ü��k�C��=@�  �L(��޸R�^�\C�z�                                    Bxi��  "          @�{@У��@  �����=qC���@У��   ��
=�Xz�C���                                    Bxi�X  "          @�@���'
=�\)��\)C�H�@����\��p��^�HC�|)                                    Bxi�"�  
�          @�ff@���<�;aG����
C��H@���   �˅�Lz�C��{                                    Bxi�1�  �          @�
=@�  �G
=�L�;���C�<)@�  �-p���G��B{C���                                    Bxi�@J  T          @�Q�@Ϯ�N�R>B�\?��RC��H@Ϯ�<(���{�-��C���                                    Bxi�N�  "          @�@���8�ÿxQ���z�C���@������R��(�C�                                    Bxi�]�  �          @ᙚ@ə��)�����
�i�C��)@ə����R�4z���=qC�\)                                    Bxi�l<  �          @�  @���1G���ff�n�RC��@����=q�9����  C��                                    Bxi�z�  
�          @�@\�HQ���H��  C�^�@\�����Mp����HC��f                                    Bxi���  �          @��
@�ff�vff��p��=p�C��\@�ff�Mp��	����\)C��{                                    Bxi��.  
Z          @�(�@�  �u�����C��{@�  �S�
��Q��|��C��                                    Bxi���  T          @�(�@�33�i�������=qC���@�33�:�H�G���Q�C�0�                                    Bxi��z  T          @���@�(��g���R��ffC���@�(��8Q��G���  C�ff                                    Bxi��   
�          @�{@�(��L(��W
=��{C���@�(����������RC���                                    Bxi���  "          @�\@�ff��Q��R�\�ָRC�N@�ff���s�
���HC�b�                                    Bxi��l  T          @陚@�\)����.�R��  C��@�\)��33�S33�ׅC�s3                                    Bxi��  
�          @�\@�G��{�Q���(�C��H@�G��O\)�I����z�C�z�                                    Bxi���  
�          @�p�@Ϯ��33�@  ��\)C��{@Ϯ�k��b�\����C���                                    Bxi�^  T          @�(�@ҏ\���#�
��
=C�j=@ҏ\�\)�N{��  C���                                    Bxi�  "          @��
@�=q�G��#33����C�t{@�=q����L���υC���                                    Bxi�*�  �          @�33@�(��	���W
=����C�P�@�(���=q�~�R�  C���                                    Bxi�9P  "          @��H@�
=��H�l����=qC��f@�
=��{����G�C�H�                                    Bxi�G�  
(          @��@�=q��z��K���\)C���@�=q>�z��^{���@)��                                    Bxi�V�  
P          @陚@�=q�33�O\)��  C��@�=q��ff�|(���C��                                     Bxi�eB  ^          @�G�@�\)����\)���C���@�\)=�\)�3�
��
=?(�                                    Bxi�s�  �          @�G�@�G������   ��  C�}q@�G����� �����C�Ff                                    Bxi���  �          @�33@�33�p��z�����C��=@�33��33�@  ����C�                                    Bxi��4  
�          @��@Ӆ���� ���~�HC�H@Ӆ�����:�H��=qC�"�                                    Bxi���  �          @�=q@���p��33��z�C��q@�녿�z��>�R���HC�H                                    Bxi���  J          @�@�{�G��
=q��=qC�\@�{�{���R���\C��H                                    Bxi��&  �          @�G�@�Q����(���ffC�&f@�Q�@  �;����\C��3                                    Bxi���  T          @�@�ff����=p���=qC���@�ff�W
=�`  ���C��                                    Bxi��r  
�          @��H@����
�B�\��(�C�E@����
�aG���C���                                    Bxi��  �          @��@����ff�AG���
=C�&f@����G��`����C��f                                    Bxi���  �          @�  @�ff��33�7����RC���@�ff�#�
�S�
��\)C���                                    Bxi�d  
�          @�R@\���^�R���HC�S3@\>u�vff�=q@p�                                    Bxi�
  �          @�@ȣ׿˅�QG���ffC��q@ȣ�>L���hQ���z�?�                                    Bxi�#�  T          @��@�ff�У��AG��Ù�C���@�ff=L���\(���z�>�G�                                    Bxi�2V  �          @�
=@ҏ\��p������C���@ҏ\���
�=p�����C���                                    Bxi�@�  �          @�ff@�\)��׿�(��]�C�5�@�\)�J=q��H��\)C���                                    Bxi�O�  �          @�33@�z��33�У��T  C�@�z�\(��ff���\C�N                                    Bxi�^H  T          @���@��H������(��AC��@��H�z�H�\)����C���                                    Bxi�l�  
�          @���@�z���׿��(�C�=q@�zῳ33�Q���{C��\                                    Bxi�{�  T          @�p�@��
�$z�z���33C���@��
���H��\�n{C�u�                                    Bxi��:  �          @�p�@θR�  �^�R��Q�C�b�@θR�����\)�{
=C�P�                                    Bxi���  "          @�p�@Ϯ��׿=p���(�C�e@Ϯ��{��  �k33C��                                    Bxi���  �          @��@�p���ÿQ����HC��@�p���
=����~=qC���                                    Bxi��,  
�          @�z�@�G�����~�RC�'�@�G���=q��Q��A��C�4{                                    Bxi���  T          @�z�@θR���aG�����C�3@θR������4(�C��R                                    Bxi��x  �          @ۅ@�\)��R>#�
?�ffC�}q@�\)�G��u�p�C�W
                                    Bxi��  �          @�z�@�  �{��G��h��C���@�  ���H��(��EC���                                    Bxi���  
�          @�(�@�
=�녾B�\��\)C�Ff@�
=��녿���.{C�ٚ                                    Bxi��j  
�          @ۅ@��H�%�=L��>�ffC��@��H�G���p��%�C�(�                                    Bxi�  �          @ڏ\@����R?:�H@ÅC�K�@��� �׿
=���C�'�                                    Bxi��  T          @�=q@ʏ\��?^�R@�\C���@ʏ\�\)��
=�`  C�E                                    Bxi�+\  �          @��H@ƸR�
=?���AY�C���@ƸR�5>k�?�33C��{                                    Bxi�:  �          @ۅ@�  �
�H?�AuG�C�q�@�  �1�?   @�ffC��                                    Bxi�H�  "          @��H@ə��z�?�\)A[
=C��@ə��&ff>\@K�C��                                    Bxi�WN  �          @�=q@ə��
=q?�A@Q�C���@ə��%�>#�
?���C��q                                    Bxi�e�  
�          @�=q@ƸR���?�z�A>�HC�w
@ƸR�1�<�>�=qC���                                    Bxi�t�  T          @�=q@�=q�?uA�C�ٚ@�=q� �׾����0��C�/\                                    Bxi��@  
�          @��@�=q�ff?\(�@�G�C��3@�=q�p���
=�aG�C�Z�                                    Bxi���  T          @ڏ\@�p��
=?k�@�Q�C���@�p���\��  �ffC�1�                                    Bxi���  
�          @ڏ\@�p��\)?#�
@�(�C�b�@�p���׿\)��ffC�L�                                    Bxi��2  "          @أ�@�����H?#�
@�
=C�y�@�����H�&ff��\)C�z�                                    Bxi���  �          @ٙ�@����?G�@ӅC��{@����R�\�Mp�C�ff                                    Bxi��~  
�          @�G�@У׿��    ���
C���@У׿У׿p����
=C���                                    Bxi��$  �          @�Q�@Ϯ��\)�W
=���C��{@Ϯ�\��\)�Q�C�l�                                    Bxi���  T          @أ�@�녿��ÿ   ��  C�@ @�녿�{��(��&ffC�/\                                    Bxi��p  T          @�Q�@ə��=q��
=�b�\C��@ə���׿����V�\C��\                                    Bxi�  �          @�  @����-p�����~{C��@�������G��r=qC���                                    Bxi��  
�          @�\)@ƸR�!G�����`  C���@ƸR��(���{�^�\C�/\                                    Bxi�$b  T          @�\)@���\)�!G����C�7
@�녿�\)���f=qC���                                    Bxi�3  
(          @أ�@ʏ\�(��Y�����C�q�@ʏ\��(����|��C�w
                                    Bxi�A�  �          @�G�@�  �ٙ��u��RC��\@�  �z�H��Q��e�C���                                    Bxi�PT  �          @ڏ\@�G���녿����  C���@�G��\(���\�pQ�C�<)                                    Bxi�^�  �          @��@��
���
��  �'\)C���@��
�0�׿����xz�C��                                    Bxi�m�  �          @�(�@�=q��zῷ
=�@(�C��@�=q�   ��������C��\                                    Bxi�|F  �          @ڏ\@�G���  ���
�-p�C���@�G��&ff��{�}G�C�(�                                    Bxi���  �          @�33@�녿�  ����-�C��3@�녿&ff��{�|��C�0�                                    Bxi���  �          @ۅ@��H��
=���R�&ffC��H@��H�(�����r{C�XR                                    Bxi��8  �          @�z�@ҏ\��{�����!�C�!H@ҏ\�G������yp�C���                                    Bxi���  �          @ۅ@�G��޸R���\���C���@�G��}p���G��n=qC��                                    Bxi�ń  T          @��H@�  ���
��33�=qC�U�@�  �u�����=qC�˅                                    Bxi��*  �          @ڏ\@�
=�\��G��L��C�ff@�
=����z���\)C��3                                    Bxi���  �          @��H@����녿����{C��@����  ����p�C��                                    Bxi��v  �          @ۅ@Ϯ�������q��C�aH@Ϯ�W
=�(���G�C��                                    Bxi�   �          @ۅ@�����R��z����C�t{@�����
�=q��Q�C��3                                    Bxi��  �          @�33@θR������u�C�!H@θR�u�\)���C��\                                    Bxi�h  �          @�(�@�{����Q�����C�k�@�{>8Q�����\)?��                                    Bxi�,  �          @ۅ@�33����ff����C�XR@�33>�z��#�
��G�@*=q                                    Bxi�:�  �          @�(�@�p��L��������C�u�@�p�>��H�=q��z�@��                                    Bxi�IZ  �          @ۅ@�\)�Ǯ����ffC�L�@�\)?E��z�����@ٙ�                                    Bxi�X   �          @��@�Q�    ���R���C��)@�Q�?�  ���H�i��Az�                                    Bxi�f�  �          @��@�=q�����  �nffC��q@�=q?(�ÿ���_33@���                                    Bxi�uL  |          @ٙ�@�G���׿�G��p  C��3@�G�>��H��  �o33@���                                    Bxi���  ,          @�=q@��
���H�����<��C��H@��
>�\)��Q��Dz�@�                                    Bxi���  T          @��H@��:�H��G��)C��)@�<#�
���H�D��=u                                    Bxi��>  T          @ٙ�@��H�n{��=q�4��C�  @��H����\)�\(�C�xR                                    Bxi���  �          @���@Ӆ�aG���z���C�33@Ӆ�.{�����E��C�C�                                    Bxi���  �          @أ�@�{�}p��Ǯ�X(�C��H@�{�L�Ϳ�����Q�C�                                    Bxi��0  �          @�\)@�Q�=����@  ��=q?h��@�Q�?�{�!���33Ap                                      Bxi���  �          @�{@���>����8�����H@7�@���?�p�����\)A�ff                                    Bxi��|  �          @׮@ƸR�\)�&ff��C�]q@ƸR?��H��
��Q�A0(�                                    Bxi��"  �          @���@\�#�
�?\)�љ�C���@\?��R�&ff��{A\(�                                    Bxi��  �          @�G�@�����
�6ff�ƸRC��f@��?�Q��{��=qAR{                                    Bxi�n  �          @أ�@ƸR�.{�+����RC�9�@ƸR?�(�������A2�\                                    Bxi�%  �          @׮@�Q�\)�(����C�]q@�Q�?����
=q��=qA#
=                                    Bxi�3�  �          @�\)@����  ���
�8��C�@��ÿ�\��
=�M�C���                                    Bxi�B`  �          @أ�@�(��N{?=p�@���C��H@�(��I����G��
ffC��                                    Bxi�Q  �          @ٙ�@�
=�'
=?E�@�Q�C��H@�
=�(Q�+���{C��f                                    Bxi�_�  T          @ٙ�@�\)�(Q��
=�e�C��\@�\)��\���H�i�C��                                    Bxi�nR  �          @ۅ@�Q��z�   ��C�'�@�Q���
���R�IG�C�e                                    Bxi�|�  �          @��H@θR�
�H��ff�s33C��
@θR�У׿�  �K33C���                                    Bxi���  �          @ۅ@ҏ\���z����\C�J=@ҏ\��G����@(�C���                                    Bxi��D  �          @��H@љ���
=�u���C���@љ��s33��Q��d��C���                                    Bxi���  �          @��@Ϯ���ÿ�G��*�RC�33@Ϯ�333������HC���                                    Bxi���  �          @�G�@Ϯ�У׿����C��3@Ϯ�W
=��\�r=qC�N                                    Bxi��6  �          @�Q�@�  �����=q��C�Y�@�  �B�\��(��k�C���                                    Bxi���  �          @ٙ�@У׿�{���
��C��@У׿Y�����H�iG�C�H�                                    Bxi��  �          @�G�@љ���
=���\�
�HC���@љ��333��{�\Q�C��                                    Bxi��(  �          @׮@Ϯ��G������5G�C���@Ϯ���Ϳ��
�up�C�=q                                    Bxi� �  �          @ָR@�{�p�׿�=q�Z�RC��{@�{<#�
����=L��                                    Bxi�t  �          @�ff@�
=�n{�����HQ�C���@�
=�L�Ϳ�(��n�\C��f                                    Bxi�  T          @׮@˅�����\�s\)C��@˅�u�������C��                                    Bxi�,�  �          @�Q�@˅��p����H�jffC�p�@˅�Ǯ��R��{C�<)                                    Bxi�;f  �          @׮@��H��Q�����RC���@��H�������
=C��)                                    Bxi�J  �          @���@ə�����G�����C�Q�@ə�>�\)��R���
@#�
                                    Bxi�X�  �          @�G�@ȣ׿J=q��R��G�C�ff@ȣ�?���!���33@�ff                                    Bxi�gX  �          @�Q�@�{�aG��#�
��C��3@�{?\)�(����=q@�{                                    Bxi�u�  �          @�@�=q���
�
=q���C���@�=q�L���(������C�\                                    Bxi���  �          @�@����ff���
�x  C��
@���!G��p���
=C��                                    Bxi��J  �          @�p�@���"�\��(��o33C�Y�@������2�\��C���                                    Bxi���  �          @ٙ�@�ff���Ϳ�ff�S\)C�R@�ff��Q��G�����C�l�                                    Bxi���  �          @���@Ϯ�����p��Ip�C�ff@Ϯ��{��
=��=qC���                                    Bxi��<  �          @�\)@�
=���H����C���@�
=�333���eC���                                    Bxi���  T          @�  @�p��Tz���H��ffC�u�@�p���ff�Y����  C�                                    Bxi�܈  T          @���@�ff�fff�L�Ϳ�
=C�/\@�ff�+��!G���(�C�*=                                    Bxi��.  �          @�=q@��ÿ��Ϳ
=����C�q@��ÿ����=q�6�\C�`                                     Bxi���  �          @�G�@ʏ\�������V�\C��@ʏ\�B�\�33��z�C��3                                    Bxi�z  �          @�G�@�녿�
=�
=���HC�*=@�녾�(��AG���\)C���                                    Bxi�   �          @ٙ�@�����\�?\)����C��)@��;���o\)�C��\                                    Bxi�%�  �          @���@ȣ��
=q����=G�C��@ȣ׿�\)��
���C���                                    Bxi�4l  �          @�G�@�����
��\���C�T{@��ͿG��J=q��\)C�7
                                    Bxi�C  �          @ڏ\@�G��$z��\����G�C�\@�G�����������C��\                                    Bxi�Q�  �          @�=q@�ff�1G��\(����C��R@�ff�
=��(���\C��f                                    Bxi�`^  �          @��@����������=qC�:�@���}p��8Q�����C�E                                    Bxi�o  �          @�33@����(Q�(����C���@��ÿ�
=�����{C�q�                                    Bxi�}�  �          @�z�@���/\)�W
=��  C�AH@����R�����W\)C�@                                     Bxi��P  
�          @��@��H�.{>u?��RC�e@��H��Ϳ��H�!�C�q�                                    Bxi���  �          @�z�@�33�!G�?E�@�
=C�+�@�33�#�
�&ff����C��                                    Bxi���  �          @�(�@�=q�A�?��A�C�@�=q�H�ÿ+���=qC�Y�                                    Bxi��B  �          @�z�@�=q���H?�(�A$z�C�=q@�=q���
��=q�ffC�)                                    Bxi���  �          @�
=@�G��U�?L��@ӅC���@�G��P�׿�ff�
�HC��3                                    Bxi�Վ  �          @���@�\)�"�\�k���
=C�H�@�\)�33�\�J{C�8R                                    Bxi��4  �          @�=q@���3�
?�p�A���C���@���[�>k�?��C�l�                                    Bxi���  �          @��H@����\@Z�HA��C���@���R�\@   A��C�q�                                    Bxi��  �          @���@�\)��
=@h��B \)C�P�@�\)�b�\@
=A�33C���                                    Bxi�&  �          @�=q@�
=�n{@9��A���C�Ff@�
=��{?��@�{C��                                    Bxi��  �          @�@�G��c33@a�A�z�C�n@�G����
?�A�C�                                      Bxi�-r  �          @�\@�ff�c33@j=qA�=qC�/\@�ff��{?��
A&{C��=                                    Bxi�<  �          @���@�z��S�
@�33B=qC���@�z����H?��RA�C�h�                                    Bxi�J�  �          @���@�Q��H��@��HB#��C���@�Q���=q@�A��
C��                                    Bxi�Yd  �          @ᙚ@���n{@z�HBC�h�@����
=?�z�A8Q�C���                                    Bxi�h
  �          @�\)@�
=�_\)@�=qB��C��)@�
=��33?�33AZ�HC���                                    Bxi�v�  �          @�ff@��N{@�Q�Bz�C��)@���
=?��HA�
=C���                                    Bxi��V  �          @�@�z��@  @�{BQ�C��
@�z����
@{A�C�                                    Bxi���  �          @�{@����#�
@�p�B5  C�"�@�����  @7
=A��C��                                    Bxi���  �          @�@����Q�@�Q�B.33C��@�����Q�@3�
A�z�C�!H                                    Bxi��H  �          @�p�@���p�@�G�B0  C�q@����33@333A��
C��                                    Bxi���  �          @�@��R�)��@���BQ�C�L�@��R���@��A�G�C�+�                                    Bxi�Δ  �          @�ff@�
=�;�@�z�B�HC�3@�
=���@ ��A��\C���                                    Bxi��:  �          @ۅ@�Q��,(�@uB
=C��@�Q�����?�\)A}G�C���                                    Bxi���  �          @��H@���p�@���B p�C�Y�@��qG�@6ffAŅC�L�                                    Bxi���  �          @ۅ@��Ϳ�p�@�
=B"�HC�AH@����s�
@:=qA���C��                                    Bxi�	,  �          @��H@���p�@�  B{C���@����33@�RA�  C��                                    Bxi��  �          @�=q@�  � ��@�{B.{C��q@�  ���H@,(�A�{C�k�                                    Bxi�&x  �          @�33@����  @�\)B.�
C�xR@�����z�@7
=A�p�C��                                    Bxi�5  �          @�=q@�
=�p�@��B6=qC�,�@�
=��@?\)A�ffC��{                                    Bxi�C�  �          @��H@�  ��@���B6C�j=@�  ��p�@A�Aң�C��3                                    Bxi�Rj  �          @ٙ�@l���{@��BC��C�+�@l������@Dz�A�p�C��                                    Bxi�a  �          @ۅ@vff��@��BI{C�R@vff��
=@Z=qA�  C�`                                     Bxi�o�  �          @�(�@��
���@�  B;33C�� @��
��G�@EAծC��                                    Bxi�~\  �          @�33@�ff�@��HB7p�C��@�ff���@A�A�C�"�                                    Bxi��  �          @��H@S33����@���Bi��C��@S33��@��B{C�&f                                    Bxi���  �          @ڏ\@�z��z�@�Q�B=�C���@�z���z�@L(�A��HC���                                    Bxi��N  T          @�33@���   @�33B(C�Z�@�����@7�A�ffC�G�                                    Bxi���  �          @�33@���  @�Q�B{C���@����z�@p�A�Q�C��                                    Bxi�ǚ  �          @�p�@�=q���@h��BffC��H@�=q�@  @�A�p�C���                                    Bxi��@  }          @љ�@�Q�>���@I��B =q@�33@�Q쿦ff@8��A�C���                                    Bxi���  
�          @�\)@�z�@K�@\)B�\B��@�z�?�R@�=qBC=qA ��                                    Bxi��  �          @�G�@�{>�@�=qB*�@��H@�{�   @�z�B=qC�H                                    Bxi�2  �          @ٙ�@u>�33@���B\��@�p�@u�'
=@��B:��C��f                                    Bxi��  �          @�Q�@�(���G�@���BO�
C�>�@�(��6ff@�{B&
=C���                                    Bxi�~  "          @��@��Ϳ!G�@�
=B(��C�Z�@����7�@^�RA�=qC��                                    Bxi�.$  "          @�\)@z�H�HQ�@��B��C���@z�H��33?�z�A�  C�}q                                    Bxi�<�  "          @�z�@e�e@���BffC�y�@e��?���A`Q�C�XR                                    Bxi�Kp  �          @Ӆ@�
=�˅@8Q�A�
=C�8R@�
=�7�?�\)Ac�C��3                                    Bxi�Z  �          @�(�@�  ���@Z=qA�G�C�� @�  �j=q?�  Au�C���                                    Bxi�h�  �          @�z�@�=q�7
=@*=qA�{C�޸@�=q�u�?E�@�{C�!H                                    Bxi�wb  �          @ָR@�p��|(�@0��AǮC�&f@�p�����>�z�@!�C�T{                                    Bxi��  
�          @�{@��\�p�@ffA�33C��H@��\�G
=?Y��@��C���                                    Bxi���  �          @�
=@�\)�6ff@z�A��\C�G�@�\)�h��>��H@�\)C�7
                                    Bxi��T  �          @׮@�\)�b�\?У�A`Q�C���@�\)�w����|��C�b�                                    Bxi���  "          @��
@�(��l��>k�@�\C��H@�(��QG��޸R�v�\C�XR                                    Bxi���  �          @��@��
��G�@�\A��C�R@��
��{�!G���z�C��                                    Bxi��F  �          @�{@Vff���
@\)B�C���@Vff��33?�  A,��C�p�                                    Bxi���  �          @��@}p���@S�
A�p�C�� @}p���G�?#�
@���C�h�                                    Bxi��  �          @�@�������@�\A�G�C�R@������׿
=q��C��
                                    Bxi��8  �          @�@�����Q�?�z�A���C���@������H�:�H��G�C���                                    Bxi�	�  "          @�(�@���  ?�p�A���C�E@����
�+����\C�%                                    Bxi��  
�          @�ff@�{��(�?�AMG�C�\@�{��{���H�.ffC��H                                    Bxi�'*  �          @У�@��\���H>8Q�?�=qC��@��\��
=�Q����C�޸                                    Bxi�5�  �          @�@�{����{�_�C�4{@�{�W
=�|���G�C���                                    Bxi�Dv  "          @�p�@�����\�^�R��C�t{@���j�H�QG���33C�ff                                    Bxi�S  �          @ָR@�ff��p��&ff���C��
@�ff�h���?\)��C�)                                    Bxi�a�  
�          @׮@�{��{�z�H��C��@�{�^�R�Q���\C��{                                    Bxi�ph  �          @�Q�@����{�ٙ��j�\C�t{@���8���qG����C�O\                                    Bxi�  �          @���@|�����?��RAP  C��
@|����(���33�C�C��f                                    Bxi���  T          @�@�=q��33?��
AS�C��q@�=q����p��*ffC��                                     Bxi��Z  "          @�p�@����(�?�G�A  C���@����
=�Ǯ�X��C�
                                    Bxi��   �          @�@����G�?�Q�Ak33C���@����  �p���p�C��                                    Bxi���  
�          @�ff@�����  @{A�G�C���@������R�
=����C�Ff                                    Bxi��L  
�          @ָR@��R��\)@(�A��RC���@��R�������C��                                    Bxi���  T          @�@�����@ ��A�ffC�|)@����p��   ����C�q                                    Bxi��  
�          @��H@
�H���@E�A��C�@ @
�H�\��G����\C��{                                    Bxi��>  �          @љ�@AG����R@2�\A�(�C�S3@AG���
=���
�@  C�n                                    Bxi��  �          @��@s33��Q�@*=qA�
=C�K�@s33��ff�aG���{C�o\                                    Bxi��  T          @���@|������@4z�A��C���@|�����\=#�
>�{C�Ff                                    Bxi� 0  �          @�z�@�=q����@'�A�  C���@�=q��\)�\)��C��3                                    Bxi�.�  �          @���@q���\)@4z�AͅC�{@q�����=u?�C���                                    Bxi�=|  �          @���@����z�@ ��A�{C�b�@�����������{C�c�                                    Bxi�L"  �          @�G�@�G���33@33A��C�E@�G���p����R�/\)C��=                                    Bxi�Z�  �          @ҏ\@����z�?�33Ah��C�8R@������aG���ffC��f                                    Bxi�in  T          @��H@��R����>u@�C��@��R�\)�
=q��Q�C��)                                    Bxi�x  �          @ҏ\@���g��#33����C��=@�녿�  ������C��q                                    Bxi���  
�          @�z�@�G��h���E�߅C���@�G���  ��G��-��C���                                    Bxi��`  
�          @��
@�p��:�H�z�H��C�C�@�p���ff���
�>p�C��                                    Bxi��  �          @�(�@�
=��  �\)���\C�l�@�
=�b�\�5����HC�Ф                                    Bxi���  �          @�G�@�\)���H?z�@��\C�O\@�\)��
=��33���C���                                    Bxi��R  "          @���@z�H��=q>�33@K�C���@z�H��G��{��33C�/\                                    Bxi���  �          @�@P  ���H>�@���C���@P  ���
�Q�����C�L�                                    Bxi�ޞ  �          @�Q�@��c33@N{B�HC��f@����R?fffA�C�k�                                    Bxi��D  �          @�p�>��H��\)@���B���C�s3>��H���@c33B#C�q�                                    Bxi���  "          @���?녿�33@�Q�B�� C��H?�����@Z=qB �C��                                    Bxi�
�  �          @�p�?z��33@��B��\C�}q?z����H@C33B��C��                                    Bxi�6  �          @�
=?���33@�\)Bm�C��R?����G�@'�A�z�C�|)                                    Bxi�'�  �          @�Q�?�\��@�(�B�\)C���?�\���@Mp�B�C���                                    Bxi�6�  �          @�(�?E���@��
B��C�� ?E��b�\@c33B1��C��                                    Bxi�E(  �          @�\)>Ǯ�G�@���B���C��f>Ǯ��
=@C33B�
C��=                                    Bxi�S�  �          @��R<��
�.�R@�BiffC�AH<��
��z�@
=A��
C�&f                                    Bxi�bt  �          @�ff    �C33@�
=B^�\C��    ��@{A�=qC�                                    Bxi�q  "          @���\)�7�@�
=Bo��C����\)����@0  A�33C�7
                                    