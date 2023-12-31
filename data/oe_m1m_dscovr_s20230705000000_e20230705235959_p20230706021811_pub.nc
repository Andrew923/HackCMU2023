CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230705000000_e20230705235959_p20230706021811_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-07-06T02:18:11.167Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-07-05T00:00:00.000Z   time_coverage_end         2023-07-05T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�5�@  "          @�G�@y���}p��%�
=C��@y���L���)����C�,�                                    Bx�5��  �          @�G�@y���s33�'
=�\)C�%@y���B�\�*�H��
C�s3                                    Bx�5��  T          @���@vff�s33�*�H��C�
@vff�B�\�.�R�(�C�n                                    Bx�5�2  T          @��@{��.{�"�\�{C�!H@{��   �%�p�C�g�                                    Bx�5��  T          @��@�Q���{��p�C�C�@�Q쾳33�   ����C�|)                                    Bx�5�~  T          @�Q�@��׿
=��R����C��3@��׾�
=�!G����C�
=                                    Bx�5�$  
�          @�ff@~{�Q��Q����
C�+�@~{�&ff�(�����C�T{                                    Bx�5��  T          @��@��׿=p��Q���p�C���@��׿z���H��z�C��                                    Bx�6	p  �          @�\)@u�:�H�*�H�	��C��)@u����.{�(�C��3                                    Bx�6  
�          @���@�G��
=q��R��Q�C�4{@�G���p�� ����C�e                                    Bx�6&�  �          @�p�@��H>u��33��Q�@N{@��H>�Q�У���(�@�                                    Bx�65b  �          @���@��>�ff��33��ff@��\@��?
=q��\)���H@���                                    Bx�6D  �          @��@�(�>���������  @}p�@�(�>�녿������@�33                                    Bx�6R�  �          @�{@��>��ٙ����H?�G�@��>�  ��Q�����@P                                      Bx�6aT  T          @�@�(�>�z��{����@s�
@�(�>��Ϳ�=q��=q@�ff                                    Bx�6o�  �          @���@�ff���
��ff����C��@�ff�L�Ϳ����\C���                                    Bx�6~�  �          @��
@=q�'
=�8Q��Q�C�Y�@=q�=q�C33�'G�C�xR                                    Bx�6�F  �          @�@�R�)���<�����C��\@�R����G��'z�C��\                                    Bx�6��  T          @���@%��ff�C�
�$�HC��3@%��	���Mp��.C��                                    Bx�6��  T          @�ff@1G��ff�>{�  C���@1G��	���G
=�&p�C�R                                    Bx�6�8  �          @�p�@=p��(��7���RC�Ǯ@=p��   �@  � Q�C��                                    Bx�6��  T          @�ff@8���(��@  �\)C�k�@8�ÿ��R�HQ��'
=C��
                                    Bx�6ք  �          @��@C�
� ���@  �33C�W
@C�
�����HQ��%
=C��=                                    Bx�6�*  
�          @��@O\)���8���{C���@O\)��p��@���G�C���                                    Bx�6��  �          @���@^{��  �0�����C��q@^{�����7���C��3                                    Bx�7v  `          @�Q�@c�
�޸R�'��{C��3@c�
�����.�R�=qC�3                                    Bx�7  T          @�p�@Z�H��Q��,���
=C��)@Z�H�\�333�33C��                                    Bx�7�  �          @��R@G���(��L���+=qC�l�@G���G��Q��1  C��q                                    Bx�7.h  �          @���@Fff��p��H���)��C�8R@Fff���
�N�R�/z�C��H                                    Bx�7=  �          @�=q@C�
���*�H�Q�C�޸@C�
��z��333��C��)                                    Bx�7K�  �          @��
@K������3�
��HC�R@K�����:�H���C�S3                                    Bx�7ZZ  �          @��@HQ����7
=�\)C�q@HQ��{�=p�� {C�aH                                    Bx�7i   �          @��@=p�����333��\C��=@=p��ٙ��:=q�!��C��                                    Bx�7w�  T          @�
=@=p�����1G��
=C���@=p����H�8Q�� G�C���                                    Bx�7�L  �          @�ff@>{���,�����C��
@>{��  �4z���HC��H                                    Bx�7��  �          @�@?\)���,���G�C�7
@?\)���3�
�G�C�e                                    Bx�7��  T          @�{@E���{�%�=qC�w
@E��ٙ��,���{C���                                    Bx�7�>  �          @�33@u��0�׿��ѮC��@u��녿�����(�C��q                                    Bx�7��  T          @��R@��?�ff�J=q�33ARff@��?���:�H��RA[
=                                    Bx�7ϊ  H          @�ff@�p�@
�H=L��?z�A��@�p�@
�H>.{@Q�A�\)                                    Bx�7�0  
4          @�@�{@�>aG�@-p�Aә�@�{@z�>�{@�{A�(�                                    Bx�7��  �          @��@�=q?�
=>.{@
=A�Q�@�=q?�>�z�@]p�A��                                    Bx�7�|  "          @�\)@��?�
==�\)?Q�A�
=@��?�
=>#�
@   A�ff                                    Bx�8
"  T          @���@���?�G��\)���A�Q�@���?\�L�Ϳ
=A���                                    Bx�8�  T          @�  @�  ?�p���33����A�{@�  ?�  ��=q�P  A��                                    Bx�8'n  �          @���@�  ?�Q�����ffA�(�@�  ?�(���\��ffA��                                    Bx�86  
�          @���@��?��8Q��
�\A��@��?����#�
��A�p�                                    Bx�8D�  �          @�\)@�ff?�녿G��  A��H@�ff?�
=�333���A���                                    Bx�8S`  �          @�p�@�(�?�{�\(��(��A�@�(�?�33�G����A�{                                    Bx�8b  �          @�z�@�33?�\)�^�R�-A���@�33?�z�L���ffA�                                      Bx�8p�  T          @��\@���?�33�\(��-�A�\)@���?����J=q�{A��                                    Bx�8R  �          @�  @�\)?���E��{A�  @�\)?��Ϳ333�33A��                                    Bx�8��  �          @���@���?���333�{A��
@���?�=q�!G���\)A�\)                                    Bx�8��  �          @�33@��?���.{���A�(�@��?��Ϳ(����A��                                    Bx�8�D  �          @��@�(�?���z���{A��@�(�?����\���A��
                                    Bx�8��  �          @��\@��
?�ff��G���  A���@��
?��þ�p���(�A���                                    Bx�8Ȑ  �          @�=q@��
?�  �����(�A�@��
?��\��������A�                                      Bx�8�6  �          @�33@��?�  ������
=A|��@��?��\��������A�Q�                                    Bx�8��  T          @�z�@�?����(�����A�{@�?����Q���{A�                                      Bx�8�  �          @��
@�{?��H��Q���  At��@�{?�p������n�RAw�
                                    Bx�9(  �          @��@�ff?��׾�����33Ac
=@�ff?�녾�=q�X��Ae�                                    Bx�9�  �          @�(�@��?��;��
�|��A\��@��?�\)����P��A_�                                    Bx�9 t  �          @��
@��R?�z�Ǯ��G�Aj{@��R?�
=�������Amp�                                    Bx�9/  �          @�{@�ff?��ÿ�R���A�(�@�ff?��Ϳ����  A��H                                    Bx�9=�  �          @�p�@�(�?�
=�:�H�\)A��\@�(�?��H�(���G�A�                                    Bx�9Lf  �          @���@�z�?�{�5���A�33@�z�?�녿&ff��
=A�Q�                                    Bx�9[  �          @�p�@��R?�(��.{��\At  @��R?�  ��R��Az{                                    Bx�9i�  �          @�@��?�z�(���ffAg�@��?�Q�(���RAmG�                                    Bx�9xX  T          @�Q�@�  ?�\)�+��A�
=@�  ?�33�(����A��
                                    Bx�9��  T          @�\)@�
=?�33�&ff��(�A��R@�
=?��z���\A�p�                                    Bx�9��  �          @��@��?�\)�(�����RA�{@��?�33�
=��p�A���                                    Bx�9�J  �          @�\)@��R?�����R��G�A�@��R?�(������
=A�=q                                    Bx�9��  �          @��
@��?����R����A��@��?Ǯ�����z�A��                                    Bx�9��  �          @�33@�33?�\)�\)��G�A�Q�@�33?�33�   ��Q�A��\                                    Bx�9�<  �          @��
@�G�?��H�h���5�A�z�@�G�?�  �W
=�(z�A�=q                                    Bx�9��  �          @�33@��
?��H�G��\)Aw\)@��
?��R�8Q��z�A}�                                    Bx�9�  `          @�33@�33?�����\�L��Aa�@�33?�녿xQ��B�RAj�\                                    Bx�9�.  "          @�{@�\)?�33�=p����Af�R@�\)?�
=�0����RAl��                                    Bx�:
�  �          @�ff@��?��=p��(�Aj{@��?����0���=qAp                                      Bx�:z  T          @�{@��?�Q�&ff���Am�@��?�(��
=����As
=                                    Bx�:(   T          @�p�@��R?����:�H�33ApQ�@��R?�p��.{��Au�                                    Bx�:6�  �          @��@�
=?�������׮Ap��@�
=?�(��   �ÅAt��                                    Bx�:El  �          @�p�@�ff?�33�333�
�RAg
=@�ff?��&ff�G�AlQ�                                    Bx�:T  T          @�33@�
=?+��@  �33A(�@�
=?333�:�H���A{                                    Bx�:b�  "          @��H@�\)>W
=�u�B{@'�@�\)>u�u�@Q�@G
=                                    Bx�:q^  �          @��
@��R=�G������p  ?��@��R>#�
��Q��n�H@                                       Bx�:�  T          @��@�
=>�������c
=?��H@�
=>L�Ϳ�\)�a��@!�                                    Bx�:��  
�          @�(�@��>��������U@s�
@��>�{����S\)@��H                                    Bx�:�P  �          @��@�  >�
=�h���7
=@�=q@�  >�ff�fff�3�@�Q�                                    Bx�:��  `          @�p�@��
?5��p����
Az�@��
?E���������A�                                    Bx�:��  
�          @�@�  ?!G���\)�^=qA (�@�  ?+������YG�Az�                                    Bx�:�B  �          @�@�\)?&ff���g\)A33@�\)?0�׿���bffA�                                    Bx�:��  
Z          @��R@�G�?�R��=q�T  @��@�G�?(�ÿ���O\)A�                                    Bx�:�  �          @�@���?�R���
�K
=@��@���?(�ÿ�G��FffA��                                    Bx�:�4  H          @��@�  ?.{��G��H��A	p�@�  ?5�}p��C�
Az�                                    Bx�;�  
�          @��R@���>\��z��d(�@���@���>�
=��33�ap�@���                                    Bx�;�  
�          @���@��>��Ϳ�
=�dz�@�Q�@��>�G���z��a��@�                                      Bx�;!&  T          @�
=@���?z῏\)�\��@�\)@���?(������X��@�ff                                    Bx�;/�  �          @��@�  ?녿����T��@�@�  ?(���ff�P��@�                                    Bx�;>r  
�          @��
@��R>���=q�YG�@��H@��R?�\�����U�@���                                    Bx�;M  �          @�z�@�\)?:�H�p���;�AQ�@�\)?B�\�k��6�HA=q                                    Bx�;[�  `          @��@�{>�Q쿕�k�
@��\@�{>Ǯ��z��iG�@��                                    Bx�;jd  "          @�33@�ff>�(�����[�@�  @�ff>���=q�X��@�                                    Bx�;y
  �          @��@�?8Q쿅��P(�A(�@�?@  ���\�K�A�\                                    Bx�;��  "          @���@�G�?�p��\)��  Ar�\@�G�?�  ����G�Au                                    Bx�;�V  �          @��@��?У׾�Q���
=A�33@��?�녾��R�i��A�{                                    Bx�;��  �          @��
@�=q?��.{����A�ff@�=q?�����\)A���                                    Bx�;��  
f          @��H@���?�p����
�c�
A�G�@���?�p������RA�p�                                    Bx�;�H  
�          @�(�@�Q�?����Ϳ��A�@�Q�?��ͽ#�
��A��                                    Bx�;��  T          @�33@�{?�Q�=u?(��A��@�{?�Q�=�G�?�ffA���                                    Bx�;ߔ  
�          @�33@��?���>.{@33A�ff@��?�>aG�@(��A��                                    Bx�;�:  T          @��@�\)?�{<��
>�=qA��H@�\)?�{=�\)?\(�A���                                    Bx�;��  "          @�(�@���?�ff    <�A��@���?�ff=L��?z�A�
=                                    Bx�<�  T          @�33@�\)?���\)�VffA���@�\)?��;u�2�\A�\)                                    Bx�<,  "          @���@��R?ٙ�=�?�
=A��\@��R?�Q�>#�
?�Q�A�=q                                    Bx�<(�  �          @�=q@�{?���=�Q�?��
A���@�{?���>�?���A��\                                    Bx�<7x  �          @��@�?��R>�?�ffA�@�?��R>8Q�@
=A�p�                                    Bx�<F  �          @��@��@
=q=u?5AУ�@��@
=q=�G�?�ffA�z�                                    Bx�<T�  �          @�z�@��H@  =u?0��A�G�@��H@  =�G�?��A��                                    Bx�<cj  
�          @�z�@���@��>��?�p�A�@���@Q�>L��@
=A�p�                                    Bx�<r  "          @�p�@�@%�>�=q@H��A�G�@�@$z�>���@s33A��R                                    Bx�<��  T          @��@�ff@"�\>�  @:=qA�33@�ff@!�>���@b�\A��R                                    Bx�<�\  
�          @��@�\)@   >W
=@�A���@�\)@\)>��@A�A�=q                                    Bx�<�  �          @���@�\)@p�>��R@fffA�@�\)@p�>�Q�@�A�33                                    Bx�<��  "          @�(�@�G�@\)?��@��A���@�G�@�R?#�
@�G�A�                                    Bx�<�N  T          @�33@�z�@{>���@���A���@�z�@p�>�ff@��HA�{                                    Bx�<��  T          @�33@��@*=q<�>�33B�
@��@)��=��
?p��B��                                    Bx�<ؚ  "          @��@���@)��>�p�@�(�B�@���@(��>�
=@�ffB\)                                    Bx�<�@  T          @��\@}p�@(��?J=qA�\B��@}p�@'�?W
=A�B
=                                    Bx�<��  "          @��\@|��@*=q?E�A33B��@|��@(��?Q�A�
B{                                    Bx�=�  �          @�=q@w
=@333?0��A�HB��@w
=@2�\?:�HA�B\)                                    Bx�=2  .          @�33@x��@/\)?fffA+�B�R@x��@.�R?s33A3�
B(�                                    Bx�=!�  
�          @��\@u�@3�
?fffA*�HBG�@u�@333?p��A3
=B                                    Bx�=0~  �          @�=q@r�\@5?c�
A*�\BG�@r�\@4z�?n{A2�\B                                    Bx�=?$  �          @��H@xQ�@2�\?.{A ��B��@xQ�@1�?8Q�AQ�B�\                                    Bx�=M�  T          @�33@s�
@7
=?aG�A'\)B��@s�
@6ff?k�A.�RB33                                    Bx�=\p  T          @��\@r�\@7�?^�RA%p�B�\@r�\@7
=?fffA,��B�                                    Bx�=k  
�          @��\@s�
@6ff?Q�Az�B=q@s�
@5?\(�A#33B�
                                    Bx�=y�  �          @��@p��@<��?^�RA%�B�@p��@<(�?h��A,z�BG�                                    Bx�=�b  �          @�z�@b�\@G�?�  AlQ�B%��@b�\@G
=?��
As
=B%�                                    Bx�=�  T          @�\)@Vff@J=q?���AQ�B-\)@Vff@I��?���AW�
B,��                                    Bx�=��  �          @���@QG�@QG�?�
=AdQ�B4(�@QG�@P��?��HAj�RB3                                    Bx�=�T  T          @�=q@J=q@Z=q?��A{
=B<��@J=q@X��?��A��RB<33                                    Bx�=��  
�          @�Q�@J=q@h��?�\)A}BDG�@J=q@hQ�?�33A��BC�H                                    Bx�=Ѡ  
�          @�(�@G
=@dz�?�
=A`  BC@G
=@c�
?��HAe��BCp�                                    Bx�=�F  T          @�\)@U�@s�
?��Ag�BC{@U�@s33?��Al��BB��                                    Bx�=��  
�          @��R@QG�@u?���Aj=qBFp�@QG�@u�?��Ao33BF(�                                    Bx�=��  �          @�ff@J=q@y��?�{Ar�\BL  @J=q@x��?��Aw33BK                                    Bx�>8  
�          @�ff@@��@���?�{Ar�HBT�@@��@�Q�?��Aw33BTz�                                    Bx�>�  �          @�{@?\)@�  ?�A}�BT��@?\)@\)?�Q�A��HBT��                                    Bx�>)�  
�          @�ff@J=q@tz�?�ffA���BI\)@J=q@s�
?���A�ffBI�                                    Bx�>8*  �          @�ff@Mp�@n�R?�z�A�p�BE�@Mp�@n{?�
=A��HBD�                                    Bx�>F�  �          @�(�@K�@c�
?��A��\B@@K�@c33?�33A�B@�                                    Bx�>Uv  "          @���@R�\@c33?�G�A�{B<@R�\@b�\?�\A�
=B<��                                    Bx�>d  
�          @���@U�@g�?�  A�  B=z�@U�@g�?�G�A���B=ff                                    Bx�>r�  �          @��
@W
=@c�
?��RA��B:@W
=@c�
?��RA���B:�                                    Bx�>�h  �          @��
@XQ�@a�?�G�A��HB9  @XQ�@a�?�G�A�33B8��                                    Bx�>�  �          @��@P  @e?�  A�p�B?\)@P  @e?�  A���B?\)                                    Bx�>��  
Z          @�G�@W�@[�?���A�{B5��@W�@[�?���A�  B6                                      Bx�>�Z  "          @���@W
=@a�?�Q�A\(�B9��@W
=@a�?�Q�A[�B9��                                    Bx�>�   
�          @���@a�@Z=q?�AV�HB0
=@a�@Z=q?�AU�B0{                                    Bx�>ʦ  I          @�G�@\(�@[�?���Ar�RB3@\(�@[�?��Aqp�B3�
                                    Bx�>�L  -          @�G�@\��@Z�H?�=qAt��B3�@\��@Z�H?���As33B3=q                                    Bx�>��  "          @���@e@R�\?�  Af�RB*  @e@R�\?��RAd��B*(�                                    Bx�>��  �          @��@`��@W
=?��AIp�B.�@`��@W�?���AF�HB/{                                    Bx�?>  
�          @��@`  @W�?���AQp�B/��@`  @XQ�?�{AN�\B/��                                    Bx�?�  T          @�  @aG�@W
=?���AP��B.�R@aG�@W�?�{AMG�B.�                                    Bx�?"�  �          @��@\��@\(�?���AG33B3�R@\��@\(�?�ffAC�B3�                                    Bx�?10  T          @�z�@I��@^�R?��Av=qB?��@I��@_\)?��\Aq�B?�H                                    Bx�??�  �          @�z�@9��@h��?�  A���BM��@9��@i��?�(�A�Q�BN�                                    Bx�?N|  "          @�(�@9��@g�?�G�A�ffBM{@9��@hQ�?�p�A��BMp�                                    Bx�?]"  T          @�(�@6ff@i��?��A��BPQ�@6ff@j�H?�G�A�z�BP�                                    Bx�?k�  �          @�z�@7
=@j=q?��A�ffBP(�@7
=@j�H?�  A��BP��                                    Bx�?zn  {          @�p�@9��@j=q?\A�(�BN�@9��@j�H?�p�A��RBN��                                    Bx�?�  
�          @�p�@@vff?�p�A�{Bj��@@w�?�Q�A�{Bk(�                                    Bx�?��  "          @�p�@=q@w�?�{A��Bhff@=q@x��?���A�p�Bh�                                    Bx�?�`  
�          @�ff@.�R@w�?�A�  B[{@.�R@xQ�?���A��B[�\                                    Bx�?�  
�          @�33@,��@w�?���A�G�B\��@,��@y��?�33A��RB]G�                                    Bx�?ì  �          @�33@-p�@w
=?�Q�A�G�B[�@-p�@xQ�?�33A�z�B\\)                                    Bx�?�R  I          @���@p�@z�H@  A�
=Bg=q@p�@|��@(�A��
Bh
=                                    Bx�?��  _          @��@�@y��@33A��Bk{@�@|(�@\)A�ffBk�                                    Bx�?�  T          @���@(�@tz�@p�A�p�Bpff@(�@vff@��A㙚BqQ�                                    Bx�?�D  �          @�{@��@xQ�@{A�Bh�@��@z�H@=qA�Bi�                                    Bx�@�  �          @�z�@�@w�@!G�A�ffBm�H@�@z=q@p�A�{Bn�                                    Bx�@�  �          @�p�@33@u@%A�RBlp�@33@xQ�@!�A�(�Bm�                                    Bx�@*6  �          @��@Q�@qG�@�RA���Bf�@Q�@s�
@�HA�=qBg�
                                    Bx�@8�  �          @�@%�@l(�?��RA��\B\p�@%�@n�R?�A�B]p�                                    Bx�@G�  T          @�
=@=p�@l(�?�G�A��
BMQ�@=p�@n{?�Q�A�33BN�                                    Bx�@V(  �          @���@b�\@Z=q?�ffA@(�B/�
@b�\@[�?z�HA3�
B0z�                                    Bx�@d�  T          @�  @c�
@X��?n{A+33B.G�@c�
@Y��?\(�A�\B.�H                                    Bx�@st  �          @�\)@\(�@\(�?��AD��B3@\(�@]p�?}p�A733B4z�                                    Bx�@�  �          @�Q�@b�\@XQ�?��A?�B.�H@b�\@Y��?xQ�A2=qB/��                                    Bx�@��  �          @���@h��@Y��?Tz�A  B,=q@h��@Z�H?B�\A
=qB,�
                                    Bx�@�f  �          @�G�@i��@X��?O\)A�
B+�@i��@Y��?:�HAB,{                                    Bx�@�  �          @���@g
=@X��?z�HA2�\B,@g
=@Z=q?fffA$  B-z�                                    Bx�@��  �          @��\@c33@`��?s33A,��B2��@c33@a�?^�RA��B3Q�                                    Bx�@�X  �          @�=q@c33@]p�?��A@��B1�@c33@_\)?xQ�A1�B1��                                    Bx�@��  �          @�33@e�@\��?���AL��B/��@e�@^{?��A<��B0�\                                    Bx�@�  �          @�z�@j=q@\��?���A@  B-ff@j=q@^�R?z�HA/�B.G�                                    Bx�@�J  �          @�z�@o\)@Vff?���A@  B'\)@o\)@XQ�?z�HA/�B(G�                                    Bx�A�  T          @��
@qG�@S�
?��\A7�
B%�@qG�@U?n{A'\)B&                                      Bx�A�  �          @��@n�R@P  ?��A@��B${@n�R@Q�?xQ�A0  B%{                                    Bx�A#<  T          @��H@u@H��?���AH(�B�@u@J�H?�G�A7�B33                                    Bx�A1�  "          @�=q@w
=@E�?���AIp�BG�@w
=@G
=?�G�A8��Bff                                    Bx�A@�  �          @�=q@qG�@N�R?��
A:�HB"(�@qG�@P  ?n{A(��B#(�                                    Bx�AO.  
�          @��\@qG�@N�R?���AB=qB"�@qG�@P��?xQ�A0  B#��                                    Bx�A]�  �          @��@l��@XQ�?xQ�A0(�B)�\@l��@Z=q?^�RA��B*�                                    Bx�Alz  
�          @�33@l��@Vff?��
A:�RB(��@l��@XQ�?k�A'
=B)��                                    Bx�A{   �          @��H@hQ�@X��?���AG\)B,\)@hQ�@[�?}p�A2�HB-z�                                    Bx�A��  T          @��\@g
=@\(�?uA.�HB.p�@g
=@^{?Y��AB/p�                                    Bx�A�l  
�          @��\@e�@XQ�?��RAaG�B-�@e�@Z�H?�\)AK�
B.�
                                    Bx�A�  �          @��\@g
=@XQ�?�AT��B,�\@g
=@Z�H?�ffA?33B-�
                                    Bx�A��  �          @��
@l��@QG�?�=qAq�B%@l��@S�
?�(�A\z�B'Q�                                    Bx�A�^  T          @��
@j�H@S�
?���Ao�B'�H@j�H@Vff?���AYp�B)p�                                    Bx�A�  �          @��@e@U�?��AlQ�B+�@e@XQ�?�AUp�B-{                                    Bx�A�  �          @�G�@b�\@X��?��HA^{B/{@b�\@[�?�=qAF=qB0�                                    Bx�A�P  T          @���@c33@W
=?��HA^�RB-@c33@Y��?�=qAF�RB/=q                                    Bx�A��  "          @�@[�@W
=?�z�AY�B1��@[�@Y��?��
A@(�B3{                                    Bx�B�  �          @��
@XQ�@Q�?�p�Ak
=B0��@XQ�@U�?���AQB233                                    Bx�BB  �          @��@Z�H@Mp�?�G�Aqp�B,��@Z�H@P��?���AX(�B.z�                                    Bx�B*�  �          @�(�@W
=@U�?�A^ffB3  @W
=@XQ�?��
AD  B4�\                                    Bx�B9�  �          @�z�@XQ�@Tz�?���Ac\)B1�
@XQ�@W
=?��AHz�B3z�                                    Bx�BH4  �          @�33@R�\@Vff?�(�Ai�B5�
@R�\@Y��?�=qAMp�B7�                                    Bx�BV�  �          @�(�@W�@P  ?�\)A�ffB0{@W�@S�
?�p�Aip�B2
=                                    Bx�Be�  
�          @�z�@W
=@S33?��At��B1��@W
=@W
=?��AX��B3��                                    Bx�Bt&  �          @�=q@Y��@J=q?���A}�B+�@Y��@Mp�?�
=Ab{B-�                                    Bx�B��  �          @��\@\(�@G�?���A}p�B(�
@\(�@K�?�
=Aa��B*�H                                    Bx�B�r  �          @�33@c33@E?�z�A^ffB$=q@c33@H��?��\AB�RB&{                                    Bx�B�  T          @��\@dz�@@  ?��\As�
B {@dz�@C33?���AXz�B"(�                                    Bx�B��  T          @��\@aG�@>�R?�
=A�(�B!
=@aG�@C33?��AxQ�B#p�                                    Bx�B�d  T          @���@\(�@=p�?�  A�{B"@\(�@A�?���A��B%\)                                    Bx�B�
  T          @���@W�@C33?�(�A�33B(�@W�@G�?���A�(�B+ff                                    Bx�Bڰ  �          @�  @Tz�@AG�?��
A��RB)(�@Tz�@E?���A�p�B+�
                                    Bx�B�V  
�          @���@Vff@H��?�p�Ao�
B,��@Vff@L��?���APQ�B.�R                                    Bx�B��  �          @���@[�@L(�?fffA-B+�@[�@N�R?=p�A{B-                                      Bx�C�  �          @��@`��@HQ�?�  A@��B&�
@`��@K�?W
=A!G�B(�\                                    Bx�CH  T          @���@^�R@C�
?���AZffB%�@^�R@G�?uA:�RB'�                                    Bx�C#�  T          @�  @^{@AG�?�Q�Ag�B$�@^{@E�?��
AG�B&G�                                    Bx�C2�  �          @���@aG�@A�?�(�Ak
=B#
=@aG�@E?�ffAJ�HB%Q�                                    Bx�CA:  �          @���@`��@@  ?��A}p�B"33@`��@Dz�?��A\��B$�                                    Bx�CO�  �          @���@aG�@:=q?��A��
B�\@aG�@>�R?�p�Ao\)B!Q�                                    Bx�C^�  �          @�
=@Z=q@6ff?���A�Q�B��@Z=q@<(�?�Q�A��
B"�                                    Bx�Cm,  "          @�@Vff@.{?�ffA�33B�@Vff@4z�?�33A���B p�                                    Bx�C{�  �          @�p�@W
=@)��?�=qA��B�@W
=@0  ?�
=A���BG�                                    Bx�C�x  �          @�ff@HQ�@N{?���A���B7  @HQ�@R�\?�Q�Ak33B9��                                    Bx�C�  
e          @��H@HQ�@_\)?���Ae��B@z�@HQ�@c�
?}p�A=��BB�                                    Bx�C��  �          @�=q@E@_\)?�(�Ak\)BA@E@c33?��\AB�RBC�                                    Bx�C�j  	�          @��
@Fff@`  ?��A�BA��@Fff@dz�?���AV=qBC��                                    Bx�C�  �          @��@G�@]p�?��A��RB@�@G�@b�\?�
=A`  BB��                                    Bx�CӶ  T          @�(�@J�H@]p�?���A�  B=��@J�H@b�\?���AVffB@p�                                    Bx�C�\  T          @�33@U�@Q�?��\As�B2�@U�@W
=?���AK�B5                                      Bx�C�  �          @��@X��@P  ?�  Ao
=B/\)@X��@U�?�ffAG
=B1�H                                    Bx�C��  T          @��\@J�H@Z=q?�ffAy�B<p�@J�H@^�R?�=qANffB>�                                    Bx�DN  �          @��\@E@`  ?�G�Ap��BB(�@E@dz�?��
AD��BDz�                                    Bx�D�  
Z          @�33@Fff@a�?��HAg
=BB�
@Fff@fff?z�HA:=qBE{                                    Bx�D+�  T          @��
@C33@g�?���AW33BGQ�@C33@k�?c�
A)�BIff                                    Bx�D:@  
�          @�z�@Dz�@i��?��AMBG@Dz�@mp�?W
=A
=BI�R                                    Bx�DH�  
�          @�z�@@��@h��?�  AmBIff@@��@mp�?�G�A>ffBK�R                                    Bx�DW�  �          @���@E@dz�?�=qA{
=BDQ�@E@i��?�=qAK�
BF�H                                    Bx�Df2  "          @�33@G
=@\��?�z�A���B?��@G
=@b�\?�
=A`Q�BB�                                    Bx�Dt�  �          @��
@Fff@`��?��Ay�BB(�@Fff@e?��AIp�BD                                    Bx�D�~  �          @��@L��@Y��?��A���B:�H@L��@^�R?���ARffB=�R                                    Bx�D�$  �          @��H@G
=@^{?�ffAx��B@p�@G
=@c33?�ffAHz�BC�                                    Bx�D��  T          @���@E�@fff?�  AlQ�BE��@E�@k�?}p�A:ffBHG�                                    Bx�D�p  �          @��@B�\@g�?�ffAu��BG�
@B�\@mp�?��AB�HBJff                                    Bx�D�  �          @���@?\)@j�H?�G�An�\BK�@?\)@p  ?}p�A:�RBM��                                    Bx�D̼  T          @��@5�@mp�?�{A�{BR��@5�@s33?��ANffBU=q                                    Bx�D�b  T          @��
@3�
@r�\?��RAk
=BU�R@3�
@w�?s33A4(�BX
=                                    Bx�D�  �          @��@333@s33?�{ATQ�BVQ�@333@w�?Tz�A��BXff                                    Bx�D��  "          @�G�@.�R@l��?��A}p�BV=q@.�R@r�\?��\AEp�BX��                                    Bx�ET  "          @�Q�@0  @l��?�Q�Ahz�BU\)@0  @q�?h��A0(�BW�R                                    Bx�E�  �          @���@4z�@i��?��HAj�HBQ33@4z�@n�R?k�A2�RBS�                                    Bx�E$�  �          @���@>{@a�?�Q�Ag\)BG�
@>{@g
=?h��A0z�BJp�                                    Bx�E3F  �          @��H@<(�@l(�?��AK�BM�H@<(�@p��?E�A
=BP
=                                    Bx�EA�  �          @�@>�R@p  ?��AU�BN{@>�R@u�?W
=A��BP\)                                    Bx�EP�  �          @�p�@<(�@o\)?��
AqBOQ�@<(�@u�?z�HA7�
BR                                      Bx�E_8  T          @�{@@  @n{?�  Aj=qBL=q@@  @s�
?p��A0z�BN�H                                    Bx�Em�  �          @���@B�\@c33?��
AF�\BE�@B�\@g�?=p�ABG��                                    Bx�E|�  �          @���@E�@]p�?�(�Al(�BAQ�@E�@c33?n{A3�BD(�                                    Bx�E�*  T          @���@<��@_\)?�{A�
=BG  @<��@e?���AO�BJ33                                    Bx�E��  T          @���@?\)@_\)?��
Ax��BE�R@?\)@e�?z�HA>ffBH                                    Bx�E�v  
�          @���@Fff@XQ�?�\)A��HB>�@Fff@^�R?�=qAPz�BA�\                                    Bx�E�  
Z          @�\)@C33@O\)?���A�ffB:�H@C33@W
=?���A��
B?33                                    Bx�E��  T          @��
@G�@W
=?��A�p�B<z�@G�@_\)?���A�ffB@�R                                    Bx�E�h  
�          @�=q@C�
@Tz�?�z�A�
=B=G�@C�
@\��?�{A��BA�                                    Bx�E�  T          @���@C33@U?��A�Q�B>�@C33@_\)?�=qA�z�BC�\                                    Bx�E�  �          @���@@��@L��?�\)A��\B;{@@��@W
=?�=qA��HB@ff                                    Bx�F Z  �          @�ff@L(�@S33?�{A��RB7�@L(�@\��?ǮA�33B<�
                                    Bx�F   �          @�ff@L��@QG�?�\)A�p�B6ff@L��@Z�H?���A��
B;��                                    Bx�F�  �          @�ff@O\)@Q�?�A�33B5=q@O\)@[�?�  A��B:Q�                                    Bx�F,L  �          @��@HQ�@R�\?���A�
=B:
=@HQ�@\��?��A��\B?G�                                    Bx�F:�  T          @��@H��@P  ?�A�=qB8
=@H��@Z=q?�\)A��B=��                                    Bx�FI�  �          @�ff@J�H@QG�?�Q�A���B7ff@J�H@[�?У�A��B=
=                                    Bx�FX>  T          @�\)@C�
@\��?��A�33BA��@C�
@g
=?�ffA���BF�                                    Bx�Ff�  
�          @�@@��@Y��?�A���BB33@@��@dz�?˅A�z�BG�\                                    Bx�Fu�  �          @��@@  @W
=?��HA�Q�BA  @@  @a�?��A��BF��                                    Bx�F�0  �          @�@@  @W�?�p�A�BA\)@@  @b�\?�33A���BG
=                                    Bx�F��  T          @�@@  @Z=q?�
=A�(�BB��@@  @e�?˅A���BH�                                    Bx�F�|  �          @�@>{@^�R?���A��RBF
=@>{@h��?�G�A��RBK33                                    Bx�F�"  T          @�{@8Q�@c�
?�{A��\BLG�@8Q�@n{?�  A��BQG�                                    Bx�F��  �          @�@7
=@g�?޸RA�BN@7
=@qG�?���A�=qBS\)                                    Bx�F�n  �          @��@6ff@a�?޸RA�(�BL�@6ff@k�?��A��\BQQ�                                    Bx�F�  �          @�G�@0  @`��?޸RA�z�BOz�@0  @j�H?���A�Q�BTG�                                    Bx�F�  �          @�G�@,(�@hQ�?˅A��BV  @,(�@qG�?�(�Al��BZ(�                                    Bx�F�`  �          @�  @*�H@b�\?�Q�A�
=BS�@*�H@l��?�=qA��
BX�\                                    Bx�G  �          @�G�@-p�@aG�?��A��BQQ�@-p�@k�?�A�Q�BVQ�                                    Bx�G�  T          @�33@0  @e�?�\A��HBQ��@0  @o\)?��A�\)BV��                                    Bx�G%R  �          @��
@1G�@hQ�?��HA�  BR\)@1G�@q�?���A|(�BW                                      Bx�G3�  �          @�33@.�R@e?��
A�z�BS33@.�R@p��?�33A�=qBX�                                    Bx�GB�  �          @�z�@+�@hQ�?�z�A�BVQ�@+�@s�
?\A���B[�\                                    Bx�GQD  _          @�(�@(��@j=q?���A�z�BY
=@(��@u?��HA���B^
=                                    Bx�G_�  "          @�=q@#�
@o\)?У�A��B^p�@#�
@x��?�(�Ak
=Bb��                                    Bx�Gn�  �          @���@{@j=q?�(�A���B`
=@{@r�\?���AU��Bc�H                                    Bx�G}6  �          @��@p�@hQ�?�z�A�Q�B_��@p�@p��?�G�AJ�\BcQ�                                    Bx�G��  �          @��
@'�@]p�?ǮA�z�BSp�@'�@g
=?�
=AmG�BX
=                                    Bx�G��  �          @��@)��@\��?�A��BQ��@)��@g
=?��
A�BV�
                                    Bx�G�(  �          @��@1G�@Q�?�ffA�{BGQ�@1G�@]p�?�
=A�33BM=q                                    Bx�G��  �          @�z�@0��@Q�?�G�A���BG@0��@]p�?��A��BM�\                                    Bx�G�t  �          @���@3�
@P  ?��
A��BD�
@3�
@[�?�z�A��HBJ�
                                    Bx�G�  "          @�(�@,��@QG�?�{A�
=BI��@,��@]p�?�p�A�
=BO�
                                    Bx�G��  T          @��
@&ff@L(�@�A�  BK�@&ff@Y��?ٙ�A�p�BRQ�                                    Bx�G�f  �          @��
@-p�@HQ�@�\A�(�BDp�@-p�@U?�
=A�Q�BK��                                    Bx�H  "          @�33@0  @C�
@33A�{B@Q�@0  @QG�?�Q�A���BG��                                    Bx�H�  T          @�33@0  @=p�@(�A�{B<G�@0  @L(�?�A��HBD�H                                    Bx�HX  "          @��\@/\)@6ff@33A�B8�\@/\)@Fff?��HAʸRBB                                      Bx�H,�  �          @�G�@.�R@7
=@\)A�\B9\)@.�R@Fff?��A�G�BB��                                    Bx�H;�  �          @�Q�@1G�@1�@p�A뙚B4�\@1G�@AG�?�\)A��HB>                                      Bx�HJJ  �          @���@/\)@7�@�A�B9p�@/\)@G
=?���A�  BBp�                                    Bx�HX�  �          @���@.�R@<��@Q�A��B<��@.�R@K�?�G�A���BE\)                                    Bx�Hg�  �          @�G�@,��@;�@
=qA��
B=��@,��@J�H?�ffA��HBFff                                    Bx�Hv<  �          @���@0��@3�
@��A���B5��@0��@C�
?�z�A���B?�R                                    Bx�H��  �          @���@  @N{@
=qA��HB\33@  @\��?�G�A��Bc                                    Bx�H��  �          @�Q�@ ��@`  @   A��Bp�\@ ��@n{?���A��\Bvff                                    Bx�H�.  �          @�Q�@
=q@X��@�\A�p�Bf(�@
=q@g
=?�{A��
Bl�                                    Bx�H��  _          @��@��@Tz�@Q�A���Bd�H@��@c�
?��HA�33Bk��                                    Bx�H�z  �          @��H@Q�@Z=q@{A��Bg��@Q�@i��?��
A���Bo
=                                    Bx�H�   �          @���@{@\(�@��A�(�Bd��@{@l(�?�G�A��
Bk                                    Bx�H��  �          @���@�@Z=q@A�ffBhQ�@�@j�H?�33A�G�Bo�                                    Bx�H�l  "          @�z�@z�@X��@��A��RBjG�@z�@j=q?���A��Br
=                                    Bx�H�  �          @���?�
=@Z�H@   B�BrG�?�
=@l��@33A���Bz                                      Bx�I�  T          @�(�?�
=@XQ�@   B�Bp�H?�
=@j�H@33A�Q�Bx�
                                    Bx�I^  �          @��@   @U@�RB{Bl(�@   @g�@�A�p�Btff                                    Bx�I&  �          @��?��@S�
@   B{Bq�?��@fff@�
A֏\ByG�                                    Bx�I4�  T          @���?�(�@P  @��B\)Bkff?�(�@b�\@ ��A�p�Bs�H                                    Bx�ICP  �          @�p�@�@W�@ ��B��Bk�@�@j�H@33A�ffBs�H                                    Bx�IQ�  T          @�(�@�@U@{BG�Bj��@�@hQ�@G�A���Bs\)                                    Bx�I`�  �          @�Q�?��H@P  @(�B�Bkp�?��H@b�\?��RA�Bs��                                    Bx�IoB  �          @���@�@QG�@ffA���BfQ�@�@c33?��A��Bn�                                    Bx�I}�  �          @�  @z�@N{@
=A��Be{@z�@`  ?�z�A�G�Bm�R                                    Bx�I��  �          @�ff?��H@P��@33A��Bk�R?��H@a�?�A�(�Bs�
                                    Bx�I�4  �          @��@
�H@E�@0  B=qB[\)@
�H@Z�H@�
A�G�Bfff                                    Bx�I��  T          @�
=@ff@R�\@
�HA�  Be�R@ff@c�
?ٙ�A�G�Bm�                                    Bx�I��  I          @�=q?��H@I��@{A�Bh�?��H@Z�H?��
A�ffBpz�                                    Bx�I�&  
�          @��?��H@S�
@G�A�ffBz�R?��H@e?�ffA�z�B�                                    Bx�I��  �          @�?�\)@O\)@Q�B
=Bp(�?�\)@a�?�A��Bx��                                    Bx�I�r  �          @�z�@�@R�\@#33B�HBiG�@�@g
=@�A�\)Br��                                    Bx�I�  �          @�{@G�@R�\@)��B	�
Bi�
@G�@hQ�@
�HA��HBsz�                                    Bx�J�  �          @�p�@ff@I��@.�RB  Ba�@ff@`  @G�A�(�Bk��                                    Bx�Jd  �          @���@	��@J=q@)��B
=B_G�@	��@`  @(�A�(�Bi�H                                    Bx�J
  �          @��@�@J=q@)��B
p�B]��@�@`  @�A��HBh�                                    Bx�J-�  �          @�=q@�@AG�@)��B��BX�
@�@W
=@��A�{BdQ�                                    Bx�J<V  
�          @�{@ff@7
=@*=qB��BW
=@ff@Mp�@�RA�BcQ�                                    Bx�JJ�  �          @�G�?�p�@0��@(Q�B(�BYz�?�p�@G
=@p�A�{Bf
=                                    Bx�JY�  
�          @��@G�@$z�@+�B�BO=q@G�@;�@�\B
=B]z�                                    Bx�JhH  �          @�Q�@�@
�H@?\)B233B6��@�@%�@(��B�\BJQ�                                    Bx�Jv�  T          @�
=?��@  @A�B7�HBH�\?��@*=q@*�HB�B[33                                    Bx�J��  T          @�{@   @z�@5B+z�BD�@   @-p�@�RBp�BU��                                    Bx�J�:  �          @��?��@  @<��B4�\BH�R?��@*=q@%B��B[                                      Bx�J��  
�          @�(�?�
=@Q�@<��B6B?�?�
=@"�\@&ffB�
BR�
                                    Bx�J��  �          @���@@�@.�RB%�B=�
@@*=q@
=B�RBO{                                    Bx�J�,  �          @��\@�@(�@$z�B�BH�
@�@2�\@�B{BW�
                                    Bx�J��  "          @�=q@
=@@%�B��B?@
=@,(�@p�Bp�BO��                                    Bx�J�x  �          @�(�@�@{@"�\B{B,G�@�@$z�@�A�p�B==q                                    Bx�J�  "          @���@�@
=@!G�B
=B7�\@�@-p�@��A��HBG\)                                    Bx�J��  
�          @�z�@�@7�@��Bz�BN  @�@L��?�
=Aљ�BY�H                                    Bx�K	j  �          @���@�@K�?�ffAȣ�B^\)@�@Z=q?�ffA���Be�H                                    Bx�K  �          @�p�@G�@R�\?���Aƣ�B]�\@G�@a�?�=qA�z�Be{                                    Bx�K&�  "          @��@=q@>�R@ffA��
BLff@=q@S�
?�{A�{BW                                    Bx�K5\  �          @��
@�
@7�@
=B�RBL��@�
@L��?�33A�\)BX                                    Bx�KD  "          @�p�@
=q@0��@��B=qBO��@
=q@E�?�A͙�B[z�                                    Bx�KR�  �          @��@�
@6ff@�B�HBX�@�
@K�?�33A�
=BdG�                                    Bx�KaN  �          @�Q�?���@@  @�B
=Bc��?���@Tz�?��A�G�Bn�                                    Bx�Ko�  �          @�  @
=@@  @Q�A���B[=q@
=@R�\?У�A�G�Be33                                    Bx�K~�  �          @���@{@2�\@�Bz�BM�@{@G�?�{A�\)BZ                                      Bx�K�@  �          @���@�R@#33@'
=B�BC\)@�R@;�@�A�33BR�                                    Bx�K��  �          @���@�\@\)@)��B(�B=�@�\@8Q�@{A���BN=q                                    Bx�K��  
�          @�  @\)@(�@)��B�HB=z�@\)@5�@�RA�(�BNG�                                    Bx�K�2  �          @�  @�
@33@.{BQ�B3�\@�
@-p�@z�B\)BF
=                                    Bx�K��  �          @�\)@�\@��@1G�B#�HBH@�\@7
=@B�BY��                                    Bx�K�~  �          @��?��@�@:�HB.(�BQz�?��@7�@\)B  BcQ�                                    Bx�K�$  �          @�
=?˅@p�@N{BG33BY=q?˅@,��@4z�B(
=Bn33                                    Bx�K��  T          @��R?\@p�@P  BI��B^
=?\@-p�@5B*�Br�
                                    Bx�Lp  "          @��R?Ǯ@
=@G
=B?
=BbG�?Ǯ@5�@,(�B��Bt�
                                    Bx�L  !          @�z�?�  @&ff@>{B7z�B�?�  @C33@ ��B�B��=                                    Bx�L�  T          @�  ?˅@��@FffB:�BdQ�?˅@;�@)��B(�Bv{                                    Bx�L.b  T          @�ff?�G�@
=@G�B@�Be�?�G�@6ff@,(�B\)Bw��                                    Bx�L=  
�          @�  ?��H@	��@O\)BF��BM?��H@)��@5B'�
Bd�                                    Bx�LK�  �          @���?ٙ�?�z�@Z�HBTBA��?ٙ�@��@C33B6��B]=q                                    Bx�LZT  T          @���?��R@ ��@]p�BX\)BU�R?��R@#�
@Dz�B8G�Bo
=                                    Bx�Lh�  �          @�  ?�G�?�z�@^�RB\(�BN{?�G�@p�@G
=B<�\Bi�\                                    Bx�Lw�  �          @�?�z�?�@c�
Bi�HBG
=?�z�@\)@N�RBJ�\Bg=q                                    Bx�L�F  "          @�z�?�Q�?Ǯ@c�
Bl�HB=\)?�Q�@��@O\)BNQ�B`\)                                    Bx�L��  
�          @�?��?�\)@_\)Bc=qB\G�?��@(�@HQ�BB  Bw{                                    Bx�L��  ^          @���?�G�?��R@fffBe=qB!p�?�G�@�@R�\BIffBG\)                                    Bx�L�8  T          @�=q?޸R?�{@mp�Bm
=Bp�?޸R?�(�@[�BR
=BA�                                    Bx�L��  
�          @�{?��?���@e�Bi��B5�H?��@
=q@P  BKG�BY�                                    Bx�Lτ  "          @�33?�  ?�\)@Tz�BYQ�B+=q?�  @
�H@?\)B<G�BL\)                                    Bx�L�*  �          @�(�?�  ?�=q@_\)Bg�B  ?�  ?�z�@Mp�BL��B>G�                                    Bx�L��  "          @��H?�z�?��H@a�Bo=qB(�?�z�?�ff@QG�BT�\B=G�                                    Bx�L�v  
�          @�(�?�Q�?��H@dz�Bop�B��?�Q�?�@S�
BT�
B<=q                                    Bx�M
  T          @���?��?�G�@fffBq
=A�?��?�\)@W�BYQ�B(�
                                    Bx�M�  �          @��
?��?c�
@fffBsA�ff?��?�G�@X��B]ffB �                                    Bx�M'h  �          @�?�G�?O\)@l��Byz�A��
?�G�?���@`  BcB��                                    Bx�M6  �          @��
?�=q?�\@i��ByAz{?�=q?�33@`  Bi�B =q                                    Bx�MD�  "          @��\?˅?W
=@i��B  Aݮ?˅?�(�@\��BgG�B*�R                                    Bx�MSZ  �          @\)?˅?#�
@fffB��A�
=?˅?��\@[�Bm\)BG�                                    Bx�Mb   ^          @z�H?�\)>��H@a�B��)A��H?�\)?�{@X��Bo�B

=                                    Bx�Mp�  �          @�G�?ٙ�?Y��@dz�Bw�HAԏ\?ٙ�?�(�@W
=B`B#��                                    Bx�ML  �          @�G�?�p�?s33@hQ�B�W
B�?�p�?˅@Y��Be�HB<{                                    Bx�M��  �          @�=q?��?��@c�
Bs�Bff?��?�(�@S33BY{B8�
                                    Bx�M��  �          @�
=?�?��
@j�HBq�A�\?�?�
=@Z�HBXB+                                    Bx�M�>  �          @��?��?:�H@k�B�=qA�p�?��?��@^�RBl��B(
=                                    Bx�M��  �          @}p�?��?c�
@hQ�B�W
B=q?��?��@Z=qBn
=BH�\                                    Bx�MȊ  �          @vff?�33>��@hQ�B��HA�p�?�33?���@`  B��3B+��                                    Bx�M�0  T          @n{?fff�#�
@eB�k�C��?fff?   @c�
B�A��                                    Bx�M��  
�          @p��?�p�=#�
@c�
B��H@?�p�?333@_\)B��fA���                                    Bx�M�|  �          @�G�?�
=>�\)@j=qB���A�R?�
=?s33@b�\Bu�\A�p�                                    Bx�N"  �          @��?�  =�Q�@i��B�Ǯ@?\)?�  ?E�@dz�Bw�A�G�                                    Bx�N�  �          @\)?z�H�L��@p  B�33C�<)?z�H?   @n{B��A�
=                                    Bx�N n  �          @x��?����33@s33B�Q�C��\?��>�p�@s33B�
=A�33                                    Bx�N/  �          @z�H?��;aG�@j=qB��C�t{?���>��@h��B���A�                                    Bx�N=�  �          @�(�?�\>���@k�B��A�\?�\?}p�@c33Bq33A���                                    Bx�NL`  T          @o\)?�(�>�@Y��B�L�A���?�(�?���@P  Brp�B(�                                    Bx�N[  �          @qG�?�p�?k�@Mp�Bj�A��?�p�?�G�@>�RBQ�B$�                                    Bx�Ni�  T          @n�R@ ��?n{@>�RBUG�A�
=@ ��?�p�@0  B>�HB\)                                    Bx�NxR  �          @qG�@{?=p�@<(�BN�A��@{?��@0  B;��A�R                                    Bx�N��  	�          @b�\?�33>Ǯ@>{Bc(�A7\)?�33?s33@5BTA�33                                    Bx�N��  �          @dz�?�(�>��@;�B]ffAW
=?�(�?��\@2�\BM�
Aۅ                                    Bx�N�D  "          @fff?�Q�?   @?\)B`�Ah��?�Q�?���@5�BP(�A���                                    Bx�N��  T          @c�
?�(�?#�
@B�\Bk�A�p�?�(�?��H@7
=BV��B{                                    Bx�N��  �          @l(�?�p�?J=q@P��Bz��A���?�p�?�z�@C33B`��B.
=                                    Bx�N�6  T          @n{?�33?�ff@W�B�aHB*ff?�33?�
=@FffBb�B_33                                    Bx�N��  "          @l��@\)>�ff@,��B;p�A$  @\)?xQ�@#�
B/33A��                                    Bx�N�  �          @p��@$z�?��@*�HB5�RAA�@$z�?��@ ��B(Q�A��                                    Bx�N�(  "          @n{@{>�p�@0  B?33A	p�@{?fff@(Q�B3��A��H                                    Bx�O
�  �          @l��@\)>��
@:�HBP�A=q@\)?c�
@333BD�
A�z�                                    Bx�Ot  �          @h��?��H?
=@@  B^z�A�Q�?��H?�@4z�BL  A�ff                                    Bx�O(  �          @n�R?���?
=q@H��Bd�RAv{?���?��@>{BR��A�33                                    Bx�O6�  �          @o\)?�{?��@L(�Bj��A��H?�{?�@@��BW�RB =q                                    Bx�OEf  �          @hQ�?�Q�>�@I��Bsz�A
=?�Q�?���@?\)B`=qB�                                    Bx�OT  �          @k�?�p�<��
@EBe(�?   ?�p�?!G�@AG�B]��A�G�                                    Bx�Ob�  T          @u@�\?#�
@9��BJ�
Az�\@�\?��H@.{B9��A޸R                                    Bx�OqX  �          @w�@(�?^�R@>�RBN�\A�G�@(�?���@0  B9
=B�H                                    Bx�O�  ^          @}p�@
�H?˅@9��B<�RB@
�H@��@!G�B��B2G�                                    Bx�O��  T          @w
=@�?�{@2�\B9��B�
@�@��@=qB�HB5{                                    Bx�O�J  �          @vff?��H?��@2�\B9�B*{?��H@z�@�B��BGff                                    Bx�O��  �          @w
=@ ��?�@0��B6(�B'�@ ��@�@�B�BD�
                                    Bx�O��  �          @u�?�p�?�{@,(�B2�
B-  ?�p�@�@��B��BH�                                    Bx�O�<  �          @vff@   ?��@,��B2{B,�\@   @��@��B�
BH                                      Bx�O��  �          @�p�?�\)@<(�@B\)Bf(�?�\)@W
=?�Q�A��Bsz�                                    Bx�O�  �          @xQ�?�\)@*�H@�B�BkG�?�\)@E�?�
=AΏ\By=q                                    Bx�O�.  �          @u�?��R@��@%B*�\Bg�H?��R@7�@33B\)Bz                                      Bx�P�  T          @xQ�?�(�@{@3�
B:Q�Bb�?�(�@0  @�\BffBw�                                    Bx�Pz  �          @|��?��\@*=q@.{B.�HB�
=?��\@J=q@�B�B�G�                                    Bx�P!   �          @y��?�
=@#�
@,��B.�B�aH?�
=@C�
@
=B�B��q                                    Bx�P/�  �          @xQ�?�ff@@333B8Bs��?�ff@7�@  B�B�.                                    Bx�P>l  �          @w
=?���@�
@1�B8(�Bn��?���@5@\)B��B�.                                    Bx�PM  �          @qG�?��@G�@1�B>Q�BR�?��@#�
@33B\)Bk�
                                    Bx�P[�  �          @dz�@��?�R@,��BI�RA�p�@��?�@ ��B7�A�                                    Bx�Pj^  �          @dz�@(�?L��@+�BD33A���@(�?���@p�B.�A��H                                    Bx�Py  �          @e�?�z�?xQ�@6ffBTG�A�p�?�z�?�ff@%B9��B{                                    Bx�P��  �          @e�?�33?��@3�
BP�A��?�33?�z�@!G�B3��B$�\                                    Bx�P�P  �          @b�\@�?xQ�@!G�B5��A�
=@�?�p�@��B��BQ�                                    Bx�P��  �          @Y��@�H=�Q�@�B1�
@�@�H?
=@33B+
=A[�                                    Bx�P��  T          @Y��@$z�#�
@	��Bp�C�9�@$z�>���@��B{@�p�                                    Bx�P�B  �          @`  @1G���=q@�BQ�C�.@1G�>8Q�@Q�B�@tz�                                    Bx�P��  T          @`��@2�\�\)@
=B�C��@2�\>��R@B��@�z�                                    Bx�Pߎ  �          @dz�@�R>���@!�B5G�@��@�R?\(�@��B)�RA��                                    Bx�P�4  �          @Vff@�
>\@��B6\)Aff@�
?c�
@  B)G�A��                                    Bx�P��  T          @[�@��>�33@�HB4�A{@��?\(�@�B'�HA���                                    Bx�Q�  �          @Y��@��?\)@!G�B?��Aep�@��?��@B.G�A�33                                    Bx�Q&  
�          @Y��@33?(�@(Q�BK�A��@33?�@(�B7��A�                                    Bx�Q(�  �          @[�@��>�(�@%BD��A/�
@��?z�H@(�B5��A�                                      Bx�Q7r  "          @s�
@-p���Q�@*=qB1��C��@-p�>��H@'�B.{A$z�                                    Bx�QF  T          @vff@%�#�
@5B>=qC��@%?(�@1G�B8p�ATz�                                    Bx�QT�  �          @tz�@'�    @.�RB8��<�@'�?��@*=qB3{AM��                                    Bx�Qcd  
�          @n�R@)����@'�B2��C���@)��?
=q@#�
B-A9G�                                    Bx�Qr
  �          @r�\@8�ü�@(�B �C�� @8��?�\@Q�BG�A                                       Bx�Q��  �          @r�\@4z�#�
@!�B'z�C�W
@4z�>���@ ��B%\)A (�                                    Bx�Q�V  �          @qG�@6ff���
@�B!(�C��H@6ff>aG�@(�B"{@���                                    Bx�Q��  �          @s�
@/\)�L��@'�B.C��\@/\)>Ǯ@&ffB-  A Q�                                    Bx�Q��  �          @s�
@(Q�k�@.�RB8
=C�y�@(Q�>\@.{B6��A
=                                    Bx�Q�H  �          @mp�@0  �
=q@�HB#
=C�u�@0  <#�
@�RB'�>aG�                                    Bx�Q��  
�          @n{@333�z�@
=B(�C�&f@333�#�
@�B#��C���                                    Bx�Qؔ  �          @g�@.{�z�@�B��C��@.{�L��@
=B#�C�t{                                    Bx�Q�:  T          @k�@�=�Q�@/\)BA�@��@�?333@)��B9ffA���                                    Bx�Q��  �          @mp�@.{����@�RB(Q�C���@.{>k�@\)B)=q@�(�                                    Bx�R�  T          @o\)@7��Tz�@\)BC���@7�����@�BC���                                    Bx�R,  �          @n�R@\)�\@.{B<C�� @\)>u@/\)B>G�@��R                                    Bx�R!�  �          @n{@�<#�
@:�HBO�R>��@�?+�@5BH�A�=q                                    Bx�R0x  T          @g
=@*=q���
@=qB'��C��@*=q>k�@�HB(@�z�                                    Bx�R?  "          @p��?�(�>Ǯ@HQ�Bd��A4��?�(�?��@=p�BR�A癚                                    Bx�RM�  �          @tz�?�\?�@UBt��A�(�?�\?�G�@H��B]\)B                                      Bx�R\j  
�          @�  ?�z�>�Q�@_\)BsffA,Q�?�z�?�33@Tz�B`Q�A��                                    Bx�Rk  
Z          @~�R?�Q�>\)@eB�u�@�z�?�Q�?s33@^{Bs�A�
=                                    Bx�Ry�  T          @s33@p�>��@?\)BT�\A&ff@p�?�=q@4z�BC�HA�{                                    Bx�R�\  
�          @o\)@1G�?E�@Q�Bz�AxQ�@1G�?�ff@��B
=A�(�                                    Bx�R�  T          @k�@AG�?�\)?�G�A�
=A��R@AG�?��R?���A��HA�ff                                    Bx�R��  "          @��
@��>B�\@_\)Bf�
@��H@��?z�H@W
=BY\)A���                                    Bx�R�N  �          @�=q@p�>���@XQ�BbQ�@�{@p�?���@N{BRA�Q�                                    Bx�R��  
�          @���@��?��@Q�BY��An�\@��?��@C33BD�HA�G�                                    Bx�Rњ  T          @z=q@ff?!G�@L��B^33A��@ff?���@=p�BG��B��                                    Bx�R�@  �          @x��@�?��@<(�BEffA�p�@�?޸R@'
=B)�B�                                    Bx�R��  �          @xQ�@(�?���@)��B,33A�Q�@(�@�@\)B{B                                      Bx�R��  T          @z=q@�?�z�@:=qB@�
B�@�@33@   Bz�B,�
                                    Bx�S2  
�          @|(�@
=q?��@?\)BG�A�Q�@
=q?�(�@'
=B'  B)                                    Bx�S�  �          @y��@��@�@�B��B�\@��@{?˅A��HB4�H                                    Bx�S)~  �          @��@ ��@#�
?�33A��B6G�@ ��@;�?�  A�33BE                                    Bx�S8$  �          @��@p�@$z�?�(�A��B933@p�@=p�?��A���BI=q                                    Bx�SF�  T          @�G�@%@{@�B�HB.��@%@<(�?��A�G�BBQ�                                    Bx�SUp  �          @��@,(�@>{@�A�33B?(�@,(�@Z�H?\A�  BO33                                    Bx�Sd  �          @�ff@3�
@8��@=qA��B7z�@3�
@XQ�?�z�A��
BI(�                                    Bx�Sr�  �          @�p�@=p�@.�R@
=A���B*��@=p�@Mp�?�33A���B=�                                    Bx�S�b  �          @�ff@AG�@'
=@{A�z�B#Q�@AG�@G�?��A��B7                                    Bx�S�  �          @�p�@?\)@'�@p�A��\B$@?\)@G�?��
A��\B9{                                    Bx�S��  "          @��@;�@!�@��A�(�B#
=@;�@A�?�p�A��\B7�                                    Bx�S�T  �          @�G�@/\)@%�@"�\B�
B-=q@/\)@G
=?���A�Q�BB�                                    Bx�S��  �          @���@1�@'
=@(�BffB,�H@1�@G
=?�  A���BA�                                    Bx�Sʠ  �          @�@/\)@"�\@��B�HB+�@/\)@B�\?�p�A��HB@                                      Bx�S�F  �          @�
=@+�@�
@�\Bp�B"��@+�@1�?�A��RB833                                    Bx�S��  �          @��@&ff@��@Q�B
=B �H@&ff@,��?��A�  B8�                                    Bx�S��  �          @���@'
=@Q�@=qB(�B  @'
=@)��?�=qA�\)B5��                                    Bx�T8  "          @�{@!�@(�@!�Bz�B#ff@!�@.�R?�
=A�p�B<                                    Bx�T�  �          @��R@��@G�@'�B  B-�@��@5�@   A���BGG�                                    Bx�T"�  "          @�@�@  @(Q�BffB/��@�@4z�@ ��A�RBIz�                                    Bx�T1*  �          @��R@�@�\@(��B�
B2{@�@7
=@ ��A���BK\)                                    Bx�T?�  �          @�ff@33@Q�@%�B�B8  @33@<(�?�
=A�ffBO��                                    Bx�TNv  "          @��R@�@(�@#33B\)B<
=@�@?\)?��AԸRBR�R                                    Bx�T]  �          @�ff@�@'�@!�BG�BN(�@�@J=q?���A�p�Bb\)                                    Bx�Tk�  �          @�ff@�@&ff@p�B�HBHQ�@�@G�?�G�A�=qB\p�                                    Bx�Tzh  �          @�@
=q@%@{B��BI{@
=q@G�?�G�AǮB]Q�                                    Bx�T�  �          @���@��@p�@   B��B@��@��@@  ?�=qA�G�BV��                                    Bx�T��  
�          @��
@��@\)@�RBz�BE�@��@AG�?�ffAϙ�BZ��                                    Bx�T�Z  �          @�z�@	��@�@$z�B��BB
=@	��@>�R?�33Aڏ\BY                                      Bx�T�   �          @tz�@z�@33@p�B �
B3�@z�@%?��A���BM�                                    Bx�Tæ  T          @qG�?��@@�B!��B[�H?��@7
=?��
A��Bp��                                    Bx�T�L  �          @�Q�?�z�@ ��@(Q�B$��Ba�?�z�@E�?�
=A�Bv�                                    Bx�T��  �          @�
=@ff@z�@5�B(��B?��@ff@<��@
�HA�{BZ=q                                    Bx�T�  T          @�
=@Q�@{@8Q�B,�B8Q�@Q�@7
=@  B �HBT��                                    Bx�T�>  T          @���@
�H@��@4z�B*�B2p�@
�H@1G�@��B {BO��                                    Bx�U�  �          @�@��@�@6ffB,  B0{@��@0��@\)Bp�BM��                                    Bx�U�  T          @���@�\@�
@1�B/33B5�@�\@,(�@�B�RBS\)                                    Bx�U*0  T          @��H@�@�
@4z�B.\)B1(�@�@,��@{Bz�BO�\                                    Bx�U8�  �          @���@
=@G�@1�B.p�B/{@
=@(��@(�B��BM�
                                    Bx�UG|  |          @��
@��@�@5B.�B0�@��@.{@�RB33BO                                      Bx�UV"  �          @��\@ff@p�@-p�B&ffB9@ff@4z�@z�A�Q�BU                                      Bx�Ud�  �          @���@@�R@%B!G�B;  @@333?��HA��
BT�                                    Bx�Usn  �          @h��?�?�z�@�B'�B2��?�@��?��A�Q�BO�\                                    Bx�U�  �          @c33?�33?ٙ�@p�B/��B'ff?�33@��?��HB
=BG�H                                    Bx�U��  �          @h��?���?��H@�B(�B:��?���@   ?�\)A�BVQ�                                    Bx�U�`  "          @k�?�
=@(�@�RB��BB{?�
=@+�?�{A�Q�BX��                                    Bx�U�  
�          @j�H?��@�@z�Bp�BA�
?��@(��?�(�Aߙ�BZ{                                    Bx�U��  "          @j�H?�33@�@
=B��B>33?�33@&ff?�G�A�33BWz�                                    Bx�U�R  
�          @g�?�?�
=@Q�B$��B4Q�?�@p�?���A��
BP33                                    Bx�U��  �          @e?�Q�?��@�B%33B0��?�Q�@�H?���A�BM(�                                    Bx�U�  
�          @dz�?�33?�z�@B#�B4�H?�33@(�?��
A�BP�                                    Bx�U�D  "          @e@   ?�p�@�B*\)B"�@   @�?�B ��BB�                                    Bx�V�  �          @g
=?�?�\)@ ��B/��B7��?�@(�?��HB{BU�
                                    Bx�V�  �          @e�?�\)?�{@�HB*
=B2�?�\)@=q?�\)A��BPp�                                    Bx�V#6  T          @i��?�\@�@p�B)�RBC��?�\@%?�\)A�=qB^z�                                    Bx�V1�  T          @h��?�(�@z�@��B)  BI�?�(�@(Q�?�A��Bb��                                    Bx�V@�  �          @j=q?��H@�@!G�B.Q�BG�H?��H@'
=?�
=A�Bc�                                    Bx�VO(  �          @k�?��H?�p�@��B"�B5\)?��H@!�?���A���BP�H                                    Bx�V]�  T          @j=q?��?�33@\)B+=qB4�R?��@�R?�A��BRff                                    Bx�Vlt  T          @i��?�{?�z�@   B,��B7\)?�{@�R?�
=A��HBT��                                    Bx�V{  T          @j�H?�\?�
=@$z�B2
=B>{?�\@!G�?��RB�B[��                                    Bx�V��  "          @xQ�?ٙ�@ff@2�\B7��BL
=?ٙ�@/\)@
=qB�Bh�
                                    Bx�V�f  �          @n{?��
@ ��@.�RB=
=BR��?��
@(��@Q�B�Bo�                                    Bx�V�  �          @l(�?�(�@   @.�RB?(�BW{?�(�@(��@Q�B  Bs��                                    Bx�V��  �          @h��?�p�?�@-p�B@��BQ��?�p�@#33@Q�BffBo��                                    Bx�V�X  T          @j=q?�  ?�@.{B@�\BP{?�  @#�
@��B�Bn\)                                    Bx�V��  
�          @n{?˅?��@1�BA�BGff?˅@"�\@��BQ�Bg��                                    Bx�V�  "          @w�?�p�?�Q�@7�B?z�BA=q?�p�@'
=@G�B  Bb�                                    Bx�V�J  �          @|��?�z�?�(�@5B7��B7�\?�z�@(��@�RB	�
BX=q                                    Bx�V��  �          @|��?�
=?��@7�B;(�B1(�?�
=@#�
@�\B
=BS�                                    Bx�W�  
�          @}p�?�p�?�{@A�BG�B�?�p�@�@ ��Bz�BG
=                                    Bx�W<  �          @vff?�z�?��H@*�HB1G�B733?�z�@%�@z�B\)BVQ�                                    Bx�W*�  |          @p  ?��
@
�H@�RB%��BJff?��
@.�R?�=qA�z�Bc�R                                    Bx�W9�  
Z          @o\)?���?��@*�HB5��B7��?���@!G�@BBX33                                    Bx�WH.  	�          @z�H?�
=@�@,��B.\)B<{?�
=@-p�@z�A�G�BY�R                                    Bx�WV�  "          @|(�?�p�@�
@.{B.��B8�\?�p�@,(�@ffB ffBV�H                                    Bx�Wez  
�          @z�H?��@�@)��B*�BD�?��@2�\@   A��B`\)                                    Bx�Wt   T          @mp�@?�\)@#�
B/��B��@@�R@33B(�B;(�                                    Bx�W��  "          @hQ�?��?�\@,(�B@G�B=(�?��@=q@Q�B��B_Q�                                    Bx�W�l  T          @dz�?�z�?Ǯ@#�
B8ffB33?�z�@�@z�B(�BB��                                    Bx�W�  �          @XQ�@�?���@(�B#�A�z�@�?��?���B��B�                                    Bx�W��  �          @S33@�H?333@�
B=qA�z�@�H?��H?�B
=Aԣ�                                   Bx�W�^  �          @\(�@,��>�p�@�
B(�@�  @,��?c�
?��BffA�=q                                   Bx�W�  �          @U@+���?��RB33C��\@+�>��H?�B��A&{                                    Bx�Wڪ  �          @XQ�@0�׾�  ?�
=BG�C�e@0��>��?�
=B(�@�p�                                    Bx�W�P  �          @S33@)���#�
?�
=Bz�C���@)��>��?��B�A!�                                    Bx�W��  �          @K�@ff?xQ�@�B-Q�A��H@ff?�G�?���BffB                                      Bx�X�  �          @>{?��?��H@�\B-{B33?��?ٙ�?�33B�B.(�                                    Bx�XB  �          @HQ�@�\?�G�@��B-
=A�G�@�\?��?�ffB�RB=q                                    Bx�X#�  �          @L(�@��?aG�@(�B-ffA��H@��?�
=?��B�B�                                    Bx�X2�  
�          @W�@�?fff@  B(G�A�(�@�?��H?�Q�B��B 
=                                    Bx�XA4  	�          @S33@��?z�H@��B&�RA��@��?��
?�{B	�B
=                                    Bx�XO�  �          @C33@33?=p�@Q�B1�\A��@33?��
?�{B�B                                       Bx�X^�  
          @E@
=q?333@B*p�A��@
=q?�p�?�=qB�A���                                    Bx�Xm&  �          @Fff@
=?E�@��B.�\A�z�@
=?��?�{B�A�z�                                    Bx�X{�  �          @A�@�?E�@�
B+G�A��@�?��
?��
B33A�33                                    Bx�X�r  �          @AG�@?#�
@�B.33A�\)@?�z�?�B  A�\                                    Bx�X�  �          @?\)@z�>��@ffB2�RAL��@z�?�G�?�33B{A��
                                    Bx�X��            @<��@�>�(�@B5  A>{@�?u?�33B!�A˅                                    Bx�X�d  �          @E@ ��>�  @�BC�@�Q�@ ��?Y��@�B3�A�                                      Bx�X�
  �          @G
=@��>k�@\)B8��@�
=@��?Q�@ffB)��A��                                    Bx�XӰ  �          @Fff@�>8Q�@  B:33@��@�?E�@�B,�A���                                    Bx�X�V  ^          @Fff@�>8Q�@  B:z�@�\)@�?E�@�B-(�A�
=                                    Bx�X��  T          @@  ?�Q�>k�@�BE�H@���?�Q�?Q�@��B5�
A�=q                                    Bx�X��  �          @?\)?�\)>\@33BH��A7\)?�\)?z�H@
=B3�A�{                                    Bx�YH  �          @C�
?��>�(�@
=BJ�ALz�?��?��@
=qB4��A�G�                                    Bx�Y�  �          @H��?���>�@��BG�\A[�?���?���@
�HB0��A�=q                                    Bx�Y+�  �          @.{?�\)���?�B,��C�R?�\)�\)@�
BH��C��H                                    Bx�Y::  T          @;�?�\�8Q�@��BD(�C�f?�\����@�
BQ�C�Z�                                    Bx�YH�  �          @=p�@G��n{?�B"�
C�� @G��Ǯ@ffB6  C���                                    Bx�YW�  �          @?\)?�33��\)@�BG��C���?�33>�{@�BF�HA!�                                    Bx�Yf,  �          @<��?��R����@Q�B9p�C�O\?��R>8Q�@
=qB<��@�\)                                    Bx�Yt�  �          @HQ�@�;���@�B1  C��@��>L��@p�B3@�p�                                    Bx�Y�x  �          @G�@  ���
@
�HB/�RC�
=@  ?�\@
=B)��AK
=                                    Bx�Y�  �          @:�H?�\)�.{@�RBG(�C�ff?�\)>�(�@(�BB�HAO33                                    Bx�Y��  �          @E?޸R��Q�@#33B^Q�C�u�?޸R?
=@�RBU�A��
                                    Bx�Y�j  T          @H��?�
=����@&ffBaC�:�?�
=>��R@'�Bc�A%                                    Bx�Y�  �          @>�R@)��?   ?��RA��A*�H@)��?O\)?��A��A�ff                                    Bx�Y̶  �          @6ff@�H>��
?�z�A�ff@���@�H?0��?��
A�
=A}p�                                    Bx�Y�\  �          @2�\@=q=�?��A�=q@0  @=q>�?��A���A6=q                                    Bx�Y�  
�          @7
=@"�\>���?�(�A��A=q@"�\?5?���A�G�Az=q                                    Bx�Y��  J          @1�@�H>��
?���A��@�@�H?(��?�
=A�=qAu�                                    Bx�ZN  
�          @:=q@#�
���
?���A���C���@#�
>�{?��A��@�                                    Bx�Z�  �          @E�@!�>B�\?޸RB	�@���@!�?&ff?�\)A��\Af=q                                    Bx�Z$�  T          @@��@Q�?#�
?�(�B�RAo\)@Q�?���?�p�A�33A�z�                                    Bx�Z3@  �          @P��@#�
?�R?��Bz�AX��@#�
?���?�33A�33A�=q                                    Bx�ZA�  �          @QG�@#�
?0��?�z�B�
Ar�\@#�
?�
=?�33A�RA�{                                    Bx�ZP�  �          @E�@
=?(��?��B
=A{33@
=?��?У�A�\)A�
=                                    Bx�Z_2  �          @7�@��?aG�?�
=B33A�p�@��?�ff?���A�p�A��H                                    Bx�Zm�  T          @E�@(�?��?�=qB�A�p�@(�?˅?���A�B��                                    Bx�Z|~  T          @QG�@�H?�?�33A�(�A�Q�@�H?�?��HA���B�                                    Bx�Z�$  T          @W�@�R?�Q�?�
=BQ�A�z�@�R?�z�?��A�=qB\)                                    Bx�Z��  �          @k�@.�R?��@G�B��A�=q@.�R?��?���AʸRB
�\                                    Bx�Z�p  T          @a�@Q�?�G�@G�B ��A��@Q�?���?���A�33B��                                    Bx�Z�  �          @`  @Q�?�z�@�\B#�\AУ�@Q�?޸R?�33B�B�                                    Bx�Zż  
�          @l��@p�?�33@ ��B+(�A���@p�?��@
=B
=B
=                                    Bx�Z�b  �          @j�H@{?��
@ ��B,A�@{?�@��B�B�                                    Bx�Z�  �          @c33@�?�33@ffB&�
Aϙ�@�?�  ?���B(�Bp�                                    Bx�Z�  �          @i��@�?�p�@B�\B (�@�@�
?���A�  B#��                                    Bx�[ T  �          @k�@(�?��@�B ��A��@(�?�p�?�z�A��
BG�                                    Bx�[�  �          @w
=@%�?�Q�@p�B�HA���@%�@33?�p�A���B��                                    Bx�[�  �          @j=q@   ?�  @�B!Q�A���@   ?�?���B G�B��                                    Bx�[,F  �          @U@	��?z�@\)B@��Ap(�@	��?�p�@\)B(ffA��                                    Bx�[:�  �          @O\)@   >�\)@!G�BMG�A{@   ?s33@ffB:Ȁ\                                    Bx�[I�  �          @O\)@   >���@!G�BL�A\)@   ?�  @B8��A���                                    Bx�[X8  �          @k�@Q�>�
=@=p�BV��A2�\@Q�?���@.�RB@z�A뙚                                    Bx�[f�  �          @vff@(�>�\)@I��B[��@�33@(�?�\)@<��BH��A�33                                    Bx�[u�  �          @p��@��?333@9��BKG�A�  @��?��H@&ffB0Q�B�                                    Bx�[�*  
�          @dz�@�?p��@)��B@�HA�{@�?�\)@33B!33Bz�                                    Bx�[��  �          @k�@?Q�@.{B>��A�z�@?\@��B"�\Bff                                    Bx�[�v  �          @q�@�>�ff@:�HBI�
A+�@�?�(�@,(�B5
=A�=q                                    Bx�[�  �          @x��@�
?
=@Dz�BP�RAf=q@�
?�33@2�\B7��A�                                      Bx�[��  �          @u�@�?(��@;�BH
=Az=q@�?�
=@(��B.�RA�33                                    Bx�[�h  �          @�Q�@�>���@Mp�BTz�A@�?�G�@>�RB?�A���                                    Bx�[�  �          @���@�H>���@L��BR�@�Q�@�H?�33@@  BA  A��                                    Bx�[�  �          @j�H@ff=#�
@?\)B[�?�=q@ff?Y��@7�BO
=A�(�                                    Bx�[�Z  �          @hQ�@��?
=@5�BLAn�H@��?�=q@#�
B3A�\)                                    Bx�\   �          @n{@�R>�Q�@<(�BQ�A33@�R?��@.�RB=A���                                    Bx�\�  �          @x��@{?�Q�@9��BDG�A�Q�@{?�Q�@{B (�B$p�                                    Bx�\%L  �          @��@�?У�@9��B6�B��@�@ff@ffB\)B4�                                    Bx�\3�  �          @���@
=?�\)@>�RB=��A�R@
=@Q�@   B�B'�H                                    Bx�\B�  �          @�=q@�?�(�@A�BA
=B�@�@�R@!G�Bz�B1��                                    Bx�\Q>  �          @\)@z�?\@5B6B
=@z�@�R@z�B=qB/�                                    Bx�\_�  �          @xQ�@�?\@+�B/��B@�@(�@
=qBz�B-
=                                    Bx�\n�  �          @�=q@?�ff@N�RBR�\A��R@@Q�@0��B*�B5�R                                    Bx�\}0  �          @p  ?�(�?��@@��BU\)A��H?�(�?�@'
=B0G�B,
=                                    Bx�\��  �          @vff?���?�Q�@E�BU\)A�
=?���?�p�@(��B-�HB5z�                                    Bx�\�|  �          @|(�@ ��?�@E�BM\)B=q@ ��@��@%�B#\)B>�                                    Bx�\�"  �          @w�?��H?�\)@B�\BO�Bz�?��H@��@#�
B%B>                                      Bx�\��  �          @��?�\)?���@G
=BD\)B8��?�\)@.{@{B\)B^                                      Bx�\�n  T          @�?�=q@2�\@C33B.G�Bq�?�=q@`��@�A�=qB��=                                    Bx�\�  T          @�
=?Ǯ@0��@J�HB4{Br33?Ǯ@`��@33A���B�.                                    Bx�\�  �          @��R?�(�@�R@R�\B=�HB]33?�(�@QG�@\)B  Byff                                    Bx�\�`  �          @�{?���@�@S�
BP�RBQG�?���@6ff@(Q�B��Bt�                                    Bx�]  �          @��\?���?��@S33BW�RBA{?���@'�@,(�B%  BjQ�                                    Bx�]�  �          @�{?�(�@   @P��BK�BD�
?�(�@3�
@&ffB  Bi�                                    Bx�]R  �          @��H@�\?�@A�B@
=B&z�@�\@#�
@�B(�BN                                      Bx�],�  �          @�33@G�?�@Dz�BB\)B'{@G�@$z�@{BQ�BO33                                    Bx�];�  �          @�?���?�Q�@J=qBD  B3��?���@.{@!G�B�BY�R                                    Bx�]JD  �          @�=q?�{?���@G
=BG�B2�
?�{@'�@   Bp�BZ(�                                    Bx�]X�  �          @mp�@�
?c�
@/\)B?�A�@�
?˅@��B!��B
G�                                    Bx�]g�  �          @vff@{?#�
@7�BAG�Af=q@{?��@%B)��A�{                                    Bx�]v6  �          @~�R@?��@G�BN��A���@@@)��B'�\B3��                                    Bx�]��  �          @�Q�@ff?�\)@N{BV{A߅@ff?�Q�@333B1ffB*��                                    Bx�]��  T          @�Q�@�?�\)@P��BZ=qA�=q@�?��H@5B4��B/�                                    Bx�]�(  �          @~�R@   ?��@QG�B]Q�A�@   ?�33@7
=B833B.z�                                    Bx�]��  �          @l(�@�?�@ffB!�B$=q@�@Q�?��
A��HBC\)                                    Bx�]�t  �          @h��@�?�@�B{B*�@�@{?�A�G�BGQ�                                    Bx�]�  �          @fff?���@\)@
=B\)BK�?���@.�R?�A�B`�                                    Bx�]��  �          @hQ�?�Q�@ ��@ ��B=qB_?�Q�@<��?�G�A�(�Bp��                                    Bx�]�f  �          @aG�?�
=@0��?У�A�33Bz(�?�
=@E?Q�AYB�L�                                    Bx�]�  �          @dz�?�G�?�ff@!�B4B6p�?�G�@=q?���B�
BW��                                    Bx�^�  �          @c�
?�@   @�HB+��BHG�?�@$z�?��A�ffBd
=                                    Bx�^X  �          @a�?˅@
=@�B%�HBS�R?˅@*=q?�
=A��
BlQ�                                    Bx�^%�  �          @e?���@��@33B�B\�?���@2�\?�{A�G�Br��                                    Bx�^4�  �          @fff?\@'
=?�
=BffBn��?\@A�?�z�A��HB}ff                                    Bx�^CJ  T          @j=q?��R@&ff@�B\)Bp�?��R@C�
?��A�=qB�8R                                    Bx�^Q�  �          @g�?�(�@"�\@ffB
=Bo��?�(�@@��?��A��
B�H                                    Bx�^`�  �          @g
=?�  @�R@��B��BkQ�?�  @>{?�33A�ffB|��                                    Bx�^o<  �          @dz�?˅@(�@z�B�\Bc��?˅@9��?��A�  Buff                                    Bx�^}�  �          @aG�?�Q�?޸R@ffB'�HB'�?�Q�@�
?��A��
BG��                                    Bx�^��  �          @b�\@33?�ff@Q�B+ffB@33@��?��BB8�
                                    Bx�^�.  �          @e@   ?�  @��B(�B%�@   @�?�A�G�BE��                                    Bx�^��  T          @j�H@Q�?���@BB$p�@Q�@=q?�G�A�\)BB�\                                    Bx�^�z  �          @i��?�p�?�{@&ffB6��B<  ?�p�@�R@ ��B
=B\z�                                    Bx�^�   �          @j=q?�z�?��H@'
=B6
=BG{?�z�@%?��RB�HBeff                                    Bx�^��  �          @e?��?�=q@*�HB@��BG�H?��@�R@B�Bh�                                    Bx�^�l  �          @e?�  ?У�@3�
BO
=B=�?�  @�@�B�Bd�\                                    Bx�^�  �          @g�?���?��
@7�BR33B0��?���@\)@ffB"�B[��                                    Bx�_�  �          @l��?���?�
=@N{Br
=B"33?���@   @2�\BC�\B]\)                                    Bx�_^  �          @n�R?��?���@FffB`�B'�
?��@
=q@'
=B2\)BZ�                                    Bx�_  T          @u?�z�?s33@N{Bm
=A�R?�z�?��
@6ffBF{B;                                    Bx�_-�  �          @x��?�׼��
@Y��Bt{C���?��?c�
@Q�BfQ�Aʏ\                                    Bx�_<P  �          @z�H@�
��@S�
Bh�\C�G�@�
?B�\@N�RB_=qA�
=                                    Bx�_J�  �          @z�H?��H���R@W�Bn  C�s3?��H?
=@U�Bi�A��H                                    Bx�_Y�  �          @s�
?�{<��
@Tz�Bs  ?�\?�{?fff@L��Bd�A�ff                                    Bx�_hB  �          @o\)@�\=�\)@HQ�BcQ�?�{@�\?c�
@@  BU\)A�G�                                    Bx�_v�  �          @qG�?���>W
=@Y��B��@���?���?���@N{BlB(�                                    Bx�_��  �          @j�H?���>��R@S33B�.A5p�?���?�
=@EBfp�B��                                    Bx�_�4  �          @l��?��\?+�@Z=qB�ffA�33?��\?�ff@G
=Bd�HBK
=                                    Bx�_��  �          @mp�?�(�>��H@]p�B�aHA�p�?�(�?��@Mp�Bp
=BB��                                    Bx�_��  �          @p��?�\)>���@dz�B�
=Aq?�\)?��R@W
=B~z�B?�                                    Bx�_�&  �          @qG�?^�R?��@hQ�B�W
B=q?^�R?��R@VffBz��Bo�                                    Bx�_��  �          @q�?��\?(�@e�B�33A�Q�?��\?��
@R�\Br�Ba=q                                    Bx�_�r  �          @qG�?�  ?!G�@_\)B�8RA�
=?�  ?��
@L��Bi  BK33                                    Bx�_�  �          @\)?���?(�@l��B�  A�=q?���?���@Z=qBk�
BG=q                                    Bx�_��  �          @w�?�ff>�p�@fffB�ǮA�Q�?�ff?��@W�Bu33B4�
                                    Bx�`	d  �          @l(�?���\)@Y��B��
C���?��>�Q�@[�B�{An=q                                    Bx�`
  �          @c�
?�Q��@S�
B���C�(�?�Q�>�p�@U�B��)A�ff                                    Bx�`&�  �          @e�?�z��\@VffB��\C�?�z�>Ǯ@W
=B��A��                                    Bx�`5V  �          @c�
?�33��=q@EBp�C�8R?�33�k�@QG�B��fC�]q                                    Bx�`C�  �          @aG�?����=q@?\)BiQ�C��?�����H@O\)B��C��f                                    Bx�`R�  �          @`��?����@1G�BQ��C��?���c�
@G
=B|�C��
                                    Bx�`aH  �          @�(�?��׾��@w�B�B�C��{?���?B�\@s�
B��A�                                      Bx�`o�  �          @��?�p�>8Q�@w
=B�G�A
=?�p�?�
=@j�HB�8RB/ff                                    Bx�`~�  �          @s33?��
?(�@eB�(�A�G�?��
?��
@S�
Bsz�B`G�                                    Bx�`�:  �          @s�
?��?
=@e�B�� Aۅ?��?�G�@S33BpBS�\                                    Bx�`��  �          @u?���?J=q@c33B���B(�?���?�Q�@N{Bd�HB[                                      Bx�`��  �          @n�R?��
?n{@XQ�B�ǮB?��
?��
@AG�BX  BX�                                    Bx�`�,  �          @mp�?�ff?p��@UB���B�?�ff?��
@>�RBV  BWp�                                    Bx�`��  �          @|(�?���?Q�@j=qB�ǮB	��?���?�  @Tz�Be��B^\)                                    Bx�`�x  �          @�  ?��?\(�@o\)B�B�B=q?��?�ff@XQ�Bg=qBf��                                    Bx�`�  �          @��?��?333@u�B�A���?��?�@aG�Bl{BQ                                      Bx�`��  �          @��?��?B�\@s33B�(�A�?��?�(�@^�RBi33BT                                    Bx�aj  �          @���?�
=?�  @�  B��B!G�?�
=@z�@vffBhBq33                                    Bx�a  �          @���?��R?=p�@�
=B�\A��?��R?��@�(�BoQ�BM��                                    Bx�a�  T          @���?���?L��@���B��\A�
=?���?���@�{Br�B_�\                                    Bx�a.\  �          @�?��H?(��@�
=B�z�A�{?��H?�{@�z�B|Bd(�                                    Bx�a=  �          @�=q?�=q?z�@��HB��fA�{?�=q?�  @���B{Q�BS=q                                    Bx�aK�  �          @��?�33>�G�@�(�B��)A���?�33?�\)@��Bz�BD�\                                    Bx�aZN  �          @��?��?�\@�z�B���A�(�?��?�Q�@�33B~��BN(�                                    Bx�ah�  �          @��?�z�>�(�@��
B�A�33?�z�?���@��B�RBC{                                    Bx�aw�  �          @�  ?E�?��\@�=qB���BS��?E�@
=q@���Bt33B��                                    Bx�a�@  �          @���?�R?�@��\B��Bx�?�R@�
@��
Bo�\B��                                    Bx�a��  �          @�(�?���>�G�@��B���A�{?���?�ff@���BG�BFff                                    Bx�a��  �          @�33?�ff?   @�(�B��A�(�?�ff?�{@��B}{BK�                                    Bx�a�2  T          @���?�  >��@�ffB��A�(�?�  ?˅@�{B�k�BO��                                    Bx�a��  �          @�p�?��\>��@���B�\A�z�?��\?�{@�  B���Bf�                                    Bx�a�~  �          @��
?�{>��@��RB�B�A���?�{?��
@��RB��=BX(�                                    Bx�a�$  �          @�(�?��H>Ǯ@�ffB�#�A���?��H?�G�@�ffB��=BMp�                                    Bx�a��  �          @�(�?�G�>�z�@�{B��AMp�?�G�?�z�@�
=B��qB@                                    Bx�a�p  �          @�?��=�Q�@�
=B��3@|(�?��?�p�@���B�L�B*ff                                    Bx�b
  �          @�=q?���?�@�z�B�#�A��?���?Ǯ@w�B~�
B[ff                                    Bx�b�  �          @��?��H>.{@�(�B�
=A Q�?��H?��H@}p�B��B4G�                                    Bx�b'b  �          @��H?��\=��
@���B���@dz�?��\?���@\)B���B&                                    Bx�b6  �          @�z�?�녽�@���B��qC�t{?��?n{@���B�  B��                                    Bx�bD�  �          @��\?��
=q@xQ�B�33C��\?�>�@x��B���AdQ�                                    Bx�bST  �          @��R?�33�aG�@�(�B�
=C�K�?�33?W
=@���B��)A�ff                                    Bx�ba�  �          @�?�G��#�
@y��B��C�G�?�G�?s33@q�B��{B ff                                    Bx�bp�  �          @�ff?�Q�>�@z=qB�p�@�=q?�Q�?�{@p  B�W
Bff                                    Bx�bF  �          @�\)>��
?�{@��\B���B�\)>��
@8��@n{BO�B�G�                                    Bx�b��  �          @��<�@	��@|��Bu��B�Q�<�@Dz�@Q�B;��B���                                    Bx�b��  �          @��R=#�
?�z�@�Q�B�\B���=#�
@7
=@Z=qBH
=B�k�                                    Bx�b�8  �          @�ff>�ff?�@�{B��HB���>�ff@�H@l��BaffB��H                                    Bx�b��  �          @�33?L��?�ff@���B��3BR��?L��@
=@\)Br(�B���                                    Bx�bȄ  �          @�33?p��?�ff@�(�B�W
B@=q?p��@
=@~�RBo�B��)                                    Bx�b�*  �          @�
=?���?\(�@��B���B{?���?��@x��Br��BoQ�                                    Bx�b��  �          @�=q?Tz�?G�@�p�B�33B,��?Tz�?���@��\B~p�B���                                    Bx�b�v  �          @�=q?Y��?\(�@���B�=qB5�\?Y��?�
=@���Bz  B�k�                                    Bx�c  �          @��H?c�
?E�@�B�B�B#33?c�
?�@�33B}�HB�u�                                    Bx�c�  �          @��?z�?^�R@��B��fB`�?z�?�
=@�Q�B}Q�B�B�                                    Bx�c h  T          @�33?#�
?�ff@�p�B���BjG�?#�
@
=@���Bu=qB�=q                                    Bx�c/  �          @���?Y��?E�@��B�aHB(��?Y��?�33@�=qB��3B���                                    Bx�c=�  T          @�\)?Q�?�  @���B�ffBJ�
?Q�@@�p�Bv�
B�W
                                    Bx�cLZ  T          @���?
=?\(�@���B���B^�?
=?�p�@�G�B��=B���                                    Bx�c[   �          @�p�>�G�?G�@��\B���Br\)>�G�?��@�  B�  B���                                    Bx�ci�  �          @�Q�?�?B�\@�p�B��{B^(�?�?��@��HB�p�B���                                    Bx�cxL  �          @���?�?O\)@�p�B���Bd�\?�?�
=@��\B�k�B���                                    Bx�c��  �          @�?&ff?h��@�G�B�G�BZ33?&ff?��R@�B}\)B��                                    Bx�c��  �          @�>�(�?:�H@��HB��
Bn�R>�(�?�\@���B�Q�B�                                    Bx�c�>  �          @�=q>��R?p��@��RB��
B�33>��R@   @��HB~��B�.                                    Bx�c��  �          @�(�?�?�  @�\)B�\Bq��?�@�
@�33By��B�{                                    Bx�c��  �          @�Q�?8Q�?\(�@��B�\BH33?8Q�?�33@���B|�B�G�                                    Bx�c�0  �          @���?\(�?J=q@��B���B)�?\(�?�{@�p�B\)B�#�                                    Bx�c��  �          @�G�?h��?h��@��B�ǮB4�?h��@   @�  Bz��B�8R                                    Bx�c�|  �          @�
=?E�?��@��B�\Bz�?E�?ٙ�@��\B�k�B�{                                    Bx�c�"  �          @�z�?��\>�@��B���Aə�?��\?�ff@��B�Bbz�                                    Bx�d
�  �          @��?Q�?0��@��B�=qB =q?Q�?�@���B�W
B�\                                    Bx�dn  �          @��?+�?aG�@��
B���BS��?+�@@�Q�B��=B��=                                    Bx�d(  �          @��>��H?�  @�(�B��\B�8R>��H@��@�\)B��B��R                                    Bx�d6�  �          @��>��H?s33@�(�B��=Bzz�>��H@p�@�  B�B�B��                                    Bx�dE`  �          @���?&ff?�{@�
=B�� BoQ�?&ff@p�@�=qBw��B�\)                                    Bx�dT  �          @���?!G�?�\)@��
B�z�Bqz�?!G�@\)@��RBy��B�p�                                    Bx�db�  �          @��?(�?xQ�@���B�\Bfp�?(�@z�@��Bp�B��                                    Bx�dqR  �          @�z�?!G�?^�R@���B��HBW�
?!G�@�\@�B��B��=                                    Bx�d�  �          @��>�(�?s33@�  B�ffB��>�(�@
=@�(�B�#�B��q                                    Bx�d��  �          @��>u?��@�33B�Q�B�ff>u@�
@�{B~z�B�B�                                    Bx�d�D  �          @���<��
?�p�@��\B���B�8R<��
@,��@��\BsQ�B�                                    Bx�d��  �          @��H��Q�?�p�@�z�B�G�B��f��Q�@-p�@�z�Bt{B�                                    Bx�d��  �          @��\��  ?���@�z�B�� B�uþ�  @#33@�p�Bu
=B�#�                                    Bx�d�6  �          @�=q�u?��@�(�B��fB�k��u@
=@�{Bq��B�k�                                    Bx�d��  �          @����@.�R@\)B
(�C��@N�R?��
A�\)B��                                    Bx�d�  �          @�\)��G�@�=q    <#�
C���G�@}p��s33�  C	Q�                                    Bx�d�(  �          @���p��@�  ?=p�@�ffC���p��@�녾�=q�-p�C#�                                    Bx�e�  �          @Ǯ�_\)@���=��
?E�B�8R�_\)@��ÿ�\)�%��B��                                     Bx�et  �          @�������@vff>���@|(�C
�����@u��\����C
+�                                    Bx�e!  �          @�G���p�@N�R>��
@_\)C5���p�@N{��G����CT{                                    Bx�e/�  �          @�����G�@QG�>��@��C����G�@Q녾�Q��u�C��                                    Bx�e>f  �          @��\��  @P��>�@�Q�C���  @Q녾����FffC��                                    Bx�eM  �          @��
��(�@E�>�p�@~{C���(�@E���33�mp�C��                                    Bx�e[�  �          @�����z�@I��>L��@��C\)��z�@G
=����33C��                                    Bx�ejX  �          @�(����H@E��Q�}p�C�����H@?\)�L����RC�
                                    Bx�ex�  �          @�{���
@H�þ����hQ�C�3���
@>�R����8��C�                                    Bx�e��  �          @�����H@P  ��z��HQ�C�����H@Fff���\�4  C�{                                    Bx�e�J  �          @������@fff�����
=C!H���@Z=q��(��K�C��                                    Bx�e��  �          @�(����\@~{<�>�=qC	z����\@xQ�aG��	�C
0�                                    Bx�e��  �          @�����=q@���>�=q@C	����=q@��H�=p��ϮC	�3                                    Bx�e�<  �          @����
@��>\@N{C�����
@��ÿ+���
=C	�                                    Bx�e��  �          @������@���L�;��C	�����@�G������HC
T{                                    Bx�e߈  �          @�  ���
@���>W
=?�ffCO\���
@��\�W
=��C�                                     Bx�e�.  �          @�������@���        C
.����@��Ϳ��\�	C
��                                    Bx�e��  �          @�ff��Q�@��>�?�z�C
����Q�@��\�\(���(�C.                                    Bx�fz  �          @�z���\)@��
=�\)?#�
C
�q��\)@��ÿh����33C\)                                    Bx�f   �          @���@�Q���ͿY��C	����@��
�����  C
k�                                    Bx�f(�  
�          @ָR����@��\�#�
���C������@���
=�!�C	޸                                    Bx�f7l  �          @��
��Q�@�녿����CG���Q�@�=q�Ǯ�Z�RC	��                                    Bx�fF  �          @����33@��׿z���  C	���33@��ÿǮ�YC
�)                                    Bx�fT�  �          @������@�  �@  ��Q�C�����@�
=��p��qC
�3                                    Bx�fc^  �          @Å���@���p�����CxR���@{���������C	�f                                    Bx�fr  �          @�ff�dz�@W
=�8Q��=qC���dz�@G���Q����C
ٚ                                    Bx�f��  �          @�\)�p  @c�
�.{��\)C� �p  @TzῸQ���z�C
s3                                    Bx�f�P  �          @��R�r�\@_\)�B�\�C	h��r�\@O\)��  ��{C��                                    Bx�f��  �          @����O\)@Vff��  �AC)�O\)@C33���H��C�=                                    Bx�f��  �          @�녿�=q@8Q������F�
B��3��=q?�Q������nCL�                                    Bx�f�B  �          @�(��b�\@n�R�)����C� �b�\@Dz��X�����C
                                    Bx�f��  �          @�(��N{@k��tz��
=C.�N{@0�������;\)CT{                                    Bx�f؎  �          @����@n�R��ff�<�RB�=���@'������e=qCǮ                                    Bx�f�4  �          @�(��'
=@c�
��ff�.ffB�p��'
=@$z����
�T33Cc�                                    Bx�f��  ]          @��Q�@S�
�{��3�RB�k��Q�@�������[ffC�f                                    Bx�g�  �          @�(��@dz��P  �=qB��=�@2�\�|(��=  C�                                    Bx�g&  �          @�����@fff�W����B�W
��@333����C\)B�Ǯ                                    Bx�g!�  �          @�
=� ��@����P���p�B�  � ��@P  ��G��:p�B�=                                    Bx�g0r  �          @�33�%�@��\��(����B����%�@dz��2�\�z�B�                                    Bx�g?  �          @�Q��\)@{��Q��Vz�B�� ��\)?Ǯ��{�|p�C�R                                    Bx�gM�  �          @��
�޸R@;�����T�
B�G��޸R?�z����\�|G�C^�                                    Bx�g\d  �          @���<��@��
�$z��ϙ�B��<��@n�R�[���C O\                                    Bx�gk
  �          @�z��+�@���L���p�B���+�@X����  �*��C T{                                    Bx�gy�  �          @�=q�(Q�@����P���33B�B��(Q�@P�������/ffC �f                                    Bx�g�V  �          @����R@�Q��c33��HB�W
��R@L(�����;�\B��R                                    Bx�g��  T          @�
=�@w��x���#  B�B��@>{����J��C (�                                    Bx�g��  �          @����#33@u�^{���B�33�#33@A���{�:ffC
=                                    Bx�g�H  �          @�G��0��@{��N{�Q�B���0��@K��}p��-
=C�                                    Bx�g��  �          @�G��<(�@~{�?\)���B�
=�<(�@QG��o\)�!�RC��                                    Bx�gє  �          @���8��@�G��@  ��
=B�.�8��@U�qG��!�HC��                                    Bx�g�:  �          @�Q��Fff@|���333��RC #��Fff@S33�c33�Q�CG�                                    Bx�g��  �          @��
�P  @r�\�$z��ٙ�C���P  @L(��R�\�Q�C�\                                    Bx�g��  �          @��\�Tz�@e�,�����C���Tz�@=p��W���\C
E                                    Bx�h,  �          @��H�U@e�*�H��ffC�3�U@>{�U�{C
W
                                    Bx�h�  �          @���Y��@^�R�#33��\)CG��Y��@9���L����C�\                                    Bx�h)x  �          @���^{@\���p��ՙ�C+��^{@8���G
=�
=qCJ=                                    Bx�h8  �          @�
=�dz�@e������\C���dz�@E�0  ��  C+�                                    Bx�hF�  �          @����l��@p  ���H��  C���l��@U�����  C
\                                    Bx�hUj  �          @����s�
@x�ÿ�p���Ch��s�
@Z=q�-p��߮C
�                                    Bx�hd  �          @����|(�@�G������{CQ��|(�@a��9����C
(�                                    Bx�hr�  �          @�ff�~{@�
=�2�\�ͅC=q�~{@e�dz���HC	�                                    Bx�h�\  �          @ۅ���@�z�����\)C\���@�33�I�����C	��                                    Bx�h�  �          @�z����R@��H�/\)��33C����R@mp��b�\�{C
�{                                    Bx�h��  �          @�p���{@���9�����
C33��{@i���l(��33C
�f                                    Bx�h�N  �          @ָR��Q�@�  �$z���{Cp���Q�@z=q�Y����C	u�                                    Bx�h��  �          @�Q����\@�  ������CxR���\@�  �>�R��
=C
޸                                    Bx�hʚ  �          @ᙚ���H@�ff��\��z�C�{���H@��R�:=q��\)C
�3                                    Bx�h�@  �          @��
��=q@�=q����C���=q@����L(����HC
��                                    Bx�h��  �          @�G���Q�@����Dz���33C����Q�@hQ��vff�
  C��                                    Bx�h��  �          @�
=��p�@�  ��R���C�R��p�@��Vff��RC�)                                    Bx�i2  T          @���\)@��׿��
�g�
CO\��\)@��\�-p���
=C��                                    Bx�i�  �          @�
=����@�(��fff��C	5�����@�������i�C
�=                                    Bx�i"~  �          @Ӆ����@�G���\)�?�
C�����@�ff�p����HC	G�                                    Bx�i1$  �          @��H���@��
�0  ���C=q���@p���a��ffC	z�                                    Bx�i?�  �          @أ����
@��A����HC�H���
@q��tz��	G�C	s3                                    Bx�iNp  �          @�ff�w�@��\�333��{C���w�@n{�dz��ffC)                                    Bx�i]  �          @����Z�H@���c33�B����Z�H@j=q����"��C\                                    Bx�ik�  �          @У��E�@���s33���B�33�E�@aG���G��0�\C8R                                    Bx�izb  �          @ҏ\�6ff@����Q��z�B�=q�6ff@b�\��Q��9\)C ��                                    Bx�i�  �          @�
=�:=q@�p������B���:=q@e�����:(�C�                                    Bx�i��  �          @���W
=@����a��=qB���W
=@j�H��G��#{Cs3                                    Bx�i�T  �          @Ӆ�g
=@�\)�P����=qC �)�g
=@s�
�����(�C��                                    Bx�i��  �          @�p��e�@w��
=����C��e�@Z=q�1����CW
                                    Bx�ià  �          @����2�\@K���z���p�CG��2�\@4z��p���33C��                                    Bx�i�F  �          @���+�@J=q�qG��)(�CE�+�@�H��G��Gz�C	�H                                    Bx�i��  �          @�=q�
=@>�R�n{�6  B���
=@����ff�V�C!H                                    Bx�i�  �          @s33��(�@
�H�6ff�CB��f��(�?�{�L���f�RB�L�                                    Bx�i�8  �          @��\�˅@���n�R�O��B�  �˅?޸R���
�p��Cz�                                    Bx�j�  �          @�z��\)?��H��p��j�HC	�H��\)?n{���B�C��                                    Bx�j�  �          @�ff����?��H�s33�j\)C�\����?�  ��G��\Cn                                    Bx�j**  �          @�����H?�33�a��g��C�����H?}p��p����C�
                                    Bx�j8�  �          @�������?�G���G��~33C(�����?E����L�C�3                                    Bx�jGv  �          @�  �!G�?=p��\)��C��!G�<��
����¢G�C1�                                    Bx�jV  T          @l(��B�\�8Q��k�«�3C_���B�\�Tz��eW
C�p�                                    Bx�jd�  T          @L�;k�����G
=§.CdO\�k��L���AG�\C}�
                                    Bx�jsh  L          @�
=@G���\@J�HB8ffC�
=@G�����@^�RBP�HC��                                    Bx�j�  \          @�p�@0���.{@p��B0z�C���@0���G�@�p�BJ�C��                                    Bx�j��  �          @�@E��8Q�@�p�B2�C�w
@E���@�33BK�\C�                                    Bx�j�Z  �          @���@H���>{@�\)B1Q�C�L�@H�����@�BJ�C��H                                    Bx�j�   	h          @�Q�@C�
�H��@o\)B!�HC�'�@C�
���@�
=B<z�C���                                    Bx�j��  "          @�33@C33�N�R@Z=qB�HC��3@C33�&ff@z=qB1
=C�˅                                    Bx�j�L  T          @��@O\)�G�@W
=B  C�\@O\)�   @uB,��C�,�                                    Bx�j��  �          @�\)@l���K�@\(�B��C��=@l���#33@{�B$�C���                                    Bx�j�  �          @˅@}p��h��@Y��B�C���@}p��AG�@}p�B�C�W
                                    Bx�j�>  
�          @˅@����n�R@K�A��C��@����H��@p��B  C�R                                    Bx�k�  �          @�z�@�\)�p��@>{A��C�/\@�\)�Mp�@c33B
=C�e                                    Bx�k�  �          @�{@�ff�e@2�\A�z�C��
@�ff�E�@VffB�C�޸                                    Bx�k#0            @Å@����N�R@:=qA��C�l�@����-p�@Y��BC��                                    Bx�k1�  
b          @�p�@��\���@/\)A�ffC�b�@��\�fff@XQ�A���C�E                                    Bx�k@|  �          @ҏ\@����ff@#33A��RC�e@���~{@P  A���C�f                                    Bx�kO"  
(          @��@����z�@�HA���C��@���{�@FffA�C��3                                    Bx�k]�  T          @�=q@�p���ff@p�A�C���@�p��\)@I��A�
=C�%                                    Bx�kln  �          @��H@��H���H@��A��C���@��H�|(�@3�
A���C��                                    Bx�k{  ~          @�Q�@u��Q�?��
Av�\C��q@u�mp�@
=qA��C���                                    Bx�k��  
�          @�=q@dz��p  ?�AK\)C��=@dz��aG�?�G�A�ffC��R                                    Bx�k�`  �          @�Q�@c33�o\)?��
A4��C���@c33�a�?�\)A�
=C���                                    Bx�k�  T          @��@U�l(�>\@��C�{@U�e�?uA1��C��                                     Bx�k��  �          @�G�@n�R�n{?ٙ�A��C���@n�R�Z=q@G�A�p�C���                                    Bx�k�R  �          @�33@w���=q?�p�Ak�C�� @w��r�\@ffA���C�Ǯ                                    Bx�k��  �          @˅@����ff?�Q�AR�HC�Ф@����p�@�A��HC��                                     Bx�k�  �          @�\)@~�R����?
=q@�{C��@~�R���
?��\A<z�C�XR                                    Bx�k�D  T          @��@n�R���R?&ff@�=qC��f@n�R���?��AS33C�K�                                    Bx�k��  �          @�{@u���R?@  @陚C�e@u��G�?�
=A_�C���                                    Bx�l�  �          @��@�����{?333@ʏ\C��q@�������?�z�ANffC�=q                                    Bx�l6  T          @�
=@�{����?8Q�@�(�C�e@�{���
?�33APz�C��                                    Bx�l*�  �          @��
@�ff��ff?c�
@��C���@�ff��Q�?˅Ag33C�y�                                    Bx�l9�  �          @���@��R���H?L��@���C�G�@��R���?�p�AZ=qC��{                                    Bx�lH(  �          @�@�(���Q�?Y��@��C�C�@�(����\?\Ac�C��R                                    Bx�lV�  �          @��H@n{����?�{A7\)C�Y�@n{�tz�?ٙ�A��C��                                    Bx�let  T          @��\@g
=���?�  AO33C��\@g
=�u�?�A�(�C��                                    Bx�lt  �          @�(�@e���?�A@��C�C�@e��}p�?��
A��C��                                    Bx�l��  �          @��@�
=��(�?�=qA@Q�C�/\@�
=��(�@   A���C��R                                    Bx�l�f  T          @ȣ�@�����?��A��HC��H@���{�@��A��C���                                    Bx�l�  �          @���@�����H?޸RA�\)C��{@���r�\@�
A�C���                                    Bx�l��  �          @�z�@�=q�vff?�ffA��RC�E@�=q�c33@A���C�o\                                    Bx�l�X  �          @˅@������?�G�A]�C�"�@����s33@�A�ffC�3                                    Bx�l��  �          @θR@�ff��G�?���A�C��\@�ff���H?�
=Aq�C�z�                                    Bx�lڤ  �          @�{@�=q��?^�R@�Q�C��@�=q����?��HAQG�C�R                                    Bx�l�J  �          @�{@�
=���R?�33A$��C�!H@�
=��Q�?޸RAz�RC��
                                    Bx�l��  �          @θR@�Q���Q�?fffA ��C��@�Q����H?�  AV�\C��=                                    Bx�m�  �          @�ff@�33���>��R@0  C�l�@�33����?fffA z�C��
                                    Bx�m<  �          @Ϯ@���ff>��R@0  C��=@����
?c�
@�p�C�3                                    Bx�m#�  
�          @�=q@�����>��@\)C�.@����G�?\(�@�Q�C�o\                                    Bx�m2�  �          @љ�@��\��(�?��@��\C��@��\��Q�?�33A!C�N                                    Bx�mA.  T          @\@��H����?���A�(�C���@��H�n�R@\)A���C��3                                    Bx�mO�  �          @�=q@��H�\)?�(�A�z�C��H@��H�k�@   A�\)C�f                                    Bx�m^z  �          @��H@����u@C�
A�{C�B�@����X��@c33B��C��                                    Bx�mm   �          @�z�@�z���Q�?�
=Av=qC���@�z��\)@\)A��C��H                                    Bx�m{�  T          @ҏ\@������?���A(��C�  @�����G�?��A}�C���                                    Bx�m�l  �          @θR@��\���?��HAP��C�aH@��\��=q@G�A��C�+�                                    Bx�m�  �          @�G�@�
=��=q?�  A���C��\@�
=�s�
@G�A�G�C��f                                    Bx�m��  �          @��@��
��
=?�{AC�C��@��
��Q�?�33A�{C���                                    Bx�m�^  �          @�=q@�  ���\?�{A?�C��R@�  ���?�A�C���                                    Bx�m�  �          @��H@����33?�33AD��C��)@����(�?��HA�Q�C�y�                                    Bx�mӪ  �          @��@�����p�?�{A?\)C�9�@������R?�A�{C��                                    Bx�m�P  �          @���@�G�����?��Ap�C��@�G����
?У�Ag33C���                                    Bx�m��  �          @�p�@�(����?Tz�@���C�+�@�(����H?�
=AEC��q                                    Bx�m��  �          @��@������?�G�AG�C���@�����{?�ffAZ�RC�n                                    Bx�nB  �          @љ�@����  ?���A=qC�l�@����=q?�\)Ae��C��                                    Bx�n�  "          @��@�����
=?�{A�
C��q@�������?У�Af{C�33                                    Bx�n+�  
�          @��
@�{��Q�?�A"�\C���@�{���\?�Q�AlQ�C�33                                    Bx�n:4  L          @�{@�{��  ?   @���C�ٚ@�{����?�ffAz�C�(�                                    Bx�nH�  
�          @أ�@�(�����>�G�@qG�C���@�(����?z�HAG�C���                                    Bx�nW�  
�          @�33@������>�ff@s33C�s3@�������?�  A�C��)                                    Bx�nf&            @��@�ff��Q�?�@�=qC��f@�ff��p�?��A��C��{                                    Bx�nt�  
�          @�33@�{��{?�@��C��R@�{��33?��A{C��                                    Bx�n�r  �          @ڏ\@����p�?.{@�
=C���@������?���A"=qC�
                                    Bx�n�  T          @�33@�����>�G�@l(�C��{@�����?p��@���C��                                    Bx�n��  �          @أ�@�
=��G�>B�\?�33C�J=@�
=���?333@�C�w
                                    Bx�n�d  
�          @ڏ\@�������>�ff@p��C�h�@�����
=?s33@�\)C��                                    Bx�n�
  �          @�\)@�p�����>��@]p�C�33@�p���ff?fff@�{C�s3                                    Bx�n̰  T          @أ�@�
=����>B�\?�{C�C�@�
=��  ?.{@��C�o\                                    Bx�n�V  T          @�\)@�
=���>8Q�?���C�u�@�
=��{?+�@�ffC��                                     Bx�n��  �          @�{@������>L��?�Q�C�=q@�����{?.{@��\C�h�                                    Bx�n��  
�          @�@����
>�=q@Q�C��@���=q?:�H@У�C�:�                                    Bx�oH  
�          @���@�z��|(�>�{@E�C���@�z��xQ�?G�@�33C��=                                    Bx�o�  �          @�{@���s�
?!G�@���C���@���n{?�ffA$(�C�0�                                    Bx�o$�  �          @��
@�Q��mp�?�@���C�O\@�Q��hQ�?n{AffC��q                                    Bx�o3:  T          @��H@�
=�n�R>\@qG�C�
@�
=�j=q?J=q@���C�U�                                    Bx�oA�  �          @�p�@�=q�3�
�k����C�z�@�=q�9���(����C��                                    Bx�oP�  �          @�@���   �
�H���RC�U�@���{��Q����
C�E                                    Bx�o_,  
�          @��R@�G�����\)��G�C���@�G����   ��Q�C��)                                    Bx�om�  	&          @�G�@�\)��z���p�C���@�\)�33��=q���C���                                    Bx�o|x  
�          @�@�\)��\)�����=qC��q@�\)���G���Q�C��R                                    Bx�o�  �          @��\@�G��s33���R����C���@�G���33�����p�C��H                                    Bx�o��  �          @�z�@��ÿ�z��Q���33C�l�@��ÿ��ÿ�  ��33C�y�                                    Bx�o�j  �          @�z�@��Ϳ�  �   ���HC�0�@��Ϳ�Q�����\C��                                    Bx�o�  �          @��
@�\)��(��У�����C��@�\)��׿�Q����\C��                                    Bx�oŶ  �          @���@��
���Ϳ�ff���C���@��
�   ��{��(�C��                                    Bx�o�\  �          @�{@�Q��
=��33��ffC���@�Q��=q��p����C���                                    Bx�o�  �          @��
@�G���
=��(���33C�G�@�G��˅������33C�AH                                    Bx�o�  �          @��@8��>�=q�W
=�D@���@8�ý�\)�W��E�\C�Y�                                    Bx�p N  �          @�
=?�z�>���l���z��@��
?�z�W
=�l���zQ�C��)                                    Bx�p�  �          @K�?�
=?���(Q��n�
A���?�
=>�z��*�H�u{A5G�                                    Bx�p�  T          @u�?���?�p��Tz��sp�B&Q�?���?fff�Z�Hu�B\)                                    Bx�p,@  �          @�=q?Y��@p��e�bp�B��)?Y��?����r�\�v�B��R                                    Bx�p:�  �          @c33?�@��&ff�?ffB�?�@��3�
�Tp�B�W
                                    Bx�pI�  �          @\��?z�@��p��4�B�k�?z�?�33����I�B��                                    Bx�pX2  �          @Z=q?�(�?���#33�E33B@z�?�(�?���,���T�\B-�H                                    Bx�pf�  �          @+�?��?�������ǅBb{?��?��Ϳ��\��\B[��                                    Bx�pu~  �          @��?�Q�?�p���33��(�BL?�Q�?�\)���
�33BC�H                                    Bx�p�$  �          ?�\)?\(�?.{��
=�R\)B��?\(�?�Ϳ�p��^z�B(�                                    Bx�p��  �          @(Q�?��H?��ÿ�(��-�
B33?��H?k����9z�B (�                                    Bx�p�p  �          @=q?�(�>�p���\)�_=qA��R?�(�>L�Ϳ���c�Ap�                                    Bx�p�  �          @#�
?�  >�׿�\�1p�Aq�?�  >��R��ff�5�HA"{                                    Bx�p��  �          @��?���?h�ÿ�z��?=qB�?���?B�\��p��J�RB                                      Bx�p�b  �          @
=?�(�?s33��\�CffB  ?�(�?J=q�����N�Bp�                                    Bx�p�  T          ?��?�ff>\�����2(�A��?�ff>�zῐ���8�As\)                                    Bx�p�  �          ?���?n{?#�
���
�{�
B
Q�?n{?(���p����BQ�                                    Bx�p�T  �          ?�{>�=q�k�?@  B���C�o\>�=q�.{?E�B�u�C�ٚ                                    Bx�q�  �          ?Ǯ?#�
>�p��8Q��/��A�?#�
>��R�@  �9  AϮ                                    Bx�q�  �          @K�?�?aG����A
=Aģ�?�?.{���H(�A�                                      Bx�q%F  �          @���@@  ?u�J�H�4�RA��\@@  ?333�O\)�9�RAR�R                                    Bx�q3�  �          @���@�\)�W
=�������C��3@�\)������Ə\C���                                    Bx�qB�  �          @��@|��<��
��p���p�>�33@|�;\)��p����HC�                                    Bx�qQ8  �          @5@ ��>��R��G���p�@�(�@ ��>W
=���
��33@��                                    Bx�q_�  �          @b�\@G�>�G���\)��Q�Ap�@G�>��R��33�ޣ�@�                                      Bx�qn�  �          @���@dz�>���ff����@��@dz�>��
��=q����@��                                    Bx�q}*  �          @��R@\)?�R����33A��@\)>�������  @�=q                                    Bx�q��  �          @n�R@P  ?����z���33A'33@P  >�׿ٙ����HA(�                                    Bx�q�v  T          @u@S33?#�
�������A1�@S33?�\��\)���
A�                                    Bx�q�  �          @?\)@ ��>�33�Ǯ��G�@�33@ ��>k��˅� ��@�ff                                    Bx�q��  �          @A�@,��>�zῪ=q�Џ\@���@,��>B�\����ә�@��\                                    Bx�q�h  �          @J�H@9���u��p�����C�c�@9�������(���ffC���                                    Bx�q�  �          @�\?��?��
�Ǯ�8�HB-�?��?k��У��C�B p�                                    Bx�q�  �          @��?�R?�=q��(��X��Bpff?�R?s33����f�\Bc��                                    Bx�q�Z  �          @
=q�!G�?��H��=q��\B�G��!G�?У׿�Q��	ffB�{                                    Bx�r   �          ?�
=����?�z�!G����HCh�����?�{�5��p�C��                                    Bx�r�  �          ?��Ϳ�\)?0�׾8Q���CB���\)?.{�aG���
C�                                    Bx�rL  �          ?˅��\)?@  ��\)�$��Cs3��\)?:�H�����B=qC!H                                    Bx�r,�  �          @\)��?��
�   �Pz�C�\��?}p����mG�C�f                                    Bx�r;�  �          ?�녿˅?}p���G��UCB��˅?z�H�.{���HC��                                    Bx�rJ>  �          ?����R?z�H��{�,��C�3���R?s33�����MC^�                                    Bx�rX�  
�          ?(���p�>�  �����(ffCs3��p�>aG���
=�0G�Cff                                    Bx�rg�  �          ?Q녾�Q�>L�Ϳ&ff�hz�C�׾�Q�>#�
�+��o  CY�                                    Bx�rv0  �          ?aG���R>.{�
=q� �C$�Ϳ�R>\)����#��C'��                                    Bx�r��  �          ?xQ�G�>u�����=qC#��G�>aG���=q��Q�C$B�                                    Bx�r�|  �          ?��ÿ�p�>�\)��p�����C'G���p�>�  �Ǯ����C(O\                                    Bx�r�"  �          ?��ÿ�  �u����d��C7J=��  ��\)��  �aG�C8+�                                    Bx�r��  �          ?z�H�h��>aG������HC&��h��>W
=�#�
��C'{                                    Bx�r�n  �          ?k��\(�>�\)�#�
��C"  �\(�>�\)���
���
C"�                                    Bx�r�  �          ?ٙ��У�>�(��L�Ϳ��
C%B��У�>�
=���
�(Q�C%aH                                    Bx�rܺ  �          ?�z�(�>�  �G��E��C�)�(�>W
=�J=q�JG�C!L�                                    Bx�r�`  �          ?333��=�G��\��C(���=��
�\�p�C*�\                                    Bx�r�  �          ?�ff�k�>��
�k��V�RC �k�>��R��  �g\)C!�                                    Bx�s�  �          ?��Ϳ�
=?�>\)@�
=C5ÿ�
=?z�=�G�@��HC�                                    Bx�sR  �          @Q����?   >�
=A"�RC's3���?�>ǮA�C&�                                    Bx�s%�  �          @ƸR��(�@"�\=�G�?��
C�R��(�@"�\���k�C��                                    Bx�s4�  �          @�=q��p�@(�?
=q@��HC!&f��p�@{>�(�@g
=C �3                                    Bx�sCD  �          @�ff���@*=q>���@R�\Cs3���@+�>��@��CQ�                                    Bx�sQ�  T          @�\�љ�@,(�=��
?(�C�3�љ�@,(��u��\C��                                    Bx�s`�  �          @�\��\)@7��u��C
��\)@7
=�L�Ϳ��C#�                                    Bx�so6  �          @ᙚ��(�@@  ����  C�\��(�@?\)��=q�p�C�H                                    Bx�s}�  �          @׮���@N{�����Z=qCk����@L(��\)���C�
                                    Bx�s��  �          @�z���=q@e�.{��{CO\��=q@c�
�Y����z�C�{                                    Bx�s�(  �          @�G���p�@5�>�p�@A�C8R��p�@5>u?�
=C)                                    Bx�s��  �          @�z����H@+�>8Q�?�(�C����H@+�=L��>�(�C\                                    Bx�s�t  T          @�\)���@\(��u��p�Ck����@[������QG�C�                                    Bx�s�  �          @�p���G�@W�>W
=?޸RC�
��G�@XQ�=u>�C�=                                    Bx�s��  �          @�z���p�@^�R>�p�@G
=C�
��p�@_\)>aG�?�=qC�                                     Bx�s�f  �          @�(���G�@Q�>B�\?�\)Cz���G�@R�\=L��>�(�Cp�                                    Bx�s�  �          @�����H@[�<�>�=qCn���H@[�����G�Cp�                                    Bx�t�  �          @��
��(�@w
=��=q�  C�{��(�@u��(��fffC��                                    Bx�tX  �          @ۅ��{@8Q�>�Q�@?\)C���{@8��>u?��RC�3                                    Bx�t�  �          @�=q�ʏ\@��?�R@�C���ʏ\@�R?�@��HC��                                    Bx�t-�  �          @أ����
@8Q�>L��?�p�C���
@8��=�Q�?B�\C�R                                    Bx�t<J  �          @�  ��=q@;�>#�
?���C8R��=q@<(�=L��>��C0�                                    Bx�tJ�  �          @����@6ff>aG�?���CB���@6ff=�G�?n{C5�                                    Bx�tY�  �          @������@Fff��G��aG�C޸���@E�aG�����C��                                    Bx�th<  T          @���
=@C�
���k�C�{��
=@C33�\)��33C�)                                    Bx�tv�  T          @��H��@:=q=#�
>���C����@:=q�u�
=qC��                                    Bx�t��  	�          @أ����H@Vff�(�����C.���H@Tz�8Q���z�C\)                                    Bx�t�.  �          @�p���G�@333�����
C!H��G�@2�\�aG�����C.                                    Bx�t��  �          @������>\?��Ax  C/xR���>�(�?���AuG�C.�                                    Bx�t�z  �          @�  ���׾��
?���A�ffC8z����׾�  ?��HAυC7��                                    Bx�t�   �          @�
=��(�=�G�?�  A��
C2�{��(�>#�
?޸RA�p�C1ٚ                                    Bx�t��  �          @����>��?޸RA�
=C.����>�?�p�A��C.B�                                    Bx�t�l  �          @�=q��=q>�ff?�A��C.c���=q>��H?�z�A�{C-�)                                    Bx�t�  �          @�������?��?��\A�p�C,�f����?
=?�G�A���C,&f                                    Bx�t��  �          @������\?+�?�33A`(�C+�f���\?333?���A\(�C+@                                     Bx�u	^  �          @�����{?��\?}p�A9�C'�q��{?�ff?uA4Q�C'k�                                    Bx�u  T          @�Q���z�?��
>��@�p�C����z�?��>�
=@�33Cٚ                                    Bx�u&�  �          @����Q�@�>8Q�?��HCff��Q�@�=�?���C\)                                    Bx�u5P  �          @�Q���p�@��#�
��C����p�@����
�G�C�q                                    Bx�uC�  �          @�33���\@2�\���
�O\)C����\@2�\������
C�                                    Bx�uR�  �          @�
=��z�@=p��L�Ϳ��HCǮ��z�@<�;�=q�,��C�{                                    Bx�uaB  �          @���  @*=q���R�I��C���  @*=q�\�s33C�q                                    Bx�uo�  �          @�����\)@.{�#�
��\)C����\)@.{��\)�0��C�=                                    Bx�u~�  �          @�=q��  @#33=u?\)C���  @#33    <��
C�                                    Bx�u�4  �          @�\)��Q�@>k�@��C����Q�@>8Q�?�Cٚ                                    Bx�u��  �          @ƸR����@�>��@
=Cu�����@�>W
=?�Ck�                                    Bx�u��  �          @�
=��ff?˅?J=q@陚C%
=��ff?�{?B�\@�  C$�f                                    Bx�u�&  �          @˅�Å?��
?\(�@�Q�C%���Å?�ff?Tz�@�\)C%�                                    Bx�u��  �          @�
=�ƸR?�p�?�  A{C&���ƸR?�  ?xQ�A
{C&p�                                    Bx�u�r  �          @�Q��Ǯ?��H?�{Ap�C&�)�Ǯ?�p�?�=qA��C&��                                    Bx�u�  T          @�����
=?�\?Y��@��C$&f��
=?��
?Q�@�C$                                    Bx�u�  �          @�  ��{?�(�?�  A{C$xR��{?޸R?xQ�A	�C$T{                                    Bx�vd  �          @�������?��H?n{A�HC"\)����?�(�?fff@���C"=q                                    Bx�v
  �          @��
���
@�?L��@�{C�H���
@Q�?B�\@�(�CǮ                                    Bx�v�  �          @��
��=q@{?k�@��RC����=q@�R?aG�@�z�CǮ                                    Bx�v.V  �          @�z����H@p�?p��A
=C���H@{?h��@�z�C�                                    Bx�v<�  T          @Ӆ��=q@p�?h��@�C�f��=q@{?aG�@���C�\                                    Bx�vK�  �          @�(�����@:�H?�R@��
C�3����@;�?z�@��C��                                    Bx�vZH  �          @�(���ff@7
=?��@��CJ=��ff@7�?\)@��C=q                                    Bx�vh�  �          @�ff���@<��?�\@�(�C�q���@=p�>�@�33C�3                                    Bx�vw�  �          @޸R���@o\)>�\)@�Cu����@o\)>u?�(�Cp�                                    Bx�v�:  �          @�ff���R@}p�>aG�?��C=q���R@}p�>8Q�?�(�C:�                                    Bx�v��  \          @�ff��33@n�R>��
@&ffC����33@n�R>�\)@�C��                                    Bx�v��  �          @޸R����@j�H>�@qG�C
����@k�>�(�@`��C�                                    Bx�v�,  �          @�\)����@[�?   @�33Cn����@\(�>��@xQ�Ch�                                    Bx�v��  �          @�
=��p�@Mp�>�@r�\CxR��p�@N{>�G�@fffCp�                                    Bx�v�x  �          @�����(�@,��?���A�C���(�@-p�?�{Ap�C�q                                    Bx�v�  �          @�\)��{@C33?s33@��C���{@C33?n{@��C�q                                    Bx�v��  �          @�G���p�@U�?
=q@�p�C����p�@U�?�@�G�C��                                    Bx�v�j  �          @�\)���@��>��?�
=C(����@��>�?��C&f                                    Bx�w
  �          @�ff���@�33>#�
?�  C�
���@�33>\)?���C�
                                    Bx�w�  T          @�33���H@���>��
@%�Cc����H@���>��R@�RCaH                                    Bx�w'\  T          @�Q����@{�>�\)@�C����@{�>�=q@p�C�                                    Bx�w6  �          @�
=��=q@u>W
=?�p�C�
��=q@u>L��?�
=C�
                                    Bx�wD�  
�          @�G���p�@����\)���CW
��p�@����\)�z�CW
                                    Bx�wSN  �          @�����@�ff��z��z�C33���@�ff��z���C33                                    Bx�wa�  �          @�Q���Q�@����  ��Q�C� ��Q�@���u��C�                                     Bx�wp�  �          @�R����@��;�p��<��C	{����@��;�p��:=qC	{                                    Bx�w@  �          @�G����
@�p��aG��ٙ�C	����
@�p��W
=�У�C	�                                    Bx�w��  �          @�����@������CG����@���  ��\CE                                    Bx�w��  T          @�����@��
�8Q쿷
=Cz�����@��
�#�
����Cz�                                    Bx�w�2  �          @�{���H@��þ�Q��5�C�����H@��þ����*�HC�3                                    Bx�w��  �          @��
��z�@��;\)��z�C����z�@��ͽ��z�HC�                                    Bx�w�~  �          @����
@�(�>k�?�{C8R���
@�(�>�  @33C:�                                    Bx�w�$  �          @�p����R@���>��@qG�C���R@���?   @~{C
=                                    Bx�w��  T          @�{���@��?+�@�(�C����@��?333@��HC�R                                    Bx�w�p  �          @�\)��ff@��?!G�@�  C�3��ff@��?(��@�\)C�R                                    Bx�x  [          @������H@\��?}p�@��CxR���H@\(�?��\A Q�C�                                    Bx�x�  �          @��
�љ�@J=q?�33AQ�C5��љ�@J=q?�
=A�CG�                                    Bx�x b  �          @�33��\)@,(�?�=qA(�C� ��\)@+�?�{A33C�{                                    Bx�x/  �          @�p�����@3�
?��A
=C�����@333?�\)A
ffC��                                    Bx�x=�  �          @����{@,��?��HA�
C�q��{@,(�?�p�A33C{                                    Bx�xLT  �          @�33�ָR@1�?�
=A\)Cz��ָR@1G�?��HA
=C��                                    Bx�xZ�  �          @�����
=@8��?�{A	p�C�q��
=@8Q�?��A��C�{                                    Bx�xi�  �          @�{��G�@.{?�=qA$��C0���G�@,��?�{A(��CO\                                    Bx�xxF  �          @�
=���H@,��?��A
=Cz����H@+�?���A#\)C��                                    Bx�x��  �          @�  ��z�@+�?��\A�C��z�@*=q?�ffA�
C�H                                    Bx�x��  �          @�\)�ڏ\@2�\?�z�A�HC�ڏ\@1�?���A�C޸                                    Bx�x�8  �          @����ۅ@   ?��\A�\C   �ۅ@�R?��A#
=C #�                                    Bx�x��  �          @�{���@'�?���A-C33���@&ff?��A2�HC\)                                    Bx�x��  
�          @�z�����@"�\?��A-p�C����@!G�?���A2�\C�                                    Bx�x�*  �          @�=q�ҏ\@>{?�(�A�C��ҏ\@<��?��\A   C�{                                    Bx�x��  �          @����H@1G�?��AffC�����H@0  ?�=qA$(�C�                                    Bx�x�v  �          @�ff��\@%�?�ffA:�\C���\@#�
?���A@  C !H                                    Bx�x�  �          @�����@,(�?�ffA7\)Cff���@*=q?���A=�C��                                    Bx�y
�  �          @�\)��33@)��?��
A6�HC���33@'�?�=qA<��C�R                                    Bx�yh  �          @��H���@$z�?�{A%G�C�����@"�\?�z�A+33C )                                    Bx�y(  �          @�Q���z�@+�?��A$��Cu���z�@)��?�Q�A+
=C��                                    Bx�y6�  �          @����H@#33?�33A-C�����H@!G�?���A4Q�C�                                    Bx�yEZ  �          @�  ��
=@�?��
A)G�C (���
=@�
?�=qA/�C aH                                    Bx�yT   �          @�\)��@�H?��A'33C ���@��?�{A-�C E                                    Bx�yb�  �          @�ff�ָR@\)?���A0z�C!�=�ָR@p�?�
=A6�HC!�                                    Bx�yqL  T          @�33��33@  ?���A;�C#  ��33@p�?�33AA��C#B�                                    Bx�y�  �          @���@�R?�=qA?33C"�\��@��?У�AE��C"�{                                    Bx�y��  �          @�33����?���?У�AF�HC$�q����?�z�?�
=AL��C%�                                    Bx�y�>  �          @�33��ff?���?�\)ADz�C%�\��ff?��
?�z�AI�C&)                                    Bx�y��  �          @����33?�?�ffAAC(����33?���?˅AF=qC(�R                                    Bx�y��  �          @�����G�?Y��?��AP��C-#���G�?L��?�z�AS\)C-xR                                    Bx�y�0  �          @��ᙚ@(�?�\)A(Q�C"���ᙚ@
=q?�
=A/33C"��                                    Bx�y��  �          A���@xQ�?�  @�Q�C�)��@vff?���@�
=CǮ                                    Bx�y�|  �          AG���p�@��?xQ�@�C33��p�@��\?�=q@�{C^�                                    Bx�y�"  �          A�
���@hQ�?���A�CxR���@e?�ffAG�C��                                    Bx�z�  �          A�����H@�ff?^�R@��C�����H@�p�?�  @�z�C��                                    Bx�zn  �          A\)�ڏ\@���?^�R@�p�CW
�ڏ\@��
?�G�@�(�C}q                                    Bx�z!  T          A\)�ᙚ@���?�@��CJ=�ᙚ@��?��A=qC�                                     Bx�z/�  �          A�����@��\?��Ap�CJ=����@�G�?�G�A&�HC��                                    Bx�z>`  �          A(���R@�G�?���A�\CB���R@�  ?�=qA��Cz�                                    Bx�zM  �          A	���@��?���@�\C0����@�=q?�G�A�Cff                                    Bx�z[�  �          A=q��@���?s33@У�C����@���?�=q@�{C                                      Bx�zjR  �          A�Ӆ@��?W
=@�p�C#��Ӆ@��\?z�H@޸RCL�                                    Bx�zx�  �          A���p�@���?fff@�33C����p�@���?��@�z�C�f                                    Bx�z��  �          A=q���H@�  ?�z�A�HC!H���H@��R?�ffA�RC^�                                    Bx�z�D  �          A���{@�  ?�33@�=qCh���{@��R?��A��C�f                                    Bx�z��  �          A�H��Q�@U?�  AEp�CJ=��Q�@Q�?�{AR=qC��                                    Bx�z��  �          A{��\@>�R?�{AS�C����\@:=q?��HA_33CY�                                    Bx�z�6  �          A�H��
=@'�@33Ah��C ����
=@#33@��As33C!+�                                    Bx�z��  "          A�\��ff@(��@G�Ae�C � ��ff@#�
@�Apz�C!                                      Bx�z߂  �          Aff��p�@-p�?��RAb=qC����p�@(Q�@�AmG�C s3                                    Bx�z�(  �          A�H��\)@+�?�
=AZffC G���\)@'
=@G�Aep�C                                     Bx�z��  �          A����z�@ff@
�HAr�HC"�H��z�@G�@  A|��C#n                                    Bx�{t  �          A	�����\@H��?�(�A8��C(����\@Dz�?�AE�C��                                    Bx�{  �          A
=��G�@`��?�=qA'33C����G�@\��?��HA5G�C)                                    Bx�{(�  �          A�
��
=@tz�?��\A=qC�3��
=@qG�?�AC                                      Bx�{7f  �          A�����@|(�?�{A  CǮ����@xQ�?�G�A   C)                                    Bx�{F  �          A
�H����@W�?�  A:ffC�)����@S33?��AH��C�                                    Bx�{T�  �          A
�\��Q�@aG�?�Q�A��C����Q�@]p�?���A'�
C��                                    Bx�{cX  �          A
{���@���?B�\@�=qC�����@���?k�@���C                                      Bx�{q�  �          A	���ff@�
=?Q�@�{Cz���ff@�?}p�@ҏ\C�3                                    Bx�{��  �          A�����@�\)?u@ʏ\C�=����@�{?��@�C�                                    Bx�{�J  
�          Az���@���?�33@��C���@�33?�{A=qC��                                    Bx�{��  �          A���
=@�ff?��\@���C����
=@��?�p�A   C��                                    Bx�{��  �          A���@�\)?�p�A{C����@�?�
=A�C�                                    Bx�{�<  �          A
{���@�33?#�
@���CaH���@�=q?Q�@�ffC��                                    Bx�{��  �          A��z�@�p��Ǯ�,(�C �f��z�@�\)���
��C c�                                    Bx�{؈  �          A   ��p�@�ff������\B�=q��p�@�Q쿢�\��B���                                    Bx�{�.  �          A*�\��@�=q>��@#33C�{��@�G�?Q�@��C��                                    Bx�{��  �          A,Q���G�@�p�>�@"�\C� ��G�@�z�?Q�@��C�)                                    Bx�|z  �          A,������@�R?0��@j=qC@ ����@�?��
@�{Cff                                    Bx�|   �          A,z����
@�>�Q�?�Q�C����
@�\?333@mp�C	                                    Bx�|!�  �          A-���=q@��?�@1G�C
���=q@��
?Y��@�  C
��                                    Bx�|0l  �          A-���G�@�>�{?�C
#��G�@�R?.{@e�C
:�                                    Bx�|?  �          A-G���@�33>�p�?�p�C	5���@�=q?8Q�@r�\C	L�                                    Bx�|M�  �          A,����
=@��>�@�C	�H��
=@�?L��@�  C	�                                     Bx�|\^  �          A+�����@�(�>k�?�p�C}q����@�?z�@G
=C�\                                    Bx�|k  �          A)�����
@��
>��?Q�C�����
@�33?�\@.�RC                                    Bx�|y�  �          A*=q���@�(�>���?˅C
���@�33?&ff@`��C.                                    Bx�|�P  �          A*�\��
=@�33>L��?���CaH��
=@��H?\)@?\)Cs3                                    Bx�|��  T          A+
=��\)@��
>��
?ٙ�Cc���\)@��H?.{@h��C}q                                    Bx�|��  �          A*=q��ff@�\>\@z�Cff��ff@陚?@  @���C��                                    Bx�|�B  �          A)���z�@���?\)@B�\C\)��z�@�  ?k�@���C��                                    Bx�|��  �          A*�R���@��H>L��?��C�����@�=q?�@Dz�C�
                                    Bx�|ю  �          A+
=���@�{>��@p�C	�����@��?G�@��C	ٚ                                    Bx�|�4  �          A*�H���@�{?�R@U�Ch����@���?xQ�@�\)C�{                                    Bx�|��  �          A+��p�@�  ?:�H@x��C&f�p�@�ff?��@�=qCW
                                    Bx�|��  �          A,���p�@��
?G�@���C
���p�@�=q?�33@ÅC
޸                                    Bx�}&  �          A/33�=q@���?Tz�@�33C
0��=q@�
=?��H@�33C
ff                                    Bx�}�  �          A/\)��@�{?Y��@�ffC
����@�z�?�p�@�C�                                    Bx�})r  �          A/\)���@�G�?5@o\)CaH���@�?�{@��\C�\                                    Bx�}8  �          A/\)��ff@��?8Q�@p  C�
��ff@�
=?�\)@��HC�                                    Bx�}F�  �          A/����@�33?��@I��C&f���@�?�G�@���CQ�                                    Bx�}Ud  �          A/��   @�
=?&ff@Z=qC��   @�?�ff@���C	)                                    Bx�}d
  �          A0����z�@��>���@z�C�\��z�@�ff?Q�@�  C�                                    Bx�}r�  �          A0�����@�  ?!G�@P��C	5����@�ff?��@�z�C	aH                                    Bx�}�V  �          A0���  @��?k�@�Q�C
p��  @�  ?���@ۅC
��                                    Bx�}��  �          A2�H�33@��?B�\@xQ�C	Y��33@�  ?�@���C	��                                    Bx�}��  �          A0��� ��@�G�?#�
@Tz�C�� ��@�?��@�  C	�                                    Bx�}�H  �          A-���=q@�>�G�@z�C
=��=q@�Q�?\(�@��C+�                                    Bx�}��  T          A/33���@�33?��@8Q�C�R���@��?z�H@��
C#�                                    Bx�}ʔ  �          A.�H��(�@�G�?8Q�@p��CJ=��(�@�\)?��@�Q�C}q                                    Bx�}�:  �          A.=q��\)@��
?Q�@��HC	G���\)@��?��R@љ�C	��                                    Bx�}��  �          A,�����
@�(�?333@n{C�{���
@�\?���@�\)C	�                                    Bx�}��  �          A-���\@�
=>�
=@p�CW
���\@�?Y��@�  C}q                                    Bx�~,  �          A%���ff@��
?(�@X��CL���ff@�=q?��\@�{C}q                                    Bx�~�  �          A ����33@�G�?E�@��
C	B���33@׮?�z�@�z�C	�                                     Bx�~"x  �          A"�R����@ٙ�?5@~�RC	������@�  ?���@�\)C
!H                                    Bx�~1  �          A �����@�  ?J=q@�  C	�����@�{?�Q�@أ�C	�                                    Bx�~?�  �          A Q����@ڏ\?��@I��C����@�G�?u@�
=C	�                                    Bx�~Nj  �          A Q���@���>�ff@%�Cc���@ۅ?\(�@�C��                                    Bx�~]  T          A$  ��z�@��H>���?�\)C:���z�@ᙚ?B�\@�  C\)                                    Bx�~k�  �          A"�\��@��>�33@   C	
��@��
?E�@�33C	:�                                    Bx�~z\  �          A"ff��ff@�(�>�33?��RC	E��ff@�33?E�@�33C	k�                                    Bx�~�  �          A-���  A Q��\)�"=qC)��  A������
=C                                    Bx�~��  �          A4(���  Az��k�����B����  A�
�HQ���p�B�k�                                    Bx�~�N  �          A2�H��Q�A���z���z�B�8R��Q�Ap����H���
B�                                    Bx�~��  �          AQ����@˅>��
?���C	�����@ʏ\?8Q�@�=qC
)                                    Bx�~Ú  �          A33��\)@���?Y��@�z�C:���\)@��R?�(�@��C��                                    Bx�~�@  �          A����\)@�33?z�@c�
C
�{��\)@��?u@��C\                                    Bx�~��  �          A\)����@�=q?=p�@��\C
������@�Q�?�{@�{C
�                                    Bx�~�  �          A�����H@�?�@X��C�����H@�z�?k�@�33C	�                                    Bx�~�2  �          A����z�@Å?   @I��C	^���z�@�=q?aG�@�33C	�{                                    Bx��  �          A  ����@��?�@W�C������@Å?k�@��C��                                    Bx�~  T          A�
��=q@��
>��H@G�C	���=q@�=q?aG�@��HC	E                                    Bx�*$  �          A���Ӆ@���>�(�@-p�C	\�Ӆ@Å?Q�@�ffC	=q                                    Bx�8�  �          A(���(�@�33>L��?�G�C	^���(�@\?
=@o\)C	}q                                    Bx�Gp  �          A  ��@��ýL�;��
C	���@���>�{@	��C	��                                    Bx�V  �          A���z�@��þ��
�z�C	� ��z�@�G�=�\)>�
=C	�R                                    Bx�d�  �          A  ��z�@�녾��E�C	�)��z�@\��Q��C	�                                    Bx�sb  �          A33��G�@�33���G�C�q��G�@�33>�\)?�\C	�                                    Bx��  �          A�H��p�@�{�\�(�C\��p�@�ff<��
>#�
C                                      Bx���  �          A�
��  @�ff��G��333C\)��  @ƸR���W
=CJ=                                    Bx��T  �          AG���(�@���33��C	���(�@�{=u>ǮC��                                    Bx���  �          A�R�׮@ƸR����h��C	W
�׮@�ff>��?�33C	\)                                    Bx���  �          A(���{@���<�>8Q�CG���{@�(�>�ff@333CY�                                    Bx��F  �          A����
=@��=��
>�C^���
=@�z�?   @FffCu�                                    Bx���  
�          A���׮@�\)��\)��G�C)�׮@�
=>�Q�@p�C&f                                    Bx��  �          A������@׮�W
=���C�\����@�\)>u?��RC�\                                    Bx��8  �          AQ���=q@љ��L�Ϳ�p�C
��=q@љ�>u?��RC�                                    Bx���  �          A  ����@ȣ׽u��Q�C	B�����@�Q�>�Q�@  C	O\                                    Bx���  �          A���Q�@��ý��Ϳ��C	{��Q�@ȣ�>���@�C	�                                    Bx��#*  �          A�\��  @�{=�G�?.{C	����  @�p�?�@R�\C	�)                                    Bx��1�  �          A����{@�p�=�\)>�G�C	W
��{@���>��H@Dz�C	n                                    Bx��@v  �          A����@�(���G��+�C�3���@�(�>��R?��RC��                                    Bx��O  �          A��У�@�p����R��(�C�\�У�@�=�G�?8Q�C��                                    Bx��]�  �          A�R�θR@Ϯ��
=�)��C�)�θR@�  <��
>�C��                                    Bx��lh  �          A�R��
=@Ϯ�Ǯ�(�C�3��
=@�  =L��>�{C�f                                    Bx��{  T          A�\��z�@�녾�\)���
C@ ��z�@��>.{?��C:�                                    Bx����  �          A��ҏ\@�
=���B�\Cu��ҏ\@�
=>��
@ ��C}q                                    Bx���Z  T          A  ����@�p��B�\��33C�����@�p�>��?���C
=                                    Bx���   �          A�H��33@��
���8Q�C���33@�33>��@%�C)                                    Bx����  �          A
=��ff@�G�<#�
=�\)C���ff@���>�ff@333C�
                                    Bx���L  �          A�
���
@�{�\)�\(�C����
@�{>��R?�C�=                                    Bx����  �          A���ҏ\@љ����H�@��C#��ҏ\@�=q���8Q�C\                                    Bx���  
�          A
=��G�@Ϯ�\(����RCJ=��G�@��þ�����C�                                    Bx���>  T          A�R��33@�{�(��l��CǮ��33@�
=�#�
�p��C��                                    Bx����  T          A����H@Ϯ>Ǯ@�C	�\���H@�{?Y��@�G�C	�                                     Bx���  �          A�R��\@�=q>�z�?޸RC	!H��\@���?B�\@�\)C	J=                                    Bx��0  �          A\)��p�@У�>aG�?��\C	�R��p�@Ϯ?+�@\)C	�)                                    Bx��*�  T          A���33@Ӆ���H�8��C	{��33@�(����
�\)C	                                      Bx��9|  T          A���\)@��}p����HC=q��\)@׮��\�AG�C�                                    Bx��H"  �          A�\��Q�@��H��R�l��C
�)��Q�@��
�.{���\C
�q                                    Bx��V�  �          A�R��{@�{���6ffC
&f��{@θR���
��C
�                                    Bx��en  �          A����@�ff�L�Ϳ�Q�C
ff���@�ff>��?\C
h�                                    Bx��t  �          A�\��p�@�\)��G��(��C	����p�@�
=>�33@�C	�                                    Bx����  �          A���G�@�G����B�\C33��G�@ȣ�>�
=@�RCE                                    Bx���`  �          A  ��z�@��R>B�\?�z�C���z�@�?�R@n�RC:�                                    Bx���  �          A(����
@�p�>���@��C\)���
@�(�?L��@��HC��                                    Bx����  �          A��\@�G�?333@�ffCG���\@�\)?��@�33C��                                    Bx���R  �          A�H���@�ff?�\@?\)C.���@���?\(�@��HCn                                    Bx����  �          A�33@��>�\)?�\)C{�33@���?#�
@o\)C@                                     Bx��ڞ  �          Aff�{@��׽��0��C�)�{@�Q�>��?�  C��                                    Bx���D  �          A=q�=q@�\)���
��C��=q@��=u>�33C�                                    Bx����  �          Aff���@��H�&ff�o\)C@ ���@�(������  C
                                    Bx���  �          A�R��@��H�@  ���\CO\��@�(���Q��C)                                    Bx��6  �          A�H��
@�Q��R�dz�C5���
@�G��aG���  C\                                    Bx��#�  �          A����@�ff�^�R����C�����@�  ���/\)C}q                                    Bx��2�  �          A�H��H@��׿�{����C�3��H@��H�333���C��                                    Bx��A(  �          A��� ��@�p���{�   C
� ��@�  �xQ����C�                                    Bx��O�  �          A����@�z῰���p�CB���@�\)�z�H��Q�C�
                                    Bx��^t  �          A  ��\)@���33��C���\)@��׿�  ���
CW
                                    Bx��m  �          Aff� z�@��������C\� z�@�ff�z�H��{C��                                    Bx��{�  �          A\)�@����
=��CL��@�ff��G����HC޸                                    Bx���f  �          A!���
@�ff������{CW
��
@��ÿp�����C�                                    Bx���  �          A#33�33@�(���p���CY��33@�
=������
C�                                    Bx����  �          A$Q���R@��H���R��{CJ=��R@���J=q��C�3                                    Bx���X  T          A"�\�G�@��\�Ǯ���C.�G�@���33��  C��                                    Bx����  �          A!���\)@�33�ٙ��33C��\)@��R�����=qC+�                                    Bx��Ӥ  �          A�\��@��ÿ����CE��@�zῗ
=�ڏ\CǮ                                    Bx���J  �          A=q��
@�p���(�� ��C�H��
@�G������G�CT{                                    Bx����  �          A���@�
=��\�&=qC)���@��\��z���C�                                    Bx����  �          A(���@���33�B�RC  ��@�  ���H�!�CG�                                    Bx��<  �          A(�� (�@��ÿ�=q�1C&f� (�@��Ϳ�p���HC�                                    Bx���  �          A  � ��@�  �ٙ��$��CxR� ��@��������\C��                                    Bx��+�  �          A���ff@�p���  ��C���ff@�  �\(���C��                                    Bx��:.  �          Az���
=@�ff�z��e�C�H��
=@�\)�8Q쿌��C�q                                    Bx��H�  �          A  ��R@��(�����HC����R@��R�����{C�=                                    Bx��Wz  �          A\)��R@�z����!G�C)��R@���#�
�L��C
=                                    Bx��f   �          A�H��(�@��!G��}p�C����(�@��R�k����HCz�                                    Bx��t�  �          A\)����@�{��z��=qC������@�ff=�\)>�(�C�                                     Bx���l  �          A����@�(��L�Ϳ�p�C33���@�(�>8Q�?�{C33                                    Bx���  �          A����
@�p�>���@�RCٚ���
@��
?G�@��\C�                                    Bx����  �          A�
��@��ÿ(�����\C(���@�녾k���Q�C                                      Bx���^  �          A�����@��\���R��C�3����@��H=��
?�\C��                                    Bx���  �          Aff����@������uC�����@�33>u?�G�C�                                    Bx��̪  �          A\)��@��R����У�C����@��R>\)?\(�C��                                    Bx���P  �          A
=��@������k�CQ���@�\)>��?˅CW
                                    Bx����  �          A
=��@�=q���Ϳ!G�C�f��@��>�z�?��C��                                    Bx����  	�          A����H@�
=>B�\?�
=Cz����H@�ff?�@c33C��                                    Bx��B  T          A\)��@���=�G�?(��C{��@�  >��H@C�
C33                                    Bx���  T          A�
��{@�
=��Q�
=qC����{@��R>��R?���C��                                    Bx��$�  T          A�\��@�=q��  ��  C�\��@��\>#�
?}p�C��                                    Bx��34  �          A����
=@��H�\��\C{��
=@�33<�>.{C                                    Bx��A�  T          A{���R@��>��?^�RC
=���R@��H?
=q@L��C(�                                    Bx��P�  �          A\)��H@��H?��
A�C�)��H@�
=?��A/\)C}q                                    Bx��_&  �          A���\@�p�?��HA=qCk���\@���?�=qA(��C�                                    Bx��m�  �          A���@���?�Q�@�33C����@�ff?���A�C&f                                    Bx��|r  T          A\)���@��?�\)@ϮC.���@�Q�?�G�A�
C�f                                    Bx���  �          A��(�@���?Y��@��C33�(�@�ff?�  @�
=C��                                    Bx����  �          A��\)@�p�>�p�@��C^��\)@�(�?G�@�  C��                                    Bx���d  �          A�H��R@��
?   @8��C}q��R@��?h��@��C��                                    Bx���
  
�          A\)� Q�@�z�>�@0  C�3� Q�@��H?fff@�ffC�                                    Bx��Ű  
�          A"{�ff@�\)��Q��C�R�ff@�
=>�{?�
=C                                    Bx���V  �          A&�R��@У׿
=q�<(�C8R��@�G��u��{C!H                                    Bx����  T          A'����@��
���H�*=qC}q���@�zἣ�
�\)Ch�                                    Bx���  �          A&ff�  @ʏ\��33����C}q�  @��H=���?z�Cs3                                    Bx�� H  
Z          A&ff�@���=q��(�C���@�>8Q�?}p�C��                                    Bx���  �          A%�(�@Ǯ��  ��{C���(�@Ǯ>L��?���C�3                                    Bx���  �          A,Q��  @�33=�\)>�p�C�  @ʏ\?�@333C)                                    Bx��,:  �          A-���@�z�>#�
?W
=CǮ��@˅?�R@QG�C�f                                    Bx��:�  �          A+33�
=@�  =���?��C@ �
=@�\)?��@<��CY�                                    Bx��I�  �          A+�
�Q�@�
==�Q�>��HC�H�Q�@�ff?
=q@7�C��                                    Bx��X,  T          A/����@ƸR�#�
��C�=���@�{>�G�@�\C��                                    Bx��f�  
�          A0(���@�z���Q�C���@��ͼ��
��Q�C                                      Bx��ux  �          A+
=�
�H@�p�>��?O\)C�)�
�H@�z�?
=@K�C��                                    Bx���  �          A$�����@�33>B�\?�=qC����@�=q?!G�@_\)C�\                                    Bx����  �          A\)��@��<#�
=#�
C� ��@�
=>�
=@��C��                                    Bx���j  �          A��{@�33?W
=@���C���{@���?�(�@�G�Ch�                                    Bx���  �          A���=q@���?��@aG�Ck��=q@�33?z�H@���C��                                    Bx����  �          A{��33@���?(�@i��C�{��33@�\)?�  @�\)C)                                    Bx���\  �          A=q����@�  ?�R@j=qC&f����@�{?�  @�
=Cn                                    Bx���  �          A33��\)@��R?333@��C����\)@���?�=q@�z�C�3                                    Bx���  "          AG�� ��@��?\)@QG�C!H� ��@��?s33@�=qCc�                                    Bx���N  T          A�\��=q@���?(�@g�C0���=q@��H?�G�@�
=Cu�                                    Bx���  �          A�\��(�@���?0��@��\C�R��(�@��R?���@�(�CE                                    Bx���  
�          A�H��@�ff?.{@���C^���@�z�?��@�p�C��                                    Bx��%@  �          A�R���R@�{?.{@��C�����R@�(�?��@��C�                                    Bx��3�  �          Aff����@���>��H@;�C�����@�33?c�
@���CO\                                    Bx��B�  �          A33��z�@��?&ff@uC����z�@���?��@�z�C�)                                    Bx��Q2  �          AQ����R@��?Q�@�G�C�
���R@�G�?���@�\C.                                    Bx��_�  �          A���=q@�33?xQ�@�33C
=��=q@���?��A z�Ch�                                    Bx��n~  �          A-���p�@�p�?:�H@w
=CT{�p�@�33?��H@�(�C��                                    Bx��}$  T          A2ff��@�>��
?У�Cu���@�ff?O\)@�{C�)                                    Bx����  �          A1��ff@���?�\@'�C
�ff@�\)?�G�@�{CJ=                                    Bx���p  �          A0Q����@�>�@Q�C
�H���@�{?s33@��RC�                                    Bx���  �          A.ff�=q@�>�?.{C
Y��=q@�
=?�R@R�\C
s3                                    Bx����  T          A.�\���@�p��u��{C	����@��>�G�@z�C	+�                                    Bx���b  T          A/���@�<#�
=L��C	\)��@��?�@,(�C	n                                    Bx���  �          A/�� ��@�
=<��
=�Q�C	�� ��@�ff?�@.�RC	+�                                    Bx���  �          A0���ff@�
=>��?G�C	}q�ff@�ff?&ff@X��C	��                                    Bx���T  �          A1���R@�
=>��?�{C	����R@�{?E�@}p�C	�                                    Bx�� �  �          A0��� ��@�>�\)?���Cٚ� ��@�Q�?J=q@��\C�q                                    Bx���  �          A1�����@�  �L�;�  C�\����@��>��@(�C�)                                    Bx��F  Z          A1p���@�  ��Q��C����@��>�(�@��C��                                    Bx��,�  �          A0����{@�{�#�
�k�C���{@�p�>��@�C��                                    Bx��;�  �          A1p�� (�@�����
���
CG�� (�@�z�>��H@#33CW
                                    Bx��J8  �          A0z��   @�\�k�����C���   @�\>�=q?�z�C��                                    Bx��X�  T          A0  ����@�(�?   @'�C������@�\?�G�@�  C+�                                    Bx��g�  �          A/����H@��>�=q?��C�f���H@�(�?E�@���C�                                    Bx��v*  �          A/����@���>�p�?�Q�C���@�?^�R@��C��                                    Bx����  �          A/�
��(�@���>��
?�z�C�{��(�@��
?Q�@���C�R                                    Bx���v  �          A1��\)@��R>���?�(�C����\)@�?Tz�@��C�                                    Bx���  �          A0�����@��R>aG�?�\)C�R���@�?8Q�@mp�C�
                                    Bx����  �          A1��33@��>��?J=qC�q��33@��\?(��@Z=qC{                                    Bx���h  Z          A1G���=q@��H>L��?�ffC���=q@��?5@j=qC�                                    Bx���  �          A0�����@�Q�>�{?��
CW
���@�\)?W
=@��Cz�                                    Bx��ܴ  T          A0��� z�@�=q>��@p�C�� z�@���?u@�
=Cٚ                                    Bx���Z            A1���@�=q?+�@^�RC�q��@�Q�?�z�@��RC	5�                                    