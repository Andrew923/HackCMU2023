CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230403000000_e20230403235959_p20230404021744_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-04T02:17:44.566Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-03T00:00:00.000Z   time_coverage_end         2023-04-03T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxtF��  T          @�G�@�녿Ǯ@\)A��C��)@��� ��?�=qA4Q�C�33                                    BxtF�&  �          @���@�Q��@33A�C�Q�@�Q��(Q�?���A1C���                                    BxtF��  �          @��@�{�33?��HA�C�/\@�{�1�?
=@�(�C��\                                    BxtF�r  T          @�Q�@�p���=q@�A��C�� @�p��(Q�?�p�AN�HC�O\                                    BxtF�  �          @�ff@�Q�˅@�A�Q�C��q@�Q����?n{AC�^�                                    BxtF�  
�          @�{@�{���@   A���C���@�{�p�?�Az�HC��f                                    BxtF�d  x          @��R@�Q쿙��@A�=qC��3@�Q��  ?�{AhQ�C�XR                                    BxtG
  
Z          @�p�@�z��@�
A�33C�
@�z�� ��?^�RA�C�Ф                                    BxtG�  "          @�{@�G���?�p�AR=qC��f@�G��   ���Ϳ���C�:�                                    BxtG!V  �          @��R@�  �>��@�=qC���@�  ���R�@  ��C�&f                                    BxtG/�  �          @�ff@��
����?�  A*ffC��@��
��G�=�Q�?k�C�~�                                    BxtG>�  "          @�p�@�p��{@G�A�\)C���@�p��<��?��@�(�C�#�                                    BxtGMH  �          @�@��H�)��?�{A=G�C���@��H�5��ff��  C�!H                                    BxtG[�  "          @�\)@�ff�0�׼��
���C��=@�ff���(��zffC���                                    BxtGj�  
�          @���@��0  ����\)C�q@���\)�@���C�g�                                    BxtGy:  T          @���@��
�333��(����RC��@��
����I���
G�C�p�                                    BxtG��  "          @���@�p��0���z����HC�G�@�p������Z�H�Q�C�k�                                    BxtG��  
�          @�  @��R�@�׿�G��*=qC���@��R��\�(��ӮC��)                                    BxtG�,  �          @�@|(��L��?���Aip�C�p�@|(��Z�H������C���                                    BxtG��  T          @�Q�@g
=�z�@\(�B�
C��f@g
=�q�?�(�A�\)C��)                                    BxtG�x  �          @�  @vff���@G
=B	�HC�
=@vff�j�H?�33Ak\)C�+�                                    BxtG�  �          @�p�@�33�(�?��RA�=qC�Y�@�33�HQ�>�
=@�\)C�(�                                    BxtG��  �          @��@�ff�.{?��Af�RC�L�@�ff�AG���\)�>{C��\                                    BxtG�j  "          @�33@�Q��<(�?�33AIG�C���@�Q��G
=������C��\                                    BxtG�  �          @��@��
�9��?��RAU�C�33@��
�HQ����(�C�8R                                    BxtH�  �          @�(�@����*�H?��
A0Q�C��{@����3�
����Q�C�*=                                    BxtH\  �          @�33@�(���Q�?s33A$(�C�%@�(��	���L���(�C�R                                    BxtH)  T          @�33@�����?J=qA�
C�~�@���G����R�X��C��{                                    BxtH7�  �          @���@�p���p�?
=q@�C�H�@�p���G���G���33C�%                                    BxtHFN  
�          @��@�  �{?�ffA5C�|)@�  �(�����1G�C�e                                    BxtHT�  
�          @���@��R��?c�
A  C��3@��R�녾L���
�HC��q                                    BxtHc�  �          @�z�@�z����>��R@S33C��H@�z���\�aG���C���                                    BxtHr@  �          @�z�@�=q�޸R=L��?�C��@�=q��G��^�R���C��3                                    BxtH��  "          @��H@�33����>#�
?ٙ�C�xR@�33���.{���C�9�                                    BxtH��  �          @�=q@�ff���?}p�A+�C�ٚ@�ff��z�#�
��Q�C�p�                                    BxtH�2  
�          @���@��H���?}p�A,��C�ff@��H������C�.                                    BxtH��  �          @�(�@�z��  ?�
=Aw�C�
@�z��*=q=L��?��C�
                                    BxtH�~  
Z          @��
@�ff�!G�    �L��C��f@�ff�	������d(�C��{                                    BxtH�$  �          @�Q�@�G��,�;.{���C�L�@�G��{��ff����C��{                                    BxtH��  �          @�G�@�\)�5�������p�C��\@�\)�p������C�|)                                    BxtH�p  �          @�@�p���>W
=@�C��@�p���녿fff�z�C��{                                    BxtH�  "          @�Q�@�z����>.{?�Q�C�=q@�z��(������6�RC�|)                                    BxtI�  �          @��H@�p��(Q�=L��?�C��{@�p��G������Z�RC���                                    BxtIb  �          @���@�33�&ff?�R@�z�C���@�33�#33�Q��Q�C�&f                                    BxtI"  �          @��@���!�?
=@��C��@���{�O\)�  C�h�                                    BxtI0�  �          @��@��
�#33������C�4{@��
�
=q�����b�\C��                                    BxtI?T  �          @���@�p���Q�?�@�z�C�� @�p���
=�\)���C�˅                                    BxtIM�  �          @��@�
=�\)?s33A�RC��
@�
=�=q��{�c�
C�
=                                    BxtI\�  �          @�33@�(����?��
AT  C��
@�(��-p��\)��C��                                     BxtIkF  �          @��
@����#33?�G�A(  C��@����,�;�ff���RC�N                                    BxtIy�  �          @�z�@����4z�?E�@��\C��=@����3�
�L����RC��{                                    BxtI��  T          @���@�Q��%�>��?��C�]q@�Q���\�����D��C��
                                    BxtI�8  �          @�(�@��\��>�33@fffC��@��\���fff�  C�ff                                    BxtI��  T          @�z�@���0��?��@�(�C�7
@���*�H�h���z�C���                                    BxtI��  �          @�{@�33�C�
?�ffATz�C�.@�33�S33����(�C�0�                                    BxtI�*  �          @�{@���C�
?
=@�\)C���@���:�H�����-G�C�,�                                    BxtI��  "          @���@��\�Q�>k�@ffC�5�@��\�;����R�t��C��3                                    BxtI�v  �          @���@�z��L(�=�\)?0��C���@�z��1G���=q��z�C��3                                    BxtI�  �          @�\)@�=q�Dz�=�G�?�{C��)@�=q�,(���p��p��C�e                                    BxtI��  �          @�  @��\�C�
>���@}p�C���@��\�5���p��F�\C���                                    BxtJh  �          @���@�
=�HQ�?n{A�C�7
@�
=�J�H�O\)���C�3                                    BxtJ  �          @���@���P��?���A4��C�]q@���XQ�333��Q�C��                                    BxtJ)�  �          @�  @��H��?˅A���C�W
@��`�׾u�(�C���                                    BxtJ8Z  T          @�Q�@�z��#33@Dz�B ��C�,�@�z��p��?�=qAU��C�޸                                    BxtJG   �          @��R@�(��:=q@(��A���C�h�@�(��vff?G�@��
C�}q                                    BxtJU�  �          @�@�33�K�@��A�=qC�)@�33�x��>��
@L(�C�C�                                    BxtJdL  �          @�@���7�@�
A�z�C�*=@���i��?�@�Q�C��                                     BxtJr�  �          @��@�{�1G�@Q�A���C�@�{�^{>��@�C�H                                    BxtJ��  �          @��\@���� ��@�A�\)C�+�@����Vff?8Q�@�C�c�                                    BxtJ�>  T          @�  @�  �5?�\)Ag�C�� @�  �H�þ�=q�333C���                                    BxtJ��  �          @�
=@��
�
�H?�(�A���C�p�@��
�7�?
=@�p�C�R                                    BxtJ��  
�          @���@�p��333?fffA
=C��=@�p��7��&ff��\)C�8R                                    BxtJ�0  T          @�  @�ff�Fff?^�RAp�C���@�ff�G
=�Tz��\)C���                                    BxtJ��  T          @�\)@�33�dz�u�\)C�|)@�33�<���G���=qC�                                      BxtJ�|  b          @��R@j�H�}p��Q��
�\C�p�@j�H�?\)�.{����C�o\                                    BxtJ�"            @�{@tz��p�׿aG��z�C��{@tz��333�*�H��G�C���                                    BxtJ��  "          @��R@x���j�H��  �)G�C�S3@x���*=q�.�R��ffC��
                                    BxtKn  
�          @�ff@j=q�g
=��(���{C��\@j=q�\)�Tz����C�E                                    BxtK  
�          @��R@�=q�e�
=q��{C�J=@�=q�4z���\��Q�C���                                    BxtK"�  �          @�ff@����QG��(���  C�b�@���� ���(����C���                                    BxtK1`  �          @�ff@����I���0����
=C�8R@����������HC�޸                                    BxtK@  �          @���@����Mp��#�
�ٙ�C��\@����������ffC��                                    BxtKN�  "          @�(�@�33�\(�?�@�Q�C�@�33�N�R����^�\C��                                    BxtK]R  T          @���@�
=�U>�  @(��C���@�
=�@�׿�(��~{C�B�                                    BxtKk�  
�          @�z�@����_\)?&ff@�ffC�� @����U�����M�C�=q                                    BxtKz�  "          @�@xQ��i��?Q�Az�C�\)@xQ��dzῐ���B{C���                                    BxtK�D  
�          @��@n{�r�\>�@�z�C�<)@n{�`�׿�  ���\C�P�                                    BxtK��  �          @�{@���\��?}p�A(z�C���@���^�R�aG��{C��3                                    BxtK��  �          @�ff@�ff�Tz�>�
=@���C��
@�ff�E�����a�C�޸                                    BxtK�6  "          @��@o\)�AG�?��
An=qC���@o\)�P  �������C�~�                                    BxtK��  "          @��\@�{�\)@1�A��C��@�{�U�?��
AV�\C���                                    BxtK҂  T          @�G�@����N{?��HA�=qC��@����h�þ\)���HC�W
                                    BxtK�(  �          @�Q�@�  �P��?�
=AFffC�>�@�  �Z=q��R��
=C���                                    BxtK��  
�          @��@��R�W
=��=q�8��C���@��R�1G������RC�Z�                                    BxtK�t  �          @�z�@�  �O\)��ff���\C�W
@�  �%��G����\C�aH                                    BxtL  �          @��
@�  �,�Ϳ�=q�9�C��H@�  ��ff��\��(�C��                                    BxtL�  �          @��@�����\��(�C�7
@������ff��z�C���                                    BxtL*f  �          @���@�p��+�>�Q�@s33C��@�p��   ���
�0z�C���                                    BxtL9  �          @�@��R�B�\?
=q@�Q�C�ٚ@��R�9����ff�2{C�s3                                    BxtLG�  �          @�(�@���G�>�
=@��RC�1�@���:�H�����M��C�q                                    BxtLVX  �          @��
@���G�>���@N{C�7
@���6ff����_33C�g�                                    BxtLd�  �          @�(�@���R�\>�
=@�C�
@���C�
���
�\  C��                                    BxtLs�  �          @�z�@�\)�Tz�>Ǯ@�(�C���@�\)�DzῨ���b=qC��q                                    BxtL�J  T          @�z�@�\)�U�>�?�=qC��H@�\)�<�Ϳ�ff��C��=                                    BxtL��  �          @�(�@����Z�H��\)�8Q�C�Ff@����;���G����\C�c�                                    BxtL��  
�          @�p�@��H�c33�#�
��
=C���@��H�C�
��ff����C��q                                    BxtL�<  T          @��
@���aG�����=qC��@���B�\��\���C��                                    BxtL��  �          @��
@��W�>8Q�?�
=C��\@��@�׿�G�����C�
                                    BxtLˈ  �          @�G�@�z��QG��#�
���C��q@�z��0�׿�G���  C�,�                                    BxtL�.  T          @���@�
=�J�H��Q�}p�C���@�
=�-p���33����C���                                    BxtL��  �          @���@����B�\��
=���
C�K�@�������\)���RC�/\                                    BxtL�z  �          @���@���!녿��ȣ�C��@�녿�z��  ���C��H                                    BxtM   T          @�G�@�G���\)�W
=��C�Q�@�G����H��z���z�C��f                                    BxtM�  �          @�G�@�(��!녾aG����C���@�(����Q��}G�C��R                                    BxtM#l  "          @�Q�@�Q��(���  �,��C���@�Q�����ff�d��C���                                    BxtM2  
�          @�Q�@�����R�#�
��C��H@�����H���\�2ffC�P�                                    BxtM@�  
�          @��@�Q��
=q>��R@W�C��\@�Q��녿J=q��C�t{                                    BxtMO^  T          @�
=@�  �Q�>��
@^{C��@�  � �׿E��z�C���                                    BxtM^  �          @�
=@�녿��R>�{@mp�C��H@�녿�33�.{����C�8R                                    BxtMl�  "          @�\)@��R��>�G�@�33C���@��R�
=�0�����C��                                    BxtM{P  �          @���@��	��?�G�A1��C���@����.{���C��{                                    BxtM��  �          @���@�\)��>L��@
=qC��f@�\)���xQ��(��C��                                    BxtM��  T          @�Q�@��޸R�������C�Ff@����ÿ����S�C�}q                                    BxtM�B  
�          @�  @�{��׾�G�����C�)@�{��G����R��z�C��{                                    BxtM��  �          @��@�p��R�\?0��@��HC��)@�p��Mp���G��.{C�8R                                    BxtMĎ  T          @�p�@��Tz�?aG�A�C���@��U��Y�����C��H                                    BxtM�4  �          @�p�@\)�c�
?J=qA�RC�q@\)�`  ��ff�2�\C�b�                                    BxtM��  T          @��
@����]p�?
=@��HC��)@����S�
���I��C�\)                                    BxtM��  �          @�(�@�(��B�\?(��@�33C���@�(��>�R�fff�=qC���                                    BxtM�&  T          @��
@�=q�\(�>k�@�RC���@�=q�G���(���(�C�C�                                    BxtN�  T          @��@y���g�������C��{@y���<(��
=q���\C�~�                                    BxtNr  �          @��@���G�����\)C��\@���{��(���G�C���                                    BxtN+  
�          @���@�p��:�H����У�C�L�@�p���R��ff���C�^�                                    BxtN9�  �          @��@�  �H�þ����ffC��@�  �#33�����=qC���                                    BxtNHd  T          @���@�G��+���G����C��q@�G��Q��
=��ffC�s3                                    BxtNW
  1          @�Q�@��3�
��  �.�RC��@����=q���C�{                                    BxtNe�            @�\)@�z��3�
>�?�p�C���@�z��!G����R�\  C��                                    BxtNtV  
�          @���@�p��5�>8Q�?�
=C���@�p��$zῚ�H�UC��=                                    BxtN��  �          @�G�@��H�?\)>��@1�C��q@��H�/\)��(��U�C��                                     BxtN��  T          @���@�G��C�
=�?���C�:�@�G��/\)��\)�p��C��3                                    BxtN�H  �          @�  @z=q�Y��?&ff@�{C�}q@z=q�S33����;�
C��\                                    BxtN��  T          @��R@����,�;����g�C�<)@����p���=q����C��=                                    BxtN��  
�          @��R@�ff��ÿ���=�C��f@�ff��=q�����
C�5�                                    BxtN�:  �          @���@��R���ÿ����RC��\@��R���{��=qC���                                    BxtN��  
�          @���@�����\��(��W33C��)@����xQ������C�J=                                    BxtN�  T          @�G�@��\��\��  �Z{C��@��\�u��Q���=qC�c�                                    BxtN�,  �          @��\@�(���(��O\)���C��@�(������z���
=C�Q�                                    BxtO�  w          @�=q@�=q��O\)�z�C�J=@�=q�������H��=qC��                                    BxtOx  T          @�  @��׿�p��^�R�Q�C���@��׿�=q���H��{C�:�                                    BxtO$  �          @���@�녿��H�h����
C��@�녿��
�޸R��C���                                    BxtO2�  T          @�Q�@���� �׿B�\���C��@�����33��\)���C��R                                    BxtOAj  
�          @�Q�@�\)�
=�aG��{C��R@�\)��Q���
��=qC��=                                    BxtOP  "          @�=q@��Ϳ�녿E��G�C�s3@��Ϳ�ff��=q��{C��{                                    BxtO^�  �          @��@�ff��ff�����;33C�L�@�ff�\(���
=��{C��                                    BxtOm\  
�          @�G�@���Ǯ��Q��P��C�.@���O\)�����z�C�T{                                    BxtO|  
�          @��@�{���
�����P��C�ff@�{�G����
����C��                                    BxtO��  "          @��@�{���
��Q��P(�C�aH@�{�G����
���\C�|)                                    BxtO�N  
}          @���@�{������uG�C�XR@�{��p����
��  C�ٚ                                    BxtO��  �          @�  @�{��\)�����tz�C�� @�{���ÿ�  ���C�3                                    BxtO��  �          @���@��H�˅��=q�j�\C���@��H�G���
=���C�q�                                    BxtO�@  
�          @���@�ff��z῔z��K\)C��@�ff�333��Q���p�C���                                    BxtO��  T          @�Q�@�����H��\)�E�C��=@�����Ϳ��Q��C���                                    BxtO�  
(          @���@�=q�
=q>�33@u�C���@�=q�z�333��(�C�aH                                    BxtO�2  �          @���@�33�
=>��@5C�=q@�33���R�B�\�  C��)                                    BxtO��  "          @�Q�@����
=q>���@�z�C�Ф@����ff�(����RC�"�                                    BxtP~  T          @���@��R��녾�(���
=C��H@��R���R��33�J=qC��R                                    BxtP$  "          @���@�녿��׿   ���RC�ff@�녿z�H����>ffC��                                    BxtP+�  
�          @�G�@�����H    <�C�"�@����p��h���33C�J=                                    BxtP:p  T          @���@�  ��33��  �.{C���@�  ����}p��-G�C���                                    BxtPI  
�          @��@��H��Q쿮{�pQ�C��{@��H�#�
��\)��C�7
                                    BxtPW�  
�          @�\)@�=q��33�ff���C�u�@�=q����!G���=qC�Z�                                    BxtPfb  �          @�\)@�G���
=��Q���(�C��R@�G��
=�   ���C�P�                                    BxtPu  �          @�  @�33���
���R��z�C�Ǯ@�33��(���R��\)C�O\                                    BxtP��  
�          @��@�녿�=q��ff����C�\@�녿J=q�(��ޏ\C��                                    BxtP�T  
�          @���@�Q��(������\C�+�@�Q�\(��'��홚C��)                                    BxtP��  �          @�Q�@�  � �׿�=q���C���@�  �n{�#33��Q�C�+�                                    BxtP��  �          @�G�@�z��\)����z�C�|)@�zΉ��/\)���C��q                                    BxtP�F  
�          @���@����\)��{��\)C��3@��׿��R�p��݅C�P�                                    BxtP��  �          @���@��\��\��33��p�C��q@��\��ff����ظRC��=                                    BxtPے  
�          @���@��\��
��\)��
=C��@��\��=q�����33C�Z�                                    BxtP�8  �          @�G�@��Ϳ�p����R����C�}q@��Ϳ�����R�ȸRC��=                                    BxtP��  "          @�\)@��\��\)�<���	��C���@��\>\)�P  ��@                                       BxtQ�  �          @�Q�@��
�\�{��ffC�c�@��
�aG��8�����C��\                                    BxtQ*  	�          @�G�@����(��   ���C��\@������G��ffC���                                    BxtQ$�  
�          @���@����\)��  ��C��H@��׿�z��,����\)C���                                    BxtQ3v  T          @���@��\�7
=���\�2�RC�E@��\�z��{�Ə\C�33                                    BxtQB  
Z          @��\@�  �G��&ff��\C���@�  ��R� ����C���                                    BxtQP�  
�          @�
=@E���G�>��R@_\)C��=@E��n�R�������C�                                    BxtQ_h  "          @�=q@333��?���A6�RC�l�@333��p���z��?�
C�u�                                    BxtQn  T          @�G�@`�����R?��@�\)C���@`����  ��
=�pz�C���                                    BxtQ|�  "          @�ff@�z��9���
=��33C�G�@�z���
������C��                                    BxtQ�Z  T          @���@����0  �&ff��Q�C�XR@����	����=q���\C�C�                                    BxtQ�   �          @�\)@s33�p��<#�
=uC�� @s33�W
=�ٙ�����C�B�                                    BxtQ��  T          @��@���J=q�8Q���33C�7
@���   �z���z�C�`                                     BxtQ�L  1          @���@����W���\���\C��@����1G���p����RC��R                                    BxtQ��  
          @���@��R�C�
�
=q����C���@��R�\)��{���HC��R                                    BxtQԘ  �          @��@����<��?��A1p�C�
@����Fff������C�k�                                    BxtQ�>  "          @��H@r�\�n�R>�33@p  C��R@r�\�_\)��\)�n�RC���                                    BxtQ��  
�          @�=q@XQ��\)�h����C�"�@XQ��I���'
=��C��                                     BxtR �  �          @���@����A녾��H��=qC�^�@�����R��ff��C���                                    BxtR0  
�          @��@����1G��Tz��Q�C���@����ff���R���HC��3                                    BxtR�  
�          @�@7
=��=q?}p�A,��C��q@7
=��������8��C��=                                    BxtR,|  �          @���@Dz����R?s33A�C�J=@Dz������33�@��C�n                                    BxtR;"  �          @���@hQ���33?&ff@�G�C��f@hQ��}p���  �Q�C�G�                                    BxtRI�  T          @���@w
=�~{>�z�@@��C��@w
=�k���G��}��C�/\                                    BxtRXn  "          @��@����l(�>�=q@/\)C�33@����Z�H��z��j�RC�Ff                                    BxtRg  "          @��@�\)�J=q�����
C�k�@�\)�1G��\��z�C�"�                                    BxtRu�  
�          @�Q�@��
�>{�Ǯ���C���@��
��R����C�޸                                    BxtR�`  "          @��@�z��C33��G���{C�\)@�z��+������rffC���                                    BxtR�  
�          @��@�=q��ff��Q��z�\C�� @�=q�xQ����ffC�H�                                    BxtR��  w          @�33@���p���=q�F{C���@����  �������C���                                    BxtR�R  
�          @���@�
=�������Q�C�1�@�
=?+������A ��                                    BxtR��  
�          @���@��\�
=��  �U��C�1�@��\�����
��ffC�>�                                    BxtR͞  �          @�33@��H�J�H=L��?   C��=@��H�7���{�a��C��q                                    BxtR�D  "          @��\@��Tz����˅C���@��:=q��������C�T{                                    BxtR��  T          @��@y���~{>�{@`  C�C�@y���mp���Q��o�
C�8R                                    BxtR��  �          @�ff@`  ��(�?s33A�RC�P�@`  ��33�����-p�C�ff                                    BxtS6  
�          @��R@c33��33?��\A$��C���@c33��33�}p�� ��C���                                    BxtS�  T          @��@�z��x��?O\)A=qC�aH@�z��vff�z�H���C��                                    BxtS%�  
�          @�Q�@�33�n�R?�@�C��
@�33�fff��\)�3�C�5�                                    BxtS4(  "          @���@�G��333�u�\)C�|)@�G��\)���\�Lz�C���                                    BxtSB�  �          @���@������  �!�C��{@���G���G��K
=C�t{                                    BxtSQt  �          @�33@���������C��H@���녿�33�]�C��q                                    BxtS`  "          @�@�33��Q������RC�s3@�33��
=�z�H���C���                                    BxtSn�  �          @��
@����
=����#33C�,�@���s33�@  ���HC�K�                                    BxtS}f  
�          @�G�@�{�����33�]p�C��f@�{���H����/33C�|)                                    BxtS�  
�          @��@����Q�>�p�@i��C��@�����
�333���C�<)                                    BxtS��  �          @��
@�  �����\)��{C�8R@�  �\�����T��C�Ff                                    BxtS�X  T          @�  @�=q���
=��
=C�G�@�=q�ٙ����
�O33C�"�                                    BxtS��  "          @��@������W
=�	��C��=@��ÿ�\)����733C�=q                                    BxtSƤ  T          @�@�=q�(�>\@u�C�+�@�=q�
=�5��  C��                                    BxtS�J  
�          @���@�G��(Q�?=p�@�C�9�@�G��,(�����z�C���                                    BxtS��  
�          @��\@��#�
?\)@���C��)@��#33�����z�C��f                                    BxtS�  T          @���@�����p�?@  @�z�C�W
@�����녽u��RC��)                                    BxtT<  �          @�
=@�����R?@  @�z�C��\@����=#�
>ǮC���                                    BxtT�  T          @�{@��ÿ�ff>\@i��C���@��ÿ���aG��
=C�xR                                    BxtT�  T          @�
=@��ÿ^�R?�=qA(  C��q@��ÿ�  ?(�@�(�C��=                                    BxtT-.  �          @��H@�
=�
=?��HAh(�C��q@�
=�!�>�
=@�33C��                                    BxtT;�  "          @�=q@�녿�  ?#�
@˅C�y�@�녿У׽u�\)C�ٚ                                    BxtTJz  "          @��\@�33��
?aG�A(�C�t{@�33�\)���
�J=qC��H                                    BxtTY   T          @�z�@�  ���?�G�A�\C��H@�  �)������p�C��H                                    BxtTg�  �          @�33@������?Q�A��C��@���� �׾�=q�(��C�AH                                    BxtTvl  �          @�(�@��R�	��>���@K�C�C�@��R����R��Q�C��                                    BxtT�  �          @���@�ff�
=q?�@�(�C�(�@�ff����G���G�C�3                                    BxtT��  
�          @�ff@�=q�33>k�@{C��@�=q���H�(���˅C�U�                                    BxtT�^  �          @�
=@�=q�z�#�
�uC���@�=q��\)�aG��Q�C��3                                    BxtT�  T          @��@�33�>�33@VffC���@�33��\�\)��z�C���                                    BxtT��  �          @�ff@����ff>��
@Dz�C���@�����\�����Q�C���                                    BxtT�P  �          @��R@�{��G�=#�
>ǮC�j=@�{��{�333�أ�C�R                                    BxtT��  �          @��R@��H�녽u��RC��@��H���h�����C��                                    BxtT�  
�          @�{@��5>#�
?�G�C���@��(�ÿ�ff�#\)C��                                     BxtT�B  
�          @��@�Q��P��>��@�
=C�y�@�Q��I���p���Q�C��                                    BxtU�  �          @��H@��H�7
=>#�
?�G�C��@��H�*=q��ff��C��H                                    BxtU�  "          @�z�@�Q��Fff�#�
��G�C��)@�Q��4zῧ��Ep�C���                                    BxtU&4  "          @��H@����A�>u@\)C��@����6ff����=qC�˅                                    BxtU4�  �          @���@�\)�@��>k�@(�C�@�\)�5�����\)C�                                    BxtUC�            @���@�z��j�H?xQ�A��C���@�z��o\)�#�
��p�C��                                    BxtUR&  T          @��R@�
=�L��?(�@�(�C���@�
=�J�H�E���\)C��                                     BxtU`�  "          @�@���aG�?�=qA(��C�O\@���i������(�C��                                    BxtUor  �          @���@�Q��w�?��\AF�HC��q@�Q�������ff���HC�8R                                    BxtU~  T          @�z�@��mp�?�33A3�C��@��vff���H��Q�C�z�                                    BxtU��  �          @��R@�z��z=q?\(�A��C�*=@�z��z=q�Tz�� Q�C�#�                                    BxtU�d  "          @�  @�  �vff?333@ָRC���@�  �r�\�s33�ffC���                                    BxtU�
  �          @���@�{��{?uA  C��H@�{���R�W
=� ��C�j=                                    BxtU��  
�          @�G�@�G��fff?.{@�C���@�G��c�
�\(��\)C��3                                    BxtU�V  E          @��@���g�=u?z�C�L�@���U���
=�]�C�t{                                    BxtU��  �          @���@����u��8Q��Q�C�H�@����Z�H��p���G�C�ٚ                                    BxtU�  T          @��@��\�u<#�
=�C��@��\�`�׿Ǯ�qp�C�H�                                    BxtU�H  "          @���@���y��>k�@
�HC��q@���j=q�����S�C�Ǯ                                    BxtV�  �          @�  @�(��r�\>�=q@'
=C�S3@�(��e���ff�G�
C�&f                                    BxtV�  
�          @���@�G��hQ�>aG�@Q�C�ff@�G��Z=q���\�C�C�C�                                    BxtV:  �          @�Q�@�p��Z=q>�@��C��@�p��S33�xQ��(�C�3                                    BxtV-�  �          @���@��R�Y��>�(�@��HC��f@��R�R�\�}p��ffC�=q                                    BxtV<�  �          @�G�@����R�\>�?��\C�xR@����C�
���H�8��C�e                                    BxtVK,  
�          @���@�\)�QG�?E�@�33C�]q@�\)�S33��R����C�<)                                    BxtVY�  "          @�G�@��R�Z=q?z�@���C���@��R�Vff�W
=� (�C���                                    BxtVhx  
�          @�=q@�G��U?��@�  C�9�@�G��Q녿W
=��C�u�                                    BxtVw  v          @�33@���L(�>��@u�C�%@���E��h���	��C��3                                    BxtV��  �          @�=q@����33?uA��C�aH@����z�E����C�=q                                    BxtV�j  
�          @��
@����,�Ϳ8Q���(�C��@����(��޸R��=qC�L�                                    BxtV�  	�          @�33@��\�Q쿎{�.ffC�  @��\��G���{���RC��                                    BxtV��  2          @��@�=q�-p��5��\C��\@�=q��Ϳ޸R����C�E                                    BxtV�\  
�          @��@����l(�>�\)@6ffC��@����`  ���H�Ep�C�p�                                    BxtV�  �          @��@.{����@G�A���C�8R@.{���;����z�C�XR                                    BxtVݨ  �          @�=q@(Q����
?��A�z�C���@(Q���ff���R�=p�C��                                    BxtV�N  "          @�ff@^{����?�G�A!p�C��R@^{��녿^�R�	��C��                                     BxtV��  
�          @�(�@������?��@��C�Y�@����������+33C��3                                    BxtW	�  �          @��@tz���G�?:�H@�RC��R@tz���\)���
�$  C��                                    BxtW@  �          @�{@@����\)?���A:�\C��3@@�������\(����C�ff                                    BxtW&�  �          @��R@Fff��z�?��AXz�C�+�@Fff��G��&ff��  C��f                                    BxtW5�  T          @�\)@<(���
=?�
=A^{C�H�@<(���(��&ff��\)C���                                    BxtWD2  
�          @�  @<(���\)?�G�Ai�C�E@<(���p�����\)C���                                    BxtWR�  "          @���@,(�����?ǮAq�C��=@,(���33�z���G�C�XR                                    BxtWa~  T          @��@p����(�>�  @\)C�W
@p�����
��p��n{C�.                                    BxtWp$  �          @��H@�ff�l(����Ϳp��C�,�@�ff�Vff���
�u��C�|)                                    BxtW~�  T          @��@�p����
?.{@�(�C���@�p���녿z�H���C���                                    BxtW�p  T          @���@w����?���A#\)C�^�@w������E���=qC�.                                    BxtW�  2          @�Q�@g���z�?�(�A;�C��{@g���  �.{��Q�C��H                                    BxtW��  
�          @�ff@n�R���?��A1�C��)@n�R���\�0����p�C��3                                    BxtW�b  D          @�  @k���=q?�z�A3�C�k�@k�����333�ָRC�"�                                    BxtW�  
(          @�  @b�\��{?���A)�C��=@b�\��  �L����\)C�^�                                    BxtW֮  �          @���@g���Q�?:�H@�  C�� @g���p���z��1�C��H                                    BxtW�T  
          @���@Z=q���?��A-C��=@Z=q��p��Q����\C�^�                                    BxtW��  	�          @��H@[���p�?��\A�\C�p�@[����s33�(�C�ff                                    BxtX�  
�          @���@c�
���\?B�\@�Q�C�5�@c�
��  ��33�/�C�o\                                    BxtXF  v          @��@n�R��p�?c�
A\)C�H�@n�R���ͿxQ��(�C�U�                                    BxtX�  �          @�G�@l(���  ?   @���C��@l(����\��{�O�
C�j=                                    BxtX.�  �          @�G�@g
=����?z�@��C���@g
=��zῥ��E�C���                                    BxtX=8  �          @���@6ff��?�{A*{C�ff@6ff���R�s33��C�T{                                    BxtXK�  �          @��@5�����?���A]��C�g�@5���녿�R��p�C��                                    BxtXZ�  T          @���@��
����?J=q@�G�C�� @��
���ÿh���33C��{                                    BxtXi*  
�          @���@u���
>8Q�?�G�C�ٚ@u���H��=q�t��C���                                    BxtXw�  T          @Å@w
=����?333@�=qC�ٚ@w
=��=q��\)�)C��                                    BxtX�v  �          @�(�@Dz�����?�ffAl  C���@Dz���������p�C�*=                                    BxtX�  
�          @���@$z����?���A��C��@$z����H�8Q��C�U�                                    BxtX��  �          @��
@\)���R@ ��A�G�C�Ǯ@\)���H��G���G�C��                                    BxtX�h  �          @��H@"�\���\@��A���C�J=@"�\����>�?���C�T{                                    BxtX�  
�          @�(�@p���\)@�A�G�C�y�@p���ff>��?�z�C��)                                    BxtXϴ             @��@z����@��A���C��@z���>�z�@-p�C��)                                    BxtX�Z  D          @��?�(���p�@$z�A�=qC�/\?�(���Q�>�ff@�Q�C�N                                    BxtX�   �          @�(�@�\��
=@0  Aՙ�C�]q@�\��z�?0��@�  C��                                    BxtX��  �          @�(�@Q����
@6ffA�(�C��@Q����\?Tz�@���C���                                    BxtY
L  "          @�z�@����@?\)A��
C��3@���p�?s33A=qC�E                                    BxtY�  T          @�?�����@Q�B p�C��3?����33?�Q�A1G�C�^�                                    BxtY'�  
(          @���?����z�@UB��C�AH?������?��
A@��C��                                    BxtY6>  T          @�p�?��H��Q�@g�B=qC��f?��H��Q�?���Aq�C��                                    BxtYD�  	�          @�@�\��z�@b�\B��C��H@�\���
?���Ak\)C��)                                    BxtYS�  
          @�p�@!���\)@Z=qB��C���@!���p�?�G�AbffC��f                                    BxtYb0  �          @�\)@���\)@c33Bp�C�<)@���\)?�33Av{C��
                                    BxtYp�  "          @�ff@z���(�@l��B��C���@z���?���A�33C��{                                    BxtY|  �          @�  @/\)�\)@|��B��C�9�@/\)���@p�A���C��
                                    BxtY�"  
(          @Ǯ@#33�s�
@�ffB)�
C��@#33���\@ ��A�=qC�U�                                    BxtY��  
�          @�
=@�
���\@��RB*��C�^�@�
���H@�HA���C���                                    BxtY�n  �          @�Q�@Q��z=q@�Q�B+�RC���@Q���{@!�A�G�C�S3                                    BxtY�  �          @ə�@)���qG�@�G�B+�HC��@)�����\@'�AĸRC���                                    BxtYȺ  T          @���@�  �y��?�Q�A@Q�C��)@�  ��=q���R�H��C�C�                                    BxtY�`  
Z          @���@�=q�<(���(���p�C���@�=q�G��9����33C�{                                    BxtY�  �          @��R@����-p��Y���33C��{@����{��\��G�C�R                                    BxtY��  �          @�
=@�33�G
=��
����C��@�33�
=q�C33���
C�n                                    BxtZR  �          @�
=@����fff��Q�����C��@����)���G
=��G�C�%                                    BxtZ�  
�          @���@��w
=���R�?33C�}q@��I���#33��33C�P�                                    BxtZ �  �          @�  @�=q��G��Q����C�q�@�=q�^{�{���HC��
                                    BxtZ/D  �          @���@�=q�\(��fff�
=qC�>�@�=q�8Q�����Q�C���                                    BxtZ=�  "          @���@����a녿�\)�{�C�)@����-p��1��ݙ�C��q                                    BxtZL�  �          @�  @�(��C33�������C�O\@�(����E���\C��                                    BxtZ[6  �          @�G�@��\�(Q���R�\C��
@��\�Ǯ�P����HC��                                    BxtZi�  �          @�G�@�������6ff��  C�� @��׿���\���  C��                                     BxtZx�  2          @��H@�=q�c�
�xQ���HC��@�=q�>�R�
�H���C�!H                                    BxtZ�(  v          @���@��R�S33�Y���C�.@��R�1녿�(����RC�Z�                                    BxtZ��  T          @�=q@��
�s33�z�H�Q�C�E@��
�Mp������\)C��)                                    BxtZ�t  �          @��H@��R�l(��u��C���@��R�G
=������RC�N                                    BxtZ�  �          @�z�@������\=u?��C�g�@�����=q��p��`(�C�J=                                    BxtZ��  "          @ƸR@�����=q?(�@�p�C�l�@�����Q�u�Q�C���                                    BxtZ�f  T          @�\)@B�\���H@K�A��C�xR@B�\��{?�Q�AW�C�+�                                    BxtZ�  �          @�{@a���\)��Q�L��C���@a���(���=q���
C��=                                    BxtZ��  �          @�
=@�(���논���=qC�{@�(���Q�У��u�C�                                    BxtZ�X  �          @�=q@�=q����.{�ǮC�,�@�=q���׿�Q��xQ�C�Q�                                    Bxt[
�            @��@�z���  �\�Z�HC�@�z��vff�����
C�*=                                    Bxt[�  �          @ə�@�G���  �G�����C��@�G��^{�Q���z�C�f                                    Bxt[(J  T          @ȣ�@�(����
�h���=qC�*=@�(��b�\��\���
C�H�                                    Bxt[6�  T          @�ff@������\�fff��C�j=@����o\)�ff���C�|)                                    Bxt[E�  �          @�p�@��R���ÿ��
�@z�C���@��R�U��&ff��=qC���                                    Bxt[T<  	�          @�ff@����{���  �`Q�C���@����J�H�1G���z�C���                                    Bxt[b�  �          @�{@�
=����:�H�أ�C��q@�
=�}p��\)��p�C�k�                                    Bxt[q�  �          @�{@�=q�����G���33C�+�@�=q�\)��Q���=qC���                                    Bxt[�.  �          @Ǯ@�p��������q�C�@�p���{��p���  C�p�                                    Bxt[��  T          @ƸR@�33���;����.�RC���@�33��Q�������C��3                                    Bxt[�z  
�          @�ff@�����
��
=�xQ�C�u�@���}p���33����C�޸                                    Bxt[�   
�          @���@�\)��{��{�H��C�w
@�\)�tz��  ����C�Ǯ                                    Bxt[��  
�          @�
=@����{�>W
=?���C�H�@����qG���33�+
=C��                                    Bxt[�l  
�          @ƸR@��
�p��?�@�G�C�%@��
�n�R�@  �߮C�C�                                    Bxt[�  �          @�{@��H���
>k�@C�
=@��H�|�Ϳ����2�\C��                                    Bxt[�  "          @���@��
����>��@��
C�y�@��
�|�Ϳn{�\)C���                                    Bxt[�^  T          @�ff@�ff��Q�?   @�(�C�@�ff�|(��c�
���C�H                                    Bxt\  T          @�\)@�
=��Q�?��@�  C���@�
=�~�R�L����z�C���                                    Bxt\�  "          @���@��
�|(�>�@�z�C��@��
�w��aG���C��                                    Bxt\!P  
(          @ȣ�@����\)>B�\?޸RC��R@��������  �8��C�\)                                    Bxt\/�  �          @�
=@�
=��=q�\)���C���@�
=���׿˅�m�C�H                                    Bxt\>�  
(          @Ǯ@�
=�fff?333@У�C��@�
=�hQ�����z�C��\                                    Bxt\MB  �          @�{@�p��QG�?n{A
=qC��R@�p��X�þaG��z�C�^�                                    Bxt\[�  �          @�33@��\�Tz�>�G�@�p�C�j=@��\�Q녿333���HC��
                                    Bxt\j�  "          @�p�@���?\)?\)@�  C�O\@���@�׾����{C�>�                                    Bxt\y4  
�          @�@���1�?�{AR�\C��q@���Dz�>�33@XQ�C�~�                                    Bxt\��  "          @��H@���L(�������C���@����\�E�����C��f                                    Bxt\��  T          @�(�@r�\���\>�
=@��\C��\@r�\���R�����.=qC�0�                                    Bxt\�&  2          @�z�@.{���@,(�A�C���@.{��?p��Az�C�G�                                    Bxt\��  
�          @���@:=q���@{A���C���@:=q���?:�H@��HC�AH                                    Bxt\�r  �          @���@1���p�@=qA��RC��R@1���{?#�
@�  C��                                    Bxt\�  "          @�{@.�R��G�@333A��C��)@.�R��{?��AG�C�T{                                    Bxt\߾  "          @ƸR@E���(�@��A��RC�)@E���33?�@���C���                                    Bxt\�d  �          @��@C�
���?��A�33C��q@C�
���\>\)?�ffC��                                    Bxt\�
  �          @Å@c33����?O\)@�33C��R@c33��z�Y�����C��q                                    Bxt]�  
�          @\@{����\?8Q�@��C�W
@{���녿Q�����C�e                                    Bxt]V  "          @�33@������>L��?���C��)@�����(�����F�\C�U�                                    Bxt](�  d          @��@~{���R�L�;�G�C�\@~{����=q�n�\C��=                                    Bxt]7�            @���@vff����>.{?�\)C�u�@vff��=q����Q�C�3                                    Bxt]FH  
(          @�=q@i������?��@��RC��{@i����{����ffC���                                    Bxt]T�  �          @\@������R?=p�@ᙚC�f@������R�@  ��33C��                                    Bxt]c�  
�          @��H@����{>��@�p�C�h�@����33��  �\)C���                                    Bxt]r:  �          @�G�@��
��z�>�
=@�  C��q@��
��G����\�  C���                                    Bxt]��  2          @��@��R��33>�?��HC��@��R���Ϳ���D  C��3                                    Bxt]��  �          @�G�@����~{>��@|(�C�^�@����x�ÿc�
��C���                                    Bxt]�,  �          @���@����`  >��R@?\)C�<)@����Z=q�Tz����C���                                    Bxt]��  �          @��@����Q�?�@�G�C�z�@����Q녿����  C�~�                                    Bxt]�x  
�          @���@����l(�=��
?8Q�C�,�@����aG���{�)C��{                                    Bxt]�  
�          @\@�=q�i��?�@�z�C�l�@�=q�g��.{��C��f                                    Bxt]��  �          @��
@�z��j�H=�G�?���C���@�z��`�׿��� z�C�*=                                    Bxt]�j  
�          @\@�Q��q�>�?�  C���@�Q��hQ쿊=q�$  C�T{                                    Bxt]�  
�          @��
@�  ������
�:�HC�"�@�  ������H�^=qC��)                                    Bxt^�  �          @\@�=q��ff�0����  C���@�=q�qG����R��
=C�t{                                    Bxt^\  �          @���@�Q��w��Tz���{C���@�Q��Z=q�G�����C�t{                                    Bxt^"  �          @���@����|�;����  C�k�@����fff�ٙ���33C��)                                    Bxt^0�  �          @���@���|�;����7
=C���@���i������k�C���                                    Bxt^?N  �          @��@s�
���
�h���	�C��H@s�
��33�z���(�C�t{                                    Bxt^M�  �          @\@~{��������Q�C��@~{��p������
=C��                                    Bxt^\�  �          @���@o\)��ff�   ��G�C�>�@o\)��녿��H���C�|)                                    Bxt^k@  �          @�=q@xQ�����.{�У�C��@xQ����
��\)�x��C��                                    Bxt^y�  
�          @�=q@�����R��  �ffC�XR@������У��yC�Y�                                    Bxt^��  T          @��@w
=���ͽ�Q�O\)C��)@w
=��z���
�k\)C���                                    Bxt^�2  
�          @�
=@tz���녾#�
��G�C��
@tz����ÿ����u�C���                                    Bxt^��  �          @�Q�@�Q���{�����uC��@�Q����H��G�����C�0�                                    Bxt^�~  �          @���@}p���Q쾏\)�*=qC���@}p���ff�����HC��f                                    Bxt^�$  �          @�Q�@�G����Ϳ&ff��C���@�G��p  ��33��C�q�                                    Bxt^��  �          @��@�(���  �J=q��\C���@�(��c�
���R��p�C�p�                                    Bxt^�p  "          @��@�����p������
=C��R@����r�\����z�C�/\                                    Bxt^�  T          @�\)@������
�����RC�"�@����o\)����
=C��H                                    Bxt^��  �          @��R@�{�w��c�
�	�C�z�@�{�Y����\���C�C�                                    Bxt_b  3          @�ff@�Q���=q�Tz�� z�C�,�@�Q��g
=��\���\C��)                                    Bxt_  
�          @�p�@�\)���׿k���\C�9�@�\)�c33�
=��
=C��                                    Bxt_)�  �          @�z�@xQ���(��#�
�ǮC�@xQ��~�R�������HC�&f                                    Bxt_8T  
�          @��H@e���ff��z��7�
C�l�@e��xQ��p���=qC�\)                                    Bxt_F�  
(          @�=q@[����ÿ��H�@z�C���@[��|(��"�\���C��H                                    Bxt_U�  
(          @��@C�
���׿�p��i�C�Y�@C�
���H�7
=��  C�ff                                    Bxt_dF            @��@p���"�\�5����C��)@p�׿Ǯ�^{�!C��                                     Bxt_r�  �          @�@�Q�0���i��� �HC�b�@�Q�>�ff�l(��"�H@���                                    Bxt_��  �          @�
=@�  �z��Z�H�
=C�XR@�  >��H�[��@�p�                                    Bxt_�8  �          @��@��Ϳ=p��@����(�C��
@���>W
=�E� �H@=q                                    Bxt_��  
�          @��R@���{�A�����C��{@����Q��h�����C��                                    Bxt_��  �          @���@����(��<(����
C�'�@��Ϳ�Q��b�\�  C�c�                                    Bxt_�*  
Z          @��H@���!G��,(����C�(�@�녿˅�U��33C��3                                    Bxt_��  
�          @��@�����(���p��:�RC��@����dz��=q��33C�
                                    Bxt_�v  �          @��@�p��fff���
�pQ�C�n@�p��>{�"�\��C��                                    Bxt_�  �          @��@�Q��W
=�����P  C�aH@�Q��G
=�����Qp�C�l�                                    Bxt_��  "          @��@�ff�l�;������C�#�@�ff�^�R���R�F�RC��R                                    Bxt`h  
�          @��@��\�|(�<�>��C��q@��\�qG���33�;33C��H                                    Bxt`  
(          @�Q�@X�����
?ǮA{�C��H@X����z�>B�\?�{C��                                    Bxt`"�  �          @���@[���33?˅A�  C�)@[���(�>aG�@�RC�B�                                    Bxt`1Z  �          @�\)@{�����?��A'�C�'�@{����;L���   C���                                    Bxt`@   �          @�33@���xQ�?J=q@���C��R@���|�;Ǯ�w
=C��                                     Bxt`N�  T          @��\@����n�R?��AMp�C�s3@����|(�>�?���C��f                                    Bxt`]L  3          @�(�@|���u@   A��C�� @|����Q�?G�@�(�C�e                                    Bxt`k�            @��H@`����(�?�p�Ak�
C�Y�@`�����
=�?���C��)                                    Bxt`z�  �          @���@G
=��������\C��=@G
=���H�У��|��C�Y�                                    Bxt`�>  
�          @�ff@n�R��(�?c�
A�
C���@n�R��p��z����C��
                                    Bxt`��  �          @�\)@s�
����>�ff@��
C��=@s�
��=q��G��G�C�'�                                    Bxt`��  �          @�G�@b�\��{>�?���C�(�@b�\��  ����L��C���                                    Bxt`�0  "          @�\)@`  ���H?Tz�@��
C�C�@`  ����0������C�7
                                    Bxt`��  
�          @�
=@`�����\?5@�G�C�Q�@`����=q�L����p�C�\)                                    Bxt`�|  "          @�  @^�R��z�?@  @�p�C��@^�R��z�E���C��                                    Bxt`�"  
9          @�  @]p���z�?Tz�@�z�C��3@]p�����333����C��                                    Bxt`��  u          @���@Vff���?Tz�@�(�C�L�@Vff��Q�8Q����HC�B�                                    Bxt`�n  "          @�G�@U��{?�Q�A/
=C�` @U��녾�Q��QG�C��                                    Bxta  
�          @���@HQ�����?�G�A9G�C�L�@HQ������
�<(�C���                                    Bxta�  
�          @�G�@A����
?�G�A8z�C��{@A���Q쾮{�HQ�C�g�                                    Bxta*`  "          @�
=@.�R���?�(�A\(�C�~�@.�R��녾���(�C�{                                    Bxta9  �          @�  @5���{?�z�A+33C��R@5�������ff���C��                                     BxtaG�  "          @�  @�R���H?���A���C�xR@�R��p�>�{@H��C�Ф                                    BxtaVR  T          @�Q�@/\)���?���AV�RC�o\@/\)���H�#�
���RC��                                    Bxtad�  �          @Ǯ@(Q���33?�@�
=C���@(Q����ÿ�����C���                                    Bxtas�  �          @�(�@{���H>�Q�@VffC��@{���R��  �<��C�+�                                    Bxta�D  "          @�z�@'���  ?�@���C���@'���p����� ��C���                                    Bxta��  �          @��@9�����#�
����C��@9����{��=q�n�\C���                                    Bxta��  "          @�z�@@  ��33>W
=?��RC���@@  ��{����EC�H                                    Bxta�6  
�          @��@Dz������B�\����C��@Dz����ÿ�Q���
C���                                    Bxta��  �          @�@AG����
��
=�z=qC��@AG����ÿ���
=C�}q                                    Bxta˂  
�          @�@N�R��Q쾣�
�<(�C�Ǯ@N�R��ff������\C���                                    Bxta�(  "          @�{@Mp�����<#�
=�Q�C���@Mp���=q��p��]��C�.                                    Bxta��  �          @���@?\)���>��@{C��R@?\)���R��  �<Q�C���                                    Bxta�t  T          @Å@9�����?L��@�=qC�O\@9����=q�8Q��أ�C�G�                                    Bxtb  T          @�z�@E���R?0��@�Q�C�XR@E��ff�L����Q�C�aH                                    Bxtb�  
�          @�@6ff��?�\@��C��{@6ff������
��\C���                                    Bxtb#f  "          @��@Z=q���H?�@�(�C���@Z=q���ÿk��	��C��                                    Bxtb2  �          @�z�@c33��
=>��@�(�C���@c33���Ϳp�����C��3                                    Bxtb@�  "          @�z�@]p���{?�=qA"�\C�~�@]p�������{�J=qC�5�                                    BxtbOX  
�          @���@U����\?!G�@�z�C���@U������O\)��(�C���                                    Bxtb]�  �          @�z�@=p�����=�G�?��C���@=p����
��{�O�
C�H                                    Bxtbl�  �          @�@G�����?.{@˅C�Ff@G���Q�O\)��Q�C�P�                                    Bxtb{J  
�          @�p�@6ff���?�G�A�C���@6ff����\���C��
                                    Bxtb��  
�          @ƸR@@  ����?�A.�\C���@@  ��p���{�G
=C�y�                                    Bxtb��  "          @�@.�R���
?���AIp�C�xR@.�R��G��B�\�޸RC�                                      Bxtb�<  �          @�z�@,(����?���AF=qC�T{@,(���Q�W
=��Q�C�H                                    Bxtb��  �          @���@Q����\?Y��A   C�l�@Q�����z���{C�S3                                    BxtbĈ  "          @�z�@z�H��ff>��@��C���@z�H���H����  C�AH                                    Bxtb�.  
�          @�33@G����?��
Aj�HC��@G���{����\)C��f                                    Bxtb��  �          @��H@�Q���{�u�ffC���@�Q��~{�����UC��                                    Bxtb�z  
�          @���@��i����G��ffC�@��O\)��(����\C��=                                    Bxtb�   "          @���@���w
=�(���ȣ�C��@���a녿�Q�����C�C�                                    Bxtc�  �          @\@�ff��33������RC���@�ff�r�\��33�{�C��=                                    Bxtcl  T          @�G�@�
=��Q�!G����C��@�
=�l(��ٙ���z�C�8R                                    Bxtc+  
�          @�G�@����j=q�xQ��Q�C�� @����P�׿����C�u�                                    Bxtc9�  �          @�G�@�=q�g
=��=q�r�\C�޸@�=q�C33�   ��p�C�+�                                    BxtcH^  �          @�=q@��\�x�ÿ@  ����C��R@��\�b�\���
��z�C�(�                                    BxtcW  �          @���@�����
=���?z�HC�Ǯ@�������=q�%�C�9�                                    Bxtce�  "          @��@����  ��z��/\)C��\@����Q쿵�Y��C��                                     BxtctP  �          @\@��R�^{�\�eC���@��R�O\)��ff�E�C�l�                                    Bxtc��  T          @�33@��\�G
=����%C�AH@��\�,�Ϳ�����C��)                                    Bxtc��  
�          @��@��H��z����Q�C�T{@��H��{��ff�B�RC��)                                    Bxtc�B  "          @���@�����z�    ���
C�7
@�����\)��
=�0��C��                                     Bxtc��  �          @�(�@�=q��33<�>��RC�j=@�=q��ff�����)�C��=                                    Bxtc��  
Z          @���@��R��
=>�  @�
C��f@��R���
�s33�
=C���                                    Bxtc�4  �          @���@�{���>��
@>�RC�(�@�{��p��Q���{C�c�                                    Bxtc��  
�          @�z�@��
����>�z�@0��C�|)@��
�|�ͿJ=q��C��R                                    Bxtc�  �          @�@�(����H������C�E@�(��y����p��8  C��                                    Bxtc�&  
�          @�p�@�Q�����  �z�C���@�Q��~{�����J�HC�S3                                    Bxtd�  �          @���@�ff��������C���@�ff��=q��(��6�\C�
                                    Bxtdr  �          @�(�@����G��L�Ϳ�=qC�
@����=q����Q��C�Ǯ                                    Bxtd$  T          @�@������?W
=@�33C��@�����p������p  C�`                                     Bxtd2�  �          @��
@�����Q�?O\)@���C���@�����녾����r�\C���                                    BxtdAd  
�          @��@�z���=q�#�
�\C��@�z������
=�5G�C�q�                                    BxtdP
  
�          @�@����w��B�\��p�C�U�@����a녿�  ��(�C���                                    Bxtd^�  
�          @���@����c�
��  �qp�C�!H@����B�\�����\)C�O\                                    BxtdmV  �          @��@�=q�_\)��\)�W33C�N@�=q�@���\)��Q�C�Q�                                    Bxtd{�  
�          @�Q�@�(��l�Ϳ^�R���C���@�(��Vff����Q�C��                                    Bxtd��  �          @��R@���w�������C���@���e�����s�
C��                                    Bxtd�H  �          @���@���xQ�#�
��(�C��\@���e��У��{�
C��3                                    Bxtd��  T          @Å@��\�c�
�����*�HC��3@��\�H���G����\C�~�                                    Bxtd��  
�          @�33@�Q��}p��\(���\C�Z�@�Q��fff��{��
=C��{                                    Bxtd�:  
�          @\@�����G����R�;�C�y�@�����녿�z��W�C�AH                                    Bxtd��  T          @��H@��H��\)��\����C�9�@��H��{��33�|(�C�*=                                    Bxtd�  �          @��@���z�L�;�C��f@���\)��Q��4��C�P�                                    Bxtd�,  �          @���@������
=��
?E�C��H@��������ff� ��C�.                                    Bxtd��  �          @���@�p���(�=#�
>�33C��@�p���������'
=C�E                                    Bxtex  T          @���@��
��z���Ϳ}p�C��
@��
��
=���R�=G�C�,�                                    Bxte  
�          @���@�����{��\)�+�C�*=@������׿��H�:ffC��
                                    Bxte+�  T          @���@u���>\)?�=qC���@u��G�����"ffC�q                                    Bxte:j  
�          @��@|(���33<��
>aG�C�Ff@|(���ff��z��0��C��q                                    BxteI  �          @�=q@x�����;�
=�\)C��3@x����(������t��C�Ǯ                                    BxteW�  �          @\@vff���Ϳ333�ӅC�Ф@vff��=q��\)��33C�޸                                    Bxtef\  �          @�=q@�����\)�z���\)C��@��������H���RC��                                    Bxteu  "          @�=q@��
���Ϳ(���ƸRC���@��
���H��G�����C���                                    Bxte��  �          @\@|(������\(���RC�p�@|(����   ��\)C���                                    Bxte�N  T          @���@s33���H�xQ��\)C��@s33��{�
=��G�C�
                                    Bxte��            @�  @~{���E���z�C��@~{���H������C��                                    Bxte��  C          @�\)@�����p���33�X��C�3@����j�H�=q���RC��                                     Bxte�@  
�          @���@��������M�C��R@��h�����G�C�xR                                    Bxte��  T          @���@���q녿�G����\C�Y�@���Mp��*=q���HC���                                    Bxteی  
�          @�
=@�z��z=q�.{�љ�C�(�@�z��g
=�������C�H�                                    Bxte�2  
Z          @��R@�����z�L�;�ffC��{@�����  ����)G�C�xR                                    Bxte��  
(          @�{@��H����>Ǯ@q�C��3@��H�\)�#�
��C���                                    Bxtf~  
�          @��@�=q���þ�����C��f@�=q���׿�ff�q�C���                                    Bxtf$  "          @��@l������\(���HC�7
@l����G��   ���C�]q                                    Bxtf$�  �          @���@�  ����>#�
?��C�ff@�  ���k��z�C���                                    Bxtf3p  
�          @�=q@�����\��=q�#�
C���@���xQ쿥��D  C���                                    BxtfB  �          @�=q@�G���Q�u��C�H�@�G��w�����!p�C�˅                                    BxtfP�  T          @�G�@���\)?(�@�Q�C�33@����Q����{�C�q                                    Bxtf_b  "          @���@�����\)>�ff@��C��R@������R�!G����RC���                                    Bxtfn  
(          @�  @��H���>\@k�C�K�@��H��=q�(����G�C�j=                                    Bxtf|�  
�          @��R@����G�>�=q@%�C��
@���~�R�@  ��  C��\                                    Bxtf�T  
�          @���@���녾L�Ϳ�\)C��@���(����\�B�RC���                                    Bxtf��  
�          @�=q@�����G���=q�%C��H@�����=q��33�U�C�p�                                    Bxtf��  
�          @��@�{����Ǯ�n{C��=@�{��(���(��`Q�C���                                    Bxtf�F  
�          @���@\)���þ�ff����C���@\)���׿Ǯ�o\)C���                                    Bxtf��  �          @���@|����녾����p�C�u�@|�������˅�s
=C�J=                                    BxtfԒ  �          @��@�  ���þǮ�k�C��R@�  ��G���  �e��C�}q                                    Bxtf�8  T          @���@vff����
=q���C��\@vff���\�����\C���                                    Bxtf��  
�          @�Q�@z�H���þ�
=�\)C�q�@z�H���ÿ��
�k�C�<)                                    Bxtg �  "          @��@�G����;�p��e�C�G�@�G���p������`z�C��                                    Bxtg*  
�          @���@�����p��!G���\)C�*=@�����(��ٙ���\)C�%                                    Bxtg�  T          @�  @U���
=��
=�~{C��@U���
=�У��|Q�C��)                                    Bxtg,v  "          @��@O\)���׾���|��C�j=@O\)���׿���}�C��                                    Bxtg;  �          @�Q�@N�R������Q��]p�C�J=@N�R��녿����v�\C���                                    BxtgI�  
Z          @���@Mp������#�
���
C�9�@Mp������\)��ffC��                                    BxtgXh  T          @\@H������
=�}p�C��q@H����p���
=��z�C�G�                                    Bxtgg  
�          @��H@:�H��=q�B�\����C�` @:�H������R�c�
C�޸                                    Bxtgu�  
(          @�(�@E��G�=�?�33C�#�@E�����
=�1��C�t{                                    Bxtg�Z  
�          @Å@P  ���;�����C�  @P  ��(���p����C��
                                    Bxtg�   T          @�p�@9����{�����4z�C��@9����ff�У��tQ�C���                                    Bxtg��  
l          @ƸR@333�����aG���\C�` @333��=q�����jffC���                                    Bxtg�L  "          @�p�@(����=q��G���ffC��3@(�����
��(��\��C��                                    Bxtg��            @���@%���=q=#�
>�Q�C�ff@%���p������G
=C��R                                    Bxtg͘  �          @�{@E���33��(�����C���@E����H��p�����C��
                                    Bxtg�>  4          @��@7
=��{��=q�!G�C��q@7
=���R�˅�o\)C�aH                                    Bxtg��  �          @�p�@5��ff��G����C���@5����G���ffC�`                                     Bxtg��  "          @�@@���������C��@@����=q�������C�U�                                    Bxth0  
�          @ƸR@Mp����ÿ���G�C���@Mp���\)�����
C�ff                                    Bxth�  �          @�ff@AG����;\�_\)C���@AG���z��
=�{33C�1�                                    Bxth%|  T          @Å@h����녿^�R�33C��3@h����ff� ����  C���                                    Bxth4"  
�          @�=q@e����׿����"�HC�s3@e����
������C���                                    BxthB�  �          @�(�@N{���
=���C��@N{��z��=q��
=C���                                    BxthQn  �          @�@N�R��
=�5�ҏ\C�޸@N�R��z������33C���                                    Bxth`  
�          @���@N�R���\�:�H��{C���@N�R��  �   ���\C�o\                                    Bxthn�  �          @��@I�����#�
��33C��@I�������
=���HC��\                                    Bxth}`  �          @�{@_\)��녿���h(�C��R@_\)�����)�����HC���                                    Bxth�  T          @�=q@AG�����(����\)C�g�@AG���p����H��
=C�%                                    Bxth��  
�          @��
@*=q��Q쾣�
�8Q�C�ff@*=q���׿ٙ��x  C��                                    Bxth�R  �          @�ff@���{���Ϳ\(�C��@���  �\�[�
C�q�                                    Bxth��  �          @ə�?�����33?�\)AL��C��=?�����Q콏\)�(��C�Y�                                    Bxthƞ  �          @ƸR?��
��z�?���AXQ�C�q?��
��=q�#�
���
C���                                    Bxth�D  �          @ƸR?xQ���  ?�p�A7
=C��R?xQ����
��  ��C��H                                    Bxth��            @ƸR?(���z�?��@��HC��R?(����H�p����C��q                                    Bxth�  B          @�
=?O\)�\?O\)@��C��\?O\)��33�+���{C���                                    Bxti6  �          @�\)?
=q��(�?c�
A\)C�� ?
=q��p��(���33C�}q                                    Bxti�  "          @�=����Å?^�RA=qC�}q=�����z�(����RC�|)                                    Bxti�  �          @�ff���
��(�?.{@�33C������
���
�L����C���                                    Bxti-(  T          @ƸR�������?+�@�\)C���������z�Q���C���                                    Bxti;�  T          @������33?Tz�@�Q�C�Ǯ������
�&ff����C���                                    BxtiJt  �          @�p��O\)�\?.{@��
C�8R�O\)��=q�J=q��\C�7
                                    BxtiY  �          @ƸR�����
=?h��A�C�s3�����Q�
=q��\)C��                                     Bxtig�  �          @Ǯ�p���\?k�A(�C��)�p�����
�\)���
C���                                    Bxtivf  �          @�{�}p���\)?�z�A-p�C�H��}p����H�����1G�C�]q                                    Bxti�  �          @�{�n{���?�
=A/�C���n{��33��z��)��C��f                                    Bxti��  T          @�ff��(���?�\)A(  C�,Ϳ�(����þ��
�@  C�E                                    Bxti�X  
�          @ȣ׿��
���H?+�@���C�\���
��=q�L����=qC��                                    Bxti��  T          @�(��E���G�?333@���C��f�E����ÿQ����
C��                                    Bxti��  
�          @������33?J=q@�G�C�H������33�=p���z�C�J=                                    Bxti�J  �          @��ÿ(���?h��A{C�.�(���
=�z���G�C�33                                    Bxti��  "          @ə������?�=qAC��=���Ǯ��
=�tz�C���                                    Bxti�  4          @ƸR��33�\?���A33C�Zᾳ33��������p  C�`                                     Bxti�<  �          @�=q������?�33AL��C�e����녽��Ϳk�C�h�                                    Bxtj�  "          @��
�#�
���?˅Ag
=C���#�
�˅=��
?:�HC�Ф                                    Bxtj�  �          @�p�������
?��A�z�C�þ����z�>\@X��C�+�                                    Bxtj&.  
�          @�  �B�\�˅?��A7
=C�&f�B�\��\)��  �\)C�*=                                    Bxtj4�  "          @Ϯ<#�
�ȣ�?�z�Al��C�
=<#�
��\)>�?�\)C��                                    BxtjCz  �          @�G��.{�˅?�G�AUG�C�8R�.{���ý#�
��33C�=q                                    BxtjR   "          @��ͽ�G�����?�A{�
C�}q��G�����>u@z�C���                                    Bxtj`�  T          @�Q�.{��G�?�Q�Ag�
C�AH�.{��  =���?Tz�C�G�                                    Bxtjol  T          @�
=�k����?��HAH��C��)�k���
=����{C��                                    Bxtj~  "          @�\)�W
=��33?�G�A,z�C�þW
=�ָR��{�7�C��                                    Bxtj��  �          @�
=���H�љ�?�
=ADQ�C��q���H��ff�#�
���C���                                    Bxtj�^  T          @�
=���
��G�?��AT  C��R���
�ָR�#�
��p�C��H                                    Bxtj�  T          @׮��
=��  ?�
=Ag\)C�'���
=�ָR=���?Y��C�7
                                    Bxtj��  �          @��@R�\�fff���\�h��C�5�@R�\�L��������C��                                    Bxtj�P  �          @��@�p������-p���=qC��{@�p���z��Dz���(�C�aH                                    Bxtj��  "          @��
@�{��\)�5���\)C�E@�{�(���E���G�C�0�                                    Bxtj�  �          @�  @���fff�5���  C���@���W
=�=p���Q�C��H                                    Bxtj�B  �          @�G�@����@��������C��{@�����H�1G����HC���                                    Bxtk�  �          @�@�������\)��  C���@��ÿ�p��>�R��G�C�f                                    Bxtk�  �          @���@��ÿ��#�
��33C��@��ÿ��\�<(���C��\                                    Bxtk4  �          @�{@`���j=q�#�
���C���@`���=p��Vff�Q�C���                                    Bxtk-�  �          @��R@qG��Z=q�#�
�ծC���@qG��.{�R�\�33C��                                    Bxtk<�  �          @�33@���C33�.�R��
=C��@���z��W��
=C�j=                                    BxtkK&  �          @�G�@�=q�/\)�5���{C��f@�=q�   �Y���	{C�1�                                    BxtkY�  T          @�Q�@�G��   �{��  C��)@�G������?\)����C��3                                    Bxtkhr  �          @��@�33���#33�ɮC��{@�33��G��C33����C��                                    Bxtkw  T          @�Q�@����� ����\)C�@�����H�@  ��p�C�N                                    Bxtk��  T          @�\)@�=q�(����R��
=C���@�=q��\�1����C���                                    Bxtk�d  "          @�=q@����!G��(����RC��@��׿�
=�.{�޸RC��)                                    Bxtk�
  "          @�Q�@���@�׿�z���C�\@����R�#�
�ԏ\C���                                    Bxtk��  4          @���@���[���ff�z=qC��@���>{��
��p�C��                                    Bxtk�V  �          @��@���I���������
C�/\@���(Q��"�\�ȏ\C�z�                                    Bxtk��  �          @���@��
�hQ쿪=q�K�
C��{@��
�Mp��������C���                                    Bxtkݢ  f          @���@�p��x�ÿ333��  C�� @�p��g��˅���RC���                                    Bxtk�H            @���@��R�{��\�o\)C�w
@��R�o\)����NffC�33                                    Bxtk��  �          @��@������H�.{��
=C�AH@����|(���\)�4z�C��                                    Bxtl	�  �          @�z�@vff����=�G�?�=qC���@vff�|�ͿW
=�33C�"�                                    Bxtl:  �          @�33@hQ���(��xQ��\)C��@hQ��r�\��z����
C��                                     Bxtl&�  �          @���@�����?\)@��
C�@���G������C��)                                    Bxtl5�  �          @�33���hQ�@^�RB.�C�<)����33@!�A��\C�ٚ                                    BxtlD,  "          @��
�(��qG�@q�B2�
C�b��(�����@2�\A�ffC�+�                                    BxtlR�  T          @�z�&ff����@]p�B �C�s3�&ff��\)@�HA֏\C�q                                    Bxtlax  T          @�p�����  @S33B�\C��׾����@p�A�G�C�0�                                    Bxtlp  T          @�p��
=�\)@e�B&ffC��׿
=���R@#33A�\)C�j=                                    Bxtl~�  �          @��
����j�H@x��B9�RC�������\)@:�HBC�`                                     Bxtl�j  �          @�z����fff@\)B?�RC��
�����{@B�\B	�C�\                                    Bxtl�  
�          @�=q>�33��p�@S33B  C�n>�33��=q@�RA��C��                                    Bxtl��  �          @��?!G����@!G�A��HC���?!G�����?�G�AJffC�B�                                    Bxtl�\  T          @��>�Q���(�@p�A���C��q>�Q����?�Q�AA�C���                                    Bxtl�  �          @�p�?�G���z�@��A��C�9�?�G���G�?�ffA/�C���                                    Bxtl֨  �          @�
=@ff���\@G�A�G�C�5�@ff��
=?�G�A��C�|)                                    Bxtl�N  �          @��
@���\)?�(�A8z�C��\@�������Ϳs33C�o\                                    Bxtl��  �          @��?��H���R?���A<��C��?��H���H����z�C�}q                                    Bxtm�  �          @���@�����?8Q�@�C�4{@����\���H���C�&f                                    Bxtm@  �          @�?��
����?Tz�A�C�W
?��
���\��(���z�C�B�                                    Bxtm�  �          @�33?���ff?L��A	p�C�?���  �Ǯ���C��                                    Bxtm.�  �          @�
=@�\��=q?p��A'
=C�J=@�\��p��#�
��p�C�\                                    Bxtm=2  �          @���@ �����?J=qAQ�C�Ff@ ��������
�Z�HC�!H                                    BxtmK�  �          @�\)@����?��AJffC�\)@���  <�>�\)C�
=                                    BxtmZ~  �          @��R?ٙ���Q�?�AP��C��?ٙ����<��
>uC���                                    Bxtmi$  4          @�z�@���=q>�  @/\)C�}q@���  �Tz����C���                                    Bxtmw�  �          @���@(�����>��?��C�z�@(����=q�c�
��
C��)                                    Bxtm�p  �          @��?������?��AG33C�<)?�����#�
��C���                                    Bxtm�  �          @���?�  ��
=?h��A&ffC��=?�  ��녾k��$z�C��f                                    Bxtm��  �          @�\)?�R��p���\)�333C�%?�R�����33�c\)C�>�                                    Bxtm�b  �          @��?h���������
=C��\?h����녿�{�m�C��                                    Bxtm�  �          @�Q�?�����{?\(�A1G�C��\?������þ�����C���                                    BxtmϮ  �          @�Q�?����xQ��
=��
=C��3?����Tz��1��(�C��                                    Bxtm�T  T          @o\)?�
=�[���z���{C�J=?�
=�QG������C�˅                                    Bxtm��  �          @e�?��<(������C���?��/\)��33��=qC�~�                                    Bxtm��  �          @�{@<(��333��33���C�33@<(����G����HC��=                                    Bxtn
F  �          @�@[��.{������C��@[��	���,���{C���                                    Bxtn�  �          @�  @Vff�33�z���{C��=@Vff�޸R�#33�=qC�AH                                    Bxtn'�  T          @���@Fff�!녿�����{C�^�@Fff�   �\)�G�C��3                                    Bxtn68  �          @�
=@S�
���H�333�C�P�@S�
��  �HQ��(z�C��\                                    BxtnD�  �          @��@dz��{�@���ffC�@ @dzΉ��W��(G�C�y�                                    BxtnS�  �          @�Q�@$z��)����
����C��@$z����7��$��C��
                                    Bxtnb*  �          @�
=@Q��r�\��z���
=C��@Q��N�R�0  �	C�/\                                    Bxtnp�  �          @�33@J�H�I���{���C��3@J�H��R�H����C��R                                    Bxtnv  �          @���@)���-p���
=��\)C�,�@)���\)�G��  C��=                                    Bxtn�  �          @|(�@2�\�33���
���C�9�@2�\�����\���C�
=                                    Bxtn��  �          @�\)@C33�Q��(��߮C��=@C33������H�p�C�%                                    Bxtn�h  �          @�@2�\�+��p���RC�3@2�\�G��A��%  C��                                    Bxtn�  �          @�Q�@b�\��Q��*=q�ffC�<)@b�\���\�>�R���C�H                                    Bxtnȴ  �          @��@QG��/\)��Q���G�C�f@QG�����!��C�                                    Bxtn�Z  �          @�(�@333�|��<��
>uC���@333�u�k��.�\C��                                    Bxtn�   �          @��\@%����?@  A��C��{@%���H��z��N�RC�j=                                    Bxtn��  �          @�Q�@���Q�?J=qAG�C��@����\�u�333C��f                                    BxtoL  �          @�Q�?��]p�@�A�C���?��w�?��RA�=qC�5�                                    Bxto�  �          @��R@
�H�C�
@�A�\)C���@
�H�aG�?��A��C�˅                                    Bxto �  �          @�{@!G��Z�H?�(�AǅC�1�@!G��r�\?���AZ�\C��\                                    Bxto/>  �          @��@�\�q�?�Q�A�\)C���@�\���\?=p�Az�C���                                    Bxto=�  �          @�33@{�l(�?z�HAG\)C��f@{�tz�=�\)?p��C�s3                                    BxtoL�  �          @�{@5�W�=���?�ffC��@5�S33�5�\)C�Z�                                    Bxto[0  �          @_\)?��H�!G�?
=qA�
C�{?��H�%����
��(�C��=                                    Bxtoi�  �          @�{@4z�u�n�R�S33C���@4z�?(���k��O�AQ�                                    Bxtox|  �          @�p�@:�H�8���p���33C��H@:�H�{�E��   C�\)                                    Bxto�"  �          @���@Y���.{�"�\��  C��f@Y����\�G���C��f                                    Bxto��  �          @�@^{�
=�!���z�C�� @^{��Q��AG���C��                                    Bxto�n  �          @��\@X���*�H���H�ÅC��f@X�����"�\��RC���                                    Bxto�  �          @��@Z�H��z��B�\� �C��f@Z�H�Ǯ�N�R�,�C��{                                    Bxto��  �          @��@J=q�=p��l(��B�\C�n@J=q>.{�p���G�@@��                                    Bxto�`  �          @��\@*�H�J=q�s�
�W33C���@*�H>���x���]@H��                                    Bxto�  �          @��\@Y���>�R�+���C�b�@Y���.�R��\)��\)C��)                                    Bxto��  T          @��@@�׿�����Z(�C���@@��>���=q�[
=Az�                                    Bxto�R  T          @�z�@{�����_\)�"�C��@{���=q�j=q�+�HC��                                    Bxtp
�  �          @�ff@w
=��{�l(��*p�C��R@w
=�B�\�vff�3�C���                                    Bxtp�  �          @��@��׿&ff�W��{C���@���>8Q��[��(�@��                                    Bxtp(D  �          @��H@S33?�p�������A�\)@S33@�Ϳٙ���B                                      Bxtp6�  �          @�33@i��?�33�0  �	A¸R@i��@���G��޸RA�z�                                    BxtpE�  �          @�{@dz�?���K��  A�ff@dz�@%�(�����RB��                                    BxtpT6  �          @�G�@
�H@�R�g��?
=BC(�@
�H@Q��:�H��Bb
=                                    Bxtpb�  �          @�(�@[�?��p  �<��A	�@[�?�z��`���-�RA���                                    Bxtpq�  �          @�@w�?���Mp��ffA|  @w�?�=q�6ff���A��H                                    Bxtp�(  �          @��\@���?�z��Dz���RA���@���@ff�%���\)A�R                                    Bxtp��  �          @��H@�Q�@p��{�݅A�@�Q�@>�R��ff��z�Bff                                    Bxtp�t  �          @���@}p�@+���p����Bp�@}p�@E����\�c33B�                                    Bxtp�  �          @�  @�33@(������}�A��
@�33@,(��8Q��B(�                                    Bxtp��  �          @�33@h��?��R��z���(�A�@h��@����\)���B�\                                    Bxtp�f  �          @�\)@fff?��������A�  @fff@�Ϳ�p����
B�                                    Bxtp�  �          @���@Dz�?���6ff�z�A���@Dz�@!G���
��p�B=q                                    Bxtp�  �          @�\)@U�@$z��`  ��
B�\@U�@U�1G����HB4\)                                    Bxtp�X  �          @�z�@Tz�@Fff�,(�����B,�@Tz�@j=q��{����B?=q                                    Bxtq�  �          @�G�>Ǯ����*�H�Q�C���>Ǯ�X���g
=�:p�C�N                                    Bxtq�  �          @�(�?��������yz�C���?��O\)��z�C��=                                    Bxtq!J  �          @�z�?Tz�(�����\�)C��3?Tz�>���33ffA���                                    Bxtq/�  �          @���?����33����=qC�AH?�����R����Q�C�<)                                    Bxtq>�  �          @�z�@ff���H�vff�TC�|)@ff�u���R�t��C��3                                    BxtqM<  �          @��@#33>������d��A"�\@#33?�
=�q��Q�A�                                    Bxtq[�  T          @XQ�?��
?�p��(��:\)B  ?��
@�� ���z�BCQ�                                    Bxtqj�  �          @��\@��?���<���0=qB
=@��@#�
����	��B;p�                                    Bxtqy.  �          @�G�@J�H@�
�B�\�  Bz�@J�H@?\)�Q�����B-G�                                    Bxtq��  �          @�
=@!G�@?\)�\���%��BGp�@!G�@o\)�'
=��ffB`G�                                    Bxtq�z  �          @�  @   @5��hQ��/=qBB(�@   @h���4z���B]�                                    Bxtq�   �          @�p�@33@H���Tz��!�
BV��@33@w
=�(���Q�Bl��                                    Bxtq��  �          @�=q@
=q@<(��E�!
=BV��@
=q@g
=�G����Blz�                                    Bxtq�l  �          @��R?���@<���{��RBi�?���@Z=q��
=��  Bw                                    Bxtq�  �          @~{?��!녾����G�C��{?��Q�aG�����C��                                     Bxtq߸  �          @[�=�G��2�\?�\)A���C��=�G��B�\?#�
A<��C��                                    Bxtq�^  �          @>�R?�\)��\)�   �BffC�,�?�\)��Q�s33��(�C��R                                    Bxtq�  �          @3�
?˅��\)��ff�1p�C���?˅>��
�\�-  A5G�                                    Bxtr�  "          @@��=��녾B�\��
=C�{=��   �������C�ٚ                                    BxtrP  �          @&ff��
=@	���B�\���HB�����
=@G��L������B�{                                    Bxtr(�  �          @7
=�5@���33���B���5@&ff�E���=qBҏ\                                    Bxtr7�  �          @O\)?^�R>��	���)Aޣ�?^�R?z�H��(��a\)BA                                    BxtrFB  �          @2�\?�����R���AC�?�����
��ff���HC���                                    BxtrT�  �          @�33?k��i��?5A)C��?k��mp��k��\(�C���                                    Bxtrc�  �          @�\)?���p���G��K
=C��=?��r�\�G���ffC��                                    Bxtrr4  �          @���@Mp�����e�/�
C���@Mp��n{�|���G  C��R                                    Bxtr��  �          @�33@u�H�ÿ�
=����C�\)@u�#�
�)������C�*=                                    Bxtr��  �          @�@�33���AG���RC���@�33���\�X����C�                                      Bxtr�&  �          @�{@����
=� ����ffC�*=@�����R�:�H���C�&f                                    Bxtr��  �          @�p�@��ÿ���H����C��@��þk��S33�\)C�o\                                    Bxtr�r  �          @��R@��R�aG��?\)�G�C�e@��R�����G
=��RC�XR                                    Bxtr�  �          @���@QG��.�R�`����C�3@QG���G����\�>��C��                                    Bxtrؾ  �          @��\@Vff�6ff�[����C���@Vff��������9p�C�C�                                    Bxtr�d  �          @�z�@N{�=p��c�
��RC��3@N{���H��{�@G�C�Q�                                    Bxtr�
  �          @��H@Tz��#�
�k��$�C�33@Tz��ff���R�C�C���                                    Bxts�  �          @�33@A��B�\�c33�=qC�p�@A���\��ff�C��C�                                      BxtsV  �          @��\@S�
�5�XQ���C���@S�
�����  �9G�C�'�                                    Bxts!�  �          @���@c33�0  �O\)�Q�C�%@c33��=q�u��/=qC�\)                                    Bxts0�  �          @�Q�@HQ��8���XQ����C��f@HQ��
=��  �=C�'�                                    Bxts?H  �          @�  @L���L(��C�
�=qC���@L����
�qG��.C��                                    BxtsM�  �          @�\)@Z=q�AG��AG��33C�7
@Z=q�	���l(��*  C�޸                                    Bxts\�  �          @��@1����R�����(�C���@1��aG��I���=qC�'�                                    Bxtsk:  �          @�z�@$z��w��.�R��33C��@$z��B�\�hQ��)33C��                                    Bxtsy�  �          @�{@�\���
�,����ffC���@�\�R�\�j=q�)C�j=                                    Bxts��  �          @�Q�@=q���H�����
C���@=q�e�Y�����C��                                    Bxts�,  �          @��@#�
�����
��{C���@#�
�`  �Tz��C��                                    Bxts��  �          @�@���{��(�����C���@����H�7�� ��C�0�                                    Bxts�x  
�          @��R?�=q������\�W�C�Ff?�=q����"�\���C�                                    Bxts�  �          @���?W
=��(�����_
=C��?W
=���
�$z���p�C�/\                                    Bxts��  �          @�ff?Tz���=q>aG�@�C�o\?Tz���ff��z��F�HC���                                    Bxts�j  �          @�  ?�G����H�k��\)C��=?�G����p�����C�O\                                    Bxts�  �          @�@*=q��������G�C��=@*=q�g
=�Dz��	�\C�/\                                    Bxts��  �          @�@G���  ���
��33C�c�@G��x���8Q���\C�'�                                    Bxtt\  �          @�@����Ϳ333��G�C�9�@�����������33C���                                    Bxtt  �          @�p�?�\)��  ��\)�?�
C���?�\)�����Q���Q�C��3                                    Bxtt)�  �          @�z�?�\)���ÿ:�H���C��=?�\)��p��G���  C��\                                    Bxtt8N  �          @�(�@Q������@  ��HC��@Q���{��p����C�˅                                    BxttF�  �          @�(�?��R��33������RC�,�?��R��{�E���
C�@                                     BxttU�  �          @��?O\)��G��>�R�	p�C�` ?O\)�W
=��  �D�
C�Ф                                    Bxttd@  �          @�(�?z�H�������p�C��?z�H�x���c33�&
=C��                                    Bxttr�  �          @�  @O\)��Q�z�H�-�C�xR@O\)�g
=�   ���C��{                                    Bxtt��  �          @�=q@�G��2�\��p��X��C�~�@�G�������p�C���                                    Bxtt�2  �          @��\@L���g��G����C��R@L���=p��9�����C��R                                    Bxtt��  �          @��R?�Q�����u�9�C���?�Q��|������ffC���                                    Bxtt�~  �          @�Q�?�=q��\)?8Q�AffC���?�=q��  ������
C���                                    Bxtt�$  �          @��@G��>�R�
=� �HC���@G��  �Dz��/G�C���                                    Bxtt��  �          @�z�@>�R�[�@(��A�\)C�|)@>�R��  ?У�A�33C�O\                                    Bxtt�p  �          @��\@W��s�
?�ffA�  C���@W����\>��H@�Q�C��f                                    Bxtt�  �          @��\@I���{���p���Q�C�]q@I���k���z���\)C�B�                                    Bxtt��  �          @���@����E����
�Z�HC�)@����8�ÿ�\)�D��C��q                                    Bxtub  �          @��@a��p  ��\)�mC��q@a��P  �z���=qC���                                    Bxtu  �          @�Q�?�=q���H�{��C�o\?�=q�e�S33�(�C���                                    Bxtu"�  T          @�?�\)�hQ������C��?�\)�8���J�H�*ffC�y�                                    Bxtu1T  �          @��>W
=�����  ���RC�H�>W
=���H�,���z�C�w
                                    Bxtu?�  �          @�G�?�z���{�����{C�@ ?�z��w��+���C��H                                    BxtuN�  �          @�=q@p��vff��z����C�J=@p��L���8Q����C��                                    Bxtu]F  �          @�33@C33�"�\�E��G�C��@C33��\)�i���:z�C���                                    Bxtuk�  �          @�@?\)�:�H�1G��33C���@?\)�z��]p��.G�C��\                                    Bxtuz�  T          @�G�?!G���(���33����C���?!G����H����C�Ff                                    Bxtu�8  �          @�  ?��R��G��%����C���?��R�[��h���1(�C���                                    Bxtu��  �          @���@{� ���qG��A��C��)@{�������j�C��\                                    Bxtu��  �          @�(�@�H�.{���w=qC��@�H?(���{�x�Ad(�                                    Bxtu�*  �          @�ff@33��
=��=q�b�C���@33��G����
�|��C��{                                    Bxtu��  �          @�p�@'
=�33�n�R�A�\C���@'
=�s33�����`��C��q                                    Bxtu�v  
�          @�z�@$z��	���s�
�B��C�  @$zῃ�
��  �c��C�
=                                    Bxtu�  �          @�  ?�z����R�(�����HC��
?�z����\��
��G�C�|)                                    Bxtu��  �          @�  ?����(��B�\��{C���?�������(���\)C�                                    Bxtu�h  �          @��\>�G�����>#�
?���C�0�>�G���33��
=�d��C�B�                                    Bxtv  �          @����G���
=?�=qA)�C��=��G���G��
=q����C���                                    Bxtv�  �          @�z῅���  ?+�@У�C�⏿�����R�p���33C�ٚ                                    Bxtv*Z  �          @����ff���>u@z�C����ff���H��{�V�\C���                                    Bxtv9   �          @��H����  ?W
=A��C������  �G���{C��)                                    BxtvG�  �          @��
�Y������?   @���C��ÿY������\)�/�C��                                    BxtvVL  �          @�(���
=���>��R@C33C�7
��
=��33����K33C�3                                    Bxtvd�  �          @�33��=q���>�G�@�G�C��f��=q���R��Q��<��C��                                     Bxtvs�  �          @��
������?+�@ӅC�Y�����Q�xQ��Q�C�XR                                    Bxtv�>  
�          @����z����?
=@��C����z���  ��ff�%p�C��=                                    Bxtv��  �          @�{���
��z�?0��@�ffC��׼��
��33�z�H�  C��                                    Bxtv��  �          @��R��Q���?!G�@�33C����Q����
��ff�!�C���                                    Bxtv�0  �          @����z���p�?B�\@��HC��q��z���z�k��ffC��)                                    Bxtv��  �          @�\)��
=���?��A(z�C��
��
=���
=��
=C��q                                    Bxtv�|  �          @\��R��{@�A��C�����R����>�33@P��C�{                                    Bxtv�"  �          @�=q��Q���=q?У�Ay�C�AH��Q�������\)�0��C�Q�                                    Bxtv��  �          @�=q��33��?�Q�A5G�C�K���33���׿���Q�C�S3                                    Bxtv�n  �          @�=q�\��?G�@�  C�+��\����k���C�*=                                    Bxtw  �          @�논���
=?G�@�C��)����ff�p���  C��)                                    Bxtw�  �          @���?z���ff�������C���?z���p���=q���\C���                                    Bxtw#`  �          @���>�ff���>��?���C�"�>�ff���ÿ�ff�p  C�5�                                    Bxtw2  �          @�Q�?h����33��
=��=qC�l�?h������z����C���                                    Bxtw@�  �          @���?����(�>��H@�
=C�H?�����׿����9G�C�R                                    BxtwOR  T          @���?��
���?�  A�\C��?��
��33�0����p�C�H                                    Bxtw]�  �          @��?c�
��{?5@ָRC�B�?c�
��zῂ�\��C�K�                                    Bxtwl�  �          @��?����  >Ǯ@mp�C��
?���������Mp�C��                                    Bxtw{D  �          @�G�=�\)����<��
>W
=C�XR=�\)���׿ٙ����C�\)                                    Bxtw��  �          @�Q�?=p���=�?�Q�C��?=p���
=�˅�v=qC���                                    Bxtw��  �          @�Q�>W
=��{�@  ��{C��q>W
=��
=����G�C�3                                    Bxtw�6  �          @�Q켣�
��?@  @�  C��=���
���Ϳ}p��Q�C���                                    Bxtw��  �          @�Q�>.{���R>#�
?�G�C��{>.{��  �����tz�C��)                                    BxtwĂ  �          @���?&ff��ff���
�AG�C�)?&ff��33�G����C�N                                    Bxtw�(  �          @�=q?333��
=>k�@	��C�Y�?333���ÿ\�ip�C�u�                                    Bxtw��  �          @��H?(����?5@�ffC���?(���{����!C��=                                    Bxtw�t  �          @�(�?������>aG�@�C��?�����\��ff�j=qC�\                                    Bxtw�  T          @�=q=�G���{?��@�G�C���=�G����H���H�:�\C��=                                    Bxtx�  �          @\���
����?��A��C�f���
��ff�:�H��p�C��                                    Bxtxf  �          @�=q>����  =�G�?��C��R>�����׿�33�\)C��                                     Bxtx+  �          @�G�?�����33�����u�C��?�����z��I���   C��                                     Bxtx9�  �          @�\)?Y�����׿O\)��ffC�5�?Y�������{�ƣ�C��R                                    BxtxHX  �          @���?���\)>�?��\C�xR?���  �У��|z�C���                                    BxtxV�  �          @��?p�����?B�\@�ffC���?p�����
���\��
C��=                                    Bxtxe�  �          @�33?k���  >�p�@^{C�W
?k����H���X(�C�t{                                    BxtxtJ  �          @���?0�����H�^�R��\C�b�?0����=q�#�
���C���                                    Bxtx��  �          @Å��z�����>�?��
C�P���z������˅�y�C�R                                    Bxtx��  �          @���(���\)@z�A�=qCwE�(���?(��@�\)Cx�                                    Bxtx�<  �          @Å������R?�{Av=qC�� ������#�
�\C���                                    Bxtx��  �          @Å���
���?У�Axz�C�p����
�\�B�\���C�~�                                    Bxtx��  
�          @�녿����R?��A�33C�*=����Q�=�Q�?Y��C�N                                    Bxtx�.  �          @�  >�(�����?B�\@陚C��>�(��������!�C��                                    Bxtx��  �          @�  ?�����G�=u?��C�f?���������
=���HC�S3                                    Bxtx�z  �          @�  @
=q��=q��G���ffC���@
=q���ÿ��
��z�C�)                                    Bxtx�   �          @�
=?����ff�#�
���C��3?������z���(�C�33                                    Bxty�  �          @�Q�?������>�G�@��RC�H�?����zῪ=q�N�\C�o\                                    Bxtyl  �          @��?������<#�
=�\)C�T{?����  ��p���z�C��                                     Bxty$  �          @�G�?����\)>�p�@b�\C���?����=q�����V=qC�0�                                    Bxty2�  �          @�G�?����G�>��@�
=C��=?���������H  C�
                                    BxtyA^  �          @���?�  ����>#�
?��
C�p�?�  ���������v{C�                                    BxtyP  �          @�33?�=q��(�>u@\)C���?�=q��p��Ǯ�m��C��f                                    Bxty^�  �          @�=q@���R���H�8(�C��R@���\�3�
�݅C��=                                    BxtymP  �          @�Q�?�����\����{C���?�����R�c�
�z�C�t{                                    Bxty{�  �          @���@z���p��G���p�C��@z���
=�o\)���C��                                    Bxty��  �          @���?�Q���z�(�����HC�u�?�Q���p��
=��\)C�B�                                    Bxty�B  �          @��R@$z���p������V�RC�8R@$z���  �9�����
C��H                                    Bxty��  �          @���@P  ���׿�p��n�RC�ٚ@P  �u�333���HC��                                    Bxty��  �          @��@p  ��
=�O\)�\)C��\@p  �p�������RC�s3                                    Bxty�4  �          @��@|����녿@  ����C�q@|���hQ��(���
=C��R                                    Bxty��  �          @���@�G��vff�\�s�
C�f@�G��b�\�Ǯ�zffC�4{                                    Bxty�  �          @���@L(���  ����{C���@L(���  ��  �t(�C���                                    Bxty�&  �          @��\@\�����#�
����C�33@\����{���R�m�C��                                    Bxty��  �          @�G�?�����z������ffC��?�����z��G���HC�n                                    Bxtzr  �          @���@z=q�~{=�\)?@  C�N@z=q�r�\���A��C��3                                    Bxtz  �          @�z�@���dz�#�
��G�C�C�@���X�ÿ����0��C���                                    Bxtz+�  �          @���@�z��\(��O\)�   C���@�z��@�׿������C��                                    Bxtz:d  �          @�33@��\�9���Ǯ�x(�C��f@��\��������\)C�u�                                    BxtzI
  �          @�z�@�ff�@�׿�G����C��3@�ff�33�'
=��Q�C��{                                    BxtzW�  �          @�(�@���Q��
=q���RC�� @���(��E����C�c�                                    BxtzfV  �          @��H@�Q��Q��  ���C�l�@�Q쿐���/\)����C��                                    Bxtzt�  �          @�(�@�  �G��1G���  C�~�@�  ����QG����C��
                                    Bxtz��  �          @�z�@�녿Ǯ�>�R��{C��@�녿
=q�Tz��	�
C�Ф                                    Bxtz�H  �          @��\@��0���>{����C��)@�>�\)�B�\����@S33                                    Bxtz��  
�          @�33@�{?���������HA9@�{?�����R��G�A�Q�                                    Bxtz��  �          @���@P  ��)���
{C���@P  �����K��+�C�]q                                    Bxtz�:  �          @��
@\)���\�!G���(�C�C�@\)�a��u�&��C���                                    Bxtz��  �          @�z�@h���n�R�'����C�'�@h���,���j�H�  C���                                    Bxtzۆ  �          @��
@�
=�.�R�#�
��ffC��\@�
=��G��S33�\)C��=                                    Bxtz�,  �          @��@O\)�r�\�7���\)C�@ @O\)�*�H�{��,��C�>�                                    Bxtz��  �          @��@,(���33�3�
��Q�C��q@,(��N{����0Q�C��{                                    Bxt{x  �          @���@A��s�
�P���(�C�AH@A��#�
����=�\C���                                    Bxt{  �          @��\@2�\�Q��qG��$��C�/\@2�\�����(��W��C���                                    Bxt{$�  �          @���@J�H�"�\����3�C���@J�H�����p��Y{C���                                    Bxt{3j  �          @�  @�H�`���l(��#p�C�O\@�H�����
�\�\C�`                                     Bxt{B  �          @�  @q�� ���P  �\)C�33@q녿����x���0��C�]q                                    Bxt{P�  
�          @�Q�@L���Fff�W
=�(�C���@L�Ϳ�=q��{�B\)C��                                    Bxt{_\  �          @�G�@5��i���\(���HC��@5�����ff�J  C�G�                                    Bxt{n  �          @���?���p��/\)���C�(�?��aG����
�7  C�J=                                    Bxt{|�  �          @��@	������&ff��z�C�}q@	���`���}p��/C���                                    Bxt{�N  �          @��R?����ÿ���C
=C�\?��|(���R��{C��=                                    Bxt{��  
�          @�33@p  ���H�u��0�C�3@p  =���Q��;�?�\                                    Bxt{��  �          @�{?�z��l(��@���ffC�p�?�z��\)����[=qC��R                                    Bxt{�@  �          @�p�@q녿�p��c�
�"G�C���@q녾��{��7��C���                                    Bxt{��  �          @�33@�p�?�33�\���ffAw\)@�p�@�R�8Q����RA���                                    Bxt{Ԍ  �          @��H@Q��  �n�R�,C���@Q녿c�
�����M��C�c�                                    Bxt{�2  �          @��
?�
=��z��'�����C��H?�
=�QG��z�H�;G�C��                                     Bxt{��  �          @��@z��|(��"�\��(�C�>�@z��8Q��l���4=qC�u�                                    Bxt| ~  �          @�(�@/\)�fff�7
=�(�C���@/\)�(��x���:�RC�'�                                    Bxt|$  �          @�\)@-p��s�
�-p���Q�C���@-p��,(��tz��4
=C���                                    Bxt|�  �          @���@  ��{�s33�(�C��
@  ���\�(Q���{C�
                                    Bxt|,p  �          @��H@{��
=�^�R�33C�� @{��(��#�
��ffC��=                                    Bxt|;  �          @�=q@G���  ��R���C���@G��h���{��)��C��\                                    Bxt|I�  T          @�G�?�  ���R�>�R��z�C�ٚ?�  �\(���z��E�C�Ф                                    Bxt|Xb  �          @��\?����33�1���=qC�^�?���hQ���  �;z�C�1�                                    Bxt|g  �          @��H?�G�����7���C��q?�G��fff���H�@{C�XR                                    Bxt|u�  �          @��\?��H��(���z�����C��{?��H��  �XQ��33C��R                                    Bxt|�T  �          @�G�>��H��33��ff�PQ�C�~�>��H���\�G���HC��                                    Bxt|��  �          @��?�p������   ��=qC���?�p���G��k����C��R                                    Bxt|��  �          @�\)?E����H��  ���C�
?E����]p��Q�C��                                    Bxt|�F  �          @���>�Q���ff�����6�RC��>�Q���  �8����=qC�/\                                    Bxt|��  �          @�Q쾀  ���\?^�RA=qC��쾀  ������=q�0��C���                                    Bxt|͒  �          @��
>�\)��Q�>�Q�@n�RC�o\>�\)��녿�ff���
C�}q                                    Bxt|�8  �          @��
@��{�g
=�A�C�� @녿�=q����u�HC���                                    Bxt|��  �          @��@*�H�!G���=q�q��C�Z�@*�H?������kQ�A��
                                    Bxt|��  �          @�z�@!녾�  ��\)�q��C�"�@!�?�ff��G��a�HAٮ                                    Bxt}*  T          @��
@2�\�G��g��4p�C�n@2�\�c�
���[C�*=                                    Bxt}�  �          @��
@.{�%�Fff�
=C�1�@.{��{�s33�MQ�C���                                    Bxt}%v  �          @��H@)�����R����
=C�\)@)����  ����G�C���                                    Bxt}4  �          @��R@�R���?n{A�C���@�R���׿E���HC��H                                    Bxt}B�  �          @��\?�{��  ?@  A ��C��\?�{��ff����4Q�C�f                                    Bxt}Qh  �          @���?n{��
=@%A�Q�C���?n{���H?^�RA�C�0�                                    Bxt}`  �          @���?#�
��(�@J�HB��C�e?#�
��\)?��
A�C���                                    Bxt}n�  �          @��?fff��G�@H��BffC��?fff���
?�Q�Ax��C��)                                    Bxt}}Z  �          @�=q?�  ����@R�\B��C���?�  ��p�?�A�p�C��q                                    Bxt}�   �          @�녽�G����@XQ�B��C�P���G����?�\)A���C�n                                    Bxt}��  �          @�  >.{��ff@aG�B��C�,�>.{��p�?���A�\)C��{                                    Bxt}�L  �          @����u���R@g
=B"�\C���u���R?�33A�z�C��                                    Bxt}��  �          @��R��\)��33@P��B\)C�����\)��
=?\A�C���                                    Bxt}Ƙ  �          @�{���R��33@O\)B�C��R���R���R?�  A�{C�N                                    Bxt}�>  �          @�p��\)���H@N{B
=C�\�\)��ff?�(�A}�C�7
                                    Bxt}��  �          @�G���Q�����@$z�A�G�C�|)��Q���  ?+�@�p�C���                                    Bxt}�  �          @����G���ff@   A��C�Uý�G����H=�G�?�Q�C�b�                                    Bxt~0  �          @�\)�u��p�?�\A�33C��=�u���R�8Q��C��                                    Bxt~�  �          @�  ������?�Q�A�ffC�+������
=�#�
�uC�9�                                    Bxt~|  �          @��H�
=q����@z�A�Q�C��ÿ
=q��G�>�Q�@p  C�:�                                    Bxt~-"  �          @�z�    ��@&ffA�ffC��    ����?B�\AC��                                    Bxt~;�  �          @��?��|��@B�\Bp�C��R?�����?�
=A�=qC�/\                                    Bxt~Jn  �          @�=q=����e@^�RB0�C���=�����33?�(�A�33C��)                                    Bxt~Y  �          @��ÿ&ff�B�\@|��BOQ�C��{�&ff��  @%A��RC���                                    Bxt~g�  �          @�z�@  �=p�@��
BUC�=�@  ���@1G�B�C��q                                    Bxt~v`  �          @�=q�(��=p�@���BT��C�+��(���\)@-p�B��C��                                    Bxt~�  T          @��þ��
�?\)@q�BMC��;��
���@(�A�RC�Ǯ                                    Bxt~��  T          @�p�@ff�
�H��\��C���@ff����G�����C��3                                    Bxt~�R  �          @��@E��7
=������C���@E����O\)�(��C���                                    Bxt~��  �          @��\?�����@��A�33C��?�����?.{@���C��                                    Bxt~��  �          @���?˅��\)@(�A��HC�H�?˅��=q?:�HA
=C��                                    Bxt~�D  �          @�{@�����?�G�A���C�l�@���z�=�\)?O\)C���                                    Bxt~��  �          @��@Q���@
�HA�ffC���@Q���{?   @�(�C�y�                                    Bxt~�  �          @���?�  ����@UB
=C���?�  ���R?��A��
C�n                                    Bxt~�6  �          @�
=?����w�@~�RB2�\C�!H?������@�\A�=qC�H�                                    Bxt�  �          @��?�G���(�?��A��
C�AH?�G���\)�#�
��ffC��=                                    Bxt�  �          @��\?�������>�@�{C�P�?������\��  �{�
C��q                                    Bxt&(  �          @��?�\)��p�    =#�
C�q?�\)��G����R��G�C��H                                    Bxt4�  �          @�=q?@  �������,��C���?@  ��  �G���\)C�G�                                    BxtCt  
�          @���?����=q�z���33C��\?����\)� ����33C��                                     BxtR  �          @��R?�����  ?=p�A
=C�*=?�����{�����C�
C�G�                                    Bxt`�  �          @���?����@A�Q�C���?����
=>�G�@��C��
                                    Bxtof  �          @�p�@\)����?�A��HC�~�@\)��zὣ�
�Y��C���                                    Bxt~  �          @��H@�
��33?���Ag
=C��q@�
��Q����p�C�^�                                    Bxt��  �          @�
=>�ff��(�@(�Aʣ�C�|)>�ff��p�>�33@^�RC�@                                     Bxt�X  �          @���>����33?�33A��C�+�>����p��u���C��                                    Bxt��  �          @���>�
=���R>W
=@ffC�q>�
=���
��������C�>�                                    Bxt��  �          @��>B�\����?���AV�\C��q>B�\��z�\(��
{C��R                                    Bxt�J  �          @���<#�
��z�?\(�A\)C��<#�
���������V{C��                                    Bxt��  �          @������
��33@,��A�G�C������
���?z�@���C���                                    Bxt�  �          @�  �����R?�G�A���C�Y�����
=��(���33C�`                                     Bxt�<  �          @����33��
=?���A2=qC�=q��33��
=��\)�0��C�=q                                    Bxt��  T          @��R>�����<�>���C�l�>�����
=��
=C��q                                    Bxt��  �          @�ff?�����R>\@s�
C���?����ff��p�����C�@                                     Bxt�.  �          @�G�@��\)?�ffAO�C��@���\�J=q���HC��3                                    Bxt�-�  �          @��H?����G�>�?�p�C��\?����p����R��=qC���                                    Bxt�<z  �          @��H?�����R��
=�:�\C�]q?����33�J�H�  C���                                    Bxt�K   �          @�(�?����H�У�����C�` ?�����c33��C�g�                                    Bxt�Y�  �          @��?�\)��(���z��<��C�#�?�\)�����HQ��{C�j=                                    Bxt�hl  �          @�(�?�ff��(��"�\���HC��?�ff�p������?G�C��f                                    Bxt�w  �          @�Q�?�ff���
�����
=C���?�ff�s�
���:�C���                                    Bxt���  �          @�ff?0����=q�\)���
C�ٚ?0���n{��  �AffC�8R                                    Bxt��^  �          @�\)?޸R��
=���R�rffC�<)?޸R����X�����C�+�                                    Bxt��  �          @��?�G���Q��%����C��R?�G��g���=q�C��C��3                                    Bxt���  �          @���?�\)��p��/\)��  C�˅?�\)�^{���G�C��f                                    Bxt��P  
�          @��H?�(���z��ff��C���?�(��u�����5��C�y�                                    Bxt���  �          @�z�?�{��z��L����\C��
?�{�A���  �X�C�                                    Bxt�ݜ  �          @�?�(�����G
=���C��)?�(��J=q���R�R��C�O\                                    Bxt��B  �          @�?������;���ffC��?����J�H�����R�HC���                                    Bxt���  �          @�\)?u�������
�U�C�"�?u��(��P  ���C�/\                                    Bxt�	�  �          @��R?������
��z��f{C�O\?�����z��Y����C���                                    Bxt�4  �          @��?�����Q���
�{�C��?�������^{���C��                                     Bxt�&�  �          @�\)?n{��������T��C���?n{��G��W��
=C��=                                    Bxt�5�  �          @��>�����Ϳ�Q��nffC�y�>�������\����RC��                                    Bxt�D&  �          @��?�\)��  �����]�C�f?�\)����Q���\C�K�                                    Bxt�R�  �          @��?���33�����]��C�\?�����N�R��
C��=                                    Bxt�ar  �          @�@,����녿
=�\C���@,����{�(��У�C�n                                    Bxt�p  �          @�\)@z���zῡG��M��C���@z���
=�L(��C�3                                    Bxt�~�  �          @�z�?���(��C33��C�s3?��3�
�����X�C�\)                                    Bxt��d  �          @���@p���ff�2�\���
C��q@p��/\)��\)�H�
C�e                                    Bxt��
  �          @��@���Q��O\)�\)C���@��������\C�~�                                    Bxt���  �          @�Q�?���b�\�p  �,�C��?�녿�  ��33�w�
C���                                    Bxt��V  �          @�33?�=q�Y������9��C�#�?�=q��  ��=qC�S3                                    Bxt���  �          @��
@
=�h���`  ��C���@
=��Q�����b��C�W
                                    Bxt�֢  �          @��@�\�\)�L���z�C���@�\�������W�C��
                                    Bxt��H  �          @�z�?����|��>�{@��HC��?����p�׿��\��(�C��                                    Bxt���  �          @�G�?��\���@P��Bz�C�Y�?��\��{?��
AW�
C��                                    Bxt��  �          @�Q�@(����@33AŮC�  @(�����>���@G�C��                                     Bxt�:  T          @�33@'���=q?^�RA�C�C�@'���Q쿓33�?
=C�j=                                    Bxt��  
(          @��
@=p���33?��\AP��C�c�@=p���
=�333���
C�                                    Bxt�.�  �          @��R@Mp����\?���A��C�>�@Mp���ff���
�W
=C�%                                    Bxt�=,  �          @�p�@mp��x��@{A�p�C��@mp����?.{@�(�C��\                                    Bxt�K�  �          @��@�(��h��@�
A���C�!H@�(����>���@uC�:�                                    Bxt�Zx  �          @�@�33�i��?��
A��C���@�33���>��?�\)C�H�                                    Bxt�i  �          @��@�p���\?���A4��C�!H@�p��ff>�=q@!�C��                                     Bxt�w�  �          @�\)@�{��ff?�  A�C���@�{����>�\)@$z�C�y�                                    Bxt��j  T          @��
@�\)��=q?�p�AW�
C��{@�\)�33?5@�z�C���                                    Bxt��  �          @ȣ�@��
�
=q�B�\��
=C���@��
��z��=q�p��C���                                    Bxt���  �          @�Q�@����
=�'��Ə\C�
@��׿Y���P����G�C�k�                                    Bxt��\  �          @���@��
���H�8Q���
=C�aH@��
����N�R��{C�/\                                    Bxt��  �          @�ff@���h���=p���RC�(�@��>\�E���ff@��H                                    Bxt�Ϩ  T          @���@��׾��5���p�C�}q@���?=p��1G���=qA ��                                    Bxt��N  
�          @��
@�G�?��ü#�
��A��@�G�?�?=p�@�\)A�ff                                    Bxt���  �          @��@�G�@�R��
=�~�RA�\)@�G�@��?
=q@�{A�p�                                    Bxt���  �          @�  @��@'
=�������Aأ�@��@#33?5@��HA�                                      Bxt�
@  �          @��@��\@333�Q���A�R@��\@8��>�
=@�\)A���                                    Bxt��  �          @��@�@:=q��  �Lz�A���@�@J=q=�\)?0��B=q                                    Bxt�'�  �          @��@�G�@333���  B
�@�G�@aG��xQ��$z�B${                                    Bxt�62  �          @�=q@Vff@E��K����B*Q�@Vff@�(���{����BK��                                    Bxt�D�  �          @���@[�@AG��G��	(�B%=q@[�@�G���=q��z�BF��                                    Bxt�S~  �          @�z�@Q�@S�
�,�����\B4��@Q�@�(������7�BN{                                    Bxt�b$  �          @��\@g
=@A��#�
���B   @g
=@tzῇ��5�B:�                                    Bxt�p�  �          @�(�@��?�p��0����\)A�p�@��@<�Ϳ�
=����B��                                    Bxt�p  �          @���@P  @�  �s33�!p�BRp�@P  @�Q�?k�A�BR��                                    Bxt��  �          @��R@i��@XQ쿰���w�
B*��@i��@h��>.{?�B3��                                    Bxt���  T          @��@c33@'��(Q����B�
@c33@^�R���
�f�\B1�
                                    Bxt��b  �          @�ff@tz�@�
�5��
A�ff@tz�@C�
���H��  B�
                                    Bxt��  �          @�\)@r�\@	���8���G�A�R@r�\@J�H��(����B�                                    Bxt�Ȯ  �          @��
@W
=@���L����B
=@W
=@R�\���R����B1��                                    Bxt��T  �          @��@Z�H@���333�	Q�B  @Z�H@O\)��=q��ffB-��                                    Bxt���  �          @���@J�H�#�
�;��*�C���@J�H?����*�H���A�                                      Bxt���  �          @���@Fff�%��@���  C�&f@Fff����tz��Ez�C��)                                    Bxt�F  �          @�33?�=q�hQ��ff����C�Z�?�=q�!G��J�H�5�C��3                                    Bxt��  �          @��
?h����\)�+����HC�,�?h����
=�.�R�=qC�{                                    Bxt� �  �          @���?��
��  ��
=��Q�C��?��
���
�����33C�0�                                    Bxt�/8  �          @�Q�?���������H���RC��?���_\)�QG��#��C���                                    Bxt�=�  �          @�  ?�(����R�8Q����C�AH?�(����1G�� �C���                                    Bxt�L�  �          @��?�����
=q��\)C�\?���(��*�H��\)C�n                                    Bxt�[*  
�          @�Q�?��R��Q쿅��5��C�Q�?��R�w
=�>�R�	��C���                                    Bxt�i�  �          @�Q�?������fff�&=qC��f?���q��2�\�p�C��
                                    Bxt�xv  �          @�  @'��\��Q����C��
@'��#�
����C�                                      Bxt��  �          @�z�@Z=q@j=q�G���=qB<(�@Z=q@�p���  �*=qBK{                                    Bxt���  �          @�p�@W�@r�\�������BA�@W�@�\)�u�.{BM��                                    Bxt��h  �          @�
=@Y��@s33��z���(�B@�@Y��@�  ���
�Y��BM\)                                    Bxt��  �          @���@��\@?\)����6ffB
�@��\@I��>�{@fffBQ�                                    Bxt���  �          @�  @���@7�������HB�
@���@g��k���B'�H                                    Bxt��Z  �          @�G�@���@<���ff���Bz�@���@dz�(����
B"��                                    Bxt��   �          @�33@��H@>{������B  @��H@n{�^�R�33B)�                                    Bxt���  �          @���@��
@AG����R�{33B
\)@��
@W�����{BQ�                                    Bxt��L  �          @��\@�G�@*=q�z����A��\@�G�@Y���k��p�B�R                                    Bxt�
�  �          @��@���@ff�����p�A��\@���@K���
=�Dz�Bp�                                    Bxt��  �          @��\@�  @��#�
��=qA�ff@�  @?\)����g33BG�                                    Bxt�(>  �          @��@�@���{��G�A�  @�@G
=��p��MG�BG�                                    Bxt�6�  �          @�G�@��R@=q�(����\A���@��R@G��h����B��                                    Bxt�E�  �          @��@�  ?�{��Q����HA�p�@�  @!녿n{���A�(�                                    Bxt�T0  �          @���@��R>��
���
��z�@b�\@��R?��ÿ��H�t��A9                                    Bxt�b�  �          @��@���?��\�!����HAn�R@���@  ���H���A�Q�                                    Bxt�q|  �          @��H@�ff?����#33��\)A�z�@�ff@#�
��=q��A��                                    Bxt��"  �          @��H@�33?u��\��{A+�@�33?�\��z��j=qA���                                    Bxt���  �          @�Q�@�
=?����
=��\)AR{@�
=@녿У����\A�{                                    Bxt��n  �          @��@�G�?��
��
��A�{@�G�@����33�j{A���                                    Bxt��  �          @��@���?�33�����p�A�{@���@"�\���l(�A�                                      Bxt���  T          @�ff@�=q@�������AΣ�@�=q@<(��.{����A��                                    Bxt��`  T          @��
@Q��k��]p��ffC��@Q��G�����n��C�4{                                    Bxt��  T          @�Q�?�p��U�y���4��C�W
?�p���  ��
=�{C��3                                    Bxt��  �          @��?�33�0  ���H�Q��C�XR?�33�����C�AH                                    Bxt��R  �          @�  @$z��=q��ff�H  C�n@$zᾏ\)���H�w\)C��R                                    Bxt��  �          @�\)@(��
=��Q��M�HC���@(��W
=���
�}=qC���                                    Bxt��  �          @�
=@z��#33��p��I�C�'�@z��
=���
B�C���                                    Bxt�!D  �          @�ff@z��������Y��C��\@z�����  �)C��f                                    Bxt�/�  �          @��R?�p��G������b=qC���?�p�<��
���H� >��H                                    Bxt�>�  �          @�\)@������H�f�RC�!H@�>L����G���@�ff                                    Bxt�M6  �          @�\)@�\�+�����F�HC�G�@�\����p��C��3                                    Bxt�[�  �          @�\)?�{����G��_��C���?�{�������C��                                    Bxt�j�  �          @�
=?�����p���p��\C���?���>�ff������A�                                      Bxt�y(  �          @�
=@�\�*=q��p��G�C�^�@�\���H��p��C��3                                    Bxt���  �          @�
=?��\����ff�s�
C��?��\=�Q���\)�@�                                      Bxt��t  �          @�33>�=q��\����k�C��R>�=q?�\����¨�By�                                    Bxt��  �          @���?��H��G��L(��  C���?��H�&ff�����j�C���                                    Bxt���  �          @�33?�\)��p��O\)�p�C��?�\)�  �����jQ�C���                                    Bxt��f  �          @��@{�i���Z�H���C���@{�ٙ����R�m  C�N                                    Bxt��  �          @��?��^{�l(��-G�C�Ǯ?���z����u�C��                                    Bxt�߲  �          @���@0  �xQ������C��@0  �&ff�\���)\)C�XR                                    Bxt��X  �          @��@7
=�Z=q��Q����C�H@7
=�G��C�
��C��\                                    Bxt���  �          @��
@c�
�@  �h���0Q�C��
@c�
�  �
�H��Q�C��R                                    Bxt��  �          @�33@o\)���У���=qC��@o\)���#�
�G�C���                                    Bxt�J  �          @�
=@����.{�\)��=qC�  @����
=��\)�u�C��=                                    Bxt�(�  �          @�
=@��
�ff>�
=@�Q�C���@��
��
�
=��33C��q                                    Bxt�7�  �          @�p�@~{�N{�   ��(�C�y�@~{�(Q��
=��{C�<)                                    Bxt�F<  T          @���@qG��@  ?.{A ��C�� @qG��>�R�E��=qC���                                    Bxt�T�  T          @�33@����$z�@33A���C�!H@����N{?#�
@߮C�3                                    Bxt�c�  �          @�G�@�p��"�\@�\A�33C�N@�p��L(�?#�
@�p�C�B�                                    Bxt�r.  �          @�G�@2�\�C33?�ffA��C�=q@2�\�a�>B�\@=qC�'�                                    Bxt���  �          @�@E�`  ?��A��HC���@E�p  �\��
=C���                                    Bxt��z  �          @��H@
=��
=?��AA�C��)@
=��ff��\)�M��C��f                                    Bxt��   �          @�Q�@   ���?�\)A�Q�C���@   ��\)�(����{C�AH                                    Bxt���  �          @�@�R��Q�?�{AF=qC�&f@�R��Q쿎{�F�HC�'�                                    Bxt��l  �          @�  ?�(���\)@33A���C�|)?�(���p�����:=qC���                                    Bxt��  �          @�(�@0����=q?���A��\C�C�@0����=q�
=���C���                                    Bxt�ظ  �          @�p�?���\)@Q�Aә�C�1�?����\<��
>�  C�
                                    Bxt��^  �          @�\)?�����@8Q�A���C��H?�����Q�>��H@��C�T{                                    Bxt��  �          @��?�ff��
=@  A�G�C��3?�ff��{���
�U�C�q�                                    Bxt��  �          @���?}p�����?0��@���C�L�?}p����������Q�C���                                    Bxt�P  �          @�ff>.{���>��
@_\)C��>.{���H�(��£�C��q                                    Bxt�!�  �          @�G�?˅��(�?���A>{C��)?˅��녿�33�lQ�C��R                                    Bxt�0�  �          @�G�@	����(�?��HAu�C��f@	������}p��%p�C���                                    Bxt�?B  �          @�  ?�\)���\?�  AR�HC��{?�\)��=q���
�W�
C��R                                    Bxt�M�  �          @��H@aG��N�R?�{A��
C���@aG��n�R=�?�\)C��=                                    Bxt�\�  �          @��\@�\)�,(�?�\)Aup�C���@�\)�AG��L�Ϳ��C�:�                                    Bxt�k4  �          @�@���{?�
=AuG�C���@���6ff=�G�?�{C�!H                                    Bxt�y�  �          @�p�@�=q�?n{A ��C�C�@�=q�녾#�
��G�C�Q�                                    Bxt���  �          @�ff@�녾�
=?xQ�A$Q�C���@�녿Q�?&ff@�{C��
                                    Bxt��&  �          @���@�(���(�?�ffA/�C��R@�(�� ��=�G�?�C�G�                                    Bxt���  �          @���@�����\�#�
��ffC�Z�@�����Ϳ#�
��(�C�9�                                    Bxt��r  T          @��@�33��\)�&ff��\)C��@�33�.{�����7�
C�b�                                    Bxt��  �          @�33@��Ϳ��G��   C�� @��Ϳ+����R�MG�C�|)                                    Bxt�Ѿ  �          @��@�\)��Q쿋��6{C�T{@�\)�@  �����{C��\                                    Bxt��d  �          @��@���#33�{��=qC��3@�녿����XQ��z�C���                                    Bxt��
  �          @��>�=q�����\����C�s3>�=q�qG���G��;�C��                                    Bxt���  �          @��>\)����0����RC��R>\)�@����Q��f��C�`                                     Bxt�V  �          @�33>�����(��]p����C��q>����p���(�C�'�                                    Bxt��  �          @�G����
���
�l(��'p�C��R���
��\)���R��C���                                    Bxt�)�  �          @�G��8Q��u��\)�8\)C��=�8Q쿼(����\��C���                                    Bxt�8H  �          @���>�����  �n�R�+Q�C��H>��Ϳ޸R��{\C��=                                    Bxt�F�  �          @��?B�\�W
=��ff�Jp�C�\)?B�\�s33��G�(�C�XR                                    Bxt�U�  �          @�ff?}p��H������R�\C�� ?}p��0����G�=qC�y�                                    Bxt�d:  T          @�?��
���\�.{��G�C��{?��
�   ���R�b��C��{                                    Bxt�r�  T          @�\)@����z�����ѮC���@���,�����R�J�C�33                                    Bxt���  �          @���?����ff������C��f?���G����\�@�HC���                                    Bxt��,  �          @���?����z��+���Q�C��?���1����\�`{C��H                                    Bxt���  �          @��?�z���Q��:=q��33C���?�z��#33��
=�i=qC�j=                                    Bxt��x  �          @��
?�p����
�Y����C��?�p���p����R�~
=C�l�                                    Bxt��  �          @��R?�R�U��G��T��C�:�?�R�=p����\ aHC���                                    Bxt���  �          @�  ?����g��#�
�G�C�5�?����7
=��
���C�l�                                    Bxt��j  �          @���@��s�
@u�B"ffC�� @���?�G�An�RC��                                    Bxt��  �          @��
?�\)���R@dz�B  C��
?�\)���?��A,Q�C��f                                    Bxt���  �          @���?�p��^{@�(�B7�C�ٚ?�p�����?�Q�A�C���                                    Bxt�\  �          @�Q�?���N{@��BH�C��f?�����R@33A��RC��                                    Bxt�  �          @�{?�Q��N�R@�  BA��C���?�Q���(�@	��A��C��R                                    Bxt�"�  �          @�z�@  �\��@tz�B+G�C���@  ���
?�A�  C�k�                                    Bxt�1N  �          @���@�R�|��@J�HB�C��@�R���?Y��A
�\C�0�                                    Bxt�?�  �          @��@2�\�S�
@b�\B{C��@2�\���?��RAv�\C���                                    Bxt�N�  �          @���@Q���G�@QG�B��C���@Q����
?fffA
=C�K�                                    Bxt�]@  �          @��?�=q����@^�RB=qC�u�?�=q���?��\A&=qC�N                                    Bxt�k�  �          @�?�Q��^�R@���B6{C��=?�Q���Q�?�A�p�C���                                    Bxt�z�  �          @�{?�{�XQ�@��
B;��C�ff?�{���R?���A��C�C�                                    Bxt��2  �          @��?�p��xQ�@|(�B.
=C�t{?�p����?��Az�\C�˅                                    Bxt���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxt��~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxt��$   N          @�p�?�ff����@5A�  C�1�?�ff���\=�?��RC�Z�                                    Bxt���  �          @�z�?�p����@'
=Aݙ�C��?�p�����L�Ϳ�C���                                    Bxt��p  �          @��?+����?���A���C���?+���녿���(��C�p�                                    Bxt��  T          @�  �������?���Ak�C��þ�����׿�Q��ip�C���                                    Bxt��  �          @���>k���p�?�Q�AmC�8R>k�����z��h��C�8R                                    Bxt��b  �          @�z�?8Q���Q�?E�@�33C���?8Q���p���\���HC��
                                    Bxt�  �          @��?k���G�?��A[33C���?k���  ��(��v{C���                                    Bxt��  �          @�=q?�R��(�?��A?
=C�C�?�R�����33��33C�Y�                                    Bxt�*T  �          @��\?E���\)?#�
@��
C��q?E����\�����
=C�K�                                    Bxt�8�  �          @��H?�G���p�>���@X��C���?�G����
�������C�E                                    Bxt�G�  �          @�\)=����Ϳ8Q���33C��H=����H�R�\���C���                                    Bxt�VF  �          @��>.{��Q쿚�H�P  C���>.{�~{�j=q�*z�C�9�                                    Bxt�d�  �          @���=p����H�G���  C�k��=p��?\)�����Rz�C��                                    Bxt�s�  �          @��H?�z�����?s33A/�C�l�?�z����Ϳ�z���\)C���                                    Bxt��8  �          @��@:�H�`��?�Q�AiG�C��@:�H�h�ÿ5�
{C�e                                    Bxt���  �          @�G�@�ff�'�?�  A��C���@�ff�I��>aG�@�C��3                                    Bxt���  
�          @�  @y���>{?�=qA�\)C�b�@y���W
=���Ϳ�=qC��H                                    Bxt��*  �          @��@K��\)@3�
Bp�C�g�@K��U?���A�Q�C�Ǯ                                    Bxt���  �          @�  @Fff�s�
?���A���C���@Fff���H�
=q���RC���                                    Bxt��v  �          @�
=@!G����?�
=A\)C�*=@!G���ff�h��� ��C��                                    Bxt��  �          @�
=@,(��p��@�\A�ffC���@,(�����=�G�?���C���                                    Bxt���  �          @�\)@'����?���A�(�C�C�@'���\)��ff��p�C�'�                                    Bxt��h  �          @�{@)����Q�?�
=At  C�4{@)�����
���\�,��C��                                    Bxt�  �          @�\)@\)���R?�A�z�C���@\)���׿(����ffC���                                    Bxt��  �          @��R@\)���?�(�A{
=C�(�@\)��\)���
�.�RC��)                                    Bxt�#Z  �          @�
=@=q��(�?�  A�=qC�� @=q��Q쿂�\�,��C�n                                    Bxt�2   �          @�{?�  ��ff?�ffA]C���?�  ��{��{�h��C��q                                    Bxt�@�  �          @�z�@+���?��Ag�
C���@+���  ����7
=C�e                                    Bxt�OL  
�          @��
@E�����?�\)AC�C�Ff@E����Ϳ�{�BffC�E                                    Bxt�]�  T          @�Q�@"�\�tz�@��A���C�Ǯ@"�\��Q�>#�
?�  C���                                    Bxt�l�  �          @�33@N�R�HQ�@{A��C���@N�R�tz�>�33@~{C�!H                                    Bxt�{>  �          @�p�@C�
�Z=q@�A�p�C��R@C�
���H>��@7�C�h�                                    Bxt���  �          @��\@E����?���AEC�Ff@E���p���{�A��C�AH                                    Bxt���  �          @��H@S33�~�R���R�]p�C���@S33�Q��G���Q�C���                                    Bxt��0  �          @���@\(��u��=q��(�C��=@\(��!G��S33��HC��)                                    Bxt���  �          @�z�@j�H�w��5���HC��)@j�H�>�R�#�
��C�o\                                    Bxt��|  �          @��@Q����J=q�
=C��@Q��Mp��2�\��ffC��=                                    Bxt��"  �          @�(�@333��  ���H���\C��R@333�hQ��,����C��{                                    Bxt���  �          @�(�@0����녿   ���\C���@0���k��.�R���C�o\                                    Bxt��n  �          @���@33��Q�>���@L(�C�#�@33��\)�G���(�C�Q�                                    Bxt��  �          @��?�33��?.{@�  C��{?�33�����
=q��C�j=                                    Bxt��  �          @�(�>�33��  ?��
A)��C���>�33��Q�����  C��=                                    Bxt�`  
�          @��?   ���R?W
=A�C�� ?   ��(���\���RC�˅                                    Bxt�+  �          @�?(����Q�?�=qA8��C��?(�����\��p���\)C��                                    Bxt�9�  �          @�p�>�����?�{A<��C��)>���33��(���z�C���                                    Bxt�HR  �          @�p�?(����G�?Q�A\)C��?(����
=��p���z�C�Ǯ                                    Bxt�V�  �          @��?E����
?h��A (�C�J=?E����
����p�C��                                     Bxt�e�  �          @�33?h����
=@
=A�{C��?h����p�����Q�C�AH                                    Bxt�tD  �          @�?p�����@��AϮC�f?p����p���33����C�l�                                    Bxt���  �          @��H?ٙ����
?��Ad��C��H?ٙ���������j�RC��                                    Bxt���  �          @�G�?����33?xQ�A!��C���?�����H�������C��f                                    Bxt��6  �          @���>\)���>���@Z=qC��)>\)��  �$z���C��3                                    Bxt���  �          @��
?Tz����R>��@*=qC�W
?Tz����\�$z���
=C��                                    Bxt���  �          @�\)?u���?�ffA��C��R?u��
=��33�L  C�xR                                    Bxt��(  �          @��?����=q@!�A��C�\?����{��z��?\)C�c�                                    Bxt���  �          @��?�33����@(��A�(�C�y�?�33��
=�8Q��C�`                                     Bxt��t  �          @��
@%��\)@$z�A��C��H@%��ff=L��>��C��3                                    Bxt��  �          @�=q@7���
=>�{@^�RC��H@7���\)�Q����HC��                                    Bxt��  �          @���@L������?xQ�A$z�C�n@L����p���{�g�C�Ǯ                                    Bxt�f  �          @��@{��tzὸQ�xQ�C�� @{��O\)��\���RC�>�                                    Bxt�$  �          @�33@����Z�H�u��C���@�������$z���C�1�                                    Bxt�2�  �          @��\@�\)�ff�   �ָRC�y�@�\)��\�N{�33C���                                    Bxt�AX  �          @��
@��׿�(��C33�
=C���@��׽��hQ��!p�C�0�                                    Bxt�O�  �          @���@����\)�+����C��H@��ÿB�\�e���
C��                                    Bxt�^�  �          @���@��R��p��Q��  C���@��R>L���l���%(�@,��                                    Bxt�mJ  �          @��
@Z�H���"�\�z�C���@Z�H��G��N�R�,�\C�]q                                    Bxt�{�  �          @��R@\���Vff�:=q��C��@\�Ϳ�\)��
=�B�C�%                                    Bxt���  �          @��R@AG�����\)�?�HC�0�@AG�>8Q���=q�g��@\��                                    Bxt��<  �          @�{@<(��
=q��=q�G=qC��q@<(�>�p���=q�iz�@�                                      Bxt���  �          @���@X�ÿh����p��N�C�|)@X��?�  ��Q��C��A�z�                                    Bxt���  �          @�\)@N{��\)��\)�_  C�b�@N{@Q����H�6ffB�H                                    Bxt��.  �          @�33@S�
������S�C�H�@S�
?����\�<�A�                                      Bxt���  �          @��@h���G��j=q�%33C�z�@h��>#�
���C�\@{                                    Bxt��z  �          @���@e� ���o\)�)G�C�^�@e>W
=����F��@X��                                    Bxt��   �          @�\)@l(��z��\(��G�C�b�@l(�<#�
��Q��=ff=���                                    Bxt���  �          @�(�@p�����A��
p�C�aH@p�׾\�q��3�C�
                                    Bxt�l  �          @�Q�@^{���������33C���@^{����8���p�C���                                    Bxt�  
�          @�=q@g���R@��A�C�/\@g��I��?fffA*{C�|)                                    Bxt�+�  �          @�Q�@X���9��@,(�A�
=C��{@X���xQ�?L��A��C���                                    Bxt�:^  T          @��H@w��5�@�A�33C���@w��fff>�@�ffC���                                    Bxt�I  �          @�(�@~�R�{@'�A�C�
@~�R�^{?uA$��C�u�                                    Bxt�W�  �          @���@~�R���@EB�C���@~�R�H��?��HA�p�C��f                                    Bxt�fP  �          @���@�=q�N{?���A3�C���@�=q�S33�B�\���C�L�                                    Bxt�t�  �          @���@���[�?�  A(��C�{@���\�Ϳn{��C�                                      Bxt���  �          @�=q@�
=�b�\?��@�ffC�H@�
=�U���=q�]C��
                                    Bxt��B  �          @���@Z=q����>Ǯ@���C�7
@Z=q�xQ��z���  C��\                                    Bxt���  �          @���@l���w�?�Q�AI�C���@l���z�H�}p��&�RC��f                                    Bxt���  �          @��@�
=�Dz�?���A��C���@�
=�\(�����-p�C�o\                                    Bxt��4  �          @��@�z��)��?�(�Aup�C�%@�z��A녾���ffC�t{                                    Bxt���  �          @���@���&ff@��A�G�C�l�@���Y��?�@�ffC��q                                    Bxt�ۀ  �          @��R@���-p�@z�A��C�J=@���a�?�@�  C���                                    Bxt��&  T          @�33@s33� ��@Q�Bp�C�
=@s33�[�?�  A�33C��
                                    Bxt���  T          @���@xQ�333@��RB;��C���@xQ��2�\@N{B{C��                                    Bxt�r  �          @��R@n{�!G�@��BAffC�4{@n{�0  @R�\BC��                                    Bxt�  �          @�{@p�׿B�\@��B=�C�O\@p���4z�@I��B\)C���                                    Bxt�$�  �          @��
@�Q����?�Q�A�{C�"�@�Q��Q�?Q�AffC�K�                                    Bxt�3d  �          @��R@�Q쾳33@   A�=qC�
@�Q쿫�?\Az�RC��{                                    Bxt�B
  �          @�p�@g
=�u@�p�B@��C��R@g
=�@  @C�
B\)C�,�                                    Bxt�P�  �          @�\)@`  �   @���B4p�C�+�@`  �tz�@=qA�\)C�Ff                                    Bxt�_V  �          @�=q@\���1G�@W
=B  C��H@\�����
?�Al��C���                                    Bxt�m�  �          @�z�@P�׾���@�  BWQ�C���@P���,(�@g�B"Q�C�=q                                    Bxt�|�  �          @��@K�<��
@��B]�
>��R@K����@|(�B2�
C�w
                                    Bxt��H  �          @���@\)<�@��B|��?��@\)�#33@�ffBFC�#�                                    Bxt���  �          @�{@K��ff@o\)B1��C�C�@K��p  @A�C�&f                                    Bxt���  T          @�{@3�
�hQ�@8Q�B Q�C���@3�
��33?��@��\C��q                                    Bxt��:  �          @�z�@\���B�\@\)A�C�O\@\���x��?�@�=qC���                                    Bxt���  �          @�(�@�=q��@	��A��
C��@�=q�Dz�?!G�@���C�H�                                    Bxt�Ԇ  �          @��@����*=q?�\A�Q�C���@����L��=�G�?�p�C�,�                                    Bxt��,  �          @�z�?��H�-p��w��N
=C�(�?��H��\)��
=
=C��=                                    Bxt���  �          @�G�@p��l���{����C�l�@p�������G��Z(�C��                                    Bxt� x  �          @�ff?�����Q��3�
��G�C�E?����	�����
�g�\C��                                    Bxt�  �          @�G�@p���33�:�H� \)C�*=@p����H��z��f(�C�AH                                    Bxt��  T          @���@,���Dz��hQ��&G�C���@,�Ϳ5��ff�m(�C���                                    Bxt�,j  �          @�@�R�\)�|(��AQ�C�o\@�R�#�
����x{C��H                                    Bxt�;  �          @��@�~�R�8Q�� 
=C�5�@��\)�����b��C��\                                    Bxt�I�  �          @��R?޸R���\�E��{C��?޸R���������wC��H                                    Bxt�X\  �          @�Q�?��H���R�)����z�C��H?��H�����33�e��C�Ǯ                                    Bxt�g  T          @�Q�?�����
=�HQ��
�C�5�?��ÿ�����(��{ffC�xR                                    Bxt�u�  �          @���?�=q��33�R�\�{C�  ?�=q��  ���R�{�HC�R                                    Bxt��N  �          @��
@/\)���R���H���C���@/\)�0�������7�C�e                                    Bxt���  �          @�(�@\)��(���Q����C�%@\)�:=q���\�;Q�C�C�                                    Bxt���  �          @�p�@&ff�����
�H����C��@&ff�-p�����A�HC��                                    Bxt��@  �          @���@A���{���R��z�C�+�@A��.{�����2�RC��                                    Bxt���  �          @�\)?�33�p  �K���C�>�?�33��G���k�C�j=                                    Bxt�͌  �          @�ff��=q�[���Q��R��C�Ǯ��=q�
=q��z�§�fCs��                                    Bxt��2  �          @������e���
�;�Cx޸��녿aG������CTO\                                    Bxt���  �          @�(���G���p��c�
��C}5ÿ�G���z���\)�Ch�)                                    Bxt��~  �          @��H=�Q��w
=�~{�7
=C��3=�Q쿘Q����33C�Ff                                    Bxt�$  T          @��
?G��Z=q��33�LC�p�?G������  �3C�:�                                    Bxt��  �          @�=q?���Q������xffC��?��?���p�Q�A��                                    Bxt�%p  T          @�Q�?�G��z������g��C���?�G�>���p�{Av{                                    Bxt�4  �          @��?��R�&ff����Q\)C���?��R<���
=p�?z�H                                    Bxt�B�  �          @�@Z�H�S�
��p����
C��
@Z�H���@���=qC���                                    Bxt�Qb  �          @��@xQ��g����
�k�C�}q@xQ��C�
��
=���HC��)                                    Bxt�`  �          @�(�@�Q��!녿���_\)C�b�@�Q쿺�H���ظRC��                                    Bxt�n�  �          @��@�z��!녿��\�XQ�C��R@�zῼ(��=q��
=C�4{                                    Bxt�}T  �          @��\@`����
=�
=��(�C�� @`���QG��.�R��p�C��                                    Bxt���  �          @���@7
=��zῨ���b{C���@7
=�A��\���\)C��                                    Bxt���  �          @��@(������
=q���C��H@(��� ������B  C�:�                                    Bxt��F  �          @�  @o\)�U�(����\C��@o\)��
=�hQ��&
=C��                                    Bxt���  �          @��@�z��*�H��=q�6�\C��@�z��Q�����{C���                                    Bxt�ƒ  �          @���@h���s33���R�|��C���@h���(��Q��(�C��                                    Bxt��8  �          @���@���J�H�ٙ���z�C�]q@�����
�G��
p�C�g�                                    Bxt���  
�          @��@��
�Dz��z����C���@��
��G��XQ���\C��{                                    Bxt��  �          @�Q�@-p��;��p  �,�RC�ff@-p��   ��\)�o  C�˅                                    Bxt�*  �          @���?�
=�����aHC�Ff?�
=?�p�����\B"ff                                    Bxt��  �          @�p�?�(���=q���\�C��\?�(�?�33���HaHBQ=q                                    Bxt�v  �          @��@�Ϳ�  ��{�l  C��H@��?�  �����~�HA��                                    Bxt�-  �          @�z�@*�H��ff��33�`z�C��=@*�H?�\)��
=�i�A���                                    Bxt�;�  �          @�z�?�{���
�1G���(�C�%?�{�G���G��k=qC�N                                    Bxt�Jh  �          @���?���
=�p���=qC���?��{��dz��  C�g�                                    Bxt�Y  �          @���?�\)����p��I�C��=?�\)�n�R�r�\�)  C�G�                                    Bxt�g�  �          @�{?��H���R�����ap�C��?��H�l(��{��0G�C�k�                                    Bxt�vZ  �          @��?�z��-p��tz��M�C��q?�zᾊ=q��p��C���                                    Bxt��   �          @��\?��Ϳ\(���
=aHC��?���@
�H���H�}ffBh                                    Bxt���  �          @�Q�@(��޸R��ff�mG�C��f@(�?��
����33AɅ                                    Bxt��L  T          @��\@��#�
����8RC�w
@�@Q���33�d\)B6                                    Bxt���  �          @�=q��33?Y�����H¡L�B����33@l(������D�\B��                                    Bxt���  �          @��R@P������@�\A�Q�C��)@P�������{�S�
C��                                    Bxt��>  �          @���@[���=q?Tz�@�ffC���@[����ÿ�{��33C��R                                    Bxt���  �          @�G�@\�����?�p�A=G�C�  @\����p���p��d��C�5�                                    Bxt��  �          @�p�@�Q�����?W
=@��C�� @�Q����ÿٙ���G�C��                                    Bxt��0  �          @��@xQ�����=���?xQ�C��\@xQ����H�������C��R                                    Bxt��  �          @���@w
=���\>�=q@%C��@w
=�����p����RC���                                    Bxt�|  �          @��@z=q�����{�L��C���@z=q�u��3�
�مC��f                                    Bxt�&"  �          @��@l(���p���
���C���@l(���
��p��/
=C��                                    Bxt�4�  �          @ə�@;��P  ��(��4{C�  @;���(���{�u��C��)                                    Bxt�Cn  T          @�G�?����fff��33�M=qC�  ?����������� C��                                    Bxt�R  T          @�Q�?�  �L(���G��X�C�Y�?�  ���
���RW
C��q                                    Bxt�`�  �          @ȣ�@Q��L����=q�I�C�]q@Q�������{C�AH                                    Bxt�o`  �          @�=q@,(��G�����D=qC�c�@,(�����p�G�C�                                    Bxt�~  �          @��
@1��G
=��  �B�HC��@1녽�G������C���                                    Bxt���  �          @ʏ\@#33���Q��g��C�S3@#33?��
��=q�~�
A�\)                                    Bxt��R  �          @�(�@|��?�\)����G33A}�@|��@a��aG��Q�B'33                                    Bxt���  �          @�z�@xQ�?J=q��\)�N{A9�@xQ�@Tz��s33�z�B"Q�                                    Bxt���  �          @�33@��?�ff�����9G�A�33@��@n�R�A���(�B*ff                                    Bxt��D  �          @�\)@W�@p����H�?�B�@W�@����%�����BUff                                    Bxt���  �          @��H@l��?�=q���JG�A��R@l��@Z=q�X����B*                                    Bxt��  �          @�33@h��?}p������O=qAtQ�@h��@X���a��z�B+��                                    Bxt��6  �          @�p�@Tz�?333�����b\)A=G�@Tz�@U��\)�!33B4(�                                    Bxt��  
�          @�Q�@3�
@U���\�9��BG�R@3�
@�Q��ff��G�Bw�                                    Bxt��  �          @���@hQ�>Ǯ�����R(�@�z�@hQ�@7��u��p�B(�                                    Bxt�(  �          @ƸR@8Q�?�\)�����kp�A�\)@8Q�@~{�o\)�ffBX{                                    Bxt�-�  �          @�@C�
?�ff���`
=A��@C�
@n{�_\)�z�BJQ�                                    Bxt�<t  �          @ʏ\@S�
�L�����
�i33C�N@S�
@,(�����=�HBff                                    Bxt�K  �          @�G�@Y��?�������I�RA���@Y��@�p��>{��BKff                                    Bxt�Y�  �          @��H@�@S33��p��M�
BeG�@�@���=q��Q�B�G�                                    Bxt�hf  �          @�(�?�ff@9�����\�eG�Bh��?�ff@�G��>{�߅B�k�                                    Bxt�w  �          @˅?�p�@����ff�o��BL(�?�p�@�
=�S�
��{B��{                                    Bxt���  �          @�G�@   ?�p�����y�A��H@   @��R�z�H���Bm{                                    Bxt��X  �          @��@Y��?�=q�����Uz�A�@Y��@����[��=qBG33                                    Bxt���  �          @�(�@C�
?�z�����]B   @C�
@����Z�H�
=B\�
                                    Bxt���  �          @��
@2�\@7
=��{�L�B7
=@2�\@����*=q��  Btp�                                    Bxt��J  �          @��@8Q�@Fff��p��?\)B<�@8Q�@����
��Br�\                                    Bxt���  �          @�  @p�@HQ���=q�I�RBOG�@p�@�
=��H���
B�u�                                    Bxt�ݖ  �          @Ǯ@��@^{�����B
=Bf��@��@�����\)B�Ǯ                                    Bxt��<  �          @�=q?��H@e���\)�P��B�z�?��H@���
��=qB�\                                    Bxt���  �          @�=q?�p�@o\)��G��G�B��)?�p�@��R�z���ffB���                                    Bxt�	�  �          @��H@�@k���\)�@�Br�@�@�(���\��z�B�p�                                    Bxt�.  �          @��H?��@.�R����u��B��3?��@����P  ����B�
=                                    Bxt�&�  �          @��H@\)@333���H�V�BA(�@\)@�=q�4z���=qBQ�                                    Bxt�5z  �          @�33?��@@  ��  �_�Bg33?��@��\�6ff�ծB��                                    Bxt�D   �          @�33@�R@������i(�B>�H@�R@�p��O\)��\)B�Q�                                    Bxt�R�  �          @˅@
=q@p����k��BB��@
=q@�
=�R�\��=qB��                                    Bxt�al  �          @��
@,��?�p���z��p  AÅ@,��@qG��l���Q�BY�                                    Bxt�p  T          @��H@�=q?\)���\�9�
@�33@�=q@:=q�e�
=qB��                                    Bxt�~�  �          @�(�@��
��(���33�9Q�C�*=@��
@����
�!�
A�\)                                    Bxt��^  �          @�z�@��׿333��z��H��C�
@���@ ����Q��4p�A�Q�                                    Bxt��  �          @�33@z=q�n{����I�C�H�@z=q?�����
�<  Aģ�                                    Bxt���  �          @�Q�@o\)���R��
=�FG�C�*=@o\)?�  ��G��J33A�                                    Bxt��P  �          @�G�@qG�����{�BC�\@qG�?�=q����Lz�A33                                    Bxt���  �          @�=q@XQ쾞�R�����ep�C�XR@XQ�@#33����>�B
=                                    Bxt�֜  �          @�=q@l��?z���Q��T��A=q@l��@I���|(��  B!�\                                    Bxt��B  �          @�(�@��ý#�
��33�I(�C��q@���@#33��(��#�B�                                    Bxt���  �          @�=q@���?�=q�@������A��@���@<(����H�|z�A��H                                    Bxt��  �          @��
@��R?G���33�"��A�\@��R@5�C�
����A�z�                                    Bxt�4  �          @�(�@��
��Q���p��=�
C���@��
?�(����H�9z�A�\)                                    Bxt��  �          @ə�@c�
�.{����[Q�C���@c�
@#33��(��4  B(�                                    Bxt�.�  �          @�(�@��Ϳ��ff��(�C���@���>���'
=��33@7
=                                    Bxt�=&  
�          @�33@��У��>{��
=C�G�@�>B�\�XQ��G�@Q�                                    Bxt�K�  �          @\@��\�(�ÿ�G����RC�>�@��\����8����33C��f                                    Bxt�Zr  �          @�  ?�
=�P  @��
B`  C��
?�
=��
=@'�A���C�3                                    Bxt�i  �          @ʏ\@j=q�x��@Y��B��C��q@j=q��33?Q�@�p�C��3                                    Bxt�w�  �          @�p�@e��z�H@dz�B��C�8R@e���
=?s33A��C�@                                     Bxt��d  �          @�Q�@���Y����
���\C�n@����ff�c33�{C�)                                    Bxt��
  �          @�G�@�=q�qG�����A�C��)@�=q� ���G
=��(�C�Ff                                    Bxt���  �          @ʏ\@����G��u��\C�  @����G��%�\C��{                                    Bxt��V  �          @��@j=q���?s33A(�C�  @j=q�����=q��=qC��
                                    Bxt���  T          @���@��
���þ���s�
C�y�@��
�L����R��=qC���                                    Bxt�Ϣ  
�          @��@b�\�n�R?��HA���C�� @b�\��ff���R�O\)C��                                    Bxt��H  T          @˅@0����{@fffB
ffC��@0����p�?5@�C��R                                    Bxt���  �          @ȣ�@h�����@7
=A�C��\@h�����R>�\)@(Q�C�!H                                    Bxt���  
          @��H@��
���H@�A��C�w
@��
���H��Q��R�\C��                                    Bxt�
:  �          @�
=@��\�Y��@(�A�33C���@��\� ��?�\)At��C��                                    Bxt��  T          @�=q@�33��Q�@�RA��HC��R@�33�'�?�  A�RC�|)                                    Bxt�'�  T          @�33@��
��G�@e�B	��C�g�@��
�0  @   A�ffC�޸                                    Bxt�6,  �          @�z�@�  �˅@H��A�Q�C��\@�  �@��?���A�(�C�3                                    Bxt�D�  �          @ʏ\@��H��z�@
=A�{C���@��H�*=q?�\)A$��C�K�                                    Bxt�Sx  �          @�=q@�=q�B�\@.�RA��HC�"�@�=q��?�A���C��H                                    Bxt�b  �          @�p�@��>�@��A��?�G�@����
=@	��A��HC�XR                                    Bxt�p�  �          @��@���?��@P  A��A^{@������@\��B�\C�                                    Bxt�j  �          @�
=@�\)@��@O\)A���A�@�\)>���@���BG�@q�                                    Bxt��  �          @�=q@���@�R@I��A�  A���@���>k�@w
=B{@\)                                    Bxt���  �          @��H@��\@Y��=�?��B{@��\@4z�?��A��A�G�                                    Bxt��\  �          @�(�@���@a녾\�S�
B=q@���@L��?��AX��A�z�                                    Bxt��  n          @�33@��R@N�R��(��qG�A�=q@��R@>�R?��A;�A�=q                                    Bxt�Ȩ  
�          @��H@��R@��?:�H@���B   @��R@I��@3�
A�ffB��                                    Bxt��N  �          @�z�@��R@�=q?333@��B$�@��R@S33@7�AͅB�\                                    Bxt���  
          @�33@�p�@i��?E�@�  B  @�p�@,��@%�A���A��                                    Bxt���  �          @�(�@��@���>�33@B�\B  @��@N�R@=qA�\)A�p�                                    