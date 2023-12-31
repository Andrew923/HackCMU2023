CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230802000000_e20230802235959_p20230803021618_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-08-03T02:16:18.907Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-08-02T00:00:00.000Z   time_coverage_end         2023-08-02T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�8�@  T          A�@x������@�B%�RC���@x����p�@��B8=qC��H                                    Bx�8��  �          A�@l(���G�@��RA�Q�C�AH@l(����@��B�C�h�                                    Bx�8Ҍ  �          AG�@]p����
@~�RA�(�C��H@]p����@��
B\)C���                                    Bx�8�2  �          @��@a���
=@a�A�  C�˅@a�����@�p�A���C���                                    Bx�8��  T          A (�@j�H��G�@Tz�A�{C�q@j�H��z�@~�RA�\C��\                                    Bx�8�~  �          A(�@�=q�Ϯ@FffA�(�C��@�=q���
@q�A��
C���                                    Bx�9$  �          A@n{��  @EA�
=C��H@n{��(�@qG�A�  C���                                    Bx�9�  �          A=q@������@>{A���C�33@����G�@h��AԸRC���                                    Bx�9*p  �          A�@u��θR@<(�A��C�XR@u���33@hQ�A�C��                                    Bx�99  
�          A�\@u����@@  A�Q�C�#�@u���{@l��A�G�C�޸                                    Bx�9G�  �          A\)@p����33@G
=A�  C��R@p����
=@s�
A�G�C���                                    Bx�9Vb  	�          A�@`  ��\)@b�\A�(�C���@`  �ə�@�Q�A�\C���                                    Bx�9e  T          Aff@^�R���@p  A�=qC��=@^�R�ƸR@��RB Q�C���                                    Bx�9s�  "          A�\@W
=��p�@vffA�=qC�Y�@W
=�ƸR@��B�\C�33                                    Bx�9�T  
�          A�H@Z=q����@x��A�  C���@Z=q��@�33B\)C�p�                                    Bx�9��  "          AQ�@Y������@��\A�G�C���@Y�����@���B	{C�t{                                    Bx�9��  
�          A	��@g����@��RA���C�s3@g�����@��Bz�C�q�                                    Bx�9�F  �          A�@a�����@��A���C�33@a���\)@�Q�B�\C�Ff                                    Bx�9��  
�          A��@S�
�ƸR@���B��C��@S�
���\@�{B)��C�W
                                    Bx�9˒  �          A�\@O\)��=q@��
B{C��@O\)����@ȣ�B3p�C�t{                                    Bx�9�8  �          A@n{���@��RB=qC�AH@n{��
=@�(�B�C��                                    Bx�9��  �          A
=q@c�
��
=@Z�HA��HC���@c�
��G�@�A�\C�C�                                    Bx�9��  �          A   @j=q���H@)��A�33C��f@j=q��  @XQ�A�{C�.                                    Bx�:*  
Z          A��@s�
�׮@@��A�=qC��)@s�
��33@p  A�
=C�w
                                    Bx�:�  �          A��@x������@Dz�A��
C�&f@x����Q�@s33A�Q�C��=                                    Bx�:#v  �          A ��@���Ϯ@��A~�\C�W
@����ff@:�HA��HC���                                    Bx�:2  �          @�=q@XQ��ҏ\@{A��C��R@XQ���Q�@L��A��
C�0�                                    Bx�:@�  
�          @���@U����@$z�A�{C�H�@U���=q@S�
Aƣ�C��                                    Bx�:Oh  �          @�ff@S�
��z�@8Q�A��C�B�@S�
��Q�@g�A�ffC��\                                    Bx�:^  T          @�@Z=q��G�@9��A��C��@Z=q���@hQ�A�{C�y�                                    Bx�:l�  T          @��
@i����@,(�A��C�Ǯ@i���\@Z=qA�G�C�|)                                    Bx�:{Z  T          @��H@L(���p�@'�A��
C��=@L(���=q@W�A�\)C�e                                    Bx�:�   T          @��@Tz����
@(�A��C�S3@Tz��ə�@K�A���C��=                                    Bx�:��  �          A Q�@}p�����@�A��
C��H@}p��ƸR@G
=A�
=C�E                                    Bx�:�L  
�          A�@|(�����@C�
A��RC��\@|(���(�@s33A�  C�^�                                    Bx�:��  �          A  @l(���ff@?\)A��HC�j=@l(����@p  A�p�C�*=                                    Bx�:Ę  �          A  @h���׮@Dz�A��HC�0�@h���ʏ\@uA�C���                                    Bx�:�>  T          A�@z�H���@4z�A�Q�C�:�@z�H����@e�A�ffC��
                                    Bx�:��  �          A(�@���\)@%A�(�C�*=@���(�@U�A�Q�C���                                    Bx�:��  �          A��@��\��{@#33A�\)C��R@��\���H@R�\A�
=C�xR                                    Bx�:�0  �          A(�@������@�Am�C��{@�����H@<��A��\C���                                    Bx�;�  �          A
�H@�=q����?��A.�RC�9�@�=q��G�@{A�  C��3                                    Bx�;|  
�          A  @�z���Q�?�ff@��
C�%@�z��ڏ\?�\)AE��C�}q                                    Bx�;+"  T          Az�@�{�ᙚ?�@X��C�1�@�{��{?�{A=qC�j=                                    Bx�;9�  �          A�\@��R���>8Q�?�33C���@��R��{?��\@��C��                                    Bx�;Hn  T          Ap�@������H��=q��(�C�E@�����=q?z�@j=qC�N                                    Bx�;W  T          A{@�ff�ᙚ�z�H����C�z�@�ff�����E�C�Y�                                    Bx�;e�  �          Aff@�=q��ff���
� Q�C��R@�=q�ᙚ��(��,��C���                                    Bx�;t`  T          A=q@��\�ۅ��  ��HC�+�@��\�߮�(�����
C��                                    Bx�;�  �          A@�=q��z�����;�C�+�@�=q��=q������z�C�Ф                                    Bx�;��  �          A�R@�G�������\�BffC���@�G���
=����׮C�w
                                    Bx�;�R  T          A�R@�����\�u���C��H@����z�        C�g�                                    Bx�;��  �          A
=@�33��
=��
>�ffC�޸@�33��R?�=q@�p�C��)                                    Bx�;��  �          A�\@���
=��\)���C��)@���=q?p��@�=qC��                                    Bx�;�D  
�          A33@��
���
��
�D��C�5�@��
��녿�{�ҏ\C��q                                    Bx�;��  �          A\)@���������j{C�� @�����\)��G���HC�\                                    Bx�;�  �          A�R@��H��\�(��Q�C�5�@��H��G���p���=qC���                                    Bx�;�6  "          A�
@�(����G��?�
C�"�@�(����
��ff�ƸRC���                                    Bx�<�  �          A��@������
���D��C���@�����=q��\)��=qC�@                                     Bx�<�  �          A�@��\��{�33�B�\C��R@��\��zῈ����=qC��H                                    Bx�<$(  �          A�
@�33��ff��p��$(�C��)@�33����8Q�����C���                                    Bx�<2�  T          A  @�=q� zῼ(��
ffC��@�=q�ff��ff�'�C���                                    Bx�<At  �          A33@�  � �ÿ�������C���@�  ��\��z��(�C�b�                                    Bx�<P  T          A@�G���þB�\����C�AH@�G��(�?\(�@���C�Q�                                    Bx�<^�  �          A33@�\)�33�B�\����C��@�\)��\?c�
@��
C��3                                    Bx�<mf  "          A�H@�{��׾��R��G�C�� @�{�(�?@  @�33C���                                    Bx�<|  "          A=q@�Q��33�#�
�L��C���@�Q��{?��@���C��                                    Bx�<��  �          A�
@�����\?W
=@��\C��@�����?�z�A0z�C�e                                    Bx�<�X  �          A�
@����	G�?!G�@g
=C�8R@�����H?�p�A\)C�q�                                    Bx�<��  �          A Q�@�ff�33>B�\?���C��3@�ff�	?��@���C��
                                    Bx�<��  "          A"ff@�z��{>��H@1G�C�g�@�z���
?��AQ�C���                                    Bx�<�J  "          A Q�@�  �Q�?u@�
=C�9�@�  �	�@�A@(�C��                                     Bx�<��  "          A#33@�G���R?�@ӅC�)@�G��
=@z�AR�HC�k�                                    Bx�<�  �          A!�@�Q��  ?�33A   C�E@�Q��  @!�Ai�C��H                                    Bx�<�<  "          A#
=@�  ��?�=qA
=C��@�  �	p�@.�RAx��C�u�                                    Bx�<��  "          A'�@S�
�33@���A��
C��@S�
��=q@�33B �HC�u�                                    Bx�=�  �          A)�?�=q���@�z�B){C�� ?�=q��\)Ap�BF�HC��
                                    Bx�=.  �          A(��?����33@�G�B=qC�'�?�����Q�@�=qB7=qC��                                    Bx�=+�  �          A)G�?�z�� ��@��B
=C��
?�z����H@�p�B9��C���                                    Bx�=:z  �          A)�@(��G�@�G�B
=C��3@(���(�@��B5p�C��                                    Bx�=I   "          A)�@G�� z�@�
=B=qC��f@G����H@�B4��C��H                                    Bx�=W�  T          A(z�@
=q��(�@�p�B$G�C��=@
=q��z�@�(�BA�RC��                                    Bx�=fl  
�          A*=q?��H��  @�Q�B,�\C�N?��H�θRA33BJQ�C�l�                                    Bx�=u  T          A*ff?������R@�
=B*�RC��
?�����p�A
=BI(�C��R                                    Bx�=��  �          A)@
=q��
=@޸RB#�
C��@
=q�ָR@�{BA��C��=                                    Bx�=�^  �          A-p�?�33����@��B%��C�� ?�33��33A�\BC��C��)                                    Bx�=�  �          A1�?(����ffA	G�BKp�C���?(����
=A33Bj�\C�J=                                    Bx�=��  �          A-�����R�ȣ�A
=BV��C�g����R����A�BuQ�C��                                    Bx�=�P  "          A*{@+��33@p  A�G�C���@+����@�Q�A�C�k�                                    Bx�=��  
(          A&�H@e����@A�A��C��=@e��z�@���A�Q�C�
                                    Bx�=ۜ  
(          A&�\@q����@#33Ab�HC���@q���\@s�
A�{C�|)                                    Bx�=�B  �          A'�@��H�z�@'
=Ag\)C���@��H�G�@w�A��C�l�                                    Bx�=��  T          A'\)@�33�ff@G�A2�\C�˅@�33�z�@S�
A���C�9�                                    Bx�>�  �          A(  @dz��G�@ffAN�HC�33@dz���\@j=qA�G�C��H                                    Bx�>4  �          A((�@fff��\@�
A4��C�8R@fff�Q�@X��A�z�C���                                    Bx�>$�  �          A(Q�@`  �\)@z�A5�C��@`  ��@Z=qA�p�C�G�                                    Bx�>3�  
�          A(Q�@_\)��H@�A?
=C��@_\)�z�@aG�A�=qC�P�                                    Bx�>B&  �          A(  @dz��p�@�
AK�
C�7
@dz���R@i��A���C��                                    Bx�>P�  �          A'�
@xQ���
@
=A:�\C�  @xQ��p�@\(�A�\)C��                                    Bx�>_r  "          A((�@l���
=?�Q�A  C�s3@l�����@C33A���C��=                                    Bx�>n  "          A(��@g
=�33@�
A4(�C�7
@g
=���@Z�HA�\)C��)                                    Bx�>|�  "          A)@fff��@��AE�C�&f@fff���@hQ�A�(�C��3                                    Bx�>�d  T          A*�R@i����
@�AMG�C�Ff@i�����@o\)A�Q�C��R                                    Bx�>�
  �          A)�@x���ff?���A!p�C���@x�����@N{A�  C�Y�                                    Bx�>��  �          A'�@n�R�p�?�
=A)��C��H@n�R�\)@R�\A��\C�f                                    Bx�>�V  �          A%G�@n{��?�A!p�C��)@n{�@J=qA���C��                                    Bx�>��  �          A$  @s�
��?W
=@�C��
@s�
�  @��AF�RC�4{                                    Bx�>Ԣ  �          A$z�@r�\�z�@�RAH��C�q@r�\���@c�
A�=qC��R                                    Bx�>�H  �          A$��@�  �G�?��A Q�C��)@�  �\)@H��A��
C�f                                    Bx�>��  T          A$��@|�����?�\A�C�u�@|����
@G�A�
=C�޸                                    Bx�? �  �          A%�@\)�?�  A(�C���@\)�  @G
=A�Q�C���                                    Bx�?:  �          A$z�@������?�p�A
=C���@�����H@EA�C�.                                    Bx�?�  �          A$Q�@|�����?�\)A'�
C���@|����R@N�RA��\C���                                    Bx�?,�  "          A%�@�\)��?��@��C�1�@�\)�z�@+�Ap��C���                                    Bx�?;,  �          A&{@�(��{?(�@XQ�C��f@�(���H@ ��A2�\C�                                    Bx�?I�  
�          A&{@�{�p���33��
=C��{@�{���?��@��C��                                    Bx�?Xx  �          A%G�@g
=��\>\@�C�AH@g
=��
?���A"�HC�k�                                    Bx�?g  	�          A$z�@{���
>��?O\)C�AH@{��?�=qAG�C�ff                                    Bx�?u�  �          A$(�@x����
>aG�?��RC�&f@x�����?�z�A��C�L�                                    Bx�?�j  T          A(  @xQ���
?.{@mp�C���@xQ��Q�@	��A=�C�{                                    Bx�?�  T          A(��@s�
��?8Q�@y��C���@s�
���@p�AA�C��{                                    Bx�?��  T          A)p�@h����\?J=q@���C�
@h����R@33AH��C�Q�                                    Bx�?�\  �          A)��@c�
�33?=p�@~�RC�ٚ@c�
��@��AD��C��                                    Bx�?�  
�          A)p�@c�
��H?aG�@���C��)@c�
��H@��AQC��                                    Bx�?ͨ  T          A)p�@`  �{?���@�z�C���@`  ���@=p�A�{C��                                    Bx�?�N  �          A'\)@a���
?�33@�
=C��
@a���\@9��A��RC�J=                                    Bx�?��  "          A$��@Z=q�  @   A3
=C�� @Z=q��@\��A��RC�L�                                    Bx�?��  �          A(��@^{�z�?�33A&ffC��H@^{�@Z=qA���C�'�                                    Bx�@@  T          A)G�@_\)�G�?�(�A=qC��f@_\)�
=@O\)A��HC�&f                                    Bx�@�  "          A)G�@`���z�@   A.=qC�޸@`���p�@`��A�
=C�K�                                    Bx�@%�  �          A)��@c33���?�
=A'�C��
@c33��@\��A��
C�aH                                    Bx�@42  �          A,��@L���
=@/\)Ak
=C��\@L���{@���A��RC�n                                    Bx�@B�  �          A0��@B�\�#�
@333Ak
=C�B�@B�\��\@�z�A��C��R                                    Bx�@Q~  �          A.�H@:=q�#\)@$z�AY��C��3@:=q��R@��A�33C�^�                                    Bx�@`$  T          A-�@?\)�"�H@�AF{C�0�@?\)��H@{�A��C��
                                    Bx�@n�  T          A.ff@<���"ff@)��A`��C�q@<���p�@��A�G�C��\                                    Bx�@}p  
�          A-��@8Q��!�@(��AaG�C��\@8Q����@�\)A�C�^�                                    Bx�@�  T          A.ff@<(��#�@��AJffC��@<(��33@�Q�A��\C�k�                                    Bx�@��  "          A-��@*�H�#�@�RAS33C�O\@*�H�
=@�33A�C���                                    Bx�@�b  �          A.=q@<(��#�@��A?�C�f@<(���@x��A��C�j=                                    Bx�@�  T          A.ff@0  �$  @{AQ�C���@0  �\)@��
A�p�C���                                    Bx�@Ʈ  T          A/33@p��%��@%�AZ{C��\@p����@��A�Q�C�                                    Bx�@�T  "          A/
=@9���#\)@+�AbffC��=@9���{@�=qA��C�]q                                    Bx�@��  �          A/�@5��#�
@0��AiG�C��)@5��=q@��A��C�0�                                    Bx�@�  �          A0��@'
=�%p�@9��As�C��@'
=�\)@�=qAÙ�C��                                    Bx�AF  �          A0��@)���#�@N�RA��C�E@)�����@�(�A��
C���                                    Bx�A�  �          A1p�@*�H�#�@VffA���C�L�@*�H�z�@�Q�AָRC��\                                    Bx�A�  "          A0��@5�� (�@n{A��C��@5���
@��HA�\)C��                                     Bx�A-8  �          A0z�@0����\@\)A�p�C��f@0�����@��HA�C�l�                                    Bx�A;�  "          A0��@G���
@��
A�G�C�e@G��ff@��A��RC��{                                    Bx�AJ�  �          A2�R@(�� ��@�(�A��\C�'�@(���R@���B33C���                                    Bx�AY*  
�          A2�\@p�� ��@�(�A�  C�� @p��\)@���A���C�z�                                    Bx�Ag�  "          A2ff@<��� ��@s�
A��C�.@<���(�@��RA�\C���                                    Bx�Avv  �          A1�@0���!�@s�
A�ffC��=@0���Q�@�
=A��C�J=                                    Bx�A�  
�          A1p�@)���!G�@q�A�p�C�^�@)�����@�ffA�C��R                                    Bx�A��  �          A0��@1�� ��@c�
A���C��)@1����@�\)A�RC�U�                                    Bx�A�h  "          A0��@6ff�"{@U�A�ffC���@6ff��\@���Aأ�C�n                                    Bx�A�  �          A0��@&ff� ��@qG�A��C�C�@&ff�  @�ffA�z�C��q                                    Bx�A��  �          A0��@(Q���@|��A��C�c�@(Q��=q@��
A�\C��                                    Bx�A�Z  �          A0Q�@�H�33@�G�A�Q�C��3@�H���@��RA�  C�p�                                    Bx�A�   �          A0��@ff�p�@�Q�A��C��@ff�ff@��B�
C��f                                    Bx�A�  �          A1��@ff�
=@��RA���C��R@ff�(�@�z�B33C���                                    Bx�A�L  �          A2{@=q�"ff@w
=A�Q�C���@=q���@��\A���C�>�                                    Bx�B�  �          A2ff@{�$  @g�A��C�@{�33@�(�A�\C�O\                                    Bx�B�  �          A2�\@%��$z�@`  A�Q�C�
=@%���
@�Q�A�33C��R                                    Bx�B&>  T          A3
=@(���H@���A�G�C��f@(���@�\)BG�C��)                                    Bx�B4�  
�          A2�R@p���\@�Q�A���C��
@p��
=@ƸRB(�C��\                                    Bx�BC�  �          A1��@�
�@��A���C��H@�
���@�{B\)C��                                    Bx�BR0  �          A1�@z��
=@�AҸRC���@z��
=q@��HBp�C��                                     Bx�B`�  T          A2ff@��z�@��\A��HC�o\@���R@޸RBC�AH                                    Bx�Bo|  "          A3
=?�33�=q@���A���C��)?�33�
=@�(�B%{C���                                    Bx�B~"  �          A2ff@   �ff@���A�C��q@   ��@�Q�B"�C���                                    Bx�B��  �          A1@���z�@�  A��C��H@����R@���B�C�T{                                    Bx�B�n  "          A-G�@�H��R@�Q�AиRC�4{@�H�=q@���B��C��                                    Bx�B�  �          A-�@1G��\)@vffA��C���@1G��p�@�=qA�  C��3                                    Bx�B��  T          A,(�@I���\)@!�AZ=qC���@I����@�=qA���C�XR                                    Bx�B�`  "          A,(�@N�R��
@��AB�\C��3@N�R�ff@��A��C�y�                                    Bx�B�  
�          A.=q@,����\@aG�A�\)C��H@,���p�@�G�A�\)C�H�                                    Bx�B�  "          A/�@����@��RA�p�C�E@�����@ڏ\B�C�:�                                    Bx�B�R  �          A0Q�?���Q�@��B33C�\?����p�@�B7��C��                                    Bx�C�  
�          A1�?����  @ȣ�B	=qC��{?�����p�@��HB3Q�C���                                    Bx�C�  �          A0z�?�{�{@ȣ�B
��C��=?�{��@��\B4C��                                    Bx�CD  T          A.�\?�����@ƸRB
�C�H�?����\)@�  B5ffC�0�                                    Bx�C-�  T          A,��?�ff���@�z�B-
=C��H?�ff�ƸRA\)BW��C��f                                    Bx�C<�  �          A*{?��\��z�@�G�B-��C�Ф?��\��(�A	BX�RC���                                    Bx�CK6  �          A)?�(�����@�  B,\)C��=?�(���z�A	�BWp�C���                                    Bx�CY�  �          A'���\)��  A�HBvC}�R��\)�:�HA�RB�Ct�                                    Bx�Ch�  T          A&�H������A�B}��Cyc׿���   A�
B���Cl��                                    Bx�Cw(  �          A%������\)A�B|33C|޸�����(Q�A�RB���CrJ=                                    Bx�C��  �          A%녿���
=A
�HBd�C��
���l(�A�B��C|p�                                    Bx�C�t  
�          A'���
=�љ�AffBL��C�(���
=���A(�Bx��C��                                    Bx�C�  T          A%�#�
����@���BC��C��
�#�
����A�Bp��C��=                                    Bx�C��  �          A%�?8Q����@�(�B6�HC�޸?8Q����HA
{BcffC��=                                    Bx�C�f  T          A%��>�G��׮@��\BD��C��>�G����\A  BqC�~�                                    Bx�C�  �          A'�@��z�@��B{C��
@���Q�@�
=B.��C�                                      Bx�Cݲ  �          A%G�?�  ��{@ʏ\B��C�9�?�  �љ�@���BC�\C�~�                                    Bx�C�X  T          A#�?��H�=q@���B=qC���?��H��@�{B2�
C��\                                    Bx�C��  �          A#
=?�p��	�@���A���C��?�p����@��
B)��C���                                    Bx�D	�  �          A#33?����p�@�ffA�  C�ٚ?�����{@�33B!=qC�k�                                    Bx�DJ  T          A"�H?��@��A�C���?���\)@�\)B  C�9�                                    Bx�D&�  �          A!�?�\)��@���A�{C�)?�\)��p�@�33B33C��f                                    Bx�D5�  �          A ��?�����@��
A��
C��q?�����
=@��\B  C�q�                                    Bx�DD<  
�          A ��@33���@�p�A���C�S3@33����@�33B  C�:�                                    Bx�DR�  �          A!@���p�@w�A��C�˅@����@��B�
C���                                    Bx�Da�  "          A!@ff���@I��A�p�C�R@ff��@�
=A�C��H                                    Bx�Dp.  �          A!p�@ff���@>{A��HC��@ff���@���A�G�C���                                    Bx�D~�  T          A!@�
�33@+�Av�RC���@�
�\)@���A�=qC�j=                                    Bx�D�z  �          A!G�@0���33@
�HAF�HC�!H@0�����@���A��C���                                    Bx�D�   T          A ��@"�\�G�@-p�AzffC�� @"�\�	G�@��A�  C�AH                                    Bx�D��  "          A�
@,���\)@0  A�  C�(�@,���33@��\Aڏ\C�ٚ                                    Bx�D�l  
�          A   @"�\��\@FffA�ffC�� @"�\��@��A뙚C�|)                                    Bx�D�  �          A"{@�@uA���C�t{@��@�z�B�C�5�                                    Bx�Dָ  T          A"�\?�  �	��@�\)A�ffC�%?�  ��\@���B+ffC�ٚ                                    Bx�D�^  �          A!G�?˅�@|��A���C���?˅���@�Q�B��C���                                    Bx�D�  �          A�H?�(��(�@��AS�
C�c�?�(��G�@�ffA��HC��R                                    Bx�E�  "          A ��@ ���=q@�A��C�b�@ ����=q@�ffBz�C�8R                                    Bx�EP  �          A z�@z���R@��A�G�C���@z����@�(�B��C�aH                                    Bx�E�  "          A�@   �=q@���A���C�W
@   ��33@��B��C�&f                                    Bx�E.�  "          A�?��R�p�@/\)A�  C�H?��R���@�z�A��C��=                                    Bx�E=B  �          A   @Q���
@AXQ�C�T{@Q��z�@�G�A�\)C��3                                    Bx�EK�  T          A{?޸R��@�A>�RC�5�?޸R�G�@�  A��C���                                    Bx�EZ�  "          Ap�?�����@�A_\)C���?���
{@��A��C�                                      Bx�Ei4  "          A��?�ff�{@!�An=qC���?�ff�	�@�
=A�Q�C�
                                    Bx�Ew�  "          A��?���=q@
=A_
=C��?���
�R@�=qA��C�f                                    Bx�E��  
�          A�?���(�@,(�A~=qC���?���\)@��
A�=qC�E                                    Bx�E�&  
�          AG�@!��(�@h��A��C�\@!�����@�{B��C�f                                    Bx�E��  T          A
=@,���33@~{A��\C���@,����(�@�Q�B�RC��q                                    Bx�E�r  
�          A!p�@'��@�Q�A��C�<)@'�����@��HBC�O\                                    Bx�E�  
(          A$z�@����
@��AˮC��q@����=q@�Q�Bz�C�ٚ                                    Bx�EϾ  �          A"ff@!G����@<��A�ffC���@!G���H@��A�p�C�S3                                    Bx�E�d  T          A!@%��@5�A��C��R@%��(�@��A��C�q�                                    Bx�E�
  �          A!@Q��z�@
=AX  C�  @Q��z�@�z�A�=qC���                                    Bx�E��  �          A"ff@$z��{?��
A!�C�z�@$z��  @w
=A��C���                                    Bx�F
V  �          A"�H@6ff�ff@*=qAr{C�o\@6ff�	G�@�p�A�Q�C�/\                                    Bx�F�  T          A"{@/\)�
=@(�A^�RC�R@/\)�
�R@�
=A�p�C��f                                    Bx�F'�  �          A#\)@W����?�G�@�C��3@W����@W
=A��C�5�                                    Bx�F6H  �          A#�@mp���R?���@�p�C��q@mp��=q@XQ�A��HC�Q�                                    Bx�FD�  �          A$��@S33��?�Q�A33C�w
@S33�  @s33A�
=C��                                    Bx�FS�  
�          A&{@*=q��\@p�A]�C��{@*=q�@�=qA�=qC�]q                                    Bx�Fb:  �          A%G�?ٙ��G�@\(�A��
C��?ٙ����@�Q�B  C���                                    Bx�Fp�  
�          A"�H?�Q��\)@XQ�A�
=C��?�Q��\)@�B ��C���                                    Bx�F�  
�          A"ff@G��(�@;�A�=qC���@G��	��@�Q�A�ffC��)                                    Bx�F�,  "          A$��@���{@2�\A{33C��{@���  @���A�33C��                                     Bx�F��  �          A#\)@�\�z�@8Q�A��C���@�\�	�@�\)A�33C�l�                                    Bx�F�x  
�          A&�H@#33��@/\)At  C�Y�@#33���@�z�A��
C��                                    Bx�F�  �          A'�@(Q��ff@(�A@��C�l�@(Q��{@���AƏ\C�                                    Bx�F��  
�          A$Q�?�(��@�
A9�C��f?�(���@���Aģ�C�R                                    Bx�F�j  
(          A$(�@��?�Q�A�C��@��Q�@j�HA��
C�XR                                    Bx�F�  	�          A$��@   ��?@  @�p�C��@   �z�@A�A��HC�]q                                    Bx�F��  
�          A)�@Tz��!�?.{@h��C�!H@Tz��=q@?\)A��RC��H                                    Bx�G\  
�          A1�@qG��&�\>#�
?L��C��3@qG��!p�@$z�AV�RC�>�                                    Bx�G  T          A0��@w��%G��.{�h��C�Ff@w��!G�@\)A;33C��                                     Bx�G �  �          A3�@~�R�&�\��\)��Q�C�u�@~�R�&{?��@��
C�|)                                    Bx�G/N  
�          A4��@�{�&=q��ff��(�C���@�{�'�?n{@�ffC���                                    Bx�G=�  �          A4��@���'
=��{��{C��H@���'�?���@�  C���                                    Bx�GL�  "          A5p�@���'33��ff���HC���@���(Q�?u@��
C���                                    Bx�G[@  �          A733@��
�)�����ffC��)@��
�)G�?�z�@��C���                                    Bx�Gi�  "          A9��@�ff�+
=��\)� ��C��R@�ff�,Q�?u@��C��f                                    Bx�Gx�  "          A:�H@�33�.{�z�H����C�T{@�33�,��?��AG�C�ff                                    Bx�G�2  
�          A;
=@~�R�/���녿��RC��R@~�R�,(�@�RA/�C�'�                                    Bx�G��  T          A:�\@�33�.�\��Q���
C�J=@�33�*�H@��A2�RC�~�                                    Bx�G�~  
�          A:�R@�Q��/33������
C�
=@�Q��+33@Q�A<  C�C�                                    Bx�G�$  �          A:ff@�Q��.�R���  C�{@�Q��+\)@�A,��C�C�                                    Bx�G��  �          A:ff@����-G���=q��=qC�z�@����,Q�?Ǯ@�
=C���                                    Bx�G�p  �          A:�H@�=q�.ff�n{���\C�:�@�=q�,��?�p�Az�C�P�                                    Bx�G�  
�          A:ff@�z��+33��ff��p�C�&f@�z��+33?�=q@ҏ\C�&f                                    Bx�G��  �          A9@�=q�*�H��=q��33C��@�=q�+
=?�ff@θRC�H                                    Bx�G�b  �          A9p�@�(��,Q쿋���p�C�}q@�(��+\)?���@���C��=                                    Bx�H  "          A8��@��
�+���ff��ffC���@��
�*�\?�{A Q�C��3                                    Bx�H�  "          A8��@z=q�+���{�   C��@z=q�,��?�=q@�33C���                                    Bx�H(T  �          A8z�@y���+��˅���C���@y���,��?���@��C���                                    Bx�H6�  �          A8��@hQ��-���33��C�C�@hQ��.ff?���@��\C�5�                                    Bx�HE�  T          A8��@c33�.ff�����p�C��@c33�.=q?�33@޸RC��                                    Bx�HTF  �          A8Q�@�  �+���Q���ffC�7
@�  �+
=?�G�@��C�@                                     Bx�Hb�  T          A8��@�ff�*�H�����p�C��q@�ff�)�?�ff@��C���                                    Bx�Hq�  �          A8��@��H�)G���ff��
=C�'�@��H�*{?��@��C��                                    Bx�H�8  T          A8��@�p��'��   ��C�l�@�p��*ff?0��@Y��C�B�                                    Bx�H��  �          A8��@����%��
�8(�C���@����)>�p�?�=qC��)                                    Bx�H��  �          A9��@���$z��\)�F�\C�` @���)G�>.{?\(�C��                                    Bx�H�*  �          A9p�@�Q��#
=�4z��aG�C���@�Q��)��#�
�J=qC�!H                                    Bx�H��  �          A8��@�Q��!�Tz����C��@�Q��*{�&ff�P  C��                                     Bx�H�v  "          A9G�@�33�!��W�����C�Ff@�33�)���5�aG�C���                                    Bx�H�  
�          A9�@��� ���Y����=qC��@���)G��=p��h��C�{                                    Bx�H��  T          A:=q@��R���o\)���C���@��R�)��������G�C���                                    Bx�H�h  T          A:�\@����z���p�����C��@����(Q���
��G�C�9�                                    Bx�I  "          A;33@������\��=qC�ff@���'����H�ffC�q�                                    Bx�I�  �          A<(�@����
=��(���
=C���@����'�
�޸R�z�C���                                    Bx�I!Z  �          A<  @�ff�����H���
C��)@�ff�'��������C��                                    Bx�I0   T          A<z�@�Q��{���\����C��q@�Q��'�
��Q��\)C���                                    Bx�I>�            A<z�@�\)���������  C��
@�\)�'��   ���C��R                                    Bx�IML  
�          A<Q�@����R��\)��=qC��3@���&=q�
=�9�C���                                    Bx�I[�  
�          A<Q�@����z���p���z�C�G�@����$���$z��J{C�                                    Bx�Ij�  
�          A;�@����p����R��ffC�4{@����$���ff�9G�C�                                    Bx�Iy>  T          A:�H@�������(���=qC�K�@����#�
�!��Hz�C��                                    Bx�I��  �          A;33@������������
C��q@����#����R��C���                                    Bx�I��  T          A=G�@�Q����o\)���C��{@�Q�����z����
C��                                    Bx�I�0  �          A<  @\��\��p���  C�˅@\��H������=qC��q                                    Bx�I��  "          A<(�@��\�  ���H��33C�)@��\�!��޸R�  C��                                    Bx�I�|  �          A;�@��
�=q���R��Q�C�s3@��
�"�R������C�w
                                    Bx�I�"  
�          A;�@�p���\��=q��{C�޸@�p�� �Ϳ�(��33C���                                    Bx�I��  �          A<��@�33�z�������C���@�33�#
=� �����C�e                                    Bx�I�n  �          A=p�@�G���\�����{C���@�G��#33������z�C��=                                    Bx�I�  
(          A<��@�=q�����  ����C��{@�=q�"ff�˅��\)C��                                    Bx�J�  T          A<(�@��  �������
C�8R@��#
=�
=�%C��                                    Bx�J`  �          A;33@��������\)���HC�Ǯ@����#��33�!p�C���                                    Bx�J)  L          A;33@��
�Q���ff���C��@��
�#
=�G��33C���                                    Bx�J7�  
�          A:=q@�p��=q��  ����C�W
@�p��!G��ff�&=qC�"�                                    Bx�JFR  �          A9@�  ����=q��\)C��@�  ��\�{�D��C���                                    Bx�JT�  �          A8��@�z��\)���H��=qC�� @�z��G��1G��]C�]q                                    Bx�Jc�  
�          A8��@��  ���R�֏\C��=@��p��'��R{C�s3                                    Bx�JrD  �          A8��@����\)���H��=qC��@����G��0  �\��C�`                                     Bx�J��  "          A9��@�{����(����HC��
@�{��1��]��C�o\                                    Bx�J��  T          A9@�  �
�H��z���33C�/\@�  �G��2�\�^�RC��                                     Bx�J�6  �          A;
=@�������ff��\)C��H@������%�M�C�<)                                    Bx�J��  �          A;\)@���
�\��\)��  C�+�@���(��(Q��O�C���                                    Bx�J��  �          A:ff@�\)�	����(����
C���@�\)�  �1��]�C�8R                                    Bx�J�(  �          A9�@�\)���������\)C��=@�\)�\)�3�
�`  C�E                                    Bx�J��  
�          A:{@�=q��
������C�5�@�=q��\�5��aC���                                    Bx�J�t  "          A9@�  �Q���z���G�C��@�  �
=�333�_
=C�Y�                                    Bx�J�  "          A8��@�����H��z���C���@�����R�C33�uG�C���                                    Bx�K�  T          A8��@��R�ff������{C���@��R��H�L����ffC��{                                    Bx�Kf  
�          A9G�@�������
��C�@ @������Tz���G�C�Q�                                    Bx�K"  �          A8��@��\�����\)��=qC���@��\�p��J=q�~{C��                                    Bx�K0�  "          A8z�@����\��p�����C�B�@���  �W���{C�H�                                    Bx�K?X  
�          A733@�G�� ����\)��(�C��H@�G��G��N{���\C���                                    Bx�KM�  
�          A7�
@�{���������C��3@�{�z��k���G�C��                                     Bx�K\�            A7\)@��R��z���G����\C��3@��R�z��c33��Q�C���                                    Bx�KkJ  
b          A6�H@����
=����33C�{@����{�.�R�]�C���                                    Bx�Ky�  
�          A5��@��
=�������C�+�@��z��Tz���{C�C�                                    Bx�K��  �          A6ff@�33�G������  C��H@�33�33�[���(�C���                                    Bx�K�<  \          A6�R@���ff��{�{C�
=@���p��e��Q�C��                                    Bx�K��  �          A6�\@��
�{������RC��{@��
�33�I����(�C���                                    Bx�K��  T          A6�H@�\)���������
=C�f@�\)��\�G��}C�5�                                    Bx�K�.  
�          A6�R@��R�	�������C��@��R����=p��pz�C�Z�                                    Bx�K��  �          A6=q@���{���\��{C���@���\)�I����=qC��                                     Bx�K�z  �          A6ff@��
�G���\)��
=C���@��
�\)�W����C�˅                                    Bx�K�   �          A6�R@����ff��z���C���@����$  �  �7\)C�S3                                    Bx�K��  
�          A7
=@��H�����\)��RC�aH@��H�z��0  �_�
C��                                     Bx�Ll  
�          A5�@@  �	p���\)�  C��H@@  �#
=�z�H��=qC�5�                                    Bx�L  �          A4��@����������C��@����p��@  �v=qC�                                      Bx�L)�  
�          A4��@�p���\�\��C���@�p��p��Z=q��z�C��
                                    Bx�L8^  
�          A4z�@�G�����ƸR���C�L�@�G����c33���\C�K�                                    Bx�LG  �          A4Q�@�p�� z���{��C�� @�p��  �b�\��Q�C���                                    Bx�LU�  �          A4��@�Q�����ƸR�z�C�\@�Q����dz���p�C��3                                    Bx�LdP  �          A5��@���=q��=q�z�C���@�����X������C��\                                    Bx�Lr�  "          A5�@��R� Q�����p�C���@��R�33�Y�����
C�o\                                    Bx�L��  	�          A4��@�ff��\)�\�33C���@�ff��H�[���33C�p�                                    Bx�L�B  �          A4��@�33��ff���� 
=C��
@�33���U����C�ٚ                                    Bx�L��  
Z          A5G�@�
=��ff��{��
=C�E@�
=��R�\��
=C�&f                                    Bx�L��  
�          A5��@�  �G���Q���  C�q@�  ��H�Dz��{33C�q                                    Bx�L�4  �          A5�@�� Q���33��33C��@���\�J�H��(�C���                                    Bx�L��  
Z          A5@�����Q��\�ffC��@������]p����
C���                                    Bx�Lـ  �          A5�@�=q�   ���
��Q�C�h�@�=q�ff�K����C�P�                                    Bx�L�&  L          A5�@�\)�����R���C�
@�\)��\�:�H�nffC�:�                                    Bx�L��  
b          A5@��\����  ��\)C�z�@��\�G��?\)�t  C���                                    Bx�Mr  �          A6=q@����Q���p����HC��3@�������8���k33C��                                    Bx�M  T          A6�H@��
�p�������
C�ff@��
��R�7
=�g�C���                                    Bx�M"�  T          A8  @��� ����(��{C���@������XQ����HC�Ǯ                                    Bx�M1d  �          A8��@��
������33�ffC�Ff@��
�33�x������C�Ф                                    Bx�M@
  �          A9�@�(���Q���\)�ffC��@�(�������H��Q�C��                                    Bx�MN�  "          A8��@��H����p���HC���@��H��R���H��{C��f                                    Bx�M]V  "          A8��@���ڏ\�p��1�HC�H@����
���H��C���                                    Bx�Mk�  "          A8z�@�=q�׮� (��0(�C��=@�=q�{��G����C�>�                                    Bx�Mz�  �          A8��@���أ������,�HC�1�@���{��p���(�C��=                                    Bx�M�H  
�          A9�@�G���p��z��>  C��@�G��(���ff��\C���                                    Bx�M��  �          A8z�@����  �G��G�C��q@���33�����
  C��                                    Bx�M��  
�          A8z�@����=q����8�RC�,�@���G����H��(�C�|)                                    Bx�M�:  T          A7
=@��H���������-(�C�Ф@��H��
��Q���ffC��                                    Bx�M��  �          A5�@�G���(��Ǯ�\)C��\@�G���H�^{��=qC���                                    Bx�M҆  "          A5p�@�\)��Q����33C��q@�\)����\(���\)C�>�                                    Bx�M�,  T          A5G�@�z����ƸR�  C�B�@�z���H�`����Q�C�Ǯ                                    Bx�M��  "          A4��@�����������  C��=@���z��Q�����C�y�                                    Bx�M�x  
�          A3�@�=q�����=q�
=C��{@�=q��y����ffC�%                                    Bx�N  T          A4  @����G�������RC��@���=q�e���Q�C���                                    Bx�N�  	�          A4Q�@�{��ff��
=��C�@�{���q�����C�g�                                    Bx�N*j  "          A4��@��\��G��������C�R@��\������
��33C�=q                                    Bx�N9  �          A4��@�����{��{��C�"�@����Q���G����RC�(�                                    Bx�NG�  
�          A4��@�������  �ffC�O\@�������(����\C�@                                     Bx�NV\  
�          A5G�@����  ��Q��p�C�4{@���33�������
C��)                                    Bx�Ne  *          A5@�p�����ڏ\���C�XR@�p���������p�C�l�                                    Bx�Ns�  
�          A5�@��R�陚��
=��\C�g�@��R�����������C���                                    Bx�N�N  T          A5G�@�
=������(��33C�Ф@�
=��������C���                                    Bx�N��  
�          A5p�@�ff�ʏ\����3Q�C���@�ff�	������陚C�H                                    Bx�N��  
�          A5�@�\)��
=��R�9G�C�� @�\)�Q������  C���                                    Bx�N�@  
(          A4��@������
��(��)�C���@��������\���
C��                                    Bx�N��  �          A4Q�@�\)����陚�!�\C��H@�\)������R�ř�C�XR                                    Bx�Nˌ            A4z�@����
=��p��=qC�H�@���G���G����\C�
                                    Bx�N�2  �          A4z�@��H������z��=qC�޸@��H�p��������HC��\                                    Bx�N��  
�          A4(�@��H��Q���\)���C��H@��H��\��=q���HC�w
                                    Bx�N�~  
Z          A4��@�z�������{�{C��f@�z��Q���\)��p�C�q�                                    Bx�O$  
Z          A4Q�@�=q��\)��R�&(�C�}q@�=q�����33��z�C��R                                    Bx�O�  �          A4(�@��������H�0z�C��f@����
=q��������C��f                                    Bx�O#p  
�          A4z�@�Q���{�����.��C���@�Q��
{�����C�3                                    Bx�O2  T          A3\)@��H��33����{C��R@��H��
������G�C�J=                                    Bx�O@�  
(          A3�@����޸R��(��
=C�ٚ@����  �~�R��{C���                                    Bx�OOb  "          A2�\@��������G����C��@����
�H�y�����
C��H                                    Bx�O^  "          A1@��H����(��4�C�O\@��H�33��\)��\C���                                    Bx�Ol�  T          A1p�@���G���
=�0��C�P�@��  ��G���{C�                                    Bx�O{T  "          A1�@�  ���
����1�C��@�  �p�������  C�|)                                    Bx�O��  �          A0��@����33��ff�0��C��R@�������\)��=qC���                                    Bx�O��  
�          A1G�@�Q���p����+��C�<)@�Q��G�����ٮC�%                                    Bx�O�F  �          A1��@�z���p������.Q�C��@�z���������\)C��=                                    Bx�O��  �          A5@�����{��p��G�C���@����	���G����C�xR                                    Bx�OĒ  
�          A5�@�  �أ���z��  C�Y�@�  �	���\)��ffC�                                      Bx�O�8  
�          A5G�@ڏ\�߮��ff���C�/\@ڏ\��
�A��x(�C�j=                                    Bx�O��  �          A3�@�{��33��p���(�C�'�@�{��H���
�ffC�%                                    Bx�O��  �          A3
=@������(���ffC�U�@��녿�  ��
C�T{                                    Bx�O�*            A4  @����
=��{���\C���@����=q���G�C�xR                                    Bx�P�  �          A7
=@�����H��z���G�C��@������Q��@��C���                                    Bx�Pv  
�          A7�
A�H��33��p��ffC�A�H��z����\����C��3                                    Bx�P+  "          A8  A���33���\��C�H�A���
=�l(����\C�y�                                    Bx�P9�  
Z          A6{A ����\)��G��p�C�RA ����=q��
=���\C���                                    Bx�PHh  �          A6�H@���(����
��RC��\@���ff�\)��C��)                                    Bx�PW  
�          A4z�@�������{�@\)C��@����G���{�
  C���                                    Bx�Pe�  
P          A0z�@�p�������\�`�\C���@�p���z������!�C��H                                    Bx�PtZ  �          A0  @y����p��p��j33C��=@y�����
��33�(�RC�^�                                    Bx�P�   "          A/�@����(��=q�bC��R@����  ���H�!��C���                                    Bx�P��  "          A1�@�
=��{�
=�O{C�^�@�
=��33��(����C���                                    Bx�P�L  
�          A0��@S�
��
=�\)�Z  C��R@S�
�\)������RC��R                                    Bx�P��  �          A1G�@�������33�W��C��\@��������p��33C��                                    Bx�P��  T          A0z�@�����  �	���L�HC�Ф@��������33��C���                                    Bx�P�>  �          A/
=@�����=q����?�C��@���������\���HC�R                                    Bx�P��  ~          A/33@�=q��  ��\)�;\)C�U�@�=q��p����\��p�C�P�                                    Bx�P�  *          A/
=@�{�����	�O��C�33@�{��{�Ǯ�Q�C��                                    Bx�P�0  
Z          A.{@��������p��3Q�C�\@����������Q�C��)                                    Bx�Q�  T          A.�R@�p������p��:=qC��\@�p�����
=����C�Ф                                    Bx�Q|  T          A.ff@�ff��\)��H�B��C��@�ff��Q����  C�U�                                    Bx�Q$"  "          A/33@�p���33�
=�AC���@�p���z���z��Q�C��                                    Bx�Q2�  T          A/33@����Q����F=qC��@����z�����RC��)                                    Bx�QAn  �          A.�H@����G�����E�RC�ff@�����ȣ��{C���                                    Bx�QP  T          A.�H@����������E��C�XR@�����{��Q��C��{                                    Bx�Q^�  �          A0Q�@�\)��ff���R�933C��=@�\)��ff�����ffC��)                                    Bx�Qm`  T          A0��@ƸR���R���H�4�RC�l�@ƸR��p���{����C�u�                                    Bx�Q|  "          A1��@����p�� Q��9p�C�1�@����ff�����p�C��                                    Bx�Q��  �          A2=q@�z��s33�33�E�C��@�z���G�������C���                                    Bx�Q�R  �          A3
=@����Q�����9�C��)@���������=qC��                                    Bx�Q��  �          A333@��\��(�����.�C��q@��\�  �����
=C��                                    Bx�Q��  �          A1�@�ff��  ���,C�!H@�ff��������33C���                                    Bx�Q�D  
�          A0z�@�
=�������/�HC��\@�
=�p����
���C��                                    Bx�Q��  �          A0Q�@�����\)�  �Kp�C���@�������p��	{C��                                    Bx�Q�  "          A0Q�@�(����\�	��M�HC��f@�(���\�ʏ\�\)C�aH                                    Bx�Q�6  T          A0��@����
=��\�F33C��@����z��\��\C��                                    Bx�Q��  T          A0Q�@�(���33�ff�>�
C�>�@�(�����G����C��                                     Bx�R�  "          A0(�@�
=������ff�933C��@�
=������G����
C��3                                    Bx�R(  �          A/�
@��H�����p��8��C�S3@��H��z���ff��(�C�\                                    Bx�R+�  
�          A/
=@���������\)��
C��)@����z���������C��                                     Bx�R:t  
�          A0  @��
���\��{�*Q�C��R@��
����H���C�Y�                                    Bx�RI  
�          A/
=@�G���=q���0=qC�#�@�G���R��Q��ٮC�w
                                    Bx�RW�  �          A.=q@�{�������&��C��)@�{�  ��=q��ffC��3                                    Bx�Rff  T          A-��@�����p���ff�&Q�C���@�������=q��p�C�*=                                    Bx�Ru  �          A.{@����������H�{C��q@����\)������C��{                                    Bx�R��  T          A/�@��������Q��,��C���@���{������=qC�H                                    Bx�R�X  T          A/�
@�������p��=C�j=@�����H���
��z�C��q                                    Bx�R��  
Z          A0z�@�z����H� (��:�C��H@�z���z�������  C�.                                    Bx�R��  
�          A0��@�=q��z���ff�0��C�T{@�=q� �����
��
=C�T{                                    Bx�R�J  
�          A1�@��R��{��G��2�\C�5�@��R�����������C��\                                    Bx�R��  T          A1��@�=q��p��ff�D�HC���@�=q�����������C�n                                    Bx�Rۖ  T          A1@�
=��=q����CC��@�
=������p��33C���                                    Bx�R�<  "          A2�R@��������ff�.{C�%@������G���ffC�G�                                    Bx�R��  T          A3�
@���(���  �'��C�g�@��33��\)��\)C���                                    Bx�S�  
�          A4z�@������R����/��C���@�����H��p��ڏ\C��
                                    Bx�S.  �          A1��@�
=�p���ff�U�C���@�
=��Q���z���\C�z�                                    Bx�S$�  "          A2=q@��fff����X��C�>�@���p���\�C��{                                    Bx�S3z  
�          A3\)@�Q��n�R�z��V\)C��
@�Q���G������HC���                                    Bx�SB   T          A2=q@�33�o\)���S(�C�'�@�33��  �ۅ�33C���                                    Bx�SP�  "          A3
=@����mp��G��Pp�C���@����ָR�ڏ\��C�`                                     Bx�S_l  T          A2ff@���i���p��Q�C��=@������ۅ��
C�aH                                    Bx�Sn  �          A2{@�p��k��p��Rz�C���@�p���{��33���C�!H                                    Bx�S|�  �          A3\)@����u��\)�T
=C��\@�����z���z���RC�o\                                    Bx�S�^  
�          A3�@����n{�Q��U�HC�3@�������߮�
=C��
                                    Bx�S�  T          A4Q�@����vff�z��U(�C��3@�������ff�Q�C�G�                                    Bx�S��  �          A333@��\�u���R�S\)C��@��\��(���33���C��                                    Bx�S�P  
Z          A3
=@��R�}p��(��NG�C��H@��R��{������C��
                                    Bx�S��  T          A3
=@��\�}p��
=�KC��@��\��p��ҏ\�  C��                                    Bx�SԜ  "          A2�R@�\)��G��33�L�C�l�@�\)��  �љ���RC���                                    Bx�S�B  �          A3
=@��
���
��
�M��C��@��
���H������C�5�                                    Bx�S��  
�          A2�H@�  ������H�<�RC�Ф@�  ������
=���C��3                                    Bx�T �  "          A2�R@����x���ff�SG�C�n@�����ff��G���C�=q                                    Bx�T4  	�          A2=q@���������
�N�
C���@�����(���G����C��q                                    Bx�T�  �          A2�\@�z������
�>�C��3@�z���p���(���Q�C�9�                                    Bx�T,�  �          A2�H@�{����   �6�HC���@�{��ff��  ���
C�Ф                                    Bx�T;&  �          A2�H@�G���Q��  �M��C�5�@�G���  �Ϯ���C��                                    Bx�TI�  "          A1�@�\)�}p����R33C��@�\)��  ��p���\C�f                                    Bx�TXr  �          A0��@�33��33�(��Q��C�E@�33��������C�xR                                    Bx�Tg  �          A0��@��\�\)����S��C���@��\������z����C��R                                    Bx�Tu�  �          A0��@��
�u���UQ�C�4{@��
��p��׮�\)C��                                    Bx�T�d  �          A1�@�ff�s�
����T\)C�~�@�ff��z��׮���C�'�                                    Bx�T�
  
�          A1��@��H�����G��S�C�j=@��H�����
��RC�xR                                    Bx�T��  
�          A1p�@�=q�����  �P�C��q@�=q��G��θR�\)C�{                                    Bx�T�V  L          A1�@����=q����S{C�=q@����(��ҏ\�(�C�W
                                    Bx�T��  �          A1�@�=q����z��Q��C��q@�=q��{�У��p�C�:�                                    Bx�T͢  "          A0z�@����33����\(�C���@����  �������C�Ǯ                                    Bx�T�H  "          A0��@�=q���H��H�W��C��@�=q��ff����
C��{                                    Bx�T��  T          A1��@�ff��(��ff�U�C�� @�ff��\)��z����C�ٚ                                    Bx�T��  �          A1�@�G���\)�
=q�L�HC��q@�G���R��  �	p�C���                                    Bx�U:  
�          A0z�@�p���Q��
ff�N�\C��
@�p���
=�����C�@                                     Bx�U�  �          A0z�@��H�ȣ���Q��'��C��@��H�	G���G���{C�S3                                    Bx�U%�  "          A0��@�\)��G���p��)=qC��@�\)�
=������C��                                     Bx�U4,  "          A0(�@����\)�����7p�C�0�@�������Q�C��                                     Bx�UB�  "          A/�
@�����=q���F(�C��{@����=q��p���(�C�1�                                    Bx�UQx  T          A.ff@���w
=��\�[C��@����Q���
=��\C��                                    Bx�U`  �          A.�\@�z��P  �33�e�RC�0�@�z������
=�%�RC�\)                                    Bx�Un�  �          A.{@�=q�Dz���
�h��C���@�=q�����\�)�\C�y�                                    Bx�U}j  �          A-�@��R�AG��(��kp�C��@��R���
���+��C�<)                                    Bx�U�  
�          A,��@�33�*=q�ff�r
=C��R@�33��33��(��3C��H                                    Bx�U��  
�          A-G�@�z��=q��
�t��C�B�@�z���������8ffC�                                    Bx�U�\  T          A,��@����\)�t��C�@�������=q�9�RC���                                    Bx�U�  "          A.{@�ff�&ff�33�q�C��f@�ff�\��{�4{C��q                                    Bx�Uƨ  T          A-��@���G
=��n�C��=@����Q�����,p�C�L�                                    Bx�U�N  �          A/33@�(��Z�H����h�HC�˅@�(�������
=�%\)C�/\                                    Bx�U��  T          A/\)@��R����33�y=qC���@��R������\)�:��C�8R                                    Bx�U�  "          A.�H@���8Q���R�n33C�]q@���ʏ\���.��C�c�                                    Bx�V@  �          A/\)@�ff�*�H����q�HC�1�@�ff��{��\)�3\)C��
                                    Bx�V�  
(          A/�@���&ff����s��C�h�@���������5\)C���                                    Bx�V�  "          A/�
@���;����n�C��@����p�����-�
C�7
                                    Bx�V-2  
�          A/�@��R�AG��33�mffC���@��R�Ϯ��  �,G�C���                                    Bx�V;�  
Z          A/\)@�=q�7��ff�lC���@�=q�ʏ\����-p�C���                                    Bx�VJ~  
�          A/\)@�p��K���
�f�C��@�p���=q���%C�l�                                    Bx�VY$  �          A/\)@�=q�a��
=�c��C��f@�=q���
�ᙚ� (�C���                                    Bx�Vg�  T          A.�\@��
�B�\��\�n33C�O\@��
��Q���{�,
=C��                                    Bx�Vvp  
�          A-�@�Q��*=q�ff�oz�C�g�@�Q�������H�1
=C��)                                    Bx�V�  T          A.�H@�{�^{����aC�xR@�{��G��߮��C��                                    Bx�V��  
�          A.�H@����  ���K��C�B�@�������ƸR�
ffC�L�                                    Bx�V�b  T          A.�H@��R���  �L(�C�H�@��R��ff������C��3                                    Bx�V�  "          A.�H@�ff�|(��	��PG�C��@�ff������33�(�C��                                     Bx�V��  
�          A/
=@�\)��ff�
{�P�\C���@�\)������  ��\C��
                                    Bx�V�T  
�          A.�R@�
=�{��	p��O�\C�&f@�
=�����=q�z�C��
                                    Bx�V��  
�          A-G�@�����R�(��F��C���@�����������
=C�3                                    Bx�V�  T          A-�@��\��  ���<��C�*=@��\��G������C�S3                                    Bx�V�F  �          A,z�@��\���
�{�C�RC��R@��\�����R� 
=C�˅                                    Bx�W�            A,��@�����p��{�Cz�C���@����陚��{���\C��H                                    Bx�W�  
�          A,��@�����  �(��GQ�C�4{@�����ff���
���C��H                                    Bx�W&8  
�          A-@�{��(��=q�BffC�33@�{������R��  C�                                    Bx�W4�  �          A+�
@��\�u�����K\)C�Ǯ@��\�ڏ\���H�
ffC��                                     Bx�WC�  T          A)�@�p���Q���33�7  C��@�p�����������
C��)                                    Bx�WR*  "          A*�R@����(�����!��C�7
@����ff������C��                                    Bx�W`�  
�          A)�@�  ������  �z�C�1�@�  ��ff�n�R���HC���                                    Bx�Wov  T          A(  @�\�ʏ\��
=�ə�C��@�\���H��ff�  C��                                     Bx�W~  
Z          A(��@�{���H�������C��\@�{��
=�����
=C��q                                    Bx�W��  
Z          A*{@�Q����H��\)��\)C�o\@�Q���33��ff�{C��{                                    Bx�W�h  T          A*ff@�  ��  ����υC��q@�  ��33��  �
=C���                                    Bx�W�  
�          A)�@��
�Å���R����C���@��
���H���4z�C��R                                    Bx�W��  T          A(��@��
��p���33���C�@��
��ff�
�H�>=qC���                                    Bx�W�Z  
Z          A(��@��
�������R���C��q@��
����ff�7�C��
                                    Bx�W�   
�          A)�@�
=���
������\C�Z�@�
=��z��	���;�C�/\                                    Bx�W�  
Z          A)�@�  ������z���z�C���@�  ���H����Ep�C�Y�                                    Bx�W�L  
�          A)p�@޸R�����{����C�z�@޸R��z��33�HQ�C�*=                                    Bx�X�  �          A(��@ۅ��=q��ff����C�C�@ۅ�����33�I�C���                                    Bx�X�  �          A'�@�(������
=��p�C���@�(���Q��Q��R=qC�<)                                    Bx�X>  
�          A'
=@�=q���
��  ����C��@�=q���
���8Q�C��=                                    Bx�X-�  �          A'�@�����z���
=C��)@���  ��\�I�C�^�                                    Bx�X<�  �          A&�\@�G�����������HC��H@�G��������\C���                                    Bx�XK0  �          A&=q@�
=��\)�|(���C��=@�
=��׿u��33C�g�                                    Bx�XY�  T          A%G�@�\)�񙚿�33�)�C��)@�\)��p�?��@�\)C���                                    Bx�Xh|  �          A"�H@�\)���
�1G��|��C��\@�\)��(�>�p�@�
C���                                    Bx�Xw"  T          A"�R@ҏ\��G��(���q�C��@ҏ\��\)?\)@H��C�0�                                    Bx�X��  
�          A"ff@�
=��=q�0  �|  C��f@�
=��\>�p�@
=C���                                    Bx�X�n  T          A"�\@����G��0  �{\)C���@����=q>��?��RC��                                    Bx�X�  �          A#�
@Ӆ����tz�����C�{@Ӆ���׿5�~{C�/\                                    Bx�X��  �          A#�@Ϯ��p������  C�q@Ϯ��=q�����p�C�ٚ                                    Bx�X�`  T          A#�@�����R�����{C���@����(��#33�f=qC��                                    Bx�X�  �          A"�H@����G���p���33C��@����ff����+�C���                                    Bx�Xݬ  �          A#
=@�\)��ff�������
C�H@�\)��G��Q��W�
C�]q                                    Bx�X�R  "          A"�\@�\��Q�������=qC���@�\���
����^�HC��\                                    Bx�X��  T          A#\)@���{��p���33C��3@���33�%��jffC�
=                                    Bx�Y	�  �          A#�@�Q���(����R��G�C�)@�Q�����(��]�C�t{                                    Bx�YD  "          A#
=@�(���  �������
C�~�@�(���
=�0���{�C�XR                                    Bx�Y&�  �          A#\)@�\��(��������C��@�\�ۅ�.{�w
=C��                                    Bx�Y5�  "          A#�@��H���\��Q���\)C�Ff@��H��p����H�{C�U�                                    Bx�YD6  "          A#�
@�ff��=q��(��ʣ�C���@�ff��33������C��3                                    Bx�YR�  T          A#33@����H�z=q��Q�C�ٚ@������ff���C���                                    Bx�Ya�  �          A"�H@����j�H��ff�4�C��R@������������z�C�                                      Bx�Yp(  T          A%��@�ff�ff���[�HC�XR@�ff��(��׮�"��C��                                    Bx�Y~�  �          A%@�  �'���W�C�B�@�  ���H�������C��                                    Bx�Y�t  
�          A*�\@�  �G��p��`(�C�Ǯ@�  ��
=���'ffC��                                    Bx�Y�  
Z          A*=q@�(��7�����O
=C��@�(���G����
�\)C�q                                    Bx�Y��  �          A)�@ҏ\�l(���z��1��C�]q@ҏ\��p���ff��{C��q                                    Bx�Y�f  "          A)�@��H�u���G��6(�C�o\@��H�Ӆ�������
C��                                    Bx�Y�  �          A)�@Ǯ�k����:��C���@Ǯ�������R��33C��R                                    Bx�Yֲ  �          A)G�@�z���z����633C���@�z�����������G�C��{                                    Bx�Y�X  
�          A)��@ƸR������\�6��C��f@ƸR�����\)���C�/\                                    Bx�Y��  �          A)��@�G��i����ff�:�\C��3@�G��У�������C�                                      Bx�Z�  �          A)��@�G��_\)�����<��C�|)@�G������(�� G�C�=q                                    Bx�ZJ  "          A)G�@�
=�\(����\�?  C���@�
=��(���{�33C�"�                                    Bx�Z�  �          A)p�@��
�QG��   �DG�C��3@��
�ə���p���C�
                                    Bx�Z.�  
(          A)@�Q��G���
=�C  C���@�Q������ff��C���                                    Bx�Z=<  "          A*{@��H��� ���E\)C��=@��ƸR�����
�C�l�                                    Bx�ZK�  �          A*{@���HQ��ff�HQ�C�S3@����  �Å�=qC��                                    Bx�ZZ�  T          A)�@����S33�p��F��C���@������
��\)�	33C��
                                    Bx�Zi.  �          A)��@�  �J�H�=q�H�HC�{@�  �����\�{C�ٚ                                    Bx�Zw�  
�          A)@���7����K{C�P�@����G��ȣ���HC���                                    Bx�Z�z  
�          A)G�@�=q�J�H���L�C��R@�=q��=q�����p�C�W
                                    Bx�Z�   T          A(��@��H�4z��ff�J�C��3@��H��
=��\)��\C��=                                    Bx�Z��  
�          A)G�@����E����HC�q�@�����ff���H���C�R                                    Bx�Z�l  
�          A'�@�{�<����J�C���@�{��=q��(��ffC�0�                                    Bx�Z�  "          A((�@��.{���Np�C���@���p����H�z�C��                                    Bx�Zϸ  T          A'33@��/\)��\�M=qC��q@�������Q��=qC��3                                    Bx�Z�^  
Z          A'�@�ff���G��S
=C���@�ff��
=������C��{                                    Bx�Z�  T          A'
=@�G���
=�Q��Z�
C��f@�G���33�����)ffC�Q�                                    Bx�Z��  "          A$��@�  �G��33�T{C�<)@�  ����Ϯ�  C�"�                                    Bx�[
P  "          A%G�@�=q��{��R�Z�C�j=@�=q������=q�,�C�W
                                    Bx�[�  �          A&ff@��\�����\�lG�C��
@��\�~�R��\)�H�C���                                    Bx�['�  "          A&�\@��0����
�offC�9�@��������
�D�
C��3                                    Bx�[6B  
�          A%�@�z�5�\)�o�C��@�z�������\�D��C���                                    Bx�[D�  �          A%��@�녿�Q��(��g�C�o\@�������33�5  C�g�                                    Bx�[S�  
�          A%p�@�ff���  �r(�C�Z�@�ff���H���?C��                                    Bx�[b4  �          A$��@��׿�������u{C�K�@��������z��?�\C���                                    Bx�[p�  �          A$  @�p������R�r=qC��=@�p���
=�����A�C��                                    Bx�[�  T          A$(�@��
��p��  8RC���@��
��\)��
�V�C���                                    Bx�[�&  "          A$Q�@�ff�#�
��3C��{@�ff�|����
�_G�C�aH                                    Bx�[��  �          A#
=@�������H�JffC�R@��������{�ffC�t{                                    Bx�[�r  T          A%G�@����   �
=�Q�HC�Ff@�����
=���
�\)C���                                    Bx�[�  "          A&{@�(������=q�C�C��3@�(���
=�\��C���                                    Bx�[Ⱦ  T          A%G�@�33��p��G��N=qC�  @�33���R�Ϯ��C�                                    Bx�[�d  T          A%@�33�������V33C��@�33��
=�׮�"�HC�!H                                    Bx�[�
  "          A$��@��ÿ��
�Q��jC�>�@��������{�2{C��3                                    Bx�[��  �          A$(�@��׿.{�Q��w��C��\@�������z��I��C��\                                    Bx�\V  �          A%�@�(��5�p��v{C���@�(���\)��{�H�\C��{                                    Bx�\�  "          A&�R@��ÿ˅���g��C���@�����p���=q�3  C�Ф                                    Bx�\ �  T          A%�@��ÿ�33�\)�o{C�=q@������������:=qC�o\                                    Bx�\/H  �          A%��@�(���\)����j��C�8R@�(����R��\�4G�C�C�                                    Bx�\=�  �          A%@���˅�  �q�\C��R@����  ���933C�k�                                    Bx�\L�  �          A%@�p���{�
�R�c��C��)@�p���(����/�C�K�                                    Bx�\[:  �          A$��@�33������`�C��{@�33��
=��z��({C�.                                    Bx�\i�  �          A!�@�
=���R�	G��h33C��@�
=��{���-  C�1�                                    Bx�\x�  "          A"ff@�G���  ����g
=C�� @�G���������9�\C��R                                    Bx�\�,  
�          A"�R@�
=��=q���r��C�t{@�
=��  ��p��L�\C��{                                    Bx�\��  
�          A"�R@�{�\(��
=�kffC�P�@�{���R��Q��>�RC��                                    Bx�\�x  
�          A"�H@��H�������l=qC�Y�@��H��G���z��:ffC�^�                                    Bx�\�  
�          A#33@�33�^�R��H�tp�C��@�33���\���R�D�C�S3                                    Bx�\��  "          A#�@�p���
=���oz�C�� @�p���G���(��9ffC�#�                                    Bx�\�j  T          A#
=@�p�����
=q�h33C��)@�p������z��*\)C��                                    Bx�\�  �          A$  @��Ϳ�=q�z��w\)C��@�����G����?��C�XR                                    Bx�\��  "          A%@�������33�h�
C��3@����\)���
�/
=C��{                                    Bx�\�\  �          A%��@��H��������z  C��@��H�������:��C�.                                    Bx�]  �          A$��@����p��33�r�C�h�@�����������-�C�\                                    Bx�]�  
�          A%�@���U����_(�C�g�@�����������C��q                                    Bx�](N  T          A$��@����g
=�{�YC�xR@����ڏ\������HC�~�                                    Bx�]6�  "          A$��@����&ff�
�\�ez�C�f@�����G��׮�#��C���                                    Bx�]E�  
�          A$��@�������^�RC�� @����������)  C�+�                                    Bx�]T@  "          A$z�@�{�5���
�U\)C�:�@�{��=q��  ���C���                                    Bx�]b�  �          A$z�@���*�H���X{C���@�����R��z���RC�!H                                    Bx�]q�  "          A!�@�z��N�R���T(�C��@�z��˅���C�y�                                    Bx�]�2  
�          A!�@�{����R�aG�C��H@�{��  �ָR�&Q�C��=                                    Bx�]��  T          A!�@�z��������\  C�'�@�z���
=�ҏ\�"p�C�L�                                    Bx�]�~  T          A!@��
�����S��C��@��
���\��G����C��R                                    Bx�]�$  T          A!@����
���
�L�C�Q�@����{��z��z�C���                                    Bx�]��  T          A ��@��� ����ff�Q  C�t{@����ff�ʏ\�=qC�{                                    Bx�]�p  
�          A ��@���p���p��P33C�ٚ@���{����C�
                                    Bx�]�  �          A#
=@�G���{����[�C��{@�G������Q��&�C�K�                                    Bx�]�  "          A#�@�녿��H����c��C��@����ff��Q��-p�C��                                    Bx�]�b  T          A#33@����(���R�^�\C�L�@����z���(��)C�o\                                    Bx�^  "          A"ff@\��������L�C�]q@\���\���H���C�                                    Bx�^�  "          A#\)@\��p���
=�M{C��R@\��{�˅�{C��                                     Bx�^!T  �          A#33@�G���z�����N\)C�:�@�G���z���p���HC���                                    Bx�^/�  �          A#�@�\)������_C���@�\)���H�޸R�+��C���                                    Bx�^>�  T          A#�@�  �������B
=C��f@�  ��\)���
��C�J=                                    Bx�^MF  
�          A#33@ə�����33�I33C��@ə������(��  C�(�                                    Bx�^[�  �          A#�@˅��G������G�C�>�@˅��p��ə��\)C��                                    Bx�^j�  �          A#�@θR���������E��C��@θR�������H�33C���                                    Bx�^y8  
�          A#�@�z��G����
�@�HC��q@�z������  ���C���                                    Bx�^��  f          A#
=@�Q�������Q��C��{@�Q������(��"p�C���                                    Bx�^��  
�          A#
=@�=q��z�� Q��O�C�Y�@�=q��p���G���RC�y�                                    Bx�^�*  T          A#
=@�  �޸R��33�IC�1�@�  ����33�p�C��)                                    Bx�^��  �          A#�@��H��33���H�HffC���@��H���H��(����C�Q�                                    Bx�^�v  �          A#33@�
=�˅����R��C���@�
=���������"��C�Q�                                    Bx�^�  �          A#33@����p��\)�V��C��@�����H��G��'(�C�7
                                    Bx�^��  T          A#
=@�  ��ff����Z�C���@�  ��
=��ff�,=qC�N                                    Bx�^�h  T          A"�R@�=q�����
�Xz�C���@�=q���R��z��*��C���                                    Bx�^�  �          A#\)@������
�W\)C�˅@���  �߮�,��C�aH                                    Bx�_�  �          A#�@�p������  �W(�C�E@�p������ff�+33C�                                    Bx�_Z  T          A#�@�  �\�{�R��C��f@�  ���H��ff�#�C���                                    Bx�_)   T          A#33@�(������(��XQ�C��3@�(�������  �-Q�C�8R                                    Bx�_7�  �          A!�@��ÿ�ff�ff�`�RC�� @���������z��3�C�Q�                                    Bx�_FL  T          A"{@�������p��T=qC�˅@����ff��33�*=qC��\                                    Bx�_T�  �          A!�@�zῥ����T��C��{@�z������  �(33C��R                                    Bx�_c�  �          A\)@�
=� �����X�C��f@�
=�������!{C�                                    Bx�_r>  �          A�R@�
=� ����H�^z�C�w
@�
=�����G��%{C�AH                                    Bx�_��  �          A (�@�z�.{�  �]��C��{@�z���(���p��6�HC��H                                    Bx�_��  T          A�
@��R�u�
=�\p�C��\@��R�mp�����<{C�|)                                    Bx�_�0  �          A�H@�Q���p��Z(�C�W
@�Q��dz���Q��;�
C�
                                    Bx�_��  �          A�R@��Ϳ\)��ff�U{C�E@����y����ff�1�
C�C�                                    Bx�_�|  T          A�@�=q�O\)����P=qC�8R@�=q���\��Q��*��C�
=                                    Bx�_�"  �          A
=@�녿#�
��33�PC��)@���{��ڏ\�-�\C���                                    Bx�_��  �          A\)@��\)��G��M�
C�n@��u�����,z�C�q                                    Bx�_�n  �          Ap�@Å����ff�N
=C��
@Å�l���أ��-�
C�aH                                    Bx�_�  �          A\)@�\)=#�
�����MG�>�{@�\)�R�\�ᙚ�4(�C�
                                    Bx�`�  T          A\)@��ý�����D�C�u�@����Tz���  �*��C��                                    Bx�``  T          A�@ҏ\>L����\)�B��?�  @ҏ\�@����33�-C��{                                    Bx�`"  �          A33@���>�
=��33�GG�@q�@����7
=�ᙚ�4��C��R                                    Bx�`0�  �          A
=@ȣ�>����ff�K
=@o\)@ȣ��:=q��z��7��C���                                    Bx�`?R  �          A
=@�Q�?���ff�K
=@��R@�Q��1G���R�9�HC�{                                    Bx�`M�  �          A=q@��
?0�������Fp�@�
=@��
�%����7�
C���                                    Bx�`\�  �          A=q@��R�0���  �m��C�f@��R��  �����B(�C��3                                    Bx�`kD  �          A�@�zΐ33��
�uQ�C��@�z�������p��@p�C�)                                    Bx�`y�  �          A33@���\(��	��m��C�#�@������z��?��C�8R                                    Bx�`��  �          A
=@��\����Q��lG�C��\@��\��(���\)�CG�C�s3                                    Bx�`�6  �          A@��þ�33���l�HC���@����{���Q��F�\C��                                    Bx�`��  �          AG�@�\)��ff���m�
C�ff@�\)������
=�E�RC���                                    Bx�`��  �          A�@�p������
�o(�C��q@�p��\)��  �GG�C��                                     Bx�`�(  �          A��@�{�\)��H�m�
C�� @�{��33��z��D  C�(�                                    Bx�`��  
�          A��@��   �	G��t��C��3@�������I�C�`                                     Bx�`�t  �          A  @��H�����}��C��R@��H�������P{C�%                                    Bx�`�  �          A(�@�\)��(��
�\�z(�C�G�@�\)��=q�����N��C��)                                    Bx�`��  �          A�@�Q�\)�	�yG�C��@�Q��r�\��\)�R�RC��q                                    Bx�af  �          A  @���>8Q���
�
=@=q@����c33��\)�\33C�8R                                    Bx�a  �          AQ�@��?���p��_��AZ=q@���G����
�V�
C�j=                                    Bx�a)�  �          A  @��?������f=qAG�@����R���Y�\C�\                                    Bx�a8X  �          A�
@�z�?����\�c�Ad(�@�z���\��{�Z�RC�H                                    Bx�aF�  �          A  @�  ?Ǯ���_ffA��\@�  �G���
=�[33C�z�                                    Bx�aU�  �          AQ�@�G�?�\)� ���^
=A���@�G�������\)�[
=C�޸                                    Bx�adJ  �          A�
@��H?���a\)A�z�@��H������aQ�C�7
                                    Bx�ar�  �          A�R@��@��   �^�A�
=@�녿Ǯ����cffC�o\                                    Bx�a��  �          A�\@�
=>u�
=�j@0  @�
=�P����Q��N�C�`                                     Bx�a�<  T          A�R@�z��=q�   �`\)C�q�@�z����\��=q�+�HC�e                                    Bx�a��  �          A(�@�\)@��� ���_��A�
=@�\)�����gG�C��                                    Bx�a��  �          A�@���@3�
���`
=A�\)@����Q����tQ�C��                                     Bx�a�.  �          AG�@���@*=q�G��f{A�33@��Ϳ��\�33�wffC�|)                                    Bx�a��  �          A�
@���?\��H�np�A���@����� ���hz�C��                                    Bx�a�z  �          A(�@��?�33��p��m{A�(�@����{���
�j��C���                                    Bx�a�   �          A  @�
=@
=����cQ�A�z�@�
=���� (��p(�C��
                                    Bx�a��  �          A�@�=q@ff���\�c  A��@�=q��Q��p��o  C��3                                    Bx�bl  �          Az�@�{?��R�����i��A���@�{��ff�   �nQ�C�aH                                    Bx�b  �          A�@�=q?˅����pQ�A��@�=q��Q���p��lz�C��H                                    Bx�b"�  �          A(�@�\)?˅���o�HA�=q@�\)��\���k(�C��                                     Bx�b1^  �          AQ�@�?��R����rG�A�Q�@��
=q�=q�k(�C��                                    Bx�b@  
�          A
=@�
=?�(�����v�A�@�
=���=q�n��C�`                                     Bx�bN�  T          A��@��?�
=�\)�u��A��R@����� ���m\)C�g�                                    Bx�b]P  �          Ap�@�33?�  �z��zffA��
@�33�
=� z��m�C��                                    Bx�bk�  �          A��@`  ?@  �
{�A@��@`  �>{�=q�r=qC��)                                    Bx�bz�  �          A@]p�>���
�\#�@�{@]p��N�R� ���n(�C�p�                                    Bx�b�B  �          A{@l��?G��	G�k�A?�@l���:=q��oz�C��                                    Bx�b��  �          A{@a�?8Q��
�\A8Q�@a��@  �ff�q�C��
                                    Bx�b��  �          A@O\)?=p��  k�AL��@O\)�AG���
�v��C�|)                                    Bx�b�4  �          A��@g
=?�������A�G�@g
=�)���\)�u�C��)                                    Bx�b��  �          A�@W�?�33�
�RffA�ff@W��'�����{C��                                    Bx�bҀ  �          A{@H��?z�H�Q��HA�33@H���333���}33C��                                    Bx�b�&  T          A=q@C�
?������{A��R@C�
�0  ��H  C�\                                    Bx�b��  �          A��@=p�?h����Q�A���@=p��8���{�  C���                                    Bx�b�r  �          Aff@O\)?�������A&�H@O\)�J=q���t�
C��
                                    Bx�c  �          A�\@n�R?}p��	G��=Al��@n�R�-p��33�r�C���                                    Bx�c�  �          A{@r�\?�=q�Q�Q�A~{@r�\�&ff��H�r��C�Ǯ                                    Bx�c*d  �          Ap�@aG�>Ǯ�
=qp�@���@aG��Q��   �l  C���                                    Bx�c9
  �          Ap�@\��>����
�Hz�@���@\���W��   �k�RC��R                                    Bx�cG�  �          AG�@\(�>�Q��
�\p�@���@\(��S�
� (��l�C�3                                    Bx�cVV  �          Ap�@XQ�>�z��33k�@��@XQ��X��� Q��l�RC�w
                                    Bx�cd�  �          A��@P  >�33�(��@�33@P  �W
=����p{C��                                    Bx�cs�  �          A�@Mp�>aG��  �{@|��@Mp��]p�� z��n33C�j=                                    Bx�c�H  �          A@R�\�\)�  ǮC�˅@R�\�r�\��z��f�C�xR                                    Bx�c��  �          A��@J=q�u�(�B�C��)@J=q�xQ���33�f
=C��\                                    Bx�c��  �          A�@G
=�aG��z��C��@G
=�xQ���(��g{C�XR                                    Bx�c�:  T          A��@L(����
��ǮC��3@L(��j�H��p��iC���                                    Bx�c��  �          A=q@G
=���R���(�C�)@G
=�\)����e�RC���                                    Bx�cˆ  �          A�H@I���k��=q�
C��{@I���z�H��\)�g(�C�h�                                    Bx�c�,  �          AQ�@W
=?!G��{�A*�\@W
=�HQ��G��t�\C�}q                                    Bx�c��  �          A�\@c�
?=p��33�qA;
=@c�
�C�
�
=�s�\C���                                    Bx�c�x  T          A�R@hQ�?Q���HǮAK
=@hQ��>�R�33�s�
C�K�                                    Bx�d  T          A�@k�?333��A+�@k��C�
�p��p��C�&f                                    Bx�d�  �          A�@l��?B�\�p��A:=q@l���?\)����qQ�C���                                    Bx�d#j  T          A�@l��?s33��#�Ae��@l���3�
��\�tQ�C�ff                                    Bx�d2  �          A@W�?�z��=qQ�A�33@W���H�
�R�
C�*=                                    Bx�d@�  �          A�@c33?��
���z�A�(�@c33�\)�z��|
=C�z�                                    Bx�dO\  �          A��@aG�?�ff���ǮA�ff@aG��\)�
=qz�C��)                                    Bx�d^  �          A=q@c33?�Q��{B�A�=q@c33�'
=�	��{=qC�޸                                    Bx�dl�  �          A=q@]p�?�ff�=q�)A���@]p�� ���
{�~��C�
=                                    Bx�d{N  �          A�H@P��@   �{aHA���@P�׿�{��\Q�C�&f                                    Bx�d��  �          A=q@\(�?����\8RA��@\(��!G��
ff��C��                                    Bx�d��  �          Aff@J=q?u��A�
=@J=q�8���
ff�~��C��=                                    Bx�d�@  �          A
=@3�
@33�33�)Bz�@3�
��{����)C�3                                    Bx�d��  �          Az�@"�\@]p����
=BV�R@"�\�   �33W
C�}q                                    Bx�dČ  T          A(�@�\@Z�H���#�B`��@�\�����{C�K�                                    Bx�d�2  �          A33@)��@P���  �3BK��@)���#�
���\)C�*=                                    Bx�d��  �          A��?��@��R��R�k�HB�
=?��?k��{��B��                                    Bx�d�~  �          Az�?�@��
�{�j�B��?�?Y������qA��H                                    Bx�d�$  �          A�
@aG�@'
=�	���|  B
=@aG�����\ǮC���                                    Bx�e�  �          A��@l(�?�{�33��A���@l(��{��  C�y�                                    Bx�ep  �          A��@h��?�
=�ffL�A�
=@h�ÿ�z��ffk�C�/\                                    Bx�e+  �          A  @dz�@z��p��A�33@dz��  ��\�C���                                    Bx�e9�  �          A  @e�@�
�Q�#�B\)@e���  �33\C���                                    Bx�eHb  �          AQ�@W�@��ffB33@W���G������C��                                    Bx�eW  �          Az�@U�@{�����B\)@U������G��qC��                                     Bx�ee�  �          A��@X��@   ���\BQ�@X�ÿ����p�8RC�%                                    Bx�etT  �          AQ�@O\)@   ��ffB�\@O\)�������C��3                                    Bx�e��  �          A��@Z=q@��{{B	\)@Z=q��G����33C�{                                    Bx�e��  �          A��@R�\@Q���\\B�@R�\��(���
C��=                                    Bx�e�F  �          AQ�@u@)���	G��uB
�@u��=q��\�qC�q                                    Bx�e��  �          A(�@mp�@#33�
�\�z�B
33@mp����H�33�fC��q                                    Bx�e��  �          A=q@^�R@!��	��}�HB�@^�R���H�ffC�j=                                    Bx�e�8  �          AQ�@L��@+����  B   @L�Ϳ���ff��C��q                                    Bx�e��  �          A(�@G�@0  ���{B%�@G��xQ���R��C�`                                     Bx�e�  �          AG�@8��@0���
�R� B.��@8�ÿ�G������C�e                                    Bx�e�*  �          Aff@;�@O\)���y(�B?�R@;���
=��RL�C���                                    Bx�f�  �          A�\@@��@E��z�
B7�@@�׿��ffC��R                                    Bx�fv  �          A�@�H@J�H�	G��BR\)@�H�
=�{k�C�,�                                    Bx�f$  �          Az�@@U�	W
B[��@��G���C��H                                    Bx�f2�  �          A�@G�@n{�Q��{�BjQ�@G��u����=C�H�                                    Bx�fAh  �          A��?�  @vff�Q��~z�B��?�  =�\)���Ǯ@
�H                                    Bx�fP  �          A��?��@�33���|  B��q?��>��
�ff£L�AW�
                                    Bx�f^�  �          Az�?��H@���ff�y=qB�L�?��H>��
���
A(z�                                    Bx�fmZ  �          A�?˅@�  �z��|�HB���?˅>W
=��R ��@�p�                                    Bx�f|   �          AG�?ٙ�@�{�
=�wG�B��q?ٙ�>�(���\�)A`(�                                    Bx�f��  �          A��?ٙ�@����=q�u  B��\?ٙ�?���\�A�(�                                    Bx�f�L  �          A{?J=q@�Q����c
=B�.?J=q?���
�\{Bq                                    Bx�f��  �          A33�@���{�\)B��f�@y�����
�e�B���                                    Bx�f��  �          A33�\)@˅��33�z�B��
�\)@qG���\)�k  B�u�                                    Bx�f�>  �          A�H�
=q@ʏ\��(���HB�Ǯ�
=q@n�R���l�RB�.                                    Bx�f��  �          A�\���@�=q�����B�aH���@p����p��j{B�\                                    Bx�f�  �          A�
��@��������B����@u���R�hffB�                                     Bx�f�0  �          A���
@ʏ\��(��  B����
@p  ��\)�j��B�\)                                    Bx�f��  �          A��Q�@ʏ\��p��=qB�33�Q�@u��ᙚ�e�B�3                                    Bx�g|  �          A���#33@ȣ������B�8R�#33@tz�����a��B�W
                                    Bx�g"  �          A�H�   @ƸR��ff�(�B��H�   @tz���G��`G�B�u�                                    Bx�g+�  �          A33�Q�@������Q�B�L��Q�@mp�����e�
B�k�                                    Bx�g:n  �          A
=��H@�p���=q�
=B����H@o\)��(��d=qB���                                    Bx�gI  �          A33� ��@�ff��\)��B��� ��@s33����`�HB��H                                    Bx�gW�  �          Az��>{@�33����� z�B�{�>{@��\����Q�HB��                                    Bx�gf`  �          AQ��333@�33�����B��333@�G������V{B�W
                                    Bx�gu  �          A���5�@�z���\)���B����5�@��H�����T��B�u�                                    Bx�g��  �          A���8��@�p���(�����B�u��8��@����=q�QffB���                                    Bx�g�R  �          A��3�
@˅��(�� �
B�R�3�
@�33��G��R��B��
                                    Bx�g��  �          A���8��@������ �B�\�8��@�z����H�R33B��H                                    Bx�g��  �          A���<��@���33����B�G��<��@�{��G��O�B�Q�                                    Bx�g�D  �          A���C33@�p���G�����B��C33@��R��\)�M
=B��
                                    Bx�g��  �          A�
�E@����
��  B�W
�E@�����=q�Hz�B���                                    Bx�gې  �          A�
�L��@�ff�����陚B��L��@��H�Ǯ�D��B��q                                    Bx�g�6  �          A�
�R�\@�p��~{��=qB�B��R�\@�33��p��B=qB�G�                                    Bx�g��  �          A
=��(�@�{�Z=q����B�\)��(�@�33��=q�+�CxR                                    Bx�h�  �          A=q��ff@�(��U�����B��q��ff@��\��
=�(C�                                    Bx�h(  �          A�R��@�(��[���ffB�����@�G������+\)CE                                    Bx�h$�  �          A
=�z=q@�
=�fff����B�aH�z=q@����  �2��CB�                                    Bx�h3t  �          A33�6ff@�=q���
� �
B��6ff@�33�Ϯ�Q��B��                                     Bx�hB  �          A�R�a�@ə��r�\�ݮB�u��a�@�=q���R�;�\C=q                                    Bx�hP�  �          A=q�y��@�Q��\���ȏ\B���y��@�����
�.�\C��                                    Bx�h_f  �          A���Q�@��Q���=qB�����Q�@�����)p�CG�                                    Bx�hn  �          AG��~�R@�z��Z=q��  B����~�R@�=q�����-{C��                                    Bx�h|�  �          A ���x��@\�e�ӅB�=q�x��@�ff��p��2�HC�=                                    Bx�h�X  �          @�ff�l(�@�\)�mp�����B�\)�l(�@������9(�C:�                                    Bx�h��  �          @�
=�g�@�G��n{�ޏ\B��f�g�@�(������9�RCJ=                                    Bx�h��  �          @��R�i��@��
�a��ҏ\B��i��@�����(��4
=Cz�                                    Bx�h�J  �          @�ff�i��@���[����HB�Q��i��@�33�����1Q�C�                                    Bx�h��  �          @�
=�g
=@��H�j=q��ffB�B��g
=@��R��\)�7��C��                                    Bx�hԖ  �          @�{�e@���j�H��  B�8R�e@���\)�8z�C��                                    Bx�h�<  �          @��R�a�@���fff��
=B��a�@�G���ff�6��Cs3                                    Bx�h��  �          @���b�\@�(��mp��݅B��b�\@�
=��G��9��C�R                                    Bx�i �  �          A (��Z�H@ƸR�n�R��(�B����Z�H@������\�;  C �                                    Bx�i.  �          A (��Z�H@�{�p����  B����Z�H@�����33�;�RC �                                    Bx�i�  �          A ���X��@�\)�s�
��  B�#��X��@�G�����<�C T{                                    Bx�i,z  �          A   �[�@����r�\��=qB�W
�[�@�\)����<\)C{                                    Bx�i;   �          @�\)�\��@�  �~{��
=B��
�\��@�����
=�A��C��                                    Bx�iI�  �          A ���a�@���{���Q�B�z��a�@��H��ff�>��C�=                                    Bx�iXl  �          A z��\��@����u��z�B�=�\��@�
=�����=
=CB�                                    Bx�ig  �          A z��U@�{�w����B���U@�  ��{�>�HC (�                                    Bx�iu�  �          A Q��w
=@����tz���p�B�W
�w
=@�  �����8ffC�                                    Bx�i�^  �          A ���z�H@�33�{���\)B��q�z�H@z=q����:\)C{                                    Bx�i�  �          A ���w�@��\�~�R��33B�(��w�@xQ������<p�C�                                    Bx�i��  �          A ���l��@�33���H��\B��{�l��@w
=��  �A=qC�q                                    Bx�i�P  �          A ���j�H@�(������p�B����j�H@xQ������A�
Cff                                    Bx�i��  �          A ���_\)@\�|������B���_\)@�z����R�>�C!H                                    Bx�i͜  �          A�hQ�@�����G���RB���hQ�@�����Q��?p�C��                                    Bx�i�B  �          Aff�p��@��
�vff�ᙚB���p��@��R���
�8�C�R                                    Bx�i��  �          A�H�e�@�p��~�R��G�B�8R�e�@�
=��Q��=�\CQ�                                    Bx�i��  �          A=q�fff@��
�b�\��z�B��fff@�=q��p��1�C G�                                    Bx�j4  �          A��e@Ϯ�c33�̣�B���e@���
=�0��B�                                    Bx�j�  �          Az��c33@Ϯ�mp���
=B�aH�c33@��
���
�4�
B�\                                    Bx�j%�  �          A\)�}p�@�Q��>�R����B���}p�@�p���ff��HC �
                                    Bx�j4&  �          A�
����@�\)�G����
B�����@��H��=q� �RC�                                    Bx�jB�  �          A���y��@У��Tz����HB�3�y��@�������&�
C�                                    Bx�jQr  �          A��k�@ƸR�{���z�B�ff�k�@�G����R�9��C��                                    Bx�j`  �          A��G
=@�������
��B�\�G
=@{���\)�UQ�C O\                                    Bx�jn�  �          A��G�@��
���H�,{Bܽq�G�@Z=q���z��B�G�                                    Bx�j}d  �          A�Ϳ�@����ָR�?�\B��Ϳ�@8Q���ǮB�aH                                    Bx�j�
  �          A�Ϳ�33@��H��p��G�B�녿�33@'��  �B�ff                                    Bx�j��  �          A����
@�p������F\)B�����
@.{�(���B�3                                    Bx�j�V  �          A�ÿ�
=@����  �TffB��ÿ�
=@���33ffB�
=                                    Bx�j��  �          A  ���
@�\)��Q��V�HB�8R���
@��
�H��B�z�                                    Bx�jƢ  �          A(����@�(���\�Y�B����@z��\)=qB��                                    Bx�j�H  �          A=q�^�R@�33�����ap�B�W
�^�R?��
�
�\8RB���                                    Bx�j��  �          A��E�@�(����e�
B��ÿE�?������u�B�                                    Bx�j�            A	���
=@���33�j��B�{�
=?��\)aHB�                                   Bx�k:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�k�  �          A	G�����@e���G�Q�B��;���>��H���«�{C��                                    Bx�k�  �          Aff�L��@�p���G��q
=B�z�L��?��H���£W
B��                                    Bx�k-,  �          A=q�#�
@r�\��\)�|Q�B���#�
?O\)���¨B�aH                                   Bx�k;�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�kJx              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�kY              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�kg�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx�kvj  	          A?�\@i����
=�{B��?�\?J=q� ��¦��BeQ�                                    Bx�k�  �          @�{?5@Y����z�{B�L�?5?
=��(�¦�\B(�                                    Bx�k��  �          @�{?fff@8Q���\33B�=q?fff=L����(�¦�@;�                                    Bx�k�\  ;          @�Q쿇�@�����=q�=  B��ῇ�@#�
��{�B�
=                                    Bx�k�  T          @�p���
=@�33��G��UQ�B�����
=?���z�Q�B͙�                                    Bx�k��  
�          @�{����@{���z��W�B��=����?�p���ff#�B�Ǯ                                    Bx�k�N  
�          @�����@�Q���{�Op�B�#׾���@���33z�B�                                      Bx�k��  
�          @ۅ>�\)@vff��z��^ffB��q>�\)?�=q��z��
B�
=                                    Bx�k�  �          @׮?���@1���  �}B���?���>�ff��33
=A�                                      Bx�k�@  "          @��?s33@1G����
�}�HB���?s33?   ��
=¡�A�(�                                    Bx�l�  T          @�=q?�G�@$z���ff8RB�(�?�G�>�\)��
=¡��A|z�                                    Bx�l�  �          @�z�?�=q@Q������ Br��?�=q=�\)��  ��@J�H                                    Bx�l&2  
�          @��?�@�����{Bz��?���\)���ǮC�W
                                    Bx�l4�  
�          @�p�?��R?�  ��z��HB%p�?��R��R��33�fC�XR                                    Bx�lC~  
�          @�z�>�(�@�\���H{B���>�(��#�
��Q�¬u�C��{                                    Bx�lR$  
�          @�33��@N{��(��kQ�B�k���?�{��p��B�W
                                    Bx�l`�  �          @��H��z�@�ff�c33�B����z�@XQ�����UQ�B�G�                                    Bx�lop  �          @Ӆ�AG�@����xQ��
=B�  �AG�@��H�<����B�W
                                    Bx�l~  �          @����?\)@��\�h���
=B�L��?\)@���3�
�Џ\B��                                    Bx�l��  �          @�{�+�@��\�\)��
=B�W
�+�@�Q��#�
���B��f                                    Bx�l�b  "          @�
=����@�������?�C�����@r�\�6ff���HC@                                     Bx�l�  �          @�ff���\@��H�xQ�� (�C����\@�p��1G���{C�
                                    Bx�l��  �          @������@�Q쿏\)��CL����@�G��8Q����Cz�                                    Bx�l�T  "          @�z���(�@��׿(����
=C����(�@��R�������C��                                    Bx�l��  "          @�=q���
@��;�{�7�B������
@�ff�(����\C                                    Bx�l�  "          @�\)��Q�@���>��?��
B�33��Q�@����p��nffC 
                                    Bx�l�F  "          @�(��l(�@��=�\)?z�B��f�l(�@�p���=q����B�{                                    Bx�m�  T          @�33�^�R@��\=�G�?n{B���^�R@��׿�=q���HB��H                                    Bx�m�  �          @�Q��b�\@I����p���G�C
c��b�\@G��<(��CQ�                                    Bx�m8  
�          @����(�@Q녿!G���33C+���(�@7
=��(���C�{                                    Bx�m-�  �          @�Q��x��@Z�H?�(�A��HC
��x��@s33>�G�@�(�C�f                                    Bx�m<�  
�          @���^�R?�p�@:�HB33CY��^�R@7
=@�
Aģ�C�
                                    Bx�mK*  �          @����#�
@qG����H�n�\B�ff�#�
@HQ�����  CQ�                                    Bx�mY�  	�          @�  �+�@y����=q�S�
B�
=�+�@dz�˅���\B���                                    Bx�mhv  
�          @���(��@���W
=�
=B�q�(��@��\�ٙ����\B���                                    Bx�mw  �          @��
�G�@��H��{�x  B�.�G�@g��0  �33B�#�                                    Bx�m��  
�          @�{� ��@mp��2�\��HB�� ��@#33�xQ��H��C :�                                    Bx�m�h  
�          @�33��
=@�  �Dz���B�녿�
=@=p����\�J�
B�B�                                    Bx�m�  
�          @��H���@�����R��(�B�33���@xQ������%��B�                                     Bx�m��  
�          @����.�R@��������  B�R�.�R@��������Q�B��{                                    Bx�m�Z  T          @����"�\@���(Q���Q�B�q�"�\@�������&=qB�8R                                    Bx�m�   "          @�\)���@�  �Fff��z�B�z����@z=q���9  B���                                    Bx�mݦ  "          @�{��(�@�
=�l���Q�B����(�@\�������T��B�                                      Bx�m�L  
�          @�G���\@���G
=��\)B�����\@n�R���
�=��B�k�                                    Bx�m��  
�          @�\)��@����:=q����B�3��@r�\���5ffB���                                    Bx�n	�  "          @�Q���R@�(��U���B�aH���R@^�R�����G�B�p�                                    Bx�n>  �          @���33@�  �b�\��Bݮ��33@b�\��  �L�
B�=                                    Bx�n&�  �          @�\)�G�@��~{��\B�R�G�@Fff�����\=qB�=q                                    Bx�n5�  
Z          @�=q���H@����g��G�B�׿��H@S33��  �Q�
B�W
                                    Bx�nD0  �          @�G�����@���W
=��RB������@O\)��ff�L�
B�\                                    Bx�nR�  
�          @�=q��p�@��
�W
=�  B�k���p�@P  ��{�K��B�                                    Bx�na|  �          @���z�@����3�
���HB�{�z�@^{���4(�B���                                    Bx�np"  
�          @�(��\)@���=p����HB��H�\)@L(���  �9��B��                                    Bx�n~�  T          @���33@����W���B�8R�33@;����\�P�B��H                                    Bx�n�n  �          @�\)=�G�@U���
=�j��B�#�=�G�?�����Q��=B��=                                    Bx�n�  
�          @�(�?n{@E���G��s��B�aH?n{?��
��\)�)B?
=                                    Bx�n��  
�          @���?�@P  ��(��n�B�
=?�?��R��(��HB�k�                                    Bx�n�`  �          @��>���@U��ff�i=qB�33>���?���Ǯ�B��=                                    Bx�n�  �          @�p�>�p�@G
=��33�r��B���>�p�?��������B�#�                                    Bx�n֬  
�          @Ǯ��Q�@]p�����`��B�#׽�Q�?�����Q�\B�Ǯ                                    Bx�n�R  �          @��H��@>�R�����qz�BŨ���?�{��\)L�B�aH                                    Bx�n��  T          @����p�@$z���(�p�B�(���p�?333��p�¤ǮB�                                    Bx�o�  T          @��
��p�@!G�����u�Bę���p�?0������¤��B��                                    Bx�oD  T          @��R��{@1G����\�WB�녿�{?�33��
=�CO\                                    Bx�o�  "          @�=q��z�@,������cp�B�G���z�?������  C\                                    Bx�o.�  
�          @��ÿ��@�H���\�t{B�B����?@  ��33p�C�                                    Bx�o=6  
�          @������@%����e�B�����?�  ��z�Q�C�{                                    Bx�oK�  T          @�(���33@
=��  �Z{C ٚ��33?aG�������C
=                                    Bx�oZ�  �          @�p���@������QffC  ��?\(������zp�C�H                                    Bx�oi(  
�          @�G��J=q?�ff�����@G�Cc��J=q>�
=����Xz�C,ff                                    Bx�ow�  "          @��H�^�R?�
=�z=q�1�
C�^�R?������J��C*J=                                    Bx�o�t  �          @��H�i��@\)�XQ����C�R�i��?���\)�7�\C xR                                    Bx�o�  "          @����|(�@%�8����C���|(�?�ff�c33�   C�=                                    Bx�o��  	�          @��U@dz�����Ə\C{�U@+��QG��\)C=q                                    Bx�o�f  �          @���:�H@`�׾k��7�C��:�H@P�׿������
Cٚ                                    Bx�o�  �          @���J�H@&ff@#�
B {C�H�J�H@QG�?�\)A���C
                                    Bx�oϲ  T          @����QG�@\��?
=@�\Cu��QG�@]p���\���C\)                                    Bx�o�X  T          @�33�8��@aG�=�\)?c�
CO\�8��@W����
�P��C�\                                    Bx�o��  �          @�z��(��@p  �&ff��B�G��(��@W
=�����\)C !H                                    Bx�o��  �          @�G���@l(��У����B���@@  �,(��z�B�=q                                    Bx�p
J  
�          @��
��\@k�������(�B�����\@;��9���ffB��                                    Bx�p�  
�          @�z���
@Z=q���R���HB�(���
@2�\�p��	{C�H                                    Bx�p'�  �          @��	��@E��������B����	��@Q��+���C.                                    Bx�p6<  �          @�Q��,��@O\)�����C�
�,��@Dzῃ�
�aG�CW
                                    Bx�pD�  "          @�p��Q�@P�׿��
�d��B�8R�Q�@2�\��p���RCu�                                    Bx�pS�  �          @�{�)��@L�ͽ#�
��C���)��@B�\��G��a�C{                                    Bx�pb.  f          @��
�,(�@A논����C�
�,(�@7��s33�Z{C{                                    Bx�pp�  �          @��\�B�\@U�����G�CB��B�\@@  ��=q���HCT{                                    Bx�pz  
�          @�=q�@��@Z�H���Ϳ��
CG��@��@N�R�����f�\C�                                    Bx�p�   �          @����>�R@W
=��G�����C���>�R@J�H��\)�f�RC:�                                    Bx�p��  �          @���8Q�@\(���  �L(�C��8Q�@L�Ϳ��\��  C�                                    Bx�p�l  �          @����:�H@P�׾�33���
CǮ�:�H@@  �������
C!H                                    Bx�p�  �          @�
=�1�@C33?�\@�CT{�1�@C�
��
=��33C33                                    Bx�pȸ  �          @����>{@N�R=�?�=qC�{�>{@G��Y���2�HC�
                                    Bx�p�^  �          @��H�>{@Fff>#�
@��C�
�>{@@�׿E��$��C�3                                    Bx�p�  �          @��R�1G�@E?�\@�ffC���1G�@Fff��G���  C�=                                    Bx�p��  �          @��H�>�R@G���
=���HC���>�R@6ff��=q����CB�                                    Bx�qP  �          @��HQ�@Tz�?xQ�A?33CB��HQ�@]p��u�E�C)                                    Bx�q�  �          @�{�:�H@Y��?��RAv�\C���:�H@g�>.{@	��C ��                                    Bx�q �  �          @�z��/\)@P��?�\A�=qC��/\)@h��?:�HA��B��                                    Bx�q/B  �          @���6ff@L(�?�
=A�=qC�R�6ff@g�?h��A1�C .                                    Bx�q=�  �          @���,(�@HQ�@�A�C�f�,(�@k�?��A�(�B�W
                                    Bx�qL�  �          @���4z�@B�\@G�A�p�C��4z�@dz�?��
A{�C \)                                    Bx�q[4  �          @���'
=@Fff@\)A�C#��'
=@g�?��RAxz�B��                                    Bx�qi�  �          @����n�R@*�H?���A]�Cc��n�R@8��>��R@mp�CE                                    Bx�qx�  �          @�\)�[�@7
=?�G�A�C!H�[�@K�?�R@�Q�C	(�                                    Bx�q�&  �          @��R�XQ�@;�?�(�A��C�XQ�@N�R?��@׮C@                                     Bx�q��  "          @����A�@=p�?�{A��C���A�@W�?k�A8  C��                                    Bx�q�r  �          @��
�(Q�@I��@Q�AۮC���(Q�@hQ�?���Ac33B��f                                    Bx�q�  "          @�  �<(�@N�R?���A���CQ��<(�@g�?O\)A�\C�                                    Bx�q��  �          @�{�U�@6ff>�@���Cu��U�@7
=��p����
CT{                                    Bx�q�d  �          @�33�Fff@P  ?k�A8��C�)�Fff@XQ�L�Ϳ&ffC�                                    Bx�q�
  �          @���)��@aG�?��@�z�B��)�)��@a녾�����B��3                                    Bx�q��  �          @���:=q@G�?�ffA^=qC\�:=q@R�\=�?�\)C��                                    Bx�q�V  �          @����:�H@I������\(�C���:�H@<�Ϳ���v=qC�3                                    Bx�r
�  �          @��H���@\��?333A�\B������@`�׾������B��f                                    Bx�r�  �          @���,��@G�>k�@P  C�
�,��@C�
�(���=qCff                                    Bx�r(H  �          @�  �333@C�
�L���1�CxR�333@8Q쿇��k\)C5�                                    Bx�r6�  �          @��
�A�@<(�>aG�@?\)C���A�@8�ÿ(��C\)                                    Bx�rE�  T          @�(��<��@U�?���Ar�\C���<��@a�>k�@333C޸                                    Bx�rT:  �          @�  �9��@k�?z�@ᙚC 0��9��@l�;�����C �                                    Bx�rb�  �          @����4z�@^�R>�p�@�p�C  �4z�@\(���R���CJ=                                    Bx�rq�  �          @�Q��'�@tz�?��AJ�HB��'�@}p��u�:�HB��)                                    Bx�r�,  �          @��'
=@w
=>��@Mp�B�=q�'
=@q녿Q��!p�B�W
                                    Bx�r��  �          @�=q��R@p  ��Q����B�\��R@`  �������\B���                                    Bx�r�x  �          @����@p  �   ��33B�Q���@]p���G���z�B�L�                                    Bx�r�  �          @�{��(�@|(���G���
B�
=��(�@\���33���B�z�                                    Bx�r��  �          @��\�У�@l�Ϳ�p����HB�k��У�@N�R�(���(�B�                                    Bx�r�j  T          @�G���z�@]p��1G��(�B�����z�@$z��g
=�YG�B��                                    Bx�r�  �          @��H��@j=q���33Bŏ\��@8Q��QG��@�B�=q                                    Bx�r�  �          @��H�5@Q��E��*(�B�k��5@z��vff�gG�B�
=                                    Bx�r�\  �          @�{�Y��@vff��=q���HB��
�Y��@Vff�z���B�p�                                    Bx�s  T          @���z�H@����:�H�   B�LͿz�H@l�Ϳ�ff�ə�B�Ǯ                                    Bx�s�  �          @����\)@z=q���H��z�B�녿�\)@hQ���
��
=B�8R                                    Bx�s!N  �          @����=q@i����  �a�B�Ǯ��=q@O\)��Q���{B��
                                    Bx�s/�  �          @�z��Q�@`�׿�  �`��B��Ϳ�Q�@G���z���B���                                    Bx�s>�  T          @��H��@i���.{�p�B�=q��@U�У�����B�B�                                    Bx�sM@  T          @�=q��{@n{������\)B���{@`�׿��
����B�33                                    Bx�s[�  T          @���z�@s�
�������\B����z�@e�����33B�3                                    Bx�sj�  �          @����z�@n�R>�?���B���z�@hQ�\(��C33B�.                                    Bx�sy2  �          @��
��=q@p  >���@�Bី��=q@n{����
=B��f                                    Bx�s��  �          @�����@S33�.{��B����@@�׿��
���C ��                                    Bx�s�~  �          @��
� ��@_\)����\)B���� ��@P  �������\B�                                     Bx�s�$  �          @���ff@Tz�Q��6=qB��R�ff@?\)��z����\C =q                                    Bx�s��  �          @��
�
=q@^�R��G����B��
=q@P  ��=q���\B�L�                                    Bx�s�p  �          @�(����@j=q��\)�|��B�q���@]p���p���Q�B�k�                                    Bx�s�  �          @����33@e�\)��B� �33@[������n�HB�                                    Bx�s߼  �          @��   @i����\)��  B�\)�   @`�׿��\�a�B�Q�                                    Bx�s�b  T          @�p���@o\)�L�Ϳ333B�G���@fff��G��ap�B�                                    Bx�s�  T          @�ff��\)@p  ���Ϳ�ffB�{��\)@fff����iB���                                    Bx�t�  �          @���-p�@S33�����33C\)�-p�@C33�������C��                                    Bx�tT  �          @�\)��R@Z=q��p����\B���R@L�Ϳ�p����B��                                     Bx�t(�  �          @��
�Q�@Vff�����HB��3�Q�@N�R�c�
�G�
B��                                    Bx�t7�  �          @�ff�@_\)<#�
>�B����@XQ�aG��A�B�\)                                    Bx�tFF  �          @�  �Q�@`��>�?�(�B�.�Q�@[��E��&�RB�u�                                    Bx�tT�  �          @�Q��#33@XQ�>�p�@�{B����#33@W
=�   �أ�B�B�                                    Bx�tc�  �          @����@_\)>�33@�  B�����@]p��
=q��Q�B�W
                                    Bx�tr8  �          @���� ��@\��>�\)@p��B�{� ��@Z=q���� (�B��R                                    Bx�t��  �          @�\)�!�@W
=>\@�p�B���!�@U�����B�L�                                    Bx�t��  �          @����#33@P  >�z�@\)C &f�#33@N{����z�C k�                                    Bx�t�*  �          @����!G�@G�>�\)@~{C �R�!G�@E��\��  C:�                                    Bx�t��  �          @�G��   @H��>�33@�C ���   @HQ��ff�ʏ\C ��                                    Bx�t�v  �          @�  �=q@HQ�?�A
=B�\�=q@K��aG��N�RB�L�                                    Bx�t�  �          @�z��'
=@G�?:�HA"{C���'
=@L�ͽ�\)�h��C.                                    Bx�t��  
�          @��
���@P��?5A\)B������@U��G�����B��\                                    Bx�t�h  �          @����@[�?z�A (�B�8R��@]p���\)�vffB���                                    Bx�t�  �          @��H��@hQ�>\@�=qB�=q��@g�����B�z�                                    Bx�u�  �          @�G��Q�@j�H?!G�A�B�L��Q�@mp���\)�p��B�3                                    Bx�uZ  �          @��ÿ�Q�@l��?E�A&�RB�LͿ�Q�@q녾����B�\)                                    Bx�u"   �          @�����@u�>�  @Z=qB�8R���@qG��.{�z�B��
                                    Bx�u0�  �          @�\)����@{�=�\)?c�
Bߏ\����@u��c�
�AG�B���                                    Bx�u?L  �          @�����
=@u=L��?�RB�k���
=@n�R�aG��<��B��                                    Bx�uM�  �          @��
���@|(�<�>��B�G����@u��h���Ap�B�                                    Bx�u\�  �          @�z��33@}p�=��
?��B�G���33@w
=�^�R�6�\B�aH                                    Bx�uk>  �          @���Ǯ@����.{���B�{�Ǯ@y�������p��Bߞ�                                    Bx�uy�  �          @�{���@�Q����(�Bҏ\���@q녿���
=B�Q�                                    Bx�u��  �          @�33��@�Q�:�H�$  B�33��@n{��z�����B�33                                    Bx�u�0  �          @\)����@x�ÿL���9p�B�𤾙��@e�ٙ���B��                                    Bx�u��  �          @�  �#�
@z=q�L���9�B�Lͼ#�
@g
=�ٙ��ə�B�Q�                                    Bx�u�|  �          @~�R>��@s33�����\)B��>��@[��   ���B�                                    Bx�u�"  �          @�  ��\)@r�\��33���\B�(���\)@Z�H�   ���HB�ff                                    Bx�u��  �          @���=���@xQ�}p��e��B�33=���@b�\��\)�޸RB��                                    Bx�u�n  �          @��\>.{@w
=��G���ffB��
>.{@^{�Q����B�=q                                    Bx�u�  �          @��>.{@r�\�\����B���>.{@U�
=�{B�{                                    Bx�u��  �          @�z�=�\)@dz����z�B���=�\)@?\)�7
=�.�B�.                                    Bx�v`  �          @��u@hQ��z���33B��q�u@C�
�6ff�+�HB�{                                    Bx�v  �          @����Q�@j�H������=qB��R��Q�@H���(��� =qB�(�                                    Bx�v)�  
(          @���>�\)@hQ��(���{B�G�>�\)@H��� �����B��                                    Bx�v8R  �          @y��>�{@hQ쿰����{B�Q�>�{@N{�
�H��\B���                                    Bx�vF�  �          @\)>u@i���������B�W
>u@L���
=�G�B�B�                                    Bx�vU�  �          @�����H@_\)�ff��Q�B�
=���H@;��5�.��B��                                    Bx�vdD  �          @�33��z�@]p���� �\B��\��z�@7��9���4��B�z�                                    Bx�vr�  �          @�(����@X����
�  B�����@1��@���;z�B�33                                    Bx�v��  �          @�(����@P  �"�\�{B�#׾��@%�Mp��K�B�p�                                    Bx�v�6  �          @�G�>���@k��˅��ffB���>���@N�R����B�8R                                    Bx�v��  �          @�=#�
@Mp��(Q���\B��3=#�
@"�\�Q��Q�B�\)                                    Bx�v��  �          @�\)�#�
@E��8Q��,z�B�Lͽ#�
@
=�_\)�_�
B��3                                    Bx�v�(  
�          @���=�@AG��@���3G�B�W
=�@��fff�fp�B���                                    Bx�v��  �          @�p��#�
@S33�\)�Q�B�z�#�
@*=q�J=q�GG�B�Ǯ                                    Bx�v�t  �          @�z��\@S�
����B�ff��\@,���Fff�B{B�L�                                    Bx�v�  �          @��
=@U��p��  B�B��
=@-p��HQ��AB�                                    Bx�v��  �          @�  ���@`  ����B��ÿ��@9���C33�7Bɔ{                                    Bx�wf  �          @�ff�Ǯ@`�����\)B���Ǯ@;��?\)�5Q�B�(�                                    Bx�w  �          @��׿�@W
=�%��p�B��ÿ�@.{�O\)�E�HB�{                                    Bx�w"�  �          @�녿��@Y���%���B��H���@0���P  �D
=B�W
                                    Bx�w1X  �          @�=q�z�@U�*�H�BǸR�z�@,(��U��I�B�aH                                    Bx�w?�  �          @��\�0��@H���8���(
=B��ÿ0��@(��`  �X=qBӽq                                    Bx�wN�  �          @��ÿz�H@1��Fff�9��B�\�z�H@33�g��gz�B�W
                                    Bx�w]J  �          @��׿��@)���I���?p�B޸R���?��h���lQ�B���                                    Bx�wk�  �          @{��\)@`  ��G����B�uþ\)@B�\�{�{B�#�                                    Bx�wz�  �          @}p���p�@Q��
=q���B��쾽p�@0  �3�
�5=qB�G�                                    Bx�w�<  T          @~{��ff@E��H���BĊ=��ff@ ���AG��G(�B�L�                                    Bx�w��  T          @|(���ff@G
=�ff�{Bę���ff@#33�<���C(�B�8R                                    Bx�w��  �          @hQ��ff@AG����H��B���ff@"�\�#33�2��B�{                                    Bx�w�.  �          @j=q�\)@?\)��\�p�B�Q�\)@   �'��6��B�k�                                    Bx�w��  �          @�  �O\)@J�H���Q�B�Ǯ�O\)@(Q��8���9�RB�p�                                    Bx�w�z  T          @~�R��{@Mp�������B�(���{@.{�)���(G�B�p�                                    Bx�w�   �          @i������@L�Ϳ�����RB�녿���@7
=��Q��33B�(�                                    Bx�w��  �          @i���G�@Q녿�����{B��f�G�@:�H���G�B�
=                                    Bx�w�l  �          @e���=q@P�׿���G�B�����=q@9����
���B�Ǯ                                    Bx�x  �          @XQ�L��@E��������B�Q�L��@0  ��
=�  B�33                                    Bx�x�  �          @Y��<#�
@B�\��z���ffB�Ǯ<#�
@,(��   �\)B�                                    Bx�x*^  �          @XQ�
=q@L�ͿB�\�R�HB���
=q@>�R��33��{BȀ                                     Bx�x9  �          @Q녾�
=@Mp���ff��(�B�  ��
=@C33������B�Ǯ                                    Bx�xG�  T          @\(���{@U��0���9�B�����{@G��������
B�aH                                    Bx�xVP  T          @^�R�L��@G
=��=q���B��)�L��@1녿�
=��HB�#�                                    Bx�xd�  �          @\�ͽ���@J=q������B��q����@5�����B�(�                                    Bx�xs�  �          @Z=q=�@U��0���<(�B��q=�@HQ쿬����z�B�u�                                    Bx�x�B  �          @S�
���@6ff�\����B��
���@\)��
��RBɅ                                    Bx�x��  �          @c33�\(�@!G��z��$p�Bٙ��\(�@ ���1G��OQ�B�G�                                    Bx�x��  T          @y����\)@��.{�133B�\��\)?��I���W��B��\                                    Bx�x�4  �          @xQ쿚�H@&ff�&ff�){B�  ���H@��Dz��Qp�B�                                    Bx�x��  �          @mp����@<�Ϳ����(�Bۣ׿��@"�\��!p�B�p�                                    Bx�xˀ  �          @hQ�?8Q�@`�׾�
=��(�B�?8Q�@W
=��������B��q                                    Bx�x�&  �          @qG��k�@Vff�޸R��z�B�Ǯ�k�@<���
=�33B��
                                    Bx�x��  �          @y����Q�@XQ��z�����B����Q�@<(��!��"�B��H                                    Bx�x�r  �          @{����@G
=�33��B����@&ff�7
=�;�B�\                                    Bx�y  �          @��
�h��@A��(���\)BՊ=�h��@p��J�H�I�Bܳ3                                    Bx�y�  �          @��p��@Dz��)�����B�#׿p��@   �L(��G�B�G�                                    Bx�y#d  �          @��Ϳs33@R�\���	33B�LͿs33@1��:�H�3z�B���                                    Bx�y2
  �          @��\�L��@W
=�p���RB��L��@7��3�
�-z�B�{                                    Bx�y@�  �          @�녿.{@O\)���p�B˨��.{@.�R�<(��9Q�B��                                    Bx�yOV  �          @�  ��ff@\(����H�뙚B����ff@@  �%��!33B��                                    Bx�y]�  �          @z=q>��@k���Q���B�ff>��@XQ��\)��
=B�                                      Bx�yl�  �          @l��>8Q�@dz�p���k33B�G�>8Q�@U��˅�̏\B��)                                    Bx�y{H  �          @z�H��@l�Ϳ��
���RB����@X�ÿ�����\)B�u�                                    Bx�y��  �          @hQ�=�G�@\(�������Q�B�G�=�G�@K��ٙ���RB���                                    Bx�y��  �          @e>B�\@Z�H�����G�B��\>B�\@J=q��
=��G�B�
=                                    Bx�y�:  �          @g��u@W����
����B��u@Dz������B�Ǯ                                    Bx�y��  �          @g
=��G�@W
=�����B�����G�@C�
��z���33B���                                    Bx�yĆ  �          @dz�k�@Q녿�\)��z�B��k�@>{������RB�                                    Bx�y�,  �          @\�;#�
@I���������HB�녾#�
@6ff��Q���
B��{                                    Bx�y��  �          @W���@DzῬ����B�\��@1G�����	(�B�.                                    Bx�y�x  �          @XQ쾅�@;���33���B�{���@%��
=q�G�B�p�                                    Bx�y�  �          @_\)���
@G
=������ffB�Ǯ���
@1G��
=�33B��H                                    Bx�z�  �          @\(��W
=@9�����G�B�L;W
=@ �����+  B��\                                    Bx�zj  �          @\�ͽ���@<(�����33B��ý���@#�
��
�(G�B��\                                    Bx�z+  �          @X��=���@:�H���H��
=B�{=���@#�
�p��#
=B��=                                    Bx�z9�  �          @XQ�>��
@:�H������B���>��
@$z��
=q��B�                                    Bx�zH\  �          @P  >��
@5���
���B���>��
@!G�� ���G�B�W
                                    Bx�zW  �          @Mp�>���@3�
�\��B��{>���@\)���R���B��H                                    Bx�ze�  T          @N�R>�
=@0  �����\)B��>�
=@�H�ff�"{B�#�                                    Bx�ztN  �          @H��?�@�R��=q�
=B��q?�@��\)�6��B��q                                    Bx�z��  �          @I��?   @#33���
�	z�B���?   @������1=qB��                                    Bx�z��  �          @C�
?�@�R���R�#\)B�k�?�?����ff�JffB�Ǯ                                    Bx�z�@  �          @<��?O\)?��H���.�
B��?O\)?�����S�Bz�R                                    Bx�z��  �          @H��?��
@33����,G�B}�\?��
?���{�Op�Bh
=                                    Bx�z��  �          @R�\?p��@������(
=B���?p��?��#�
�Lp�B{�
                                    Bx�z�2  �          @O\)?�G�@  �ff�!��B��q?�G�?�{�p��Ep�Bu                                    Bx�z��  �          @G
=?Tz�@��Q��,�B�G�?Tz�?�(��p��Q
=B��=                                    Bx�z�~  �          @Q�>���@������#  B��>���@33�!��J
=B�#�                                    Bx�z�$  �          @S�
>��R@�H��R�)z�B�ff>��R@   �'
=�P��B�ff                                    Bx�{�  �          @_\)?z�@{�Q��,ffB��R?z�@��1G��R�B�=q                                    Bx�{p  �          @c33?   @p��   �3p�B��f?   @   �8Q��Y�RB�                                    Bx�{$  T          @g
=?L��@���$z��5�
B�.?L��?�
=�<(��Zz�B�                                      Bx�{2�  �          @a�?J=q@�!G��6G�B���?J=q?���8Q��Z�RB�Q�                                    Bx�{Ab  �          @mp�?0��@{�+��8�B���?0��?�p��C33�^  B��H                                    Bx�{P  �          @vff@�R?�(��&ff�+G�Bz�@�R?�  �7
=�@�RA��                                    Bx�{^�  �          @u@(�?�33�'
=�+�A�\)@(�?p���3�
�<=qA���                                    Bx�{mT  �          @vff@��?�
=� ���"��B�\@��?��R�0  �6�A�p�                                    Bx�{{�  �          @x��@%�?���"�\�#��A��@%�?s33�/\)�3Q�A���                                    Bx�{��  �          @~�R@&ff?����(Q��%A�33@&ff?�  �5��5��A��                                    Bx�{�F  �          @r�\@Q�?���-p��7p�Bp�@Q�?����;��K=qAծ                                    Bx�{��  �          @c�
?��?���%��=�
B"�\?��?����2�\�T33A�G�                                    Bx�{��  �          @w�@&ff?}p��,(��0G�A�(�@&ff?��4z��:�
A8��                                    Bx�{�8  T          @l��?��
?�ff�<(��T��B{?��
?O\)�G��g�RA�33                                    Bx�{��  �          @a�?��?����B�\�p�BDp�?��?O\)�N{B�H                                    Bx�{�  �          @_\)?c�
?�
=�G
=B�BTG�?c�
?(���P���\B��                                    Bx�{�*  �          @aG�?���?p���@  �j\)A�
=?���>�G��G
=�y=qA|                                      Bx�{��  �          @^{?�(�?����0���U��B�?�(�?+��:=q�fA��
                                    Bx�|v  �          @`  ?�ff?O\)�A��p�A܏\?�ff>��R�G��|�\A6ff                                    Bx�|  �          @hQ�?��?0���AG��a��A�Q�?��>L���E�j{@�{                                    Bx�|+�  �          @l��@
=q?�ff�333�E��Aϙ�@
=q?
=�;��R�\At��                                    Bx�|:h  �          @j�H@	��?�Q��.{�?A���@	��?=p��8Q��N��A��                                    Bx�|I  �          @u�@G�?�=q�2�\�:��A��@G�?^�R�=p��Jz�A�ff                                    Bx�|W�  �          @p  @��?��\�1G��@33A��@��?Q��;��P
=A���                                    Bx�|fZ  �          @��
@{@��2�\�+p�B)ff@{?Ǯ�E��B�Bp�                                    Bx�|u   �          @�{@  @ff�5��*�B,�@  ?У��HQ��A��B�R                                    Bx�|��  �          @�
=@ff?����9���.  B�
@ff?�p��J�H�C�B �                                    Bx�|�L  �          @���@:=q?�  �P  �:33A�(�@:=q>��H�W
=�C
=A�                                    Bx�|��  �          @���@K�>��
�e��@��@�G�@K��u�e�Ap�C��H                                    Bx�|��  �          @�p�@>�R>�ff�b�\�Fp�A
�\@>�R�����dz��H�\C�                                    Bx�|�>  �          @��\@,��?����aG��Ip�A�  @,��?���i���S��A8Q�                                    Bx�|��  �          @��@$z�?�z��K��8p�B=q@$z�?�33�X���IQ�A���                                    Bx�|ۊ  �          @�G�@\)@ff�333�"��B 33@\)?���E��7�B��                                    Bx�|�0  �          @s33@�?�\)�+��4B�@�?�Q��9���H�A�33                                    Bx�|��  �          @q�@�
?޸R�(Q��1=qB @�
?����7
=�F
=B�                                    Bx�}|  �          @��@@��"�\�Q�B&z�@?��5��2
=B�                                    Bx�}"  �          @n{@\)@{�˅��  B?��@\)@p���Q���{B2�                                    Bx�}$�  �          @s33@33@
=�(��Q�B)�@33?�  ��R�"z�B                                      Bx�}3n  �          @tz�@
�H@  �����B833@
�H?�z���� p�B%\)                                    Bx�}B  �          @dz�?�33@녿�Q��\)BH�?�33?�(��  ��HB7�
                                    Bx�}P�  T          @G
=?���@   ��{���B��q?���@G���(��{Bx��                                    Bx�}_`  �          @K�>aG�@I����p���B�\>aG�@C�
�Tz��r�RB���                                    Bx�}n  �          @P  >�  @Mp��#�
�6ffB�  >�  @I���!G��4��B���                                    Bx�}|�  �          @I��?O\)@>{��G���
B���?O\)@:�H����#�B�#�                                    Bx�}�R  �          @Q�?^�R@7�>�AB�(�?^�R@:=q=u?�ffB��{                                    Bx�}��  �          @S�
=�Q�@7�?�ffA�B�B�=�Q�@C�
?�\)A��\B��                                     Bx�}��  �          @hQ�>���@U�?�ffA��\B��>���@\��?��AQ�B���                                    Bx�}�D  �          @xQ�?O\)@n{?!G�A�B�z�?O\)@qG�=�Q�?���B���                                    Bx�}��  �          @p  ?��@e��L�ͿW
=B���?��@a녿����B�8R                                    Bx�}Ԑ  �          @r�\?�\@j�H�:�H�3\)B�(�?�\@a녿�G���G�B��                                    Bx�}�6  �          @mp�?��
@_\)����B�?��
@X�ÿu�r=qB�{                                    Bx�}��  �          @���?�ff@j�H��������B�#�?�ff@dz�n{�UB�
=                                    Bx�~ �  T          @y��?��@fff�.{� ��B�\?��@^{��Q����B���                                    Bx�~(  T          @�{?�G�@g���z���33B��?�G�@U�	������B���                                    Bx�~�  �          @�@ff@N{�\)� ��BWQ�@ff@G
=��G��iG�BSp�                                    Bx�~,t  �          @�Q�@333@Q녿��\�TQ�BE�H@333@Fff��(�����B?��                                    Bx�~;  �          @��@3�
@P�׿��
��{BE33@3�
@AG���(���Q�B<=q                                    Bx�~I�  �          @�Q�@J�H@%���R��p�B=q@J�H@�\������Bp�                                    Bx�~Xf  �          @�(�@@  @2�\�
=��  B+�
@@  @{��R�33B�                                    Bx�~g  �          @�G�@4z�@>�R��
=��(�B:G�@4z�@+�������B.33                                    Bx�~u�  �          @�Q�@7�@A녿����\B:
=@7�@1G��z����B/�                                    Bx�~�X  �          @��@9��@0�׿����33B.p�@9��@\)�
=q���B"z�                                    Bx�~��  �          @�@.{@p�����B(�@.{@ff�0����B�
                                    Bx�~��  �          @�G�@#�
@\)�$z��\)B$�H@#�
?�{�6ff�(�Bz�                                    Bx�~�J  �          @�p�@A�@5�������B,��@A�@'������îB#=q                                    Bx�~��  �          @���@E�@*=q>�?��B#\)@E�@)���k��P  B#{                                    Bx�~͖  �          @�  @^{@Q�=��
?��B	�H@^{@��u�Q�B	ff                                    Bx�~�<  �          @y��@J=q@�>���@�Q�B
��@J=q@{=���?�ffBG�                                    Bx�~��  �          @n�R@8Q�@=�?���B=q@8Q�@��L���E�B��                                    Bx�~��  �          @j=q@0  @��>#�
@%�B$��@0  @���#�
�!�B$��                                    Bx�.  �          @u@Fff@�׽�\)����Bz�@Fff@�R�\��z�B
=                                    Bx��  �          @u�@=p�@�<#�
>L��B\)@=p�@=q���R��p�Bp�                                    Bx�%z  �          @k�@,(�@\)��  �|(�B*�
@,(�@��z��B(=q                                    Bx�4   �          @n�R@�
@7�����  BLff@�
@4z���
=BJ��                                    Bx�B�  �          @~{@@Vff�8Q��%BhG�@@S33��R���Bf��                                    Bx�Ql  �          @�ff@=q@Y����p���z�BZ�R@=q@Tz�Q��5�BX(�                                    Bx�`  �          @~{@�@G��������HB\ff@�@<(������{BV
=                                    Bx�n�  �          @w�@@@�׿�  ��z�B]
=@@4z�У��Ǚ�BU��                                    Bx�}^  T          @w
=?�z�@:�H�У���  Bc33?�z�@+����R��BZ�                                    Bx��  �          @s�
@G�@'���ff��G�BD
=@G�@�ÿ����B9                                    Bx���  �          @���@=p�@*�H��{�xz�B(�@=p�@ �׿������B!33                                    Bx��P  �          @���@HQ�?�p�������B(�@HQ�?�G�����ӮA��H                                    Bx���  �          @���@#33@��Q����B�@#33?ٙ��(Q��"\)B�
                                    Bx�Ɯ  �          @�(�@C�
@
�H��(�����BG�@C�
?�
=���R��  B �H                                    Bx��B  �          @�(�@H��?�Q��=q��Q�A�\)@H��?�Q���
���A�                                      Bx���  �          @���@7
=@#33��33�ң�B&�
@7
=@��p����RBz�                                    Bx��  �          @�z�@\)@C33��{��BKff@\)@2�\��R���\BA=q                                    Bx��4  �          @�G�@�@b�\���\�]p�Bl�@�@XQ쿺�H����Bgp�                                    Bx���  �          @u@p�@8Q�n{�g33BQ�@p�@.�R�����p�BLG�                                    Bx���  �          @{�@;�@Q��G���\)B�@;�?���\��z�B{                                    Bx��-&  �          @x��@R�\?�33������
A�Q�@R�\?n{��\��\)A}                                    Bx��;�  T          @x��@N{>�=q�Q���
@�  @N{    �	���
=C��R                                    Bx��Jr  �          @|��@K�<��z���\?�@K������
��C���                                    Bx��Y  �          @p��@A�>������@!G�@A녾\)�����HC��                                    Bx��g�  �          @k�@6ff���
�G��p�C�Ф@6ff����p���HC��                                    Bx��vd  �          @j=q@C33��\)���R�(�C�ff@C33����Q�� ffC�!H                                    Bx���
  �          @Y��@{?��Ϳ˅��G�B�R@{?�녿��
� ffA���                                    Bx����  �          @X��@
=@z΅{��(�B1�R@
=?�׿�{��
=B'
=                                    Bx���V  �          @S�
?�@  ��ff���BL��?�@�\������BA��                                    Bx����  �          @a�?�\)@1녿}p���=qB`p�?�\)@(�ÿ�=q����BZ                                    Bx����  �          @a�?��\@E��:�H�F{B�33?��\@=p�������
=B��                                    Bx���H  �          @mp�?�@i������ffB�#�?�@g�������B���                                    Bx����  �          @fff>aG�@_\)>��@�{B��3>aG�@a�=#�
?333B�Ǯ                                    Bx���  �          @fff<��
@]p�?k�AmG�B�z�<��
@b�\>��H@��HB��                                     Bx���:  �          @{���{@n{?�33A���B�L;�{@u�?0��A"�HB�                                      Bx���  �          @i��@L(�?�(���(����A��
@L(�?��ÿ�{��z�A��
                                    Bx���  �          @p  @Z�H=u���H��
=?��@Z�H��G����H���\C��                                    Bx��&,  �          @p  @\(��k���  �{33C���@\(����
�aG��]�C���                                    Bx��4�  T          @���@Z=q?����33A�{@Z=q?��Ϳ(���&=qA��                                    Bx��Cx  �          @���@�  >�ff�k��333@�@�  >�
=��z��[�@��                                    Bx��R  �          @���@��(�þ��
�x��C���@��333�u�9��C��                                    Bx��`�  �          @�(�@��H�\)���
�s33C���@��H�����  �?\)C�}q                                    Bx��oj  �          @��
@��\���H�#�
��C��@��\��\���Ϳ�(�C�H                                    Bx��~  �          @�p�?���@E�=��
?�Q�B��3?���@C�
��\)��ffB��\                                    Bx����  T          @�����H@|(�>\@�(�Bܔ{���H@}p���Q쿞�RB�k�                                    Bx���\  �          @�Q쿑�@�33<�>�p�B�����@�=q�����B�L�                                    Bx���  �          @�녿�\)@{�>u@O\)B��f��\)@{��aG��@  B��H                                    Bx����  �          @�=q����@�  ?�@�(�B��3����@�G�<#�
>.{B���                                    Bx���N  �          @���?&ff@�ff?E�A%�B�W
?&ff@�Q�>�\)@j�HB���                                    Bx����  �          @�33�#�
@}p�?�  A���B�G��#�
@�z�?��\A�
=B�B�                                    Bx���  �          @����5@B�\@!�B�HB�(��5@Tz�@
=qBB�{                                    Bx���@  �          @��H>�@�Q�=��
?��B�>�@�  �Ǯ��=qB���                                    Bx���  �          @p  ?�ff@U�G��DQ�B���?�ff@N�R��z���B�B�                                    Bx���  �          @~{>�@vff>�?��HB��>�@u���R��z�B�z�                                    Bx��2  �          @s33?+�@l(�>�@�33B�u�?+�@n{=L��?Q�B���                                    Bx��-�  �          @�  ?�@o\)>��H@�B�ff?�@q�=u?\(�B��                                    Bx��<~  �          @�z�?��R@tz�?G�A/�B��f?��R@xQ�>���@�ffB�u�                                    Bx��K$  T          @y��?�33@`  �(����B�u�?�33@Y���������B�L�                                    Bx��Y�  T          @�ff?�33@X�ÿ�����p�Br�
?�33@Mp���=q���HBm(�                                    Bx��hp  �          @�
=?���@h��?���A���B��?���@p��?fffAJ�RB�\)                                    Bx��w  �          @��R?�{@dz�?�A�Bz{?�{@k�?B�\A&ffB|�                                    Bx����  �          @�33@�
@qG�?��@�\)Bu��@�
@s�
=�G�?��RBv�\                                    Bx���b  �          @��R@
�H@c�
>W
=@<(�Bj��@
�H@c�
�B�\�'�Bj�                                    Bx���  �          @���@(Q�@U>�{@�z�BOG�@(Q�@W
=�#�
��BO�
                                    Bx����  T          @��R@�@c�
��Q쿣�
Bqff@�@a녾��H���Bpz�                                    Bx���T  �          @n{?�(�@HQ�<#�
=��
Bg(�?�(�@G
=��33��z�Bf�                                    Bx����  �          @L(�?��@%���{�ƸRB\��?��@!G��!G��7�
BZG�                                    Bx��ݠ  �          @hQ�?�(�@:�H���
=B`G�?�(�@5�c�
�h(�B]33                                    Bx���F  �          @�33?�\@e��h���L��B~?�\@\�Ϳ�ff��=qB{z�                                    Bx����  �          @u?�33@U��ff����B��?�33@L�Ϳ���
=B�ff                                    Bx��	�  �          @W�@��@�>u@�(�B.(�@��@�\���
��z�B.��                                    Bx��8  �          @��@?\)@I��>�G�@��HB:  @?\)@K�=�Q�?�Q�B;{                                    Bx��&�  �          @�z�@E�@z�H?��HAZ=qBOp�@E�@���?E�A	p�BR�                                    Bx��5�  �          @�{@H��@��?8Q�@�
=BQ=q@H��@��
>��@:�HBR�R                                    Bx��D*  �          @�@S�
@}p�>�@��HBH\)@S�
@~�R=#�
>�ffBI(�                                    Bx��R�  �          @���@hQ�@a�?\(�A�\B0�@hQ�@fff>��@��RB3
=                                    Bx��av  �          @��@fff@E?��
AC33B"p�@fff@K�?.{A ��B%�
                                    Bx��p  �          @��@o\)@c33?�{AB�\B-�@o\)@i��?5@���B1�                                    Bx��~�  �          @�33@hQ�@h��?�
=Axz�B4z�@hQ�@q�?��\A0Q�B8��                                    Bx���h  �          @�=q@vff@[�?�ffAa��B&@vff@c33?h��AB*�R                                    Bx���  �          @�\)@g
=@hQ�?��A:�HB4�@g
=@n{?&ff@�{B7��                                    Bx����  �          @��@mp�@j=q?��A5G�B2�@mp�@p  ?#�
@�z�B5\)                                    Bx���Z  �          @��R@e@e?��AJ�\B3�@e@l(�?=p�A
=B7(�                                    Bx���   �          @�(�@l��@\(�?L��AQ�B+��@l��@`��>�
=@��B-��                                    Bx��֦  T          @�p�@4z�@tz�?}p�A=�BVff@4z�@z=q?�@أ�BX�
                                    Bx���L  �          @�z�@*=q@\)?E�AffBa�@*=q@���>���@|��Bb�R                                    Bx����  �          @��R@���@�?��AJ�RAծ@���@��?c�
A�A�z�                                    Bx���  �          @��@�  @4z�?�\)A{33B��@�  @<��?��A@��B��                                    Bx��>  �          @��@U�@s�
?z�HA.�HBC\)@U�@x��?\)@ǮBE��                                    Bx���  �          @�Q�@#33@���?�  A.�RBr=q@#33@��?�\@�Q�Bs��                                    Bx��.�  �          @�p�@ ��@�ff?s33A!G�Bw��@ ��@���>�(�@��By(�                                    Bx��=0  �          @�33@(�@�  ?E�A�B��@(�@���>�\)@J=qB��=                                    Bx��K�  �          @�
=@'
=@�{?G�A	p�Bn(�@'
=@�  >���@P��Boz�                                    Bx��Z|  �          @��@(Q�@�
=?�\)AK�
Bh�@(Q�@�=q?(��@�Bj�
                                    Bx��i"  �          @��H@Dz�@g�?L��A��BF��@Dz�@l(�>��@�(�BH��                                    Bx��w�  T          @�\)@Dz�@n�R?��AB�HBJz�@Dz�@tz�?#�
@��BM�                                    Bx���n  T          @�=q@�@AG�?�(�ATz�BG�@�@HQ�?c�
A=qBQ�                                    Bx���  �          @�p�@n{@j=q?��HAz�HB1�@n{@r�\?���A5p�B6{                                    Bx����  �          @��\?�\@��?
=q@��B�G�?�\@�z�=���?��
B���                                    Bx���`  �          @���?(�@�>��@��RB�33?(�@�ff��\)�Y��B�G�                                    Bx���  �          @�>��@�?�=qA�(�B��>��@���?aG�A733B�p�                                    Bx��Ϭ  �          @�{?�ff@x��?��
A�{B�.?�ff@���?�{Ak\)B�ff                                    Bx���R  �          @�ff?��R@n{?���A�=qB�\?��R@y��?���A��
B���                                    Bx����  �          @�ff@�H@i��?�=qA�33Ba�R@�H@qG�?p��A<��Be�                                    Bx����  �          @�  @HQ�@hQ�?n{A.�HBD@HQ�@mp�?
=q@ə�BG(�                                    Bx��
D  �          @��@��@�\?���AK�A�  @��@��?Y��AG�A�z�                                    Bx���  �          @��@�  @@�׿   ��\)B�
@�  @<(��Q��BG�                                    Bx��'�  �          @��@�p�@(�>�  @,(�A�{@�p�@p�<��
>W
=A�33                                    Bx��66  �          @��@���@�\>B�\@��A�Q�@���@33������A�
=                                    Bx��D�  �          @���@�z�?�  ��R���HA�=q@�z�?�
=�E��  AxQ�                                    Bx��S�  �          @���@��?�=q�\��z�Ak�@��?��
������Ac�                                    Bx��b(  
�          @���@�@Q�=u?.{A�{@�@Q�#�
��=qAî                                    Bx��p�  T          @�Q�@�G�@
=q<#�
>�A�{@�G�@	���W
=�ffA�33                                    Bx��t  �          @��@���?Ǯ���R�UA�(�@���?\������A��R                                    Bx���  �          @�  @������ÿ�Q��^{C�f@�����ff��33�V=qC�K�                                    Bx����  T          @���@���@��>�  @;�A�(�@���@��        A�33                                    Bx���f  
�          @��
@��\@%�>k�@.{B33@��\@%�#�
�   B��                                    Bx���  �          @�33@�  ?�\)>���@c�
A�ff@�  ?�33>�?�
=A�Q�                                    Bx��Ȳ  �          @�ff@���?���>��
@UAM��@���?�p�>B�\?��RAQ�                                    Bx���X  �          @�{@���?�=�Q�?z�HAG33@���?������
AG�                                    Bx����  T          @�ff@��?W
=�k����A{@��?O\)���R�S33A	p�                                    Bx����  �          @�ff@���@?\)>8Q�?��B�
@���@@  �\)��
=B�H                                    Bx��J  �          @�33@�=q@H��>u@&ffB=q@�=q@I����Q�n{B�                                    Bx���  T          @��
@��\@^�R=�\)?G�B!�@��\@^{��z��G
=B!�\                                    Bx�� �  �          @�z�@�?�=q?�@��A�  @�?У�>���@�ffA�=q                                    Bx��/<  �          @��@�33��33?(�@���C��@�33��z�?&ff@�z�C�xR                                    Bx��=�  �          @���@���Ǯ>��
@\��C��@����33>�Q�@w
=C�                                      Bx��L�  �          @��\@���?&ff=�Q�?z�H@ᙚ@���?(��<��
>u@��
                                    Bx��[.  �          @�z�@���?��>��H@�{A��R@���?�=q>���@J=qA�                                      Bx��i�  �          @���@�  @	��?O\)A��A��H@�  @{?z�@�G�Aȣ�                                    Bx��xz  �          @��\@�p�@��?B�\A
=A�p�@�p�@!G�?   @�(�A�z�                                    Bx���   �          @�(�@���@)��?�\@�
=A��
@���@,(�>k�@�RA�R                                    Bx����  T          @��
@��@)��>��?���A���@��@)������{A��H                                    Bx���l  �          @�(�@��@
=>aG�@��A�
=@��@����
��  A��
                                    Bx���  �          @��@��
@*=q?
=q@�=qA��@��
@-p�>��@3�
A�
=                                    Bx����  �          @��H@��@ff?z�@���A��H@��@	��>�Q�@x��A���                                    Bx���^  �          @�33@�?�(�?z�@ȣ�A�{@�@G�>\@���A�{                                    Bx���  �          @��R@��\?��>��H@���A��\@��\?�>�z�@C33A�                                    Bx����  �          @�
=@��@�>�z�@A�A���@��@33=��
?L��A�z�                                    Bx���P  �          @�(�@��H@�R?0��@��HA�(�@��H@�\>�ff@���A���                                    Bx��
�  �          @�
=@��H?���>#�
?�A�p�@��H?����#�
���A��                                    Bx���  �          @��H@�\)?��H������A��@�\)?�
=��33�g�A��
                                    Bx��(B  �          @��@��H?��=�\)?333A�
=@��H?�׾����
A���                                    Bx��6�  �          @��\@���@�
?�R@���A��H@���@�>���@���A���                                    Bx��E�  �          @��@���@   ?fffA�Aۮ@���@%�?#�
@�A�                                    Bx��T4  �          @�@��R@\)?��
A/�
Aޣ�@��R@%�?E�AffA�                                    Bx��b�  �          @�\)@�G�@��?���A=�A�@�G�@\)?^�RA�\Aۮ                                    Bx��q�  �          @�ff@��H@(�?��HAM��A�z�@��H@33?z�HA%p�A�p�                                    Bx���&  �          @�G�@�p�@�?Tz�A  Aԏ\@�p�@=q?z�@���A�Q�                                    Bx����  �          @�ff@��H@1�?#�
@�=qBz�@��H@5�>�33@tz�Bff                                    Bx���r  �          @���@��@B�\?k�A\)Bz�@��@G
=?
=@�{B=q                                    Bx���  �          @��\@��\@6ff?��RAV�RB��@��\@>{?n{A!G�B	�R                                    Bx����  �          @��H@��@>�R?�p�A��RBff@��@G�?�z�AH��B\)                                    Bx���d  �          @��@~{@Vff?�(�A}p�B ��@~{@_\)?�\)A>�HB%G�                                    Bx���
  �          @�z�@���@K�?У�A���B{@���@U?��A]��BQ�                                    Bx���  �          @�z�@{�@[�?�ffA`Q�B$=q@{�@b�\?p��A ��B({                                    Bx���V  �          @�z�@k�@g�?�ffA�B2(�@k�@p��?�z�AG�B6�\                                    Bx���  '          @�(�@q�@a�?��Ap��B,(�@q�@j=q?��\A.�RB033                                    Bx���  �          @��\@�z�@C�
?��RAYp�B(�@�z�@K�?k�A�B(�                                    Bx��!H  
Z          @���@q�@Y��?���A~�RB'��@q�@a�?��A=��B,                                      Bx��/�            @��R@~�R@HQ�?�G�A`z�B@~�R@O\)?n{A$(�B��                                    Bx��>�  
_          @�z�@HQ�@Y��?�=qA�33B=G�@HQ�@c33?�(�Af�RBB�                                    Bx��M:  �          @�
=@^{@b�\?ǮA�\)B6{@^{@k�?�
=AQ�B:��                                    Bx��[�  �          @���@hQ�@L��?�\)A|(�B%��@hQ�@U�?��\A;�B*                                      Bx��j�  �          @�33@,(�@g�?��
A�=qBU��@,(�@r�\?��A��BZ�\                                    Bx��y,  
Z          @�p�?�  @w�?�z�A���B�ff?�  @���?��RA�\)B�Q�                                    Bx����  �          @��\=���@o\)?���A�(�B��=���@z�H?��A��
B�{                                    Bx���x  "          @�\)�c�
@e@ ��A�=qB��c�
@q�?�\)A�33BΏ\                                    Bx���  
�          @����  @U�?�G�A�=qB�\��  @`  ?�33A��B�G�                                    Bx����  Y          @����xQ�@L(�@#�
B=qBոR�xQ�@\(�@p�A��B�\)                                    Bx���j  
Z          @��R@Mp�@1G�?�
=A��B#33@Mp�@:=q?���Aj�RB(��                                    Bx���  T          @�(�@���?��H������=qA�ff@���?�z�\)��  A��                                    Bx��߶  
�          @��@�p�?�=q�W
=�!A�33@�p�?�p���  �A��A�(�                                    Bx���\  
�          @��H@��R?�33�!G���A�Q�@��R?��ÿL���ffA��                                    Bx���  "          @��@��?�
=�8Q���33A}�@��?��Ϳ^�R��An�H                                    Bx���  
�          @��@��
?��
����A`��@��
?�(��(���(�AV�R                                    Bx��N  "          @��
@�ff?��R�
=q��=qAV=q@�ff?��+���ffAK\)                                    Bx��(�  T          @�(�@��?������Q�A<z�@��?��
�!G���
=A2=q                                    Bx��7�  �          @�\)@��
?O\)�#�
��ffA��@��
?:�H�8Q���=q@���                                    Bx��F@  
�          @��
@��H=�G���(���z�?�33@��H=�\)��G���\)?.{                                    Bx��T�  
�          @�  @�p�?��
?���AL��A4��@�p�?�z�?�=qA8z�AI��                                    Bx��c�  
�          @���@�  ?�?�p�A��A��@�  ?���?���Al��A�Q�                                    Bx��r2  
Z          @��R@�=q?��?��
Ahz�AB{@�=q?�?�z�AR�\AY                                    Bx����  �          @��H@�\)@�R?�(�A�{A��@�\)@(Q�?�Q�AO\)A���                                    Bx���~  �          @��@���@:=q?�{Ak33B	(�@���@B�\?��
A2�HB�
                                    Bx���$  �          @��@��\@2�\>�
=@��A�=q@��\@4z�=�?��HA�Q�                                    Bx����  T          @�@���@   >�=q@-p�A��@���@G�=L��?�\A�
=                                    Bx���p  "          @�{@�?�\)>��@���A���@�?�z�>���@?\)A�=q                                    Bx���  �          @��R@��
?^�R?�@�33A��@��
?k�>�G�@��
A��                                    Bx��ؼ  �          @��@��>�?�\@�\)@���@��?�\>�@�{@�{                                    Bx���b  
�          @��
@��\=�Q�?z�@��?u@��\>#�
?\)@���?���                                    Bx���  �          @�Q�@�ff�\)?B�\@�{C�>�@�ff�u?E�A ��C��\                                    Bx���  
�          @�33@�
=��Q�?��A3\)C�#�@�
=�u?�\)A8��C���                                    Bx��T  T          @�p�@�ff��ff?�p�Ar�RC���@�ff��\)?\Ayp�C���                                    Bx��!�  "          @��H@�녾��?У�A��\C��H@�녾k�?�A�p�C��\                                    Bx��0�  
(          @��
@n{?�G�?�\)A�A�ff@n{?�
=?���A�{A��
                                    Bx��?F  
�          @�z���@p  �@  �
=B����@h�ÿ�z��v=qB�W
                                    Bx��M�  �          @�z��I��@��
�=p���B��\�I��@�Q쿞�R�K�B��
                                    Bx��\�  O          @��
�7�@�\)��\)�EB�aH�7�@�p��=p����B�{                                    Bx��k8  
-          @����{@�{�(����B�p���{@��H�����QG�B�W
                                    Bx��y�  �          @�  �ٙ�@�p��8Q��(�B�33�ٙ�@��
�8Q���{Bڏ\                                    Bx����  	�          @��\���@�(��#�
��G�B�u����@��H�z��ǮB�q                                    Bx���*  �          @��p�@�p�>�(�@���B�W
�p�@�{�����RB�33                                    Bx����  
�          @�����p�@�p�?0��@�B�
=��p�@��R=�G�?�33B�                                    Bx���v  
Z          @��\��z�@���?W
=A�B��H��z�@��\>�  @%�Bя\                                    Bx���  "          @����@�z�?��
A��\B�B���@���?z�HA%��B�33                                    Bx����  "          @��?L��@���@��Aՙ�B��
?L��@���?��HA�G�B��
                                    Bx���h  
�          @�=q@�@L(�@R�\B (�B\)@�@b�\@:=qA�(�B �                                    Bx���  
�          @��H@���?��@H��A��A��
@���@�R@:=qAޏ\A�=q                                    Bx����  
�          @�(�@������
@^�RB(�C��{@���>��R@^{B�\@W�                                    Bx��Z  
�          @�  @���    @7
=A�
=    @���>��
@5Aٙ�@U�                                    