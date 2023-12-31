CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230409000000_e20230409235959_p20230410021720_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-10T02:17:20.279Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-09T00:00:00.000Z   time_coverage_end         2023-04-09T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxv5    "          @��H@{�>�{�����@���@{�?�zῡG���Q�A��                                    Bxv5�  �          @���@��=��Ϳ�����
?��@��?aG���G���G�AC�
                                    Bxv5L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxv5+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxv5:�  	          @��\@n�R�&ff>��R@vffC���@n�R�Q쿋��\z�C��3                                    Bxv5I>  �          @�
=@U��/\)����
=C�AH@U��33��33�ϮC�(�                                    Bxv5W�            @���@\�Ϳ�(��z���
=C�xR@\�;B�\�!���
C�t{                                    Bxv5f�  
�          @�\)@\(��.�R�
=q�߮C�� @\(���
����ȏ\C��f                                    Bxv5u0  �          @��@I���(�>#�
@ffC��@I������\)���HC��                                    Bxv5��  T          @��
@Mp��J=q=�Q�?���C���@Mp��0  �Ǯ���
C��{                                    Bxv5�|  "          @��\@C33�Fff��p��}G�C�<)@C33�G��*=q��C�:�                                    Bxv5�"  
�          @�{@]p��G
=>�33@��C�H@]p��6ff�����=qC�E                                    Bxv5��  �          @�p�@W
=�I��?
=q@�\)C�c�@W
=�>�R��z��hQ�C�1�                                    Bxv5�n  �          @��@E��E�?�33A���C�xR@E��W��Ǯ��z�C�8R                                    Bxv5�  �          @���@2�\�Z=q?(�@��C��f@2�\�O\)��p�����C�b�                                    Bxv5ۺ  5          @�G�@녿�@!�B!\)C���@��<��?�
=A�ffC��q                                    Bxv5�`  
�          @�{@:=q�	��@ ��A�=qC���@:=q�8Q�?(�AQ�C���                                    Bxv5�  T          @�ff@,���Q�@A���C�Ff@,���G�?\)@�Q�C�o\                                    Bxv6�  "          @���@���e?#�
A
ffC�� @���Z=q��ff��(�C�l�                                    Bxv6R  T          @�ff?��`�׿�z���33C���?���\�@���;=qC��                                    Bxv6$�  
�          @�  ?�(��\�Ϳ�  ��  C��q?�(���
�6ff�,��C�7
                                    Bxv63�            @��R?�\)�h�ÿ^�R�@��C���?�\)�+��'��Q�C�q�                                    Bxv6BD  �          @���@+��>{?�ffA��C�f@+��N{�����z�C��f                                    Bxv6P�  �          @�G�@333�#�
?���A�G�C��@333�L��>�Q�@���C��H                                    Bxv6_�  
�          @���@*=q�5?��HA���C���@*=q�Tz�<#�
=�Q�C�`                                     Bxv6n6  "          @��@��,��@G�A���C��{@��W
=>�33@�G�C���                                    Bxv6|�  �          @�@	���G�?ٙ�A��C�>�@	���c33���� ��C���                                    Bxv6��  �          @�{?��R�Vff?�p�A�ffC�W
?��R�hQ��ff��C�U�                                    Bxv6�(  "          @���@Q��@  @	��A�\C���@Q��k�>��R@���C��                                    Bxv6��  �          @�  @1G���@ffB�HC�H�@1G��A�?n{AL��C�=q                                    Bxv6�t  �          @���@4z���H@*�HB  C�c�@4z��5?�A�=qC�k�                                    Bxv6�  
Z          @��H@4zῬ��@@��B/��C�=q@4z��,(�?��A�33C�%                                    Bxv6��  T          @�33@0�׿&ff@Q�BD�
C�Q�@0���\)@\)B�HC�o\                                    Bxv6�f  
Z          @��\@,(��(��@Tz�BHG�C�3@,(��G�@ ��B�HC��
                                    Bxv6�  T          @���@(�ÿL��@Q�BG�C���@(����@=qB�C��                                    Bxv7 �  
�          @�=q@)����(�@A�B3�C���@)���3�
?�{A�33C��\                                    Bxv7X  
(          @��@-p����@'
=B
=C�j=@-p��N�R?�\)Amp�C��q                                    Bxv7�  
�          @�z�@1G��)��@33A�p�C�!H@1G��U�>��@�ffC���                                    Bxv7,�  �          @�p�@H���@p�A��C�&f@H���;�?W
=A0(�C�}q                                    Bxv7;J  T          @�ff@7
=�&ff@p�A��
C��
@7
=�W
=?�@�(�C�,�                                    Bxv7I�  �          @��
@;��6ff?ǮA�\)C��@;��O\)��G���z�C��                                    Bxv7X�  �          @���@!G��C�
?�Q�A���C���@!G��`  ��G���C��q                                    Bxv7g<  
�          @��H@(���Mp�?��\A��
C��@(���Z=q�
=q��C��R                                    Bxv7u�  �          @�=q@.�R�C33?���A��C��\@.�R�S�
�Ǯ���RC��                                    Bxv7��  "          @�33@.{��H@�B�C�+�@.{�P��?G�A&=qC��                                     Bxv7�.  
�          @��@(���:�H?�Q�A�  C��@(���`��>B�\@(�C�}q                                    Bxv7��  T          @�(�@+��%@33A��C��q@+��X��?+�Az�C�                                      Bxv7�z  T          @��@Q�� ��@.{B�
C���@Q��c33?���AaG�C��                                    Bxv7�   �          @��@-p����H@0��Bp�C��@-p��E?��A�p�C���                                    Bxv7��  g          @��\@@�׿��R@Q�A�=qC�K�@@���2�\?Tz�A6{C���                                    Bxv7�l            @���@L(��G�?޸RA��
C�K�@L(��5>�33@��RC�+�                                    Bxv7�  �          @�@@����?�(�A�C�g�@@���5>���@�Q�C�U�                                    Bxv7��  
�          @���@#�
��G�@7�B0  C�� @#�
�0  ?��HA��
C�u�                                    Bxv8^  T          @�(�@�H����@<��B6��C�xR@�H�6ff?�  A�
=C�#�                                    Bxv8  T          @�@\)��{@:�HB4  C��@\)�E�?���A�p�C��                                    Bxv8%�  �          @�?˅�(��@J=qB6�
C��f?˅�xQ�?��A��C�'�                                    Bxv84P  
(          @�p�?���-p�@HQ�B4�C���?���z�H?��A���C��{                                    Bxv8B�  
�          @���@z��\)@>�RB*��C���@z��\(�?�Q�A�
=C���                                    Bxv8Q�  
�          @���@#�
�z�@.{B��C���@#�
�XQ�?�
=Axz�C��{                                    Bxv8`B  
�          @��@I����=q@+�BffC�}q@I����\?�\AîC��q                                    Bxv8n�  "          @�
=?����
=q@\��BK  C�R?����fff?�33AɅC�h�                                    Bxv8}�  T          @��?�(��
=q@c33BPz�C�Ff?�(��i��?�p�A�
=C��q                                    Bxv8�4  �          @�\)?���@b�\BP��C�J=?���e@ ��A�\)C�>�                                    Bxv8��  �          @��R?˅��@g
=BX(�C���?˅�g
=@z�A��C��f                                    Bxv8��  T          @�\)?�33��@g�BXp�C���?�33�e�@
=A�G�C�c�                                    Bxv8�&  �          @�(�@녿�G�@Q�BB�HC�4{@��J�H?��HA��HC���                                    Bxv8��  
�          @��\@A녿�z�@/\)Bp�C�}q@A��&ff?�z�A���C���                                    Bxv8�r  
�          @���@R�\����@\)BQ�C��@R�\�{?�=qA�=qC���                                    Bxv8�  T          @���@Q녿�@{B(�C�&f@Q��  ?�ffA��
C���                                    Bxv8�  �          @���@6ff���@6ffB(��C�ff@6ff�%?�ffA�=qC��                                     Bxv9d  �          @�Q�@I����@&ffB�
C�ٚ@I����
?�A���C��                                    Bxv9
  
Z          @�Q�@QG���ff@!G�B�C��@QG��
�H?�33A��HC�8R                                    Bxv9�  
�          @���@N{��G�@   BQ�C�L�@N{�?��
A�z�C��q                                    Bxv9-V  
(          @��\@A녿��H@$z�B��C�N@A��0��?�\)A�=qC��3                                    Bxv9;�  
�          @��@9����G�@/\)B�C�h�@9���8��?�  A��RC���                                    Bxv9J�  
�          @�(�@(��ff@<(�B)z�C���@(��R�\?�  A��RC�G�                                    Bxv9YH  T          @�{?�=q�{@W�BF�C��H?�=q�fff?�A��C�u�                                    Bxv9g�  
�          @��
?�(��   @c�
B\�RC�  ?�(��`��@ffA���C�W
                                    Bxv9v�  �          @���?�(����@j�HBd  C��?�(��]p�@  A��C���                                    Bxv9�:  "          @�p�@�Ϳ�p�@QG�B?Q�C��@���Vff?���Aƣ�C��                                    Bxv9��  
Z          @��
@
=��@HQ�B7=qC�k�@
=�N{?�G�A�z�C��                                    Bxv9��  
�          @�(�@���\)@S33BC��C��
@��P��?�
=A��
C���                                    Bxv9�,  �          @��@p��G�@=p�B+��C�H�@p��N{?ǮA���C���                                    Bxv9��  
�          @�{@(���@Z=qBH�C��@(��R�\@�
A�C��q                                    Bxv9�x  �          @�p�@	���z�@E�B0��C�k�@	���b�\?��
A��C���                                    Bxv9�  �          @��
@�
���@5�B!��C��)@�
�^�R?��\A�C�Ǯ                                    Bxv9��  
�          @��
@ ���(�@@��B.Q�C�@ ���fff?�z�A��RC��3                                    Bxv9�j  T          @�{@���@9��B"��C��@��b�\?���A�Q�C���                                    Bxv:	  T          @�z�@  �!�@2�\B��C��3@  �dz�?�Axz�C�
                                    Bxv:�  "          @��\@�\���@%B�C��=@�\�W�?���Ak
=C�
                                    Bxv:&\  
�          @���@1녿���@>{B/C�N@1��'
=?�
=AָRC�g�                                    Bxv:5  �          @��\@>{���H@2�\B ��C���@>{�)��?��HA�C�)                                    Bxv:C�  
�          @�z�@���@3�
B��C��@��Z=q?�ffA�Q�C���                                    Bxv:RN  "          @��R@���4z�@p�Bp�C�%@���j�H?@  A(�C���                                    Bxv:`�  �          @�p�@-p��p�@{B
=C��R@-p��W
=?p��AD��C�n                                    Bxv:o�  
�          @�{@.{�   @��B\)C��
@.{�XQ�?h��A=��C�g�                                    Bxv:~@  
�          @�@/\)�%@G�A�=qC�P�@/\)�W�?5A�
C���                                    Bxv:��  
�          @�z�@AG��8��?��HA�p�C�#�@AG��G
=��Q����C��                                    Bxv:��  "          @���@E��>�R?L��A*=qC��3@E��?\)�B�\� z�C���                                    Bxv:�2  "          @���@@���>{?��AqC���@@���I������
=C���                                    Bxv:��  �          @��@7
=�J�H?�=qAb=qC��@7
=�R�\�(����C�~�                                    Bxv:�~  
(          @�{@=p��G�?�=qAa��C��)@=p��P  �z���G�C�'�                                    Bxv:�$  
�          @�@<���E�?�{Ah��C�ٚ@<���N�R�
=q���C�.                                    Bxv:��  
�          @�Q�@X���#33?�(�A�ffC���@X���<(�=#�
>�C���                                    Bxv:�p  "          @���@S�
���?�=qA��
C���@S�
�B�\>��@��C��                                     Bxv;  T          @��@W
=�33@�\A�G�C��@W
=�@  ?+�AffC��                                    Bxv;�  �          @���@B�\�!G�@��A�RC�  @B�\�S�
?@  A�
C�E                                    Bxv;b  �          @�  @J=q�ff@=qB�C�(�@J=q�@��?��A`  C�+�                                    Bxv;.  
�          @�Q�@L�Ϳ�33@!�B��C���@L���8��?��
A��RC���                                    Bxv;<�  "          @�\)@\(���\)@ffA���C�k�@\(��#�
?��\A�  C��
                                    Bxv;KT  
�          @�ff@XQ쿺�H@{B
=C�L�@XQ���R?���A�=qC��                                    Bxv;Y�  
�          @�ff@H�ÿ�@9��B#�C���@H����?�(�A�33C�!H                                    Bxv;h�  �          @��
@`  ����?�Q�A�  C��@`  �z�?n{AHQ�C�E                                    Bxv;wF  
(          @��@k���(�?�Q�A���C�!H@k��ff?E�A$z�C�#�                                    Bxv;��  
�          @�Q�@dzῑ�@ ��A�
=C�  @dz��(�?���A��C���                                    Bxv;��  
�          @���@^{�Tz�@Q�B�RC�H�@^{��\)?�
=A��HC���                                    Bxv;�8  
(          @�z�@XQ�W
=@*�HB��C��)@XQ���?�
=AЏ\C��=                                    Bxv;��  �          @�\)@c33��ff@ ��Bz�C���@c33�Q�?�Q�A�C���                                    Bxv;��  
�          @�@\(���z�@ ��B
�C���@\(��{?��A��HC��\                                    Bxv;�*  
G          @���@\�;u@BQ�C��@\�Ϳ�=q?�Q�A�G�C�s3                                    Bxv;��  T          @�
=@[�>aG�@�HB��@j�H@[��xQ�@�RB 
=C�)                                    Bxv;�v  
�          @��H@E�����@ffB�RC�S3@E���?�p�A��
C�Q�                                    Bxv;�  
�          @��@:�H��\)@�BffC�z�@:�H�#�
?�ffA�Q�C�b�                                    Bxv<	�  T          @��H@*=q�ff@�RB�C��
@*=q�:=q?s33AX  C�4{                                    Bxv<h  �          @��@3�
� ��@�RB�\C�33@3�
�5�?�  A`��C�^�                                    Bxv<'  �          @�33@AG��xQ�@$z�B(�C�q@AG���?��A���C���                                    Bxv<5�  �          @�z�@AG����@�
B	�C�� @AG��p�?��A�33C�q�                                    Bxv<DZ  
�          @�
=@E��{@G�BG�C�9�@E�   ?�(�A�
=C���                                    Bxv<S   �          @��@XQ�=���@/\)BG�?�(�@XQ쿜(�@p�B	��C�{                                    Bxv<a�  �          @�(�@Y����@.�RB(�C��@Y����z�@ffB33C��                                    Bxv<pL  T          @��@W���G�@+�B{C��@W�����@�
Bp�C��                                    Bxv<~�  
�          @���@N�R<#�
@5B%\)>W
=@N�R���@ ��Bz�C���                                    Bxv<��  �          @���@G��0��@7
=B'G�C��\@G����H@(�A�C��\                                    Bxv<�>  
�          @���@N{��
=@0  B!{C�J=@N{��@{A��
C�C�                                    Bxv<��  T          @�  @P�׿#�
@(Q�B\)C�j=@P�׿�@ ��A�z�C�w
                                    Bxv<��  "          @���@@�׿��H@1�B"�\C��@@���Q�?��A���C��3                                    Bxv<�0  "          @�
=@}p��E�?\(�A@��C���@}p���=q>Ǯ@��C�XR                                    Bxv<��  
�          @��R@~{�J=q?�{At(�C�e@~{��(�?��A�
C�}q                                    Bxv<�|  �          @��@�zᾀ  ?:�HA�HC�K�@�z�\)?�@�z�C�&f                                    Bxv<�"  �          @��@�33�fff?!G�A�C���@�33���>�?�p�C��3                                    Bxv=�  	�          @�
=@p�׿B�\?ǮA��\C�L�@p�׿�33?��\Ac
=C�Ǯ                                    Bxv=n  
Z          @��@s33�O\)?ǮA�p�C�  @s33����?�  A]�C���                                    Bxv=   �          @��@x�ÿn{?�ffA��C�AH@x�ÿ�
=?5A�\C��                                    Bxv=.�  T          @�p�@u�����?��A��
C�
=@u�����?�A���C�0�                                    Bxv==`  
�          @�\)@\�;�\)@�
A�p�C���@\�Ϳ��H?�
=Ař�C�S3                                    Bxv=L  "          @�  @P  ���@&ffB�HC�7
@P  �ٙ�@�\A�=qC�1�                                    Bxv=Z�  "          @���@a녿�=q@(�A���C�xR@a녿�p�?�
=A��C�U�                                    Bxv=iR  T          @���@\�Ϳ�Q�@p�A��
C���@\����?�33A�G�C�p�                                    Bxv=w�  T          @�=q@:�H���?��AиRC�y�@:�H�2�\?��@��\C�,�                                    Bxv=��  �          @��@8���\)?�(�AɮC��@8���2�\>�@�Q�C�                                      Bxv=�D  T          @��H@C�
���R?�=qA��C�w
@C�
�'
=?0��A  C��)                                    Bxv=��  T          @��\@Vff��=q?�A��C�  @Vff��?�=qAuG�C�
                                    Bxv=��  "          @��\@X�ÿ�G�?ٙ�Aď\C���@X���Q�?L��A4��C��                                    Bxv=�6  
(          @��\@K���
?�p�A�
=C��=@K�� ��>�Q�@���C��R                                    Bxv=��  �          @��H@X�ÿ޸R?�p�A�p�C�k�@X����R?   @�G�C�\)                                    Bxv=ނ  �          @���@`  ��\)?�
=A���C��3@`  ���>8Q�@!G�C��f                                    Bxv=�(  �          @���@b�\��{?xQ�A`Q�C���@b�\��\)=�?�C�\                                    Bxv=��  5          @z=q@]p���  >���@�G�C���@]p��ٙ�����{C��=                                    Bxv>
t  s          @z=q@]p�����  �j=qC�)@]p���\)�}p��nffC�7
                                    Bxv>  T          @tz�@P  ��  �����C���@P  ����Tz��P��C�W
                                    Bxv>'�  
�          @u@\(���ff��\)��\)C��H@\(���  �u�k�C��                                    Bxv>6f  "          @u@J=q��\�u�z�HC�c�@J=q����^�R�^=qC�H                                    Bxv>E  �          @u@N{��z�?�AC��@N{�����������C�p�                                    Bxv>S�  �          @xQ�@aG��Ǯ��{���HC��@aG���p����\�uG�C�]q                                    Bxv>bX  
�          @xQ�@S33��  >�(�@��HC��@S33��  ��
=���C��                                    Bxv>p�  
�          @x��@dzῷ
=>��H@�G�C�
@dz��  �aG��N{C��H                                    Bxv>�  �          @w�@O\)��p�?(�A�\C��@O\)���þ�  �w
=C�Y�                                    Bxv>�J  
�          @xQ�@L�Ϳ�z�?B�\A733C���@L���33�L���C�
C��3                                    Bxv>��  T          @z=q@n{���?�@�p�C��{@n{���H�����HC�                                      Bxv>��  �          @z=q@q녿Tz�?
=q@��C�ٚ@q녿z�H=�G�?���C���                                    Bxv>�<  �          @z=q@vff�B�\?
=q@�ffC��q@vff���>Ǯ@�=qC��{                                    Bxv>��  T          @y��@mp�=�Q�?��Ay�?�{@mp��Ǯ?xQ�Ag�
C��R                                    Bxv>׈  T          @z=q@j=q>�?��A��@�z�@j=q��G�?�(�A���C�&f                                    Bxv>�.  T          @r�\@^{�8Q�?�
=A��HC��=@^{�O\)?�Q�A�{C�u�                                    Bxv>��            @tz�@mp�>\?��Ap�@���@mp�=u?333A+\)?}p�                                    Bxv?z  r          @l��@Z�H?O\)?�AffAT  @Z�H>�?^�RAb=q@�{                                    Bxv?   �          @k�@L(��\)?�33A��C��@L(�����?�  A��C�7
                                    Bxv? �  �          @p��@AG�?E�?�
=A�33Ad��@AG���\)?���A�  C�L�                                    Bxv?/l  �          @w
=@@��?���?�\)A�33Aߙ�@@��?0��@
=qB{AP                                      Bxv?>  �          @u@U���Q�?�{A�RC�<)@U��k�?�\)A�z�C�Q�                                    Bxv?L�  T          @vff@B�\��?��RA��C��R@B�\�33>�Q�@��C��q                                    Bxv?[^  
�          @xQ�@J=q���?�(�A��
C��@J=q��?uAg�C�S3                                    Bxv?j  T          @u@J=q�u?�\)A�G�C���@J=q�˅?��\A�ffC���                                    Bxv?x�  "          @w
=@\(�����?���A¸RC���@\(����?�(�A�C�l�                                    Bxv?�P  �          @xQ�@c�
?W
=?
=qA�AUp�@c�
?�\?\(�AW
=A\)                                    Bxv?��  T          @~{@n�R>�G�?�{A�{@���@n�R����?�Q�A��C�8R                                    Bxv?��  
�          @�Q�@Y����@G�A�p�C���@Y������?���A�G�C���                                    Bxv?�B  
�          @|(�@\)��p�@6ffB:�
C�=q@\)�&ff?��A�RC�^�                                    Bxv?��  T          @x��?�\����@FffBV33C�K�?�\�+�@Q�B  C��3                                    Bxv?Ў  
(          @~�R?�p���  @[�Br  C���?�p��7�@��BQ�C���                                    Bxv?�4  �          @~�R?�  ��\)@S33B`\)C���?�  �,(�@ffB�
C��                                     Bxv?��  h          @�  @zῸQ�@FffBK��C���@z��*�H@��Bz�C��                                    Bxv?��  "          @���@���=q@:=qB9p�C��{@���R@G�A�33C�>�                                    Bxv@&  �          @���@)����=q@4z�B2�C���@)���{@�
A�  C�\                                    Bxv@�  
n          @���@%��Y��@=p�B=�C��=@%��z�@�B
\)C���                                    Bxv@(r  �          @���@.�R�u@4z�B1(�C�P�@.�R�
=@
=A��C�*=                                    Bxv@7  T          @\)@-p��u@0  B.�
C�C�@-p��z�@33A�\)C�L�                                    Bxv@E�            @~{@(Q�\(�@5�B6�\C��R@(Q��G�@
�HBz�C�C�                                    Bxv@Td  r          @�  @.{�\)@6ffB6��C�<)@.{��  @z�BQ�C��)                                    Bxv@c
  
Z          @��@7
=�
=@333B.�HC�9�@7
=��  @  Bz�C�AH                                    Bxv@q�  
�          @�  @1녿�R@333B1�HC��H@1녿��
@\)B�RC��\                                    Bxv@�V            @~�R@<(���@'
=B$
=C�z�@<(���z�@A�(�C�K�                                    Bxv@��  
�          @~{@B�\�:�H@(�B��C�AH@B�\��(�?�\)A��C�>�                                    Bxv@��  "          @~�R@H�ÿ��
@��BC��@H�ÿ��?��
A��C�~�                                    Bxv@�H  �          @���@QG����@�A�p�C��@QG���{?�z�A�Q�C�/\                                    Bxv@��  T          @\)@%��\@(��B5Q�C�@%����R@��B  C��                                    Bxv@ɔ  
�          @�  ?�G�?333@hQ�B��RA�  ?�G��s33@dz�B~
=C���                                    Bxv@�:  �          @|(�?�R?�Q�@fffB��fB�Ǯ?�R��=q@w
=B�W
C�S3                                    Bxv@��  
�          @��H?.{?��H@uB�B�Bq��?.{�z�@~�RB���C��)                                    Bxv@��  T          @��\>�?�(�@xQ�B�\B�#�>��
=@���B�  C�\                                    BxvA,  T          @��>Ǯ?c�
@~�RB���B��>Ǯ�k�@~{B�C�Z�                                    BxvA�  "          @�z�W
=?(�@��\B��B���W
=���H@|��B�aHC��                                    BxvA!x  
(          @�녾�  ?z�@�  B�{B�녾�  ���H@w
=B���C�q                                    BxvA0  
�          @����33��\)@�=qB�
=C>Ǯ��33��@h��B|p�C���                                    BxvA>�  T          @��
>\)�.{@|��B��C�B�>\)����@_\)Bw�
C�R                                    BxvAMj  
�          @�33?�ff?(��@mp�B�8RA��R?�ff��  @h��B|p�C���                                    BxvA\  �          @��@Q�?:�H@=p�BG�HA��@Q�(�@?\)BJ33C��H                                    BxvAj�  
�          @��H@QG�?��\@�RB(�A�33@QG���@��Bz�C��)                                    BxvAy\  �          @�33@J�H?�\)@B=qA�G�@J�H<#�
@%B  >��                                    BxvA�  �          @��H@6ff?�z�@+�B$z�A��R@6ff����@:�HB6��C��R                                    BxvA��  T          @�=q@9��?�G�@%B{A�z�@9��=#�
@8Q�B3(�?E�                                    BxvA�N  "          @��@6ff?�=q@#33B�HA�ff@6ff=�@7�B4�R@=q                                    BxvA��  �          @��@0��?��@(Q�B!�A�@0��>�@>{B<�\@2�\                                    BxvA  
Z          @��@0��?�{@   B33A�p�@0��>Ǯ@<��B:G�A ��                                    BxvA�@  �          @���@-p�?�@ffBG�B�H@-p�?#�
@9��B8�ATz�                                    BxvA��  "          @�33@(Q�@�?�
=A�RB+{@(Q�?���@1�B,ffA�{                                    BxvA�  
�          @��@3�
?�=q@(�B��A�
=@3�
>.{@1G�B2�@aG�                                    BxvA�2            @|��@%�?�{@/\)B1{A�(�@%����@=p�BC�C�Q�                                    BxvB�  T          @\)@(��?��@1�B1�A�  @(�þk�@=p�B@��C��                                     BxvB~  "          @���@9��?k�@(Q�B#=qA�\)@9����=q@1�B.\)C�W
                                    BxvB)$  �          @���@I��?(�@=qB�RA/�@I����@(�B{C���                                    BxvB7�  T          @�=q@U>�  @33B	�R@��@U�:�H@(�B�\C���                                    BxvBFp  
(          @�=q@]p�>�{@A�p�@�@]p��\)@�\A�33C�b�                                    BxvBU  �          @���@u�h��?�(�A�C�O\@u���?:�HA"ffC�]q                                    BxvBc�  "          @�@w
=��z�?�=qAo
=C���@w
=��  >��H@���C�aH                                    BxvBrb  "          @�
=@vff���?��\A_�C��@vff��Q�>�{@��C�#�                                    BxvB�  �          @�\)@w
=��
=?��RA�{C�� @w
=�˅?!G�A	p�C���                                    BxvB��  
�          @��
@s�
�L��?��A���C�{@s�
���\?\(�A@��C��f                                    BxvB�T  
�          @��@u��\(�?�Q�A�z�C��3@u����\?8Q�A"ffC���                                    BxvB��  
�          @�G�@p�׿���?xQ�A^�HC��R@p�׿�z�>�(�@\C�                                    BxvB��  r          @�Q�@qG��aG�?��\Al��C�t{@qG���(�?�A�\C�                                    BxvB�F  
�          @���@s33�=p�?��AqG�C���@s33����?&ffA  C���                                    BxvB��  T          @�Q�@tz�\(�?\(�AEG�C���@tzῑ�>�
=@���C���                                    BxvB�  
�          @\)@p�׿G�?��Aw�C�!H@p�׿�33?&ffAG�C�}q                                    BxvB�8  T          @~�R@g���{?��HA��
C��@g���G�?&ffA�C��
                                    BxvC�  T          @~�R@fff����?�ffAu�C���@fff���>�
=@���C���                                    BxvC�  T          @�Q�@p  �L��?�\)A���C��\@p  ����?0��A (�C�&f                                    BxvC"*  �          @\)@h�ÿW
=?��HA���C�z�@h�ÿ�G�?@  A0��C�n                                    BxvC0�  
�          @�Q�@fff���?��A��C���@fff��
=>��H@�  C�|)                                    BxvC?v  T          @\)@l�Ϳ��
?=p�A*�HC�s3@l�Ϳ�(�>��@	��C�'�                                    BxvCN  "          @~�R@l�Ϳ�  ?:�HA(��C��=@l�Ϳ�Q�>��@p�C�`                                     BxvC\�  �          @|��@i����(�?^�RAL  C���@i����(�>���@��C��                                    BxvCkh  T          @|��@n{��{?8Q�A)G�C���@n{���>W
=@B�\C�K�                                    BxvCz  T          @~{@mp���Q�?=p�A+
=C�
@mp����>B�\@,��C��q                                    BxvC��  �          @\)@n{���?��@�p�C�h�@n{��33���
��33C���                                    BxvC�Z  
Z          @~�R@j�H��ff?uAap�C�H@j�H����>�ff@��C��                                    BxvC�   6          @\)@l(��:�H?�G�A�ffC�j=@l(���
=?\(�AH(�C�'�                                    BxvC��  @          @���@C�
�'
=��  �g�C���@C�
��\�������C��                                    BxvC�L  
�          @�G�@<(��0  ����  C�q�@<(��{��(���\)C��q                                    BxvC��  
�          @���@C�
�#�
��
=��ffC��@C�
�
�H��z����RC�Q�                                    BxvC��  �          @�  @A��&ff�k��Q�C���@A���\��  ��=qC�z�                                    BxvC�>  T          @\)@;��,(�=u?n{C��)@;��\)���\�mC��{                                    BxvC��  
(          @}p�@>�R�%=�?��C�~�@>�R���n{�X��C�p�                                    BxvD�  "          @|(�@;��(Q�=�?�C�@;��p��p���\z�C�H                                    BxvD0  T          @z�H@E��{>\@�ffC�@E��(��
=q��\C�C�                                    BxvD)�  
�          @r�\@%�G�?Tz�AW�C�\)@%��H�����C��H                                    BxvD8|  �          @w
=@e���?k�Ad��C�@ @e�L��?(��A"�\C���                                    BxvDG"  �          @u�@H�ÿ���?&ffA#
=C��@H�ÿ�
=������C�4{                                    BxvDU�  T          @w�@b�\���H?fffAX(�C���@b�\��(�>�33@�z�C��                                     BxvDdn  T          @x��@E���Ϳ�33��C�33@E�����z�C��=                                    BxvDs  �          @z�H@*�H�,(�����  C�j=@*�H�=q��������C���                                    BxvD��  
�          @�  @p��E=���?���C�L�@p��8Q쿐������C�L�                                    BxvD�`  �          @���@0���8Q�>�@�33C��H@0���5��333��C��                                    BxvD�  T          @�Q�@2�\�5>�ff@�ffC�H�@2�\�2�\�333�!C���                                    BxvD��  "          @}p�@5�.�R>�{@�{C��@5�(�ÿB�\�1p�C��H                                    BxvD�R  "          @\)@O\)�   ?�z�A�
=C�!H@O\)�33>�\)@\)C�T{                                    BxvD��  �          @�G�@*�H�-p�?fffAUG�C�G�@*�H�6ff�k��Z�HC��3                                    BxvDٞ  
�          @�  @z��Z=q�u�Tz�C���@z��G
=�������C��                                    BxvD�D  
�          @���?�\�g
=���
���
C�\?�\�L(����H��z�C�}q                                    BxvD��  "          @���@�
�^�R������C�U�@�
�H�ÿ��R���C���                                    BxvE�            @\)@(Q��z�>L��@P  C�AH@(Q��{�8Q��=p�C��=                                    BxvE6            @y��@tz�Ǯ?�RA�C��@tz���>�
=@�(�C��f                                    BxvE"�  "          @xQ�@w
=���
>W
=@B�\C��)@w
=����>8Q�@,(�C�G�                                    BxvE1�  
�          @xQ�@w��#�
=�?���C���@w��L��=L��?=p�C��f                                    BxvE@(  T          @z=q@w
=<�>�
=@�
=>�@w
=��>���@�p�C��                                    BxvEN�  
�          @�G�@���L��>�@ӅC�b�@���HQ�O\)�9p�C��\                                    BxvE]t  
�          @��H@���[�>B�\@,(�C�P�@���O\)��33��z�C��                                    BxvEl  "          @��?��\�u��\)��Q�C�/\?��\�^�R�������RC��                                    BxvEz�  �          @�G�?����n�R�k��N{C��f?����W
=�����33C���                                    BxvE�f  �          @~�R?�{�hQ�B�\�1��C�B�?�{�C�
�ff� {C��
                                    BxvE�  
Z          @~{?�=q�mp�?aG�AN�\C�!H?�=q�p�׿����C��                                    BxvE��  
(          @x��?p���n�R?�\@�  C�)?p���h�ÿu�dz�C�G�                                    BxvE�X  T          @{�?c�
�tz�>\)@G�C���?c�
�e���=q��Q�C���                                    BxvE��  T          @~�R?k��w
=>\)?�p�C���?k��g������(�C�'�                                    BxvEҤ  "          @~{?�ff�tz�>aG�@N{C��?�ff�g���  ��z�C�
                                    BxvE�J  T          @y��?�G��p  >�z�@�G�C��=?�G��e���33���
C��H                                    BxvE��  �          @|(�?W
=�u�>�Q�@�Q�C�'�?W
=�k���\)���C�e                                    BxvE��  �          @|��?Tz��u�>�@�C�
?Tz��n�R��G��l��C�AH                                    BxvF<  �          @|��?333�vff>�@߮C�"�?333�p  ��G��mp�C�E                                    BxvF�  "          @~{?s33�tz�>��@��RC���?s33�l(�����xQ�C�9�                                    BxvF*�  
�          @|��?@  �s�
?+�A�C��
?@  �q녿O\)�=�C��H                                    BxvF9.  "          @~{?��\�o\)>�@ڏ\C���?��\�h�ÿz�H�g33C���                                    BxvFG�  
�          @\)@
=q�S�
����(�C���@
=q�8�ÿ�
=��  C�h�                                    BxvFVz  �          @{�?�=q�^{����G�C��=?�=q�J�H��z���ffC�                                      BxvFe   T          @x��?��H�_\)>��@(�C�?��H�R�\������C���                                    BxvFs�  �          @y��?�  �^{>k�@Y��C�` ?�  �S�
��=q���HC��                                    BxvF�l  T          @~{?�Q��e�>��@s33C��)?�Q��Z�H����
=C��                                    BxvF�  �          @\)?�G��b�\>�33@��C�<)?�G��Z�H�}p��g33C��                                    BxvF��  T          @~�R?��R�Z=q>��
@�ffC��?��R�R�\�xQ��ap�C���                                    BxvF�^  "          @�  ?����^{?��A
{C��?����\�Ϳ:�H�)G�C�!H                                    BxvF�  �          @���?���k�>�G�@�Q�C�\)?���e��s33�\  C���                                    BxvF˪  �          @�Q�?�(��c�
?\)AG�C��?�(��aG��J=q�6�HC��                                    BxvF�P  
�          @�Q�?�Q��l(�>��@�ffC���?�Q��e�z�H�b�RC��
                                    BxvF��  "          @���?�=q�hQ�>��@�=qC���?�=q�c33�c�
�O\)C�                                      BxvF��  
�          @���?�  �`  ?O\)A;�
C�L�?�  �c�
�����HC�                                      BxvGB  
(          @���?�=q�hQ�=���?�(�C�?�=q�Z=q��p���p�C�j=                                    BxvG�  T          @��?����j�H?5A#\)C��?����j�H�.{�Q�C��=                                    BxvG#�  6          @�33?�ff�u?�\@�  C�T{?�ff�p�׿k��P��C��H                                    BxvG24  �          @��?У��j=q>Ǯ@���C�?У��c33�xQ��_�C�Y�                                    BxvG@�            @��?�\�g
=?�@�
=C��?�\�c33�Q��;33C�:�                                    BxvGO�  
�          @�G�?�Q��g�>�@ۅC�� ?�Q��c33�^�R�G33C���                                    BxvG^&  
�          @���?�Q��hQ�?�\@陚C�u�?�Q��dz�W
=�?�C��                                    BxvGl�  
�          @�=q?��g
==���?�
=C�N?��Z=q���H���C�                                      BxvG{r  �          @���?�(��a�=�?�G�C��\?�(��U��z���\)C�AH                                    BxvG�  �          @�=q@Q��\(�>�{@�=qC��f@Q��U��k��Q�C�P�                                    BxvG��  T          @�=q@G��Vff������
=C��@G��@�׿\��C�|)                                    BxvG�d  
�          @�G�?Ǯ�l�;�=q�vffC�l�?Ǯ�Vff��=q���C�u�                                    BxvG�
  
�          @���?�(��i���E��0z�C��q?�(��HQ��G����C��R                                    BxvGİ  T          @�G�?У��fff�G��2{C�/\?У��E�G���(�C��\                                    BxvG�V  
�          @���?�\)�p  �z���C��?�\)�S33�����  C�Q�                                    BxvG��  �          @��\?�Q��r�\������{C�b�?�Q��Y����p����HC�xR                                    BxvG�  �          @��?����mp���G��ȣ�C�t{?����S�
��p��ʸRC��                                    BxvG�H  "          @�Q�?�ff�u���{���C���?�ff�]p���
=��p�C�n                                    BxvH�  
�          @~{@?\)�!G����(�C��@?\)�
=q��
=���C��                                    BxvH�  �          @}p�@E��ÿ.{��\C�'�@E���R��p���C���                                    BxvH+:  "          @~�R@>�R�&ff�Ǯ���C�q�@>�R�33�������C�4{                                    BxvH9�  T          @~{@N{�33>k�@W
=C�=q@N{��R�
=�	C��H                                    BxvHH�  T          @���@U��  >8Q�@(��C���@U��
�H��R���C�q�                                    BxvHW,  �          @���@G��#�
    ��C�Y�@G���ÿfff�N=qC�Ff                                    BxvHe�            @}p�@33�XQ�k��U�C��{@33�E��33���
C�                                    BxvHtx  q          @\)@   �E��B�\�0��C��f@   �4zῡG���C�Ǯ                                    BxvH�  
�          @��H@E�*�H���\C��R@E�\)�u�Y�C���                                    BxvH��  �          @���@(���AG����R��{C��R@(���.{�����=qC�{                                    BxvH�j  T          @�G�@,(��:=q�z��z�C�e@,(��!녿�ff���\C�j=                                    BxvH�  
�          @�G�@.{�>�R    �uC�,�@.{�2�\����o
=C��                                    BxvH��  ?          @�p�@%�P  >��R@���C�C�@%�J=q�W
=�9�C���                                    BxvH�\  �          @��@�
�W��B�\�-p�C�33@�
�Fff������{C�Z�                                    BxvH�  
�          @�33@`  ��녿s33�W�
C���@`  ������ff��  C��f                                    BxvH�  
�          @�=q@^�R�޸R�������HC���@^�R��(���(���z�C�XR                                    BxvH�N  
�          @���?Tz��Q�?��A��HC��?Tz��h��>��@�Q�C�ff                                    BxvI�  
�          @�G�?s33�W
=?��RA�G�C�޸?s33�u?5A#\)C��                                    BxvI�  �          @�Q�?����L��?�p�A�Q�C�E?����l(�?B�\A1G�C��                                    BxvI$@  "          @�Q�@G��L(�?W
=A@��C��
@G��Q녾��R���C�S3                                    BxvI2�  T          @���@   �@��?��An�RC��@   �L(��L�Ϳ.{C�
                                    BxvIA�  �          @�G�@333�)��?�Q�A�C�L�@333�9��>aG�@G�C��                                    BxvIP2  �          @\)@9���(�?��RA���C�  @9���-p�>�{@��HC�xR                                    BxvI^�  7          @���@'
=�+�?��RA�  C��@'
=�A�>��H@�Q�C�Y�                                    BxvIm~  
�          @��H@(Q��'
=?޸RA���C��
@(Q��B�\?@  A(��C�e                                    BxvI|$  "          @��@'
=�ff@ffA�33C�H@'
=�:�H?�
=A�Q�C��                                    BxvI��  
�          @�z�@!G����@��Bp�C��@!G��;�?��RA��C�L�                                    BxvI�p  
�          @�33@
=q���@:=qB6
=C��=@
=q�0  @
=A���C��                                    BxvI�  
�          @���?�(��W�?��A�33C�|)?�(��g�=���?��HC��\                                    BxvI��  
�          @�  ?�=q�N{?�G�A�\)C��{?�=q�b�\>�{@�(�C���                                    BxvI�b  
Z          @�=q?�(��c33?}p�Ab=qC���?�(��j�H��  �e�C���                                    BxvI�  "          @��H?����mp�?��\Ahz�C��
?����u��\)�z=qC���                                    BxvI�  �          @��\?�(��`��?�{A�\)C���?�(��u>�{@��
C���                                    BxvI�T  �          @�G�?�G��Y��?�G�A�=qC��\?�G��mp�>�\)@�=qC��                                    BxvI��  �          @��\?�G��^�R?�A�G�C�ff?�G��j�H�u�O\)C���                                    BxvJ�  "          @��H?޸R�U?�\)A��C��R?޸R�aG��u�Y��C�q                                    BxvJF  �          @��\@�>�R?���A���C��@�Z=q?:�HA&=qC�                                    BxvJ+�  7          @�Q�?����L(�?�Q�A�=qC���?����_\)>���@��C���                                    BxvJ:�  
�          @���?��H�\��?0��A��C��=?��H�^�R���H��\)C���                                    BxvJI8  �          @��@��\(�<�>�
=C�4{@��P�׿�{�|  C��                                    BxvJW�  T          @���@
�H�`  >�p�@��C���@
�H�[��L���2�\C�.                                    BxvJf�  T          @��?�(��dz�>��@�{C�k�?�(��`�׿J=q�1C��                                    BxvJu*  �          @���@.{�G��W
=�7�C���@.{�8Q쿜(�����C���                                    BxvJ��  �          @��
@,���C33����p�C��f@,���,(��Ǯ��ffC��
                                    BxvJ�v  �          @�G�@{�Fff�5�!��C�B�@{�,�Ϳ�
=��\)C�7
                                    BxvJ�  "          @�33@-p��>�R��� Q�C�"�@-p��(�ÿ�G���{C��                                    BxvJ��  �          @�(�@S33�G��u�W�
C��R@S33����33����C�s3                                    BxvJ�h  �          @��@7
=�0  �u�Yp�C�\@7
=��\��ff��\)C���                                    BxvJ�  T          @��\@/\)�%����
��p�C�Y�@/\)��Q����	
=C�O\                                    BxvJ۴  
�          @��H@333�p���
=��z�C�U�@333���
������C��q                                    BxvJ�Z  "          @��@.�R�)�����
����C���@.�R�G��33�p�C��f                                    BxvJ�   T          @��\?��
�A녿����C�>�?��
�=q�z��C�@                                     BxvK�  
Z          @���?�p��"�\������C���?�p��ٙ��9���?�RC��\                                    BxvKL  
�          @���@33�L�Ϳ��\�pz�C�S3@33�,�Ϳ��R��p�C��
                                    BxvK$�  
�          @��
?��\�y����=q�s�
C���?��\�g
=��G���=qC���                                    BxvK3�  
�          @�(�?����x�þ�  �a�C��=?����g
=��p����RC�<)                                    BxvKB>  
�          @�(�?�33�w
=�����{C��)?�33�a녿�\)���HC��
                                    BxvKP�  
�          @�(�?�z��p�׿#�
��
C�G�?�z��W���ff��=qC�`                                     BxvK_�  "          @�z�?fff���׾������HC�J=?fff�mp��˅��33C��                                    BxvKn0  
�          @�  ?n{���
��{��G�C�aH?n{�s33��\)��z�C��                                    BxvK|�  
�          @���>L���������C�XR>L����Q쿯\)��33C�k�                                    BxvK�|  T          @�ff?
=���
��33���HC�3?
=�s33�����z�C�j=                                    BxvK�"  T          @��
?W
=�{��c�
�H(�C��?W
=�\�����C��q                                    BxvK��  �          @��?�G��xQ�k��O�
C�J=?�G��X���ff��p�C�J=                                    BxvK�n  �          @��?=p��{��s33�VffC�L�?=p��[�������HC��                                    BxvK�  �          @��H?(��|�ͿTz��;
=C�\)?(��_\)�����
C���                                    BxvKԺ  T          @���>���Q�n{�O33C�C�>��aG������z�C��R                                    BxvK�`  �          @�z�>�ff���׿Y���<��C�*=>�ff�c�
��
���C��3                                    BxvK�  T          @��?��\)�u�UC��
?��_\)�	������C�<)                                    BxvL �  �          @�ff�u���׿���s
=C�H��u�^�R��\�(�C�                                    BxvLR  T          @�  �s33��Q쿀  �Z=qC�XR�s33�`  ������C~�
                                    BxvL�  �          @���>.{��z�c�
�B{C�'�>.{�j=q�Q���p�C�O\                                    BxvL,�  
�          @�  =��
��ff�(���  C��=��
�r�\��
=��(�C��3                                    BxvL;D  �          @���=���
=������C��=��w
=��G���{C��H                                    BxvLI�  "          @����G���ff����ffC�>���G��u���ff��33C�+�                                    BxvLX�  "          @��R>�Q���(��
=��C��H>�Q��p  ��=q��G�C�                                    BxvLg6  T          @�
=�8Q����H�}p��Z�\C��׾8Q��e������(�C��
                                    BxvLu�  T          @�ff=�Q���녿��\�`��C��H=�Q��c33�p���33C��
                                    BxvL��  T          @�>�p��\)��\)�x��C���>�p��]p����{C��                                    BxvL�(  �          @��R?��|(������ffC�?��Vff�����\C�j=                                    BxvL��  T          @��>�G���G������RC�"�>�G��_\)����C��                                     BxvL�t  "          @��R?333�~{��z�����C��?333�Z�H��
���C���                                    BxvL�  T          @�
=?Q��\)��ff�h(�C�Ǯ?Q��^�R�{��(�C��R                                    BxvL��  �          @�\)?G����׿}p��X��C�y�?G��a��
=q���RC�5�                                    BxvL�f  �          @��R?=p���=q�8Q���\C�!H?=p��j=q���ٙ�C��\                                    BxvL�  
�          @��?.{��(�����Q�C���?.{�r�\��
=���\C�q                                    BxvL��  "          @��?�G��@  �%�p�C�aH?�G����W��R��C��                                    BxvMX  T          @��R?���,(��,(���C��{?����G��XQ��U�C��f                                    BxvM�  �          @��R?�z���<���1\)C��?�zῬ���aG��a�C�q�                                    BxvM%�  �          @�{?p���-p��;��6z�C���?p�׿��H�fff�vz�C�e                                    BxvM4J  �          @�z�?(����׾���\)C�Q�?(��l�Ϳ�z���33C���                                    BxvMB�  
�          @}p��+��S33@G�A�p�C�H��+��p  ?p��A^{C��{                                    BxvMQ�  
�          @�z�������?Tz�A:=qC������33�������HC�q                                    BxvM`<  
w          @�\)���
���?333A�C����
��z�\)��p�C��                                    BxvMn�  
�          @�Q�?
=q�����
����C��?
=q�~{��������C��                                     BxvM}�  �          @�G�>aG���ff�Ǯ���C��H>aG��y����{��p�C���                                    BxvM�.  �          @���?&ff�}p���(����C���?&ff�Vff�%��=qC��H                                    BxvM��  �          @�Q�?Q���G��O\)�1��C��{?Q��hQ��(���
=C�XR                                    BxvM�z  
Z          @�G�?���p��E��%p�C��f?��p�׿��H��z�C��                                    BxvM�   �          @��>����zῑ��t��C��>���hQ��33�ffC��                                    BxvM��  T          @�=q>��������\�[\)C�=q>����l(��������C���                                    BxvM�l  �          @��H=L�����H��\)���RC�Y�=L���`  � ���ffC�h�                                    BxvM�  T          @��H?L�����E��$��C�aH?L���q녿��H�י�C��3                                    BxvM�  �          @�33?+���G����H��33C��
?+��\(��%��Q�C��f                                    BxvN^  �          @�\)?
=�tz��
=���C�]q?
=�J�H�.{� �RC�@                                     BxvN  �          @�  ?����dz��ff���HC�L�?����8���0���%�RC�G�                                    BxvN�  �          @���>�=q�j=q��33��
=C�%>�=q�<���8Q��0�C���                                    BxvN-P  "          @�p�>����h���   ��p�C�!H>����9���>{�5z�C��=                                    BxvN;�  �          @��?!G��a�� �����C�\?!G��333�<(��6��C�W
                                    BxvNJ�  
�          @�(�>��b�\����G�C��\>��2�\�@���:��C���                                    BxvNYB  �          @�{?J=q�g���z���z�C�0�?J=q�:�H�7��.=qC��                                     BxvNg�  "          @��>�ff�X����	�C���>�ff�%��N{�Kz�C�H                                    BxvNv�  T          @�{?:�H�>{�7
=�,Q�C��?:�H���fff�l=qC��H                                    BxvN�4  
�          @�ff?=p��Vff�����C�1�?=p��"�\�L(��IQ�C��                                    BxvN��  T          @���?}p��l(������G�C���?}p��@  �6ff�(G�C�"�                                    BxvN��  �          @���?�=q�z=q��z��{�C�o\?�=q�Z=q��R��=qC��                                    BxvN�&  T          @���?�=q�q녿�(�����C���?�=q�Mp���R�33C�B�                                    BxvN��  
�          @�\)?���g���p����C���?���>�R�+��{C��H                                    BxvN�r  �          @��R?�(��I���\)�z�C��R?�(��z��Q��M�\C��                                    BxvN�  �          @��?���A��   ��C���?������P  �NffC��                                    BxvN�  
(          @��H?��7�� ���{C�.?��33�N{�P��C�ff                                    BxvN�d  
�          @��H?��
�0  ��R� �\C�B�?��
��Q��J=q�\z�C��R                                    BxvO	
  "          @�?8Q���=q>��@�G�C��?8Q���G��0�����C��                                    BxvO�  T          @�z�?s33�~�R>�z�@���C��3?s33�y���W
=�=G�C��{                                    BxvO&V  
�          @���?��
��Q�k��K�C�0�?��
�q녿����\)C���                                    BxvO4�  �          @�=q?�33�r�\�u�_\)C�l�?�33�dzῥ���  C���                                    BxvOC�  �          @���?���<�Ϳ����\)C�S3?����
�'��%ffC��f                                    BxvORH  T          @���?�\�G�������\)C��q?�\�\)�'��"z�C���                                    BxvO`�  �          @~{@��AG���(���ffC��\@��\)����Q�C���                                    BxvOo�  
�          @~�R?���P�׿����ffC�k�?���0���(���C��                                    BxvO~:  
�          @\)@�
�Q녿n{�W�
C�
=@�
�9����ff��G�C���                                    BxvO��  "          @\)?�(��L(���=q���\C��R?�(��,(��
=q��RC��                                    BxvO��  "          @~{?�(��H�ÿ�\)���\C��?�(��(���(��Q�C�^�                                    BxvO�,  
�          @~�R@�\�?\)��G���(�C���@�\�!���\����C�                                    BxvO��  �          @~{@�H�#�
��=q��\)C��)@�H���H�p��=qC��                                    BxvO�x  
�          @z=q?�G��P  ��G����\C�s3?�G��-p���33C���                                    BxvO�  "          @y��?�p��5���
��G�C�k�?�p��  �\)��C��\                                    BxvO��  T          @y��?�
=�0  �
=q��C��{?�
=�33�5��;C��                                    BxvO�j  T          @tz�?�z��  ����C�8R?�z�\�7
=�F{C���                                    BxvP  �          @z�H?�\)�(Q���
�z�C���?�\)��33�<(��M��C��f                                    BxvP�  �          @\)?n{�U������G�C���?n{�+��2�\�2
=C��                                    BxvP\  �          @�  ?&ff�[������(�C�\)?&ff�2�\�/\)�.��C���                                    BxvP.  "          @�Q�?�G��Y��������(�C�K�?�G��1G��,���)�C�
=                                    BxvP<�  	�          @�Q�?���j�H��
=��
=C��)?���H�����ffC�]q                                    BxvPKN  
�          @�  ?:�H�hQ쿾�R��  C��?:�H�E��H�Q�C���                                    BxvPY�  "          @�Q�?.{�hQ쿷
=���RC�L�?.{�G
=�ff�C�'�                                    BxvPh�  T          @|��?Tz��s�
��Q���\)C�  ?Tz��dz`\)���
C��                                    BxvPw@  �          @\)?G��r�\�B�\�2ffC��=?G��\(���  ��33C�Z�                                    BxvP��  "          @~{?�
=�`  ��z���\)C�XR?�
=�?\)�33�33C��=                                    BxvP��  �          @~{?z��g���  ��C��f?z��E��=q��C�J=                                    BxvP�2  "          @~�R?����XQ��p���\)C�B�?����333�$z��!�C��                                    BxvP��  S          @�Q�?����L�����Q�C�AH?����!G��7��9(�C�}q                                    BxvP�~  �          @�(�@
=�`  ��p����C��=@
=�QG������
=C�ff                                    BxvP�$  �          @�Q�@�H�G���z���p�C�� @�H�;���{��  C���                                    BxvP��  �          @}p�@��R�\�0���"�\C��\@��?\)������HC��                                    BxvP�p  
�          @�Q�@%�A�>8Q�@#33C�G�@%�>{�!G��C���                                    BxvP�            @�Q�@�
�L�;�33���C��@�
�?\)��Q���p�C��
                                    BxvQ	�  T          @���@*�H�@  ������z�C���@*�H�2�\��
=���C��
                                    BxvQb  �          @��H@��L(���ff�s�C�,�@��333��������C��{                                    BxvQ'  "          @\)?��fff�h���S33C��R?��O\)������z�C��                                    BxvQ5�  
�          @{�?�=q�`�׿!G��  C�q?�=q�N�R���
��ffC�\                                    BxvQDT  �          @|��?�\)�c�
�c�
�R�HC���?�\)�L�Ϳ����Q�C���                                    BxvQR�  �          @}p�?L���^�R��Q��ʣ�C��H?L���:=q�"�\� =qC���                                    BxvQa�  �          @~{>�ff�a녿�Q�����C��>�ff�>{�#33�!�C�T{                                    BxvQpF  �          @u�>�Q��N{�����\)C�1�>�Q��+����'C���                                    BxvQ~�  "          @tz�@B�\>Ǯ@��B@�33@B�\�W
=@�B�\C��                                    BxvQ��  
�          @y��@E=�\)@  B  ?�  @E��@(�B�\C�<)                                    BxvQ�8  
�          @xQ�@Dz��G�?���A�G�C�U�@Dz��\)?�A��C��=                                    BxvQ��  "          @~�R@  �O\)>u@`  C�Z�@  �L(��(����C��                                    BxvQ��  �          @�  @���U���Ϳ�(�C�W
@���L�Ϳu�`��C��                                    BxvQ�*  8          @���?�
=�e��(��z�C��?�
=�S33��G���  C�y�                                    BxvQ��  �          @���?ٙ��`�׿p���YC��?ٙ��I�������י�C�(�                                    BxvQ�v  �          @���?�\)�j�H������  C��f?�\)�\(��������C��R                                    BxvQ�  �          @���@���Z�H��G���C��@���L(���ff����C��                                    BxvR�  �          @~�R?�G��a녾��
��ffC�G�?�G��Tz῜(����C��
                                    BxvRh  "          @|(�@<(��\)?(�AffC�T{@<(��z�<��
>��C���                                    BxvR   T          @{�@G
=���H?�G�A���C��H@G
=�{?5A)�C�AH                                    BxvR.�  �          @y��@1G��?���A��\C�� @1G��&ff?.{A"�HC�`                                     BxvR=Z  "          @{�@ ���5�?B�\A5p�C���@ ���;�<#�
>8Q�C�J=                                    BxvRL   �          @}p�@p��<(�?L��A=�C���@p��C33<�>ǮC�t{                                    BxvRZ�  �          @z�H?�{�S33����p�C���?�{�C�
�������C��\                                    BxvRiL  "          @x��@�AG���Q���p�C�U�@�5���\)��G�C�9�                                    BxvRw�  �          @z=q@=p��z�?:�HA0��C��{@=p���=�?�C�N                                    BxvR��  �          @xQ�@#33�8��>\)@	��C��q@#33�5��
=���C�f                                    BxvR�>  �          @z�H@W
=��z�?�@��C�*=@W
=��p�<�>�(�C��{                                    BxvR��  �          @�  @dz��=q?Y��AD  C��@dz�޸R>�
=@��C��q                                    BxvR��  �          @�G�@j�H����?�@�  C�j=@j�H��33=�G�?�{C��H                                    BxvR�0  
�          @���@o\)��
==#�
?#�
C��f@o\)��33���
��G�C�                                    BxvR��  �          @\)@o\)��논#�
��C�Ф@o\)�����Q���(�C�!H                                    BxvR�|  T          @}p�@vff�333>��H@�{C�ٚ@vff�L��>���@���C��                                    BxvR�"  �          @�Q�@~�R���=�\)?��
C��@~�R��논����C��                                    BxvR��  T          @���@qG���G�������C���@qG����ͿG��3�
C��                                    BxvS
n  
(          @��@w
=������G�C���@w
=���\�=p��((�C��
                                    BxvS  
�          @���@aG���z�J=q�7\)C�e@aG���zῗ
=���C�R                                    BxvS'�  T          @��@@���0��>��@��HC�Ǯ@@���2�\�k��L(�C���                                    BxvS6`  �          @�33@W��z�>B�\@)��C��q@W���\�Ǯ��  C���                                    BxvSE  �          @�=q@Tz��>.{@
=C�aH@Tz���
��
=��p�C���                                    BxvSS�  T          @�G�@P  ��#�
��C�q@P  ��׿�R���C���                                    BxvSbR  8          @���@:�H�%���H��33C�@ @:�H��ÿ�{��G�C�c�                                    BxvSp�  
�          @|��@�333��=q����C�^�@����R���C��q                                    BxvS�  �          @tz�?����>{��z���C��H?�������%��+��C���                                    BxvS�D  j          @u@���6ff������C�s3@����H�33�Q�C��)                                    BxvS��  p          @vff@HQ���
��Q���
=C�K�@HQ��z�\(��R=qC�H�                                    BxvS��  
�          @w
=@6ff?�\)@�B
=qA�@6ff?L��@��Bz�Ayp�                                    BxvS�6  T          @xQ�@%�?�@�RB�B��@%�?��H@&ffB)�A�Q�                                    BxvS��  �          @xQ�@{?�
=@)��B0{A�p�@{>��H@7
=BB(�A4                                      BxvSׂ  T          @tz�@%>�{@.{B8�H@��@%����@.{B8��C�P�                                    BxvS�(  T          @x��@�L��@H��Ba�C�Q�@�Q�@B�\BVQ�C�Q�                                    BxvS��  �          @y��@!G�?��@9��B@��AV=q@!G���@=p�BF33C��3                                    BxvTt  
�          @|(�@3�
>u@0  B1{@���@3�
��G�@.�RB/  C���                                    BxvT  	�          @|(�@C33�W
=@�B��C�O\@C33����@�
A���C���                                    BxvT �  �          @���@R�\>�p�@G�B	�@�\)@R�\�B�\@�\B�C�]q                                    BxvT/f  �          @��@Z�H>��H@�A��A\)@Z�H��@�B�RC���                                    BxvT>  �          @�33@Z=q?�@p�B\)A��@Z=q���
@G�B�C���                                    BxvTL�  �          @|��@{�(�@J�HBW�
C�C�@{��{@:�HBA33C�<)                                    BxvT[X  �          @w�@  ���@8��BE�C�g�@  ��p�@#33B'C�C�                                    BxvTi�  �          @w
=?�\��Q�@3�
B;C�/\?�\�#�
@��B�HC�S3                                    BxvTx�  
�          @{�@{��Q�@�RB�HC���@{�\)@   A�=qC��                                    BxvT�J  T          @xQ�@5��(�@B33C�XR@5��  ?�(�A���C�/\                                    BxvT��  �          @y��@4z῕@��B��C��f@4z���H@33A���C�g�                                    BxvT��  �          @z=q@>�R��=q@
�HB33C���@>�R��?��
A�z�C�XR                                    BxvT�<  �          @z=q@N{���H?У�A�z�C��f@N{��?�p�A�  C�N                                    BxvT��  �          @xQ�@33����@\)B!(�C���@33��?�p�A��C�R                                    BxvTЈ  �          @w�@����@&ffB(��C�~�@�!G�@�
B �C�޸                                    BxvT�.  �          @xQ�@����@
�HB�C��)@��6ff?��
A���C�U�                                    BxvT��  �          @tz�?����/\)?�{A�C���?����G
=?�33A�C�1�                                    BxvT�z  �          @|(�@{�C33?�=qA�ffC��@{�N{>���@���C�G�                                    BxvU   
�          @}p�@�E?O\)A<  C��q@�L(�=L��?333C�*=                                    BxvU�  �          @w
=@G��(��?�z�A�\)C��R@G��AG�?�p�A�C��f                                    BxvU(l  	�          @w�?�
=�6ff?޸RA��C��?�
=�K�?�G�Ar�RC���                                    BxvU7  
�          @w�@��A�?�(�A��\C���@��N�R>��@�G�C�3                                    BxvUE�  �          @u�?��>{?��A�{C��q?��P  ?J=qA?33C��)                                    BxvUT^  �          @vff@
=�6ff?��HA��C�=q@
=�G�?@  A4  C��                                    BxvUc  T          @w�@�\�3�
?��A�ffC�� @�\�B�\?(�A��C���                                    BxvUq�  T          @xQ�@*�H�+�?Tz�AE�C�t{@*�H�333>.{@$z�C��{                                    BxvU�P  �          @x��@2�\�#33?L��A@(�C�Ǯ@2�\�*�H>8Q�@*�HC�"�                                    BxvU��  �          @x��@W���Q�=���?�
=C�@W���zᾸQ���33C�8R                                    BxvU��  T          @u@Dz���\=��
?�  C���@Dz��  ���ۅC��f                                    BxvU�B  �          @u�@8���{=��
?�  C���@8����H�   ��C��                                    BxvU��  T          @s33@7��p������HC��\@7���ÿ(��Q�C�!H                                    BxvUɎ  T          @r�\@1��%�#�
�
=qC���@1��   �#�
��C���                                    BxvU�4  "          @q�@�:�H>���@���C�^�@�;��������C�P�                                    BxvU��  �          @s�
@Q��<��>���@���C�t{@Q��<(���������C���                                    BxvU��  "          @tz�@ ���4z�L���AG�C��3@ ���,�ͿY���P��C�xR                                    BxvV&  
�          @s33@5��\)�����C�T{@5���
���
�|z�C�`                                     BxvV�            @q�@��333?(��A"{C�u�@��8Q�#�
�\)C�3                                    BxvV!r  �          @q�@.�R� ��?�A
=C�� @.�R�#�
��\)�}p�C�o\                                    BxvV0  �          @r�\@:�H������y��C�u�@:�H�\)�L���Dz�C�7
                                    BxvV>�  
Z          @r�\@(Q���?z�HAt(�C�� @(Q��%>\@�ffC���                                    BxvVMd  T          @p  @)���!G�?G�AAp�C�>�@)���(Q�>8Q�@-p�C��q                                    BxvV\
  �          @p  @&ff�&ff?
=qAG�C�� @&ff�)�����
��p�C�4{                                    BxvVj�  T          @q�@"�\�#�
?J=qAD��C�b�@"�\�*�H>.{@)��C��                                    BxvVyV  T          @s�
@�
�4z�>��@˅C�� @�
�5�������C��=                                    BxvV��  "          @w
=@E����H?���A�  C��
@E��
�H?&ffA(�C�`                                     BxvV��  �          @u@<(��
=q?�33A�(�C��H@<(��
=?�RA��C���                                    BxvV�H  T          @vff@8���?��\At��C�~�@8��� ��>�ff@�C��                                     BxvV��  "          @xQ�@7
=��?�\)A�(�C�&f@7
=�$z�?
=qA z�C��                                    BxvV  "          @x��@:�H��?���A�  C��@:�H��R?z�A��C��{                                    BxvV�:  �          @xQ�@0���(��?&ffA�\C�(�@0���.{<�>�p�C��q                                    BxvV��  T          @xQ�@!G��%�?:�HA733C�(�@!G��+�=�?���C���                                    BxvV�  "          @{�@!G���?�ffA��HC��)@!G��+�?��HA�C��)                                    BxvV�,  �          @u@"�\�,(�?}p�Ap��C��{@"�\�6ff>�33@��C��                                    BxvW�  
�          @vff@p��7�?��A���C���@p��E?��A
=C��f                                    BxvWx  
�          @w
=@
=�G�?=p�A1��C�
=@
=�L��    =uC��                                    BxvW)  T          @u�@��E�?z�A(�C��@��H�þ\)�ffC�k�                                    BxvW7�  �          @xQ�@ff�L��?�A��C��
@ff�P  �.{�%�C�ff                                    BxvWFj  T          @~{@��Vff���Ϳ\C�޸@��O\)�^�R�K�C�P�                                    BxvWU  
�          @~{?�z��\��>��@  C�s3?�z��Y���#�
���C��f                                    BxvWc�  �          @\)?�
=�k���p�����C���?�
=�`  ������p�C��                                    BxvWr\  "          @|��?Y���n�R�Tz��B�RC�t{?Y���\(���z���{C���                                    BxvW�  T          @}p�?����mp��u�L��C�˅?����fff�fff�S\)C�3                                    BxvW��  �          @���?�=q�k������C��H?�=q�]p�������C�E                                    BxvW�N  �          @�  ?�  �W����R����C���?�  �@  ��p�����C�'�                                    BxvW��  "          @|(�?�=q�4z�� ������C�z�?�=q��\�&ff�&(�C�N                                    BxvW��  �          @|(�?����Mp��G��F{C�?����<�Ϳ��R����C��                                    BxvW�@  �          @z=q?�(��Vff?�Ap�C�� ?�(��X�þW
=�L��C���                                    BxvW��  T          @{�?Q��mp��#�
���C�9�?Q��]p����H���RC��                                    BxvW�  
�          @y��>���E����\)C�j=>���{�>{�HC���                                    BxvW�2  �          @x��?�33�c33�G��<��C��\?�33�Q녿�����  C��q                                    BxvX�  
�          @w�?p���i���O\)�C
=C�B�?p���W���\)����C���                                    BxvX~  
�          @z�H@ ���J�H�c�
�U��C�8R@ ���8Q�˅��{C�w
                                    BxvX"$  
�          @y��?���l�;W
=�G
=C�.?���c�
����zffC�}q                                    BxvX0�  
�          @s�
@��1�?��A�=qC��@��<��>\@�33C�K�                                    BxvX?p  �          @r�\@-p��p�?�33A�G�C�h�@-p��{?^�RAU��C��{                                    BxvXN  �          @{�@Dz��(�?�G�Ao\)C�=q@Dz��
=>��H@��C�5�                                    BxvX\�  
�          @�G�@O\)��?�Q�A��C���@O\)�33?0��A{C�T{                                    BxvXkb  
�          @��
@\(���p�?�\)A�33C�  @\(���\)?��A���C�XR                                    BxvXz  �          @��@j=q�ٙ�?5A ��C��@j=q��>��R@��
C��\                                    BxvX��  	�          @�33@K��\)?0��A  C��)@K��%�=�G�?���C�y�                                    BxvX�T  �          @�33@Tz��
=>�@�{C�N@Tz�������
��
=C�\                                    BxvX��  �          @��\@O\)���>��@l��C���@O\)�����z����\C���                                    BxvX��  �          @��\@U���ͿJ=q�2{C�<)@U���(���G����\C��                                    BxvX�F  
�          @�=q@XQ��Q����أ�C�� @XQ��(��p���X��C��                                     BxvX��  �          @��
@7
=�<(����
��z�C�q@7
=�7
=�.{��C���                                    BxvX��  "          @��@c�
���H?
=q@�z�C���@c�
��=�Q�?�G�C��                                    BxvX�8  
�          @�=q@XQ��(�?�@陚C��@XQ��  <#�
>\)C�*=                                    BxvX��  T          @�33@Z�H�p�>�
=@�(�C���@Z�H�  ��Q쿝p�C�]q                                    BxvY�  �          @�(�@dz��   >�@�p�C�\)@dz��33    �#�
C��                                    BxvY*  �          @�p�@j�H��
=>��
@��RC�R@j�H���H�����C��{                                    BxvY)�  �          @��@X���{�k��QG�C�ff@X���
=�8Q��$Q�C�                                    BxvY8v  T          @��@h�ÿ�녾8Q��"�\C�B�@h�ÿ�ff�����HC�Ф                                    BxvYG  T          @�33@e����H�8Q��!G�C���@e���\)��R�
�\C�33                                    BxvYU�  �          @��@K��������\)C���@K���ÿ���w33C���                                    BxvYdh  T          @�33@�\�<�Ϳ�
=����C��3@�\�\)�33�p�C�K�                                    BxvYs  
�          @��\@��<�Ϳ޸R��\)C�<)@���R�
=��C���                                    BxvY��  
Z          @�33@33�9����\��\)C��)@33���(��� �\C�j=                                    BxvY�Z  T          @�(�?�{�?\)��
��C�&f?�{����:�H�5Q�C���                                    BxvY�   �          @��?�z��C�
����C�j=?�z��(��C33�=�C��                                    BxvY��  �          @�{?Ǯ�G���\�(�C�C�?Ǯ�!��;��2z�C���                                    BxvY�L  T          @�{?��@������C�=q?�����333�(33C�
=                                    BxvY��  "          @�?��>{��\�(�C��H?��Q��9���/C���                                    BxvY٘  �          @�z�@33�4z��p��33C��@33����1��)ffC�(�                                    BxvY�>  �          @��@��*�H�ff�{C��{@����8���3ffC�                                      BxvY��  �          @�z�@���(����
ffC��H@���33�7��0�\C�)                                    BxvZ�  T          @�z�?��R�"�\�'
=�C�
?��R����G
=�B{C�Ff                                    BxvZ0  �          @��?���)���#33�(�C�?��� ���E��@=qC���                                    BxvZ"�  T          @���?�G��C�
��	�HC�&f?�G��p��>{�7(�C���                                    BxvZ1|  T          @��?����H�'��"�C��{?�׿�\�E�Hz�C�^�                                    BxvZ@"  T          @��H@{�G��#33��C�(�@{��33�?\)�=
=C��R                                    BxvZN�  �          @��
@ff�
=�=q��C�y�@ff��G��7��1z�C���                                    BxvZ]n  �          @��
@7
=�   ��� {C�~�@7
=���H�$z���
C�o\                                    BxvZl  �          @�(�@333��p����G�C�W
@333��
=�*=q�!�C���                                    BxvZz�  �          @�z�@@�׿��
�(�� �C��=@@�׿�  �!��\)C��
                                    BxvZ�`  T          @���@Fff��p��������C�b�@Fff���H�{��C�Q�                                    BxvZ�  �          @�(�@Fff���H����\)C���@Fff��Q�������C�w
                                    BxvZ��  �          @�p�@]p����׿���֏\C��@]p��n{�����\)C�}q                                    BxvZ�R  �          @�@b�\��
=�����(�C�@b�\�:�H�ff���C�&f                                    BxvZ��  �          @�@p  �+���
=���RC��f@p  ��\)������
C�ٚ                                    BxvZҞ  �          @�{@�  >\)���
�g�
@�@�  >�p��z�H�Z�R@���                                    BxvZ�D  "          @�z�@s�
<���ff��p�>�(�@s�
>�p���G����@��                                    BxvZ��  
�          @��@n�R?E���{��p�A:�\@n�R?�{��33��(�A��\                                    BxvZ��  �          @�{@x�þaG���p���  C�h�@x��=����R��G�?�                                      Bxv[6  T          @���?�����p��7��?  C�1�?�����ff�L(��\�
C��H                                    Bxv[�  
          @���@
=q�����=p��?�C���@
=q�c�
�O\)�XC�                                    Bxv[*�  	�          @��\@G��!�����p�C��3@G����H�1G��*�RC���                                    Bxv[9(  �          @��@ ���8�ÿ������
C�w
@ ���   ��
��p�C���                                    Bxv[G�  �          @�@{�AG�����\)C�&f@{�!��{�
=C���                                    Bxv[Vt  �          @�z�@{�1G��
=��C�Z�@{��R�*�H�!z�C�q�                                    Bxv[e  �          @�z�@�\�(��{�C��{@�\�����<���4�C��3                                    Bxv[s�  �          @�(�@*�H���
�%��C�+�@*�H��z��:�H�4\)C�N                                    Bxv[�f  �          @��ý��7��#�
�'  C��ͽ��{�H���Z��C�t{                                    Bxv[�  �          @��H?����
=�@  �C{C�E?��ÿ���Y���iz�C�5�                                    Bxv[��  �          @��
?�=q��
=�HQ��F��C��?�=q��Q��_\)�h
=C���                                    Bxv[�X  �          @�33?�Q��ff�9���5�\C�^�?�Q쿳33�R�\�X{C��                                    Bxv[��  �          @��@
=�  �
=��C�*=@
=���333�0�C�aH                                    Bxv[ˤ  �          @�33@  �
=�,(��$\)C�q�@  �����E�D\)C���                                    Bxv[�J  �          @���?˅��R�=p��={C�� ?˅��G��XQ��dQ�C�>�                                    Bxv[��  �          @�G�?����
�H�@  �@
=C�:�?��Ϳ�Q��Z=q�f��C��q                                    Bxv[��  �          @���?����{�O\)�U(�C��R?������dz��x�C�k�                                    Bxv\<  �          @�{?�
=��z��[��\�RC�b�?�
=����qG���C�C�                                    Bxv\�  �          @��?�p���33�s�
��
C���?�p���\����z�C��                                    Bxv\#�  �          @�z�?���Fff���H��
=C��=?���%��'
=��C�{                                    Bxv\2.  �          @�(�?�{�:�H��H�33C�p�?�{�33�AG��<G�C�}q                                    Bxv\@�  �          @��
?����8���p��33C�>�?�������B�\�?p�C�Y�                                    Bxv\Oz  �          @�(�?����7��0���'�C�t{?������U��W  C�Z�                                    Bxv\^   �          @�z�?�(��8Q��-p��#�
C�s3?�(�����R�\�R�\C�|)                                    Bxv\l�  �          @��
?����7
=�.�R�&�C���?������S33�V33C�ff                                    Bxv\{l  �          @���@Q��B�\��(����C��@Q��,(���\)��C��f                                    Bxv\�  �          @���@���@  ������{C��)@���*�H���
����C�B�                                    Bxv\��  
�          @��
?5��\�_\)�hC���?5���H�w
={C�,�                                    Bxv\�^  �          @��
?E��{�L(��K��C���?E���Q��j=q�|z�C�,�                                    Bxv\�  �          @�(�?�=q��H�J=q�H  C�  ?�=q��33�g��u�
C���                                    Bxv\Ī  �          @��?�
=��J=q�I�\C�]q?�
=�����g
=�u��C�u�                                    Bxv\�P  �          @��?u�.�R�:=q�4ffC��?u� ���\���d�\C��f                                    Bxv\��  �          @�p�?�G��8���*=q� �C�˅?�G��{�P  �O(�C��3                                    Bxv\�  �          @�{?B�\�/\)�C�
�<G�C��=?B�\��p��fff�m��C���                                    Bxv\�B  �          @�Q�?.{�7��Dz��8�HC��3?.{�ff�h���k{C��                                    Bxv]�  �          @���>��H�333�Mp��A�HC���>��H� ���p���t��C��
                                    Bxv]�  �          @�=q?E�����`  �X�RC���?E���ff�|���{C�9�                                    Bxv]+4  �          @���?�� ���p  �s��C���?녿�{���8RC��H                                    Bxv]9�  �          @�  ?
=q�33�j=q�oC�Y�?
=q�������\C�XR                                    Bxv]H�  �          @�
=>�
=�����n{�w=qC�&f>�
=�������C��                                    Bxv]W&  T          @�  ?8Q�����p���x��C��
?8Q�s33���\W
C���                                    Bxv]e�  �          @���?E���(��u��33C��?E��O\)���
��C��q                                    Bxv]tr  �          @�?�p����>�R�;z�C���?�p��У��\(��ez�C�3                                    Bxv]�  �          @��H@�
�*=q��
�	��C���@�
�z��6ff�1Q�C�aH                                    Bxv]��  �          @�33@	���>{�����C��=@	���\)�=q��C�`                                     Bxv]�d  �          @�33?��B�\�
�H� 33C�\)?��{�333�-
=C�                                      Bxv]�
  �          @�33?�33�3�
��H�z�C�4{?�33�(��@  �>=qC��H                                    Bxv]��  �          @�33?���'��"�\�(�C��3?�׿�(��Dz��AC�˅                                    Bxv]�V  �          @��\?����;������
C�'�?�����H�!���C��)                                    Bxv]��  �          @��@
=�p����33C��@
=�����7
=�.�C��                                    Bxv]�  �          @�ff@Fff����!G��z�C��\@Fff�.{�/\)�#p�C���                                    Bxv]�H  �          @�z�@Z�H��{�   ���C��@Z�H��R����=qC��)                                    Bxv^�  T          @��@l(��#�
�ٙ����HC�@l(��u��ff��p�C�#�                                    Bxv^�  �          @��
@r�\�z��  ����C��)@r�\�k��˅���C�Ff                                    Bxv^$:  �          @�  @s�
�fff�У����\C�Y�@s�
��\������
C�8R                                    Bxv^2�  �          @��R@hQ쿳33�˅���C�y�@hQ쿀  ��\)�ң�C�P�                                    Bxv^A�  �          @�(�@a녿�{�������\C���@a녿�  ���H��=qC�G�                                    Bxv^P,  �          @�(�@c�
��(���z���G�C�q@c�
��z���
���C�8R                                    Bxv^^�  �          @��
@`�׿��
��Q����HC���@`�׿��H������G�C���                                    Bxv^mx  �          @�33@^�R��{����v�RC��@^�R�Ǯ���R��Q�C��f                                    Bxv^|  �          @��@[���33������ffC���@[����ÿ�{��ffC��3                                    Bxv^��  �          @��H@U��  �+��
=C��)@U���\�����C�H�                                    Bxv^�j  �          @��H@S33�녿L���4z�C��f@S33��\��ff����C�#�                                    Bxv^�  �          @��
@^{��p���  �ap�C�(�@^{��Q쿷
=����C��q                                    Bxv^��  
(          @��@U����z���p�C��@U���\��{����C��                                    Bxv^�\  T          @�z�@e��p�����z�C���@e�������h  C���                                    Bxv^�  �          @��@tz��\)=L��?@  C���@tz�˅���
��C��
                                    Bxv^�  �          @��@i����>�  @b�\C��R@i�����.{�=qC��=                                    Bxv^�N  �          @�\)@dz��\)��=q�n�RC��{@dz��
=�G��+33C��                                    Bxv^��  �          @��R@`  ��
�.{�C�@ @`  �p��333��C��
                                    Bxv_�  �          @�  @vff���
>��@�\C���@vff��G�����dz�C��\                                    Bxv_@  �          @��R@o\)��33>�Q�@���C��=@o\)��
=��Q쿝p�C�Y�                                    Bxv_+�  �          @�z�@b�\��>\)?�C���@b�\��
��33���C��\                                    Bxv_:�  �          @��@P�������(�C�'�@P���
�H���\�hz�C�4{                                    Bxv_I2  �          @��
@S�
���=�Q�?�(�C��@S�
�ff�����z�C�E                                    Bxv_W�  �          @�33@\(���R>k�@I��C��f@\(��{������p�C��3                                    Bxv_f~  �          @��@Z=q��\=���?�{C�
=@Z=q�  ��G���(�C�Ff                                    Bxv_u$  T          @��H@]p��Q�>���@���C�*=@]p���þW
=�<��C��                                    Bxv_��  �          @���@K��   ����(�C��f@K���ÿ:�H�'
=C�~�                                    Bxv_�p  �          @�G�@>{�,��>���@�p�C�޸@>{�-p��u�\��C��                                    Bxv_�  �          @���@{�>�R?�AG�C��@{�A녾�����C��3                                    Bxv_��  �          @��@Fff���H?�{A�G�C��)@Fff��?�{A�  C���                                    Bxv_�b  �          @~�R@(���ff@�\A��
C���@(���!G�?�  A�ffC�'�                                    Bxv_�  �          @|(�@=q�0  ?�  As�C��f@=q�:=q>��
@���C��q                                    Bxv_ۮ  �          @�G�?�33�0�����ffC�C�?�33���-p��,p�C�~�                                    Bxv_�T  �          @�33?�
=�*�H�#33�C��?�
=�   �G
=�G�
C�                                    Bxv_��  �          @�Q�?���p��?\)�1�
C�?����Q��^�R�Z�C�`                                     Bxv`�  �          @�p�?��H�'
=�#33���C�o\?��H��Q��E�A33C���                                    Bxv`F  �          @�
=?����  �C33�9z�C��\?��Ϳ�(��`  �`
=C���                                    Bxv`$�  �          @���@Dz�L��>��A33C�+�@Dzᾙ��>��@��HC�1�                                    Bxv`3�  T          @���@E?��
@(�B�A�G�@E?(��@*�HB z�AA�                                    Bxv`B8  �          @��
@e?�?�
=A�  AG�@e=L��@   A���?^�R                                    Bxv`P�  �          @�G�@e��?aG�AQp�C��@e����?�AQ�C���                                    Bxv`_�  �          @�Q�@h�ÿ�G�?@  A-G�C�@h�ÿ��>�p�@��
C��)                                    Bxv`n*  �          @�G�@r�\����>u@\(�C�  @r�\��33��Q쿦ffC��H                                    Bxv`|�  �          @�G�@x�ÿ���>��@�
C�Y�@x�ÿ��ý�G��˅C�S3                                    Bxv`�v  �          @���@s�
���;\)��p�C�=q@s�
���
���ϮC���                                    Bxv`�  �          @���@w
=��zᾔz���=qC��f@w
=��������z�C�XR                                    Bxv`��  �          @���@��\�B�\�#�
�{C���@��\�333��{���C�'�                                    Bxv`�h  �          @��
@�녿+��#�
��\C�XR@�녿(����
��{C��                                     Bxv`�  �          @�(�@�Q�c�
��ff�ȣ�C���@�Q�@  �&ff���C��                                    Bxv`Դ  �          @���@�G��\(���
=���HC��{@�G��=p���R���C���                                    Bxv`�Z  �          @�@����p�׾�ff��
=C�s3@����O\)�+���RC�^�                                    Bxv`�   T          @�ff@��\�aG��Ǯ����C���@��\�E��
=� ��C���                                    Bxva �  �          @�33@\)�aG����
��  C��f@\)�G������HC�|)                                    BxvaL  �          @�(�@����B�\��Q�����C��{@����&ff�����C�t{                                    Bxva�  �          @�{@�(��녾Ǯ����C�\@�(�����\��G�C�Ф                                    Bxva,�  �          @�ff@�{��  �#�
�.{C�H�@�{�u��\)�}p�C�Z�                                    Bxva;>  �          @�ff@�p���ff>#�
@�RC���@�p���=L��?.{C��)                                    BxvaI�  �          @�{@�p�����>�\)@xQ�C�U�@�p��#�
>�  @\(�C��                                    BxvaX�  �          @���@���#�
>�ff@���C��f@����>�(�@�G�C�+�                                    Bxvag0  �          @��@�ff��z�?   @�\)C��@�ff����>��@��\C�G�                                    Bxvau�  �          @�\)@���  ?(�AG�C�J=@��Ǯ?�@�  C�S3                                    Bxva�|  �          @�  @�{=u?.{A�\?c�
@�{����?+�A��C�N                                    Bxva�"  �          @�  @�
=�B�\>�\)@s�
C��{@�
=��  >k�@Dz�C�G�                                    Bxva��  �          @���@�{�+�>�G�@�C�p�@�{�B�\>��@b�\C��)                                    Bxva�n  
�          @���@��
�@  ?aG�A>=qC���@��
�p��?+�Az�C���                                    Bxva�  �          @�33@��׾Ǯ?5A\)C�Z�@��׿\)?��@�(�C�Ff                                    Bxvaͺ  �          @�p�@�33��>�G�@�  C��@�33��R>���@x��C��{                                    Bxva�`  �          @��R@��Ϳ(��=L��?(�C�� @��Ϳ&ff����p�C��                                    Bxva�  �          @�p�@���.{>8Q�@Q�C��@���5<#�
>�C�ff                                    Bxva��  �          @���@�33���>L��@*=qC�{@�33�!G�=L��?(��C��                                     BxvbR  �          @�@��\�5?��@�G�C�T{@��\�W
=>��@���C��                                    Bxvb�  �          @��R@�33�h��>\@��C��@�33�z�H>\)?�C���                                    Bxvb%�  �          @�\)@�=q��\)>�  @L(�C���@�=q��33�#�
���C��\                                    Bxvb4D  �          @�\)@�G���(�>���@�{C��@�G����\<#�
>8Q�C���                                    BxvbB�  �          @��R@�
=��
==��
?}p�C��f@�
=��33��\)�fffC��                                    BxvbQ�  �          @�@����  �#�
�.{C�
@�������Ǯ��  C�e                                    Bxvb`6  �          @�(�@��H��ff>�?޸RC��
@��H�������XQ�C���                                    Bxvbn�  �          @��@vff��p�>��@��C�k�@vff�G����Ϳ�  C�/\                                    Bxvb}�  �          @�p�@mp��Q�L�Ϳ(�C��@mp���\�#�
�p�C�(�                                    Bxvb�(  �          @���@j�H�ff�#�
�	G�C�q@j�H��\)����w�
C�z�                                    Bxvb��  �          @�\)@]p���Q쿼(�����C�c�@]p���G���z���ffC�<)                                    Bxvb�t  �          @�\)@R�\����z���
=C��=@R�\������H���C�
=                                    Bxvb�  �          @���@W
=���\�33��\C���@W
=�&ff�"�\��C��f                                    Bxvb��  �          @�Q�@\(��@  �
=�{C�� @\(����{�z�C���                                    Bxvb�f  �          @�Q�@y���B�\������C���@y��>u�����R@c33                                    Bxvb�  �          @�ff@u���G��ٙ����C�33@u�>��
����@��H                                    Bxvb�  �          @��@p�׽u�����HC��@p��>�(�������@У�                                    BxvcX  �          @���@a녿���������HC��{@a녾��H���C��                                    Bxvc�  �          @���@w
=�   ��p����C�P�@w
=�#�
��ff��  C���                                    Bxvc�  T          @�ff@}p���G���{����C�9�@}p�>u�������@_\)                                    Bxvc-J  T          @��R@|(���\)�����(�C���@|(�=�\)��\)��G�?��                                    Bxvc;�  �          @��@�=q>�G���ff���H@�p�@�=q?B�\��33�w\)A(z�                                    BxvcJ�  �          @��\@�G�?(������  AQ�@�G�?p�׿�Q���AQG�                                    BxvcY<  �          @�=q@���>�G���  �W
=@\@���?.{�Y���7
=AG�                                    Bxvcg�  �          @��@���?�R�}p��Tz�A�
@���?Y���L���*�RA9��                                    Bxvcv�  �          @��H@�z�?k��aG��;�
AI�@�z�?�\)��R��RAq�                                    Bxvc�.  �          @�\)@�z�=u�����?O\)@�z�>W
=�\)��\)@:=q                                    Bxvc��  �          @���@�ff�W
=�=p��
=C��{@�ff�#�
�E��%G�C��=                                    Bxvc�z  �          @�Q�@���J=q��Q����HC��f@���+������p�C�q�                                    Bxvc�   �          @�Q�@���������\���
C�� @������
�L���-C��H                                    Bxvc��  �          @���@\)���ÿG��(��C��
@\)���ÿ����m�C�xR                                    Bxvc�l  �          @���@�  ��{�&ff��C��q@�  ��녿z�H�S�C��                                    Bxvc�  T          @���@|�Ϳ��R������C��@|�Ϳ��
�xQ��Q��C��                                    Bxvc�  �          @��@fff�z�>u@P��C���@fff�33��p�����C���                                    Bxvc�^  �          @��
@a��!�>B�\@��C�'�@a��\)�����HC�]q                                    Bxvd	  �          @�(�@c33� ��>�z�@vffC�Y�@c33�   ��p����C�h�                                    Bxvd�  �          @�(�@j�H���Q��-G�C��@j�H��녿�{���C�^�                                    Bxvd&P  �          @��\@]p��{��{��  C���@]p�����������C�Z�                                    Bxvd4�  �          @��H@Z=q�"�\�:�H���C��f@Z=q��׿������\C�:�                                    BxvdC�  �          @��\@^�R�"�\�\���HC���@^�R�ff����^{C��R                                    BxvdRB  �          @���@K��-p��L���-�C��=@K��=q���R����C�w
                                    Bxvd`�  �          @�G�@p  � �׾�����  C��@p  ���ͿY���8z�C��R                                    Bxvdo�  �          @��R@��ÿ�ff>�  @Y��C���@��ÿ��ý#�
��\C��=                                    Bxvd~4  �          @�
=@\)��\)?��AQ�C�*=@\)���R>�\)@y��C�aH                                    Bxvd��  �          @�Q�@�녾�33?�33A{
=C��=@�녿#�
?�G�A]G�C��                                    Bxvd��  �          @���@����u?��\A��\C�J=@�����?�z�A|Q�C�                                      Bxvd�&  �          @�\)@\)��(�?�{A�p�C��{@\)�E�?�Q�A�
=C���                                    Bxvd��  �          @��@z=q���
?�=qA���C���@z=q�:�H?�Q�A���C��3                                    Bxvd�r  �          @��R@s�
�8Q�?У�A��C��@s�
����?���A��
C�                                    Bxvd�  �          @�
=@u���(�?�A���C��{@u��\(�?�  A�
=C���                                    Bxvd�  �          @�\)@w
=�333?�  A���C��3@w
=��=q?�G�A�p�C�,�                                    Bxvd�d  �          @��@p����
>8Q�@
=C��@p���녾\��33C��R                                    Bxve
  �          @��
@u��>B�\@{C�� @u��
�\��
=C��f                                    Bxve�  �          @��
@~�R��  >\@��C�#�@~�R�����G�����C��=                                    BxveV  �          @��@��H��(�>��H@���C��@��H���>\)?��C�                                      Bxve-�  �          @���@�Q쿪=q?333Ap�C���@�Q쿼(�>��R@��C��R                                    Bxve<�  
�          @�Q�@}p����R?\)@��RC�J=@}p�����>L��@333C���                                    BxveKH  �          @���@y����  ?�G�A��\C�Ǯ@y�����?p��AMG�C��)                                    BxveY�  �          @���@[���ff@	��A��RC�:�@[����\?���A�{C��
                                    Bxveh�  �          @��@_\)>��H@   Bff@��@_\)�W
=@"�\B�
C�>�                                    Bxvew:  �          @���@e>�{@   B
=@�\)@e��33@   B
��C�4{                                    Bxve��  �          @�(�@l(�>�\)@�B �R@�G�@l(���p�@z�A��
C�#�                                    Bxve��  �          @�(�@p��=#�
@\)A�Q�?!G�@p�׿�@
=qA�\)C��=                                    Bxve�,  T          @�z�@hQ켣�
@p�BC���@hQ�333@
=Bp�C��                                    Bxve��  �          @�z�@^{���
@,(�B{C�` @^{�O\)@#�
B  C�j=                                    Bxve�x  �          @�z�@U>.{@5�B �H@:�H@U�(�@1G�B��C���                                    Bxve�  �          @�p�@U?#�
@4z�BffA,(�@U�.{@8Q�B"�HC��{                                    Bxve��  �          @�p�@x�þ.{@�\A���C��q@x�ÿ:�H?�z�A�(�C��{                                    Bxve�j  �          @�z�@g�=�@p�B��?���@g���@��B��C���                                    Bxve�  
�          @�(�@aG�>W
=@%�B�@X��@aG���\@!�B�RC��R                                    Bxvf	�  �          @���@l�ͼ�@�BffC���@l�Ϳ0��@��A�Q�C��q                                    Bxvf\  �          @��@l�ͽ�G�@�\A���C�"�@l�Ϳ=p�@
�HA�RC�P�                                    Bxvf'  �          @���@j=q�#�
@{A��
C��\@j=q�(��@�A�C���                                    Bxvf5�  �          @�
=@_\)=u@ffB�H?�  @_\)���@�B�HC�!H                                    BxvfDN  �          @�
=@XQ�#�
@!G�B�RC���@XQ�=p�@��B=qC��                                    BxvfR�  �          @���@S33=L��@ ��B�H?c�
@S33�&ff@�HB�C�c�                                    Bxvfa�  �          @�ff@c33�   @	��A�33C���@c33��=q?�Aڣ�C��f                                    Bxvfp@  �          @�@mp����R?�A�Q�C��R@mp��Q�?�
=A��\C���                                    Bxvf~�  �          @���@p  >�\)@G�A�{@��R@p  ���R@G�AᙚC���                                    Bxvf��  �          @�=q@l(�>u@\)A��@q�@l(�����@{A���C��=                                    Bxvf�2  �          @��H@R�\��\)@4z�B"ffC�g�@R�\�\(�@+�B(�C���                                    Bxvf��  �          @�33@`  ���@#�
B�\C���@`  �xQ�@Q�B{C�7
                                    Bxvf�~  
�          @�=q@g�����@�B�
C���@g��s33@	��A�
=C��q                                    Bxvf�$  �          @���@Tz�z�@*=qB  C�
@Tzῧ�@Q�B�C�@                                     Bxvf��  �          @�=q@a녿0��@�B�RC�n@a녿��@z�A�z�C��{                                    Bxvf�p  �          @�33@c33���@�\A��C���@c33��33?��A�z�C���                                    Bxvf�  �          @��@dzῦff@Q�A�=qC��{@dz��{?�A�  C�9�                                    Bxvg�  �          @�33@e���=q?��A�{C�!H@e��33?�{A��C��                                    Bxvgb  �          @�33@b�\��?�(�A��HC�@ @b�\���?�\)Ao
=C��R                                    Bxvg   �          @�33@Vff��?�ffA��HC�@Vff� ��?���AqG�C��=                                    Bxvg.�  �          @��@S�
�(�?�p�A�(�C�H�@S�
�%?��\AZ�\C���                                    Bxvg=T  �          @��@Vff��R?��
A�
=C��H@Vff�/\)?�\@׮C�W
                                    BxvgK�  �          @���@Y�����?��A��HC�  @Y���.{?
=q@�C���                                    BxvgZ�  �          @�p�@Vff���?�\)A�G�C�7
@Vff�0��?Y��A0��C�@                                     BxvgiF  T          @�{@N�R�+�?�Q�A��C�"�@N�R�>�R?��@���C��)                                    Bxvgw�  �          @�@W
=�&ff?�G�A�z�C�"�@W
=�7
=>�ff@�=qC���                                    Bxvg��  �          @���@N{�*�H?�=qA���C�+�@N{�<(�>��H@��
C���                                    Bxvg�8  �          @��@C�
�4zΐ33�vffC��\@C�
�
=����p�C�1�                                    Bxvg��  �          @�(�@333�Fff���R��=qC��@333�&ff�ff���C���                                    Bxvg��  �          @��
@(Q��H�ÿ��
��ffC��@(Q��#33����z�C���                                    Bxvg�*  �          @�(�@{�P  ��=q���
C�'�@{�$z��-p��z�C�k�                                    Bxvg��  �          @�(�@��\�Ϳ�����G�C��@��5��#33�  C�(�                                    Bxvg�v  �          @��@G��^�R��p����\C��=@G��8���(��ffC��                                    Bxvg�  �          @�@\)�j=q��ff�\z�C�� @\)�K����C��{                                    Bxvg��  �          @��
@�R�P  ��p���G�C�=q@�R�%�'
=���C�^�                                    Bxvh
h  �          @��\?ٙ��y���8Q���C��f?ٙ��j�H��=q���C�n                                    Bxvh  �          @��?�{�s33�����C���?�{�L���\)�G�C�|)                                    Bxvh'�  �          @�  @(��Tz�=�G�?˅C�*=@(��L�ͿaG��E�C���                                    Bxvh6Z  T          @�
=@ ���^�R�O\)�733C��@ ���E������ҏ\C���                                    BxvhE   �          @�  ?�(��_\)��=q�x��C�R?�(��@������  C��q                                    BxvhS�  �          @�z�?��R�b�\=#�
?
=C��\?��R�X�ÿ��\�i��C�9�                                    BxvhbL  �          @�\)?�
=�p�׿E��*=qC�H?�
=�Vff��\)��{C�Ff                                    Bxvhp�  �          @�?����w���=q�p��C���?����W
=������HC��                                    Bxvh�  �          @��R?�ff�_\)�z���ffC�g�?�ff�,���AG��8�
C���                                    Bxvh�>  �          @���?�=q�`�׿�������C�� ?�=q�7��#�
���C�=q                                    Bxvh��  �          @�=q?˅�e���=q�w33C��R?˅�E�����\)C���                                    Bxvh��  �          @��H?�Q��_\)�������RC�8R?�Q��5�#�
��
C�l�                                    Bxvh�0  �          @��?�\)�c�
��{�ϮC���?�\)�5��7
=�){C��                                    Bxvh��  �          @�?�z��`�׿�Q���\)C�%?�z��0  �:�H�1ffC�o\                                    Bxvh�|  �          @�?W
=�g�����Q�C���?W
=�7
=�;��2(�C�(�                                    Bxvh�"  �          @��H?L���c33��{���HC�S3?L���3�
�7
=�1�RC��                                    Bxvh��  �          @��?O\)�N�R�$z��C��?O\)���\(��[ffC��)                                    Bxvin  �          @�?
=�J=q�*�H�(�C�T{?
=���aG��e(�C���                                    Bxvi  �          @�ff<#�
�N�R�*=q��C��<#�
�  �a��eC�&f                                    Bxvi �  �          @�ff=�\)�G��3�
�'��C���=�\)�ff�h���o��C��                                    Bxvi/`  �          @�33=#�
�B�\�0  �(�C�c�=#�
��\�c33�p�C���                                    Bxvi>  �          @��H��Q��Fff�%���C��
��Q��	���Z�H�f=qC�Ff                                    BxviL�  �          @�G��:�H�G��6ff�&��C�ff�:�H���k��l\)Cz��                                    Bxvi[R  �          @�=q���
�>{�=p��,�Czٚ���
��33�o\)�o��Cq�                                    Bxvii�  �          @��þ��H�7
=�H���<�HC�,;��H�޸R�w�  C~aH                                    Bxvix�  �          @�{�p���:�H�L���8��C|0��p�׿��
�|���|33Cr!H                                    Bxvi�D  �          @�
=����AG��Fff�/�RCz
�����33�x���rQ�Cp
=                                    Bxvi��  �          @�{�s33�8���N{�:�C{޸�s33��p��~{�~=qCqT{                                    Bxvi��  �          @�ff�Tz��AG��I���4ffC~��Tz����|(��y�RCv\                                    Bxvi�6  �          @������HQ��:�H�%�Cz�f�����33�p���i=qCq�                                    Bxvi��  �          @�p��k��E��A��-=qC}Y��k����H�vff�r\)Ct��                                    BxviЂ  �          @��Ϳu�O\)�4z��G�C}}q�u���l���d��Cv:�                                    Bxvi�(  �          @�G����R�B�\�(�����Cw�{���R�33�^{�]\)Cn�{                                    Bxvi��  
�          @�\)��{�H����H��Cv���{�p��R�\�O  Cn�                                     Bxvi�t  �          @��R���H�I���-p��!\)C��q���H�Q��dz��j\)C��                                    Bxvj  �          @��þ����C�
�4z��)�C�E�����   �i���s(�C�U�                                    Bxvj�  �          @��H�333�@  �A��233C�o\�333��\)�u��y��Cy��                                    Bxvj(f  �          @�33����:=q�Mp��>=qC�H��녿�(��~{�fC�Y�                                    Bxvj7  �          @��\��33�2�\�2�\�3(�C�h���33�޸R�b�\�}�C�O\                                    BxvjE�  �          @�33@_\)��>�@�G�C��H@_\)���L���333C��R                                    BxvjTX  T          @�=q@G
=�#33���ۅC�W
@G
=�  ���
��G�C�f                                    Bxvjb�  �          @�G�@:=q�.{�&ff�  C�}q@:=q�
=���R���C�y�                                    Bxvjq�  �          @�G�@!��>�R���
��C�'�@!��\)�)���=qC�4{                                    Bxvj�J  �          @�Q�@p��9����Q���  C�0�@p��
=�1G��"
=C��3                                    Bxvj��  �          @��
@#�
�9�������33C��R@#�
�z��
=q���
C��H                                    Bxvj��  �          @�@8���;���R���C�Q�@8���$z�������C�4{                                    Bxvj�<  �          @�
=@H���4z�u�E�C��@H���)���xQ��UG�C��                                    Bxvj��  �          @�Q�@C�
�=p�=#�
?�C��
@C�
�3�
�n{�J=qC���                                    BxvjɈ  �          @��@5�?\)�
=q��ffC��@5�(�ÿ��R��C���                                    Bxvj�.  �          @�@'��L(��B�\�$z�C��@'��<�Ϳ�(����C��f                                    Bxvj��  �          @�
=@:�H�?\)?   @�G�C�'�@:�H�?\)��\��  C�*=                                    Bxvj�z  �          @�@7��@  >L��@0��C��@7��9���L���1G�C�`                                     Bxvk   T          @�ff@6ff�C33=��
?��C��\@6ff�9���n{�N=qC�AH                                    Bxvk�  �          @�@A��1�?!G�A�C���@A��5����
��\)C�q�                                    Bxvk!l  �          @�ff@:=q�8Q�?Tz�A7\)C��q@:=q�@  �.{��C��                                    Bxvk0  �          @�@@���3�
?+�A  C�|)@@���7�������(�C�+�                                    Bxvk>�  �          @�z�@<���7�>��@�
=C��@<���6ff�����
C�                                    BxvkM^  �          @�p�@,���J�H����\C�>�@,���<(���
=��G�C�J=                                    Bxvk\  �          @���@��N�R��Q쿨��C�t{@��AG�����=qC�k�                                    Bxvkj�  �          @��@
=q�[���z���=qC�
@
=q�HQ쿵���C�E                                    BxvkyP  �          @��@{�[���
=��=qC�}q@{�E�����ffC��                                     Bxvk��  �          @�z�@O\)���R?�=qA��C��@O\)��
=?}p�Aj�HC���                                    Bxvk��  �          @�33@A��(��?(��A��C�|)@A��-p���=q�n�RC��                                    Bxvk�B  �          @�(�@K��!G�?
=A�C���@K��$zᾙ�����C���                                    Bxvk��  �          @�p�@R�\�   ��(����C�c�@R�\��Ϳ�  ��ffC��                                    Bxvk  T          @��@]p���\����
=C��@]p��޸R��Q���{C���                                    Bxvk�4  �          @�33@p�׿�{���Ϳ���C�o\@p�׿�p��#�
�
=C�E                                    Bxvk��  �          @��
@Z=q�{>��@��C�w
@Z=q��R��p���{C�l�                                    Bxvk�  �          @��@Mp���
����	�C�!H@Mp���(��������C�=q                                    Bxvk�&  �          @�Q�@>�R��ff����33C�q�@>�R�\(��:=q�,�C���                                    Bxvl�  �          @���@G
=��
=�=q�	(�C��{@G
=�@  �5�&G�C�4{                                    Bxvlr  �          @��@L�Ϳ��H�����
C���@L�Ϳ\(��(Q����C�q�                                    Bxvl)  �          @�
=@c33��녿�G����HC���@c33���\��p���\)C��                                    Bxvl7�  �          @�  @Mp���(��	����(�C��q@Mp��#�
�!G��33C�aH                                    BxvlFd  �          @��@�?\)��\)��  C��@�
=q�1��$C��f                                    BxvlU
  �          @��@ff�,(�����
=C���@ff�޸R�C33�8p�C���                                    Bxvlc�  �          @�Q�@6ff�����#�
��\C���@6ff�Tz��A��6�RC��                                    BxvlrV  �          @�Q�@5��=q����C��f@5����
�333�${C���                                    Bxvl��  
�          @�G�@$z��� ���p�C��)@$zῨ���J=q�>z�C�j=                                    Bxvl��  T          @���@p����0����
C�AH@p���=q�U�L��C�!H                                    Bxvl�H  �          @���@6ff��(��(Q��ffC�w
@6ff�0���C�
�9{C�/\                                    Bxvl��  �          @�Q�@7���Ϳ�(�����C���@7���{�-p���HC�Z�                                    Bxvl��  �          @�Q�@/\)�"�\�����C��R@/\)���2�\�#��C�U�                                    Bxvl�:  �          @��@E��ÿ�ff�ŮC�!H@E��{�!G����C�@                                     Bxvl��  �          @���@ff�L(�������C���@ff���8Q��*�C�
=                                    Bxvl�  �          @���@ff�A녿�
=���C���@ff�	���7��(�C���                                    Bxvl�,  �          @�G�@��6ff��\��ffC��)@������:=q�.{C�G�                                    Bxvm�  �          @�G�@
=�<�Ϳ��R��(�C�Q�@
=��
�:=q�+p�C�s3                                    Bxvmx  �          @�ff?�ff�Q녿�33���C�b�?�ff����<(��1��C�xR                                    Bxvm"  �          @�p�?�\)�mp���(���Q�C�j=?�\)�<���,(��!��C�l�                                    Bxvm0�  T          @�\)?���fff��ff�˙�C���?���.{�=p��4(�C�c�                                    Bxvm?j  T          @�  @%�1G�����33C���@%��
=�-p����C���                                    BxvmN  �          @���@Vff�
=q��  ����C���@Vff��p��
�H����C�\                                    Bxvm\�  �          @��R@P  �  �����C��=@P  �˅�Q���G�C��\                                    Bxvmk\  �          @�@)���7����H��=qC�XR@)���	�������C�l�                                    Bxvmz  �          @��@\)�7
=��z�����C���@\)���%���RC��                                    Bxvm��  �          @��?�p��P�׿�=q�ң�C�H?�p��Q��8Q��133C��                                    Bxvm�N  �          @�z�?У��R�\������{C�(�?У�����:=q�4p�C�                                    Bxvm��  �          @�{?���S33��=q��Q�C�:�?���,(�������C��3                                    Bxvm��  �          @��?�
=����@Q�BY=qC���?�
=�(�@$z�B(�C�/\                                    Bxvm�@  �          @�ff?�(����@o\)Bd�
C�T{?�(��*=q@>�RB(G�C�B�                                    Bxvm��  �          @��@�
��
=@HQ�B8ffC�\@�
�;�@p�A�G�C�)                                    Bxvm��  �          @�33@?\)�=q?�(�A�33C��
@?\)�=p�?uAMp�C���                                    Bxvm�2  �          @��H@:=q�!G�?��HA�G�C��@:=q�C�
?fffA@��C���                                    Bxvm��  
(          @��@)���  @(�BG�C��=@)���?\)?���A���C�                                    Bxvn~  �          @��@)���#33@\)A���C�
=@)���L(�?�33Ax  C��
                                    Bxvn$  �          @�(�@*=q�(Q�@G�A�p�C���@*=q�Q�?��ArffC���                                    Bxvn)�  �          @�33@H���=q?�ffA�G�C�=q@H���9��?J=qA&�RC��H                                    Bxvn8p  �          @�33@C33�   ?�Ař�C�Y�@C33�?\)?E�A#
=C�˅                                    BxvnG  �          @��@:=q�0  ?�(�A�{C�K�@:=q�L(�?
=@�\)C�+�                                    BxvnU�  �          @�(�@6ff�4z�?�G�A�=qC��f@6ff�QG�?��@�z�C��f                                    Bxvndb  �          @�z�@.{�=p�?�G�A�33C�T{@.{�Y��?
=q@�33C�\)                                    Bxvns  �          @��
@>{�,(�?�(�A��C��{@>{�HQ�?��@�p�C�                                    Bxvn��  �          @��@>�R��R?��RA���C�)@>�R�B�\?k�AC\)C�7
                                    Bxvn�T  �          @��H@?\)�=q@ ��A��HC���@?\)�?\)?xQ�AN�\C���                                    Bxvn��  �          @��\@<���   ?�33A���C���@<���A�?Tz�A0��C��                                    Bxvn��  m          @�(�@7
=�5?�Q�A�ffC���@7
=�QG�?�\@�
=C���                                    Bxvn�F  ;          @�33@AG��1�?��A�33C���@AG��Fff>�  @O\)C�                                      Bxvn��  �          @��@N�R�#�
?�A��C�Ǯ@N�R�:=q>�33@�z�C�                                      Bxvnْ  �          @�33@Tz����?�{A�ffC�˅@Tz��1�>�33@��C�                                    Bxvn�8  �          @�33@S�
�%?��ArffC��)@S�
�4z�=��
?���C��                                     Bxvn��  �          @��@U�(��?fffA>ffC���@U�1녾#�
���C�)                                    Bxvo�  �          @�33@Vff�\)?�Q�A~=qC��\@Vff�0��>#�
@�C�G�                                    Bxvo*  �          @��
@U�$z�?��AqC�=q@U�3�
=��
?���C���                                    Bxvo"�  �          @�=q@L���%?��A��C��H@L���8Q�>W
=@4z�C���                                    Bxvo1v  �          @���@U��&ff>�
=@��
C��@U��#�
����{C�:�                                    Bxvo@  �          @�G�@Z�H�(�?O\)A.�\C�C�@Z�H�#�
�8Q���C��{                                    BxvoN�  �          @�G�@Z�H�33?�z�Az�RC��@Z�H�$z�>B�\@%�C���                                    Bxvo]h  �          @���@W
=�=q?�
=A\)C�/\@W
=�+�>.{@�C���                                    Bxvol  �          @�33@Z=q��?�z�Aw�
C�AH@Z=q�,(�>\)?�33C��)                                    Bxvoz�  �          @��@_\)�Q�?�G�A�z�C�ٚ@_\)�+�>�  @P  C�=q                                    Bxvo�Z  �          @�(�@]p����?���A��C�p�@]p��'�>��@��C�w
                                    Bxvo�   �          @���@`  ��
?��A�G�C��@`  ��>��H@ӅC��
                                    Bxvo��  �          @��\@c33�G�?n{AG
=C��
@c33��ͼ#�
�B�\C���                                    Bxvo�L  �          @�G�@Z�H�   >�@���C���@Z�H�\)���޸RC���                                    Bxvo��  �          @��H@c�
��(�?�  A�=qC�� @c�
�33>Ǯ@�\)C��                                    BxvoҘ  �          @�=q@Y����
?Tz�A=��C�e@Y���{���\C�k�                                    Bxvo�>  �          @��
@b�\�=q�
=����C�޸@b�\�   ��p���p�C�J=                                    Bxvo��  �          @�=q@W
=�	����33��{C��@W
=����
=�p�C��                                    Bxvo��  �          @�Q�@S�
����Q���33C�o\@S�
��
=�   ��RC���                                    Bxvp0  �          @�\)@Tz��!G��+��=qC�k�@Tz���
�������C�q                                    Bxvp�  �          @�  @K��2�\�aG��:�HC�^�@K��\)������C�                                      Bxvp*|  
�          @���@QG��/\)�������C�f@QG��=q�������C��R                                    Bxvp9"  �          @�G�@~�R���
�^�R�<z�C�\@~�R�W
=��ff��=qC�
=                                    BxvpG�  �          @��@\)��{�h���C33C���@\)�c�
��\)��=qC��{                                    BxvpVn  �          @���@s�
��z�z�H�S\)C�4{@s�
��33��ff��ffC��H                                    Bxvpe  �          @���@z=q���H�n{�H��C���@z=q�z�H��
=��=qC��R                                    Bxvps�  �          @�G�@xQ쿸Q쿋��k\)C��\@xQ�fff�������
C��                                     Bxvp�`  �          @�G�@e�������ep�C��R@e����R�����=qC���                                    Bxvp�  �          @���@u���z�Y���8Q�C�J=@u���Q쿸Q���C�c�                                    Bxvp��  �          @���@}p���
=�L���+\)C��@}p���  ��ff��33C��                                    Bxvp�R  �          @�G�@{���\)��  �X��C�g�@{��\(����H��  C�ٚ                                    Bxvp��  �          @�33@|(���=q��G����RC���@|(��8Q��
=��ffC��3                                    Bxvp˞  T          @�33@|(���  �����C�/\@|(��(���p���p�C���                                    Bxvp�D  �          @�33@tz`\)�������C�&f@tz�.{�����=qC���                                    Bxvp��  �          @��\@\(��{�O\)�-��C�!H@\(���Q��p����C�B�                                    Bxvp��  �          @�Q�@E��:=q�Ǯ��\)C�P�@E�� �׿\��p�C�ff                                    Bxvq6  �          @���@P  �.�R�������C��)@P  �ff���H��C��                                    Bxvq�  T          @��H@dz��p��n{�Hz�C�&f@dz��33�޸R����C��H                                    Bxvq#�  �          @��H@n{�G��}p��R�\C��
@n{������(���z�C�O\                                    Bxvq2(  T          @�(�@k��  �.{�ffC�B�@k�������
����C��                                    Bxvq@�  �          @�(�@r�\�ff�����HC�~�@r�\���H������33C���                                    BxvqOt  T          @�33@p  �
=q�8Q���C�  @p  ��z῅��]p�C���                                    Bxvq^  �          @��
@p�׿����
�]G�C���@p�׿��\��Q���Q�C��3                                    Bxvql�  
�          @��@Z=q��=q�(��	Q�C�4{@Z=q    �*�H�Q�=#�
                                    Bxvq{f  T          @��@U��У��
�H���HC���@U��(��)���{C��{                                    Bxvq�  �          @��@_\)��p�������(�C��q@_\)�W
=�
=�Q�C�Ff                                    Bxvq��  �          @�G�@^�R��p������C�@^�R�Y������C�)                                    Bxvq�X  �          @���@7���G��8Q��*p�C�'�@7�<��H���>�?
=q                                    Bxvq��  
�          @�Q�@<(���z��5�'��C�C�@<(�=�G��Dz��8�@33                                    BxvqĤ  �          @�Q�@1G���{�2�\�$G�C��@1G������Mp��C�C��                                    Bxvq�J  
�          @��@-p����H�<(��.��C��H@-p����Q��I�C���                                    Bxvq��  
�          @���@-p���G��<���.�C�o\@-p��.{�S�
�Jz�C�=q                                    Bxvq�  T          @��@7���Q��5��%p�C��\@7�����J�H�?(�C���                                    Bxvq�<  
�          @��@ �׿&ff�`  �U��C���@ ��?Y���\���R{A�\)                                    Bxvr�  �          @��\@ �׿h���Vff�M�
C�3@ ��?\)�[��Tp�AH(�                                    Bxvr�  
�          @���@N�R����n{�PQ�C���@N�R��ff���ә�C�l�                                    Bxvr+.  T          @�=q@a��(�����p�C��\@a��
�H�����s�C�<)                                    Bxvr9�  �          @���@a��p���G��Y�C��q@a녿˅��=q�ʣ�C��                                     BxvrHz  �          @��@c�
�녿��
���RC�&f@c�
����� ������C���                                    BxvrW   �          @���@Z�H�"�\>�?�\C��@Z�H���p���K\)C��f                                    Bxvre�  �          @���@A��>{=�?�
=C���@A��0�׿����uC���                                    Bxvrtl  
�          @���@N{�333��Q���=qC�w
@N{�����  ����C��f                                    Bxvr�  
�          @�G�@J=q�2�\�8Q����C�C�@J=q�p���=q��
=C�
=                                    Bxvr��  T          @���@Z�H�"�\���
���C��
@Z�H�
�H�������
C���                                    Bxvr�^  �          @���@`  ��H����a�C��{@`  ����  ��z�C��H                                    Bxvr�  �          @���@dz��(��z�H�R{C�9�@dz��=q��ff�ƸRC��                                    Bxvr��  �          @�G�@e�����j=qC�� @e��Q��{���C��                                    Bxvr�P  �          @�G�@c�
��R�O\)�.�\C�H@c�
��
=����  C�c�                                    Bxvr��  �          @��@[���
���|z�C���@[���{��\��(�C�u�                                    Bxvr�  �          @�=q@Z�H�%�����\C�y�@Z�H�
=�������HC�!H                                    Bxvr�B  �          @��@hQ��33������33C��{@hQ��Q쿢�\���HC���                                    Bxvs�  T          @���@W��(�þL���-p�C���@W��33��ff����C�˅                                    Bxvs�  �          @���@S�
�,�;�Q���=qC�ff@S�
��\��(����\C��f                                    Bxvs$4  �          @���@`  ��;B�\�#33C���@`  ��ÿ��H���C�Q�                                    Bxvs2�  �          @���@hQ���R����G�C�=q@hQ������=q�k�C��                                     BxvsA�  �          @�G�@l�����33���C�@ @l�Ϳ�  ������\)C�]q                                    BxvsP&  �          @���@S�
�-p��u�L��C�\)@S�
�����H��p�C��                                    Bxvs^�  �          @���@W
=�,(����
��G�C���@W
=�����(���p�C�=q                                    Bxvsmr  �          @�G�@Mp��5��L�Ϳ!G�C�H�@Mp��"�\��  ��Q�C���                                    Bxvs|  �          @���@?\)�B�\=�\)?uC�B�@?\)�1녿��R��
=C��                                    Bxvs��  �          @�G�@K��5�����(�C�*=@K��   ��=q����C��                                    Bxvs�d  �          @���@L(��2�\?&ffA
=C�^�@L(��333��R�Q�C�U�                                    Bxvs�
  �          @�33@b�\�p�?�\@׮C��R@b�\�(������C��{                                    Bxvs��  �          @��@dz��?E�A#�C�^�@dz���;��R����C�˅                                    Bxvs�V  �          @��@\���z�?��\A�C�@\���(��>\)?�33C�L�                                    Bxvs��  �          @��H@H���
=?���A���C��H@H���:�H?�RA�C���                                    Bxvs�  �          @��
@O\)�1G�?h��A?�
C�Ǯ@O\)�8Q�\��Q�C�.                                    Bxvs�H  �          @�(�@P���1G�?h��A?33C���@P���8�þǮ����C�9�                                    Bxvs��  �          @���@Z�H�*�H?��@�C�  @Z�H�(�ÿ+��z�C�%                                    Bxvt�  �          @�{@L(��=p�?L��A&�RC���@L(��AG�����33C�O\                                    Bxvt:  �          @��R@S33�8��?J=qA"ffC�^�@S33�<�Ϳ�����
C�)                                    Bxvt+�  m          @�  @Z�H�333?��@�C�\)@Z�H�1녿.{��C�w
                                    Bxvt:�  �          @��R@j�H�   >�z�@r�\C��@j�H���Tz��+
=C���                                    BxvtI,  �          @�ff@\(��,��?W
=A-�C��\@\(��2�\��(�����C�t{                                    BxvtW�  �          @�@dz��$z�>��@��C�  @dz��!G��333�G�C�ff                                    Bxvtfx  �          @��
@_\)�(Q�#�
�\)C���@_\)���Q��|��C��                                    Bxvtu  �          @��
@\(��*=q��Q���ffC��@\(���R���R��C�z�                                    Bxvt��  �          @�ff@`���'
=������C���@`������Q����C��=                                    Bxvt�j  �          @��@Z=q�\)��  ��Q�C���@Z=q��
=�{��ffC���                                    Bxvt�  �          @�(�@S33��Ϳ�p���(�C���@S33��ff�=q��C�p�                                    Bxvt��  �          @��H@H���G�������
C�\@H�ÿ��H�,����HC��H                                    Bxvt�\  �          @��H@Q���ÿ����C���@Q녿�(��(��ffC���                                    Bxvt�  �          @�33@G
=�$z������\)C�4{@G
=�����%���\C�h�                                    Bxvtۨ  �          @�(�@H���)����  ��ffC��@H�ÿ��H�!G���C���                                    Bxvt�N  �          @��
@<���@�׿���a�C�4{@<����R����p�C�t{                                    Bxvt��  �          @��
@7��AG����\���C�Ǯ@7������R�
�RC��                                    Bxvu�  �          @���@9���5�������C��R@9����� ����C�E                                    Bxvu@  �          @�G�@���Y���L���1�C�Ф@���=p���Q��ÅC���                                    Bxvu$�  �          @��\?��H�z=q?=p�AC��R?��H�u���=q�f�HC�\                                    Bxvu3�  T          @��?�  ����?�\)ArffC���?�  ��33�J=q�*{C�ٚ                                    BxvuB2  �          @���?�
=�r�\>���@\)C�w
?�
=�a녿�
=����C�T{                                    BxvuP�  T          @��@��j�H>��R@�Q�C��=@��Z�H��{��(�C���                                    Bxvu_~  �          @��?�Q��p��>aG�@?\)C���?�Q��]p����R��z�C��f                                    Bxvun$  �          @�  ?�Q��q�>8Q�@{C���?�Q��]p����
��
=C���                                    Bxvu|�  �          @���@(Q��W
=<��
>���C��@(Q��@�׿��R��\)C���                                    Bxvu�p  �          @���@,���Tz�������C���@,���9����\)��ffC�xR                                    Bxvu�  �          @��@)���S33�B�\�'
=C�ff@)���7
=��z���Q�C�n                                    Bxvu��  �          @��R@0  �J�H���R���RC��H@0  �,(��ٙ���33C�ٚ                                    Bxvu�b  �          @��R@1��I���u�P  C���@1��,�Ϳ�����HC��                                    Bxvu�  �          @��@.{�HQ쾊=q�n{C�� @.{�*�H��z����C��                                    BxvuԮ  �          @��@+��J=q�������HC�"�@+��,(��ٙ�����C�w
                                    Bxvu�T  �          @�{@&ff�Q�=#�
?�C�0�@&ff�<(�������=qC���                                    Bxvu��  �          @�p�@G��X�ÿL���2ffC�� @G��*=q����p�C�:�                                    Bxvv �  �          @�ff@���QG�?@  A'33C�R@���P  �Q��7�
C�+�                                    BxvvF  �          @���@/\)�N�R>�ff@��HC�*=@/\)�E������h  C��{                                    Bxvv�  �          @���@@  �?\)?+�A\)C���@@  �>{�E��%�C��                                    Bxvv,�  �          @��@>{�>{?uAMG�C��f@>{�E��   ��{C�H                                    Bxvv;8  �          @���@333�<��?�(�A�G�C���@333�K�����dz�C���                                    BxvvI�  �          @�ff@#33�J=q?��Aj�RC�y�@#33�R�\�   ��=qC��                                    BxvvX�  �          @�p�@���L��?�\)A���C���@���^{�u�W�C���                                    Bxvvg*  �          @�{@��W�?�ffA���C��@��e�Ǯ����C�E                                    Bxvvu�  �          @��R@��Z�H?�  A^ffC�C�@��`  �.{�G�C��R                                    Bxvv�v  �          @��R@G��Y��?Q�A4��C��q@G��Y���W
=�9�C��                                    Bxvv�  �          @�{@�
�H��?��A�z�C�0�@�
�_\)���
���C���                                    Bxvv��  �          @�ff@#�
�J=q?��
Ab�RC���@#�
�Q녿
=q��C�                                    Bxvv�h  �          @�{@
=�P  ?���A���C���@
=�[���G����C�>�                                    Bxvv�  �          @���@
=�I��?��A���C�g�@
=�c33        C��=                                    Bxvvʹ  �          @��@�H�A�?��A��HC�P�@�H�c33>�=q@c33C�!H                                    Bxvv�Z  �          @���@Q��Fff?�\A£�C��f@Q��dz�>�?��
C��                                     Bxvv�   �          @�\)@33�<(�?�z�A��HC��)@33�_\)>���@���C��=                                    Bxvv��  �          @�(�?.{�g���\����C�K�?.{�	���mp��j�
C���                                    BxvwL  T          @���?�ff�n{�  ��G�C���?�ff�  �n{�[�C��                                    Bxvw�  �          @���?�(��o\)��
�أ�C���?�(��
=�c�
�P(�C��                                    Bxvw%�  �          @�{?�(��y�����H��{C�P�?�(��1G��Fff�2�C��                                    Bxvw4>  �          @�  @\)�hQ쿳33���C�ٚ@\)�#�
�;��#{C��q                                    BxvwB�  �          @�z�@   �l(���33�uG�C�B�@   �.�R�/\)��C�(�                                    BxvwQ�  �          @�p�@#�
�`  �Q��+�
C�\@#�
�-p��
=�p�C��                                    Bxvw`0  �          @��R@6ff�W
=���ָRC�+�@6ff�.�R���ٙ�C�%                                    Bxvwn�  �          @�p�?�p��{���ff���C�Z�?�p��P���\)��C���                                    Bxvw}|  �          @�@�\�z�H��  �K�C���@�\�Vff�33��33C���                                    Bxvw�"  �          @�
=?�Q����׽���=qC��?�Q��_\)���R��{C���                                    Bxvw��  �          @���@���|(�=���?��\C��R@���a녿�G����C��)                                    Bxvw�n  �          @��R@��w
=�!G��G�C�g�@��G���� z�C�%                                    Bxvw�  �          @�
=@z��|�ͽ�G���Q�C���@z��[�������(�C���                                    Bxvwƺ  �          @�{?���  ���
���C��R?��_\)��������C�o\                                    Bxvw�`  �          @�?�z����H��������C�\?�z��\���{��RC��)                                    Bxvw�  �          @��@ ���x�þ���  C��H@ ���Mp�������HC��                                    Bxvw�  �          @�p�@33�x�þ�
=���RC��@33�N�R�p����C�5�                                    BxvxR  �          @��?�z��|�;�p����HC��?�z��S33�(����C�
=                                    Bxvx�  �          @�?����G����ǮC��
?���`  �G���G�C��                                    Bxvx�  �          @�ff?޸R��=q���R��Q�C��\?޸R�\(������{C�h�                                    Bxvx-D  "          @�  ?�Q���\)>�Q�@��C�^�?�Q��y����Q���33C�q                                    Bxvx;�  �          @�\)?������?z�@��C���?�����Q��G����C�!H                                    BxvxJ�  �          @�{?�z���33>L��@'�C�
=?�z��mp���\��z�C�
                                    BxvxY6  �          @�p�@ ���{�=��
?�G�C��=@ ���_\)��ff����C��3                                    Bxvxg�  �          @�33?�p��}p�>�
=@���C��\?�p��k��\���C��
                                    Bxvxv�  �          @�33?�����\?   @ҏ\C�b�?���tz��G�����C��
                                    Bxvx�(  �          @��?�z��{�>�@�ffC�y�?�z��j�H��(����C�/\                                    Bxvx��  �          @��R@���mp��aG��5�C���@���J=q��(��хC���                                    Bxvx�t  �          @�ff@&ff�fff�u�EC���@&ff�B�\��
=��  C�:�                                    Bxvx�  �          @���@'��Y����G��\C��f@'��;���p����C��q                                    Bxvx��  �          @��@#33�XQ�W
=�2�\C�|)@#33�7���ff��33C���                                    Bxvx�f  �          @�  @;��C33��  �[�C��@;��#�
��
=���C�o\                                    Bxvx�  �          @�\)@G��5�\)��33C���@G���H���R���HC��                                    Bxvx�  �          @�(�?����
=>Ǯ@��C���?���x�ÿ�
=����C�%                                    Bxvx�X  �          @��?�=q��  >Ǯ@�z�C���?�=q�l(���=q��(�C��
                                    Bxvy�  �          @�  @
=�a논��
����C��q@
=�E���(���G�C��3                                    Bxvy�  �          @���@=q�^�R�����Q�C�\)@=q�6ff�����C��                                    Bxvy&J  �          @��H@.{�L�Ϳ�=q�eC�/\@.{��\�\)��C��q                                    Bxvy4�  �          @��@.�R�H�ÿ����{C�}q@.�R���*=q���C�R                                    BxvyC�  �          @��\@%�U��p���Ip�C���@%�p����	{C�>�                                    BxvyR<  �          @��H@\)�Vff��Q��
=C�Q�@\)�
=�*=q�\)C�L�                                    Bxvy`�  �          @��\@$z��Fff�˅��  C���@$z�����8���'��C�c�                                    Bxvyo�  �          @���@+��QG�������(�C��@+��*�H��
=�ظRC��=                                    Bxvy~.  �          @�Q�@5�HQ�#�
���C�)@5�*=q��z���ffC�n                                    Bxvy��  �          @�  @<(��6ff�u�S�C���@<(����{��G�C��)                                    Bxvy�z  �          @�G�@<���3�
�����(�C�/\@<�Ϳ�=q�\)��\C��                                    Bxvy�   �          @��@<���>�R��33�v{C�\)@<���33�p��	�\C��f                                    Bxvy��  �          @��@=p��7���
=��\)C��\@=p����)���z�C�Ff                                    Bxvy�l  �          @�33@?\)�8Q쿧����C��@?\)����"�\��C��{                                    Bxvy�  �          @��@C�
�.{��  ����C�,�@C�
��33�(����RC��)                                    Bxvy�  �          @�33@B�\�0�׿�
=��{C���@B�\���H�&ff�C�U�                                    Bxvy�^  n          @��
@L���{��33��33C�'�@L�Ϳ����)���{C���                                    Bxvz  �          @�33@<���/\)��z���
=C���@<�Ϳ�=q�1��
=C���                                    Bxvz�  �          @��H@AG��"�\������HC��3@AG�����333�!33C�H                                    BxvzP  �          @���@C�
��R�   ��
=C��@C�
�xQ��5�&{C�=q                                    Bxvz-�  �          @�=q@B�\��=q�'���RC�AH@B�\��Q��C�
�4C�q                                    Bxvz<�  �          @�33@E��˅�'��p�C�XR@E������C�
�3ffC��                                    BxvzKB  �          @�(�@7
=��z��HQ��5�\C��\@7
=?��R�\�BffA%��                                    BxvzY�  �          @��
@p��B�\�^�R�U��C�o\@p�?�ff�Y���OQ�A�z�                                    Bxvzh�  �          @���@N�R�����$z���C��@N�R    �<���)��<��
                                    Bxvzw4  �          @��@H�ÿ�p��0���G�C�L�@H��>�z��@���.�@��                                    Bxvz��  �          @�z�@R�\��  �{�	{C�� @R�\��Q��8Q��$�
C�1�                                    Bxvz��  �          @��@XQ��  �Q��  C��@XQ���3�
���C��H                                    Bxvz�&  �          @�p�@.�R���\(��L(�C�xR@.�R?�p��P  �=�A�Q�                                    Bxvz��  �          @�@5����W
=�EG�C�U�@5�?�z��L���9�A��H                                    Bxvz�r  �          @�ff@/\)��G��]p��M{C�g�@/\)?����N�R�:A�=q                                    Bxvz�  �          @�{@0  ��z��^{�M�HC��
@0  ?��H�J=q�5��A�{                                    Bxvzݾ  �          @�{@#�
�L���g��Zp�C��@#�
?�\)�O\)�;��B �                                    Bxvz�d  �          @�ff@(��u�mp��b  C�9�@(�?У��U�B�B��                                    Bxvz�
  �          @�ff@�;�\)�l(��`�C�@��?˅�Vff�Cp�B��                                    Bxv{	�  �          @��@#33���l���]��C���@#33?�ff�N�R�7�RB(�                                    Bxv{V  �          @�  @=q>��r�\�f�@HQ�@=q?��R�N{�7�\B�                                    Bxv{&�  �          @��R@'
=>#�
�fff�X(�@e@'
=?�
=�B�\�,B
=                                    Bxv{5�  �          @�
=@7
=�u�Z�H�H(�C��@7
=?��R�E�.��A���                                    Bxv{DH  �          @�
=@<(��Ǯ�U��A��C�AH@<(�?����E��/�A�{                                    Bxv{R�  �          @�  @J=q����I���1�C��R@J=q?���A��)p�A�z�                                    Bxv{a�  �          @���@P  �5�C�
�*ffC��{@P  ?h���@���&�
Ayp�                                    Bxv{p:  �          @�Q�@U���p��@���(  C��\@U�?�
=�333���A�(�                                    Bxv{~�  �          @�Q�@U��\)�B�\�)��C�Ф@U�?�33�-p��p�A�=q                                    Bxv{��  �          @�\)@b�\�\(��
=q��p�C�#�@b�\>�����
��
@�=q                                    Bxv{�,  �          @�z�@w
=�޸R����g
=C��f@w
=�u����C�f                                    Bxv{��  �          @�p�@q녿��Ϳ����ŮC�.@q녾�  �G����
C�%                                    Bxv{�x  �          @�z�@n�R���ÿ�\)��{C�@ @n�R�W
=�����HC�b�                                    Bxv{�  �          @�
=@p  �������؏\C�q@p  ������  C�޸                                    Bxv{��  �          @���@w
=�Ǯ��G����C��q@w
=����33���C��                                    Bxv{�j  �          @�Q�@vff�����G����
C�{@vff��ff�33��
=C���                                    Bxv{�  �          @�
=@s33��Q��������C��H@s33�����33��(�C�}q                                    Bxv|�  �          @��R@HQ���G��1��C�ٚ@HQ�?����>�R�(
=A��                                    Bxv|\  �          @��R@Z=q�Y���/\)��
C�f@Z=q?#�
�2�\�p�A+
=                                    Bxv|   �          @�{@S33�����0����C��@S33>��;��%(�@�
=                                    Bxv|.�  �          @��R@J�H�����;��$33C�g�@J�H?��E�/=qA�
                                    Bxv|=N  �          @��@Mp��.{�Dz��,��C���@Mp�?u�@  �'�A��H                                    Bxv|K�  �          @�
=@Dz῕�AG��*G�C���@Dz�?�\�L���7(�A                                    Bxv|Z�  �          @��@S33���\�0  ��C�}q@S33>�z��AG��)33@�Q�                                    Bxv|i@  �          @���@X�ÿ����!G����C�]q@X�ý�G��>�R�%Q�C�                                    Bxv|w�  �          @���@L�Ϳ\�1G��  C�S3@L��=�Q��J=q�2G�?��
                                    Bxv|��  T          @�\)@>{�����Dz��-  C��@>{>\�U��@33@��H                                    Bxv|�2  �          @�
=@;���p��A��*�
C���@;�>�  �W
=�C�@��H                                    Bxv|��  �          @�Q�@Dz��G��;��"33C��q@Dz�>.{�R�\�;�H@Mp�                                    Bxv|�~  �          @�{@G���33�,����
C��@G��u�J�H�5��C�u�                                    Bxv|�$  �          @�{@HQ��  �1G��=qC�4{@HQ�=�G��H���4z�?�Q�                                    Bxv|��  �          @�\)@H�ÿ�  �4z��(�C�>�@H��>\)�L(��5@{                                    Bxv|�p  �          @�
=@Tz`\)�*=q�Q�C�Ǯ@Tz�>.{�?\)�(  @8��                                    Bxv|�  �          @�
=@U���
�,(���RC�� @U>���>{�&(�@���                                    Bxv|��  �          @�ff@TzῙ���,(��=qC��@Tz�>����;��$��@��R                                    Bxv}
b  �          @�{@Z=q����(���Q�C���@Z=q>�ff�333���@�                                      Bxv}  �          @�ff@\�ͿE��,�����C���@\��?333�.{��
A8��                                    Bxv}'�  �          @�\)@`  �W
=�*=q�C�>�@`  ?!G��.{��A"{                                    Bxv}6T  �          @���@o\)�\��
��Q�C��\@o\)����"�\�C��q                                    Bxv}D�  �          @��R@c�
��33����C�@c�
>u�+���\@w�                                    Bxv}S�  �          @�  @e��z������HC��@e>u�,���z�@y��                                    Bxv}bF  �          @�ff@_\)�^�R�'��=qC�  @_\)?z��,(��G�A                                      Bxv}p�  �          @��@Z=q�z��)����HC�7
@Z=q?\(��%���
AaG�                                    Bxv}�  �          @��@^�R�:�H�'
=�p�C�{@^�R?333�'���
A7\)                                    Bxv}�8  �          @��R@c33���'��Q�C�p�@c33?Y���#33��AVff                                    Bxv}��  �          @�{@b�\��33�)����C�#�@b�\?�ff�p���A��                                    Bxv}��  �          @�z�@e�����!���C��@e�?����
���A�Q�                                    Bxv}�*  �          @�@e�\)�%�{C�ٚ@e?����33��=qA���                                    Bxv}��  �          @�@e�>B�\�'
=�G�@Fff@e�?�  �����p�A��                                    Bxv}�v  �          @�p�@c�
?�\�$z��  A
=@c�
?�  ��Q�����Aљ�                                    Bxv}�  �          @�{@g�>�(��!G��
�@�  @g�?�z��������A��                                    Bxv}��  �          @���@n{�#�
�ff��C���@n{?�Q������A�Q�                                    Bxv~h  �          @��H@p��<��	����Q�?�\@p��?��׿�=q��(�A��
                                    Bxv~  �          @��
@s�
>�
=��
��ff@�33@s�
?��Ǯ���A���                                    Bxv~ �  �          @�(�@s�
?   ��=q�˅@�\)@s�
?�\)������
A�                                    Bxv~/Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxv~>               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxv~L�  &          @�ff@����=p���z���  C�Ф@���>aG�����=q@Dz�                                    Bxv~[L  �          @�\)@|�Ϳ�{��ff����C��@|�ͼ����ᙚC��                                    Bxv~i�  T          @�\)@|(���(����H����C�e@|(��.{���G�C��                                     Bxv~x�  �          @�  @}p��Q녿��H�Ώ\C��@}p�>����ff��33@��                                    Bxv~�>  �          @���@u��ff���ԏ\C��@u��\)���� 33C�z�                                    Bxv~��  �          @���@u��ff������HC�^�@u>k���H� �R@Z=q                                    Bxv~��  �          @��\@w
=�z�H�  ��(�C��@w
=>��
�(�� ��@���                                    Bxv~�0  �          @��@tzῂ�\�  ��p�C���@tz�>�\)�p���R@�                                      Bxv~��  T          @�G�@tz῀  ��R��C��@tz�>�z�����@��\                                    Bxv~�|  �          @�  @hQ쿱��{��  C���@hQ�#�
�'��33C��f                                    Bxv~�"  �          @�{@a��  ������\C���@a녿��\�������C�"�                                    Bxv~��  �          @��@S33�3�
�@  ���C���@S33�G����
=C�=q                                    Bxv~�n  �          @��@X���5��=p����C��@X����\���
=C�z�                                    Bxv  �          @�  @^�R�.{�L���$Q�C��)@^�R������C��\                                    Bxv�  �          @�  @c33�z��  ��{C�k�@c33���H��R�
=C��3                                    Bxv(`  �          @���@e�����H��Q�C�` @e���G��p����C�P�                                    Bxv7  �          @�  @Z=q�"�\���R����C���@Z=q��33�%�G�C���                                    BxvE�  �          @�\)@s33�u��  ��p�C�'�@s33?0�׿�\)��z�A%��                                    BxvTR  �          @�=q@|(�?��R��Q���p�A�=q@|(�@zῂ�\�N�\Aݮ                                    Bxvb�  T          @�G�@{�?\�ٙ����A��R@{�@��+���A�                                    Bxvq�  �          @�G�@|��?}p���(���z�Aa��@|��?�\)���pz�A�ff                                    Bxv�D  "          @�33@�33>�z���\��33@��@�33?�ff��������A�33                                    Bxv��  T          @�33@���>k��
�H��G�@S�
@���?��ÿ޸R��p�A��H                                    Bxv��  
�          @���@�=q�\�У���G�C�~�@�=q?��˅��
=@���                                    Bxv�6  T          @���@��H���H��������C���@��H>�녿�\)��G�@�z�                                    Bxv��  �          @��@����p��������C���@��>��H�����z�@θR                                    Bxvɂ  �          @��@�33��33��\)��ffC��3@�33?
=q�Ǯ����@�G�                                    Bxv�(  �          @�p�@�녿+�������\C���@��>�\)��  ��{@l(�                                    Bxv��  �          @�z�@�{�^�R�����\)C�#�@�{>8Q��p���{@ ��                                    Bxv�t  �          @�p�@�Q�n{��33��{C�ٚ@�Q�=u������
?Y��                                    Bxv�  �          @��@�G��(�ÿ�33��(�C�� @�G�>�z��  ��33@w
=                                    Bxv��  �          @��@��H�333�\���\C�b�@��H>B�\��z����
@��                                    Bxv�!f  �          @�{@�(��k�������Q�C��@�(���Q�У�����C�l�                                    Bxv�0  �          @��R@�z�=p����R��  C�5�@�z�>\)��z���p�?�=q                                    Bxv�>�  T          @�ff@�\)�z�H�s33�;�
C��@�\)��33��=q��p�C�˅                                    Bxv�MX  T          @�@�ff��=q�Ǯ��Q�C���@�ff�p�׿��\�J�HC��                                    Bxv�[�  
�          @�ff@��Ϳ��ÿ�33�d(�C�%@��;��
�����{C���                                    Bxv�j�  �          @љ�@�  �%�	�����C�ff@�  �����K���z�C�^�                                    Bxv�yJ  �          @�z�@�=q�*�H� ����\)C���@�=q�xQ��a���33C�B�                                    Bxv���  �          @�G�@���6ff��R����C�
=@������fff���HC�}q                                    Bxv���  �          @�=q@����8�ÿ�����
C���@��ÿ�G��E��ڏ\C���                                    Bxv��<  �          @޸R@����K��
�H��\)C���@����˅�`����{C�\)                                    Bxv���  �          @��
@�G���  ��33�5�C�Ff@�G��9���Y����ffC�33                                    Bxv�  �          @�z�@�z��������3�C��{@�z��2�\�Tz��޸RC��R                                    Bxv��.  �          @�\@�
=��{��R����C�u�@�
=�^{�5���C��                                    Bxv���  �          @��@��R�~�R��
=�~�HC���@��R����n{���C�/\                                    Bxv��z  
�          @��
@��
���\���R�Ap�C��@��
�-p��Y����z�C�"�                                    Bxv��   �          @�ff@�
=�xQ��{�p��C��@�
=�ff�g
=��=qC���                                    Bxv��  �          @�  @�������?��\A'�C�@�����(���ff�pQ�C�ff                                    Bxv�l  �          @��
@����G�?:�H@�{C��q@�������������C���                                    Bxv�)  �          @��H@��R���@ ��A~�\C�Q�@��R��(����R�
=C��                                    Bxv�7�  "          @���@������R?�{Ai��C���@������H����-�C�1�                                    Bxv�F^  T          @���@�����G�?�ffA`��C���@�����p���{�)��C�b�                                    Bxv�U  T          @�@�(���
=?�{AiG�C�  @�(���zῡG���C���                                    Bxv�c�  �          @�@����{@[�AۮC�c�@����G�>�@mp�C�c�                                    Bxv�rP  �          @�R@�����(�@z=qA��C���@�����ff?J=q@�=qC��{                                    Bxv���  �          @�=q@Z�H���
@�(�B�C��@Z�H��Q�?��HA>=qC��{                                    Bxv���  
�          @��H@q����@p��A��C�g�@q��ȣ�?\)@��C���                                    Bxv��B  
�          @�p�@^{��\)@���B�C�%@^{����?��A-��C�>�                                    Bxv���  �          @�  @�p����@EA���C���@�p���
=�B�\��
=C��                                    Bxv���  �          @�  @l(����
@g�A�Q�C�AH@l(���
=>�\)@
=qC��)                                    Bxv��4  T          @�@~{����@j�HA�C���@~{�ə�>�G�@XQ�C�R                                    Bxv���  T          @�
=@n{���@C�
A���C��@n{��p���{�)��C�                                    Bxv��  T          @�@s�
����@c33A��C��q@s�
��ff>W
=?�=qC�G�                                    Bxv��&  �          @�  @W���p�@b�\A�C�U�@W���{=L��>\C�\)                                    Bxv��  
�          @�=q@fff���@n�RA�33C���@fff���>�Q�@/\)C�^�                                    Bxv�r  �          @�@@  ��@|(�A�{C��@@  ����>Ǯ@=p�C��)                                    Bxv�"  �          @��@:�H����@�
=B(�C��3@:�H���?5@�p�C�u�                                    Bxv�0�  T          @�G�@4z���
=@��B-G�C�"�@4z���p�?�z�AtQ�C�޸                                    Bxv�?d  T          @�\@(����H@��BCz�C��@(���Q�@��A��\C��f                                    Bxv�N
  T          @޸R?�=q�]p�@���BbffC�J=?�=q��
=@G
=A��C�n                                    Bxv�\�  
�          @�\@
=����@�=qBC=qC�4{@
=��ff@(�A�33C�ff                                    Bxv�kV  
Z          @�33@aG�����@�A�33C�"�@aG���  ����)�C���                                    Bxv�y�  "          @���@Z=q����@	��A��
C��f@Z=q��  �����'�
C�O\                                    Bxv���  �          @�=q@N{��
=@L(�A�p�C��
@N{��G���=q��C��                                    Bxv��H  T          @�ff@>�R��{@~{B\)C���@>�R��  ?+�@�z�C�Q�                                    Bxv���  �          @��@G���
=@p  A��C�k�@G����>��@p��C���                                    Bxv���  T          @�z�@%����
@���B-(�C�G�@%���\)?��Am�C�AH                                    Bxv��:  "          @�=q@ff�b�\@�33BV�RC�Q�@ff��ff@:�HA���C���                                    Bxv���  �          @��?8Q��N�R@�Q�Bx�\C�O\?8Q�����@i��A��C�k�                                    Bxv���  T          @�p�?�z��xQ�@�BR��C�R?�z�����@3�
A�\)C�t{                                    Bxv��,  
�          @�p�@(��qG�@�33BM(�C�p�@(���z�@333A�C���                                    Bxv���  
�          @�{@���Y��@�{B^�HC�u�@����  @Q�A�p�C��                                    Bxv�x  T          @�G�?��A�@���Bmp�C�,�?���  @dz�A�(�C�:�                                    Bxv�  T          @�z�@��S�
@��B^(�C�N@���z�@P��Aڏ\C���                                    Bxv�)�  
�          @�ff@@  ��33@�z�B �RC���@@  ��33?�ffAF�RC���                                    Bxv�8j  	w          @ᙚ@J=q�:=q@���BQ  C���@J=q��z�@Mp�A�p�C�4{                                    Bxv�G  
�          @�z�@���(�?�Ab�HC��3@���  ��G��*=qC�B�                                    Bxv�U�  �          @�  @�����׾�Q��4z�C��=@����\)�Tz��ڏ\C�U�                                    Bxv�d\  �          @��@�G���=q<��
>�C�0�@�G���
=�A���Q�C�e                                    Bxv�s  �          @���@�{��������C��@�{��(��Vff��C�U�                                    Bxv���  �          @�  @����\)?fff@�C���@�������H��G�C��3                                    Bxv��N  �          @�(�@�  ����@#33A��C��@�  ���
���c�
C��                                    Bxv���  �          @���@�33��Q�@7
=A�(�C�z�@�33��녽u��(�C�G�                                    Bxv���  �          @���@�������@,��A�z�C��@������H�����(�C���                                    Bxv��@  T          @��H@��
����@?\)A��\C��R@��
��\)>�z�@33C�0�                                    Bxv���  �          @�p�@�����R?�A333C�3@����
=����/33C��                                    Bxv�ٌ  �          @�@����Q�=p�����C���@������`����G�C�=q                                    Bxv��2  �          @�=q@���G��\�9��C���@���33�=p����C�z�                                    Bxv���  �          @�=q@�  ���H�L�Ϳ��RC�N@�  �s�
�$z�����C��                                    Bxv�~  �          @�G�@�(����R���uC��
@�(����
�5���C�q�                                    Bxv�$  �          @�\)@����(�?˅AF�\C�}q@����(��˅�E��C�|)                                    Bxv�"�  �          @��
@Z=q���
?�\)AD��C��H@Z=q��  ���}G�C��)                                    Bxv�1p  �          @��R@j=q��  ?(�@���C�:�@j=q��=q�@����(�C���                                    Bxv�@  �          @��@vff��녾���C�
C�1�@vff��(��q����
C��=                                    Bxv�N�  �          @��H@�ff��{�#�
����C�5�@�ff�����0  ���C���                                    Bxv�]b  �          @��R@�
=��\)�.�R��33C��@�
=�'
=������C�=q                                    Bxv�l  �          @�G�@�z���33�����:=qC��@�z��a���  ��(�C��)                                    Bxv�z�  �          @�G�@����������XQ�C�s3@����]p���Q���\C��)                                    Bxv��T  �          @�Q�@�\)��{��ff�V=qC�E@�\)�_\)��  ��RC�                                    Bxv���  �          @�\)@�\)��ff��G��4(�C��R@�\)�[��w�����C���                                    Bxv���  �          @�  @�(����ÿ�\)�@��C�p�@�(��N�R�xQ���Q�C��)                                    Bxv��F  �          @�Q�@����{�\)���
C�
=@���{��C�
��z�C�4{                                    Bxv���  �          @�\)@�����\)���Z=qC��R@����S33��z��{C���                                    Bxv�Ғ  �          @��@������\�p  ��p�C��{@�����p����R�6��C���                                    Bxv��8  �          @��@�
=�c33����C�8R@�
=�8Q���(��A{C���                                    Bxv���  �          @�=q@�\)�tz������RC��=@�\)�h����=q�I�\C���                                    Bxv���  �          @�=q@�z���p��}p����C�1�@�z�ٙ���{�D  C��                                    Bxv�*  �          @�@���tz���G��z�C��\@�����\�����Ep�C���                                    Bxv��  �          @�R@�����(���{�	��C�˅@��׿�=q��\)�J=qC���                                    Bxv�*v  �          @�@������j�H���HC�R@��׿�33��=q�4C��                                    Bxv�9  �          @�@�p���33�P  ��p�C��R@�p�� ����G��)  C�]q                                    Bxv�G�  �          @���@�p����
��G��  C��@�p���=q����P�\C�#�                                    Bxv�Vh  �          @���@S33�����g
=��{C�ff@S33�5����R�W�\C��f                                    Bxv�e  T          @��H@��\��\)�(������C�@��\�7���33�
=C�=q                                    Bxv�s�  �          @��
@�������G��v�RC�'�@����P������(�C�.                                    Bxv��Z  �          @�@��
��������RC�  @��
�w
=�g���p�C�                                      Bxv��   �          @�33@�p�����޸R�T��C��\@�p��q���33��C�>�                                    Bxv���  �          @�z�@��
��
=�U�����C�y�@��
�   ���7�C�Ǯ                                    Bxv��L  T          @�z�@��R�����{����C�p�@��R�33����H(�C���                                    Bxv���  �          @�p�@�  ����Vff�ϮC��\@�  � �����R�5�
C��                                    Bxv�˘  �          @�
=@��H��{�p  ����C�o\@��H��������A�C�޸                                    Bxv��>  �          @��
@�
=���\�l������C�w
@�
=�=q����>ffC��                                     Bxv���  �          @�p�@�33��G��g
=��\)C�q@�33�p����\�CQ�C���                                    Bxv���  �          @�\)@�{����dz��㙚C�33@�{��\���\�@�\C�U�                                    Bxv�0  �          @�\)@�����ff�a���C��3@��������33�9�\C���                                    Bxv��  �          @�\@�\)���e���  C��@�\)�ff��(��@(�C�*=                                    Bxv�#|  �          @�\@��H�����j=q��
=C�l�@��H�Q�����D��C���                                    Bxv�2"  �          @�{@�����p��p����ffC�aH@����  �����B�RC��)                                    Bxv�@�  �          @��@����  �~�R��33C�7
@�����R���
�D�HC�p�                                    Bxv�On  9          @���@�G�����������
C���@�G���Q�����J33C�j=                                    Bxv�^  �          @���@����������\���C���@��Ϳ�\)�����Kp�C�j=                                    Bxv�l�  �          @���@�����=q��
=���C��=@��Ϳ�\)��=q�M�C��=                                    Bxv�{`  �          @���@��
��Q���33���C���@��
��  ��z��PG�C�o\                                    Bxv��  
�          @�
=@�Q��q�����*(�C�XR@�Q�����G��d�C��
                                    Bxv���  
�          @�@Z=q�E������H��C���@Z=q>u��=q�vG�@~{                                    Bxv��R  
�          @�@Z=q�3�
�����M�
C�9�@Z=q>����ff�s�@�p�                                    Bxv���  
�          @�p�@Vff�C�
��{�C�C���@Vff=��
�����s��?�\)                                    Bxv�Ğ  �          @߮@i���Mp������7�\C�Y�@i���\)���R�i��C���                                    Bxv��D  �          @��@��\(���p��)  C�Ff@��Ǯ����[�
C�S3                                    Bxv���  
�          @�@r�\�h����(��+z�C�\@r�\�
=����fC��)                                    Bxv��  
�          @��H@�Q��J=q��{�6�HC���@�Q�=�G���G��_��?\                                    Bxv��6  "          A�@����W
=��{�.��C��@���=�\)�Ӆ�U�R?Q�                                    Bxv��  
�          A�
@�  �U���(��,�\C�&f@�  =u�љ��Rz�?=p�                                    Bxv��  
�          A��@�(��_\)��{�)  C�8R@�(��#�
�θR�S��C�
=                                    Bxv�+(  
�          A�H@��\�tz������#�C��f@��\����ָR�S  C��\                                    Bxv�9�  
�          A{@�=q�l���������C���@�=q��(���ff�Ip�C���                                    Bxv�Ht  	�          A�@�=q�z�H��(�� (�C�!H@�=q����z��Q�HC�Ǯ                                    Bxv�W  
�          A�R@�G����R��
=��C�e@�G���z���
=�>C�R                                    