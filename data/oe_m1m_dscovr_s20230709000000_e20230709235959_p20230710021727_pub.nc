CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230709000000_e20230709235959_p20230710021727_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-07-10T02:17:27.124Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-07-09T00:00:00.000Z   time_coverage_end         2023-07-09T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�+@  T          @��H�vff@�@�\)B433CaH�vff@@��\B9
=C�\                                    Bx�9�  T          @ȣ��z=q@�@�(�B1�\C�f�z=q?��R@�
=B6�C
=                                    Bx�H�  
�          @����w�@��@�B3�CL��w�@G�@���B8(�Cu�                                    Bx�W2  �          @�\)�w�@G�@��B/z�C���w�@@���B4(�C�f                                    Bx�e�  
(          @�=q�c33@.�R@�(�B1p�Cs3�c33@"�\@��B7�C^�                                    Bx�t~  �          @ȣ��j=q@�@��\B=�C  �j=q?�@�p�BA�RCJ=                                    Bx��$  "          @ȣ��qG�?��@��
B>\)CaH�qG�?ٙ�@�{BBffC                                    Bx���  �          @����n{?�(�@�p�B?�C��n{?��
@��BD�Cn                                    Bx��p  �          @�G��}p�?�\@���B9ffC��}p�?�=q@��
B=  C:�                                    Bx��  �          @˅����?��H@��B9�HC�����?\@�p�B=Q�CL�                                    Bx���  "          @�33���H?�ff@�33B9�CE���H?�{@��B=  C!��                                    Bx��b  T          @������?���@��B5p�CB�����?��@���B8�C!u�                                    Bx��  �          @��H����?���@���B7G�CW
����?�z�@��
B:�HC��                                    Bx��  "          @ʏ\�|��?�33@���B=�C\)�|��?��H@��RB@�
C��                                    Bx��T  �          @��H�u�?�{@�{B?
=C
�u�?�
=@�Q�BBC\)                                    Bx���  T          @ʏ\�p��@   @�B>�C  �p��?���@�  BB�C5�                                    Bx���  �          @ə��z�H@�@��RB5G�C���z�H?�{@�G�B933C�H                                    Bx��$F  T          @���}p�@�@�B2Q�C���}p�?���@�  B6Q�C�                                     Bx��2�  T          @ə��{�@
�H@���B1��C��{�@   @�\)B5�HC��                                    Bx��A�  T          @�  ��  ?��@��B3��C�R��  ?�(�@�\)B7�C�3                                    Bx��P8  
�          @�\)���?�p�@���B3��C�3���?Ǯ@��RB6�
C�                                    Bx��^�  �          @ȣ����?���@��B3(�C�
���?�33@�\)B6�C��                                    Bx��m�  T          @Ǯ��{?���@�(�B2��C ���{?��@�B533C"�                                    Bx��|*  "          @�\)���?}p�@�G�B.�C'aH���?Tz�@�=qB/��C)T{                                    Bx����  �          @�ff��\)?��
@���B0�RC#33��\)?�\)@��HB2��C%#�                                    Bx���v  ^          @�ff���R?���@���B5ffC%aH���R?p��@�{B7Q�C'h�                                    Bx���  J          @Å�}p�?��@�{B:�C ���}p�?�p�@�\)B={C"�                                    Bx����  "          @Å����?��@��\B4��C$�)����?}p�@��
B6��C&�{                                    Bx���h  
�          @�=q����?��\@�B.  C&�{����?aG�@��RB/�RC(p�                                    Bx���  T          @�33���
?@  @�B-(�C*G����
?��@�ffB.\)C,�                                    Bx���  �          @�(�����?
=@�
=B.\)C,Q�����>�ff@��B/G�C.+�                                    Bx���Z  
�          @�(����\>W
=@�=qB&Q�C1k����\=�\)@�=qB&�\C3!H                                    Bx��    
�          @�(���p�<�@|��B!  C3���p���G�@|��B �C5L�                                    Bx���  �          @��H��(�����@}p�B"(�C5:���(��k�@|��B!�C6�
                                    Bx��L  T          @Å���B�\@z=qBz�C6L������
@y��B{C7�)                                    Bx��+�  �          @Å������@�{B:��C8xR�����@�B:{C:k�                                    Bx��:�  
�          @Å���H���@�  B0�\C;�����H�=p�@�\)B/p�C=��                                    Bx��I>  T          @�G���ff���@���B'�HC;\��ff�.{@�Q�B&�C<��                                    Bx��W�  ,          @�=q�����(�@i��BG�C;E�����:�H@hQ�BQ�C<��                                    Bx��f�  
�          @�=q���
�c�
@tz�B33C>����
��G�@r�\B��C@T{                                    Bx��u0  "          @�����Q쾏\)@~�RB%z�C7�=��Q����@~{B$��C9{                                    Bx����  "          @�Q���=q���@C�
A�ffC?�H��=q���@AG�A�p�C@��                                    Bx���|  
�          @�
=��  �O\)@J=qB   C=8R��  �h��@H��A��C>E                                    Bx���"  �          @������u@J�HA���C>��������@H��A��HC?�                                    Bx����  "          @�Q����Ϳ�@333A߅C@�
���Ϳ�G�@1G�A�ffCA�3                                    Bx���n  
�          @�=q��ff��=q@s�
B{C7=q��ff�\@s33B��C8��                                    Bx���  
�          @�(���p�>�ff@��RB-�C.33��p�>���@�
=B.z�C/�q                                    Bx��ۺ  T          @����{?E�@y��Bp�C*�H��{?(��@z�HBffC+�                                    Bx���`  �          @�p���  ?5@��
B(z�C+
��  ?
=@�z�B)ffC,��                                    Bx���  T          @�(�����?\)@��\B3��C,�)����>�G�@��HB4�\C.+�                                    Bx���  
�          @������H?E�@���B=�\C)O\���H?&ff@�G�B>��C*�R                                    Bx��R  
P          @�z��w
=?���@���BE(�C$� �w
=?s33@�BF�C&8R                                    Bx��$�  
�          @�(����?Y��@�p�B8�C(u����?:�H@�{B9��C*                                      Bx��3�  ,          @�(���  ?.{@��
B5��C*���  ?�@�(�B6��C,k�                                    Bx��BD  |          @�z����?(�@�z�B7  C+ٚ���?   @��B7�RC-T{                                    Bx��P�  "          @Å����?��@�=qB3�\C,� ����>�G�@��\B433C..                                    Bx��_�  "          @�33����>�G�@��B4(�C.)����>�{@�=qB4�C/�                                    Bx��n6  �          @��H���>�ff@�33B633C-�3���>�{@��B6�RC/\)                                    Bx��|�  "          @��
��\)?�@���B7C,�
��\)>�(�@��B8ffC.=q                                    Bx����  �          @�(���\)?!G�@���B7(�C+�
��\)?�@��B7�HC,��                                    Bx���(  �          @����z�?J=q@��B;G�C)8R��z�?0��@�Q�B<33C*�)                                    Bx����  "          @�z�����?Y��@�B8�HC(^�����?@  @�{B9�HC)�3                                    Bx���t  �          @�z���  ?�{@�Q�B0��C%aH��  ?�G�@�G�B1�HC&�
                                    Bx���  �          @\���R=�@uB�RC2�{���R<�@uB��C3��                                    Bx����  �          @\���>u@s33B�HC1)���>#�
@s�
B{C2
                                    Bx���f  
�          @�33���>\@s�
B�RC/aH���>���@tz�B{C0W
                                    Bx���  �          @�33��>�ff@x��B=qC.�\��>�p�@y��B��C/��                                    Bx�� �  T          @����
=?�@p  B33C-����
=>�@p��B�C.z�                                    Bx��X  �          @��H����?B�\@j�HB(�C+
=����?.{@k�B�
C+��                                    Bx���  "          @������?0��@k�Bp�C+�����?�R@l(�B
=C,��                                    Bx��,�  T          @��H����?�@n�RBQ�C-�\����>�ff@o\)BC.��                                    Bx��;J  �          @�����׼#�
@�Q�B&=qC4����׽��
@�Q�B&33C5\                                    Bx��I�  �          @�����G��#�
@}p�B$Q�C4����G���G�@}p�B$G�C5s3                                    Bx��X�  �          @�Q���녾���@u�B��C9���녾��@tz�BG�C9��                                    Bx��g<  �          @�G���\)���@y��B#��C9�3��\)��@x��B#(�C:�=                                    Bx��u�  �          @�=q���R?�@r�\B�C-�H���R>�@s33B{C.c�                                    Bx����  T          @�G��}p��Y��@���B:Q�C@{�}p��k�@���B9z�CA
=                                    Bx���.  "          @������\��(�@�Q�B3�
CD�f���\���@��B2CE��                                    Bx����  �          @�G���녿�@�{B/C;����녿#�
@�B/=qC<p�                                    Bx���z  "          @\����>��R@���B&33C0\����>�  @���B&ffC0�\                                    Bx���   
�          @�����z�#�
@�(�B-(�C4�{��z����@�(�B-{C5Y�                                    Bx����  �          @�=q��z὏\)@�B.��C4����z��@�B.�C5�f                                    Bx���l  
�          @����z�=L��@�p�B-�C3^���z�#�
@�p�B-�C4�                                    Bx���  
�          @�G���ff>��
@��\B7(�C/�3��ff>��@��\B7\)C0xR                                    Bx����  
�          @�����>u@��B8�C0�q��>8Q�@��B8�
C1z�                                    Bx��^  "          @�G���=q>��R@z�HB"G�C0+���=q>��@{�B"z�C0                                    Bx��  �          @�=q��Q�>B�\@p��B(�C1��Q�>\)@p��BG�C2J=                                    Bx��%�  T          @������>�@~�RB$��C2k�����=��
@~�RB$�C2��                                    Bx��4P  "          @�33��{>�33@�p�B,z�C/n��{>��R@�p�B,�C0                                      Bx��B�  
�          @��
��Q�>aG�@��B)33C133��Q�>8Q�@��B)G�C1��                                    Bx��Q�  T          @�33����>�@�G�B%�C.E����>�
=@���B&(�C.                                    Bx��`B  "          @��
����?}p�@~{B!�
C'������?u@~�RB"G�C(!H                                    Bx��n�  
�          @�z���\)?�33@|��B z�C"����\)?�\)@}p�B!{C#                                    Bx��}�  "          @�(����R?��@}p�B!=qC"�R���R?���@~{B!C#�                                    Bx���4  
�          @��H��p�?��
@l(�B(�C$�)��p�?�G�@l��B��C$�                                    Bx����  "          @�G����R?s33@i��B�C(�\���R?n{@j=qB=qC(޸                                    Bx����  
�          @�\)��=q?\@��HB-{C����=q?��R@�33B-��C�)                                    Bx���&  �          @�  �p��?��@���B5��CW
�p��?�{@���B6G�C��                                    Bx����  �          @����u@@�(�B-�RCxR�u@�
@�z�B.Q�C�                                     Bx���r  "          @���p  @z�@���B*G�C@ �p  @33@��B*�C}q                                    Bx���  �          @�Q��u@
=q@��HB+�\C�{�u@��@�33B,{C��                                    Bx���  T          @���hQ�@*=q@�G�B'��C�=�hQ�@(��@���B(=qC�R                                    Bx��d  
�          @�z��l(�@%�@��B*��C��l(�@#�
@��B+G�C5�                                    Bx��
  
�          @Å�c33@,(�@�B,�C�\�c33@+�@�{B-  C�3                                    Bx���  �          @��
�c�
@7�@���B&(�C(��c�
@7
=@��B&�\CE                                    Bx��-V  
�          @����w�@\)@�G�B%33C.�w�@�R@���B%z�CG�                                    Bx��;�  �          @�=q�\��@AG�@}p�B#33C
�)�\��@@��@}p�B#z�C
�                                    Bx��J�  
�          @�33�Z=q@K�@z=qB   C	�Z=q@J�H@z�HB 33C	\                                    Bx��YH  
�          @�=q�`��@QG�@n�RBffC	\�`��@P��@n�RB�C	{                                    Bx��g�  T          @�G����\@  @uB�
C!H���\@  @uB�HC#�                                    Bx��v�  
�          @������H?�@���B'�C{���H?���@���B'�C\                                    Bx���:  �          @�=q�y��@*=q@u�B33C���y��@*�H@u�B{C�)                                    Bx����  
(          @��H�}p�@*=q@s33B�C��}p�@*=q@r�\B�C�                                    Bx����  �          @�����H@=q@qG�B�RC�����H@�H@p��B�Cn                                    Bx���,  �          @���~{@&ff@qG�B��C�\�~{@'
=@qG�B��C�R                                    Bx����  �          @����{@��@g�B�HC����{@p�@g
=B�\C��                                    Bx���x  �          @�G���(�@*�H@`��B
=C#���(�@+�@`  B��C                                    Bx���  �          @�  �~�R?�p�@�{B/��C�\�~�R?޸R@�B/p�C\)                                    Bx����  
�          @�Q��s33?��
@�ffB?��C!W
�s33?�ff@�ffB?��C!\                                    Bx���j  "          @������?��@�  B3z�C"(����?�=q@��B3�C!�H                                    Bx��	  
�          @�Q��z�H?���@�=qB+p�C� �z�H?�p�@���B*�HC:�                                    Bx���  
�          @�G��w
=?�  @���B5��C���w
=?��
@�G�B5
=CO\                                    Bx��&\  
�          @����\)?�(�@�ffB0(�C�R�\)?޸R@�{B/�CaH                                    Bx��5  "          @����(�?�(�@�p�B.33C n��(�?�  @��B-��C �                                    Bx��C�  "          @�G���G�?У�@�{B/z�C����G�?�z�@�p�B.��C�
                                    Bx��RN  
�          @����|(�@33@��\B*p�C�=�|(�@�@��B)�C&f                                    Bx��`�  "          @���p  ?�(�@��B;��Cc��p  ?�G�@�z�B:��C�f                                    Bx��o�  
�          @�(��H��>�p�@�Q�Bk��C-:��H��>�
=@�  Bk�C,T{                                    Bx��~@  "          @�(��L(�>W
=@�
=BjG�C033�L(�>��@�
=Bj�C/E                                    Bx����  
�          @\�G
=>��
@�
=BlG�C.0��G
=>�p�@��RBl  C-0�                                    Bx����  "          @����O\)>�(�@���BdQ�C,}q�O\)>�@���Bc��C+�                                    Bx���2  �          @�\)�i��?   @�
=BP�C,B��i��?��@��RBO�RC+h�                                    Bx����  
�          @�{�vff?���@�33B<C"���vff?�G�@��HB;�C!�                                    Bx���~  "          @�  �`��?�ff@�\)BQ{C#L��`��?�{@�
=BP=qC"c�                                    Bx���$  
�          @���Y��?W
=@��BX�C&��Y��?h��@�33BX(�C%
=                                    Bx����  T          @�{�l(�?���@�{BD\)C#�
�l(�?���@�p�BCp�C"��                                    Bx���p  �          @����G�?�ff@c33B�\CL���G�?���@a�B=qC��                                    Bx��  �          @��
��=q?�\)@r�\B �
CY���=q?�
=@p��B\)C��                                    Bx���  	�          @�33���?�@n�RB�C�����?�33@l��B
=C:�                                    Bx��b  T          @�33��33?��
@r�\B!\)Cz���33?�@p��B�HC                                    Bx��.  T          @������H?��H@~�RB)�HC ^����H?��
@|��B(�C�=                                    Bx��<�  �          @�(����?���@s�
B!�HC$h����?��\@r�\B C#��                                    Bx��KT  "          @��H��ff?���@z=qB(z�C%�3��ff?��@x��B'ffC$��                                    Bx��Y�  "          @����{?��\@���B,\)C&J=��{?���@�Q�B+Q�C%Q�                                    Bx��h�  T          @�z���(�?p��@��HB0�C'33��(�?��\@��\B/�C&+�                                    Bx��wF  T          @������?aG�@�33B/��C({����?u@��\B/  C'�                                    Bx����  
�          @�p����?\(�@��B0(�C(E���?s33@��HB/(�C'+�                                    Bx����  
          @�{���?\)@�ffB4{C,c����?#�
@�B3ffC+33                                    Bx���8  
�          @��
���=�@���B.��C2W
���>W
=@���B.��C1(�                                    Bx����  T          @��\��G�>�@|(�B*=qC2G���G�>aG�@|(�B*
=C1�                                    Bx����  T          @�33��(�=���@w�B%�
C2����(�>B�\@w
=B%��C1��                                    Bx���*  �          @��\���;W
=@s�
B#G�C6�����ͽ�@s�
B#z�C5�\                                    Bx����  �          @��\��(��W
=@u�B$��C6��(���@uB$�
C5�\                                    Bx���v  �          @���u>\@��BC
=C.Y��u>��H@�z�BBffC,                                    Bx���  
�          @����?h��@��RB5Q�C'\)���?��\@�{B4  C%�3                                    Bx��	�  �          @�
=�p��?�{@�Q�BD{C#� �p��?�p�@�\)BBQ�C!��                                    Bx��h  "          @����Vff?��@�z�BX�HC"
=�Vff?��H@�33BV�C {                                    Bx��'  
�          @�  �Q�?W
=@�ffB^ffC%�)�Q�?z�H@�B\C#s3                                    Bx��5�  
�          @����Vff?333@��RB]Q�C(.�Vff?W
=@�{B[�C%�R                                    Bx��DZ  
�          @�Q��Vff>��@�
=B_
=C+��Vff?(�@��RB^
=C)��                                    Bx��S   "          @����P  >�=q@�=qBe{C/J=�P  >��@��BdffC,�q                                    Bx��a�  �          @����X��>8Q�@�
=B^�RC0���X��>���@�
=B^33C.s3                                    Bx��pL  "          @���Y��>\)@�B]z�C1���Y��>�z�@�p�B]
=C/
                                    Bx��~�  T          @�  �J�H>�=q@��\Bg��C/33�J�H>�
=@�=qBg{C,n                                    Bx����  
�          @�\)�^�R>�33@��\BX\)C.0��^�R?   @��BWz�C+�q                                    Bx���>  "          @���~�R>���@��RB@C/��~�R>�G�@�ffB@{C-��                                    Bx����  �          @����mp�>�=q@��RBO  C/�\�mp�>�
=@�ffBNG�C-u�                                    Bx����  T          @����b�\>W
=@�33BWQ�C0�)�b�\>�p�@��HBV�RC.�                                    Bx���0  
�          @�G��xQ�=�@��
BG�RC2:��xQ�>�\)@��BGQ�C/�                                    Bx����  I          @���qG�=�@�\)BM�
C2��qG�>�z�@�\)BMp�C/�f                                    Bx���|  
�          @�G�����=�G�@�\)B@  C2s3����>�=q@�\)B?��C08R                                    Bx���"  
Z          @�p���Q�    @��HB={C3�3��Q�>#�
@��HB<�C1�q                                    Bx���  "          @�z��~{<��
@�33B>G�C3���~{>8Q�@��HB>{C1u�                                    Bx��n  �          @���\)=u@��B>
=C333�\)>aG�@�33B=C0�f                                    Bx��   	�          @��\)>���@�(�B>(�C/W
�\)>��H@��B=G�C-�                                    Bx��.�  "          @�p��|��?.{@��HB=(�C*+��|��?Y��@��B;z�C'޸                                    Bx��=`  T          @�(���p����
@�z�B3�C48R��p�>\)@�(�B3  C2�                                    Bx��L  T          @�33����\)@���B.�C5�����<��
@���B.G�C3�q                                    Bx��Z�  
�          @�33���׽���@\)B,�C5W
����=u@\)B,(�C30�                                    Bx��iR  
�          @������=u@~{B)�C3E���>aG�@}p�B(�
C1&f                                    Bx��w�  
�          @�(���G�>8Q�@���B,z�C1�H��G�>�{@�Q�B+�HC/k�                                    Bx����  T          @�(����
=@��HB0�C<�����
=@��B1�C9�3                                    Bx���D  "          @��
��z��@�33B3
=C4ff��z�>��@�33B2�HC1��                                    Bx����  T          @�33���?333@��HB3
=C*@ ���?^�R@���B1�C'ٚ                                    Bx����  
Z          @����xQ쿕@��B5CD�R�xQ�z�H@��B8p�CB8R                                    Bx���6  "          @�ff�g����@�ffB=�CG�3�g�����@�  BA33CEE                                    Bx����  �          @����a녿�@�ffB:��CO8R�a녿�\)@���B?�CL�3                                    Bx��ނ  
�          @��\�e���
@��
B3CQ���e���\)@��RB8��CO�)                                    Bx���(  
�          @����e�33@�=qB2Q�CQ��e����@��B7ffCOQ�                                    Bx����  
Z          @������\����@�=qB3�C8�����\�\)@��\B4(�C5��                                    Bx��
t  
�          @�����z�n{@uB(�\C@����z�=p�@xQ�B*�RC>(�                                    Bx��  �          @����~{>��H@��B7  C,���~{?0��@��\B5Q�C*)                                    Bx��'�  �          @��R�\)?��\@{�B.��C%���\)?��H@xQ�B+�\C#�                                    Bx��6f  �          @��\)?
=q@\)B3
=C,Q��\)?=p�@}p�B1=qC)��                                    Bx��E  "          @�{�z=q?�{@~{B1Q�C$
�z=q?��@z=qB-�C!n                                    Bx��S�  �          @������?xQ�@}p�B.�C&�����?�@z=qB+�C#޸                                    Bx��bX  "          @�Q�����?=p�@�G�B2�\C)�{����?s33@�  B0�C&�q                                    Bx��p�  T          @�������?@  @��B4��C)p�����?u@��B2{C&�                                    Bx���  	�          @�Q��u?���@��
B7{C#���u?���@���B3\)C ��                                    Bx���J  "          @����mp�?�=q@�
=B;�RC T{�mp�?�ff@�z�B7G�CO\                                    Bx����  
�          @�(��l��?�\)@���B4=qC)�l��@ff@�G�B.Q�Ck�                                    Bx����  T          @�p��u�?���@�=qB-�RC��u�@
�H@|��B'�RC�                                     Bx���<  
�          @���\)?��H@�Q�B+�C���\)?�
=@z=qB%�
C.                                    Bx����  
�          @��H�e�@	��@��HB1�C
=�e�@Q�@}p�B*Cn                                    Bx��׈  
�          @���|��?�
=@\)B+��C�q�|��?�33@x��B&=qCG�                                    Bx���.  -          @�33�p  ?��@���B5��C\)�p  ?��@���B0
=Cc�                                    Bx����  	�          @��
�I��@G�@�Q�BIffCc��I��@�@�z�BA�HC&f                                    Bx��z  {          @�G��QG�@ ��@��\BA�RCk��QG�@��@��RB:Q�CQ�                                    Bx��   
�          @��\�^�R@�\@�{B8�C�)�^�R@�\@��B0�HC�                                    Bx�� �  �          @�Q��]p�?�Q�@�p�B9�RC�]p�@(�@���B2��C�R                                    Bx��/l  
�          @����O\)?�=q@�Q�BM��C
=�O\)?���@��BG{CE                                    Bx��>  T          @���|��@�@n{B�
C�=�|��@��@eB{C@                                     Bx��L�  
�          @�G��p��@�
@z=qB)��CO\�p��@33@q�B"p�C�\                                    Bx��[^  T          @�
=�U@�@�33B7(�C��U@�@}p�B.��C��                                    Bx��j  T          @�{�G
=@@���BB�
C��G
=@
=@��
B:Q�Cٚ                                    Bx��x�  T          @��L��@@�ffB>�HC�f�L��@
=@��B6p�C�                                    Bx���P  �          @��R�=p�@\)@��\BE��C�)�=p�@!G�@�B<=qC�
                                    Bx����  �          @�Q��L(�@�@��RB<  C���L(�@#33@���B2�Ck�                                    Bx����  �          @�
=�~�R?��@��B5{C+Y��~�R?^�R@�Q�B2Q�C'��                                    Bx���B  �          @�  ��  >���@uB(  C/����  ?
=@s�
B&\)C,�                                    Bx����  �          @�  ������@^{B\)C;&f��������@`  B�HC833                                    Bx��Ў  �          @�
=��G����@]p�BC933��G��.{@^�RBC65�                                    Bx���4  �          @����Dz�@3�
@g�B$(�C	�=�Dz�@C�
@Z�HB33C(�                                    Bx����  "          @����Fff@5�@p  B'=qC	���Fff@E�@c33B(�C(�                                    Bx����  �          @����{<#�
@Z�HB\)C3����{>�  @Z=qB�C0�                                    Bx��&  
Z          @�33���\��=q@L��B��C7Q����\�#�
@Mp�B(�C4u�                                    Bx���  �          @��
��ff    @Z�HBQ�C3���ff>�  @Z=qB�
C0�                                    Bx��(r  T          @�(���\)���R@XQ�B  C7�f��\)�u@Y��B�C4�q                                    Bx��7  T          @����ÿ��
@`  B�CD�f���ÿ��\@e�B�CAff                                    Bx��E�  �          @�����׿��R@a�B�CGB����׿�(�@hQ�B{CD�                                    Bx��Td  �          @�
=��=q��Q�@VffB�\CIW
��=q��
=@^{BffCFW
                                    Bx��c
  "          @����
��=q@QG�B��CG�)���
��=q@XQ�B
=CD�H                                    Bx��q�  �          @�{������
=@\��BffCM}q������z�@eBz�CJO\                                    Bx���V  �          @�\)�z=q��@hQ�B\)CO�q�z=q���@q�B%33CL�H                                    Bx����  �          @�
=�y���ff@]p�B  CS
=�y���z�@hQ�BCO�R                                    Bx����  �          @�ff�xQ��"�\@S�
BG�CU33�xQ��G�@`  BCRW
                                    Bx���H  
�          @�{��녿ٙ�@Q�B(�CIu���녿�
=@Y��BQ�CFY�                                    Bx����  �          @�Q������&ff?�{A�p�CP�������(�@�
A�z�CO{                                    Bx��ɔ  T          @��R����p�@�A���CM�����G�@��A��
CK{                                    Bx���:  �          @�
=���\��@(�A��CH�����\�У�@%�A�z�CF��                                    Bx����  T          @�����{���R@)��AۮCB!H��{���\@/\)A�{C?�3                                    Bx����  T          @��H���\�
=@5�A��C:�)���\��33@7�A�\)C7�R                                    Bx��,  T          @�(�����B�\@7
=A陚C6)���=L��@7�A�(�C3n                                    Bx���  
�          @�G����?��@6ffA�\C$�3���?��@.�RA�C"ff                                    Bx��!x  �          @����ff?W
=@$z�A�  C*�=��ff?��@\)A�
=C(��                                    Bx��0  �          @�=q���?s33@'
=A�33C)k����?�@!G�A�33C'�                                    Bx��>�  �          @��H��p�?���@
=A��RC%����p�?\@�RA�=qC#�H                                    Bx��Mj  �          @������?�z�@p�A���C.����@�@�\A�(�C&f                                    Bx��\  T          @������@
=@\)A��C�\����@$z�@�A���C�
                                    Bx��j�  "          @����G�@8��@�A��C}q��G�@E@�
A���CǮ                                    Bx��y\  
�          @������@4z�@/\)A�\Cc����@C33@�RA�G�CB�                                    Bx���  
�          @����=q@<��@1G�A��HC\��=q@L(�@\)A�{C�                                    Bx����  �          @�p����@@��@$z�A�  Ck����@N�R@�\A��RCu�                                    Bx���N  T          @�����=q@?\)@!G�A�=qC�q��=q@Mp�@\)A���C��                                    Bx����  "          @���G�@>{@*=qA�  C����G�@L��@Q�A�ffC��                                    Bx��  "          @�ff��(�@@��@!G�A��
C�f��(�@N�R@�RA�=qC��                                    Bx���@  �          @�(����\@E�@
=A�ffC�����\@R�\@�
A�  C)                                    Bx����  �          @�=q�z�H@N{@33A¸RC�
�z�H@Z�H?�p�A��HC
޸                                    Bx���  �          @����u@P��@ffA�  C�f�u@^{@�A�\)C	�f                                    Bx���2  T          @���z=q@L(�@�A�{C���z=q@Y��@ ��A��
C
�3                                    Bx���  T          @����}p�@Dz�@�
A�(�CB��}p�@Q�@ ��A���Cn                                    Bx��~  �          @����=q@<(�@\)A�
=C5���=q@J=q@(�A�ffC(�                                    Bx��)$  
�          @�����@:�H@  A�33C  ���@G�?��HA��HC#�                                    Bx��7�  	�          @��H��\)@8Q�@  A�{C��\)@E?��HA�{C�H                                    Bx��Fp  
�          @�������@0��@
=qA��CT{����@=p�?��A�  Cz�                                    Bx��U  �          @����=q@0��@��A��HCp���=q@=p�?���A��C��                                    Bx��c�  �          @�����H@.�R@�A���C�����H@;�?�=qA�z�C�                                    Bx��rb  
�          @�G����@,��@(�A�ffC�����@9��?�z�A��C                                    Bx���  �          @�\)���@#�
@��A�CW
���@1G�?�
=A�33CO\                                    Bx����  T          @��R����@'�@��A��HC}q����@5�?�{A���C��                                    Bx���T  
�          @�Q����
@%�@	��A�=qCp����
@2�\?��A�\)Cs3                                    Bx����  
Z          @�\)��\)@�@G�A�
=C�=��\)@'�?�G�A���C�H                                    Bx����  
�          @��R��\)@�@�A�33C��\)@\)?�\)A��\C�                                    Bx���F  �          @�����\@�@A���C����\@��?���A�33Cp�                                    Bx����  
�          @��R��G�@ff@(�A�33C!H��G�@z�?���A��
C�H                                    Bx���  "          @�z���33@p�@�A�=qC���33@(�@�A��C�f                                    Bx���8  T          @�����R@  @G�A�C8R���R@�R@G�A��\C�f                                    Bx���  �          @�����R@��@�A��
C����R@\)@G�A�z�C�=                                    Bx���  "          @�z���{@?�(�A��C:���{@"�\?ٙ�A�\)C=q                                    Bx��"*  �          @����@33?�
=A�33C����@   ?�A�33C��                                    Bx��0�  "          @�����  @(�@ ��A�
=C{��  @��?�G�A�  C�R                                    Bx��?v  "          @�(���Q�@p�?�z�A�=qC����Q�@=q?�33A�
=C�                                    Bx��N  "          @��
��z�@�
?�  A���C  ��z�@  ?�G�A���C#�                                    Bx��\�  �          @�z����H@��?�A��C  ���H@�?ǮA���C{                                    Bx��kh  T          @���(�?��@�A��C�q��(�@�?��A���C^�                                    Bx��z  T          @��R��\)?�?��RA��CǮ��\)@33?�\A�33C��                                    Bx����  "          @�
=��G�?��?���A��C�=��G�@�?˅A�\)C��                                    Bx���Z  "          @��R��  @   ?��A�\)C#���  @��?�ffA��C0�                                    Bx���   �          @��R����@   ?�p�A��
C=q����@(�?��RA|��C\)                                    Bx����  T          @�
=���?�p�?�z�A��C�f���@
=q?�AqG�C�{                                    Bx���L  "          @�\)���H?���?�33A�{C)���H@Q�?�z�An=qCL�                                    Bx����  "          @������@   ?���A��C5�����@p�?���A��C&f                                    Bx����  �          @�ff��?�  ?�{A���C%� ��?�(�?ٙ�A�ffC#s3                                    Bx���>  �          @��R��33?p��?�
=A��C)�=��33?��?ǮA���C's3                                    Bx����  "          @�{����?���?�G�A�p�C$������?�ff?��Ac�
C"�
                                    Bx���  �          @�ff��ff?��
?���Aj�RC @ ��ff?�
=?�33AC�C��                                    Bx��0  �          @�ff��z�@	��?�{A�{C���z�@
=?˅A���C�q                                    Bx��)�  T          @����@   @��A��HCW
���@0  ?�A�=qC�                                    Bx��8|  T          @��R���R?�
=?�(�A�p�C�q���R@
=q?�(�A��HCff                                    Bx��G"  
(          @����Q�@��@��A�33C\��Q�@��?�z�A�
=C}q                                    Bx��U�  T          @�\)���\@#�
@
=A�  Ck����\@333?�\A�C
                                    Bx��dn  
Z          @�Q�����?���@�A�Q�C������@\)@
�HA�  C�
                                    Bx��s  
Z          @�\)��{@@=qA���C���{@Q�@�A���C��                                    Bx����  T          @�ff���@�R@ffA��
CY����@.�R?�\A��
C��                                    Bx���`  �          @����=q@"�\@{A�\)C���=q@3�
?��A�ffC�q                                    Bx���  �          @�Q�����@z�@�A�33C����@%�?�ffA�
=CG�                                    Bx����  �          @��R��(�@8Q�@A�  C���(�@HQ�?ٙ�A�
=C��                                    Bx���R  
�          @�{�l��@N{@
=A��C
��l��@`  ?�A���C��                                    Bx����  "          @�
=�s33@Mp�@��A�C���s33@^{?���A�(�C	�=                                    Bx��ٞ  
(          @�
=�o\)@U@��A��C
:��o\)@e?�
=A�{C&f                                    Bx���D  "          @�Q��fff@dz�@
=A���C33�fff@tz�?�\)A�p�CG�                                    Bx����  
�          @�ff�^�R@^�R@A�G�C��^�R@p��?���A�G�C��                                    Bx���  
�          @��R�mp�@R�\@G�Aď\C
u��mp�@c�
?�A���C.                                    Bx��6  "          @��R�r�\@I��@�A�{C@ �r�\@\(�?��A���C	��                                    Bx��"�  	�          @�p��[�@_\)@�
A�=qCz��[�@qG�?���A��CG�                                    Bx��1�  
�          @�ff�S33@dz�@��A�Q�C���S33@w�?�33A��C}q                                    Bx��@(  �          @�\)�QG�@a�@%A�(�C� �QG�@vff@�A�33CJ=                                    Bx��N�  
�          @�{�N{@XQ�@1�A�  C���N{@n�R@�\A�G�C�
                                    Bx��]t  
�          @�  �QG�@Z�H@0��A�ffC��QG�@p��@G�AÙ�C                                      Bx��l  �          @���Fff@[�@<(�BC#��Fff@s�
@(�A�33C5�                                    Bx��z�  �          @�  �QG�@]p�@/\)A�\)Ch��QG�@s�
@�RA��
C�                                    Bx���f  �          @�ff�]p�@]p�@��A���C�]p�@p��?��A�Q�C�H                                    Bx���  
�          @�{�Y��@^{@(�A�Ch��Y��@q�?�
=A���C�R                                    Bx����  �          @�ff�XQ�@c�
@�A�{Cz��XQ�@w
=?�A�  C.                                    Bx���X  �          @����AG�@k�@�A�p�CW
�AG�@~�R?���A�{B�W
                                    Bx����  "          @��\�HQ�@n{@��A���C��HQ�@�  ?�33A��C                                       Bx��Ҥ  T          @����@  @qG�@
=qA���C ���@  @�G�?���A�G�B�=q                                    Bx���J  
(          @�G��7�@p��@ffA�33B�Ǯ�7�@��?��
A�B���                                    Bx����  
�          @��
�@��@r�\@z�A���C n�@��@��H?�  A�  B��3                                    Bx����  
(          @�(��/\)@u@#�
A㙚B��f�/\)@�p�?�(�A�=qB�z�                                    Bx��<  
�          @�(��)��@qG�@/\)A��B�.�)��@�(�@
=qA�B�\)                                    Bx���  
�          @���C33@i��@p�A��C޸�C33@~{?�33A�Q�B�                                      Bx��*�  �          @��\�N�R@aG�@z�A�
=C���N�R@tz�?�\A�  C.                                    Bx��9.  "          @����G�@c33@�
AЏ\CT{�G�@vff?�G�A�Q�C�                                    Bx��G�  �          @����Fff@c�
@z�Aљ�C\�Fff@w
=?�G�A��HC ��                                    Bx��Vz  �          @����8��@l��@ffA���C \�8��@�Q�?�\A��
B��                                    Bx��e   �          @�\)�0  @p  @��A��B�aH�0  @�=q?�ffA�33B�                                      Bx��s�  	�          @�{�B�\@^�R@�A�33C)�B�\@s33?�A��C �f                                    Bx���l  
�          @���u@;�?��
A�C���u@J=q?��Alz�C��                                    Bx���  T          @�
=�(Q�@e@0  A��B���(Q�@~{@
�HA�  B�#�                                    Bx����  T          @�G���R@y��@$z�A�z�B��f��R@�  ?�Q�A�(�B��\                                    Bx���^  �          @����33@~{@%�A���B�8R�33@�=q?�Q�A���B��                                    Bx���  
�          @�\)�Vff@U@��A¸RC��Vff@hQ�?���A�\)C�3                                    Bx��˪  �          @�����Q�@8Q�?�  A/�C�q��Q�@@  ?
=q@���C�f                                    Bx���P  
�          @�Q���z�@<��?���Ah��C����z�@G
=?W
=A�C
=                                    Bx����  �          @��R��Q�@=q?�z�A�=qC}q��Q�@(��?�G�Ab�RC0�                                    Bx����  �          @�
=��G�@33?޸RA��C��G�@#33?�{As�CE                                    Bx��B  �          @�Q��p��@Fff?У�A�{Cz��p��@Tz�?�\)AIp�C
��                                    Bx���  �          @��R�Z�H@QG�?��RA�z�CL��Z�H@b�\?�Q�A��RC�q                                    Bx��#�  �          @�  �'
=@��H?���A���B���'
=@�33?��\A`��B���                                    Bx��24  "          @��
���\��\?���ArffC9� ���\��\)?���A}�C70�                                    Bx��@�  T          @�(���Q��?z�HA)p�C5k���Q�<�?}p�A*�RC3��                                    Bx��O�  "          @��
��\)>Ǯ?�  A,��C/����\)?��?k�A\)C-��                                    Bx��^&  
Z          @������
?5?�\)AB�HC,����
?aG�?}p�A+�
C*=q                                    Bx��l�  �          @�������?�ff?\)@�{C%�\����?�\)>�{@q�C$�                                    Bx��{r  T          @�\)���R@5?�=qA@��C����R@>�R?��@�p�C��                                    Bx���  �          @��\�g�@l(�?��HAS�Ch��g�@u?
=@�(�CG�                                    Bx����  
�          @�33�g�@r�\?p��A#
=C���g�@x��>���@P��C�                                    Bx���d  T          @�33�P  @~{?�z�Av�\CQ��P  @���?=p�@�ffC {                                    Bx���
  T          @����Z�H@|��?�G�AYC���Z�H@�33?
=@�G�C�\                                    Bx��İ  �          @�{��(�?n{?�z�A~ffC)33��(�?�33?��RA^{C&��                                    Bx���V  "          @����z�@1�?�  A��C&f��z�@?\)?�G�A2�RC.                                    Bx����  T          @�Q����
@(�?�Q�AUC�{���
@'
=?B�\A  C33                                    Bx���  
(          @��\���\?�ff?˅A�
=C"0����\?�ff?�ffAb�HC��                                    Bx���H  T          @�33��ff?�z�?У�A��RC�f��ff@
=q?��A_
=CT{                                    Bx���  
�          @����33?�33?�A��\C#ٚ��33?�?�z�Au��C!�                                    Bx���  �          @�
=��\)@33?�p�A�Q�C�=��\)@z�?���Af=qC�H                                    Bx��+:  "          @�����G�@(��?�z�Aq�C�{��G�@5?p��A
=C�                                    Bx��9�  T          @�33��Q�@8��?У�A��CY���Q�@HQ�?�{A6�HC@                                     Bx��H�  �          @�z�����@7
=?�z�A���C������@G
=?��A;33C��                                    Bx��W,  T          @����
=@;�?�p�A���C����
=@L(�?�Q�AD(�CxR                                    Bx��e�  "          @�ff��ff@L��?��
AyG�CL���ff@Z�H?s33A��C�                                     Bx��tx  "          @����33@7�?�{A�z�C��33@G
=?�=qA0��C�                                    Bx���  T          @����(�@+�?�p�Aw�
C  ��(�@8��?}p�A$  C�                                    Bx����  T          @��H��G�@:�H?�Q�An�HC33��G�@HQ�?fffA  C^�                                    Bx���j  �          @�����=q@7�?��AV�\C�H��=q@C33?B�\@��\CE                                    Bx���  
�          @�����ff@>{?�{Ac�CG���ff@J�H?L��A�C�
                                    Bx����  T          @�=q����@>�R?�  APQ�C������@J=q?333@�
=C#�                                    Bx���\  �          @��H����@AG�?���A;33Ch�����@K�?�@��C
                                    Bx���  �          @��\��\)@Fff?�Q�ADQ�C@ ��\)@QG�?(�@�\)C޸                                    Bx���  �          @�������@3�
?�{A8��Cٚ����@>{?z�@���CxR                                    Bx���N  
(          @����G�@A�?���A1�CG���G�@J�H?   @�ffC
=                                    Bx���  �          @������@A�?xQ�A ��CY�����@J=q>���@��
CE                                    Bx���  T          @�G���Q�@=p�?�(�AK33C�q��Q�@HQ�?&ff@�G�C=q                                    Bx��$@  �          @�G���=q@9��?�z�AAC����=q@C�
?(�@�=qC33                                    Bx��2�  "          @����@1�?�33A?\)CO\��@<(�?�R@���C�
                                    Bx��A�  "          @��\��{@7
=?�  A$z�C����{@?\)>�ff@��Ck�                                    Bx��P2  �          @��H��p�@=p�?Tz�A��C�H��p�@C�
>�=q@/\)C�q                                    Bx��^�  T          @�33���H@E�?c�
A�RC����H@L��>���@EC(�                                    Bx��m~  T          @��\���\@B�\?k�A�Cu����\@J=q>���@Z�HCs3                                    Bx��|$  �          @�����@N{?\(�A33C  ��@Tz�>u@ ��C�                                    Bx����  T          @������
@R�\?L��Ap�C����
@XQ�>.{?޸RCE                                    Bx���p  �          @��H��Q�@L��?.{@���C�f��Q�@QG�=u?(��C�                                    Bx���  �          @����\)@S33?333@�p�C�H��\)@W�=u?�RC�                                    Bx����  �          @�����33@@  ?�@���C�f��33@C33�#�
���Cn                                    Bx���b  �          @������@QG�?Y��A��C����@W�>L��@�CB�                                    Bx���  T          @�G���\)@H��?\(�A=qC���\)@O\)>u@{C!H                                    Bx���  "          @�����  @HQ�?0��@�\)C0���  @L��=��
?Tz�C�=                                    Bx���T  
�          @�������@Z=q?O\)AffCs3����@`  >�?��C��                                    Bx����  �          @�������@]p�?=p�@�C������@b�\=u?z�CY�                                    Bx���  �          @������\@j=q?@  @�Q�C
!H���\@o\)<��
>aG�C	��                                    Bx��F  
�          @����x��@r�\?uA ��C�q�x��@z=q>W
=@Q�C�H                                    Bx��+�  T          @�  ����@i��?J=qA  C	ٚ����@n�R=u?(�C	33                                    Bx��:�  T          @�
=��33@L��?B�\@��RC�H��33@R�\=�G�?�C�f                                    Bx��I8  T          @����G�@N{?(�@θRC\��G�@Q녽#�
��
=C�{                                    Bx��W�  
�          @���(�@J=q>�G�@�p�C&f��(�@L(��B�\��\C�                                    Bx��f�  T          @�{���@N�R>Ǯ@���C}q���@O\)��  �+�C^�                                    Bx��u*  T          @�(����@Dz�?(��@��C�)���@H��=#�
>ǮC@                                     Bx����  
�          @������@:�H?�@���C&f����@=p���\)�G�C�                                     Bx���v  �          @�ff��{@.�R?�@���C�\��{@2�\<��
>.{CB�                                    Bx���  �          @�{��z�@H��?&ff@��Ck���z�@Mp�    <#�
C�)                                    Bx����  T          @�(���=q@HQ�?!G�@���C{��=q@L�ͼ#�
��G�C��                                    Bx���h  �          @�����@Z�H?:�H@�  Cz�����@`  <�>�=qCٚ                                    Bx���  �          @�����
@Z�H?B�\A�CB����
@`��=L��?�C�
                                    Bx��۴  
�          @������@\(�?.{@�G�C�3����@`  ����Q�C(�                                    Bx���Z  
Z          @��
���@HQ�?0��@�{C����@L��=#�
>�ffC\)                                    Bx���   T          @����ff@N{>��@�{C�=��ff@O\)����1G�Cff                                    Bx���  �          @��
��Q�@P  >���@�  C����Q�@P�׾�\)�>{C��                                    Bx��L  �          @�
=�z=q@Y��?�R@ٙ�C��z=q@\�ͽ�Q쿅�C
�{                                    Bx��$�  �          @�z��\)@L��?\)@�Q�C:��\)@P  ��G���G�C��                                    Bx��3�  T          @�(��p��@Z�H?�R@���C	���p��@^{��G���Q�C	E                                    Bx��B>  T          @�z��}p�@Mp�>�@��C���}p�@O\)�L����RC�R                                    Bx��P�  T          @���s33@Z=q?5@��
C
��s33@^�R���
�k�C	xR                                    Bx��_�  
(          @�z��qG�@Y��?:�HA{C	��qG�@^�R    ���
C	O\                                    Bx��n0  "          @�G��\(�@c�
?aG�A!�C��\(�@j�H=�?��C0�                                    Bx��|�  �          @���s�
@O\)?z�HA1��C�H�s�
@W�>�\)@HQ�C
z�                                    Bx���|  
�          @�(����@>{?8Q�A ��C#����@C33=�\)?Tz�CaH                                    Bx���"  �          @�33���H@>�R?�\@�G�C����H@AG�������C��                                    Bx����  �          @�����@>�R?�@�z�C����@AG�����{C��                                    Bx���n  
�          @�(�����@1G�>�33@}p�C=q����@1녾u�.{C!H                                    Bx���  
�          @�z����@8Q�>��
@g
=C�{���@8Q쾙���U�C�\                                    Bx��Ժ  
e          @�����@1�?�@��HC0����@5��#�
��
=C�f                                    Bx���`  
�          @��
��
=@Q�?&ff@陚C�R��
=@{=�?�\)C#�                                    Bx���  
�          @�(���z�@%=u?!G�Cn��z�@"�\�   ��p�C�f                                    Bx�� �  �          @�{��=q@HQ�>��@��\Cu���=q@H�þ�=q�C33CT{                                    Bx��R  �          @�����H@E?
=q@���C����H@HQ�\)���C�=                                    Bx���  �          @�
=���\@0  ?@  AG�C�H���\@5>\)?�G�C��                                    Bx��,�  
�          @�
=��ff@8��?k�A$  Cs3��ff@AG�>�=q@?\)CB�                                    Bx��;D  �          @�z����H@?\)?�@��
C�����H@C33���Ϳ���CY�                                    Bx��I�  	�          @��
��ff@33?k�A&�HC����ff@(�>���@��RC5�                                    Bx��X�  �          @�����G�@
�H?�  A3�
C}q��G�@�?   @�33C��                                    Bx��g6  "          @��
����@{?p��A(��C�)����@�>�(�@�G�CT{                                    Bx��u�  
�          @�z����\@ ��?�33AN�HC޸���\@,��?�@��HC�                                    Bx����  
�          @����y��@A�?���A�z�C+��y��@Q�?=p�A(�C�q                                    Bx���(  S          @�p�����@5�?��A���C�����@Fff?aG�A(�Cp�                                    Bx����  "          @�{��ff@,��?�33AzffCL���ff@<(�?B�\A�C�                                    Bx���t  "          @�33���@Q�?���AM�Ch����@$z�?�@�{C�                                     Bx���  �          @�z���\)@z�?#�
@���C�f��\)@��=�G�?��RC��                                    Bx����  T          @��H��=q@��>�z�@N�RCQ���=q@�;W
=�=qC=q                                    Bx���f  "          @�(���{@ ��>�(�@���Cu���{@#33�����
=C(�                                    Bx���  T          @�33���\@'
=?�@�=qC����\@*=q��Q�xQ�Cs3                                    Bx����  �          @�����@(Q�>�z�@L��C޸���@(Q쾣�
�c�
C��                                    Bx��X  
(          @�����@�=�G�?���C�=��@��G���
=C.                                    Bx���  �          @�
=��(�@�
��\)�N�RC&f��(�@(��L���  Cn                                    Bx��%�  H          @�{���@&ff>u@2�\C����@%��Q���ffC\                                    Bx��4J  �          @�p���@ff=#�
>��C����@�
��G�����C)                                    Bx��B�  
�          @�\)����@\)?z�@�p�C����@�
=L��?�CE                                    Bx��Q�  �          @��R��{@��>�G�@�(�CE��{@(��u�333C�\                                    Bx��`<  
�          @�����(�@�þ\��\)C�3��(�@   �\(��"�RC�                                     Bx��n�  �          @��
��=q@{>L��@��C���=q@p����R�l��C�f                                    Bx��}�  �          @�z����@��?
=@޸RC����@�=�\)?E�C!H                                    Bx���.  
�          @��\��@��?c�
A)G�C����@=q>�{@���C�                                    Bx����  
N          @�G��l(�@>{?�@���C��l(�@AG��.{��C��                                    Bx���z  "          @��
�^�R@S�
?B�\A��Cs3�^�R@X�ü����
C��                                    Bx���   
Z          @���z�H@$z�?��RAm�C�=�z�H@2�\?(�@�C��                                    Bx����  �          @��H�}p�@   ?�z�A_
=C�q�}p�@,��?��@ϮC��                                    Bx���l  �          @�33�p  @1�?�(�Ak
=C}q�p  @?\)?
=q@��Cz�                                    Bx���  
�          @���z=q@(Q�?�
=A`��C��z=q@5?�@�\)C�                                    Bx���  �          @��H�u@333?uA7�C�f�u@<��>�\)@S33Cz�                                    Bx��^  "          @����fff@?\)?^�RA(z�CJ=�fff@G
=>\)?�
=C+�                                    Bx��  �          @���W�@HQ�?���AX(�C	#��W�@S33>���@���C��                                    Bx���  �          @�ff�j=q@1�?xQ�A?\)C�3�j=q@<(�>�z�@^�RC:�                                    Bx��-P  T          @�{�qG�@(Q�?fffA0z�C.�qG�@0��>�  @AG�C�=                                    Bx��;�  
�          @���hQ�@3�
?�\)A[�C=q�hQ�@@  >�
=@��HCp�                                    Bx��J�  
�          @�G��h��@9��?���AUp�Cp��h��@E�>\@�G�C                                    Bx��YB  �          @���Tz�@N{?}p�AAC�
�Tz�@W�>B�\@�C�
                                    Bx��g�            @�  �R�\@S�
?\(�A'�C��R�\@Z�H=#�
?�C�{                                    Bx��v�  
�          @��R�h��@;�>�  @B�\C#��h��@:=q����Cc�                                    Bx���4  
�          @����Tz�@R�\?k�A1G�CB��Tz�@Z�H=���?�33C33                                    Bx����  
�          @���Tz�@QG�?�
=AdQ�Cu��Tz�@\��>�Q�@��Cٚ                                    Bx����  	�          @���_\)@H��?uA8��C
  �_\)@Q�>.{@�\C�                                    Bx���&  �          @����c33@G�?��@�C
�R�c33@J=q�aG��(Q�C
W
                                    Bx����  "          @����n�R@6ff?.{A
=C�{�n�R@;��#�
��ffC�{                                    Bx���r  
Z          @�\)�n�R@/\)?G�A(�C�H�n�R@6ff=��
?��
C�
                                    Bx���  	�          @��R�QG�@L(�?��AQ�C��QG�@Vff>�  @Dz�CW
                                    Bx���  T          @�{�^�R@?\)?k�A6ffCT{�^�R@G�>.{@�C
�                                    Bx���d  "          @��R�>{@]p�?��AP��C���>{@g
=>8Q�@
=qCaH                                    Bx��	
  �          @����R@|��?5A
�\B�33��R@�  ��\)�Z=qB�                                    Bx���  �          @�  �Q�@}p�?=p�A��B�  �Q�@��׾���L��B�=q                                    Bx��&V  T          @����2�\@j=q?�  As
=B��\�2�\@w
=>��R@mp�B��R                                    Bx��4�  
�          @�\)�C33@Q�?�(�A��C��C33@b�\?(�@�RC�                                     Bx��C�            @���L(�@H��?�  A�ffCs3�L(�@Z=q?.{A(�C�                                    Bx��RH  �          @�
=�l��@J�H?c�
A%p�CY��l��@S33=�\)?\(�C
G�                                    Bx��`�  
�          @���e�@S�
?(��@�p�C	.�e�@W��B�\��C�                                    Bx��o�  "          @��H�L(�@aG��0���\)C5��L(�@O\)������C��                                    Bx��~:  
�          @�z��P��@S33>L��@\)C���P��@O\)�!G�����C)                                    Bx����  �          @��H�W
=@G
=>aG�@3�
C	&f�W
=@Dz�\)���C	�=                                    Bx����  �          @�z��\(�@B�\?�R@�p�C
���\(�@E�#�
�   C
{                                    Bx���,  T          @��R�XQ�@J�H?.{AC�)�XQ�@O\)�\)��z�C=q                                    Bx����  
�          @����}p�@*=q?�@�
=C
=�}p�@-p��.{���RC�
                                    Bx���x  T          @��
�}p�@1�>�@��RC���}p�@3�
��  �>�RC�f                                    Bx���  "          @������@\)?�R@�G�C����@$z������CW
                                    Bx����  
�          @�z�����@�?�@ǮC������@�u�=p�CW
                                    Bx���j  �          @��H��  ?޸R=�Q�?��
C����  ?ٙ���Q����CG�                                    Bx��  �          @���z�H@\)?���A��RC���z�H@1�?E�A�C��                                    Bx���  "          @����i��@>{?W
=A!��C�H�i��@E=u?+�C�\                                    Bx��\  �          @��\�s33@2�\?s33A6{C���s33@<(�>W
=@p�C8R                                    Bx��.  
�          @�(���z�@�?}p�A<��C(���z�@#33>�Q�@�ffC^�                                    Bx��<�  �          @�z�����@=q?k�A,��C�)����@#�
>�=q@J=qCL�                                    Bx��KN  "          @�����@%?fffA)p�C\)����@/\)>L��@��C�                                    Bx��Y�  "          @�
=�
=@��R>\@��
B��
=@��ͿG����B�33                                    Bx��h�  "          @����  @��>k�@)��B�G���  @�ff��G��<  B��H                                    Bx��w@  "          @������@���>\)?���B�녿���@��Ϳ�z��O�
B̏\                                    Bx����  
�          @��Ϳ��@�\)=�Q�?xQ�BϞ����@��\��G��X��B�k�                                    Bx����  
�          @�\)���@�33����{B�����@�z��G���
B�{                                    Bx���2  �          @����  @�z��G���
=B�녿�  @���  �~�RB�                                    Bx����  �          @�{���
@��\��=q�6ffB��f���
@��\�������B���                                    Bx���~  
�          @����33@�G���33�i��BѨ���33@��׿��H��p�B�33                                    Bx���$  �          @�p���p�@�ff��(���G�B���p�@����\��(�BՊ=                                    Bx����  �          @�\)��z�@��\��ff��Q�B̞���z�@��ÿ�=q��  B�
=                                    Bx���p  �          @����(�@��\��Q쿃�
B�Ǯ��(�@�(������HB�.                                    Bx���  z          @�=q�u@��þ�ff��(�B�aH�u@����  ��Q�B�z�                                    Bx��	�  �          @��׿!G�@���&ff��z�B��
�!G�@�녿�p�����B��
                                    Bx��b  
�          @�
=��=q@�=q�����=qB��Ὴ=q@����
=���B�u�                                    Bx��'  
�          @�(��z�H@���8Q���B�G��z�H@��
�z����B��H                                    Bx��5�  
�          @��H����@���B�\��B������@��������B���                                    Bx��DT  
Z          @�zῐ��@�(������=B��)����@����H��  B�G�                                    Bx��R�  T          @�{�k�@�\)����A�B��
�k�@�  ��R��p�B���                                    Bx��a�  �          @�33�xQ�@�ff�L���	p�B�=q�xQ�@���������B��                                    Bx��pF  "          @��
��33@�ff�(������B��ÿ�33@��H� ����{B�                                    Bx��~�  T          @��H���@��
�������B�p����@�G������B�k�                                    Bx����  �          @��׿�  @�{��ff���
B���  @�z�޸R����B�L�                                    Bx���8  T          @�zῡG�@�ff���˅BЊ=��G�@��
��{��Q�BҀ                                     Bx����  �          @����@��������>{Bٮ���@��H��
��p�B�Q�                                    Bx����  T          @��
��@��
��{�H��B��f��@z�H��R��{B�Q�                                    Bx���*  �          @�z��\)@��\��z��Pz�B����\)@w
=�G���ffB���                                    Bx����  
�          @���<(�@��׿�
=�U�B�L��<(�@c�
�p����HC�=                                    Bx���v  
�          @�z��Y��@l�Ϳ�G��5��C���Y��@S33��(���=qC�H                                    Bx���  "          @��dz�@e���\�6{C���dz�@L(��������C
@                                     Bx���  "          @����l(�@^{�k��#�
C���l(�@Fff������33C�3                                    Bx��h  
�          @�
=�w�@Tz῏\)�E��Ch��w�@8�ÿ�(���G�C8R                                    Bx��   
�          @���fff@h�ÿ���:�\C���fff@N{�   ���
C
+�                                    Bx��.�  
�          @���k�@X�ÿ��R�_33C	\)�k�@;��
=��p�C}q                                    Bx��=Z  T          @�(�����@0�׿����(�C�����@\)��R�Σ�C�)                                    Bx��L   �          @�=q����@33���H��Q�C������?��������C�                                    Bx��Z�  
�          @�\)����@Q쿱����
C�����?�����33C��                                    Bx��iL  
�          @��R��ff@�
���
�n�\C#���ff?�׿�{���
C޸                                    Bx��w�  T          @���(�@����p��f�\C��(�?�p���=q����C\)                                    Bx����  T          @��R���@%��{�O�Cff���@(���\��Q�C��                                    Bx���>  T          @�\)�g�@S33�E��=qC	���g�@>{�����=qC��                                    Bx����  �          @�\)�mp�@E�����M�C(��mp�@*�H��z���
=C0�                                    Bx����  �          @�\)�(Q�@�>�
=@�(�B�\)�(Q�@�(��=p����B��f                                    Bx���0  �          @�ff�6ff@���>�\)@P  B��\�6ff@|�ͿTz��33B���                                    Bx����  �          @�{�Mp�@o\)>L��@�C�f�Mp�@i���Tz��\)CT{                                    Bx���|  �          @�  �%@�(��s33�/�
B�(��%@mp���\��ffB�Ǯ                                    Bx���"  	�          @����G�@z�H�u�2�\C u��G�@l�Ϳ���r�RC�                                    Bx����  "          @��\�R�\@vff���
�c�
Cz��R�\@k������O\)C                                    Bx��
n  �          @�  �W�@l(�    <��
Cp��W�@c33��G��;\)C�\                                    Bx��  �          @�G��^{@h�þu�1�C�
�^{@[����R�d  CG�                                    Bx��'�  "          @����W�@l(���������Cff�W�@\(���33���
Cn                                    Bx��6`  �          @�=q�c�
@e����H��  C���c�
@S�
���H����C	{                                    Bx��E  �          @�
=�aG�@^�R�Ǯ����CT{�aG�@O\)����z�HC	h�                                    Bx��S�  
�          @�33�^{@W��.{��\C���^{@L(������R=qC	\)                                    Bx��bR  
�          @����p  @G
=�����\)CT{�p  @7
=����x��C��                                    Bx��p�  "          @�Q��e�@C33�B�\�=qC�)�e�@.{������Q�C�\                                    Bx���  �          @�\)�h��@?\)�\)���C���h��@-p���\)��ffCJ=                                    Bx���D  "          @�(��W
=@`�׿   ��CǮ�W
=@N�R���H��G�C&f                                    Bx����  "          @��R�E�@vff���
�j=qC ���E�@g
=������\)C��                                    Bx����  �          @�
=���
@)���O\)��RCG����
@�
�\���\C��                                    Bx���6  
Z          @����@{�.{� (�Cs3��@������~�HCxR                                    Bx����  �          @�{��\)@(��8Q��
=C���\)@�ÿ������C0�                                    Bx��ׂ  T          @�{����@�H������
Ch�����@
�H�����a�C                                    Bx���(  "          @�\)����@
=�z�H�5�Cff����?�  ���
���CO\                                    Bx����  
�          @����G�?����=q�w�C!G���G�?��Ϳ��H��z�C&T{                                    Bx��t  �          @��\��p�?��Ϳ�����C&� ��p�?
=�����z�C,�q                                    Bx��  
�          @�����33?��H��\)��Q�C%L���33?333�����
=C+Q�                                    Bx�� �  �          @�G����?У׿}p��5�C �����?����z����C$�\                                    Bx��/f  T          @����|��@%��
=q��
=C�{�|��@����R�rffC�                                     Bx��>  
(          @�p��Q��(��,(��	33CU��Q��7
=���H��(�C]�                                    Bx��L�  "          @��2�\���Dz��$��CY@ �2�\�9�����p�Cb�                                    Bx��[X  �          @�G��1G��%�7
=�{C_\�1G��R�\� ����ffCe�                                    Bx��i�  T          @�=q�&ff���>�R�"�C]8R�&ff�A��p����
CeW
                                    Bx��x�  �          @��H�;��
=�:=q�p�CW���;��6ff�(����C`!H                                    Bx���J  "          @�  �mp��˅� ����\)CK#��mp���R���H���HCS�                                    Bx����  �          @���?\)��p��,(��z�CU���?\)�*�H�G����
C]��                                    Bx����  �          @��R�+��У��Dz��1z�CSG��+��(���R�	��C^:�                                    Bx���<  T          @�
=�Tz῔z��0  ���CGO\�Tz��33�33��=qCQ�R                                    Bx����  
�          @��R�P  �Q��<(��%  CB{�P  ��\)�%����CNu�                                    Bx��Ј  "          @��R�O\)����7�� �CF)�O\)��=q�����CQ�                                    Bx���.  "          @����G����\�2�\�z�CJ0��G�� ����
���CT�)                                    Bx����  �          @�(��@�׿�G��@  �-=qCF���@�׿����%��  CS5�                                    Bx���z  
�          @�z��l(��������CKc��l(���Ϳ�����RCRǮ                                    Bx��   T          @�p������׿�����33CK���������R�_�CO�
                                    Bx���  T          @�{��\)��������CN����\)�(Q쿰���xQ�CSٚ                                    Bx��(l  T          @�{��ff���H����
=CK�q��ff��ÿ��P��CPB�                                    Bx��7  �          @�z���z��(������HCL.��z�������R=qCP�q                                    Bx��E�  
�          @�33��
=�G���p����CPW
��
=�+���G��6�\CTff                                    Bx��T^  �          @������ �׿�Q���(�CS����8�ÿfff�#�CWz�                                    Bx��c  �          @�p��z�H��ÿ�33��  CP�f�z�H�&ff�����e�CU��                                    Bx��q�  T          @�
=�qG���G����R��G�CFn�qG����
��ff���CM=q                                    Bx���P  T          @�z��c�
������R���CH���c�
��{�\��  CO�
                                    Bx����  "          @���N{�\(��2�\��CB��N{�У�����\CNǮ                                    Bx����  "          @��
�Fff��{�XQ��=33C:8R�Fff��ff�H���,33CJ��                                    Bx���B  �          @�=q�k�����Q��"\)CD��k����H�5���CP                                    Bx����  "          @��
�l�Ϳ��*=q�CL=q�l���
=��
���CT��                                    Bx��Ɏ  
�          @�ff�u����H��Q����CL
�u��33�c�
�8��CP:�                                    Bx���4  
�          @�\)�~{���H��
=��Q�CH#��~{������
=��
=CN0�                                    Bx����  	�          @�33�c�
��\)��H���CL���c�
�  ������G�CT\)                                    Bx����  �          @�ff���ÿ�\��ff���CK�f������ÿz�H�@��CO��                                    Bx��&  
Z          @����=q����7
=�\)CI���=q�������̣�CR�                                     Bx���  T          @��U�ٙ��1���\CO��U���
�H���CX�                                    Bx��!r  �          @�Q��B�\�
=�5�{CV���B�\�5�
=����C_                                    Bx��0  "          @��H�L(��#33�#�
� ffCZ�3�L(��L(����H��=qCa                                      Bx��>�  
�          @����`���5�,(���(�C[��`���`  ��  ��G�C`�                                    Bx��Md  T          @��R�w
=�@���Q���p�CY��w
=�dzῳ33�o
=C^                                    Bx��\
  f          @��~�R�:�H����ģ�CX:��~�R�\�Ϳ���_�C\��                                    Bx��j�  �          @�Q���G��AG��{���
CX��G��a녿��R�Pz�C]+�                                    Bx��yV  �          @�Q����\�E��\��{CY!H���\�c33��ff�0z�C]                                    Bx����  �          @����H�C33��{��Q�CX�3���H�]p��c�
�33C\8R                                    Bx����  �          @�Q��6ff�W
=��
=���
Ce��6ff�hQ�Ǯ����Cgٚ                                    Bx���H  �          @�G��
=�_\)���
����Cn��
=�mp��L���,��Cps3                                    Bx����  �          @�{���Z�H�}p��]��Cn+����c�
=L��?.{Co33                                    Bx��  �          @����e����]Cl����o\)<�>�
=Cm�)                                    Bx���:  �          @����(��i�����\�P(�Cl@ �(��r�\=���?�p�Cm8R                                    Bx����  �          @�{�   �W���(����Co^��   �n�R�(���Cq�{                                    Bx���  T          @�z��
=q�b�\�
=q��=qCn���
=q���׿��
�K�Cq�=                                    Bx���,  �          @����G��e��  ��Cm���G��w��Ǯ���Co�
                                    Bx���  �          @�ff���hQ쿚�H�{�
Cm�H���tzὸQ쿗
=Co5�                                    Bx��x  �          @��H�(��fff���H��(�Ck���(��w���{��
=CmǮ                                    Bx��)  �          @��
���\�Ϳ�33���Cnn���q녿���\)Cp�q                                    Bx��7�  �          @��������QG��G���
=Co#׿����n{�xQ��O�CrE                                    Bx��Fj  �          @�p��   �|(���ff���\Cs��   ��
=���
�~{Ct�                                    Bx��U  �          @�������y����  ��p�Cq5����������ə�Cs0�                                    Bx��c�  �          @���G��}p���G�����Cp#��G���녿���(�Cr(�                                    Bx��r\  �          @��\�%�n�R�У���{Ck=q�%��������Q�CmaH                                    Bx���  �          @���"�\�p  ��{��z�Ck���"�\������G����RCm��                                    Bx����  T          @�z��'
=�c33���\���HCi���'
=�qG��.{�	��CkQ�                                    Bx���N  �          @�Q�����Dzῧ���z�Ck�����Tzᾳ33��=qCm&f                                    Bx����  �          @��R�8���8�ÿ����jffCa��8���E��.{�33Cb�)                                    Bx����  �          @��
�8Q��N{������\Cd8R�8Q��c�
��R��
=Cg\                                    Bx���@  �          @�G�����C33�
=��Cy�Ϳ���g
=��{��{C|��                                    Bx����  �          @�G���=q�HQ���
�Q�Cz𤿊=q�j�H���
���C}�{                                    Bx���  �          @�33�����L����R��Cy�������n{��Q�����C|(�                                    Bx���2  �          @�{��(��Vff�
�H���\Cy�3��(��u����q�C|W
                                    Bx���  �          @�\)�޸R�j=q��33��33Ct�׿޸R��녿=p���
Cvٚ                                    Bx��~  �          @�33�У��k���{����CvͿУ��\)����G�Cw��                                    Bx��"$  �          @��\�
=q�_\)����=qCn8R�
=q�p  ���
��Q�Cp�                                    Bx��0�  �          @�����p��c33����  Cs���p��xQ�����p�Cu�R                                    Bx��?p  �          @k����:�H�s33�v=qCm����Dz�#�
�8Q�Co�                                    Bx��N  �          @�����
�R�\�У���(�Cq����
�g��z���Cs��                                    Bx��\�  �          @fff�=q�   �E��G\)Cb��=q�'
=<#�
=�Q�CcL�                                    Bx��kb  �          @hQ��\)��ÿ�=q���RC_ٚ�\)�'
=���
���
CbG�                                    Bx��z  �          @qG��#33��H�����z�C_� �#33�.{�(����Cb޸                                    Bx����  �          @e����Ϳ���(��R  C_�����Ϳ��H�����=qCkT{                                    Bx���T  �          @n{��(����
�Z�H��C5&f��(����
�P���G�C\:�                                    Bx����  �          @�녿�(���\)�9���B��C[^���(������\��Cf�H                                    Bx����  �          @�(������l�Ϳ�����CrE�����x�ýL�Ϳ#�
Csn                                    Bx���F  �          @����Q�����=q�V�HCw�R��Q����>W
=@%�Cx�{                                    Bx����  �          @���ff�z�H�ٙ�����Cq���ff��  ����G�Cs��                                    Bx����  �          @��
�
=q���
���H��\)CrQ��
=q���
�.{�33Cs�                                    Bx���8  �          @��������\��G��r{Cq��������    =L��Cr�                                    Bx����  �          @��H���{�>��@P��Cp�����k�?�z�A��
CoQ�                                    Bx���            @���0  �e��8Q����Ch}q�0  �hQ�>�p�@�ffCh��                                    Bx��*  �          @�ff�Dz��Z�H�aG��.ffCd\�Dz��aG�>.{@33Cd�3                                    Bx��)�  �          @��\�Fff�>�R��������C_޸�Fff�;�?
=@�C_p�                                    Bx��8v  �          @�
=�I���2�\=���?�=qC]�
�I���(��?k�AJ=qC[��                                    Bx��G  �          @g
=�.{��>�(�@��CZ� �.{����?��A�ffCW��                                    Bx��U�  �          @H����>u?�Bz�C.p���?J=q?�G�B
(�C!��                                    Bx��dh  �          @<�Ϳ�{����?��A�ffC=
=��{��?(�A�C7��                                    Bx��s  �          @_\)����Ϳ�  �ɅCfY����1녿333�8z�Ci��                                    Bx����  �          @w��'��$z῎{��33C`c��'��1녾�����G�Cb�3                                    Bx���Z  �          @����4z��*�H�s33�[\)C_ff�4z��5���G���\)Ca{                                    Bx���   �          @�����Vff��33��Ck���p�׿Q��(z�Cn�                                    Bx����  T          @������Tz��
=��
=Cm)����w����\��ffCp�q                                    Bx���L  T          @��R�8���6ff���H��ffC`�)�8���I�������=qCc�                                    Bx����  �          @��R�Z�H�'
=�W
=�0  CYY��Z�H�/\)���
���
CZ�3                                    Bx��٘  �          @��
�O\)�,(���=q�f=qC[���O\)�8�þk��EC]�q                                    Bx���>  �          @����HQ��1녿�����C]�)�HQ��4z�>�\)@r�\C^\                                    Bx����  �          @k��'
=�\)�������C\�
�'
=�!녿
=�\)C`
=                                    Bx���  �          @L(��33���R�p�����
C\�{�33�����R��33C_n                                    Bx��0  �          @.�R�G����+��e�C[� �G�����\)�Dz�C]�                                    Bx��"�  �          @@������^�R��\)C]c���G���z���p�C`                                      Bx��1|  z          @8Q��
=��33�:�H�nffCR�\�
=��ff��\)��\)CU@                                     Bx��@"  �          @"�\�33���\�W
=��\)CK���33���
=���@CL8R                                    Bx��N�  �          @p���R��  �k�����CL{��R���\=��
?�CL�\                                    Bx��]n  �          @.{��R���������\CK!H��R���=u?���CK�R                                    Bx��l  �          @:�H�,(������33����CJ  �,(���논#�
���CK�                                    Bx��z�  T          @:=q�-p��xQ쾽p���CG�-p������\)����CI                                      Bx���`  �          @p��ff�
=q�����	�CBc��ff�(�����~�RCD=q                                    Bx���  �          @#33�\)>�׾�  ����C)B��\)>\�\�	�C+c�                                    Bx����  �          @'��%�>�녾L�����\C*���%�>�{���R��=qC,�
                                    Bx���R  �          @\)���>\=�\)?�C+=q���>\�#�
�}p�C+!H                                    Bx����  �          @*�H�%�?(�>W
=@�\)C&� �%�?#�
<��
>ǮC&�                                    Bx��Ҟ  �          @&ff�!G�?
=>�\)@�\)C&�
�!G�?&ff=���@ffC%��                                    Bx���D  �          @���ff>\�����z�C*�3�ff>�{�W
=���RC+��                                    Bx����  �          @7
=�5�>�(�=��
?�Q�C+E�5�>�G��L�ͿuC+(�                                    Bx����  �          @(���$z��G��8Q��z=qC=���$z���#�
�\(�C>��                                    Bx��6  �          @<(��#�
�����ff��CN���#�
��\)�#�
�Tz�CP�                                    Bx���  �          @S�
�"�\��p��@  �S
=CY�
�"�\�
=�\)�
=C[�                                    Bx��*�  T          @H������{�G��e�CY����� �׾L���e�C[��                                    Bx��9(  T          @Fff�  ��z�xQ���Q�C\n�  ����p���Q�C_L�                                    Bx��G�  �          @/\)�$z�?@  �L�Ϳ��\C#�3�$z�?5��=q���C$�                                    Bx��Vt  �          @hQ��G
=?\?p��At(�C�3�G
=?�p�>�ff@�  C�f                                    Bx��e  �          @`���@  ?�=q?�p�A�  C)�@  ?У�?G�AMCs3                                    Bx��s�  �          @P���5?��?Q�AjffC�5?\>Ǯ@�(�C��                                    Bx���f  �          @G
=�.{?�33>�Q�@�ffC�)�.{?�Q콏\)���
C!H                                    Bx���  �          @'��  ?�=q=L��?�ffCc��  ?��
��33��(�C@                                     Bx����  �          @9���'�?=p��aG���33C$&f�'�?&ff��(��z�C&&f                                    Bx���X  �          @\���5�\(���Q�����CD�5���ÿ�{��CL�
                                    Bx����  �          @]p��.{�u�z���C9��.{�W
=��z��p�CE@                                     Bx��ˤ  �          @Z=q�1�?�\����C)��1녽u��z��

=C5:�                                    Bx���J  �          @Tz��3�
?�ͿУ��홚C(�)�3�
=L�Ϳ�p���=qC3�                                    Bx����  �          @I����  ?�׿�(��(�C ����  ?������?�
CT{                                    Bx����  �          @<�Ϳ0��@z��ff�
=B�B��0��?�\�
=q�B�B��                                    Bx��<            @U���@*=q��  �׮B����@ff�{�(�B��H                                    Bx���  �          @mp�����@/\)�ٙ����B������@����$ffC��                                    Bx��#�  �          @W��G�@�
>�Ap�Cu��G�@ff�k��|��C                                      Bx��2.  �          @R�\��@�����
�ǮCz���@�\�:�H�QC�
                                    Bx��@�  �          @QG���=q@)������(z�B�=q��=q@
=��������B���                                    Bx��Oz  �          @S�
��@#�
�����C h���@33���H����CW
                                    Bx��^   �          @W���@�\�����C�H��@zῇ����C�                                     Bx��l�  �          @P���p�?�p�>���@�z�C5��p�?��R��=q���C
                                    Bx��{l  �          @2�\��33?�?�  A�ffC�=��33@Q�?&ffAZ=qB�z�                                    Bx���  �          @@�׿�{@p�?�33A�G�B�Q��{@��>�A�B��q                                    Bx����  T          @U���@1G�?xQ�A�\)B�\���@;�=�@��B�{                                    Bx���^  �          @5�aG�@+�=���?�Q�B�8R�aG�@%�0���c�B�aH                                    Bx���  �          @]p��u@Z�H�u�z�HB��þu@P�׿������B�\)                                    Bx��Ī  �          @N{��?˅?E�A��C���?�  >�=q@��C�f                                    Bx���P  �          @O\)�1�?Q�?�Q�A�=qC#��1�?��?fffA��HC�\                                    Bx����  �          @Q���\?���?��A�\)C33��\@
=q?=p�AS�C�=                                    Bx���  �          @z=q�޸R@X��>��@(�B�W
�޸R@Q녿Y���N{B���                                    Bx���B  �          @|(���ff@fff>�p�@�33B�z��ff@c�
�333�#�
B�\                                    Bx���  �          @n{�xQ�@XQ�?�33A���B��f�xQ�@dz�=�G�?�G�B�L�                                    Bx���  �          @c33���
@�\?�(�B��B��f���
@��?�ffA�33B�{                                    Bx��+4  �          @i����ff@.�R?���A�
=B�B���ff@A�?��A33B�.                                    Bx��9�  �          @{��   @vff>Ǯ@�{B�Ǯ�   @s33�@  �1p�B���                                    Bx��H�  �          @w����R@c�
?�@���Bڅ���R@c�
�
=q��Bڊ=                                    Bx��W&  �          @�  ����@q�?\)A Q�B�=q����@q녿���HB�B�                                    Bx��e�  �          @y�����@e�?J=qA:�RB�33���@i����z���
=B�p�                                    Bx��tr  �          @tzῳ33@X��?��A�=qB��
��33@c�
=u?\(�B��f                                    Bx���  �          @w
=�n{@i��?333A(��B�z�n{@l�;Ǯ���B��                                    Bx����  �          @w��fff@hQ�?L��A@  B��)�fff@l�;�z����
B�Q�                                    Bx���d  �          @a녿�33@8Q�?��A�33B�p���33@J=q?   AffB��                                    Bx���
  �          @i���^�R@Y��?^�RA_�BЮ�^�R@`�׾���B��
                                    Bx����  �          @e�Ǯ@'
=?��HA�z�B��Ǯ@@  ?aG�Aip�B��H                                    Bx���V  �          @`���!�?���?�
=B
G�C�)�!�?��H?�  A�p�C�q                                    Bx����  �          @`  �/\)>W
=@��BffC/���/\)?Q�?��RB
=C#n                                    Bx���  �          @c33�A�>L��?�=qA���C0@ �A�?8Q�?�Q�A�z�C&��                                    Bx���H  �          @g
=�A�>�ff?�A�ffC+�H�A�?s33?���A�=qC"��                                    Bx���  �          @c33�*=q?^�R@ffB�\C!��*=q?�Q�?�G�A�C��                                    Bx���  �          @j=q��\@(�@
=BC�H��\@-p�?��A�ffB��H                                    Bx��$:  �          @k���@{@Q�BG�B�8R��@>�R?�=qA�z�B��                                    Bx��2�  �          @l�Ϳ�z�@�@�B  B�{��z�@;�?�G�A��RB�33                                    Bx��A�  �          @�33��=q@0��@!�B�B���=q@W�?У�A��B�Q�                                    Bx��P,  �          @z=q���?�ff@,(�B2��Cp����@G�@��BC��                                    Bx��^�  �          @u����@��?�
=B
�HB� ���@%?z�HA�p�Bߣ�                                    Bx��mx  �          @w
=?=p�@mp�?#�
A�RB�u�?=p�@n�R��G����B���                                    Bx��|  �          @�G�=���@qG�?�\)A�p�B��)=���@�Q�>k�@S�
B�\                                    Bx����  �          @��
���
@hQ�?�Aߙ�B������
@���?O\)A5p�B�\)                                    Bx���j  �          @��>�(�@z�H?�ffA�ffB�p�>�(�@�(�>�?�ffB�{                                    Bx���  �          @��?@  @~�R?c�
AF=qB���?@  @�=q��\)�z=qB�{                                    Bx����  �          @{�?fff@q녾W
=�H��B��?fff@dzΰ�
���
B��=                                    Bx���\  �          @�녿��@U�?�p�A�\)B����@g
=>��@�
=Bܞ�                                    Bx���  T          @����@e�?�z�A�  B�
=���@z=q?�@�B��H                                    Bx���  �          @vff�*�H?@  @B33C$=q�*�H?���?��A��\C��                                    Bx���N  �          @����Mp��(��@
=B{C?�H�Mp�=#�
@��Bp�C3W
                                    Bx����  �          @���G�=���@0  B%�\C2(��G�?aG�@'
=B\)C$O\                                    Bx���  �          @�  �AG�?!G�@"�\B��C(O\�AG�?���@��B	��Cff                                    Bx��@  T          @s�
��=�@?\)BR�C1���?u@5�BC�C&f                                    Bx��+�  �          @x���
�H>.{@L(�B^�HC/�H�
�H?��@@��BM33C�                                    Bx��:�  �          @`���
�H?��@*�HBG\)C$�)�
�H?�=q@��B,�HC��                                    Bx��I2  �          @^�R�{?8Q�@$z�B>��C!��{?�@��B"�\Cff                                    Bx��W�  �          @`  �G�?��@(�B1(�CG��G�?�Q�@�\B\)CT{                                    Bx��f~  �          @^{�!�?��@33B��C�R�!�?�?��A㙚C��                                    Bx��u$  �          @hQ��p�@#33>Ǯ@�
=C���p�@#�
��33��33C��                                    Bx����  
�          @n�R�?\)@�?fffA`Q�C�)�?\)@��>���@��C��                                    Bx���p  T          @i���8Q�@33?h��Af{C�
�8Q�@{>���@�z�CaH                                    Bx���  �          @l���A�?�(�?�p�A���Cn�A�?��R?0��A-G�C�f                                    Bx����  �          @j=q�Fff?���?��
A��RC���Fff?�  ?Tz�AQp�C��                                    Bx���b  �          @k��J�H?���?�p�A�Q�C W
�J�H?�  ?�{A�=qC��                                    Bx���  �          @mp��@  ?�
=?���A�p�C�H�@  ?��?�A�=qCh�                                    Bx��ۮ  �          @k��:=q?��?�p�Bz�C c��:=q?�ff?�{A�=qC                                      Bx���T  �          @dz��&ff?�  @p�Bp�C
=�&ff?���?�A��C�)                                    Bx����  �          @aG��\)?��@ ��BG�CW
�\)?�?���A�C�                                    Bx���  �          @[��!G�?��H?ǮA�
=Cٚ�!G�@�?��
A�G�C�                                    Bx��F  �          @����4z�?��@�RB�C��4z�@?�33A�C@                                     Bx��$�  �          @~�R�0��?�@ffA��C5��0��@�H?�p�A��C
�R                                    Bx��3�  T          @y���(Q�@?��HA�ffC�\�(Q�@#33?��A��C�f                                    Bx��B8  �          @~�R�:�H?��H?�\)A߅C��:�H@��?�  A�ffC�\                                    Bx��P�  �          @w��<(�?ٙ�?�A�\)C�R�<(�@��?�ffA���C�                                    Bx��_�  �          @qG��4z�?��
?�A���CxR�4z�@   ?�A��\C�H                                    Bx��n*  �          @qG��7
=?���?���A��C���7
=@(�?��A�Q�C��                                    Bx��|�  �          @aG��\)@��?��A�  C\)�\)@�?333A6�HC�                                    Bx���v  �          @u�$z�@�?��A���C	O\�$z�@,��?Tz�AG�C�{                                    Bx���  �          @{��(��@�R?��
A�(�C�\�(��@(Q�?���A���C�                                    Bx����  �          @\)�E�?�33?�33A�{C�E�@
=?�\)A��C��                                    Bx���h  �          @�\)�R�\?�Q�?�G�Ař�C���R�\@?�z�A�(�C��                                    Bx���  �          @�p��H��?��?�Q�A�33C��H��@ff?���A�  C.                                    Bx��Դ  �          @���S�
?�?˅A�(�C���S�
@��?��
Ah  CaH                                    Bx���Z  
�          @�{�Z=q?�p�?�G�A�{C�{�Z=q@  ?.{A=qC��                                    Bx���   �          @�ff�r�\?У�>�33@�G�C�r�\?�z��G����
CW
                                    Bx�� �  �          @�z��tz�?�ff>���@���Cٚ�tz�?�=q��G���  Cu�                                    Bx��L  �          @��x��?�{?�RAz�C �R�x��?�p�>k�@FffC33                                    Bx���  �          @�ff�tz�?��?��
Ab�RC�R�tz�?�\)?(�A{C
=                                    Bx��,�  �          @�{�u?�
=?�33A�  C"�)�u?���?J=qA.{CJ=                                    Bx��;>  �          @��R�q�?\?�ffAh(�C��q�?�  ?(�AQ�C33                                    Bx��I�  �          @~�R�c�
?�33?���A��C"��c�
?�z�?E�A6�HCc�                                    Bx��X�  �          @i���Vff?z�H?���A�ffC#���Vff?�p�?E�AB=qCٚ                                    Bx��g0  �          @n{�^�R?5?���A�p�C(xR�^�R?z�H?^�RAW�
C$J=                                    Bx��u�  �          @�=q�tz�>B�\?��A��RC15��tz�?
=q?��RA��HC+��                                    Bx���|  �          @�G��mp�=u?���A��C3��mp�>�?�  A��RC,�
                                    Bx���"  �          @|���j=q<#�
?�(�A��\C3�\�j=q>���?�z�A��C-�3                                    Bx����  �          @x���W����?��HA��CI  �W��\(�?�  A�CBG�                                    Bx���n  �          @p���Z=q�+�?�=qA��
C?!H�Z=q��z�?��HA�33C8�H                                    Bx���  �          @u��hQ�=�Q�?�(�A��HC2���hQ�>��?�33A�Q�C-�=                                    Bx��ͺ  �          @u��c33>u?�A��
C0&f�c33?(�?�ffA�33C*L�                                    Bx���`  �          @�Q��mp�=���?�\)A��C2c��mp�>�?��A�=qC,�H                                    Bx���  �          @��\�h��?��?�p�A�ffC� �h��@�?s33AL  C�                                     Bx����  �          @�=q�u�?�?�
=A�(�C�H�u�?�  ?�  AV{Cs3                                    Bx��R  �          @��
�qG�?�
=?�{A�C��qG�?�p�?^�RA8  C@                                     Bx���  �          @��
���H?�Q�?p��AFffC#ٚ���H?��?
=@�G�C!=q                                    Bx��%�  �          @��
���?�G�?�G�AV�RC"����?�p�?&ffAQ�C�3                                    Bx��4D  �          @��H����?}p�?�Q�A�  C&E����?��\?c�
A>=qC"��                                    Bx��B�  �          @�����?�\)?�\)Ak
=C$�R���?���?G�A$  C!z�                                    Bx��Q�  �          @�(����H?�?�G�AU��C${���H?��?+�A��C!:�                                    Bx��`6  T          @���r�\?˅?�  A�ffC:��r�\?�
=?��
AZffC
=                                    Bx��n�  �          @��H�tz�?��?�G�A��C ���tz�?�
=?�{An�\C8R                                    Bx��}�  �          @�33�w�?��
?�G�A�z�C!� �w�?У�?���Ao�
C:�                                    Bx���(  �          @�(��tz�?˅?�A���CaH�tz�?�z�?uAJffCz�                                    Bx����  �          @��
�~{?��?�
=A|(�C!h��~{?���?L��A(z�C
                                    Bx���t  �          @��H�|��?�
=?��Aa��C (��|��?�33?&ffA
{CT{                                    Bx���  �          @�33�{�?�G�?���Ad��C���{�?޸R?&ffA��C&f                                    Bx����  �          @��\�z=q?��H?�Az{C���z=q?��H?@  A   Cff                                    Bx���f  �          @�G��z=q?���?�{A�  C$�f�z=q?��?��
A`(�C z�                                    Bx���  �          @�Q��u�?��R?�{A���C"�u�?�ff?}p�AW�C�R                                    Bx���  �          @����xQ�?�=q?n{AG�C�=�xQ�?�\>��H@��C}q                                    Bx��X  �          @��R�mp�?�=q?�=qAo
=C�R�mp�?�ff?#�
AQ�C!H                                    Bx���  �          @���q�?^�R?�\)A�33C'��q�?���?\(�AE��C#^�                                    Bx���  �          @}p��n{>�
=?���A~{C-�{�n{?.{?k�A[�C)�3                                    Bx��-J  �          @�p��g�?��?Y��A;�
C}q�g�@�\>�{@�{C�)                                    Bx��;�  �          @��R�U@{?��HA���CO\�U@p�?(�A��C�)                                    Bx��J�  �          @�
=�Dz�@��?�G�A��Cn�Dz�@0  ?W
=A7�C
&f                                    Bx��Y<  �          @���G�@�
?�Q�A��RC��G�@&ff?O\)A4Q�C=q                                    Bx��g�  �          @���`  ?�?aG�AHQ�CO\�`  @   >Ǯ@�
=CE                                    Bx��v�  �          @���L��@   ?���A�  C��L��@33?fffAMG�C:�                                    Bx���.  �          @����@��@�?�\)A�C���@��@#33?@  A-G�C��                                    Bx����  �          @�=q�<(�@(�?��A�\)CT{�<(�@,(�?(��AC	�\                                    Bx���z  �          @�z��H��@(�?���A~ffC(��H��@(��>��H@�Q�C��                                    Bx���   �          @�p��@��@   ?�33A�\)CG��@��@1G�?:�HA!�C	^�                                    Bx����  �          @��H�?\)@\)?��A�=qC:��?\)@.�R?!G�AQ�C	��                                    Bx���l  T          @�(��?\)@%?�A�C��?\)@333>��H@��C��                                    Bx���  �          @��\�,��@5?�\)A�=qC���,��@A�>Ǯ@�{C�3                                    Bx���  �          @���'
=@@��?�Q�A�33C���'
=@Mp�>��@�z�C{                                    Bx���^  �          @���1G�@0  ?���A���C(��1G�@@  ?!G�Az�C�                                    Bx��	  �          @���;�@(�?ǮA�=qC#��;�@0  ?h��AL��C                                    Bx���  �          @���G
=?��H?��A�\)C���G
=@?�  A�ffC�                                    Bx��&P  �          @��\�Y��?��R?��Ay��Cn�Y��?�Q�?#�
A�C�=                                    Bx��4�  �          @��
�*�H@:=q?�A�  C���*�H@Fff>�
=@���C�q                                    Bx��C�  �          @����   @aG�?B�\A (�B����   @fff���˅B�p�                                    Bx��RB  �          @�(��(�@n{?333A  B����(�@q녾k��<��B�                                    Bx��`�  T          @�zῙ��@}p�>�{@�Bը�����@z�H�!G��z�B���                                    Bx��o�  T          @�
=>\)@�33�n{�AG�B�W
>\)@�  ���H����B�                                      Bx��~4  �          @��?Tz�@w
=��
=��B�?Tz�@U�#�
�p�B�#�                                    Bx����  �          @�?�Q�@��׿������B�#�?�Q�@e�\)���B�ff                                    Bx����  T          @�  ?��R@~�R������\)B�
=?��R@b�\��\��
=B�W
                                    Bx���&  �          @�
=?�
=@��������yB�
=?�
=@h��������B�{                                    Bx����  �          @��
?�{@��ÿ}p��RffB��q?�{@j�H��
=��B�k�                                    Bx���r  �          @�
=?��@��Ϳfff�9�B�  ?��@s�
��{��p�B�                                      Bx���  �          @���?���@��\��z�����B��R?���@hQ�����\B�p�                                    Bx���  �          @���?��@�zῙ���v{B��
?��@o\)�	������B�#�                                    Bx���d  �          @�33?��@�z�J=q���B�=q?��@��\��ff���B�8R                                    Bx��
  �          @�p�?J=q@����0����RB�(�?J=q@�Q��
=����B��R                                    Bx���  �          @�\)?Tz�@��H�0����RB�Q�?Tz�@�녿ٙ���z�B��
                                    Bx��V  �          @���?L��@�z�0�����B�B�?L��@�33���H���B��)                                    Bx��-�  
�          @���?c�
@�=q�z�H�H(�B���?c�
@~{��(���Q�B���                                    Bx��<�  �          @�=q?�  @��H���
�Q�B�#�?�  @~�R�G���  B��H                                    Bx��KH  �          @��?O\)@�������W\)B�B�?O\)@�G��z����
B�ff                                    Bx��Y�  T          @��?h��@�=q��ff�UG�B�Q�?h��@}p������B�8R                                    Bx��h�  �          @��\?��@�
=����ffB���?��@qG��
=��Q�B�z�                                    Bx��w:  �          @���?u@�  �����  B���?u@u������Q�B�                                      Bx����  �          @���?J=q@��ÿ�
=�rffB�\?J=q@x�������(�B�                                    Bx����  �          @��H?h��@����  ����B��
?h��@q��(�� z�B�                                      Bx���,  �          @��\?(��@��\���H����B��\?(��@aG��5�B��3                                    Bx����  �          @��?�\)@��ÿ޸R����B�{?�\)@a��'��(�B��f                                    Bx���x  �          @�Q�?z�H@x�ÿ�p����B��)?z�H@U�4z��  B�k�                                    Bx���  �          @�p�?��@y���˅��G�B�ff?��@\(��(��ffB��
                                    Bx����  �          @���?#�
@`�׿�{��Q�B�u�?#�
@@  �'
=�!�RB�                                    Bx���j  T          @�zᾔz�@N�R�!���HB�{��z�@$z��L���L
=B���                                    Bx���  �          @�����
@J=q�'
=�  B�
=���
@\)�P  �R=qB��H                                    Bx��	�  �          @��?��@c33��\)��  B�Q�?��@B�\�(Q��!\)B�ff                                    Bx��\  �          @���?Y��@Y����Q���B��?Y��@7��)���&
=B��                                    Bx��'  �          @��?�ff@E��Q���\B�G�?�ff@{�@���A  B��
                                    Bx��5�  �          @�=q?:�H@H���p��\)B���?:�H@ ���Fff�G\)B��{                                    Bx��DN  �          @�{?s33@C�
�,(�� (�B���?s33@���S�
�PB��                                    Bx��R�  �          @�p�?!G�@2�\�AG��:p�B�p�?!G�@33�dz��l{B���                                    Bx��a�  �          @��\?!G�@%�AG��B�B���?!G�?����aG��sB��                                    Bx��p@  �          @�Q�?�z�@C33�$z����B�33?�z�@���K��C{Bn�                                    Bx��~�  �          @�z�?��@>�R�(���(�B�Ǯ?��@z��N�R�N33B��R                                    Bx����  �          @�  ?k�@S�
��R�(�B��
?k�@,(��I���?��B�\                                    Bx���2  �          @��R?�@S33�!��=qB�B�?�@*�H�L(��E�HB��3                                    Bx����  T          @�=L��@XQ���
�	=qB�z�=L��@333�?\)�;��B�(�                                    Bx���~  �          @���?Y��@S�
��
�(�B�.?Y��@.�R�>�R�8�\B�\)                                    Bx���$  �          @�?��\@&ff�=p��6�B�?��\?���\���b
=B`�                                    Bx����  �          @�p�@ ��?�z��L(��K  B
=@ ��?n{�^{�e�HA�ff                                    Bx���p  �          @�p�>L��@R�\��
� �B�
=>L��@1G��.{�1B��q                                    Bx���  �          @��Ϳ+�@qG��������\B�{�+�@X���ff��(�B�=q                                    Bx���  �          @�  �z�@]p�?�33A33B���z�@hQ�>\@�Q�B�L�                                    Bx��b  �          @��ÿ���@s33?�@�33B��ÿ���@tzᾙ���\)B��                                    Bx��   �          @�녿�33@y��?�33Ax  B�aH��33@���>�z�@y��B���                                    Bx��.�  �          @��ÿ�33@y��?���A�B��H��33@�33>�@�ffB�k�                                    Bx��=T  �          @�Q쿜(�@z=q?�
=A��B֊=��(�@�=q>���@��RB�=q                                    Bx��K�  �          @�\)��Q�@�33?��A�G�B�p���Q�@�G�>�ff@��RB�\                                    Bx��Z�  �          @�����
@\)?�\)A�G�Bה{���
@�{?�\@ҏ\B���                                    Bx��iF  �          @��H�޸R@r�\?��
A^ffB�Q�޸R@z�H>aG�@8Q�B��
                                    Bx��w�  �          @��  @Y��?^�RA@Q�B���  @`��>\)?�B�z�                                    Bx����  T          @��Ǯ@g�?���A�  B��Ǯ@q�>�33@�B�Ǯ                                    Bx���8  �          @����G�@j=q?z�HAS
=B��H�G�@q�>L��@+�B�G�                                    Bx����  �          @����{@P��?�G�A��B�G��{@\��?\)@���B�.                                    Bx����  �          @����&ff@C�
?�Q�A�{CW
�&ff@R�\?G�A(��C J=                                    Bx���*  �          @��H�=q@*=q@
=B33C5��=q@E�?�G�A��
C                                       Bx����  �          @�����@%�@333B  C� ���@Fff@p�A�p�B��)                                    Bx���v  �          @��
�G�@@��@�B�B����G�@\(�?�\A�G�B��H                                    Bx���  �          @���   @HQ�@��A��RB�(��   @aG�?�=qA�ffB�(�                                    Bx����  T          @�p���@I��@
�HA�ffB����@aG�?��RA�=qB�                                    Bx��
h  T          @���0  @7
=?��AʸRC�H�0  @K�?��\A��
Cٚ                                    Bx��  �          @����(Q�@U�?ǮA���C G��(Q�@e�?\(�A/�B���                                    Bx��'�  �          @�
=�)��@X��?\(�A4z�B����)��@_\)>#�
@
=B�Q�                                    Bx��6Z  �          @����:=q@Q�?�z�A�C�q�:=q@.{?�33A��C	                                    Bx��E   �          @��L��?�G�@	��A�{C�\�L��?�33?�A���CE                                    Bx��S�  �          @��B�\?�@�HB(�C���B�\?��@A�RCY�                                    Bx��bL  �          @����;�?�ff@�RB=qC#��;�@ ��@Q�A�33C�{                                    Bx��p�  �          @�z��4z�?�
=@*=qB {C0��4z�?�@z�B��C�                                    Bx���  �          @��R�G
=?�{@�
A�
=C��G
=@�R?�z�A�G�C^�                                    Bx���>  �          @��R�/\)?�@"�\B�HC�=�/\)@G�@��A��CW
                                    Bx����  �          @����
?˅@J�HBAC����
@
�H@333B%�C�=                                    Bx����  �          @�p��z�?��
@>{B5�HCs3�z�@z�@$z�BG�C�                                    Bx���0  
�          @��ÿ\?�R@e�B���C�H�\?�ff@X��Bn
=CQ�                                    Bx����  �          @�  ��=q?�G�@\(�Bu�CaH��=q?�33@K�BY=qC��                                    Bx���|  �          @�Q��Q�?�
=@I��BQ33C��Q�?�G�@7
=B8=qC�                                     Bx���"  �          @�Q��  ?�@.�RB,��Cٚ�  @z�@�B(�C#�                                    Bx����  �          @{����@	��@=qBz�C�����@$z�?��HA���C�=                                    Bx��n  �          @�\)��@
=q@<��B2(�C����@+�@\)B�B��R                                    Bx��  �          @������@�
@*=qB�C�����@1G�@�A�z�CxR                                    Bx�� �  �          @�z��#33@"�\@��B	  C5��#33@<��?�
=Aң�C�H                                    Bx��/`  �          @��(��@%@(�Bz�C���(��@@  ?�z�A�ffCk�                                    Bx��>  �          @�{�-p�@#�
@��BQ�C���-p�@=p�?��A�\)Cz�                                    Bx��L�  �          @�ff���@
=q@3�
B'�C\)���@(��@
=B��C�{                                    Bx��[R  �          @�(�����@@=p�B7�C{����@&ff@!G�B33B��
                                    Bx��i�  �          @����
@=q@333B%�C}q��
@8Q�@�
B33B�                                    Bx��x�  �          @�����@-p�@"�\Bp�B����@HQ�@   A�{B�(�                                    Bx���D  y          @���{@�
@<(�B0�
C	(��{@#�
@!G�B�\C�                                    Bx����  �          @p  ����@(�@��BG�C�ÿ���@$z�?���A��
B��\                                    Bx����  �          @xQ��
�H?�=q@\)B$��C�)�
�H@  @
=BffC�R                                    Bx���6  �          @o\)���?�(�@�\B�C
����@ff?��A�C�                                    Bx����  �          @~�R�Q�@�@�\A��C^��Q�@0��?���A��RC�                                    Bx��Ђ  �          @g��
=?�z�@{B��C	�f�
=@G�?�A��C��                                    Bx���(  �          @l�Ϳ�
=?���@(��B@33Ck���
=?�=q@ffB&
=C��                                    Bx����  �          @�����\@.{@A�p�C���\@B�\?���A��B��                                    Bx���t  �          @�\)�ff@1G�@'
=B��Cc��ff@K�@�A��
B���                                    Bx��  �          @��&ff@��@1G�B33C
�R�&ff@-p�@�A�z�C�                                    Bx���  �          @�ff�5@
=@ ��BCE�5@0��@�
Aۙ�C�=                                    Bx��(f  �          @�Q��>{@&ff?ǮA���C
�{�>{@5�?��Am�Cff                                    Bx��7  T          @�z��;�@!�@33A�33C(��;�@5?�=qA��C��                                    Bx��E�  �          @���#33@	��@,��B33C��#33@%�@�\B33C��                                    Bx��TX  �          @����.{@N�R?��Ag�C&f�.{@W
=?�@��HC ��                                    Bx��b�  �          @�{�Dz�@E�?Q�A*�\C�3�Dz�@J�H>�\)@h��C
                                    Bx��q�  �          @�p��8Q�@U�>u@C�
C�)�8Q�@Tzᾨ����33C�                                    Bx���J  �          @����Tz�@=p�?G�A�C
:��Tz�@C33>�=q@Z=qC	aH                                    Bx����  �          @����U�@?\)?O\)A$��C
��U�@E>���@n�RC	�                                    Bx����  �          @����C�
@Tz�?z�@��
C���C�
@W�<��
>uC8R                                    Bx���<  �          @�=q�=p�@U?p��A>�\C���=p�@]p�>�p�@��C�)                                    Bx����  �          @�(��7
=@c33?O\)A z�C �f�7
=@hQ�>W
=@"�\C =q                                    Bx��Ɉ  �          @����7�@X��?s33AA�C@ �7�@`  >\@�  CY�                                    Bx���.  �          @�G��=p�@XQ�?!G�A (�CB��=p�@[�=�\)?Y��CǮ                                    Bx����  �          @����L(�@K����Ϳ��\C
�L(�@G���R����C�H                                    Bx���z  �          @�G��Tz�@C33�����ffC	k��Tz�@<(��fff�8  C
s3                                    Bx��   �          @��\�C33@XQ쾞�R�|��C
=�C33@Q녿\(��-p�C�)                                    Bx���  �          @����;�@\�;��\CW
�;�@Tzῃ�
�QC^�                                    Bx��!l  �          @���0  @a녾���R�\B��0  @\�ͿQ��(��C �{                                    Bx��0  �          @�
=�#33@j�H�u�5B��=�#33@g
=�#�
�  B�aH                                    Bx��>�  �          @�  ���@n{?��@��HB������@p�׽#�
��B�
=                                    Bx��M^  �          @�  ��\@l(�?Q�A,Q�B����\@qG�>k�@@  B�z�                                    Bx��\  �          @�{��G�@~�R?5A�B�׿�G�@���=��
?��\B���                                    Bx��j�  �          @�  ��
=@\)?5A=qB���
=@��=��
?��\B���                                    Bx��yP  �          @��Ϳ�(�@u?:�HAffB�p���(�@y��>�?�33B鞸                                    Bx����  �          @��R��ff@�  ?:�HA�B�z��ff@��=���?�=qB�Ǯ                                    Bx����  T          @�
=�Ǯ@�z�?z�@��B�\)�Ǯ@���\)�fffB�                                      Bx���B  �          @��R���@�Q�?@  A�
B�녿��@��\=�Q�?�Bπ                                     Bx����  �          @�
=�p��@�Q�?h��A<��B��p��@�33>�  @Mp�B̅                                    Bx��  �          @�Q쿔z�@�Q�?fffA8��Bҏ\��z�@��H>�  @G�B���                                    Bx���4  �          @�{����@z=q?:�HA
=B��f����@~�R>�?�33B�(�                                    Bx����  �          @����z�@z=q?5A��B���z�@~�R=�?�G�B�L�                                    Bx���  �          @�
=�&ff@e�=�?��
B���&ff@c33��
=��p�B�G�                                    Bx���&  �          @�{�J�H@A녿!G��\)CB��J�H@9����=q�`��C	}q                                    Bx���  �          @��R�-p�@_\)�\)��p�B����-p�@W������]p�C �\                                    Bx��r  �          @�{�   @e�8Q����B����   @\(����R��  B��H                                    Bx��)  �          @�33��p�@��?h��A7�B�\��p�@�  >���@l��B�8R                                    Bx��7�  �          @��Ϳ�(�@��?�33A��B����(�@�
=?L��A�RB�ff                                    Bx��Fd  �          @����{@�p�?�ffA���B�G���{@��?.{A�HB�                                      Bx��U
  
�          @����\@���?G�A�B�z���\@��
>B�\@z�B�3                                    Bx��c�  �          @��H�z�@w�?L��A!��B���z�@|(�>u@A�B���                                    Bx��rV  �          @�p���R@xQ�?5AQ�B�33��R@|(�>��?�{B�ff                                    Bx����  �          @�33�p�@�Q�>�{@��B��p�@��׾k��3�
B���                                    Bx����  �          @�33���R@��?}p�AH(�B�{���R@���>\@��B�ff                                    Bx���H  �          @����@�=q�u�=p�B����@��׿!G���p�B��                                    Bx����  �          @���	��@}p�>\@��B���	��@~{�.{�
=qB��                                    Bx����  �          @�p��33@�p��\��z�B�B��33@��\�s33�<(�B�G�                                    Bx���:  �          @�=q���@�(����R�z=qB�����@�G��^�R�1G�B���                                    Bx����  �          @�  ��33@��׿�\�ϮB����33@z=q��ff�Y�B��)                                    Bx���  �          @�
=��Q�@�Q쿨�����B�uÿ�Q�@r�\�����ď\B݅                                    Bx���,  T          @�z῰��@}p����\���HB�ff����@p  �����G�B�\)                                    Bx���  T          @��H�p��@s33������Bϳ3�p��@aG����Q�B���                                    Bx��x  �          @�G�����@xQ�p���O
=B�LͿ���@n{������G�B���                                    Bx��"  �          @�33��@i��=�\)?^�RB�  ��@hQ��
=����B�Q�                                    Bx��0�  �          @�����@mp�>L��@*�HB�{��@mp���z��{�B��                                    Bx��?j  �          @�
=>�p�@���L���/
=B��R>�p�@~{��=q��\)B�\)                                    Bx��N  �          @�  ?�@�����
=B��H?�@�=q�����j=qB��                                    Bx��\�  �          @�Q�?�R@��Ϳ333��\B�#�?�R@��ÿ�p���  B���                                    Bx��k\  �          @�\)?s33@�  �k��H��B�W
?s33@vff�����B�W
                                    Bx��z  �          @���?Tz�@�(������B�=q?Tz�@��׿�\)�t  B���                                    Bx����  �          @�G�?\(�@�z�(����RB�� ?\(�@��׿�
=��ffB���                                    Bx���N  �          @��?+�@��H�Tz��4Q�B�L�?+�@|�Ϳ�=q��G�B��                                    Bx����  �          @���?��@�G��h���D��B��=?��@x�ÿ�z�����B�k�                                    Bx����  �          @��
?�=q@�=q��33�s�B�\)?�=q@y�������
=B��                                    Bx���@  
�          @��H?��@���
=����B��?��@�녿����j�HB�=q                                    Bx����  �          @�G�>�@�\)�\)��{B�#�>�@��(���B���                                    Bx����  �          @�Q���@�ff�aG��<(�B��f���@�z�:�H�ffB�{                                    Bx���2  �          @��R�
=q@��H�:�H� ��B�(��
=q@~{��(����BÙ�                                    Bx����  �          @�\)�\)@���+��G�B���\)@�����z���B��                                    Bx��~  �          @��þ#�
@����=q�h��B�33�#�
@��E��'�
B�B�                                    Bx��$  �          @�
=����@�G��z�H�Y�B�G�����@y����������B��3                                    Bx��)�  �          @��ÿz�@��\�����tQ�B�B��z�@z�H������B��                                    Bx��8p  �          @�  ���@z=q��G�����B�W
���@mp����H��{B�G�                                    Bx��G  �          @�Q��@w������=qBÅ��@j�H��p����B�aH                                    Bx��U�  �          @�Q쾏\)@tz����{B�L;�\)@e�\)���HB��
                                    Bx��db  �          @����@\���
=�p�B�#׿�@J=q�/\)�"p�BƳ3                                    Bx��s  �          @�Q�p��@g�?�A�ffB�
=�p��@s33?���A�Bϳ3                                    Bx����  �          @�z῝p�@vff?�z�A���Bי���p�@�Q�?�(�A�p�B�=q                                    Bx���T  T          @����   @l��?:�HAz�B��)�   @p��>��R@�33B��                                    Bx����  �          @�=q��p�@z=q?c�
AA�B�\)��p�@\)>�ff@\Bܣ�                                    Bx����  �          @�=q���@}p�?8Q�A�B�z���@���>�=q@h��B��                                    Bx���F  �          @�=q��  @��׼���G�B��
��  @\)�����33B��                                    Bx����  �          @��
�Ǯ@�Q�@  ��HB�aH�Ǯ@z=q��Q��|  B�k�                                    Bx��ْ  �          @��H��G�@�(���z��u�B��)��G�@��\�:�H�Q�B�L�                                    Bx���8  �          @����u@�p�=L��?&ffB���u@��;�����33B��                                    Bx����  �          @�33�Y��@��&ff�	B�\�Y��@��H����i��Bˏ\                                    Bx���  �          @�=q�5@��ÿ����=qB���5@w����H��  B��                                    Bx��*  �          @�{��{@x�ÿ���g
=B��H��{@p�׿�Q���B��                                    Bx��"�  �          @�{��ff@y���@  �$��B��
��ff@s33��33�33Bٽq                                    Bx��1v  �          @��Ϳ�{@a녿��
�g33B�3��{@Y�������\)B�k�                                    Bx��@  �          @�p�� ��@B�\��33��{B�
=� ��@5�����p�B��H                                    Bx��N�  �          @����@'
=����
33C)��@
=�(���ffC�                                    Bx��]h  �          @�p��z�?�ff�H���V  C(��z�?:�H�N�R�_{C ��                                    Bx��l  �          @��R�B�\@��\?�RA z�B���B�\@��
>8Q�@�B���                                    Bx��z�  �          @��Ϳ+�@�\)?��
AM�B���+�@��?z�@�\)B���                                    Bx���Z  �          @��Ϳ��@��?�p�A�ffB�Ǯ���@�\)?��AO\)B�W
                                    Bx���   �          @�(��L��@�(�?�Q�Ap  BȨ��L��@�\)?B�\A�
B�8R                                    Bx����  �          @�{�O\)@��>aG�@3�
B���O\)@���W
=�)��B��                                    Bx���L  �          @�{�
=q@���������B�#׿
=q@�G��xQ��I�B�aH                                    Bx����  �          @�{�B�\@��;u�<��B��R�B�\@���.{��\B�Ǯ                                    Bx��Ҙ  �          @�\)�u@�
=�L�Ϳ#�
B��\�u@�ff�   ���B��{                                    Bx���>  T          @�
==���@�(�?B�\Az�B���=���@�>��
@z=qB���                                    Bx����  �          @�  �#�
@�?:�HAG�B�Lͼ#�
@�
=>�z�@`��B�L�                                    Bx����  �          @��R��\)@��R?�p�A���B��ͽ�\)@��\?��AQ�B��q                                    Bx��0  �          @��R��@���?�=qAU�B����@�(�?(��AB�Ǯ                                    Bx���  �          @��R�:�H@�33?&ffA   B�#׿:�H@�z�>u@7�B���                                    Bx��*|  �          @�\)��p�@�{>���@fffB�
=��p�@�ff�����
B�                                    Bx��9"  �          @�{�
=q@��
>W
=@'
=B�G��
=q@��
�L�����B�B�                                    Bx��G�  �          @��R�xQ�@�=q��=q�S�
B�{�xQ�@��ÿ+��  B�L�                                    Bx��Vn  �          @��R����@�����
�n{B�
=����@�
=������RB�8R                                    Bx��e  �          @�\)���H@����Q쿎{B�����H@�
=����=qB�L�                                    Bx��s�  �          @��׿�
=@�G��#�
��
=B��ῷ
=@�Q������
B�33                                    Bx���`  �          @����R@�
=���H��=qB��῞�R@�p��^�R�,(�B�\)                                    Bx���  �          @����ff@��׾����k�B�\��ff@�\)�.{�{B�L�                                    Bx����  T          @�{���@��������BΞ����@�{�W
=�'\)B��                                    Bx���R  �          @�G���p�@��׾�Q����HB�LͿ�p�@�
=�:�H�p�Bأ�                                    Bx����  �          @�z῾�R@z�H��R�{Bݏ\���R@vff�p���L(�B�33                                    Bx��˞  �          @�(�����@�p���z���  B�����@�Q��\)����B�z�                                    Bx���D  �          @��\)@�{�   ��z�B�33�\)@�Q������HB��
                                    Bx����  �          @��c�
@��R�������
B��c�
@����	����p�B̨�                                    Bx����  �          @�  �}p�@�G����
���B�\�}p�@�z��
=����B���                                    Bx��6  �          @��׿}p�@�33��33����Bͨ��}p�@�ff��p���p�B�z�                                    Bx���  �          @�  ��p�@�Q��ff���B♚��p�@w
=�ff�ӅB�33                                    Bx��#�  �          @�(�����@z�H�5�  B�zᾨ��@l���HQ�� p�B�{                                    Bx��2(  �          @�=q��  @S33�_\)�9��B�����  @A��n�R�K
=B�aH                                    Bx��@�  �          @�G��@  @=p��k��I
=B�z�@  @+��x���Y��B�\)                                    Bx��Ot  �          @�=q�5@>{�W��?p�B��)�5@-p��e��O�B�L�                                    Bx��^  �          @�G��E�@0���`  �J��B�LͿE�@   �l(��ZB�k�                                    Bx��l�  �          @�Q��@�H�p���a��BΏ\��@���z�H�r{B��)                                    Bx��{f  �          @��ÿ
=@\)�mp��]�B��f�
=@p��xQ��m��B�\                                    Bx���  �          @�녿0��@
=�|(��rz�B�\�0��?�=q���\�B�=q                                    Bx����  �          @��þ��?�
=�����33B�p����?������aHB�                                    Bx���X  �          @��.{@
�H�p���k\)B��Ϳ.{?�33�y���zp�B�Q�                                    Bx����  �          @�(��z�H@��o\)�kp�B�ff�z�H?�\�w��yp�B���                                    Bx��Ĥ  �          @�녿��?�=q�tz��{B��{���?����z�H��C�
                                    Bx���J  �          @�G����
@ ���c33�`p�B��쿣�
?�\�j�H�m33B���                                    Bx����  T          @����33@Q��QG��J  B��R��33?�33�Y���UC�                                    Bx���  �          @�����
@��L���EQ�B��)���
@z��U�Q��B�33                                    Bx���<  T          @�\)�(��@6ff�A��8
=B��(��@)���Mp��F�B��)                                    Bx���  �          @�=q�O\)@`���z��{B��ÿO\)@Vff�"�\�\)B��                                    Bx���  �          @�{�&ff@��
�������B��)�&ff@��ÿ�����p�B�B�                                    Bx��+.  �          @���=p�@U��/\)���B���=p�@I���<(��(�B΀                                     Bx��9�  �          @���u@>{�AG��0Q�B��u@1G��L���=B�B�                                    Bx��Hz  �          @�������@N�R�ff���B�����@E��"�\�(�B��                                    Bx��W   �          @�z��ff@I���p��ffB�aH��ff@@  �(���Q�B��H                                    Bx��e�  T          @�����@]p��.{�p�B����@Z=q�aG��HQ�B�                                    Bx��tl            @�(����H@P  �Ǯ����B�#׿��H@I����  ���B�                                    Bx���  T          @��Ϳ��H@r�\?�33Aw33B�=���H@vff?n{AF�\B��)                                    Bx����  �          @����(�@ ���2�\�,��C	}q�(�?����9���5G�C��                                    Bx���^  �          @�33��ff@���>�(�@��B�.��ff@�G�>L��@-p�B�                                    Bx���  �          @��H��
=@}p������
B���
=@|�;�33��
=B�.                                    Bx����  �          @�ff�n{@r�\@��A�Bϙ��n{@y��?�
=A��B��
                                    Bx���P  �          @��
�:�H@h��@�A�
=Bʳ3�:�H@p��@�A�\B�                                      Bx����  �          @���W
=@g�@\)A��\B�
=�W
=@o\)@�\A��\B�B�                                    Bx���  �          @��ͿxQ�@l��@��A�\)B�\)�xQ�@s�
?�Q�A��BЏ\                                    Bx���B  �          @��ͿQ�@e�@�HB�\B��Q�@l��@�RA�B��                                    Bx���  �          @�(��@  @^�R@"�\B{B�Q�@  @fff@
=B�\Bˀ                                     Bx���  �          @�{�L��@�\)?��A�33B�� �L��@�G�?��Ad  B�k�                                    Bx��$4  �          @�
=    @��?#�
A��B���    @�{>�
=@�p�B���                                    Bx��2�  �          @��R�k�@���?!G�A{B�  �k�@�>��@��B���                                    Bx��A�  �          @�
=�u@�>�@�p�B�=q�u@�ff>u@H��B�8R                                    Bx��P&  �          @�ff��Q�@�(�?(��A
{B�Lͽ�Q�@���>�ff@�z�B�G�                                    Bx��^�  �          @�ff���R@�>W
=@)��B��þ��R@�    �L��B���                                    Bx��mr  
�          @��R��=q@�p�>�p�@���B��ᾊ=q@�>.{@��B���                                    Bx��|  �          @�\)�B�\@�p�?��@�G�B�\�B�\@�{>���@�B�                                    Bx����  �          @���=�Q�@�\)?��@���B��{=�Q�@�  >�33@��RB���                                    Bx���d  �          @���=�\)@�\)>��@�  B�Q�=�\)@��>�=q@]p�B�Q�                                    Bx���
  �          @�Q�>�(�@�ff���
���B���>�(�@�ff�W
=�/\)B���                                    Bx����  �          @�\)?\(�@�33>�\)@c33B���?\(�@�33=�Q�?�33B��                                    Bx���V  �          @���?�p�@�Q�>��R@\)B���?�p�@���>�?��HB��f                                    Bx����  �          @�  ?�Q�@�G�>B�\@�RB��
?�Q�@���<�>�{B��H                                    Bx���  �          @�
=?=p�@�G��u�F�\B�ff?=p�@�  �����k\)B�33                                    Bx���H  �          @�\)?�\)@�G����H��33B��
?�\)@��׿+��	�B��3                                    Bx����  �          @���?��H@�  <#�
=��
B��?��H@�  �#�
�33B�{                                    Bx���  �          @���?��
@�����\B�L�?��
@���B�\�{B�B�                                    Bx��:  �          @���?\@�  =�\)?n{B��?\@�  ��Q쿏\)B��                                    Bx��+�  T          @��?��
@�>Ǯ@�(�B���?��
@�{>k�@:=qB��                                    Bx��:�  �          @��H@ff@�녾B�\��BzG�@ff@�����{��\)Bz{                                    Bx��I,  �          @��\?��@��;�\)�]p�B��?��@�z��
=���\B���                                    Bx��W�  �          @��\@Q�@����u�G�Bx�R@Q�@�G��W
=�$z�Bx��                                    Bx��fx  �          @�33@��@���?�@�{Bw��@��@���>Ǯ@�{Bx\)                                    Bx��u  �          @�=q?��
@��H?^�RA0  B��?��
@��
?:�HA�B�33                                    Bx����  �          @�G�?�=q@���=p���\B�ff?�=q@�z�aG��3
=B�#�                                    Bx���j  �          @�G�@33@tzᾨ����33Bv�
@33@s33��ff���Bv�                                    Bx���  �          @��\@�@~{�����33Bw@�@|�Ϳ:�H�\)BwG�                                    Bx����  �          @��\@�R@|�;�Q���Q�Br{@�R@|(�����=qBq�R                                    Bx���\  �          @���@��@z=q>8Q�@G�Bo��@��@z=q=u?B�\Bo�                                    Bx���  �          @�G�?�p�@�=��
?��\B��?�p�@��#�
�\)B���                                    Bx��ۨ  �          @��?�(�@��R�aG��1G�B��\?�(�@�ff��{��=qB�z�                                    Bx���N  
�          @�=q?�\)@�p��=p��p�B���?�\)@��Ϳ\(��-�B�p�                                    Bx����  �          @��?���@��\�0����
B��R?���@�녿O\)�$z�B��{                                    Bx���  �          @��H?�=q@�p�������
B��?�=q@���z��陚B��=                                    Bx��@  �          @��H@
�H@~�R�333��
Bu�R@
�H@}p��L���!�Bu=q                                    Bx��$�  �          @��?��H@��\�J=q�\)B���?��H@�녿fff�4��B��{                                    Bx��3�  �          @��\?�\)@��Ϳ�ff�T  B�aH?�\)@��
��33�ip�B��                                    Bx��B2  �          @��\?�G�@��
�\(��-�B���?�G�@�33�u�B�RB��{                                    Bx��P�  �          @���?��@�녿@  ��B�Q�?��@�G��Y���.ffB�33                                    Bx��_~  �          @���?��@�  �c�
�5�B�B�?��@�
=�}p��I��B��                                    Bx��n$  �          @���?���@����33�k�B��q?���@�z῞�R��B��=                                    Bx��|�  �          @���?�p�@����\��ffB�(�?�p�@��׿�{��{B��H                                    Bx���p  �          @���?O\)@�p���=q�\��B�Q�?O\)@����p�����B�L�                                    Bx���  �          @��R?�{@��׿!G��G�B���?�{@�  �5��B��q                                    Bx����  
�          @�{?s33@�  �&ff��B��R?s33@���=p��B���                                    Bx���b  �          @��?�\@�=q�����^{B��{?�\@�����z��o�
B��                                    Bx���  �          @�  >��
@�ff����z�B��)>��
@�{�\)��
=B��
                                    Bx��Ԯ  �          @���=�G�@��׿���\)B�33=�G�@�Q��R����B�33                                    Bx���T  �          @��\>��@�
=�W
=�*=qB���>��@�ff�k��:ffB��                                    Bx����  T          @��\?z�@���  �Ip�B�
=?z�@�������X��B���                                    Bx�� �  �          @�
=��G�@�=q��\)��ffB�aH��G�@�����Q����B�u�                                    Bx��F  �          @��\�!G�@�
=�����  B��f�!G�@�{�ٙ����B�                                    Bx���  �          @��H���@����33����B��H���@�
=��(����
B���                                    Bx��,�  �          @����Ǯ@������G�B��׾Ǯ@�(���\)���B��3                                    Bx��;8  �          @����B�\@y���(���RB�.�B�\@w��\)����B�\)                                    Bx��I�  �          @����aG�@s33��\��G�B�(��aG�@qG����G�B�\)                                    Bx��X�  �          @����=q@c33� ����Bսq��=q@`���#�
�Q�B�
=                                    Bx��g*  �          @���33@G
=�!G���B��f��33@E��#�
�(�B�aH                                    Bx��u�  �          @�G����R@�p��޸R����B�uþ��R@��Ϳ�����B��                                     Bx�v  �          @��þ�\)@�p��ٙ���p�B�����\)@��Ϳ�  ��z�B���                                    Bx�  �          @��R>���@�G���ff��G�B���>���@��׿���{B��\                                    Bx�¡�  �          @�  ?=p�@z�H�z���
=B��?=p�@y���ff��p�B��\                                    Bx�°h  �          @���?W
=@u���R��RB�k�?W
=@s�
������HB�L�                                    Bx�¿  �          @�G�?^�R@w������Q�B�Ǯ?^�R@vff�\)��=qB���                                    Bx��ʹ  �          @��\?O\)@��
��\����B���?O\)@�33��ff��=qB��q                                    Bx���Z  �          @��?O\)@�녿�Q��s\)B���?O\)@�녿�p��z=qB�                                    Bx���   �          @�=q?Q�@�\)��  ��\)B�?Q�@�
=���
��z�B���                                    Bx����  �          @�G�?u@�p����H��z�B���?u@����p���\)B��                                    Bx��L  �          @�p�?W
=@\���'
=�z�B�� ?W
=@[��(Q��B�aH                                    Bx���  �          @�?�(�@l���G���p�B���?�(�@l(���\��B��                                     Bx��%�  �          @��R?n{@�G��������B�\?n{@��ÿ�z����RB�                                      Bx��4>  �          @��?n{@~�R�����RB��{?n{@~�R�������\B��=                                    Bx��B�  T          @�?^�R@vff���H��\)B���?^�R@vff��p�����B���                                    Bx��Q�  �          @�p�?@  @|(�����
=B�L�?@  @{�������ffB�B�                                    Bx��`0  �          @�{?aG�@��׿�33���\B�ff?aG�@��׿�z�����B�\)                                    Bx��n�  �          @��?��\@�  �\��33B�z�?��\@�  �\��{B�u�                                    Bx��}|  �          @�{?���@p�׿˅���B�?���@p�׿˅��z�B���                                    Bx�Ì"  �          @��@�R@O\)�	����p�B]�@�R@O\)�
=q��B]��                                    Bx�Ú�  �          @�
=@z�@X���33��  Bj\)@z�@X���33��(�Bj\)                                    Bx�én  �          @�?�z�@l�Ϳ�Q����B�Q�?�z�@l�Ϳ�Q���
=B�Q�                                    Bx�ø  �          @�(�?\(�@�(������o
=B��{?\(�@�z῏\)�n=qB��{                                    Bx��ƺ  �          @�z�?z�H@�녿�ff��=qB��q?z�H@�녿������B�                                    Bx���`  �          @�(�>�@��
��{���HB��q>�@��
������  B�                                    Bx���  �          @�(�?�ff@�{��R�33B���?�ff@�{�(���B�                                      Bx���  �          @�p�?(�@�
=��{�h��B��\?(�@�\)�����f=qB��{                                    Bx��R  �          @�(�>��@�(������33B��3>��@�zῪ=q���B��R                                    Bx���  �          @��=#�
@��\���R����B���=#�
@��\��p���33B���                                    Bx���  �          @���Ǯ@�=q��G���Q�B�
=�Ǯ@�=q���R��Q�B�                                    Bx��-D  �          @�z�L��@\)�У���
=Bʔ{�L��@�  ��{����Bʊ=                                    Bx��;�  �          @��׾��@\)�����{B������@�  ��G�����B���                                    Bx��J�  �          @�G����@�(����\�\Q�B�����@�(��}p��V�HB��                                    Bx��Y6  �          @��
�\)@����
=���B��)�\)@����\)��{B��)                                    Bx��g�  �          @�=q�u@���!G���\B��\�u@������ Q�B��\                                    Bx��v�  �          @�ff��=q@�p��aG��4z�B��H��=q@��B�\���B��H                                    Bx�ą(  �          @�  ��@�\)=�Q�?���B�8R��@�\)=�?�=qB�8R                                    Bx�ē�  �          @��=L��@���=L��?+�B��3=L��@�G�=�Q�?�z�B��3                                    Bx�Ģt  �          @��\>\)@��>u@A�B�� >\)@��>�\)@b�\B��                                     Bx�ı  �          @�G�?\(�@���@  ���B�Ǯ?\(�@��
�5�(�B���                                    Bx�Ŀ�  �          @�  ?�z�@Tz������Bpff?�z�@Vff�\)��Bq
=                                    Bx���f  �          @�Q�?�\)@c33�����\)B��
?�\)@dz�����B�{                                    Bx���  �          @�\)?��
@��׿�  ����B��R?��
@�G����H��=qB��
                                    Bx���  �          @�\)?�ff@��׿��R��Q�B��H?�ff@��ÿ�Q��w
=B�                                    Bx���X  �          @��\?�@��
�O\)�"�\B���?�@�(��B�\���B��q                                    Bx���  �          @���?�@�(���ff���RB��R?�@�(�������=qB�Ǯ                                    Bx���  �          @���?ٙ�@�녿����[�B���?ٙ�@�=q���\�P��B�                                    Bx��&J  �          @��\@�
@u��33���HBwQ�@�
@w
=�������Bw��                                    Bx��4�  �          @�{�Y��@U�?W
=A[
=B���Y��@S�
?aG�Ag
=B��                                    Bx��C�  �          @�z��{@2�\?�z�A��C���{@1G�?ٙ�AĸRC�                                    Bx��R<  �          @��
�"�\@@  ?��HA��CB��"�\@>�R?�G�A�=qCn                                    Bx��`�  �          @�����@K�?�
=A�
=B�����@J=q?�p�A��\B�L�                                    Bx��o�  �          @�z���\@HQ�?�z�A��B�\)��\@G
=?��HA�\)B�                                    Bx��~.  �          @�p���@J=q?�G�A��B��R��@H��?��A��HB�{                                    Bx�Ō�  �          @��=q@G
=?��A�=qB�p��=q@E?��A�(�B��
                                    Bx�śz  �          @�  �!�@;�?�  A�  C�)�!�@9��?��A�C&f                                    Bx�Ū   
~          @�{��R@>{?���A���C�
��R@<(�?�33A���C)                                    Bx�Ÿ�  T          @��R�\)@AG�?��A�p�CxR�\)@?\)?˅A��C��                                    Bx���l  T          @���*�H@E?��\A`��C��*�H@Dz�?�=qAmp�C�                                    Bx���  �          @����(�@[�?+�A��B����(�@Z�H?:�HA�RB�.                                    Bx���  T          @����{@i��>��
@��B���{@h��>\@��B�                                    Bx���^  	�          @�  �{@c33?!G�A��B��f�{@b�\?333A�
B�{                                    Bx��  "          @����Q�@h��?+�A(�B��Q�@hQ�?=p�A�B�Ǯ                                    Bx���  "          @�G���
=@s�
>Ǯ@��B���
=@s33>�@�B�Ǯ                                    Bx��P  �          @����(�@p  >B�\@#33B�=��(�@o\)>�=q@e�B뙚                                    Bx��-�  �          @�z��z�@j=q>W
=@5B��ÿ�z�@j=q>�\)@y��B�                                    Bx��<�  
�          @��H��\@`��>��R@�(�B�W
��\@`  >Ǯ@�B�p�                                    Bx��KB  �          @���=q@u�?z�HAX��B�aH��=q@s33?��Ak\)Bڔ{                                    Bx��Y�  �          @�p���
=@s�
?^�RA@��B�8R��
=@r�\?s33AS�B�k�                                    Bx��h�  "          @��
��@i��>\@��\B陚��@h��>�@�
=B�R                                    Bx��w4  �          @�G��%@E���{����C
�%@E���=q�qG�C�                                    Bx�ƅ�  T          @����-p�@<(��Ǯ���HC� �-p�@<(����
���C��                                    Bx�Ɣ�  T          @\)�(��@:�H�8Q��%C��(��@:�H��G���=qC
=                                    Bx�ƣ&  "          @�  �%@AG���z����C���%@AG��W
=�C�
C�=                                    Bx�Ʊ�  
Z          @\)�33@Z=q�u�`��B��f�33@Z�H�#�
�z�B���                                    Bx���r  
�          @�G��
=@P�׾���j=qB��
=@QG��8Q�� ��B��                                    Bx���  �          @���/\)@>{=�\)?p��C���/\)@>{>\)@   C�R                                    Bx��ݾ  
�          @������@L��=L��?=p�B��f���@L(�>�?�33B��                                    Bx���d  �          @�=q�!G�@L(�=#�
?z�C J=�!G�@L(�=�?�  C O\                                    Bx���
  �          @��\�0  @@��=���?�z�C^��0  @@��>8Q�@!G�Ch�                                    Bx��	�  �          @���8��@7
=<��
>��CQ��8��@6ff=�Q�?�=qCW
                                    Bx��V  �          @�G��4z�@7�>\)?�C���4z�@7�>W
=@AG�C��                                    Bx��&�  �          @�G��.�R@=p�>\)?�Q�C�f�.�R@=p�>aG�@E�C��                                    Bx��5�  �          @�=q�0��@>{>.{@
=C�)�0��@>{>�  @aG�C�                                    Bx��DH  T          @�=q�.�R@AG�=�Q�?��\C33�.�R@@��>.{@��C:�                                    Bx��R�  �          @�G��33@R�\>���@�B��33@Q�>�
=@�Q�B��                                    Bx��a�  �          @����%�@E�=L��?&ffC�H�%�@E�>�?�z�C��                                    Bx��p:  �          @����)��@<(�?!G�A�C��)��@:�H?8Q�A$��CO\                                    Bx��~�  
(          @�����R@U�>�(�@�B��\��R@Tz�?�@�=qB���                                    Bx�Ǎ�  �          @�33��
@`��>�{@�33B�����
@`  >�ff@��B���                                    Bx�ǜ,  T          @�=q��
@\(�>���@���B�3��
@[�>���@���B��)                                    Bx�Ǫ�  "          @�z��333@>{?
=Az�CJ=�333@<��?.{A��Cz�                                    Bx�ǹx  �          @��aG�?�{?��\Af�RC8R�aG�?���?�=qAs�C�H                                    Bx���  
�          @�
=�|�ͽ�\)?�Q�A�{C5
=�|�ͽ�?�
=A�C5��                                    Bx����  "          @�{�z=q?
=?�\)A�G�C+ff�z=q?��?��A�\)C,                                    Bx���j  
�          @��R�w�?!G�?\A�33C*�R�w�?z�?��A�p�C+k�                                    Bx���  �          @���q�?���?�\)A��C ���q�?��
?�A�=qC!J=                                    Bx���  "          @�Q��xQ�?�p�?�(�A���C"c��xQ�?�Q�?�G�A�  C"�                                    Bx��\  "          @�G��{�?^�R?���A�(�C'���{�?Q�?�(�A�G�C(+�                                    Bx��   
�          @����|(�?�?��RA��
C+�q�|(�?�?�G�A��C,p�                                    Bx��.�  �          @�
=��  ?5?��Ap(�C)���  ?+�?�{Aup�C*s3                                    Bx��=N  T          @��R�p  ?Q�?�
=A��C'�q�p  ?B�\?��HA�=qC(�\                                    Bx��K�  T          @�\)�|��>�{?��HA�\)C/��|��>�z�?�(�A��\C/�{                                    Bx��Z�  
�          @��R�u�>�p�?��HA��C.xR�u�>��R?�(�A���C/T{                                    Bx��i@  
�          @�{�u?��?���A�ffC+���u>��H?˅A�z�C,�3                                    Bx��w�  �          @�{�tz�?\)?У�A��C+�R�tz�?   ?�33A�33C,��                                    Bx�Ȇ�  
Z          @��R�{�?W
=?�  A�(�C'�q�{�?J=q?��
A�\)C(��                                    Bx�ȕ2  �          @�{��G�?Y��?5A��C(
��G�?Tz�?=p�A#33C(p�                                    Bx�ȣ�  T          @�Q���  >�Q�?��A��RC.�\��  >��R?�33A�{C/��                                    Bx�Ȳ~  
�          @���~{?@  ?�G�A�z�C)Q��~{?0��?��A�\)C*33                                    Bx���$  T          @����\)?���?L��A+�C!�q�\)?��?W
=A5C"�                                    Bx����  
�          @����z�H?��
?#�
A33C�H�z�H?�G�?333A33C�                                    Bx���p  "          @��R�x��?���?5A33C�
�x��?�?B�\A&�\C�                                    Bx���  T          @������?L��?xQ�AUp�C(�3����?E�?�  A[�
C)0�                                    Bx����  T          @��R�z�H?�ff?s33AS
=C$�3�z�H?��\?}p�A[�C%n                                    Bx��
b  T          @����w
=?�z�?p��AQG�C#J=�w
=?�\)?z�HAZ�HC#�                                    Bx��  �          @���{�?c�
?��
Ad��C'G��{�?Y��?��Al(�C'��                                    Bx��'�  
Z          @��R�{�?z�?�{A���C+�)�{�?�?���A�  C,W
                                    Bx��6T  !          @��p��?�p�>�@��C8R�p��?�(�?�@�RCs3                                    Bx��D�  �          @�G��G�@   ��
=���C^��G�@ �׾������RC5�                                    Bx��S�  
�          @���J=q@ �׾���{Cz��J=q@!녾�Q����\CL�                                    Bx��bF  �          @���L(�@�ÿ:�H�%�C8R�L(�@=q�!G��(�C�                                    Bx��p�  T          @����N�R@��\)�   C5��N�R@ff�����ffC��                                    Bx���  �          @�G��U�@ff�O\)�8��C�U�@Q�8Q��%Cff                                    Bx�Ɏ8  �          @�G��QG�?��R��  ��33C���QG�@�\��
=��C                                      Bx�ɜ�  �          @�G��>{@G���\��=qC�)�>{@���Q���{C��                                    Bx�ɫ�  "          @�G��E?�G����י�CaH�E?��ÿ޸R�Ώ\Ck�                                    Bx�ɺ*  
7          @��H�Y��@���G�����CJ=�Y��@�;�33��p�C�                                    Bx����  �          @�33�`  @�\��G���  C�R�`  @�
��Q���33C��                                    Bx���v  �          @����g�?�p��xQ��_�
C�
�g�?�G��h���QCT{                                    Bx���  �          @�G��l��?L�Ϳ�����C'ٚ�l��?Y�������  C'�                                    Bx����  T          @����dz�?�
=��(�����C!���dz�?��R����
=C �{                                    Bx��h  T          @�z��N�R@ff�W
=�<��C�3�N�R@Q�=p��'
=C�
                                    Bx��  "          @�(��U@G��J=q�1p�C�\�U@33�333�z�Cs3                                    Bx�� �  c          @��
�W�@�\�����z�C�R�W�@�
�\��=qC�                                    Bx��/Z  �          @��\�P��@���
=q��p�C�)�P��@�H��ff����CaH                                    Bx��>   �          @����O\)@녿Y���A�C�f�O\)@�
�B�\�,(�C��                                    Bx��L�  �          @}p��Q�@
=�.{���CL��Q�@�ÿ
=�z�C��                                    Bx��[L  T          @~{�QG�@��Y���D��C���QG�@
=�B�\�0(�C(�                                    Bx��i�  T          @�Q��Fff@\)��Q���  C=q�Fff@녿���}��C��                                    Bx��x�  f          @�G��@  @33��z���C� �@  @
=�����  C��                                    Bx�ʇ>  T          @�Q��@��@\)�����(�CY��@��@33������\C��                                    Bx�ʕ�  �          @~{�C33@p����R��\)C\�C33@�׿����  Cp�                                    Bx�ʤ�  �          @����7
=@Q���
��33C33�7
=@(���
=���RCu�                                    Bx�ʳ0  �          @���:�H@���
=����C.�:�H@\)������{C��                                    Bx����  �          @��\�5@   ��G���z�C
���5@#�
��z����C
�                                    Bx���|  �          @���,(�@,�Ϳ�����p�C���,(�@0�׿�G����CL�                                    Bx���"  �          @�  �%�@/\)��{���HC:��%�@333���R���\C��                                    Bx����  �          @~{� ��@.�R��z�����C�{� ��@2�\��ff��ffC��                                    Bx���n  �          @\)�(Q�@/\)��(���(�C�=�(Q�@2�\������CG�                                    Bx��  �          @�G���@(�ÿ�=q�؏\C����@.{��(����
C��                                    Bx���  �          @�G��   @*�H�ٙ���C{�   @/\)��=q��33CT{                                    Bx��(`  �          @}p��!G�@����{��RCk��!G�@{��  ���HC�                                     Bx��7  �          @���:=q@*=q�}p��`��C	���:=q@,�Ϳ^�R�E�C	#�                                    Bx��E�  T          @����=p�@'
=�h���P(�C
���=p�@(�ÿL���5��C
:�                                    Bx��TR  �          @�  �G
=@���B�\�.�RCT{�G
=@��&ff�{C��                                    Bx��b�  �          @�  �B�\@�ͿW
=�B{C.�B�\@�R�:�H�(��CǮ                                    Bx��q�  �          @\)�(Q�@*�H��\)��(�C�H�(Q�@.{��  ��C                                    Bx�ˀD  �          @�G��(��@*�H���H��p�C���(��@.�R�����
=C                                    Bx�ˎ�  �          @�Q��(Q�@2�\���
�pQ�CJ=�(Q�@5��fff�R�RC�)                                    Bx�˝�  �          @���E@,�Ϳ�����C
���E@.�R����(�C
��                                    Bx�ˬ6  �          @�{�L��@&ff�&ff�=qC޸�L��@(Q����G�C�
                                    Bx�˺�  �          @�z��]p�@�;���
=C���]p�@p��Ǯ���Cff                                    Bx��ɂ  �          @�z��fff@ �׾u�Tz�Cٚ�fff@G������C�q                                    Bx���(  �          @�(��g�?�{��R�	C�g�?�녿���{Cp�                                    Bx����  �          @�(��\(�@   ����j{C޸�\(�@�\�s33�Up�CQ�                                    Bx���t  �          @�z��P��@�h���L(�CO\�P��@Q�L���4(�C޸                                    Bx��  �          @����;�@7��(��(�C���;�@8�þ��ָRCc�                                    Bx���  �          @���J�H@
=���H��33CW
�J�H@
�H��\)���C�3                                    Bx��!f  �          @��H�<(�@0  �+���
C��<(�@1녿����\)C�f                                    Bx��0  �          @��\�;�@3�
���
����C5��;�@4z�L���1G�C
                                    Bx��>�  �          @�=q�G
=@'��#�
��Cٚ�G
=@'�=��
?���C�)                                    Bx��MX  �          @�=q�P  @�;8Q��"�\C��P  @�ͽ�\)�z�HC�                                    Bx��[�  �          @���H��@'
=��{���CJ=�H��@'��k��J=qC&f                                    Bx��j�  �          @��H�C33@-p����R��p�C
W
�C33@.{�B�\�,��C
8R                                    Bx��yJ  �          @�33�;�@5��Q����HC��;�@6ff�k��R�\C��                                    Bx�̇�  �          @�33�2�\@?\)��  �a�C�R�2�\@@  ���У�C��                                    Bx�̖�  �          @��
�#�
@Mp����Ϳ�Q�C � �#�
@N{=#�
?
=C }q                                    Bx�̥<  �          @�z��6ff@@  �L���5�CxR�6ff@@�׽�\)�s33Ch�                                    Bx�̳�  �          @���6ff@>�R�����(�C���6ff@@  ��
=��  Cu�                                    Bx��  �          @�{�J=q@{��
=���\C��J=q@!G�����k�Cu�                                    Bx���.  �          @����N{@ff�����v=qCٚ�N{@���}p��]��CO\                                    Bx����  �          @�ff�G
=@,�Ϳ=p��"�RC  �G
=@.�R��R�\)C
�                                    Bx���z  �          @�
=�C33@5��   ��z�C	\�C33@6ff��p����
C�)                                    Bx���   �          @����G�@{��33��33C���G�@ �׿���hz�C+�                                    Bx���  �          @�z��E@
�H������C޸�E@\)��Q�����C
=                                    Bx��l  �          @�z��AG�@0�׿!G����C	�\�AG�@2�\��\����C	J=                                    Bx��)  �          @�(��<��@2�\�:�H�$  C���<��@4z�(��\)CB�                                    Bx��7�  �          @�p��@  @1녿G��,��C	!H�@  @4z�&ff�(�C��                                    Bx��F^  �          @�p��O\)@ �׿0���p�C(��O\)@"�\�z����C�
                                    Bx��U  �          @�z��?\)@.�R�Q��6=qC	�\�?\)@1G��0���=qC	5�                                    Bx��c�  �          @�(��0  @@  �����RC���0  @AG������  CG�                                    Bx��rP  �          @��
�-p�@C33�����C���-p�@Dz��(���
=Cs3                                    Bx�̀�  �          @�p��0��@E�����G�C�q�0��@G
=������33C��                                    Bx�͏�  �          @���/\)@C33����G�C���/\)@E�������C��                                    Bx�͞B  �          @�
=�/\)@G��333���CJ=�/\)@I���\)��33C�                                    Bx�ͬ�  
�          @����5�@HQ�#�
�
ffC(��5�@J=q�   ��
=C�                                    Bx�ͻ�  T          @���7
=@AG��E��'�Ch��7
=@C33�!G��	G�C)                                    Bx���4  �          @�
=�2�\@>{���\�`(�C+��2�\@@�׿aG��A�C                                    Bx����  T          @�
=�J=q@.{�
=� ��CT{�J=q@/\)����˅C{                                    Bx���  T          @�  �+�@K��O\)�0��C.�+�@Mp��+����C��                                    Bx���&  "          @�  �(��@L�ͿG��)Cz��(��@O\)�!G��	�C5�                                    Bx���  
(          @��
�Tz�@  ?B�\A,Q�C�
�Tz�@{?\(�AC\)CG�                                    Bx��r  T          @����Mp�@'
==u?J=qC���Mp�@'
=>.{@�HC޸                                    Bx��"  �          @�z��?\)@6ff=�\)?s33Cc��?\)@5>L��@.�RCs3                                    Bx��0�  T          @�  �J=q@4z�>�z�@~�RC
T{�J=q@333>�
=@�C
}q                                    Bx��?d  "          @�=q�`��@\)>��R@��C�H�`��@�R>�
=@��HC�\                                    Bx��N
  �          @�(��dz�@!G�>�  @S33C���dz�@ ��>�Q�@���C�H                                    Bx��\�  
�          @��
�dz�@!G�>\)?���C���dz�@ ��>�  @S33C�\                                    Bx��kV  �          @�z��b�\@"�\?   @љ�CG��b�\@!G�?(�A z�C�=                                    Bx��y�  T          @��
�mp�@\)>�p�@�(�Cٚ�mp�@{>��@�ffC\                                    Bx�Έ�  
�          @���g�@
=?��@���C��g�@�?333Az�C=q                                    Bx�ΗH  
�          @�ff�o\)@�>�p�@�
=C���o\)@ff>��@\Cٚ                                    Bx�Υ�  T          @����o\)@!�>��?�33C�q�o\)@!G�>��@Tz�C{                                    Bx�δ�  "          @����c33@p�?��@�RC@ �c33@(�?&ffA
=qC�=                                    Bx���:  �          @����Z=q@Q�?�=qAh  C��Z=q@�?�
=A~�HC��                                    Bx����  "          @�z��l(�@  ?!G�A��C���l(�@{?:�HAC                                      Bx����  �          @�{�l��@�\?\(�A2{CJ=�l��@  ?uAG
=C��                                    Bx���,  �          @�\)�tz�@  ?\)@�
=C�{�tz�@{?(��A�C�H                                    Bx����  "          @��H�fff@
=>�ff@���C��fff@�?\)@�z�C�                                    Bx��x  
�          @�ff�~�R?�(�>aG�@4z�C�)�~�R?��H>���@z�HC�q                                    Bx��  �          @�\)�s33@>�  @P��CaH�s33@�>�Q�@��C�                                    Bx��)�  
�          @���w
=@=q>u@>{C��w
=@��>�{@���C#�                                    Bx��8j  "          @����q�@{>k�@7�C޸�q�@p�>���@�
=C                                      Bx��G  "          @�\)�p  @�H>���@�z�C!H�p  @��?�\@�
=CW
                                    Bx��U�  T          @��R�fff@#33>�p�@�\)C���fff@"�\>��@���Cٚ                                    Bx��d\  �          @���s33@Q�>B�\@��C��s33@�>�z�@l(�C�                                    Bx��s  �          @�G���Q�@>�{@��C� ��Q�@z�>�(�@�{C�                                    Bx�ρ�  T          @�  �u�@{?=p�Az�C���u�@(�?W
=A,  CE                                    Bx�ϐN  �          @�p���{?�G�?
=q@�p�C&n��{?}p�?z�@��C&�q                                    Bx�Ϟ�  �          @�����?\(�?&ffA�HC(�f����?Tz�?.{AffC)                                    Bx�ϭ�  T          @�������?�
==�\)?uC k�����?�
=>�?�\C xR                                    Bx�ϼ@  �          @�  �s�
?��ý�Q쿕Cu��s�
?��ü#�
��Cp�                                    Bx����  
�          @���z=q?��;���G�C�R�z=q?�{�u�Y��C��                                    Bx��ٌ  �          @�=q�}p�?����
��{C!H�}p�?�
=��  �P��C��                                    Bx���2  T          @��H�x��?����
��{C#E�x��?�����
=C#=q                                    Bx����  
Z          @��\�tz�?^�R?xQ�A]�C'33�tz�?Tz�?�  Af{C'Ǯ                                    Bx��~  
�          @�33�xQ�?
=?uA[�
C+L��xQ�?��?z�HAap�C+�H                                    Bx��$  T          @���u>�\)?��\A��C/޸�u>aG�?��
A��C0�f                                    Bx��"�  
�          @����z�H=L��?fffAN{C3G��z�H<#�
?fffAN=qC3�\                                    Bx��1p  T          @�33���׿z�>�@�p�C<B����׿��>�(�@\C<��                                    Bx��@  �          @��
���H�aG�=L��?=p�C7����H�aG�=#�
?�RC7\                                    Bx��N�  �          @�(��xQ쾳33?��\A�Q�C9+��xQ����?�  A��\C9�                                    Bx��]b  �          @���z�H����?�33A���C8ٚ�z�H�\?��A��C9�                                    Bx��l  "          @~�R�r�\�aG��u�j�HC7G��r�\�W
=��  �s33C7!H                                    Bx��z�  �          @����x�þB�\>�Q�@�33C6�\�x�þL��>�Q�@�\)C7                                    Bx�ЉT  "          @��\�p  ���?!G�A��CG���p  ��{?z�A��CG��                                    Bx�З�  
�          @�{�HQ쿳33@  B�CL\�HQ쿽p�@(�BQ�CMaH                                    Bx�Ц�  T          @}p��XQ쿗
=?У�A�CGO\�XQ쿠  ?�=qA��CH@                                     Bx�еF  "          @k��N{�h��?&ffA/33CC�\�N{�p��?(�A%G�CD5�                                    Bx����  �          @�  ��{@J�H�33�ffB�L;�{@P������B���                                    Bx��Ғ  "          @�G��k�@Q��>{�$�B�8R�k�@Y���5��\B�=q                                    Bx���8  �          @�33��(�@N�R�9����
B��ÿ�(�@U�1G����B�\                                    Bx����  �          @������@J�H�;��=qB陚����@Q��3�
�ffB�
=                                    Bx����  "          @�zῸQ�@C�
�H���+p�B�\��Q�@K��AG��#�B��                                    Bx��*  T          @�zΐ33@Vff�?\)�!  B��)��33@]p��7����Bؽq                                    Bx���  T          @�33�ٙ�@*=q�S33�9=qB�#׿ٙ�@1��L���1�B��
                                    Bx��*v  "          @�=q��
=@%��W��>33B�8R��
=@-p��QG��6��B�Ǯ                                    Bx��9  "          @��H��p�@4z��XQ��>�B�B���p�@<(��Q��7{B�z�                                    Bx��G�  �          @�녿Tz�@@  �Tz��;��B���Tz�@G��Mp��3p�BѮ                                    Bx��Vh  �          @���   @HQ��S�
�8��B�B��   @P  �L(��0z�Bř�                                    Bx��e  T          @�z���>k������C+����>Ǯ��G�#�C&��                                    Bx��s�  T          @�z῜(�>�z���W
C&�{��(�>�ff��p��C�3                                    Bx�тZ  
�          @�33�˅>.{�����)C-�q�˅>��
��Q�=qC(�
                                    