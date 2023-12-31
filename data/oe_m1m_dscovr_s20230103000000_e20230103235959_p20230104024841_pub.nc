CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230103000000_e20230103235959_p20230104024841_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-04T02:48:41.123Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-03T00:00:00.000Z   time_coverage_end         2023-01-03T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill         �   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxWN�   �          @i������@\��?��A�\)B�(�����@C33?��HB��B��f                                    BxWN�  "          @xQ�z�@c33?�z�A�G�Bƨ��z�@C�
@�\B\)Bə�                                    BxWN�L  �          @aG��8Q�@XQ�?
=A�HB��8Q�@Fff?���A��
B�\                                    BxWO�  �          @c33��R@QG��h���u�B�ff��R@X�ü����Bȣ�                                    BxWO�  T          @aG��aG�@:=q��  ����B��R�aG�@QG��n{�|Q�B�Ǯ                                    BxWO!>  
�          @z=q��\)@,���333�7��B��R��\)@U�   ����B��                                     BxWO/�  
�          @}p��fff?���Y���m�B�p��fff@'
=�3�
�6
=B�                                    BxWO>�  �          @{��xQ�?��Q��bB�uÿxQ�@-p��*�H�+{B�Q�                                    BxWOM0  T          @\(���(�@�G����B�=q��(�@2�\�����(�B�u�                                    BxWO[�  �          @Y����\)@p��33���B��)��\)@:=q������B�
=                                    BxWOj|  
�          @K��xQ�@�׿�p��{B➸�xQ�@,�Ϳ������
B۸R                                    BxWOy"  T          @H��>.{?�p��{�@��B�>.{@\)�У���B�
=                                    BxWO��  T          @U?=p�?��*=q�U{B�G�?=p�@{���B��3                                    BxWO�n  "          @Q�?=p�@
=���:p�B�� ?=p�@*=q�޸R� ��B��                                    BxWO�  �          @J�H��@,(���\)����B����@AG��^�R��z�B���                                   BxWO��  
�          @J=q=L��@1G���G���ffB��=L��@Dz�=p��Xz�B�L�                                    BxWO�`  T          @Z�H>.{@<(���p���p�B�ff>.{@R�\�h���w33B��                                    BxWO�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWO߬   �          @c�
>�33@HQ��33��p�B�G�>�33@\�ͿJ=q�LQ�B�p�                                    BxWO�R  T          @^�R>aG�@HQ쿼(���(�B��>aG�@Z=q�(��!G�B��\                                    BxWO��  �          @H�ÿ��
@p�?k�A��B��f���
?�33?��HBffB��f                                    BxWP�  �          @]p���p�@:=q?��A�B�  ��p�@{@ ��BG�B�\                                    BxWPD  T          @p  ��
?�ff@�B-  C.��
?^�R@/\)BK
=C33                                    BxWP(�  T          @s�
�z�?�Q�@7
=BBQ�C@ �z�?&ff@HQ�B]
=C"�                                    BxWP7�  �          @w��z�?��H@2�\B8\)Cp��z�?n{@HQ�BXG�C�3                                    BxWPF6  �          @qG�����@\)@�
B�
B�k�����?�@7�BG��Ck�                                    BxWPT�  T          @w
=���@\)@�B�\B��R���?��
@?\)BKC�
                                    BxWPc�  �          @x�ÿУ�@=q@$z�B%�B�.�У�?�@FffBT
=CQ�                                    BxWPr(            @z=q����@G�@#�
B$C8R����?��@C�
BOG�CT{                                    BxWP��  T          @y���@
=q@�RB(�C
=�?���@<��BD�C0�                                    BxWP�t  
Z          @w���?�{@%�B(33C�)��?���@>{BI��C�\                                    BxWP�  
(          @y����R?�@(Q�B*G�C����R?���@@��BJffC:�                                    BxWP��  
�          @u��	��@�
@Q�B�\CQ��	��?���@5�B?p�CW
                                    BxWP�f  "          @q���?�\)@\)B%=qC
����?�@8Q�BG��C�                                    BxWP�  T          @tz��ff?޸R@�B��C}q�ff?��@2�\B=(�C�                                    BxWPز  S          @tz��,(�?�(�@�B��CO\�,(�?W
=@%�B)��C"�)                                    BxWP�X  
Z          @tz��%�?�z�@G�B��C:��%�?��
@'�B-
=C=q                                    BxWP��  
(          @p  �1G�?�Q�@z�B�Cz��1G�?^�R@�B��C"��                                    BxWQ�  
�          @p  �%?�
=@�B	�C\�%?��@�RB%ffC@                                     BxWQJ  
�          @l(��(��?��H?�z�A��
C��(��?�@�B�C{                                    BxWQ!�  	�          @p  �   ?�G�@
=qB�C�
�   ?�33@!�B*�C.                                    BxWQ0�  
�          @g
=���@��?�\)A��
B�
=���@G�?���B�
C�                                    BxWQ?<  T          @\�Ϳ�@=p�?B�\AP  B�Q쿵@*�H?�p�AУ�B�
=                                    BxWQM�  	�          @mp���
=@1�?�G�A�ffB�z��
=@�
@
=qB�HC�)                                    BxWQ\�  T          @n�R�{@-p�?��A�\)CL��{@33?�
=A���C�q                                    BxWQk.  
�          @n�R�G�@7�?�(�A�p�B�B��G�@{?�z�A���CE                                    BxWQy�  T          @o\)��33@?\)?�A�B��)��33@&ff?��A�=qB�L�                                    BxWQ�z  �          @]p���Q�@2�\?��A�  B�uÿ�Q�@�?�(�A��
B���                                    BxWQ�   T          @I��>���@G�����p�B�W
>���@  ��R�tz�B�                                    BxWQ��  "          @J�H���
@{�xQ����B=���
@(�þ��
�ڏ\B���                                    BxWQ�l  
�          @N�R��33@5�?�A\)B���33@'
=?�(�A��HB�z�                                    BxWQ�  �          @Fff����@!G�?.{AN=qB��H����@G�?��AǙ�B�p�                                    BxWQѸ  
�          @>�R��z�@%�>\@��B�k���z�@��?�G�A�{B��                                    BxWQ�^  
�          @@  ���
@$z�#�
�@  B����
@   ?��A:=qB��                                    BxWQ�  
(          @0�׿�\)@p��u���B�\)��\)@��>��
@�  B���                                    BxWQ��  
�          @7��W
=@���33��=qB��H�W
=@%����-�B��                                    BxWRP  	�          @?\)��
=@ �׾�����=qB�k���
=@ ��>���@�ffB�W
                                    BxWR�  "          @P�׿�  @)�����
���B�Ǯ��  @)��>�{@ƸRB��
                                    BxWR)�  
�          @U���\@/\)?�A�B��q��\@!G�?�p�A�Q�B�L�                                    BxWR8B  
Z          @G�����@7
=>B�\@aG�B�LͿ���@.{?c�
A�ffB�B�                                    BxWRF�  
�          @1녿�\@'
==��
?�=qB�B���\@   ?:�HA}B�#�                                    BxWRU�  
�          @'
=��@�ü#�
�uB��ͽ�@z�?z�A`z�B���                                    BxWRd4  �          @"�\�L��@	���}p����B�uþL��@���(��(z�B���                                    BxWRr�  
�          @\)�aG�@�>�  @��
B���aG�@	��?L��A�
=B�p�                                    BxWR��  T          @0�׾��@*�H>���@�G�B�ff���@!G�?s33A���B�p�                                    BxWR�&  
�          @)�����
@#�
>u@��RB�aH���
@�H?\(�A�G�B��{                                    BxWR��  T          @'
==�\)@   ���5�B��R=�\)@p�>��HA5��B���                                    BxWR�r  
�          @S�
?(�@+�����ffB��?(�@@  �u���B�8R                                    BxWR�  T          @`�׿fff@Tz�L�Ϳc�
B�aH�fff@N�R?B�\AL  B�(�                                    BxWRʾ  �          @l�Ϳ:�H@aG��������B�uÿ:�H@_\)?�A(�B˨�                                    BxWR�d  "          @i����G�@Z=q>�@�
Bܨ���G�@Q�?uAt  B�.                                    BxWR�
  
�          @`�׿��@Q�>�\)@��B�=q���@G�?���A�  B�                                      BxWR��  "          @a녿�Q�@�R?��A��C��Q�?�p�@�B$�C
Y�                                    BxWSV  
�          @hQ��\@R�\�
=q�  B�Ǯ��\@Tz�>�=q@��Bř�                                    BxWS�  "          @a녿�=q@333?���A���B��)��=q@��?�  A��B���                                    BxWS"�  �          @j=q=#�
@X��?�p�A�(�B���=#�
@?\)@G�B  B�p�                                    BxWS1H  "          @]p��&ff?�?��A���C�=�&ff?��H?�  A���C�q                                    BxWS?�  
�          @h���{@&ff?�p�A�ffC�\�{@�R?���A��\C��                                    BxWSN�  T          @k��33@#�
?�A��\C�3�33@��?�  A�RC@                                     BxWS]:  T          @j=q�	��@#�
?��RA��
C)�	��@�@�
B�
Cc�                                    BxWSk�  �          @w
=�Q�@   ?ٙ�A�33C�
�Q�@G�@  B�
C��                                    BxWSz�  "          @n�R�G�@\)?�z�A��
Cz��G�?��H@��B G�C�                                    BxWS�,  
�          @[��ff?�Q�@G�B  C���ff?h��@33B)C�                                    BxWS��  "          @]p��{?���@G�B$p�C�f�{?E�@!�B<33C ��                                    BxWS�x  "          @w
=�)��@�R?�  A�Q�C���)��@
=?�A��Cs3                                    BxWS�  
�          @z=q�8��@
=�B�\�7�C��8��@>\@��\C�                                    BxWS��  	�          @{��E�@��������C^��E�@�>�=q@\)CT{                                    BxWS�j  �          @z�H�+�@!G�?n{Ab�\C���+�@�R?�  A�(�C8R                                    BxWS�  T          @y���Q�@!�?�(�AҸRC0��Q�@33@G�BffC+�                                    BxWS�  "          @|���	��@6ff?�A���B�\�	��@�@33B��C(�                                    BxWS�\  �          @\)�Q�@6ff?��A��\C���Q�@(�@�A�C8R                                    BxWT  
�          @�Q��*=q@"�\?�  A�
=CY��*=q@�@33A���C��                                    BxWT�  "          @����O\)@?���A�{C#��O\)?�\?˅A��
CY�                                    BxWT*N  �          @����6ff@�H?�  A�ffC���6ff@ ��@G�A���C�H                                    BxWT8�  
�          @z�H�B�\?�z�?��HA�z�C�B�\?k�@�RB��C#.                                    BxWTG�  
�          @xQ��B�\?�  ?�33A��C
=�B�\?���@ ��A��C�=                                    BxWTV@  T          @z=q�HQ�?�  ?�\A�Q�CL��HQ�?�ff@z�B ��C!k�                                    BxWTd�  �          @|(��Q�?��
?��A��\C�3�Q�?���?�{A�33C �3                                    BxWTs�  
(          @{��Z�H?�=q?��A�  C"}q�Z�H?0��?�  A�
=C(�\                                    BxWT�2  T          @�  �R�\?�(�?�p�A�
=C�H�R�\?��
@G�A�G�C"��                                    BxWT��  �          @|���I��?�ff?��RA�  C���I��?L��@�RB	�C%�                                    BxWT�~  
�          @p  �C33?.{@33B��C'}q�C33>B�\@	��Bp�C0��                                    BxWT�$  "          @~{�:=q@33?�\)A��RC���:=q?�?�\)A�\)C�                                    BxWT��  T          @{��1G�@�?ǮA�p�C�\�1G�?�{@�\A��C�                                    BxWT�p  �          @|(��E?���?��
A�(�C�E?���?�z�A�=qC�=                                    BxWT�  �          @�  �Z=q?���?�p�A��C!���Z=q?333?�Q�A�\)C(aH                                    BxWT�  �          @�G��Z=q?�33?�A�33C!\)�Z=q?5@G�A���C(E                                    BxWT�b  �          @~{�[�?��?�p�A�Q�Cp��[�?h��?޸RAУ�C%\                                    BxWU              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWU�             @x���c�
?�  ?���A���C$\)�c�
?333?��A�  C(��                                    BxWU#T  T          @z=q�a�?�=q?��
A�33C#
=�a�?B�\?��RA���C'޸                                    BxWU1�  �          @u��S�
?�G�?�{A��
C#��S�
?�R?�ffA��HC)c�                                    BxWU@�  "          @s33�W�?���?�{A�\)C!���W�?B�\?���A�33C'8R                                    BxWUOF  �          @qG��W
=?���?��RA��
C!aH�W
=?Tz�?��HA���C&5�                                    BxWU]�  T          @qG��S33?�Q�?��A�ffC 33�S33?Y��?�\)A�
=C%��                                    BxWUl�  �          @l���8��?Ǯ?�{Aљ�C�H�8��?�z�?�A��C#�                                    BxWU{8  
�          @vff�Z�H?��?��A�
=C"L��Z�H?@  ?���A�  C'�\                                    BxWU��  
�          @����e�?�z�?�(�A��RC"��e�?L��?�Q�A��C'ff                                    BxWU��  T          @l���N�R?��
?�(�A��
C"E�N�R?.{?�A�=qC(�                                    BxWU�*  
�          @w��XQ�?n{?У�A�  C$��XQ�?
=q?�ffA�C*��                                    BxWU��  T          @dz��HQ�?:�H?ǮAθRC&�3�HQ�>�Q�?�
=A�Q�C-aH                                    BxWU�v  
�          @dz��G�?J=q?��A�Q�C%�q�G�>�(�?�
=A�  C,{                                    BxWU�  
�          @a��G
=?Tz�?�Q�A��C%
=�G
=>��H?˅A�Q�C*�q                                    BxWU��  
�          @j=q�H��?u?˅AΣ�C"���H��?
=?�G�A��C)O\                                    BxWU�h  �          @s33�P��?n{?�Q�A�Q�C$#��P��?�?�{A�z�C*��                                    BxWU�  �          @r�\�W�?Y��?��RA�\)C%޸�W�?   ?��A�G�C+��                                    BxWV�  �          @n�R�U?O\)?�Q�A���C&c��U>��?˅AɮC+�                                    BxWVZ  �          @k��Vff?L��?��\A�(�C&���Vff?   ?�A�\)C+xR                                    BxWV+   T          @c33�Tz�>\?�Q�A��\C-���Tz�=�G�?��RA��
C2)                                    BxWV9�  �          @XQ��J=q>u?�Q�A�G�C/�3�J=q��?�(�A���C4}q                                    BxWVHL  "          @Vff�G
=>W
=?�(�A���C0#��G
=�u?�p�A�33C5\                                    BxWVV�  T          @X���J=q>\)?�(�A���C1���J=q��?�(�A��RC6W
                                    BxWVe�  
�          @]p��J�H>���?��A�ffC.�f�J�H    ?�\)A��C3�3                                    BxWVt>  T          @_\)�N�R>B�\?��A�p�C0�f�N�R��Q�?�ffA���C5�)                                    BxWV��  
�          @^{�H��?8Q�?��
A��
C'!H�H��>�(�?�z�A��C,@                                     BxWV��  �          @g
=�L��?L��?��HA�
=C&��L��>��?���A�(�C+�                                    BxWV�0  �          @g
=�J�H?�ff?�\)A��HC!���J�H?:�H?ǮA�p�C'�                                    BxWV��  �          @g��HQ�?s33?��A��C#.�HQ�?��?��HA�p�C).                                    BxWV�|  
�          @`  �C�
?\(�?�33A��C$aH�C�
?
=q?�ffA�p�C)��                                    BxWV�"  �          @XQ��C33?.{?�  A��
C'aH�C33>��?�{A���C,p�                                    BxWV��  "          @\���W
=>B�\?�A(�C0���W
==�\)?\)A�C2��                                    BxWV�n  T          @c�
�W
=?W
=?+�A0Q�C&  �W
=?.{?W
=A[
=C(�
                                    BxWV�  T          @b�\�J�H?�?��A�\)CǮ�J�H?k�?�G�A��C#�                                     BxWW�  �          @e��<��?���?�{A��
Cٚ�<��?���?У�Aڏ\C��                                    BxWW`  "          @n{�HQ�?�{?��RA�z�Ch��HQ�?��
?޸RA���C!��                                    BxWW$  T          @n{�H��?�{?���A�(�C}q�H��?��
?ٙ�A�Q�C!�                                     BxWW2�  �          @j=q�HQ�?�=q?���A��HC�q�HQ�?��\?˅A�z�C!�                                    BxWWAR  �          @j�H�N{?��H?s33Aq��C�{�N{?�p�?�p�A���C�                                    BxWWO�  
�          @i���K�?�  ?p��Ao�
C�K�?��\?�p�A��
C(�                                    BxWW^�  "          @a��R�\?���?
=A{C!��R�\?n{?J=qAQG�C$8R                                    BxWWmD  T          @a��@  ?���?��RA�\)C���@  ?���?��RAȸRC \)                                    BxWW{�  �          @]p��Dz�?��
?�Q�A�
=C!c��Dz�?E�?�\)A���C%��                                    BxWW��  �          @a��G�?�Q�?��\A�  C
�G�?u?��RA��\C"��                                    BxWW�6  �          @dz��P  ?B�\?��RA���C&�=�P  ?   ?�\)A��RC+Q�                                    BxWW��  
�          @g��P��?.{?��A��C(#��P��>Ǯ?�  AĸRC-.                                    BxWW��  "          @k��Q�>�?��A��C1�\�Q녾W
=?У�A���C7�
                                    BxWW�(  "          @i���R�\�B�\?���Ȁ\C7W
�R�\�   ?�  A���C<��                                    BxWW��  "          @g
=�R�\���?�
=A��HC;{�R�\�0��?���A��C?�=                                    BxWW�t  T          @g��L�ͽu?���A��
C5#��L�;\?�ffA�p�C:�                                    BxWW�  
�          @l���
�H����@$z�B5ffCSk��
�H��@��B��C[�
                                    BxWW��  
Z          @hQ��3�
�B�\@
=qB��CC0��3�
��
=?��HB{CJ�                                    BxWXf  "          @e�5����
@
=qB�C:aH�5��=p�@�
BQ�CB�{                                    BxWX  T          @fff�;���@�B=qC6���;���@ ��B(�C>W
                                    BxWX+�  T          @g��+��L��@�B(��C5\�+���@�B#��C?�                                    BxWX:X  �          @g��"�\�L��@%�B5�\C533�"�\�\)@!G�B033C@W
                                    BxWXH�  T          @h���&ff?��@p�B*�
C(��&ff=u@!G�B0
=C2�3                                    BxWXW�  �          @g
=�2�\?#�
@��B�C'!H�2�\>W
=@�RB�
C/�3                                    BxWXfJ  �          @g��9��?8Q�@   BG�C&
=�9��>��
@ffB(�C-�3                                    BxWXt�  �          @fff�S�
?J=q?�33A�
=C&���S�
?��?��
A���C*�{                                    BxWX��  "          @c�
�*=q?��@
=B  C���*=q?�R@G�B=qC&�f                                    BxWX�<  �          @dz��<��?��\?�G�A�C�R�<��?u?�p�A���C"
=                                    BxWX��  "          @b�\�Q�>�33@��B3p�C+���Q��@�HB5��C6Ǯ                                    BxWX��  
�          @`���   >B�\@5BZ�
C.�H�   ��Q�@5�BY  C>�                                    BxWX�.  T          @b�\�
�H>�
=@-p�BK�C)#��
�H����@/\)BN=qC6�                                     BxWX��  
�          @b�\�Q�>�33@1�BP��C*���Q�8Q�@333BRp�C8�=                                    BxWX�z  �          @a녿��H>���@:=qB^��C*�����H�k�@:�HB_�C:�{                                    BxWX�   T          @aG�����>���@>�RBi�C*�����þ��@>�RBi�C<L�                                    BxWX��  �          @c33����?L��@8Q�B[��C�H����>�  @>�RBg��C,^�                                    BxWYl  �          @a���
?�(�@
=B,ffC����
?�  @&ffBBffC�                                    BxWY  
�          @s33�}p�@b�\?&ffA (�B���}p�@W
=?��A�Q�BԳ3                                    BxWY$�  
�          @u���G�@Tz�?�z�A�33B�Ǯ��G�@C33?�  A�=qB�\                                    BxWY3^  �          @p�׿�\)@B�\?�  A�G�B�ff��\)@*�H@�B�B�B�                                    BxWYB  T          @n{���R@P  ?��A���B����R@<��?��A�B�                                    BxWYP�  "          @k����\@R�\?}p�A{\)B�Q쿢�\@C33?�=qA��B�G�                                    BxWY_P  �          @P�׿���?�ff@�
B9  CW
����?�{@#33BT
=C�
                                    BxWYm�  
Z          @C�
�޸R?s33@�RBA��CJ=�޸R?�@�BR  C#                                      BxWY|�  �          @?\)��?��H@ ��B*��C=q��?Tz�@��B?Q�Cs3                                    BxWY�B  T          @Y������@�R?��
A�ffC +�����@p�?��HA�=qCn                                    BxWY��  
�          @c33���
@;�?G�AM�B����
@/\)?��A��\B�{                                    BxWY��  T          @^�R���
@5�?@  AK33B�G����
@)��?�  A��RB��R                                    BxWY�4  �          @a녿���@L��>�G�@�(�B➸����@Dz�?�  A�{B�W
                                    BxWY��  �          @^�R��  @J�H?��A�B��ῠ  @@��?�33A���B�                                      BxWYԀ  T          @dz�s33@U�?Tz�AW33B��Ϳs33@HQ�?�33A��BոR                                    BxWY�&  �          @hQ쿆ff@Y��?0��A/�B�Q쿆ff@N�R?��
A�
=B�\                                    BxWY��  "          @mp��Y��@c�
?z�A�Bνq�Y��@Y��?���A���B��                                    BxWZ r  �          @tz�(��@l��?(��A Q�B���(��@aG�?�ffA�33B��                                    BxWZ  
�          @l�;�G�@g
==���?ǮB�  ��G�@b�\?:�HA7�
B�G�                                    BxWZ�  
�          @l�Ϳ\)@\��?�\)A���B�W
�\)@L��?ٙ�A�\)BǸR                                    BxWZ,d  �          @l(��(�@e?\)A��B�Q�(�@\(�?�A��RB�(�                                    BxWZ;
  �          @l�;��R@c�
?��A��B��ᾞ�R@Y��?���A��B�p�                                    BxWZI�  �          @r�\?�ff@U?�(�A�(�B�� ?�ff@E�?�\A��B�L�                                    BxWZXV  
�          @y��?��R@h��?!G�AG�B�u�?��R@^�R?��RA���B��)                                    BxWZf�  �          @~{?�Q�@p  >�p�@��B��
?�Q�@h��?�  Ak
=B�Ǯ                                    BxWZu�  �          @�G�?�  @n�R>B�\@.�RB��?�  @i��?Q�A;�B�(�                                    BxWZ�H  T          @���?�=q@xQ�   ��ffB�.?�=q@z=q>#�
@��B�p�                                    BxWZ��  T          @���?���@j�H�s33�]��B��
?���@qG����
��z�B�                                    BxWZ��  "          @~�R>\@`  ������B��\>\@p�׿�p�����B�ff                                    BxWZ�:  �          @xQ�>Ǯ@B�\����B��=>Ǯ@X�ÿ���p�B�                                    BxWZ��  "          @j=q>���@1����(�B�Ǯ>���@G�������B��                                    BxWZ͆  T          @\)>�
=@R�\�(���B���>�
=@g
=��\)��=qB��)                                    BxWZ�,  "          @z=q>u@U�   ��\)B��R>u@g���
=��=qB�\)                                    BxWZ��  "          @n{>Ǯ@Q녿�
=��(�B��>Ǯ@`�׿������B�ff                                    BxWZ�x  
�          @a�>B�\@W��k��t  B���>B�\@^�R��Q���{B���                                    BxW[  
�          @Y��>\@U��L���W
=B��
>\@Tz�>�{@��B�Ǯ                                    BxW[�  �          @w
=?�\@B�\�33�
=B��?�\@W�������B�Ǯ                                    BxW[%j  �          @r�\>���@@������B��R>���@U���  �܏\B�.                                    BxW[4  
�          @w
=?\(�@c�
�������B���?\(�@k���\��{B��R                                    BxW[B�  
(          @w
=?@  @a녿����33B��H?@  @l(��:�H�/�B��                                    BxW[Q\  "          @{�?xQ�@\�Ϳ�������B��=?xQ�@j=q��  �mp�B�8R                                    BxW[`  
�          @��?�@c33���
��  B�p�?�@p  �s33�Z�\B�=q                                    BxW[n�  "          @��?�ff@c�
�У���p�B��?�ff@qG���ff�p��B��
                                    BxW[}N  
�          @���?�z�@U��(����B~{?�z�@a녿n{�W33B��                                     BxW[��  
�          @��?˅@W
=��\)��  B�u�?˅@dzῊ=q�v�RB�{                                    BxW[��  "          @���?�=q@X�ÿ�(��ʣ�B���?�=q@g
=��
=���\B�k�                                    BxW[�@  
�          @�Q�?�{@\(��\��(�B��R?�{@hQ�xQ��`��B���                                    BxW[��  	�          @}p�?��R@S�
�Ǯ���B��\?��R@`  ����s\)B�                                    BxW[ƌ  �          @z�H?�\)@P  ���H��z�B�(�?�\)@^{��Q����B���                                    BxW[�2  T          @xQ�?��H@U���H��{B�(�?��H@a녿p���aG�B�\                                    BxW[��  
(          @s33?^�R@?\)����HB�aH?^�R@QG������\)B�{                                    BxW[�~  
�          @tz�?��
@%��ff�z�Bm\)?��
@9����Q���
=Bx�
                                    BxW\$  
(          @mp�?@  @B�\����Q�B�=q?@  @QG�������B�(�                                    BxW\�  
�          @g
=?
=q@P�׿���Q�B�{?
=q@\(��n{�n�\B�                                    BxW\p  
Z          @e>�@G����߅B�{>�@U���������B�#�                                    BxW\-  "          @S33=�Q�@,�Ϳ�����B�  =�Q�@:=q��p���Q�B�G�                                    BxW\;�  
(          @2�\��(�@�R<#�
>�=qB�=��(�@��>�p�@�Q�B�\                                    BxW\Jb  
�          @5�����@�>.{@Z=qB�G�����@Q�?�A)�B�L�                                    BxW\Y  "          @5��޸R@?
=qA4(�C�H�޸R?��R?W
=A�Q�CE                                    BxW\g�  
�          @�H��=q?�(�?�AZ�\C�H��=q?�\)?O\)A�
=CQ�                                    BxW\vT  �          @33���?���?�A��C5ÿ��?��?B�\A�ffC	=q                                    BxW\��  �          @)���(�?���?Q�A��C=q�(�?�=q?z�HA��C�H                                    BxW\��  "          @	�����?!G�?8Q�A��C!�����?�?O\)A���C$��                                    BxW\�F  
�          @��>#�
?s33A���C/�\��<�?uA�C30�                                    BxW\��  
�          @'
=���>�p�?��\A���C*�����>B�\?�ffA���C/#�                                    BxW\��  
�          @�R��(�?h��?�=qA��HC0���(�?@  ?���B=qCxR                                    BxW\�8  
�          @\)�Ǯ?�33?B�\A�33Cc׿Ǯ?��
?z�HA�ffC�=                                    BxW\��  "          @5��:�H@)�����R�ϮBң׿:�H@*�H=L��?��B�p�                                    BxW\�  
�          @G����H@B�\�Ǯ��B�B����H@C�
<�?�B��                                    BxW\�*  
�          @Z=q��G�@Vff��z����RB����G�@W
=>#�
@/\)B�Ǯ                                    BxW]�  "          @dz�#�
@^{�E��G�B�.�#�
@b�\������Q�B�(�                                    BxW]v  �          @aG����@[��.{�3�
B�� ���@_\)�aG��eB�\)                                    BxW]&  
(          @Z=q?=p�@Mp��0���=B���?=p�@Q녾����B�u�                                    BxW]4�  
�          @_\)?��\@L�;\����B���?��\@N�R=L��?B�\B��                                    BxW]Ch  �          @HQ�?�@A녾�z����RB�z�?�@B�\=�G�?�(�B��=                                    BxW]R  
�          @!�>B�\@
�H���E�B�  >B�\@{�B�\���B�33                                    BxW]`�  �          @P  =�G�@L(�?�\A=qB��=�G�@Fff?k�A��\B���                                    BxW]oZ  
Z          @P��>k�@L(�?�A=qB���>k�@E?n{A�(�B��\                                    BxW]~   "          @J=q�#�
@<(�?�{A��\B�8R�#�
@1�?�(�A�G�B���                                    BxW]��  T          @HQ�>�@{?��B�HB�{>�@p�@
�HB2
=B�aH                                    BxW]�L  T          @>{��?���@G�BIQ�B�.��?\@�RBeG�B�u�                                    BxW]��  	�          @;�����?��R@{BiffB�\)����?�z�@(��B��B�
=                                    BxW]��  �          @<(����H?�=q@=qB^\)B֞����H?��\@%Byp�Bހ                                     BxW]�>  �          @2�\<�?���@\)B\=qB���<�?��
@�HBx�B�
=                                    BxW]��  
�          @HQ�>k�@ff@�B#{B���>k�@�@�
B?=qB�33                                    BxW]�  �          @aG�?�G�@N�R>�=q@�ffB�aH?�G�@J�H?(��A/�
B��                                    BxW]�0  
�          @{�?��
@g
=���Ϳ\B�
=?��
@fff>�{@��B��H                                    BxW^�  
(          @|��?��@mp�?(��A�HB�W
?��@fff?���A��\B�p�                                    BxW^|  
�          @~{?c�
@qG�?Tz�A@Q�B�aH?c�
@h��?��\A��B�z�                                    BxW^"  
�          @\)?u@r�\?@  A-��B�� ?u@j�H?�Q�A�B���                                    BxW^-�  
�          @w
=?��@j�H>���@�
=B�.?��@e?Tz�AH  B��{                                    BxW^<n  T          @y��?�=q@p  =��
?��HB���?�=q@mp�?�@�{B��                                    BxW^K  
�          @w�?��@l�ͽL�Ϳ8Q�B��?��@k�>\@�ffB��                                    BxW^Y�  �          @u�?}p�@l(��u�s33B��H?}p�@j�H>�Q�@�
=B��q                                    BxW^h`  T          @qG�?L��@h��>�@��B�33?L��@c�
?aG�AX(�B��                                    BxW^w  �          @g
=?��R@N�R��{��B�aH?��R@O\)<�?�B���                                    BxW^��  �          @^�R?�z�@C�
���H��B�aH?�z�@E�\)���B��f                                    BxW^�R  	�          @P��?n{@@�׿
=�*{B��?n{@C�
��  ��ffB�                                    BxW^��  �          @QG�>�p�@J�H���!p�B���>�p�@Mp��W
=�n{B���                                    BxW^��  "          @U>��R@R�\�aG��vffB�\)>��R@S33>\)@\)B�\)                                    BxW^�D  
�          @Z�H?�p�@"�\���
��z�BP33?�p�@(�ÿ@  �L��BTff                                    BxW^��  
�          @b�\?�\)@:�H��33��=qBt(�?�\)@A녿Q��XQ�Bw�                                    BxW^ݐ  	�          @hQ�?���@K��0���/\)B|�H?���@N�R��{��z�B~z�                                    BxW^�6  
�          @j�H?�ff@P�׿
=�p�B�8R?�ff@S�
�u�p��B���                                    BxW^��  
�          @g
=?���@W
=>�33@�G�B��3?���@S33?333A4��B�{                                    BxW_	�  �          @u�?��H@aG�>���@�=qB���?��H@^{?.{A"�HB�33                                    BxW_(  
�          @mp�?�  @N{>�(�@׮Bv
=?�  @J=q?E�A@��Bt�                                    BxW_&�  �          @p��?�z�@\�;�33��p�B��3?�z�@^{<#�
>�B��                                    BxW_5t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW_D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW_R�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW_af              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW_p  
e          @���?��
@mp�?��HA��
B���?��
@dz�?�=qA�{B���                                    BxW_~�  �          @q�?�
=@`��?+�A$z�B��f?�
=@[�?��\Az�HB�\                                    BxW_�X  "          @��?�z�@hQ�?�=qAw�B���?�z�@`��?�
=A��
B�B�                                    BxW_��  
�          @���?�  @^�R?���A�Q�B�G�?�  @U�?�(�A�p�B�aH                                    BxW_��  "          @k�?�G�@O\)?\(�AYB�?�G�@H��?�A��B���                                    BxW_�J  
�          @hQ�?z�H@S33?�  A�\)B��)?z�H@K�?��A���B�Ǯ                                    BxW_��  
(          @p  ?W
=@Z=q?���A��\B�33?W
=@QG�?��A���B�
=                                    BxW_֖  T          @o\)?+�@P  ?�(�Aۙ�B��R?+�@Dz�@G�B
=B�k�                                    BxW_�<  �          @p��?!G�@Tz�?�z�A�\)B�Q�?!G�@I��?��HA��B�.                                    BxW_��  
(          @g�>�33@J�H?�
=A�Q�B�ff>�33@@  ?�(�B\)B��3                                    BxW`�  
(          @mp�>Ǯ@S33?�{A�z�B���>Ǯ@H��?�z�A�z�B��                                    BxW`.  �          @l��>��@W
=?��RA�33B��R>��@Mp�?��A�
=B�z�                                    BxW`�  �          @fff>L��@E�?���A��
B��=>L��@9��@�B��B�{                                    BxW`.z  �          @`  >�@&ff@��B�B�>�@��@�HB3Q�B�Q�                                    BxW`=   "          @#�
��G�@=q>\A�BȀ ��G�@�?
=A]G�B��)                                    BxW`K�  �          @#33�J=q@�\��
B�k��J=q@
=�.{�}p�B�\                                    BxW`Zl  
�          ?��H��p�?�=q��G��S33C���p�?�=q�#�
��Q�C�H                                    BxW`i  
Z          @(�����?��>�(�AY�C5ÿ���?�  ?�A�p�CJ=                                    BxW`w�  
�          @.�R�
�H>u?�\)B�\C-�
�H=���?��B
=C1k�                                    BxW`�^  �          @.{�  ?8Q�?�=qA�G�C".�  ?(�?��A�=qC$�H                                    BxW`�  
�          @(Q���?c�
?���A�z�C�\��?G�?��B ffC��                                    BxW`��  T          @%��
?(�?�B  C#}q��
>��H?�(�B
�C&�)                                    BxW`�P  �          @%��?(��?�Q�B
=C!�q��?
=q?��RBffC%�                                    BxW`��  "          @,����
?�?���B=qC$����
>�G�?��B�C'�R                                    BxW`Ϝ  �          @*�H�33?��\?���A�
=C��33?h��?�BffC�                                    BxW`�B  �          @#33�?s33?��Aʏ\Cu��?\(�?�z�A��C��                                    BxW`��  
�          @#�
��R?Q�?p��A�{C�{��R?=p�?�  A�{C!�{                                    BxW`��  "          @"�\�
�H?!G�?���A�Q�C#�
�H?
=q?�AݮC%��                                    BxWa
4  T          @0  �{>��
?�ffB
�C+� �{>L��?���BC.�{                                   BxWa�  "          @,�Ϳ��?���������C�Ϳ��?�
=��G���C��                                   BxWa'�  T          @#�
��Q�?n{�����\CLͿ�Q�?��
������p�C��                                   BxWa6&  
�          @\)��\?�  ��  ��{C����\?��ÿk����\C33                                   BxWaD�  �          @#33��
?�(��^�R���HCE��
?��
�E���33C
=                                   BxWaSr  �          @(����?^�R�8Q���(�C����?k��&ff�z�\CǮ                                   BxWab  
Z          @!G�����?z�Af�HCA�R����?
=qAV�\CB�R                                   BxWap�  "          @6ff��Ϳ���?
=qA.=qCP5���Ϳ���>�G�A�HCP�
                                   BxWad  
�          @,���ff��\)?��A=p�CM���ff��z�>�A!�CNL�                                   BxWa�
  �          @1G��G�������Q���HCT�)�G���Q�B�\���
CTn                                   BxWa��  �          @.{�
=��ff�#�
�\(�CP�
�
=������1G�CP��                                   BxWa�V  �          @,(��(������\)���CK� �(������33��RCK{                                   BxWa��  �          @+��Q쿁G���ff�G�CK��Q�z�H���5�CJY�                                   BxWaȢ  �          @#�
��\�z�H��p��CK���\�s33��(��G�CJz�                                   BxWa�H  
�          @(Q���R���ͽu��=qCS33��R�������Q�CS�                                   BxWa��  
�          @,�Ϳ�p���=q���1G�C^���p����þ�  ����C^�H                                   BxWa��  "          @&ff��J=q��R�`��CF�
��@  �+��s\)CE�                                   BxWb:  �          @1���R��{���Ǚ�C;޸��R��=q��Q����C:.                                   BxWb�  
�          @0�����#�
��p���RC4&f��=��Ϳ�p��ffC1:�                                   BxWb �  T          @7
=�z�>aG�����(��C.��z�>��ÿ�\)�&��C*޸                                   BxWb/,  "          @4z��
�H>\�޸R�G�C*.�
�H>���(����C'�                                    BxWb=�  �          @-p��{�\)��{��Q�CB��{��������\C@
                                   BxWbLx  T          @/\)�33�c�
������p�CI)�33�O\)��  ���HCG�                                   BxWb[  
�          @4z��   �\)��\)���
C@���   �   ��z���\)C?8R                                   BxWbi�  T          @5�%�>�zῊ=q��(�C-�=�%�>�33������HC,33                                   BxWbxj  
�          @5�(Q�>�=q���\��33C.��(Q�>��ÿ�G���=qC,��                                    BxWb�  �          @333�*�H>���Q����HC0�\�*�H>B�\�O\)��33C/�{                                    BxWb��  	�          @1G��*�H�k��+��_�C8�3�*�H�B�\�.{�c�C8(�                                    BxWb�\            @/\)�"�\�:�H�.{�f=qCD�"�\�0�׿8Q��s�CC8R                                   BxWb�   n          @4z��!G����������(�CAn�!G��
=q��������C@!H                                    BxWb��   �          @.{�
=��p����\���
C<��
=���������G�C;T{                                    BxWb�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWb�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWb�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWc
�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWc�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWc(2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWc6�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWcE~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWcT$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWcb�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWcqp              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWc�  �          @5��/\)�Ǯ����2�\C<��/\)��p��\)�7�
C;�)                                    BxWc��  T          @8Q��3�
��=q�z��;\)C9n�3�
�u����>�RC8�                                    BxWc�b  T          @<(��333�
=q�!G��G
=C>��333��\�&ff�N{C>ff                                    BxWc�  �          @;���H��\������HC@��H��ff�˅��C>��                                    BxWc��  
�          @;���
�fff������CI@ ��
�W
=�����CG��                                    BxWc�T  "          @:�H�
�H�aG���p��(�CJ&f�
�H�O\)��\�ffCH��                                    BxWc��  �          @@�����+���
=� �\CD�R���
=�����"��CC0�                                    BxWc�  T          @>�R������z��N(�CB������\��P{C?�f                                    BxWc�F  
�          @;���(���ff�ff�8\)C@Ǯ��(���p����:{C>��                                    BxWd�  �          @;��\)�#�
��\)��C8��\)��Q��\)��\C6^�                                    BxWd�  
�          @8Q��>�\)����F�
C+Q��>�Q��(��Ez�C)�                                    BxWd!8  �          @9���Q�>����=q��\C-� �Q�>��R�������C,z�                                    BxWd/�  �          @6ff�(�<��
��
=��ffC3�H�(�=u����(�C2�                                    BxWd>�  �          @:=q�*=q��\)��z���G�C5�
�*=q�#�
������C4�                                    BxWdM*  	�          @5�!G�>�����  �ѮC-33�!G�>�{���R���
C,Q�                                    BxWd[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWdjv              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWdy              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWd��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWd�h  
2          @Fff�'�>W
=�У�����C/}q�'�>�  �У����
C.�                                     BxWd�  "          @Dz��#33���Ϳ���� �RC6.�#33�L�Ϳ�{� �C533                                    BxWd��  �          @<(���׿&ff���
�ffCD���׿(���ff�
=CC#�                                    BxWd�Z  �          @@�׿aG���(����XffCo{�aG���z��=q�]ffCn                                      BxWd�   	�          @AG��s33���
�   �f�CiQ�s33���H�!��j��Cg�                                    BxWdߦ  �          @@  �aG�����(Q��y  Cf@ �aG���  �)���}33Cd}q                                    BxWd�L  
Z          @@�׿Tz�
=�333��CWLͿTz���3�
u�CT+�                                    BxWd��  �          @:=q��33�:�H�
=�E�
CK�f��33�.{�Q��G�HCJ}q                                    BxWe�  �          @=p���
>����G���C-���
>�z��  ��C,�                                     BxWe>  
�          @<�Ϳ�p��\�
=�9p�C>�\��p��������:G�C=��                                    BxWe(�  �          @=p��p���Q���H�&�C6p��p��L�Ϳ��H�&G�C5h�                                    BxWe7�  T          @B�\��>�=q�����C-����>�����\)�(�C,�R                                    BxWeF0  
(          @<���
�H>�33���H�'�C*��
�H>\�����&ffC)��                                    BxWeT�  �          @=p��33?���G���C')�33?\)��  ��\C&\)                                    BxWec|  �          @C�
��>����
�
=C)c���>��H���
�G�C(��                                    BxWer"  
�          @@����?@  ���\)C"W
��?G���33�33C!�q                                    BxWe��  "          @;���?z��\)�(�C&L���?����{�G�C%��                                    BxWe�n  T          @5���?#�
��p���G�C%E��?(�ÿ�p���p�C$ٚ                                    BxWe�  �          @<(��Q�?�Q���((�C	�
�Q�?�����\�
=C	z�                                    BxWe��  "          @,(���  ?�p��L������Cc׿�  ?��R�.{�l(�CY�                                    BxWe�`  
�          @7����@G�����C�C&f���@녿��:�RC�                                    BxWe�  �          @=p��z�?�녿fff��
=C	�R�z�?�33�aG���33C	�\                                    BxWeج  
�          @;��   ?�=q������(�C	���   ?��������\C	Y�                                    BxWe�R  �          @7����R?��(���V=qC�q���R?�
=�#�
�O\)C��                                    BxWe��  T          @/\)��p�?��;���P��C�׿�p�?��;\)�7
=C޸                                    BxWf�  �          @0�׿��?��R>���@ʏ\C\)���?��R>��
@�\)Ch�                                    BxWfD  
�          @0  �	��?�=���@
�HC0��	��?�=�@�RC5�                                    BxWf!�  "          @.�R�
=q?У�>L��@�C  �
=q?У�>W
=@��RC�                                    BxWf0�  T          @1G��=q?���=u?�ffCT{�=q?���=�\)?�  CW
                                    BxWf?6  T          @6ff�"�\?h�ÿ!G��Qp�C E�"�\?k���R�O\)C 0�                                    BxWfM�  
�          @9������>u�
�H�x{C'�H����>�  �
�H�w��C'B�                                    BxWf\�  
�          @]p���G�?
=q�Y��¡��B�  ��G�?\)�Y��¡
=B�Q�                                    BxWfk(  "          @]p�����?\)�Z=q¡(�B�녽���?��Z=q ��B�ff                                    BxWfy�  �          @]p���=q?�  �R�\
=BҨ���=q?�G��Q���B�G�                                    BxWf�t  �          @\�Ϳ��?z�H�N�R��B�=q���?}p��N�R��B�3                                    BxWf�  �          @\(���Q�?�{�N{B׮��Q�?�\)�Mp�B�p�                                    BxWf��  
Z          @P  �=p�?�{�\)�L��Bߊ=�=p�?�{��R�L=qB�p�                                    BxWf�f  
�          @G
=��@�ÿ�Q��G�B�aH��@�ÿ�Q��{B�W
                                    BxWf�  T          @G
=��p�@���ff��\)C�Ϳ�p�@���ff��33C�=                                    BxWfѲ  
�          @K��{?�ff�����z�C��{?�ff����ȏ\C�3                                    BxWf�X  
�          @@�׿�33@   ������z�CxR��33@   ��������C}q                                    BxWf��  
�          @G
=���@�
��
=��\)C�R���@�
��
=���C�                                    BxWf��  
�          @B�\���?�녿�Q���G�C�
���?�׿�Q���  C��                                    BxWgJ  
�          @C�
����?У׿�Q��(�C{����?У׿�Q����C(�                                    BxWg�  "          @J�H��p�?ٙ����
�	33CW
��p�?ٙ�����	Cs3                                    BxWg)�  �          @O\)���?�  ��=q�
=qC����?��R���
��C��                                    BxWg8<  "          @Q녿��R?������&��C s3���R?�\)�	���'�HC �)                                    BxWgF�  	�          @O\)��?��\)�1{C )��?�ff�  �2{C O\                                    BxWgU�  �          @G����\?�33��R�;��B�녿��\?���\)�<��B�ff                                    BxWgd.  
�          @Fff��  ?�p���Q��0(�B����  ?�(������1�B��                                     BxWgr�  "          @Mp���p�@���33�:��B�z᾽p�@��z��<\)Bǣ�                                    BxWg�z  �          @J=q����?�=q��\�;G�B�#׿���?�����
�<�HB�                                    BxWg�   "          @Mp��#�
?�ff�#33�T��B�녿#�
?��
�$z��V�\B�aH                                    BxWg��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWg�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWg�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWgʸ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWg�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWg�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWg��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWhP              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh"�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh1B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWhN�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh]4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWhk�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWhz�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh�r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWhþ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWh�V              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi*H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi8�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWiG�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWiV:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWid�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWis�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi�,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi�x              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi�j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWi�\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj#N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj1�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWjO@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWjl�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj{2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj�~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj�$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj�p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj�b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWj�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWkT              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk*�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk9�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWkHF              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWkV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWke�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWkt8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk�*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk�v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk�h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWk�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWl�  
�          @=p��\)@8��>��
@�ffB�녿\)@9��>.{@R�\B��
                                    BxWlZ  �          @(��>aG�@33?5A�  B��
>aG�@�?
=A`Q�B�                                      BxWl$   �          @(Q�>k�@�R?=p�A��HB�z�>k�@ ��?(�AYB���                                    BxWl2�  
�          @@  >�{@8Q�?=p�Ad  B���>�{@:=q?z�A3�B�                                    BxWlAL  "          @L��?+�@@��?J=qAdz�B��?+�@C33?�RA4��B�=q                                    BxWlO�  �          @E�?J=q@<��>�=q@��B��?J=q@=p�=�G�@   B�.                                    BxWl^�  �          @@��>��@=p�=L��?c�
B��>��@=p���G���B�{                                    BxWlm>  "          @>{>�
=@:=q���
��  B�� >�
=@8�þ��B�\)                                    BxWl{�  �          @Dz�>�33@?\)�   �  B�>�33@=p��+��J=qB���                                    BxWl��  
�          @A녿c�
@(�?
=qA<��B�33�c�
@{>��A�B���                                    BxWl�0  "          @8Q��ff?�z�?�z�A�p�C=q��ff?�(�?��A���CQ�                                    BxWl��  
Z          @<(�����@�>�
=A  B�����@��>�\)@�G�B�Q�                                    BxWl�|  "          @7
=��p�@=q>���@�
=B��´p�@�>��@�{B�{                                    BxWl�"  �          @7����@��>�33@��B�8R���@�H>W
=@���B��H                                    BxWl��  �          @C33��(�@(Q�>���@ʏ\B�p���(�@(��>8Q�@\��B�(�                                    BxWl�n  
�          @S33�\@9��>��@�  B�LͿ\@:=q=�Q�?�ffB�#�                                    BxWl�  �          @^{��ff@E>#�
@.�RB����ff@Fff���
��=qB�\                                    BxWl��  "          @a녿˅@J=q<#�
>8Q�B�\)�˅@I���.{�4z�B�k�                                    BxWm`            @P  ��G�@&ff>�=q@�z�B�Q��G�@'
==�G�@�B�{                                    BxWm  
�          @AG���Q�@�>�Q�@�z�C� ��Q�@�\>k�@���CJ=                                    BxWm+�  ]          @H�ÿ�ff@#33�#�
��  B�\)��ff@"�\�.{�C33B�z�                                    BxWm:R  �          @U����@:�H>�33@�
=B�����@<(�>.{@;�B�B�                                    BxWmH�  
Z          @Y�����\@@�׿n{�\)B�녿��\@<�Ϳ�{��33B�Ǯ                                    BxWmW�  �          @Z=q��  @G
=����"�\B߸R��  @Dz�J=q�V=qB�G�                                    BxWmfD  
�          @\(����@G����H�z�B�\���@E�.{�8  B�\                                    BxWmt�  S          @Tzῥ�@@�׾����RB�Q쿥�@>�R�(���6�RB���                                    BxWm��  �          @U��Q�@3�
�!G��-p�B�
=��Q�@1G��L���^{B��
                                    BxWm�6  �          @W
=��p�@,�Ϳu��p�B�#׿�p�@(�ÿ�����p�B�k�                                    BxWm��  �          @Q녿�z�@'���������B��
��z�@#33��p�����B�L�                                    BxWm��  "          @L�Ϳ޸R@%��@  �Yp�B�
=�޸R@!G��h������B�#�                                    BxWm�(  �          @H�ÿ���@(��5�P��B������@�ÿ\(��~�HB�B�                                    BxWm��  �          @P  �33@��!G��2�RC#��33@�ÿG��_33C��                                    BxWm�t  "          @R�\��\@!G��
=q��
C ���\@�R�333�Ep�Cc�                                    BxWm�  �          @J�H��{@!G���ff��HB��f��{@�R�(��2=qB���                                    BxWm��  �          @L�Ϳ޸R@'��\)�"�RB�k��޸R@$z�:�H�T  B�G�                                    BxWnf  �          @K���\)@&ff�Q��p��B���\)@"�\�}p�����B�#�                                    BxWn  �          @I����ff@,(����z�B����ff@)���333�LQ�B�q                                    BxWn$�  "          @H�ÿ�z�@(Q쾞�R��p�B��׿�z�@'
=�����B�(�                                    BxWn3X  �          @H�ÿ�{@+��aG����B���{@*=q�����陚B�ff                                    BxWnA�  T          @L�Ϳ�@%��\��G�B����@#33����=qB��q                                    BxWnP�  T          @Q��{@��#�
�4(�C�
�{@녿J=q�`(�C.                                    BxWn_J  �          @c33�{@/\)��������C
�{@,�Ϳz���Ck�                                    BxWnm�  �          @b�\��@*=q���C�
��@'��0���4z�C�                                    BxWn|�  �          @^�R��@"�\�   �Q�C}q��@   �+��1C�                                    BxWn�<  �          @^�R�\)@'
=�
=q�33C�)�\)@$z�8Q��>=qC{                                    BxWn��  "          @\(��+�@�����\C{�+�@�\�0���8��C�f                                    BxWn��  �          @U��{@
�H�   �
�HC
���{@Q�&ff�4  C33                                    BxWn�.  �          @`  ��
@#�
���Q�C���
@!G��0���6�HC�                                    BxWn��  
�          @_\)��@0�׿   �z�B����@.{�0���6�HB���                                    BxWn�z  
�          @\�Ϳ��@3�
�+��4z�B��3���@0  �^�R�h��B��R                                    BxWn�   �          @S�
����@333�J=q�^�\B�z����@/\)�}p���
=B�{                                    BxWn��  "          @S33��ff@6ff��R�.�\B�{��ff@333�Q��f�\B��                                    BxWo l  �          @N{��33@*�H�0���G33B�z��33@'��aG��}��B�=                                    BxWo  
�          @J�H�Ǯ@(�ÿTz��s
=B���Ǯ@$z῁G����B�\)                                    BxWo�  �          @@�׿�  @$z�Ǯ����B�k���  @"�\���0(�B�{                                    BxWo,^  �          @=p����@�>��
@�=qB��ÿ��@��>��@=p�B���                                    BxWo;  "          @3�
��=q@녽L�Ϳ���B�k���=q@녾aG���{B���                                    BxWoI�  �          @5���=q@�����
���HB��Ϳ�=q@���B�\���B���                                    BxWoXP  
�          @:=q��(�@Q�?5Ac�B�8R��(�@�?
=qA+�B�G�                                    BxWof�  �          @7��Ǯ?���?��A��C �3�Ǯ@�?�33A�G�B��                                    BxWou�  4          @2�\��?�33?k�A��\C� ��?�(�?G�A�z�C�
                                    BxWo�B  T          @4z��p�@�?�RAK�
C:��p�@
=q>�Az�C �R                                    BxWo��  
�          @4z���?�?
=AG33CB���?��H>��Ap�C�H                                    BxWo��  �          @<(��
=?�p��L���w�C�
=?��H���
�ʏ\C\                                    BxWo�4  �          @AG����@G���=q���HC	}q���?��R������C	ٚ                                    BxWo��  
R          @<�Ϳ��R@
=q��  ��ffC�����R@Q���� z�C                                    BxWò  �          @>�R��@p��B�\�g�C����@(���33��G�C+�                                    BxWo�&  T          @@  ��33@�\�u��G�C�3��33@G���������C                                      BxWo��  �          @@�׿��H@�׾�\)��33C���H@�R��ff�z�CG�                                    BxWo�r  "          @A녿�Q�@�
�k����\C�R��Q�@�\������=qC@                                     BxWp  
�          @B�\��@�������\)C�{��@	�����H���C��                                    BxWp�  �          @@  ��@ff��G��(�C {��@���z���G�C B�                                    BxWp%d  T          @A녿�ff@z���0��B��3��ff@G��@  �f�RC s3                                    BxWp4
  T          @AG��Ǯ@녿������
B�Ǯ�Ǯ@���\)��  B��                                    BxWpB�  �          @E���@&ff����%�B�ff���@#33�=p��`(�B�W
                                    BxWpQV  �          @;��^�R@)���O\)��(�B�W
�^�R@%���G����HB�=q                                    BxWp_�  �          @<�Ϳ(��@/\)�Y�����B���(��@*�H��ff���BϮ                                    BxWpn�  �          @L�ͿG�@>�R���p�B�aH�G�@;��B�\�`Q�B��
                                    BxWp}H  �          @Z=q����@L(��xQ���z�B�L;���@G
=���H����B®                                    BxWp��  T          @W���z�@C33�����33B�녾�z�@<(��������
B�Q�                                    BxWp��  �          @\(��#�
@Dz���
��{B��׽#�
@<(���G�����B��R                                    BxWp�:  
�          @`��<#�
@J=q�\���
B��\<#�
@B�\��G���RB��=                                    BxWp��  �          @_\)��\)@G��Ǯ��Q�B�k���\)@?\)�����\)B��=                                    BxWpƆ  T          @Y��<�@I��������B���<�@B�\�\����B�                                    BxWp�,  T          @Z�H�\)@H�ÿ��\��ffB�{�\)@B�\��G��ӅB�B�                                    BxWp��  
�          @]p��Ǯ@N{���H��{B��
�Ǯ@G
=���H���B�L�                                    BxWp�x  
Z          @a녽L��@Q녿��
��33B��f�L��@J�H���
��z�B���                                    BxWq  �          @X�ý�G�@P�׾\���
B����G�@N{�!G��0��B��)                                    BxWq�  	�          @^{��p�@Y����Q쿺�HB����p�@XQ쾳33���
B��{                                    BxWqj  "          @e�>�33@R�\��=q���B��f>�33@K��˅����B�z�                                    BxWq-  
�          @c�
=u@O\)���H��(�B��=u@G
=���H��B���                                    BxWq;�  "          @c�
>�{@?\)��\)���HB��>�{@5��ff��B�=q                                    BxWqJ\  	.          @Y���333@Dz῀  ���B���333@>�R���R��Q�BΊ=                                    BxWqY  
Z          @[��B�\@J�H�����G�B�녿B�\@E����
��{BϮ                                    BxWqg�  
(          @b�\�u@Q녿�ff��Q�B�L;u@J�H��ff��  B��{                                    BxWqvN  
�          @c�
�W
=@P  ��
=��Q�B�k��W
=@HQ��
=��{B��R                                    BxWq��  
(          @`��>��@Fff��G����B�W
>��@>{��  ��B��                                    BxWq��  
�          @[�>��@@  ��ff�؏\B�33>��@7����
��{B�ff                                    BxWq�@  �          @\��>�
=@@�׿�\)���B�
=>�
=@7��������B�G�                                    BxWq��  
�          @@��?(�@%��33��=qB�p�?(�@�R����� ��B�8R                                    BxWq��  
Z          @Fff>��@1녿��
�Ù�B��>��@*�H��  ��33B�.                                    BxWq�2  T          @G
=>��H@0�׿�=q��33B��f>��H@)���������B�\                                    BxWq��  L          @.�R?�\@����
=��p�B��?�\@33��\)����B��                                    BxWq�~  
�          @6ff>\@#33��Q���Q�B�  >\@�Ϳ����(�B�L�                                    BxWq�$  �          @2�\>��R@{���H�ϮB���>��R@
=��33��B���                                    BxWr�  
�          @*�H=��
@�H��=q��{B�p�=��
@zῡG���ffB�L�                                    BxWrp  
�          @<��>��@ �׿�(���p�B���>��@�ÿ��	�\B��q                                    BxWr&  
�          @:=q>��R@$zῨ���ׅB�33>��R@p��\���B��{                                    BxWr4�  T          @:�H>�@'����\�υB�#�>�@ �׿�p����B��f                                    BxWrCb  �          @O\)�\)@>�R���\��
=B�� �\)@7���G��݅B��R                                    BxWrR  
�          @R�\��G�@;���(�����B�(���G�@3�
�ٙ���G�B�W
                                    BxWr`�  
�          @Fff<��
@4zῘQ�����B�<��
@.{��z���33B���                                    BxWroT  
�          @@  ��=q@0�׿�ff����B�=q��=q@*=q��G���  B���                                    BxWr}�  
�          @Dz�n{@$z`\)��z�Bۊ=�n{@p���������B�G�                                    BxWr��  
�          @H�ÿp��@*=q������\B��)�p��@#�
��  ��
=B�#�                                    BxWr�F  T          @J=q���R@>�R�=p��^�RB��)���R@:�H�z�H���
B�#�                                    BxWr��  "          @:=q����@   ����/�
B��
����@�Ϳ@  �p  B��)                                    BxWr��  �          @;����R@��h����{B�.���R@  �������HB�
=                                    BxWr�8  T          @B�\��
=@G������G�B�{��
=@
�H���\�Ə\B�p�                                    BxWr��  
R          @>{�˅@���(����B�(��˅@z῱����\B��H                                    BxWr�  �          @@�׿�R@0  �z�H���B�z��R@*�H��������B�G�                                    BxWr�*  
�          @B�\�
=@+����
����B��)�
=@#�
���R���B��                                    BxWs�  
�          @AG��Ǯ@*�H���\����BĊ=�Ǯ@#�
��p���\)B�8R                                    BxWsv  
�          @?\)��@3�
�G��u�B�  ��@/\)��G����RBɊ=                                    BxWs  
�          @=p��J=q@'���{��{B�k��J=q@!G������G�B֣�                                    BxWs-�  	_          @;��E�@ �׿��\�Ώ\B�\�E�@����(���B׊=                                    BxWs<h  �          @5��&ff@���p���
=B�녿&ff@z῵��\B�8R                                    BxWsK  
�          @)���u@���ff��RB�uÿu?�(����H�  B��H                                    BxWsY�  
�          @6ff��\)@G���33��G�B�uÿ�\)?�33�Ǯ��RB��q                                    BxWshZ  
Z          @8Q쿵?��H��z����HB���?��Ǯ�33B�33                                    BxWsw   �          @@  ��ff?��H��33���C Q��ff?��ÿ��33CaH                                    BxWs��  	�          @A녿�@
=q��ff��B��Ϳ�@G���(���B�(�                                    BxWs�L  
�          @C�
�}p�@p���  ��(�B߳3�}p�@�ٙ���B��H                                    BxWs��  
�          @H�ÿxQ�@%���p����B�LͿxQ�@�Ϳ�
=�(�B�=q                                    BxWs��  
�          @HQ�0��@,�Ϳ�����ffBД{�0��@%��˅��  B��
                                    BxWs�>  
S          @I���u@1녿��H�݅B�k��u@*=q��
=�33B��                                    BxWs��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWs݊              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWs�0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWs��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWt	|  
�          @K����
@+���
=��
=B��;��
@!녿���B{                                    BxWt"  T          @E���
@'���=q����B����
@\)�����RB                                     BxWt&�  �          @C�
��R@(�ÿ�33����B�ff��R@!G���{���\BϞ�                                    BxWt5n  
�          @?\)���@"�\��(���z�B�uÿ��@�H��z��{B�Ǯ                                    BxWtD  T          @>{��@�ÿ�z����B��Ϳ�@  �����z�B�ff                                    BxWtR�  �          @C�
���@p���(��	  B�����@z��z���
BΙ�                                    BxWta`  "          @G
=�333@+���33�ָRB�33�333@#�
��{��{B҅                                    BxWtp  �          @N�R�Ǯ@%��{�z�B�
=�Ǯ@���
��\B��                                    BxWt~�  
�          @N�R�B�\@ff�33�$  B�\)�B�\@���R�6=qB��                                    BxWt�R  
�          @G���@ �׿�\�
Q�B��;�@����H�=qB�#�                                    BxWt��  
�          @N�R����@*=q���
�=qB��þ���@ �׿��R�Q�B���                                    BxWt��  
�          @K���G�@!녿�{��B���G�@���
�!�HB�                                    BxWt�D  
�          @5��z�@z��G��ffB�Q�z�@(���Q��
=B��f                                    BxWt��  �          @@�׼��
@��� ���)p�B��
���
@����;�B��                                    BxWt֐  "          @AG���@�H�޸R��RBʙ���@G�����\B�                                      BxWt�6  
�          @G����
@'
=������HB�G����
@!G����R��p�B��f                                    BxWt��  �          @C33����@$z�k���G�B�\����@\)��\)��\)B��                                    BxWu�  �          @HQ쿣�
@,�Ϳ^�R��G�B�׿��
@'���=q��  B��                                    BxWu(  T          @B�\��p�@(Q��G��Q�B��쿽p�@'
=���
�\B�#�                                    BxWu�  �          @C33�
=q@\)��33�z�B̊=�
=q@
=���(�B��                                    BxWu.t  "          @E����@{��(��{B�W
���@���33���B��                                    BxWu=  
�          @2�\�.{@33��Q���(�B��ÿ.{@���\)�\)B֮                                    BxWuK�  �          @>{=L��@$zΌ����B��
=L��@�Ϳ�33��B��q                                    BxWuZf  
�          @.{���@z΅{��RB�33���@�Ϳ���\)B��{                                    BxWui  "          @<(���@�
��ff��HB�ff��@
=q��p��)�HB��
                                    BxWuw�  �          @9��>��@����H�*��B��R>��?��H���<�RB�
=                                    BxWu�X  T          @$z�?
=q?�Q��=q�7=qB�?
=q?�������HG�B���                                    BxWu��  
�          @��?.{?��ÿ�\)�N
=Bz
=?.{?����H�]�RBnG�                                    BxWu��  �          @��?O\)?�(���\)�PffBb
=?O\)?��ÿ��H�_  BS�
                                    BxWu�J  �          @*=q?���?�ff��
�K�BI?���?����	���X�B:\)                                    BxWu��  �          @�R?��?����
=�L�
B4
=?��?p��� ���Y  B"�
                                    BxWuϖ  �          @  ?��\?Tz��=q�YffBG�?��\?.{����c��BG�                                    BxWu�<  
�          ?�G�?=p�?5��z��W�HB.�?=p�?�����H�c��B{                                    BxWu��  �          ?�{?=p�?E������A��B9{?=p�?+���G��N
=B)��                                    BxWu��  
�          ?�?^�R?.{�����MG�B33?^�R?녿�
=�W�\BQ�                                    BxWv
.  �          @p�?�G�>�����p��T��AX��?�G�>.{�޸R�W�
@��                                    BxWv�  �          @(��?��
    ��
=�<�H=��
?��
������<33C���                                    BxWv'z  
�          @\)?�zᾀ  ���<��C�?�z�Ǯ����9\)C�u�                                    BxWv6   �          @5@��0�׿�����C��H@��Q녿�p��  C�<)                                    BxWvD�  
�          @8��@ff�k���  �p�C�&f@ff��ff��
=���C��\                                    BxWvSl  
�          @5@Q쿀  �����	{C�ff@Q쿏\)�\�=qC�q                                    BxWvb  �          @<��@���ÿ��H��33C�E@��zῌ������C�w
                                    BxWvp�  
�          @G�?�Q�(����\��\C�q?�Q�.{�xQ���(�C���                                    BxWv^  �          ?Ǯ�#�
?��\�
=��ffB�3�#�
?�p��0����G�B�                                    BxWv�  	�          @p��Y��?�  ���
��z�B��Y��?���z�����B��H                                    BxWv��  
�          @$z�L��?�׿�ff�ffB��
�L��?�G���
=�$G�B��                                    BxWv�P  
�          @,�Ϳk�@   ����
=B�=q�k�?�׿�Q���B�                                    BxWv��  
Z          @0  ���\@���(��
=B�aH���\?�(���\)�p�B�                                    BxWvȜ  
�          @.{��=q?�ff��(���B��)��=q?������,�B��)                                    BxWv�B  �          @5���Q�?�Q��z��p�B�{��Q�?���ff��B��                                    BxWv��  K          @333��
=?��H�z��?Q�C{��
=?�ff���LG�CW
                                    BxWv�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw�   2          @C�
��Q�?�������33CW
��Q�?�33�����G�C��                                    BxWw �              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw/&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw=�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWwLr              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw[              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWwi�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWwxd              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw�V              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw�H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWw�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx
�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx(,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx6�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWxEx              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWxT              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWxb�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWxqj              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx�\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx��  �          @*�H�   ?�������\)C���   ?��R����C8R                                    BxWx�  
�          @,(��G�?��H>L��@��C�=�G�?�(�=�Q�?���C��                                    BxWx�@  
�          @*�H�
=q?\>W
=@��
C���
=q?��
=�G�@(�C�3                                    BxWy�  T          @,����?��>\A(�C����?�\)>���@��HC�                                    BxWy�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWy!2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWy/�  	�          @333��{?�p���
=��\)C	\��{?�33�����G�C
h�                                    BxWy>~  
�          @5��ff?��
��G���
C�q��ff?�
=�У���HC��                                    BxWyM$  
(          @.�R���@�<#�
>B�\B��3���@
�H���3�
B���                                    BxWy[�  "          @(Q��33?�Q�}p���{C����33?�\)��=q��p�C0�                                    BxWyjp  �          @)���ff?�����33��Q�Cu��ff?���G����C�                                    BxWyy  
Z          @,���	��?�=q���R��ffC�3�	��?Ǯ����G�C�                                    BxWy��  	�          @*�H��?��׿��4z�C��?��Ϳ
=�Mp�C��                                    BxWy�b  
Z          @(Q��p�?G��\���C"n�p�?@  ��(���RC"��                                    BxWy�            @%� ��>\)�\)�G�
C0Ǯ� ��=�G��\)�J�RC1��                                    BxWy��  
`          @ ���=q>������Ap�C,�f�=q>�=q�
=q�H(�C-��                                    BxWy�T  T          @   �ff>��G�����C0�
�ff=��
�J=q���C2                                    BxWy��  �          @#�
�{�=p��p�����CF^��{�J=q�fff����CG��                                    BxWyߠ  "          @"�\�33>�=q�aG���33C-T{�33>W
=�fff��(�C.��                                    BxWy�F  
�          @&ff�\)��=q����X��C:G��\)���R�
=�R=qC;
                                    BxWy��  
(          @!G��33��R�B�\���CC��33�(�ÿ8Q���z�CD�                                    BxWz�  
�          @*�H��ͽ����H�33C7���;W
=�����33C9T{                                    BxWz8  �          @*�H�
�H�#�
�\��RC85��
�H��  ��G��
p�C:�\                                    BxWz(�  N          @'���R��=q������
=C:���R��33��ff��G�C<�H                                    BxWz7�  
.          @#33�
=�Ǯ��������C>ff�
=����ff��=qC@k�                                    BxWzF*  �          @%�	�������R��ffCA��	���
=���H��CCO\                                    BxWzT�  
�          @)���33=�\)��\)�ϙ�C28R�33<#�
��\)��  C3�
                                    BxWzcv  T          @0  ����?�33�\�Cz����?����=q�=qCz�                                    BxWzr  T          @*�H��p�?���W
=���RC\��p�?��R�n{���RC�q                                    BxWz��  
�          @*�H� ��?���
=q�=p�C&��� ��?녿��IG�C'J=                                    BxWz�h  �          @,(��p�?k���G��Q�C}q�p�?c�
���H�*=qC �                                    BxWz�  "          @-p��Q�>�Q쿜(���z�C+u��Q�>�z῞�R���C-
                                    BxWz��  	�          @,���(��W
=���
��C9�
�(�������G��	��C;Ǯ                                    BxWz�Z  
Z          @0  �G�>aG���  �  C.z��G�>\)��G����C0��                                    BxWz�   T          @+��{?   �^�R��z�C(�\�{>�ff�fff��G�C)�f                                    BxWzئ  "          @(Q��p�>�׿@  ��{C)
�p�>�(��G�����C*
=                                    BxWz�L  
�          @.{�\)?녿h����33C'&f�\)?��p�����\C(@                                     BxWz��  �          @)��� ��>Ǯ�.{�o\)C+)� ��>�Q�333�v�RC+�                                    BxW{�  �          @(����>�Q�h�����
C+� ��>��R�k���33C,�H                                    BxW{>  
.          @!��   >�zᾀ  ��  C-n�   >�=q��=q���HC-�q                                    BxW{!�  �          @!�� ��=�\)���
��C2ff� ��=�\)���
���C2�                                     BxW{0�  �          @#33�!G�<��
���(�C3xR�!G�<��
���#�
C3�                                     BxW{?0  Z          @#33�   >�녽�Q��
=C*�q�   >��ͽ�G����C*ٚ                                    BxW{M�  �          @$z��#33�#�
�#�
��  C4�H�#33�#�
�#�
�s33C4�                                    BxW{\|  �          @#33�!�>�=q�#�
�#�
C-�3�!�>�=q���
���C-�R                                    BxW{k"  �          @!G��{>�(�<�?�RC*.�{>�(�<#�
>�C*(�                                    BxW{y�  �          @'��"�\?������R�\C&�"�\?
=�8Q��|��C&�                                    BxW{�n  �          @%�"�\>��ͽ�\)���HC+&f�"�\>Ǯ��Q���C+:�                                    BxW{�  �          @"�\�   >�\)>aG�@�z�C-� �   >���>L��@�=qC-B�                                    BxW{��  �          @%�#33>��=���@��C*���#33>��=��
?�  C*��                                    BxW{�`  �          @#�
��R?
==�G�@{C&���R?
==��
?���C&�{                                    BxW{�  �          @*=q�#�
?(����R����C&���#�
?
=��{��G�C&�3                                    BxW{Ѭ  �          @&ff�!G�>�
=��33��33C*�H�!G�>��;�p����C+�                                    BxW{�R  �          @'��%>��������C0���%>\)��{��z�C0�q                                    BxW{��  �          @&ff�%�<��
��=q����C3��%�<#�
��=q���C3��                                    BxW{��  �          @$z��   �����:ffC7\�   �#�
��\�7�
C7�)                                    BxW|D  �          @#33�"�\<#�
�W
=��Q�C3�q�"�\    �W
=����C3��                                    BxW|�  �          @ ���   �#�
���.�RC4T{�   ���
���.{C4s3                                    BxW|)�  �          @p����>\)�����G�C0�{���>������=qC0�                                    BxW|86  �          @#�
�#33<#�
���
�\)C3���#33<#�
����C3                                    BxW|F�  �          @!G��   =��.{�{�C10��   =��8Q�����C1^�                                    BxW|U�  
�          @\)�p�<���p��
�\C35��p�<��
�\�
=C3�
                                    BxW|d(  �          @�R�Q��=���@\)C@u��Q��=�@1G�C@\)                                    BxW|r�  �          @�H�ff��>8Q�@�p�C?.�ff��>L��@�z�C>��                                    BxW|�t  �          @\)�(���G�>\)@P  C>B��(���(�>#�
@k�C>)                                    BxW|�  
�          @�R����=u?�
=C>������=��
?��C>�                                     BxW|��  �          @����
�
=q>�=q@ҏ\CA:���
��>���@��
C@�3                                    BxW|�f  �          @
=�{�!G�>�ffA0  CC�3�{��R>��A:{CC}q                                    BxW|�  �          @
=�G���
=>�p�A=qC>�=�G����>ǮA��C>(�                                    BxW|ʲ  �          @
=�녾��H>��
@�z�C@8R�녾�>�{A�C?�f                                    BxW|�X  �          @������>���@�  C=�������>��
@�(�C=��                                    BxW|��  �          @���  �333>\A��CEaH�  �0��>���A�CE                                    BxW|��  T          @��   �Q�>��@ڏ\CJB��   �O\)>�z�@�p�CJ                                      BxW}J  T          ?�{��p����\>�Adz�CV����p���G�?   AxQ�CV5�                                    BxW}�  �          ?�zῨ�ÿk�>���AaG�CW����ÿh��>�G�At��CV��                                    BxW}"�  �          ?�p�����Y��?   A��CZ)����Tz�?�A�(�CYz�                                    BxW}1<  �          @��?c�
>�Q��z�aHA���?c�
>�\)����A�                                      BxW}?�  �          @�=q@��?333�XQ��a{A���@��?z��Y���cz�Ar�R                                    BxW}N�  �          @l(�@�?n{�1��FG�A��H@�?Tz��3�
�I\)A���                                    BxW}].  �          @��@   ?@  �o\)�p�HA���@   ?(��p���s�A�Q�                                    BxW}k�  �          @���@�?5�l���o(�A��
@�?z��n{�q�\A~�R                                    BxW}zz  �          @�G�@��?Y���g
=�e��A�p�@��?:�H�h���h�A��                                    BxW}�   �          @��
@{?����fff�]�A��H@{?z�H�hQ��`��A�(�                                    BxW}��  �          @��@G�?L���hQ��a�A�=q@G�?.{�j=q�dQ�A�                                      BxW}�l  �          @��@
�H?k��j=q�e(�A���@
�H?L���l(��g�A�p�                                    BxW}�  �          @��R@5�?�=q�S33�=�RA�  @5�?z�H�U��@G�A���                                    BxW}ø  �          @�  @<��?����N{�6ffA�(�@<��?�  �P  �8�HA�                                      BxW}�^  T          @�G�@7�?h���X���A��A�z�@7�?L���Z�H�C�RAy��                                    BxW}�  �          @�G�@333?8Q��_\)�I\)Ag�@333?�R�`���K  AE�                                    BxW}�  �          @��@)��?(��dz��R�HAO
=@)��?   �e�TQ�A+
=                                    BxW}�P  �          @��@8��?0���\���E(�AV�H@8��?
=�^{�F��A7�                                    BxW~�  �          @�Q�@'
=?�R�fff�Uz�AU�@'
=?�\�g��V�A2=q                                    BxW~�  T          @�
=@{?k��e��V�\A�
=@{?O\)�g
=�X�RA��                                    BxW~*B  �          @�ff?�?�  �p���g{B(�?�?���s33�k�B�                                    BxW~8�  �          @���?�
=?����q��a�\B�\?�
=?�  �tz��e�\B�                                    BxW~G�  �          @���?�?�ff�r�\�c��B33?�?�Q��u��gz�Bz�                                    BxW~V4  �          @�
=?���?Ǯ�p  �d��B ��?���?��H�r�\�hz�Bz�                                    BxW~d�  �          @��@&ff?L���n{�W=qA��@&ff?333�o\)�X�Ap                                      BxW~s�  �          @���@�H?h���l���\(�A�\)@�H?Q��n{�^(�A�\)                                    BxW~�&  �          @���@�
?��
�i���X\)A�G�@�
?�
=�k��[{A�
=                                    BxW~��  T          @�z�@��?���l(��S�
Bz�@��?ٙ��o\)�WffB��                                    BxW~�r  �          @�
=@�\@   �fff�GQ�B$�@�\?�z��i���J�HBz�                                    BxW~�  �          @�@{?�(��g
=�J
=B&33@{?���i���M��B!(�                                    BxW~��  T          @�ff@�
@���\���<�B.��@�
@��`  �@�B*ff                                    BxW~�d  T          @�@@��R�\�2\)B5G�@@33�U�6  B1��                                    BxW~�
  
�          @�
=@'�?�p��X���7�\BQ�@'�?�33�[��:�\B{                                    BxW~�  �          @�  @=q@(��S33�/��B5�\@=q@��Vff�3=qB2(�                                    BxW~�V  �          @Z=q?��@=p����H��z�B��q?��@<(����
��p�B�ff                                    BxW�  �          @k�?�\@_\)��������B�ff?�\@^{�����B�L�                                    BxW�  T          @q�>�(�@k��@  �6=qB��q>�(�@j=q�Q��H(�B��3                                    BxW#H  �          @z=q?Q�@e�����Q�B�(�?Q�@dz΅{���HB���                                    BxW1�  �          @xQ�?G�@`�׿�����B�Ǯ?G�@^�R���R��33B���                                    BxW@�  �          @P  >���@L(��u��\)B�G�>���@K����	��B�G�                                    BxWO:  �          @2�\>���@0�׼���B�33>���@0�׽��
����B�.                                    BxW]�  �          @<��?(�@7����(Q�B��?(�@7��8Q��dz�B��f                                    BxWl�  �          @1�?&ff@,�;.{�\��B��f?&ff@,(��W
=���B��H                                    BxW{,  �          @*=q�#�
@"�\?(��Aip�B�=q�#�
@#33?�RA[
=B�33                                    BxW��  �          @$z�=u@�R>aG�@��
B�ff=u@�R>8Q�@�  B�k�                                    BxW�x  �          @=p�?(��@5��
=�p�B�?(��@5����\B���                                    BxW�  �          @j=q?�G�@[��\(��YG�B�(�?�G�@Z=q�h���eB�\                                    BxW��  �          @�?��H@_\)����ɮBs{?��H@^{��
=��
=Br\)                                    BxW�j  �          @�z�@�\@N{�
=��p�Bf�R@�\@L���	����ffBe�
                                    BxW�  �          @�ff@��@p�׿�=q��p�Bqff@��@n�R��\)��ffBp��                                    BxW�  �          @�(�@�@h�ÿ�
=��\)Bo33@�@g
=��(���  Bn�\                                    BxW�\  �          @�
=@��@\���(����\Bh�
@��@Z�H��R���HBh                                      BxW�  �          @���@��@P���"�\� Q�BV�
@��@N�R�$z��=qBU�                                    BxW��  �          @��@!G�@HQ��'��p�BM
=@!G�@G
=�)���=qBL{                                    BxW�N  �          @���@��@N�R�*�H��B[�
@��@L���-p��
p�B[                                      BxW�*�  �          @�Q�@=q@E�,���
(�BP33@=q@Dz��.�R�BOQ�                                    BxW�9�  �          @�
=@
=@@  �1G����BO�@
=@>�R�2�\�G�BN=q                                    BxW�H@  �          @�{@p�@1G��8Q��\)BA��@p�@0  �9����B@�R                                    BxW�V�  T          @�=q@,��@�\�7����B!
=@,��@G��8Q��
=B                                       BxW�e�  �          @�G�@,(�@���7
=���B \)@,(�@  �8Q����Bp�                                    BxW�t2  �          @��@!G�?���.{�#��B��@!G�?�\)�/\)�$ffB��                                    BxW���  �          @�(�@	��?�ff�?\)�;�B@	��?��
�@  �<G�B                                    BxW��~  �          @��@��?����N{�Gz�BQ�@��?˅�N�R�H{BG�                                    BxW��$  �          @��@Q�?����=p��8�B  @Q�?Ǯ�=p��8��B33                                    BxW���  �          @Mp�@	��?Tz��{�/z�A�z�@	��?Q��{�/A�
=                                    BxW��p  �          @H��@
�H?.{�
=q�.{A�G�@
�H?+��
=q�.G�A�{                                    BxW��  �          @�G�@'
=?�=q�H���;��A�=q@'
=?����H���<33A�33                                    BxW�ڼ  �          @��
@7
=>��^�R�H��Az�@7
=>���^�R�H�
A�\                                    BxW��b  �          @�p�?:�H�G��-p��!
=C���?:�H�G��-p�� ��C��R                                    BxW��  �          @�?p���I���=p��({C�XR?p���I���=p��'��C�W
                                    BxW��  �          @�Q�?���Fff�G��.p�C�q�?���Fff�G��.z�C�q�                                    BxW�T  �          @��\?��R�E�H���-\)C��
?��R�E�I���-�C��R                                    BxW�#�  �          @�z�?����8���S33�7{C�?����8���S33�7\)C�Ǯ                                    BxW�2�  �          @�33?�=q��_\)�F33C��?�=q���_\)�F�\C��                                    BxW�AF  �          @��\?�=q�
=�^�R�E=qC��f?�=q�ff�^�R�E�RC��{                                    BxW�O�  �          @�33?}p��A��S33�7C��?}p��AG��S�
�8z�C�
=                                    BxW�^�  �          @���?�
=�7��QG��:{C�(�?�
=�7
=�Q��:�HC�4{                                    BxW�m8  �          @�33?Q��R�\�Fff�)�C��?Q��QG��G��*�C�\                                    BxW�{�  �          @�ff?=p��S�
�P���/�\C�G�?=p��R�\�Q��0�RC�O\                                    BxW���  �          @���?\(��J�H�Q��4  C��
?\(��I���S33�5=qC���                                    BxW��*  �          @�
=?��H��R�L(��:�C�N?��H�p��Mp��;�HC�k�                                    BxW���  �          @��@dz�B�\�W��-{C�}q@dz�#�
�W��-(�C��                                    BxW��v  �          @�z�@`�׾B�\�Y���033C�k�@`�׾.{�Y���0G�C���                                    BxW��  �          @��@h�þL���Z=q�,=qC�t{@h�þ.{�Z=q�,Q�C���                                    BxW���  �          @�ff@j=q���R�S33�'�HC��\@j=q��\)�S�
�(
=C��=                                    BxW��h  �          @���@�G�=��J�H��\?�z�@�G�>���J�H�z�@                                    BxW��  �          @���@u>�=q�XQ��$��@�  @u>����XQ��$��@�                                      BxW���  �          @���@`  >�Q��g
=�6�@�=q@`  >����fff�6ff@�ff                                    BxW�Z  �          @���@Vff?333�k��<ffA=��@Vff?=p��j�H�;�
AH��                                    BxW�   �          @���@Z=q?�33�^{�/�HA�z�@Z=q?�Q��]p��/  A��                                    BxW�+�  �          @�ff@XQ�?���W
=�+ffA���@XQ�?���U�*\)A��\                                    BxW�:L  �          @�  @Y��?�{�^�R�1{A���@Y��?�z��^{�0�A�ff                                    BxW�H�  �          @�Q�@H��?\�dz��6�\A�
=@H��?����c33�5(�A��H                                    BxW�W�  �          @�=q@(��@!��^�R�.Q�B/(�@(��@%��\(��+��B1p�                                    BxW�f>  �          @��\@/\)@%�Y���((�B-@/\)@(���W
=�%�RB0
=                                    BxW�t�  �          @���@-p�@'
=�W��'(�B/�@-p�@*�H�Tz��$��B2                                      BxW���  T          @�p�@=p�@=q�Z�H�'\)BG�@=p�@p��XQ��$�B�H                                    BxW��0  �          @���@r�\>����^{�)G�@��@r�\>�Q��]p��(�@��R                                    BxW���  �          @���@s�
?E��dz��*Q�A6ff@s�
?Tz��c33�)z�AE�                                    BxW��|  �          @�@s�
?8Q��[��&  A+�@s�
?J=q�Z�H�%(�A:=q                                    BxW��"  �          @�ff@s33?L���\(��&\)A=p�@s33?\(��[��%ffAL��                                    BxW���  �          @�(�@o\)?���S�
�!=qA��@o\)?����R�\��HA�z�                                    BxW��n  �          @��\@w
=?���@�����A�\)@w
=?�33�?\)�
=A��
                                    BxW��  �          @���@n�R?���Fff��RA��H@n�R?����Dz��  A��                                    BxW���  �          @�G�@dz�?�
=�Vff�&�
A��R@dz�?�G��U��%33A��                                    BxW�`  �          @���@P  ?���b�\�2Q�A���@P  ?�\)�`���0{A�{                                    BxW�  �          @��@Tz�?��R�`���/��A�G�@Tz�?����^�R�-Aʏ\                                    BxW�$�  
�          @���@S33?�ff�e�5�\A�Q�@S33?���c�
�3z�A�z�                                    BxW�3R  �          @�G�@U�?���c33�3Q�A��@U�?����aG��1=qA��
                                    BxW�A�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�_D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�m�   i          @�
=@g�?��
�[��$Q�A��R@g�?�\)�X���!�
A��\                                    BxW�|�  �          @��
@Tz�@Dz��9���p�B+
=@Tz�@I���3�
��(�B.                                      BxW��6  �          @�33@Mp�@R�\�.{��G�B6�H@Mp�@W
=�(Q����
B9z�                                    BxW���  �          @��H@E@S�
�3�
���\B;��@E@X���-p����RB>�\                                    BxW���  �          @���@HQ�@C33�<���{B1(�@HQ�@H���7
=�=qB4ff                                    BxW��(  �          @�\)@E�@^{�=q��G�BA��@E�@b�\�33�иRBC�                                    BxW���  �          @��H@+�@{���\)��z�B^�R@+�@~�R��  ���\B`
=                                    BxW��t  �          @�(�@#�
@��ÿ�
=�U��Bl�@#�
@�녿�ff�<z�Bm\)                                    BxW��  �          @�Q�@8Q�@|�Ϳ�(���BWz�@8Q�@�Q������BY(�                                    BxW���  �          @��\@(��@��R��z���Q�Bg��@(��@��׿�\��\)Bi�                                    BxW� f  �          @���@5@z=q�
=q����BX  @5@~�R����(�BY��                                    BxW�  �          @�=q@/\)@�  �
=q��B^p�@/\)@�=q�G�����B`\)                                    BxW��  �          @���@
=@��h���<\)B7�@
=@#�
�c33�6z�B=\)                                    BxW�,X  �          @�{@	��@\)�j�H�@p�BE  @	��@'��dz��:�BJ                                    BxW�:�  �          @�ff@>�R@n�R�ff���
BM��@>�R@s�
���H����BO��                                    BxW�I�  �          @��H@C33@q��{����BLQ�@C33@w
=�����BN�                                    BxW�XJ  �          @�ff@5@|�Ϳ�����BY33@5@�Q��33���RB[                                      BxW�f�  �          @�=q@�R@��׿������BiQ�@�R@��\�ٙ���ffBk
=                                    BxW�u�  �          @���@�R@������
��{B@�R@�������k�B�p�                                    BxW��<  �          @�=q@7
=@���������RB]��@7
=@�\)��
=�{
=B_�                                    BxW���  �          @�(�@B�\@��Ϳ�=q���RBW�@B�\@��R��z��s33BX�R                                    BxW���  T          @���@Dz�@w
=����
=BN(�@Dz�@{������  BP33                                    BxW��.  �          @�  @Y��@g��
=��\)B;ff@Y��@n{�p����B>ff                                    BxW���  �          @�G�@`��@aG�����{B4ff@`��@g��G���  B7��                                    BxW��z  �          @�
=@P��@Vff�3�
��z�B733@P��@^{�*�H��  B;33                                    BxW��   �          @��@<��@p��n�R�0��B��@<��@(Q��g��*  B&�H                                    BxW���  �          @�
=@<(�@(Q��qG��.��B'z�@<(�@333�h���'��B.�                                    BxW��l  �          @�33@G�@>{�J=q�{B.ff@G�@G
=�AG��	B3�R                                    BxW�  �          @���@P��@S�
�>�R���B5��@P��@\���4z����B:�                                    BxW��  �          @���@^�R@!��qG��$��B�
@^�R@,���i���z�B33                                    BxW�%^  �          @�Q�@q�@�H�c33�=qB�R@q�@%�[��33B	                                    BxW�4  �          @��H@w
=@'
=�`����HB  @w
=@1G��XQ���B�                                    BxW�B�  �          @�
=@~�R?�
=�\)�(33AΣ�@~�R@��x���#{A�Q�                                    BxW�QP  �          @���@��@  �^�R�Q�A��
@��@�H�W
=��A�                                    BxW�_�  T          @��@��@=q�O\)��A�  @��@$z��G����\A�\                                    BxW�n�  �          @�G�@�G�@���G����A�\)@�G�@'
=�?\)��  A�p�                                    BxW�}B  �          @\@��@�_\)�A�  @��@!G��W
=���A�Q�                                    BxW���  �          @��@��
@	���G
=��RA�z�@��
@�
�?\)���A�
=                                    BxW���  �          @���@�{@   �E���HA���@�{@
�H�>�R��\)A���                                    BxW��4  �          @�
=@�Q�?�p��p���\)A�(�@�Q�@(��h���(�A���                                    BxW���  �          @�Q�@��\?�ff����1\)A��@��\@�\��  �,  A���                                    BxW�ƀ  �          @ƸR@��?��H��33�%  A���@��@(��~�R�ffA�ff                                    BxW��&  �          @�{@k�?��
��
=�G=qA�  @k�?����(��A��A�G�                                    BxW���  �          @��R?�
=�h�����
33C��q?�
=�!G�����C���                                    BxW��r  �          @�G����l(����H�>\)C������\�������KffC�T{                                    BxW�  T          @�  ���c�
��p��E(�C�J=���S�
���
�Rp�C��                                    BxW��  �          @�=q��ff�^�R����K��C�W
��ff�N{��Q��Y
=C��                                    BxW�d  T          @��R��p��P������T33C��f��p��?\)�����a��C�}q                                    BxW�-
  �          @���(��aG���z��K�C�Ϳ(��O\)���H�Y(�C���                                    BxW�;�  �          @��Ϳ�\�W
=��\)�S\)C�����\�E����a=qC�C�                                    BxW�JV  �          @�=q�\)�0  ���H�qp�C�� �\)�����  ���C�P�                                    BxW�X�  �          @�33=��
�*�H��p��u�
C���=��
�
=��=q�C��                                    BxW�g�  �          @�ff��33����ff�~ffC��ᾳ33�
=��33aHC�=q                                    BxW�vH  �          @�Q쿅��mp������@  C~T{����[���Q��N�C}�                                    BxW���  �          @�{��  �XQ����R�T��C}����  �Dz���p��c  C{�                                    BxW���  �          @�
=������z��_\)��C�R���������qG��%{C33                                    BxW��:  �          @�  ������  ����Q�C�Ϳ������
�	����Ch�                                    BxW���  �          @�녿�p��w��j�H�&{Cy&f��p��g��z�H�4\)Cw��                                    BxW���  �          @�\)�
�H��  �
=���CtB��
�H���H�=q��
=CsxR                                    BxW��,  �          @�녿�
=��녿�=q���\Cw
=��
=��p������\)Cvh�                                    BxW���  X          @��R���H�����(���z�Cvh����H����G���p�Cu�=                                    BxW��x  �          @���ff����z�����Cp33�ff�{��&ff���Co!H                                    BxW��  �          @�33��p������  �n=qC|!H��p���  ��\)���C{�                                     BxW��  T          @��R��
���R�ٙ�����Cv#���
���\�z�����Cu�{                                    BxW�j  �          @�G������Q��\)���CtxR�����(���p����Cs�H                                    BxW�&  �          @���dz����Ϳ�p��j�\Cf��dz����ÿ�ff��
=Cf&f                                    BxW�4�  �          @�{�_\)�����Q��:ffCh�)�_\)��Q���
�o�
Ch@                                     BxW�C\  �          @��\�p  �����\)�YCdxR�p  ��(���
=��(�Cc�3                                    BxW�R  �          @���o\)��{��p��d��Ce��o\)��=q�����\Ce�                                    BxW�`�  �          @����c�
��p���G��A�Ch���c�
��녿����x  Ch                                    BxW�oN  �          @�33�q���z῅���Cf�H�q���������S33CfQ�                                    BxW�}�  �          @��
�E���Ϳ�
=��33Cm��E����33����Cl��                                    BxW���  �          @�z��O\)��
=�ٙ��w\)Cn(��O\)��=q�ff���
Cmk�                                    BxW��@  T          @ָR�*�H��(�����=qCt���*�H��p��7
=�ʣ�Cs�{                                    BxW���  �          @����>{���\�1G�����Cr  �>{��33�L�����HCp��                                    BxW���  �          @�\)�I�������@�����HCq\)�I����Q��]p����HCp=q                                    BxW��2  �          @�\�J=q��(��;���\)Cp�3�J=q���
�XQ���p�Co��                                    BxW���  �          @��
�@����G��S33���Cq� �@����  �o\)���Cp5�                                    BxW��~  �          @�33�G������I����=qCp�H�G������e���HCo\)                                    BxW��$  �          @���Mp���33�[���  Cp#��Mp���G��xQ�� ffCn�q                                    BxW��  �          @�ff�XQ������k����Cn}q�XQ���{��(��Cl��                                    BxW�p  �          @�ff�aG����mp����Cm  �aG����H�����\)CkQ�                                    BxW�  �          @��^�R��p��b�\��(�CmJ=�^�R��33�\)�Q�Ck��                                    BxW�-�  �          @�ff�x����{�G���{Cjk��x�������dz����
Ch�                                    BxW�<b  �          @�  ��  ��  ����Q�Cc����  �����4z���p�Cb��                                    BxW�K  T          @�p�����u���  �=qCZ�����]p���=q�p�CX(�                                    BxW�Y�  �          A ���
�H��=q���8RCS���
�H�(����  ffCD��                                    BxW�hT  �          A�\��33�W
=��p��C:aH��33>��H���.C%xR                                    BxW�v�  �          @�Q쿱�>\)��(�.C.=q���?Q����H�fC}q                                    BxW���  �          @����>�=q���¡�C'{��?s33��Q�L�C��                                    BxW��F  �          @��
�s33>�=q���¥#�C#޸�s33?u��  ��C�H                                    BxW���  �          @�
=��zἣ�
��(�¢.C4�쿔z�?(����33u�CT{                                    BxW���  �          @�Q�Y��>�p���{¥�RCz�Y��?�����
�C �3                                    BxW��8  �          @��þ�ff?���¨��C ;�ff?�G����� B��
                                    BxW���  �          @�녾�?=p���Q�¦��B����?�Q�������B���                                    BxW�݄  �          @�\���H?s33��  £�)B�G����H?�33��(��qB�                                    BxW��*  �          @�{��Q�?����=q¡� B֮��Q�?�=q���fB���                                    BxW���  T          @�=q��=q?����¢��BЊ=��=q?�  ��33BŊ=                                    BxW�	v  �          @���Q�?�����qB�aH��Q�?�p���\��B�u�                                    BxW�  �          @��
���?��
����£(�B�Q���?��H����\B��H                                    BxW�&�  �          @�녾��H?���  §��C�
���H?�  ���
=B޽q                                    BxW�5h  �          @�G����H>�G���Q�©8RC	Ǯ���H?�����ǮB�                                    BxW�D  �          @�z���>��ᙚ©��C����?�
=��
=�B��
                                    BxW�R�  �          @�>��
�#�
��33®��C���>��
?+���=q§�B�=q                                    BxW�aZ  �          @�
=?fff�����=q¥aHC��)?fff>����¤�A�
=                                    BxW�p   �          @�\)�&ff?�Q�����8RB�{�&ff@Q���
=G�B��)                                    BxW�~�  �          @ᙚ�(��?�z���  �3B�\)�(��@%�У�k�B�                                    BxW��L  �          @�  ��ff@����(�k�B�zῆff@B�\��33�x��B�8R                                    BxW���  �          @�(�>�>k��ָR«  AΣ�>�?n{����¢33B{=q                                    BxW���  T          @�=q?�p�@Dz��ȣ��x�HB�aH?�p�@mp���p��bG�B�\)                                    BxW��>  �          @�ff?Y��@����ff�#z�B�?Y��@��\�����\B�k�                                    BxW���  �          @�p�?W
=@��\����ffB���?W
=@ȣ��vff���\B�                                    BxW�֊  �          @��H?W
=@�(����
�Q�B���?W
=@��H�\)�
=B�=q                                    BxW��0  �          @��H?5@�\)����G�B��?5@�{�z�H� ��B���                                    BxW���  �          @�  ?L��@�{�����B��3?L��@��H�Y����B���                                    BxW�|  �          @�  ?�\)@\�qG�����B�.?�\)@θR�E��ə�B�\)                                    BxW�"  �          @�=q?��@����w�� �B��
?��@���L(���B�Q�                                    BxW��  �          @�  ?�33@��k�����B���?�33@ٙ��<������B��H                                    BxW�.n  �          @�p�@:�H@�(����
� ��Bg�@:�H@����=q�
�RBq33                                    BxW�=  �          @�(�@Tz�@�����G��!�
BV�H@Tz�@�����  �BbG�                                    BxW�K�  �          @�\)@hQ�@w
=�����*�B:�@hQ�@�z�����BIp�                                    BxW�Z`  �          @�{@Z�H@s33��Q��1�B@=q@Z�H@��H�����BOQ�                                    BxW�i  T          @�p�@Z�H@��R��{�#ffBK�\@Z�H@��R�����BX(�                                    BxW�w�  �          @�@#�
@��������-ffBq�@#�
@�����33�(�B|�\                                    BxW��R  �          @��
@:=q@�  ����%�Bdff@:=q@�����{�G�Boz�                                    BxW���  �          @�{@S33@   �����`��A�@S33@*=q��  �Qp�BQ�                                    BxW���  �          @�{@)��?�z������vG�Bp�@)��@&ff�����d�HB233                                    BxW��D  �          @�Q�@;�@�\��{�g�HB{@;�@>{��(��U�
B5�R                                    BxW���  �          @�\@Mp�?�Q����v
=A�Q�@Mp�@{��{�gG�BQ�                                    BxW�ϐ  �          @�z�@-p�@p����qz�B(�
@-p�@L�����H�]BG33                                    BxW��6  �          @�@6ff@�
��  �v33B�\@6ff@5���ff�dQ�B3                                      BxW���  �          @��H@[�@?\)���H{B$G�@[�@g������4�HB:=q                                    BxW���  �          @�33@e�@ff���X�\B\)@e�@C�
��33�G��B!��                                    BxW�
(  �          @�  @a�@ ����=q�Z  B�H@a�@N�R��
=�HQ�B*{                                    BxW��  �          @��R@hQ�@=p��\�Q�RB��@hQ�@k����>�RB5�R                                    BxW�'t  �          @ָR@<��@5�����O�HB/=q@<��@]p���33�;=qBE��                                    BxW�6  �          @�{@B�\@L(���z��:  B9�@B�\@o\)���R�$�\BK                                    BxW�D�  �          @�G�@>�R?u���H��A�33@>�R?���p��y�\A�                                      BxW�Sf  �          @�
=>�G�@:�H��
=�h
=B���>�G�@^�R��=q�Kp�B��{                                    BxW�b  �          @�R�
=@����.�R���B�� �
=@�{��33�|Q�B�
=                                    BxW�p�  �          @��
�n{@�׿����X��B�.�n{@�{�\(��ə�B��)                                    BxW�X  T          @�녿�z�@�G��u��G�BΊ=��z�@��L�;���B�L�                                    BxW���  �          @�\)���@�\������HBȞ����@��H>�p�@:=qBș�                                    BxW���  �          @�\����@��
�=p�����Bͮ����@��>��?���B͏\                                    BxW��J  �          @��H��\@�\)�L���θRBսq��\@���=��
?+�BՊ=                                    BxW���  �          @���fff@���@I��A�p�B�\�fff@�{@w
=A�p�B��=                                    BxW�Ȗ  �          @�
=��\)@�33@9��A���B�k���\)@�p�@k�A�  B�k�                                    BxW��<  �          @�����(�@��@:�HA�Q�B��f��(�@�
=@j�HA�\)B�{                                    BxW���  �          @�33�qG�@��@EA��B�z��qG�@���@uA�=qB��
                                    BxW��  �          @�(����@z�H@���Bz�C	���@Q�@�Q�B+�C�                                    BxW�.  �          @�R�l��@u@��HB(��C���l��@J=q@��B=�C��                                    BxW��  �          @�Q��b�\@~�R@�p�B*��C��b�\@Q�@�p�BA�C	5�                                    BxW� z  �          @�p��L(�@�(�@�{B#ffB�{�L(�@mp�@��B;�
C��                                    BxW�/   �          @��H�AG�@�33@��B'\)B����AG�@\(�@�=qB?CO\                                    BxW�=�  �          @���1�@}p�@�=qB-\)B�  �1�@S33@�=qBF\)C�                                    BxW�Ll  �          @����G�@���@�  B'B�W
�G�@aG�@���BC  B��                                    BxW�[  �          @Ϯ���@���@��B&�HB�Ǯ���@h��@��BBz�B�q                                    BxW�i�  �          @�p���@��R@o\)B�\B�8R��@y��@��\B8�
B��                                    BxW�x^  �          @�=���@�Q�@'
=A�G�B��R=���@��H@Tz�B	B��                                    BxW��  �          @��R?(��@�(�?�\A���B��=?(��@�=q@"�\AʸRB�Ǯ                                    BxW���  �          @��?h��@�=q?�p�A�=qB��?h��@���@   A�=qB��                                     BxW��P  �          @��R?�z�@��?z�HA��B���?�z�@���?�  A�  B�Q�                                    BxW���  �          @��@
�H@�Q�?aG�A�HB���@
�H@��\?�33A�=qB���                                    BxW���  �          @�@{@��?   @�(�B�.@{@�?�G�AEG�B��                                    BxW��B  �          @���@�@�
=?5@��B��@�@��?��RAf=qB��)                                    BxW���  �          @��@  @���?s33A(�B��R@  @��H?޸RA��B�B�                                    BxW��  �          @Å@.�R@�{>Ǯ@h��B}ff@.�R@��\?�A1p�B{p�                                    BxW��4  �          @��
@-p�@��R>��@��
B~��@-p�@��H?�G�A>{B|p�                                    BxW�
�  �          @��H@1�@�z�>�(�@�Q�Bz�
@1�@���?��HA7�
Bx�R                                    BxW��  �          @�Q�@�@��?0��@ӅB�\@�@��\?��RAf=qB���                                    BxW�(&  �          @�  @Q�@��?:�H@�
=B�
=@Q�@�z�?��An�\B��
                                    BxW�6�  �          @��@�@��?�R@�B�\)@�@��?�Q�A]G�B�8R                                    BxW�Er  �          @�=q@333@��#�
��33Bv�@333@�(�?=p�@�RBu�\                                    BxW�T  �          @�(�@mp�@�
=��33���RBIQ�@mp�@�ff�����5�BO
=                                    BxW�b�  W          @�ff@p  @1��:�H� (�B\)@p  @L(��{��
=B!��                                    BxW�qd  �          @��@�=����u�i33@��@�?.{�q��c�A�ff                                    BxW��
  �          @�{@1G�����n{�T  C��@1G�>#�
�o\)�U�\@U                                    BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��V              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���   �          @�  @?\)���
�\)�TffC��@?\)?
=�|(��Q
=A3�                                    BxW��H  �          @�  @$z�Tz��w��\Q�C�
@$z�W
=�}p��cffC��q                                    BxW���  �          @���@8�ÿ��tz��Q��C��@8��=�Q��vff�T��?�(�                                    BxW��  �          @�  @=p��W
=�k��L\)C���@=p�>�p��j�H�K\)@�p�                                    BxW��:  �          @�z�@HQ�=p��k��C33C�Y�@HQ����p  �HQ�C��q                                    BxW��  �          @��@K��(���~�R�J�HC�"�@K����
�����NC��=                                    BxW��  �          @�(�@P  �8Q��{��Ip�C�p�@P  >�ff�z=q�G�@��
                                    BxW�!,  �          @��R@B�\���c33�E�\C��@B�\>�G��aG��C�A(�                                    BxW�/�  �          @�  @@  >����!���@�\@@  ?L�����z�An�H                                    BxW�>x  �          @s33@=p�?�z��z��33A��@=p�?�p���{��(�Aԏ\                                    BxW�M  �          @tz�@QG�?���ff���RA���@QG�?�33������A�\)                                    BxW�[�  �          @�G�@fff?��H�������HA���@fff?�Q쿦ff��
=A���                                    BxW�jj  �          @u�@B�\?�=q����z�A��R@B�\@�\������(�Bp�                                    BxW�y  �          @q�@;�@Q�^�R�W�B�@;�@  ��\���\B�
                                    BxW���  �          @��@@��@Tz������B>��@@��@W
=    ��B@�                                    BxW��\  �          @�33@\(�@aG������O
=B6�R@\(�@j=q����=qB;�                                    BxW��  �          @���@`  @XQ�����B0
=@`  @hQ쿠  �^�HB8p�                                    BxW���  �          @���@r�\@hQ��7���B/=q@r�\@����R��
=B;��                                    BxW��N  �          @�G�@�
=@��������B.�@�
=@������qp�B7�                                    BxW���  T          @θR@���@�=q�z�����B'�@���@�33�����B=qB.��                                    BxW�ߚ  �          @�@��@����\)����B)ff@��@��H���
�^�HB2(�                                    BxW��@  �          @θR@���@e��
=����B=q@���@w
=����8��BQ�                                    BxW���  �          @љ�@�
=@i�������C�
B�@�
=@u��=p���G�B(�                                    BxW��  �          @���@���@}p�����
=B  @���@�\)��(��,(�B%ff                                    BxW�2  �          @�@u@��\�N{���HBA�\@u@���(����\BM��                                    BxW�(�  �          @�\)@�\)@s33�:=q����BQ�@�\)@�  �p����B'                                    BxW�7~  �          @Ӆ@��R@J=q�W���B�@��R@l���1�����B�                                    BxW�F$  �          @�p�@�=q@��\)�%ffAΣ�@�=q@2�\�s33��
Bff                                    BxW�T�  �          @˅@u�?�(������F  A�@u�@���\)�4p�A�                                      BxW�cp  �          @�33@x��?��H���OG�A�z�@x��@ff��z��?��A��                                    BxW�r  T          @ٙ�@�?�(����\�A\)A�\)@�@%��ff�.�A�                                      BxW���  �          @�
=@��\?�(�����1{A�  @��\@2�\��{�z�A���                                    BxW��b  �          @�p�@��@.{���H�=qA�\@��@\(��s33��B�                                    BxW��  �          @��
@�p�@W
=�o\)�33B
=@�p�@~{�E��  B!�                                    BxW���  �          @�{@��@_\)�j�H��p�Bff@��@��\�?\)��Q�B"��                                    BxW��T  �          @��@��@h���a���33BQ�@��@��R�4z�����B#��                                    BxW���  �          @��@���@a��tz��
=B  @���@����HQ���ffB$=q                                    BxW�ؠ  �          @�z�@�
=@s33�qG���\)BQ�@�
=@���A����B,\)                                    BxW��F  �          @�@�\)@{��mp����RB@�\)@����<(���\)B/                                      BxW���  �          @�(�@���@q��l(����
Bz�@���@�(��<���îB*=q                                    BxW��  �          @���@��@l(��o\)� �HBz�@��@����@����G�B+                                      BxW�8  �          @޸R@�Q�@Fff�����
=BQ�@�Q�@q��X������B��                                    BxW�!�  �          @���@��\@K���Q��
��BQ�@��\@vff�W
=��ffB\)                                    BxW�0�  T          @�Q�@�p�@N�R�s33�Q�B=q@�p�@xQ��H�����B�H                                    BxW�?*  �          @�ff@�Q�@Fff�n{�33A��@�Q�@n�R�E�ң�B�                                    BxW�M�  �          @��
@��\@C�
�L����Bz�@��\@fff�%���\)B                                    BxW�\v  �          @��
@��@7��2�\��p�A�G�@��@U�p����B��                                    BxW�k  �          @��H@��\@;��$z���\)BQ�@��\@Vff��p�����B�                                    BxW�y�  �          @�{@���@@  �G
=���B ��@���@a��   ���\B�H                                    BxW��h  �          @��@�Q�@k��(Q�����B��@�Q�@�33�����(�B#�                                    BxW��  �          @ٙ�@���@��\��{��B8\)@���@�ff�8Q���
B;33                                    BxW���  �          @��@�
=@�p���\�h��B8�
@�
=@�p��O\)��(�B>ff                                    BxW��Z  �          @�z�@�z�@��Ϳ�
=�Z{B@  @�z�@��
�.{��ffBD�
                                    BxW��   �          @�p�@�\)@�녿Tz���(�BA=q@�\)@��
>B�\?�ffBB�                                    BxW�Ѧ  T          @�ff@�ff@��;.{����BO  @�ff@��\?h��@�Q�BM�\                                    BxW��L  �          @��H@���@��\���
�L��B]��@���@�\)?��AG�B[�R                                    BxW���  �          @�ff@~{@��?�\@�B[�H@~{@��R?�=qAR�RBW��                                    BxW���  �          @�  @���@��?z�@���BP�@���@���?У�AW33BL�                                    BxW�>  �          @�G�@�ff@�33>�{@4z�BT�\@�ff@�?�A:�\BQ�                                    BxW��  �          @��@�{@���?L��@���BS@�{@�G�?���Au��BNz�                                    BxW�)�  �          @�
=@|��@�p�?E�@˅B\ff@|��@��?���Aw\)BWff                                    BxW�80  �          @��
@�p�@��\=��
?+�BNG�@�p�@�ff?�z�A�RBK��                                    BxW�F�  �          @���@���@��>�(�@[�BO�R@���@�p�?�G�AC�BK�
                                    BxW�U|  �          @�(�@�(�@�(�?�  A
{BL�R@�(�@��H?�p�A��\BF
=                                    BxW�d"  �          @�  @���@\)@A�{B@���@`  @5�A�=qBz�                                    BxW�r�  �          @�(�@��@|��@�A���BG�@��@^�R@1G�A�p�B(�                                    BxW��n  R          @ҏ\@��H@�Q�?���A�(�B��@��H@c33@-p�A�=qB�                                    BxW��  V          @љ�@���@�G�?�33A�
=B \)@���@e�@*=qA��B                                      BxW���  T          @�
=@��\@�?�A�(�B)�\@��\@n�R@(Q�A�(�B�R                                    BxW��`  T          @ə�@��@���?�z�At��B2�R@��@vff@�RA��B'                                      BxW��  �          @˅@�ff@�z�?�{A�{B9=q@�ff@{�@,��A�
=B,�\                                    BxW�ʬ  �          @�p�@\)@��\@!�A�=qB=p�@\)@o\)@VffA�  B,�                                    BxW��R  �          @�
=@l(�@�=q@8��Aݙ�B?
=@l(�@Z=q@h��BB*�R                                    BxW���  �          @���@]p�@�Q�@9��A�BD@]p�@Vff@i��B��B0
=                                    BxW���  �          @�
=@J=q@��@#33A��HBX�@J=q@p��@XQ�B
=qBG�                                    BxW�D  �          @�p�@L(�@��H@6ffA�ffBP33@L(�@[�@g�B�
B<=q                                    BxW��  �          @��H@C33@z�H@A�A��RBP�R@C33@N�R@p��B!
=B:��                                    BxW�"�  �          @�=q@U�@�\)?�  A��RB[33@U�@���@+�A�Q�BP{                                    BxW�16  �          @�33@U�@�  ?aG�AG�Baff@U�@��R?��A�z�B[�                                    BxW�?�  �          @�  @a�@��\?���A��BWp�@a�@�33@1G�A�(�BL                                      BxW�N�  �          @�ff@W�@�
=@�A���BT  @W�@}p�@:=qA�BFff                                    BxW�](  �          @Å@c�
@�
=@#�
A�G�BG�@c�
@fff@XQ�B�
B5p�                                    BxW�k�  T          @���@��H@�  ?���Ae�BD�@��H@�=q@#33A�p�B:
=                                    BxW�zt  �          @�Q�@�z�@���?�(�Ak�BJ�@�z�@��H@.�RA�ffB?                                    BxW��  �          @�G�@���@��?���AE��BG�@���@���@{A���B=z�                                    BxW���  $          @�@�=q@�ff?s33A33BO�@�=q@�(�@ ��A�z�BH                                    BxW��f  �          @�(�@x��@�33?��HA0  BL��@x��@��@(�A�(�BDG�                                    BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�ò              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��  	b          @���@��\@o\)@#�
A��RB#33@��\@G�@R�\A�p�B�
                                    BxW��J  "          @ȣ�@�=q@j�H@)��Aȏ\B!�\@�=q@A�@W�BQ�B(�                                    BxW��  �          @ȣ�@q�@XQ�@j=qB=qB'Q�@q�@!�@�G�B-p�B�                                    BxW��  �          @ҏ\@��@xQ�@�A�  B{@��@Vff@3�
A�ffB	G�                                    BxW�*<  �          @�  @���@�z�?�Ayp�B*Q�@���@y��@-p�A��B��                                    BxW�8�  !          @�G�@��H@�ff?�
=Ae�B*z�@��H@\)@%A��B
=                                    BxW�G�  �          @�Q�@��\@��?���A~=qB4@��\@�33@2�\AÙ�B'z�                                    BxW�V.  T          @�G�@��
@�?�z�Ab�\B)33@��
@~{@%�A�ffB                                    BxW�d�  �          @�\)@�p�@���?���A[�
B$z�@�p�@vff@\)A���B(�                                    BxW�sz  �          @Ӆ@��
@i��?h��@�
=B�
@��
@W�?�
=Am�B ff                                    BxW��   �          @ʏ\@��\@W
=>�@��\B
=@��\@K�?���A.ffA��\                                    BxW���  �          @�
=@�=q@e?:�H@�{B�@�=q@Vff?��RAU�B                                     BxW��l  �          @��H@�(�@j�H?�  AQ�B	
=@�(�@W
=?��
Az�RA�                                    BxW��  �          @��H@�ff@`  ?^�R@�{B�R@�ff@N�R?�\)Al��A�Q�                                    BxW���  �          @��
@��@\��?5@��
B�@��@Mp�?�Q�AQA��                                    BxW��^  �          @�(�@���@`��>�
=@tz�B�@���@U�?�
=A+\)B z�                                    BxW��  �          @ƸR@�
=@U�>�
=@z�HB{@�
=@I��?�33A*=qA�
=                                    BxW��  �          @���@�p�@N�R>\@l(�B
=@�p�@Dz�?�=qA)�A�G�                                    BxW��P  �          @�33@��@N{=u?�B 
=@��@HQ�?E�@�G�A�{                                    BxW��  �          @�(�@�z�@"�\=���?�Q�A�ff@�z�@p�?(��@��\A�p�                                    BxW��  T          @�{@��@,�ͽ�G���
=A�@��@)��?   @�  A�z�                                    BxW�#B  �          @�  @���@���=q��ffA��
@���@*=q��G��*�RA�{                                    BxW�1�  T          @�ff@���@P  �u� ��B=q@���@N{>��H@�  BG�                                    BxW�@�  T          @�=q@�{@0�׿�  �G�
A�@�{@>�R�(����A�Q�                                    BxW�O4  �          @�z�@���@8Q쿈���'�A�z�@���@C33�����x��A�                                    BxW�]�  T          @��
@�  @6ff��33�.�HA�@�  @A녾���=qA�                                      BxW�l�  �          @�{@�Q�@J�H�u��A���@�Q�@Fff?(��@��A�ff                                    BxW�{&  T          @�\)@�
=@X��>�
=@w
=B��@�
=@L��?�A-p�A�z�                                    BxW���  �          @��H@���@Tz�>�@��HBG�@���@HQ�?���A6=qA�Q�                                    BxW��r  �          @�G�@�G�@Q�>���@r�\B(�@�G�@Fff?���A-�A���                                    BxW��  �          @��H@��
@P��>�p�@^�RB�@��
@E?���A&�\A���                                    BxW���  T          @���@�Q�@3�
�����'33A�R@�Q�@>�R��(����HA�z�                                    BxW��d  �          @Ǯ@�G�@"�\��G���33A��@�G�@7������*{A�                                      BxW��
  �          @ə�@���@,�;����HA�z�@���@/\)>��?�z�A���                                    BxW��  T          @�=q@�Q�@$zᾏ\)�   A�=q@�Q�@#�
>��
@9��A��                                    BxW��V  �          @���@�ff@)����G���  A�\)@�ff@&ff?   @�=qA�ff                                    BxW���  T          @���@�  @{�����A��@�  @p�>��
@:�HA�\)                                    BxW��  �          @�
=@��?�Q쿫��O33A�G�@��?����h�����A���                                    BxW�H  T          @�G�@�(�?�׿����#
=A�  @�(�@�������A�(�                                    BxW�*�  "          @��@�(�?�
=���\��HA���@�(�@��
=q��z�A��R                                    BxW�9�  T          @���@�?�ff��R���RA�
=@�?�33�B�\���A��                                    BxW�H:  �          @�
=@��@   �L������A�p�@��@'��L�Ϳ��HAѮ                                    BxW�V�  �          @��@�z�?Ǯ�8����Q�A�
=@�z�@(������\)A��\                                    BxW�e�  �          @�(�@�z�?��R�\)�$�HA{�
@�z�@Q��e��\A�p�                                    BxW�t,  T          @���@�=q>�ff�c33�{@�33@�=q?��\�Vff���Ak�
                                    BxW���  �          @�p�@��R?����W
=��AL��@��R?���@  ���A�{                                    BxW��x  �          @��H@�p�@p��K���{A�ff@�p�@HQ�� ����ffB
=                                    BxW��  �          @��@��?�33�Vff� 33A���@��@(Q��333��\)A�z�                                    BxW���  �          @�@�=q�L����Q��+=qC���@�=q?@  �|(��'��A�                                    BxW��j  T          @��@�
=���~{�-{C�*=@�
=?Q��x���(��A/\)                                    BxW��  T          @\@����R����)p�C��)@��>�33���H�+{@�33                                    BxW�ڶ  �          @��@�ff��p��\)�!  C��H@�ff?
=�}p���R@�33                                    BxW��\  �          @���@�(���\��Q��#��C�f@�(�?���Q��#�@�G�                                    BxW��  
�          @�ff@�  ?=p��\)��A=q@�  ?�z��l���G�A�G�                                    BxW��  �          @�=q@��?��|(���H@��
@��?��R�l(��  A�=q                                    BxW�N  T          @��@�33?���<(����
A�  @�33@�\�{��A�\)                                    BxW�#�  T          @�33@��?����G����A�  @��@#33�%���A��                                    BxW�2�  T          @Ϯ@��\@��XQ���\)A�  @��\@;��0���ɅA                                    BxW�A@  �          @�  @�z�@ff�Mp�����Aģ�@�z�@C33�#33��
=A���                                    BxW�O�  �          @�p�@��@,���[���Aޣ�@��@\(��,(���ffB�H                                    BxW�^�  
�          @��@���@X���b�\��Q�A���@���@�z��(Q���z�B��                                    BxW�m2  T          @��@���@��
�4z����B�@���@�p���
=�G33B!Q�                                    BxW�{�  �          @أ�@�(�@_\)��H���RB=q@�(�@~{��  �d(�B(��                                    BxW��~  �          @�G�?(�@ҏ\?W
=@�Q�B�z�?(�@�@=qA��B���                                    BxW��$  �          @�ff�fff@��
@(��A�Q�B�  �fff@�G�@�33B��B�\)                                    BxW���  �          @�  ��33@�G�@�\A��\B�G���33@��@g�BQ�B�{                                    BxW��p  T          @����@���>.{?��
B�L;��@�z�?��HAt��B��\                                    BxW��  �          A���33@�
=?�G�@�RB�{��33@�\)@:�HA�(�B�ff                                    BxW�Ӽ  �          @�p�@i��@�=q��Q��>=qBGG�@i��@�\)���Ϳ�  BK\)                                    BxW��b  �          @�@�ff=��������?Q�@�ff?+�����p�@�
=                                    BxW��  �          @�ff@У�?�ff�Q���33AU��@У�@G���Q��^ffA�                                    BxW���  T          @���@���@��H�k���z�BF�@���@�\)?�=qA�BDp�                                    BxW�T  T          AG�?˅@ᙚ@e�A��HB���?˅@��R@�ffB ��B�.                                    BxW��  �          A  ?���@�@  Au�B�W
?���@�z�@��
A��B�ff                                    BxW�+�  �          AG�@�
Aff?�  @љ�B�.@�
@���@B�\A�  B�Q�                                    BxW�:F  �          A33@{A�R?�A,z�B�L�@{@�  @l��A�p�B���                                    BxW�H�  �          A\)@33A��@{A�  B�\@33@�@�{A�B���                                    BxW�W�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�f8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�t�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�$�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�3L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�A�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�_>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�m�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�|�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��n              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW� `              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�,R              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�:�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�I�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�XD              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�f�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�u�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�%X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�3�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�B�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�QJ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�n�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�}<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��l              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�-              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�;�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�JP              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�X�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�g�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�vB              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�&
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�4�  r          Az�L��@X���p���B��\�L��@�����33�U�\B��                                    BxW�CV  
�          A��  ?����H¡�Ck���  @a������Bӣ�                                    BxW�Q�  "          A(�?�  @�33��\)�M
=B���?�  @��H��ff�33B��)                                    BxW�`�  �          A#�?�ff@�Q���ff�M�\B��3?�ff@��
�����{B�L�                                    BxW�oH  
�          A'�
��  @�G�����\B��3��  @�ff� z��H��B�L�                                    BxW�}�  "          A$�;aG�@�
=�
�R�p�B�\�aG�@�(���Q��1��B��                                    BxW���  
�          A$��@33@���׮�$  B��\@33A\)�����\)B�=q                                    BxW��:  "          A)?��@����  �@�
B�Q�?��A  ����{B��                                    BxW���  
�          A)��?�@�{�33�q��B�ff?�@�\)��ff�3=qB�=q                                    BxW���  T          A)����z�@�(�����x��B�k���z�@�
=���R�<\)B�                                    BxW��,  T          A+
=�u@��R��H�|  B�녿u@�\����>{B�\)                                    BxW���  
�          A*=q=u@�
=�(��rp�B��3=u@����  �3�B��                                    BxW��x  �          A(  ?aG�@ƸR�=q�T��B���?aG�A  �����z�B�Ǯ                                    BxW��  T          A#�?��@陚�ָR�&��B�
=?��A����\)���HB��                                    BxW��  
�          A"=q@s33@�\�������Bv��@s33A���  ��ffB���                                    BxW�j  �          A%�@8Q�@����  �#(�B��q@8Q�AG�������  B���                                    BxW�  
Z          A%�@Vff@ٙ��߮�*B(�@VffA
=��z���ffB��q                                    BxW�-�  �          A&{@Y��@ָR���-�B|p�@Y��A=q������ffB��
                                    BxW�<\  T          A'
=@j�H@����  �/�
Bs�@j�HAz���ff���HB�33                                    BxW�K  
�          A'33@c�
@�(���Q��0  Bv�@c�
A����ff��Q�B�                                    BxW�Y�  �          A'�
@QG�@أ���\�133B�u�@QG�A  ��
=��RB���                                    BxW�hN  �          A(z�@J=q@��
��=q�0G�B��{@J=qA	������  B���                                    BxW�v�  �          A((�@K�@���\)�-�B��3@K�A
=q���\��G�B��\                                    BxW���  �          A&=q@N�R@޸R��  �)ffB�(�@N�RA	����33��z�B�Ǯ                                    BxW��@  T          A%G�@X��@�������$33B�p�@X��A	p���(�����B���                                    BxW���  �          A)p�@���@�33�أ��
=Be33@���A�H�������By��                                    BxW���  T          A+\)@�@�z�����BuG�@�Az��xQ���Q�B�W
                                    BxW��2  �          A)��@�G�@�������\)Bm(�@�G�A�R�\����{B|
=                                    BxW���  T          A)G�@��
@�p���33� �Bk�@��
Aff�S�
��  Bz=q                                    BxW��~  �          A)��@�Q�@����=q����Bh�@�Q�A{�Q���(�Bw\)                                    BxW��$  �          A(Q�@�=q@�ff��p��  Bu�@�=qA���z=q��
=B���                                    BxW���  
Z          A'�@fff@�  �����(�B�.@fffA���n{��=qB��
                                    BxW�	p  
�          A*�R@�G�A Q����H����Bwp�@�G�A��L������B�.                                    BxW�  �          A-�@xQ�@�(���33���B|33@xQ�A{��������B�                                      BxW�&�  T          A-p�@i��@�G����+p�Bz�@i��Az���p���=qB��H                                    BxW�5b  �          A,��@i��@����Q��0�RBw(�@i��A	����(���\)B���                                    BxW�D  �          A,��@s33@����
=�'�Bv��@s33A(�������  B�\                                   BxW�R�  �          A-�@�=q@�G���R�&33Bo�H@�=qA�
������(�B�
=                                   BxW�aT  �          A/�
@�@����=q��HBvz�@�A�
���R��ffB�G�                                    BxW�o�  T          A0z�@��RAz���
=�  B|�@��RAp��`  ���B��{                                    BxW�~�  T          A0��@��RA  �����B|  @��RA���!��S�B��                                    BxW��F  
�          A-�@��A����H��ffB{=q@��Aff��\�-��B�#�                                    BxW���  
�          A'�@��A(���{����BrQ�@��Aff���2�RB|�                                    BxW���  �          A'\)@�(�A�R��33���HBt@�(�A\)��z���B}Q�                                    BxW��8  T          A((�@���A(��:=q����Bp��@���A���
=��\Bu��                                    BxW���  
Z          A%@�
=A
�R�.�R�t��Bp�@�
=AG���=q��  Bu=q                                    BxW�ք  T          A0(�@���A�R�����Bg  @���A�
?�G�@�=qBg��                                    BxW��*  �          A2=q@�{AQ��(���RB^�@�{Aff?L��@���B_��                                    BxW���  �          A0��@��RA���{���\BwG�@��RA���Mp����B���                                    BxW�v  �          A/
=@�A���=q��{Bip�@�A�\?�(�@�Q�Bh�
                                    BxW�  
�          A.�R@��AQ��Q��5�Bi�\@��A(�>���@��Bl33                                    BxW��  "          A733@��
A�\����Bc��@��
AQ�?s33@�G�Be
=                                    BxW�.h  T          A;
=@ϮA�H��G���B`��@ϮAp�?�z�A�HB_��                                    BxW�=  
�          AEG�@��
A$z῜(���Bd��@��
A#�?�\)@��HBd=q                                    BxW�K�  T          AR�H@��A0z�\���
Be��@��A0z�?��@�{Be��                                    BxW�ZZ  	�          AO�
@߮A,�׿��H��33Bd=q@߮A-��?��
@�p�Bd�
                                    BxW�i   
�          AP��@��HA/
=��Q��z�Bg��@��HA0��?�{@�(�Bi                                      BxW�w�  
�          AU@��
A4�׿���ffBj�R@��
A6{?�  @��Bk�                                    BxW��L  
�          AT��@ٙ�A2ff�#33�1�Bjff@ٙ�A6�R?�@��Bl�
                                    BxW���  "          AQ�@��HA0���'��8Q�Bl�R@��HA5p�>�@�Bo\)                                    BxW���  �          AQ�@θRA0Q��1��DQ�Bnff@θRA5>��
?�33Bqff                                    BxW��>  
Z          AS�
@ϮA1p��L���_�Bn��@ϮA8�׽�\)��z�Br��                                    BxW���  �          AU�@��HA2�H�L���]Bn
=@��HA:{�L�;B�\Bq�                                    BxW�ϊ  �          AR�R@љ�A-��h����z�Bk=q@љ�A6ff���{Bp�                                    BxW��0  �          AT��@�33A0Q��W��k
=Bl\)@�33A8Q쾀  ��=qBp��                                    BxW���  �          AR=q@��
A0z��(���9p�Bl
=@��
A5G�>�ff?�(�Bn                                    BxW��|  T          AS\)@ڏ\A1�
=�
=Bi�\@ڏ\A4(�?z�H@�Q�Bk                                      BxW�
"  "          AR�R@أ�A1G��Q����BjQ�@أ�A3�
?u@�Bk��                                    BxW��  �          AR=q@��A1p��=p��PQ�Bo�H@��A7�>��?+�BsG�                                    BxW�'n  
�          AH��@�\)A(���^{����Bu��@�\)A1p����(�Bz�R                                    BxW�6  �          A:�R@�=qA  �qG�����Bz�@�=qA&�\�}p���z�B�\                                    BxW�D�  
�          A.ff@��A(��n{���
B��=@��A�R��=q��p�B��                                    BxW�S`  
�          A+�@��
A��������33B�  @��
A�Ϳ�����  B�aH                                    BxW�b  
�          A0(�@�A��\����=qB��)@�A �ͿB�\�~{B�k�                                    BxW�p�  �          A=��@��RA%��W���Q�B��@��RA-p���ff��B�
=                                    BxW�R  T          A9�@��HA��\����(�B�8R@��HA(�׿�R�C�
B���                                   BxW���  
�          A>�R@�p�A$z��^{��G�B��H@�p�A-p��\)�+�B�.                                   BxW���  �          A733@��RA
=�S�
��B�Q�@��RA'��   �{B��\                                    BxW��D  �          A;
=@�33A"�R�B�\�qB�  @�33A)�8Q�fffB��H                                    BxW���  �          A5�@�p�A���XQ���z�Bv{@�p�A!���.{�[�B{z�                                    BxW�Ȑ  "          A5p�@�Q�A���o\)��B~z�@�Q�A$  ��G����HB�.                                    BxW��6  �          A>�H@���Ap��^{��\)Bm�H@���A&�\�0���Tz�Bsz�                                    BxW���  T          A=@�  A���X������Bn�@�  A%��R�?\)Bs�                                    BxW��  "          AB=q@��
A��L(��t  Bi�H@��
A'��\����Bn�R                                    BxW�(  �          AD(�@���A"�\�@���c�Bk33@���A)p��#�
�E�Boff                                    BxW��  �          AA�@���A"ff�-p��N�HBm
=@���A(  =�?z�Bpz�                                    BxW� t  T          AA�@��RA ���>�R�e�Bm\)@��RA'�
�.{�J=qBq�\                                    BxW�/  "          AFff@�33A$Q��S�
�w�
Bm�@�33A,z��(���(�Bq��                                    