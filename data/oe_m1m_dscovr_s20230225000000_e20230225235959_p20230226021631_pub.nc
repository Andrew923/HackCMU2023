CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230225000000_e20230225235959_p20230226021631_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-26T02:16:31.321Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-25T00:00:00.000Z   time_coverage_end         2023-02-25T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxh]��  �          @�
=��z�>��H�0�����RC.J=��z�?E���{�z�HC+�                                    Bxh]�f  "          @������R>��8Q��
=C.n���R?G���p���{C+�                                    Bxh^  
�          @�����ff>��333� Q�C.}q��ff?E���Q���=qC++�                                    Bxh^�  T          @�  ��p�?�Ϳ+�����C-����p�?O\)��z��Tz�C*�)                                    Bxh^#X  �          @�\)��p�?���R���C-����p�?E�����B�\C+�                                    Bxh^1�  T          @�{���
?녿���p�C-E���
?J=q�L���33C*��                                    Bxh^@�  
�          @�ff��z�?\)����G�C-n��z�?@  �#�
��C+E                                    Bxh^OJ  "          @�Q����R?   �   ���C.0����R?0�׾.{��(�C,{                                    Bxh^]�  
�          @�ff����?������=qC-�q����?8Q�L����C+��                                    Bxh^l�  �          @�
=��p�>�=q�0�����RC0�)��p�?z��ff��\)C-8R                                    Bxh^{<  T          @���(�>�=q�(����p�C0�\��(�?녾�(���
=C-W
                                    Bxh^��  �          @����
>�=q�5�  C0�����
?
=�����ffC-�                                    Bxh^��  �          @�
=����>�  �B�\�  C1�����?������Q�C,�q                                    Bxh^�.  T          @�{��33>�Q�=p��{C/���33?0�׾�ff��=qC+}q                                    Bxh^��  
Z          @������>k��u�P  C0�
���?0�׿5���C*��                                    Bxh^�z  
�          @�G���>�G��W
=�3�C-���?O\)���H�ҏ\C)�                                    Bxh^�   �          @��R���>����\)��C/����?
=q��������C,�{                                    Bxh^��  
Y          @��
��=q>���!G����C1ٚ��=q>�ff����\)C-��                                    Bxh^�l  
�          @��\��Q�>\)�=p��'
=C2���Q�?   ����
=C,�3                                    Bxh^�  #          @����G�>8Q�:�H�$Q�C1c���G�?��
=q��  C,�                                     Bxh_�  �          @����>���0���G�C1�3���>�����{C-T{                                    Bxh_^  T          @�����>.{�!G��C1�����>�׾�����C-h�                                    Bxh_+  T          @�=q����>���
=q����C0s3����?   ������33C-O\                                    Bxh_9�  �          @�=q����>��R����=qC/������?�;�����{C,�H                                    Bxh_HP  T          @�����>�33������C/T{���?(���������C+��                                    Bxh_V�  T          @��\��  ?.{����33C*���  ?Q녽L�Ϳ+�C)�                                    Bxh_e�  T          @�����ff>�33�����C/p���ff?�����R�|��C,c�                                    Bxh_tB  �          @�ff���
�u�O\)�(Q�C4�����
>�Q�:�H��\C/:�                                    Bxh_��  �          @��
����=�Q�@  ���C2�
����>������RC.8R                                    Bxh_��  �          @�{��(�>���(���Q�C0�\��(�?\)��(����HC,��                                    Bxh_�4  T          @����ff=u�}p��S�C3���ff?\)�Q��.�RC,u�                                    Bxh_��  T          @����p�>#�
��=q�g�C1ٚ��p�?.{�Y���5��C*�R                                    Bxh_��  �          @�
=��녽���z��pQ�C5�����>��H����Z�RC-�=                                    Bxh_�&  
�          @|���n�R���Ϳ�z���
=C:{�n�R>u������(�C0B�                                    Bxh_��  �          @�Q��g��.{�˅��(�C>�f�g�>aG����H��  C0�{                                    Bxh_�r  T          @r�\�U�.{��33��G�C?���U>�  ��\��33C/��                                    Bxh_�  
�          @u�]p�����ff��\)C<�q�]p�>��ÿ�{�ƣ�C.�{                                    Bxh`�  
�          @����p�׾��ÿ������\C9
=�p��>�녿�{���C-�q                                    Bxh`d  �          @����s33���Ϳ�(���C:  �s33>�=q��G����
C/�H                                    Bxh`$
  
�          @y���g
=���H��{��  C;�3�g
=>�=q������C/�                                     Bxh`2�  
�          @����s33�����������RC8u��s33>�녿������C-�{                                    Bxh`AV  
�          @���y���k����
��
=C7W
�y��>�ff��(���  C-h�                                    Bxh`O�  
Z          @����|(����
���R���RC8���|(�>�33��p���  C.�q                                    Bxh`^�  �          @����|�;��ÿ�(���ffC8�|��>��ÿ�(���ffC/8R                                    Bxh`mH  "          @�(��z�H������  ����C8p��z�H>�p����R���C.��                                    Bxh`{�  T          @�33�x�þ�����\����C7�{�x��>�
=��p����C-޸                                    Bxh`��  T          @�(��|�;��R��z�����C8�=�|��>��
��z����RC/p�                                    Bxh`�:  �          @��
�|(���33����C9
�|(�>�zῘQ����
C/�
                                    Bxh`��  �          @���z�H���
���H���RC8�)�z�H>�{������{C/�                                    Bxh`��  
�          @�33�xQ쾊=q������C7��xQ�>�
=��  ��(�C-�\                                    Bxh`�,  T          @��H�xQ�u��  ���RC7xR�xQ�>�(��������RC-��                                    Bxh`��  T          @�=q�xQ�B�\���H����C6�{�xQ�>�G���33���RC-u�                                    Bxh`�x  �          @xQ��e��.{�����G�C>�R�e�=�Q��  ���C2�)                                    Bxh`�  "          @~{�g��5��Q����RC?{�g�=�G���������C2=q                                    Bxh`��  T          @~�R�k���\��z���ffC;�3�k�>����p�����C/��                                    Bxhaj  �          @����p�׾Ǯ��\)���
C9���p��>�33�����G�C.�3                                    Bxha  �          @p���[���������p�C<���[�>�\)��G���{C/\)                                    Bxha+�  �          @p  �[���p�������  C:5��[�>��Ϳ�Q�����C-O\                                    Bxha:\  
�          @~{�h�þ�z�Ǯ��p�C8�\�h��?���  ��\)C+�                                    BxhaI  
�          @q��Z=q��G��У���=qC5�\�Z=q?8Q쿺�H��Q�C(�                                    BxhaW�  �          @e��Tzᾞ�R������C9@ �Tz�>\���\����C-h�                                    BxhafN  !          @g
=�U���ff�����33C;��U�>�=q��{��C/^�                                    Bxhat�  
)          @dz��P  �
=q������(�C=p��P  >aG���(���{C0�                                    Bxha��  T          @g
=�R�\��������Q�C=:��R�\>k���(����C0�                                    Bxha�@  "          @hQ��U��ff������C;��U>�\)�����C/:�                                    Bxha��  T          @i���U��G���33��(�C;u��U>��
��
=���\C.�\                                    Bxha��  �          @g
=�S�
�
=q�������C=W
�S�
>B�\����
=C0�                                    Bxha�2  "          @i���[��\�������HC:Q��[�>�=q��p����HC/�\                                    Bxha��  
�          @i���^�R���R������C9
=�^�R>����=q���C/��                                    Bxha�~  "          @j=q�b�\���h���f�\C6(��b�\>�{�\(��Xz�C.�\                                    Bxha�$  "          @r�\�fff���ÿ�������C98R�fff>�\)������\C/�)                                    Bxha��  
�          @w��h�þ�׿�(����RC;Y��h��>L�Ϳ�����
C0��                                    Bxhbp  T          @x���l�;�׿�������C;=q�l��>\)��
=���C1�H                                    Bxhb  �          @xQ��n{���Ϳ�G��r=qC:)�n{>#�
��=q����C1�{                                    Bxhb$�  T          @tz��i����G����
�{\)C:�\�i��>���{���C1��                                    Bxhb3b  !          @tz��i����׿��\�w\)C;B��i��=��Ϳ�\)��C2k�                                    BxhbB  �          @p  �e���s33�j�\C;�
�e=L�Ϳ�����C3#�                                    BxhbP�  T          @g��c33?+����
����C)=q�c33?!G�>��@���C)��                                    Bxhb_T  
�          @tz��mp�?k�<�>���C&��mp�?J=q>�@�ffC(
=                                    Bxhbm�  
�          @vff�n�R?s33<#�
>.{C%��n�R?Q�>�@�C'�{                                    Bxhb|�  T          @vff�o\)?n{<#�
=�C%���o\)?O\)>��@�\)C'�\                                    Bxhb�F  "          @r�\�k�?fff<��
>�p�C&Q��k�?E�>�@���C(:�                                    Bxhb��  �          @r�\�l��?Y��������C'��l��?@  >���@�Q�C(z�                                    Bxhb��  �          @s�
�o\)?B�\����z�C(�\�o\)?8Q�>��@|(�C)
                                    Bxhb�8  S          @q��l��?B�\���G�C(p��l��?8Q�>��@y��C(�                                    Bxhb��  #          @r�\�mp�?@  ����(�C(�{�mp�?8Q�>��@w�C){                                    BxhbԄ  �          @p  �k�?.{�.{�*=qC)�\�k�?.{>8Q�@333C)��                                    Bxhb�*  �          @p���l��?(�þ.{�!�C)�f�l��?&ff>8Q�@/\)C)�3                                    Bxhb��  �          @p���l(�?.{�W
=�J�HC)�)�l(�?0��>��@�
C)c�                                    Bxhc v  �          @p  �j=q?G���=q����C'�q�j=q?O\)>��@�RC'�                                     Bxhc  "          @s�
�mp�?J=q�Ǯ���C'�3�mp�?aG�=L��?333C&��                                    Bxhc�  �          @u��n{?E���G����C(^��n{?aG��#�
�#�
C&��                                    Bxhc,h  �          @s�
�n{?@  ��p���=qC(���n{?W
==L��?5C'L�                                    Bxhc;  �          @s�
�n{?B�\��33���C(z��n{?Tz�=u?uC'^�                                    BxhcI�  �          @p���j�H?@  ��  �w�C(xR�j�H?G�>��@�RC(�                                    BxhcXZ  �          @l���g
=?333�������C(�q�g
=?G�=L��?W
=C'��                                    Bxhcg   �          @n{�h��?#�
��(���{C*��h��?E��u��  C(
=                                    Bxhcu�  �          @mp��hQ�?(�þ�
=�θRC)�3�hQ�?G��#�
�&ffC'�H                                    Bxhc�L  �          @l���g�?+���33��
=C)���g�?B�\<��
>���C(0�                                    Bxhc��  �          @mp��h��?(�þ�z���p�C)�f�h��?8Q�=�\)?�33C(�\                                    Bxhc��  T          @i���e?
=��\)��ffC*�q�e?&ff=#�
?��C)�                                     Bxhc�>  �          @g
=�c33?zᾏ\)��C*���c33?#�
=#�
?#�
C)�3                                    Bxhc��  �          @g
=�b�\?&ff���R��
=C)��b�\?8Q�=L��?8Q�C(p�                                    Bxhc͊  �          @e�a�?zᾙ����Q�C*�)�a�?(��<��
>�p�C)}q                                    Bxhc�0  �          @dz��aG�>�׾�p���p�C,Y��aG�?
=��Q쿽p�C*s3                                    Bxhc��  �          @fff�c33>�G���(���33C,ٚ�c33?���#�
�"�\C*p�                                    Bxhc�|  �          @g��dz�>��������
C/G��dz�>��H��\)��(�C,.                                    Bxhd"  �          @g��a�?   ��R�p�C+���a�?=p�������
=C(0�                                    Bxhd�  �          @hQ��`  ?��L���L  C+s3�`  ?Y����ff���C&aH                                    Bxhd%n  �          @g
=�`��?�\�!G��"{C+��`��?B�\���R��(�C'ٚ                                    Bxhd4  �          @e��`��>�=q�&ff�'
=C/�=�`��?�;�G��߮C+
=                                    BxhdB�  �          @`���Z�H>�녿(���-��C-8R�Z�H?+��\��  C(�
                                    BxhdQ`  �          @^{�Z=q>\���33C-���Z=q?
=��\)���C*:�                                    Bxhd`  �          @`  �Z�H>�{�+��0z�C.aH�Z�H?�R��
=��
=C)�                                     Bxhdn�  
�          @^�R�Z=q>\���\)C-��Z=q?(����
����C)�                                    Bxhd}R  �          @`  �[�>�{�!G��%�C.k��[�?���Ǯ���C*�                                    Bxhd��  �          @aG��Z=q>�Q�E��I��C-��Z=q?0�׿   �=qC(�)                                    Bxhd��  �          @Z�H�U>������HC,#��U?+��������C(�R                                    Bxhd�D  �          @Vff�Q�?���
=����C*�f�Q�?(�ý��33C(�=                                    Bxhd��  
�          @Tz��N{>���R�+�C+�\�N{?333���
����C'�3                                    BxhdƐ  �          @Vff�P��?녾����\C*��P��?8Q���(�C'��                                    Bxhd�6  �          @W��QG�?��#�
�/
=C+
=�QG�?B�\���R���C&�                                    Bxhd��  �          @Z=q�P  ?
=�O\)�[\)C)���P  ?h�þ�
=��C$p�                                    Bxhd�  �          @\���S33?��E��O�
C*���S33?Tz��(�����C%��                                    Bxhe(  �          @`���Z=q?�\��R�"�\C+xR�Z=q?@  ������z�C'�                                    Bxhe�  T          @]p��U>��H�:�H�C�C+�{�U?G�����أ�C&��                                    Bxhet  �          @[��O\)?Tz�=p��I�C%�\�O\)?�������=qC!n                                    Bxhe-  �          @Y���O\)?333�:�H�FffC'Ǯ�O\)?u���R��C#n                                    Bxhe;�  �          @W��P��>��.{�:�RC+��P��?:�H�\��ffC'^�                                    BxheJf  �          @XQ��QG�?�\�&ff�333C+��QG�?B�\������p�C&�f                                    BxheY  �          @X���R�\?\)�(��&�RC*\)�R�\?G���=q����C&�)                                    Bxheg�  T          @Z�H�S33?녿.{�8  C*��S33?Tzᾨ�����C%�f                                    BxhevX  �          @X���O\)?녿W
=�f�RC*{�O\)?fff��� ��C$z�                                    Bxhe��  T          @Z�H�Q�?���5�?33C)���Q�?^�R��{��p�C%:�                                    Bxhe��  �          @X���QG�?녿.{�:�RC*��QG�?Tzᾨ������C%�\                                    Bxhe�J  �          @Y���R�\?   �(���3�C+J=�R�\?B�\��{���C'\                                    Bxhe��  �          @Z�H�S�
>��H�333�<��C+�{�S�
?B�\�Ǯ��
=C'�                                    Bxhe��  �          @[��Tz�?�Ϳ(��$��C*�
�Tz�?E���\)��ffC&�H                                    Bxhe�<  �          @_\)�W
=?�ͿG��M��C*��W
=?Y����(���G�C%�
                                    Bxhe��  �          @`���W�?
=q�J=q�Pz�C*���W�?Y����G���G�C%�                                    Bxhe�  �          @`���W
=?��Y���_
=C+(��W
=?\(��   �
=C%��                                    Bxhe�.  �          @aG��W�?
=q�c�
�iG�C*�{�W�?fff���	C%
                                    Bxhf�  �          @`���W
=?�\�c�
�k�
C+s3�W
=?^�R������C%�\                                    Bxhfz  �          @`  �W�>�G��W
=�]�C,� �W�?G�����C&�f                                    Bxhf&   �          @^{�Vff>�
=�J=q�RffC,ٚ�Vff?=p��   ���C'��                                    Bxhf4�  �          @^{�W�>�Q�B�\�IG�C-���W�?+��   �Q�C(��                                    BxhfCl  �          @]p��U>��ͿL���W\)C-@ �U?:�H���(�C'�q                                    BxhfR  �          @XQ��N{?��^�R�n=qC*�3�N{?^�R����C$�)                                    Bxhf`�  �          @Vff�Mp�>�(��Q��c�
C,Q��Mp�?E������C&��                                    Bxhfo^  �          @Vff�L(�>�G��c�
�v=qC,{�L(�?O\)�z��33C%�{                                    Bxhf~  �          @U��K�>�
=�c�
�y�C,z��K�?J=q����$��C&
                                    Bxhf��  �          @U�K�>��Ϳk��~�RC,ٚ�K�?G���R�,��C&@                                     Bxhf�P  �          @U��J=q>�ff�p����C+���J=q?W
=�!G��-��C%.                                    Bxhf��  T          @P���Dz�>��z�H��z�C+��Dz�?\(��(���:ffC$T{                                    Bxhf��  T          @Q��Fff>��k���33C+���Fff?W
=�(��*�RC$�                                    Bxhf�B  �          @R�\�Fff>��}p����\C+�=�Fff?^�R�+��;33C$Y�                                    Bxhf��  "          @U��H��>�ff�}p���Q�C+޸�H��?\(��+��9��C$Ǯ                                    Bxhf�  �          @Vff�I��>�G���ff��ffC+�R�I��?aG��:�H�H��C$c�                                    Bxhf�4  �          @Y���L��?�\���\���\C*���L��?n{�+��6�HC#�{                                    Bxhg�  T          @U�J=q>��u���C+s3�J=q?\(��!G��.�RC$��                                    Bxhg�  
�          @Vff�N{>�Q�^�R�o�
C-�
�N{?8Q����%�C'aH                                    Bxhg&  T          @Tz��K�>\�Y���m�C-+��K�?:�H�z�� ��C')                                    Bxhg-�  �          @W��N�R>�
=�\(��mp�C,���N�R?E����(�C&��                                    Bxhg<r  "          @Vff�N�R>Ǯ�L���\(�C-+��N�R?5���C'��                                    BxhgK  �          @S�
�K�>�33�L���ap�C-��K�?.{����=qC'��                                    BxhgY�  �          @Vff�L��>��
�p�����\C.L��L��?8Q�0���>ffC'ff                                    Bxhghd  �          @Z=q�QG�>�녿fff�s�
C,Ǯ�QG�?G��(��$z�C&��                                    Bxhgw
  �          @Y���P  >\�p������C-E�P  ?E��+��4Q�C&��                                    Bxhg��  T          @Z=q�N�R>�(��z�H����C,p��N�R?Tz�+��6�HC%��                                    Bxhg�V  T          @Y���N{?   �u��G�C+:��N{?aG��!G��*ffC$�R                                    Bxhg��  �          @Z�H�N{?!G��n{�{�C(�3�N{?z�H�
=q���C#\                                    Bxhg��  �          @\(��QG�?z�aG��l��C)�3�QG�?k����
�HC$\)                                    Bxhg�H  �          @X���N�R?\)�c�
�s
=C*8R�N�R?fff�
=q�ffC$n                                    Bxhg��  T          @Y���O\)?\)�Y���g
=C*5��O\)?aG��   ��C$                                    Bxhgݔ  "          @W��L��?\)�c�
�t��C*��L��?fff�
=q��HC$=q                                    Bxhg�:  
�          @W��Mp�?녿W
=�g�C)�H�Mp�?c�
���H�{C$xR                                    Bxhg��  
�          @Z�H�P  ?�R�W
=�c�
C)33�P  ?n{�����z�C#�q                                    Bxhh	�  �          @Z=q�P��?�ͿQ��]�C*^��P��?\(����G�C%0�                                    Bxhh,  T          @W
=�L��?   �\(��l��C+��L��?W
=�
=q��C%W
                                    Bxhh&�  T          @X���O\)?��Q��aC*� �O\)?W
=�   ��C%h�                                    Bxhh5x  �          @\���P��>���G����RC+���P��?aG��0���8z�C$�\                                    BxhhD  "          @\(��N{>Ǯ��33���
C-��N{?^�R�Y���dz�C$�3                                    BxhhR�  "          @Z=q�Mp�>�G���ff��G�C,!H�Mp�?^�R�=p��G�
C$�f                                    Bxhhaj  T          @XQ��L��>��xQ���\)C+s3�L��?\(��&ff�1C$�H                                    Bxhhp  �          @W��L��?녿c�
�t  C)���L��?fff�
=q�33C$=q                                    Bxhh~�  �          @Z�H�O\)>�G��z�H��p�C,8R�O\)?Tz�.{�7�C%�\                                    Bxhh�\  �          @\���N�R>�\)��
=��C/)�N�R?E��n{�z�\C&��                                    Bxhh�  
�          @Z=q�L��>��ÿ�����33C..�L��?G��W
=�d��C&@                                     Bxhh��  �          @Z=q�N�R>�ff�z�H���C,��N�R?W
=�.{�7\)C%c�                                    Bxhh�N  T          @X���Mp�?녿fff�u��C)�3�Mp�?h�ÿ���p�C$5�                                    Bxhh��  �          @Z=q�P  >Ǯ�u��\)C-#��P  ?G��0���:�RC&��                                    Bxhh֚  �          @W
=�L(�>k����
��=qC/�)�L(�?(�ÿQ��aG�C(G�                                    Bxhh�@  �          @Y���Mp�>\)��{���\C1n�Mp�?�R�n{�~�\C)
=                                    Bxhh��  
Z          @S�
�J=q>\)�u��G�C1p��J=q?�ͿL���`  C*{                                    Bxhi�  "          @S33�J�H>�G��E��W\)C,��J�H?:�H���H�	�C'                                      Bxhi2  "          @Vff�Mp�>�33�\(��n�RC-�{�Mp�?0�׿�R�*�RC'ٚ                                    Bxhi�  
�          @Vff�N�R>��
�W
=�hQ�C.W
�N�R?(�ÿ(��(��C(}q                                    Bxhi.~  T          @S�
�K�>�Q�O\)�b=qC-}q�K�?.{���p�C'��                                    Bxhi=$  
�          @Q��H��>�=q�\(��s\)C/��H��?�R�&ff�8(�C(�H                                    BxhiK�  
�          @P���G�>#�
�s33���C1��G�?\)�G��]G�C)�                                    BxhiZp  T          @Q��J=q>W
=�\(��t(�C0&f�J=q?녿.{�@(�C)�                                    Bxhii  
�          @S�
�J=q>W
=�s33���C0.�J=q?(��B�\�T��C)(�                                    Bxhiw�  �          @S�
�J=q>B�\�xQ���{C0�
�J=q?
=�J=q�\��C)aH                                    Bxhi�b  
�          @U�O\)>�\)�@  �O
=C/��O\)?z�����C)�)                                    Bxhi�  
�          @Vff�P  >��ÿ=p��K�C.��P  ?�R���=qC)&f                                    Bxhi��  �          @S33�Mp�>W
=�@  �QC033�Mp�?��z��"=qC*�{                                    Bxhi�T  T          @U��L��=�\)�h���|��C2�{�L��>�ff�J=q�Z�HC+��                                    Bxhi��  �          @U�L��=��
�s33����C2�=�L��>��H�Q��d��C+Q�                                    BxhiϠ  �          @P  �G�=�\)�h����Q�C2��G�>��J=q�aG�C+�)                                    Bxhi�F  
�          @U�L(�=u�}p�����C2��L(�>��\(��p  C+h�                                    Bxhi��  "          @XQ��N{=��
��G����C2���N{?�\�aG��q��C*��                                    Bxhi��  T          @X���N�R����  ��C4z��N�R>�녿k��{
=C,��                                    Bxhj
8  
�          @Z�H�P  �L�Ϳ��
��
=C4�H�P  >��Ϳs33���
C,��                                    Bxhj�  "          @Vff�L(��#�
�����
=C4��L(�>�
=�u����C,��                                    Bxhj'�  
�          @X���N{�L�Ϳ�ff���RC4���N{>�녿xQ���G�C,�=                                    Bxhj6*  
�          @W
=�L�ͽ#�
���
��33C4Ǯ�L��>�녿s33����C,�=                                    BxhjD�  S          @X���N�R���
���\��C5h��N�R>�Q�s33���RC-��                                    BxhjSv  	�          @[��QG�������\���C6���QG�>����z�H��33C.�                                    Bxhjb  
(          @Z�H�O\)�#�
������HC6� �O\)>��R���
���
C.z�                                    Bxhjp�  
�          @\(��P�׾B�\��=q��=qC7J=�P��>�zῆff��
=C.�f                                    Bxhjh  "          @W��L�;8Q쿆ff��z�C7L��L��>�\)���\��p�C/                                    Bxhj�  "          @O\)�C33��\)�����{C5Q��C33>Ǯ�}p���\)C,��                                    Bxhj��  �          @QG��G���\)�}p�����C5\)�G�>�33�n{��C-�                                    Bxhj�Z  	�          @R�\�G��#�
���\��z�C4�R�G�>��Ϳp�����HC,                                    Bxhj�   
�          @Vff�L(����
���
��G�C5s3�L(�>�Q�xQ���=qC-z�                                    BxhjȦ  T          @Z�H�P  ����ff���C6:��P  >��ÿ�G����RC.5�                                    Bxhj�L  T          @W
=�N{����n{��ffC8���N{>���s33���
C1E                                    Bxhj��  T          @W
=�J=q���
�����=qC5aH�J=q>�녿������C,�
                                    Bxhj��  
�          @U��I���\)�����Q�C6���I��>��ÿ�ff��Q�C.
=                                    Bxhk>  	�          @XQ��L�;\)��=q��33C6��L��>��ÿ����\)C.(�                                    Bxhk�  
�          @_\)�S�
�aG�������HC7�{�S�
>����=q���
C/�
                                    Bxhk �  
�          @c33�W��k���{��(�C7��W�>����{��p�C/��                                    Bxhk/0  
�          @c�
�W��u��\)���\C7�q�W�>�  ��{��  C/��                                    Bxhk=�  
�          @dz��XQ쾊=q�������C8� �XQ�>aG���{����C0Q�                                    BxhkL|  
�          @fff�Z�H���������\C8h��Z�H>W
=������  C0k�                                    Bxhk["  
�          @dz��X�þ��������  C9  �X��>.{��=q���C1)                                    Bxhki�  �          @dz��X�þ�33�������C9���X��=���\)���C1�                                    Bxhkxn  T          @dz��X�þ���������
C9!H�X��>#�
�����  C1B�                                    Bxhk�  	�          @\(��S33��\)�n{�{�C8�=�S33>\)�u��  C1�f                                    Bxhk��  �          @c�
�W���{�������C9���W�>���������\C1�                                    Bxhk�`  �          @e�Z=q��zῈ������C8��Z=q>8Q쿌����=qC1\                                    Bxhk�  
�          @h���[����R��33��33C90��[�>B�\�����RC0�{                                    Bxhk��  �          @a��U���Q쿏\)��Q�C:&f�U�>\)������C1�f                                    Bxhk�R  	�          @`���P  ��녿�G���(�C;&f�P  >��������  C1h�                                    Bxhk��  U          @_\)�P  ��{��(���{C:��P  >B�\��G�����C0�                                    Bxhk�  �          @b�\�Vff���
�������\C9^��Vff>#�
��������C133                                    Bxhk�D  	�          @`���U����
��ff���C9� �U�>\)������HC1��                                    Bxhl
�  
)          @^�R�U���p��k��u�C:c��U�=#�
�}p����HC3^�                                    Bxhl�  
�          @c�
�W
=��{������{C9�=�W
=>\)�����  C1�f                                    Bxhl(6  T          @g
=�\(����ÿ�ff��33C9z��\(�>��������C1�H                                    Bxhl6�  
Z          @g
=�\(���p���G����
C:
�\(�=��
������(�C2�                                     BxhlE�  
�          @j=q�^�R�\������RC:(��^�R=�Q쿏\)���RC2�\                                    BxhlT(  
�          @l���a녾�G���G��~{C;
�a�<#�
�������HC3�
                                    Bxhlb�  �          @n{�c33��G���G��{33C;
�c33    �������C3�                                    Bxhlqt  �          @o\)�b�\�   ��{��=qC<��b�\�#�
��(���  C4)                                    Bxhl�  
[          @o\)�b�\��\������Q�C<@ �b�\�����H����C4n                                    Bxhl��  !          @mp��`�׿�\������ffC<5��`�׼��
���H���RC4J=                                    Bxhl�f  T          @l���_\)�녿�=q��\)C=Y��_\)��Q쿜(����C5�                                    Bxhl�  �          @n�R�`  ��R������RC=���`  ����  ���HC6
=                                    Bxhl��  "          @u��hQ�#�
���\�xz�C>��hQ�8Q쿙�����C6�                                    Bxhl�X  
�          @q��c�
�&ff������  C>J=�c�
�#�
���R���C6�)                                    Bxhl��  T          @p  �b�\�.{�����
=C>�)�b�\�L�Ϳ��R���\C733                                    Bxhl�  �          @s�
�e�5�����
=C?:��e�k���G����C7��                                    Bxhl�J  
Z          @tz��g
=�0�׿���}�C>��g
=�W
=��p���(�C7ff                                    Bxhm�  �          @u�hQ�5�����(�C?��hQ�k���G���ffC7�{                                    Bxhm�  �          @x���i���B�\��{���HC?� �i����  ��������C7��                                    Bxhm!<  
�          @y���i���G���{���RC@��i����=q��=q��C8J=                                    Bxhm/�  "          @w
=�g��E���{���C@��g���=q������{C8G�                                    Bxhm>�  
�          @g��Tz�n{�������CC���Tz��
=��\)��Q�C;J=                                    BxhmM.  
�          @c�
�N�R�p�׿�33��33CD(��N�R��녿�
=��p�C;&f                                    Bxhm[�  �          @e��Q녿\(���\)��z�CB�q�Q녾�33��\)����C:)                                    Bxhmjz  #          @k��U��u��33����CD#��U���(���Q���p�C;ff                                    Bxhmy   �          @q��X�ÿk������G�CCE�X�þ�����33�Ώ\C9�                                    Bxhm��  �          @u��_\)�^�R�������HCA�R�_\)��\)��ff��33C8��                                    Bxhm�l  �          @y���e�O\)���
���HC@��e��  ���R��(�C7�R                                    Bxhm�  �          @vff�a녿W
=���\���CAc��a녾�\)���R����C8��                                    Bxhm��  T          @u�a녿Q녿��\���RCA��a녾�����R���RC8.                                    Bxhm�^  �          @vff�a녿^�R���\��  CA���a녾��R��G����\C9�                                    Bxhm�  "          @x���dz�aG���  ��z�CA���dzᾨ�ÿ�  ��\)C9=q                                    Bxhmߪ  �          @xQ��b�\�^�R������CAٚ�b�\������ff��  C8�=                                    Bxhm�P  	�          @xQ��b�\�aG�������p�CB�b�\�����Ǯ���
C8��                                    Bxhm��  
�          @x���e�h�ÿ�Q�����CB.�e�\�������C9�R                                    Bxhn�  T          @y���g��\(���
=���RCAY��g���{������C9Q�                                    BxhnB  "          @y���i���G�������Q�C@
�i����zΎ����HC8�                                     Bxhn(�  �          @x���g��O\)��z����HC@���g�������������C8�\                                    Bxhn7�  T          @vff�e��G������C@O\�e���=q������z�C8T{                                    BxhnF4  
)          @q��aG��E���33��G�C@W
�aG���=q�������C8h�                                    BxhnT�  �          @r�\�aG��L�Ϳ�33���C@�=�aG�������\)���C8�
                                    Bxhnc�  
�          @s33�a녿J=q��
=��
=C@�f�a녾�\)��33��Q�C8��                                    Bxhnr&  �          @tz��^�R�\(�������\)CA���^�R��z��ff����C8��                                    Bxhn��  �          @q��Z=q�h�ÿ�\)����CB��Z=q���
��{�ə�C9\)                                    Bxhn�r  �          @q��[��Y����������CA�H�[���=q�������
C8��                                    Bxhn�  �          @tz��^�R�J=q��\)��  C@�q�^�R�W
=�Ǯ��p�C7n                                    Bxhn��  
�          @tz��_\)�J=q��=q���C@Ǯ�_\)�k����
����C7�R                                    Bxhn�d  
Z          @tz��^{�L�Ϳ������
C@�R�^{�aG���=q���
C7�{                                    Bxhn�
  #          @u��\�ͿE���(���Q�C@���\�;#�
��33��{C6��                                    Bxhnذ  !          @tz��a녿=p���(���p�C?��a녾aG���z���p�C7��                                    Bxhn�V  
�          @tz��e��0�׿�{����C>ٚ�e��aG������  C7}q                                    Bxhn��  �          @z�H�l�Ϳ(�ÿ�=q��Q�C>!H�l�;W
=��  ���C75�                                    Bxho�  T          @{��mp��.{��\)���C>aH�mp��W
=������C7G�                                    BxhoH  
�          @z=q�hQ�333���
���\C>��hQ�.{������C6��                                    Bxho!�  "          @z=q�`�׿E����
��ffC@h��`�׾�����H��\)C6aH                                    Bxho0�  �          @|(��aG��=p���=q��  C?�{�aG����Ϳ�  �ҸRC5��                                    Bxho?:  �          @�G��e��Q녿�z��\C@�H�e��\)������  C6J=                                    BxhoM�  
�          @�=q�hQ�:�H�ٙ���p�C?^��hQ�L�Ϳ�����(�C4��                                    Bxho\�  �          @��H�j=q�B�\��33����C?��j=q���Ϳ�����33C5��                                    Bxhok,  �          @���l(��:�H�����=qC?8R�l(���G��ٙ���{C5�R                                    Bxhoy�  	�          @����l(��&ff��  ���\C=���l(��L�ͿУ���
=C4Ǯ                                    Bxho�x  "          @~�R�g��:�H��  ���RC?\)�g�����z����C5�R                                    Bxho�  �          @\)�h�ÿ8Q쿺�H��C?33�h�þ���\)��{C6�                                    Bxho��  T          @�G��k��E����R����C?޸�k��.{������C6��                                    Bxho�j  �          @����k��333���H����C>�q�k���G���\)����C5�                                    Bxho�  
�          @�G��l(��B�\��Q�����C?���l(��8Q��{����C6��                                    BxhoѶ  T          @~�R�fff�5���
���C?!H�fff���Ϳ�Q�����C5�{                                    Bxho�\  �          @}p��hQ�:�H�������C?O\�hQ�.{�Ǯ���
C6�                                    Bxho�  T          @{��fff�=p���33����C?���fff�8Q�������RC6޸                                    Bxho��  
�          @w��c�
�8Q쿫�����C?z��c�
�B�\��G���Q�C6��                                    BxhpN  
�          @vff�e�&ff��33���\C>5��e�8Q쿧���=qC6�3                                    Bxhp�  "          @u�`�׿zῷ
=��p�C=^��`�׼��
��ff��{C4\)                                    Bxhp)�  T          @s33�Y���.{������HC?Y��Y�����
��
=��  C5h�                                    Bxhp8@  "          @u��\(��8Q��  ��33C?ٚ�\(�����z���(�C633                                    BxhpF�  
[          @r�\�[��333��(����C?���[�����\)��(�C6�                                    BxhpU�  �          @q��Z�H�.{������Q�C?&f�Z�H��G��˅�Ǚ�C5ٚ                                    Bxhpd2  
Z          @tz��^�R�&ff��Q���p�C>�)�^�R��Q��=q��p�C5�                                    Bxhpr�  �          @s�
�\(��.{��(���=qC?8R�\(���G���\)�ə�C5ٚ                                    Bxhp�~  
�          @vff�`  �!G���p���
=C>G��`  �u��{��C5�                                    Bxhp�$  �          @vff�`  �\)��  ���C=)�`  <#�
�������HC3�
                                    Bxhp��  
�          @y���dz�   ���R��\)C;��dz�=u�������C3                                      Bxhp�p  
�          @vff�aG���녿�p���\)C:�{�aG�>�����
����C1��                                    Bxhp�  �          @tz��aG�����33��Q�C<W
�aG�<#�
��  ��=qC3�                                    Bxhpʼ  �          @r�\�`�׿�\�������C<W
�`�׼#�
��
=��  C4!H                                    Bxhp�b  �          @q��`�׿
=���\���HC=���`�׽�G���33��p�C5�                                    Bxhp�  �          @~�R�h�ÿ녿��R��z�C<���h�ü#�
������C40�                                    Bxhp��  T          @���mp��0�׿�  ���HC>� �mp����������C5�
                                    BxhqT  �          @y���e��:�H�����p�C?xR�e��W
=��G����HC7Y�                                    Bxhq�  
�          @|(��hQ�#�
�����\)C>��hQ��G����
��ffC5��                                    Bxhq"�  "          @����l(��G����H����C@  �l(��k������G�C7��                                    Bxhq1F  !          @�G��j�H�L�Ϳ��R��33C@G��j�H�k�����  C7�{                                    Bxhq?�  T          @�G��k��5���
���C>޸�k�����
=��{C6
=                                    BxhqN�  �          @����o\)�O\)�����ffC@+��o\)��z�����(�C8u�                                    Bxhq]8  T          @z=q�c33�Q녿���ffC@�q�c33��\)��\)��G�C8s3                                    Bxhqk�  �          @w
=�^{�J=q��  ����C@�=�^{�aG���
=��ffC7�H                                    Bxhqz�  T          @w
=�]p��E����
��Q�C@�=�]p��L�Ϳٙ�����C7=q                                    Bxhq�*  �          @w��_\)�E���p���G�C@z��_\)�aG���33��{C7��                                    Bxhq��  �          @w
=�]p��8Q�����z�C?�R�]p��\)�ٙ���z�C6c�                                    Bxhq�v  �          @w
=�]p��G��\����C@���]p��W
=��Q���C7xR                                    Bxhq�  
�          @xQ��`  �&ff������C>���`  ���
�����C5\)                                    Bxhq��  
�          @u��\(��5���R���C?���\(��#�
������
C6�f                                    Bxhq�h  �          @tz��`  �G���������C@��`  ��\)��G�����C8��                                    Bxhq�  "          @y���dz�TzΎ���ffCA�dzᾣ�
������
C9&f                                    Bxhq�  �          @�G��j=q�c�
��Q����CA�q�j=q��33��z��\C9}q                                    Bxhq�Z  �          @�=q�g
=�����ff���CD  �g
=��ff���ӮC;{                                    Bxhr   T          @~{�e��.{��=q��33C>���e����Ϳ��H��z�C5��                                    Bxhr�  �          @\)�dz�:�H�У����C?���dz�����
��(�C6#�                                    Bxhr*L  T          @z�H�Z�H�xQ��z����HCC�f�Z�H��Q��33���C9�3                                    Bxhr8�  �          @x���W��O\)��G��ׅCAs3�W��.{��
=��{C6ٚ                                    BxhrG�  
�          @z�H�\�ͿO\)����  CAE�\�;W
=������
=C7k�                                    BxhrV>  "          @�=q�hQ�n{��������CBn�hQ쾳33������  C9p�                                    Bxhrd�  �          @����n�R��  ��  ��  CB��n�R��G���  �ǮC:�                                    Bxhrs�  
�          @�p��n{����\��\)CDT{�n{����ff��z�C;�R                                    Bxhr�0  �          @����j�H��Q����ffCF�j�H��\�����C;�3                                    Bxhr��  �          @����l�Ϳ�(�����CFJ=�l�Ϳ������{C<z�                                    Bxhr�|  �          @����l(����
��ff�ƣ�CG!H�l(��(��Q���33C=^�                                    Bxhr�"  
�          @���e��
=��  ���CI���e�E�����ffC@�                                    Bxhr��  
�          @����hQ�\��33���\CJ���hQ�c�
��
��G�CA�R                                    Bxhr�n  �          @�=q�hQ�Ǯ��(���(�CK5��hQ�fff������
CA��                                    Bxhr�  
�          @���S33��33�z�� ��CN���S33�J=q�.�R�p�CA}q                                    Bxhr�  �          @�(��E��  �-p���CM���E���C33�0�C=��                                    Bxhr�`  
�          @�{�E���  �3�
�=qCM�R�E��   �H���4��C=5�                                    Bxhs  �          @�p��J=q��\)�'���RCO��J=q�+��@���+�C@�                                    Bxhs�  �          @���P  ��z��\)�	(�CO�P  �B�\�8���#�CA�                                    Bxhs#R  
�          @���S�
��33������CN���S�
�G��333�{CAQ�                                    Bxhs1�  
�          @��R�P  ��
=�#�
���COW
�P  �B�\�>{�&��CA!H                                    Bxhs@�  "          @��U��\�   �	p�CL��U��!G��7
=� ��C>�f                                    BxhsOD  �          @��N�R��ff�&ff��CM�)�N�R�!G��=p��'��C>�                                    Bxhs]�  �          @��K���  �,(��z�CMG��K��\)�A��,��C=�f                                    Bxhsl�  
�          @�z��S33���H�   �
=CK�3�S33�z��5�!Q�C=�q                                    Bxhs{6  
�          @�(��U���33��R�	CJ���U����333�C=�                                    Bxhs��  	�          @����Y�����
�� Q�CL5��Y���333�,�����C?��                                    Bxhs��  �          @�
=�\�Ϳ˅���G�CL�q�\�ͿB�\�.�R��RC@^�                                    Bxhs�(  �          @�z��L�Ϳ�=q�!��\)CN=q�L�Ϳ0���9���&33C@�                                    Bxhs��  "          @�z��I����  �*=q��CMu��I���z��@  �,\)C>\)                                    Bxhs�t  �          @�G��G
=��
=�%���CL�R�G
=���:=q�*��C=��                                    Bxhs�  �          @����E���H�%���CMaH�E���;��+�C>c�                                    Bxhs��  �          @�Q��C�
����!���HCN�q�C�
�(���8���*�HC@.                                    Bxhs�f  �          @����G
=��=q�{�Q�CN���G
=�8Q��5�&�CA�                                    Bxhs�  �          @��O\)�޸R��(����HCP@ �O\)���
��H�p�CE�                                    Bxht�  �          @���Mp���
=��Q���ffCS{�Mp���ff�p��\)CJ)                                    BxhtX  T          @���AG���z��\)��CM��AG��\)�3�
�)��C>u�                                    Bxht*�  �          @�{�H�ÿ�
=�
=�	�\CLz��H�ÿ�R�,(�� =qC?!H                                    Bxht9�  T          @�
=�G
=����   �=qCL
=�G
=���3�
�&C=                                    BxhtHJ  
Z          @����G�����'
=��CK���G��   �:=q�*�C=�                                    BxhtV�  
�          @���>�R��=q�333�"�CL��>�R�Ǯ�E��6��C;xR                                    Bxhte�  �          @���;�����=p��*��CL��;������N{�>
=C:c�                                    Bxhtt<  �          @��H�7���G��@  �/
=CK���7�����O\)�AffC95�                                    Bxht��  �          @��H�@  �����5�#Q�CK�\�@  �\�G
=�7  C;B�                                    Bxht��  T          @����E����*�H�Q�CJ���E�����<(��-\)C;u�                                    Bxht�.  
\          @���I����ff�(����
CJn�I����(��:=q�)��C;�                                    Bxht��             @����J�H����%����CJ(��J�H��(��6ff�&C;��                                    Bxht�z  "          @����@�׿����/\)��CK�H�@�׾���@���2�
C;�{                                    Bxht�   	�          @�\)�>{����,(���CLc��>{���>{�2�
C<                                    Bxht��  
�          @�z��:�H��{�&ff���CM��:�H�   �8���1Q�C=��                                    Bxht�l  T          @�ff�;���{�+����CL�R�;����=p��3��C=Y�                                    Bxht�  "          @�ff�8�ÿ�{�.�R�"Q�CM.�8�þ��AG��7�\C={                                    Bxhu�  
�          @�p��6ff��ff�0  �$��CLp��6ff�����@���9
=C<                                      Bxhu^  
�          @�z��1녿��
�1��)�CL�3�1녾\�B�\�=(�C;�                                    Bxhu$  
�          @�(��2�\��G��1G��(��CL8R�2�\��Q��AG��<{C;T{                                    Bxhu2�  
�          @����/\)��p��7
=�.��CL��/\)�����Fff�Ap�C:T{                                    BxhuAP  T          @�p��(�ÿ����?\)�7Q�CLu��(�þ�  �Mp��I�RC9T{                                    BxhuO�  T          @���(�ÿ�z��:�H�5=qCK� �(�þk��HQ��F��C9�                                    Bxhu^�  "          @�  �%��=q�6ff�5�HCJ���%�8Q��B�\�FQ�C7�                                    BxhumB  
�          @w��ff���
�8���ACK���ff���Dz��R33C6�H                                    Bxhu{�  �          @u�p��z�H�1G��9=qCI�R�p���G��<(��H�C6�                                     Bxhu��  "          @~{�7
=���
�#�
� �CG� �7
=�aG��0  �/=qC8T{                                    Bxhu�4  �          @~{�9����G��!G��=qCGB��9���aG��,���+��C8T{                                    Bxhu��  $          @~�R�?\)�xQ������CE���?\)�L���'
=�$=qC7�f                                    Bxhu��  R          @|(��Mp��Y���
=�z�CBٚ�Mp��B�\�����C7^�                                    Bxhu�&  V          @|���G��E���\���CA�H�G��u��H�
=C5
                                    Bxhu��  �          @�Q��K��L���33�33CB
�K����
�����C5�                                    Bxhu�r  �          @�Q��C�
�O\)����(�CB���C�
�u�%�� ��C5
=                                    Bxhu�  "          @��\�!G��Q��E�E�CE�3�!G�>��L���N�C1�                                    Bxhu��  
�          @�33�33�fff�Q��S��CIY��33=����Y���_z�C1}q                                    Bxhvd             @��\����\�J�H�L��CK����L���U��[�C5L�                                    Bxhv
  "          @���"�\�z�H�Dz��A��CI&f�"�\�#�
�N�R�O�C4�R                                    Bxhv+�  
�          @�
=�@  �B�\�7��+(�CB.�@  =�G��=p��2p�C1�                                    Bxhv:V  �          @�\)�;����?\)�4Q�C?
=�;�>��
�A��7(�C-�3                                    BxhvH�  �          @���(Q�\)�J�H�F�C?��(Q�>Ǯ�L���I
=C+��                                    BxhvW�  	�          @�\)�$z�
=q�S33�N
=C?�=�$z�>�G��Tz��O=qC*=q                                    BxhvfH  "          @�����\��
=�e�c��C>p���\?!G��c33�`�C$�
                                    Bxhvt�  T          @�Q��\)�L���u��p�C9�R��\)?k��n{�s(�C�H                                    Bxhv��  �          @��Ϳٙ��aG������C;LͿٙ�?s33�z�H�~\)C�=                                    Bxhv�:  "          @��������z�#�C7p���?�=q��  �t��C�3                                    Bxhv��  �          @��R�Z�H��G�����	�HCD}q�Z�H��=q�(���(�C8}q                                    Bxhv��  	�          @�Q���
=���������Pz�CC���
=�Y����\)��(�C>��                                    Bxhv�,  
�          @���ff�����(���\)CB����ff��Ϳ��H���HC;u�                                    Bxhv��  "          @����{��Q녿��R�ҏ\C?ٚ�{��k������{C7G�                                    Bxhv�x  �          @�{�tz�aG����ߙ�C@�R�tzᾀ  �  ��
=C7Ǯ                                    Bxhv�  T          @���B�\�   �C�
�3=qC=O\�B�\>Ǯ�E��4ffC,��                                    Bxhv��  "          @���)�����`���Sz�C6���)��?aG��Y���JQ�C!��                                    Bxhwj  "          @�Q��p�>W
=����aHC,�ÿ�p�?����z�H�r=qC�                                    Bxhw  �          @�����=L�������x�\C2�=��?�
=�u�f��C�)                                    Bxhw$�  �          @���޸R��ff�}p�8RCB}q�޸R?.{�z�H�C��                                    Bxhw3\  �          @��ÿ�33�aG��mp��rG�CLٚ��33>L���s�
�}z�C.{                                    BxhwB  �          @�\)����&ff�XQ��V�\CC@ ���>��
�[��[
=C,p�                                    BxhwP�  "          @�z��,(��aG��G
=�D=qC8�H�,(�?(���B�\�>�C&5�                                    Bxhw_N  "          @���\)�.{�\(��cp�C8W
�\)?L���W
=�Z��C n                                    Bxhwm�  
�          @���
=?+����RB�C5ÿ�
=?����vff�r��B���                                    Bxhw|�  
�          @�
=��z�?��\�z=q�fC
�R��z�@z��^�R�^�
B�                                    Bxhw�@  "          @���z�?L���fff�p33Cff��z�?�G��O\)�L��C	\)                                    Bxhw��  
�          @����{?z��i���wC"����{?����Vff�X
=C��                                    Bxhw��  �          @�{���H=�G��g��vG�C0ٚ���H?����\���c�C��                                    Bxhw�2  T          @����  ?!G��q��  C ���  ?�33�]p��\��C�R                                    Bxhw��  
�          @��R��z�?B�\�j�H�s33CJ=��z�?޸R�Tz��P��C	��                                    Bxhw�~  �          @��׿��H?s33�x����C�H���H?�p��^�R�[  B���                                    Bxhw�$  "          @��
��(�?����e�{Q�CO\��(�@�
�Dz��Fz�B�u�                                    Bxhw��  T          @�  ����@z��U�O�\B�\)����@:=q�)���=qB��3                                    Bxhx p  
�          @�(���=q@��\���P
=B��ÿ�=q@B�\�.{�  B�
=                                    Bxhx  "          @�  ��=q?���p���cz�C�3��=q@5�G
=�/{B�(�                                    Bxhx�  
�          @�=q���@�H�XQ��?33B��f���@P  �&ff�
B�L�                                    Bxhx,b  
�          @��Ϳ���@z��N{�=��C (�����@G
=�{�	B���                                    Bxhx;  
�          @�녿�{@*�H�6ff�)�\B�8R��{@U�G���ffB�                                     BxhxI�  T          @w�?}p�@.{@!�B$�B��H?}p�?�Q�@J�HB^�B{ff                                    BxhxXT  "          @w
=>�G�@P  ?�A�z�B��=>�G�@'�@.�RB7{B��
                                    Bxhxf�  
�          @n{�\)@L��?޸RA��
B��H�\)@'
=@"�\B.33B�=q                                    Bxhxu�  
�          @k���@_\)?c�
Ac�BÏ\��@G�?�ffA�
=B�aH                                    Bxhx�F  T          @q녽L��@n�R>�@��HB�uýL��@^{?���A�
=B��\                                    Bxhx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxhx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxhx�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxhx��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxhẍ́             @�{���R=��
������C1�Ϳ��R?��~�R�o��Ck�                                    Bxhx�*  T          @���Q�?�  ��=q�sC
c׿�Q�@!G��aG��E  B���                                    Bxhx��  
�          @��Ϳ�Q�?�33���
�}
=C\)��Q�@ ���u�O{B�{                                    Bxhx�v  T          @��
��
=?(�����RC�{��
=?���G��v��C ^�                                    Bxhy  �          @�(��Ǯ��Q���33�{C78R�Ǯ?������Rk�C�                                    Bxhy�  �          @�33��(��\(���{CN���(�>�����  z�C)�                                    Bxhy%h  
�          @�녿��H�n{�����qCYc׿��H>������C'�                                    Bxhy4  �          @�
=���H@{��Q��m�Bʏ\���H@_\)�]p��1�
B�
=                                    BxhyB�  �          @��H=�G�@L(��g
=�A��B���=�G�@����)���p�B���                                    BxhyQZ  %          @�(�>#�
@HQ��n�R�G��B�33>#�
@�  �1��G�B�p�                                    Bxhy`             @��þǮ?ٙ�����fB�
=�Ǯ@2�\�tz��Vp�B�
=                                    Bxhyn�  "          @�z�\(�>\)���z�C+�\(�?������#�B��H                                    Bxhy}L  �          @�z�=p�?5�����qC
�=p�?���\B�                                    Bxhy��  T          @�Q�z�?������\B�uÿz�@���z��q��BУ�                                    Bxhy��  T          @��׿5?^�R��z���CO\�5@�\��  �|=qBڞ�                                    Bxhy�>  �          @�  �p��>�ff���
�=C��p��?�\)��33L�B�G�                                    Bxhy��  "          @�  ��\)>��H��G�B�CG���\)?�=q������B���                                    BxhyƊ  "          @�z��\?У���G�(�B���\@'
=�_\)�R�\B�G�                                    Bxhy�0  �          @��׾�@
=��(��q
=B�(���@U��X���5�RB�\)                                    Bxhy��  "          @�
=�.{@.{�n�R�Wp�B�G��.{@e��:=q�G�B��                                    Bxhy�|  
�          @�{=���@Z=q�=q���B��==���@z=q���H���B���                                    Bxhz"  Q          @��>�ff@e����B���>�ff@\)����h��B�8R                                    Bxhz�  
Z          @�{?B�\@W��ff�	  B�� ?B�\@w
=������B���                                    Bxhzn  �          @�>�  @C�
�4z��)�B���>�  @k������ޏ\B�\)                                    Bxhz-  �          @����G�@4z��AG��;�B�Q��G�@_\)�p��=qB�z�                                    Bxhz;�  �          @�p��#�
@U���R��RB�k��#�
@vff�Ǯ��ffB��3                                    BxhzJ`  �          @��׿B�\@Dz��8Q��)\)BϮ�B�\@l(�� �����\B��                                    BxhzY  �          @��
���@   �G
=�L�\B��H���@L���Q��ffB�G�                                    Bxhzg�  �          @��
��=q@K��333�$�B��;�=q@q녿�z��ծB�=q                                    BxhzvR  �          @�z�
=@C�
�@���/��B��Ϳ
=@n{�������B�                                    Bxhz��  "          @�ff��p�@`  �(Q��B����p�@�����z���  B�z�                                    Bxhz��  �          @��R���@��R�\��  B֮���@�{��33��z�B�                                    Bxhz�D  �          @�  ��@����(��a��B���@��ͽu�&ffBޏ\                                    Bxhz��  �          @�p�� ��@��\����V�\B�3� ��@�\)���
��=qB�B�                                    Bxhz��  "          @�p���@�=q����MB�\��@�ff<#�
=��
B��                                    Bxhz�6  �          @����z�@�  �333�ffB�{�z�@�G�>\@�G�B��                                    Bxhz��  �          @�
=�z�@�(���=q�J=qB��z�@��?Q�AQ�B�                                    Bxhz�  �          @�{�G�@ff@4z�B#33C{�G�?�=q@S�
BH�\CE                                    Bxhz�(  �          @��\� ��@	��@I��B;��C
=� ��?��@e�B`��C�3                                    Bxh{�  "          @�G��G�@\)@C33B5(�C�R�G�?�@`��B[p�C��                                    Bxh{t  "          @�  ��?���@S�
BL�
C�f��?���@l(�Bq{CT{                                    Bxh{&  
�          @�=q��G�?���@_\)BV��C�H��G�?p��@u�By�HC                                    Bxh{4�  
�          @��\�˅?�ff@`��Bn�C�ÿ˅>���@n�RB�ffC%�
                                    Bxh{Cf  �          @j=q�Tz�@G�@*�HB?ffB��ÿTz�?�ff@I��Bs\)B�#�                                    Bxh{R  
�          @aG���33@{?�A��B�
=��33?��R@�RB0=qB�=q                                    Bxh{`�  �          @`  ���?���@��B3(�C�����?u@1G�BR�HC�                                    Bxh{oX  �          @\(���(�@��?!G�Ah��B�z��(�@
�H?�Q�A�33Bʏ\                                    Bxh{}�  �          @o\)>�33@A��Q���B��H>�33@]p���=q��
=B��                                     Bxh{��  �          @p��>�(�@R�\�޸R��\)B�=q>�(�@g��c�
�[�B��\                                    Bxh{�J  �          @l(�?!G�@X�ÿ��\���B��)?!G�@fff�������B�
=                                    Bxh{��  �          @p�׾�\)@l(�>�  @|(�B���\)@aG�?��A�33B�.                                    Bxh{��  
Z          @s�
>�G�@fff�����(�B�\>�G�@p  �#�
���B���                                    Bxh{�<  �          @o\)>�
=@_\)��������B�.>�
=@i���k��eB�Ǯ                                    Bxh{��  �          @�Q�?�R@p  ��R��B�(�?�R@r�\>���@�B�\)                                    Bxh{�  
(          @���?(�@_\)�5��ffB�\?(�@�=q��z�����B��
                                    Bxh{�.  "          @��R?�R@N�R�@  �)G�B�#�?�R@vff�Q����B���                                    Bxh|�  T          @�33?\)@S�
�^{�7�RB��R?\)@�G��$z�� �B�.                                    Bxh|z  
+          @���>��@_\)�K��(Q�B���>��@��������RB�
=                                    Bxh|   �          @��
�#�
@\)�33��B�.�#�
@�p����
���B�(�                                    Bxh|-�  T          @�  ��@�=q�������B��=��@���\(��&�RB�G�                                    Bxh|<l  T          @��#�
@���(Q��G�B�=q�#�
@�p�������Q�B�8R                                    Bxh|K  T          @�ff=#�
@����7
=�  B�=#�
@��\��=q���B��                                    Bxh|Y�  �          @�G����R@�  �+�� �\B�Q쾞�R@��׿˅���B�p�                                    Bxh|h^  �          @��þ�@��
�6ff�	��B����@�������\B�L�                                    Bxh|w  
�          @��\���H@��R�2�\�=qB�8R���H@�  ���H���B��q                                    Bxh|��  T          @�(����@��H�,(���G�B����@���˅��ffB���                                    Bxh|�P  �          @�����@����-p���  B�G�����@���˅��G�B�33                                    Bxh|��  �          @�{���@���,�����B������@�p��˅���RB��                                    Bxh|��  �          @��;aG�@���*=q��{B��{�aG�@�p������p�B�                                    Bxh|�B  T          @��Ϳ0��@���AG���B�  �0��@�ff��(���z�BĞ�                                    Bxh|��  �          @�(��B�\@��H�@  �
=B���B�\@����H��33Bƀ                                     Bxh|ݎ  �          @���Q�@��\�B�\�p�B���Q�@��   ��ffB���                                    Bxh|�4  �          @�33�z�H@y���H����B�\)�z�H@��������z�B̏\                                    Bxh|��  �          @�Q�=�\)@����#33��33B��=�\)@��׿�(���33B�G�                                    Bxh}	�  "          @���<#�
@��������B��R<#�
@�=q��\)�}B�                                    Bxh}&  �          @�����@����J�H�(�B�W
���@��������G�B��\                                    Bxh}&�  �          @�\)�z�@y���Z�H�#�RB��H�z�@��H��H���HB�aH                                    Bxh}5r  "          @��R�=p�@s33�^{�'z�B��=p�@�Q��   ��G�BƔ{                                    Bxh}D  �          @�{�=p�@s33�[��&p�B�  �=p�@���p���\)BƮ                                    Bxh}R�  T          @�
=��G�@Z=q�u��<�B�𤿁G�@�ff�<(��B���                                    Bxh}ad  �          @�p��#�
@���7���B��#�
@�33�����(�B��
                                    Bxh}p
  �          @��?�R@�G��\)��Q�B��\?�R@�{�����L��B�Ǯ                                    Bxh}~�  �          @��\>8Q�@����/\)��RB�B�>8Q�@�G���
=����B�Ǯ                                    Bxh}�V  "          @��\>.{@�p��   �뙚B��R>.{@�(���z���G�B��                                    Bxh}��  �          @�    @�
=�&ff��B�      @�ff��  ��33B�                                      Bxh}��  
�          @��R��Q�@����!���=qB�(���Q�@�Q쿴z��|(�B���                                    Bxh}�H  �          @�(��#�
@���\)���B�G��#�
@�{����{�
B�B�                                    Bxh}��  �          @���=�\)@�����R���HB�G�=�\)@�33��33��
=B�p�                                    Bxh}֔  "          @�����@�������B������@�ff���R�`Q�B�\)                                    Bxh}�:  �          @��׽�\)@���Q���
=B���\)@��H����s
=B���                                    Bxh}��  �          @�zὣ�
@�=q�G���{B�(����
@�\)��p��ip�B���                                    Bxh~�  T          @����\@�������(�B��3��\@��
�W
=�
=B��H                                    Bxh~,  
�          @��ÿW
=@����p���=qB�
=�W
=@�ff�����S�
B�L�                                    Bxh~�  
�          @�  �B�\@�����|  B�33�B�\@���  �1G�BĔ{                                    Bxh~.x  �          @��׿@  @�Q쿸Q��}�B��@  @��R����6ffB�ff                                    Bxh~=  �          @��R�^�R@���z��{�
B�\�^�R@��
����5B�Q�                                    Bxh~K�  T          @�����@��R��=q�l  B�.���@�(��8Q����B�aH                                    Bxh~Zj  �          @�{��ff@��R�����C33B��ῆff@��\=L��?z�B�p�                                    Bxh~i  �          @�p���p�@������8��B�8R��p�@���=���?���Bϣ�                                    Bxh~w�  
�          @��ÿ�
=@��\�+����Bυ��
=@�33>�G�@�  B�aH                                    Bxh~�\  �          @�Q쿞�R@�녾���G�B�p����R@���?z�@��B҅                                    Bxh~�  
�          @�����\@�G����
���B�𤿢�\@�  ?(��AffB�8R                                    Bxh~��  T          @�p���Q�@�  ��\)�k�B�ff��Q�@�ff?.{A�BӸR                                    Bxh~�N  �          @����Q�@��
<#�
>.{B�ff��Q�@�Q�?}p�AG�B�Q�                                    Bxh~��  
Z          @�{��33@�
=>\)?�p�Bֳ3��33@��\?��AaG�B�Ǯ                                    Bxh~Ϛ  �          @�33���@���>L��@#33B��Ύ�@��?�
=An�\B�(�                                    Bxh~�@  �          @����Q�@�  =�G�?�=qBљ���Q�@��
?�{A\��BҀ                                     Bxh~��  �          @��
��
=@�ff��������Bя\��
=@�p�?.{A�RB���                                    Bxh~��  T          @��H���H@����
=��Q�B��Ϳ��H@�z�?�@�p�B��                                    Bxh
2  �          @�녿���@�녿5�\)B�\����@��>�=q@W�Bծ                                    Bxh�  �          @�Q�}p�@�z�u�AG�B�Q�}p�@��H?@  Az�Bͣ�                                    Bxh'~  T          @����xQ�@�(��5��B�
=�xQ�@�>�\)@dz�B�Ǯ                                    Bxh6$  �          @�Q�z�H@�G���  �LQ�Bͮ�z�H@��ͼ��
��  B�
=                                    BxhD�  �          @���s33@����{�j=qB͸R�s33@����#�
�Q�B��f                                    BxhSp  �          @��
�\(�@��R�E��"�RB�(��\(�@���>8Q�@Q�B���                                    Bxhb  �          @��ͿL��@���Q��,(�B�G��L��@�=q>\)?���B��                                    Bxhp�  �          @�  �u@�Q쿇��Z�HB�G��u@�zὸQ쿘Q�B̏\                                    Bxhb  �          @�p��O\)@�{��{�i�B�  �O\)@��\�#�
��B�L�                                    Bxh�  �          @��
��R@��\��
=��ffB�LͿ�R@�G���(����BÏ\                                    Bxh��  �          @��H�Y��@�z῅��R{B��
�Y��@�Q�L�Ϳ(�B�B�                                    Bxh�T  T          @��׿333@��Ϳ=p��ffB�  �333@��R>�  @K�B�Ǯ                                    Bxh��  �          @��\�\(�@��Q��$��B�
=�\(�@�  >.{@�Bɮ                                    BxhȠ  �          @�(��\(�@�ff�^�R�,��B��\(�@���>\)?�
=B�\)                                    Bxh�F  T          @�
=���@�ff��(���p�B�����@�������\B�z�                                    Bxh��  �          @��ÿ
=@�  ���R��p�B��H�
=@�
=�����33B�8R                                    Bxh��  �          @����+�@�\)������p�B���+�@�
=�z���z�B�B�                                    Bxh�8  �          @��\�(��@�  �����ffBĨ��(��@�  �(���Q�B���                                    Bxh��  �          @��ÿ(��@�Q쿰�����BĞ��(��@��R��Q����
B��                                    Bxh� �  �          @��ÿ(�@�=q��
=����B�\�(�@�z�p���4��B���                                    Bxh�/*  �          @�=q��@����
�H�ՅB�
=��@��Ϳ�
=�bffB��                                    Bxh�=�  
�          @��H��ff@��\�ff�ͅB���ff@�{�����R{B�
=                                    Bxh�Lv  �          @�33���@�z���R���
B�LͿ��@�
=�z�H�;\)B�L�                                    Bxh�[  �          @���
=q@��R���z�B³3�
=q@��
��{��z�B�k�                                    Bxh�i�  �          @����G�@���(Q��G�B�p���G�@��ÿ�
=��ffB�#�                                    Bxh�xh  �          @�G����H@}p��'��(�B�8R���H@�{�ٙ���ffB��3                                    Bxh��  T          @�  �5@����
=��\B��
�5@�
=����\)B�
=                                    Bxh���  �          @�\)�Q�@|���p���33B�z�Q�@�z��ff����B�.                                    Bxh��Z  �          @��H�@  @��H�����BȨ��@  @��׿�  ����BƸR                                    Bxh��   �          @����@�{�
=��B�𤿅�@�33��33��  B̀                                     Bxh���  "          @�����@��
�����
=B�녿���@�녿�  ���B�.                                    Bxh��L  �          @��׿B�\@y���'
=�33B���B�\@��
�ٙ����B�Ǯ                                    Bxh���  "          @�Q�#�
@s�
�7
=�ffB��þ#�
@��H��(�����B�\)                                    Bxh��  �          @��\����@r�\�?\)��B��ý���@�33�ff���
B��{                                    Bxh��>  "          @���>L��@l���C33��B��)>L��@����
�H�ׅB��                                    Bxh�
�  �          @�{?(��@g
=�:�H��
B�B�?(��@���z���33B��                                    Bxh��  �          @���?+�@e�7��z�B��q?+�@�(��G��Ώ\B�p�                                    Bxh�(0  
�          @�33?!G�@u��{��B�G�?!G�@��׿˅��G�B�.                                    Bxh�6�  �          @��
��@s33�&ff�	z�B�����@��ÿ�(���G�B�33                                    Bxh�E|  
�          @�=q����@s�
�\)�\)B�  ����@�  ��{����B��q                                    Bxh�T"  
�          @��\���@l���*�H�{B�Ǯ���@�������B�8R                                    Bxh�b�  �          @��
����@c33�;��=qB������@�33�ff��
=B��R                                    Bxh�qn  �          @�p����H@w
=���Q�B��쿚�H@��׿�������BӞ�                                    Bxh��  �          @�G����
@z=q�(���
=B�W
���
@�33�����(�B��                                    Bxh���  �          @��׿��@r�\�#33�B�
=���@�  ������B�.                                    Bxh��`  �          @�p���{@l(���R�  B�ff��{@�(��У���(�B�Q�                                    Bxh��  �          @�z��G�@vff�33��p�B����G�@�ff��
=�l��Bۮ                                    Bxh���  T          @��Ϳ�z�@e��%��
B��f��z�@�����G����\B�G�                                    Bxh��R  T          @��
���@L(��C33�%
=B�{���@qG��33���B�u�                                    Bxh���  �          @����33@*�H�XQ��<��B󙚿�33@U��.�R�33B���                                    Bxh��  T          @�z῞�R@j=q�#�
�ffBُ\���R@��
��(���ffBՔ{                                    Bxh��D  �          @��Ϳ��
@^�R�*�H��\B�{���
@~{��\)���B�=q                                    Bxh��  �          @�녿��@s�
�z��ٮB�zῥ�@�p������v�HB�z�                                    Bxh��  �          @�G���33@Q��5�
=Bڨ���33@tz��z����BՔ{                                    Bxh�!6  �          @�33����@.{�a��Hz�B�8R����@Z=q�7
=�{Bؽq                                    Bxh�/�  
�          @����k�@%�fff�R��B�LͿk�@S33�>{�#��B�G�                                    Bxh�>�  �          @�G��xQ�@,(��aG��K�\B��ͿxQ�@XQ��7
=��B�#�                                    Bxh�M(  "          @�p��^�R@Dz��AG��-��B�z�^�R@i����\���BθR                                    Bxh�[�  
�          @���W
=@�Q쿈���b=qB��\�W
=@�z�\)��B�ff                                    Bxh�jt            @�G�<��
@��׾�{��33B���<��
@��?!G�@��B��{                                    Bxh�y  X          @�    @��H�=p���B���    @���>L��@'
=B���                                    Bxh���            @�녾\@�녿����Q�B�#׾\@�Q��G���  B��3                                    Bxh��f  T          @�zᾅ�@�\)�����nffB�#׾��@��
�L���%B��f                                    Bxh��  �          @�녾aG�@������i��B��aG�@����.{��\B��q                                    Bxh���  &          @���(��@mp��(�����B�LͿ(��@��H������
B�p�                                    Bxh��X  P          @�G��^�R@\���5��G�B�p��^�R@~{���ԏ\B���                                    Bxh���  �          @��
��@>{�>{�+�
B�녿�@a��G����HB؏\                                    Bxh�ߤ  "          @��\��ff@0  �[��AB�uÿ�ff@Z�H�1G��\)B݅                                    Bxh��J  T          @�
=���@Q��`���O\)B�{���@E��;��#�B�                                    Bxh���  "          @��
��Q�@G��\(��O�B�����Q�@<���7��$��B��f                                    Bxh��  
^          @��Ϳ�
=@��\���O�B��f��
=@G
=�6ff�"p�Bݣ�                                    Bxh�<  �          @�녿�=q@@  �   ���Bۊ=��=q@]p������B֏\                                    Bxh�(�  
�          @��ÿ��H@9���!G��33B�B����H@W�������z�B�p�                                    Bxh�7�  
�          @|(���G�@<(��33��
B�\)��G�@W
=��\)��=qB�\                                    Bxh�F.  �          @��Ϳu@J�H�   �33B��H�u@hQ��\��B���                                    Bxh�T�  �          @�����\@U������HB�𤿂�\@o\)��  ��33B�p�                                    Bxh�cz  �          @��\��\)@k���(��؏\B�녿�\)@�Q쿑��t(�B�=q                                    Bxh�r   T          @�\)���@qG�����{Bٞ����@��
��
=�t��B֨�                                    Bxh���  
�          @�녿�\)@tz������G�B��Ϳ�\)@�=q�@  �!G�B��)                                    Bxh��l  �          @��ÿJ=q@��ÿ��}��B�{�J=q@�{��\)�qG�B�G�                                    Bxh��  
�          @��׿�=q@z�H�����z�B�Ǯ��=q@����G���B�p�                                    Bxh���  �          @�=q���
@|(����H���
B�  ���
@����{��  B֞�                                    Bxh��^  T          @��
���@��׿�ff�^�HB��ÿ��@��;.{�p�B��f                                    Bxh��  �          @�
=���R@�Q쿝p���  B��Ϳ��R@���{��{B�B�                                    Bxh�ت  
�          @������@{���{��G�B��Ϳ���@��=p��Q�B؞�                                    Bxh��P  
�          @�=q���R@��\>�33@�
=B�.���R@���?���A���B�\)                                    Bxh���  T          @�
=��33@��>�G�@�(�B��ÿ�33@�G�?�Q�A�G�B�\)                                    Bxh��  �          @����\@�>�\)@P  B�G����\@���?��\AqB�W
                                    Bxh�B  �          @����
=@�p���\)����B��
��
=@��������
=B�p�                                    Bxh�!�  �          @�=q���@��׾B�\�G�B�녿��@�ff?G�A��B�p�                                    Bxh�0�  T          @��\�ٙ�@�\)���H���
Bݨ��ٙ�@�\)>��@�z�Bݣ�                                    Bxh�?4  
�          @��׿��R@��=���?�(�B�{���R@�?��\AE�B��
                                    Bxh�M�  T          @�ff�ٙ�@���?W
=A$��B�W
�ٙ�@�  ?�p�A�p�B�
=                                    Bxh�\�  "          @�zΌ��@�(���  �HQ�B؏\����@��H?0��A��B��                                    Bxh�k&  �          @�33���R@��\�#�
��(�B�
=���R@�Q�?E�A�\Bڔ{                                    Bxh�y�  
�          @�p���{@��H?   @�B��Ϳ�{@�(�?�z�A���Bި�                                    Bxh��r  
�          @��R���@��H?:�HA�B�W
���@��\?��A���B߳3                                    Bxh��  �          @�33��@��R?s33A5��B�\��@���?��A��RB���                                    Bxh���  X          @��
��\)@��?s33A4  B۽q��\)@�?��A�Q�B�ff                                    Bxh��d  �          @���33@���?n{A-�B��
��33@�  ?��A�\)B�p�                                    Bxh��
  �          @��\��=q@�Q�?L��A�Bڙ���=q@�\)?޸RA��\B��f                                    Bxh�Ѱ  �          @��ÿ��R@��R?G�A\)B�𤿾�R@�{?�(�A���B�#�                                    Bxh��V  "          @�녿���@�
=?L��A��Bڣ׿���@�ff?�p�A��B���                                    Bxh���  "          @�G��޸R@��
?n{A2ffB�\)�޸R@�=q?�A�z�B�8R                                    Bxh���  
Z          @�Q쿽p�@��R?0��ABظR��p�@��R?У�A�ffBڽq                                    Bxh�H  
�          @��\����@�>u@8Q�Bͮ����@���?��RAn�\BΏ\                                    Bxh��            @�  ���@�33>�  @AG�Bγ3���@�{?�p�ApQ�Bϣ�                                    Bxh�)�  �          @���}p�@�G�>aG�@1G�B̨��}p�@�(�?�Q�AmG�B̀                                     Bxh�8:  T          @�����
@���>�{@�p�B͔{���
@��?�ffA��BΙ�                                    Bxh�F�  T          @�\)���R@���?�@���Bҽq���R@��?�p�A��B�33                                    Bxh�U�  
�          @�����
=@�z�>L��@��BО���
=@��?�Q�Af�RBъ=                                    Bxh�d,  
�          @�{��\)@�
=>��@H��B�.��\)@��?��HAo\)B�\)                                    Bxh�r�  T          @�����ff@�33>�
=@��BӮ��ff@���?�33A��B�                                    Bxh��x  &          @����R@��>�{@�  B�ff���R@��?���A�BҊ=                                    Bxh��  �          @��׿�ff@��
>��H@��B�G���ff@���?��
A�{B�ff                                    Bxh���  
�          @��H���@�
=>�  @7�B̮���@���?���Ao�B͊=                                    Bxh��j  T          @�33����@�{>�ff@��HB��쿐��@�\)?�  A�\)B��                                    Bxh��  T          @��׿z�H@����Q쿃�
Bʙ��z�H@��?xQ�A2{B�                                    Bxh�ʶ  T          @�33���@�
=��ff���B�\���@�ff?�R@���B�(�                                    Bxh��\  �          @��׿p��@��H�\(��=qB�
=�p��@��>L��@�
Bɽq                                    Bxh��  "          @��R��p�@���У�����B�\��p�@��Ϳ
=�ٙ�B���                                    Bxh���  �          @���aG�@�  �Q��ʏ\B����aG�@�33��{�N=qB�.                                    Bxh�N  "          @�33���
@������B�.���
@��Ϳ�{��=qB���                                    Bxh��  �          @�Q�=���@��R����{B�\)=���@�33��G��h��B��{                                    Bxh�"�  �          @�z��G�@�p��\)��\)B�
=��G�@�����
��(�B��                                    Bxh�1@  T          @�
=���@��������  B���@�
=������B�=q                                    Bxh�?�  �          @�Q��@�p��+��
=Bã׿�@��Ϳ�(����B�
=                                    Bxh�N�  "          @�ff�\)@����Q����B��
�\)@��R������HB��\                                    Bxh�]2  
�          @��׽�G�@~�R�AG���HB�G���G�@�������
B��H                                    Bxh�k�  
�          @�G��0��@�z�����B��ÿ0��@�G���=q�uB�z�                                    Bxh�z~  �          @����\)@�z��\)��(�B��q�\)@��Q����B��                                    Bxh��$  T          @����s33@������z�B�Q�s33@�z�E��z�B�                                      Bxh���  T          @��H�@��׿s33�,Q�B�{�@��=u?!G�B��                                    Bxh��p  �          @�
=�8Q�@~{>��@�
=B��H�8Q�@r�\?�G�Al��B�u�                                    Bxh��  
�          @����;�@e�?�
=Ae�C=q�;�@O\)?�
=A���C{                                    Bxh�ü  �          @�p��-p�@h��?�{A\Q�B�33�-p�@S�
?��A��C:�                                    Bxh��b  
�          @�33�8��@U�?�G�A~ffC�q�8��@>�R?���A�G�C.                                    Bxh��  
�          @�z��-p�@e�?�\)A`  B�G��-p�@P  ?�\)A��HC�{                                    Bxh��  
Z          @�{�
=q@��\?Q�A!�B��)�
=q@s33?�Q�A�G�B�B�                                    Bxh��T  �          @�����@~{?�z�Ag�B�(���@g�@   A��
B��                                    Bxh��  
�          @�ff�  @e?��A�G�B��  @N{@�
A݅B�Ǯ                                    Bxh��  �          @�Q��   @j=q?O\)A&�\B����   @Y��?˅A��B��                                    Bxh�*F  
�          @���%@q�>�ff@��B��q�%@e?��\A33B��                                    Bxh�8�  �          @�(��"�\@j�H?��
AO\)B�ff�"�\@Vff?�ffA��B�B�                                    Bxh�G�  �          @���	��@l�Ϳ�{��33B�.�	��@z=q����(�B�                                     Bxh�V8  
Z          @����{@mp�=#�
>��HB�W
�{@g
=?Y��A/�
B�                                    Bxh�d�  T          @�33�.�R@dz�?O\)A$  B��.�R@S33?���A�p�C��                                    Bxh�s�  �          @���,��@U?��RA���C ���,��@;�@(�A�C��                                    Bxh��*  
�          @�
=�.�R@E?��HA��\C���.�R@(Q�@ffA�  C�                                    Bxh���  "          @����0��@I��?�Q�A��C8R�0��@,��@A��C��                                    Bxh��v  T          @�  �0��@E�?�G�A��
C�f�0��@'�@��B z�C��                                    Bxh��  "          @�p��J=q@$z�?��HA�{C��J=q@Q�@�RA��HC�                                    Bxh���  
�          @��9��@E?��
A�{C:��9��@.�R?�A�Q�C�                                     Bxh��h  �          @�{�,(�@=p�?�
=A�p�CW
�,(�@'�?�ffA���C                                    Bxh��  
�          @q��8��@
=<#�
>8Q�C�)�8��@�\?�A
=C�
                                    Bxh��  
�          @�p��7�@?\)>L��@/\)C�7�@8Q�?\(�A>=qC��                                    Bxh��Z  
�          @��
�6ff@(�?��A���Ck��6ff@?�A�Q�C�\                                    Bxh�   
�          @�Q��9��@W�?   @�{C���9��@K�?��RA�Q�C\)                                    Bxh��  	�          @�(��8Q�@N{>�{@��C�=�8Q�@Dz�?�ffA`Q�C5�                                    Bxh�#L            @�z���\@x�����p�B�.��\@�Q쿕�_33B�\                                    Bxh�1�  "          @�zῷ
=@}p��Q���=qBۨ���
=@�zῸQ���
=B�
=                                    Bxh�@�  
�          @������@�  �<���B��쿌��@������R��  B�33                                    Bxh�O>  
#          @��\��  @�(��)����{B���  @��
��z����B�aH                                    Bxh�]�  "          @����(�@�z����Ώ\B��)��(�@�G���  �\(�B۔{                                    Bxh�l�  "          @�G����@�ff�%�����B�𤿅�@�p���=q��ffB�8R                                    Bxh�{0  
�          @�ff�Ǯ@�p��	���ĸRB�Ǯ�Ǯ@�G������EB�                                      Bxh���  �          @��R���R@�=q�{��G�B�  ���R@�Q쿸Q�����B֮                                    Bxh��|  '          @��\�޸R@��
�8����\B���޸R@����33����B��                                    Bxh��"  
�          @��R���@��Ϳ���q�B�zῑ�@�=q�W
=��RBΊ=                                    Bxh���  
�          @�33��  @�z��p�����BӸR��  @��ÿ��Up�B�G�                                    Bxh��n  Y          @�(���  @���\)���B��)��  @�녿�����33B˅                                    Bxh��  
�          @�ff��\@���.�R����B��R��\@�
=����z�RB��                                    Bxh��  �          @�>���@�{�:�H��p�B���>���@��R��\)�s
=B�#�                                    Bxh��`  �          @��?W
=@�\)�5��33B��
?W
=@����33��Q�B���                                    Bxh��  
�          @��>���@���X���G�B��>���@�=q���У�B�p�                                    Bxh��  "          @�\)>�{@|(��X���"\)B�8R>�{@���=q�܏\B���                                    Bxh�R  �          @���>�
=@\���n�R�<  B�G�>�
=@��R�6ff�
=B��R                                    Bxh�*�  �          @�G�?��@Z=q�����EffB��?��@�\)�H����\B�33                                    Bxh�9�  "          @�  >8Q�@n{��(��F\)B��\>8Q�@��
�Z�H�  B���                                    Bxh�HD  
(          @�p���\)@tz������G  B�  ��\)@�  �a���B���                                    Bxh�V�  "          @�{=u@_\)��  �P�B���=u@�p��e�ffB�aH                                    Bxh�e�  T          @��׿c�
@vff��p��:{B�#׿c�
@�ff�K���\Bɏ\                                    Bxh�t6  �          @�Q�fff@s33����=G�B�Ǯ�fff@�p��P���	B��                                    Bxh���  
�          @���333@z�H����7��B�Q�333@�Q��G
=��\B���                                    Bxh���  "          @��Ϳ:�H@^{��
=�TG�B��f�:�H@�ff�s33� \)BƸR                                    Bxh��(  �          @�{��R@l�����H�J�B����R@���hQ����B�=q                                    Bxh���  �          @��
����@\)�����;��B�������@��
�P  ���B��                                    Bxh��t  �          @�{�5@�����33�A��B�(��5@�
=�c�
�p�B�\)                                    Bxh��  �          @�녾��
@�����G��4�B�k����
@����L����Q�B�                                      Bxh���  �          @�G��#�
@�(�����1��B�.�#�
@��R�@  ����B�#�                                    Bxh��f  �          @��
>�{@�33�{��((�B��>�{@�(��4z���(�B�Q�                                    Bxh��  �          @��>���@��R�vff�#  B�8R>���@�\)�.{�ۙ�B�\)                                    Bxh��  �          @��R=�G�@�z���G��*=qB�
==�G�@��R�:=q��B��                                     Bxh�X  �          @���}p�?�{��=qB�B�
=�}p�@   �e�Tp�B���                                    Bxh�#�  T          @���{��{��G��d�
CS���{�Ǯ��  �xQ�C>�                                    Bxh�2�  �          @�33��G�@���\�|p�B�p���G�@Fff�~{�J�\B�
=                                    Bxh�AJ  T          @��
>�p�@�{�`����HB��>�p�@�(��z�����B�{                                    Bxh�O�  �          @���>\@�=q�^�R�B��=>\@�Q��z���Q�B�                                    Bxh�^�  �          @�
=?E�@����B�\� \)B���?E�@������=qB��\                                    Bxh�m<  �          @�  ?Q�@����C�
� ffB���?Q�@�(�������Q�B��R                                    Bxh�{�  �          @���>�@�
=�c�
��\B��3>�@�p���H�Ǚ�B�\)                                    Bxh���  �          @�ff>aG�@���u����B���>aG�@��\�*=q���
B�aH                                    Bxh��.  �          @��>aG�@��R�q����B���>aG�@�
=�%��ɮB�\)                                    Bxh���  �          @�z��@�ff���R�:��B��=��@�(��W
=�B��f                                    Bxh��z  �          @��þ�@�(���\)�6z�B�z��@���Tz�� 33B��)                                    Bxh��   �          @�=�\)@�=q���6�HB�=q=�\)@���R�\� z�B��=                                    Bxh���  �          @�p��8Q�@\)��{�F\)B�\�8Q�@�\)�g���B�\                                    Bxh��l  T          @\���@�������@G�B�L;��@�Q��\(��	B�z�                                    Bxh��  �          @�녾��@�ff���R�A�B�#׾��@�{�e��
�B�W
                                    Bxh���  T          @Ǯ��G�@������@��B�  ��G�@�z��b�\�
=qB�k�                                    Bxh�^  �          @Å>��@�z����.\)B�{>��@�  �AG����B��R                                    Bxh�  �          @��
>�p�@���\)�#(�B�>�p�@�p��1�����B�#�                                    Bxh�+�  �          @�p�>���@�����
�(
=B�  >���@���:�H��ffB��{                                    Bxh�:P  �          @�ff?�@��R�QG����
B�k�?�@��\��Q���=qB�                                    Bxh�H�  �          @�{>���@�33�Fff��{B��>���@���p����B�#�                                    Bxh�W�  �          @�33?W
=@�ff�p�����B�?W
=@��R�!����HB��=                                    Bxh�fB  �          @�G�>���@�{�r�\�ffB�Q�>���@��R�#�
��z�B��                                    Bxh�t�  �          @Å?
=q@��R�u�  B�  ?
=q@���&ff��B��
                                    Bxh���  �          @�=q>��H@��������&33B��3>��H@���5����B���                                    Bxh��4  �          @�G�>�@�Q��~�R�$��B���>�@��H�1���G�B��                                    Bxh���  �          @���?333@�Q��{��"�RB�p�?333@��\�.{��33B�{                                    Bxh���  �          @���?���@�{���R�=B�k�?���@�ff�c�
�\)B���                                    Bxh��&  �          @��H?��@q����R�Q�B�� ?��@�p���z���B�aH                                    Bxh���  �          @��
?�z�@l(���=q�U��B�
=?�z�@������� z�B��{                                    Bxh��r  �          @�z�?��R@P�������_=qB�
=?��R@�ff���\�*��B�                                    Bxh��  �          @���?�ff@�����=q�'B�#�?�ff@�Q��8Q���B�aH                                    Bxh���  �          @�(�?��R@�=q�����*�RB��H?��R@�ff�>{���B��                                    Bxh�d  �          @ə�?�(�@�G�����$p�B��
?�(�@���8Q���p�B�z�                                    Bxh�
  �          @�z�?\(�@�{��  �'
=B�G�?\(�@��\�>�R�޸RB��\                                    Bxh�$�  �          @�?.{@�=q���\�%B�
=?.{@�p��5�ۅB��3                                    Bxh�3V  �          @�(�>u@���Vff��HB���>u@����   ���\B�8R                                    Bxh�A�  �          @�  >\@�p����H�1z�B�#�>\@�33�G����
B��
                                    Bxh�P�  �          @�>�  @�z��X���Q�B�p�>�  @�=q�G����B�{                                    Bxh�_H  T          @�>��@�
=�h����B�p�>��@��R��
��  B��f                                    Bxh�m�  �          @�>k�@��H���
�'�\B�aH>k�@��R�7
=���B�G�                                    Bxh�|�  �          @��=#�
@��\���H�:�\B�  =#�
@��\�XQ��p�B�33                                    Bxh��:  �          @�\)>aG�@����(��@  B���>aG�@��]p���
B�                                    Bxh���  �          @�
=?333@�(���33�1�RB��
?333@�=q�HQ����B�                                    Bxh���  �          @���?B�\@������%�B�  ?B�\@�p��333�ظRB���                                    Bxh��,  T          @�(�?.{@������(��B���?.{@��
�7���33B��{                                    Bxh���  �          @ƸR?z�@���y���\)B�33?z�@��
�%���(�B�33                                    Bxh��x  �          @��?333@�  ���H�!�B�#�?333@��
�1G��У�B��R                                    Bxh��  �          @�  ?
=@����\)��
B��?
=@��
�+��ʸRB�
=                                    Bxh���  �          @ȣ�?��@�p��s33��
B�#�?��@��R������\B�\                                    Bxh� j  �          @�z�?z�@�G���z��"��B�33?z�@�p��3�
���
B�W
                                    Bxh�  �          @�ff?
=@�\)��33�6{B��?
=@���U����RB��                                    Bxh��  T          @�z�?��@�\)�mp��z�B�� ?��@�\)�p���=qB�\                                    Bxh�,\  �          @�?�p�@��\�\����33B��?�p�@ȣ׿�Q����HB�Ǯ                                    Bxh�;  �          @�G�?z�H@����r�\��B���?z�H@ʏ\������B�ff                                    Bxh�I�  �          @أ�?�  @���i����B��)?�  @�33�
=���B�(�                                    Bxh�XN  �          @ڏ\?�  @��H��(��
=B��R?�  @ƸR�)����z�B���                                    Bxh�f�  �          @޸R?Q�@�=q�o\)��HB���?Q�@�=q�����33B�Ǯ                                    Bxh�u�  �          @�\)?Y��@�\)��p����B�ff?Y��@˅�(����
=B���                                    Bxh��@  �          @�ff>��
@l����33�fG�B�
=>��
@�=q��
=�+p�B��R                                    Bxh���  �          @�(�?��@L(���=q�wB�� ?��@�(������==qB��                                    Bxh���  �          @���?�R@}p������]{B�#�?�R@�����=q�"ffB���                                    Bxh��2  �          @�Q�>Ǯ@��\��  �N�
B��3>Ǯ@��H���R��RB�                                    Bxh���  �          @���?
=@����{�J��B��q?
=@����(����B�{                                    Bxh��~  
�          @�=q>���@�33��=q�O�
B�33>���@�(������ffB���                                    Bxh��$  T          @�33>���@������\�]�B�>���@�z���33�!��B�{                                    Bxh���  �          @�{�u@�33���\�V�B����u@������\��B�33                                    Bxh��p  �          @�  ��Q�@X������r�B�.��Q�@��H��\)�6�HB��=                                    Bxh�  �          @�
=��Q�@9�������B��{��Q�@���33�IQ�B�W
                                    Bxh��  �          @�p�>��@333��=q.B��R>��@��H��(��LQ�B���                                    Bxh�%b  �          @��
�#�
@����p�  B�#׽#�
@\)���\�YB�G�                                    Bxh�4  �          @�p�>.{@0  ���H��B��>.{@�������M�B���                                    Bxh�B�  �          @�p�>��@7���G��
B��=>��@�����\�IffB�.                                    Bxh�QT  �          @�>�Q�@=p���    B�=q>�Q�@����Q��E�RB��f                                    Bxh�_�  �          @߮>��R@7
=��(�� B��R>��R@�����J�B�\                                    Bxh�n�  �          @�z�>�{@9���Ǯ�fB���>�{@���Q��GG�B�=q                                    Bxh�}F  
�          @�{>���@B�\��
=�33B��\>���@�=q��ff�Bz�B�\)                                    Bxh���  T          @�{>�@@  ��\)��B��\>�@�G���
=�C\)B�ff                                    Bxh���  T          @�ff>�z�@<(��ə���B���>�z�@�������F��B��=                                    Bxh��8  �          @�\)>�\)@ ����\)W
B�Q�>�\)@�(����\�U�B�G�                                    Bxh���  �          @�ff?��R@7��ƸR�}33B��?��R@����\)�C=qB��                                     Bxh�Ƅ  �          @ᙚ?p��@O\)��{�u�B��
?p��@�����33�9�B���                                    Bxh��*  �          @��?��@S33�ƸR�v��B���?��@��\����9��B��                                    Bxh���  �          @��H?#�
@S33�Ǯ�v��B��f?#�
@��H��z��9�
B��
                                    Bxh��v  
�          @߮>�@:=q�����B���>�@�\)�����F��B�(�                                    Bxh�  �          @�p�>�=q@G���Q��B�\)>�=q@w
=���Z(�B���                                    Bxh��  �          @�33>��
@���{B�
=>��
@vff���H�Xp�B��                                     Bxh�h  �          @��H>�Q�@.�R�ȣ�� B�#�>�Q�@�������K�B��\                                    Bxh�-  �          @��>8Q�@@  ���H=qB�(�>8Q�@��H�����D33B��=                                    Bxh�;�  �          @��>��
@���Ӆ��B���>��
@�z����R�W��B�\                                    Bxh�JZ  T          @��H>�=q@����\)�B�
=>�=q@|(���z��`�B�(�                                    Bxh�Y   T          @��
>���?�=q���H=qB��=>���@c33��33�jz�B�8R                                    Bxh�g�  �          @�=q?��@0  ��\){B�=q?��@�����  �L=qB�aH                                    Bxh�vL  �          @��
>�{@���\)�B��>�{@�����33�]��B�aH                                    Bxh���  �          @�
=?z�?�(������B�=q?z�@mp����
�e��B�8R                                    Bxh���  �          @�ff?z�@  �љ��B��f?z�@|����{�[��B�.                                    Bxh��>  T          @�\)?��\@<�����|�B�  ?��\@�����z��?�HB��3                                    Bxh���  �          @߮?�p�@R�\�����o(�B��)?�p�@�������2�B�=q                                    Bxh���  �          @��?���@xQ���p��X{B��f?���@�����z��33B�Ǯ                                    Bxh��0  �          @�z�?���@�=q���\�J��B��?���@�(���Q��Q�B��                                    Bxh���  �          @�p�?�  @w������T�B���?�  @����Q���B�\                                    Bxh��|  �          @��?\@`  ��{�e{B�{?\@�  ����(�HB�33                                    Bxh��"  �          @�@��@fff����Q�
Bj33@��@�
=������
B�B�                                    Bxh��  �          @�z�@�@\�������R�Bkff@�@�������G�B���                                    Bxh�n  T          @�
=@z�@`����33�S  Bm��@z�@�33��p��\)B��)                                    Bxh�&  �          @�G�?�(�@c33��{�U{Bs�?�(�@�p�������B�G�                                    Bxh�4�  �          @�  @Q�@g����H�TG�Bm�
@Q�@�������\)B�                                      Bxh�C`  �          @�?�@������\�IQ�B��H?�@��H��  �{B�{                                    Bxh�R  �          @�z�?�33@_\)��  �e
=B�W
?�33@�����G��(��B���                                    Bxh�`�  �          @��
?��
@>{��G��x  Bz�?��
@����ff�;�HB�.                                    Bxh�oR  �          @��H?�p�@�
��G�� Bep�?�p�@�����(��R33B��H                                    Bxh�}�  �          @�
=?˅@���ʏ\k�Bd{?˅@�z���z��JQ�B��H                                    Bxh���  �          @��?���?�Q����
u�B<\)?���@_\)���
�c��B�ff                                    Bxh��D  �          @��?���?�  �Ӆ(�B  ?���@S33��p��d33Bm��                                    Bxh���  �          @ڏ\@�@G�����vffB8p�@�@tz���{�AG�Bq{                                    Bxh���  �          @�Q�?��\@����\)�%�
B�\?��\@�
=�9����p�B�\                                    Bxh��6  �          @ۅ?�z�@�  �����RB��?�z�@�
=�   ��\)B�k�                                    Bxh���  �          @ڏ\?���@�=q��=q�p�B��?���@�  �����  B�.                                    Bxh��  �          @�?�=q@�{�q����B�� ?�=q@����   ���RB�8R                                    Bxh��(  �          @�?��\@��\�fff���\B�.?��\@��
����o33B�z�                                    Bxh��  �          @޸R?u@�=q�k�����B�B�?u@�(���{�xQ�B��                                    Bxh�t  �          @�?�ff@�\)�p  ��B�Q�?�ff@�=q������z�B��f                                    Bxh�  �          @��?��@�����Q����B�?��@�{�  ��ffB��R                                    Bxh�-�  �          @ٙ�?��@�����33�33B��q?��@�Q�������B�\)                                    Bxh�<f  �          @ٙ�?��@����z���HB�G�?��@ƸR�������B�z�                                    Bxh�K  �          @��H?�G�@�33��z���B�G�?�G�@����.�R��=qB��H                                    Bxh�Y�  �          @��?�@�ff�u��z�B���?�@���G����RB���                                    Bxh�hX  �          @�z�?��R@�p����H�  B�  ?��R@ƸR�*=q��(�B�ff                                    Bxh�v�  �          @�ff?���@mp���G��`��B�  ?���@�\)���R��B��=                                    Bxh���  �          @�
=?�{@vff���R�[�
B��)?�{@��H���H�
=B��\                                    Bxh��J  �          @�\)?���@�Q���33�UffB��=?���@�
=���ffB���                                    Bxh���  �          @�=q?s33@�
=��  �J
=B���?s33@�=q�r�\�G�B��                                    Bxh���  �          @�
=?���@�Q���\)�MQ�B��q?���@��
�tz��Q�B��
                                    Bxh��<  �          @�Q�?�\)@��������Np�B��?�\)@��xQ��
=B��{                                    Bxh���  �          @ָR?��@��H��Q��A�B�p�?��@��
�`����ffB��\                                    Bxh�݈  T          @أ�?�=q@�{����:�B��3?�=q@�{�XQ���ffB��R                                    Bxh��.  �          @���?�@�=q�����4p�B�u�?�@����O\)��Q�B�ff                                    Bxh���  �          @�=q?���@��H�����?Q�B���?���@�z��aG����B�k�                                    Bxh�	z  �          @ڏ\?�{@�
=��z�� Q�B�  ?�{@����.{��=qB��                                    Bxh�   �          @��
?���@��
����=�B��\?���@�p��b�\���B�#�                                    Bxh�&�  �          @ٙ�?˅@�����
=�H�HB�33?˅@�p��q��33B�\)                                    Bxh�5l  
�          @��
?�
=@�Q���z��N��B��3?�
=@�{�|���=qB�z�                                    Bxh�D  �          @ۅ@�@�p�����$(�B��@�@����8�����B�                                    Bxh�R�  �          @ۅ@z�@�  ������B�  @z�@��H�.�R��{B��3                                    Bxh�a^  �          @�z�?�{@�33��ff�D�\B�=q?�{@�
=�n�R�z�B��                                     Bxh�p  �          @�=q?޸R@r�\��z��Q�B���?޸R@����  ��
B�33                                    Bxh�~�  �          @��?�p�@h������V��B�  ?�p�@��
��z����B���                                    Bxh��P  �          @أ�?��@Y����G��[�Bs�
?��@����  �B���                                    Bxh���  �          @��
?�@u��  �U(�B�u�?�@��\���\��B�#�                                    Bxh���  �          @�G�?˅@k���\)�W(�B�Q�?˅@�p���33���B���                                    Bxh��B  T          @�=q?���@qG���\)�VQ�B�\?���@�Q����\�
=B�=q                                    Bxh���  �          @�{?�z�@c33���H�e�\B���?�z�@����\)�!(�B���                                    Bxh�֎  �          @�G�?��\@Fff��\)�yp�B��=?��\@�33����4B�=q                                    Bxh��4  �          @�?c�
@?\)�����B��f?c�
@������:�B�\                                    Bxh���  �          @�G�?��\@\����ff�i=qB�z�?��\@������$��B�                                      Bxh��  �          @�z�?�Q�@�  ���OB�ff?�Q�@�
=�|(��p�B��                                    Bxh�&  �          @�z�?�G�@{���ff�QG�B���?�G�@�p��~�R�(�B��
                                    Bxh��  �          @�33?���@~{���\�M\)B���?���@�p��vff�	\)B���                                    Bxh�.r  �          @�=q@
=q@����9G�Bz��@
=q@�  �Y����G�B�{                                    Bxh�=  �          @�33@�@�p���  �;�\B|�@�@�Q��]p���33B��H                                    Bxh�K�  
�          @ᙚ?�Q�@z=q�����O33B~=q?�Q�@�������B���                                    Bxh�Zd  �          @޸R?���@Vff��\)�m��B��3?���@�G���z��'�HB��R                                    Bxh�i
  �          @�ff?��\@L(��\�t�B��{?��\@���G��.G�B���                                    Bxh�w�  �          @�ff?�@J�H��=q�s��B��{?�@��������-��B�=q                                    Bxh��V  �          @�G�>��
@���  Q�B��>��
@�������^G�B��                                    Bxh���  �          @陚>�\)?���������B�p�>�\)@�������`�B���                                    Bxh���  �          @�>���@%���(�33B��\>���@�������I��B�Q�                                    Bxh��H  �          @�>8Q�@{��z��B��>8Q�@������H�W�B�G�                                    Bxh���  �          @��
=���@�
�߮Q�B�G�=���@��������UG�B��                                    Bxh�ϔ  �          @�\)<#�
@Q����H�)B�W
<#�
@�  ��\)�T33B���                                    Bxh��:  �          @�ff�u@���ᙚ� B�{�u@�  ���S\)B���                                    Bxh���  �          @�ff��\)@�H��G�  B�.��\)@�������R(�B��3                                    Bxh���  �          @�R�u@z����HǮB��)�u@�ff����U�\B�z�                                    Bxh�
,  T          @�ff��@\)��{B�Ǯ��@�(������X  B��f                                    Bxh��  �          @�����
@
�H���H��B����
@�������Y�B��                                     Bxh�'x  �          @��ͽ�G�@��33#�B�8R��G�@������[�B�\                                    Bxh�6  T          @�(��aG�@ ����33B�B����aG�@�p����H�^(�B��                                    Bxh�D�  �          @�ff��33?��H���B���33@������`  B��                                     Bxh�Sj  �          @���  ?����aHB����  @��H��ff�b(�B���                                    Bxh�b  �          @��aG�?����{p�B����aG�@��H��ff�b
=B�G�                                    Bxh�p�  �          @��u?�G���
=\)BÙ��u@\)��Q��e��B��                                    Bxh�\  �          @��;��R?�z���R��B�𤾞�R@y����G��hz�B�                                      Bxh��  �          @��;���?�����R33B��;���@vff�ə��i�
B�                                    Bxh���  �          @�R��Q�?�\)����=qB�B���Q�@x���˅�i�\B��                                    Bxh��N  �          @���?�\)���
=B�.��@y����z��jQ�B�aH                                    Bxh���  �          @�\)��G�?����=q=qB�#׽�G�@u��p��l�\B�B�                                    Bxh�Ț  T          @�\)�#�
?�ff���
B�L;#�
@hQ���G��sB�#�                                    Bxh��@  �          @�׼��
?��
��� B������
@vff�θR�l�RB���                                    Bxh���  �          @�ff>8Q�?�Q���  �\B���>8Q�@~{��G��f�HB��R                                    Bxh��  �          @�=q>B�\?У���z��fB�u�>B�\@~{��ff�iffB�ff                                    Bxh�2  �          @�  >L��?�G����H�{B��f>L��@u��{�l��B�
=                                    Bxh��  �          @�  >��?��
��\\B�� >��@w
=��p��k�B�33                                    Bxh� ~  
�          @�  >���?�\����33B���>���@�=q��G��c�HB�Ǯ                                    Bxh�/$  �          @�\)>�{?�(���=q�B��f>�{@s�
��p��l�HB�                                    Bxh�=�  �          @�R>��?�{��=q��B�  >��@l����ff�o�
B���                                    Bxh�Lp  �          @���?5?��H��  (�BnG�?5@c33��{�rB�aH                                    Bxh�[  �          @�p�>�?\��  G�B�W
>�@u�ʏ\�j33B�\)                                    Bxh�i�  �          @��>\?�������33B���>\@l������o��B�L�                                    Bxh�xb  �          @��H?�G�?�����z���BQ?�G�@hQ���G��lffB�                                    Bxh��  �          @�p�?У�?����W
B��?У�@fff�ȣ��g�B�33                                    Bxh���  �          @�?�z�?\(������\B?�z�@P  ��=q�y33B���                                    Bxh��T  �          @�p�?!G�?.{���H¥��B>33?!G�@G
=��{=qB�=q                                    Bxh���  �          @�G���\)?�33���B�\��\)@�����(��fB�{                                    Bxh���  �          @���\)?�ff��=qB��
��\)@{��˅�iG�B��                                    Bxh��F  �          @�{<��
?�G���G��B��=<��
@x����33�j{B�p�                                    Bxh���  T          @�    ?��R��ff��B���    @u�ȣ��j�B�                                      Bxh��  �          @�=q>L��?��
��{�B�(�>L��@i���ʏ\�p�B��                                    Bxh��8  �          @�\)>�33?��
��z�£8RB��q>�33@^�R��33�xG�B��{                                    Bxh�
�  �          @�>�33?�����R¢�
B���>�33@c33�����wQ�B��q                                    Bxh��  �          @�Q�>�=q?��\��£��B�G�>�=q@_\)��z��x��B��                                    Bxh�(*  �          @�ff>aG�?�G���(�¤{B�z�>aG�@^{���H�x��B�                                    Bxh�6�  �          @�G��L��?�ff��p� �B�\�L��@p�������p33B�ff                                    Bxh�Ev  �          @��H?�?@  ����¦{BZ
=?�@Q������HB��=                                    Bxh�T  �          @�=q?333?(���Q�¥�
B$  ?333@J=q���Hk�B��
                                    Bxh�b�  �          @��>��?J=q��33¥�3Bm=q>��@Q���(��}�HB���                                    Bxh�qh  �          @�\)>�Q�?�p�����B��>�Q�@fff��  �o�RB��=                                    Bxh��  �          @�Q�>B�\?�
=��(� ��B�\>B�\@dz���G��q\)B�                                    Bxh���  T          @�\�#�
?�����\)¢��B�L;#�
@a������tffB�{                                    Bxh��Z  T          @�녾Ǯ?�(���\�B���Ǯ@��������a�\B��
                                    Bxh��   �          @�녾�G�@�R���
{B���G�@�����=q�Ip�B�k�                                    Bxh���  �          @�
=�B�\@?\)��33L�B�8R�B�\@������
�;33B��                                    Bxh��L  �          @�{>�@Vff��ffB��\>�@������\�/33B�Q�                                    Bxh���  �          @�  �#�
@b�\�����{Q�B�#׾#�
@�
=��\)�)�\B�.                                    Bxh��  �          @�=q��=q@8Q��ָRaHB��쾊=q@��������9  B�.                                    Bxh��>  T          @��H�.{@9���׮\)B��=�.{@��\�����8��B��q                                    Bxh��  �          @���n{@Dz���(��}33Bծ�n{@��
�����-
=BȔ{                                    Bxh��  �          @�33�xQ�@AG���33�}�Bר��xQ�@����(��-��Bɽq                                    Bxh�!0  �          @�  �}p�@C�
��  �~��B׳3�}p�@����  �.\)Bɨ�                                    Bxh�/�  �          @�
=����@9����
=�~��B��)����@�  �����0z�B�33                                    Bxh�>|  �          @�G���  @P  ��z��u�B�#׿�  @������\�&(�BΔ{                                    Bxh�M"  �          @���z�@\����Q��b�HB����z�@����z���RB�(�                                    Bxh�[�  
�          @�G��ٙ�@<����{�x��B�
=�ٙ�@�G���\)�,G�B�L�                                    Bxh�jn  �          @��H��\)@*�H��{�3B�(���\)@�(������933B�G�                                    Bxh�y  �          @��k�@!G����
�B��ÿk�@�
=��G��?
=B���                                    Bxh���  �          @��H�W
=@�
��z�W
B�\�W
=@�����(��E�\B�{                                    Bxh��`  T          @���xQ�@%��33
=B���xQ�@��
��
=�?
=B�z�                                    Bxh��  �          @��ͿTz�@Fff��z��}\)B��Tz�@�{����+  B�33                                    Bxh���  �          @�\�\(�@,����Q��fB�p��\(�@�G���=q�=B�p�                                    Bxh��R  �          @���O\)@(Q�����B�
=�O\)@����p��Cz�B��                                    Bxh���  �          @�{�=p�@<(���\\B�.�=p�@�z������:��BÙ�                                    Bxh�ߞ  �          @�p���
=@@  �陚�HB��þ�
=@�{��
=�9�B��)                                    Bxh��D  �          A Q���@?\)���L�Bʞ����@�
=���\�:��B�z�                                    Bxh���  �          A��G�@>�R��
=z�B�G��G�@��\�Å�=ffB��f                                    Bxh��  S          A�
�L��@l�������
B�p��L��@�������.�B��
                                    Bxh�6  �          AG���\)@x��������B�#׽�\)@�  ��ff�)��B�G�                                    Bxh�(�  �          AG��\)@�(��G��{��B�  �\)@ٙ������&�B�p�                                    Bxh�7�  �          A���\)@���� (��}�B�{��\)@�p�����'�\B��H                                    Bxh�F(  �          A�(�@n{����\BƏ\�(�@�(��\�-�B��)                                    Bxh�T�  �          A�����@dz�� z�=qB�����@�Q�����0�B�=q                                    Bxh�ct  �          A�H���\@K��z�{B�p����\@�  ��G��<(�B�=q                                    Bxh�r  �          A��z�@:�H�Q�aHB�B���z�@�Q����
�A�\B�                                    Bxh���  �          A�
��?��H� (�ǮC����@�������V  B�B�                                    Bxh��f  T          A�
��  ?�33���C����  @���ٙ��Up�B܀                                     Bxh��  �          A�
�9��?��
����ffC }q�9��@xQ��޸R�\��B���                                    Bxh���  �          A�H�1G�?@  ����C$���1G�@hQ���=q�d�B��R                                    Bxh��X  �          A���?��R�����C��@�=q���H�^33B�R                                    Bxh���  �          A	p��G�?����R(�Cp��G�@�����\�^ffB��                                    Bxh�ؤ  �          A33���@���B�C=q���@��H���H�R33B�Ǯ                                    Bxh��J  �          Az��G�?޸R�G�#�C�
�G�@�����Y�\B��                                    Bxh���  �          A�\�z�?�ff��z�C33�z�@�����  �]��B癚                                    Bxh��  �          A{��R?�z��  Q�C�3��R@�����a�
B�                                      Bxh�<  �          A�\�33?�33�\)�fCT{�33@�z���{�[�B�ff                                    Bxh�!�  �          A�ͿУ�@5����)B���У�@�����Q��Cz�B��f                                    Bxh�0�  �          A�R��=q@.{�G�B�B��R��=q@��H�����E��Bֳ3                                    Bxh�?.  �          A=q���@�\�ffL�C\)���@����\)�Sz�B�W
                                    Bxh�M�  �          Ap��s33@L����
=B�
=�s33@�Q��ۅ�=33B�B�                                    Bxh�\z  �          Aff�c�
@K����ǮB�33�c�
@ȣ����>Q�B�#�                                    Bxh�k   T          A���
@Q��
{C��
@����p��OffB��                                    Bxh�y�  �          AQ���?�\��
  C��@����{�S��B�8R                                    Bxh��l  �          A�R�ff?��	�8RC�ff@����  �Q�B��H                                    Bxh��  �          A�H�Dz�?���	G���C!8R�Dz�@�\)��G��]  B��)                                    Bxh���  �          A\)�`  >���  �)C,W
�`  @k����b{C�
                                    Bxh��^  �          AG��}p�>.{�33�RC1�{�}p�@X����  �`=qC�                                     Bxh��  �          A�\���R��\)� ���k�CAh����R@�\����cQ�C��                                    Bxh�Ѫ  �          A�H����0����G��Bp�CNp����>������VC2u�                                    Bxh��P  �          A�
�Å��\)���
�33CX8R�Å��p�����<G�CE�                                    Bxh���  �          A(������c33��G��0CR����׿(����(��NQ�C:E                                    Bxh���  �          A����p��G
=�ٙ��.�CMٚ��p���  ��\)�EG�C65�                                    Bxh�B  �          AG��ə��A���(��2{CM���ə��\)��Q��H  C5G�                                    Bxh��  �          A ����G��=q��\)�8�CH:���G�?���33�D��C/                                      Bxh�)�  
�          A   ��G��!G���(��6  CI
��G�>�G����D33C0+�                                    Bxh�84  �          A�R���5�ڏ\�4\)CL����<���z��HffC3��                                    Bxh�F�  �          A�R��\)�U�˅�$Q�COB���\)�(������>��C9W
                                    Bxh�U�  �          A �����
�j�H����+=qCQ����
�:�H���R�H�HC:��                                    Bxh�d&  �          A!�����\)��=q�JCJ�H����?B�\��R�W��C,��                                    Bxh�r�  �          A!���녿��R� ���Rz�CF����?�ff��H�W��C'W
                                    Bxh��r  �          A!������ff�V�CF����?��  �Z�C%�                                    Bxh��  �          A!G���=q��G��	��k�CA����=q@{��H�b��C^�                                    Bxh���  �          A"ff��zῘQ��(��d
=C@h���z�@  ����[G�C^�                                    Bxh��d  �          A$����(������o33C:c���(�@=p��{�[  C                                    Bxh��
  �          A#�
��(�>aG�����|
=C1L���(�@r�\�z��X��C�q                                    Bxh�ʰ  T          A$�����R>��H�(�B�C-�q���R@�33���W��C	h�                                    Bxh��V  �          A$������>����33�|�C0E����@z=q�G��W�C�                                    Bxh���  
�          A&�\���>�\)����}\)C0�{���@|(���H�Xp�C�{                                    Bxh���  �          A'����?�R�z��y��C,����@�  �z��P��C
�{                                    Bxh�H  �          A&�\���?z���
�z�RC-����@�ff�  �Q��C
n                                    Bxh��  �          A'33��\)>�����{�HC.B���\)@�z�����T=qC
�=                                    Bxh�"�  �          A&�H��=q>�Q���
�y��C/�3��=q@�Q��G��T(�C5�                                    Bxh�1:  �          A&�R���H?(���33�x(�C,33���H@�����R�N�RC
�                                    Bxh�?�  �          A'���z�?�G����{��C'���z�@���=q�LQ�C�                                    Bxh�N�  �          A&�\���R?��
����|\)C!����R@��
�����E=qC�                                    Bxh�],  �          A'\)���?�G��Q��yp�C!޸���@�33��z��C�C!H                                    Bxh�k�  �          A'
=���H?����  ��C#�����H@�����Pz�C�=                                    Bxh�zx  �          A&ff�y��?�(���
��C.�y��@�z�����H�B�                                    Bxh��  �          A&�R�xQ�?�Q�����Cz��xQ�@��H��z��D��B��=                                    Bxh���  �          A%��\)@ ������HCY��\)@�33����Az�B���                                    Bxh��j  
�          A�H�@��@s33�
{�r�C ^��@��@޸R�˅��HB�q                                    Bxh��  �          A��8Q�@\)��
�o�\B��q�8Q�@�\�����33B�Q�                                    Bxh�ö  �          A!G��`��@Fff��R�y{C
���`��@�ff�����,B�#�                                    Bxh��\  �          A#\)�W
=@P  �p��{33C���W
=@����߮�,��B�                                    Bxh��  �          A$���P��@9�������C
\)�P��@���\�5�HB�q                                    Bxh��  �          A%���U�@9���u�C
��U�@θR���
�5�B�=                                    Bxh��N  �          A%���P  @(Q��\)G�C���P  @�Q�����;��B��)                                    Bxh��  �          A%G��B�\@.�R��\)C
\�B�\@��
��G��;��B�\                                    Bxh��  �          A&�H�AG�@Dz��  =qC���AG�@�{����533B䙚                                    Bxh�*@  �          A#\)�/\)@2�\��H�Ch��/\)@����ff�;�B�=q                                    Bxh�8�  �          A"�H�>{@Tz��{C޸�>{@أ���
=�-=qB�ff                                    Bxh�G�  �          A$(��:=q@w
=����w�\B���:=q@�����"  B߽q                                    Bxh�V2  �          A#�
�$z�@�������x33B����$z�@����=q��
B�B�                                    Bxh�d�  T          A$Q��\)@�33���s\)B��\)@����z����B���                                    Bxh�s~  �          A&�\�!G�@�33���y�HB�{�!G�@�G��ָR� �B��f                                    Bxh��$  �          A#\)��@��H�\)�uffB�k���@������
�Q�B�#�                                    Bxh���  �          A$  �z�@��  �v
=B���z�@�Q��˅���B��)                                    Bxh��p  T          A!����{@�Q��=q�vB�ff��{@�G���
=��B�k�                                    Bxh��  �          A �׿�(�@�G��p��vz�B��쿼(�@���������B�Q�                                    Bxh���  �          A �Ϳ�  @������uffB�33��  @������H��B�\                                    Bxh��b  �          A zῡG�@��Q��tG�B�8R��G�@��������
B�(�                                    Bxh��  �          A!녿J=q@�33���z(�BǙ��J=q@�p��Ǯ�ffB�p�                                    Bxh��  �          A!��u@��R�\)�|{B�ff�u@�G��������B��                                    Bxh��T  �          A�����@�\)�
=�r��B������@�ff�������B�p�                                    Bxh��  �          A�\����@�G��\)�w=qB�q����@�G����
��HBϙ�                                    Bxh��  �          A\)�s�
@���=q�[�C��s�
@������	p�B�\)                                    Bxh�#F  T          A ���E�@u�(��r�C �q�E�@�����H�ffB�=                                    Bxh�1�  �          A"�H�'�@������o�
B�{�'�@��������HB��
                                    Bxh�@�  �          A"�\�&ff@�Q����l��B���&ff@�Q���Q���B�
=                                    Bxh�O8  �          A!��Fff@���(��e\)B�W
�Fff@����=q��RB��                                    Bxh�]�  T          A"ff�333@�(�����f�B�\)�333@��������B�p�                                    Bxh�l�  �          A"ff�>�R@����\)�XG�B�
=�>�RAff��Q����B�(�                                    Bxh�{*  �          A!G��E�@�  �
=�Yp�B�Q��E�@���������B�#�                                    Bxh���  �          A ���8Q�@�������`��B��H�8Q�@�z������G�B�{                                    Bxh��v  �          A Q��Z�H@������QB�(��Z�H@�ff��G���B�\                                    Bxh��  �          A z��b�\@���� ���U�
B��b�\@��������ffB���                                    Bxh���  T          A z��QG�@�z���X=qB���QG�@�(���  ���B�\                                    Bxh��h  �          A��7
=@�������M��B�q�7
=A����R��33B��H                                    Bxh��  �          A�\�9��@�=q���Vp�B�9��@�
=��  ��(�B�\                                    Bxh��  T          A���@�
=�=q�c��B�z���@�  ���H�\)B���                                    Bxh��Z  �          A��0  @�p�� Q��[�B�G��0  @�(���(����RB�ff                                    Bxh��   �          A=q�:=q@�33�(��bp�B����:=q@�ff��\)�B�p�                                    Bxh��  �          Az��,��@�G����R�YffB�ff�,��@�
=��������B�p�                                    Bxh�L  �          AG��Mp�@�{��\�`p�B��q�Mp�@�Q���ff��
B�B�                                    Bxh�*�  �          A�
��@��R����]Q�B����@�p���=q���
B�{                                    Bxh�9�  �          Az��<(�@�\)� ���]�B����<(�@�\)���R� �
Bݮ                                    Bxh�H>  �          Ap��N�R@����]��B��{�N�R@���33��B��                                    Bxh�V�  �          A(��QG�@�������S�B�Q��QG�@�G���(���(�B�\                                    Bxh�e�  �          Ap��P  @��
��\)�O�B��H�P  @��R��  ��G�B�u�                                    Bxh�t0  �          A���<(�@�ff��p��W�B�p��<(�@��������
=B��H                                    Bxh���  �          A��1G�@����\�M�B�#��1G�A�\��\)�ۙ�B�k�                                    Bxh��|  �          A��N{@�p���ff�J�HB��H�N{@�z����R���HB�k�                                    Bxh��"  �          A���*=q@����
�RffB�\)�*=q@�\)��33��B��H                                    Bxh���  �          Ap��p�@�p���{�UffB��f�p�A   ��p���33B�(�                                    Bxh��n  �          A���
=@�{���U��B���
=A Q���z���Q�B���                                    Bxh��  �          A��G�@�(������T=qB���G�A�H�������
BϸR                                    Bxh�ں  �          A=q���@�{��z��[�HB�{���A���=q��(�B�B�                                    Bxh��`  �          A���
=q@������W�B�Ǯ�
=qA�����z�B��
                                    Bxh��  �          A=q��
=@��
��Q��V��Bۊ=��
=A����
����B�ff                                    Bxh��  �          A{�	��@�z����L�B�Ǯ�	��A��  ��B��f                                    Bxh�R  �          A���=q@�����(��H�HB�33�=qA���z���ffB�L�                                    Bxh�#�  �          A���@�ff�陚�F��B����AG���G��ǮB�u�                                    Bxh�2�  T          A���\)@�{��Q��E�HB�8R�\)A����Q���z�B�Q�                                    Bxh�AD  �          AG��%�@�
=���CffB䙚�%�A���z�H��{B֏\                                    Bxh�O�  �          A��!G�@����
=�FG�B�p��!G�A��\)��G�B��                                    Bxh�^�  �          A���*�H@����z��K{B� �*�HA���\)�ң�B�aH                                    Bxh�m6  �          A��0��@�������G(�B�(��0��A
=��33��
=B�L�                                    Bxh�{�  �          A���(��@�{��z��J��B��
�(��Aff���R��p�B��f                                    Bxh���  �          A��7
=@�  ��ff�=(�B�  �7
=A�
�k����HB�\)                                    Bxh��(  �          Aff�U�@Ϯ������B�L��U�A33����m�B�\                                    Bxh���  �          A���P  @��
����{B�B��P  A
=���R�HB�\                                    Bxh��t  �          A=q�7
=@�ff����4�B�Q��7
=A���S33��
=B�
=                                    Bxh��  �          A���W�@�33����\)B�ff�W�A  �ٙ��((�B�B�                                    Bxh���  �          A=q�c�
@陚������=qB�
=�c�
A	�c�
��\)B���                                    Bxh��f  �          Aff�?\)@�{���\�B��?\)A33����9�B���                                    Bxh��  �          A�����@�\�j�H��33B�����A33    <�B�                                    Bxh���  �          A�
��ff@����t��B����ff@�ff?��
@�
=B�ff                                    Bxh�X  �          A����
=@�(��Mp���B�����
=A(�?\)@\(�B�.                                    Bxh��  �          A�����@��
�A�����B�
=���A�R?8Q�@�
=B�                                    Bxh�+�  �          A��@߮���*�\B���@߮?�z�A)B��q                                    Bxh�:J  �          A�׿�(�@�{���H�
=B��f��(�A�
��p��MB��                                    Bxh�H�  �          A33���@�\)�ٙ��9=qB�=q���A���H����=qB�
=                                    Bxh�W�  �          Az�<��
@�G����:B��3<��
A\)�N�R����B�Ǯ                                    Bxh�f<  �          A>��@ҏ\��Q��;Q�B�Q�>��A���Q���p�B�
=                                    Bxh�t�  �          A  ?Y��@�=q��G��?G�B��3?Y��A��Z=q��ffB�                                      Bxh���  �          A��?}p�@�\)��ff�:��B���?}p�A�R�P������B�W
                                    Bxh��.  �          A(�?���@�\)���
�9�B�Q�?���A=q�K���Q�B�W
                                    Bxh���  �          A��?}p�@�=q�ۅ�7�B��f?}p�A��G�����B�p�                                    Bxh��z  �          A�?W
=@�=q�����6��B�\)?W
=A
=�C33��  B�33                                    Bxh��   
�          A33?333@У��ٙ��8�B��R?333Aff�E���Q�B�                                      Bxh���  �          A��?B�\@�Q�����5�
B��q?B�\AG��=p����B�33                                    Bxh��l  �          Aff?333@����Ӆ�2��B��?333A
=�5���RB�                                      Bxh��  �          A
=?G�@��˅�)\)B��?G�A��\)�tQ�B�#�                                    Bxh���  �          Az�?=p�@�G���z��0B��?=p�AG��333����B���                                    Bxh�^  �          A��?�=q@�ff��z��(ffB�G�?�=qA�   �s
=B�W
                                    Bxh�  �          A�
?��\@�G���Q��-(�B��)?��\AQ��*�H����B�                                      Bxh�$�            A�?�\)@ۅ�����)B�k�?�\)A���"�\�x  B�                                   Bxh�3P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxh�A�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxh�P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxh�_B  �          A\)?У�@�G���Q��(
=B���?У�A�
�!��q�B�\                                    Bxh�m�  	.          A33?n{@�z���\)�(  B�33?n{A�����j�HB���                                    Bxh�|�  
�          AQ�?��@���ff�%�\B�aH?��A�\���aG�B�(�                                    Bxh��4  �          A\)?��
@�Q��˅�#�
B��)?��
A{���YB�z�                                    Bxh���  �          Az�?aG�@���33�"G�B�Q�?aG�A\)�{�R{B�W
                                    Bxh���  "          A33>Ǯ@��
��\)�ffB�33>ǮAz��\�((�B�\)                                    Bxh��&  �          A (�?aG�@��
��p���RB��)?aG�A��	���F�RB��                                    Bxh���  �          A ��?��@�
=���H�#�
B��3?��A�H�
=�YB�B�                                    Bxh��r  T          A!�?��@�R�ҏ\�$�B�aH?��A�R�
=�Z=qB�#�                                    Bxh��  �          A Q�   @�p���33�Q�B�G��   A����(��{B��)                                    Bxh��  �          A!녾�Q�A�����p�B����Q�A������B�#�                                    Bxh� d  �          A!녿�A�
���H�{B��
��A (�������Q�B�u�                                    Bxh�
  "          A!����Az���  �
�RB��)��A (����H����B��                                     Bxh��  �          A"{�z�HA\)��\)�(�B�=q�z�HA �׿fff��=qB�(�                                    Bxh�,V  �          A"=q���\@����ff��B�33���\A=q���
�!��B�                                    Bxh�:�  "          A"=q�Dz�A ����=q��\)B��Dz�A���n{��  B�u�                                    Bxh�I�  "          A ���4z�@���Q���
B���4z�Ap���33���B�                                    Bxh�XH  T          A!���A{������B�\���A�׿�����\)B�                                      Bxh�f�  "          A"{��A33��G���B�Ǯ��Ap�������
B���                                    Bxh�u�  �          A"=q�3�
A�����G�B�k��3�
A�R�z�H��G�B�p�                                    Bxh��:  T          A"{�k�@������ffB��k�Aff�E���z�B޸R                                    Bxh���  T          A!녿�=qA�R���
�ffB�=q��=qA��{��G�B�
=                                    Bxh���  "          A"{�>�RA=q������B�8R�>�RAff�G����B�Q�                                    Bxh��,  T          A�
���A z���ff�=qB�{���A(���p���33B�\                                    Bxh���  "          A�H���@���=q��B�8R���A녿�z��ָRB��                                    Bxh��x  T          A���C33@�  �������
B���C33A(��^�R���B�k�                                    Bxh��  
Z          A�
�:�H@������H��(�B�(��:�HA(��O\)��  B�                                      Bxh���  
�          A{�E�@�z���\)��
=B��
�E�Ap��B�\����B�aH                                    Bxh��j  �          A\)�7�@�����=q����B܀ �7�A  �E����\B�u�                                    Bxh�  T          AQ���@��
�����B�L���A33�n{��\)B��                                    Bxh��  T          A
=��RA ��������B����RAff�n{��(�B���                                    Bxh�%\  T          AG�����@�p����H�
�RB��f����A녿�33��\)B��                                    Bxh�4  "          A  ����@������R��BǨ�����A���������B�\                                    Bxh�B�  T          A(����@�(������
�B�(����A���\)��33B�Q�                                    Bxh�QN  "          A�׿�@������H�\)B�aH��A����ָRB���                                    Bxh�_�  "          A�Ϳh��@�(���Q���HB�33�h��A�R��ff��B�                                    Bxh�n�  �          A�H��@�33��{���B�p���A녿O\)����B�aH                                    Bxh�}@  "          A(���\@��H���\��\BԞ���\A
=�p����  B�Q�                                    Bxh���  �          A"�R����A   ��\)���B�uÿ���A=q��� z�B�\)                                    Bxh���  "          A#���
=@�
=��p���B��)��
=A33�˅��B�8R                                    Bxh��2  T          A$  ����@��R�ə���RBǞ�����A   ���H���Bè�                                    Bxh���  "          A$  ���
A (���Q��B�#׿��
A z��33���B                                    Bxh��~  
�          A!��Q�@���Ǯ���B�𤿸Q�A����(����BĮ                                    Bxh��$  �          A#�
�c�
AQ������B�녿c�
A"�R�Q����
B�                                      Bxh���  �          A#��O\)A�\��33�z�B�Q�O\)A!p�����
=B�(�                                    Bxh��p  "          A"=q��p�A����ff��HB��
��p�A (���  ��B���                                    Bxh�  "          A#����A33��p���
B�G����A �׿��R��
=B���                                    Bxh��  "          A$�׿�=q@�p��ə���B���=qA��ٙ��(�B���                                    Bxh�b  
�          A#\)��ff@�ff�����p�BƊ=��ffA
=��ff���B��H                                    Bxh�-  
Z          A"�R�(�A\)���
�(�B�LͿ(�A!녿Tz����RB��                                    Bxh�;�  
�          A&�\��A	����\�(�B�p���A%녿n{���
B�\)                                    Bxh�JT  
�          A&�H=uA
�R������B�(�=uA&=q�aG����HB�L�                                    Bxh�X�  �          A&�R�L��A
�\�����(�B���L��A&{�aG���=qB��\                                    Bxh�g�  T          A&�H��{A
�R��G���
B�zᾮ{A&=q�\(���\)B��q                                    Bxh�vF  T          A'\)��A	��p��	��B�p���A&ff��  ��Q�B�Q�                                    Bxh���  �          A'33�(��A����ff�
�HB��
�(��A%녿�����B�L�                                    Bxh���  �          A((����Ap���Q���Q�B��H���A(  �8Q�xQ�B���                                    Bxh��8  	�          A((����A  �������B��)���A((�>\)?@  B���                                    Bxh���  L          A'\)�!G�A������G�B�(��!G�A&�H���6ffB��                                    Bxh���  T          A&�\���A���33���\B�k����A%����33��
=B��                                    Bxh��*  
�          A*=q?
=A��������G�B���?
=A)��?��@N{B���                                    Bxh���  �          A*�\?�33A�
������B�?�33A)p����
��Q�B��{                                    Bxh��v  T          A)�?aG�A���p���G�B�(�?aG�A)G��#�
�#�
B��=                                    Bxh��  
�          A*{?�A{������B�k�?�A(�þ.{�p��B�aH                                    Bxh��            A+�?J=qAG������
=B�?J=qA*=q������B�z�                                    Bxh�h  T          A/
=�\)A&�H�R�\��  B��{�\)A+\)@\)A<��B��=                                    Bxh�&  �          A0(��#�
A%�o\)����B�p��#�
A-?�A�B�k�                                    Bxh�4�  
�          A/�>�Q�A�����\)B��)>�Q�A.�\?333@j=qB�B�                                    Bxh�CZ  
�          A/33>�A\)���R��B�k�>�A.�H���!G�B�(�                                    Bxh�R   �          A0  ?.{A
=���H��z�B���?.{A/��u��p�B���                                    Bxh�`�  �          A.�H?&ffA  ��Q����HB��?&ffA.=q���   B�(�                                    Bxh�oL  
�          A-��?�A
=��\)���B��H?�A-�����   B���                                    Bxh�}�  �          A/33>�A33��G��\)B��)>�A.{��=q���B��                                    Bxh���  "          A2=q?�A������RB�ff?�A1��Q���\)B��                                     Bxh��>  
�          A3
=>�ffA���� �B��=>�ffA2�\�   �#33B�\)                                    Bxh���  �          A2�H?��A�R��
=�G�B�W
?��A2=q����5�B�\)                                    Bxh���  
�          A3\)>�  Ap������{B��>�  A1p��!G��N�RB�ff                                    Bxh��0  
�          A,Q�>L��A���fff����B��>L��A$��?ٙ�A�B���                                    Bxh���  T          A)����A(Q켣�
����B�����Aff@�A뙚B�z�                                    Bxh��|  
Z          A&�R��ffA%p�?z�@K�B�����ffA33@��Bz�B�                                    Bxh��"  �          A$�;ǮA ���
=q�B{B�k��ǮA��@FffA��B��=                                    Bxh��  |          A)��=���A%�����*�HB��=���A�@[�A��
B��H                                    Bxh�n  �          A4  ��z�A1������HB��ᾔz�A"�\@�33A¸RB�B�                                    Bxh�  �          A4  ���A0��?��RA"�HB�녾��A	�@�ffBp�B�L�                                    Bxh�-�  �          A4�þ�\)A3\)?�\)@޸RB��쾏\)A��@�  B�RB�z�                                    Bxh�<`  �          A7
=��z�A3�@{A333B�𤾔z�A
=q@�B#�RB���                                    Bxh�K  �          A5����p�A/�@5AhQ�B��쾽p�A@��B1{B�.                                    Bxh�Y�  �          A*�H�p��A ��@S�
A�\)B����p��@�\@��
B?(�B��                                    Bxh�hR  �          A  ?B�\@�33��ff�G�B���?B�\@��z����B�                                    Bxh�v�  �          A/\)@K�@2�\�!��� B%  @K�@�ff���
�)=qB��3                                    Bxh���  T          A,Q�@L(�?�(��"=q�3A���@L(�@�G���{�>  B�\                                    Bxh��D  
�          A$  @HQ�?(������A>=q@HQ�@�p���Qp�Bp{                                    Bxh���  �          A+\)@J�H?���!�#�A�(�@J�H@����33�I�By                                      Bxh���  �          A+�
@B�\?�{�"�H��A��
@B�\@�
=� Q��A
=B��R                                    Bxh��6  �          A,(�@J�H?�  �!���A癚@J�H@�=q�����=�B��                                    Bxh���  �          A1p�@S33@�R�$��G�B�@S33@�G����/Q�B�=q                                    Bxh�݂  �          A(��@S�
@g
=�p��y\)B=�H@S�
@�G���G��p�B���                                    Bxh��(  �          A(Q�@[�@|(��{�p�BC�R@[�@�\)��{�	�B�p�                                    Bxh���  �          A/�
@H��@x���Q��{�\BL�@H��A(���Q���\B�aH                                    Bxh�	t  �          A5��@,��@\)�$Q��3B_p�@,��A	p���z���B�\                                    Bxh�  �          A333@3�
@�z�� (��}�RB_p�@3�
A	�����H�=qB���                                    Bxh�&�  �          A0Q�@<(�@����
=�l�RBj  @<(�A{��Q�����B�\)                                    Bxh�5f  �          A
=@3�
@��\�=q�^
=Bo33@3�
A(����
���B�ff                                    Bxh�D  T          @��?L��@�Q쿷
=���B�?L��@���?fffA6=qB�k�                                    Bxh�R�  T          @���<#�
@���?�ffA�33B���<#�
@G�@x��BM�B��q                                    Bxh�aX  �          @�(�?
=@�ff>���@���B���?
=@i��@%�Bz�B��q                                    Bxh�o�  �          @��H?�{@�=q��G��<  B���?�{@Ӆ�����HB��=                                    Bxh�~�  �          @���?�G�@{��S�
�B�33?�G�@�=q�aG���B�u�                                    Bxh��J  �          @�\)>��@s33�dz��,z�B�B�>��@��\���O�B�33                                    Bxh���  �          @u�>��@h�ÿ�=q����B�B�>��@k�?fffAZ�RB�k�                                    Bxh���  �          @\��>�=q@Q녿p���~=qB�� >�=q@S�
?W
=Ac�
B��{                                    Bxh��<  �          @W�?&ff@:�H���ʸRB�Ǯ?&ff@O\)>�  @���B�.                                    Bxh���  �          @|��?h��@<���p��p�B��3?h��@j=q��
=��(�B�{                                    Bxh�ֈ  �          @s33?xQ�@2�\���(�B���?xQ�@fff�#�
�Q�B��f                                    Bxh��.  T          @���?�R@HQ�����33B�aH?�R@{��
=q��{B���                                    Bxh���  �          @vff?(��@@���  �p�B�33?(��@n�R��
=���HB��f                                    Bxh�z  �          @]p�?L��@p��33�&p�B��?L��@Q녿=p��EB��{                                    Bxh�   
�          @j=q?.{@*=q�33���B��?.{@]p��#�
�$  B��H                                    Bxh��  �          @���?�=q@]p���z���=qB�p�?�=q@l��?�@��B��=                                    Bxh�.l  �          @���?xQ�@x���"�\��B��)?xQ�@�z�k��.�RB�G�                                    Bxh�=  �          @w
=?E�@<(��
=�Q�B�u�?E�@n�R����
=B��=                                    Bxh�K�  �          @�  ?��@c�
�{�B���?��@�=q������p�B��\                                    Bxh�Z^  �          @��\?�G�@K��8���!{B���?�G�@��R�c�
�6�RB��3                                    Bxh�i  �          @p��?�ff?��G
=�`ffBoff?�ff@H�ÿ�\����B���                                    Bxh�w�  �          @]p�?E�?�{�E��|Bq��?E�@.�R��(���B��{                                    Bxh��P  �          @n�R?�{?˅�J�H�j�B\�?�{@>�R����p�B�(�                                    Bxh���  �          @j=q?���?�{�Dz��g=qBa�H?���@<�Ϳ�=q��{B�(�                                    Bxh���  T          @XQ�?E�?z�H�E
=BP\)?E�@�H�
=q�!�HB���                                    Bxh��B  T          @Z�H?�p�?�ff�8���h�B:G�?�p�@%��=q�33B�#�                                    Bxh���  T          @�  @	��@��@  �433B/��@	��@S�
���
��
=Bc�
                                    Bxh�ώ  �          @y��?�?�Q��Fff�WG�A�ff?�@&ff��
��HBV
=                                    Bxh��4  �          @hQ�?�33=�Q��L(��z�@Dz�?�33?ٙ��,���C{B7ff                                    Bxh���  �          @k�?�
=�Q���R�\)C�u�?�
=�h���HQ��v�C���                                    Bxh���  �          @,(�?��
����G��'�
C���?��
�
=q�
=�Q�C�4{                                    Bxh�
&  �          @%?u���H��(��$��C���?u����ff�=C�3                                    Bxh��  �          @1�?=p��	����ff�	(�C���?=p�������w�
C���                                    Bxh�'r  �          @8Q�?k���33��\�<z�C���?k��\�&ff�C���                                    Bxh�6  �          @�
?��H��������4�\C���?��H���
��z��f�C�7
                                    Bxh�D�  �          @\)?�(��}p��޸R�1C��?�(�=#�
�   �Vz�?�
=                                    Bxh�Sd  �          ?��>�33>\)>�  B{A�(�>�33    >�z�BQ�C���                                    Bxh�b
  �          ?�ff����?�p��W
=�Q�B�8R����?�z�>�ffA��\B�                                    Bxh�p�  �          @����@l(�����  B�녾�@��>���@�{B��                                    Bxh�V  
�          @�{��@�ff�����p�B��ÿ�@�
=������Bҙ�                                    Bxh���  T          @�33��ff@y����ff�X�B���ff@�z��G���p�B�p�                                    Bxh���  �          @�ff�J=q@o\)��G��bG�B���J=q@��Dz���B\                                    Bxh��H  �          @�G����@�33��{�;{B�
=���@�=q��p���33B�G�                                    Bxh���  �          @��ͿxQ�@����  �=qBȔ{�xQ�@��������\B�
=                                    Bxh�Ȕ  �          @陚�:�H@�\)��
=�-33B��)�:�H@�G������K�
B���                                    Bxh��:  �          @�  ��{@��
���R�;z�B�\��{@�33��p���
=B��q                                    Bxh���  �          @�33�#�
@˅�k���{B��H�#�
@�33=���?G�B��                                     Bxh��  �          @��\��33@�������Q�B����33@�p����\���B�(�                                    Bxh�,  �          A�þ�ff@��\��  �;�RB��H��ffA\)�Q���\)B�Q�                                    Bxh��  �          A33��@���p��*
=B�LͿ�A�ÿ��
�<  B�                                      Bxh� x  �          @��þ��R@�=q�   ���RB�\)���R@�p�?�(�A�B�{                                    Bxh�/  �          Az῔z�@�����  � �HBȊ=��z�A�ÿ�Q���B��H                                    Bxh�=�  �          AG�����@�=q��=q�ffB�G�����@���@ffA�{B�z�                                    Bxh�Lj  �          A	���!�@�=q��������B�Q��!�Aff>��?�G�B�p�                                    Bxh�[  �          A��G�@ڏ\�������B����G�A녾�=q��B�B�                                    Bxh�i�  �          A���8Q�@�ff��  �B��)�8Q�@�z῎{����B�#�                                    Bxh�x\  �          A
=q��(�@�������1\)Bي=��(�A ���
=q�ip�Bυ                                    Bxh��  �          A�R�Q�@�
=���\����B���Q�@����R�	��B��                                    Bxh���  �          AQ��S33@����p��p�B�{�S33@�
=�+����RB��                                    Bxh��N  �          @�
=�(�@��S�
�ڏ\B��(�@�G�>W
=?��Bۀ                                     Bxh���  �          @�33�{@�ff�!����B��H�{@���?c�
@�B�33                                    Bxh���  �          @���z�@�z��w����B��z�@Ӆ�����\)B�                                    Bxh��@  �          @�p��
�H@�  ������B����
�H@��Ǯ�>�RB�u�                                    Bxh���  �          A�׿��H@��
�o\)�ָRB��
���HA Q�>�@K�B�z�                                    Bxh��  �          A��%�@���������RB�u��%�A�H�   �FffB�=q                                    Bxh��2  �          Ap��?\)@�
=���\�  B�L��?\)A�
=�\��B�aH                                    Bxh�
�  �          A��ff@�{�����
=BӅ�ffA����<��B�\)                                    Bxh�~  �          A�R�AG�@�Q���p���\)B�33�AG�A
{�����\)Bڞ�                                    Bxh�($  �          A����HA\)�h����(�B�����HA�?�p�@��B�B�                                    Bxh�6�  �          Ap��-p�@��
��������B�.�-p�A녾\��B�#�                                    Bxh�Ep  �          A ���\)A
=q�����ffB�{�\)Az�?   @7�B���                                    Bxh�T  �          A,z��%�A�������  B���%�A'
=?8Q�@s�
BϸR                                    Bxh�b�  �          A+\)�'
=Ap������\)B��'
=A$��?�@5B�u�                                    Bxh�qb  �          A�\�Q�@�Q���Q�����B���Q�A��\)��ffB�Q�                                    Bxh��  �          A#��/\)A{�����B�G��/\)A������\)BӀ                                     Bxh���  �          A,���~�RA	���  ����B��
�~�RA�R>\)?B�\B�                                    Bxh��T  
�          A5�У�@陚�Q��B��B���У�A*ff�a���\)B�\)                                    Bxh���  �          A<Q��{A Q��ff�6G�B����{A333�C�
�p��B��)                                    Bxh���  �          A-p����
@�\���/�B�uÿ��
A%���   �V=qBǅ                                    Bxh��F  �          A��У�@�
=���H�:��B�W
�У�A33�.�R��z�B�#�                                    Bxh���  �          A'�
��ff@�G���\�9B��Ϳ�ffA
=�7
=�}��Bų3                                    Bxh��  �          A/\)��
=@��
�  �K�RBԏ\��
=A ���w
=���B�                                    Bxh��8  �          A1녿�z�@�Q������1  BϸR��z�A*�\�*=q�\��B�\)                                    Bxh��            A4����
@�33���
�0�\B�k���
A,z��,(��\(�Bɞ�                                    Bxh��  
�          A5����33A
{��p����Bɨ���33A1����
=�(�B��                                    Bxh�!*  �          A7
=��33A
=��\�&{B��)��33A2=q�Q��+�B�W
                                    Bxh�/�  
�          A+33����A����z��B�LͿ���A'33��z��ƸRB�#�                                    Bxh�>v  
�          A�R�a�@�G��Q��ep�B�#��a�@��@�A_�B�\                                    Bxh�M  "          Az����@��@�{B{C�����?�G�@��BL�HC'��                                    Bxh�[�  �          A{��  @��H@��RB��C�
��  @ ��@�  BHz�C�                                    Bxh�jh  "          A
�\����@�@�ffB=qC
����?xQ�@�ffBV
=C))                                    Bxh�y  
�          A   ���@��@�p�BQ�Cs3���?�33@���BKz�C&�3                                    Bxh���  T          A����p�@��@XQ�A���C8R��p�@S�
@�ffB!ffC�                                    Bxh��Z  T          @�
=��=q@�p�@w�A�  C޸��=q@��@��B:�C}q                                    Bxh��   "          A�����@�
=@�
=B  C
� ����?���@�33BJ  C$�3                                    Bxh���  �          @�����z�@�
=@o\)A��HC���z�?�@�z�BC�RC�                                    Bxh��L  �          @��
�n{@���@��B��C���n{?���@�33B[z�C z�                                    Bxh���  �          @�{��R@�z�@�G�B)�\B�\��R?���@�=qB���CJ=                                    Bxh�ߘ  T          @��Ϳ��@�{@}p�B=qB�����@�@�ffB���B�B�                                    Bxh��>  
�          @�  ��@���@�BB�ff��@%�@�ffB�B�                                    Bxh���  "          A\)���
@ʏ\@qG�A���C�
���
@^{@�  B.��Cp�                                    Bxh��  
�          A4(��Q�@ٙ�@^{A���C���Q�@���@�\)BC�                                    Bxh�0  T          A0����
=@�ff@I��A���C
���
=@��@�G�B{C�{                                    Bxh�(�  
�          A&{���@�(�@FffA�{C	�H���@�=q@��HB��C�                                    Bxh�7|  �          A�R���H@��>��@8Q�B�B����H@�{@�G�A��C
                                    Bxh�F"  T          A\)��  @��
?�33@���C ����  @��@��A��CQ�                                    Bxh�T�  
�          A
=���
@�\?L��@��C�����
@���@�33A܏\C	J=                                    Bxh�cn  �          A���(�@��?�A3�CB���(�@�\)@��B��C��                                    Bxh�r  "          A  ����@�
=?�Q�A7\)C#�����@��R@�Q�BCǮ                                    Bxh���  "          AG��ָR@�
=?��@��HC���ָR@��@�ffA��Cp�                                    Bxh��`  "          A����(�@��
���
� ��C	B���(�@���@9��A��
C�\                                    Bxh��  
�          @�
=���@��H@\(�A��C޸���@$z�@��\B;G�C�{                                    Bxh���  �          @���?\)@[�@8Q�Bz�C!H�?\)?�Q�@��BOCO\                                    Bxh��R  �          @Vff�Q�?���Q��  CJ=�Q�@�5�W\)C�                                     Bxh���  �          ?�zῃ�
?5������C�=���
?W
=����
=C��                                    Bxh�؞  "          ?�(�����>�׿Tz���p�C aH����?Tz�����p�C�                                     Bxh��D  "          ?�녿�=q?�논��
�
=Ck���=q?z�H?�A�Q�C	��                                    Bxh���  
Z          ?�(��O\)?p�׾�G�����C�R�O\)?��
=�G�@�z�C aH                                    Bxh��  "          ?�\)?�=L�;�Q����@�z�?�>k���\)����A��\                                    Bxh�6  "          ?�녿����#�
��ff�#�HC;� ����>�녿z�H���C!0�                                    Bxh�!�  
�          @;��(��>����\��ffC)� �(��?p�׿�R�I�C z�                                    Bxh�0�  T          @<���,��?�녽�Q��
=C��,��?��\?�A"�RC@                                     Bxh�?(  T          @8���*�H?���#�
�I��C���*�H?xQ�>���AG�C                                       Bxh�M�  "          @q��_\)?�  �����C J=�_\)?��>�  @x��Cu�                                    Bxh�\t  "          @N{�9��?�(��8Q��P��C33�9��?���>�A\)C�)                                    Bxh�k  �          @-p����?��>aG�@�z�Cff���?G�?:�HA�C!��                                    Bxh�y�  
�          @W��G
==��
?�(�A�33C2s3�G
=�\)?��A��
C>.                                    Bxh��f  
�          ?�녿�
=�L��?��B%{C6�\��
=�z�?\(�B�CN8R                                    Bxh��  �          @`  �.�R?���?W
=As33C��.�R?O\)?�(�A�ffC#�                                    Bxh���  �          @A��!�?���>�
=A Q�C���!�?�33?�Q�A�G�C�                                    Bxh��X  �          @w��P  @녽#�
�#�
C��P  ?�G�?�G�At��C��                                    Bxh���  
Z          @P  �1G�?У׽�Q�ǮCz��1G�?�Q�?E�A]p�Cz�                                    Bxh�Ѥ  "          @8Q��ff?����p���  C��ff?޸R�Ǯ��Cn                                    Bxh��J  �          @��R�Z=q@�>�G�@�ffC���Z=q?�{?У�A�(�Cu�                                    Bxh���  �          @}p��E@�?@  A0z�C
=�E?���?���A߮C��                                    Bxh���  �          @�z��K�@n{��ff���C���K�@��>\@�z�B�ff                                    Bxh�<  �          @�p��L��@%�>�(�@��C)�L��?��R?ٙ�A�{C!H                                    Bxh��  
�          @�p���\)@l(�?uA�RC}q��\)@*=q@.�RA�\)C@                                     Bxh�)�  �          @�ff��(�@�{?p��AG�C	����(�@Tz�@FffA�p�C�=                                    Bxh�8.  "          A�\��z�@���G��>�RC	33��z�@��H@#�
A��\CQ�                                    Bxh�F�  	�          A33���@�G����
�RC����@��H@
=qAS�Cn                                    Bxh�Uz  �          A�H��Q�@ȣ׿\�Q�C���Q�@��?�Q�AC�C��                                    Bxh�d   �          @�=q����@�������2{C������@�p�?�A-�C��                                    Bxh�r�  T          @�{��=q@��ÿG���\)C5���=q@��R?�
=AyG�C	{                                    Bxh��l  �          @�ff���\@����!G�����CL����\@��
@��A�(�C�                                     Bxh��  
(          A	���z�@�p��z��^=qC޸��z�@�z�?��A	p�C�{                                    Bxh���  "          AQ����@�ff�z��[33C�����@�p�?�ffAz�C}q                                    Bxh��^  �          Az����
@�\)��{��p�C:����
@�ff@��Ao\)C�H                                    Bxh��  "          @����(�@G��(���Q�C����(�@?\)?�=qA8��C��                                    Bxh�ʪ  �          @��H��=q?����{�:�\C ����=q@
=����=qCh�                                    Bxh��P  �          @�G���G�?}p����;\)C)�{��G�?������H���C%O\                                    Bxh���  T          @�����\@g��L�Ϳ���C(����\@L��?ٙ�A���Cu�                                    Bxh���  �          @�����=q@`��>B�\?��C���=q@:=q?�(�A��
C�\                                    Bxh�B  
�          @Ǯ����@QG�����(�C{����@7�?���Ar�RCff                                    Bxh��  
(          @��H���\@�p�����H��CJ=���\@��H?u@��CE                                    Bxh�"�  
�          @��H���R@�  ��  �G�C
�3���R@�
=?�\)A,Q�C#�                                    Bxh�14  �          @�z����@w������!�Cn���@^{?�p�Ak\)CE                                    Bxh�?�  "          @�{��(�@?\)��Q��P��C��(�@0  ?�p�A1p�C�3                                    