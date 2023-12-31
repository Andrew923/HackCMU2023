CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230211000000_e20230211235959_p20230212021622_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-12T02:16:22.856Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-11T00:00:00.000Z   time_coverage_end         2023-02-11T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxc�X@  "          @����  �:�H?5A (�C<Q���  �z�H>�\)@H��C?�                                    Bxc�f�  
�          @�G���\)��>�@�Q�C9}q��\)�&ff>L��@��C;p�                                    Bxc�u�  
�          @�  ��\)�u>L��@C4����\)���>��?޸RC5�R                                    Bxc܄2  �          @����{?(�����޸RC,����{?(�>#�
?��C,�R                                    Bxcܒ�  T          @��H����?8Q�>aG�@{C+޸����?�?�@���C.                                      Bxcܡ~  �          @�=q����?�\>\@��HC.8R����>�=q?z�@љ�C0��                                    Bxcܰ$  �          @�z����?:�H?   @���C+W
���>��?J=qA��C/&f                                    Bxcܾ�  �          @�����{?O\)?\)@�
=C*@ ��{>�ff?aG�A(Q�C.�{                                    Bxc��p  
�          @��
���R?�p�>W
=@ ��C%aH���R?z�H?E�AG�C(:�                                    Bxc��  �          @�{���H?z�H>��@7�C)\���H?=p�?0��@�C+�                                    Bxc��  T          @�����p�?��>#�
?ٙ�C(k���p�?\(�?#�
@�ffC*�\                                    Bxc��b  T          @�Q����H?��
��z��Mp�C%�����H?�G�>�{@p��C&�                                    Bxc�  T          @��R����?�������QG�C%�����?�=q>�p�@��C%5�                                    Bxc��  T          @������?������
�fffC#�����?���?(�@�
=C%T{                                    Bxc�%T  �          @����ff?���=�Q�?�  C#����ff?��R?B�\A(�C%��                                    Bxc�3�  "          @������?�(�?
=q@���C%G�����?Q�?�ffAH��C)��                                    Bxc�B�  T          @�����?��>�=q@H��C$���?�ff?^�RA#�C'T{                                    Bxc�QF  "          @�{���\?p�׼#�
���C(c����\?W
=>�(�@��C)��                                    Bxc�_�  �          @�Q�����?n{>\@��\C(�3����?#�
?E�A��C,(�                                    Bxc�n�  
�          @�����p�?:�H?@  A
=C*�H��p�>�z�?�  AK�C05�                                    Bxc�}8  �          @�����z�?@  ?aG�A2�HC*=q��z�>�  ?���Ag
=C0�R                                    Bxc݋�  �          @�Q����?�33�Ǯ��ffC&�����?���>L��@�C&#�                                    Bxcݚ�  �          @��H���R?������{C'�����R?�z�=�\)?L��C&�{                                    Bxcݩ*  �          @��\���R?}p���(���=qC(�����R?�=q=�Q�?��
C'�q                                    Bxcݷ�  �          @�
=���?�  >�?\C(aH���?Q�?
=@���C*k�                                    Bxc��v  �          @�Q���G�?��?!G�@�{C$����G�?\(�?�AW�
C)��                                    Bxc��  T          @�
=��G�?�\)?.{@���C&�)��G�?+�?���AR{C,{                                    Bxc���  "          @��R���?n{<#�
=��
C)5����?Q�>�G�@��
C*z�                                    Bxc��h  �          @�p���G�?��>���@c33C'�=��G�?J=q?E�A\)C*��                                    Bxc�  �          @�ff���\?��
<��
>k�C'�����\?fff?   @��C)u�                                    Bxc��  
�          @�����?W
==��
?aG�C*L����?5>�@�=qC+�{                                    Bxc�Z  
(          @�  ��?B�\<��
>��C+J=��?(��>�p�@���C,h�                                    Bxc�-   
�          @�����
=?=p����\C+����
=?5>k�@%C+�
                                    Bxc�;�  T          @�Q����R?5<��
>uC+�H���R?(�>�33@�  C,�                                    Bxc�JL  
�          @�G���Q�>��H=#�
>�(�C.^���Q�>�
=>��@@��C/8R                                    Bxc�X�  "          @�G���Q�>�ff=�\)?Tz�C.���Q�>�Q�>�=q@G
=C/��                                    Bxc�g�  T          @�G�����>��>�\)@N{C1{����=�Q�>�p�@��RC2�                                    Bxc�v>  T          @�������>��>���@Z=qC/c�����>aG�>�ff@�p�C1�                                    Bxcބ�  T          @������?�>���@r�\C-�����>��R?
=q@�p�C0k�                                    Bxcޓ�  �          @�����
=>�{?+�@��RC0
=��
=�#�
?B�\A
�\C4!H                                    Bxcޢ0  T          @�\)����?\(�?z�HA5�C)�H����>���?��\Al(�C0n                                    Bxcް�  �          @�Q���33?#�
?�ffAAp�C,}q��33=��
?�(�Ab�HC3�                                    Bxc޿|  T          @�����H?\)?�ffABffC-Y����H<#�
?�Q�A]�C3�H                                    Bxc��"  �          @�Q���z�>��?}p�A6=qC/8R��z����?���AD��C5+�                                    Bxc���  
�          @������>k�?   @�ffC1c������?��@�Q�C4\)                                    Bxc��n  "          @�
=��
=<�=L��?!G�C3�3��
=    =u?333C4�                                    Bxc��  T          @����\)>B�\�8Q���C1޸��\)>�  ��\)�O\)C1+�                                    Bxc��  T          @����\)>L��=�?���C1����\)=�>L��@�
C2�{                                    Bxc�`  "          @�\)��
==�=�G�?���C2����
==u>#�
?�C3Y�                                    Bxc�&  T          @�ff��{�L��=L��?
=C4�
��{��\)<��
>�  C4��                                    Bxc�4�  T          @���p���\)>aG�@"�\C4�{��p��#�
>#�
?�{C5��                                    Bxc�CR  
�          @�{���L��>L��@C6G�����=q=�Q�?��\C7�                                    Bxc�Q�  
�          @�{����\)>k�@,(�C78R����33=��
?c�
C8�                                    Bxc�`�  T          @��R��{��  >���@��HC6�
��{����>u@0  C8�f                                    Bxc�oD  T          @�
=��{���>�(�@�
=C7���{��(�>��@@��C8�R                                    Bxc�}�  
�          @�Q�����.{>Ǯ@���C5�3�������>�=q@FffC7��                                    Bxcߌ�  T          @�\)��
=���R>\)?�\)C7����
=��{���
�k�C7�                                    Bxcߛ6  �          @�\)���R��{=�?�=qC7�����R��33�L�Ϳ�RC8\                                    Bxcߩ�  T          @�
=��ff�\=�Q�?�ffC8ff��ff�\���Ϳ���C8c�                                    Bxc߸�  T          @�����G���Q�        C8���G����
�.{��\)C7��                                    Bxc��(  "          @��\��녾�{�#�
��\)C7�
��녾����#�
��=qC7^�                                    Bxc���  �          @��H��=q���R���
�k�C7����=q��=q�#�
��C7
=                                    Bxc��t  
�          @��������(�������\C8Ǯ����B�\�\)�ȣ�C6!H                                    Bxc��  �          @�����(���
=��\)�Dz�C8����(��u��G����
C6��                                    Bxc��  T          @�p���z���;�33�z=qC8p���z�B�\�   ��Q�C6#�                                    Bxc�f  "          @�����
��������C9+����
�.{�.{���C5��                                    Bxc�  "          @�{��(����(���ffC9)��(����@  ���C5p�                                    Bxc�-�  �          @�ff���;�{�#�
���HC7�����ͼ#�
�:�H� ��C4)                                    Bxc�<X  
�          @����
���ÿ333���C7�����
<��
�E��	G�C3��                                    Bxc�J�  �          @�p����
�#�
�0����(�C5ٚ���
>.{�.{���
C2�                                    Bxc�Y�  �          @���������
�:�H�33C4G�����>��R�(������C0u�                                    Bxc�hJ  �          @��
���\���#�
��RC5z����\>8Q�!G���\C1�q                                    Bxc�v�  T          @����Q쾏\)�&ff��p�C7&f��Q�=L�Ϳ5� ��C3aH                                    Bxc���  "          @�z���=q�L�ͿG���C6����=q>8Q�J=q�z�C1�q                                    Bxc��<  T          @����p�����=p��(�C7O\��p�=�G��G��\)C2��                                    Bxc��  �          @��\���׾����&ff��C7ٚ����<��5��C3�                                    Bxcై  �          @�{����8Q�Tz���\C6+����>aG��Q���C1xR                                    Bxc��.  �          @�
=���;����B�\�Q�C7u�����=��
�O\)��C3
=                                    Bxc���  "          @��
���þ�z�c�
�((�C7u�����>���k��.�RC28R                                    Bxc��z  �          @����p��L�ͿL���$(�C6�=��p�>B�\�L���$z�C1��                                    Bxc��   �          @�\)�������0����HC6  ���>8Q�0�����C1�                                    Bxc���  �          @��\��\)��녿(����C9E��\)��\)�G��p�C4ٚ                                    Bxc�	l  �          @��������R�8Q��p�C8  ���=L�ͿG���\C3c�                                    Bxc�  �          @�����;�ff�(���(�C9�����ͽ��@  �ffC5�\                                    Bxc�&�  �          @�G����R��33�B�\�ffC8xR���R<��W
=�)�C3��                                    Bxc�5^  T          @�����  �����\�W\)C5���  >�33�u�K�C/G�                                    Bxc�D  T          @����ff���L���,(�C:J=��ff�L�Ϳn{�F�HC4��                                    Bxc�R�  �          @�����p��.{�
=��z�C=G���p���{�Tz��3�C8��                                    Bxc�aP  �          @�������Ǯ����G�C9.�����������ffC5�                                     Bxc�o�  �          @�����  =�\)�&ff�33C3���  >�Q�
=q��(�C/n                                    Bxc�~�  T          @����G�?��Q��6�RC,����G�?W
=�   ���
C(B�                                    Bxc�B  T          @������?&ff�L���2=qC*�)����?p�׾�
=���C&ٚ                                    Bxc��  �          @��H��  ?z�   ��C+�=��  ?=p��B�\�*�HC)z�                                    Bxc᪎  �          @�����33?�;W
=�<��C,L���33?
==u?Tz�C+��                                    Bxc�4  �          @��H����?��>W
=@<��C,L�����>Ǯ>�(�@�33C.s3                                    Bxc���  �          @���~�R?��?Tz�A:{C,+��~�R=�?}p�A^=qC2Q�                                    Bxc�ր  �          @�(���Q�>��
?c�
AH  C/c���Q��G�?p��AS�C5�)                                    Bxc��&  �          @�  ��=q>aG�?�{As�
C0�)��=q����?��Ao�C85�                                    Bxc���  �          @�=q���>B�\?E�A#�C1aH����.{?E�A$��C6L�                                    Bxc�r  �          @l���hQ�\>Ǯ@�(�C9�3�hQ��\>8Q�@4z�C<\                                    Bxc�  �          @h����Ϳ���������CNG���Ϳ   ��
=��
C@�3                                    Bxc��  �          @z�H�5�8Q��!G��!z�C�
=�5��Q��b�\�Cs�R                                    Bxc�.d  T          @y���L���B�\����C33�L�Ϳ�\�P���l��Cu��                                    Bxc�=
  �          @Q�>u�$z��ff��\C���>u��  �0���u{C���                                    Bxc�K�  �          @y���@  �*�H�'
=�,C~O\�@  ��p��a���Cn�=                                    Bxc�ZV  �          @z�H�z���H�,���-�
C_p��z����R�\�c�
CB�
                                    Bxc�h�  T          @r�\�B�\��\)�W�=qCl��B�\?   �aG��C�H                                    Bxc�w�  �          @a녿��   �QG��C_�H��?�G��I��B�BꞸ                                    Bxc�H  �          @����vff��녿�{��  CK
�vff�@  �
�H��C?�                                    Bxc��  T          @�p���(�����������CK�3��(��h���#33�홚C?��                                    Bxc⣔  �          @�
=��G��33������CP0���G���(��-p����\CC�H                                    Bxc�:  �          @����33�!G��������CR{��33��Q��1G���G�CFO\                                    Bxc���  �          @�z�����{��(���p�CQ���������7
=� G�CE�                                    Bxc�φ  �          @�Q�����������CQ)��������#33��{CFaH                                    Bxc��,  T          @�\)��p��#�
��=q��(�CS����p����R�1�� \)CG�)                                    Bxc���  �          @����ff� �׿����(�CR����ff�\�%����
CG��                                    Bxc��x  �          @������� �׿������CS�)���ÿ����\)���CIB�                                    Bxc�
  �          @�  �\)�(Q쿺�H��  CUaH�\)��(��{���CK@                                     Bxc��  �          @�Q��w��,(����H����CV�\�w���z��.{�z�CK:�                                    Bxc�'j  T          @����|(��!녿�\)��Q�CT���|(���Q��2�\���CH&f                                    Bxc�6  �          @�  �u� �׿���\)CU!H�u��z��5��
z�CH!H                                    Bxc�D�  �          @����Q���� ����(�CQ�3��Q쿕�3�
��RCDG�                                    Bxc�S\  �          @����s33�<�Ϳ������
CYǮ�s33�����;��
�CM�                                    Bxc�b  �          @�=q�q��:�H��
=���\CY���q녿���2�\��\CN�=                                    Bxc�p�  �          @���a��N{��z����C^c��a��
�H�9���{CS��                                    Bxc�N  �          @�(��x���@�׿�p���(�CY�x����
�)�����\CO�f                                    Bxc��  �          @�Q��g
=�Fff��������C\���g
=�
=�0�����CRB�                                    Bxc㜚  �          @����b�\�Z=q��  ���C_��b�\�=q�5�
=CVG�                                    Bxc�@  T          @��R�fff�\�Ϳ��R����C_�fff�p��5�ffCVG�                                    Bxc��  T          @����Z�H�c33��p����Cb!H�Z�H�#33�8Q��G�CX�                                     Bxc�Ȍ  �          @�33�S33�fff������=qCcz��S33�'
=�7��	G�CZQ�                                    Bxc��2  �          @���B�\�l�Ϳ�{�|��Cf���B�\�/\)�5��
�C^�                                    Bxc���  �          @��R�A��l(���=q�xz�Cf�
�A��/\)�333�	�C^+�                                    Bxc��~  �          @�z��1G��q녿�{��G�Ci�R�1G��3�
�7
=�Cak�                                    Bxc�$  �          @���333�xQ쿸Q����Cj0��333�7��>{�Q�Ca�q                                    Bxc��  
w          @�
=�8���vff�����`z�Ci&f�8���<���0  ���Ca��                                    Bxc� p  �          @����9���qG����]�Chk��9���8���,(��33C`��                                    Bxc�/  T          @�Q��8Q��|�Ϳ�G��:�RCi�3�8Q��HQ��'���33Ccc�                                    Bxc�=�  T          @����HQ��xQ�\)��(�Cg\�HQ��P������Cb�                                    Bxc�Lb  �          @����N{�[��Q���Cb�=�N{�/\)�{��\)C\c�                                    Bxc�[  �          @���aG��G��c�
�+�C]�\�aG����	����G�CV�f                                    Bxc�i�  �          @�G��n{�5���  �B{CYE�n{������
=CQ��                                    Bxc�xT  �          @��
�W
=�$z῎{�k33CYh��W
=���
=��ffCP�                                     Bxc��  �          @�\)�j=q�(���\)��  CR��j=q�������Q�CH�)                                    Bxc䕠  T          @����S33�#33���
��ffCY�3�S33��G��  ��=qCP                                    Bxc�F  �          @�Q쿓33���R�}p��L  C~����33�XQ��,(��(�C{.                                    Bxc��  �          @�\)��R��=q�z�H�I�C��׿�R�^�R�.�R��\C��                                    Bxc���  T          @���������h���8z�C������fff�,���z�C��R                                    Bxc��8  �          @��<#�
���׿���C��<#�
�g���G�C�\                                    Bxc���  "          @�=q?5��{��R�G�C��\?5�a��ff���C��3                                    Bxc��  �          @��
?�Q���Q켣�
����C�q�?�Q��g���p����C���                                    Bxc��*  T          @���@C�
�>�R?Tz�A/�
C��
@C�
�C33�����HC���                                    Bxc�
�  
�          @�p�@N�R�G�?��
A~�\C���@N�R�W��.{��\C��f                                    Bxc�v  �          @���@U��Vff?5A  C�g�@U��U��G���HC�z�                                    Bxc�(  �          @���@^�R�O\)?!G�@�  C��@^�R�L�ͿO\)��HC���                                    Bxc�6�  T          @�p�@N�R�S�
>�Q�@�ffC�*=@N�R�I�������V=qC�޸                                    Bxc�Eh  �          @���@[��N{?O\)A��C�j=@[��P�׿�R���C�:�                                    Bxc�T  �          @��@qG��2�\?Tz�A   C��q@qG��8Q��(����C�O\                                    Bxc�b�  �          @��@������?Tz�A!p�C��f@����ff=�\)?aG�C���                                    Bxc�qZ  T          @��@s�
�#�
?+�A(�C��@s�
�&ff�����(�C��\                                    Bxc�   
�          @y��?�p��U�>�\)@��C�e?�p��H�ÿ�33��p�C�&f                                    Bxc厦  �          @r�\?333�i������\)C�p�?333�L�Ϳ�\��RC�,�                                    Bxc�L  "          @j�H���R�Q녿����{C�Ff���R�"�\�z��(�C�}q                                    Bxc��  �          @�(�@/\)�,(�?�ffA�\)C���@/\)�?\)=L��?5C�AH                                    Bxc庘  T          @����u�Tz�?���A���C��3�u�l(�=u?p��C�'�                                    Bxc��>  
�          @�=q�.{�Q�?�Q�A��
CfE�.{�e��u�:�HCh�                                     Bxc���  T          @���P���\��?.{Ap�Cb�)�P���Z�H�Tz���HCbT{                                    Bxc��  �          @���������ÿ�����p�Cz=q����U�>�R�33CuJ=                                    Bxc��0  "          @�(���p��XQ��p���G�CvJ=��p��	���Z=q�R=qCkh�                                    Bxc��  �          @�G��3�
�`  ���
�~�\Cg@ �3�
�)���'
=��C_Y�                                    Bxc�|  �          @����U���G������\)Cf���U��\������\)Cb                                      Bxc�!"  �          @���Z=q���\�z���ffCf
�Z=q�^{�{���
Ca��                                    Bxc�/�  �          @��
�fff�P��<#�
>��C^(��fff�>�R�����
C[�=                                    Bxc�>n  T          @����j=q�r�\�����ffCb  �j=q�N{�z���C]aH                                    Bxc�M  "          @���S33�qG��=p���\Cd�\�S33�G���R���C_xR                                    Bxc�[�  �          @����fff�e��\��\)C`ٚ�fff�(Q��7
=��CX33                                    Bxc�j`  �          @����tz��P�׿�������C\u��tz��z��1G���Q�CSO\                                    Bxc�y  �          @����{��I����ff����CZ� �{��\)�-p�����CQ��                                    Bxc懬  T          @�\)�n{�x�ÿ��
�-p�CbG��n{�HQ��"�\�ܣ�C\                                    Bxc�R  �          @��
�{��w
=�����/�C`n�{��E��#33�ظRCZ�                                    Bxc��  �          @�33�qG��|�Ϳ�\)�9p�CbO\�qG��H���(����  C[�\                                    Bxc泞  �          @��R�e�z�H��  �S�Cc�\�e�Dz��/\)��\)C\��                                    Bxc��D  T          @�  �b�\�\)�����^=qCdff�b�\�Fff�5���Q�C]8R                                    Bxc���  �          @�z��S33�l(������yp�Cd5��S33�3�
�1G��G�C\c�                                    Bxc�ߐ  
�          @�\)�fff�fff��z��N=qCa��fff�3�
�!����
CZ                                    Bxc��6  �          @�
=�}p��c�
���
�/�
C]���}p��5������33CW�                                     Bxc���            @�(��tz��`�׿�z��r�\C^���tz��(���-p���=qCV��                                    Bxc��  
�          @�
=�a��g
=��p��[
=Ca���a��333�%����CZh�                                    Bxc�(  �          @�(��C�
�\�Ϳ������HCdn�C�
� ���7
=�\)C[W
                                    Bxc�(�  �          @���7
=�S33��=q���Ce��7
=�Q��1��(�C[�                                    Bxc�7t  �          @��\�7
=�Fff��\��CcT{�7
=���7��\)CX�)                                    Bxc�F  T          @����A��QG����R���\Cc(��A��(��H��� {CW�{                                    Bxc�T�  
�          @�  �Dz��]p�������Q�Cd^��Dz���H�E��  CZ:�                                    Bxc�cf  �          @����9���c�
��
��G�Cf�)�9�����S�
�$��C\�                                    Bxc�r  �          @�p��9���h���\)���
Cgz��9���(��aG��+�C\�                                    Bxc瀲  T          @��\�1��e��33��G�Ch.�1��
=�c33�0�RC\O\                                    Bxc�X  T          @��R�#33�hQ��p����
Cj���#33�(��^{�233C_�q                                    Bxc��  "          @�=q�1��l������p�Ci)�1��#�
�XQ��'Q�C^�3                                    Bxc笤  T          @�Q��1G��e�Q���G�ChO\�1G��(��XQ��)��C]\)                                    Bxc�J  T          @��
�333�e��ff��z�Cg��333�ff�e��1�C[��                                    Bxc���  �          @�(��1G��b�\������Cg���1G��G��i���6G�C[k�                                    Bxc�ؖ  
�          @����G��K��E��Cm�\�G���Q���33�e�C[�f                                    Bxc��<  "          @�Q��G��HQ��P���,z�CtE��G��˅��\)�z��Cbc�                                    Bxc���  �          @�녿�(��S�
�4z��
=CoB���(����y���[Q�C`L�                                    Bxc��  �          @����J=q�U��  �s�Cb�{�J=q�$z��{���C[&f                                    Bxc�.  �          @�Q��E�[���33�`��Cc���E�,(��=q��  C]
                                    Bxc�!�  �          @��H�G
=�XQ쿵���Cc^��G
=�"�\�(���G�C[J=                                    Bxc�0z  �          @���G
=�L(���p��w�Ca���G
=���������HCZ5�                                    Bxc�?   �          @��R�K��A녿޸R��z�C_���K��
=�2�\��CU��                                    Bxc�M�  �          @��\�5�;���
=����Ca�H�5�Q��p��
=CX�f                                    Bxc�\l  "          @�p��L���333���up�C].�L�����
�H��CU�                                    Bxc�k  "          @���hQ��.{�=p����CX���hQ��{��  ���HCSk�                                    Bxc�y�  T          @�Q���z��+��@  �
=qCT޸��z��
�H��p����RCO��                                    Bxc�^  "          @�p��G
=�H�ÿ��R����CaY��G
=��
�&ff�G�CX�3                                    Bxc�  T          @�=q��
�c�
�{���Co����
�z��j=q�F�RCdk�                                    Bxc襪  "          @�ff�b�\�U�����Q�C_J=�b�\�ff�?\)��CU�f                                    Bxc�P  "          @��
�U�<�Ϳ�Q���ffC]��U��(��<(����CR��                                    Bxc���  �          @��S�
�>�R�ff��C]���S�
��
=�E�Q�CRG�                                    Bxc�ќ  �          @��R�dz��6ff��
=��
=CZ�{�dz����8Q����CO��                                    Bxc��B  �          @�G��e��Dz���
���C\���e��	���5��z�CR�                                    Bxc���  
�          @����5�>{�Q���=qCbE�5��=q�U�2��CT�R                                    Bxc���  �          @�Q���=p��6ff�z�Cg�f��У��qG��S�\CV��                                    Bxc�4  "          @��\�vff�+���
=��33CV�{�vff��p��3�
��CL5�                                    Bxc��  �          @����}p��!G���=q���RCTz��}p��У��*=q����CJQ�                                    Bxc�)�  "          @�=q����Q��33��(�CRp���녿�(��*�H���CG��                                    Bxc�8&  "          @�33���  ��{��G�CPE����\)�%���
=CF)                                    Bxc�F�  T          @����  �33�������\CM����  �����p���
=CC�                                    Bxc�Ur  �          @�G���ff�녿�33��
=CM�{��ff��z��!���
=CCh�                                    Bxc�d  �          @����|(���
��ģ�CR^��|(�����3�
�
=CF�\                                    Bxc�r�  "          @�=q�R�\�;��  ��z�C]�q�R�\��{�L(�� �HCQ��                                    Bxc�d  
�          @���Z�H�,(��{�׮CZ33�Z�H��33�Dz���CM�=                                    Bxc�
  �          @���Y���/\)����G�CZ�)�Y����\�:=q��CO�                                     Bxc鞰  T          @�  �S�
�0�׿�p���p�C[���S�
���8Q��=qCP��                                    Bxc�V  T          @��
�_\)�%�p���\)CX���_\)�����AG��33CL:�                                    Bxc��  �          @�33�\(��/\)��\��p�CZ�{�\(���\�;��ffCOE                                    Bxc�ʢ  �          @���R�\�?\)�����C^=q�R�\���5��  CTG�                                    Bxc��H  �          @�G��AG��I�����H���
Cb:��AG��(��@  �ffCW��                                    Bxc���  
�          @���333�K��ff��Cd���333�z��W
=�/��CXk�                                    Bxc���  
�          @�ff�-p��]p������=qCg���-p��
=�X���-ffC]!H                                    Bxc�:  
�          @�ff�1��`����\�îCg���1��   �L���"33C]�                                    Bxc��  �          @�  �P  �B�\�  ��(�C_��P  ���R�N{� �CSxR                                    Bxc�"�  T          @�z��l���?\)��Q����CZ��l���33�:=q�
(�CQ
=                                    Bxc�1,  T          @��H�q��5��33���HCX���q녿�
=�3�
�(�CO�                                    Bxc�?�  "          @�Q��q��$z�����{CV8R�q녿У��5�
z�CKff                                    Bxc�Nx  �          @��s�
�/\)�(���{CW���s�
�޸R�B�\�(�CL��                                    Bxc�]  �          @�{�e��3�
������CZ{�e���33�4z��
=CO�q                                    Bxc�k�  
�          @���a��'
=��(���G�CXk��a녿ٙ��2�\�CM��                                    Bxc�zj  �          @��
�U�6ff���(�C\xR�U����?\)�
=CQc�                                    Bxc�  �          @�z��X���)����\��z�CY�q�X�ÿ�\)�G
=�33CM�{                                    Bxcꗶ  "          @�  �_\)�)���=q��\CY#��_\)��=q�Mp����CLW
                                    Bxc�\  �          @�p��P  ��H��
��CX�
�P  ��33�B�\�"Q�CKY�                                    Bxc�  "          @���W���R�{���CUp��W����G
=�$Q�CG0�                                    Bxc�è  
�          @��\�`  ��\����
=CN���`  �G��6ff��
C@�f                                    Bxc��N  "          @�G��fff��G��#33�  CN�fff�5�@����C?+�                                    Bxc���  �          @���y����\)�'����CM���y���J=q�G��ffC?�                                     Bxc��  �          @�{�s�
��\�%��33CL���s�
�5�C33���C>��                                    Bxc��@  �          @��R�i���˅�   � 33CK���i���z��9���p�C={                                    Bxc��  �          @�=q�`  ����%�

=CI���`  ��p��:�H���C9��                                    Bxc��  "          @��R�H������   � =qCZ�H�ÿ���Mp��,Q�CK�                                    Bxc�*2  �          @���Dz��(�ÿ�Q����C\���Dz��Q���\��CT\)                                    Bxc�8�  �          @����=p��*�H��33���RC]�R�=p����R����p�CU�
                                    Bxc�G~  
�          @����C33�,�Ϳ��R��  C]���C33���4z��
=CR                                    Bxc�V$  �          @�ff�!��S33��p���ffCh���!��#33�#�
���CaL�                                    Bxc�d�  !          @���(���@  ��=q���RCd���(���\)�#33�\)C\\)                                    Bxc�sp  �          @�ff����@  ��Q����Cg\)����p��)���33C^�
                                    Bxc�  
(          @�(���\�:=q��=q��z�Cg�R��\�z��/\)�&ffC^
=                                    Bxc됼  �          @�\)����.{�
=q����Cd�
��ÿ�\�?\)�4=qCX�                                    Bxc�b  �          @���,���#�
�p���p�C_p��,�Ϳ�{�>{�-\)CR�
                                    Bxc�  "          @���\)�:�H����(�Ch�=�\)����QG��@�\C\.                                    Bxc뼮  "          @�
=��R�/\)�-p���HCf޸��R��{�`  �OQ�CW�)                                    Bxc��T  "          @�����R��R�5��!=qCd
=��R��=q�aG��VffCRǮ                                    Bxc���  �          @�(���Q��33�R�\�EQ�Cbuÿ�Q�B�\�r�\�t��CIu�                                    Bxc��  T          @��ÿ�ff����@  �D(�Cp)��ff����fff  C[c�                                    Bxc��F  �          @��ÿ
=���mp�\Cwn�
=���~{£�C7#�                                    Bxc��  �          @8Q��{��  ���H�ʏ\C_@ ��{��(���  �Q�CU=q                                    Bxc��  T          @\���=q�G���z���G�C\��=q��z����
=CRQ�                                    Bxc�#8  �          @����5�(Q���
���
C^��5�����%��G�CU                                    Bxc�1�  T          @���!G��*=q�ٙ���ffCb���!G���z�� ����RCY!H                                    Bxc�@�  T          @�G��(Q��9���\��(�Cc���(Q���������C[�                                    Bxc�O*  �          @���ff�q녿����ffCw�q��ff�Dz��'
=���Cs@                                     Bxc�]�  T          @�=q�O\)��ff�   ����C���O\)��  �   ��ffC�>�                                    Bxc�lv  
�          @�=q��G��~�R���H��Cv5ÿ�G��dz��=q��\)Cs�{                                    Bxc�{  �          @�
=�7
=�e��\)�Tz�Cgp��7
=�Vff������HCe}q                                    Bxc��  T          @�\)�7��`  ��(��p��Cf���7��8Q����Ca{                                    Bxc�h  �          @����<���`  �B�\�{Ce޸�<���A녿�33��  CaǮ                                    Bxc�  T          @�ff�@  �e>aG�@+�Cf)�@  �\�Ϳ��\�H��Cd�q                                    Bxc쵴  T          @����L���b�\��G����Cc޸�L���R�\��ff�}Ca��                                    Bxc��Z  T          @���.�R�k���z��g
=Cin�.�R�W��\���RCf�                                    Bxc��   T          @�=q�9���s�
>�
=@�\)Ch�3�9���n{�fff�,  Ch{                                    Bxc��  �          @���L���hQ�>��@A�Cd�)�L���`  �z�H�;�Cc��                                    Bxc��L  �          @�{�p��]p���\)���Cj�)�p��(Q��:�H�=qCb�                                    Bxc���  �          @����p��j=q�z��θRCn���p��0  �K��((�CgB�                                    Bxc��  T          @����h�ÿ����=qCmaH���5��:=q��
Cf�                                    Bxc�>  �          @�����Mp��{��p�Ch�\�����J�H�.G�C_
                                    Bxc�*�  �          @���'��*�H�(Q��ffCa���'�����XQ��=�CT                                      Bxc�9�  "          @�33��\�E���p����
Cic���\�  �8���({C`u�                                    Bxc�H0  �          @��׿�
=�XQ��(���z�Cs����
=�!��>�R�1�HCl�                                     Bxc�V�  �          @�{��z��K�����͙�Co\��z��=q�/\)�&��Cg�H                                    Bxc�e|  �          @����$z���H�
=���C_=q�$z��  �A��5�RCRJ=                                    Bxc�t"  �          @��\��H�(Q�������Cck���H��Q��H���;33CV�                                    Bxc��  �          @�����*=q��{��=qCc������33�(Q��!�CZ�                                    Bxc�n  �          @��R����9���0  ���Cm{��׿��c33�U�HC`\)                                    Bxc��  T          @�\)�
�H�7��>{�{Ch�f�
�H�޸R�p  �U�CZ�f                                    Bxc���  �          @�(��=q�C�
�<(��(�Cg���=q��
=�q��K
=CZ��                                    Bxc��`  �          @�����\�8Q��(���p�Cg�=��\��{�\(��EG�C[!H                                    Bxc��  �          @�  �   �'��{��(�CbJ=�   ��  �<���033CV�R                                    Bxc�ڬ  �          @�Q��+��	��� ���\)CZ��+���(��Dz��8CLxR                                    Bxc��R  �          @{��:=q�G���{��{CV�=�:=q������z�CM}q                                    Bxc���  �          @y���녿�33��(���C[�{�녿�z���R�0G�CN�q                                    Bxc��  �          @5��zῠ  ��  � �RCYͿ�z��R��
�E��CHk�                                    Bxc�D  �          @G��   ��
=�h����=qC@  �   ��G���  ��=qC7
                                    Bxc�#�  T          @*�H�ff�B�\��  ��G�C8��ff>aG���  ��ffC.�f                                    Bxc�2�  
�          @3�
�"�\=���33��\)C1@ �"�\>��H��ff��p�C)(�                                    Bxc�A6  �          @&ff��
�L�Ϳ����p�C8޸��
>8Q쿑���  C/��                                    Bxc�O�  �          @n�R�Y��?333������(�C(c��Y��?���h���e��C"�                                    Bxc�^�  �          @g���{���
���E��C4� ��{?.{�z��9  C�                                    Bxc�m(  T          @aG��8Q��z��=p��j�Cv�Ϳ8Q�!G��U�HC]L�                                    Bxc�{�  �          @?\)��p���{����K33CY  ��p������   �kQ�C@�=                                    Bxc�t  T          @a��6ff>k���\)��=qC/Y��6ff?:�H���H��p�C%�{                                    Bxc�  T          @mp��QG�>��Ϳ�Q���p�C-��QG�?h�ÿ�p���Q�C$k�                                    Bxc��  �          @p  �G
=?}p��ٙ���{C"T{�G
=?�p���ff����C�)                                    Bxc�f  T          @l���X��?\(���z���z�C%�q�X��?�
=�Tz��O�C �                                    Bxc��  �          @o\)�N{>.{����  C0��N{?:�H��z��ՙ�C'8R                                    Bxc�Ӳ  T          @�  �[�>���G���C/�
�[�?aG�����G�C%�=                                    Bxc��X  �          @8���z�?����33��
CE�z�?�ff��p���33C0�                                    Bxc���  
�          @b�\�!G�?5�
=�({C$.�!G�?�z��G����C�3                                    Bxc���  �          @\(��%�>�\)�z��\)C-���%�?h�ÿ���
(�C ��                                    Bxc�J  T          @j�H�����&ff�z����Cn�Ϳ��ÿ�=q�1G��C�HCeJ=                                    Bxc��  �          @���33�<(�����p�Ck��33����/\)�)(�Cc                                    Bxc�+�  �          @u��p�׽�G��'�{C:���p��?333�!��Cp�                                    Bxc�:<  �          @p�׿@  ?���S33{B���@  @�\�.{�B
=B�\)                                    Bxc�H�  �          @q녿�  ?5�^�R\Czῠ  ?�  �Fff�\�HB�W
                                    Bxc�W�  T          @g
=��Q�=�\)�Q��\C1���Q�?����Fff�o��C=q                                    Bxc�f.  �          @e��p���Q��(��,�\C\h���p��\(��5�R�HCKu�                                    Bxc�t�  
�          @g
=�z�8Q��0  �M�RCG)�z�>#�
�5�W(�C/�3                                    Bxc�z  �          @x��=L���\)�:=q�E��C��=L�Ϳ��R�a�=qC���                                    Bxc�   �          @g
=?�G����H�(Q��Az�C�b�?�G������Fff�wp�C��R                                    Bxc��  "          @|(���R��p��HQ��a��C|�{��R�p���e��Cl}q                                    Bxc�l  �          @����#�
����N�R�L  C@+��#�
>�ff�P  �M\)C*                                      Bxc�  
�          @����Q��
=�^�R�]
=C>#��Q�?&ff�\���Y�
C$�\                                    Bxc�̸  
�          @c�
�33���
�%��@33C;�)�33>���#�
�=��C(W
                                    Bxc��^  �          @vff�G��#�
�>{�RQ�C4���G�?\(��6ff�F  CO\                                    Bxc��  T          @b�\����?�(���z���=qC�����@��}p����B��H                                    Bxc���  "          @�z��A녾L���Z�H�AffC7��A�?Tz��Tz��:(�C$�)                                    Bxc�P  �          @���R����z=q�d�
C=aH��R?E��vff�_�HC"��                                    Bxc��  �          @���5����n{�Q�\C=#��5�?+��l(��N��C&��                                    Bxc�$�  �          @�ff�)����Q��`  �S\)C5޸�)��?u�W��HQ�C .                                    Bxc�3B  �          @��H�A녾��7��,G�C=
�A�>Ǯ�8Q��-ffC,�q                                    Bxc�A�  "          @�G��q녿xQ��G���CBc��q녾���p���C7�)                                    Bxc�P�  "          @���|(��O\)�,(��\)C?�
�|(�<��
�3�
�33C3�                                    Bxc�_4  T          @����녽�G��4z���C5�
���?8Q��.�R�  C)�                                    Bxc�m�  �          @��R�l(������O\)�$\)C:=q�l(�?\)�Mp��"�HC+Q�                                    Bxc�|�  �          @�
=�aG��!G��Z=q�.�C>33�aG�>Ǯ�\���1  C-��                                    Bxc��&  �          @�p��s�
�Y��?E�A	C]�=�s�
�^�R��  �3�
C^n                                    Bxc��  �          @��qG��\��?Q�AffC^z��qG��b�\�aG����C_8R                                    Bxc�r  
Z          @�  �u��`  ?G�A��C^aH�u��dzᾊ=q�<(�C_                                      Bxc�  
�          @�  �vff�`��?
=@�\)C^\)�vff�a녾�ff��p�C^��                                    Bxc�ž  T          @����z=q�`��?��@��C]�H�z=q�aG����H����C]��                                    Bxc��d  "          @����~{�\(�>�
=@���C\��~{�Z�H�
=��
=C\�3                                    Bxc��
  "          @�G�����W����Ϳ�ffC[�q����L�Ϳ���9CZL�                                    Bxc��  T          @�\)�|(��Z=q�����k�C\�\�|(��J�H��ff�f{CZ                                    Bxc� V  �          @�p������L�Ϳ
=q��G�CZ�������:=q��Q����CW�                                    Bxc��  �          @�z��y���QG��5��p�C\�y���<(���{��=qCX��                                    Bxc��  �          @�������XQ�333���HC[������B�\�У���\)CX�q                                    Bxc�,H  T          @��
�����`�׾��H��{C]\�����N�R���H�|��CZ�3                                    Bxc�:�  �          @�\)���\�HQ�J=q��
CYz����\�1G������\)CV=q                                    Bxc�I�  T          @��~{�O\)�Y����RC[G��~{�7���p���CW��                                    Bxc�X:  �          @��
�O\)�p  �E��z�Ce#��O\)�XQ�����\)Cb33                                    Bxc�f�  �          @����8����33��=q�B{Cj��8���hQ��(���Cgn                                    Bxc�u�  T          @�=q�U��c33��=q�tQ�Cb޸�U��B�\�G���  C^c�                                    Bxc�,  �          @�=q�N�R�n�R�����Ip�Ce��N�R�QG��ff��Q�CaW
                                    Bxc��  �          @���j�H�@  ��  �?�C[B��j�H�&ff��ff���\CWQ�                                    Bxc�x  "          @���i���?\)����Y��C[Q��i���#33��
=��33CV��                                    Bxc�  T          @�z��a��G
=���\�p��C]h��a��(������{CX�=                                    Bxc��  �          @�{�e��E���\�p��C\�=�e��'�������CX(�                                    Bxc��j  �          @�����  ��33���H���CEJ=��  �^�R����G�C>��                                    Bxc��  �          @��\��z῰���ff���
CE� ��z�@  �����=qC=�q                                    Bxc��  �          @�z�������Q������\CB�f�������z����HC;.                                    Bxc��\  T          @�����Q쿫��z����CD�\��Q�8Q��
=��(�C=)                                    Bxc�  T          @�33��녿����z��أ�CF����녿B�\�(Q���ffC>\                                    Bxc��  "          @�����zΉ��p���z�CA�R��z�\�*�H���C8�                                    Bxc�%N  �          @�ff��{�����{�㙚CKs3��{��=q�7��p�CBff                                    Bxc�3�            @����
=��ff�{��z�CH0���
=�Q��333�\)C?
=                                    Bxc�B�  >          @��H���ÿ�ff������CA
���þ�33�(����33C8s3                                    Bxc�Q@  
�          @�\)��>�Q��*=q��  C/�\��?����p�����C'(�                                    Bxc�_�  
�          @�ff���?
=�.�R��C,�
���?�ff�{�ظRC$k�                                    Bxc�n�  �          @�����
�k��<(���HC6�H���
?��9����33C-�                                    Bxc�}2  �          @������&ff�G���ffC;��������
�
=��
=C4@                                     Bxc��  �          @�z���  ��������(�CAff��  ���   ����C9�=                                    Bxc�~  �          @��
�����p��p����HCE�f����W
=�!����HC>5�                                    Bxc�$  �          @�ff��Q������
��
=CQk���Q��\�(Q���\)CJ��                                    Bxc��  �          @�G��{��]p���\���C]aH�{��7��(Q�����CX!H                                    Bxc��p  �          @��R�I����Q쿜(��P  Ci���I���q��33����Cf8R                                    Bxc��  �          @�����(����{��
=CRk���(���G��2�\��ffCK�                                    Bxc��  �          @�ff��z�Q��I���=qC>
=��z�=#�
�P���\)C3��                                    Bxc��b  �          @������-p��33���CU�����z��,(���ffCNu�                                    Bxc�  �          @�z��[���Q쿬���^=qCg!H�[��p����H��p�Cc�{                                    Bxc��  �          @�=q�g
=�z=q��=q����Cc:��g
=�W
=�#33�ڣ�C^�f                                    Bxc�T  �          @�33�j=q�z�H��������Cb��j=q�W
=�#�
���C^�{                                    Bxc�,�  
�          @�{�fff���׿�ff��Cd��fff�Z=q�1���ffC_s3                                    Bxc�;�  "          @��
����ff�L(��
z�CO  ��������g�� ��CD=q                                    Bxc�JF  �          @��\�}p��<���   ����CX�R�}p��p��K��=qCQ!H                                    Bxc�X�  T          @���xQ��n�R��{��33C_�f�xQ��L(��!G����HC[s3                                    Bxc�g�  �          @�=q�{��C33�(����HCY�{��z��H���
{CR��                                    Bxc�v8  
Z          @�  �^�R�`  �=q��  Ca&f�^�R�1G��N�R���CZxR                                    Bxc��  �          @�(��hQ��\(��p���33C_n�hQ��,���P  ��HCX�{                                    Bxc�  9          @�z��`  �N{�B�\�(�C^���`  ��p���&�HCU��                                    Bxc�*  =          @��
�j=q�[��"�\����C_&f�j=q�*�H�U��33CX#�                                    Bxc��  "          @�=q�tz��X���333��C]�)�tz��$z��dz����CU��                                    Bxc�v  
�          @�  �u��P���2�\��  C\k��u�����a��{CT��                                    Bxc��  �          @����{��Z�H�$z���=qC]�{��*=q�Vff���CV{                                    Bxc���  "          @���z�H�_\)������
C]�f�z�H�0���P  �Q�CW#�                                    Bxc��h  
�          @����vff�\���"�\�љ�C]���vff�,���U��33CW�                                    Bxc��  =          @�(��r�\�j=q�'
=����C`�r�\�8���\(��z�CYW
                                    Bxc��  �          @�33�i����G��z���(�Cc���i���Y���@������C^�                                    Bxc�Z  �          @���S33��p���ff��(�CiG��S33�u�5��z�CeO\                                    Bxc�&   �          @��U�~{��
��{Ce���U�QG��N{�
\)C`\)                                    Bxc�4�  �          @���S�
�����
�H���\Cf�q�S�
�XQ��G
=�33Ca��                                    Bxc�CL  T          @��U��������
CgaH�U�e��7�����Cc�                                    Bxc�Q�  T          @���^�R�|����\��  Cd�\�^�R�P���L(���C_�                                    Bxc�`�  
�          @���K����
�����ChO\�K��Z=q�U�ffCb�                                    Bxc�o>  �          @�Q��HQ���ff�����{CiG��HQ��_\)�Vff��Cd�                                    Bxc�}�  �          @��
�U��p��(�����CgT{�U�\(��X���
=Ca��                                    Bxc�  T          @���`  �����#�
���HCe��`  �Q��]p��G�C_.                                    Bxc��0  "          @�=q�c�
�l���.{����Cb��c�
�;��b�\��C[s3                                    Bxc���  �          @��\�h���`  �:=q��p�C_��h���,(��j�H�33CX��                                    Bxc��|  
�          @�=q�L���u��<(���{Cf��L���@  �q��"��C_(�                                    Bxc��"  �          @���Z�H�r�\�1G���Cc���Z�H�@���fff�Q�C]c�                                    Bxc���  "          @��H�Q��~�R�/\)��Cf���Q��L���g��{C`Q�                                    Bxc��n  �          @�(��_\)�w��,������Cd��_\)�G
=�c33�C]�                                     Bxc��  
�          @�33�X���l(��AG����\Ccp��X���7
=�s�
�"�
C\(�                                    Bxc��  
�          @�  �Q��z=q�#33�ԸRCe�3�Q��K��Z=q���C`)                                    Bxc�`  "          @�33�Q��g��N�R�  Cc޸�Q��0  ��  �,p�C[�R                                    Bxc�  �          @��H�L���q��E��G�Ce� �L���<(��y���'z�C^�\                                    Bxc�-�  �          @��
�6ff�QG��a���\Cd���6ff����R�C  C[W
                                    Bxc�<R  �          @�G�� ���\(��a���\Ciٚ� ��� ������HQ�C`�                                    Bxc�J�  �          @�Q��@���`���Z�H���Cep��@���'
=�����8�HC\�                                    Bxc�Y�  �          @�G��J=q�^�R�W
=�G�Cc�{�J=q�%���H�4  C[n                                    Bxc�hD  "          @��\�g��XQ��Dz���Q�C_\�g��$z��q��!��CWW
                                    Bxc�v�  "          @�{�qG��P���N�R���C\�=�qG��=q�y���$(�CT��                                    Bxc���  
�          @�{�_\)�Y���W��
�
C`:��_\)�!G���=q�-CW�=                                    Bxc��6  
�          @�G��Z=q�c33�`  ��\Cb#��Z=q�(����\)�1CY�R                                    Bxc���  T          @�=q�\(��[��hQ��  C`��\(��   ���\�6
=CW�R                                    Bxc���  �          @�=q�S�
�e�b�\��RCc^��S�
�+������4�\CZ��                                    Bxc��(  �          @��0�����H�E���\)Co��0���p������$G�Ci��                                    Bxc���  T          @ƸR�K�����]p��=qCg���K��J=q�����/=qC`�\                                    Bxc��t  �          @��
�P  �~�R�QG��CfǮ�P  �G����H�)  C_޸                                    Bxc��  �          @��H�<�����R�;����Cl���<���j�H�w���HCg0�                                    Bxc���  �          @��
�@����\)�7���=qCl
�@���l���tz��Cf��                                    Bxc�	f  �          @�p��E�{��h���(�CgǮ�E�?\)��{�7z�C`�                                    Bxc�  �          @����B�\�w��l(���\Cg�\�B�\�:�H��
=�:�RC_�H                                    Bxc�&�  �          @����J�H�h���h���{Cd��J�H�.{���
�933C\�
                                    Bxc�5X  "          @�G��B�\�i���k��
=Cf5��B�\�.{��p��<�C]�
                                    Bxc�C�  �          @�=q�(���W
=�x���)ffCg�=�(����������O��C^!H                                    Bxc�R�  �          @���	����33���
�h�CY���	���!G���(�W
CD33                                    Bxc�aJ  �          @����R�   �����\�\C]�q��R�}p���z��y=qCK�f                                    Bxc�o�  �          @����R��(���=qk�Cm����R�W
=��z�p�CU�                                    Bxc�~�  �          @�=q��ff���R��Q�=qCl𤿦ff�Q����\�)CT^�                                    Bxc��<  �          @�=q��\)�����R�j��Ck�)��\)��G����
Q�CY��                                    Bxc���  �          @�녿�=q�,(���=q�^Co�H��=q��{�����Ca��                                    Bxc���  "          @�33��  �ff�����g{CiG���  ��G����CW�3                                    Bxc��.  �          @��H��Q��P�����
�N\)Cy�ÿ�Q��p�����}33Cq��                                    Bxc���  �          @�=q����?\)����cp�C��=��Ϳ����ff�\C}�                                    Bxc��z  �          @��
�}p��AG���(��^
=C{�}p���
=����Cr�                                    Bxc��   �          @�{���H�@����{�]\)Cx!H���H��33��\)aHCm�H                                    Bxc���  �          @�z΅{��R����o(�Cq:΅{��{����
=C`�)                                    Bxc�l  �          @�녿�ff�L(������O��Ct{��ff������{{Ci�H                                    Bxc�  �          @��׿����=q�n�R�!�Cx����J=q�����O\)Cr�                                    Bxc��  �          @�녿��
�O\)��
=�I��CqB����
�(���=q�s�Cf��                                    Bxc�.^  �          @��׿����[���=q�F��Cr������
=���R�p�ChE                                    Bxc�=  �          @�z��ff�S33����Q=qCqaH��ff�
�H���R�z�
CfO\                                    Bxc�K�  �          @Ǯ�33�S33��(��M�
Cn33�33�
�H��\)�u��Cb��                                    Bxc�ZP  �          @�  �����p  ��33�=�Cr�������*�H�����h�Ci�H                                    Bxc�h�  "          @�\)����u�����:ffCs�{����1�����effCk��                                    Bxc�w�  T          @�Q�޸R�r�\����@�CuLͿ޸R�,������l33Cm.                                    Bxc��B  �          @�G��ٙ��tz���ff�A\)Cv\�ٙ��.�R�����m�Cn!H                                    Bxc���  "          @����  ���H�\)�!ffC��ÿ�  �hQ���(��Q=qC~�\                                    Bxc���  T          @ƸR��ff��Q�����#��C}�3��ff�b�\���R\)Cy�H                                    Bxc��4  T          @���z���ff�o\)�=qC}T{��z��s33�����CCy�                                    Bxc���  �          @ȣ׿������o\)�p�C�����}p����AQ�C{�                                    Bxc�π  �          @��H��=q�����n{��
Cy��=q�x����z��<�RCt                                    Bxc��&  �          @��H��33��\)�Q����HC}@ ��33�������%�Cz��                                    Bxc���  �          @��
�����(��Dz����HC}�3����������RC{J=                                    Bxc��r  �          @ҏ\�����G��+����\C
�����33�u���
C}.                                    Bxc�
  �          @ҏ\���H����S�
��C(����H����(��'=qC|�R                                    Bxc��  �          @�{���R��z��K����C.���R��33��G�� �
C|��                                    Bxc�'d  �          @ҏ\��{�����(����C�箿�{���
�Mp���Q�C�k�                                    Bxc�6
  �          @љ���z��ƸR�������C�����z���33������C�8R                                    Bxc�D�  "          @��Ϳ�  ��G���  �t��C�c׿�  ��G��AG����C��                                    Bxc�SV  T          @��Ϳ�\��z����aG�C}@ ��\��ff�.�R�ˮC{�R                                    Bxc�a�  �          @���������Ϳ��IG�Cw������\)�'
=��Q�Cvk�                                    Bxc�p�  
�          @�
=�33�����p��/\)C{)�33��������Cy�                                    Bxc�H  �          @�
=���(�������CxT{������z��r{Cw�{                                    Bxc���  T          @�(����z�?��A33Cy������������Cy��                                    Bxc���  T          @��׿�{���H?
=q@��C�lͿ�{��=q�(���ə�C�h�                                    Bxc��:  �          @�녿�Q���\)?z�HAz�C}����Q���녾W
=���RC}�q                                    Bxc���  �          @���� ����p�?���A4  Cz��� ����G�<��
>8Q�Cz�f                                    Bxc�Ȇ  �          @�녿�(���������{C{���(����\��{�G33C{Q�                                    Bxc��,  T          @�  ��  �������z�C}Ϳ�  ���\�Tz����Cz�                                    Bxc���  �          @��\���H����P  �z�Cw�H���H�Z�H�����9�HCsc�                                    Bxc��x  �          @�=q����)���}p��B��Ce����׿�ff����c�CZz�                                    Bxc�  T          @�\)�G����\��33�  CT&f�G���=q��Q���C;��                                    Bxc��  �          @�z���������=q�p��CU\)�������G��C@��                                    Bxc� j  �          @�G��(���{���
�yz�CL\)�(��u���  C5h�                                    Bxc�/  �          @�Q���
�����
=8RC?�\��
?���
=��C'#�                                    Bxc�=�  �          @��
� �׼���Q��}ffC4��� ��?u����u33C\                                    Bxc�L\  �          @�����>�Q�������C*���?�33���H�y
=Cff                                    Bxc�[  T          @�33��?����ǮC%:���?�{��
=�|�
C5�                                    Bxc�i�  �          @��H�ff?#�
��p���C"�q�ff?�����ff�r�RC��                                    Bxc�xN  "          @�녿�z�?�����33�C^���z�@!���{�c��B�W
                                    Bxc���  �          @�Q���R?�\)��\)�]=qC��R@-p������>�\Cn                                    Bxc���  �          @�G����@G������p��B�33���@J=q����IG�B���                                    Bxc��@  �          @��H�Ǯ@������u=qB�(��Ǯ@Fff��(��Np�B�u�                                    Bxc���  T          @��\�\@	����p��w(�B���\@Dz������PQ�B�R                                    Bxc���  �          @�\)��{@,(����H�hffB���{@e���
=�?33Bݮ                                    Bxc��2  "          @���(�@�����v(�BꞸ��(�@Q�����M
=B��f                                    Bxc���  T          @�
=����@ff����\)B��q����@C33��33�X  B�q                                    Bxc��~  �          @�=q��33@	�����H�s  B��H��33@C33���\�M\)B�Ǯ                                    Bxc��$  �          @���z�?������rC
���z�@.{���R�Q��B�z�                                    Bxc�
�  �          @�\)�
�H?�33��ff�o  C
�{�
�H@4z�����NffB�B�                                    Bxc�p  
�          @����
=@Q���
=�v(�B��Ϳ�
=@?\)��\)�Oz�B�                                      Bxc�(  �          @�����
@ �����
�z{B�\)���
@Y�������N{B���                                    Bxc�6�  T          @��H���H@�
�����3B�\���H@N�R��  �W�B�W
                                    Bxc�Eb  
�          @��R?��@J=q����W�RB�p�?��@|(��n�R�,=qB���                                    Bxc�T  �          @���=L��@N�R��
=�Xz�B�33=L��@����qG��,�B��=                                    Bxc�b�  T          @�����
@W����\�P33B��3���
@�(��fff�$Q�B��{                                    Bxc�qT  
�          @��
=���@p  ��p��@{B��=���@�
=�XQ��G�B��\                                    Bxc��  �          @��=�@U����Up�B�Ǯ=�@�(��q��)B��{                                    Bxc���  �          @�G���=q@HQ������]{B��;�=q@z�H�vff�1�\B���                                    Bxc��F  T          @��\�#�
@���s�
�,�B��#�
@�ff�>{�(�B��H                                    Bxc���  T          @�G�����@o\)���\�=�
B�녽���@�{�S33�p�B�u�                                    Bxc���  �          @�p���ff��=q��\)��C<���ff?�R��ffC �                                    Bxc��8  T          @����ff�p�����`��Cbu��ff�����z��|��CT�                                    Bxc���  �          @�\)�&ff��
��G��R  C]�)�&ff����z��k
=CP��                                    Bxc��  T          @��H�4z��H���5��ffCd\�4z��#�
�W��&  C^+�                                    Bxc��*  
�          @����5�U�-p���{Ce���5�1G��R�\��C`G�                                    Bxc��  
�          @����AG��;��K��33C`&f�AG���\�j�H�0G�CY(�                                    Bxc�v  T          @���G
=�>{�@�����C_��G
=�
=�`���'��CY33                                    Bxc�!  T          @���!G��\�Ϳ������Ci���!G��I����
=����Cg^�                                    Bxc�/�  �          @�\)���������(�Cm&f���׿�(�����?�
Cg.                                    Bxc�>h  �          @��R���?�����p�\)C@ ���@z����
33B��H                                    Bxc�M  
�          @�p��p��?Y����  ��C	��p��?�  ��Q���B�G�                                    Bxc�[�  T          @���ff?�
=���W
C� ��ff@�
����}G�B�q                                    Bxc�jZ  �          @�{���?fff�����C
�׿��?޸R�����B�p�                                    Bxc�y   
�          @��
���\��=q���\�{C@����\?������\CxR                                    Bxc���  
�          @�Q��������G�\)CST{����u����{C;��                                    Bxc��L  �          @�
=����  �o\)�gQ�C[�
���J=q�|���}��CK�{                                    Bxc���  
�          @�\)����s33�#�
�aG�C~  ����n�R�5�$��C}��                                    Bxc���  �          @�33���Y��@z�A��\C��Ϳ��n{?�A���C���                                    Bxc��>  
�          @�G���G����n{�,��C~�f��G���{��G���
=C~�                                    Bxc���  T          @��\��p����ÿO\)��RC|� ��p�������z���
=C{�3                                    Bxc�ߊ  �          @�����z����\��R���
C:ῴz���(�������p�C~�                                    Bxc��0  �          @�녿�=q��z���Ϳ��C�
��=q���ÿ����2=qC��                                    Bxc���  
�          @������
=�W
=�G�C�׿����33��33�Ep�C�\                                    Bxc�|  �          @��ÿ޸R��{���H��=qC{���޸R���׿�z��n=qCz�                                    Bxc�"  �          @��\�\���H>��H@���C~
�\���H����C~�                                    Bxc�(�  �          @��ÿ�(�����?0��@��C�z῜(���=q��=q�333C��f                                    Bxc�7n  T          @���?Q���z�@Mp�B�
C��q?Q���z�@z�A�Q�C���                                    Bxc�F  T          @��>aG����@:�HA���C�0�>aG���?�Q�A���C��                                    Bxc�T�  T          @�ff>B�\��=q@{Aƣ�C�H>B�\��?�p�Ai�C���                                    Bxc�c`  �          @�z�>.{��33@�
A�33C��f>.{��p�?�=qARffC��R                                    Bxc�r  T          @���=�\)��=q@P  B��C�` =�\)���@33A�\)C�W
                                    Bxc���  T          @��
�8Q���z�@/\)A�Q�C����8Q�����?��
A��\C��                                    Bxc��R  �          @���>B�\��(�@G�BQ�C�  >B�\��33@p�A��C�f                                    Bxc���  �          @��=�����=q@hQ�B  C��=�����z�@1G�A�z�C��3                                    Bxc���  
Z          @\>�z���=q@S33B(�C���>�z����\@
=A�
=C�xR                                    Bxc��D  �          @��?����
=@fffBQ�C�˅?������@.{A�p�C�R                                    Bxc���  T          @�=q?������R@K�A��HC�  ?�����ff@��A��RC�p�                                    Bxc�ؐ  �          @�(�?�  ��z�@l(�B��C���?�  ���R@4z�A�C�
=                                    Bxc��6  
(          @��\?aG���33@L��B	z�C�c�?aG����H@ffA�p�C��H                                    Bxc���  �          @�  =�\)��ff@��A�p�C�t{=�\)����?�Q�A��RC�l�                                    Bxc��  �          @��=L����G�?���A�  C�W
=L����=q?���AR�RC�Q�                                    Bxc�(  T          @�
=>aG���@\)A�(�C�^�>aG���  ?��Aw�C�G�                                    Bxc�!�  �          @�ff�\)���\?��RA�
=C�|)�\)����?+�@���C��                                     Bxc�0t  �          @�z�J=q����?��@��HC�O\�J=q��=q�u�333C�W
                                    Bxc�?  �          @�\)�������>�\)@I��C��������  �
=���
C���                                    Bxc�M�  T          @�Q���H��\)������ffC{����H���\����[33C{n                                    Bxc�\f  
�          @����.{��  �L���	��C�S3�.{��(���{�@��C�=q                                    Bxc�k  �          @���h�����H�0����z�C����h�����Ϳ�������C���                                    Bxc�y�  �          @��Ϳ�����p����ȣ�C{ٚ�����~{�:=q�
  Cy��                                    Bxc��X  �          @���ٙ��qG��=p��\)Cu��ٙ��L���c�
�2�Cr
=                                    Bxc���  �          @��H�z��9���|���?�Cj���z�����z��^�Cb}q                                    Bxc���  �          @�����
�}p�������CM�=��
�#�
���
\C8c�                                    Bxc��J  �          @�녿��ͼ#�
��
=.C4k�����?W
=�����C�=                                    Bxc���  �          @�{�G��n{��
=�zQ�CJL��G��������� C6�f                                    Bxc�і  T          @�33�
=��ff��(�L�C?���
=>�����z��=C)^�                                    Bxc��<  �          @�{�\)�fff��z�ǮCI���\)�#�
��
=��C4!H                                    Bxc���  �          @�
=�
�H���R���
�{z�CQ�{�
�H�Ǯ��Q���C>5�                                    Bxc���  �          @�p��
�H��{���H�h�C\�)�
�H��ff���H�}�CM�
                                    Bxc�.  �          @������5�����@��Ce�������=q�]  C]0�                                    Bxc��  �          @�\)�{�z=q�Tz����Cm�q�{�R�\�|(��/  Ci�                                    Bxc�)z  �          @��H��R����O\)�{Co33��R�c33�y���({Ck�                                    Bxc�8   �          @��R�33����;��{Csp��33�c33�e��$��Co�3                                    Bxc�F�  �          @��Ϳ���q��"�\��=qCs�{����R�\�H�����CpO\                                    Bxc�Ul  �          @�G���������p���=qCv�����r�\�I���z�Csz�                                    Bxc�d  �          @�����R����� ����Q�Cth���R����1G���(�CrL�                                    Bxc�r�  T          @�G����
��{�������Cy@ ���
�����ӮCw�\                                    Bxc��^  �          @��Ϳ�ff�c�
����ffCs+���ff�Fff�?\)�z�Co�
                                    Bxc��  �          @��\�����|���C�
���CuT{�����XQ��l(��/p�Cq�                                     Bxc���  �          @��R�����w��>{�
=CtǮ�����Tz��e��-�Cq0�                                    Bxc��P  �          @�{�G��Z=q�c�
�#�ClQ��G��0�����\�C(�Cf��                                    Bxc���  �          @����R�E�e�(�Cg:���R�(�����E\)C`��                                    Bxc�ʜ  �          @�\)�333�S33�U��\)Ce���333�,(��u�2��C_�)                                    Bxc��B  T          @��
�/\)�K��Q��(�Ce8R�/\)�%��qG��433C_E                                    Bxc���  T          @����(�����
��
=����Cn�
�(���~�R�(Q�����Cls3                                    Bxc���  T          @��H�>{�����=q�2�HCm5��>{��������
Ck�                                     Bxc�4  �          @�  �6ff��z῀  � ��Co�R�6ff���Ϳ�����CnxR                                    Bxc��  �          @��R�8Q����ͿJ=q��(�Cp� �8Q���ff�����  Co�                                     Bxc�"�  �          @�  �A�����B�\����Cn��A����׿�  �   Cm��                                    Bxc�1&  �          @�=q�4z���
=>�(�@�{CmǮ�4z���\)���
�\��Cm�{                                    Bxc�?�  T          @��
�HQ��fff>�z�@a�Cd�q�HQ��fff���R�n{Cd��                                    Bxc�Nr  T          @�p��W
=���Ϳ޸R����Cg��W
=�s33�=q��G�Cd�\                                    Bxc�]  �          @���N{��\)�����33Ch��N{�tz��3�
��33Ce�H                                    Bxc�k�  �          @�������H>���@ECu����=q�����33Ct�                                    Bxc�zd  �          @�=q�	������?���A2�\Cwٚ�	������>k�@�CxE                                    Bxc��
  �          @��
�z�����@.�RA�RCqh��z����?�p�A��HCs}q                                    Bxc���  T          @������(�@��AͮCq�f���
=?У�A��Cs��                                    Bxc��V            @����!�����?�\)A��\Cp���!�����?��A6�HCr{                                    Bxc���  n          @�  �C33��G�?��AD(�Cj�H�C33��p�>�
=@�Ckp�                                    Bxc�â  �          @�Q��^{���?�  A&�RCf+��^{����>�z�@C33Cf�H                                    Bxc��H  �          @�z��:=q����?�Q�AC
=Cm���:=q��G�>���@��\Cn�3                                    Bxc���  T          @�Q��/\)��?�Q�A>�HCp���/\)���>�Q�@dz�Cq�{                                    Bxc��  T          @�33�C33��33?���A.ffCmٚ�C33��
=>�z�@3�
Cnu�                                    Bxc��:  �          @��\�9����=q?��AZffCm���9����
=?
=q@��\Cnc�                                    Bxd �  T          @���2�\���?�=qA]p�Cn�{�2�\����?��@�p�Co��                                    Bxd �  �          @�(��>{���?��RALQ�Cm=q�>{��Q�>��@��Cn                                    Bxd *,  �          @��
�^{��>�@�\)Cih��^{��{���
�I��CixR                                    Bxd 8�  T          @���P  ��ff>��@��HCi�)�P  ��
=��=q�5�Ci�3                                    Bxd Gx  �          @��\�XQ���33�����O\)Cf� �XQ���  �xQ��(Q�Ce�                                    Bxd V  �          @����C�
��G���\)�A�Cju��C�
���z�H�+�Ci�=                                    Bxd d�  �          @��AG����=u?&ffCj��AG�����!G���  Ci�R                                    Bxd sj  T          @���O\)���.{��ffCh.�O\)���H�Y���  Cg�)                                    Bxd �  �          @��'
=��  ?\A�Q�Co�H�'
=��?B�\A ��Cp��                                    Bxd ��  �          @�{�%����
?�(�AO�
Cp���%���Q�>�ff@�ffCq��                                    Bxd �\  �          @�z���H���\?�\)AA��Cw녿��H��{>��R@Tz�Cxc�                                    Bxd �  �          @�\)�����H?�Q�A���Cu33����33?�z�AEG�Cv\)                                    Bxd ��  T          @�����R��G�@G�A��Cu
=��R���?��HAE��Cv5�                                    Bxd �N  �          @��
�E�Tz�@S33BG�Cc
�E�tz�@-p�A�p�Cg
=                                    Bxd ��  �          @��
�{����@�A���Cr�=�{��33?���A�\)Ct8R                                    Bxd �  �          @��\��=q��\)@1G�A�Q�Cw�Ϳ�=q���
@   A�  Cyn                                    Bxd �@  �          @�ff��33����@3�
A��Czz��33����@G�A�=qC{��                                    Bxd�  �          @�(���{���H@1�A�p�Cx��{��\)?���A��RCzaH                                    Bxd�  �          @��Ϳ�����  @,(�A�\)C|�=������(�?�A�33C}�                                    Bxd#2  �          @�(���Q����@0��A�Cz�ÿ�Q�����?�A�=qC|B�                                    Bxd1�  �          @������p�@3�
A��C{G�����=q?�(�A��C|��                                    Bxd@~  �          @�(��޸R���@8��A�Cz
�޸R��
=@�
A�z�C{��                                    BxdO$  "          @��׿����z�@6ffA�
=Cw녿������@33A���Cy�=                                    Bxd]�  �          @�
=���R��G�@(Q�A�ffC{�׿��R���?���A��\C}+�                                    Bxdlp  �          @���\)�g�@/\)B  Cn=q�\)����@
=A��Cp�)                                    Bxd{  �          @�33�<(���Q�@P  B/\)CQ��<(����@;�B�\CXٚ                                    Bxd��  �          @�{��z��9��@.{B\)Cl����z��S33@p�A��Co�R                                    Bxd�b  �          @�p���33��\)@33A���C�޸��33��  ?�  A_�C��)                                    Bxd�  �          @�(��:�H���\@	��A���C�t{�:�H���?�\)AyG�C��R                                    Bxd��  T          @����&ff���?�\A���C�  �&ff��z�?z�HA4z�C�/\                                    Bxd�T  T          @�\)��33��ff?�=qA�=qC�⏾�33����?J=qAffC��
                                    Bxd��  �          @�����{?�{A��C�:����33?z�@���C�Q�                                    Bxd�  �          @�<#�
���\?Y��A33C�<#�
���=L��?��C�                                    Bxd�F  
Z          @��>aG����
>�@�p�C�4{>aG���(��\��
=C�4{                                    Bxd��  T          @�=q���H�g�@��HB;�
C{�\���H���@\(�B�HC~)                                    Bxd�  
�          @�{��{�y��@~�RB1��C~���{��  @Q�BG�C��                                    Bxd8  �          @��ÿ��R��\)@n�RB ��C}�����R����@?\)A�\)CxR                                    Bxd*�  �          @�\)��\)���@Y��B
=C�R��\)���@'�A��HC���                                    Bxd9�  T          @����ff���\@=p�B
=C}O\��ff��  @{A��C~��                                    BxdH*  �          @��׿��s33@I��BG�C��3����Q�@�RA�{C�(�                                    BxdV�  �          @��R�p������?��A���C�!H�p������?��\A>ffC�h�                                    Bxdev  T          @����33��\)?�ffA�p�C�+���33��p�?B�\Az�C�l�                                    Bxdt  �          @����  ��>�p�@��
C�R��  ��p���G����C��                                    Bxd��  �          @������
��{=�Q�?��\C�j=���
��z�B�\�\)C�Z�                                    Bxd�h  �          @��������
�#�
���C�=q��������aG��
=C�'�                                    Bxd�  T          @�Q쿆ff���;8Q��Q�C�:ῆff�������\�2�\C��                                    Bxd��  �          @�{�8Q���(�=�\)?@  C�f�8Q���=q�E��	G�C���                                    Bxd�Z  �          @�z�!G����H�#�
��G�C�� �!G����׿^�R���C�s3                                    Bxd�   T          @�G��E���
=>�\)@O\)C��
�E���ff����=qC��3                                    Bxdڦ  �          @�33�5��ff�^�R�&ffC���5����У���C���                                    Bxd�L  T          @��R�Y�����ÿY���=qC���Y�����\��\)��\)C���                                    Bxd��  "          @��Ϳ�p����H��(�����Cy^���p���ff��
=�]G�Cx��                                    Bxd�  �          @����ff����aG��!�C{�f��ff��ff�У���ffCz�)                                    Bxd>  �          @�=q��Q����Ϳ�ff�o33C|�q��Q����
�33���C{�R                                    Bxd#�  �          @���B�\��33�n{�#\)C��=�B�\��(���  ��ffC���                                    Bxd2�  �          @��0����녿aG��G�C�N�0����33�޸R���C�&f                                    BxdA0  �          @��׿&ff����O\)��\C�n�&ff��ff��33��=qC�H�                                    BxdO�  T          @�\)�n{��(���{�e�C�{�n{��  ��(��N=qC���                                    Bxd^|  �          @��\�333��=q��z���p�C��H�333�����(���Q�C��q                                    Bxdm"  �          @�녿=p����R>\)?���C��3�=p�����5����C���                                    Bxd{�  T          @��H�@  ���?
=@�z�C���@  ���׾�\)�>�RC��3                                    Bxd�n  �          @��
�G���G�>��?˅C�˅�G���  �8Q�����C�                                    Bxd�  T          @���(����33��=q�B�\C�Uÿ(�������{�G
=C�@                                     Bxd��  �          @�
=�.{��z�k��'�C���.{��G����
�@  C���                                    Bxd�`  
�          @�\)�!G���p��W
=��RC�XR�!G���녿��\�>=qC�C�                                    Bxd�  �          @�녿@  ��
=>���@n{C���@  ���R�   ���C��\                                    BxdӬ  �          @�\)�(����z�>�{@y��C�%�(����(������Q�C�"�                                    Bxd�R  �          @��þ�����=L��?�C�N�����p��G���HC�Ff                                    Bxd��  "          @��ͿTz���Q�>��@���C�ÿTz����þ��
�u�C�R                                    Bxd��  �          @��Ϳ�=q��
=?�A�\)C}s3��=q��z�?(��@���C~�                                    BxdD  �          @�G���{����=���?���C��H��{��  �0���G�C�p�                                    Bxd�  T          @��\�0����(������D  C��)�0����z��{��p�C��f                                    Bxd+�  �          @�zῥ���p��
=��=qC\)�����Q쿱��|z�C~�H                                    Bxd:6  �          @�������L���(�Cx� ����\)�Ǯ���HCw��                                    BxdH�  �          @��Ϳ�ff��
=����p��C{�{��ff���ff���HCz�q                                    BxdW�  �          @�����
��Q�
=q��\)C�,Ϳ��
��33��{�uG�C��q                                    Bxdf(  �          @�\)�u������p�����C��u��ff�{��Q�C��3                                    Bxdt�  �          @��.{��
=�����b{C�箿.{��ff��(����RC��=                                    Bxd�t  �          @�(���R���\�����33C�����R��\)�}p��2�\C�w
                                    Bxd�  �          @�{�s33���\>\)?\C���s33��G��5����C���                                    Bxd��  T          @��þ�(���z�?�\)AC�C�� ��(���  >�  @*=qC��                                    Bxd�f  �          @�Q쾣�
��
=?�  A�=qC�"����
����?&ff@�ffC�33                                    Bxd�  T          @��
��Q���@A�p�C�����Q����R?��RAUp�C�                                    Bxd̲  �          @��þ�(���z�?�33A��C�~���(���z�?��A9��C��q                                    Bxd�X  �          @�녾�����Q�@33AͅC���������=q?�(�A��RC���                                    Bxd��  �          @�(���(�����@=qA�p�C�h���(����
?�=qA���C��{                                    Bxd��  
(          @�{�k����@!G�A܏\C��q�k���p�?�A��C���                                    BxdJ  �          @��\>\)��33@+�A��
C���>\)��\)?�\)A���C���                                    Bxd�  �          @�Q����33@#33A�ffC�8R�����R?�  A��C�Ff                                    Bxd$�  ;          @��׾�����p�@=qA�\)C��R������Q�?˅A�(�C�)                                    Bxd3<  �          @�p��(���p�@#33A�Q�C�ÿ(���G�?�\A��C�aH                                    BxdA�  �          @�ff�5��\)@   A���C���5���H?��HA�(�C��R                                    BxdP�  T          @�Q�h����33@0��A�Q�C���h����  ?�p�A�(�C���                                    Bxd_.  �          @���s33�qG�@]p�B&�C���s33����@1G�B �RC�˅                                    Bxdm�  
�          @�z�J=q���?�p�A�=qC���J=q��?333A33C�4{                                    Bxd|z  "          @�33��Q���p���33�qG�C��=��Q���G���(��T(�C�^�                                    Bxd�   T          @����\)��\)>�?�33C��R��\)���B�\�{C���                                    Bxd��  T          @���������+����C�������
=��=q���C��{                                    Bxd�l  "          @�G������\)��Q��p��C�ٚ������H����W�C��                                    Bxd�  
�          @�(��   ��33��=q�2�\C�lͿ   ��
=��(��Hz�C�]q                                    BxdŸ  "          @��ÿ����\)��p��r�\C��׿�����H����XQ�C��\                                   Bxd�^  T          @��;��
���\?z�@�{C�L;��
��33��{�fffC�N                                   Bxd�  �          @��
������p�?���Ah��C�33�������\>�G�@���C�@                                     Bxd�  �          @�������Q�@0��B��C�7
����p�?��RA���C�H�                                    Bxd P  �          @��
>\)�e@~{B?\)C�#�>\)��
=@S33B
=C��R                                    Bxd�  �          @���>��R�P��@�
=BW�C���>��R�~�R@vffB/�
C�5�                                    Bxd�  �          @�(�?G��<(�@��
BVQ�C�o\?G��fff@c�
B/��C�)                                    Bxd,B  �          @��?���.{@��
BZ�C�~�?���X��@eB5�\C���                                    Bxd:�  �          @�  ?L���,(�@��
B^��C�:�?L���W
=@fffB8�C���                                    BxdI�  T          @�z�?n{�\)@���Bk\)C�1�?n{�Mp�@y��BE�RC��                                    BxdX4  "          @��R?���Q�@�BhG�C��?���Dz�@mp�BC\)C���                                    Bxdf�  	�          @��?c�
��(�@�\)B�C��)?c�
�{@��Bp\)C��\                                    Bxdu�  "          @�z�?��ٙ�@�G�B��C��?��p�@�p�Bk�C��                                    Bxd�&  
�          @����u��\@�(�B�ffC�7
�u�3�
@��RBa  C�n                                    Bxd��  �          @��þ8Q��P��@s�
BE��C�b��8Q��w�@L(�BC���                                    Bxd�r  
�          @�녾�z��J�H@xQ�BJ��C�c׾�z��s33@QG�B"��C���                                    Bxd�  �          @�녾W
=�5@Z�HBH�
C�箾W
=�X��@7�B ��C�>�                                    Bxd��  
�          @�G�?fff�$z�@@  B?Q�C���?fff�C33@ ��B33C�@                                     Bxd�d  
�          @�  ?z�H�k�@�{B��HC�ff?z�H���@��\B�aHC��\                                    Bxd�
  �          @���?�>�
=@�  B��A_�
?���\@�  B�(�C��
                                    Bxd�  
(          @��H?�Q�>�@��RB��A`z�?�Q��(�@�
=B�\)C��\                                    Bxd�V  �          @��?���?�33@�(�B�8RB/  ?���>k�@�  B��HA.ff                                    Bxd�  �          @��
?�ff?fff@��B��A�ff?�ff��@�=qB��C�s3                                    Bxd�  
�          @��\?�Q�?@  @�z�B���A�\)?�Q�#�
@�{B���C���                                    Bxd%H  �          @�ff>�=q��G�@��B��\C�g�>�=q�(Q�@�
=Bs  C���                                    Bxd3�  �          @��?(�ÿ   @���B��C�b�?(�ÿ�(�@�33B��\C�                                      BxdB�  
          @��?�
=��
=@�
=B�G�C��?�
=����@��B�L�C�8R                                    BxdQ:  �          @��?�=q?�33@�(�B��RB#\)?�=q>B�\@�  B�z�@�                                    Bxd_�  
�          @��H?��?��@�(�B�k�Bk��?��?s33@�(�B��B                                    Bxdn�  �          @���?�33@%@�
=Bl(�B�.?�33?��H@�33B�B`��                                    Bxd},  m          @���?��
@H��@�
=BI�HB��?��
@@��RBmz�Bcp�                                    Bxd��  �          @�z�?��@`  @�ffBA��B��=?��@,��@�  BgQ�B��f                                    Bxd�x  
�          @��?s33@g
=@��BE��B�k�?s33@1�@�{Bl�HB��                                    Bxd�  �          @��H?�(���z�@`��Bz�C�� ?�(���{@*�HA݅C��=                                    Bxd��  �          @��R?.{���@��B5��C��R?.{����@\(�BQ�C��                                    Bxd�j  �          @�?5��  @uB��C�9�?5���
@;�A�C��                                     Bxd�  �          @�?c�
�~{@���BC33C�J=?c�
���@w
=BC�K�                                    Bxd�  �          @�p�?���s33@�
=BHffC��?�����H@}p�B Q�C�}q                                    Bxd�\  ;          @�  ?�\�q�@��RBQ��C��R?�\��33@�ffB(�RC�(�                                    Bxd  
s          @�ff?Q��`  @��BZ�\C��)?Q���33@�33B1��C�XR                                    Bxd�  �          @�p�?O\)�Vff@��B`(�C���?O\)���R@�B7�\C�t{                                    BxdN  T          @���?z�H�`  @��BZz�C��R?z�H���@���B2�C�W
                                    Bxd,�  
�          @���?aG��|��@��BG�HC�AH?aG���  @���B  C�9�                                    Bxd;�  �          @�?��|��@�BH�\C�>�?�����@�z�B =qC��                                     BxdJ@  "          @�z�?�z��x��@�BJ  C�Ff?�z���
=@�z�B!��C��                                    BxdX�  
Z          @��H?�33�dz�@�G�BR��C���?�33��@��B+G�C�˅                                    Bxdg�  
�          @У�?�  �z�H@�\)BG�\C�w
?�  ��Q�@�B�HC��q                                    Bxdv2  "          @˅?У���  @�B1
=C��?У���\)@e�B	ffC�3                                    Bxd��  "          @��þ#�
��=q@#�
A�33C�'��#�
��{?�  Ab{C�5�                                    Bxd�~  �          @�33��ff���R@	��A�{C����ff��  ?��\A�
C���                                    Bxd�$  �          @ə��L����(�@3�
A�Q�C�  �L������?�p�A�C��                                    Bxd��  
Z          @ȣ׽��
��z�@.{A�(�C��3���
����?��Ar�HC���                                    Bxd�p  �          @�Q�aG����@*=qA�p�C�⏾aG�����?���AiG�C���                                    Bxd�  �          @�
=�\)���R@
=A��C�4{�\)����?��\A<��C�\)                                    Bxdܼ  �          @Ǯ������\@
=qA�{C��
�����(�?��A=qC��3                                    Bxd�b  �          @�\)>�����?�\)A��RC�G�>������?@  @�C�:�                                    Bxd�  ;          @ə�?=p����H@	��A���C���?=p���z�?��
A�C�q�                                    Bxd	�  
s          @�G�>W
=���R@'�A�33C�{>W
=��33?�G�A_
=C��                                    Bxd	T  T          @�\)<���{@   A�{C�&f<����?�33APz�C�%                                    Bxd	%�  
Z          @ƸR�u���\@.{A��
C��\�u��\)?У�As�C��{                                    Bxd	4�  
�          @�\)>B�\��G�@333A�Q�C��)>B�\��
=?��HA�(�C��=                                    Bxd	CF  
�          @�  ���
���@>{A�p�C��=���
��ff?��A��C��                                    Bxd	Q�  "          @�ff?.{����@L(�A�=qC���?.{����@	��A�Q�C�^�                                    Bxd	`�  �          @Ǯ?�ff����@z�A�{C�=q?�ff���?�(�A5p�C��3                                    Bxd	o8  "          @�\)?��\��
=@�
A���C�@ ?��\��Q�?p��A\)C��
                                    Bxd	}�  f          @�=q?�ff����@!�A��
C�|)?�ff����?�(�A`��C�!H                                    Bxd	��  �          @���?���  �����
=C��f?�������
�D��C���                                    Bxd	�*  �          @�G�=�\)����>�  @ffC�Y�=�\)���R�fff�	G�C�Z�                                    Bxd	��  �          @��
=u������
�   C�K�=u�����Q�����C�O\                                    Bxd	�v  �          @��H�.{��ff��33�/\)C�,;.{��(��G���p�C�!H                                    Bxd	�  �          @���>aG����
��=q�LQ�C��>aG���Q�����  C�#�                                    Bxd	��  �          @���8Q����O\)����C�#׾8Q������
=��(�C��                                    Bxd	�h  �          @�Q�\)��ff�L����z�C�Uþ\)��{����p�C�N                                    Bxd	�  �          @��=L�����H�0���أ�C�>�=L����33��ff����C�AH                                    Bxd
�  �          @���<��
���R�B�\��C�R<��
���R����
=C��                                    Bxd
Z  �          @�33?z���Q�#�
�ȣ�C���?z����ÿ�p����\C���                                    Bxd
   �          @�=q�8Q����ÿ����HC�R�8Q���녿����C�\                                    Bxd
-�  �          @����#�
��
=�\(��(�C�/\�#�
��{������G�C�%                                    Bxd
<L  �          @�\)>���{����Q�C���>���\)��\)��p�C��\                                    Bxd
J�  �          @�33>�(����R>\)?��HC�:�>�(���(��k��\)C�C�                                    Bxd
Y�  �          @��R?(����p�?��A{�C�|)?(�����>�ff@��\C�]q                                    Bxd
h>  �          @�  ?(������@Q�A�=qC���?(����33?��A)C�b�                                    Bxd
v�  �          @���?z���?W
=Ap�C���?z���\)��z��8Q�C��f                                    Bxd
��  �          @��?����
?�z�A���C�� ?����H?�@��C��                                    Bxd
�0  �          @��?}p���33@��A��
C�<)?}p���p�?�ffA%G�C��3                                    Bxd
��  �          @�  ?����H@�A�C�*=?���
=?�=qAM�C��f                                    Bxd
�|  �          @�  ?�=q���@
=A�
=C���?�=q��Q�?��RA?�C�S3                                    Bxd
�"  �          @�
=?�{��@'
=AϮC�T{?�{��33?��
AnffC��\                                    Bxd
��  �          @�{?ٙ���{@5�A�C�xR?ٙ�����?�ffA��HC���                                    Bxd
�n  �          @��?z�H��{@(�A�Q�C�S3?z�H����?�\)A4z�C��                                    Bxd
�  �          @�(�?�{��ff@,(�AڸRC�H?�{��z�?�33A��C�Q�                                    Bxd
��  �          @�{?�33���@L��B�C�7
?�33����@{A�(�C�!H                                    Bxd	`  �          @��@ ���E@�G�B5��C��@ ���u@UBQ�C���                                    Bxd  �          @�  @4z��
=@��RB\p�C�� @4z��%@���B@�C��q                                    Bxd&�  �          @�p�@:�H���@��B]
=C�H�@:�H��\@�G�BDp�C��=                                    Bxd5R  �          @��@#�
���
@���Bc��C���@#�
�-p�@��HBEp�C���                                    BxdC�  �          @�Q�@7���  @�{BYp�C�N@7��*=q@��B=33C���                                    BxdR�  �          @���@'
=� ��@��BL�\C�{@'
=�W
=@y��B)��C��=                                    BxdaD  �          @��@���G
=@���B=p�C��=@���z=q@c33B�
C��q                                    Bxdo�  �          @�
=@7
=�333@��HB6��C��f@7
=�dz�@\(�BC�P�                                    Bxd~�  �          @�33@(��Q�@r�\B+G�C�\)@(��~�R@B�\B\)C�Ǯ                                    Bxd�6  �          @�G�@1G�����@�BXG�C�e@1G��7�@�B9z�C�H                                    Bxd��  �          @�G�@;���p�@��HBQffC�H@;��7�@��HB3p�C���                                    Bxd��  �          @�  @)����p�@�=qBg\)C�g�@)����@��BKffC���                                    Bxd�(  �          @�G�@Dz��R@��Bd�HC�N@Dz����@�(�BT�C�7
                                    Bxd��  �          @�ff@E��5@�\)B`�C��@E���@�\)BO��C���                                    Bxd�t  �          @��@E��
=q@�=qBd  C��q@E��\@�33BT��C��{                                    Bxd�  �          @�Q�@U���@�p�BYp�C��@U��(�@���BO=qC�                                      Bxd��  �          @�  @2�\�8��@=p�B�\C�@2�\�\(�@33A�p�C��=                                    Bxdf  �          @��
@z��{�@O\)B��C�J=@z���Q�@
=A�33C���                                    Bxd  �          @���@33��
=@EB\)C���@33����@��A�\)C���                                    Bxd�  �          @��
?�ff����@33A��RC���?�ff��z�?��A4z�C�>�                                    Bxd.X  �          @��
?�  ��  @E�A��
C���?�  ��G�@G�A�\)C��                                    Bxd<�  �          @���@ff�o\)@j�HB"33C��@ff��@3�
A�ffC��                                    BxdK�  �          @��R@8�ÿ��@�B_�\C���@8���G�@���BF�C��                                    BxdZJ  �          @�=q@:=q>u@�  Bn��@�  @:=q�Q�@�{Bj�C�.                                    Bxdh�  �          @�=q@C�
?B�\@�(�BdA^ff@C�
��z�@�Bhp�C�H�                                    Bxdw�  �          @�  @�33�5@z�HB-33C��@�33���
@l(�B \)C��=                                    Bxd�<  �          @�@�p��(�@`  Bz�C�{@�p����@R�\B��C��
                                    Bxd��  T          @��@q�>\@�G�BHQ�@��R@q녿
=@���BF�C���                                    Bxd��  �          @�z�@n{>�G�@�G�BJ�@�\)@n{�
=q@���BIp�C��H                                    Bxd�.  �          @�z�@hQ��@�33BN�HC���@hQ쿁G�@��BH  C�AH                                    Bxd��  �          @���@@�׿��@�p�B[�
C�o\@@�����@�G�BB�C���                                    Bxd�z  �          @Ӆ@L(��<(�@�(�BA�C���@L(��z=q@�z�BQ�C��{                                    Bxd�   �          @Ӆ@Z=q��@���BM�HC��@Z=q�U�@�G�B.ffC��\                                    Bxd��  �          @��
@]p��ff@���BO\)C�U�@]p��J=q@�=qB1G�C��                                    Bxd�l  �          @�  @4z��z�H@�B0z�C���@4z����H@n�RB�\C�)                                    Bxd
  �          @ڏ\@&ff��33@�G�B=qC���@&ff��p�@L(�A�C��
                                    Bxd�  �          @ۅ@ ����
=@h��B �C�ٚ@ ����(�@�HA�{C��                                    Bxd'^  �          @ڏ\@%����@Q�A�\)C�@%����@G�A��
C���                                    Bxd6  �          @�{@���z�@&ffA��\C�aH@���=q?��A6ffC��                                    BxdD�  T          @Ӆ?�z���ff?�A#�C���?�z��ə��W
=����C�n                                    BxdSP  �          @��@�R��{>.{?�p�C�T{@�R���\����"�RC��                                    Bxda�  �          @ҏ\?��R��(�?�A��RC��{?��R����?�@��C�xR                                    Bxdp�  �          @Ӆ?u�ʏ\?���A\��C�Q�?u����>\)?�p�C�1�                                    BxdB  �          @Ӆ?�
=��@   A�{C�?�
=���H?���A��C�W
                                    Bxd��  �          @�z�?�=q��{@�A�33C��{?�=q��  ?333@��C��R                                    Bxd��  T          @�33?������
@	��A��C��=?�����{?=p�@�p�C�C�                                    Bxd�4  �          @�(�?������@0  AĸRC���?�����=q?���AA�C��                                    Bxd��  �          @�p�?z�H����@H��A�\)C��?z�H�ʏ\?�\Aw�C�e                                    BxdȀ  �          @�\)@\)���@n�RB�
C��@\)����@ ��A�Q�C���                                    Bxd�&  �          @�{@p���\)@���B�HC��=@p���G�@?\)A���C���                                    Bxd��  �          @�ff@z���\)@��B�HC��@z�����@<��A�G�C�b�                                    Bxd�r  �          @׮@,(����\@���B#33C��=@,(����R@S�
A��
C��f                                    Bxd  
�          @�G�@0�����@�  B��C���@0�����\@HQ�A�z�C���                                    Bxd�  �          @�Q�@$z���ff@��HB �RC���@$z����@N{A�
=C��                                    Bxd d  �          @�  @6ff����@�  B'33C�7
@6ff���@\(�A�\)C���                                    Bxd/
  �          @�\)@&ff����@��B �\C�S3@&ff��Q�@L��A���C�,�                                    Bxd=�  �          @�\)@��
�xQ�@�Q�B5  C���@��
��(�@�p�B$33C��R                                    BxdLV  �          @�
=@��\?�@�Q�B1�@�(�@��\��\@���B1��C��3                                    BxdZ�  �          @�@�z�aG�@�z�BJ�C��q@�z���@���B8��C���                                    Bxdi�  �          @�@����  @�z�B<��C��R@���ff@���B+G�C�K�                                    BxdxH  �          @У�@�p���{@��B2ffC�u�@�p���@�(�B��C��                                    Bxd��  �          @�ff@�����@�B.��C���@���7
=@xQ�B��C�                                      Bxd��  �          @�33@y���ff@�=qB7\)C���@y���Fff@}p�B��C��                                    Bxd�:  �          @�{@���@�p�B"�C�U�@���?\)@dz�BffC��                                    Bxd��  �          @���@��R�Ǯ@{�B�C�AH@��R��@]p�B {C��3                                    Bxd��  �          @θR@p���	��@�ffB=\)C�q@p���K�@�=qBG�C�޸                                    Bxd�,  �          @�=q@'
=�0��@��HBZffC��3@'
=�z�H@���B0  C���                                    Bxd��  �          @ҏ\@$z��5@�33BYp�C��@$z���Q�@���B.G�C�S3                                    Bxd�x  �          @��@��A�@���BW�C�H@���{@�{B*�RC��H                                    Bxd�  �          @�33@ff�I��@���BU�\C�]q@ff���@��B'�RC�Q�                                    Bxd
�  �          @�z�@
=�Fff@�(�BX�C���@
=����@�  B*�RC�o\                                    Bxdj  �          @�z�@1G��2�\@��HBVffC�c�@1G��~{@���B,G�C�t{                                    Bxd(  �          @�z�@7��&ff@�(�BY�C���@7��s33@��B0z�C���                                    Bxd6�  �          @�  @9���1G�@��BOz�C�%@9���y��@���B&�C�L�                                    BxdE\  �          @У�@<���1G�@��HBN�C�h�@<���y��@�G�B%
=C��                                    BxdT  �          @�G�@6ff�G�@��BF�HC�9�@6ff���R@�33Bp�C�
=                                    Bxdb�  �          @У�@C�
�Fff@��
B@�C�Ff@C�
��@~�RBffC��                                    BxdqN  �          @�G�@Vff���H@�\)Bc�
C�)@Vff��R@���BI\)C���                                    Bxd�  �          @У�@Y������@�(�B_z�C�c�@Y���$z�@��BD{C�q�                                    Bxd��  �          @�=q@Q녿u@�=qBe=qC�ٚ@Q��p�@�BM=qC��                                    Bxd�@  �          @���@<(����H@�G�B{
=C�8R@<(����@�  Bf�C���                                    Bxd��  �          @���@U��\(�@���BhQ�C�Ǯ@U��
=q@���BQffC�z�                                    Bxd��  �          @θR@`�׿p��@��\B^�
C�}q@`���p�@�ffBH{C��\                                    Bxd�2  �          @�ff@^{���@�p�BU�\C�c�@^{�5@��B7=qC�Z�                                    Bxd��  �          @��@g���{@�G�BWffC��=@g��'
=@���B<ffC�
                                    Bxd�~  �          @�=q@\�Ϳ��@�p�B_(�C���@\���%�@�{BC�RC��H                                    Bxd�$  �          @�=q@O\)���@�Q�Be��C�e@O\)�,��@�Q�BG��C�                                      Bxd�  �          @��@`�׿�G�@�=qBY(�C�e@`���0��@�G�B;��C��                                    Bxdp  �          @ҏ\@S33��z�@��\BY��C�H@S33�J=q@�ffB7�C�&f                                    Bxd!  �          @ҏ\@{��z�@��BP�RC��@{�����@��B@(�C��\                                    Bxd/�  
�          @��
@��
���@���BM�C�/\@��
�\@�p�BA
=C��R                                    Bxd>b  T          @�(�@o\)���\@�=qBU�\C���@o\)�#33@��HB;��C��)                                    BxdM  �          @�33@n�R��33@��BJ�RC�w
@n�R�G
=@�\)B*�RC��                                    Bxd[�  �          @Ӆ@��׿�{@�G�BG�C��@����$z�@�G�B.G�C��3                                    BxdjT  �          @��H@|����@��HB=C�j=@|���J�H@�B{C��H                                    Bxdx�  �          @��
@x���p�@��
B=�RC�7
@x���Vff@��B{C��q                                    Bxd��  �          @Ӆ@x�ÿ���@�{BB�C��3@x���G�@�G�B"�C���                                    Bxd�F  �          @��@s�
��@�33B?z�C�)@s�
�U�@���Bp�C�l�                                    Bxd��  �          @���@e���\@�=qBK��C�,�@e��P  @���B)=qC��H                                    Bxd��  �          @У�@�p��E�@��
BC�RC��@�p���(�@���B1�C�W
                                    Bxd�8  �          @��@w���z�@��BL�\C���@w��*=q@��HB1ffC��                                     Bxd��  �          @љ�@}p����@���BHp�C�L�@}p��'�@�  B-��C�=q                                    Bxd߄  
�          @��H@��H��  @��HB=��C�� @��H���@�B)=qC��                                    Bxd�*  �          @��H@���
=@���B9�
C�=q@�����@��RB*�RC�&f                                    Bxd��  �          @ҏ\@��H��(�@���BA=qC�/\@��H��@�(�B3p�C���                                    Bxdv  �          @ҏ\@��׿
=q@��B8�
C��\@��׿޸R@�ffB*G�C�u�                                    Bxd  �          @ҏ\@�ff�(�@��RB9�C��@�ff��ff@���B*
=C�                                      Bxd(�  �          @�G�@�녿�Q�@�  B"�C�s3@���>{@g
=B33C�q�                                    Bxd7h  �          @�=q@��Ϳ�33@��B �C��@����;�@fffB��C��f                                    BxdF  �          @��H@����@�\)B {C��@��:=q@fffBC��                                    BxdT�  �          @�G�@���@��B(��C��q@����@z=qB  C�)                                    BxdcZ  �          @�{@��R��@�\)B@��C�0�@��R��z�@���B6ffC�1�                                    Bxdr   �          @�@��?
=@�(�B?�@�Q�@���E�@��B>Q�C�@                                     Bxd��  �          @�33@���?:�H@��B>��Aff@����(�@��\B?p�C�*=                                    Bxd�L  �          @޸R@�z�?L��@��
B=�\AG�@�z�\)@���B?�C��=                                    Bxd��  �          @�  @��H?�  @�(�B<��As�
@��H�\)@���BC�C�!H                                    Bxd��  �          @߮@��
?�
=@�  BC
=A�
=@��
    @�{BL�
<��
                                    Bxd�>  �          @�  @�Q�?�  @���BCG�A���@�Q�>���@�G�BQ�@��\                                    Bxd��  �          @��@�p�?�33@�Q�B8�HA]��@�p��W
=@�z�B>��C��                                    Bxd؊  �          @�=q@�Q�>�G�@��RB&�H@�  @�Q�Q�@���B$��C���                                    Bxd�0  �          @�  @���>8Q�@�\)B*
=?��H@��Ϳ�=q@��B$�C�\                                    Bxd��  �          @ᙚ@�33>��@�33B.  @;�@�33���@�  B)Q�C�@                                     Bxd|  �          @�\@�=q>��@�B0�?�(�@�=q��@�G�B*�C���                                    Bxd"  �          @�  @��>�\)@���B1��@P  @�����@���B,��C�#�                                    Bxd!�  �          @ᙚ@�Q�>��@�ffB2��?�G�@�Q쿗
=@��B,Q�C�`                                     Bxd0n  �          @��H@�(�=��
@�z�B9�
?p��@�(���ff@�
=B2�C���                                    Bxd?  �          @�
=@��\>���@�  BCz�@q�@��\��\)@�z�B=�C��                                    BxdM�  �          @�G�@�\)>k�@��RB?  @3�
@�\)��
=@��\B8C���                                    Bxd\`  �          @�33@�\)>�\)@���B@G�@Z=q@�\)��z�@���B:�C�"�                                    Bxdk  �          @�z�@���=��
@�p�BEp�?�G�@��Ϳ��@��B<�RC��                                    Bxdy�  �          @�z�@�G�    @�  BI�HC���@�G���  @�G�B?�C��q                                    Bxd�R  �          @���@�
==�\)@��BL�H?n{@�
=����@��
BC\)C�
=                                    Bxd��  �          @�(�@�녾��R@�
=BH��C��@�녿��@�B:z�C�Ff                                    Bxd��  �          @�(�@���{@�=qBM�\C�Ф@�����@�Q�B>p�C���                                    Bxd�D  �          @���@��;��@�p�BE(�C�b�@��Ϳ޸R@�z�B7�
C��H                                    Bxd��  �          @�@��H�+�@�z�BE  C��R@��H�Q�@��B2z�C��{                                    Bxdѐ  �          @�
=@�\)�p��@���B8�\C�h�@�\)��@�=qB$  C�                                      Bxd�6  �          @���@z=q�У�@��BO��C��@z=q�Dz�@�  B.�HC��\                                    Bxd��  �          @�=q@`  ��z�@��
B]�RC�P�@`  �I��@�{B9�C�                                      Bxd��  �          @��@XQ��  @���Bb\)C�W
@XQ��Q�@���B<�C��                                    Bxd(  �          @�@]p��{@�{BO��C�4{@]p��x��@�G�B$G�C�Ф                                    Bxd�  �          @ۅ@n�R��R@�{BD�C�+�@n�R�u�@�G�B�C��                                    Bxd)t  �          @�ff@\���3�
@��BH33C�s3@\����@�=qB{C���                                    Bxd8  �          @�z�@k��,��@��\B@Q�C���@k�����@��
Bz�C�=q                                    BxdF�  �          @��@[��K�@��B>33C��q@[���\)@�Q�B��C��R                                    BxdUf  �          @�p�@R�\�L��@�p�BA�C���@R�\����@��B�HC��R                                    Bxdd  �          @�{@,(��g�@�B>{C�L�@,(����
@mp�B�RC�xR                                    Bxdr�  �          @��
@@���\(�@�{B0��C��{@@������@R�\A��\C��
                                    Bxd�X  �          @�  @b�\�
=q@���BG  C�K�@b�\�]p�@��HB�C���                                    Bxd��  �          @ָR@(���\)@�B1G�C��q@(�����@VffA�(�C��\                                    Bxd��  �          @��
@�����@�=qB#  C���@����H@6ffA��
C�e                                    Bxd�J  T          @љ�?�ff���@��HB
=C���?�ff���\@"�\A���C�p�                                    Bxd��  �          @���?����\)@eB�C���?����G�?�Q�A�Q�C���                                    Bxdʖ  �          @���?�33���@Y��B��C�Ff?�33��p�?��
A�\)C�}q                                    Bxd�<  �          @��?s33��
=@P��A��C���?s33���?�p�AN{C�4{                                    Bxd��  �          @�Q쿪=q��z�?�=qA  C��f��=q��ff�=p����
C��3                                    Bxd��  �          @�\)��33��\)?�=qA>{C�� ��33��33����(�C��R                                    Bxd.  �          @��
��\��=q?���A���C��ÿ�\���H���
���C��\                                    Bxd�  �          @ʏ\������?�=qA��HC�}q����G�<#�
=uC��R                                    Bxd"z  �          @ə�<����@��A�ffC�+�<���Q�?
=@���C�(�                                    Bxd1   �          @��
������p�@�A�(�C�� �������H?�\@��\C���                                    Bxd?�  �          @���>�����@A�Q�C���>���z�>k�@�
C���                                    BxdNl  �          @�
=�������?�
=A��HC�}q�����θR=#�
>�Q�C��\                                    Bxd]  �          @�ff����33@'�A��HC�k����˅?J=q@��C�w
                                    Bxdk�  T          @θR��ff���R@�A�Q�C��)��ff��=q>�=q@�RC��)                                    Bxdz^  �          @ƸR?����R@7�A݅C��?����\?��HA4z�C��                                    Bxd�  
�          @�33@W��\)@���B?��C�@W��p  @mp�BQ�C��{                                    Bxd��  �          @��H@L���_\)@��B'ffC�Ff@L�����H@AG�A���C�u�                                    Bxd�P  �          @�33@>�R�p  @�p�B${C�=q@>�R��=q@7�AָRC��                                     Bxd��  �          @���@p����@��
B5��C���@p���R�\@c33B��C�k�                                    BxdÜ  �          @Å@n�R��@���B3
=C��
@n�R�W
=@\(�B	�C�                                      Bxd�B  �          @\@vff��ff@��HB6z�C�y�@vff�@��@fffB�C��R                                    Bxd��  �          @�=q@o\)�ff@���B3=qC�XR@o\)�Q�@\(�B
��C�`                                     Bxd�  �          @���@i���@�p�B/�C��@i���^�R@QG�B  C�.                                    Bxd�4  �          @���@s33�@���B(�
C�1�@s33�\��@J=qA���C��                                    Bxd�  �          @��@�
=��\@u�B�HC��@�
=�Fff@A�A��C���                                    Bxd�  �          @���@hQ���R@�z�B-�C��3@hQ��g
=@L(�A�C��                                    Bxd*&  �          @�  @x���   @��HB,Q�C�]q@x���I��@R�\BffC��                                     Bxd8�  �          @�Q�@�33�{@qG�B��C�˅@�33�P��@:�HA�\)C���                                    BxdGr  �          @���@s33����@��RB2�C�l�@s33�H��@Z�HB(�C�9�                                    BxdV  �          @ƸR@�ff��@n{B�HC��q@�ff�Dz�@:�HA��C��R                                    Bxdd�  �          @�
=@��$z�?�p�A]C�W
@��9��?!G�@�=qC��3                                    Bxdsd  �          @ȣ�@��
�@{A�Q�C�:�@��
�:=q?�33AN�RC�Ǯ                                    Bxd�
  �          @�G�@�  �  @)��A�
=C�h�@�  �<��?�=qA�G�C�S3                                    Bxd��  �          @�G�@��R��{@\)A��RC���@��R�"�\?�ffA��HC��=                                    Bxd�V  �          @�=q@����Q�@(��A�G�C�k�@����H?��RA��\C��                                    Bxd��  �          @ʏ\@�����H@"�\A��
C���@����Q�@�\A���C���                                    Bxd��  �          @ə�@�(��=p�@z�A�C�ff@�(�����?޸RA�Q�C��                                    Bxd�H  �          @�Q�@����?�33A���C��H@����
?�z�Av�HC��                                    Bxd��  �          @Ǯ@�  ����@�A�
=C�W
@�  ���@	��A���C��                                    Bxd�  �          @�p�@��z�@8Q�A�=qC��
@���Q�@#33AÅC���                                    Bxd�:  �          @��@����8Q�@*=qA�z�C�@�����  @{A���C��H                                    Bxd�  
�          @�33@>{�(�@�p�BN��C���@>{�b�\@p  B33C��                                    Bxd�  �          @���@Dz��33@�33BR�HC��@Dz��^{@}p�B"(�C���                                    Bxd#,  �          @�{@h���   @��HB?Q�C���@h���U@n{B{C��
                                    Bxd1�  �          @ƸR@n{��
=@���B=
=C�J=@n{�P��@l(�B�C�b�                                    Bxd@x  �          @ȣ�@�����@i��B33C��@����N{@1�Aԣ�C�>�                                    BxdO  �          @�p�@����
@VffB�RC�E@���@��@!�A�C�˅                                    Bxd]�  �          @�{@�{����@n�RB{C�� @�{�*�H@B�\A�33C�,�                                    Bxdlj  �          @��@�zῨ��@�=qB&p�C��H@�z��#33@\(�B��C��3                                    Bxd{  �          @��@�����@~{B �C�aH@���%�@Tz�B��C�
=                                    Bxd��  �          @�p�@�������@xQ�B\)C�'�@����'
=@N{A�  C��                                    Bxd�\  �          @�G�@���Y��@^{Bz�C��@����z�@AG�A�{C�B�                                    Bxd�  �          @���@�녿
=q@Mp�B �C���@�녿��@7�A�C���                                    Bxd��  
�          @���@����@HQ�A���C�j=@����Q�@3�
A�G�C�!H                                    Bxd�N  �          @�G�@�Q쿃�
@K�A��HC�/\@�Q��   @+�A�G�C�                                      Bxd��  �          @�G�@��H���@l(�B33C���@��H�
�H@J=qA���C�XR                                    Bxd�  �          @���@�G���@�(�B-��C��)@�G��:�H@W�B\)C��                                    Bxd�@  �          @��R@~�R��Q�@qG�B(�C���@~�R�ff@K�B
=qC��q                                    Bxd��  �          @�p�@E�q�@UB	p�C���@E���?�
=A�z�C��H                                    Bxd�  �          @�(�@E�g
=@Z=qB�RC�O\@E����@33A��\C�1�                                    Bxd2  �          @�\)@^{��(�@1�Bz�C�S3@^{��
@�RA��C���                                    Bxd*�  �          @�\)@���@�
?���Ab{A���@���?Ǯ?�A�z�A�G�                                    Bxd9~  �          @�Q�@mp�@n�R=���?��B4�\@mp�@_\)?��Af�RB-
=                                    BxdH$  �          @�ff@dz�@��ÿ+���BA@dz�@���?.{@�  BA�                                    BxdV�  T          @���@0��@�G��xQ��,(�Be�@0��@��
>��H@�{Bg
=                                    Bxdep  �          @�G�@  @�G��333�Q�ByG�@  @�G�?@  A(�By�                                    Bxdt  �          @��@N�R@Q�@(Q�B�B�@N�R?��@L(�B+�A��
                                    Bxd��  �          @��
@g�@�@�
A�  B��@g�?�\)@0  B
��A��\                                    Bxd�b  �          @��@c33@��@�A��
B  @c33?�=q@B�\B�HA�ff                                    Bxd�  �          @�=q@^{>�  @N{B+
=@�@^{�Tz�@G�B$�C�K�                                    Bxd��  �          @��
@��?\)@�(�Bj��AL��@�ͿY��@��\Bfz�C�p�                                    Bxd�T  �          @��@�>u@�=qB|�@��@����H@���Bl=qC�s3                                    Bxd��  �          @�33@z�?\)@��\B~��Ap(�@z�k�@�Q�Bx{C�                                    Bxdڠ  �          @�(�@�R<��
@��RBs�
?��@�R��(�@��RB^�\C���                                    Bxd�F  �          @�@$z�>B�\@�ffBo@���@$zῧ�@�Q�B_�\C�}q                                    Bxd��  �          @�33@@�׿!G�@�33BZ��C�\@@����
@xQ�B;(�C��R                                    Bxd�  T          @��\@1G����H@��HB\��C�0�@1G��&ff@k�B0p�C�ff                                    Bxd8  �          @�@J�H���
@�{BS�\C�H�@J�H���R@z�HB@�C�p�                                    Bxd#�  �          @���@9������@�\)Bc�C��f@9������@��BH�RC��                                    Bxd2�  �          @�z�@^�R��\)@���BD33C��\@^�R��\)@l(�B/p�C�z�                                    BxdA*  �          @�33@�  ���
@EB��C�޸@�  ����@333B �\C�7
                                    BxdO�  �          @��@I���#�
@�  BU�HC��=@I������@�  BD33C���                                    Bxd^v  �          @�33@Y��>�{@��\BH�@�z�@Y������@|��B?�C�B�                                    Bxdm  �          @��H@U���  @�33BK=qC�޸@U��У�@qG�B5��C��
                                    Bxd{�  �          @�{@��>k�@ ��A��H@AG�@���#�
@(�A�p�C��                                    Bxd�h  =          @��@��u@   A�33C���@���  @G�A�{C�1�                                    Bxd�  �          @�Q�@r�\=u@`��B+\)?s33@r�\��33@Tz�B 
=C���                                    Bxd��  �          @�  @
=q>���@�  B���@���@
=q����@��BsG�C�H                                    Bxd�Z  �          @�33?�z�>W
=@�B��
A��?�z��  @�ffB���C��{                                    Bxd�   T          @�>�p�>#�
@��
B��A�p�>�p���\)@��B�aHC�]q                                    BxdӦ  T          @���?p��>.{@�z�B�A ��?p�׿��@���B��RC��                                    Bxd�L  �          @�Q�@�{>��H��(���\)@�33@�{?(��u�%@��                                    Bxd��  �          @�\)@���?   �   ���R@���@���?#�
��z��C�
@���                                    Bxd��  �          @�G�@�\)>�z�>��?Ǯ@Dz�@�\)>aG�>u@!�@                                    Bxd>  �          @��H@�\)?
=q�.{�߮@��@�\)?@  ��G����H@��                                    Bxd�  �          @�{@��?xQ쿣�
�Q��A"�\@��?��Ϳ\(��Q�A`��                                    Bxd+�  �          @�{@���?z�H�(���A   @���?��׾k��z�A8(�                                    Bxd:0  �          @���@��?:�H��R����@�{@��?fff���
�R�\A�\                                    BxdH�  �          @�@�33�#�
�+��ۅC��@�33>L�Ϳ#�
���@33                                    BxdW|  �          @��@�=q��=q�B�\����C���@�=q<��
�O\)��
>L��                                    Bxdf"  �          @�{@�p���=q�L���G�C���@�p��8Q쾏\)�6ffC�{                                    Bxdt�  �          @�{@�(��
=>�ff@�  C�  @�(��5>W
=@ffC�g�                                    Bxd�n  �          @�p�@�Q�Ǯ?�(�AG�
C���@�Q�L��?�G�A$  C��                                    Bxd�  
�          @�@��;��H?�{A��\C�e@��Ϳ��
?��AZffC��
                                    Bxd��  �          @��@�G���=q?���A���C���@�G��p��?�p�A��\C��3                                    Bxd�`  �          @���@�p�@z�?�\A��A�\)@�p�?��@��A�(�Az{                                    Bxd�  �          @�\)@��R?�@�A��RA���@��R?�  @#33A�G�A@                                      Bxd̬  �          @��@�z�@�
?ǮA�ffA�  @�z�?���@�A��HA���                                    Bxd�R  �          @�  @���@2�\@!�A��Bff@���?ٙ�@W
=B��A�\)                                    Bxd��  �          @�
=@���>�{@x��B,(�@���@��Ϳ�=q@p  B$C���                                    Bxd��  �          @�
=@Z=q�.{@�  BQQ�C�U�@Z=q�  @}p�B0\)C�Ff                                    BxdD  �          @���@8Q쿬��@��BW\)C�p�@8Q��5�@e�B&G�C�Ǯ                                    Bxd�  �          @�Q�@��
���@dz�B"z�C�9�@��
���@HQ�Bp�C�>�                                    Bxd$�  �          @���@��
�W
=@S33B  C�� @��
����@@  BC�5�                                    Bxd36  �          @���@vff��\)@z�HB6�C�z�@vff��p�@hQ�B%p�C�y�                                    BxdA�  �          @�{@?\)���R@���BM�C���@?\)�(Q�@U�B��C�XR                                    BxdP�  
�          @�z�@p�׾\)@s�
B5\)C���@p�׿�G�@`  B#33C��                                    Bxd_(  �          @�p�@Y�����@l(�B.33C�|)@Y���Fff@0��A���C���                                    Bxdm�  �          @�Q�?�p��\��@|(�B<�C��\?�p���z�@�HA�C�h�                                    Bxd|t  �          @�  ?#�
�i��@�G�B>
=C��R?#�
���
@(�A�
=C��q                                    Bxd�  �          @�{>�{�tz�@s�
B3G�C��f>�{��{@	��A��
C��3                                    Bxd��  �          @��?��
�>�R@�=qBP��C���?��
��Q�@,��A��HC��f                                    Bxd�f  �          @�Q�@Dz�
=q@\)BP  C��@Dz���H@aG�B/�C���                                    Bxd�  �          @��@��
�n{@G�BC��=@��
� ��@#�
A陚C�                                    BxdŲ  �          @��
@�=q���@=p�B�C�)@�=q��@�A�=qC�,�                                    Bxd�X  �          @���@���(�@>{B	\)C��3@���R@�
A�=qC��                                    Bxd��  �          @�p�@}p����H@Mp�B�
C�XR@}p��1G�@
=A�(�C��                                     Bxd�  �          @�@1����@\)B<C�O\@1��p  @3�
A�ffC�C�                                    Bxd J  "          @�p�@%��!�@~�RB?(�C���@%��u�@0��A�\)C��q                                    Bxd�  �          @��@!��(��@{�B<G�C��@!��z=q@*�HA���C�s3                                    Bxd�  �          @�(�@,(�� ��@p��B6�\C���@,(��n{@#�
A��C���                                    Bxd,<  �          @�33@mp��^�R@�=qB;Q�C�b�@mp��z�@\��B�C���                                    Bxd:�  �          @��@qG���ff@xQ�B0�
C�~�@qG��)��@G
=B=qC�q�                                    BxdI�  �          @��\@��33@C33B�
C��3@��B�\@�
A�{C�                                      BxdX.  �          @��@����R@�Aȣ�C��@���J�H?�p�AO
=C��\                                    Bxdf�  �          @��@��\�G�?�G�A'�C�'�@��\�  >8Q�?�{C��                                    Bxduz  �          @�p�@�ff��p�?�G�A$  C���@�ff�p�>B�\?�Q�C��                                     Bxd�   �          @�ff@��ÿ��H@�
A��C���@����ff?��\AM��C�q�                                    Bxd��  �          @�p�@��
��Q�?��HA���C�ٚ@��
�p�?s33A�\C�O\                                    Bxd�l  �          @�(�@��H�\?�ffA�\)C���@��H�?��A333C�ٚ                                    Bxd�  �          @��\@�p���ff@   A֏\C�j=@�p���{@	��A�ffC�L�                                    Bxd��  �          @�\)@�  ?z�@AG�B��@�=q@�  �(�@@��BQ�C�(�                                    Bxd�^  �          @���@�{�L��@N�RB�C���@�{��z�@:�HB =qC�33                                    Bxd�  �          @�  @]p���G�@���B>=qC��R@]p��,��@P  B
=C��                                    Bxd�  �          @��\@h�ÿ���@W
=B%�
C�Ǯ@h���G�@,��B �C�
=                                    Bxd�P  �          @��R@Y�����@���B@33C��
@Y���/\)@P  B�RC���                                    Bxd�  �          @�  @c�
�0  @<��B
=C�#�@c�
�j�H?�
=A�p�C�\                                    Bxd�  �          @�G�@Tz��I��@A�B  C�:�@Tz���=q?���A��C��
                                    Bxd%B  �          @�p�@1G��AG�@x��B.  C�Ff@1G�����@(�Ȁ\C�xR                                    Bxd3�  �          @�=q@
=�u@O\)B�
C��@
=��G�?\A~�\C�"�                                    BxdB�  �          @���@Z=q�Dz�@7
=A���C��@Z=q�{�?��HAz=qC�z�                                    BxdQ4  �          @�  @��\���H@VffB�HC�,�@��\�(Q�@!�A�ffC���                                    Bxd_�  �          @�  @^�R����@�G�B=  C�ff@^�R�333@L��B�RC��
                                    Bxdn�  �          @��@l(�����@�ffBB�\C�k�@l(���
=@p  B((�C�1�                                    Bxd}&  �          @�33@W�����@�BRp�C�n@W����R@~{B5�\C���                                    Bxd��  �          @��H@7
=��{@��B]{C�B�@7
=�AG�@k�B&(�C���                                    Bxd�r  �          @��\@;��p��@��
Ba�\C�q@;��)��@y��B2z�C��                                    Bxd�  �          @���@2�\?�  @���B\�\A���@2�\�\@���Bn
=C�%                                    Bxd��  �          @�Q�@R�\?Y��@~{BF  Ag�@R�\�B�\@\)BG(�C�y�                                    Bxd�d  �          @��@���\)?��@���C���@���=q����l��C���                                    Bxd�
  �          @��\?����
=��  �#�
C���?�����R�����z�C��q                                    Bxd�            @���@�  �G�@
�HA�=qC�:�@�  �<(�?�{AJffC��q                                   Bxd�V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxd �              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxd�   �          @��H@{���@~{B5G�C���@{��˅@h��B"C��                                    BxdH  T          @�G�@S33?\(�@�  BL��Ah��@S33�aG�@�  BL�C���                                    Bxd,�  
(          @���@`��?
=q@��\BCA��@`�׿���@}p�B<�C�E                                    Bxd;�  
�          @�  @aG��8Q�@��HBB�C�@ @aG��G�@^�RB�C��)                                    BxdJ:  "          @�  @e���@~{B:��C��H@e�"�\@N�RB33C�U�                                    BxdX�  
�          @�G�@j�H��Q�@`��B C��@j�H�J�H@��A֣�C���                                    Bxdg�  
(          @���@_\)�j=q@p�A���C��@_\)��\)?z�@���C��)                                    Bxdv,  
�          @���@q��W
=?�Q�A�p�C�+�@q��w
=>�G�@�ffC�9�                                    Bxd��  	�          @�
=@G
=�j=q@Q�A�33C�/\@G
=��ff?   @�\)C�@                                     Bxd�x  �          @��
@�H�n{@=p�BC�}q@�H���H?��RAVffC��                                     Bxd�  �          @��@-p��`��@<��BffC��
@-p���z�?��Ac33C��
                                    Bxd��  �          @��@(Q��w
=@�RA�z�C�'�@(Q���?�\@��C�]q                                    Bxd�j  "          @�@g
=�i��?�Ax(�C�XR@g
=�z=q�����\)C�W
                                    Bxd�  "          @��\?�  �XQ�@n{B4�\C��3?�  ���H@�
A��\C��                                    Bxdܶ  T          @��
?����XQ�@w�B:C�
?�����z�@(�A���C�AH                                    Bxd�\  �          @��
?��\�i��@fffB+��C���?��\��G�?�=qA�(�C�n                                    Bxd�  "          @���?�\)�J=q@�Q�BEp�C��)?�\)��  @=qAأ�C�w
                                    Bxd�  
�          @�Q쾊=q��33@S33B�\C�:ᾊ=q���?��\AW�C���                                    BxdN  
�          @�
==u�c33@�z�BEQ�C�xR=u��p�@ffA̸RC�W
                                    Bxd%�  	�          @������j�H@�Q�B=��C��þ����\)@(�A��
C���                                    Bxd4�  �          @�33>\)�Z=q@���BG�C�"�>\)��Q�@�AУ�C�Ф                                    BxdC@  �          @���>�G��W
=@���BH\)C��R>�G����R@ffA�
=C���                                    BxdQ�  �          @���?5�[�@z�HB@�C��q?5��
=@��AŮC�H�                                    Bxd`�  "          @���?�  �>{@���BP�
C�l�?�  ����@%A�z�C��                                    Bxdo2  �          @�=q@33�1�@y��B=  C��H@33��(�@(�A�Q�C��=                                    Bxd}�  �          @���@
=q�L(�@c33B*ffC�@
=q���?�Q�A�  C�1�                                    Bxd�~  T          @��H�#�
�@��@y��BQ�C�|)�#�
���H@A�=qC���                                    Bxd�$  �          @������[�@L(�BG�Cjz������?�G�A���Cq�                                    Bxd��  "          @�33?fff�[�@{�B?C�S3?fff���@(�A�33C�\)                                    Bxd�p  �          @��R?�(��hQ�@i��B)33C��\?�(�����?���A��C�ٚ                                    Bxd�  
�          @���@���q�@Z=qB��C��H@����33?���A��
C���                                    Bxdռ  �          @���?���q�@aG�B33C�O\?����z�?�z�A���C���                                    Bxd�b  "          @�=����Z�H@�z�BI�HC�Ф=�����=q@Q�A�  C��{                                    Bxd�  
�          @���u�A�@���BVG�C|\)�u����@+�A�z�C�                                      Bxd�  �          @���!G��I��@�Q�BS��C�Y��!G���(�@%�A�p�C��                                    BxdT  
�          @�=q�W
=�Q�@�=qBLp�C�,;W
=��p�@
=A֏\C��
                                    Bxd�  
Z          @��׿���@�G�Btp�Cdff���~{@n�RB!�RCu{                                    Bxd-�  �          @�G��	���\)@�Q�B]��Ce33�	����\)@S�
B�\Cs�                                    Bxd<F  T          @�������@�BW�\C`�\�����
@QG�B	p�Cok�                                    BxdJ�  �          @�=q��\�!G�@�p�Bh  Cj�ÿ�\���\@\(�B=qCw��                                    BxdY�  	�          @��
����H@��B]=qCb� ����ff@W�BCq�                                     Bxdh8  
�          @����z���
@���Bi33Cd{�z���@g
=B��Cs��                                    Bxdv�  
�          @����33@�G�Bh��CcL�����@g�BCs�                                    Bxd��  
Z          @�ff�
=�   @��HBZ�\Cb���
=��G�@W
=B	�Cq8R                                    Bxd�*  �          @��"�\�%@�BP�RCa�H�"�\���@J�HBz�Co}q                                    Bxd��  �          @�{�!G��#�
@��RBR��Caz��!G���G�@Mp�Bp�Co��                                    Bxd�v  "          @����R�(��@��BP��Cb�{��R��33@HQ�B �CpQ�                                    Bxd�  
�          @�33�=q�p�@�
=BW�Ca���=q���R@P��B�Cp:�                                    Bxd��  
�          @��\��\�<��@��\BO�Ck=q��\���H@:�HA�\)Cv                                      Bxd�h  T          @�녿�p��У�@�=qB�\Cc�R��p��mp�@�z�B8Q�Cx=q                                    Bxd�  �          @�\)���H�xQ�@�
=B�aHCZ�Ϳ��H�J�H@��BUG�Cy)                                    Bxd��  
          @��R��
=��  @�G�B��=Ck�ÿ�
=�s�
@��B633C|�=                                    Bxd 	Z  8          @�
=�����@���Bv��Co.�����\)@e�BG�C{��                                    Bxd    p          @�p�>Ǯ��z�@��B��qC��{>Ǯ�p��@��B>��C���                                    Bxd &�  
�          @�@Fff=u@�
=Bb�H?��
@Fff��Q�@��BF��C��
                                    Bxd 5L  
�          @�(�@AG���ff@��RBd=qC�˅@AG����@��\B:�\C��=                                    Bxd C�  8          @�(�@X�þu@�\)BS33C���@X����@}p�B2��C�f                                    Bxd R�  
�          @�\)@�Q�?�ff?�(�A��\A=�@�Q�=�G�@�RA��?��                                    Bxd a>  �          @�G�@�\)?\(�@�RA�=qA�@�\)�#�
@Q�A�ffC�\                                    Bxd o�  �          @�G�@��H?�@B�\B��@��
@��H�O\)@>�RB�\C��q                                    Bxd ~�  �          @�Q�@�(�?&ff@8��A�A Q�@�(��#�
@9��A�{C�3                                    Bxd �0  
�          @���@L��?�@�33B[(�A"�R@L�Ϳ�Q�@��BN
=C��H                                    Bxd ��  �          @��@S33?5@�z�BW�
ABff@S33��=q@�  BN��C���                                    Bxd �|  �          @��R@HQ�?�R@�
=B_��A333@HQ쿹��@���BRC��{                                    Bxd �"  �          @�@K�?Q�@��
BZQ�Ag�@K���p�@���BT
=C�o\                                    Bxd ��  "          @���@u�?��\@s33B/(�An�\@u��333@w�B3Q�C��                                    Bxd �n  �          @�G�@�33?�Q�@G
=BA��R@�33��@[�B{C��\                                    Bxd �  �          @�G�@���?�\)@]p�Bz�A��@��þ�  @mp�B*\)C�<)                                    Bxd �  
v          @�G�@�{?��@.{A��A��@�{?�@P��B\)@ۅ                                    Bxd!`  
p          @�Q�@�  ?��@S33B
=A���@�  >L��@p  B,�\@6ff                                    Bxd!  "          @���@���?��@9��A�p�A���@���>�ff@Z�HBz�@��                                    Bxd!�  T          @�  @y��@�@P��BQ�A�=q@y��>�(�@tz�B0��@ə�                                    Bxd!.R  "          @�ff@q�@�@VffB��A�@q�>Ǯ@y��B6�@�33                                    Bxd!<�  T          @���@5?E�@��Bb�HAr�\@5���R@�(�BZ��C�7
                                    Bxd!K�  "          @�p�@/\)?���@���B`(�A��@/\)�aG�@��RBd�RC�q                                    Bxd!ZD  
�          @���@
=?�@��B|�
AZ�\@
=�Ǯ@��HBiG�C�AH                                    Bxd!h�  �          @��H@8��?��@���BY�A�
=@8�ÿfff@�=qB[�C�aH                                    Bxd!w�  �          @�(�@C33?���@���BL��A�G�@C33�z�@��\BYp�C���                                    Bxd!�6  �          @�z�@N�R?8Q�@�
=BO�AI�@N�R��
=@��
BH��C��
                                    Bxd!��  T          @��\@R�\?�(�@{�B@�HA��H@R�\�(�@�=qBJG�C��=                                    Bxd!��  "          @���@g
=?�@fffB+G�A�{@g
=��\)@w
=B;33C��f                                    Bxd!�(  T          @���@��?�  @"�\A���A�z�@��>�  @<(�B�R@W�                                    Bxd!��  	`          @�33@���?�\)@#�
A��A���@���>�{@@��B	33@�{                                    Bxd!�t  �          @�(�@�G�?��@6ffB 33A��@�G�>\)@O\)B(�?��                                    Bxd!�  �          @��H@��
?�
=@C�
Bp�A��@��
�L��@XQ�B�C���                                    Bxd!��  T          @���?��H@�@�p�Bjp�BJ{?��H�u@�(�B�L�C�f                                    Bxd!�f  T          @�
=?���@��@�{Bk�BT�
?��ͼ��
@�B�
=C���                                    Bxd"
  �          @�G�?\(�?��
@�z�B���B�ff?\(��
=q@�B���C��R                                    Bxd"�  �          @��R?��@%�@�
=BZz�Bl�R?��>��@�p�B��HA��R                                    Bxd"'X  �          @�Q�@5@
=@l��B4G�B  @5?   @��Bb(�A\)                                    Bxd"5�  �          @�G�@$z�@(�@�  BG=qB!�H@$z�>W
=@��Bq�@���                                    Bxd"D�  �          @��@H��@�R@\��B#{B=q@H��?5@�{BQ�
ALz�                                    Bxd"SJ  �          @��\@P��@#33@UB�B  @P��?Q�@��
BK{Aa�                                    Bxd"a�  �          @���@^{@�R@R�\BffB��@^{?��@|(�B@�HA�                                    Bxd"p�  �          @���@r�\?��
@L(�B{A�
=@r�\>B�\@h��B/Q�@9��                                    Bxd"<  �          @��@hQ�?�ff@Q�B{A�
=@hQ�>.{@o\)B7Q�@(��                                    Bxd"��  �          @�G�@�Q�?�@*�HA�ffA�  @�Q�>�G�@L��B��@�Q�                                    Bxd"��  �          @�Q�@��\?޸R@��A�=qA���@��\>��H@:�HB�@���                                    Bxd"�.  �          @���@�{?��
@*=qA�=qA�z�@�{>�
=@J�HB  @���                                    Bxd"��  T          @���@��?�{@B�\B�A�{@����@U�B
=C�3                                    Bxd"�z  �          @���@|��?�{@L��B�HA�  @|�;L��@^{B%(�C���                                    Bxd"�   �          @�G�@hQ�?�\)@_\)B%
=A�ff@hQ����@vffB:�C�>�                                    Bxd"��  �          @��\@y��@G�@@  B	p�A��H@y��>��@e�B)�\@�{                                    Bxd"�l  �          @��\@z=q@�@8Q�B�A��@z=q?
=q@^�RB%p�@�=q                                    Bxd#  �          @���@�G�?�?�{ApQ�A�p�@�G�?c�
?��HA�G�A'�                                    Bxd#�  �          @�\)@��\?�\)@)��A�
=A��@��\>�=q@FffB�\@s�
                                    Bxd# ^  �          @�
=@R�\@  @S33B��B	Q�@R�\?
=q@}p�BG�\A��                                    Bxd#/  �          @��@s33@�
@:�HB
=A�@s33?
=q@a�B*ffA�                                    Bxd#=�  �          @�Q�@:=q@G�@l(�B4�B��@:=q>\@��B_=q@�R                                    Bxd#LP  �          @���@@��@"�\@^{B%\)B ��@@��?:�H@��BWffAY�                                    Bxd#Z�  �          @�z�@���?�
=?ǮA���A���@���?L��@	��A�
=A%G�                                    Bxd#i�  �          @�(�@�G�?�?333@��A��@�G�?�
=?�\)Ax(�A]��                                    Bxd#xB  �          @��
@��
?���?��@�ffA}G�@��
?}p�?�{AI��A7�                                    Bxd#��  �          @��@�?Ǯ?�  Ac�
A��@�?Tz�?���A��A!��                                    Bxd#��  �          @�33@�?�
=?���AH  A�{@�?�G�?޸RA�ABff                                    Bxd#�4  
�          @�z�@�p�?�{?���A�33A��\@�p�?Tz�@\)A�  A)                                    Bxd#��  �          @���@��@�?���A��A��@��?�G�@,��B �
A[
=                                    Bxd#��  �          @��@�z�@��@A�\)A�z�@�z�?xQ�@5�B��ARff                                    Bxd#�&  �          @�\)@r�\@p�@0��B �
A��@r�\?:�H@]p�B'��A-�                                    Bxd#��  �          @���@_\)@   @Z=qB!=qA��H@_\)>�  @|��BA�H@��H                                    Bxd#�r  �          @��\@Tz�?�Q�@hQ�B-�A�  @Tz�=�Q�@��
BL\)?�ff                                    Bxd#�  �          @�=q@l(�?�
=@g�B,Q�A��@l(��
=@p��B4��C�}q                                    Bxd$
�  �          @���@c33?h��@n{B5�RAep�@c33�^�R@n�RB6=qC��                                    Bxd$d  �          @���@mp�?E�@j�HB0(�A;�
@mp��xQ�@g�B-Q�C���                                    Bxd$(
  �          @�  @c33@�@6ffB
=B�H@c33?Q�@g�B3�APz�                                    Bxd$6�  �          @�ff@U�@z�@J=qB�RB(�@U�?!G�@w�BC
=A+�                                    Bxd$EV  �          @�@`��?�z�@U�B%z�A��@`�׾�  @g
=B6�HC��\                                    Bxd$S�  �          @��R@B�\?8Q�@�p�BT�
AVff@B�\��  @���BK��C�Ф                                    Bxd$b�  �          @��H@.{>�{@�  Bd�
@�(�@.{�У�@|(�BL�HC��=                                    Bxd$qH  T          @��@p��L��@�ffB~ffC�e@p���
@|(�BR{C�xR                                    Bxd$�  �          @�?��
��@�=qB�W
C��?��
���@�Q�B[Q�C��                                    Bxd$��  �          @�  ?�Q���@�z�B�{C�\?�Q��&ff@xQ�BM��C���                                    Bxd$�:  �          @�
=@/\)=��
@���B_p�?ٙ�@/\)��\@h��B@p�C��3                                    Bxd$��  �          @�  @�Q�?G�@{AمA&{@�Q쾣�
@�A�G�C���                                    Bxd$��  �          @���@l��?�@VffB'G�Az�@l�Ϳ��\@O\)B �RC�K�                                    Bxd$�,  �          @�p�@e�?G�@QG�B&ADz�@e��L��@P��B&�C���                                    Bxd$��  �          @�ff@�  ?\(�@/\)BG�AC33@�  ��@5�Bz�C��                                    Bxd$�x  8          @�  @n�R?�=q@I��BffA�33@n�R��@S33B%{C�P�                                    Bxd$�  �          @���@g�>�{@Q�B((�@�G�@g�����@Dz�B=qC���                                    Bxd%�  �          @�  @U?\)@g
=B;p�A��@U��z�@^{B1�C�u�                                    Bxd%j  
�          @�G�@^{?(�@e�B5�
Aff@^{����@]p�B.(�C�0�                                    Bxd%!  
�          @�
=@^{?���@W�B+��A��H@^{���@_\)B2�C�)                                    Bxd%/�  
�          @��R@o\)?W
=@FffB�AJ�R@o\)�+�@H��B�C��f                                    Bxd%>\  �          @�@q�?�  @=p�B�Al(�@q녾�@EB(�C�h�                                    Bxd%M  
�          @�ff@p��?��\@A�B�Ar�H@p�׾��H@J=qBffC�K�                                    Bxd%[�  
Z          @�z�@o\)?u@>�RB  Aep�@o\)��@EBG�C�H                                    Bxd%jN  
�          @�ff@tz�?�{@4z�B33A�{@tzᾞ�R@@��B�RC��{                                    Bxd%x�  �          @��@���?�z�@!G�A���A���@���>��R@@  B�\@�z�                                    Bxd%��  "          @�=q@�\)?��@Q�A�\)A�G�@�\)?��@.{B��A�                                    Bxd%�@  
Z          @�(�@��?�p�@z�A���A�G�@��?z�@(��A�@�=q                                    Bxd%��  "          @�(�@c33?Tz�@L(�B$��AS
=@c33�=p�@N{B&p�C�#�                                    Bxd%��  
Z          @�Q�@AG���@hQ�BG\)C�@AG����@>�RB=qC�P�                                    Bxd%�2  
�          @�ff@g�?+�@U�B(�A'�
@g��p��@QG�B$ffC���                                    Bxd%��  "          @��H@b�\?��\@G�B!
=A��\@b�\�
=q@O\)B(z�C��                                    Bxd%�~  �          @�{@n{?B�\@HQ�B�A8(�@n{�G�@G�B�
C�R                                    Bxd%�$  
�          @�G�@\)?n{@9��B�HAQ@\)��@@  B��C�Q�                                    Bxd%��  �          @�(�@8��?G�@��BY{Ar�\@8�ÿ�p�@���BP��C�p�                                    Bxd&p  T          @�33@2�\?333@�ffB^�\Ab{@2�\��=q@���BRC�Ff                                    Bxd&  	�          @�p�@dz�?��@c33B-��A�@dz�!G�@j�HB5ffC��R                                    Bxd&(�  �          @��@��?�@1�B�A{�@����  @@��BG�C�AH                                    Bxd&7b  
�          @��
@��\?��\@333B�HA��@��\�8Q�@Dz�B��C��H                                    Bxd&F  �          @�(�@\)?���@7�B�A�=q@\)�#�
@Mp�Bz�C��                                    Bxd&T�  
�          @�33@�  ?�  @>{B
=AaG�@�  ���H@EB�C�~�                                    Bxd&cT  q          @��\@|��?Q�@Dz�B�A:�\@|�Ϳ5@FffB��C��                                    Bxd&q�  �          @��@K�?�p�@H��B*z�A���@K���p�@VffB9�C���                                    Bxd&��  T          @�G�@3�
?�Q�@VffB>�\A�  @3�
�   @aG�BKC���                                    Bxd&�F  
Z          @�=q@L��?�  @QG�B1�A�=q@L�Ϳ#�
@W
=B733C�T{                                    Bxd&��  	�          @�
=@xQ�?��@\)A�\)A�z�@xQ�>\)@'
=B��@�
                                    Bxd&��  �          @��@��?�\)?��A�
=A���@��?��@�A�  @�Q�                                    Bxd&�8  T          @�\)@�  ?�  @	��A�{Aap�@�  ��G�@�A��HC�.                                    Bxd&��  
�          @�33@�z�<��
?�
=A�  >aG�@�z�c�
?�(�A�Q�C��3                                    Bxd&؄  �          @���@���?�
=?�33A�  A�ff@���>�=q@ ��A��@tz�                                    Bxd&�*  
�          @��\@���?��H@z�Aڣ�Aυ@���?&ff@=p�B�A�\                                    Bxd&��  �          @��@��?���@(��A�ffAl(�@����=q@6ffB	\)C�*=                                    Bxd'v  �          @��@s33?��@��A��A�Q�@s33>�
=@-p�BG�@�                                      Bxd'  
�          @�z�@~{?�ff@{A�\)A��@~{�#�
@333B��C���                                    Bxd'!�  
�          @��@s33?�  @{A��A���@s33�L��@1G�BG�C��R                                    Bxd'0h  T          @�=q@|(�?�z�?�  A�
=A��\@|(�?B�\@ffA�p�A.�\                                    Bxd'?  �          @tz὏\)?��<�@*�HB����\)>��>�\)A�\)B��)                                    Bxd'M�  T          @�\)���
��녿��R���
CC�����
��z���H�ҏ\C7u�                                    Bxd'\Z  
�          @�
=���ÿ�\��R��  CIG����þ����AG��33C8��                                    Bxd'k   
�          @��H������Q��z���(�CGW
���������5��(�C8��                                    Bxd'y�  	`          @�G���p����ff��Q�CI}q��p����<(����
C:^�                                    Bxd'�L  
�          @��H������������CK���Ϳ5�E���
C<E                                    Bxd'��  �          @�p���(����
�H��  CI����(��.{�4z���ffC;�H                                    Bxd'��  �          @�����������z����HCF��������
�333��{C7��                                    Bxd'�>  
�          @�G���ff��{�p���
=CI�=��ff����4z���ffC;:�                                    Bxd'��  
Z          @�  ��z�˅�>{��G�CF���z�#�
�W����C4{                                    Bxd'ъ  T          @�
=���R��  �.�R�㙚CHh����R����N�R�	�C733                                    Bxd'�0  �          @�������'��ܣ�CI33����p��J�H�{C8u�                                    Bxd'��  
�          @�(���(��G��{��{CK����(���R�HQ��G�C;��                                    Bxd'�|  "          @����j�H�xQ�>�@�(�Cb���j�H�h�ÿ�
=�w\)C`�q                                    Bxd("  �          @�=q���
�����   �˅CA5����
=��1G���p�C2��                                    Bxd(�  T          @��
����  ��\����CD(�����  �.{�݅C6��                                    Bxd()n  T          @�z����\��p��p�����CIG����\�333�8�����HC;��                                    Bxd(8  �          @�����Ϳ�=q�����  CG�
���Ϳ���7
=��\)C:
=                                    Bxd(F�  T          @�  ���\��{�(����
CD���\�\�,(��ծC8�                                    Bxd(U`  �          @����\)�����G���{CDE��\)��ff�"�\��=qC8��                                    Bxd(d  �          @�z����H����z����HCD�����H���{���
C9Ǯ                                    Bxd(r�  "          @�=q��  ��=q��\��  CF^���  �E������RC;�R                                    Bxd(�R  
�          @�\)��������
��G�CBE���u�{��G�C6�\                                    Bxd(��  
�          @�z������33�����q�CF�3����k���
��z�C=G�                                    Bxd(��  "          @������׿��
� ����{CA����׾.{����  C5�=                                    Bxd(�D  �          @�\)���׿��׿���33CB����׾�Q��  ����C7��                                    Bxd(��  "          @�ff���\���xQ����CL�����\�˅������33CEff                                    Bxd(ʐ  
�          @�����=q�Q�(���أ�CM!H��=q���
���H���CGT{                                    Bxd(�6  
�          @�=q��G���\)���RCL�)��G����R���R�Mp�CI��                                    Bxd(��  	�          @������!�>���@X��CO+����Q�h���\)CM�=                                    Bxd(��  "          @�G������C�
=u?z�CT�=�����.{��33�dQ�CQ�\                                    Bxd)(  �          @�����ff�K���\)�Q�CT����ff�	���-p���p�CKs3                                    Bxd)�  �          @\��33�P  ��ff��  CT�=��33�)����Q���\)COu�                                    Bxd)"t  "          @�����QG�>\)?�ffCS������<(���
=�VffCQW
                                    Bxd)1  T          @������XQ�?�@�ffCU=q�����N{����)G�CT
=                                    Bxd)?�  T          @�������h��?@  @�(�CYc������c�
�����#\)CX�                                    Bxd)Nf  
Z          @�=q����~{>��@�
=C]\����mp���(��`��C[33                                    Bxd)]  
�          @�(��|�����?�ffAM��Cb)�|����\)�O\)��\)Cb��                                    Bxd)k�  
Z          @�ff�z=q���?
=@�Cdp��z=q�����=q�x(�Cb                                    Bxd)zX  �          @�  �z�H����>#�
?\Ce!H�z�H����� ����\)Cb                                      Bxd)��  T          @�Q�������>�{@R�\Ccc������G���ff��p�C`�f                                    Bxd)��  �          @�
=��{�/\)?�G�Az�CO�
��{�8�þ�
=��Q�CQ�                                    Bxd)�J  
�          @�
=�����z�?��Ao�
CK�������1G�>�  @�HCO�q                                    Bxd)��  
�          @�G����H���H?�(�A��\CI����H�*=q?J=q@�33CO��                                    Bxd)Ö  T          @�p�����
=q?�A�33CK����1�?z�@��CQz�                                    Bxd)�<  
�          @�33��33���@�
A�G�CE����33�p�?�ffAVffCNٚ                                    Bxd)��  �          @�������mp�?z�HA�
C[k������n�R�h�����C[�=                                    Bxd)�  
�          @�Q���33�Vff?�z�A2�RCV����33�`  �z�����CW�)                                    Bxd)�.  T          @������N{?�p�A<z�CU33���Z�H����z�CV                                    Bxd*�  "          @�����(��>�R?�(�A:{CR#���(��L�;�33�S�
CS��                                    Bxd*z  
�          @����{�333?�{AP  CPaH��{�G
=�\)����CR��                                    Bxd**   "          @������H�7�?�(�Ab�\CQ\)���H�N{��\)�0��CTQ�                                    Bxd*8�  
�          @�\)���H�1�?���A`z�CP�����H�H�ýu��CS�f                                    Bxd*Gl  
�          @�{�����I��?ǮAt  CUs3�����`�׾\)����CX^�                                    Bxd*V  T          @�ff��ff�QG�?ǮAs�CVǮ��ff�g
=�B�\��{CY�=                                    Bxd*d�  "          @����  �L��?���A2{CU�q��  �Vff�
=q��=qCW5�                                    Bxd*s^  �          @�����z��N�R��=q�7�CXaH��z��-p��������CS��                                    Bxd*�  
�          @�=q��Q��E�����,  CVff��Q��p������\)CN(�                                    Bxd*��  "          @�������C33������ffCU�������\)������CP�=                                    Bxd*�P  T          @�{��p��L(�?��@��\CVW
��p��Dz῅��(��CUJ=                                    Bxd*��  �          @�  ����@  ?�R@�{CS�q����;��c�
�33CS#�                                    Bxd*��  �          @�\)��G��E>�ff@��RCT����G��;�����/\)CSk�                                    Bxd*�B  
�          @�\)��  �H��?��@�Q�CUz���  �B�\�z�H��CT��                                    Bxd*��  �          @�����R�K�?G�@��HCV\���R�J�H�W
=�=qCU�3                                    Bxd*�  T          @�\)����C33?\)@��CTW
����<(��xQ���CSs3                                    Bxd*�4  T          @�����  �J�H?=p�@��CU�q��  �H�ÿ^�R�
ffCUxR                                    Bxd+�  �          @��R����P��?@  @���CW:�����N{�fff�=qCV��                                    Bxd+�  �          @�Q������J�H?�\@��
CU�������A녿����+
=CTff                                    Bxd+#&  "          @�p��+��7�@^�RB&Q�Cb�f�+���?�
=A�CmW
                                    Bxd+1�  T          @�p���
=�&ff@��\BT��CiaH��
=��33@#33A�Cv{                                    Bxd+@r  
�          @�����
�p�@�z�BW\)Ce����
��Q�@+�A��Ct(�                                    Bxd+O  "          @�G��*�H�E@n{B)\)Ce��*�H���?��A���Co@                                     Bxd+]�  T          @�\)�>�R�:=q@^�RB�\C`G��>�R��
=?�z�A��RCj                                    Bxd+ld  
�          @�z��Dz��=p�@FffB�C`�Dz����H?�ffAd��Ci�                                    Bxd+{
  �          @�������@��B���CT����L(�@j=qB6
=Crff                                    Bxd+��  "          @�{�   ��Q�@��HBW�
CYǮ�   �q�@8Q�B�HCl��                                    Bxd+�V  �          @������Tz�@`��B&�Cm������33?�p�A���CuG�                                    Bxd+��  T          @�(���
�g
=@N�RB��Cmff��
���R?���A=Cs�                                    Bxd+��  �          @�33�7
=�k�@#�
A��Ch.�7
=���R>�@�p�CmT{                                    Bxd+�H  �          @���Dz��fff@�HA���Ce���Dz���=q>���@�=qCj��                                    Bxd+��  T          @��8Q��<��@@��BffCa� �8Q�����?�(�A]Cj�                                     Bxd+�  "          @�{�R�\�  @QG�Bp�CVs3�R�\�c33?��A�Q�Cc+�                                    Bxd+�:  
�          @�z��`  �{@0��B33CWB��`  �`  ?��RA`��C`�R                                    Bxd+��            @����K���@K�B�
CYn�K��j=q?�\)A�(�Ce�                                    Bxd,�  
�          @���^{� ��@N�RB\)CW޸�^{�p  ?��A�{Cc@                                     Bxd,,  �          @����XQ��"�\@I��B�HCX�R�XQ��o\)?�ffA��\Cc�                                    Bxd,*�  
�          @�
=�>{��@)��B��CS�>{�8Q�?�Q�A���C`&f                                    Bxd,9x  
(          @�G��}p��/\)?��A��HCV�3�}p��DzὸQ쿊=qCY�\                                    Bxd,H  
Z          @����p��333?^�RA��CU����p��8Q�z���Q�CV�)                                    Bxd,V�  �          @�{�\)�0��?�\A�{CV��\)�QG�>k�@!G�C[W
                                    Bxd,ej  
(          @�
=�{��\)@$z�A�z�CQ���{��L(�?�
=AQp�C[�                                    Bxd,t  
�          @��R�vff�"�\@  A�=qCUaH�vff�S�
?B�\A(�C\�f                                    Bxd,��  �          @�p������=q@p�A�p�CR.�����S33?�  A+�
CZ}q                                    Bxd,�\  
Z          @�
=��z��5�?�\)A��
CT�=��z��P��=u?.{CX�{                                    Bxd,�  "          @�������U�?n{A��CY\)����W
=�G��G�CY��                                    Bxd,��  T          @����=q�W�?c�
A(�CY�3��=q�X�ÿTz��	�CZ{                                    Bxd,�N  T          @��������W
=?Tz�A	CZ�����U�c�
��
CY��                                    Bxd,��  
�          @�����H�Vff?h��A�CY�����H�XQ�O\)�ffCY�)                                    Bxd,ښ  	�          @������X��?(�@�G�CY������P�׿���4��CX��                                    Bxd,�@  
�          @������H�W�>��R@L��CY�����H�Fff�����bffCW��                                    Bxd,��  	�          @�����hQ����  C\����=p������p�CWL�                                    Bxd-�            @�Q���=q�hQ�J=q�  C]����=q�3�
�(���CV�
                                    Bxd-2  T          @�Q��\)�i������.�RC^}q�\)�-p��*�H��CV(�                                    Bxd-#�  �          @�������\�Ϳ   ��G�C\�����2�\����\CV0�                                    Bxd-2~  "          @�ff��G��Mp��n{�ffCX�
��G����
=��CP�                                    Bxd-A$  �          @�ff���*=q�z���{CR����33���
����CLٚ                                    Bxd-O�  "          @����ff�5�>��R@aG�CV���ff�(�ÿ�=q�EG�CT�                                    Bxd-^p  T          @�p�����?\)>�33@w
=CW�3����2�\�����K33CUٚ                                    Bxd-m  �          @�ff�{��S�
?=p�A
=C\{�{��P�׿n{�&{C[�f                                    Bxd-{�  �          @�\)��Q��I��?���AHz�CZ.��Q��S�
��\���C[��                                    Bxd-�b  "          @��{��R�\?8Q�A ��C[��{��N�R�p���'\)C[s3                                    Bxd-�  T          @�����,��@=p�B�RCe!H��r�\?��A��HCnW
                                    Bxd-��  T          @��\�������@H��B>{Cl33�����fff?�{A�{Cv�                                    Bxd-�T  �          @�녿�  ��{@q�BsQ�Cq�
��  �\(�@�Bz�C}޸                                    Bxd-��  !          @|�Ϳ�����H@QG�Ba�Cg�H����C�
@33A�Q�CvaH                                    Bxd-Ӡ  
�          @�G��!���@:�HB p�C_#��!��^�R?�
=A�
=Ci��                                    Bxd-�F  
�          @����-p���@>{B%
=CXǮ�-p��L��?��A���Ce��                                    Bxd-��  T          @���.{�,��@"�\B{C`��.{�e?p��A<Q�Ch��                                    Bxd-��  �          @��H�1��6ff@�A�{Ca���1��e?#�
@�
=ChB�                                    Bxd.8  
(          @�ff�@���AG�?�Q�A�{Ca
=�@���e�>��@O\)Ce��                                    Bxd.�  
�          @�Q��>�R�L(�?�ffA�\)Cb���>�R�j�H=u?.{Cf޸                                    Bxd.+�  "          @���,���8��@,(�B	  Cb޸�,���tz�?z�HA>{Cj��                                    Bxd.:*  �          @�ff�B�\�[�?�A��HCdc��B�\�x�ü��
��=qCg�3                                    Bxd.H�  
�          @���QG��-p�@�A�{C[���QG��^{?5A�Cb��                                    Bxd.Wv  �          @��H�C33�c33?��RAm�CeQ��C33�mp��(���
=Cf��                                    Bxd.f  �          @�33�L(��dz�?8Q�A��CdB��L(��^�R��=q�MG�Cc��                                    Bxd.t�  	�          @�G��Z�H�j�H=#�
>�ffCc{�Z�H�P  ��(���ffC_�=                                    Bxd.�h  �          @��H�x���R�\=L��?�C\33�x���:=q���
���
CX��                                    Bxd.�  
Z          @�G��r�\�K�<�>�{C\��r�\�3�
���R����CX�{                                    Bxd.��  	`          @�G���\)�\)@e�BR�Cu�
��\)�y��?���A�C}�q                                    Bxd.�Z  
(          @��;aG���p�@vffBz��C��H�aG��dz�@(�B	  C�E                                    