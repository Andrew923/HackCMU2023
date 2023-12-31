CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230323000000_e20230323235959_p20230324021610_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-24T02:16:10.903Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-23T00:00:00.000Z   time_coverage_end         2023-03-23T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxp�>@  
�          @���U@'
=��
=�yC���U@5>���@�ffC�H                                    Bxp�L�  �          @��R�\��@'
=����[�
C��\��@1G�?   @�\)CE                                    Bxp�[�  �          @�  �b�\@#�
����Z�HC��b�\@.�R>��@���CW
                                    Bxp�j2  T          @�  �j=q@��xQ��G�Cn�j=q@$z�>�@��
C��                                    Bxp�x�  �          @����dz�@{�+���CO\�dz�@(�?@  A�C�\                                    Bxp��~  
�          @����^�R@p��u�N{C���^�R@p�?�\)ArffC��                                    Bxp��$  �          @���e@�
�
=q��
=C33�e@\)?G�A'
=C�q                                    Bxp���  T          @��
�hQ�@{�p���F�HC���hQ�@Q�>Ǯ@��C�
                                    Bxp��p  T          @�33�j=q?�Q쿦ff���C{�j=q@���\)�aG�Cp�                                    Bxp��  �          @�33�h��?����=q���Cٚ�h��@ff�\��\)C�                                    Bxp�м  
Z          @�=q�l(�?޸R�������
C�{�l(�@
�H�B�\�%�C��                                    Bxp��b  �          @xQ��[�?�(��\���RCJ=�[�?�z�?��AffC�                                    Bxp��  
�          @w
=�Z�H?������
�HCc��Z�H?˅?Tz�AF�HC\                                    Bxp���  
�          @xQ��`��?�\)=�?�=qC(��`��?��?xQ�Ah��C�\                                    Bxp�T  �          @x���W�?���\)���CaH�W�?�z�?uAe��C�=                                    Bxp��  T          @��H�]p�@�
�(����C5��]p�@ff?
=q@��C�                                    Bxp�(�  T          @�ff�\��?�녿�����CE�\��@
=�B�\�(��C��                                    Bxp�7F  "          @����S33?�\��\��ffC���S33@(��\)���RC��                                    Bxp�E�  
�          @���@��?�p��	����{C��@��@4z�B�\�'33C�f                                    Bxp�T�  
�          @�  �J=q@  �ٙ�����C���J=q@3�
��  �U�C
ff                                    Bxp�c8  T          @���j=q?�33��ff���
C"���j=q?�z���H��\)C��                                    Bxp�q�  �          @��R��Q�?�녾������RC$���Q�?��>���@��C$!H                                    Bxp���  �          @�33�G
=@'����Ϳ���C��G
=@G�?��A�=qC�f                                    Bxp��*  �          @���5@G�>���@��
C8R�5@\)?�A�C
�                                    Bxp���  T          @��\�8��@I��?
=@�(�Cu��8��@
=@�A�ffC                                    Bxp��v  T          @�33�@��@Dz�?(�A��Cp��@��@�@	��A�G�C��                                    Bxp��  �          @��H�@��@A�?@  A�C��@��@
�H@  A��C.                                    Bxp���  T          @����7�@HQ�?&ffA(�C���7�@33@{A�=qC@                                     Bxp��h  T          @����1�@J=q>�{@��\CQ��1�@   ?��HAݙ�C
                                    Bxp��  T          @���:=q@?\)?!G�A��C&f�:=q@p�@�A�G�C�\                                    Bxp���  �          @�\)�@��@<��>�
=@�Q�C�{�@��@G�?�A�(�C�f                                    Bxp�Z  T          @���L(�@4z�?#�
A  C
��L(�@33@�AᙚC5�                                    Bxp�   T          @�
=�XQ�@ff?W
=A8��C+��XQ�?�ff?��HA�=qCaH                                    Bxp�!�  
�          @���Y��@�?�A�ffC���Y��?�33@A�C!W
                                    Bxp�0L  �          @�\)�Y��@
=q?��
A�z�C�)�Y��?�z�@�RA�  C!8R                                    Bxp�>�  �          @����Q�?��?���A�=qC{�Q�?fff@�A��C$�H                                    Bxp�M�  
�          @u�Fff?��
?�33A���C
�Fff?E�@Q�B
=C&�                                    Bxp�\>  
�          @����e@z�?���AmG�C�q�e?�
=@�A�C!�\                                    Bxp�j�  "          @����c33@p�?aG�A=��C)�c33?�z�?�z�Aԣ�CO\                                    Bxp�y�  "          @����c�
@
=q?aG�A?�CǮ�c�
?�\)?��A���C                                      Bxp��0  "          @��
�X��@(�?=p�A&ffC0��X��?�(�?��
A�{C�)                                    Bxp���  "          @����Q�@p�?��@�
=C��Q�?˅?��A�p�C33                                    Bxp��|  T          @��
�Y��@
=q?#�
Az�C� �Y��?�  ?�
=A�33C(�                                    Bxp��"  T          @�G��fff@Q�?n{AHz�Ch��fff?���?�z�A��C�H                                    Bxp���  
�          @��H�g�@�?#�
A��C��g�?���?�  A��RC#�                                    Bxp��n  �          @��H�i��@  ?�@��C^��i��?У�?��A��C��                                    Bxp��  T          @��H�fff@Q�>�z�@y��C���fff?�\)?�  A�z�C�{                                    Bxp��  "          @����Z=q@"�\>��@�
CaH�Z=q@�?��HA���C��                                    Bxp��`  "          @�33�L��@p�?   @��Cu��L��?�=q?�(�A��
C@                                     Bxp�  "          @�Q��N{@33>�\)@\)Cff�N{?���?���A���C��                                    Bxp��  "          @����7
=@,��?Tz�A>�RC�
�7
=?�\)@Q�A�p�C޸                                    Bxp�)R  T          @��
�L(�@!�?�@��C�{�L(�?�\)?��
A�\)C�=                                    Bxp�7�  "          @y���@  @�H>���@��RC{�@  ?��?ǮA��C�{                                    Bxp�F�  �          @w
=�E�@33���
���
C=q�E�?�(�?�Q�A�  Cc�                                    Bxp�UD  �          @{��XQ�?���<�>�C�q�XQ�?��?�ffAy�C
=                                    Bxp�c�  �          @{��U@ ��=�Q�?�{C��U?�?���A��C}q                                    Bxp�r�  T          @|���Vff@�
=L��?333C\)�Vff?�p�?�\)A�z�C�H                                    Bxp��6  T          @s33�P��?�z�\)�(�C���P��?ٙ�?^�RAU�Cu�                                    Bxp���  "          @���dz�?��R>��
@�C�)�dz�?��
?��A�33C�=                                    Bxp���  "          @�p��c33@�\?5A��C��c33?�\)?�A�C�)                                    Bxp��(  �          @�(���G�@��=���?�
=CǮ��G�?��?�G�A|��C{                                    Bxp���  
(          @����w�@
=���Ϳ��C�H�w�@z�?��AiC�f                                    Bxp��t  
�          @�
=�w�@+���{���
CQ��w�@p�?�{AYG�C��                                    Bxp��  �          @���x��@8Q������
=Cs3�x��@*�H?�z�A]p�C�=                                    Bxp���  �          @���vff@.{��ff��\)C���vff@#�
?��
AIp�Cc�                                    Bxp��f  �          @�33�~{@*�H�:�H�
�RC
�~{@*=q?E�A�C0�                                    Bxp�  
�          @�Q��w�@+��(����C@ �w�@(Q�?W
=A"{C                                    Bxp��  �          @����x��@,(��B�\���C\)�x��@,(�?@  Az�CW
                                    Bxp�"X  �          @�{�|��@�����Cp��|��@�?Y��A((�C�=                                    Bxp�0�            @������@�׾�{��ffC�\����@
=?aG�A/33Cz�                                    Bxp�?�  \          @�Q��w�@�R�����vffC�w�@�
?fffA9�C�                                    Bxp�NJ  �          @�
=�s�
@z�u�G�C�H�s�
@
=?�  ANffC                                      Bxp�\�  T          @�{�fff@�H�Tz��-�C�fff@   ?��@�(�C+�                                    Bxp�k�  �          @|(��(��@����Q����C	�=�(��@333<#�
>\)CaH                                    Bxp�z<  �          @\)�7�@�R����p�C+��7�@.{>�\)@��Cp�                                    Bxp���  �          @����=p�@#33���
�k�
C.�=p�@.{>�G�@���C	k�                                    Bxp���  �          @��H�@��@   �������CE�@��@0��>�=q@q�C	}q                                    Bxp��.  �          @~�R�333@"�\���H��{C	� �333@333>�\)@�  C�R                                    Bxp���  �          @��\�5�@���Q��£�CE�5�@<�;B�\�'�C�R                                    Bxp��z  �          @����Vff@�Ϳ����s
=C�Vff@+�>���@���CJ=                                    Bxp��   �          @�
=�j=q@��k��?�C\�j=q@ ��>�(�@���C�{                                    Bxp���  
�          @�{�g
=@\)�(�����Cp��g
=@{?8Q�A�C��                                    Bxp��l  �          @�{�g
=@\)�&ff�
=Ck��g
=@{?8Q�A�\C��                                    Bxp��  �          @�z��s�
@ �׿J=q�{C�H�s�
@#�
?(�@�=qC�                                    Bxp��  �          @����w
=@   �(���\C)�w
=@p�?B�\A�HC�=                                    Bxp�^  �          @���s�
@!G��������RC� �s�
@�?uA@��C&f                                    Bxp�*  �          @���vff@(��B�\��C���vff@(�?��A]��CY�                                    Bxp�8�  �          @��\�x��@���W
=�(��CL��x��@
�H?�ffAT��Cٚ                                    Bxp�GP  �          @��
�n�R@-p�=L��?z�C���n�R@33?�
=A���CL�                                    Bxp�U�  
�          @��e@G�����z�C���e?�p�?�RA	�C(�                                    Bxp�d�  �          @u�Tz�?��=�G�?���Cs3�Tz�?Ǯ?�ffA�ffC�\                                    Bxp�sB  �          @tz�k�?\@:=qBjG�B�G��k��8Q�@Q�B�{C?33                                    Bxp���  "          @W
=�#�
?�@H��B�  B�=q�#�
���@S33B�u�C�b�                                    Bxp���  "          @l�Ϳ�ff?��H@E�Bn��B����ff���R@X��B�G�CD��                                    Bxp��4  �          @�ff��@QG������mp�B��f��@Vff?@  A%p�B�u�                                    Bxp���  �          @�Q��4z�@hQ쿑��_�B����4z�@mp�?^�RA)�B��                                    Bxp���  �          @���-p�@e��Q��$��B�33�-p�@`  ?�\)Ad(�B��                                    Bxp��&  T          @���=p�@[������T��C�{�=p�@_\)?Q�A#\)CG�                                    Bxp���  
�          @����@��@O\)����\��C���@��@Vff?8Q�A��C�3                                    Bxp��r  "          @�G��*�H@5������}C\)�*�H@@  >��H@�C��                                    Bxp��  T          @��H�8Q�@AG���z��x  C���8Q�@L(�?
=q@�(�C!H                                    Bxp��  �          @����?\)@g��c�
�+�C���?\)@dz�?���AO�C��                                    Bxp�d  �          @qG���(�@AG�?uAw�B�Q��(�@
=@
=B#�RC#�                                    Bxp�#
  �          @l(��\@L(�?Y��AXz�B��f�\@z�@ffB!=qB�p�                                    Bxp�1�  T          @��\�6ff@C33�k��F�HC��6ff@Fff?@  A"ffC�H                                    Bxp�@V  
�          @����^�R@>�R��ff�|��Ck��^�R@N{>Ǯ@��RC	(�                                    Bxp�N�  �          @�Q��U@0  ���R��C� �U@@  >��
@�(�C
�                                    Bxp�]�  T          @�33�U@7��������CT{�U@H��>��
@}p�C�                                    Bxp�lH  �          @�ff�dz�@333�����l��C���dz�@AG�>\@���C�                                    Bxp�z�  �          @��g�@-p����H�o�
C.�g�@<��>��
@}p�C��                                    Bxp�  "          @����g
=@5���z�����C��g
=@J=q>W
=@{C
ٚ                                    Bxp:  
�          @�\)�l(�@-p����ep�C�3�l(�@;�>�Q�@�=qC�=                                    Bxp¦�  �          @����q�@#�
�^�R�-p�C���q�@)��?�@��HC��                                    Bxpµ�  �          @�  �xQ�@'
=�W
=�"{C)�xQ�@*�H?
=@�z�Cs3                                    Bxp��,  �          @�ff��Q�@C33����f�RC���Q�@Q�>��@���C��                                    Bxp���  �          @�{�u@G������  C�f�u@^{>u@)��C	��                                    Bxp��x  �          @���y��@<�Ϳ��H��33C�)�y��@R�\>aG�@(�C�)                                    Bxp��  �          @���~{@0  ����o�
CO\�~{@A�>��@>�RC��                                    Bxp���  �          @�=q�u�?�����������C��u�?��?!G�A�RC�                                    Bxp�j  T          @����|(�?�33<�>�33C33�|(�?�?Y��A8  C .                                    Bxp�  "          @��|��?���>�z�@\)C$n�|��?O\)?L��A333C(k�                                    Bxp�*�  �          @�������?�=q=u?Q�C!������?�\)?8Q�A  C$k�                                    Bxp�9\  T          @��H���
?�=q>W
=@.{C".���
?�ff?W
=A2�RC%�                                    Bxp�H  �          @��H���
?�=q    <�C"!H���
?�33?+�AG�C$ff                                    Bxp�V�  �          @��\���H?��;.{�(�C!�q���H?�  ?�@��C"��                                    Bxp�eN  �          @�=q��(�?�  �����\C#8R��(�?�z�>��H@ϮC$\)                                    Bxp�s�  �          @����(�?��H��{��=qC$����(�?�(�>��R@uC$k�                                    BxpÂ�  �          @��
��ff?�33���H��z�C%}q��ff?��R>\)?��
C$^�                                    BxpÑ@  �          @��\����?��׿�����C%������?��
=#�
>��HC#                                    Bxpß�  �          @�G���z�?}p��!G���p�C'B���z�?��L�Ϳ#�
C%�                                    Bxpî�  
�          @�{���?�Ϳ
=��p�C,�\���?E��u�G�C)�R                                    Bxpý2  �          @�Q���p�>��@  ���C.���p�?E���
=���C*+�                                    Bxp���  �          @�=q��{?�ͿW
=�3\)C,z���{?c�
��ff����C'�                                    Bxp��~  �          @�����?}p���Q��~�HC&Q����?��������  C ^�                                    Bxp��$  �          @�  �n�R?��������
C��n�R@(��������CǮ                                    Bxp���  T          @�(��p��@(���{��{C���p��@,�;��
�z�HC^�                                    Bxp�p  �          @�G��333@O\)��Q���z�C���333@a�>�33@��C h�                                    Bxp�  �          @����<��@7
=����p�C�)�<��@Y���L���\)C�3                                    Bxp�#�  �          @���:�H@7�������{Cs3�:�H@Y���.{�{C�f                                    Bxp�2b  �          @��
�7
=@5���z�����CG��7
=@Q녽L�Ϳ!G�C
                                    Bxp�A  �          @���<(�@#33���H����C)�<(�@<(����
����C
=                                    Bxp�O�  �          @���J�H@#�
�Ǯ����C\�J�H@@  ���Ϳ���C��                                    Bxp�^T  T          @�\)�R�\@#�
�������C#��R�\@AG��\)���C	�                                    Bxp�l�  �          @�Q��Vff@ �׿�33��(�C)�Vff@@  �W
=�)��C
&f                                    Bxp�{�  �          @�(��Mp�@"�\�����p�C�H�Mp�@>{���Ϳ�\)C	33                                    BxpĊF  �          @�Q��G�@+�����h  CY��G�@7
=>\@��C	�                                    BxpĘ�  �          @��
�b�\@7
=�Y���)�C)�b�\@:=q?&ffA{C�)                                    Bxpħ�  �          @�
=�i��@'���{��=qCT{�i��@<��=���?�C{                                    BxpĶ8  �          @����|(�@G��c�
�0Q�C{�|(�@=q>���@��C�=                                    Bxp���  �          @�
=��  @#33��G����\C���  @�?�33AUC��                                    Bxp�ӄ  �          @�ff�s33@,(��
=q�ӅC�s33@'
=?W
=A$Q�C�=                                    Bxp��*  T          @��n{@.{�k��3�
C�)�n{@\)?�\)Aap�CB�                                    Bxp���  �          @�����@2�\�fff�#
=C����@8Q�?�@�CJ=                                    Bxp��v  "          @��H��Q�@#33��ff�a�C����Q�@7
==���?��C�)                                    Bxp�  "          @����z�@8Q��ff��33C����z�@X�þ8Q��CG�                                    Bxp��  T          @�����ff@333�������CǮ��ff@Tz�W
=�(�C@                                     Bxp�+h  �          @������@4z�� �����\CJ=����@\�;������C�                                    Bxp�:  �          @�p�����@.�R����=qC�����@W�����p�CT{                                    Bxp�H�  �          @�{���@333�33���
C�=���@\�;�ff���Ch�                                    Bxp�WZ  �          @�����
@9����=q���C޸���
@Z�H�L�Ϳ�p�CxR                                    Bxp�f   
(          @�p����
@8�ÿǮ��
C�3���
@R�\<��
>uC��                                    Bxp�t�  "          @�(�����@=p��˅��G�C������@W
=<�>�\)Cff                                    BxpŃL  "          @�
=��Q�@E��ٙ���{C����Q�@aG��#�
��\)C�                                    Bxpő�  T          @���@Dz�����RCO\��@c�
����Q�C8R                                    BxpŠ�  T          @��H����@9����G��G�
C^�����@I��>�\)@2�\CB�                                    Bxpů>  �          @�ff��z�@J�H���H�;�C���z�@W
=>�@�z�C�                                     BxpŽ�  �          @�ff���
@O\)�}p��CY����
@Tz�?.{@��HC��                                    Bxp�̊  T          @����=q@QG�������C����=q@J=q?��
A"�HC��                                    Bxp��0  
(          @�����  @O\)=#�
>�33C����  @5�?�=qA
=C.                                    Bxp���  �          @��(Q�?�  �w
=�Op�CY��(Q�@J=q�+��=qC��                                    Bxp��|  �          @�(��?\)?�Q��K��&��C�?\)@N{��\)���\C޸                                    Bxp�"  T          @�p��_\)@ff�.�R�z�C��_\)@H�ÿ�z���\)C
                                      Bxp��  �          @�Q��s33@33��\���C�H�s33@G
=�k��)�C�                                     Bxp�$n  �          @��\�Z=q@333�+��
ffC���Z=q@1�?=p�A(�C�\                                    Bxp�3  )          @�=q����@5�@%�BB�\����?��\@g�Be(�C�R                                    Bxp�A�  �          @�Q��%@333?�=qA�
=C�%?���@<(�B/��CJ=                                    Bxp�P`  �          @�33�@��@0  ?�  A��C	���@��?��H@'�B��Cp�                                    Bxp�_  M          @��
�C33@9��?�z�Aw33C��C33@   @��B  C�                                     Bxp�m�  �          @��\�;�@I��>�\)@qG�C���;�@(��?޸RA�  C	�q                                    Bxp�|R  �          @�
=�,(�@L(���G����RC#��,(�@7
=?�z�A�C5�                                    BxpƊ�  M          @���9��@Mp��#�
��C��9��@5�?�  A���C��                                    Bxpƙ�  
�          @���Mp�@S33?�R@�C.�Mp�@'�@ffA�33C�                                     BxpƨD  
�          @����W
=@
=?�
=A�z�C�)�W
=?��@&ffBQ�C�                                    Bxpƶ�  
�          @���e@�?�z�A���Cs3�e?��@p�B��C#{                                    Bxp�Ő  "          @�\)�x��@p�?z�HA@��C��x��?��H@G�A˙�C=q                                    Bxp��6  M          @�ff�vff@
=?�
=Ai�C� �vff?��
@	��A�
=CJ=                                    Bxp���  
�          @����mp�@�?s33AC\)CaH�mp�?�33?���AͅC�q                                    Bxp��  �          @�(��hQ�@5�?�\@ʏ\C��hQ�@G�?��
A�=qC�                                    Bxp� (  �          @�Q��dz�@I��������C
�{�dz�@333?�Q�A��HC��                                    Bxp��  �          @�
=�n{@R�\���
�aG�C
���n{@<��?�(�A�\)C��                                    Bxp�t  �          @�=q�~�R@aG�>W
=@��C
���~�R@@��?���A��\C�
                                    Bxp�,  �          @����tz�@P  ��33�~{C�H�tz�@B�\?���A]p�C��                                    Bxp�:�  �          @�  �y��@`  �W
=�C

=�y��@L��?�Q�A~�\C�{                                    Bxp�If  "          @�Q��x��@a녾W
=�G�C	Ǯ�x��@N{?���A�Q�C\)                                    Bxp�X  �          @�Q��p��@e�&ff��(�CaH�p��@^�R?��A?\)C	=q                                    Bxp�f�  T          @��R�j=q@l(�<�>�p�C�\�j=q@P  ?޸RA�{C
c�                                    Bxp�uX  
�          @�Q��W
=@g
=?Y��A�C  �W
=@333@�HA�C0�                                    Bxpǃ�  T          @���J=q@o\)>�  @7�C:��J=q@L��?�Q�A��HC�H                                    Bxpǒ�  
�          @�(��<(�@y��>�?���B����<(�@Y��?�A��HC�
                                    BxpǡJ  "          @�z��8Q�@|��>\@��B�8R�8Q�@U�@	��A�ffCٚ                                    Bxpǯ�  �          @�z��A�@�z�=�G�?�G�B�u��A�@g�@   A��
C�                                    BxpǾ�  �          @�  �K�@�p���\)�=p�B���K�@n�R?�{A��Ck�                                    Bxp��<  T          @�Q��e�@u���R�Z�HC�3�e�@c�
?��RA�33C&f                                    Bxp���  "          @����hQ�@u�������\Ch��hQ�@e?�33AuCJ=                                    Bxp��  �          @���4z�@��H�G���\B�\�4z�@�  ?�
=AY�B�\)                                    Bxp��.  �          @�p��fff@:�H=u?J=qC���fff@$z�?��A�C�                                     Bxp��  �          @�G��Y��@Y���8Q��	��C��Y��@G
=?��A�ffC	��                                    Bxp�z  �          @�Q��Z�H@g
=�Ǯ��Cff�Z�H@X��?��Ar�RC5�                                    Bxp�%   �          @�p��U@z�H��p���(�Cc��U@j=q?���A���CY�                                    Bxp�3�  �          @���Z=q@Z=q���Ϳ�
=C  �Z=q@E?��HA��C	��                                    Bxp�Bl  �          @�\)�N{@Dz�>\@��\CaH�N{@#�
?�p�A�Q�Cz�                                    Bxp�Q  T          @�{�QG�@Tz�>�
=@�(�C�{�QG�@1G�?��A�\)C                                    Bxp�_�  
�          @��
�AG�@I��>�Q�@�  C�\�AG�@(��?�  A�z�C
��                                    Bxp�n^  �          @��\�>�R@q녽���
=C Q��>�R@Z�H?���A�\)C
                                    Bxp�}  �          @�=q�(��@�Q�8Q��ffB��(��@i��?�33A��B��                                    Bxpȋ�  �          @�p��@i��=�\)?p��B�W
�@N�R?��HA��
B��                                    BxpȚP  �          @�p��0��@>�R<�>�C޸�0��@(��?�\)A��CG�                                    BxpȨ�  �          @����p  @/\)��z��aG�C�H�p  @$z�?z�HAA�C��                                    Bxpȷ�  �          @����G�@\)�@  �
=CW
��G�@#�
>�ff@��C�H                                    Bxp��B  T          @�Q���  @�H�8Q���
Cٚ��  @�R>�G�@�(�C.                                    Bxp���  �          @�ff�dz�@@  �����mp�C���dz�@4z�?�=qAU�C��                                    Bxp��  "          @�33�\(�@&ff���
��=qC���\(�@
=?��Aj�\C��                                    Bxp��4  T          @���W�@.{>���@z=qC+��W�@33?�p�A�\)C�                                     Bxp� �  �          @��\�.{@<��>�G�@�\)C�f�.{@��?��HA�z�C
�                                    Bxp��  
�          @��H�?\)@,(�?J=qA1p�C
��?\)@33?�33AݮCxR                                    Bxp�&  �          @\)�(Q�@<(�>�@�ffC���(Q�@�H?޸RA�C	c�                                    Bxp�,�  
�          @����dz�?�33?��A��C  �dz�?�\)@G�A��C"�)                                    Bxp�;r  �          @y���4z�@{?c�
AV{C
� �4z�?���?�33A��C+�                                    Bxp�J  �          @r�\�0��@�
?��A��C!H�0��?�{?��HA��C��                                    Bxp�X�  �          @a��8��?�(�?&ffA(Q�C�R�8��?�p�?���A��C��                                    Bxp�gd  �          @e��.{@ ��?�\)A�Q�Cs3�.{?��?��B p�C:�                                    Bxp�v
  T          @j�H���@�@"�\B=�B�ff���?=p�@L��B�.C��                                    BxpɄ�  T          @x���(�?�Q�@�\B33C���(�?:�H@:=qBC(�C#^�                                    BxpɓV  �          @q녿�\)?��@*=qB4�Cٚ��\)?�@N{Bkp�C$�                                     Bxpɡ�  �          @^�R��
=?�33@,(�BK��C  ��
=>���@H��B�  C(@                                     Bxpɰ�  )          @[����?��H@/\)BT��B�\)���>���@Mp�B�u�C#�                                    BxpɿH  �          @c�
�h��@�@0  BLQ�B�
=�h��?�R@W�B��C��                                    Bxp���  �          @c�
�0��@ff@1G�BMz�B���0��?.{@Z=qB�k�CT{                                    Bxp�ܔ  �          @p�׿�Q�@C�
?޸RA��Bޔ{��Q�?��R@9��BM\)B���                                    Bxp��:  �          @fff��p�@.�R?�33B�
B�\)��p�?�\)@9��B[�B��                                    Bxp���  �          @_\)��@!G�@�B�B��
��?�{@>{BkG�C��                                    Bxp��  T          @[�����@�R@B�B�Ǯ����?�=q@=p�Bp33C                                    Bxp�,  �          @\(��+�@�R@  B$�\B�=q�+�?�G�@FffB���B��                                    Bxp�%�  
�          @Tz�
=@\)@B5�B�aH�
=?�  @Dz�B���B�R                                    Bxp�4x  T          @i����@(��@=qB'�HB�Ǯ��?�=q@Tz�B��B��                                    Bxp�C  "          @Y���u@+�@�
B�B�#׾u?\@AG�B{�B�Ǯ                                    Bxp�Q�  �          @i��=�Q�@4z�@\)B�HB��=�Q�?�=q@O\)B�#�B�                                    Bxp�`j  T          @hQ�#�
@333@��B��Bͽq�#�
?���@H��Bt�Bߏ\                                    Bxp�o  "          @J�H�\@Q�?�p�B�B�\�\?��@3�
B���BԞ�                                    Bxp�}�  T          @QG���=q@{@B   B�����=q?�=q@<��B�u�B�B�                                    Bxpʌ\  �          @N�R��=q@%?��B�B��þ�=q?��
@3�
Bt{B��                                    Bxpʛ  �          @Z�H�G�@�H@p�B$�B��G�?��R@B�\B���B��{                                    Bxpʩ�  
�          @c�
�B�\@��@��B1=qB�B��B�\?�{@O\)B��)B�Ǯ                                    BxpʸN  
�          @_\)�^�R@�R@G�B#ffB��
�^�R?��\@G
=B~��B�.                                    Bxp���  "          @e��p�@%@
=B  B��
��p�?�Q�@@��Bg{CxR                                    Bxp�՚  T          @g��k�@'�@��B�RBڣ׿k�?�33@J�HBxQ�B�k�                                    Bxp��@  �          @fff��
=@ff@{B,��B�=q��
=?���@O\)B=qC	�                                    Bxp���  �          @e�@  @�H@��B0  B�p��@  ?�33@P  B�#�B�B�                                    Bxp��  �          @s33�8Q�@?\)@�B��B�{�8Q�?��@O\)Bm(�B���                                    Bxp�2  �          @h�þ���@2�\@�\B33B�Q����?Ǯ@P��B~�HB��
                                    Bxp��  T          @e��   @%@��B(�Bɽq�   ?��@P��B���B��H                                    Bxp�-~  �          @c33��ff@(Q�@33B"�HBǊ=��ff?�z�@L��B�33Bה{                                    Bxp�<$  �          @p�׾�
=@1G�@\)B&\)B�p���
=?��H@Z�HB��B�u�                                    Bxp�J�  T          @fff�\)@>�R@ ��B��B�=q�\)?���@E�Bk��B�ff                                    Bxp�Yp  �          @H�þu@�R?�BG�B�{�u?�Q�@1�By�B��                                    Bxp�h  �          @Tz�(�@AG�?�{A�(�B��(�@G�@�B0  B�#�                                    Bxp�v�  �          @dz῀  @HQ�?���A�p�B�z῀  @�\@!G�B4�B�W
                                    Bxp˅b  �          @��H��=q@k�?k�AS�B���=q@=p�@Q�B
=B�u�                                    Bxp˔  �          @�  ���@e�?�A�Q�B�k����@0��@#�
B"\)B���                                    Bxpˢ�  
�          @r�\�k�@H��?�{A�\BԔ{�k�@�@@  BSp�B�                                    Bxp˱T  �          @���@N{?���Aأ�B��
��@	��@A�B@�B���                                    Bxp˿�  �          @�=q��\)@�@AG�B233B����\)?s33@p��BsQ�C�                                    Bxp�Π  
�          @������@�\@UBI
=B������?0��@\)B�ffC@                                     Bxp��F  T          @������@=p�@1�B �B�G�����?��@p��Brz�C^�                                    Bxp���  �          @�z���@mp�@{A��B�p����@p�@c�
B[��Bɨ�                                    Bxp���  "          @��R��(�@e�@p�B	33B��R��(�@\)@n{Bi��B�Ǯ                                    Bxp�	8  �          @�Q쿫�@_\)@{B�
B��Ύ�@	��@l(�B]��B�                                    Bxp��  �          @�33��G�@x��@ffA�G�B�\��G�@*�H@`��BG��B枸                                    Bxp�&�  T          @��;��
@u�?�
=A�\)B�����
@,��@UBK�\B��q                                    Bxp�5*  
�          @��\��\)@aG�@?\)Bp�B�z`\)?�p�@�p�Bp(�B�\)                                    Bxp�C�  �          @�\)=�Q�@`��@?\)B!�B���=�Q�?�(�@�p�B�=qB���                                    Bxp�Rv  
�          @�{��{@2�\@G
=B4G�B��H��{?��\@~�RB���C�                                    Bxp�a  "          @xQ켣�
@ff@C�
BR(�B��f���
?c�
@p��B�p�B�\)                                    Bxp�o�  N          @e>�@'�@
=B%��B�.>�?�
=@N{B�  B�u�                                    Bxp�~h  	�          @h��?u@6ff@ ��B��B���?u?��
@@  B_��BvG�                                    Bxp̍  �          @��?E�@3�
@4z�B0Q�B�u�?E�?�@n{B�33Bv33                                    Bxp̛�  T          @�=q?k�@+�@P  BC�\B�{?k�?���@��B��=BK                                      Bxp̪Z  
�          @�G�>�p�@#�
@]p�BTB���>�p�?p��@�ffB�{B���                                    Bxp̹   "          @�\)<#�
?�(�@n�RBxB�\)<#�
>�\)@��RB�=qB���                                    Bxp�Ǧ  �          @��Ϳ�>��R@�  B�z�C�
�녿��@q�B��Cv�                                    Bxp��L  �          @w
=�!G�?�@5Bg(�B�(��!G�>\@QG�B���C��                                    Bxp���  "          @���=#�
@xQ�?���A�  B�=#�
@C�
@*�HB$=qB�k�                                    Bxp��  �          @�(�?Tz�@�ff>��@�
B�?Tz�@r�\?�A���B�\)                                    Bxp�>  
�          @�{?��@�
=�������B�?��@���?�G�A�Q�B��R                                    Bxp��  �          @�p�?Ǯ@�������ffB��?Ǯ@��\?�p�A��HB�Q�                                    Bxp��  �          @��R?�@�  =#�
>�(�B��3?�@��\?�Q�A�(�B��R                                    Bxp�.0  T          @�{?��H@hQ�?��HA�G�B��?��H@5@#�
B��B{�                                    Bxp�<�  
�          @���>�ff@�G�?5A
=B�Q�>�ff@Z�H@G�B\)B�
=                                    Bxp�K|  �          @���?G�@l(�?��A���B�?G�@<��@{B�B�=q                                    Bxp�Z"  "          @{�?�@   @(�BQ�BX\)?�?���@N�RB](�Bff                                    Bxp�h�  �          @n{?У�@>�R?�{A�ffBuQ�?У�@��@�B&�BU�\                                    Bxp�wn  �          @p  ?}p�@^�R>�p�@��HB�.?}p�@C33?�(�A�  B��                                    Bxp͆  
�          @w
=?fff@g�?J=qA@  B��?fff@@��@
=qB
=qB��q                                    Bxp͔�  �          @��R?���@|(���
=���B��q?���@r�\?�A���B��\                                    Bxpͣ`  �          @�{?��@g
=?333AQ�By�?��@B�\@z�A�RBi                                      BxpͲ  �          @���@W�?�(�>�\)@��A�@W�?ٙ�?��
At��A�\)                                    Bxp���  "          @�  @*=q@g
=��Q����
BV\)@*=q@]p�?��A`(�BQ�R                                    Bxp��R  "          @��@AG�@Y������
=BA�\@AG�@Tz�?fffA5��B>��                                    Bxp���  �          @|(�?�@q�>�p�@��RB�p�?�@U?���A���B�aH                                    Bxp��  �          @��R?\@y��������=qB�ff?\@l��?�  A�33B�aH                                    Bxp��D  �          @�=q@�
@e��\)�q�Bp�\@�
@o\)>\@���Bt��                                    Bxp�	�  �          @��\@%@g
=��ff�T  BY\)@%@n�R>�ff@�(�B\��                                    Bxp��  T          @���@x��@P  ?�@��B=q@x��@1�?�G�A�33B(�                                    Bxp�'6  �          @�ff@O\)@Q����z�BG�@O\)@��>�ff@��HB                                      Bxp�5�  �          @�\)@*=q@@  ��ff����BA��@*=q@QG������BKz�                                    Bxp�D�  �          @��@@��@G����
�U��B8=q@@��@Q�>��@X��B=                                    Bxp�S(  �          @�\)@Tz�@g�������{B=�R@Tz�@_\)?��
A@��B9�                                    Bxp�a�  T          @��@%@��׾�(����RBp��@%@��H?���Ao�
Blp�                                    Bxp�pt  �          @���@4z�@�(��#�
�\)Bo�
@4z�@�Q�?�\)A�
=Bg                                    Bxp�  T          @���@0��@�Q�@5A�\)Bt��@0��@a�@���B5�HBO�
                                    Bxp΍�  �          @��@�@�=q@5Aۙ�B�W
@�@e@�=qB?��Bq��                                    BxpΜf  �          @�\)?��@�
=@{A��B��q?��@`  @w
=B3�
B��                                    BxpΫ  �          @aG�>#�
@_\)�k��n�RB�Ǯ>#�
@S�
?���A�Q�B��                                     Bxpι�  �          @hQ�=�Q�@dz�<��
>\B�{=�Q�@R�\?��A�
=B��
                                    Bxp��X  "          @i����Q�@e����R��ffB��)��Q�@Z�H?�=qA�{B���                                    Bxp���  T          @|(����@u����{B�\���@q�?h��AV�HB�=q                                    Bxp��  �          @�������@�  �����
=B�=q����@�=q?�=qA���B���                                    Bxp��J  �          @��.{@���W
=�1�B�Q�.{@��?�p�A�ffB��{                                    Bxp��  �          @���=��
@��׾�{��  B��)=��
@���?�33A�ffB�                                    Bxp��  T          @�\)<�@������G�B�\)<�@���?��RA��
B�Q�                                    Bxp� <  �          @�z�B�\@��H�
=q��(�B����B�\@�{?�=qA}G�B�                                    Bxp�.�  �          @��׾�p�@�>8Q�?�Q�B�(���p�@�\)@�A��\B��                                    Bxp�=�  T          @�녿(�@��?uAD��B�
=�(�@k�@!�BB���                                    Bxp�L.  �          @�p����
@��?Tz�A��B�W
���
@�G�@/\)A�Bͨ�                                    Bxp�Z�  �          @��� ��@j�H@   A�B��R� ��@)��@N�RB&�Cn                                    Bxp�iz  T          @�\)�%@fff?xQ�AF=qB��\�%@>{@  A��C�                                    Bxp�x   T          @��R��{@��
��(���  B܀ ��{@��R?��RAuB��H                                    Bxpφ�  �          @����Z=q@Y��>�@�ffC!H�Z=q@>{?��HA�{C
�3                                    Bxpϕl  T          @��{@aG�?0��A�\B���{@@  ?���AՅCff                                    BxpϤ  �          @�z��33@s�
?�  AK33B�8R�33@J=q@ffA�  B�
=                                    Bxpϲ�  T          @�33��  @5��ff�m��C�
��  @G�������HC�                                    Bxp��^  T          @����\@����k��-p�B��)�\@��?�p�A�p�B�                                    Bxp��  �          @��Ϳ��H@�ff=���?���B��
���H@��\?�=qA�33B�.                                    Bxp�ު  �          @��H����@�����
�n{B�
=����@�33?޸RA�  B�                                    Bxp��P  �          @�{�z�@�zᾀ  �C�
B�=q�z�@��?�p�A�p�B�                                      Bxp���  T          @�33=u@�G�>�(�@�{B�z�=u@��@z�A�z�B�L�                                    Bxp�
�  N          @��
?�(�@��?   @�=qB��?�(�@�=q@A��B�aH                                    Bxp�B  
`          @�\)?�{@{�?:�HA��B�W
?�{@X��@�A�=qBt�H                                    Bxp�'�  �          @�  ?��@�p��u�5B��R?��@�z�?���A�z�B�G�                                    Bxp�6�  �          @��R?��@���=��
?��
B��H?��@~{?�33A���B�u�                                    Bxp�E4  �          @�>�Q�@�(�����z�B���>�Q�@�
=?�=qA{�B�Q�                                    Bxp�S�  T          @��?���@�  �k��5B�.?���@���?�
=A���B���                                    Bxp�b�  T          @��@AG�@:�H?�p�A�z�B0=q@AG�@@*�HB  B
�                                    Bxp�q&  �          @�G�?�Q�@e?z�A ��Bvp�?�Q�@HQ�?�A�Q�Bh��                                    Bxp��  T          @�33��p�@}p��L(���
B�Ǯ��p�@�33�����(�B���                                    BxpЎr  
�          @������@{��Dz��B�uÿ��@��ÿ�Q���G�B�k�                                    BxpН  
�          @�  ��@i���e��0z�Bţ׿�@�ff�G�����B�                                    BxpЫ�  T          @�ff@Vff@��?�@�(�BN��@Vff@p��@�A�33BA(�                                    Bxpкd  �          @�Q�@L(�@��R>���@FffBY@L(�@���?��A�z�BO=q                                    Bxp��
  
�          @��@U@���?E�A=qBI
=@U@^{@
=qA�p�B8p�                                    Bxp�װ  T          @��@mp�@^�R?O\)A(�B,��@mp�@<��@   A���B=q                                    Bxp��V  T          @�ff@p��@b�\>�ff@��B-�@p��@H��?�Q�A�(�B��                                    Bxp���  "          @�@Tz�@|(�>��
@b�\BGp�@Tz�@c�
?��HA�ffB<{                                    Bxp��  �          @�\)@Fff@xQ쾔z��W�BM��@Fff@n�R?�{AN�RBI(�                                    Bxp�H  �          @��
@Z�H@qG�������\B?(�@Z�H@j�H?uA-B<                                      Bxp� �  �          @�G�@u�@fff�\)���HB,��@u�@dz�?B�\A�B+��                                    Bxp�/�  �          @�=q@b�\@aG��^�R�G�B3(�@b�\@fff>�G�@�\)B5�                                    Bxp�>:  �          @�z�@u@I�������X��BG�@u@W�    �#�
B$�                                    Bxp�L�  "          @�p�@���@AG�����K�B�\@���@N�R<#�
=uB�                                    Bxp�[�  
�          @�  @e�@>�R��\���RB  @e�@Z=q�(���  B.z�                                    Bxp�j,  
�          @�ff@S33@@���ff��33B)ff@S33@c�
�fff�(z�B<z�                                    Bxp�x�  �          @��@E@>�R������B/�R@E@e�����K
=BD�
                                    Bxpчx  
�          @�33@Mp�@E�����z�B/\)@Mp�@c�
�5�\)B?�
                                    Bxpі  �          @��?�z�@L(��Z=q�-�RBy�?�z�@��z���B���                                    BxpѤ�  
�          @�33@J�H@Q��=q���B��@J�H@1녿#�
�
=B%=q                                    Bxpѳj  "          @�=q@mp�?�Q��=q���RA�33@mp�@p����
�Qp�B33                                    Bxp��  
�          @���@S33@(��p����
B�H@S33@C�
��(��q�B+(�                                    Bxp�ж  
�          @�{@\(�@33������
B=q@\(�@;���G��y��B!z�                                    Bxp��\  T          @�33@hQ�@����陚A�\)@hQ�@7
=���H��ffB��                                    Bxp��  �          @�{@U@ff�����B�\@U@?\)��ff���B'=q                                    Bxp���  
�          @���@2�\@C33�=q���B>33@2�\@l�Ϳ��H�iG�BS��                                    Bxp�N  �          @�(�@�\@j=q��Q��ƸRBsz�@�\@���\)���B~Q�                                    Bxp��  �          @��?�=q@�33��  ��(�B���?�=q@��    <��
B�u�                                    Bxp�(�  �          @�
=?��R@��H��{�[33B�.?��R@�
=>�p�@�  B���                                    Bxp�7@  T          @�(�?�{@��Ϳ�Q��n�RB��=?�{@���>��R@x��B��                                    Bxp�E�  
�          @���?�(�@|�Ϳ(�����B��\?�(�@|(�?=p�A"=qB�p�                                    Bxp�T�  
�          @�@7�@Z=q�����ffBGp�@7�@k��\)��Q�BO��                                    Bxp�c2  
�          @��\@J=q@0��<�>�(�B$��@J=q@%�?xQ�AV�\B�                                    Bxp�q�  
�          @���@9��@G
=����B<  @9��@<(�?��
A`(�B5\)                                    BxpҀ~  �          @�=q@HQ�@(��?�=qAl��B ��@HQ�@�?�A�\)B�                                    Bxpҏ$  �          @z�H@%@+��\)���B7��@%@-p�>\@��HB8��                                    Bxpҝ�  �          @xQ�@7
=@#33>8Q�@.{B'  @7
=@?��Az�HB�                                    BxpҬp  �          @��
@1�@,(�?�=qAx��B0\)@1�@
�H?�
=A�33B�
                                    Bxpһ  "          @��@H��@z�?���A��B�@H��?�  ?�(�A�  Aˮ                                    Bxp�ɼ  �          @��H@o\)?�����z��\)A��
@o\)?�
=>\@���A�ff                                    Bxp��b  
�          @�(�@l��@�����_�A���@l��@�
��z��uB 33                                    Bxp��  	�          @���@1�?���,(��#�A��@1�@p��z���B�H                                    Bxp���  �          @�Q�@e�?�  ��R��p�A�p�@e�@
=q������A���                                    Bxp�T  �          @��@L��?�p���H�	�HAƣ�@L��@p����
��G�B
\)                                    Bxp��  T          @�z�@C33@$z��(����\B Q�@C33@@  �B�\�!p�B1�H                                    Bxp�!�  �          @�p�@Fff@ �׿�G���(�B�
@Fff@<�ͿQ��-�B.G�                                    Bxp�0F  �          @���?���?��XQ��S33B333?���@5��%�{Be                                      Bxp�>�  T          @�\)@3�
@�
�\)��33BQ�@3�
@:�H������
=B8=q                                    Bxp�M�  �          @��@e�@&ff��  ���RB�@e�@A녿J=q���B!{                                    Bxp�\8  "          @�z�@`  @9����ff��
=B\)@`  @U��=p��\)B.33                                    Bxp�j�  T          @���@I��@!���R���\B(�@I��@Mp���G���{B633                                    Bxp�y�  �          @�(�@|(�?���?aG�A>{A�33@|(�?z�H?��\A�33A`Q�                                    Bxpӈ*  T          @��
@�{?�ff?�=qAp��A��@�{?��\?�G�A�\)AD��                                    BxpӖ�  �          @���@��?�p�?��HAS�A�z�@��?�  ?У�A�A8��                                    Bxpӥv  �          @���@��R?�33?O\)A(�A��@��R?�ff?�ffAdz�Aj�H                                    BxpӴ  �          @�=q@�  ?�Q�>�p�@�{A��H@�  ?�p�?fffA$��A�=q                                    Bxp���  �          @�
=@�{?Ǯ>8Q�@G�A��\@�{?�?.{@��HA�(�                                    Bxp��h  �          @�  @�z�?�G�=�?��
A���@�z�?У�?0��@�\A�\)                                    Bxp��  
�          @�@��
?��Ϳ��\�dQ�A�(�@��
@
=q�������A�(�                                    Bxp��  
Z          @���@�p�@�R��{�v�\A�{@�p�@#33�\)���A��                                    Bxp��Z  T          @��H@�Q�@�Ϳ����HA�ff@�Q�@+����\�0Q�A��                                    Bxp�   �          @��@���?˅����A��\@���@Q쿦ff�UA�p�                                    Bxp��  
�          @�{@�=q@�������
=A�@�=q@?\)��\)�ip�B
�                                    Bxp�)L  T          @���@�(�@��7
=��A��
@�(�@N�R����z�B�                                    Bxp�7�  �          @�G�@��
@���W
=�\)A�(�@��
@Y���������B�\                                    Bxp�F�  
�          @��H@�p�@�H�Y�����A��
@�p�@X��������B�
                                    Bxp�U>  
-          @�{@�=q@���n{���A�  @�=q@]p��0  �ӅB�H                                    Bxp�c�  T          @Å@���@�R��  �#p�A��@���@X���Dz���\)B�H                                    Bxp�r�  O          @���@_\)?��|(��6�A̸R@_\)@6ff�L����\B��                                    Bxpԁ0  
�          @�=q@p  @������*�B�@p  @e��HQ�����B.�                                    Bxpԏ�  "          @���@QG�@X���}p����B8�@QG�@�\)�*�H��\)BW\)                                    BxpԞ|  
�          @�  @8��@����r�\�BY33@8��@�Q���
��(�BpG�                                    Bxpԭ"  T          @�=q>B�\@|���1��Q�B��\>B�\@�(�������ffB�\)                                    BxpԻ�  T          @�  �
=@dz��\)���B�.�
=@y�������  B��
                                    Bxp��n  
�          @�  �,��@S�
��ff�]��CG��,��@^{=L��?!G�B��)                                    Bxp��  �          @��
�8��@^{�5��RCǮ�8��@aG�>�(�@��HCh�                                    Bxp��  �          @�ff�.�R@J�H�z��ҏ\C���.�R@j=q�u�>�\B��=                                    Bxp��`  �          @���@��@$z��'
=��
C��@��@P�׿���33C�3                                    Bxp�  T          @��G�@
=q�,(��(�C\)�G�@8�ÿ�\)���C	0�                                    Bxp��  T          @�\)��R@��E��-�HCB���R@Mp�����G�B�k�                                    Bxp�"R  T          @�p��{@
�H�J=q�6
=C���{@C33�z����RB�\                                    Bxp�0�  T          @�(��*=q@�\�1���\CxR�*=q@3�
�   �ڸRCn                                    Bxp�?�  �          @��H�33@���I���:�
C�=�33@AG��z���HB�u�                                    Bxp�ND  �          @�=q��{?��
�p  �^(�C8R��{@7
=�@���%��B�                                      Bxp�\�  �          @o\)��33?�(��G��v�HCY���33@Q��%��;z�B���                                    Bxp�k�  T          @�33�s33?�G��*=q�`Q�B�#׿s33@���33��RB�u�                                    Bxp�z6  "          @�  �(Q�?�  �����XffC�{�(Q�@��[��/(�C	33                                    BxpՈ�  T          @�녿�?�ff��  �e��C:��@<���P  �,�
B�#�                                    Bxp՗�  �          @�z��(�?�
=�u�Q�\Cz��(�@1��HQ��!�C:�                                    Bxpզ(  '          @�33�'
=?�Q��~�R�O��C#��'
=@5��QG��!�C�                                    Bxpմ�  �          @�z��0  ?�����b33C)\)�0  ?�ff�xQ��F�RCٚ                                    Bxp��t  �          @��H�A녿�R�i���G{C?� �A�?
=q�j=q�H  C)�3                                    Bxp��  �          @����l�ͿaG��X���&��CAc��l��>L���`  �-Q�C0�H                                    Bxp���  �          @���s33��
=�E�\)CH�H�s33����XQ��&33C:33                                    Bxp��f  T          @�(��@  �+��*�H�
=C]�q�@  �޸R�W
=�033CR�                                    Bxp��  �          @�
=����(��L���,�Ca���������r�\�X\)CQO\                                    Bxp��  �          @�  �Q녿�G��<�����CL�R�Q녿
=q�QG��2{C=W
                                    Bxp�X  T          @���(���\�s33�b�
C?�\�(�?+��qG��`��C$��                                    Bxp�)�  �          @�
=�z�H>�ff��
=�=CaH�z�H@\)����u�B�=q                                    Bxp�8�  �          @�Q�aG�?�\)����¡=qB�Q�aG�@J=q�Ǯ�|\)B�                                      Bxp�GJ  T          @�  >.{>���ff«��B�8R>.{@#�
��\)�
B�L�                                    Bxp�U�  �          @�G�>�?�33��� 33B���>�@P����\)�|{B�33                                    Bxp�d�  �          @�{>�
=?�Q������B�W
>�
=@{�����bp�B��3                                    Bxp�s<  �          @�33?
=?˅�ڏ\\B�
=?
=@dz������l
=B��                                    Bxpց�  �          @�G�?\?c�
�ᙚ��A�{?\@=p���ff�z��B{\)                                    Bxp֐�  �          @�?�?�  �޸R�3B�?�@aG���{�e��B{Q�                                    Bxp֟.  �          @��?�(�@333�Ӆk�B��3?�(�@�����
=�BB�u�                                    Bxp֭�  �          @�(�?8Q�@;���aHB�8R?8Q�@�G���  �Bp�B���                                    Bxpּz  �          @׮��p���Q���Q��
CDO\��p�?\�\G�C �                                    Bxp��   �          @�
=�\)�G���=q�u=qCaff�\)��(��Ǯ�
C>��                                    Bxp���  T          @�
=���\�=p����fCW����\?������
��C�                                    Bxp��l  �          @�녿}p��L�����\\)CZ޸�}p�?fff���
=C	��                                    Bxp��  �          @�33�p�׾\��
=¢ǮCJ5ÿp��?�\)�����B�8R                                    Bxp��  �          @�ff���;��
���u�C=������?�z��θR�
C
�                                    Bxp�^  T          @��ff��R��=q��CD}q�ff?��\��
=�C�                                     Bxp�#  �          @��
���������CA&f��?��\����}(�C(�                                    Bxp�1�  �          @�{�^�R�u�����K�C8�^�R?��
���DG�C#u�                                    Bxp�@P  "          @׮�n{�u��Q��_z�C7��n{?�����\�U33C�=                                    Bxp�N�  �          @��
����8Q���
=�933C68R���?��
�����1ffC$�H                                    Bxp�]�  �          @љ����H�u����4�C4Ǯ���H?�����+�C$J=                                    Bxp�lB  �          @У��w
=�Ǯ��ff�U33C9���w
=?�33���H�N��C#s3                                    Bxp�z�  T          @�z���Q�>�
=��{�>ffC.W
��Q�?�\����-�Cff                                    Bxp׉�  "          @�\)��33�������I��C9�q��33?����z��D\)C%��                                    Bxpט4  "          @�Q��|(��#�
����R��C4�=�|(�?�(���ff�F�C��                                    Bxpצ�  T          @ҏ\�XQ�=��
����k�C2�3�XQ�?ٙ���33�ZCB�                                    Bxp׵�  
Z          @��G�=����\�w=qC1���G�?���G��c��C��                                    Bxp��&  T          @����R>�����H8RC)c���R@����ff�l
=CL�                                    Bxp���  T          @ʏ\�0��>�(������~��C+!H�0��@33��G��d
=Cff                                    Bxp��r  �          @�
=�Q�=�����=q�
C1G��Q�?��
�����}��C�                                    Bxp��  
�          @ʏ\����?B�\�\�fC.����@�R��33�|z�B�G�                                    Bxp���  "          @�=q��33?p����G�C�R��33@-p���  �xG�B�u�                                    Bxp�d  �          @Ӆ�s33��G����¢��C:���s33?�p���p�p�Bힸ                                    Bxp�
  "          @�zῡG�>����Q��3C-\��G�@��z�B�W
                                    Bxp�*�  
�          @�ff�8Q�>�=q��ff¦\)C�=�8Q�@ff�Å��B�                                      Bxp�9V  �          @�G��=p�?���ff£� C8R�=p�@����Q���B�Q�                                    Bxp�G�  �          @��ÿ�p�?���=q8RC ����p�@33���\)B�L�                                    Bxp�V�  �          @�\)��ff>�z���(� �C$����ff@��G���B�.                                    Bxp�eH  �          @��ÿL��<#�
��
=¥\)C3ff�L��?��
���RǮB�k�                                    Bxp�s�  
(          @�(��Ǯ�B�\�Å«��CM�׾Ǯ?Ǯ�����B��H                                    Bxp؂�  
�          @���8Q�=u�ʏ\¦��C.��8Q�?��������{B�k�                                    Bxpؑ:  "          @�z�?Q녿�33����C�5�?Q�>�����£�A�                                    Bxp؟�  T          @�  ?(���
=��=q\)C��=?(�>���ff¦B
=                                    Bxpخ�  �          @�(�?
=q��Q���  Q�C�Ff?
=q?333��=q¤�3BP��                                    Bxpؽ,  "          @�p����ÿ(�����H¦33CsǮ����?�=q����ǮBՅ                                    Bxp���  �          @����5�k���  ¥��CF5ÿ5?��H���\ǮB�                                    Bxp��x  T          @��ÿ:�H���H��
=£��CV\�:�H?��
��33��B�G�                                    Bxp��  
�          @�33���
�����\)�CL����
?�p���(�z�C)                                    Bxp���  T          @����;����� C=33����?�z�����k�C
}q                                    Bxp�j  T          @�ff��{�#�
��
=�qC8���{?����\�C                                      Bxp�  �          @�p��У׾u�����C<aH�У�?��\���R��C+�                                    Bxp�#�  
�          @`  ?�녿�(�?��A���C�R?�녿ٙ�?
=Aa�C��q                                    Bxp�2\  
�          @�Q�?�
=�'�@A�B9�C�%?�
=�W
=@�A���C���                                    Bxp�A  �          @�?����fff@FffB��C���?�����=q?�p�A��\C���                                    Bxp�O�  T          @���@'
=��33?��A�  C�Ǯ@'
=��33>L��@�C�%                                    Bxp�^N  �          @�Q�@���G�@A�B�\C�33@���
=?��A��
C�T{                                    Bxp�l�  "          @�\)?��B�\@�
=BL�RC�q?����@\��B
=C�`                                     Bxp�{�  �          @�  ?��H�P  @fffB.{C��H?��H���
@$z�A���C��\                                    Bxpي@  �          @�(�@%��Q�?���A5p�C��=@%���
��z��<(�C�e                                    Bxp٘�  T          @�=q@�����R���\�4��C�` @����(��7���33C�b�                                    Bxp٧�  �          @ٙ�@!G���{�����G�C�\@!G������0����Q�C�H                                    Bxpٶ2  �          @���@(������{�333C�k�@(���G��Dz���{C�l�                                    Bxp���  �          @�\@=q��
=��\)��HC�4{@=q��p��6ff��(�C��                                    Bxp��~  
�          @�33@p�������Q���C��H@p���z����A�C���                                    Bxp��$  �          @�=q@&ff���
���H�(�C��@&ff�z=q��(��Mz�C��\                                    Bxp���  �          @��
@G��������8�C��f@G��333�Å�m�
C��                                    Bxp��p  "          @�p�@z�����@����p�C�g�@z���Q������ 
=C�33                                    Bxp�  T          @�33?������H���\�1
=C�)?����aG������n=qC��{                                    Bxp��  �          @�\��=q�vff��
=�hC���=q� ��������C�:�                                    Bxp�+b  T          @陚��G��`  ��z��tQ�C�b���G��У���33(�C~��                                    Bxp�:  T          @��ýu�Vff�θR�zQ�C�|)�u��(���(���C��{                                    Bxp�H�  "          @�G�=����;���z���C��q=��Ϳ����{¢C���                                    Bxp�WT  T          @���녿�  �Ӆ�HC{���=������Hª=qC)�R                                    Bxp�e�  
�          @���z������u�C��H��z�>\��=q¬.B�Q�                                    Bxp�t�  �          @�?333�J�H�Ӆ�G�C�AH?333���\��
=8RC�y�                                    BxpڃF  T          @�?=p��vff�ƸR�f�C�y�?=p���\����C��                                    Bxpڑ�  �          @���G��z��ָR33C��)��G���(���\ª
=C`E                                    Bxpڠ�  �          @��8Q쿳33��{� Cr��8Q�>�ff��=q¥��C0�                                    Bxpگ8  T          @�\)>�p���=q��z��Q�C�o\>�p��(�����H�fC��{                                    Bxpڽ�  �          @�z���W������y��C�%����  ��ffp�C}�                                    Bxp�̄  �          @��\)��z���p��HC;L��\)?����׮��C޸                                    Bxp��*  �          @��ÿ���3�
��(��C�����^�R��(�£�3Ck:�                                    Bxp���  �          @�zᾮ{�<(����Q�C��
��{�u��¤��Cz��                                    Bxp��v  �          @��H��R�����z���C���R������¨B�CX��                                    Bxp�  �          @�(�����%����
��Cw�3�������G�¢�fCN�                                     Bxp��  �          @��H���\� ����� Cs\���\=L����Q�¤��C1
=                                    Bxp�$h  �          @���33?n{��=q�C�f�33@(���\�t�
B�Ǯ                                    Bxp�3  �          @���Vff?�33�����j��C�
�Vff@N{��  �G(�C+�                                    Bxp�A�  �          @����j=q?�G���z��i�C$���j=q@&ff�����NG�C��                                    Bxp�PZ  �          @����8��@�\��(��{G�C���8��@r�\��33�P�B��                                    Bxp�_   
�          A��z=q?333��\)�qQ�C)���z=q@ ����G��ZffCE                                    Bxp�m�  "          A�H��z�L����=q�r  C4����z�?����\�e{C�
                                    Bxp�|L  �          A{���>8Q���  �p\)C1� ���@���{�`��Ck�                                    Bxpۊ�  
�          A���{��  ��33�p��C7h���{?�Q������f��C�                                    Bxpۙ�  �          AQ����þ�
=��p��fp�C9\)����?�(������_�C!�R                                    Bxpۨ>  �          A�H���ÿ�����33�f�CE������?���
=�l��C,��                                    Bxp۶�  
Z          A  ��33?�
=�ٙ��V�C&B���33@;���  �?Q�C�H                                    Bxp�Ŋ  �          A����\)@p���\)�]�\CǮ��\)@��������;{C
�                                    Bxp��0  T          A�H��p�@��� ���l��C:���p�@����z��F��Ck�                                    Bxp���  T          A#�
�i��?�{��H�C#
�i��@b�\���p{C�                                    Bxp��|  
�          A!p��x��?\)���fC+���x��@<���(��s��C�{                                    Bxp� "  "          A!p����@
=��`
=C33���@��R��ff�>�RC
��                                    Bxp��  �          A5�����?�(��Q��n33C$����@y������S��C+�                                    Bxp�n  �          A9G���z��������
�&ffCU����z��'��	���B  CH#�                                    Bxp�,  T          A4����=q�Q����[��CR}q��=q�E��  �p{C;��                                    Bxp�:�  �          A3\)����dz��=q�R
=CS�
��녿���(��hffC?#�                                    Bxp�I`  �          A5���أ����R�   �4\)CS���أ��   ���MG�CDk�                                    Bxp�X  �          A:�H���
������!�\C]����
�p  ���D�HCQ��                                    Bxp�f�  �          A2�\��G���
=��Q���G�Ca��G����
��(��  CZ\)                                    Bxp�uR  T          A3\)������\)����ޏ\Cd��������(���  ��C]��                                    Bxp܃�  
�          A<z�������
��
=���
Cj^�������p���z��(�Ccff                                    Bxpܒ�  �          A<  ��33��������Q�Ce���33��{���@�C[�                                    BxpܡD  �          A:{�ƸR��
=�p��1Q�C]^��ƸR�L�����S\)COG�                                    Bxpܯ�  �          A=p���\)�i���z��F��CPxR��\)������\�Z�HC>&f                                    Bxpܾ�  �          A8  ��p���G����� �\C[����p��^�R��
�A�HCO�=                                    Bxp��6  "          A2{�*=q���R�,    C:�q�*=q@p��'��
C	Q�                                    Bxp���  "          A4���_\)�(���+\)L�C>���_\)@
=�(Q���C�{                                    Bxp��  
�          A6ff�q녿��
�*{CJ��q�?�  �*�R\C!Ǯ                                    Bxp��(  
�          A1���ff��G�� ���=CC����ff?���� z�=qC#�                                    Bxp��  �          A+�
��\)?�  �'�¨�RB�L;�\)@h���{k�B�                                    Bxp�t  "          A+
=���?\(��*{©  B��)���@c33� ���BŔ{                                    Bxp�%  �          A(�׿���?�\�#\)aHB�������@�=q��R�3BָR                                    Bxp�3�  T          A$  ?=p�@����\  B��?=p�@����\)�t33B��)                                    Bxp�Bf  �          A(  ��{@��"�H(�B��쾮{@����G���B�p�                                    Bxp�Q  �          A)��G�@p��!�  Ch��G�@��R���s�B�ff                                    Bxp�_�  �          A&ff�B�\@(�� ��u�B֊=�B�\@�p���u\)BŊ=                                    Bxp�nX  �          A%G���G�@C�
�p�ffB�8R��G�@��R��
�g��B�\)                                    Bxp�|�  T          A)�>u@o\)��k�B��\>u@��
�	p��Z�B�p�                                    Bxp݋�  �          A!p��8Q�@y���=q{B��
�8Q�@��H��33�P
=B�p�                                    BxpݚJ  �          A*�\�hQ�@����C:��hQ�@���\)�d�RC ��                                    Bxpݨ�  �          A)��33>�=q��
�vffC0���33@"�\�
{�f\)CaH                                    Bxpݷ�  �          A*�R�i��?�p��
=�C���i��@���33�g�C�3                                    Bxp��<  �          A2�H�33@��H�!33B���33@�  �
=�K��B���                                    Bxp���  �          A1G����R@�����G�B�R���R@�������I��B�Ǯ                                    Bxp��  �          A1G����R@��� z��B�aH���R@�(��
=q�M�B�z�                                    Bxp��.  �          A"�H�$z�?��
�Q�k�C0��$z�@]p��33�)B�.                                    Bxp� �  �          A.=q�8Q�@�����d(�B�8R�8Q�@����p��/p�B�Ǯ                                    Bxp�z  T          A1���?����)�3Cc���@|���
=z�B�                                      Bxp�   �          A1���S�
?��&�\��C��S�
@�
=�p��oz�B�{                                    Bxp�,�  "          A7
=���
@�� ���u��B����
@�G���R�=��BΣ�                                    Bxp�;l  �          A3�?(��@����
=�=(�B���?(��A����G��=qB��f                                    Bxp�J  
�          A0  ?���@��H���
�6��B���?���A����R����B��                                    Bxp�X�  �          A3�
?=p�@ٙ��ff�Q�RB���?=p�A�������B�k�                                    Bxp�g^  
�          A2�H?��@�(��G��bG�B��q?��AQ���  �(z�B���                                    Bxp�v  �          A-녿��@h���!�Q�B�����@���{�[�BЅ                                    Bxpބ�  T          A+�
��
@J�H� ���RB����
@��\��H�aQ�Bܙ�                                    BxpޓP  �          A'\)���@3�
��
=B������@�����e�RB�                                      Bxpޡ�  �          A&�\�Q�@=q���Ck��Q�@����33�nz�B���                                    Bxpް�  �          A!������@HQ���
z�B� ����@�����\�b  B���                                    Bxp޿B  "          A33���@Q��=q�fCٚ���@�z��  �k\)B���                                    Bxp���  �          A+\)��Q�����z�� CIn��Q�?h����ǮC'#�                                    Bxp�܎  �          A,�����H��  �  �{CG)���H?xQ���G�C'^�                                    Bxp��4  X          A*=q������
��
�X��CG�{����=���
�b
=C2�=                                    Bxp���  �          A0z���zῚ�H���c�C?�{��z�?�����
�d{C).                                    Bxp��  �          A1������=L����p�C3xR����@(�����d
=C+�                                    Bxp�&  "          A0Q���>��(��e33C/^���@0������V33C�q                                    Bxp�%�  T          A0����Q�?�G��p��g  C'�f��Q�@b�\���Q�Ck�                                    Bxp�4r  �          A/
=�У�?���
�H�RC*��У�@P���{�@�Cp�                                    Bxp�C  
�          A&{�e@��
����c33C)�e@����=q�4ffB�W
                                    Bxp�Q�  �          A���Q�@�{���i=qB���Q�@��H����3
=B��f                                    Bxp�`d  �          A!�p  @�\)��(��C�B�Ǯ�p  @�ff��G����B�
=                                    Bxp�o
  �          A��(��@����>�B�8R�(��@�������HB�Q�                                    Bxp�}�  �          A�
� ��@�
=�����OQ�B�L�� ��@�{����{B�z�                                    BxpߌV  T          A���z�@��\��ff�Y�\B�(���z�@�Q������ \)B��                                    Bxpߚ�  T          A�׾��@�����(��_�B�33���@�\�ʏ\�&�HB���                                    Bxpߩ�  T          A��?��
@�����33B�8R?��
A�H��Q���
=B�p�                                    Bxp߸H  �          A=q@�@������H�"�\B�8R@�A���\)��33B��=                                    Bxp���  
(          AG�@\)@�R����
=B�k�@\)Ap��b�\��Q�B��H                                    Bxp�Ք  "          Aff?h��@�ff�����RB�  ?h��A	��^�R��  B���                                    Bxp��:            A  >�@�Q���(��N��B�Q�>�@��R��p���
B�8R                                   Bxp���  
�          A(�?�@ʏ\�ᙚ�?�HB�\)?�@��
�����  B�=q                                    Bxp��  �          A=q>��@�����o��B���>��@�����ff�6�RB���                                    Bxp�,  T          A�?O\)@�
=��p��kp�B�z�?O\)@У���Q��3  B���                                    Bxp��  �          A�\��\)@�����=q��RB�(���\)A�H�j�H��p�B�                                      Bxp�-x  �          A33��G�@�p������
B�����G�A  �S�
���
B�ff                                    Bxp�<  
Z          A�H>�
=@��\�����8  B�>�
=@�(�����ffB�B�                                    Bxp�J�  T          A�\��z�@����O�B�8R��z�@޸R��(���B��
                                    Bxp�Yj  T          AG�?�@��H��\�Z�
B��R?�@�G���G��"(�B�k�                                    Bxp�h  �          A��?��@�{��p��j�\B��{?��@Ϯ�����2�
B�                                      Bxp�v�  T          A{?У�@�33��\)�t=qB���?У�@���\)�>=qB�B�                                    Bxp��\  T          A�?�@�{����u{B���?�@Å��=q�?��B��=                                    Bxp��  "          A�?�G�@�ff���o  B�G�?�G�@�z�����8�B�G�                                    Bxpࢨ  "          A"{?�z�@���
ff�j��B�p�?�z�@�(����4��B��                                    Bxp�N  �          A ��?���@���\)�}z�B�u�?���@�p����E�
B��                                    Bxp��  "          A'�>.{@vff�  �HB��H>.{@��
�(��Y
=B�Ǯ                                    Bxp�Κ  �          A%�>u@��33�|{B�=q>u@�G������CffB�                                    Bxp��@  "          A z�?�@�(��	G��k��B�8R?�@�\����3p�B�p�                                    Bxp���  S          A�R?�R@�����Xp�B�Ǯ?�R@������   B���                                    Bxp���  T          A��>�z�@�z���ff�Nz�B�W
>�z�@�G�������
B��{                                    Bxp�	2  �          A ��>u@�=q�����K��B���>uA z������Q�B��\                                    Bxp��  �          A,��>8Q�@�G���R�~�\B���>8Q�@�Q��\)�E�HB�                                      Bxp�&~  "          A,(�?8Q�@\)�$����B�Ǯ?8Q�@��=q�x�B�p�                                    Bxp�5$  �          A0��?E�����-��¡��C�J=?E�?�z��.ff¥ffBb
=                                    Bxp�C�  �          A.{>�������)��aHC���>���>�\)�-¯k�B��                                    Bxp�Rp  �          A.{>��
�`���$Q�(�C��
>��
�c�
�,��ª
=C��q                                    Bxp�a  
�          A.�R=�G��c33�%���C��=�G��k��-�ª=qC�e                                    Bxp�o�  T          A2=q�=p���33�/�¡L�Cu�
�=p�?�=q�0��¦p�B���                                    Bxp�~b  �          A1�����
�l��� ��33C~ff���
�����*�\£��Cc��                                    Bxp�  T          A;
=��33��\)�33�X��C�"���33��
=�,����C~��                                    Bxpᛮ  �          AF=q�L����ff�9��=C��R�L�Ϳ��R�D��¦Q�Cm�                                    Bxp�T  �          AG
=�0���J�H�?�
�RC��f�0�׾\)�Fff­�=C?��                                    Bxp��  �          AO\)����]p��G��{C|}q��������O
=ª  CC��                                    Bxp�Ǡ  �          AP  ��
����>{Q�Cu�q��
�\�J=q�CXu�                                    Bxp��F  �          AIG��!���(��8(�aHCns3�!녿�Q��B�R�CM�                                    Bxp���  �          AA���R�\��
=���^�Cq0��R�\�W
=�2{8RCa��                                    Bxp��  �          ABff�aG��0���<  �C��\�aG�>L���A�±(�C
@                                     Bxp�8  T          AJ=q>���(��Hz�¦C��=>�?���H(�¤� B���                                    Bxp��  �          APQ��{�b�\�E
=Cu����{����M��¥=qCBaH                                    Bxp��  T          AT  �:�H��\)�C\)Q�Ckn�:�H��\)�M��3CI
=                                    Bxp�.*  �          AL  �|����z��4���y��Cc�q�|�Ϳ��R�@z�G�CH��                                    Bxp�<�  �          AJ�H�l����p��,���j��Ck���l���'��<z��CWE                                    Bxp�Kv  �          A:=q�h���ᙚ���Qz�C�P��h�����\�*�\�C�^�                                    Bxp�Z  �          A:�R���R�߮�(��R{C��῞�R��Q��*�H�C~��                                    Bxp�h�  T          A9녿��
��z����V�C~�q���
�����+\)G�CvǮ                                    Bxp�wh  T          AF=q�<(��G�����,��Cz�{�<(������!G��a�Ct)                                    Bxp�  
�          AHz��J=q�p��߮�=qC|33�J=q���H�
=�=�\Cx�                                    Bxp┴  T          A?
=���{��  ��{C��\��� �����6�C8R                                    Bxp�Z  T          A&�H�	���G������G�C
�	���ʏ\��{�G�C{=q                                    Bxp�   T          A-���3�
�L������C8�3�
@�����  C�H                                    Bxp���  T          A?33��=q=L����R�Z{C3���=q@33����P�C �f                                    Bxp��L  �          AIp���Q���=q�  �7�Cbp���Q��z�H�$z��\G�CU&f                                    Bxp���  �          AW
=��Q���(��/��\��CV�R��Q�Ǯ�;��t��CB                                      Bxp��  
�          AT����p��(��6=q�l��CFٚ��p�?(���9G��sffC.!H                                    Bxp��>  T          AT(��љ��u�733�p��C4}q�љ�@0  �1��e�RC:�                                    Bxp�	�  �          AUp���{>�z��<���y�C1O\��{@J�H�5��j(�C��                                    Bxp��  "          AV{��  @�:�H�u�HC �=��  @�z��,���Y�C�{                                    Bxp�'0  T          AL����ff@U��0Q��l\)C����ff@����H�HC��                                    Bxp�5�  T          AG���
=@U��,z��p�Cn��
=@��
�=q�K=qC�H                                    Bxp�D|  T          A>�\�+�@�����H�ep�B��+�A z��\)�0��B��H                                    Bxp�S"  T          A=���U�@��H����a��B��U�@�=q�=q�/
=B�#�                                    Bxp�a�  �          A?�
����@�������\(�C������@ʏ\�(��5ffC�                                    Bxp�pn  T          A<Q��S�
@�G��%�y�\B�W
�S�
@�ff��
�I�B虚                                    Bxp�  �          A<���׮@j�H�33�E�\Cz��׮@�  �����%�C��                                    Bxp㍺  T          A;33���
@O\)�
ff�?ffC�����
@�Q����H�"Q�C�=                                    Bxp�`  �          A9G���ff@<(����N��C}q��ff@�����
=�0�
C�                                    Bxp�  �          A=p����@a���R�.�C�����@��ᙚ�=qC0�                                    Bxp㹬  "          A;
=�G�@�\��\�%{C$�f�G�@��\��G����C��                                    Bxp��R  �          A9���@P����{���C!����@���\)���C�R                                    Bxp���  �          A,����R@�{?�  A�C���R@��@.�RAn{C{                                    Bxp��  "          A5���z�@Ӆ��\)�p�B�(���z�@�{������
B��                                    Bxp��D  
�          A2=q�
=q@�=q����I=qB��R�
=qA33��G��ffB��H                                    Bxp��  T          A(z���H@�(���=q�;z�B��)���HA����z���
Bʞ�                                    Bxp��  �          A(Q�?���A33@C�
A�  B���?���@�{@�G�BffB���                                    Bxp� 6  �          A&ff�J�HAz���R�c\)B��J�HA���0��Bؔ{                                    Bxp�.�  �          A'�
�=qA���y����\)BӀ �=qA  ���ffB�=q                                    Bxp�=�  
�          A5��z�@θR����T�\Bי��z�A�
�����B�p�                                    Bxp�L(  T          A:�H���@��\�'33�B�Ǯ���@�  ���G(�B��                                    Bxp�Z�  T          A=��
=q@�33��
�vB����
=q@�����=�
B��                                    Bxp�it  
�          A?33?�RA\)����1�B���?�RA$  ��
=��G�B��                                    Bxp�x  
�          A8(���\)A\)� z��1z�B��þ�\)A
=��Q����B�G�                                    Bxp��  f          A4�Ϳ�G�Aff��\�*p�Bŏ\��G�AQ����H�㙚B®                                    Bxp�f  �          A7\)��\)@�{�Q��F  B�Ǯ��\)A���\)�G�B�#�                                    Bxp�  �          A2ff��(�@��
=�>�B�Ǯ��(�A����z���HBə�                                    Bxp䲲  T          A8���#33@����Q��=��B����#33Az���p���
BҮ                                    Bxp��X  �          A8(��5�@���z��Wp�B�aH�5�AQ���
=�"{B���                                    Bxp���  �          A9��ff@�����
�e��B�W
�ffA   � (��/\)BԽq                                    Bxp�ޤ  T          A<��@<(�@����H�m�BoQ�@<(�@��
���:�B�p�                                    Bxp��J  �          A?
=��z�@ڏ\��\�YffB�B���z�A(����� �\B¨�                                    Bxp���  �          AD�ÿTz�@�p��0��W
B�B��Tz�@�  ��H�I�B�8R                                    Bxp�
�  �          AR=q��@��
�?
=��Bۏ\��@�p��$���N�B�(�                                    Bxp�<  �          AT(���@�  �>ff�~��B�ff��A���"�\�GB�(�                                    Bxp�'�  �          AS
=��p�@ٙ��4���i�B̀ ��p�Aff�(��0�B��)                                    Bxp�6�  �          AC\)���RAp��\)�7(�B�33���RA#�
�˅���\BĀ                                     Bxp�E.  �          AF=q��33@�ff�'33�j��Bͅ��33A	�	G��2(�Bƅ                                    Bxp�S�  �          AJ�R��R@|(��?��B��H��R@ٙ��)��d��B�k�                                    Bxp�bz  �          AP(��8Q�A��H�G33B�Ǯ�8Q�A)G���G����B��q                                    Bxp�q   �          AS����A{�%��Mp�B�ff���A'33��
=�G�B�(�                                    Bxp��  T          AY��A (��
=�*�B�=q��A?�
�������B�aH                                    Bxp�l  �          AT��>L��A)p�� (��G�B��H>L��AD(�������HB�(�                                    Bxp�  �          AU�>�A$����  B���>�AA�������Q�B�k�                                    Bxp嫸  T          AP(�?(�A/����H��G�B��{?(�AEp��xQ�����B�L�                                    Bxp�^  
�          AQ���
=q@�\)�@���fB�녿
=q@��H�'��X
=B�\                                    Bxp��  	�          AW��aG�@����A�B��aG�@�p��'�
�V��B�{                                    Bxp�ת  �          AJ�\���AQ��
�H�/z�B�����A.�\������
=B�Ǯ                                    Bxp��P  �          AG�>�33A&�R�ۅ�\)B�#�>�33A<���������\B���                                    Bxp���  �          AM��?���ABff�w
=��z�B�aH?���AK��0���E�B��H                                    Bxp��  �          AI�?��HAE녿�
=��B��{?��HAG\)?��R@�(�B���                                    Bxp�B  �          A>�R@��A8z�G��s�
B�\)@��A5@��A'�
B�
=                                    Bxp� �  �          AH��@33A&{��
=���B���@33A9�Z=q��z�B��\                                    Bxp�/�  �          A'\)?�z�A#33�k���=qB���?�z�A!?�
=AQ�B���                                    Bxp�>4  h          A/\)?ٙ�A'�
<��
=���B��=?ٙ�A"ff@*=qAh(�B��                                    Bxp�L�  h          A=G�?E�A<(��aG����B�� ?E�A7
=@0  AX  B�G�                                    Bxp�[�  �          AA��?��HA?���R�<(�B�#�?��HA<  @�HA9G�B��)                                    Bxp�j&  h          AB{?���A>�\���R��p�B���?���A9@-p�AO�
B�#�                                    Bxp�x�            AC33?�A=����
��p�B��f?�A<(�?�  A�B��H                                    Bxp�r  �          A8���s�
A����Q��
=B�Q��s�
A��^{��(�B�                                    Bxp�  �          A6{��ff@�(���33�(ffB��)��ffA���G���=qB�z�                                    Bxp椾  �          A6ff��\)@������H�#Q�C�=��\)@�{��=q��  B�                                    Bxp�d  �          A3�
�e�A��
=��
=B�Q��e�A�\�;����
Bݙ�                                    Bxp��
  �          A?�?:�HA=G�����θRB��f?:�HA<z�?��HA�B��)                                    Bxp�а  |          A;33?333A:{?Tz�@�=qB�\?333A0z�@p��A�=qB��3                                    Bxp��V  �          A-?���A)�?��@���B�� ?���A�
@p��A��\B�L�                                    Bxp���  �          A/�@
=A*�H?k�@��B���@
=A!��@g
=A�=qB�ff                                    Bxp���  �          A2ff?!G�A0��>�Q�?��B�u�?!G�A)p�@J�HA��B�.                                    Bxp�H  �          A-@��A'�?5@s33B�#�@��A
=@W
=A���B���                                    Bxp��  �          A Q�@aG�@�ff@���A�\)B�L�@aG�@θR@�B(�Bu                                    Bxp�(�  �          A ��@�A{@e�A��B}�@�@�{@�33B�
Bo=q                                    Bxp�7:  T          A�@4z�A  @��AQB�33@4z�A=q@�AٮB�Ǯ                                    Bxp�E�  �          A��@���@����,������Bz��@���A��(��tz�B��                                    Bxp�T�  �          A
=@
=@<����p��{��BMQ�@
=@�Q���p��Iz�B~\)                                    Bxp�c,  �          A(�@�?�������Bff@�@n�R���|  Bs=q                                    Bxp�q�  �          A{@n{>�����k�@�(�@n{@$z��  �w��B
                                    Bxp�x  �          A\)?�33>Ǯ�¥��A��\?�33@5��H\B��                                    Bxp�  �          A&�H>����z�H�G�k�C�1�>��������$Q�¥C�q�                                    Bxp��  �          A2�H����p���
�I  Cz����  ��R��HCqL�                                    Bxp�j  �          A:�\�$z���ff�z��J�\Cy���$z���(��(Q�u�Co�f                                    Bxp�  �          A2�H�aG���������lz�Ch&f�aG���33�'33{CPL�                                    Bxp�ɶ  �          A9���H�e�+�
W
Cq^����H�0���4��33CGJ=                                    Bxp��\  �          A'33���
����$Q�¡{C]� ���
?Ǯ�#\)�CL�                                    Bxp��  T          AQ�?E�@�Q���Q��l�RB�k�?E�@�  �����1G�B�L�                                    Bxp���  �          A(�@$z�@�p���\)���HB��H@$z�@�G��ff����B�aH                                    Bxp�N  �          A$��@@  Ap����?\)B�u�@@  A��>��@�B�B�                                    Bxp��  �          A7�@uA$���S�
���B��@uA,�ÿ�\�#33B��)                                    Bxp�!�  "          A@Q�@���A���=q��\)B�� @���A-G��5�[�B��                                    Bxp�0@  �          AD  @*=qA������!�HB���@*=qA*{�����
=B��f                                    Bxp�>�  �          A733@$z�@�p��   �633B��
@$z�A  ��G�����B�                                      Bxp�M�  �          A)G�?��@j�H�{�RB��\?��@�(��	G��YG�B��H                                    Bxp�\2  �          A7
=?c�
@Vff�-���B�aH?c�
@\��e�RB�k�                                    Bxp�j�  �          A4��?��R@�p��!p�8RB��=?��R@����Q��E��B�k�                                    Bxp�y~  �          A/�
?�z�@���z��r��B�{?�z�@�=q��(��8B��3                                    Bxp�$  �          A4  @%�@�{����uz�Bt��@%�@�33���=B��                                    Bxp��  
�          A:ff@hQ�@��
�\)�d�\BZ�\@hQ�@�� Q��/��B�G�                                    Bxp�p  �          A6=q?���@�
=�   �v�
B��3?���@�p��Q��;  B��                                    Bxp�  �          AN{��  @q��D  �{Bљ���  @�(��-��e�Bą                                    Bxp�¼  �          AJ�\���
@z�H�@z��fB�33���
@޸R�)��b�\B�=q                                    Bxp��b  �          AJ�R?�G�@�G��4��{B�#�?�G�AG���
�DffB��)                                    Bxp��  �          AF�\@N�R@��
�=q�E��B��@N�RA�����=qB���                                    Bxp��  �          AR=q?   >����P��®�A�33?   @p  �HQ�ffB���                                    Bxp��T  �          AX(�<��h���W�¬=qC��q<�@,(��S���B���                                    Bxp��  �          AC�
?�  @E��;33L�B���?�  @\�'��m\)B��                                    Bxp��  �          A=�?�ff@I���5G��=B�  ?�ff@����!G��i�B��                                    Bxp�)F  �          A9�?��H@�z��((�Q�B�
=?��H@���=q�H�B�G�                                    Bxp�7�  �          A?�?�@����'\)�tz�B�Q�?�A�	�8ffB��R                                    Bxp�F�  �          A6ff?�@�\)�  �b��B��R?�A������&��B��                                    Bxp�U8  T          A4(�?��@�
=�
=q�JG�B��?��Az��θR��B��\                                    Bxp�c�  �          A8z὏\)@�  ����\��B�Lͽ�\)A(������B���                                    Bxp�r�  �          A6�R@�\@����33�j��B��=@�\@�����/G�B�(�                                    Bxp�*  �          A4��?��@������f  B�� ?��A����\�)�RB�{                                    Bxp��  �          A(z�?���@{��\)� B���?���@�������O��B�
=                                    Bxp�v  �          A5G�?L��@g
=�+�
  B���?L��@��
���^\)B���                                    Bxp�  �          A7�?��\@���� (��q��B�  ?��\A (��ff�4z�B��                                    Bxp��  �          A7
=?���@��H�"=q�z  B���?���@�z��=q�<�
B�z�                                    Bxp��h  �          A8��?��R@�{�)G��HB�Ǯ?��R@�z���
�Lz�B�B�                                    Bxp��  �          A8��?n{@���+��=B��?n{@�����U33B�aH                                    Bxp��  T          A.ff?�\)@�
=�ff�e�B�?�\)@�\)��Q��'p�B��                                     Bxp��Z  �          A(z�@@ə��=q�K=qB�=q@AQ����
��B���                                    Bxp�   
�          A'�?�p�@�\)�33�W�B�G�?�p�A ����  �B�=q                                    Bxp��  �          A4Q�?��R@����e�\B�u�?��RA����  �'G�B�                                    Bxp�"L  �          AD(�?���@�33�&ff�h�B���?���A
=�Q��*
=B��R                                    Bxp�0�  �          A<Q�?�G�Aff�  �3z�B�{?�G�A!�����R���B�
=                                    Bxp�?�  �          A8z�?�\)A33��������B�(�?�\)A/
=�AG��s�B���                                    Bxp�N>  �          A:=q@0  A����p���33B���@0  A/\)�1G��\z�B��
                                    Bxp�\�  �          A?�@=p�A (����H��B�\@=p�A3��7��]�B�z�                                    Bxp�k�  �          A1p�?�G�A z���(��'�B�8R?�G�A  ��z���=qB�z�                                    Bxp�z0  �          A,��?\@ҏ\� ���G��B��f?\Az���p��	\)B��q                                    Bxp��  �          A(�@
�H@�Q�@�=qB%B��@
�H@_\)@���B^��Bh\)                                    Bxp�|  �          A��?˅@�=q@x��B �HB��=?˅@E�@�B[�Bz��                                    Bxp�"  �          A�@4z�A�\��\)�>{B��@4z�A	��?�@VffB��                                    Bxp��  �          AW�@�=qA@  ������B�=q@�=qAMp����R����B��)                                    Bxp��n  ~          A��\@�=qAap����R����B���@�=qAs
=��=q���HB���                                    Bxp��  �          A���@�ffAa�������
=B���@�ffAp�Ϳ�z�����B�.                                    Bxp��  �          Ay�@�33A[�
������B��@�33Ak����H���B��=                                    Bxp��`  �          AxQ�@�33AV�R�Å��G�B�=q@�33Aip��Q���(�B�33                                    Bxp��  �          A
=@�Q�A]����(�����B�33@�Q�Ap(����陚B�\                                    Bxp��  �          Ao�@���AV�R��33���B��@���Ab�R�+��%�B��                                    Bxp�R  T          Ak\)@N�RAQ�����=qB�W
@N�RAc
=����\)B�\)                                    Bxp�)�  �          At��@e�AS���\)��G�B��@e�Aip��0���&�\B�aH                                    Bxp�8�  �          AyG�@��A[\)������B���@��Am��(���Q�B�(�                                    Bxp�GD  �          A|(�@�G�A^�H��  ����B�@�G�Ap�ÿ���
=B�.                                    Bxp�U�  �          Ax��@���A]����(����B��f@���Al�Ϳ�G����B�#�                                    Bxp�d�  �          Ak
=@|(�AQ����H��
=B�z�@|(�A`�׿�Q�����B��                                    Bxp�s6  �          Ad��@FffAF=q��(���=qB��H@FffA[
=�'��)G�B�p�                                    Bxp��  
�          A^�H@9��A<���أ���
=B�\)@9��AS��J=q�R{B�B�                                    Bxp됂  �          AO�
@�HA/33������B��@�HAE���J=q�a�B��
                                    Bxp�(  �          Ap(�@Y��ATQ����
����B�=q@Y��Ag\)�
=�G�B��                                    Bxp��  |          A��
@�  Ar�H����=qB�B�@�  A�=q�&ff��RB�                                      Bxp�t  �          A���@��\Ak
=��������B���@��\Au녾#�
�z�B��                                    Bxp��  �          AxQ�@�z�Ac
=�\)�nffB�  @�z�Ak�
>aG�?L��B�Q�                                    Bxp���  �          Ajff@�G�AR�\�l���i�B�=q@�G�AZ�R>W
=?Q�B��R                                    Bxp��f  �          A~�R@�Q�A`z��s�
�_
=B��@�Q�Ahz�>�Q�?��B�k�                                    Bxp��  �          A~�\@ۅA\���}p��f�HB~33@ۅAe��>��?
=qB��H                                    Bxp��  �          Ai@�G�AE�������
B�R@�G�AS�
��
=��z�B���                                    Bxp�X  �          A�33@��Am���ff�~=qB���@��Ay���k��B�\B��3                                    Bxp�"�  �          A�p�@ȣ�A�
=�i���9��B��@ȣ�A�?��@��\B�u�                                    Bxp�1�  �          A���@��RA�
=�Tz����B�=q@��RA�  @{@���B�k�                                    Bxp�@J  T          A�
=@�A������6=qB��@�A�=q?���@�  B�\)                                    Bxp�N�  �          A�  @�z�A�\)����uG�B�Ǯ@�z�A�\)>�p�?��\B�                                    Bxp�]�  �          A�(�@}p�A�z����B���@}p�A�33�'
=��  B���                                    Bxp�l<  �          A�Q�@�33A�33�ڏ\��\)B���@�33A�=q�W
=�{B��=                                    Bxp�z�  �          A�{@��A��R��z��k�B���@��A��H@�(�AB�RB�Ǯ                                    Bxp쉈  �          A���@�=qA�Q��z��ÅB�  @�=qA���@p  AffB���                                    Bxp�.  �          A��@�
=A�����ff�R�HB�Q�@�
=A�\)?���@>�RB�=q                                    Bxp��  �          A��@�(�A�{�	G���ffB�� @�(�A���*=q��p�B�p�                                    Bxp�z  T          A��@(�A����-�����B��
@(�A�����Q��X(�B��\                                    Bxp��   �          A��@Z=qA�z������G�B�u�@Z=qA����w
=�%B�p�                                    Bxp���  �          A��
@p��A�{���י�B�Q�@p��A�=q�r�\�"�HB�z�                                    Bxp��l  �          A�ff@�
=A���  ����B�8R@�
=A�(����H��ffB���                                    Bxp��  �          A�(�@���A�  � z���G�B�33@���A�{�   ��33B�W
                                    Bxp���  �          A��@���A�����
���B�\@���A���l����
B�k�                                    Bxp�^  �          A���@�
=A�(�������HB���@�
=A�=q�n�R�!�B���                                    Bxp�  �          A�G�@���A�(�����B�� @���A�
=�����-G�B���                                    Bxp�*�  �          A��R@{�A�Q��1p�����B��@{�A�=q��=q�g
=B�{                                    Bxp�9P  �          A��@�33A��H�{���B��=@�33A��\�`����B��                                    Bxp�G�  �          A���@��HA��
��Q����HB�G�@��HA���8Q���B�                                      Bxp�V�  �          A��A	�A�Q��`  ���B�ffA	�A���@�R@��HB��q                                    Bxp�eB  �          A��R@�A�33��{�H��B���@�A�33?��@l(�B���                                    Bxp�s�  �          A���A
=A�G������
B��A
=A�z�@��
A/�B�Q�                                    Bxp킎  �          A�\)A��A��녿�G�B�\A��A��@�G�AaB|{                                    Bxp�4  �          A��
@�A��������,��B�\)@�A�  @z�@�G�B��                                    Bxp��  �          A��R@�G�A�
=������
B�(�@�G�A�  �B�\��B��                                    Bxp���  T          A��\@�Q�A����   ���B���@�Q�A�������-�B�z�                                    Bxp��&  �          A�@y��A���+�����B�aH@y��A�Q����\�R�\B�8R                                    Bxp���  �          A�\)@\(�A����3\)���
B��@\(�A������j�RB��                                    Bxp��r  �          A��R@HQ�A�\)�C
=�G�B�@HQ�A�z���  ����B�
=                                    Bxp��  �          A�{@-p�A��\�I��\B�p�@-p�A����߮���B�G�                                    Bxp���  �          A�z�@0��A��
�Fff�Q�B���@0��A��������B��
                                    Bxp�d  �          A�Q�@6ffA�  �J�\�  B�{@6ffA�������  B�33                                    Bxp�
  �          A��@5�A�z��F�\��HB���@5�A�ff������p�B��\                                    Bxp�#�  �          A��H@   A����P�����B�W
@   A�Q���  ���RB�=q                                    Bxp�2V  �          A���@{A�  �M���{B��3@{A�G��������B�u�                                    Bxp�@�  �          A�z�@A|(��V�R� �B��H@A�����ff���B��                                    Bxp�O�  T          A��?��
Ax  �[
=�%G�B��3?��
A����Q���{B��q                                    Bxp�^H  T          A�\)?}p�Atz��^{�(�RB���?}p�A�z��  ��z�B���                                    Bxp�l�  T          A��?�Q�Ar{�`���+{B��?�Q�A��
�\)���B�.                                    Bxp�{�  �          A���?��\Ah���h(��3Q�B�  ?��\A�z������G�B�\                                    Bxp�:  T          A��?h��A_33�o
=�;�\B��?h��A���=q��\)B�(�                                    Bxp��  �          A��R?�(�Ao��]�*p�B��H?�(�A�=q�����G�B�aH                                    Bxp  �          A��@>{A�33�D���B�z�@>{A�G��Ӆ���HB���                                    Bxp�,  �          A�33@r�\A�  �$  ��=qB��{@r�\A�z������@z�B�\)                                    Bxp���  �          A���@l(�A�  �"ff��\B�.@l(�A�Q���{�<Q�B��)                                    Bxp��x  �          A�Q�@c�
A�(��=q��{B�\)@c�
A���g��"ffB��R                                    Bxp��  �          A��@W
=A����?\)�33B�p�@W
=A�(��ʏ\��G�B���                                    Bxp���  T          A��\@h��A����8Q��

=B��@h��A�
=��=q���B�z�                                    Bxp��j  �          A�=q@���A�  ����ͅB�B�@���A��H�333��B��f                                    Bxp�  �          A��\@ÅA�������L��B��@ÅA��\?��H@�ffB��R                                    Bxp��  T          A��@�\)A�33�j�H�!��B���@�\)A�ff@,��@���B�{                                    Bxp�+\  �          A��
@ȣ�A�ff�9�����RB��H@ȣ�A���@_\)A�B��R                                    Bxp�:  �          A��@�{A����ff��ffB�ff@�{A���@���A2{B��                                    Bxp�H�  �          A�p�@���A���p���B�{@���A��\@�{AD(�B�p�                                    Bxp�WN  �          A�
=@�z�A����\)��33B�p�@�z�A��@}p�A/
=B�{                                    Bxp�e�  �          A�33@�33A�
=�,(���(�B�z�@�33A��
@g
=A
=B�.                                    Bxp�t�  �          A��@߮A���)���陚B��=@߮A���@l(�A"�RB�8R                                    Bxp�@  �          A��@��
A��
�U���RB�{@��
A�Q�@>�RA33B�.                                    Bxp��  �          A�z�@��HA�  �:�H�G�B���@��HA�G�@]p�A�B��                                     Bxp  �          A�ff@���A���!��߮B�L�@���A�@y��A-�B��                                    Bxp�2  �          A��\@��
A����Q����B�.@��
A�ff@�\)AF�RB�p�                                    Bxp��  �          A��\@�
=A��R�녿�=qB�.@�
=A��
@��\A�(�B��3                                    Bxp��~  T          A��\@�33A��
�   ��\)B�  @�33A��R@��
A��B�ff                                    Bxp��$  T          A���@�A�33��\)�B�\B�#�@�A��@�=qA�p�B�L�                                    Bxp���  �          A���@�=qA�
==���>�=qB�ff@�=qA�ff@�A���B�G�                                    Bxp��p  �          A�z�@��A�
=�����ffB�
=@��A�33@���AO
=B�33                                    Bxp�  �          A�ff@��A�33��33�K�B��\@��A�@���Ak\)B�Q�                                    Bxp��  �          A�ff@�  A����
�5�B��=@�  A��@�Ar=qB�8R                                    Bxp�$b  �          A��\@���A�ff�L�����B�(�@���A��@�A}G�B��q                                    Bxp�3  �          A�
=@��HA��H�h��� ��B�=q@��HA��R@���Aup�B��q                                    Bxp�A�  �          A�G�@��
A��R��z��K�B��q@��
A�G�@��RAg
=B�G�                                    Bxp�PT  �          A�G�@��
A�Q�fff�{B���@��
A�{@�G�AuB�{                                    Bxp�^�  �          A�\)@�33A�z�aG���HB�Ǯ@�33A�(�@�=qAw33B�8R                                    Bxp�m�  �          A���@�Q�A����+�����B�z�@�Q�A��R@�  A
=B���                                    Bxp�|F  T          A���@��A�z��\)��ffB���@��A��@�\)A]G�B���                                    Bxp���  
�          A�p�@�z�A�녿O\)�p�B��=@�z�A�G�@��A~�RB���                                    Bxp�  �          A�p�@��A�  ���Ϳ��B�p�@��A�=q@�z�A�z�B���                                    Bxp�8  �          A�p�@�\)A��<��
=L��B�\@�\)A���@�G�A���B��f                                    Bxp��  �          A�\)@�  A��>8Q�?�\B��H@�  A�Q�@�ffA�p�B���                                    Bxp�ń  �          A���@���A�
=?
=q?��RB��{@���A��R@��A�p�B�
=                                    Bxp��*  �          A��R@У�A�Q�k���RB��@У�A��@˅A�=qB���                                    Bxp���  �          A��\@�33A��?.{?��B��@�33A�(�@�Q�A�p�B��{                                    Bxp��v  �          A�z�@��A�p�?���@���B��H@��A�{AffA��\B��                                    Bxp�   �          A�z�@���A�33?�@���B���@���A���A\)A�B���                                    Bxp��  �          A�z�@���A��R@�@���B�k�@���A�33A�A��B��                                    Bxp�h  �          A��
@��A�ff?�?�\)B��R@��A��
@�33A�33B��                                    Bxp�,  �          A���@��A�녿�G���ffB��f@��A���@�z�A^�\B�
=                                    Bxp�:�  �          A��\@�z�A�33�0�׿�B���@�z�A��@�
=A�33B�{                                    Bxp�IZ  �          A�z�@�Q�A��R����ͅB�.@�Q�A�(��ff���B�33                                    Bxp�X   �          A��@��HA�z��$����33B�
=@��HA�ff��  �7
=B���                                    Bxp�f�  �          A�=q@�\)A�z���33��z�B�=q@�\)A�\)��{�xQ�B���                                    Bxp�uL  �          A��@�{A��R�z���
=B�G�@�{A�@��RAH  B���                                    Bxp��  �          A��
@�
=A��
�2�\��  B�8R@�
=A�(�@\)A1��B��)                                    Bxp�  �          A�  @��
A���xQ��,z�B�@��
A���@>{A�B�L�                                    Bxp�>  �          A��@���A����|(��/�B���@���A�
=@:�HA��B��                                    Bxp��  �          A���@�{A�=q���\�L��B�Ǯ@�{A��@�@��B�p�                                    Bxp�  �          A��H@�
=A��������g�B���@�
=A��?ٙ�@�\)B���                                    Bxp��0  �          A�  @���A��������b�RB��\@���A�33?�=q@��B�p�                                    Bxp���  �          A���@�ffA�������@(�B�{@�ffA���@#33@��
B���                                    Bxp��|  �          A�G�@�A�G������?
=B��H@�A��@%�@�p�B�k�                                    Bxp��"  �          A�G�@��A�G���p��EB��=@��A�{@��@�=qB�(�                                    Bxp��  
�          A��@�
=A�����\�{
=B�@�
=A���?�ff@g�B�G�                                    Bxp�n  �          A�G�@��\A�  �ҏ\����B��
@��\A���>��R?^�RB���                                    Bxp�%  �          A�
=@�=qA��\��z���  B�(�@�=qA��׿Q���B���                                    Bxp�3�  �          A��@�G�A�������B�u�@�G�A��ÿ333��
=B���                                    Bxp�B`  �          A���@��A�z���
=���
B��\@��A�������Dz�B�                                      Bxp�Q  �          A��\@�G�A�=q���
��{B�L�@�G�A�ff�G��
=qB��q                                    Bxp�_�  �          A�(�@���A��\�����G�B�W
@���A��ÿ\(����B���                                    Bxp�nR  �          A�@�
=A��R��������B�#�@�
=A�33�^�R�(�B�Q�                                    Bxp�|�  �          A�  @�{A���\��p�B�� @�{A����!G���  B��=                                    Bxp�  �          A�Q�@��RA����p�����B��H@��RA���>��
?fffB���                                    Bxp�D  �          A��\@�  A�z���ff��Q�B���@�  A��
>�  ?333B��                                    Bxp��  �          A���@\A����G�����B�Q�@\A��ÿ\��
=B��\                                    Bxp�  �          A�{@���A��H��
����B�(�@���A��ÿ�z��|(�B�33                                    Bxp��6  
�          A�@��
A�=q�
=���HB��@��
A��ÿ�{����B�                                    Bxp���  �          A��@��A�z��  �ʣ�B���@��A�(����H��  B�8R                                    Bxp��  �          A��@�{A�=q������HB��\@�{A���%���B�k�                                    Bxp��(  �          A�33@�  A�
=�+33��33B�Q�@�  A������=G�B��                                    Bxp� �  �          A�G�@��
A}��4Q��(�B���@��
A��R��z��]��B�                                    Bxp�t  �          A�  @�33Aap��V�H�&�
B�p�@�33A�������
=B��
                                    Bxp�  �          A�p�@�\)AT���f�\�6\)B��
@�\)A��\�33��z�B��)                                    Bxp�,�  T          A��R@�=qA[�
�]���.=qB�\)@�=qA�Q�� (����B��                                    Bxp�;f  �          A�  @�Q�AiG��M�� �\B���@�Q�A��
��G�����B�.                                    Bxp�J  �          A�ff@��HA�  �%p���ffB�
=@��HA�
=�tz��/�B�                                    Bxp�X�  �          A�ff@��A|���/���
B���@��A������R�\B�W
                                    Bxp�gX  �          A��@��A~�H�.ff�33B�{@��A�Q���p��Lz�B�B�                                    Bxp�u�  
�          A�G�@�ffA�\)��\��z�B�aH@�ffA����0  ��
=B�W
                                    Bxp�  T          A�
=@��A����=q���HB�\)@��A�(��#�
��=qB��3                                    Bxp�J  �          A�=q@�\)A�33��\)��G�B�ff@�\)A�����
�uB�W
                                    Bxp��  �          A��@��A�(���{��Q�B��
@��A�p�?���@L��B�{                                    Bxp�  �          A��
@�(�A�G�������  B�  @�(�A�p�?�(�@�Q�B�B�                                    Bxp�<  �          A��@��RA�(���\)��33B���@��RA�{?�ff@���B�                                    Bxp���  �          A�=q@�A�\)�z=q�6�RB�  @�A�z�@J�HA�
B�8R                                    Bxp�܈  �          A�G�@�G�A��������:�HB�
=@�G�A�  @H��A�B�G�                                    Bxp��.  �          A�G�@�p�A�G��e��&=qB���@�p�A�33@eA&�RB���                                    Bxp���  �          A���@�{A�Q��\(��=qB��@�{A��
@r�\A.ffB�k�                                    Bxp�z  �          A��
@�{A�  �p��ȣ�B��@�{A�@���Af�\B�33                                    Bxp�   �          A���@�p�A�(��)����G�B���@�p�A��@�(�ATQ�B�                                      Bxp�%�  �          A�\)@��A���
=���B�8R@��A�G�@�z�AlQ�B�B�                                    Bxp�4l  T          A���@�G�A�G��G���Q�B�\)@�G�A�ff@��RAp  B�Q�                                    Bxp�C  �          A��\@��RA�p��{���HB�=q@��RA��@�G�Ah��B�\)                                    Bxp�Q�  �          A�Q�@���A��
�8�����B��@���A�p�@�ffAMG�B��                                    Bxp�`^  �          A��\@�ffA��׿����w
=B��)@�ffA���@�A��B��                                     Bxp�o  �          A��\@�Q�A��
�\��33B�  @�Q�A�G�@��A��B���                                    Bxp�}�  �          A��R@���A��
��ff���B�p�@���A�{@�  A��B�Q�                                    Bxp�P  �          A��\@���A����G����\B�ff@���A��
@�Q�A���B�B�                                    Bxp���  �          A�=q@���A��
�
=q��{B��q@���A�33@�33Al  B��R                                    Bxp���  �          A��
@�z�A���<(��\)B���@�z�A�G�@��
AJffB�W
                                    Bxp��B  �          A��@�33A����/\)��p�B�p�@�33A��R@��\AT��B��)                                    Bxp���  �          A���@���A���)����p�B���@���A�z�@�AY�B�(�                                    Bxp�Վ  �          A���@��A���Vff�\)B��@��A�ff@���A;�B��q                                    Bxp��4  �          A���@�A�ff��Q��:ffB���@�A�\)@X��A�B���                                    Bxp���  �          A��\@��RA�p������X��B��@��RA�ff@0��A Q�B�33                                    Bxp��  �          A��
@���A��
���R�t��B��@���A��\@�@�33B�\)                                    Bxp�&  �          A�33@�33A�z��N{�z�B��3@�33A��@��A@��B�u�                                    Bxp��  �          A���@���A���@ff@�\)B�#�@���A�(�A33A�
=B��                                    Bxp�-r  �          A�Q�@�A��\@!G�@��B�  @�A�A��A�
=B�z�                                    Bxp�<  �          A��\@�  A�(�@AG�A  B�� @�  A{�
A(�A��HB�z�                                    Bxp�J�  �          A�\)@��A�Q�?��
@�Q�B���@��A�ffA�
Aϙ�B��H                                    Bxp�Yd  �          A�33@��
A��@��@�  B��@��
A��Ap�A�p�B��q                                    Bxp�h
  �          A��\@���A��H@%�@�{B�8R@���A�A�HA�B���                                    Bxp�v�  �          A�z�@�ffA�G�@/\)A�\B��q@�ffA{�A  A��B��=                                    Bxp��V  �          A�p�@���A���@/\)A33B�u�@���Az{A�A���B�Q�                                    Bxp���  T          A�Q�@��
A��H?��H@��B��=@��
A�33AG�A�B���                                    Bxp���  �          A��@��A���@
=@�p�B�.@��A|z�A�RA�B��                                    Bxp��H  �          A���@��A�ff?�{@���B�ff@��A~�HA
=AۮB�p�                                    Bxp���  �          A���@�{A���?�  @��HB�k�@�{A�
A	�A�  B���                                    Bxp�Δ  �          A��H@��RA�z�?\@��B���@��RA�ffAffA�B�
=                                    Bxp��:  �          A��\@��HA�z�?��
@���B��@��HA�ffA�RAԸRB���                                    Bxp���  �          A�{@���A�ff?�  @�33B�=q@���A
=A
=qA�\)B��                                    Bxp���  �          A��@�(�A�?�G�@��B���@�(�A}A
{A��
B�Ǯ                                    Bxp�	,  �          A��H@��A�\)?�=q@�z�B��
@��A|z�A
�HAޏ\B�{                                    Bxp��  �          A��@�ffA���@(�@��B�� @�ffAx��A�
A�G�B�Ǯ                                    Bxp�&x  �          A�
==�G�A=�>�H�4�B��=�G�AyG��ʏ\���RB�33                                    Bxp�5  |          A������@�(��s�W
Bę����AHz��333�&B�z�                                    Bxp�C�  �          A��ÿ�p�@�{�~{L�B��H��p�A?
=�B�\�5G�B�#�                                    Bxp�Rj  �          A�=q�!G�@���{
=��B���!G�A<  �@���4(�B�.                                    Bxp�a  �          A�p��W
=@���{\)ffB����W
=A<Q��@���2(�B��
                                    Bxp�o�  �          A����7�@�G��yG�ǮB�aH�7�AB�\�;��,BΊ=                                    Bxp�~\  �          A���Fff@�(��mG��sB�=q�FffAN{�)p����B�{                                    Bxp��  �          A�{�W�@����j�\�n�\B�B��W�AQ��%G���RB���                                    Bxp���  �          A����:=q@���o\)�t=qB�(��:=qAP���*�\�33B��                                    Bxp��N  �          A����N{AG��k��m��B�p��N{ATQ��$�����B�G�                                    Bxp���  �          A�=q�,��A��s�
�q��B�\)�,��AY�+33��B�p�                                    Bxp�ǚ  �          A�=q�H��A�i���bG�B����H��Ab�H�  �z�B��                                    Bxp��@  �          A����w
=A���b�\�Z=qB��w
=Ac33�(�� �RB�k�                                    Bxp���  �          A�33�Mp�@�\)���B�G��Mp�A+\)�U��G�B�ff                                    Bxp��  �          A�ff�%�@�
=���B����%�A,���X���J�\B���                                    Bxp�2  �          A��\�A�@�ff��Q��3B��=�A�A)p��Z�H�L��B��f                                    Bxp��  T          A��\�~{@w���
=#�C�R�~{A#�
�Z�R�L�B�aH                                    Bxp�~  �          A��hQ�@�����R�C� �hQ�A(���\(��K�HB�                                    Bxp�.$  �          A��
�fff@�z���(��\C\)�fffA,���Y��H{B��                                    Bxp�<�  h          A�
=�\(�@b�\��\)z�C:��\(�A!G��`���SQ�BٸR                                    Bxp�Kp  �          A���H��@Z=q������C�H�H��A z��c��V33Bֽq                                    Bxp�Z  �          A��
�I��@a�����C�I��A"�\�b�H�T�B�z�                                    Bxp�h�  �          A����S33@W���ff{Cp��S33A (��c\)�UB؊=                                    Bxp�wb  �          A����/\)@Dz���G�C�R�/\)A���f�H�[G�B�B�                                    Bxp��  �          A���i��@G����� C)�i��A��h���a=qB�G�                                    Bxp���  �          A�Q��s�
@2�\��
=�qC�{�s�
A
=�dQ��Z
=B���                                    Bxp��T  �          A�=q�k�@8������  C�{�k�Az��c\)�Y{B�.                                    Bxp���  �          A���fff@6ff���\u�C��fffA�
�b�H�Y��Bݔ{                                    Bxp���  �          A�����
@K����aHCQ����
A  �^�R�S  B���                                    Bxp��F  �          A��R�u@(����B�Cp��uA{�f{�[G�B��                                    Bxp���  �          A�Q��}p�@^�R��  C
���}p�A ���\���O��B�                                      Bxp��  �          A����|��@8����33�RCٚ�|��A���c\)�Wp�B�q                                    Bxp��8  �          A��R�h��?�33���R��Cn�h��A���lz��d��B�                                    Bxp�	�  �          A�
=��{?8Q����Hz�C*=q��{@��H�x(��pB�G�                                    Bxp��  |          A��H����@����H� CE����A(��m�]
=B�
=                                    Bxp�'*  �          A�{�w
=@��p��\C�
�w
=A  �j�\�^�\B�Q�                                    Bxp�5�  �          A�{�|(�?&ff��Q���C*���|(�@����w\)�r\)B�Ǯ                                    Bxp�Dv  �          A�z��xQ�?�������{C+���xQ�@�\)�y��s��B�W
                                    Bxp�S  �          A�{���\?B�\����C*
=���\A��|���op�B�u�                                    Bxp�a�  h          A�z��y��?W
=���\ffC'�f�y��A���y��pQ�B�                                     Bxp�ph  �          A����
=��
=�����)C9)��
=@�
=��\)�y�B�33                                    Bxp�  �          A��H���>u��  p�C1  ���@��|���s  B��
                                    Bxp���  �          A������>L����ffaHC1O\���@�G��}���u{B�u�                                    Bxp��Z  �          A��R���\���������RCC�f���\@�G����\L�C ��                                    Bxp��   �          A�G����
��������RCJQ����
@����=q33C��                                    Bxp���  �          A�33�����J=q���CQ� ����@�G�����C.                                    Bxp��L  �          A�p���=q�����33��CX����=q@Dz�����C.                                    Bxp���  �          A�����zῑ����R��C@���z�@�����p��}��C ��                                    Bxp��  �          A��
��{�����  C@n��{@�p����\�~ffB�(�                                    Bxp��>  �          A�33����>�Q���=q��C/������@���33�p��B�.                                    Bxp��  |          A�p���G�?����\33C-Y���G�A ���~�\�op�B��                                    Bxp��  �          A����;�@*�H��G��C	���;�A�H�m���\�B��H                                    Bxp� 0  �          A�ff�^�R?�\)���H�3C�3�^�RA
=�r{�d
=B�u�                                    Bxp�.�  �          A����y��?�(���
=�C"���y��A	��v�H�i�HB��                                    Bxp�=|  �          A����s33?�������C(��s33A��tQ��e�RB��f                                    Bxp�L"  T          A����S33@(����C���S33Az��o��]�B�L�                                    Bxp�Z�  �          A���fff@(Q���
=��C���fffA�H�l���Z
=B���                                    Bxp�in  �          A����h��@A����\CL��h��A%G��k33�UG�B��)                                    Bxp�x  �          A�\)�i��@:=q��z�B�Cn�i��A"�R�i�V{B�u�                                    Bxp���  �          A�\)�[�@Z�H��{
=C\�[�A)��ep��P\)B���                                    Bxp��`  �          A�(��c33@\)��C���c33A1��`���I{B�k�                                    Bxp��  �          A�z��_\)@�p����u�C  �_\)A4���_��G
=B�\)                                    Bxp���  �          A�������@vff����p�Cp�����A0(��a��H�B�k�                                    Bxp��R  �          A�z���Q�@��
�����
B��R��Q�AK\)�Ep��)��B�\                                    Bxp���  �          A�z����@����z���B����AM�C
=�'{B�{                                    Bxp�ޞ  �          A�\)����@��R����  B�  ����AK��J�H�.�B�{                                    Bxp��D  �          A���:�H@�33��(��B���:�HAC\)�Y��<�B��                                    Bxp���  �          A�  �X��@�����G�ffB�p��X��AE��V{�9G�Bҽq                                    Bxp�
�  �          A����U�@��
��  L�B���U�AG\)�V�R�8�B��                                    Bxp�6  �          A�
=�w
=@��\���HL�B�B��w
=AI���R�H�4{B�
=                                    Bxp�'�  �          A�p��l��@�{���\)B�
=�l��AK��R�\�3=qB�p�                                    Bxp�6�  �          A����s33@�
=��z��=B��s33AO
=�O\)�/\)B�                                    Bxp�E(  �          A�  �y��@���(��B� �y��AQ��L���,p�B��                                    Bxp�S�  �          A����fff@�����33Q�B��fffAM��Q�233B�k�                                    Bxp�bt  �          A��
��p�@У����B�B�B���p�AU��Fff�%��Bس3                                    Bxp�q  �          A��
����@��
��(��3B������AS\)�G��'(�B�                                    Bxp��  �          A�{��z�@љ���  8RB��3��z�AU��F{�%G�B�aH                                    Bxp��f  �          A����p�@�=q�����B�����p�APz��L(��,  B�p�                                    Bxp��  �          A����
@��R��  ffB�\)���
AO��Mp��-Q�B�G�                                    Bxp���  T          A�Q����
@�������B�Ǯ���
AO��O33�.G�B�8R                                    Bxp��X  �          A�����Q�@�=q����r�B����Q�AaG��6=q���B��                                    Bxp���  �          A�  ���@�G���ff�w�B����A^{�9���Bۅ                                    Bxp�פ  �          A��
���\@�p���33�z�HB��=���\AY���=p����B���                                    Bxp��J  �          A��
����@޸R����vC s3����AY��:�R�33B�L�                                    Bxp���  �          A�=q����A���|z��effB�������Ag\)�&�H�=qB߅                                    Bxp��  �          A�p���\)A�R�y�^p�B�q��\)Ar�R�{���
B۽q                                    Bxp�<  �          A�p���(�@�Q���=q�s(�B�B���(�Aa���6�H��\B�Ǯ                                    Bxp� �  �          A�=q���@��
���
�vG�C�3���AY��>=q�B��)                                    Bxp�/�  �          A������@�=q��8RC�����AN�R�L���(�\B��H                                    Bxp�>.  �          A�\)��
=@�������C:���
=AR�\�L���'�RB��                                    Bxp�L�  �          A�p���  @љ���z�{B��q��  AY��HQ��#(�B�L�                                    Bxp�[z  �          A�p����
@�{��z��C�����
AO��R{�-
=B�
=                                    Bxp�j   �          A����z�@��������C
��z�AM��UG��/�\B߀                                     Bxp�x�  �          A����{@����#�C����{AM��V{�033B��                                    Bxp��l  �          A�������@��\��z�B�C�����APQ��U��/�B�G�                                    Bxp��  �          A�\)����@�������
C�����AO��W\)�/p�B�Q�                                    Bxp���  �          A�{��33@�����  �RCh���33AM���Z�\�1�HB�33                                    Bxp��^  �          A�p���=q@�=q����\CW
��=qAJ�R�[��3�B�                                     Bxp��  �          A����@�\)��  � C�H���AV{�Q���)=qB�L�                                    Bxp�Ъ  �          A�  ���@׮��(�B�B��f���A_��H��� ffBܔ{                                    Bxp��P  �          A�z�����@r�\���8RC�{����A6�R�`Q��>\)B��
                                    Bxp���  �          A�=q��33�L����
=� C5�q��33A33�����e��C �{                                    Bxp���  �          A�z���\)�#�
����G�C5aH��\)Az�����e�C \                                    Bxp�B  �          A�ff��Q��G���p��C4���Q�AG������e{C                                     Bxp��  �          A��R��p�    ��{�qC4���p�A\)����d�B�aH                                    Bxp�(�  �          A�p��Ϯ=L������u�C3���ϮA������c�RB�k�                                    Bxp�74  �          A�G���(������R�C5���(�A�\�����f  B�aH                                    Bxp�E�  �          A�z����ÿ(����{W
C9������@�p���  �k��C n                                    Bxp�T�  �          A�  ��ff�:�H��ff.C:����ff@�(���z��nG�B��                                    Bxp�c&  �          A���=q�����z�C9T{��=qA (���p��k�RB�\)                                    Bxp�q�  �          A�����\)=#�
���R{C3����\)A�
��  �bB��R                                    Bxp��r  �          A��H����?!G����\C.c�����A33�}���\�B�#�                                    Bxp��  �          A��R�ƸR@c�
����=C(��ƸRA7\)�dz��>�B��                                    Bxp���  �          A���Å@Y�����R�RC޸�ÅA6{�g\)�@�HB�p�                                    Bxp��d  �          A����=q@1G���=q�HC(���=qA,���k33�E=qB�{                                    Bxp��
  �          A��\����?s33����B�C+�f����A��yG��W�\B��\                                    Bxp�ɰ  �          A�Q���33�c�
������C:޸��33@�
=����effC�)                                    Bxp��V  �          A�ff��
=��������C<����
=@�z���G��eG�C=q                                    Bxp���  |          A�Q����ÿ����=q�C8+�����A Q���p��`��C5�                                    Bxp���  �          A�Q���\)�s33���
� C;@ ��\)@�G����H�d
=C�=                                    Bxp�H  �          A�(����H����������C<�R���H@�(����\�c��C�q                                    Bxp��  �          A�z����ÿ����G�
=C=h�����@��H��G��e33C��                                    Bxp�!�  �          A��H� ���,(����aHCFs3� ��@�Q����H�nQ�C=q                                    Bxp�0:  �          A�
=��������HffCD+����@�
=���\�l�
C�=                                    Bxp�>�  �          A��R��  �{����
=CC�q��  @�����ff�m�C޸                                    Bxp�M�  �          A�z���{�*=q��G���CG�3��{@�p���Q��s33CQ�                                    Bxp�\,  �          A�z���33�������k�CG���33@�ff�����t�C	                                    Bxp�j�  �          A�z���=q�#�
��{G�CI@ ��=q@�ff��Q��y33C��                                    Bxp�yx  �          A�z��	p�������\(�C;  �	p�@�R���[C�                                    Bxp��  �          A������J=q��p��|{C9���@�z����V\)C�{                                    Bxp���  
�          A����\)��Q���33��C<
=�\)@�z����R�^  C
�
                                    Bxp��j  �          A�ff�Q쿞�R����
=C<� �Q�@�z���33�_�RC
8R                                    Bxp��  �          A�{��
=��p���{�=C>����
=@�
=���\�d{C	�
                                    Bxp�¶  �          A�����33���R��  �C>�q��33@�R��z��d�C	n                                    Bxp��\  �          A����{�W
=���  C:��{@��������^33C��                                    Bxp��  �          A�����R�
=q������C7�\��R@�\)�33�Y�\C��                                    Bxp��  �          A�G������  ��{��C5�{���Ap����Y\)CT{                                    Bxp��N  �          A��R���
�.{��p��{C58R���
A{�}���X(�C5�                                    Bxp��  �          A��\�G���\���\#�C7�H�G�A z��~�\�Y��C&f                                    Bxp��  �          A����=q�k����\�HC5���=qA���|z��V�CxR                                    Bxp�)@  �          A�33����>u��Q�G�C2E����A���{�
�T�Cs3                                    Bxp�7�  �          A��H��ff>8Q���\)�C2����ffA��zff�S�C\)                                    Bxp�F�  |          A�
=��p�?s33���H��C,����p�A(��p(��N�RC �R                                    Bxp�U2  T          A�ff���?��\��z�k�C,���A���n�H�NffC !H                                    Bxp�c�  �          A��H���
=����
=C3����
A	G��w\)�T�C�                                     Bxp�r~  �          A����=q���
���B�C4)�=qA��xz��Q�C��                                    Bxp��$  �          A�ff��H��=q��
=�HC5����HA�
�yp��S�C��                                    Bxp���  �          A�  ��H�.{����aHC5(���HA�y���T�C\)                                    Bxp��p  �          A��
�
=��
=��\)33C6��
=A=q�{
=�V��C0�                                    Bxp��  �          A���녿!G���\)� C8s3��@��R�|���X�C�{                                    Bxp���  �          A���p��z���G���C8��p�A (��|  �X�\CJ=                                    Bxp��b  �          A����
����Q��{C7����
@�
=�z{�W
=C�3                                    Bxp��  �          A�����þ\����C6�
���A���w\)�T\)C��                                    Bxp��  �          A�=q����#�
���H�C8^����@��H�x  �V(�C�=                                    Bxp��T  �          A���	녾�������}Q�C7��	�@�p��tQ��R�C	s3                                    Bxp��  �          A�(��Q��R��(��~�C8+��Q�@��\�vff�T\)C	n                                    Bxp��  �          A�z��
ff�p������|��C::��
ff@�G��xz��V33C
��                                    Bxp�"F  �          A�{�(��n{��
=�{(�C:{�(�@�Q��v�H�T�
Cff                                    Bxp�0�  �          A���׿�\�����z��C7O\���@���r�R�P��C
+�                                    Bxp�?�  �          A�������  ���
�x�\C5����A ���o33�L�\C
!H                                    Bxp�N8  �          A�������&ff��  �s{C7�3���@�33�o
=�Lz�C�f                                    Bxp�\�  �          A����=q�=p���{�yG�C8���=q@����s33�QffCG�                                    Bxp�k�  �          A��
��
�s33�����{(�C:8R��
@���v=q�T��CL�                                    Bxp�z*  �          A�  ��׿������}��C:����@�Q��xQ��W�C
�                                    Bxp���  �          A�=q��\�QG���
=G�CL�\��\@������v�HC�\                                    Bxp��v  �          A�p��G�������{�=C==q�G�@�z��~�H�\(�C
ff                                    Bxp��  �          A�����R������HǮC;�R��R@�(��~�\�[��C�3                                    Bxp���  �          A�33�=q�xQ����ffC:�
�=q@�\)�{
=�X  C	Y�                                    Bxp��h  �          A��\��ÿ
=q��z��~��C7�
���A ���u���RQ�C�\                                    Bxp��  �          A����
{�E���=q�}p�C9��
{@�=q�v�H�S��C	��                                    Bxp��  �          A�p��z�5��Q��z�C8�f�z�@�Q��s
=�Q\)C
��                                    Bxp��Z  �          A�
=�(����\�����v�HC:n�(�@���r�\�Q�RC�=                                    Bxp��   �          A����
=��z����R�w  C<�3�
=@�\�u���U�\C�H                                    Bxq �  �          A��H�(���ff��G��y
=C>
=�(�@߮�w��XQ�CaH                                    Bxq L  �          A�
=�
=�������y33C@:��
=@�\)�z�\�[�RC@                                     Bxq )�  �          A���녿�
=����w\)C@@ ��@ָR�{
=�Z�C��                                    Bxq 8�  �          A�
=��� ����33�xffC@����@��
�z�H�\Q�C�\                                    Bxq G>  �          A�
=��ÿ��H���H�~  CAJ=���@أ��}p��_�C�\                                    Bxq U�  �          A��H�G��   �����}z�CA� �G�@�\)�}G��_��C\                                    Bxq d�  �          A��R�33��{��(��|(�C@k��33@ڏ\�{33�]=qC�                                    Bxq s0  �          A�ff�=q�����  �|��C@���=q@ٙ��{
=�]C�R                                    Bxq ��  �          A��\�������
�{C@8R��@�33�zff�\�\C�                                    Bxq �|  �          A��H�33��33���{C?^��33@��
�{��]z�C�                                    Bxq �"  �          A��H��Ϳ޸R��  �{Q�C?xR���@޸R�y�[
=Cٚ                                    Bxq ��  �          A�z��(���ff�����x{C?�H�(�@ڏ\�w�
�Yp�C�                                    Bxq �n  �          A��R�����p����H�~�\C>����@�  �x���ZffC�                                    Bxq �  �          A�������(���G��y
=C?#���@�ff�x  �Y{Cp�                                    Bxq ٺ  �          A��H��\��{�����s�\C=�R��\@�ff�tQ��T{C�\                                    Bxq �`  �          A�\)���#33��G��e�CBY���@��
�tQ��SQ�C��                                    Bxq �  �          A�Q��#
=�+���33�c�CB�#
=@�Q��u��R�C��                                    Bxq�  �          A�ff�\)��  ��\)�iz�C=�q�\)@ָR�qp��M�C
=                                    BxqR  �          A�Q����.{����i��CC�)��@�z��yp��W�\CǮ                                    Bxq"�  �          A�z��\)�   ��  �j�
CBh��\)@��
�xz��V
=CǮ                                    Bxq1�  �          A��\�(��(�����jz�CB
=�(�@�p��w�
�U(�C�                                    Bxq@D  �          A�z����K����\�fCE���@��R�{33�Y�C�                                    BxqN�  �          A�ff�\)�HQ�����hffCE޸�\)@�G��{�
�Zz�CJ=                                    Bxq]�  �          A�z��G��I�����\�fCE� �G�@�  �z�H�YG�C�                                    Bxql6  �          A�Q�����?\)��z��f��CD�����@�(��yp��W�
C=q                                    Bxqz�  �          A�ff�Q��@  �����h{CE��Q�@�p��zff�X�RC��                                    Bxq��  �          A������C�
��G��h�\CEs3��@�z��{\)�Y��C�H                                    Bxq�(  �          A�ff����>�R��{�kG�CET{���@����|  �Z�
C��                                    Bxq��  �          A��R���9�����H�lCE
=��@�z��|���[�C�                                    Bxq�t  �          A�ff����K������mffCF�)���@���~�H�^�C��                                    Bxq�  �          A�33�p��E���\�j��CE�
�p�@�\)�}���[G�C&f                                    Bxq��  �          A�G����AG���=q�i�CEJ=��@����|z��Y��CL�                                    Bxq�f  T          A���"�H�����z��e(�CA��"�H@���t���Pz�C�{                                    Bxq�  �          A�
=�)��R��(��_{CA(��)@���p���L{C�=                                    Bxq��  �          A�33�"�R�S33��p��b�CE�3�"�R@�(��y�VffC!H                                    BxqX  �          A���%��N{��\)�a  CEY��%�@�ff�y��T��C(�                                    Bxq�  �          A������j=q����f�
CH�H��@��R��(��]C�\                                    Bxq*�  �          A��
��
�}p���33�eG�CJ���
@�{��
=�_�\C�                                    Bxq9J  �          A�\)�
=�Z=q���\�d��CF��
=@�(��|z��Y��C��                                    BxqG�  �          A���   �w
=��=q�b��CI!H�   @�\)���\33C�                                    BxqV�  �          A�Q�� Q��|(���ff�b33CIz�� Q�@���=q�\�\C��                                    Bxqe<  �          A��\� Q��tz���
=�c(�CH�)� Q�@��\��=q�[�C)                                    Bxqs�  �          A���� ���l����G��cz�CH:�� ��@�ff��  �Z�C�H                                    Bxq��  �          A��R� ���b�\��p��d{CGaH� ��@���~�R�Y��C�                                    Bxq�.  �          A�z��   �h����G��d�CG�q�   @����\)�Z�RC8R                                    Bxq��  �          A��R�!G��z=q���R�b
=CI33�!G�@�Q���Q��[�RC�\                                    Bxq�z  �          A���&ff�l(���  �_G�CG���&ff@���}���W
=C�)                                    Bxq�   �          A��\�&�\�a�����_\)CF�R�&�\@����{\)�U�\C�                                    Bxq��  �          A�Q��&�\�b�\��33�_  CF��&�\@�  �z�H�UffC=q                                    Bxq�l  �          A�z��%�b�\����_��CF�)�%@�G��{��V
=C��                                    Bxq�  �          A����'��xQ����\�\CHO\�'�@��|z��V�HC�=                                    Bxq��  �          A���'33��z����R�\=qCI�)�'33@�
=�~�H�X��C�                                    Bxq^  �          A�\)�)���ff��ff�Z�RCI���)�@���~�R�X(�C5�                                    Bxq  �          A���+33���
��{�Y�\CI��+33@�
=�}���V=qC5�                                    Bxq#�  �          A�\)�(Q���p����R�[�CI���(Q�@�
=�~�H�XffC�
                                    Bxq2P  �          A���,���w
=��
�YQ�CG���,��@�{�{
=�S\)Cn                                    Bxq@�  �          A���+\)������(��Y��CH�H�+\)@���|���U�C�\                                    BxqO�  �          A���)�����
�Z
=CIk��)@�{�}p��W�C&f                                    Bxq^B  �          A��R�,����  �}���X  CHT{�,��@�Q��z{�S�
C0�                                    Bxql�  �          A�  �*�\�Y�����[��CE��*�\@�33�v�H�Q33CY�                                    Bxq{�  �          A��,���<���~�\�[\)CCG��,��@��R�rff�L\)C.                                    Bxq�4  �          A�\)�)��ff�����_C@u��)�@�=q�o��I�HC=q                                    Bxq��  �          A���$�Ϳ�  ��\)�f�C=�f�$��@�Q��o33�H��C�                                    Bxq��  �          A����%�,�������b(�CB�{�%@��H�t  �O  C�=                                    Bxq�&  �          A�33�'��n�R�~=q�\
=CG�{�'�@���xQ��T�\C@                                     Bxq��  �          A��H�'��k��}���[�CG^��'�@��\�w\)�T
=C&f                                    Bxq�r  �          A����)G��c�
�}G��[G�CF���)G�@��v{�R33C�f                                    Bxq�  �          A�33�)��qG��|���Y�HCG�=�)�@�\)�w33�S�C޸                                    Bxq�  �          A�(�����Q����R�^{CPk���@xQ����
�f\)CB�                                    Bxq�d  �          A��R��H������
�[ffCP{��H@r�\��G��d=qC�                                    Bxq
  �          A�ff�  ���H��  �\z�CS�
�  @U�����k��C �H                                    Bxq�  �          A����$����z��}G��W�HCN}q�$��@w
=����_(�Cu�                                    Bxq+V  �          A��\�$Q���p���
�[(�CLxR�$Q�@��H�����]
=C�                                    Bxq9�  �          A�Q��&ff���\�}��X33CL��&ff@��
��(��\=qCaH                                    BxqH�  �          A�{�)���  �|(��W�CK��)�@�z��|���XQ�Cp�                                    BxqWH  �          A�ff�*�\��p��|z��W=qCJ��*�\@�
=�|(��W  CE                                    Bxqe�  �          A�{�,  �����{33�VffCI���,  @����y��T�C\                                    Bxqt�  �          A����1�����y���R��CH� �1�@���w��Pz�CxR                                    Bxq�:  �          A����3��~{�x���RQ�CGz��3�@�  �uG��N�C�                                    Bxq��  �          A��H�6�R�x���w33�P  CF�=�6�R@����s33�KQ�CW
                                    Bxq��  �          A�G��4Q���33�x(��PQ�CI)�4Q�@���w�
�P  C��                                    Bxq�,  �          A�33�9p���p��q�HCK�9p�@n{�w33�O
=C"33                                    Bxq��  �          A�G��6�\��33�r=q�I�
CL
�6�\@e�y��Q�
C"��                                    Bxq�x  �          A�G��/33����z�H�S�\CJ�H�/33@��H�{�
�T��Cff                                    Bxq�  
�          A�\)�(  ��ff��=q�Z=qCJ���(  @��
���YQ�CE                                    Bxq��  �          A���#33��33��z��Zz�CN�{�#33@��H�����`\)C(�                                    Bxq�j  �          A���+������{\)�T��CL!H�+�@���~=q�X\)CǮ                                    Bxq  �          A�
=�5���G��xQ��Q  CG���5�@��R�uG��MffCaH                                    Bxq�  �          A�
=�4z����\�vff�N�RCJ)�4z�@�ff�x(��PC��                                    Bxq$\  �          A����+
=��  �{��Up�CK�3�+
=@���}��XQ�Cc�                                    Bxq3  �          A�\)�.�R��\)�z=q�R��CKn�.�R@��R�|���U�
C��                                    BxqA�  	�          A��3
=�����y���Q{CJ
=�3
=@���z=q�R  C��                                    BxqPN  T          A��
�4����z��v�\�M�\CKc��4��@~{�z�R�Rz�C �f                                    Bxq^�  "          A�  �8  �����qG��G
=CM�f�8  @P  �{��S  C$5�                                    Bxqm�  T          A�  �8�����R�qG��F�CMO\�8��@S�
�{
=�RG�C$                                      Bxq|@  "          A�G��8(���ff�r=q�I��CKB��8(�@r�\�w��OC!Ǯ                                    Bxq��  �          A�\)�/
=��ff�x���QQ�CLY��/
=@�Q��}G��V�\C�f                                    Bxq��  "          A�  �$z���=q��33�[�CM��$z�@�{��  �]C��                                    Bxq�2  �          A�(��,������{��R��CM���,��@{������Y�C �                                    Bxq��  T          A���4�����
�up��L=qCLc��4��@p���{��S�\C!�\                                    Bxq�~  T          A�
=�@����\)�h���?��CKu��@��@P  �q��IC$�f                                    Bxq�$  	�          A��I����{�e���:�CIh��I��@Y���l���BQ�C$��                                    Bxq��  "          A���>�\��(��pz��Fz�CIB��>�\@����s33�I��C!Y�                                    Bxq�p  �          A��� Q���33��33�\��CN��� Q�@�
=��33�a��C#�                                    Bxq   T          A��
�&�H���}���V��CNff�&�H@�Q�����]�\C�q                                    Bxq�  �          A��
�'�����|���U33COL��'�@s33��=q�^p�C {                                    Bxqb  �          A�(��0����=q�|���T33CIQ��0��@��R�z�H�R
=C�                                    Bxq,  �          A�  �3���33�{��R��CH\�3�@���x  �N�
C��                                    Bxq:�  �          A��
�2�H����z�R�RffCI��2�H@���yG��P�C^�                                    BxqIT  �          A����9���Q��w33�LffCI33�9�@�(��w�
�M{CW
                                    BxqW�  �          A�
=�=�����r�H�G=qCJ�{�=��@x���w��L��C!��                                    Bxqf�  T          A�p��<Q���33�t  �G�RCKs3�<Q�@q��z{�N�C"33                                    BxquF  �          A��
�?33����r�R�E��CK!H�?33@o\)�y��L�RC"�H                                    Bxq��  �          A���A����p(��BCKh��A@b�\�x  �Kp�C#��                                    Bxq��  T          A�\)�2ff��\)�|z��R  CJ�q�2ff@�z��~{�S�
C��                                    Bxq�8  "          A���������  �d�HCK�
��@�
=��{�_�RC�                                    Bxq��  
�          A����&�R������R�[��CLs3�&�R@�{�����\=qCǮ                                    Bxq��  �          A���+���(���
=�W\)CLp��+�@��R��  �Y�RCn                                    Bxq�*  �          A�{�-G�������(��U(�CL��-G�@�������Yp�Cu�                                    Bxq��  "          A�G��+33����}p��S33CO(��+33@s33�����\C u�                                    Bxq�v  �          A���/�����zff�O�CN�\�/�@l����G��Yz�C!c�                                    Bxq�  �          A����/������y��O=qCN��/�@i����G��Y�RC!�)                                    Bxq�  "          A����3
=��G��u�J�\CO\)�3
=@S33�����X(�C#�
                                    Bxqh  T          A�
=�5������v�R�K��CM��5��@o\)�~{�T{C!                                    Bxq%  "          A����333����z�\�Pp�CKB��333@�G��|���SQ�C
=                                    Bxq3�  �          A��\�2{��ff�{��Q�
CJ�H�2{@�p��|���S\)CQ�                                    BxqBZ  "          A��\�3�����x  �M�CLu��3�@}p��}G��T33C �{                                    BxqQ   �          A����:�\��
=�q��F  CM#��:�\@[��z�\�P��C#�
                                    Bxq_�  �          A��H�<������o�
�D
=CL���<��@W��y���O{C$\                                    BxqnL  
�          A��H�?�
�����lz��@p�CM:��?�
@HQ��x(��M�C%aH                                    Bxq|�  �          A����?������m��@�CM@ �?�@J=q�x���M�
C%=q                                    Bxq��  T          A�\)�=����{�p���DffCL���=��@]p��z{�NC#�R                                    Bxq�>  �          A�33�8����
=�uG��I��CLJ=�8��@s33�{�
�Q33C!��                                    Bxq��  �          A���8Q�����w�
�K��CKff�8Q�@��\�|(��P��C z�                                    Bxq��  T          A��3\)�����y��M�
CM�3�3\)@s�
�����V�\C!=q                                    Bxq�0  
�          A��
�*�R���������Vz�CMǮ�*�R@�Q����R�[�\C:�                                    Bxq��  T          A��
�,z���
=���Y{CJ��,z�@�p����R�V�\Cz�                                    Bxq�|  "          A��2=q��
=��{�U\)CH�R�2=q@����|z��P�C�                                     Bxq�"  �          A���B�H�����o33�A�\CKn�B�H@c�
�w33�JffC#��                                    Bxq �  �          A��
�AG���\)�p���CG�CKk��AG�@i���x  �Kp�C#+�                                    Bxqn  "          A���=p����H�u�H�
CJ@ �=p�@����x���Lz�C ��                                    Bxq  �          A��?�
��(��s\)�FffCJ!H�?�
@�G��w33�JC!\)                                    Bxq,�  �          A���?\)��G��t  �G33CI�\�?\)@�z��w
=�J�\C �f                                    Bxq;`  �          A����@Q������rff�E��CJ+��@Q�@�  �v�\�JQ�C!�)                                    BxqJ  �          A�  �@����
=�t(��F�CIff�@��@��R�v�\�I�C ��                                    BxqX�  �          A���A����
�s\)�F(�CH�
�A�@����t���G�C �{                                    BxqgR  T          A���=����z��w�
�K�HCG:��=��@��\�tz��H=qC�{                                    Bxqu�  �          A���=�����R�uG��I  CI���=��@�Q��w33�KQ�C 33                                    Bxq��  T          A�p��=�����tz��H(�CJ�=�=�@�=q�xQ��L�\C!                                      Bxq�D  �          A����<z�����u��Iz�CJ5��<z�@�ff�x���L�RC \)                                    Bxq��  �          A��=p����\�r�H�E��CL@ �=p�@j=q�z�\�N��C"��                                    Bxq��  �          A�(��?������tz��F�HCJB��?�@��\�xQ��K(�C!33                                    Bxq�6  �          A�=q�A��33�tz��F��CHǮ�A@��H�u���H�C G�                                    Bxq��  "          A�z��D������t(��F
=CGT{�D��@�=q�r�H�D�C��                                    Bxq܂  �          A�z��G33����r�H�DCFB��G33@�ff�p(��ACJ=                                    Bxq�(  �          A�ff�Hz������q��C��CE���Hz�@���n�\�@(�CG�                                    Bxq��  �          A�  �K��p  �o��A�
CDs3�K�@�z��j=q�<33C��                                    Bxq	t  �          A��
�G��y���q��D�CEaH�G�@�33�mp��?�HC�q                                    Bxq	  �          A��
�F�R�s33�r�R�E�RCE  �F�R@�
=�mG��?C33                                    Bxq	%�  T          A��E���s�
�s��F�CE!H�E��@���n{�@��C                                      Bxq	4f  �          A��
�F�\�u�r�H�E��CE.�F�\@�{�m�@33CG�                                    Bxq	C  �          A��
�C��w��u��H\)CE�
�C�@���p  �B��CǮ                                    Bxq	Q�  
�          A��C
=�x���up��H��CE�3�C
=@���pQ��C  C                                    Bxq	`X  �          A����Dz��o\)�tQ��GCD��Dz�@��\�n{�@�C�=                                    Bxq	n�  �          A���C��p  �uG��H��CE
=�C�@�33�n�H�A�CY�                                    Bxq	}�  �          A�  �C
=�qG��v�\�I�CE+��C
=@�(��p(��Bz�C(�                                    Bxq	�J  �          A����C
=�tz��w��I�
CEff�C
=@�(��q���C�C33                                    Bxq	��  �          A�
=�EG��l(��w��I
=CD���EG�@���pQ��A{C�R                                    Bxq	��  T          A����D���p���w33�H��CD���D��@�p��p���A��C=q                                    Bxq	�<  �          A��R�D���~{�v=q�G�CE���D��@�ff�q���B�C\                                    Bxq	��  
�          A����C�
�{��w33�H��CE���C�
@����r{�CQ�C�                                    Bxq	Ո  �          A�=q�<  �{��{�
�O33CFz��<  @��u��H�C0�                                    Bxq	�.  �          A�Q��AG����R�v�R�IQ�CG:��AG�@����t(��FffCn                                    Bxq	��  �          A��R�@����33�w�
�I�RCG޸�@��@�{�v=q�G�C�3                                    Bxq
z  	�          A��R�@����33�w��IffCG�
�@��@��u��G�C�                                    Bxq
   �          A�Q��B�\��{�t���G�CH��B�\@����tz��F�C�
                                    Bxq
�  �          A��H�A���z=q�yG��K�CE��A��@���s��D�C
                                    Bxq
-l  �          A�33�@���tz��{
=�L�\CE���@��@�Q��tQ��E
=Cp�                                    Bxq
<  �          A�
=�E������vff�G�CF{�E�@��r=q�C
=C5�                                    Bxq
J�  �          A�33�D���g��x���J  CDk��D��@��H�p���A33C��                                    Bxq
Y^  �          A����G\)�w
=�v�R�G{CE33�G\)@��\�p���@�C�{                                    Bxq
h  �          A����D���j=q�y���J33CD���D��@��H�q�A�\C��                                    Bxq
v�  �          A�G��J{��Q��p���A=qCG�f�J{@��\�q���B�C!�                                    Bxq
�P  �          A���?33�,(���z��R�RC@��?33@���o��?Q�C�=                                    Bxq
��  �          A��
�@���1���  �Q�C@�R�@��@ə��o��>�HCk�                                    Bxq
��  T          A�{�?��!������S=qC?���?�@ҏ\�n�H�>33C33                                    Bxq
�B  �          A�=q�E��33��  �P�\C:u��E@�R�c��2G�C޸                                    Bxq
��  �          A�{�C���{�����R�C933�C�@�Q��bff�1\)C�)                                    Bxq
Ύ  �          A�Q��>{�S33���H�RQ�CC��>{@�p��up��D�C}q                                    Bxq
�4  �          A����=��6ff����TG�CA��=�@��
�s\)�A�HCǮ                                    Bxq
��  �          A��R�;\)��R�����W�
C?���;\)@�G��r�H�A�C޸                                    Bxq
��  �          A��H�9�����R�ZffC>+��9@�R�qp��?\)C&f                                    Bxq	&  �          A����<zῼ(���ff�YffC;��<z�@�ff�k��9(�C�{                                    Bxq�  �          A��H�;��L�������[  C7�f�;�A���f�R�4p�C�                                    Bxq&r  �          A��H�?33���H��G��W(�C9���?33@��H�g��5\)C�q                                    Bxq5  �          A����>ff�B�\�����SG�CBO\�>ff@�{�tz��B��C��                                    BxqC�  �          A����;�������ff�YG�C=s3�;�@�G��o��=\)C�                                    BxqRd  T          A�33�>�H�Q���33�V33C>)�>�H@�=q�o33�<��CY�                                    Bxqa
  �          A�G��=���4z������U=qCA^��=��@θR�tz��B33Ch�                                    Bxqo�  �          A�p��A������  �R��C?s3�A�@ָR�p  �=(�C                                    Bxq~V  �          A���D���3�
���\�Op�C@�H�D��@���p���=��C�
                                    Bxq��  �          A���I��N�R�}��Jz�CBc��I�@��\�p���=z�C#�                                    Bxq��  �          A��
�@���������\�N��CF�{�@��@����{\)�HQ�C^�                                    Bxq�H  �          A��
�@  �����{�MCG�R�@  @�\)�}G��J�CxR                                    Bxq��  �          A�  �@z���\)���L�HCHp��@z�@�33�}��J��C                                    Bxqǔ  �          A���F�R�dz��~{�Kp�CD\�F�R@�=q�t���A\)Cٚ                                    Bxq�:  �          A�\)�C
=��Q��r{�@=qCNB��C
=@G
=���O(�C%�3                                    Bxq��  �          A����.=q����c��0Q�C^��.=q�(���Q��f��C78R                                    Bxq�  �          A�=q�0���(z��U���$��C_�H�0�׿�33��\)�bQ�C<�                                     Bxq,  �          A�p��.=q�(���T���%{C`!H�.=q�ٙ���G��c��C<�f                                    Bxq�  �          A�\)�+\)�6=q�D����
Cb�+\)�<(���(��`G�CC^�                                    Bxqx  �          A�  �)��=��;����Cd��)��g�����]p�CF�{                                    Bxq.  �          A����*�R�3\)�C
=��HCbn�*�R�7
=���\�_��CC�                                    Bxq<�  �          A����+33�:�\�;\)�
=CcxR�+33�`  ��33�\�\CF�                                    BxqKj  �          A�  �,  �?��6�R�\)Cd��,  �{������Z33CH\                                    BxqZ  �          A��1p��Bff�.{��Cc���1p�����{��SQ�CIu�                                    Bxqh�  �          A�p��+��#��N�H�$�C_���+��У���33�b{C<��                                    Bxqw\  �          A�(����=p��E��{Ce�f���Tz���(��g�CFn                                    Bxq�  �          A��R���-G��TQ��(�Cc\)�����H��(��l��C?!H                                    Bxq��  �          A�33�+\)�6=q�E���CbǮ�+\)�<����(��`Q�CCh�                                    Bxq�N  �          A����*{�333�J{�{Cbz��*{�'���\)�b��CA�)                                    Bxq��  T          A����4(��:{�8  ��Ca��4(��fff�33�U�HCE�R                                    Bxq��  
�          A��H�8���5p��7��ffC`z��8���W
=�|���RCD8R                                    Bxq�@  
�          A��R�9��4Q��8Q��{C`@ �9��Q��|z��R��CC��                                    Bxq��  
�          A����1��0(��B{�(�C`�R�1��/\)��G��Z�CA��                                    Bxq�  �          A�{�*�R�/��N=q� p�Ca���*�R��\��(��c��C@#�                                    Bxq�2  �          A�  �.�H�.=q�K��{C`�f�.�H�z����R�`33C?�R                                    Bxq	�  �          A����1��.�\�H(��G�C`�
�1������\)�]z�C@�                                     Bxq~  �          A����0z��1���C�
�
=Ca.�0z��0����Q��\G�CB\                                    Bxq'$  �          A�\)�3
=�)p��I��Q�C_n�3
=�����R�\z�C>�                                     Bxq5�  �          A����4���(���G
=�33C_��4�������\)�Z(�C?                                    BxqDp  
�          A��\�4���,  �C�
���C_���4���{���H�Y\)C@T{                                    BxqS  
�          A��R�1��3
=�@Q��33Ca+��1��=p���33�Z�CB�f                                    Bxqa�  T          A�Q��+��:�H�<����RCck��+��^�R����\�
CE�q                                    Bxqpb  
�          A�=q�0���4Q��=��Ca���0���G
=���\�Y��CC��                                    Bxq  T          A�\)�7��!G��L�����C]L��7��������
�Y�
C;��                                    Bxq��  
�          A�G��4(��$z��M��� z�C^ff�4(���G����H�\�\C<�)                                    Bxq�T  
�          A�G��8���0���>�\��HC_� �8���9�����U\)CB!H                                    Bxq��  �          A�p��>{�,  �=G��C^!H�>{�,���|(��Q�C@�\                                    Bxq��  �          A�  �=��(  �D(��(�C]���=�������T
=C>�{                                    Bxq�F  "          A��
�<z��'��DQ���\C]���<z��\)���Tp�C>�=                                    Bxq��  �          A����B�\�"ff�B{�C[ٚ�B�\��\�z�H�OQ�C=�=                                    Bxq�  T          A�G��Dz��!�?���C[xR�Dz���xz��M�C=��                                    Bxq�8  T          A�33�EG��"�H�=�G�C[�\�EG��{�w��L
=C>33                                    Bxq�  T          A�\)�F=q�$���;\)��C[�3�F=q����v�\�J�RC>��                                    Bxq�  �          A���D���(  �:�H�\)C\xR�D���%�w�
�KC?޸                                    Bxq *  "          A����F�H�'
=�9���33C\�F�H�%��v=q�I�C?��                                    Bxq.�  T          A��H���(���6�\�ffC\)�H���2�\�t���G��C@��                                    Bxq=v  T          A���Hz��)G��5���
C\0��Hz��5�t(��G��C@Ǯ                                    BxqL  �          A����Ip��)G��4���	�HC\��Ip��8Q��s\)�F��C@�H                                    BxqZ�  "          A�p��J�\�)�2ff��C[�R�J�\�>�R�q���E{CA:�                                    Bxqih  
�          A��H�J{�*ff�0����C\#��J{�Dz��p���D��CA��                                    Bxqx  �          A��Jff�&ff�/��=qC[k��Jff�9���mp��C=qC@�                                    Bxq��  "          A��
�Lz��&ff�-��C[!H�Lz��>{�k�
�AG�CA\                                    Bxq�Z  �          A�p��M��$  �,���G�CZ�=�M��7
=�i��?�
C@��                                    