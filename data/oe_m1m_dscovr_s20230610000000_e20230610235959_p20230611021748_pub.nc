CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230610000000_e20230610235959_p20230611021748_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-06-11T02:17:48.794Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-06-10T00:00:00.000Z   time_coverage_end         2023-06-10T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�)��  "          @�{?�����@k�BzC���?���  @N{BL��C��R                                    Bx�)�&  �          @�?��
�Q�@8��B0�RC�c�?��
�AG�@{BG�C�E                                    Bx�)��  "          @�33@   �8Q�?�ffA�p�C�e@   �O\)?��At��C��                                     Bx�)�r  
�          @��?��
�$z�@#33B(�C�T{?��
�G�?���A�{C��
                                    Bx�)�  
�          @�{?�Q��z�@\(�B\C�}q?�Q��,��@7�B,��C��                                    Bx�)��  
�          @�?����@O\)BL  C��q?����4z�@(��B��C���                                    Bx�)�d  
�          @��
?�p��z�@Mp�BN�\C��?�p��2�\@'
=BQ�C��{                                    Bx�*
  "          @�  ?���
�H@S�
BL�
C���?���:�H@+�B�\C��f                                    Bx�*�  	�          @��?˅� ��@l(�B\��C�=q?˅�5@EB-�C���                                    Bx�*(V  
�          @�?��H��p�@r�\Bl��C���?��H�ff@S�
BB��C�                                      Bx�*6�  
�          @�p�?ٙ��   @vffB^C�/\?ٙ��8Q�@P  B0�RC�L�                                    Bx�*E�  �          @��R@��:=q@@  B   C�Ф@��c33@p�AᙚC�5�                                    Bx�*TH  
Z          @�\)?���>�R@AG�B"G�C�33?���hQ�@p�A��C�Ǯ                                    Bx�*b�  �          @�ff?���L(�@@  B�C���?���tz�@��A�=qC��
                                    Bx�*q�  
�          @�  ?�z��o\)@��A�33C��?�z����R?�A�C���                                    Bx�*�:  �          @���@ff�
�H@fffBH�C���@ff�>�R@>{B��C��3                                    Bx�*��  
�          @�\)?��H��Q�@��Bw  C��?��H���@l(�BM��C��                                    Bx�*��  �          @�\)?�  ��=q@�33Bp33C��)?�  �!G�@fffBF  C�h�                                    Bx�*�,  �          @�{?�G����@��Bp=qC�XR?�G��{@dz�BF�C��                                    Bx�*��  
�          @�p�?�����
@���Bn�RC��\?�����@b�\BE��C�)                                    Bx�*�x  T          @�z�?�=q��=q@���B~(�C�  ?�=q��\@mp�BT��C�Y�                                    Bx�*�  �          @�(�?��H��G�@���B��HC��{?��H� ��@y��BfG�C���                                    Bx�*��  �          @�(�?�(��z�H@���B�=qC�*=?�(���(�@z�HBg�C�O\                                    Bx�*�j  T          @�(�?�(����@�  B�aHC���?�(���@w
=Bb��C��                                    Bx�+  T          @���?�z῅�@��B�W
C��?�z��   @r�\Bd�C��                                     Bx�+�  "          @�Q�?�ff���H@�=qB��fC�w
?�ff�	��@j�HB^p�C��\                                    Bx�+!\  T          @�\)?�=q����@s33Bh�C�.?�=q�333@N�RB8�RC���                                    Bx�+0  �          @�
=?�p���{@r�\BgC�<)?�p��-p�@O\)B9��C�O\                                    Bx�+>�  
�          @��H?\(���G�@~{B��C��?\(��
�H@c�
Bb��C��                                    Bx�+MN  "          @}p�>�Q�
=@xQ�B��C���>�Q쿾�R@hQ�B�=qC��                                     Bx�+[�  �          @�z�?�G���p�@j=qB���C�Ф?�G���
@P��BV  C���                                    Bx�+j�  "          @�?�ff�	��@g�B]
=C���?�ff�<(�@@��B,\)C��                                    Bx�+y@  "          @�{?��R��G�@j=qBw��C���?��R�@L��BI�HC��)                                    Bx�+��  �          @��?xQ��z�@h��Bm��C�|)?xQ��.{@EB;C�ٚ                                    Bx�+��  �          @|(�?@  ��Q�@Tz�Bg�
C��f?@  �*�H@1G�B4
=C���                                    Bx�+�2  
�          @|��>��
�p�@P  B]�C�+�>��
�:�H@(��B'�RC�,�                                    Bx�+��  
Z          @\)?�\�33@L��BV��C�B�?�\�?\)@$z�B!G�C�ٚ                                    Bx�+�~            @\)?�ff���@UBf  C�˅?�ff���@>�RBC33C���                                    Bx�+�$  
�          @�Q�?��ÿ��H@U�Bc
=C�33?��ÿ���@=p�B?�C�xR                                    Bx�+��  
�          @��@	�����R@P��BS{C��@	����(�@8��B2�C���                                    Bx�+�p  "          @��\@zῳ33@N{BP�RC��@z��
=@333B-�
C�,�                                    Bx�+�  �          @���?��H����@UB`\)C��=?��H�(�@:=qB9(�C���                                    Bx�,�  
Z          @�=q?+��P��@J�HB.G�C��3?+��z=q@z�A��HC��H                                    Bx�,b  �          @�p�?
=�U@L��B-ffC��?
=�\)@A�RC�5�                                    Bx�,)  
�          @��
?�\�Y��@E�B'p�C�B�?�\����@p�A��C��)                                    Bx�,7�  
�          @��
?h���W
=@@��B#Q�C��\?h���~{@	��A޸RC�s3                                    Bx�,FT  T          @�33?�
=�K�@EB)G�C�'�?�
=�s�
@G�A�G�C��
                                    Bx�,T�  �          @��H?#�
�l��@'�B�C���?#�
���R?ٙ�A�C�T{                                    Bx�,c�  
Z          @��?���:�H@E�B3=qC���?���c33@�B �\C�0�                                    Bx�,rF  
�          @�?����\@N�RBOp�C�Ǯ?���>{@'�B\)C�q                                    Bx�,��  
�          @���?�p��33@O\)BV�C�y�?�p��/\)@+�B&�C�R                                    Bx�,��  
�          @�z�?�{�	��@I��BF  C�o\?�{�3�
@$z�B��C��                                    Bx�,�8  �          @���?�(��ff@EBI(�C�t{?�(��0��@!�B�C��                                    Bx�,��  
�          @���?�(�� ��@Q�BR��C��?�(��-p�@.�RB&(�C�8R                                    Bx�,��  
Z          @���?���.{@>�RB1ffC���?���Tz�@�B�C�T{                                    Bx�,�*  "          @���?�\)�(Q�@FffB8��C��)?�\)�QG�@=qB	{C�]q                                    Bx�,��  
�          @�  ?���Q�@Tz�BN�C��{?���Dz�@,(�B�RC�4{                                    Bx�,�v  
�          @��?����@J=qBG��C�f?��B�\@!�Bp�C�~�                                    Bx�,�  
�          @���?��\����@P��BY��C���?��\�(��@/\)B,G�C��H                                    Bx�-�  
�          @��R?��
���R@W�BU(�C�?��
�,��@5�B)p�C�                                    Bx�-h  �          @�{?˅�33@R�\BN�C��?˅�/\)@/\)B#�\C��                                    Bx�-"  
�          @�?��R��ff@dz�Bj33C�AH?��R�#�
@Dz�B=
=C���                                    Bx�-0�  
(          @�z�?��� ��@U�BX
=C�޸?���-p�@333B*��C�1�                                    Bx�-?Z  
�          @\)?���Q�@K�BS��C�&f?��&ff@*=qB'C�XR                                    Bx�-N   
�          @�z�?�{��@R�\BS��C���?�{�0��@/\)B&�C��                                    Bx�-\�  T          @��?��Q�@XQ�BSz�C���?��5�@4z�B&�RC�S3                                    Bx�-kL  
�          @�
=?�\)�   @\(�B[G�C�AH?�\)�.{@:=qB.��C�k�                                    Bx�-y�  �          @��?����p�@`��B_�\C���?���.{@>�RB2�RC��                                    Bx�-��  �          @��?���
=q@Y��BU�C��R?���7�@5B(  C�G�                                    Bx�-�>  T          @��R?��R��\@S33BN�
C�1�?��R�>{@-p�B �C�T{                                    Bx�-��  "          @��?\�ff@Z�HBSQ�C���?\�3�
@7�B'�C�9�                                    Bx�-��  "          @�=q?����%@C33B;\)C�k�?����L��@��B=qC�Ff                                    Bx�-�0  �          @�z�?��R�AG�@ ��B�C�)?��R�`  ?��
Ạ�C��
                                    Bx�-��  �          @�ff?�Q��.�R@5�B*{C��?�Q��R�\@
=qA�z�C��R                                    Bx�-�|  �          @�?�z��*�H@8��B.�HC�޸?�z��O\)@�RB{C���                                    Bx�-�"  �          @��R?�(��1�@333B&p�C��?�(��U�@�A�C��f                                    Bx�-��  
�          @w
=?���
=@;�BF�C��
?���,��@��B\)C��)                                    Bx�.n  O          @p  >�Q�=L��@mp�B��R@���>�Q�E�@hQ�B��qC���                                    Bx�.  
�          @j�H>����H@>{Bz\)C��>��z�@%�BI�
C�}q                                    Bx�.)�  '          @i��>\)�.{@hQ�B�8RC���>\)�s33@`��B�\)C�R                                    Bx�.8`  "          @j�H=#�
�+�@b�\B�p�C���=#�
��@S�
B�k�C��R                                    Bx�.G  �          @n{>����H@eB��
C�P�>���G�@Y��B�.C��                                    Bx�.U�  "          @u�#�
���@g�B�
=C����#�
���@S�
Bv=qC���                                    Bx�.dR            @y����Q�h��@q�B��
C����Q�ٙ�@_\)B�C�h�                                    Bx�.r�  
�          @w��#�
�L��@vffB�C�
�#�
���\@n{B�8RC��                                     Bx�.��  "          @l��>��þ�@i��B��fC���>��ÿh��@b�\B��C�3                                    Bx�.�D  
�          @���>�׿z�H@xQ�B��
C��\>�׿��
@e�B{
=C�Y�                                    Bx�.��  
Y          @~{>����O\)@w
=B��3C�8R>�����\)@fffB��)C�L�                                    Bx�.��  T          @�Q�>�  �   @|��B��fC�b�>�  ��=q@p��B�\)C�]q                                    Bx�.�6  
�          @�(�>8Q�   @��HB�#�C�"�>8Q쿮{@x��B�.C�޸                                    Bx�.��  "          @��\>�(����
@�Q�B��\C��3>�(��n{@y��B���C�h�                                    Bx�.ق  
�          @��?
=q�E�@|��B��C�l�?
=q�˅@l(�B�\C�Z�                                    Bx�.�(  
�          @��
>�p���\)@��HB�  C�l�>�p����@|(�B�.C��                                    Bx�.��  �          @��\?E��333@z�HB�33C��{?E���G�@l(�B��{C��3                                    Bx�/t  
�          @u�?����@^{B�{C�'�?����\)@H��B^��C���                                    Bx�/  
�          @s�
?z�5@c�
B�33C��f?zῸQ�@UB�(�C��                                    Bx�/"�  �          @vff�L�;#�
@u�B���C|\)�L�Ϳu@mp�B��HC�p�                                    Bx�/1f  
�          @w�?0�׿+�@n�RB���C���?0�׿�Q�@`  B�\C���                                    Bx�/@  �          @n�R>���  @l��B�B�C��q>����\@dz�B���C��H                                    Bx�/N�  "          @`�׾aG�>L��@`  B�L�C	0��aG���@]p�B�=qCw��                                    Bx�/]X  
Z          @b�\>L��=�Q�@a�B�A���>L�Ϳ#�
@^{B�{C���                                    Bx�/k�  T          @a�>�33���@_\)B��RC��>�33�z�H@W
=B���C��
                                    Bx�/z�  �          @e?��׿��\@N�RB���C��?��׿�z�@<��B_
=C�\                                    Bx�/�J  
�          @y��?�{�B�\@g�B�W
C���?�{��p�@X��BuQ�C�aH                                    Bx�/��  �          @|(�?Ǯ��(�@X��Bo  C�f?Ǯ���@Dz�BMC��=                                    Bx�/��  �          @q�?�
=�Q�@W�B�RC��?�
=���R@HQ�BbQ�C��                                    Bx�/�<  T          @q�?�  �B�\@U�Bx��C��3?�  �\(�@N�RBk=qC��                                     Bx�/��  
�          @tz�?�33�B�\@VffBv=qC��?�33��@HQ�B\�C��)                                    Bx�/҈  
�          @l��?��Ϳk�@S�
B~��C��?��Ϳ���@C�
B_=qC�XR                                    Bx�/�.  "          @w�?����@b�\B�{C�}q?����@VffBp\)C��                                    Bx�/��  T          @���?��ÿ#�
@h��B��C��?��ÿ�{@[�BkffC���                                    Bx�/�z  �          @y��?�
=���@dz�B��{C���?�
=��G�@X��Br�C�\)                                    Bx�0   �          @vff?�>��
@\(�B~��A,z�?��\@\(�B}�
C��q                                    Bx�0�  �          @�G�?޸R���R@g
=BC��?޸R���
@^�RBoQ�C���                                    Bx�0*l  �          @���?\�   @l(�B�{C�  ?\��(�@aG�Bs�
C��                                     Bx�09  �          @w�?�p����@c33B���C�'�?�p��Y��@\��B~�
C��                                    Bx�0G�  T          @w
=?Tz῔z�@e�B���C��{?Tz��@Q�BiG�C�)                                    Bx�0V^  �          @vff?Tz��(�@n�RB�C�W
?Tz῔z�@dz�B�\)C���                                    Bx�0e  �          @tz�?��Ϳz�@`��B�u�C�q�?��Ϳ�G�@U�Bs��C���                                    Bx�0s�  �          @u�?�33�O\)@b�\B���C���?�33���R@S�
Bqz�C��                                    Bx�0�P  
�          @��?�(���z�@b�\Bnp�C��{?�(���=q@O\)BPC��                                    Bx�0��  
�          @���@녿���@`  BZ(�C�:�@��
=@I��B<ffC��{                                    Bx�0��  �          @�ff?�=q���H@_\)B`�C��3?�=q�
=@HQ�BA=qC�n                                    Bx�0�B  T          @�33?�z῾�R@c33Bo�HC��)?�z��	��@L(�BLp�C��H                                    Bx�0��  �          @�(�?�p�����@U�BV�C���?�p��#�
@8��B1z�C���                                    Bx�0ˎ  "          @�ff?˅���
@\(�B\��C��3?˅�=q@A�B9p�C���                                    Bx�0�4  T          @���?�����@^{BZ
=C�S3?��p�@FffB:C�p�                                    Bx�0��  
�          @�Q�?�녿��
@_\)B]G�C�]q?����H@EB:\)C�
                                    Bx�0��  T          @���@ �׿���@e�Bep�C���@ �׿�\@S33BK��C�\)                                    Bx�1&  T          @�?��H���@`��Bd=qC��H?��H��ff@N{BI�
C�Ǯ                                    Bx�1�  �          @�\)?��H���@g
=Bj�RC���?��H�33@N�RBG\)C�.                                    Bx�1#r  
�          @�  ?�\)��Q�@h��Blp�C�y�?�\)�
=@P  BH�C�{                                    Bx�12  "          @�\)?�
=����@a�Bb��C�,�?�
=�\)@J=qBB
=C�ff                                    Bx�1@�  
�          @�z�?���Ǯ@XQ�B[�RC�ff?����@A�B<33C���                                    Bx�1Od  
�          @�{?�
=�Ǯ@`  Bc  C��3?�
=�(�@H��BB�C���                                    Bx�1^
  
�          @�  ?���@b�\Bb�HC�/\?���
@Mp�BE(�C���                                    Bx�1l�  
�          @�  ?�33��@h��Bl�C��?�33�z�@S�
BM(�C�AH                                    Bx�1{V  
�          @�Q�?��R����@o\)Bu�
C���?��R��
@Z=qBU\)C��{                                    Bx�1��  "          @�Q�?�����@tz�B�(�C�)?����p�@aG�B_G�C��                                    Bx�1��  �          @�
=?����G�@eBiz�C��{?�����@L��BE�\C���                                    Bx�1�H  "          @��?����p�@aG�Bip�C���?���ff@H��BE�C���                                    Bx�1��  �          @��
?�{��  @^{Be�C���?�{�
=@EBB=qC���                                    Bx�1Ĕ  �          @��?�녿�{@]p�Bi��C�}q?���{@G
=BG��C�R                                    Bx�1�:  
�          @���?�\)���@Z=qBg��C���?�\)��R@C33BE33C�Ǯ                                    Bx�1��  T          @~�R?˅��@UBe��C�R?˅� ��@AG�BF��C�.                                    Bx�1��  
�          @~{?�������@P��B]�C���?������H@@  BDp�C�j=                                    Bx�1�,  �          @{�?�33��G�@L(�BY��C�AH?�33����@9��B?Q�C�(�                                    Bx�2�  "          @x��?˅��\@Dz�BQ  C���?˅�33@,��B0�C�S3                                    Bx�2x  �          @y��?У׿޸R@Dz�BP�\C���?У��G�@-p�B0\)C��f                                    Bx�2+  
�          @}p�?�ff���R@N{Ba=qC��?�ff��\@9��BB{C���                                    Bx�29�  �          @|(�@33�@  @QG�B`�
C��@33��=q@E�BN33C���                                    Bx�2Hj  
�          @z=q?����
@Mp�B]\)C�Q�?���@;�BB�C�G�                                    Bx�2W  T          @xQ�?�녿��@P��Be��C���?�녿���@>�RBI\)C�Ф                                    Bx�2e�  �          @x��?�ff����@X��Bt
=C���?�ff��33@I��BY�C���                                    Bx�2t\  �          @z=q?޸R���
@UBkG�C���?޸R����@FffBR�RC��H                                    Bx�2�  
�          @~{@
�H��=q@:�HB=��C���@
�H�z�@%B#��C�,�                                    Bx�2��  �          @~�R@녿�@333B3  C��
@����@p�B  C�b�                                    Bx�2�N  T          @~{@ �׿���@A�BG�\C��@ ����@-p�B,z�C��                                    Bx�2��  �          @}p�@G���Q�@<��BA�C�  @G���@'
=B%G�C�`                                     Bx�2��  T          @~{?���z�@E�BL��C��{?��
�H@/\)B/�RC�"�                                    Bx�2�@  �          @~{?��Ϳ�@=p�BB=qC��3?�����@&ffB$ffC�AH                                    Bx�2��  
V          @|(�?��R�   @1G�B1��C�xR?��R���@Q�BQ�C��
                                    Bx�2�  �          @�  @�
��\)@7
=B7\)C�޸@�
�@   B
=C���                                    Bx�2�2  "          @|(�?������R@1G�B3(�C�>�?�����@Q�B�\C�`                                     Bx�3�  T          @~{@녿�
=@2�\B3�
C�=q@���@�HB33C�@                                     Bx�3~  �          @���@)���	��@Q�B 
=C�z�@)����R?޸RA�z�C�q�                                    Bx�3$$  �          @�G�@.{�?�=qA�z�C��H@.{�'�?�z�A�C�
=                                    Bx�32�  
�          @�  @$z��?���A�C���@$z��'
=?�33A��C�>�                                    Bx�3Ap  "          @���@�ÿ�33@5�B3
=C�5�@���ff@{B�RC�*=                                    Bx�3P  �          @�Q�@���@8��B8ffC�E@��33@"�\B(�C�
                                    Bx�3^�  �          @~�R@�G�@-p�B,{C��)@�(�@�B  C�B�                                    Bx�3mb  
�          @x��@ ���	��@"�\B"��C��3@ ���"�\@��B�HC�33                                    Bx�3|  �          @u�@z����@'�B-G�C��@z����@�B\)C�<)                                    Bx�3��  �          @r�\@33�@�HB(�C�<)@33�p�@�\B�HC���                                    Bx�3�T  
�          @k�@�
�z�@�RB�C�` @�
�=q?�{A�C�7
                                    Bx�3��  �          @l(�@
�H��z�@�\BC�O\@
�H���?���A�33C��                                    Bx�3��  
�          @j=q@�
��@��B��C�
=@�
���?�\A�C��                                    Bx�3�F  T          @xQ�@��ff@   B!{C�` @���R@Q�B\)C�H                                    Bx�3��  
�          @��?޸R��G�@c�
Bq�HC���?޸R��=q@UB[�RC��=                                    Bx�3�  
�          @�33?��;���@o\)B�\)C��?��Ϳ�  @hQ�Bz
=C��                                    Bx�3�8  �          @��\@,(��=p�@>�RB;p�C�E@,(����H@4z�B.��C��)                                    Bx�3��  T          @��H@{���@Mp�BN�HC��f@{��ff@E�BC�\C�xR                                    Bx�4�  T          @��@p�����@aG�Bg\)C���@p��(��@]p�Ba�C��=                                    Bx�4*  "          @���@�>8Q�@l��Bp�\@�@����@k�Bn��C��                                    Bx�4+�  
�          @��?�Q�#�
@xQ�B��
C��3?�Q��R@u�B��=C���                                    Bx�4:v  "          @�z�?˅���
@tz�B�C���?˅�!G�@qG�B�33C�Ff                                    Bx�4I  �          @u�?�ff>�@fffB��@��H?�ff��G�@dz�B��
C���                                    Bx�4W�  �          @p  ?�\)=��
@^�RB��H@S�
?�\)���@\��B�k�C���                                    Bx�4fh  T          @u�?�33>��@a�B�  A&�\?�33��z�@a�B�C�3                                    Bx�4u  �          @s�
?�(��\@\��B���C�� ?�(��h��@VffB{
=C��                                    Bx�4��  T          @j�H?��\=��
@Y��B���@l(�?��\��G�@W�B��C�^�                                    Bx�4�Z  
�          @j=q?�\)?\(�@W
=B�p�B��?�\)>���@]p�B��{A���                                    Bx�4�   
�          @vff>��H?޸R@W�Bw33B��>��H?���@eB�G�B��f                                    Bx�4��  �          @j=q>�?��@\(�B�z�B�  >�?�@c�
B�G�B=��                                    Bx�4�L  "          @dz�>k�?u@XQ�B��=B�{>k�>�(�@_\)B�  Bx�\                                    Bx�4��  �          @k�?�p�>#�
@N�RBv�@���?�p�����@N{Bu(�C���                                    Bx�4ۘ  T          @l��?��
>Ǯ@`��B�.A��?��
�\)@a�B�B�C�+�                                    Bx�4�>  "          @y��>#�
?�=q@\(�B��B��
>#�
?\)@c�
B��B�B�                                    Bx�4��  "          @�z�5?�(�@b�\Bm�\B�p��5?�@r�\B��{B��
                                    Bx�5�  T          @�녿#�
@ ��@^{Bj��B׏\�#�
?�(�@n�RB�L�B�B�                                    Bx�50  	�          @\)>\)?�  @o\)B���B�(�>\)?.{@xQ�B���B�                                    Bx�5$�  �          @w�>��R?+�@q�B�ǮB��
>��R=�G�@u�B�{A�p�                                    Bx�53|  �          @y��<#�
?���@i��B��=B�L�<#�
?
=@qG�B�\)B���                                    Bx�5B"  �          @q녾�?�Q�@\��B��\B��)��?k�@g�B�
=B�                                      Bx�5P�  �          @qG��8Q�?�\)@aG�B��=B�Ǯ�8Q�?��@h��B�#�Cz�                                    Bx�5_n  
�          @qG����R?�  @Z=qB��C녿��R?   @aG�B�z�C�                                    Bx�5n  
�          @p  ��{?Tz�@QG�Bt{C����{>�33@W
=Bp�C'�\                                    Bx�5|�  �          @mp���ff?L��@\��B�=qCzῆff>���@a�B���C#�3                                    Bx�5�`  
�          @k��^�R?:�H@^{B���C#׿^�R>k�@b�\B�u�C%p�                                    Bx�5�  �          @l�;.{?B�\@g
=B�B�Ǯ�.{>u@k�B��HB�33                                    Bx�5��  T          @l�;8Q�?=p�@g�B��B���8Q�>k�@l(�B���Cc�                                    Bx�5�R  
�          @n{��{?c�
@eB�(�B�33��{>\@k�B��C#�                                    Bx�5��  
�          @p  �
=?��@c33B��B�LͿ
=?
=q@j=qB�W
C	�)                                    Bx�5Ԟ  �          @w
=�Tz�?�z�@dz�B��B�LͿTz�?&ff@l��B�.C�                                    Bx�5�D  
�          @vff�\)?�p�@b�\B�Q�B�녿\)?:�H@k�B�B�{                                    Bx�5��  �          @u�=���?���@i��B�ffB��=���?
=@qG�B�{B��=                                    Bx�6 �  
Z          @tz�>aG�?��@c�
B��B���>aG�?W
=@mp�B��B��                                    Bx�66  T          @tz�?n{=�G�@l(�B�k�@޸R?n{�Ǯ@j�HB�33C��                                    Bx�6�  T          @w�?:�H?z�@o\)B�  B
=?:�H=�\)@r�\B�#�@���                                    Bx�6,�  
(          @r�\>��H?�@dz�B��B��)>��H?.{@l(�B��=BYz�                                    Bx�6;(  
�          @p��>aG�?�z�@^�RB��\B�� >aG�?n{@h��B��3B��3                                    Bx�6I�  �          @o\)?E�=u@i��B�\@���?E���
=@hQ�B��3C��
                                    Bx�6Xt  �          @b�\?8Q�>���@\��B���A���?8Q�#�
@]p�B�
=C���                                    Bx�6g  �          @o\)?L��?�@g
=B�8RB��?L��=L��@i��B�Q�@y��                                    Bx�6u�  �          @j=q?0��?8Q�@aG�B�.B9  ?0��>��@eB��A�=q                                    Bx�6�f  �          @j=q?5?333@aG�B��{B2G�?5>k�@eB��{A�(�                                    Bx�6�  
�          @\(�?s33>u@R�\B��
A`��?s33�B�\@R�\B�33C�O\                                    Bx�6��  �          @^{?s33>�
=@Q�B��HA��R?s33�#�
@S�
B���C��                                    Bx�6�X  
�          @\��?W
=?@  @P��B���B'G�?W
=>��
@U�B���A�G�                                    Bx�6��  �          @QG�?aG�?E�@C33B�\B$�R?aG�>�p�@HQ�B���A�(�                                    Bx�6ͤ  
�          @P��?���?5@?\)B�p�B�
?���>���@C33B�A�ff                                    Bx�6�J  �          @^�R?:�H?��@N{B���Ba{?:�H?.{@U�B��RB+z�                                    Bx�6��  
�          @^{>��?�G�@R�\B�u�B��>��?z�@X��B�=qB\��                                    Bx�6��  �          @\(�<�?aG�@Tz�B�p�B���<�>�ff@Y��B�ǮB��\                                    Bx�7<  T          @W�<�>�@U�B���B�  <�=u@W
=B��qB��                                    Bx�7�  
�          @Mp�>�  >��@K�B�=qB7ff>�  �\)@L(�B���C�H�                                    Bx�7%�  
�          @O\)?�\>��R@J�HB��\A�=q?�\���
@L(�B��=C��=                                    Bx�74.  �          @O\)?.{?
=@G
=B��B#��?.{>L��@J=qB��
A�ff                                    Bx�7B�  �          @O\)?:�H?&ff@E�B��
B&33?:�H>�=q@H��B��A��                                    Bx�7Qz  �          @Mp�?:�H?(��@C33B�u�B(\)?:�H>�\)@G
=B��HA�                                      Bx�7`   T          @K�?^�R?E�@<��B�33B%��?^�R>��@AG�B��
A�=q                                    Bx�7n�  T          @Q�?�  ?G�@AG�B�p�B�
?�  >��@FffB�k�A�                                      Bx�7}l  T          @S�
?�=q?(�@C�
B�A�z�?�=q>�  @G�B��qAMG�                                    Bx�7�  "          @U�?��?
=q@FffB�
=AظR?��>.{@I��B�Q�A�                                    Bx�7��  T          @P��?}p�?\)@C33B���A���?}p�>L��@EB�ǮA8��                                    Bx�7�^  "          @L(�?�G�>��H@?\)B���A�{?�G�>\)@A�B�Ǯ@�ff                                    Bx�7�  �          @AG�?�z�>���@0��B��fA|(�?�z�    @1�B�>�                                      Bx�7ƪ  �          @C33?�G�>�{@/\)B�
=Arff?�G�<��
@0��B��?�=q                                    Bx�7�P  �          @AG�?�z�>��@0  B��
A�p�?�z�=���@1�B���@���                                    Bx�7��  �          @C33?�  >��@/\)B�33A�=q?�  >#�
@1�B�aH@�
=                                    Bx�7�  T          @<(�?h��?�\@.�RB�33A�ff?h��>L��@1G�B��AIp�                                    Bx�8B  "          @:=q?W
=?�R@-p�B���Bz�?W
=>��R@0��B��A�z�                                    Bx�8�  �          @;�?h��?333@)��B�
=B��?h��>���@-p�B��{A��R                                    Bx�8�  �          @<��?5?Y��@.{B��BHff?5?
=q@333B��=B�\                                    Bx�8-4  �          @333?����Q�@\)B���C��)?����Q�@p�B�\C���                                    Bx�8;�  T          @3�
?��Ϳn{@z�BN(�C��\?��Ϳ�33?���B>�C��q                                    Bx�8J�  �          @5?c�
��=q@{B�\)C��?c�
��\@
�HB�aHC��                                    Bx�8Y&  
�          @<(�?�Q�.{@�Boz�C���?�Q�Ǯ@33Bi�
C���                                    Bx�8g�  
Z          @7�?Q�>�{@.{B�  A��?Q�=#�
@/\)B��q@AG�                                    Bx�8vr  �          @3�
?�33>.{@!G�B��\Az�?�33����@!G�B��C�t{                                    Bx�8�  T          @5�?��H��33@��Bgp�C�<)?��H���@B_�C��=                                    Bx�8��  
�          @7�?˅���
@z�B\
=C�b�?˅�\)@G�BU��C�XR                                    Bx�8�d  �          @8��?�\)�B�\@33B5��C��?�\)�xQ�?���B+�C�>�                                    Bx�8�
  "          @@  @G��0��@�B3(�C��q@G��fff@�B*z�C��
                                    Bx�8��  �          @Fff@
=q�8Q�@ffB*��C�Ǯ@
=q�n{@ ��B"\)C�S3                                    Bx�8�V  �          @:�H?�=q��z�?У�B��C�1�?�=q����?�p�A�  C���                                    Bx�8��  �          @4z�?��R����?��HA��
C�<)?��R���H?���A�33C��q                                    Bx�8�  T          @5�@(����?���A�  C�Ff@(�����?�ffA��C�7
                                    Bx�8�H  �          @@  @�ÿ�  ?G�Aw�
C���@�ÿ���?!G�AF=qC�aH                                    Bx�9�  T          @1�@   ��33>���@�=qC���@   ��
=>W
=@�p�C�b�                                    Bx�9�  T          @E@@�׾�(�>�33@��C��@@�׾�>���@�
=C���                                    Bx�9&:  �          @E�@?\)�   ��\)��33C�:�@?\)���H���z�C�Q�                                    Bx�94�  T          @C33@/\)>����G���33@��\@/\)>\��p���{@��H                                    Bx�9C�  �          @@  @ ��<��
�����z�>�Q�@ ��>.{�������@xQ�                                    Bx�9R,  �          @@��@�R>�������@dz�@�R>��R��33��@�R                                    Bx�9`�  �          @C�
@ ��?�녿�����33A���@ ��?��R�u����A�G�                                    Bx�9ox  
(          @:�H?��Ϳ.{���h�HC���?��;�G��\)�q(�C��q                                    Bx�9~  �          @>{?�z�?h���{�F(�A�R?�z�?�\)���:��B�
                                    Bx�9��  �          @<(�?��R>��R�'
=��RAc33?��R?���$z��x�RA���                                    Bx�9�j  �          @<(�?�p�?:�H�
=�\�Aљ�?�p�?p�����QB�                                    Bx�9�  �          @@  @�\?����33�(�A�33@�\?�p���ff��A�{                                    Bx�9��  �          @?\)?޸R>�33�Q��T��A6ff?޸R?���O�A��
                                    Bx�9�\  �          @<��?�
=�!G��!G��x\)C���?�
=�����#�
G�C���                                    Bx�9�  �          @AG�?�33    ���J�
<#�
?�33>W
=�z��I�R@Ǯ                                    Bx�9�  �          @>{?�z�>8Q��\)�F
=@���?�z�>\�{�C�A3�                                    Bx�9�N  �          @:=q?�(�>W
=� ���D@�?�(�>\��p��AQ�AI��                                    Bx�:�  �          @7
=@Q�?��׾��R��Q�A��\@Q�?�33�B�\�\)A�                                    Bx�:�  �          @1�?�  ?c�
��Q��2��AָR?�  ?�ff��{�)  A���                                    Bx�:@  
�          @)��?��?˅��(��   B ��?��?�\)�������B"��                                    Bx�:-�  �          @\)?n{?���?O\)A�G�B[�?n{?�  ?k�A��HBUp�                                    Bx�:<�  �          @#33?�Q�?�p��c�
��{B ��?�Q�?�ff�J=q���B�
                                    Bx�:K2  T          @ ��?�  >�=q��(��0A�R?�  >�
=��Q��-
=AV�\                                    Bx�:Y�  �          @33@z�?&ff�333����A��@z�?5�#�
��
=A���                                    Bx�:h~  T          @�?�(�?B�\�p����A��?�(�?Tz�^�R��A���                                    Bx�:w$  �          @!G�@G�>�ff������AH(�@G�?\)������Aw�                                    Bx�:��  �          @#33@{=��
��  ����@ff@{>8Q쿞�R����@�                                    Bx�:�p  �          @   @�R��  ��ff��p�C�Ǯ@�R�.{��������C��)                                    Bx�:�  "          @!G�@�
�aG��n{��33C�8R@�
����s33��{C�#�                                    Bx�:��  �          @=q@ff?0�׿aG���p�A��
@ff?B�\�Q����A��H                                    Bx�:�b  
�          @=q@ �׽�G��Y������C�n@ �׽#�
�\(�����C�`                                     Bx�:�  �          @>�{���O\)��C��q>�{� �׿u�ʏ\C��                                    Bx�:ݮ  �          @>����
=q�.{��G�C�Ff>����
=�W
=����C�j=                                    Bx�:�T  T          @\)?\(��33��\)���HC�<)?\(��G����*=qC�Y�                                    Bx�:��  �          @%�?\)��R<#�
>���C�aH?\)��R�.{�vffC�e                                    Bx�;	�  �          @$z�>���� �׾�������C��>�����R���H�/�C�Ф                                    Bx�;F  �          @��?J=q�    =���C�U�?J=q�����u�C�\)                                    Bx�;&�  T          @�
?��Ϳ�
=>��A'�
C�� ?��Ϳ��H>�=q@��
C��R                                    Bx�;5�  T          @�?Q��ff�#�
���C��?Q���B�\����C��\                                    Bx�;D8  �          @
=q?(���ff?:�HA�G�C�\)?(���?��A�Q�C�"�                                    Bx�;R�  T          ?�z�>�
=��\>��AJ�RC��
>�
=���>�\)A  C��                                     Bx�;a�  �          @�
?B�\���=u?�\)C��\?B�\��׽�\)��C��                                    Bx�;p*  �          @?�Q��
=������C���?�Q��z����;�C��)                                    Bx�;~�  
�          @
=?�33���
�.{���RC��?�33��\��z�����C�0�                                    Bx�;�v  T          @�?������u��\C�
=?�����
�8Q����C�R                                    Bx�;�  �          @ff?�{��{>��
@�{C��?�{����>k�@�p�C���                                    Bx�;��  �          @�H?�p���=q>.{@�z�C���?�p��˅=�\)?˅C��                                     Bx�;�h  �          @�?�Q��ff>���A�\C��?�Q����>���@��
C��\                                    Bx�;�  �          @��?�����=�\)?�C�b�?����녽L�Ϳ���C�aH                                    Bx�;ִ  �          @{?�Q����>L��@�\)C�c�?�Q��{=��
@	��C�S3                                    Bx�;�Z  
�          ?���?k���p�?�A���C��)?k��\>��AmC���                                    Bx�;�   "          ?�
=?��Ϳ�  >��@��\C��
?��Ϳ�G�=�\)@	��C���                                    Bx�<�  
�          @�?Q녿�p�?��A~=qC���?Q녿�G�>�G�AK
=C�xR                                    Bx�<L  �          @�\?��
��Q�>�@O\)C��?��
�ٙ�<�?&ffC�f                                    Bx�<�  T          @(�?�ff��(�>�  @��
C�33?�ff���R>#�
@��C�{                                    Bx�<.�  �          @33?ٙ����
=�\)?�ffC���?ٙ����
���
��C��
                                    Bx�<=>  
�          @�R?�z��{>\A�C���?�z���>�=q@ۅC���                                    Bx�<K�  �          @�?�녿��>W
=@�\)C�4{?�녿�33=���@#33C�"�                                    Bx�<Z�  �          @=q?��H���>�=q@�=qC��R?��H���>.{@~�RC��                                     Bx�<i0  T          @ff?�����>ǮA(�C�L�?����z�>�\)@ڏ\C�+�                                    Bx�<w�  T          @%?�녿���>u@��
C�
=?�녿��H>�@:=qC��
                                    Bx�<�|  �          @!G�?�G����H>��A-p�C��?�G��޸R>�p�A
{C��3                                    Bx�<�"  �          @�?�
=��33>ǮA�C���?�
=��>�z�@�\C���                                    Bx�<��  �          @(��?����?W
=A�C��=?����Q�?8Q�A~=qC�g�                                    Bx�<�n  �          @:�H?�\��\)?�\)A��\C���?�\��Q�?��\AθRC�"�                                    Bx�<�  T          @8Q�?�=q��Q�?�
=Bz�C��\?�=q�\?���B��C��                                    Bx�<Ϻ  �          @:�H@ �׿���?�\)A��C���@ �׿�?��
A���C�)                                    Bx�<�`  �          @6ff?�Q���?J=qA��C���?�Q��?0��Ab�RC��                                     Bx�<�  �          @*�H?�Q쿷
=?�Q�B�HC���?�Q��  ?�{A��RC�5�                                    Bx�<��  �          @&ff?�G�����?�{A�G�C��q?�G���z�?��
A�RC�+�                                    Bx�=
R  T          @0��?�33��33?��HA�z�C�&f?�33���H?�{A�Q�C��                                     Bx�=�  �          @1G�?����p�?�G�B\)C���?����ff?�
=A�{C�E                                    Bx�='�  �          @-p�?���33?��\A��C��?�����?���A�(�C�q�                                    Bx�=6D  �          @{?�{��33?�p�A��C��3?�{����?�z�A��
C��                                    Bx�=D�  �          @��?��H��33?�ffA��HC�k�?��H��Q�?z�HA�p�C��{                                    Bx�=S�  �          @�ÿ333?��
�����&{B�B��333?��ÿ�ff�p�B���                                    Bx�=b6  �          @333��\)?�z��
=�033Cuÿ�\)?��R��\)�*�C�=                                    Bx�=p�  T          @8�ÿ�33?�z�����
=C�\��33?��R��G���C޸                                    Bx�=�  �          @4z���\>����\�#  C,��\>��ÿ�G��!��C*��                                    Bx�=�(  �          @��
=�u��p��qp�Cn�R�
=�aG��G��y��ClT{                                    Bx�=��  
�          @ �׿fff>W
=�{��C'8R�fff>����p�Q�C!�\                                    Bx�=�t  �          @ �׿��
?����_�C�ΰ�
?(���33�[��C��                                    Bx�=�  �          @�Ϳ�p�?���
�c�HC�H��p�?(����_��Cz�                                    Bx�=��  �          @!녿��>�\)�Q��i
=C'�Ϳ��>�Q����f�
C$J=                                    Bx�=�f  �          @5��(���p��!G��|ffCD�׿�(���=q�!��~��C@Y�                                    Bx�=�  �          @0  ��{�Q��{�Y�\CS+���{�=p��  �]�CPxR                                    Bx�=��  T          @2�\��p��333�
=�l(�CQ����p��(��Q��p(�CNp�                                    Bx�>X  �          @8Q콸Q쿎{�(��33C��
��Q쿁G��+�aHC�\)                                    Bx�>�  T          @.{��Ϳ\(���R#�Cmk���ͿE�� ���\Cj�                                     Bx�> �  �          @!G��333������;��Cuz�333��(���{�C�Ct�=                                    Bx�>/J  �          @��J=q��=q��{��
Cv��J=q������{CvE                                    Bx�>=�  T          @�H�z��{��
=��HC|���z�����R�G�C|=q                                    Bx�>L�  T          @�Ϳ
=q�������{C}33�
=q��p������%\)C|��                                    Bx�>[<  "          @��=u��\��=q���C�� =u���R��33�=qC��                                    Bx�>i�  T          @{���Ϳ�Q��G��p�C��3���Ϳ�녿�����C��=                                    Bx�>x�  �          @/\)��\)�\)����	z�C�~���\)�(���{��\C�k�                                    Bx�>�.  "          @*�H��33�p��������C��;�33�
�H�\�
�C�u�                                    Bx�>��  "          @(Q쾏\)����Q��(�C�` ��\)�Q��  ��HC�L�                                    Bx�>�z  
�          @�þ�����������C�}q������\��33����C�g�                                    Bx�>�   
�          @�R�u��zῐ����p�C�y��u��\)��
=� (�C�j=                                    Bx�>��  T          @�����׿�=q��C�f����������(�C��q                                    Bx�>�l  �          @��\)��
��=q�ݮC�þ\)�녿����  C�\                                    Bx�>�  �          @ff���녿�z����HC�G����   ���H���HC�AH                                    Bx�>��  	�          @p�=�Q������G����
C�T{=�Q�������C�Y�                                    Bx�>�^  �          @
=>�녿�p�������HC��)>�녿��H��=q���C��{                                    Bx�?  �          ?�(�?333���������C��q?333�������C��                                    Bx�?�  
�          @?G���G���z��	  C�� ?G����R�����C��
                                    Bx�?(P  �          @�\?��������33�{C��?��������
=��C�Ff                                    Bx�?6�  
�          @?�p�����\)���C��f?�p���녿���C�#�                                    Bx�?E�  
�          @\)?�
=���׿���	�C�
=?�
=���Ϳ�z��\)C�e                                    Bx�?TB  T          @\)�L�Ϳ��ÿ������CvE�L�Ϳ�����R��Cu�                                    Bx�?b�  T          @ �׿���p�������C{��������
=�{C~�H                                    Bx�?q�  
�          @?��ٙ������=qC���?���
=�����z�C���                                    Bx�?�4  
�          @�?5�������\��C�  ?5��
=��ff��RC�P�                                    Bx�?��  T          ?ٙ�?�R��p��p���	G�C�g�?�R���H�u���C���                                    Bx�?��  "          ?�G�?����\)�fff��ffC��q?�����Ϳk�����C��q                                    Bx�?�&  T          ?�(�?녿�
=�#�
����C��\?녿��(���י�C���                                    Bx�?��  �          ?�33?J=q���R�(�����C�O\?J=q��p��.{��{C�l�                                    Bx�?�r  
�          ?���?\)���׿333�癚C�5�?\)��{�5���
C�P�                                    Bx�?�  	�          ?���>�ff��ff�=p���C�s3>�ff����B�\���
C��f                                    Bx�?�  �          ?���>Ǯ��ff�fff��RC�N>Ǯ����h����C�k�                                    Bx�?�d  "          ?�p�>��R��R�}p��[�\C�>��R�(���  �^G�C�K�                                    Bx�@
  T          ?�z�>�
=�
=��(��\)C��
>�
=�녿�p��
C�&f                                    Bx�@�  T          ?�G�?   ����=q�yffC��?   ��׿�=q�{G�C�}q                                    Bx�@!V  
�          ?�{>��ÿ�(��\�A�C�w
>��ÿ�(��Ǯ�K33C�z�                                    Bx�@/�  �          @
=>Ǯ��\��33�
ffC���>Ǯ�녾\��
C��=                                    Bx�@>�  "          @�\?���Q�#�
��G�C�5�?�����(�����C�:�                                    Bx�@MH  
�          ?�{?�ͿǮ�L���Ώ\C��R?�Ϳ�ff�O\)��z�C�                                    Bx�@[�  
�          ?�\?8Q��G�����(�C���?8Q��  ���H���C�˅                                    Bx�@j�  
(          ?��H?G����!G����C���?G���z�#�
��33C��{                                    Bx�@y:  �          @�
?E���  �E���p�C��R?E��޸R�G���ffC��H                                    Bx�@��  �          @G�?�ff�˅�&ff����C�Ǯ?�ff�˅�(����33C�Ф                                    Bx�@��  
�          @�?8Q��\�aG���(�C�f?8Q��\�c�
�ȣ�C�                                    Bx�@�,  �          @
�H?8Q����xQ���ffC��?8Q���
�z�H�؏\C��R                                    Bx�@��  T          @�?k���z�s33�ԣ�C�p�?k���z�u��z�C�y�                                    Bx�@�x  
�          ?��?s33��=q�n{���C���?s33��=q�n{��(�C���                                    Bx�@�  T          ?�
=?5������
  C�S3?5�������
��C�Y�                                    Bx�@��  
�          ?���?Q녿�{�p����z�C���?Q녿��Ϳp�����C���                                    Bx�@�j  �          ?���?B�\���H�0���θRC��?B�\���H�0����p�C��                                    Bx�@�  �          ?��H?}p��
=q�k��(�C��?}p��
=q�k��Q�C��{                                    Bx�A�  "          ?���?�{�0�׿aG��Q�C��?�{�0�׿aG��ffC�)                                    Bx�A\  "          ?Ǯ?��
�녿.{��(�C��H?��
�녿.{��(�C��H                                    Bx�A)  
�          ?��?�\)���H�#�
��C�!H?�\)���H�#�
�ۙ�C��                                    Bx�A7�  T          ?���?�(��aG��}p��33C���?�(��k��}p��(�C��H                                    Bx�AFN  T          ?�=q?�(����ÿ}p����C���?�(����ÿ}p��z�C���                                    Bx�AT�  "          ?��?�Q�Y���E����C�Ǯ?�Q�Y���E���z�C��                                     Bx�Ac�  �          @G�?��G���
=�@z�C�U�?��G���
=�?\)C�P�                                    Bx�Ar@  "          @   ?ٙ��@  �333��\)C��?ٙ��@  �333����C��                                    Bx�A��  
�          @z�?�z�B�\�u����C���?�z�B�\�u��{C��H                                    Bx�A��  T          @�?У׿W
=�z�H���C�P�?У׿Y���xQ��߅C�>�                                    Bx�A�2  T          @ ��?�z�.{�c�
�ҸRC�޸?�z�0�׿c�
�ѮC�˅                                    Bx�A��  �          @
=?ٙ��n{�Y�����\C���?ٙ��n{�Y����33C���                                    Bx�A�~  "          ?�\?�
=�W
=�\)��\)C��
?�
=�Y�������C���                                    Bx�A�$  T          ?�
=?�\)��p��s33��C���?�\)�\�s33��RC�e                                    Bx�A��  �          @�\?�{>�����R�z�@��?�{>\)���R���@���                                    Bx�A�p  	�          @
=?�Q�!G�������=qC�� ?�Q�#�
��������C���                                    Bx�A�  �          @p�?�녿(��h���ř�C�{?�녿�R�h����=qC���                                    Bx�B�  
�          @��?�
=�����G��ծC�t{?�
=���Ϳ�  ���C�U�                                    Bx�Bb  T          @�?�33��Q����c
=C�z�?�33��Q�
=q�\z�C�k�                                    Bx�B"  "          @�?�녿�녿@  ���RC���?�녿�33�=p���G�C���                                    Bx�B0�  
�          @�
?��ÿ���fff��Q�C��?��ÿ�ff�c�
���C���                                    Bx�B?T  �          @#33?����=q�aG���p�C�O\?���˅�\(�����C�7
                                    Bx�BM�  �          @�R?�\)���Y����=qC��?�\)��
=�Tz����C���                                    Bx�B\�  T          @"�\?���G��=p���(�C���?���\�8Q����C��                                    Bx�BkF  T          @#�
?�(��޸R�=p���(�C�S3?�(���  �5��\)C�=q                                    Bx�By�  T          @(��?��Ϳ��
����Q�C��?��Ϳ�����HQ�C���                                    Bx�B��  
�          @,��?�33���þ��H�*ffC�\?�33��=q���� Q�C�                                      Bx�B�8  �          @0��@ �׿�ff���>ffC�f@ �׿��
=q�4Q�C��{                                    Bx�B��  �          @5�@33���Ϳz��>=qC��\@33��{����3�
C��)                                    Bx�B��  
�          @9��@G���z�333�`z�C��@G����+��V�HC��                                    Bx�B�*  "          @2�\@ff���R�Tz���z�C�R@ff��  �O\)���\C��{                                    Bx�B��  �          @1G�?��R��(��Tz���=qC��)?��R��p��L������C�|)                                    Bx�B�v  
(          @6ff@33��\)����33C�ff@33�У׾�G����C�S3                                    Bx�B�  
�          @5�@33���Ϳ���C\)C��@33��{�\)�7
=C���                                    Bx�B��  
�          @7�?��H�����:�H�k33C��\?��H��(��0���]�C�t{                                    Bx�Ch  �          @Fff@p��ff����2�HC�8R@p��
=�\)�%p�C�"�                                    Bx�C  
�          @C33@Q����+��J�HC�Ф@Q����R�<��C��
                                    Bx�C)�  
�          @>�R@�녿�R�?�
C��
@��\���1��C��                                     Bx�C8Z  
�          @:=q@ ����\�   �C�XR@ ���33��ff��RC�E                                    Bx�CG   "          @:=q?�p����
=�p�C���?�p��ff��p���33C��=                                    Bx�CU�  
�          @:�H@   ���p����
C���@   �ff���
���
C�˅                                    Bx�CdL  
�          @;�?����	����(��(�C�3?����
=q�\��RC��                                    Bx�Cr�  �          @>�R@   �������z�C�E@   �(���33��=qC�5�                                    Bx�C��  �          @<(�?����
=q��(��G�C��q?����
�H��p���
=C���                                    Bx�C�>  �          @@  ?�(��(����"�HC�
=?�(���;�����C��3                                    Bx�C��  T          @=p�@   �Q���z�C��H@   ��þ�
=�ffC���                                    Bx�C��  
�          @8Q�@  ����>\)@5C�O\@  �˅>8Q�@p  C�Y�                                    Bx�C�0  "          @7�@#33��
=?�A(z�C���@#33��?\)A3\)C���                                    Bx�C��  �          @5�@!녿�
=>�Q�@��C�z�@!녿�>���AC��{                                    Bx�C�|  
Z          @5�@$z῕>��@C33C��{@$z῕>8Q�@p  C��                                     Bx�C�"  
�          @1�@!G���z������\C���@!G���z�u����C��q                                    Bx�C��  �          @0��@���녽�Q��\)C�Ǯ@���녽L�Ϳz�HC�                                    Bx�Dn  �          @1G�@���ff�\)�7�C��{@���ff������C��                                    Bx�D  
�          @0��@�ÿ�=q������C���@�ÿ�=q��  ����C�o\                                    Bx�D"�  �          @1�@ �׿��þ���  C�t{@ �׿����(��Q�C�P�                                    Bx�D1`  "          @333@{���R���Q�C���@{��G���(���HC��                                     Bx�D@  "          @4z�@{��
=�(���X  C�4{@{������R�J�RC�H                                    Bx�DN�  �          @.�R@{��p��8Q��q�C��@{���R���,��C�
=                                    Bx�D]R  �          @/\)?��R��{���
��\C�� ?��R��{���
��
=C�}q                                    Bx�Dk�  �          @/\)?�����녾W
=����C��?�����33�\)�@  C��                                     Bx�Dz�  �          @1�?�\)��
=�+��`��C��?�\)��������IG�C��=                                    Bx�D�D  �          @0��?�p��녿
=q�6�RC�5�?�p��33����p�C��                                    Bx�D��  
�          @"�\?�p���(�>�{@�=qC�s3?�p�����>��A  C���                                    Bx�D��  �          @-p�?�����@  ��33C��q?����ÿ.{�n=qC�ff                                    Bx�D�6  
�          @*�H?Ǯ������G��33C���?Ǯ��  �����=qC�
=                                    Bx�D��  
�          @:�H?��
=q�B�\����C��?��
=q��G��z�C��)                                    Bx�D҂  �          @:=q?޸R��
>��@=p�C�|)?޸R�33>u@��C���                                    Bx�D�(  T          @1G�?�����׿8Q��t  C�t{?�����\�!G��T��C�O\                                    Bx�D��  
�          @^{?��R�@  �5�?33C�9�?��R�A녿
=�ffC�                                      Bx�D�t  T          @c�
?��R�I��������HC��?��R�L(��h���o\)C��                                    Bx�E  �          @dz�?��\�E�������{C�!H?��\�H�ÿ�  ��=qC��)                                    Bx�E�  �          @Y��?}p��Fff�n{�
=C��R?}p��H�ÿL���Z�RC��                                     Bx�E*f  T          @\(�>aG��+���
��C�b�>aG��0�׿�Q���C�P�                                    Bx�E9  "          @X�ý�\)������9��C�q��\)���ff�0
=C�&f                                    Bx�EG�  �          @X��>�G��
=����4{C�8R>�G��p��33�*\)C��                                    Bx�EVX  T          @C�
>�Q�����Sp�C��f>�Q��z��ff�I��C�XR                                    Bx�Ed�  �          @n{?+��!G��(Q��4��C�z�?+��(Q�� ���+  C�+�                                    Bx�Es�  T          @XQ�?���<(���
=�˅C�?���@  ��ff��p�C��                                    Bx�E�J  T          @&ff>aG��33��
=��33C��H>aG��ff��=q�ď\C���                                    Bx�E��  "          @
==�Q��G���������C�:�=�Q���
�xQ����C�33                                    Bx�E��  �          @�R���H�녿��� C�<)���H����(����HC�k�                                    Bx�E�<  
�          @333@ff��p���  ��{C��@ff�����Q���
=C�:�                                    Bx�E��  T          @xQ�?\(��_\)��������C��?\(��b�\�p���g33C��                                    Bx�Eˈ  "          @�\)?���=q�xQ��T  C��q?�����E��(��C��3                                    Bx�E�.  �          @1G�@����{    ��Q�C�4{@����{=u?��
C�8R                                    Bx�E��  
�          @<��@  ���#�
�G
=C�P�@  ��\)����*�\C�R                                    Bx�E�z  
�          @vff?Ǯ�Z=q�J=q�=G�C�Q�?Ǯ�\(���R�Q�C�5�                                    Bx�F   �          @�33?���dz�h���MC�U�?���g
=�:�H�$��C�4{                                    Bx�F�  "          @�33@�\��Q�xQ��B{C��@�\�����B�\��C�e                                    Bx�F#l  T          @�{?�(���
=�L���p�C��?�(���  �z���C��\                                    Bx�F2  T          @���?�z����5��C�N?�z����R���H���C�9�                                    Bx�F@�  �          @�{?�������\���C���?����(���=q�G�C�                                    Bx�FO^  "          @���?�����p������C��3?�����{��\)�HQ�C���                                    Bx�F^  T          @�p�?�ff��녿\)��  C�>�?�ff���\���R�Z�HC�33                                    Bx�Fl�  �          @�Q�?�  ��p���G���  C�C�?�  ��{�B�\�	��C�:�                                    Bx�F{P  T          @��
?�=q���׿   ���\C��)?�=q��G��k��(�C��3                                    Bx�F��  
�          @�G�?�33��������C��?�33��ff��  �'
=C��                                    Bx�F��  
�          @���?�����(�����(�C�h�?������;u�p�C�^�                                    Bx�F�B  
�          @�Q�@   ��33������RC���@   ��(���=q�1G�C���                                    Bx�F��  �          @��R?�(���녿
=q��(�C��?�(����\��  �*=qC���                                    Bx�FĎ  �          @��H?�
=��ff�����
C���?�
=��
=��  �(Q�C��{                                    Bx�F�4  T          @�{?����׿(�����HC���?������Ǯ���HC���                                    Bx�F��  �          @�Q�?�����������C��{?����zᾊ=q�<��C���                                    Bx�F��  
�          @�33?У����׿L���
�\C���?У���녿�\����C���                                    Bx�F�&  �          @�Q�?��
��
=�^�R��C�+�?��
���׿���z�C�)                                    Bx�G�  �          @�G�?�����
=��  �%C�^�?������׿0�����C�K�                                    Bx�Gr  �          @��
?�p���녿���*�\C�?�p����
�8Q����C���                                    Bx�G+  
�          @��?�����H��=q�0��C��?����z�E�����C���                                    Bx�G9�  �          @�?�����ff�����BffC�H�?�����Q�aG��33C�,�                                    Bx�GHd  
Z          @�
=?�33��  ��p��G�C��R?�33��녿k���C��)                                    Bx�GW
  
(          @�(�?�=q��{��(��H  C��{?�=q��  �fff��C��R                                    Bx�Ge�  �          @�ff?�{�������`z�C�Q�?�{��  ��G��,(�C�/\                                    Bx�GtV  �          @���?޸R����333�   C�AH?޸R��{�����ffC�/\                                    Bx�G��  �          @�z�@�\��p��Ǯ����C�c�@�\����G����C�Z�                                    Bx�G��  
�          @�33@G��z�H?z�@��C�
=@G��xQ�?Q�A&=qC�.                                    Bx�G�H  
Z          @��H@���p��@\(�BT��C�=q@���:�H@_\)BYQ�C���                                    Bx�G��  
�          @���@'����H@HQ�B=G�C�� @'���G�@L��BB�C�z�                                    Bx�G��  "          @��@%���\@L��BC�
C�=q@%�Q�@P��BH�RC�AH                                    Bx�G�:  T          @�G�@+��p��@N{BB\)C�Y�@+��:�H@QG�BF�RC�`                                     Bx�G��  "          @�@2�\��=q@A�B-\)C�:�@2�\����@G�B4G�C�Ф                                    Bx�G�  T          @�  @*�H��@4z�B�C���@*�H�@=p�B$�C��\                                    Bx�G�,  �          @��
@(�����@ ��B�\C���@(���{@*=qB\)C�                                      Bx�H�  �          @��@,���&ff@33A��RC��@,���(�@{B��C��                                    Bx�Hx  
�          @���@%�8Q�?��A��HC��R@%�0  ?�p�A�33C���                                    Bx�H$  T          @��R@!��e?��A��C��q@!��^{?�\)A�=qC��                                    Bx�H2�  
�          @��@
=q��G�?��A^�HC�
@
=q�|��?�33A�ffC�XR                                    Bx�HAj  T          @�=q@8���w
=?�ffA��\C�e@8���p  ?�A��C��                                    Bx�HP  
�          @��@HQ��h��?�=qA�G�C�]q@HQ��a�?���A���C���                                    Bx�H^�  �          @�(�@L���h��?�z�A�p�C��=@L���`��?�33A��RC�&f                                    Bx�Hm\  	�          @�z�@G
=�c�
@G�A�G�C���@G
=�Z=q@��AУ�C�33                                    Bx�H|  
�          @�33@:�H�n�R?��A���C��@:�H�e@Q�A�p�C��\                                    Bx�H��  �          @�33@7
=�p��?�A�
=C��q@7
=�g�@�A�(�C�&f                                    Bx�H�N  
�          @�z�@5��tz�?�(�A��
C�AH@5��j�H@�RA�\)C��=                                    Bx�H��  �          @���@.{�~{?�=qA�Q�C�/\@.{�u@ffA���C���                                    Bx�H��  �          @�
=@z���G�?\(�A (�C�5�@z���
=?�z�AX��C�h�                                    Bx�H�@  �          @�z�@
�H���
>���@j�HC��R@
�H���H?&ff@�\C��                                    Bx�H��  T          @��@����>�  @2�\C�@����?��@���C��3                                    Bx�H�  
(          @�33?�Q���
==#�
>��C�1�?�Q���ff>�p�@�ffC�9�                                    Bx�H�2  "          @�G�?�{���=�?��RC�l�?�{��z�>�G�@���C�y�                                    Bx�H��  T          @��R?�����>L��@�RC��
?������?�\@ȣ�C��                                    Bx�I~  
�          @�33@ff���>�  @:�HC��R@ff���\?\)@�p�C���                                    Bx�I$  
Z          @�\)@   ����<�>�{C��f@   ����>�{@��
C��\                                    Bx�I+�  �          @�ff?������>\@��C��?�����?0��A\)C��                                    Bx�I:p  �          @�z�?�33��
=>��?�\C���?�33��{>�@��C��
                                    Bx�II  "          @��
?�������z��t(�C��q?������׿���3\)C���                                    Bx�IW�  
�          @�33?�=q��=q����F�HC��?�=q��z�B�\�  C��\                                    Bx�Ifb  "          @�z�?(����&ff��RC���?(�����\)�ĸRC���                                    Bx�Iu  �          @�{?������=p��=qC��
?���\)�'���=qC�n                                    Bx�I��  T          @�Q�?
=��Q��Fff�	�C���?
=��\)�0  ���C��=                                    Bx�I�T  
�          @�?�{��33�.�R���
C�q�?�{��G��ff���C�4{                                    Bx�I��  T          @�G�?E���
=�\)��z�C�k�?E����
��{���C�H�                                    Bx�I��  "          @�33?\(����ÿ�����\C��?\(���z῝p��K�
C���                                    Bx�I�F  T          @�(�?   ��\)��H��{C��)?   ��������
=C���                                    Bx�I��  "          @�Q�?���  �1G����C���?���{�Q����C��H                                    Bx�Iے  
(          @�ff?:�H���\�@  ���C��?:�H�����&ff��p�C��                                    Bx�I�8  
�          @���?O\)��G��'
=��ffC��=?O\)��\)�p����C�aH                                    Bx�I��  �          @��?W
=���(Q����C���?W
=���
�\)���\C��                                    Bx�J�  �          @�\)?+���
=�J=q��C�
=?+����R�1���(�C�ٚ                                    Bx�J*  �          @�  ?�\��G��H���p�C��?�\�����0������C��                                    Bx�J$�  
�          @���>W
=��G��Mp��\)C�AH>W
=�����4z���z�C�1�                                    Bx�J3v  T          @��\>#�
��33�Mp����C��R>#�
���H�3�
��33C���                                    Bx�JB  �          @��\>W
=���
�L(����C�@ >W
=����2�\����C�1�                                    Bx�JP�  
�          @��
>�{��=q�U��
\)C��>�{��=q�;���{C��                                    Bx�J_h  
�          @��H>L����{�Fff� p�C�%>L�����,����{C�
                                    Bx�Jn  
%          @�G�>����\)�P  �	�
C��3>����\)�7
=��RC�~�                                    Bx�J|�  %          @��>�������_\)���C��3>������HQ��=qC��                                    Bx�J�Z  �          @�
=>k���p��g��33C�u�>k���ff�P  �
��C�^�                                    Bx�J�   
]          @��>���
=�o\)�&{C��f>���Q��XQ��ffC��R                                    Bx�J��  T          @�\)>#�
��Q��_\)�
=C�\>#�
�����HQ��
G�C���                                    Bx�J�L  �          @�  @33����>Ǯ@���C��R@33���?G�A	��C��3                                    Bx�J��  
�          @��
=�G�����   ���RC��H=�G������
=q���HC��R                                    Bx�JԘ  "          @�z����]p��p���<��C�%����p���^{�)�
C�s3                                    Bx�J�>  �          @�z���I����G��O�C�����^�R�qG��<G�C�<)                                    Bx�J��  T          @�33�.{�_\)�h���6�
C�xR�.{�q��U�${C��                                    Bx�K �  
�          @���:�H�Y���dz��6�
C���:�H�l(��QG��${C�g�                                    Bx�K0  T          @�ff�=p��Vff�c�
�8�C��q�=p��h���QG��%\)C�8R                                    Bx�K�  �          @�33�z��Y���qG��>=qC�*=�z��l���^{�+=qC��                                    Bx�K,|  �          @�  �+��hQ���=q�?p�C�Ǯ�+��}p��p  �,ffC�4{                                    Bx�K;"  4          @�z�aG��c�
��G��F{C�׿aG��z=q�~�R�3=qC���                                    Bx�KI�  �          @�(������`  �����F
=C}�����vff�~{�3\)C~}q                                    Bx�KXn  �          @�\)�G��W��~�R�D  C�s3�G��l(��k��0�C��                                    Bx�Kg  �          @���:�H�Z=q�\(��2p�C����:�H�l(��HQ��G�C�l�                                    Bx�Ku�  �          @���.{�p  �c�
�,(�C���.{��G��N{��
C�7
                                    Bx�K�`  �          @�����
�����B�\��HC��R���
��G��(����\C��                                    Bx�K�  �          @��H��  ���H�  ��(�C��3��  ���׿�ff���RC���                                    Bx�K��  �          @��
����������G�����C�\)������{��ff�V�HC�ff                                    Bx�K�R  �          @�=q����ff���H��C�J=�����\��  �T  C�O\                                    Bx�K��  �          @��R=���33���\�=C���=���p��
=��z�C��3                                    Bx�K͞  �          @��=��
��=q��{�w�C�l�=��
���\=���?�\)C�l�                                    Bx�K�D  
Z          @�zᾞ�R����?aG�A'33C�#׾��R��{?�ffAw
=C��                                    Bx�K��  �          @��
��Q���\)?���AU�C��;�Q����?��A���C���                                    Bx�K��  �          @��þ�\)���?��A��HC�J=��\)���\@33A���C�:�                                    Bx�L6  �          @��
�u��G�?�(�A�{C�� �u���
@��A�ffC��)                                    Bx�L�  T          @��>aG���=q?���A��RC�XR>aG���?���A��C�c�                                    Bx�L%�  �          @���#�
��\)?���A��C��ͼ#�
����@\)A�(�C���                                    Bx�L4(  �          @���.{���?�=qA���C��׾.{��{@p�A�(�C��R                                    Bx�LB�  �          @�G�=#�
���?�z�A��C�>�=#�
���\@33Aʣ�C�AH                                    Bx�LQt  T          @�=q>W
=��{?���A��HC�]q>W
=��Q�@\)AݮC�k�                                    Bx�L`  �          @�  =�\)����?�  A���C�y�=�\)��
=@Q�A֏\C�~�                                    Bx�Ln�  �          @�
=>����?���A�ffC��>������?�A�\)C��{                                    Bx�L}f  �          @���?
=����@%�A��
C���?
=����@=p�BG�C�(�                                    Bx�L�  �          @�z�>������\@G�A�{C���>������@*�HA��C���                                    Bx�L��  �          @���=u��33?�33A`��C�c�=u��\)?ǮA�C�ff                                    Bx�L�X  �          @��\�B�\����?0��@���C�箾B�\��?��AO\)C��                                    Bx�L��  �          @�����R���R=�Q�?��
C�G����R��?\)@�ffC�E                                    Bx�LƤ  "          @�Q쾣�
��\)������z�C�C׾��
���=�\)?J=qC�E                                    Bx�L�J  �          @�(�>.{���<#�
=�C���>.{��G�>��@�=qC���                                    Bx�L��  �          @���?
=q���>�z�@QG�C�{?
=q��{?=p�A33C�)                                    Bx�L�  �          @�  >�z������Q쿂�\C��>�z���
=>�Q�@�{C��f                                    Bx�M<  �          @�  ����\)�����p�C�ٚ����  =L��?\)C�ٚ                                    Bx�M�  �          @�  �����p��\)��
=C��ÿ����ff��G���G�C���                                    Bx�M�  �          @�Q쿠  ���ÿ5��Cc׿�  ���\��=q�H��C�                                    Bx�M-.  �          @�녿+���  ��z��R�\C�+��+���  >.{?�C�,�                                    Bx�M;�  �          @��H��33��=q�#�
��C�  ��33���>�(�@���C���                                    Bx�MJz  �          @��
��  ��33>u@+�C�����  ���?5@�ffC��)                                    Bx�MY   �          @��;�����(�>.{?�\)C�S3�������H?#�
@���C�O\                                    Bx�Mg�  �          @��\�Ǯ���=�G�?��\C����Ǯ����?z�@љ�C��                                    Bx�Mvl  
�          @��   ���;W
=�%�C��\�   ����>aG�@,(�C��\                                    Bx�M�  �          @�Q��G����<#�
=�Q�C�W
��G���
=>�ff@�(�C�T{                                    Bx�M��  �          @�=q��{����=#�
>�ffC��R��{����>�@�  C���                                    Bx�M�^  �          @�Q�#�
��  >�33@~�RC���#�
��{?O\)A��C�                                    Bx�M�  �          @�Q쾏\)��\)>�ff@���C�g���\)���?h��A'�C�b�                                    Bx�M��  �          @����\)���H>�
=@�
=C��὏\)����?c�
A ��C���                                    Bx�M�P  �          @�p��L����(�?(��@�=qC����L����G�?���AJ�RC��R                                    Bx�M��  �          @�{�#�
����?+�@�ffC��3�#�
���?�33AL��C���                                    Bx�M�  �          @�  ���
��?\(�A�C��=���
���\?��AmG�C���                                    Bx�M�B  �          @��ý#�
��?}p�A-p�C�Ǯ�#�
��=q?�(�A��C��f                                    Bx�N�  �          @�\)�L������?h��A(��C��ýL����G�?�\)A
=C��3                                    Bx�N�  �          @��H�k�����?:�HA
=qC���k���?�A`Q�C���                                    Bx�N&4  �          @�<��
���>�
=@�  C�&f<��
����?\(�A*=qC�&f                                    Bx�N4�  �          @��H�z���ff�W
=�)C�E�z���Q��
=��Q�C�Q�                                    Bx�NC�  �          @�Q�W
=�����ff����C��ÿW
=���ÿ����\��C��)                                    Bx�NR&  T          @��ÿ�Q��xQ��(�����C|�q��Q������˅��\)C}��                                    Bx�N`�  �          @�ff��{�{���G����\C~33��{���\�������HC~�                                     Bx�Nor  �          @�ff��G����H��33���\C�
=��G����R��  �P��C�8R                                    Bx�N~  �          @�  �^�R��{��\)���
C���^�R�����xQ��F�RC�C�                                    Bx�N��  �          @�{�G����Ϳ����C��3�G���Q�}p��N{C��
                                    Bx�N�d  �          @����n{��G����o\)C�޸�n{��(��B�\�ffC���                                    Bx�N�
  �          @�������(������(�C���������}p��L  C�                                    Bx�N��  �          @��G�����33�r=qC���G����׿@  ���C��
                                    Bx�N�V  �          @�Q�G���Q쿞�R�\)C�Ф�G�����Tz��)��C��                                    Bx�N��  �          @����\)�r�\��33���
Cz
��\)�{����
����Cz�q                                    Bx�N�  �          @�p���  �tz��=q�Ù�C{�f��  �~{������  C|��                                    Bx�N�H  �          @�(��Y���qG���
=��  C�� �Y���x�ÿ���n�HC�Ф                                    Bx�O�  �          @�G��
=�w
=�xQ��`z�C���
=�{�����
{C���                                    Bx�O�  �          @{������mp���p���{C��3�����s�
�^�R�M�C�                                    Bx�O:  �          @}p���33�j�H�����RC�AH��33�r�\�����zffC�XR                                    Bx�O-�  �          @\)��
=�j�H��p���{C�����
=�s33��\)����C�ٚ                                    Bx�O<�  �          @�ff�(���mp�������(�C��3�(���w����H��
=C�&f                                    Bx�OK,  �          @��׿
=q�fff�  ��(�C��{�
=q�s33��33���HC���                                    Bx�OY�  �          @�33�#�
�u���ԣ�C��\�#�
��  ������HC��                                    Bx�Ohx  �          @����\�tz������\)C�.�\����� ���ӮC�T{                                    Bx�Ow  �          @�p��h���a��R�\�(Q�Cz�h���u��<(��
=C�H�                                    Bx�O��  �          @����u�n{�7����C���u�~�R�   ��
=C�>�                                    Bx�O�j  �          @��Ϳ���j�H�=p����Cy�f����{��%���
C{(�                                    Bx�O�  �          @��
��33�u��)����Cy���33����G����HC{�                                    Bx�O��  �          @�������s33��R��  Cz���������ff�ң�C{�                                    Bx�O�\  �          @�녿�z��`���3�
��Ct����z��p���p����Cv(�                                    Bx�O�  �          @�z´p��W
=�N{�%�CvG���p��j=q�8����Cw�q                                    Bx�Oݨ  �          @�(�����e��@  ��HCx�=����vff�(���Q�Cz(�                                    Bx�O�N  �          @�=q�&ff�o\)�{��HC���&ff�}p��ff��z�C�P�                                    Bx�O��  �          @�Q�=�\)�n{��
��(�C��=�\)�y���ٙ���=qC���                                    Bx�P	�  �          @��ý�G��g
=��� C�)��G��s�
���ՙ�C�'�                                    Bx�P@  �          @��þ���`����H�	�C��q����n{�z���{C��q                                    Bx�P&�  �          @�\)=u�^�R���	(�C�~�=u�l(�����ffC�xR                                    Bx�P5�  �          @�G���Q��_\)�   �ffC�J=��Q��mp��	������C�T{                                    Bx�PD2  �          @��ý��`  �(���C�\���n{�ff��33C�q                                    Bx�PR�  �          @���<��
�X���)���=qC�4{<��
�g��z��Q�C�1�                                    Bx�Pa~  �          @����#�
�Tz��-p��C���#�
�c�
�Q���
C��                                    Bx�Pp$  �          @�33�u�Tz��333� �\C��׾u�dz��{�
�C�
=                                    Bx�P~�  �          @��׿B�\�b�\�,���C����B�\�q��ff���\C�Y�                                    Bx�P�p  �          @��׾�(��e��.{�(�C����(��u������RC���                                    Bx�P�  �          @��\��
=�b�\�7��Q�C�����
=�s33�!G���\C��R                                    Bx�P��  �          @���&ff�`  �;����C���&ff�qG��%��{C�
                                    Bx�P�b  �          @���#�
�`  �;���C��ÿ#�
�qG��%���C�/\                                    Bx�P�  �          @����s33�`  �0  �z�C~޸�s33�p  �������C��                                    Bx�P֮  �          @�G��W
=�Vff�<(��!�C��W
=�g��&ff�=qC�|)                                    Bx�P�T  �          @�\)�\(��Q��:=q�"�C@ �\(��b�\�%��\)C�&f                                    Bx�P��  �          @��c�
�S33�4z��  C~�ÿc�
�c33�\)��
C�q                                    Bx�Q�  �          @�p����
�J=q�:=q�$C|���
�[��&ff��C}Q�                                    Bx�QF  �          @��
����J=q�3�
� p�Cz�R����Z=q�   ��RC|J=                                    Bx�Q�  �          @�33�����HQ��5��!�
Cz�������XQ��!G��33C|                                    Bx�Q.�  �          @�
=��Q��L(��:=q�"(�Cy�{��Q��\���%���C{�                                    Bx�Q=8  �          @�  ��{�L(��=p��$��Cz�H��{�]p��)���G�C|B�                                    Bx�QK�  �          @��R���
�N{�:=q�"�C|Q쿃�
�^�R�%��HC}��                                    Bx�QZ�  �          @�ff��  �Mp��9���"��C|��  �^{�%���C}��                                    Bx�Qi*  �          @�\)�����G
=�A��)Cy�ῐ���XQ��.{�=qC{z�                                    Bx�Qw�  �          @�
=��p��K��8���!=qCxǮ��p��[��%���Cz@                                     Bx�Q�v  �          @�
=���\�L���5�(�Cx^����\�]p��!��	�HCy�{                                    Bx�Q�  �          @�\)��ff�E�>{�&ffCw+���ff�Vff�*�H�Q�Cx��                                    Bx�Q��  �          @�p���=q�<���@���+�RCu�Ϳ�=q�N{�.{��
Cw��                                    Bx�Q�h  �          @�����z��C�
�B�\�(G�Cu8R��z��U��/\)��Cw�                                    Bx�Q�  �          @��R��
=�+��Vff�5�Cj:��
=�?\)�E�#�
Cm
                                    Bx�Qϴ  �          @��׿�\)�+��G��.Ck��\)�=p��6ff���Cm��                                    Bx�Q�Z  �          @p���33��  ��H���CYO\�33��(��\)�{C\�)                                    Bx�Q�   �          @b�\�
=��\�����33C`��
=�{��  ���
Cbh�                                    Bx�Q��  �          @]p���(���ÿ�����\)CjLͿ�(��#33��=q��  Cl
=                                    Bx�R
L  �          @\�Ϳ�\)�5��
=��z�Cx�Ϳ�\)�=p���33��ffCyY�                                    Bx�R�  �          @e���  �Y����Q���\)C}�쿀  �Z�H�#�
�
=C}�f                                    Bx�R'�  �          @dz�
=�`��>.{@*�HC�8R�
=�^�R>��H@�ffC�.                                    Bx�R6>  �          @I����  �G�>B�\@`��C��
��  �E�>�Ap�C���                                    Bx�RD�  �          @L�ͽ���Ϳ�ff���C������#�
�����ffC��                                    Bx�RS�  �          @g���Q��   �@  �_Q�C��;�Q��G��333�JG�C��                                    Bx�Rb0  �          @e���Q��ff�8Q��V�C�0���Q��ff�*�H�A�C���                                    Bx�Rp�  �          @e�������<(��[=qC�q������\�.�R�FQ�C��                                    Bx�R|  �          @g��Ǯ�G��>�R�]Q�C����Ǯ��\�1��HffC�/\                                    Bx�R�"  �          @l�;������@  �Y{C������=q�2�\�D  C��                                    Bx�R��  �          @hQ��(���  �H���p�\C�3��(����=p��[�C�H                                    Bx�R�n  �          @fff��(���(��HQ��r�C����(��   �=p��]�C���                                    Bx�R�  �          @e���G���z��?\)�c
=C�����G����333�NffC�N                                    Bx�RȺ  �          @l�;�
=�   �E�b�C�׾�
=�G��8���Mz�C��R                                    Bx�R�`  �          @fff��׿��R�>{�]��C�` �������1G��I{C�                                      Bx�R�  �          @g
=��(��ff�#33�;�C��\��(��$z��z��&�
C�AH                                    Bx�R��  �          @e���=q�7����HC�H���=q�C33������{C�q�                                    Bx�SR  �          @g���ff�@  ��Q���C�����ff�J�H��z��ۮC��
                                    Bx�S�  �          @k�����9����R�=qC��{����E���H� �\C�33                                    Bx�S �  �          @h�þaG��3�
��\�p�C��aG��@������C��                                    Bx�S/D  �          @n�R�aG��,���$z��-�C��\�aG��:�H��
�33C��)                                    Bx�S=�  �          @n�R�u�   �0  �>C�5þu�.�R�!G��*{C�t{                                    Bx�SL�  �          @q녾�\)���:�H�K  C��{��\)�(Q��,���6p�C���                                    Bx�S[6  �          @p�׽�����C33�XffC�lͽ��{�5�C��C��R                                    Bx�Si�  �          @u��  ����HQ��[(�C��q��  �{�:�H�F�C��                                    Bx�Sx�  �          @p  �������:�H�Lp�C�S3�����%��,���8
=C��                                    Bx�S�(  �          @j�H�Ǯ�Q��0���CQ�C�Q�Ǯ�'��"�\�/{C��)                                    Bx�S��  �          @j�H�\�!��(���7�C��;\�0  ����#p�C�f                                    Bx�S�t  T          @j�H����*�H�{�)z�C�\����8Q���R�Q�C�j=                                    Bx�S�  �          @p  �\�1���R�%�C�  �\�?\)�{�C�e                                    Bx�S��  �          @g
=��
=�$z��p��-C�]q��
=�1���R��C��{                                    Bx�S�f  �          @hQ���H�!G��$z��4=qC��H���H�/\)�� Q�C��                                    Bx�S�  �          @w
=�\)���?\)�K=qC�Y��\)�(Q��1G��7�C���                                    Bx�S��  �          @}p���ff�z��J�H�UG�C��f��ff�%�=p��Ap�C�{                                    Bx�S�X  �          @{��
=q�z��G��RffC�y��
=q�%�9���>C�                                      Bx�T
�  �          @z=q���H��R�>�R�F��C�p����H�.�R�0  �3{C��                                    Bx�T�  �          @r�\���33�=p��N33C�P����#33�0  �:�C���                                    Bx�T(J  �          @s�
�����C�
�WQ�C�/\�����7
=�C�HC��                                    Bx�T6�  �          @j�H�Ǯ�(��)���;��C�n�Ǯ�*�H���'��C���                                    Bx�TE�  �          @fff    �%��\)�/C��    �2�\�����C��                                    Bx�TT<  �          @g��#�
�<�����
{C�uþ#�
�G
=��\��RC��=                                    Bx�Tb�  �          @b�\�8Q��*=q���$�HC���8Q��6ff�ff�33C�+�                                    Bx�Tq�  �          @dz�\)�Dz����C���\)�N{�����z�C��)                                    Bx�T�.  �          @hQ켣�
�O\)��{��33C��R���
�W���=q��  C�ٚ                                    Bx�T��  �          @fff�.{�H�ÿ�p�����C�h��.{�Q녿��H��C�y�                                    Bx�T�z  �          @e�W
=�:=q�z���C�箾W
=�E�������(�C�                                    Bx�T�   �          @mp����P  ���
�噚C��׾��X�ÿ�  ����C��\                                    Bx�T��  �          @tz�<��
�^{������\)C�  <��
�e��ff���RC��                                    Bx�T�l  �          @tz�B�\�Vff��=q��RC�]q�B�\�_\)�����=qC�o\                                    Bx�T�  �          @r�\�aG��L(��   � {C��aG��Vff��p����
C�q                                    Bx�T�  �          @s�
��G��P  ��(���=qC�H��G��Y���ٙ����C��                                    Bx�T�^  �          @qG����K����G�C��þ��U���G���ffC���                                    Bx�U  �          @u��aG��O\)�G���ffC��aG��Y���޸R��z�C�%                                    Bx�U�  �          @w
=>.{�XQ����(�C�ff>.{�aG���ff��=qC�W
                                    Bx�U!P  �          @x��>�ff�\�Ϳ�p��ә�C��
>�ff�e���Q���(�C��{                                    Bx�U/�  �          @y��>�ff�Y����{��C�>�ff�b�\������ffC��)                                    Bx�U>�  �          @x��>�=q�U��p���Q�C�H�>�=q�_\)�ٙ�����C�0�                                    Bx�UMB  �          @w����
�Tz��   ���C��ͼ��
�^{��p���C��\                                    Bx�U[�  �          @u��Q��Vff������HC�@ ��Q��_\)��{�ŮC�G�                                    Bx�Uj�  �          @vff�u�X�ÿ����
C��)�u�a녿��
���HC�\                                    Bx�Uy4  T          @w��W
=�X�ÿ���p�C�5þW
=�a녿Ǯ����C�G�                                    Bx�U��  �          @xQ쾏\)�W������
=C��H��\)�`�׿�{��ffC���                                    Bx�U��  �          @w
=���Vff������{C����_\)��ff��C�4{                                    Bx�U�&  �          @vff��ff�7
=��Cy�쿆ff�B�\�ff���Cz�f                                    Bx�U��  �          @w
=�n{�G
=���
C}p��n{�QG���=q����C~0�                                    Bx�U�r  �          @u�Y���J�H��(����C~�R�Y���Tz��(����HC��                                    Bx�U�  �          @|(�����@  ����Cz������J�H�ff� ��C{��                                    Bx�U߾  �          @y���}p��4z��   ���Cz���}p��@  ���  C{�=                                    Bx�U�d  �          @y���aG��5��!��!�
C|��aG��AG��33�C}�                                     Bx�U�
  �          @|�ͿG��;��!���\C(��G��G��33�Q�C��                                    Bx�V�  T          @|(������%��.�R�0\)CvJ=�����2�\�!G�� {Cwٚ                                    Bx�VV  �          @{����H� ���0  �2�RCtO\���H�-p��#33�"Cv�                                    Bx�V(�  T          @��׿�ff�1G��-p��)Cy:ῆff�>{�\)�Q�Cz��                                    Bx�V7�  �          @z�H�=p��AG��Q��C�#׿=p��L(�����C�~�                                    Bx�VFH  �          @�����=q�n{��{��z�C��3��=q�u��������
C��                                    Bx�VT�  T          @z�H��\)�e���������C���\)�k������z�C��3                                    Bx�Vc�  �          @\�;8Q��J=q������{C�j=�8Q��P  �����C�u�                                    Bx�Vr:  �          @s33?Tz��dzὸQ쿽p�C���?Tz��dz�>.{@,(�C��                                    Bx�V��  �          @��?����~�R?��RA��HC��3?����xQ�?��
A���C��\                                    Bx�V��  �          @��R?�����G�@	��A�G�C��{?����w�@(�A��
C�'�                                    Bx�V�,  �          @��?��
�z�H?E�A!C�<)?��
�vff?�ffA_
=C�k�                                    Bx�V��  �          @}p�?�=q�_\)�.{�p�C��\?�=q�`  =�Q�?�  C���                                    Bx�V�x  �          @�=q?���e����33C��H?���e=�?��C��                                     Bx�V�  �          @��@����=q>W
=@'�C���@����G�?�\@�  C��q                                    Bx�V��  T          @�=q@�
���?���A\��C��@�
���R?�G�A�\)C�`                                     Bx�V�j  �          @�(�@����33?��A��RC�g�@���}p�@�AȸRC��q                                    Bx�V�  �          @�z�@p����?ǮA�p�C�q�@p����?�{A�ffC��f                                    Bx�W�  �          @�{@ff��=q?�=qAl��C�XR@ff���R?�33A�C���                                    Bx�W\  �          @�(�@,�����
=�Q�?}p�C��)@,����33>���@�p�C���                                    Bx�W"  �          @�
=@+���33?\(�A��C��\@+�����?�AP(�C�                                    Bx�W0�  "          @�ff@p�����?˅A�G�C��)@p�����?��A���C�W
                                    Bx�W?N  �          @��@����p�?��AK�
C�E@�����\?�Q�A��C���                                    Bx�WM�  �          @�@ff��?��
Ae�C��@ff���H?�=qA�Q�C�4{                                    Bx�W\�  �          @�ff@���ff?�G�Aa�C��3@���33?ǮA�(�C��                                    Bx�Wk@  �          @�p�@z����?Q�AffC��R@z����H?��AK33C�                                      Bx�Wy�  �          @�(�@������?�=qA��HC�޸@����?��A�{C�.                                    Bx�W��  �          @�33@
=q���?�Q�A��C��
@
=q��33@�RAŅC�1�                                    Bx�W�2  �          @�  @#33����?�=qA�
=C�c�@#33��p�?�\)A�33C��)                                    Bx�W��  �          @��R@&ff�n�R@�RA�z�C�o\@&ff�c�
@.{A��C��                                    Bx�W�~  �          @��\@$z�����?У�A��\C�J=@$z��y��?��A�C���                                    Bx�W�$  �          @��@{����?�=qAMC��@{���\?��A�33C�C�                                    Bx�W��  �          @���?�=q���R?B�\A(�C�,�?�=q����?�ffAG
=C�O\                                    Bx�W�p  �          @��H@<(���  ?��Aqp�C�P�@<(�����?�z�A���C��H                                    Bx�W�  T          @��\@=p����?�{An�RC���@=p���=q?�\)A��C��                                    Bx�W��  "          @�ff@>�R��z�?��A��RC��q@>�R��Q�@	��A�(�C�K�                                    Bx�Xb  �          @�@)�����?޸RA�33C�z�@)������@ ��A��
C��3                                    Bx�X  �          @�ff@.�R��33?�A���C��@.�R��\)@z�A��C�o\                                    Bx�X)�  �          @�(�@@  ��=q?���A���C�7
@@  �|��@ffA�\)C���                                    Bx�X8T  �          @�z�@Dz��\)?�z�A�Q�C�˅@Dz��w
=@	��A�z�C�=q                                    Bx�XF�  �          @���@A����?�(�A��\C�/\@A���  ?��HA�
=C���                                    Bx�XU�  �          @���@8����=q?�p�A~=qC���@8����
=?޸RA�{C�0�                                    Bx�XdF  �          @�p�@4z����?�Q�Au��C�Q�@4z����?ٙ�A��
C���                                    Bx�Xr�  �          @�
=@1G����H?�ffA1��C���@1G�����?���A`Q�C���                                    Bx�X��  �          @��@,(����?���A@(�C�J=@,(���\)?��An�RC��                                     Bx�X�8  �          @�p�@>�R�z�H?�ffA�  C��@>�R�tz�?��
A�33C���                                    Bx�X��  �          @�G�@H���x��?�p�A���C�u�@H���q�?��HA�p�C���                                    Bx�X��  �          @�(�@N{�~{?�A���C��=@N{�w�?��A�
=C���                                    Bx�X�*  T          @��@Mp�����?\A��C�N@Mp��z�H?޸RA��
C���                                    Bx�X��  �          @�=q@?\)��33?˅A�p�C�f@?\)��Q�?���A�Q�C�Y�                                    Bx�X�v  �          @���@>{�~{?�p�A��
C�c�@>{�w�?���A�ffC��H                                    Bx�X�  �          @��H@3�
�O\)@p�A�=qC�� @3�
�G
=@��A�G�C�3                                    Bx�X��  �          @�G�@1��Z=q?�z�A�(�C���@1��S33@ffAϙ�C��                                    Bx�Yh  �          @��@#�
�Y��?�\)A���C�u�@#�
�S�
?�ffA��\C��R                                    Bx�Y  �          @|(�?����dz�>\@�Q�C�� ?����b�\?�Ap�C��3                                    Bx�Y"�  �          @p  ?˅�Vff>��R@�  C��{?˅�U�>��H@��C��f                                    Bx�Y1Z  �          @p��?��R�aG�����C��{?��R�aG�<�?�C���                                    Bx�Y@   �          @w���\)�X�ÿ������HC����\)�]p������HC���                                    Bx�YN�  �          @�  �8Q��dz��
�H��=qC�K��8Q��k���p����HC�u�                                    Bx�Y]L  �          @�����\���5��&��Ce�q��\�%��-p����Cg�{                                    Bx�Yk�  �          @�����G��C�
�9�\Cg�����H�<(��0\)Ch�
                                    Bx�Yz�  �          @����(��{�>�R�2�HCk(���(��'��6ff�)Q�Cl��                                    Bx�Y�>  �          @�����
����>{�1z�Ci�R���
�&ff�5�((�Ck�                                    Bx�Y��  �          @�����{�?\)�6��Ce�\����
=�8Q��.{CgT{                                    Bx�Y��  �          @�Q��녿���<���=��C]}q�녿�
=�7
=�6ffC_��                                    Bx�Y�0  T          @����z��=q�<���;Q�C]� �z��(��7
=�3�HC_��                                    Bx�Y��  �          @�=q�Q��  �@  �=��C[u��Q����:=q�6�RC]�f                                    Bx�Y�|  �          @���#33��\)�@���<��CK�3�#33��G��<���8
=CNG�                                    Bx�Y�"  �          @��
�ff��Q��A��>��CSs3�ff��=q�=p��8�HCUٚ                                    Bx�Y��  �          @������33�G��:
=C_�\����(��A��2��Ca�R                                    Bx�Y�n  �          @�녿�(��Q��@���0�HCfQ��(�� ���8���(��Cg�)                                    Bx�Z  �          @�33��
=�%�9���'��CiG���
=�.{�2�\�Q�Cj�
                                    Bx�Z�  T          @��H� ���,���/\)�Q�CiL�� ���4z��'
=��CjxR                                    Bx�Z*`  �          @��� ���<(��!G��33Ck�H� ���C33�����Cl�{                                    Bx�Z9  �          @�p���z��L�������Co{��z��S33��R��=qCoٚ                                    Bx�ZG�  �          @����
=�\�������Cp�Ϳ�
=�b�\�   �ң�Cqh�                                    Bx�ZVR  �          @�p�����\��������Cqn����a녿����{Cq�q                                    Bx�Zd�  �          @�(���
=�aG���  ��Q�Cq5ÿ�
=�e�������Cq�                                    Bx�Zs�  �          @�녿����j=q������RCv������n�R���R���Cw�                                    Bx�Z�D  �          @�ff���j=q��Q���C|E���n�R�����(�C|�\                                    Bx�Z��  �          @�z῏\)�g
=��
=��33C|�ÿ�\)�k����
���C}                                      Bx�Z��  �          @��
��
=�e�����C{�=��
=�i����  ���RC|�                                    Bx�Z�6  �          @��
��\)�h�ÿ�����
C|ٚ��\)�l�Ϳ�33��
=C}�                                    Bx�Z��  �          @�{�0���o\)�ٙ���
=C��׿0���s33�Ǯ��  C��R                                    Bx�Z˂  �          @�p�����s33��z����HC�f����w
=�\���C�                                    Bx�Z�(  �          @��
��G��s�
��  ����C�����G��w
=��{��  C��
                                    Bx�Z��  �          @~{�W
=�y���#�
��C�}q�W
=�z�H�   ���C��                                     Bx�Z�t  �          @y���B�\�w
=������C����B�\�w��Ǯ���C���                                    Bx�[  �          @\)����q녿�G��lz�C�xR����tz�^�R�L��C��                                     Bx�[�  �          @��xQ��w
=���\��33C���xQ��y�������|  C�f                                    Bx�[#f  �          @�{��  �y����33�33C�)��  �|(����\�aG�C�q                                    Bx�[2  �          @�\)�+����׿�\)�up�C�G��+������z�H�W�C�Q�                                    Bx�[@�  �          @���fff�|(���  �^�\C���fff�~{�^�R�A��C��q                                    Bx�[OX  T          @��H�p���u����p��C�"��p���w��p���TQ�C�0�                                    Bx�[]�  T          @�����\�x�ÿ��\�c�C@ ���\�z�H�fff�H  C\)                                    Bx�[l�  �          @�(��xQ��{��B�\�)p�C�\�xQ��|�Ϳ#�
�=qC��                                    Bx�[{J  �          @�ff�}p��~�R�\(��=�C�
=�}p���Q�=p��#\)C�{                                    Bx�[��  T          @�����
�q녿��H���RC~�3���
�tzΉ��w�C~��                                    Bx�[��  �          @|(��&ff�s�
<�>��C�&f�&ff�s�
>\)@�C�&f                                    Bx�[�<  �          @����\)�S33@Q�B\)C�lͽ�\)�N�R@{BC�j=                                    Bx�[��  �          @�p�=����I��@,(�B!�C���=����E�@1G�B'�C��=                                    Bx�[Ĉ  �          @�>�z��L��@Dz�B.�C��
>�z��G�@I��B4C���                                    Bx�[�.  R          @�\)?���Mp�@C�
B,G�C�L�?���HQ�@H��B2�C�n                                    Bx�[��  "          @���?fff�>�R@FffB3\)C�e?fff�9��@J�HB8�HC��)                                    Bx�[�z  �          @�G�?�\�`��@�B  C�'�?�\�\��@p�B��C�9�                                    Bx�[�   �          @�z�#�
�hQ�?��HA�
=C����#�
�e�@33A�{C���                                    Bx�\�  T          @���\)�j�H?�p�A�  C�˅��\)�hQ�@z�A��C���                                    Bx�\l  �          @�Q쾊=q�q�?��HAڣ�C��R��=q�n�R@�\A���C���                                    Bx�\+  �          @�녾\)�}p�?У�A��HC��R�\)�z�H?�(�A���C��
                                    Bx�\9�  �          @��<���z�?���A��C�=q<����
?�
=A��C�>�                                    Bx�\H^  �          @�Q�=�\)��G�?�=qA�Q�C�� =�\)��Q�?�A�C��H                                    Bx�\W  �          @�=�\)�}p�?��A��C�� =�\)�{�?�A�
=C��H                                    Bx�\e�  �          @}p�=u�u?h��AUG�C�y�=u�tz�?z�HAf�HC�y�                                    Bx�\tP  �          @|��>�=q�x��?�RA  C�  >�=q�xQ�?0��A!�C�H                                    Bx�\��  �          @�p�>#�
���\?5A��C�  >#�
��=q?G�A-p�C�!H                                    Bx�\��  R          @j�H�����e?:�HA733C�:�����e�?J=qAG
=C�:�                                    Bx�\�B  T          @��R�W
=��G�?�z�A~�HC����W
=����?�(�A�
=C���                                    Bx�\��  �          @��׾�33��=q?�  A��C����33����?��A�33C���                                    Bx�\��  �          @��׿����?�  AYG�C�]q����33?��Ag33C�Y�                                    Bx�\�4  T          @���������
?fffAC\)C�0�������33?uAP��C�.                                    Bx�\��  �          @�p�>.{�w�?�  A��C�8R>.{�vff?ǮA��C�9�                                    Bx�\�  �          @�{?���vff?�p�A��\C�{?���u�?��
A�z�C��                                    Bx�\�&  �          @�����o\)?�(�AîC�˅����n{?�\A�\)C��                                    Bx�]�  �          @�\)<#�
�w
=?�(�A�Q�C�R<#�
�u?�\A�C��                                    Bx�]r  T          @��
=��
�w�?��A��C���=��
�vff?�
=A���C��)                                    Bx�]$  �          @�33�Ǯ�|(�?�G�Ac�
C�33�Ǯ�{�?�ffAmG�C�0�                                    Bx�]2�  �          @���\�s�
?�=qA���C���\�r�\?�\)A�  C��                                    Bx�]Ad  �          @�=q��R�^�R?��RA�33C��)��R�^{@�A�G�C���                                    Bx�]P
  �          @����Ǯ�p��?�33A��HC�;Ǯ�p  ?�
=A��RC�
=                                    Bx�]^�  �          @w��\�aG�?��A��RC��=�\�`  ?���A�(�C��                                    Bx�]mV  �          @~�R����k�?���A���C��=����j�H?�p�A�{C�Ǯ                                    Bx�]{�  �          @��ý�\)�q�?��A�\)C�uý�\)�qG�?�z�A�(�C�u�                                    Bx�]��  �          @|(�<��n�R?�G�A�p�C�AH<��n{?��
A�  C�AH                                    Bx�]�H  �          @xQ�����e�?�A���C��{�����dz�?�Q�A���C��3                                    Bx�]��  T          @�z�?8Q���Q�?
=A�RC�R?8Q���Q�?��AffC�R                                    Bx�]��  �          @�ff?�
=����>�Q�@�p�C���?�
=����>\@�33C���                                    Bx�]�:  �          @���?�(����H>���@s�
C��
?�(����H>��R@|��C��
                                    Bx�]��  �          @���?�=q���
>�Q�@���C��?�=q���
>�Q�@�(�C��                                    Bx�]�  �          @���?�G�����>u@@  C�p�?�G�����>u@C�
C�p�                                    Bx�]�,  �          @�  ?����33>�{@��HC��{?����33>�{@��C��{                                    Bx�]��  �          @�G�?�  ����>��
@�G�C���?�  ����>��
@���C���                                    Bx�^x  �          @��R?������>W
=@,��C�%?������>W
=@(��C�%                                    Bx�^  �          @���?����>�=q@Z=qC��?����>��@S�
C��                                    Bx�^+�  �          @�\)?�����H>�p�@��C��)?�����H>�Q�@�33C���                                    Bx�^:j  �          @�ff?Tz���33>�
=@�
=C�l�?Tz���33>��@�G�C�l�                                    Bx�^I  �          @��R?xQ���=q>�@�C�P�?xQ���=q>�@��RC�P�                                    Bx�^W�  �          @�ff?n{���\>�33@�  C��?n{���\>���@��C��                                    Bx�^f\  T          @��?Y������>��@QG�C�xR?Y������>k�@>{C�w
                                    Bx�^u  �          @�=q?\(���\)>k�@9��C�u�?\(���\)>L��@#33C�u�                                    Bx�^��  �          @�Q�?�=q��33>Ǯ@�  C���?�=q��33>�Q�@��C��                                    Bx�^�N  �          @�{?J=q���H>�33@�G�C�/\?J=q���H>��
@�33C�/\                                    Bx�^��  �          @��?O\)��=q=�G�?�33C�U�?O\)���\=�\)?k�C�U�                                    Bx�^��  �          @�ff?=p���(�<��
>��C�Ǯ?=p���(����
���C�Ǯ                                    Bx�^�@  �          @��R?�z����?�@�C���?�z����?�@ٙ�C��q                                    Bx�^��  �          @��?Tz���G�?}p�AW�
C��?Tz�����?s33AN=qC��=                                    Bx�^ی  �          @�?Y���~{?z�HAYG�C�\?Y���~�R?n{AO
=C��                                    Bx�^�2  �          @�=q?s33����?8Q�A��C�l�?s33���?+�A�\C�h�                                    Bx�^��  �          @���?�G����?
=q@�=qC���?�G���  >��H@��HC��
                                    Bx�_~  �          @��?����ff>��@�z�C�\?����ff>�33@�(�C��                                    Bx�_$  �          @���?^�R��p��u�W
=C��H?^�R��p�����z�C��H                                    Bx�_$�  �          @��?G����׼���p�C��H?G����׽�Q쿝p�C��H                                    Bx�_3p  �          @���?Tz����=���?��C�Ǯ?Tz����<�>�G�C�Ǯ                                    Bx�_B  �          @��\?����z�H>�\)@|(�C��)?����{�>W
=@@��C���                                    Bx�_P�  �          @��H?\(��\)=�Q�?��
C�{?\(��\)<��
>�z�C�3                                    Bx�__b  �          @�(�?J=q��G�����G�C���?J=q��G��aG��C33C���                                    Bx�_n  �          @�z�?:�H���\��G�����C�{?:�H��=q�B�\�)��C��                                    Bx�_|�  �          @��?\(����׽��Ϳ�Q�C��?\(���Q�8Q��#�
C�
=                                    Bx�_�T  �          @�p�?c�
��=q��\)�z�HC�#�?c�
��=q�#�
���C�%                                    Bx�_��  �          @�?G���33����C�j=?G���33�u�S33C�k�                                    Bx�_��  �          @��R?L�����
����=qC��H?L�����
�k��E�C���                                    Bx�_�F  �          @�{?=p����
�W
=�<(�C��?=p�������R���C�{                                    Bx�_��  �          @��?
=q��ff�\)��
=C���?
=q��ff�u�Q�C���                                    Bx�_Ԓ  �          @�
=?�������{���
C�q?�����;�G���Q�C�                                      Bx�_�8  T          @�ff?333���
������
=C���?333��33�   ����C�Ф                                    Bx�_��  �          @��?^�R���׿#�
�G�C�#�?^�R��  �=p��$��C�+�                                    Bx�` �  �          @��?\)��Q�&ff�ffC�  ?\)�\)�B�\�*�HC�                                    Bx�`*  �          @��\>�(��~�R�&ff�(�C��>�(��}p��B�\�-p�C��                                    Bx�`�  �          @�p�>�
=��(��\����C��H>�
=���
�   ����C���                                    Bx�`,v  �          @�{>�G������z��~{C��)>�G����;����z�C���                                    Bx�`;  �          @���>�p����׾�p�����C�y�>�p���  ���H�ӅC�z�                                    Bx�`I�  �          @���>�\)��  ��p���  C��>�\)����   �أ�C��                                    Bx�`Xh  �          @�\)>#�
��{��\�޸RC�3>#�
���#�
�Q�C�{                                    Bx�`g  �          @�z�?8Q����ý�G���p�C��f?8Q����þ�  �S�
C���                                    Bx�`u�  �          @��R?�33����=�G�?�\)C�t{?�33��녽#�
���C�t{                                    Bx�`�Z  T          @�  ?xQ���z�����C�<)?xQ���zᾏ\)�c33C�>�                                    Bx�`�   �          @��?s33��z��G���Q�C��?s33��(���=q�X��C�R                                    Bx�`��  �          @�Q�?O\)���\)��p�C�'�?O\)��p���z��o\)C�*=                                    Bx�`�L  �          @�
=?��H��������G�C��{?��H�����B�\���C���                                    Bx�`��  �          @��?�����33�B�\�\)C��?������H��Q����C��                                    Bx�`͘  �          @�{?��\��=q�u�G�C���?��\����������C���                                    Bx�`�>  T          @�{?n{��=q��p���33C�3?n{�����
=q���C��                                    Bx�`��  T          @���?W
=���R��{���C�P�?W
=��{����G�C�U�                                    Bx�`��  �          @�p�?E���33���R�s�
C�Ǯ?E����\�   ���HC�˅                                    Bx�a0  �          @���?��\���H�Tz��(Q�C���?��\������G��M�C���                                    Bx�a�  �          @�G�?�  ����Q��%��C�q�?�  ��=q��  �K
=C��                                     Bx�a%|  �          @���?u����Tz��)�C�/\?u��=q���\�O�C�=q                                    Bx�a4"  �          @�?p����  �Y���&�HC��?p����ff����N{C��                                    Bx�aB�  T          @�\)?+���(��0����\C��?+����H�fff�.�HC�!H                                    Bx�aQn  �          @���?���{�Y���#�C�5�?����Ϳ���L��C�=q                                    Bx�a`  �          @�
=?.{���\�n{�6=qC�>�?.{���ÿ���`  C�J=                                    Bx�an�  �          @��R?(����\�^�R�)p�C�Ф?(���G���=q�T(�C�ٚ                                    Bx�a}`  T          @�
=>�����=q����UG�C�}q>�����Q쿦ff��ffC��                                    Bx�a�  T          @�p�>���Q쿎{�[�C��=>���ff�������
C��{                                    Bx�a��  �          @�(�>�����(���Q����C���>���������z�����C��=                                    Bx�a�R  �          @�p�>�����
�˅���C���>�����ÿ�ff��{C���                                    Bx�a��  �          @�  >W
=��
=��ff��G�C�ff>W
=��z��G�����C�n                                    Bx�aƞ  �          @�  >����  ��
=��  C��f>����p���33���C��{                                    Bx�a�D  �          @~�R>��q녿�p���33C�xR>��mp���
=���C���                                    Bx�a��  �          @G
==�G��7
=�xQ���
=C�q=�G��3�
��\)��p�C�#�                                    Bx�a�  T          @\)��\)�33�\(����RC��;�\)�  �z�H��p�C�xR                                    Bx�b6  �          @������{�8Q���G�C������
�H�W
=���C��3                                    Bx�b�  �          @(��0���
�H�^�R����C|ff�0�����}p���  C{�q                                    Bx�b�  �          @\)��Q��ff��G�����C�/\��Q��녿����33C��                                    Bx�b-(  �          @�R�0�׿�Q쿪=q�ffCzxR�0�׿�{��Q����Cy�3                                    Bx�b;�  �          @$z�c�
����Q��&�\Cq�c�
���ÿ��
�2G�Cpp�                                    Bx�bJt  �          ?�>�\)�B�\?=p�B(C��>�\)�L��?0��B  C��\                                    Bx�bY  �          ?��?B�\���
?\(�BA�C��?B�\��?Y��B?=qC��                                    Bx�bg�  �          ?��H?(�ÿG�?#�
A�  C�#�?(�ÿQ�?
=A�\C�}q                                    Bx�bvf  �          ?��
?�Ϳ\(�?333B	(�C�` ?�Ϳfff?#�
A�ffC��                                    Bx�b�  �          ?�p�?�R�=p�?@  BC�Ǯ?�R�J=q?333B��C���                                    Bx�b��  �          ?�33?G����H?J=qA�z�C�w
?G���G�?8Q�A�  C��
                                    Bx�b�X  �          ?�33?B�\���\?:�HA��
C�XR?B�\����?&ffA�z�C��                                    Bx�b��  �          ?�(�?J=q��{?333A�
=C�?J=q��33?(�A�33C���                                    Bx�b��  T          @�\?s33��{?O\)A��HC�\)?s33��z�?333A��HC���                                    Bx�b�J  �          @�?����(�?L��A��HC��)?���G�?+�A�C���                                    Bx�b��  �          @��?�{��z�?��A�Q�C�  ?�{��p�?k�A�p�C��{                                    Bx�b�  �          @
=?�ff���H?��RA���C��?�ff���?�\)A��HC�9�                                    Bx�b�<  �          @�R>L��?�Q�?�(�B#  B��>L��?˅?�=qB2�B���                                    Bx�c�  �          @(�>�  @G�?��B��B��q>�  ?�?�p�B�B�                                      Bx�c�  �          @�>�p�?�\)?�  B33B��\>�p�?�G�?У�B)
=B�.                                    Bx�c&.  �          @��>�33?��
?�ffB"\)B�>�33?�z�?�
=B2ffB�=q                                    Bx�c4�  �          @��?\)?�\)?ٙ�BC�HB��?\)?��R?��BS�B�{                                    Bx�cCz  �          @ff?
=?�33?У�BM�
B{p�?
=?��
?��HB\��Bp�R                                    Bx�cR   �          ?�33?�R?��?�
=BEBm��?�R?s33?�G�BT(�Bb��                                    Bx�c`�  �          ?�  ?G�?   ?�(�Bf��B�R?G�>Ǯ?�G�Bo{A�Q�                                    Bx�col  T          ?�?k�>�  ?ǮBi��Av{?k�>\)?�=qBm\)Ap�                                    Bx�c~  �          ?�  ?Y��=#�
?\Bsff@,��?Y�����
?\Bs
=C�l�                                    Bx�c��  �          ?�Q�?Tz�=L��?�(�Bq�@fff?Tz�u?�(�Bq�
C���                                    Bx�c�^  �          ?��?J=q=��
?�ffBk33@��?J=q���
?��Bk�C�J=                                    Bx�c�  �          ?�\)?u��Q�?��RBI\)C��)?u��?���BB33C�.                                    Bx�c��  �          ?�{?�
=�s33?��B	p�C���?�
=���
?uA�p�C��                                    Bx�c�P  �          ?�Q�?�(�����?c�
A�33C�� ?�(����\?J=qA���C��R                                    Bx�c��  �          ?�\)?�Q쿇�?uA���C�(�?�Q쿐��?aG�A�Q�C�/\                                    Bx�c�  �          ?���?����Q�?k�A��HC�?����G�?Q�A�=qC��                                    Bx�c�B  �          ?�z�?Y������?�A��C�e?Y����{>��Ac�C�f                                    Bx�d�  �          ?�z�?W
=��ff?
=A�{C�q�?W
=����>�A���C�H                                    Bx�d�  �          ?�=q?}p���33?�\A�C��\?}p���
=>ǮAL  C�P�                                    Bx�d4  �          @   ?�녿�?O\)A�33C�Z�?�녿��R?.{A�ffC���                                    Bx�d-�  
�          @
=q?�z����?c�
A�Q�C�:�?�z��33?@  A�=qC��                                     Bx�d<�  �          @�?�G���p�?}p�AƏ\C�H?�G���?W
=A�  C�ff                                    Bx�dK&  �          @-p�?��ÿ�\)?�A̸RC���?��ÿ��H?�  A��C�Ff                                    Bx�dY�  �          @Fff?�ff�"�\?k�A�\)C��=?�ff�'
=?0��AL��C�O\                                    Bx�dhr  �          @(�?�{���R?�33A�{C�j=?�{��=q?�G�A���C��                                    Bx�dw  �          @�\?��
����?���B=qC��)?��
����?xQ�A�
=C��3                                    Bx�d��  �          @G�?�G���G�?�B�\C�^�?�G���\)?�ffA��C�L�                                    Bx�d�d  �          @�\?�{����?��HB��C�Y�?�{��ff?���B��C�#�                                    Bx�d�
  �          @G�?��
��
=?��\B\)C���?��
���?�z�B
=C�O\                                    Bx�d��  �          @   ?�ff�˅?�B
�RC���?�ff���H?��\A�C���                                    Bx�d�V  �          @\)?�ff��z�?�=qBp�C�Q�?�ff��ff?�Q�B�C���                                    Bx�d��  �          @%�?����?ٙ�B$=qC���?������?ǮB��C�:�                                    Bx�dݢ  T          @(Q�?��Ϳ�Q�?޸RB%p�C���?��Ϳ���?˅B��C�R                                    Bx�d�H  �          @)��?��ÿ�33?���B-p�C��=?��ÿ���?�
=B�
C�
=                                    Bx�d��  �          @#33?��\���?��
B1Q�C��?��\��p�?��B C�P�                                    Bx�e	�  �          @�
?�녿�\)?�B8�
C���?�녿��
?�ffB(z�C��                                    Bx�e:  �          @?�
=��p�?˅B,(�C��=?�
=����?��HB=qC�AH                                    Bx�e&�  �          @G�?�Q쿨��?�z�B�C��?�Q쿹��?��\B{C���                                    Bx�e5�  �          @33?��R��(�?�  B#33C��)?��R��{?�\)Bz�C��                                    Bx�eD,  �          @"�\?z�H?�?�B;�B]ff?z�H?�p�?�(�BM�HBM=q                                    Bx�eR�  �          @�H?Y��?�G�?ٙ�B1��BrQ�?Y��?�=q?���BF(�BeG�                                    Bx�eax  �          @  ?L��>.{@G�B�AC�
?L�ͽ#�
@�B�C���                                    Bx�ep  "          @�?5?�ff?��B9�RB��\?5?���?�Q�BO33Bx�                                    Bx�e~�  �          @p�?:�H?�G�?�ffB<
=B�aH?:�H?��?���BQp�Bs��                                    Bx�e�j  �          @!G�?L��?�
=?�33BD�RBr��?L��?�(�@�\BY�Bb                                    Bx�e�  �          @�
?�(�>aG�?˅BP
=A!p�?�(�=L��?���BR�@(�                                    Bx�e��  �          @��?��׿���?��B(�C���?��׿�p�?�
=B=qC��                                    Bx�e�\  �          @�?�G���G�?��B33C��?�G����?�\)A㙚C���                                    Bx�e�  �          @=q?��ÿ�p�?�ffB��C���?��ÿ�{?���A��C���                                    Bx�e֨  �          @�?Ǯ?��?޸RB:�A�Q�?Ǯ>�33?��BA�AK
=                                    Bx�e�N  �          @�?�G��333?�z�B3��C���?�G��aG�?�=qB((�C��
                                    Bx�e��  �          @�?��
�Ǯ?��B�C��)?��
��Q�?���Aߙ�C���                                    Bx�f�  �          @&ff?�\)��ff?��HA��\C��f?�\)��
=?�  A�
=C��q                                    Bx�f@  T          @,��?����?@  A��C��f?�����?   A3�C�g�                                    Bx�f�  "          @QG�?Y���I���L�ͿfffC���?Y���G������RC��H                                    Bx�f.�  �          @\��?#�
�XQ�=u?k�C�P�?#�
�W��������C�W
                                    Bx�f=2  �          @c33?�\�`  ���\)C�  ?�\�^�R��G����
C�'�                                    Bx�fK�  �          @_\)?�R�\(����
��z�C�R?�R�Z=q��
=�ۅC�!H                                    Bx�fZ~  �          @Tz�>��QG�=L��?Tz�C�*=>��P�׾�{��=qC�0�                                    Bx�fi$  �          @n�R?��l(�<�>�ffC�f?��j�H�����Q�C��                                    Bx�fw�  �          @p��?.{�j�H��33���
C�9�?.{�fff�G��A�C�Q�                                    Bx�f�p  �          @mp�?.{�e��+��%C�XR?.{�^�R�������C��                                     Bx�f�  �          @p  >B�\�l(������\C�q�>B�\�e�����z�C�|)                                    Bx�f��  �          @r�\=�G��o\)�
=�33C��=�G��h�ÿ���~ffC��3                                    Bx�f�b  �          @u<#�
�l�Ϳz�H�n=qC��<#�
�c33��
=���C�)                                    Bx�f�  �          @l�ͽ�Q��XQ쿼(����C�@ ��Q��K������Q�C�33                                    Bx�fϮ  �          @Dz�>�\)��\>�(�A*ffC�p�>�\)�z�>��@fffC�c�                                    Bx�f�T  �          @Q�>��33?���Aܣ�C��\>��
�H?Tz�A��HC�5�                                    Bx�f��  �          @$z�k��G�>�Q�AffC�  �k��33=�\)?�  C�'�                                    Bx�f��  �          @��#�
�z�?�ffA�=qC|��#�
�(�?G�A��HC}�H                                    Bx�g
F  �          @{�\)�?   A=p�C�8R�\)�Q�>L��@���C�XR                                    Bx�g�  �          @ �׾�G��(�>��R@�33C��f��G��p�    ����C��                                    Bx�g'�  �          @&ff��G��#33>W
=@��C�  ��G��#�
���(��C�"�                                    Bx�g68  �          @(�ý��
�&ff�B�\��C�  ���
�#�
���9p�C�)                                    Bx�gD�  �          @�<#�
��<�?c�
C�{<#�
��
�u���C�{                                    Bx�gS�  �          @ff>��G�>�AMG�C��R>���
>B�\@��HC��                                    Bx�gb*  �          @�=u�>��
A�
C���=u��=#�
?���C��H                                    Bx�gp�  �          @�\�#�
�G�>L��@���C�|)�#�
�녽����#�
C�}q                                    Bx�gv  �          @�;B�\��=�G�@,(�C��\�B�\���W
=��p�C��                                    Bx�g�  �          @0  ���
�0  <��
>�ffC�(����
�.{��33��
=C�'�                                    Bx�g��  �          @Dzᾣ�
�AG���(��   C�𤾣�
�;��Tz��z�HC�ٚ                                    Bx�g�h  �          @R�\���H�I���Q��g\)C��R���H�@�׿��R��
=C�e                                    Bx�g�  �          @Vff�   �H�ÿ��\���HC�t{�   �>{��Q���z�C�4{                                    Bx�gȴ  �          @K���ff�=p������z�C��f��ff�2�\���H�ڏ\C�b�                                    Bx�g�Z  �          @N�R�(�������
���C~���(������<�HC|^�                                    Bx�g�   �          @I���=p�������1��Cz�׿=p�����{�Oz�Cwu�                                    Bx�g��  �          @.�R�\)��\)��ff�+  C}(��\)������\�I\)Cz��                                    Bx�hL  �          @\)�L���p�>��R@�{C�N�L����R    ��Q�C�P�                                    Bx�h�  �          @=q�.{�
=>ǮA��C��q�.{���=L��?��C��                                    Bx�h �  �          @   �#�
�p�>��
@�
=C�)�#�
�\)���#�
C�                                      Bx�h/>  �          @=q�������>L��@���C��f��������\)�XQ�C��f                                    Bx�h=�  �          @�H<#�
���>�{A33C�&f<#�
�=q    �#�
C�&f                                    Bx�hL�  �          @"�\>��R�=q?0��A}�C��3>��R�\)>�{@�(�C��
                                    Bx�h[0  �          @��>�p����>�z�@��C�j=>�p��=q�L�Ϳ��C�b�                                    Bx�hi�  �          @{>��
��>k�@�(�C��)>��
�(����8��C���                                    Bx�hx|  �          @�R>��R�(�>8Q�@��C��>��R�(��8Q�����C��                                    Bx�h�"  �          @ ��?�p���{@33Bh�\C�.?�p��#�
?�p�B\{C�8R                                    Bx�h��  �          @2�\?�33>�ff@
=Bh�\A�(�?�33=���@=qBo�@��                                    Bx�h�n  �          @/\)?�33>u@Bk(�A��?�33��G�@ffBl��C��                                    Bx�h�  �          @%�?��\?J=q@�RBo��B�?��\>�ff@�B��RA�                                      Bx�h��  �          @�\?L�Ϳ��\?k�B{C��?L�Ϳ��?@  A��HC�y�                                    Bx�h�`  �          @#33>��H�   =L��?�=qC��{>��H��R������z�C��                                     Bx�h�  �          @�?�G���p�>�=q@ڏ\C�w
?�G��   �#�
�k�C�\)                                    Bx�h��  �          @$z�?\(���=u?��\C��?\(��=q���R��ffC��
                                    Bx�h�R  �          @��?Tz��
�H>�\)@�(�C���?Tz����L�Ϳ�ffC�p�                                    Bx�i
�  �          @�
?Ǯ�B�\?��
B%\)C��R?Ǯ�}p�?��B�C��3                                    Bx�i�  �          @�?���?��BG�C�� ?��J=q?�Q�B8=qC�h�                                    Bx�i(D  �          @�H?��
�8Q�?�BR=qC�L�?��
��G�?��
B>Q�C��                                     Bx�i6�  �          @��?�녿!G�?�z�BL��C�?�녿k�?��B;�\C�=q                                    Bx�iE�  �          @p�?����z�?�33BI��C�  ?����^�R?��B9��C���                                    Bx�iT6  �          @(�?��H��?�z�BM=qC�K�?��H�@  ?�B?p�C�aH                                    Bx�ib�  �          @�?�z�8Q�?��BX=qC��?�z῁G�?޸RBB�\C�q�                                    Bx�iq�  �          @�?��\�8Q�?�{BO\)C�E?��\��  ?�(�B:��C��q                                    Bx�i�(  �          @�?���333?���BN
=C���?���}p�?�(�B:
=C�N                                    Bx�i��  �          @#�
?�G����?��HBJG�C��\?�G��\(�?�B:C�0�                                    Bx�i�t  �          @(��?�p����?�(�BA�RC���?�p���?�z�B9�C��
                                    Bx�i�  �          @*�H?�33��ff@�\BG�RC�b�?�33�E�?�Q�B:�HC�|)                                    Bx�i��  "          @1�?�{�#�
@\)BX�C�5�?�{��@�BQ{C�
                                    Bx�i�f  T          @-p�?���>W
=@
�HBW�@�Q�?��þ\)@
�HBW��C�q�                                    Bx�i�  �          @ff?
=q�\(�?O\)BQ�C�f?
=q�z�H?(��A�C�b�                                    Bx�i�  �          @=q�����>aG�@�G�C��{����þ.{��(�C��{                                    Bx�i�X  �          @��<��
=>�z�@�\C�g�<��Q�������C�ff                                    Bx�j�  �          @��>���
=?\)Amp�C��>����>W
=@���C�j=                                    Bx�j�  �          @�>8Q��?
=Az�HC�q�>8Q��
=q>k�@��C�]q                                    Bx�j!J  �          @�
>���\>�  @�
=C��H>��33�\)�W
=C���                                    Bx�j/�  �          @p�=�\)��ͼ#�
�uC��=�\)�
�H�\�C��f                                    Bx�j>�  T          @p�>���(�=#�
?���C��{>���
�H�����Q�C���                                    Bx�jM<  �          @��=�Q��
=q��(��3�C�,�=�Q��33�J=q��\)C�<)                                    Bx�j[�  �          @�\>���p����W
=C�H>����fff��\)C��                                    Bx�jj�  �          @�H>L���
=���H�;�C�p�>L���\)�c�
���
C���                                    Bx�jy.  �          @ff>W
=�33��(��*ffC���>W
=�(��Q���p�C��                                    Bx�j��  �          @�
>�Q���׾�����p�C���>�Q��
�H�.{��G�C���                                    Bx�j�z  �          @G�>��R��R����ϮC���>��R�
=q�#�
��ffC�#�                                    Bx�j�   �          @33>Ǯ�  �k����C��>Ǯ�
�H��R�y��C��                                    Bx�j��  �          @>���33����l��C�>���\)����XQ�C�'�                                    Bx�j�l  �          @�
?�R�p�=���@'
=C��{?�R�(���\)�ᙚC��H                                    Bx�j�  �          @�
?��\)=�\)?�  C��
?��{���
� ��C���                                    Bx�j߸  �          @��>�33�����
��33C�8R>�33�z���H�>=qC�O\                                    Bx�j�^  �          @�R?   ��þk���ffC���?   ��
�&ff�u�C��                                    Bx�j�  �          @$z�?.{�p��B�\���\C��)?.{��ÿ!G��b{C���                                    Bx�k�  �          @%?\)� �׾�����z�C�>�?\)��H�=p�����C�y�                                    Bx�kP  �          @�>�z���R��p���C���>�z����B�\��=qC��                                    Bx�k(�  �          @ff>�{��\���� ��C�7
>�{���O\)��C�l�                                    Bx�k7�  �          @!�>���;����C�^�>����W
=��{C��                                     Bx�kFB  �          @��?:�H��\<��
>ǮC�Ф?:�H��׾�����
C��                                    Bx�kT�  �          @��?333��=�\)?���C�e?333�33��Q��=qC�|)                                    Bx�kc�  �          @33?333�(�<#�
>���C��?333�	���\�=qC��                                    Bx�kr4  �          @
=?+���׾��C33C�O\?+���Ϳ��QG�C��f                                    Bx�k��  �          @=q?0����
���:=qC�P�?0���\)�
=q�O�C���                                    Bx�k��  �          @�?W
=�p���G��1�C�s3?W
=�	�����I�C���                                    Bx�k�&  �          @
=?Tz��p�=�\)?�33C�B�?Tz�����{�Q�C�^�                                    Bx�k��  �          @�?��  >W
=@��C��?��  �W
=���C��                                    Bx�k�r  �          @�>�G����>�\)@�Q�C�~�>�G��G��\)�]p�C�w
                                    Bx�k�  �          @��?O\)�  ���&ffC��3?O\)��;�G��+33C�!H                                    Bx�kؾ  �          @=q?G��G�>��@c33C���?G���׾�\)��33C��\                                    Bx�k�d  �          @�>�  �>B�\@�\)C��>�  ���  ��\)C�                                    Bx�k�
  �          @"�\?@  ��=�\)?�  C���?@  ��þǮ���C���                                    Bx�l�  �          @$z�?0���{=L��?���C���?0������
=��C��
                                    Bx�lV  �          @%?���!G�=#�
?n{C��f?����R��(��ffC��                                     Bx�l!�  �          @'
=?8Q��   <#�
>uC��?8Q��p����#�
C�.                                    Bx�l0�  �          @#�
?5�p�=u?�G�C�?5��H����(�C�                                      Bx�l?H  �          @%�?&ff�\)=�\)?˅C�C�?&ff�p��������C�Z�                                    Bx�lM�  �          @$z�?(��\)��\)���C���?(�������Ep�C�
                                    Bx�l\�  T          @$z�?+���R�L�Ϳ��C��?+������<(�C��                                    Bx�lk:  �          @\)?   ��<��
>��HC��H?   ��þ�ff�&{C���                                    Bx�ly�  �          @   ?z���=#�
?��\C��f?z���þ�
=���C��                                     Bx�l��  �          @\)>�����=��
?�ffC��{>����H�����\)C���                                    Bx�l�,  �          @'
=>Ǯ�$z�>L��@�ffC�E>Ǯ�#�
�������
C�H�                                    Bx�l��  �          @%>��R�"�\>\AC�w
>��R�#�
���2�\C�n                                    Bx�l�x  T          @(Q�>����#33?
=qA<  C���>����'
=<�?z�C��q                                    Bx�l�  �          @,��>�=q�(Q�?�A3
=C���>�=q�+�    �#�
C���                                    Bx�l��  �          @1�>�=q�-p�?�A+�C��R>�=q�0�׼��
=qC�˅                                    Bx�l�j  �          @0��>�Q��,(�>��HA%G�C��\>�Q��/\)�#�
�n{C��                                     Bx�l�  �          @333>�p��.�R?   A&=qC��>�p��1G��#�
�fffC��{                                    Bx�l��  �          @1G�>aG��-p�?�A/\)C�XR>aG��0�׼��
��(�C�N                                    Bx�m\  �          @/\)=��+�?��A8Q�C�AH=��.�R    =�\)C�:�                                    Bx�m  �          @(��=�Q��%�?�A;
=C���=�Q��(��<#�
>W
=C��                                    Bx�m)�  �          @*=q=�\)�%?��AR{C��q=�\)�)��=�\)?�{C��R                                    Bx�m8N  �          @/\)>8Q��*=q?�RAP��C���>8Q��.�R=�\)?�G�C���                                    Bx�mF�  �          @,��>��R�(Q�?
=qA7�
C�]q>��R�+�    ���
C�K�                                    Bx�mU�  �          @*�H>�33�'
=>�ffA�C��>�33�(�ý��
���C��R                                    Bx�md@  �          @,(�>����&ff?\)A>�HC�ff>����*=q<��
>��C�N                                    Bx�mr�  T          @+�>���$z�?!G�AYC�~�>���(��=���@�
C�^�                                    Bx�m��  T          @(��>�Q��!G�?0��Ar�\C��>�Q��'
=>#�
@c33C��3                                    Bx�m�2  �          @0  >�ff�'�?5An�HC���>�ff�,��>#�
@VffC��                                     Bx�m��  �          @5�?�\�*�H?L��A��RC�s3?�\�1G�>k�@�Q�C�>�                                    Bx�m�~  �          @8Q�?:�H�,��?&ffAQC���?:�H�1�=�Q�?ٙ�C�]q                                    Bx�m�$  �          @8��?
=q�-p�?Y��A���C���?
=q�4z�>�=q@�ffC�n                                    Bx�m��  �          @:=q?G��/\)?z�A9G�C��R?G��333<#�
>��C���                                    Bx�m�p  �          @@  ?k��4z�>\@�(�C�?k��5�W
=�|��C��
                                    Bx�m�  �          @I��?����<(�>�{@��
C��?����<�;�\)��33C�                                      Bx�m��  �          @Tz�?���Fff>Ǯ@ٙ�C��\?���G������  C��                                    Bx�nb  �          @W
=?�=q�H��>�Az�C�}q?�=q�J�H�8Q��Dz�C�ff                                    Bx�n  �          @U�?p���I��?��A(�C�W
?p���L(����G�C�9�                                    Bx�n"�  �          @J�H?fff�>{?�RA6{C�n?fff�B�\�#�
�8Q�C�B�                                    Bx�n1T  T          @HQ�?k��:=q?.{AI��C��f?k��>�R=u?��C��\                                    Bx�n?�  �          @C�
?aG��6ff?5AV=qC���?aG��;�=���?��C�T{                                    Bx�nN�  �          @@��?L���3�
?8Q�A]p�C��f?L���9��=�G�@Q�C���                                    Bx�n]F  �          @>{?^�R�2�\?�A&{C��H?^�R�5��\)��G�C�|)                                    Bx�nk�  �          @@  ?c�
�4z�?�A�C���?c�
�7
=��Q��C��q                                    Bx�nz�  �          @C33?k��5�?&ffAF=qC��{?k��:=q=#�
?:�HC��q                                    Bx�n�8  �          @8��?Tz��-p�?�A6�RC�}q?Tz��1G����
���
C�Q�                                    Bx�n��  T          @3�
?8Q��*=q?
=qA0��C��)?8Q��-p��#�
�Q�C�w
                                    Bx�n��  �          @7�?5�.{?�A733C�Ff?5�1논��
��C�                                      Bx�n�*  T          @A�?!G��8Q�?+�AL(�C�*=?!G��=p�=#�
?0��C��                                    Bx�n��  �          @A�?&ff�8��?#�
AC�
C�` ?&ff�=p�<#�
>#�
C�:�                                    Bx�n�v  �          @9��?(���0  ?(��AS�C��)?(���4z�=u?�z�C��                                    Bx�n�  
�          @C�
?333�7�?E�Aj�HC�Ф?333�>{>\)@%�C��R                                    Bx�n��  
�          @K�?k��8��?k�A�33C��{?k��A�>�\)@�  C�xR                                    Bx�n�h  �          @QG�?}p��>�R?h��A���C�+�?}p��Fff>u@�
=C���                                    Bx�o  �          @P��?}p��@  ?J=qAa�C�(�?}p��Fff=�@
�HC��                                    Bx�o�  �          @J�H?aG��;�?W
=Aw\)C�aH?aG��B�\>B�\@[�C��                                    Bx�o*Z  �          @J�H?h���;�?G�Ae�C���?h���A�>�@�\C�U�                                    Bx�o9   �          @G
=?O\)�7
=?c�
A�C���?O\)�?\)>�  @��\C���                                    Bx�oG�  �          @@��>W
=�:=q?.{ARffC��>W
=�>�R<�?\)C��q                                    Bx�oVL  �          @B�\>��
�:=q?G�An�HC�!H>��
�@��=�@
=C�f                                    Bx�od�  �          @@��>�
=�9��?0��ATz�C�#�>�
=�>�R=#�
?:�HC��                                    Bx�os�  �          @@  >�33�8Q�?333AYG�C�k�>�33�=p�=L��?xQ�C�S3                                    Bx�o�>  �          @AG���R�4z�?&ffAK\)C����R�8��<#�
>�=qC���                                    Bx�o��  �          @DzῡG��-p�?.{ALQ�Cu#׿�G��333=u?�z�Cu�=                                    Bx�o��  T          @Fff�fff�6ff?B�\Aep�C|z�fff�<(�=�G�@Q�C}�                                    Bx�o�0  T          @E���(��2�\>�
=@�G�Cv^���(��3�
�W
=�|��Cv�=                                    Bx�o��  �          @Fff��z��5�?�A=qCw����z��7����\)Cw�R                                    Bx�o�|  �          @G
=����0  ?
=A.�\Ct�׿���4z�#�
�:�HCu!H                                    Bx�o�"  �          @E���
�2�\>�p�@ۅCuc׿��
�333������Cuz�                                    Bx�o��  �          @C�
���H�1�>�ffAz�Cv�����H�3�
�8Q��Tz�Cv�                                    Bx�o�n  �          @Fff����/\)?aG�A��HCwff����7
=>�  @�(�CxO\                                    Bx�p  �          @G
=��G��4z�?Q�At��CzT{��G��<(�>.{@E�C{�                                    Bx�p�  �          @G
=�u�7
=?:�HAZ�HC{xR�u�<��=�\)?�{C|�                                    Bx�p#`  �          @Fff�}p��4z�?O\)Ar�RCz�׿}p��;�>#�
@8Q�C{O\                                    Bx�p2  �          @J=q�\(��;�?=p�AYC}��\(��AG�=u?���C~33                                    Bx�p@�  	�          @I���@  �?\)?��A/
=C���@  �C33��Q���HC�{                                    Bx�pOR  "          @Dz�.{�<��>�A
{C�z�.{�>�R�aG���33C��=                                    Bx�p]�  
�          @J�H���E?   AG�C��q���HQ�W
=�w�C���                                    Bx�pl�  
�          @G���=q�@��?=p�A\��C�xR��=q�Fff=#�
?:�HC���                                    Bx�p{D  "          @E��#�
�@  >��HA  C�n�#�
�B�\�W
=�w
=C�s3                                    Bx�p��  T          @E��.{�AG�?
=qA"�HC�ff�.{�Dz�#�
�<��C�l�                                    Bx�p��  
�          @HQ콣�
�B�\?.{AIp�C�<)���
�G����z�C�@                                     Bx�p�6  
(          @HQ�>�p��;�?z�HA�33C��
>�p��E�>�z�@���C�k�                                    Bx�p��  
Z          @G�?�\�:=q?s33A�(�C�  ?�\�C33>��@��C�Ǯ                                    Bx�pĂ  
�          @E>���<(�?c�
A���C�~�>���Dz�>L��@j=qC�c�                                    Bx�p�(  
Z          @C�
>u�:=q?fffA�
=C�e>u�A�>W
=@z=qC�K�                                    Bx�p��  
�          @Dz�>�z��7�?uA��\C��f>�z��AG�>�=q@��HC�                                    Bx�p�t  �          @H��>�G��C33?\)A%��C��>�G��E����5�C��                                    Bx�p�  
�          @Fff?aG��:�H>��HA\)C�h�?aG��<�;B�\�c�
C�P�                                    Bx�q�  
(          @C�
?z��:�H?+�AI�C��?z��?\)����C��H                                    Bx�qf  "          @@  ?#�
�5�?&ffAI�C�ff?#�
�:=q���
��(�C�>�                                    Bx�q+  �          @>{?c�
�2�\?   A
=C�� ?c�
�5��#�
�Dz�C��H                                    Bx�q9�  �          @=p�?u�1G�>���@�C��
?u�1G����
��  C���                                    Bx�qHX  "          @:=q?L���1�>���@��RC�\?L���1G���33��{C�3                                    Bx�qV�  �          @<(�?����,(�>�p�@陚C�q?����,�;����p�C��                                    Bx�qe�  T          @>{?��R�*=q>���@��HC�� ?��R�+��k���z�C�h�                                    Bx�qtJ  T          @B�\?�\)�)��?��A&ffC��=?�\)�-p���Q��\)C�l�                                    Bx�q��  �          @K�?���1�?O\)Amp�C�l�?���8��>\)@!�C���                                    Bx�q��  �          @HQ�?��H�1�?E�Ac�C���?��H�8Q�=���?��C�q�                                    Bx�q�<  
�          @G�?�G��7
=?333AO33C���?�G��<(�<#�
>��C��H                                    Bx�q��  "          @Fff?���@  ?�\A(�C��=?���A녾W
=�r�\C��R                                    Bx�q��  �          @Fff>�p��A�>�A=qC���>�p��C�
��  ��33C�z�                                    Bx�q�.  "          @@  >Ǯ�<(�>�A�C��{>Ǯ�=p��u���C�˅                                    Bx�q��  T          @>{?���6ff?(�A<��C�}q?���:�H���
�˅C�`                                     Bx�q�z  �          @O\)?:�H�G
=?�A!�C��R?:�H�I���8Q��J=qC��                                     Bx�q�   "          @U>����Mp�?E�AV�RC��3>����S33�#�
�\)C�z�                                    Bx�r�  
�          @[�    �N{?��\A�(�C��R    �W�>k�@vffC���                                    Bx�rl  �          @O\)?+��Dz�?
=qA\)C�+�?+��G
=�L���j�HC�
                                    Bx�r$  
�          @B�\>W
=�<(�?@  Ac
=C��>W
=�A�<�?�C���                                    Bx�r2�  �          @I��>L���B�\?G�AeC��H>L���H��=#�
?+�C���                                    Bx�rA^  "          @H��<��C�
?5APz�C�AH<��H�ý#�
�5C�>�                                    Bx�rP  "          @H��>�=q�A�?5AS33C��=>�=q�G�����\C�xR                                    Bx�r^�  �          @E?Tz��8Q�?:�HA[�
C��?Tz��>�R<�>�C�Ǯ                                    Bx�rmP  �          @G
=?Q��9��?:�HAX��C��R?Q��?\)<#�
>�=qC���                                    Bx�r{�  �          @E?8Q��9��?E�Af�HC��q?8Q��@  =L��?��\C���                                    Bx�r��  
(          @G
=?&ff�;�?Q�Au�C�9�?&ff�B�\=���?�C�                                      Bx�r�B  T          @Fff?(���;�?@  A`��C�T{?(���A�<�?
=qC�"�                                    Bx�r��  �          @HQ�?z��=p�?J=qAj�HC��=?z��C�
=u?���C�\)                                    Bx�r��  �          @I��?J=q�:=q?aG�A�Q�C���?J=q�B�\>#�
@8Q�C�C�                                    Bx�r�4  �          @H��?��>�R?W
=Ax(�C�H?��E=���?�=qC��3                                    Bx�r��  �          @N{>��Dz�?W
=AqC�H�>��K�=��
?�{C�#�                                    Bx�r�  
(          @R�\>\�J�H?J=qA]G�C�` >\�QG�    <#�
C�Ff                                    Bx�r�&  �          @Mp��#�
�Fff?L��Af�HC��{�#�
�L��<�?   C��{                                    Bx�r��  �          @C�
<��@  ?
=qA#�C�H�<��C33�L���n{C�G�                                    Bx�sr  T          @A�=��>�R>��HA  C�"�=��@�׾�  ��ffC�                                      Bx�s  "          @Dz�aG��C33>�=q@��HC��þaG��AG�������C��                                    Bx�s+�  
�          @8��?(��,(�?
=AAp�C�^�?(��0  ���
����C�:�                                    Bx�s:d  �          @I��?����0��?��A�  C��?����:�H>�{@�C�O\                                    Bx�sI
  
�          @J�H?����1�?���A��C�Ǯ?����<��>�p�@���C�0�                                    Bx�sW�  "          @H��?��\�1�?��\A�  C��?��\�<(�>��
@���C���                                    Bx�sfV  "          @H��?h���.{?�  A��C�>�?h���<(�?��A!�C��R                                    Bx�st�  T          @mp�?�\�g
=?(�A�C�H?�\�i����z���{C���                                    Bx�s��  �          @w
=?J=q�dz�?�(�A��
C�7
?J=q�p��>��
@�\)C���                                    Bx�s�H  "          @q�?333�_\)?��\A��
C���?333�l(�>Ǯ@���C�T{                                    Bx�s��  "          @g
=?Tz��J=q?�(�A��C�c�?Tz��[�?(��A(��C��R                                    Bx�s��  �          @fff?�ff�5�?޸RA�33C�XR?�ff�J�H?}p�A���C�*=                                    Bx�s�:  T          @p��?aG��P��?�33A���C��
?aG��c�
?O\)AF�HC��R                                    Bx�s��  �          @j�H?L���P��?��RA��
C���?L���aG�?&ffA#�C�b�                                    Bx�sۆ  �          @[�?��Mp�?}p�A���C��
?��Vff>8Q�@?\)C���                                    Bx�s�,  "          @\��?+��L��?��A�(�C��?+��W�>�\)@���C���                                    Bx�s��  "          @Z�H>����O\)?�G�A�=qC��\>����X��>B�\@N{C�g�                                    Bx�tx  �          @\(�?0���Q�?333A;�C��\?0���Vff�����C��\                                    Bx�t  �          @R�\>8Q��N�R?�A�C���>8Q��P�׾�z���G�C���                                    Bx�t$�  �          @[�?s33�I��?p��A}�C�Y�?s33�R�\>�@  C��                                    Bx�t3j  �          @Z�H?\(��L��?Q�A\��C���?\(��S�
    =��
C�Q�                                    Bx�tB  
�          @^{?��W�?z�AG�C��H?��Z=q��=q���RC�t{                                    Bx�tP�  �          @n{?�33�S�
?p��Ap(�C��R?�33�\(�=���?��C�B�                                    Bx�t_\  
�          @i��?У��C�
?���A��C��?У��O\)>�33@��C�XR                                    Bx�tn  �          @`��?�z��8��?���A��RC��\?�z��Dz�>�Q�@�z�C�0�                                    Bx�t|�  T          @dz�?�
=�4z�?��
A�C�:�?�
=�>�R>���@�33C�xR                                    Bx�t�N  "          @c�
?��8Q�?�G�A�z�C��?��B�\>�=q@�(�C�h�                                    Bx�t��  �          @i��@��;�?@  A>�HC�^�@��A�<#�
>�C���                                    Bx�t��  �          @p  ?��G
=?Q�AJ�RC�Ф?��N{<�>�G�C�c�                                    Bx�t�@  �          @o\)?���G�?W
=AO�
C���?���N�R=L��?5C�1�                                    Bx�t��  
�          @mp�?���Dz�?fffAap�C��q?���L��=�?���C�9�                                    Bx�tԌ  �          @e�?�(��@  ?fffAip�C���?�(��HQ�>�@
=C�e                                    Bx�t�2  �          @b�\?޸R�>�R?G�AK�C�R?޸R�E�<��
>�z�C���                                    Bx�t��  �          @<(�>����)��?�(�A�=qC�5�>����7�?�A$��C��R                                    Bx�u ~  "          @G
=?��/\)?��A���C�Y�?��>�R?
=A0  C��                                    Bx�u$  
�          @Fff?��\�,��?J=qAk�
C���?��\�3�
=�Q�?�(�C�+�                                    Bx�u�  4          @j=q?�z��J�H?L��AHz�C��R?�z��QG��#�
�aG�C�}q                                    Bx�u,p  �          @I��?�  �333?��\A��
C��\?�  �=p�>���@�p�C�L�                                    Bx�u;  �          @u?�=q�Z�H?8Q�A,z�C�` ?�=q�_\)�����\C�'�                                    Bx�uI�  �          @\)?�(��a�?�RA�HC��?�(��dzᾏ\)��  C��H                                    Bx�uXb  �          @{�?���W�?+�Az�C��f?���[��B�\�2�\C�o\                                    Bx�ug  T          @x��?�=q�XQ�?�RAC�0�?�=q�[��u�dz�C�f                                    Bx�uu�  �          @��@33�\��?
=AC�c�@33�^�R��\)����C�AH                                    Bx�u�T  �          @�  @
=�fff?=p�A z�C�33@
=�j�H�8Q���HC���                                    Bx�u��  �          @��@ ���^{?c�
AG\)C��@ ���e�    �uC���                                    Bx�u��  �          @E�?k��:=q��=q��  C��\?k��0  �z�H���C�H�                                    Bx�u�F  �          @Mp�?��R�:�H�#�
�(��C�t{?��R�4z�B�\�^ffC���                                    Bx�u��  �          @a�?ٙ��C�
>�G�@�p�C���?ٙ��Dzᾨ����ffC�z�                                    Bx�u͒  �          @_\)?�Q��B�\>��R@�=qC��3?�Q��AG�������C��f                                    Bx�u�8  �          @]p�?У��B�\>�\)@���C�?У��AG����H��C�(�                                    Bx�u��  �          @U?����@  >�  @��C�� ?����>{�   �
{C��q                                    Bx�u��  �          @U?�=q�;�>L��@^{C�!H?�=q�8�ÿ���C�L�                                    Bx�v*  �          @c�
?��HQ�>��@�(�C��?��Fff���	��C�.                                    Bx�v�  T          @c�
?�Q��G�>u@w�C�1�?�Q��E������C�XR                                    Bx�v%v  �          @c�
?�z��HQ�>���@���C�  ?�z��Fff���H��
=C�R                                    Bx�v4  �          @l(�?���L��>�G�@ۅC���?���Mp���p���  C���                                    Bx�vB�  �          @l(�?�{�J=q>\@���C�8R?�{�J=q��
=���C�=q                                    Bx�vQh  �          @k�?����K�<��
>\C�!H?����E��E��A�C��H                                    Bx�v`  T          @c�
?�\)�@�׾�=q��C���?�\)�6ff���\��
=C���                                    Bx�vn�  �          @aG�?�ff�A�    �#�
C�XR?�ff�;��B�\�F�HC��H                                    Bx�v}Z  �          @g�?��Fff���G�C�W
?��>{�fff�fffC��                                    Bx�v�   �          @r�\?����R�\>u@k�C��3?����P  �
=�ffC��q                                    Bx�v��  �          @�(�@	���s33>���@x��C���@	���p�׿(���33C��                                     Bx�v�L  �          @�p�@
�H��33?��@�G�C��3@
�H��33���H��Q�C���                                    Bx�v��  �          @�Q�@����z�?z�@޸RC�S3@���������Q�C�J=                                    Bx�vƘ  �          @�ff@��z=q>��@�  C�
=@��y��������C��                                    Bx�v�>  �          @c33?��
�@��>B�\@H��C�P�?��
�=p������C���                                    Bx�v��  
�          @{?��
�L�Ϳ�z��N��C��3?��
�����33�ez�C�j=                                    Bx�v�  �          @#33?�\)�s33���D{C���?�\)��G��ff�^(�C�                                    Bx�w0  �          @*�H?�(���  ���R�@�C��3?�(����
�H�Z�\C��                                    Bx�w�  �          @$z�?��H��z��\�.G�C��
?��H�.{� ���M�C�t{                                    Bx�w|  �          @(Q�?�Q�\��=q�p�C��q?�Q쿊=q��z��;(�C���                                    Bx�w-"  �          @(Q�?���\��p��	\)C��H?����{���/  C�"�                                    Bx�w;�  �          @+�?��
��\�����\)C�h�?��
��녿�Q��G�C���                                    Bx�wJn  �          @?��ÿ�{�����z�C���?��ÿ�ff��z��{C��                                     Bx�wY  �          @�?�ff��Q쿕�噚C��{?�ff�����ff��RC��                                    Bx�wg�  �          @��?����녿�{�߮C�U�?��������R�{C���                                    Bx�wv`  T          @�?�Q�W
=���133C�S3?�Q���˅�L��C�ff                                    Bx�w�  �          @G�?�����G�����p�C�=q?�������G��'�
C���                                    Bx�w��  �          @\)?�{��\)�Tz���ffC�
=?�{���Ϳ��\����C�/\                                    Bx�w�R  �          @*=q?����33��G����C��?�����Ϳ.{�n=qC�z�                                    Bx�w��  �          @>{?�33�&ff�aG�����C�"�?�33�p��\(�����C��=                                    Bx�w��  �          @J�H?�G��0  �\����C�\)?�G��$zῆff��  C�33                                    Bx�w�D  �          @e�?��H�AG��Y���\��C��=?��H�-p���=q���
C�'�                                    Bx�w��  �          @h��?�33�E�xQ��w33C��?�33�0  ���H��Q�C��                                     Bx�w�  �          @|��?�Q��Tz�c�
�O�C�!H?�Q��@  ��Q���z�C�s3                                    Bx�w�6  �          @o\)?�\)�Fff�}p��uG�C��?�\)�0  ��p���(�C�
                                    Bx�x�  �          @^�R?����7
=�+��1G�C�p�?����&ff��{���\C���                                    Bx�x�  �          @Vff?޸R�5��\����C��f?޸R�)���������C��                                    Bx�x&(  T          @QG�?޸R�.�R>��@�33C�B�?޸R�-p���(���\C�\)                                    Bx�x4�  �          @I��?�z��)��>�33@�33C�f?�z��)�����
���C��                                    Bx�xCt  
�          @N�R?���!�?Y��At  C��q?���*=q>B�\@UC��                                    Bx�xR  T          @E�?����?��A�\)C���?����%?.{AO�C���                                    Bx�x`�  �          @b�\?��,��?�\)A��
C�R?��=p�?&ffA)C���                                    Bx�xof  �          @`��?����333?��A��C���?����>{>���@��C��f                                    Bx�x~  �          @j�H?ٙ��C33?333A5p�C���?ٙ��HQ콣�
���C�B�                                    Bx�x��  �          @�?�(��hQ쿕���C���?�(��N{����RC�f                                    Bx�x�X  T          @��?��q녿h���C�C�  ?��[������ȏ\C�q                                    Bx�x��  "          @��
@33�u�������\C��@33�h�ÿ��\���
C���                                    Bx�x��  �          @��?�z��x�ÿ����
C�?�z��h�ÿ�(����
C��3                                    Bx�x�J  �          @��R?�33�~�R�!G��{C��?�33�l�Ϳ�����ffC���                                    Bx�x��  T          @�?�z����
�����S�C�w
?�z��n{��
��p�C���                                    Bx�x�  �          @��@�
��\)�h���0  C��3@�
�w���Q����RC���                                    Bx�x�<  �          @�Q�@  ������ə�C�|)@  ���ͿУ���ffC�=q                                    Bx�y�  �          @���@����(�������p�C��3@����z���
����C�O\                                    Bx�y�  �          @��H@�R��녾u�*=qC�f@�R�����{�xz�C���                                    Bx�y.  �          @��@�R���Ϳ���c�C���@�R����H��z�C��                                    Bx�y-�  �          @��R?޸R��z������G�C�C�?޸R��33�,����{C�}q                                    Bx�y<z  �          @�=q?�(���  �G���ffC���?�(����\�P  �=qC��                                    Bx�yK   T          @ƸR?�G����R�����l(�C��3?�G���z��<(���G�C�u�                                    Bx�yY�  �          @���@�R�\��@Tz�B�HC�u�@�R��p�@�AԸRC��                                    Bx�yhl  T          @�G�?��
��R@�  B��C�o\?��
��@�(�B�Q�C���                                    Bx�yw  �          A��?L��?�33A
�RB���B\(�?L�Ϳ��
A
�HB��HC��3                                    Bx�y��  �          A�?�Q�?�Q�AQ�B�G�BIff?�Q�5A	B��C��                                    Bx�y�^  �          A\)?�?�(�A33B���B7�R?����A��B��C���                                    Bx�y�  �          A
=?�p�?�{A
=B�p�B)�?�p��5AQ�B��=C�8R                                    Bx�y��  �          AG�?�{?Y��A=qB��)A���?�{��\)A�B�\C���                                    Bx�y�P  �          Ap�?У�?�A�HB���A���?У׿�(�Az�B���C�˅                                    Bx�y��  �          A33?���?5A�
B�#�A��?��ÿ˅A{B�aHC�l�                                    Bx�yݜ  �          A?�z�>�ffA\)B��fAr�R?�z���HAQ�B���C�*=                                    Bx�y�B  �          A!�?\>��A   B�k�A��?\�p�A(�B�=qC�K�                                    Bx�y��  "          A(�?�?�{A(�B��
A���?���G�A�
B��
C��f                                    Bx�z	�  �          A�
@ ��>\A(�B�Q�A)@ ����\A��B���C�S3                                    Bx�z4  �          A   @z��(�A\)B���C��@z��333A��B�G�C���                                    Bx�z&�  "          A
=@��&ffA{B�\)C�aH@��?\)A�RB���C�33                                    Bx�z5�  �          A!�@��Q�A  B��=C�^�@��L(�A  B�C�q                                    Bx�zD&  
�          A"�H@����A��B�\)C�G�@�l(�A�HB�#�C�33                                    Bx�zR�  �          A&�H@�\���RA�B�(�C��H@�\��(�A
=Bv�HC�Ф                                    Bx�zar  �          A'�@#�
���
A ��B�#�C���@#�
�mp�A�RB��HC�U�                                    Bx�zp  �          A)G�@+��L��A#�B��C��=@+��,(�AB��
C�p�                                    Bx�z~�  �          A-��@8�ÿ���A%��B��HC��{@8���s�
A\)B�C��\                                    Bx�z�d  �          A1@5�$z�A&�HB���C��@5��G�A�BjffC��\                                    Bx�z�
  T          A1��@>{��(�A((�B��{C�8R@>{��\)A\)Bt=qC�                                    Bx�z��  �          A1@C33����A)B�\C�W
@C33�hQ�A Q�B�\)C��                                    Bx�z�V  �          A3
=@8�ÿ��A,  B��C��@8���k�A"�\B��C�\                                    Bx�z��  �          A1�@K��B�\A)��B��C�K�@K��Tz�A!��B�  C��f                                    Bx�z֢  �          A3�
@\�;�G�A*�RB��)C�b�@\���AG�A#�
B��RC�e                                    Bx�z�H  �          A5p�@W��W
=A,��B�Q�C��@W��\(�A$Q�B��qC�9�                                    Bx�z��  �          A5�@`  �ǮA+33B���C��@`  ���A   Bv�\C�1�                                    Bx�{�  �          A4��@@  �ٙ�A,  B�ffC�33@@  ����A (�By��C��\                                    Bx�{:  �          A4  @HQ쿦ffA+�B��fC���@HQ��w
=A!G�B  C���                                    Bx�{�  �          A.�R@[���p�A$��B�{C�%@[��l(�A
=Bz33C�p�                                    Bx�{.�  �          A.ff@e���\A#\)B��C�Ff@e�mp�A��Bv��C��                                    Bx�{=,  �          A0(�@^{�\(�A&�RB��C��@^{�VffAffB�C��q                                    Bx�{K�  T          A2�R@g
=�p��A(z�B�  C���@g
=�]p�A�
B}�C�)                                    Bx�{Zx  �          A4(�@u�����A(z�B���C�9�@u��e�A\)Bx�C�y�                                    Bx�{i  �          A7\)@p�׿Y��A,��B��3C���@p���[�A$(�B~�\C���                                    Bx�{w�  �          A7�@s33�L��A,��B�p�C�
@s33�XQ�A$z�B~��C�/\                                    Bx�{�j  �          A6=q@w
=�Q�A*�HB�z�C��)@w
=�W�A"�RB}
=C�h�                                    Bx�{�  �          A9G�@�녿�=qA,Q�B��qC��\@���xQ�A"{Bs��C�!H                                    Bx�{��  T          A5@k��fffA+33B�  C�(�@k��\��A"�RB~�C�s3                                    Bx�{�\  �          A8z�@`  �aG�A/\)B��=C��{@`  �_\)A&�HB�G�C��                                    Bx�{�  �          A8  @[���A/\)B���C��)@[��I��A((�B�33C��                                     Bx�{Ϩ  �          A5p�@W
=��\)A-G�B�p�C��{@W
=�7�A'33B�#�C���                                    Bx�{�N  T          A3
=@[��aG�A*ffB�G�C�+�@[��1G�A$��B���C��=                                    Bx�{��  "          A/
=@z=q<�A#\)B�
=>�@z=q�=qA�RB�B�C�.                                    Bx�{��  �          A1@��H>8Q�A$��B��R@!�@��H��\A ��B�C�]q                                    