CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230308000000_e20230308235959_p20230309021617_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-09T02:16:17.655Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-08T00:00:00.000Z   time_coverage_end         2023-03-08T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxk�H   "          @fff>�����@'
=B;  C���>���]p�?h��Al��C�<)                                    Bxk�V�  �          @g�=�G��{@(��B;(�C�8R=�G��_\)?n{Am�C��q                                    Bxk�eL  T          @j�H���,��@�RB*\)C��\���fff?.{A*{C��                                    Bxk�s�  �          @j�H��  �4z�@z�BG�C�|)��  �g�>��@�ffC�
=                                    Bxk肘  �          @hQ켣�
�4z�@G�BG�C�����
�e>�(�@��
C��R                                    Bxk�>  �          @i���8Q��1�@B   C�#׾8Q��fff?�\A��C���                                    Bxk��  "          @e����7�@Q�BG�C��
���c�
>�z�@���C��                                    Bxk變  �          @e��#�
�C33?�A���C�z�#�
�dz�L�ͿO\)C��3                                    Bxk�0  �          @_\)>Ǯ���@
=B#�HC���>Ǯ�I��?�A��C���                                    Bxk���  "          @\(�?�z��p�@=qB9G�C�9�?�z��<��?�  A�G�C�                                    Bxk��|  �          @^{?Tz��!G�@Q�BC�R?Tz��QG�>��@�\)C��                                    Bxk��"  T          @\��?�33��p�@7
=Bc=qC��?�33�0��?�33A�\C�AH                                    Bxk���  "          @_\)?�=q���
@8��Bd��C���?�=q�4z�?�33A�
=C�l�                                    Bxk�n  
�          @^�R?��׿�33@4z�B\G�C�S3?����+�?�z�A�(�C���                                    Bxk�  
�          @e?�p�����@8��BW�C���?�p��"�\?���A��HC�"�                                   Bxk�#�  �          @dz�?Ǯ��z�@   B5ffC���?Ǯ�<(�?�\)A�(�C��                                   Bxk�2`  
�          @g�?�Q��@ffB"  C�˅?�Q��N�R?B�\AB{C�f                                    Bxk�A  "          @dz�?�33�#33@��B��C�%?�33�Tz�?�A33C���                                    Bxk�O�  �          @dz�?�Q��+�@ ��B	��C���?�Q��U>�z�@�\)C�˅                                    Bxk�^R  T          @k�?@  ��H@*�HB:=qC���?@  �]p�?�  A{�
C�                                      Bxk�l�  �          @~{?�z��\)@5�B7�C�xR?�z��g
=?���A���C��                                    Bxk�{�  
�          @\)?����,��@-p�B*��C�aH?����n{?aG�AK
=C�xR                                    Bxk�D  �          @z�H?k��1�@'
=B'33C�+�?k��o\)?B�\A3\)C��=                                    Bxk��  �          @�  ?s33�>�R@!G�B�C���?s33�w
=?z�A  C��                                    Bxk駐  �          @tz�?��C33@{B�C��=?��p��>�z�@�G�C�                                      Bxk�6  �          @s33>���:�H@Q�BG�C���>���o\)>��H@��C��)                                    Bxk���  �          @s�
>��<��@
=B
=C��>��p  >�@��
C��=                                    Bxk�ӂ  �          @u�?&ff�B�\@�RB�C�{?&ff�p��>���@���C���                                    Bxk��(  �          @y��?����333@�B  C���?����i��?��AffC�1�                                    Bxk���  �          @w�?��\�2�\@\)B��C���?��\�k�?&ffA=qC��
                                    Bxk��t  r          @w
=?fff�8Q�@=qB�\C��\?fff�n{?
=qA�C��\                                    Bxk�  
�          @w�?���>�R@\)B\)C�� ?���mp�>�{@��\C���                                    Bxk��  �          @u?���?\)@	��B33C�
=?���j�H>��@|(�C�G�                                    Bxk�+f  	L          @vff?��
�:�H@�B�C��{?��
�k�>��@���C��                                    Bxk�:  
�          @tz�?O\)�@��@��B��C��f?O\)�n{>���@�ffC�#�                                    Bxk�H�  "          @u?L���G
=@�B�
C�0�?L���o\)>��@�C�                                    Bxk�WX  �          @vff?z�H�Dz�@B=qC��)?z�H�mp�>8Q�@,(�C�e                                    Bxk�e�  �          @u?��B�\@ ��A�{C�� ?��h��>�?�z�C��                                     Bxk�t�  
�          @�Q�?޸R�:=q@��B  C�l�?޸R�fff>���@��C���                                    Bxk�J  
�          @~�R?�  �=p�?��RA�z�C�P�?�  �dz�>��@
=qC�
                                    Bxk��  
�          @\)?��H�AG�@	��B��C���?��H�l��>�  @j�HC���                                    Bxkꠖ  h          @|��?�(��7
=@�\B�C���?�(��h��>�ff@ӅC��                                    Bxk�<  "          @r�\?n{�"�\@)��B1��C�
?n{�c33?p��AeG�C�\)                                    Bxk��            @xQ�?���,��@Q�B�
C��
?���c33?!G�A=qC�'�                                    Bxk�̈  
�          @\)?��>{@�A�p�C���?��g�>k�@P  C�Y�                                    Bxk��.  
�          @\)?�{�?\)?��A��C���?�{�a�=#�
?�RC��                                     Bxk���  
�          @|��?���7�?��HA�  C���?���^{>8Q�@%C�9�                                    Bxk��z  �          @r�\@   �/\)?�z�A�z�C��@   �Mp��#�
�8Q�C���                                    Bxk�   "          @fff@ �׿У׿�Q���  C���@ �׿E���\)��C�|)                                    Bxk��  T          @k�?�33�(�?�Q�Bz�C�f?�33�E>Ǯ@ə�C��                                    Bxk�$l  �          @y��@(��#33?�33A�C�XR@(��J�H>�z�@�(�C�Z�                                    Bxk�3  
�          @z�H@C�
�ff>�33@�(�C�5�@C�
�(��n{�]��C�1�                                    Bxk�A�  �          @xQ�@R�\��p�=u?s33C���@R�\��(��xQ��j�\C�1�                                    Bxk�P^  T          @tz�@N�R�   �W
=�N{C�(�@N�R���Ϳ�(���p�C��                                     Bxk�_  "          @s33@J=q�ff�L���E�C�+�@J=q��Q쿢�\���C���                                    Bxk�m�  "          @q�@@���{������z�C���@@�׿޸R��z�����C��R                                    Bxk�|P  T          @p  @A��z�
=�C��3@A녿�(���=q����C�R                                    Bxk��  
�          @n�R@?\)���þ�{����C�J=@?\)���׿��R����C���                                    Bxk뙜  
�          @n{@I�����ÿY���Y�C���@I���fff�����C�H                                    Bxk�B  
�          @l(�@B�\�\    �#�
C���@B�\��ff�J=q�U�C�q�                                    Bxk��  
�          @fff@5��G�?��A�ffC�%@5�33    ��\)C��                                    Bxk�Ŏ  
o          @e@<(���(�?8Q�A?\)C��{@<(����;�=q��\)C��                                    Bxk��4  �          @k�@C�
��
=>�(�@��HC��R@C�
��׿!G���\C�5�                                    Bxk���  �          @qG�@B�\�
=q?�\@�
=C�U�@B�\�
=�.{�&�HC��                                    Bxk��  "          @u@Dz��(�?\)A�C�H�@Dz��
=q�&ff��C�t{                                    Bxk� &  T          @s�
@8���
=>�G�@���C�\)@8���  �W
=�N=qC�                                    Bxk��  �          @u�@2�\�!G�?@  A4��C���@2�\�#33�(�����C�˅                                    Bxk�r  "          @s33@.�R�!�?#�
A33C���@.�R�   �B�\�9p�C�Ǯ                                    Bxk�,  �          @n�R@>�R��
=��ff��\)C���@>�R��{�����C��q                                    Bxk�:�  T          @p��@C�
��
�&ff�   C�H@C�
��
=��\)���HC�q�                                    Bxk�Id  "          @qG�@J=q�   ����C�Ф@J=q��녿�z����\C�J=                                    Bxk�X
  �          @q�@L(���p����
��Q�C��@L(���zΉ���z�C�@                                     Bxk�f�  
�          @k�@K����>�p�@�G�C�G�@K���p������C���                                    Bxk�uV  7          @k�@S33��=q>L��@L��C�*=@S33���H�#�
� z�C�f                                    Bxk��  �          @mp�@H�ÿ�G�?:�HA6=qC�Z�@H�ÿ�׾�\)��z�C��                                    Bxk쒢  �          @k�@L(��У�?�A
=C�z�@L(����\��  C�.                                    Bxk�H  "          @o\)@Dz��\?��A�(�C��@Dz���<��
>\C��
                                    Bxk��  
�          @u�@G
=��Q�?���A�z�C���@G
=�
=q>��
@��\C��)                                    Bxk쾔  �          @vff@:�H����?�A�ffC��@:�H��?Y��AL(�C���                                    Bxk��:  T          @r�\@*=q��@ffB��C��3@*=q� ��?s33Ah��C�Y�                                    Bxk���  "          @u@G����
?�(�A�\)C�ٚ@G����?�z�A��C�aH                                    Bxk��  "          @u�@S33��\)?�ffA�C��)@S33��G�?B�\A7�C��q                                    Bxk��,  �          @w�@a녿k�?��RA�=qC��)@a녿�?��A33C��                                    Bxk��  �          @w�@^�R�xQ�?���A�{C�9�@^�R�\?+�A�C�/\                                    Bxk�x  
E          @s�
@Z�H�+�?�(�A�G�C�w
@Z�H��=q?n{Ab�RC�ff                                    Bxk�%  �          @vff@�R?8Q�@A�BQ(�A��@�R�n{@>�RBKC��q                                    Bxk�3�  
�          @vff@?�@@��BM\)ALQ�@����@7
=B?�RC���                                    Bxk�Bj  
�          @w�@   >L��@:�HBEff@�=q@   ����@'�B+=qC�"�                                    Bxk�Q  T          @vff@9���#�
@ ��B#�C�n@9����z�@�Bp�C���                                    Bxk�_�  
�          @vff@<(�����@�Bp�C��{@<(��\?�
=A�C�^�                                    Bxk�n\  �          @|(�@9����@(Q�B'ffC�H�@9����  @G�A��HC�n                                    Bxk�}  
�          @|��@S�
���@z�A�Q�C���@S�
����?�ffA��C�'�                                    Bxk틨  �          @{�@`  ����?��A���C��@`  �\>�Q�@��RC�B�                                    Bxk�N  �          @z�H@e���p�>���@�(�C�� @e���p������Q�C���                                    Bxk���  T          @{�@l(���  >��H@�{C��H@l(���=q�W
=�Dz�C��                                    Bxk���  �          @~�R@@  ��G�@$z�B"Q�C��q@@  ���@
�HB��C��                                    Bxk��@  "          @\)@^{�Q�?�  A��C�U�@^{�˅?��A�
C���                                    Bxk���  �          @|��@U�
=@ ��A�
=C�H@U�\?�
=A���C��                                     Bxk��  �          @|��@G
=��  @�HB{C���@G
=��Q�?��HA�  C��3                                    Bxk��2  "          @~{@Fff�#�
@��B  C�.@Fff��G�?�  Aљ�C�+�                                    Bxk� �  	�          @{�@S�
�&ff@   A���C�s3@S�
����?�33A���C�O\                                    Bxk�~  7          @|(�@W
=�.{?�z�A�G�C�H�@W
=��ff?��A�  C��q                                    Bxk�$  ?          @{�@[���p�?�Aݙ�C��=@[���p�?���A�C��                                    Bxk�,�  T          @{�@i���W
=?���A���C���@i�����
?�@���C�Z�                                    Bxk�;p  "          @w�@g���{?@  A3�C�xR@g����=�Q�?�{C���                                    Bxk�J  
�          @xQ�@h�ÿ�\)?8Q�A+\)C�y�@h�ÿ�=q=u?k�C��)                                    Bxk�X�  T          @z=q@n{��=q>�(�@���C���@n{��33�.{�!�C�p�                                    Bxk�gb  �          @z=q@qG��L��?   @�C��@qG��p��<�?   C���                                    Bxk�v  T          @z�H@n{���?��A  C��@n{���H�#�
�W
=C��R                                    Bxk  T          @{�@l�Ϳ���?
=A	C�f@l�Ϳ�=q��Q쿧�C��                                    Bxk�T  �          @|��@s33�L��?#�
Az�C�3@s33��G�>��@(�C���                                    Bxk��  �          @}p�@p�׿�{>��@ۅC�Ǯ@p�׿��������C�,�                                    Bxk  �          @|��@r�\�fff?�\@�C�O\@r�\����#�
��Q�C�Y�                                    Bxk�F  �          @~�R@{���
=>���@��\C��R@{���=��
?�C�,�                                    Bxk���  �          @}p�@z=q>�p�?   @���@�\)@z=q=�\)?�RAz�?���                                    Bxk�ܒ  �          @\)@}p�>8Q�>u@aG�@*=q@}p�=#�
>���@��
?��                                    Bxk��8  "          @|(�@xQ�?�\=��
?�\)@�G�@xQ�>��>��
@�
=@�                                      Bxk���  �          @~�R@xQ�<�>W
=@I��>�ff@xQ콣�
>L��@=p�C�h�                                    Bxk��  T          @|��@X�ÿ��R�.{�%C�q@X�ÿu��=q��\)C��                                    Bxk�*  "          @|(�@U��p��333�#�
C��f@U��\)�˅��z�C��{                                    Bxk�%�  T          @�  @aG��޸R�.{���C�ٚ@aG���
=��������C�Ǯ                                    Bxk�4v  
�          @~�R@_\)����&ff�33C�q�@_\)��p�������p�C�N                                    Bxk�C  �          @|(�@Z�H��(��Tz��C\)C���@Z�H��=q�������C�9�                                    Bxk�Q�  �          @|��@hQ쿮{�333�"�HC��f@hQ�Tzΰ�
���C���                                    Bxk�`h  �          @|(�@c�
����5�%�C�L�@c�
�}p���\)��\)C�9�                                    Bxk�o  �          @~�R@g
=���
����
=qC�� @g
=������\��  C���                                    Bxk�}�  �          @}p�@u�J=q��p���ffC�(�@u��\�8Q��'�
C�>�                                    Bxk�Z  "          @z�H@xQ�>�����(�@׮@xQ�>�=�G�?���@ڏ\                                    Bxk�   �          @{�@vff�z��ff��=qC���@vff��z�.{�\)C�޸                                    Bxk辶  �          @z=q@l��?.{�k��\��A&�\@l��?�ff�����
=A|��                                    Bxk�L  �          @x��@q�?   �!G��33@�
=@q�?=p����R���\A2{                                    Bxk���  �          @{�@tz�=u�^�R�Mp�?fff@tz�>��:�H�+�@��                                    Bxk�՘  "          @{�@q녾��
��G��m��C���@q�>aG����
�s�
@S33                                    Bxk��>  T          @y��@l�;�  ��
=����C��@l��>�Q쿓33����@��                                    Bxk���  �          @{�@p�׾�\)�����~ffC��{@p��>�\)�����~�H@�\)                                    Bxk��  �          @~�R@p  �^�R���
�o�C�|)@p  ��  ������
=C��                                    Bxk�0  �          @�  @j=q��ff��p���C�f@j=q���R�˅���C���                                    Bxk��  �          @|(�@n{�@  ����yC�E@n{�\)�����\)C���                                    Bxk�-|  �          @z�H@n{�:�H�s33�a��C�l�@n{�8Q쿘Q�����C���                                    Bxk�<"  �          @z�H@p  ��R�n{�[�C�T{@p  ��Q쿎{���C�XR                                    Bxk�J�  �          @y��@l(��5����w\)C��
@l(���G���  ��C�&f                                    Bxk�Yn  �          @z=q@e�h�ÿ�p���Q�C��H@e�L�Ϳ�G����\C�g�                                    Bxk�h  T          @z=q@b�\�p�׿�\)���
C��@b�\�#�
��33��{C���                                    Bxk�v�  �          @z=q@Y�������{��z�C�N@Y����������
C���                                    Bxk��`  
�          @x��@U���
���H�У�C�c�@U��\)�   ��33C�\)                                    Bxk�  T          @{�@Q녿�z����ffC�G�@Q녾��	�����C��
                                    Bxk�  �          @y��@Mp��^�R�G���  C�g�@Mp�>aG��(���H@u�                                    Bxk�R  T          @u@Dzῌ���   ���C��@Dz�    ���p�<#�
                                    Bxk��  �          @w
=@8Q�޸R��\)���HC�t{@8Q��R��R� \)C��R                                    Bxk�Ξ  �          @xQ�@<�Ϳ�=q�Q��ffC��q@<�;\)� ���!Q�C���                                    Bxk��D  �          @s�
@8�ÿٙ�������C���@8�ÿ=p��
�H���C���                                    Bxk���  �          @p  @.{�B�\�Q��  C�=q@.{>�Q���R�Q�@�                                    Bxk���  "          @h��@1녿s33�
=q�33C���@1�>L���ff� ��@�                                    Bxk�	6  "          @u�@6ff�0����H��C�+�@6ff?\)�p�� �HA0��                                    Bxk��  �          @|(�@\)�Ǯ�AG��HffC�~�@\)?����5��8�\A�
=                                    Bxk�&�  T          @x��@{��G��I���Yp�C�^�@{?����>�R�H�\A�\)                                    Bxk�5(  �          @x��@z�#�
�Q��f�\C���@z�?�
=�=p��FQ�B
z�                                    Bxk�C�  T          @w�@����G��@���M��C��3@��?����,���1��A�=q                                    Bxk�Rt  7          @p  @z��(��8Q��J��C��R@z�?�  �/\)�=�A�(�                                    Bxk�a  
�          @j=q?��R�.{�@���a�RC��q?��R?���.{�C�RB
=                                    Bxk�o�  �          @g
=?�{>aG��S�
�3Az�?�{?��
�333�Mz�BQ��                                    Bxk�~f  T          @e?��׿u�E��u��C��H?���?�R�J�HA���                                    Bxk�  �          @i��@G�����.{�F
=C�P�@G�>�z��;��\z�AQ�                                    Bxk�  �          @k�@&ff�����#33�0��C��@&ff?\(����&�A�33                                    Bxk�X  �          @e�@���@  ��R�2��C�XR@��?��"�\�8  AEG�                                    Bxk��  �          @Z�H@
=�&ff�   �B�C�n@
=?(��   �C{A�                                    Bxk�Ǥ  �          @Tz�?�33�k��*�H�Y�\C���?�33?����p��A�A�Q�                                    Bxk��J  T          @\��?�\��z��33�-�HC�j=?�\��p��4z��e
=C��                                    Bxk���  �          @b�\@�\)�'��IC��\@?@  �$z��D��A��                                    Bxk��  T          @fff@  �(���#33�==qC��H@  ?!G��#33�>
=Ax                                      Bxk�<  �          @e�@�\��p����%�C�ٚ@�\�#�
�%�BG�C���                                    Bxk��  �          @e�?�
=�!������C��3?�
=��  �A��g�C�e                                    Bxk��  �          @h��?������H�-p��Ip�C�0�?����k��L(��C�z�                                    Bxk�..  �          @���?#�
�333�x��� C�B�?#�
?��H�p���Bx�                                    Bxk�<�  "          @���?
=q�u�~{£
=C�?
=q?�33�g���B���                                    Bxk�Kz  �          @|(�?=p��#�
�p  ��C��\?=p�?����g
=Q�Bi=q                                    Bxk�Z   �          @xQ�>�z�?
=q�Tz��By
=>�z�@�\�+��Q��B�\                                    Bxk�h�  �          @l��?aG��C�
������HC��?aG��z��'��E33C���                                    Bxk�wl  �          @g��#�
�Q��L����C��þ#�
?E��N{� B��                                    Bxk�  �          @i����\)<��A��i�\C3\��\)?�Q��*�H�A��C@                                     Bxk�  
�          @q녿�z�?E��!G��f33C#׿�z�?�{��\)���B�
=                                    Bxk�^  "          @s33?����0  ��ff��=qC��?��Ϳ˅�7��F��C���                                    Bxk�  �          @o\)?\(��\�P  �w{C��)?\(�>.{�e�Q�A5�                                    Bxk���  T          @u?�
=�Q��%��0��C�<)?�
=�fff�X���C�\)                                    Bxk��P  T          @xQ�?}p��7
=��H��RC��f?}p���z��^{�~�RC��=                                    Bxk���  "          @x��?�33�R�\��G��v=qC�S3?�33�(�����RC�f                                    Bxk��  T          @y��@z��AG��L���>ffC��@z��33�
=�ffC���                                    Bxk��B  T          @|(�@(��B�\�
=q���
C�aH@(��(������
=C�z�                                    Bxk�	�  S          @|��?�z��XQ쿔z����\C�
=?�z��p��%�$ffC��3                                    Bxk��  �          @~{?���]p���
=��33C�E?���7���(����HC���                                    Bxk�'4  �          @�Q�?��dz�����C�o\?��G
=��G���ffC�R                                    Bxk�5�  T          @}p�?��_\)�W
=�@��C���?��@  ������
C���                                    Bxk�D�  �          @z=q?Ǯ�aG��\���
C��?Ǯ�<�Ϳ��H��33C��f                                    Bxk�S&  �          @�Q�?�ff�XQ쿎{���C�
=?�ff�\)�"�\�\)C��{                                    Bxk�a�  �          @~{@Q��K����
�q��C��@Q��ff�Q��z�C�3                                    Bxk�pr  �          @{�@  �<�Ϳ�\)��Q�C���@  �
=�ff��C�p�                                    Bxk�  
�          @y��?��H�AG���\)����C�}q?��H��
�'
=�)�HC��\                                    Bxk�  �          @w
=@���/\)�����ŅC��@�ÿٙ��*�H�1G�C��                                    Bxk�d  T          @s�
@�
�p��G�� �C���@�
���\�:�H�Iz�C�33                                    Bxk�
  �          @w
=@L(���\�8Q��,z�C���@L(����R�������\C��                                    Bxk�  �          @xQ�@:=q����ff��33C��@:=q�J=q���33C�]q                                    Bxk��V  �          @vff@#33�G���(���(�C�33@#33��p��$z��(�\C��                                    Bxk���  �          @u�@���
=��ff��RC��
@������*�H�2  C��                                    Bxk��  �          @r�\@(��녿�Q���z�C�}q@(���G��"�\�*�HC�P�                                    Bxk��H  �          @q�?��AG���33���RC�/\?��
�H����z�C���                                    Bxk��  T          @qG�?��R�XQ�=p��4��C��H?��R�+��(��(�C��f                                    Bxk��  �          @p��?\�XQ�\)�
{C��?\�0�����=qC�aH                                    Bxk� :  "          @r�\?�
=�W
=�W
=�K�C�>�?�
=�:=q���H�י�C��)                                    Bxk�.�  �          @p��?��A녿�����Q�C��H?��
=q�(��"��C�4{                                    Bxk�=�  "          @n�R?ٙ��3�
��(���33C�� ?ٙ���p��333�DG�C�Ff                                    Bxk�L,  �          @n{@
�H�#�
�Ǯ�Ǚ�C�%@
�H��=q�#33�-�HC��                                    Bxk�Z�  �          @n{@�����޸R��{C���@�׿�=q�'��3��C��)                                    Bxk�ix  �          @mp�@p��z������C�z�@p��fff�4z��F�C��
                                    Bxk�x  �          @l��?����   �G���
C��?��ÿ����;��Q��C�                                    Bxk��  �          @mp�?�  �>�R��������C�@ ?�  ��� ���*�HC�)                                    Bxk��j  �          @mp�?�
=�9���������C��?�
=�G�����%(�C��R                                    Bxk��  �          @l��@�
�/\)��\)��=qC���@�
������&�C�(�                                    Bxk���  "          @n{?�{�.{��33����C�9�?�{�ٙ��,(��;z�C�ٚ                                    Bxk��\  �          @l(�?��#�
��z����HC���?���Q��6ff�K�RC��q                                    Bxk��  �          @mp�@�\�#�
�������C�G�@�\�Ǯ�&ff�5��C�E                                    Bxk�ި  T          @mp�@33��ÿ����(�C��@33���!��,�C�+�                                    Bxk��N  �          @k�@Q��	����p�����C��@Q쿔z��   �-��C��q                                    Bxk���  �          @j�H@.�R�zΰ�
��p�C�l�@.�R�����
�	=qC�Z�                                    Bxk�
�  �          @l(�@z��'
=��
=���
C��3@z���p���RC���                                    Bxk�@  �          @mp�@�R�!G������z�C�G�@�R���
��	\)C�*=                                    Bxk�'�  
�          @mp�@2�\�ff��{��33C���@2�\���H������C�~�                                    Bxk�6�  �          @n�R?�Q��>�R�������
C���?�Q��(��33���C��=                                    Bxk�E2  T          @p  @G��A녿c�
�\��C�ٚ@G����Q��
�HC�z�                                    Bxk�S�  T          @p  @p��>�R=u?Y��C�B�@p��-p���G�����C��                                     Bxk�b~  
�          @n{@���6ff�0���*�RC�7
@����׿�\)��(�C��H                                    Bxk�q$  �          @n�R@	���>�R�\)�	C��@	������ff��\C���                                    Bxk��  
�          @p��@2�\�{��33����C���@2�\���R� ����RC��                                    Bxk��p  >          @p��@:=q�s33�Q��\)C���@:=q=��
�����?��R                                    Bxk��  �          @w
=@:�H��=q��
=��C���@:�H�z������C�b�                                    Bxk���  
�          @z=q@7���  ��p���  C�L�@7��5�"�\�"�HC��                                    Bxk��b  >          @{�@O\)��33����ffC�B�@O\)�u�Q����C�ٚ                                    Bxk��  �          @x��@Tz�n{��p����HC�/\@TzὸQ���H��Q�C�AH                                    Bxk�׮  �          @�  @Q녿(���	���{C�K�@Q�>�{��R�33@�(�                                    Bxk��T  �          @�Q�@J�H��G�������C��@J�H?#�
�
=�ffA7\)                                    Bxk���  �          @���@P�׿Tz��\)���C��H@P��>aG��Q���@y��                                    Bxk��  �          @��@\�;�(�����p�C�o\@\��?�\��
���A�\                                    Bxk�F  �          @�33@g
=��33��33�ܣ�C�8R@g
=?   ��{��ff@�(�                                    Bxk� �  �          @�33@l�;����p��ȏ\C���@l��?(��У���A                                    Bxk�/�  T          @z=q@���{���R��ffC��@���ff������p�C�\)                                    Bxk�>8  �          @vff?�Q��\��=��
?�C�f?�Q��I����z���Q�C��                                    Bxk�L�  T          @u�?�Q��P  �W
=�K�C�j=?�Q��6ff��=q�ŅC�"�                                    Bxk�[�  
�          @qG�@���,(��}p��uC��f@��� ����\�  C���                                    Bxk�j*  �          @p  @   �0�׾Ǯ��Q�C��@   �����
��\)C���                                    Bxk�x�  T          @p  @�H�4zὸQ쿵C�T{@�H� �׿����  C��)                                    Bxk��v  �          @p  @ ���0��>�=q@��
C�#�@ ���'
=�n{�g33C��\                                    Bxk��  
�          @p  @�\�6ff�:�H�4  C�]q@�\�G���\)���C���                                    Bxk���  T          @p  @��B�\�
=q�C�|)@��!G����
��p�C�3                                    Bxk��h  
�          @n�R@ff�>�R�(���#�C���@ff��H��{��
=C�y�                                    Bxk��  �          @mp�?��H�N{��R�\)C��
?��H�*=q��
=���C�aH                                    Bxk�д  �          @j�H@	���;������{C�'�@	����Ϳ��أ�C��H                                    Bxk��Z  T          @j�H?�p��;��h���f=qC��?�p�����
�	ffC���                                    Bxk��   �          @j�H?�p��G
=�\(��Yp�C���?�p��p���ffC��=                                    Bxk���  
�          @k�?˅�J=q�p���n�HC�^�?˅�{�(����C�g�                                    Bxk�L  �          @k�@{�8Q��\��{C��R@{�������ffC�k�                                    Bxk��  �          @l(�@	���;���R�C�"�@	����������=qC��                                    Bxk�(�  T          @k�?�(��C33�
=��
C�s3?�(��!G�����Q�C�                                    Bxk�7>  �          @k�?��H�E��
=���HC�/\?��H�(Q���أ�C�XR                                    Bxk�E�  �          @k�?����HQ����C�H�?����(Q���
���
C���                                    Bxk�T�  T          @j=q?�p��I���333�1G�C�^�?�p��$z�����  C��{                                    Bxk�c0  T          @j�H?�\)�P  ����
=C�8R?�\)�1G���\��C�(�                                    Bxk�q�  �          @j�H@��@�׾.{�(Q�C�XR@��*�H��z���Q�C���                                    Bxk��|  �          @l��@�B�\�����\C�:�@�,�Ϳ�33���
C��3                                    Bxk��"  
Z          @j�H@Q��>�R���Ϳ�G�C���@Q��*�H������p�C�@                                     Bxk���  T          @l(�@p��.�R=���?�p�C�H@p��!G�������C�                                      Bxk��n  
�          @j=q@ff�0�׾��
��G�C�4{@ff�Q쿷
=��\)C�O\                                    Bxk��  "          @j=q@ff�2�\��\)���C��@ff����33����C��                                    Bxk�ɺ  �          @k�?��H��G��\)�[�HC�)?��H<#�
�p��Q�>�{                                    Bxk��`  �          @k�>L�;aG��`��ªW
C��=>L��?��R�R�\(�B��f                                    Bxk��  
�          @j=q?:�H�B�\�a�C��?:�H?��
�R�\�
Bpff                                    Bxk���  �          @n�R?�=q�k��c33aHC�?�=q?�  �U��~��BD{                                    Bxk�R  �          @s�
?�=q���aG��RC��?�=q?s33�[���B�R                                    Bxk��  �          @u�?���   �hQ���C��?��?��\�aG��B1��                                    Bxk�!�  �          @~�R?����\�q��C�1�?��?�=q�j=q�B6Q�                                    Bxk�0D  T          @�Q�?������b�\�x��C��?��>aG��q�p�A�                                    Bxk�>�  �          @�Q�?���{�O\)�X�\C���?��\�n{�)C�j=                                    Bxk�M�  �          @�G�?�G���  �S�
�\\)C�e?�G���  �n�Rp�C�8R                                    Bxk�\6  �          @�33?�G���p��QG��S��C���?�G�����l(�z�C���                                    Bxk�j�  �          @��@�ÿ�Q��N�R�M�C���@�ü#�
�b�\�kffC���                                    Bxk�y�  �          @�=q>���U������C�y�>���Q��[��f�C�g�                                    Bxk��(  T          @~�R?
=�Vff��
��p�C���?
=�\)�O\)�Y�
C�^�                                    Bxk���  T          @{�?5�B�\�
=��C��3?5���XQ��o=qC���                                    Bxk��t  �          @|��?}p��:=q�{�\)C�ff?}p����[��r(�C�XR                                    Bxk��  �          @}p�?�\)�K��33��z�C���?�\)��J=q�T\)C��                                    Bxk���  T          @|��?�  �U��{��  C�AH?�  ��H�3�
�7�RC���                                    Bxk��f  �          @z=q?�Q��c33��ff�z�RC�J=?�Q��5��Q���C�k�                                    Bxk��  T          @y��?}p��l(��=p��0Q�C�y�?}p��E�
=q��\C���                                    Bxk��  �          @\)?�
=�l(�>\@���C��3?�
=�b�\��\)��
=C���                                    Bxk��X  �          @xQ�?s33�n�R��G��θRC�%?s33�P  ������HC�&f                                    Bxk��  �          @w�?�  �k������C�� ?�  �I�����R��Q�C���                                    Bxk��  T          @tz�?�
=�P�׿�{���HC���?�
=�p��"�\�'(�C��                                    Bxk�)J  �          @l(�?���Mp���Q���  C�Y�?���333�������
C��{                                    Bxk�7�  �          @r�\>#�
�N{?��RA�{C�t{>#�
�o\)?\)A�
C�@                                     Bxk�F�  T          @s33>�
=�e�?���A��C�P�>�
=�p�׾��R��
=C�(�                                    Bxk�U<  T          @q�?8Q��e�?s33Aj=qC��{?8Q��j=q���{C��3                                    Bxk�c�  �          @qG�?Y���dz�?W
=AMC���?Y���g
=�!G��33C���                                    Bxk�r�  �          @o\)?(���hQ�>�@�(�C�,�?(���a녿}p��u�C�S3                                    Bxk��.  �          @n{?xQ��dz�>�{@�ffC���?xQ��Z=q�����z�C��
                                    Bxk���  �          @n�R?�G��e�>��@���C��?�G��Y����z���
=C�K�                                    Bxk��z  �          @n�R?����a�>�=q@�{C���?����W
=������  C�\                                    Bxk��   T          @n�R?�33�aG�>�33@��C�f?�33�W���ff����C�b�                                    Bxk���  �          @l��?���Z=q=�G�?�  C�?���L(���(���\)C��H                                    Bxk��l  �          @s�
?h���j�H>�33@���C���?h���`�׿�����\)C�<)                                    Bxk��  �          @q�>aG��n�R>���@�  C���>aG��c�
��33��Q�C��                                    Bxk��  �          @s33>\)�q논#�
�B�\C��>\)�^�R��p���C�'�                                    Bxk��^  �          @p��?�
=�aG��������C�J=?�
=�H�ÿ������C�N                                    Bxk�  T          @q�?��
�Q녾�{���C�:�?��
�8�ÿ�=q��C���                                    Bxk��  T          @l(�?�  �J�H��(����C�o\?�  �0�׿�\)��33C�.                                    Bxk�"P  �          @k�?0���fff�L�ͿG�C�k�?0���S33��
=���
C���                                    Bxk�0�  
�          @n{?\)�j=q>u@i��C�\)?\)�^{��Q���\)C��R                                    Bxk�?�  �          @p��?�  �Tz�333�3�C�XR?�  �2�\��� ffC�3                                    Bxk�NB  "          @s�
?����fff����\C�L�?����H�ÿ�=q��ffC�o\                                    Bxk�\�  T          @tz�?����dz�5�+�
C�Z�?����A��G�� z�C��q                                    Bxk�k�  "          @xQ�?�ff�a녿p���`��C��?�ff�9���{�p�C�                                    Bxk�z4  �          @z=q?�z��R�\��G���ffC�aH?�z��$z��=q�33C�s3                                    Bxk���  T          @{�?�z��K����
���\C�}q?�z��p�����ffC��                                    Bxk���  
�          @z�H?��5�������C��?���33�3�
�8�C���                                    Bxk��&  �          @w�?��.�R��  ��p�C���?���\)�)���3(�C�޸                                    Bxk���  �          @w
=@  �\)�
�H�	�C���@  ��G��6ff�?z�C�p�                                    Bxk��r  "          @qG�@��z��z���(�C�AH@����(���7�RC��                                    Bxk��  "          @p��?��R�E��p���z�C��H?��R��
�!��*��C�s3                                    Bxk��  T          @p  @��p���\)��33C�n@������)���4ffC��q                                    Bxk��d  �          @p��?�\�,(���z���p�C��=?�\���
�1��?�
C�p�                                    Bxk��
  
�          @q녾�z��i���
=q�ffC��H��z��K���\)���C�l�                                    Bxk��  "          @s�
��R�n�R�����C�H���R�S�
���
�ޏ\C���                                    Bxk�V  "          @u��@  �n{��G���=qC�LͿ@  �R�\�����{C���                                    Bxk�)�  �          @s�
���n{�����C��׿��S33��G���=qC�R                                    Bxk�8�  "          @r�\>����k��#�
�p�C��>����K���(���=qC���                                    Bxk�GH  "          @q�>���j=q�G��=�C�1�>���G
=���{C��H                                    Bxk�U�  "          @q�?��i���B�\�9��C�\?��G
=�33�p�C��H                                    Bxk�d�  �          @p��?�z��XQ�(���"�RC�N?�z��8�ÿ�{��(�C���                                    Bxk�s:  �          @o\)=����hQ����ffC��=����Tz´p���{C��
                                    Bxk���  
�          @o\)?Y���e����  C���?Y���H�ÿ����C��                                    Bxk���  �          @o\)?�\)�W��Q��K\)C��?�\)�5��   ���C��                                    Bxk��,  T          @l(�@?\)�xQ��{��
=C��@?\)���R�������C��                                    Bxk���  T          @u@QG�?c�
���
��AtQ�@QG�?��R��=q���\A�33                                    Bxk��x  "          @x��@XQ�?0�׿�ff���HA9��@XQ�?�ff����A�
=                                    Bxk��  �          @z�H@S�
?.{��p���z�A:�R@S�
?�{�˅���A�z�                                    Bxk���  �          @s33@J�H>�(���\�  @�ff@J�H?�녿޸R�ڏ\A��\                                    Bxk��j  �          @u�@I��?���
=���A�@I��?��
��G���33A�\)                                    Bxk��  �          @x��@P��?�������\AQ�@P��?�  ��Q���
=A�p�                                    Bxk��  �          @xQ�@W�>\�����@�ff@W�?��\��=q��
=A�
=                                    Bxk�\  "          @z=q@]p�>k���G��׮@n�R@]p�?Y���Ǯ��p�A\z�                                    Bxk�#  "          @{�@N�R?B�\������AT  @N�R?�������A��                                    Bxk�1�  
Z          @}p�@:�H?�����G�Aϙ�@:�H@�ÿ˅��{B��                                    Bxk�@N  �          @~�R@E?��
�
�H���A�(�@E?�p��Ǯ��=qBQ�                                    Bxk�N�  "          @\)@A�?��R��� G�A��@A�@	����Q���p�BQ�                                    Bxk�]�  "          @\)@J�H?����   ��\)A�G�@J�H?���������A���                                    Bxk�l@  "          @}p�@G�?���������RA�\)@G�@33�����{B\)                                    Bxk�z�  
�          @{�@C�
?�녿�G����Aᙚ@C�
@
=q����|Q�B��                                    Bxk���  �          @w
=@QG��L�Ϳ�Q����C�E@QG�?�\������A��                                    Bxk��2  �          @q�@G
=��Q����	
=C�4{@G
=?+�� ���(�AC�                                    Bxk���  
�          @u�@J�H>�����
=@z�@J�H?aG�������Axz�                                    Bxk��~  
�          @s�
@P��>L�Ϳ��H��{@\��@P��?aG���G�����Aq                                    Bxk��$  �          @u�@S33>u��z���(�@��@S33?fff�ٙ����
As�
                                    Bxk���  �          @s�
@J�H>�
=�z����@��H@J�H?��׿����A��\                                    Bxk��p  �          @w
=@G
=>�
=�{�
=@�\)@G
=?�
=��
=����A�                                      Bxk��  �          @x��@J=q>\�{�@��@J=q?�녿�������A�=q                                    Bxk���  "          @x��@hQ�^�R������(�C�8R@hQ��G�������
C��                                    Bxk�b  "          @y��@j=q�5��������C��f@j=q��\)���
��C��
                                    Bxk�  
�          @xQ�@`  ?��ÿ�  ��ffA�Q�@`  ?�Q�J=q�<��A�
=                                    Bxk�*�  "          @xQ�@\��>��˅�ĸR@�
=@\��?�G��������
A�z�                                    Bxk�9T  T          @xQ�@c�
����
=��C�0�@c�
=L�Ϳ�G����?Y��                                    Bxk�G�  T          @w
=@b�\��Ϳ�
=���
C���@b�\���
���
��33C���                                    Bxk�V�  �          @u@W
=�p�׿˅��(�C�7
@W
=���
������ffC�=q                                    Bxk�eF  o          @w
=@S�
�c�
��p���=qC��H@S�
�k���
=��p�C�H                                    Bxk�s�  T          @u@W
=��Q��ff���HC���@W
=>�z������ff@�ff                                    Bxk���  "          @tz�@P�׿\)�����  C�q@P��>���(�����@�R                                    Bxk��8  
�          @vff@XQ����\�ۙ�C���@XQ�>#�
��=q��G�@.�R                                    Bxk���  �          @vff@Tz����\)��\)C�R@Tz�>aG���z���33@n�R                                    Bxk���  
(          @vff@N{����33���C���@N{>��G����RAp�                                    Bxk��*  T          @x��@G
=�L���
�H�=qC���@G
=�#�
��
��\C��3                                    Bxk���  �          @x��@C33��=q����HC�AH@C33��z��
=�p�C�K�                                    Bxk��v  �          @z=q@N�R�O\)�   ���C��@N�R���
�
=q�  C�H�                                    Bxk��  
�          @xQ�@;���=q�
=q�z�C��@;���\��R�p�C��                                    Bxk���  �          @w�@.�R���
����
C�j=@.�R�����*�H�033C���                                    Bxk�h  
�          @w�?�\)��G��S�
�o��C�g�?�\)?5�P���j�A��                                    Bxk�  �          @qG�?�\)���`  {C�h�?�\)?����U��y�
B��                                    Bxk�#�  �          @qG�?��>��R�G
=�i�A{?��?���4z��J=qB�H                                    Bxk�2Z  T          @u�?�\)?W
=�U�k�A���?�\)?��6ff�I��BY��                                    Bxk�A   
�          @y��?�Q�>��Z=q�{(�A|��?�Q�?����C33�RB-�H                                    Bxk�O�  �          @z�H@Q�?Q��J�H�X��A�{@Q�?��-p��/��B#{                                    Bxk�^L  "          @|��@��?W
=�8Q��KQ�A�G�@��?�G��(��#�B��                                    Bxk�l�  T          @}p���G��*=q�.�R�.�Cy0���G������]p��uz�Cm�)                                    Bxk�{�  �          @~�R��\)�!��333�533Cv{��\)�����_\)�y  Ch:�                                    Bxk��>  �          @}p��=p��1G��,���,C~�ÿ=p����H�^{�v�RCv}q                                    Bxk���  "          @}p��h�ÿ�  �b�\�~z�Cn�3�h�þk��u�k�CBff                                    Bxk���  �          @z=q>�ff�*�H�*=q�1��C���>�ff�У��Y���~  C���                                    Bxk��0  �          @x��>L���A������C��>L���z��P���e�HC��\                                    Bxk���  
�          @u�>�Q��=p��
=�33C�z�>�Q�� ���L���e�
C�
                                    Bxk��|  �          @r�\���
�1��� p�C�,ͽ��
�����HQ��m�C�                                    Bxk��"  �          @s33?fff�<(��
=�	��C���?fff���=p��R�C���                                    Bxk���  �          @p��?��*�H�����ffC�O\?���
=�$z��/\)C���                                    Bxk��n  T          @qG�?�  �3�
��(��ۮC��R?�  ��#33�,\)C�                                      Bxk�  �          @s�
?�33�.{�������C�z�?�33���H�(���0�C�                                    Bxk��  "          @xQ�?�p��Dz��ff��(�C��R?�p�����{���C��
                                    Bxk�+`  �          @vff?��R�\����|  C�s3?��1G���
�  C���                                    Bxk�:  �          @l��?�ff�^�R>�z�@�G�C�g�?�ff�XQ�\(��[33C���                                    Bxk�H�  �          @c33?�  �L�ͽ�Q�˅C��{?�  �@  ��������C�B�                                    Bxk�WR  �          @j=q@'��(������C��@'���녿���� ��C��q                                    Bxk�e�  T          @k�@A녿У׿��
���HC���@A녿��׿޸R��{C��                                    Bxk�t�  	E          @h��@)����׿�����Q�C�]q@)�����\����C�=q                                    Bxk��D            @i��@333��녿�G���33C���@333�@  �����C���                                    Bxk���  T          @l(�@N{�.{��Q���p�C�q�@N{>\��z��أ�@׮                                    Bxk���  
�          @g
=@I����(���p����C�e@I���B�\�Ǯ���C�33                                    Bxk��6            @fff@:�H�У׿������C�l�@:�H��
=�������C�
=                                    Bxk���  "          @mp�@E��ٙ���\)��\)C��{@E���  ������Q�C���                                    Bxk�̂  
Z          @s�
@Z�H�\�����  C��@Z�H����aG��W
=C���                                    Bxk��(  �          @tz�@\�Ϳ��R�
=q�\)C�H�@\�Ϳ��R�}p��r=qC�R                                    Bxk���  �          @tz�@J�H��p��:�H�1��C��R@J�H��33��=q��G�C�E                                    Bxk��t  "          @xQ�@Z�H��ff���
����C�"�@Z�H��
=�#�
�C��                                    Bxl   T          @y��@g���Q콸Q쿱�C�%@g��������p�C�޸                                    Bxl �  "          @y��@p  ���
����qG�C�P�@p  �fff����\C�G�                                    Bxl $f  �          @|��@n�R��Q�\��=qC�.@n�R��  �=p��,��C�w
                                    Bxl 3  �          @\)@qG���(���ff��G�C�  @qG����\�Q��=p�C�q�                                    Bxl A�  �          @�  @mp�����\)�p�C�E@mp����ÿp���\  C���                                    Bxl PX  �          @|(�@xQ�\�\)���RC�8R@xQ쾣�
��  �h��C���                                    Bxl ^�  
�          @|(�@z=q���
<�>�ffC���@z=q���
��\)���C��\                                    Bxl m�  "          @\)@e���G��\(��O�C�
@e��0�׿�����  C���                                    Bxl |J  9          @w
=@>�R���˅�ÅC��@>�R��  ����C��{                                    Bxl ��  o          @u�@>�R���Ǯ���RC�Q�@>�R���R�33��RC���                                    Bxl ��  T          @u�@Dz�� �׿�����HC�k�@Dz��=q�У��ʏ\C�j=                                    Bxl �<  �          @u@I�������Tz��Ip�C�'�@I���˅�����(�C���                                    Bxl ��  T          @vff@J�H��{���
�yC��
@J�H������ff��(�C�                                    Bxl ň  �          @z=q@G���33�������C�U�@G���녿�{���C��q                                    Bxl �.  �          @z�H@]p����������=qC�y�@]p��8Q�����
=C��                                    Bxl ��  
�          @z=q@P�׿�z��=q��=qC�Q�@P�׿Y����Q��C���                                    Bxl �z  �          @vff@:�H��Q����C���@:�H��ff�\)��C�)                                    Bxl    "          @xQ�@=p����\��\��HC���@=p�����\)� {C�w
                                    Bxl�  T          @u�@I����G������陚C��@I�������z��z�C�k�                                    Bxll  T          @r�\@3�
��(�����ffC�E@3�
��=q�  ��C�|)                                    Bxl,  
�          @mp�@{�   ��33��\)C���@{������H�#�C���                                    Bxl:�  �          @qG�@7
=��녿�ff��RC��@7
=���\�{�p�C�=q                                    BxlI^  T          @p��@L�Ϳ�ff��p����C��@L�ͿJ=q����=qC�3                                    BxlX  "          @q�@R�\���Ϳ������
C���@R�\�����(���p�C��{                                    Bxlf�  �          @s33@Q녿�
=����G�C��@Q녿0�׿��H��Q�C��                                    BxluP  T          @qG�@Q녿�\)���H��z�C���@Q녿!G���p���33C���                                    Bxl��  
�          @r�\@K��xQ��������C��f@K���p�����C��                                    Bxl��  �          @p��@J�H��(���\)���
C�t{@J�H�.{��z���ffC���                                    Bxl�B  �          @i��@J�H�c�
��G����C�'�@J�H��׿�(����C�˅                                    Bxl��  T          @h��@,�Ϳ�  ��Q����C���@,�Ϳ�z�����G�C�c�                                    Bxl��  �          @p  @���(�������C��H@�ÿ�\�   �'�HC�33                                    Bxl�4  "          @r�\@p��.�R��33����C�}q@p����
=q�Q�C���                                    Bxl��  �          @e�@  �ff�޸R���C�~�@  ���R���!
=C�L�                                    Bxl�  !          @g
=@��Q��\���
C���@���  �z��C��
                                    Bxl�&  T          @g
=@
�H�&ff��(���=qC��@
�H�
=��
=�Q�C�ٚ                                    Bxl�  �          @s�
?�ff�G���  ���RC���?�ff�'
=���(�C�K�                                    Bxlr  
�          @j=q?�\)�5���\��  C��H?�\)���
�
=qC�L�                                    Bxl%  T          @n�R?����9����
=���RC�>�?������R��HC�)                                    Bxl3�  "          @w
=?�{�S�
���H��C�'�?�{�.�R���C�:�                                    BxlBd  T          @vff?����K���=q��Q�C�C�?����$z��(����C��R                                    BxlQ
  T          @u?�Q��Dz�����33C��
?�Q�����'��,Q�C�~�                                    Bxl_�  
�          @u?�(��;���G���
=C�4{?�(����"�\�&C��H                                    BxlnV  �          @y��?�\)�@  ��\)�ď\C��3?�\)�������\C��)                                    Bxl|�  �          @y��?�z��AG��\��z�C�"�?�z��(���G�C�                                      Bxl��  "          @|(�@�
�>{�Ǯ���C�ff@�
�Q��
=���C�u�                                    Bxl�H  �          @|(�@z��?\)��  ���C�` @z���H��
���C�K�                                    Bxl��  �          @|��?��H�E��p���C�7
?��H�!G���
���C��\                                    Bxl��  �          @w�@��7
=�������C���@��G��
=�(�C���                                    Bxl�:  "          @q�?�z��6ff��\)�ʸRC��?�z��  �Q����C��                                    Bxl��  �          @���?��P�׿�p���ffC��=?��'��'
=���C�@                                     Bxl�  �          @~�R?Ǯ�Tz῾�R��33C���?Ǯ�0  ������C��\                                    Bxl�,  �          @{�@  �(�ÿ����ݙ�C�9�@  �   � ����C�0�                                    Bxl �  �          @�G�@=q�\)��\��C��@=q����*�H�&p�C��                                    Bxlx  �          @�Q�@(���
�
�H�p�C�J=@(��˅�0  �-=qC�~�                                    Bxl  T          @�G�@ ���8�ÿ�����
=C�w
@ ������p���\C��)                                    Bxl,�  T          @�G�@7��,�ͿW
=�@��C�b�@7����=q���HC�c�                                    Bxl;j  "          @�G�@<���'
=�\(��D��C�J=@<���  ������G�C�XR                                    BxlJ  T          @��@<(��'
=�z�H�_�C�.@<(��{��Q��ĸRC�q�                                    BxlX�  �          @���?�  �g���ff�j{C��?�  �J�H��\��G�C�s3                                    Bxlg\  �          @�?�
=�b�\����r=qC�J=?�
=�E��
���
C��                                    Bxlv  �          @��?��H�j�H��p�����C�޸?��H�J=q��R�G�C�g�                                    Bxl��  �          @�z�?&ff�w
=��=q����C�?&ff�Tz��Q��Q�C���                                    Bxl�N  �          @��?���c33��=q���C��\?���A���\�	C�T{                                    Bxl��  T          @�z�@�\�I����Q��¸RC�xR@�\�"�\�!G���HC�c�                                    Bxl��  
�          @�z�@ ���K���
=��ffC�&f@ ���%�� ����C��)                                    Bxl�@  
�          @�@�R�?\)��=q��G�C�T{@�R�ff�&ff�{C��
                                    Bxl��  �          @�z�@�\�:=q����(�C�R@�\���#�
�=qC��=                                    Bxl܌  �          @���@
=�B�\����ׅC�XR@
=����*=q�p�C���                                    Bxl�2  T          @��@���0�׿��R��ffC�xR@����+�� �C�o\                                    Bxl��  T          @�?���N�R����z�C�*=?���%�(����C�
=                                    Bxl~  �          @��?�(��J�H��=q�Џ\C��3?�(��!��(���C��
                                    Bxl$  "          @�p�@��HQ��ff����C��f@��   �&ff��C���                                    Bxl%�  �          @��@Q��C�
��=q���C�k�@Q���H�'
=���C��                                     Bxl4p  T          @��
?�z��J=q���
���C���?�z��"�\�%�C�q�                                    BxlC  �          @��H?�(��Z�H�����HC�XR?�(��8����
�
33C�^�                                    BxlQ�  �          @��?�p��aG���33����C�j=?�p��?\)�z��33C�*=                                    Bxl`b  T          @��
?�Q��H�ÿ�����z�C��{?�Q�� ���'��\)C�Ф                                    Bxlo  �          @�z�@Q��Q��&ff�z�C�)@Q쿪=q�E�B�C�p�                                    Bxl}�  T          @�@.�R���H�4z��)z�C��@.�R�!G��G
=�@Q�C���                                    Bxl�T  T          @��@6ff��\)�.�R�&�
C�B�@6ff�����;��6�C��q                                    Bxl��  �          @���@:�H�h���$z�� {C�U�@:�H�.{�.{�+�C�W
                                    Bxl��  �          @�Q�@5��\�/\)�.{C��=@5>���1G��0�
@�G�                                    Bxl�F  �          @~�R@7
=<��
�0  �/\)>�Q�@7
=?E��(���&�AqG�                                    Bxl��  �          @~{@U���G����RC�u�@U=#�
�� {?.{                                    BxlՒ  �          @���@^�R���������C�XR@^�R�
=��33���C�7
                                    Bxl�8  "          @��\@:�H�   �333�-\)C�.@:�H>�z��5��/��@��H                                    Bxl��  �          @�33@E�^�R�!���C�
@E����*�H�#Q�C���                                    Bxl�  �          @���@N�R���\������C�B�@N�R�+��33���C�#�                                    Bxl*  �          @�  @X�ÿ�{�����  C��@X�ÿ\(�������p�C�޸                                    Bxl�  �          @�Q�@U���=q����Q�C��q@U��������C��                                    Bxl-v  �          @���@S33�У׿�\)��C��H@S33�����   ��ffC���                                    Bxl<  �          @���@\�Ϳ\���H���C�)@\�Ϳ��ÿ���\)C�c�                                    BxlJ�  �          @��@\�Ϳ��H��G�����C��@\�Ϳ����
=��G�C��q                                    BxlYh  �          @�(�@@  �z����C���@@  ��p������
C�ٚ                                    Bxlh  �          @�ff@]p��   ��ff��  C��)@]p���=q����ɅC���                                    Bxlv�  "          @��H@Z�H��
=�����G�C�H�@Z�H��ff��\)��{C��=                                    Bxl�Z  "          @\)@aG���ff������HC�w
@aG���{�p���ZffC��q                                    Bxl�   �          @��@b�\��(�=u?\(�C�u�@b�\��z�����z�C��\                                    Bxl��  �          @���@%�����
=����C���@%��Ǯ�'��$
=C�ff                                    Bxl�L  �          @���?���.{�33�G�C�\)?���   �<(��;�C���                                    Bxl��  T          @��?����<���33���
C��?����33�0���,z�C�g�                                    BxlΘ  T          @�=q?�{�7��
�H�G�C�q�?�{�(��6ff�3{C�"�                                    Bxl�>  �          @�=q?�(��-p����	{C�  ?�(�� ���:�H�8ffC�<)                                    Bxl��  T          @���@�����\)�p�C���@�����A��C33C��q                                    Bxl��  �          @�G�@�\�   �
=q�C�K�@�\���/\)�+��C���                                    Bxl	0  T          @��@=p��  ��
=���\C�]q@=p������p���  C�e                                    Bxl�  �          @�=q@Z�H���H��G��f�\C��@Z�H��\)���R��ffC�P�                                    Bxl&|  �          @�33@J=q��Q��
=�Ù�C�9�@J=q��
=�����G�C�ٚ                                    Bxl5"  �          @���@O\)��
�������C��=@O\)��녿����C���                                    BxlC�  �          @���@X����\�Tz��<��C�xR@X�ÿ�  �����=qC�T{                                    BxlRn  T          @�  @N{���fff�RffC�N@N{����
=����C�Q�                                    Bxla  �          @}p�@L(���R�5�%G�C���@L(����H���\���C�33                                    Bxlo�  �          @z=q@Fff�{�Y���H��C�0�@Fff����33��  C��                                    Bxl~`  �          @\)@AG��G���G����C���@AG���{����z�C�1�                                    Bxl�  �          @z=q@'
=�=q������(�C��q@'
=��z����C��                                    Bxl��  
�          @vff@{�Q��p����C���@{���
�!��&Q�C���                                    Bxl�R  
�          @x��@.{���
�H��RC�5�@.{��ff�!��#C�q�                                    Bxl��  "          @u�@ ���33� ������C�ff@ �׿��H��R�!�
C���                                    BxlǞ  
�          @u�@(Q��{�
=��C�Y�@(Q쿠  �!G��#\)C�G�                                    Bxl�D  
�          @w�@,(���\)�����HC�u�@,(��z�H�&ff�)
=C��)                                    Bxl��  "          @z=q@3�
������Q�C��q@3�
�=p��'
=�'�\C��                                    Bxl�  �          @y��@S�
�z�H��33��C�Ǯ@S�
������=qC���                                    Bxl6  �          @y��@QG��ٙ���33��{C�>�@QG����Ϳ�ff���HC��=                                    Bxl�  �          @xQ�@E����Ϳ�{���C��=@E���
=�����=qC���                                    Bxl�  �          @u@J=q��\��G����HC��\@J=q��{�s33�g\)C��                                     Bxl.(  �          @tz�@Mp����Ϳfff�Z{C�H@Mp��Ǯ�����33C��                                    Bxl<�  �          @r�\@^{��p�<��
>�p�C�o\@^{��Q쾸Q���C��q                                    BxlKt  �          @s33@O\)��Q�W
=�O\)C��f@O\)���ÿ333�+�
C�Q�                                    BxlZ  �          @u@W
=��=q�B�\�2�\C��
@W
=��(��&ff�  C�p�                                    Bxlh�  �          @vff@QG���zᾳ33���\C��
@QG���G��Tz��I�C�޸                                    Bxlwf  �          @w
=@5���Ϳ�\)����C�{@5����
�����
=C��\                                    Bxl�  �          @w�@1G���\�����
=C�33@1G���׿����Q�C��{                                    Bxl��  �          @w
=@G
=��
�xQ��i�C�AH@G
=�޸R���H��Q�C�^�                                    Bxl�X  T          @tz�@@�׿�=q��
=��(�C�` @@�׿�z����{C�~�                                    Bxl��  �          @vff@E�� �׿����ffC�n@E�����������C���                                    Bxl��  T          @r�\@-p��
=q�������C��q@-p���G���ff���C���                                    Bxl�J  �          @q�?�G��8�ÿ����G�C��\?�G���
�#�
�+p�C��{                                    Bxl��  T          @s33?����Dz��33���
C���?����\)�'
=�/�C���                                    Bxl�  �          @xQ�?5�=p�����p�C�?5����C�
�P��C��q                                    Bxl�<  �          @y��?@  �8���\)�(�C�<)?@  �
�H�H���WG�C�}q                                    Bxl	�  �          @w
=?����,(��"�\�%{C��\?��ÿ��H�HQ��Z(�C�AH                                    Bxl�  �          @u?h���$z��*=q�1Q�C��?h�ÿ��N{�g�C�aH                                    Bxl'.  T          @w�?�\)�ff�5�>33C��{?�\)�Ǯ�U�p�RC��                                    Bxl5�  
�          @u�?��\����9���E��C�&f?��\���H�XQ��x�HC�~�                                    BxlDz  �          @r�\?z��Q�(���r{C���?z���ÿ��R��C��3                                    BxlS   �          @s33>#�
�5�@�RB$��C���>#�
�W�?�Q�A�33C�]q                                    Bxla�  T          @r�\>Ǯ�E@Q�B	ffC���>Ǯ�a�?��A��C�*=                                    Bxlpl  T          @qG�>�\)�G
=@B�C���>�\)�a�?��RA�Q�C�B�                                    Bxl  
�          @qG�>����E�@B
=C���>����`  ?�G�A��C�C�                                    Bxl��  
�          @qG�?�R�<(�@�RB�C��R?�R�Z=q?�A��RC�,�                                    Bxl�^  T          @w�>�p��S�
?�(�A�p�C�5�>�p��l(�?��A��\C��                                     Bxl�  
�          @x��?���U?�\)A�
=C��?���mp�?z�HAj�\C��q                                    Bxl��  �          @u=��
=q@E�B[�
C��=��7
=@��B"G�C�>�                                    Bxl�P  �          @xQ�>����@>{BI�C�z�>��G
=@G�Bp�C�*=                                    Bxl��  T          @vff>�  �?\)@=qB�C�j=>�  �_\)?˅AÙ�C��                                    Bxl�  �          @vff?#�
�5@ ��B#33C�^�?#�
�XQ�?�p�A��
C�b�                                    Bxl�B  n          @w�>�Q��J�H@
�HB�C�C�>�Q��g
=?���A��C�޸                                    Bxl	�  �          @z=q?��G�@�
B{C�,�?��fff?��HA��C��H                                    Bxl	�  �          @w
=?8Q��H��@
=BC�y�?8Q��c�
?�G�A��C���                                    Bxl	 4  �          @n�R>�
=�:=q@  B�
C�  >�
=�XQ�?�(�A�Q�C��\                                    Bxl	.�  �          @s33>#�
�P��?��
A�
=C�n>#�
�fff?k�AeC�K�                                    Bxl	=�  �          @tz�>��H�J�H@�
Bz�C�e>��H�e�?��HA�  C��f                                    Bxl	L&  �          @tz�>�\)�N{@   A�p�C���>�\)�g�?��A�C�=q                                    Bxl	Z�  �          @p��>��A�@��B33C���>��^{?���A�C��R                                    Bxl	ir  T          @vff>��C�
@�\B33C�G�>��aG�?�(�A�C��R                                    Bxl	x  �          @qG�?\)�E@G�B  C�q?\)�_\)?���A��C���                                    Bxl	��  �          @u�>��R�7�@   B#�C��>��R�Y��?�(�A�  C��H                                    Bxl	�d  �          @u�<#�
�I��@��BC��<#�
�dz�?�ffA�{C�                                    Bxl	�
  �          @w���Q��QG�@z�B=qC�8R��Q��j�H?���A��C�N                                    Bxl	��  T          @k�������@:�HBS�HC�þ���5@�\BQ�C��                                    Bxl	�V  T          @l�;#�
�AG�@
=B�\C�xR�#�
�\(�?��A�(�C���                                    Bxl	��  
�          @i��?(��p��@�RB�� C��H?(���G�@�BT�C��                                    Bxl	ޢ  �          @o\)?�G�@>�R?���A���Bm��?�G�@#�
@�\B33B^
=                                    Bxl	�H  �          @mp�?�\)@2�\?��A��B`?�\)@�@�B�BN�H                                    Bxl	��  T          @o\)?У�@�@�Bz�B`�?У�?�ff@333BD{B?z�                                    Bxl

�  �          @q�?˅@�@+�B5(�BT�?˅?�z�@G
=B^�HB&
=                                    Bxl
:  �          @q�?���@@4z�BA\)Ba�R?���?��@O\)Bl��B0=q                                    Bxl
'�  �          @n�R@A�?�=q?�33A��A���@A�?�  ?ǮA�{A��H                                    Bxl
6�  
�          @y��@R�\?��?k�AY�A���@R�\?�{?���A��AиR                                    Bxl
E,  �          @~{@N{?�33?��
A���A�@N{?��
?ٙ�A�p�A��                                    Bxl
S�  �          @}p�@G
=?�
=?��RA�G�A�Q�@G
=?�G�?�z�A癚A�
=                                    Bxl
bx  �          @vff@%�?�=q@	��B�Bp�@%�?�G�@!�B%Q�A��
                                    Bxl
q  �          @s33@,��?���?�A��
B�@,��?��@33B
=A�ff                                    Bxl
�  �          @l(�@;�?��?�  A���B��@;�?�=q?�
=A��A��                                    Bxl
�j  �          @g
=@/\)?޸R?�p�A�B��@/\)?��?�{A���A��                                    Bxl
�  T          @\(�@E?�p�������\Ạ�@E?�  =�G�?�G�A���                                    Bxl
��  T          @l��@Y��?���?0��A/�A��@Y��?^�R?n{Am�Ae��                                    Bxl
�\  �          @k�@S�
?���?.{A.{A��@S�
?�{?z�HA{33A���                                    Bxl
�  �          @l(�@W
=?�=q?#�
A Q�A�
=@W
=?��?p��Am��A��
                                    Bxl
ר  �          @s33@I��?�?\(�ARffA��\@I��?�z�?�ffA�=qA݅                                    Bxl
�N  �          @r�\@[�?��
>�\)@���A�ff@[�?�?#�
A��A��                                    Bxl
��  T          @hQ�@E�?���?�ffA���A�@E�?��?���A�=qA�(�                                    Bxl�  
�          @Y��?�?�=q@�RB'�\B�H?�?�  @"�\BFz�A��                                    Bxl@  �          @b�\@B�\�#�
��(���z�C��@B�\>L�Ϳ�(��ͮ@q�                                    Bxl �  �          @��
@B�\�.{�+��"�C���@B�\���
�0���(C�ٚ                                    Bxl/�  �          @�33@Dz�@  �&ff�  C�)@Dz��G��-p��%�C��                                    Bxl>2  "          @�=q@A녿E��'
=�p�C�Ф@A녾��.{�'��C��=                                    BxlL�  �          @�=q@>{�G��*=q�#Q�C�� @>{���1G��+�C��                                    Bxl[~  �          @�=q@C33�!G��'�� (�C�4{@C33<��
�,(��%��>�33                                    Bxlj$  �          @�=q@@�׿5�*=q�"C�^�@@�׽L���0  �)�RC�~�                                    Bxlx�  
�          @��\@J�H�L�������C��@J�H�L���#33�=qC�9�                                    Bxl�p  T          @�=q@O\)�J=q����
C�)@O\)�W
=����\)C��                                    Bxl�  
�          @~�R@N{������
=C��=@N{=��
��
�z�?�{                                    Bxl��  �          @�Q�@H�ý�G�����{C���@H��>��=q��HA
{                                    Bxl�b  �          @���@P  �k��Q����C��{@P  >�{�����@�Q�                                    Bxl�  "          @�=q@HQ�u�%���C�o\@HQ�?���!G����A ��                                    BxlЮ  �          @\)@Mp����R�z����C�H�@Mp�>�����=q@���                                    Bxl�T  �          @n�R@S33>8Q�ٙ���\)@Dz�@S33?z��{����Aff                                    Bxl��  T          @k�@U�?�\���H��A  @U�?Y����ff����Ad                                      Bxl��  �          @qG�@c33?L�ͿG��@z�AI@c33?u�\)��Aq�                                    BxlF  �          @r�\@b�\?���>�@�p�A�@b�\?}p�?=p�A6{Axz�                                    Bxl�  �          @q�@j�H?Y����\)���AP  @j�H?W
=>�@�AN=q                                    Bxl(�  �          @q�@`��?��=#�
?��A���@`��?��>�p�@��A��                                    Bxl78  �          @r�\@R�\?�=q>���@��\A�ff@R�\?ٙ�?B�\A9p�A�(�                                    BxlE�  
�          @p  @C33?�=q�h���|��A�  @C33?�G��!G��-A�ff                                    BxlT�  T          @��@S�
=u�ff�Q�?���@S�
?�R�G���RA(��                                    Bxlc*  �          @���@XQ�>Ǯ����z�@љ�@XQ�?fff��\��ffAn=q                                    Bxlq�  �          @�G�@QG�?!G����	��A/33@QG�?�33��
��Q�A��                                    Bxl�v  T          @���@N�R?��z��{A ��@N�R?�{����z�A�G�                                    Bxl�  �          @~{@Fff?\(���
�Aw
=@Fff?�����\��=qA��
                                    Bxl��  T          @\)@1G�?�=q�$z��#33A���@1G�?�z��\)�
�HA���                                    Bxl�h  
�          @��\@P  ?(���
=��\A7
=@P  ?����Q���
=A�                                    Bxl�  �          @��@J�H?B�\����33AW�
@J�H?���	��� 33A��                                    Bxlɴ  �          @�=q@S33?=p��p���HAI@S33?��R��p���=qA��H                                    Bxl�Z  
�          @��\@g
=>��H�����@��R@g
=?fff��33���A`��                                    Bxl�   �          @��H@u>��
�����@�\)@u?!G���p���Q�A(�                                    Bxl��  �          @�=q@~�R>u�=p��(��@Z�H@~�R>�녿(����H@��\                                    BxlL  �          @��@r�\>��
��������@��@r�\?녿}p��g�A��                                    Bxl�  �          @��@!G�?��5��1��A���@!G�@33��H���B\)                                    Bxl!�  �          @�G�@]p�?Tz�������AXz�@]p�?�  ������ffA�z�                                    Bxl0>  �          @��
@\��?8Q��33��33A;�@\��?�
=������(�A�
=                                    Bxl>�  �          @�(�@vff<#�
���H��  =��
@vff>�33������@�{                                    BxlM�  T          @�p�@y���u��ff�j{C�\@y���0�׿�  ��C��q                                    Bxl\0  �          @��@z=q��ff��z����HC��3@z=q�.{��p����C�Ǯ                                    Bxlj�  �          @�(�@vff�
=������z�C���@vff���������z�C�R                                    Bxly|  �          @�z�@���>aG��B�\�*�\@E�@���>Ǯ�0���{@���                                    Bxl�"  T          @�z�@~�R>�������n�R@��@~�R?
=q�xQ��X  @�{                                    Bxl��  �          @�33@~{=�G��u�YG�?���@~{>��ÿh���L��@��                                    Bxl�n  �          @�=q@~�R<#�
�L���6�\=�@~�R>L�ͿG��0��@5                                    Bxl�  T          @��
@~{���
����j�HC�j=@~{>.{���
�hz�@��                                    Bxlº  �          @��@��þ�G��
=��C��@��þ�z�+��33C��
                                    Bxl�`  �          @���@w
=�G��\(��Ep�C�K�@w
=��Ϳ��\�j�RC��                                    Bxl�  �          @�=q@{��Y���
=�p�C�޸@{��0�׿E��/�C��                                    Bxl�  
�          @��@�G��5����Q�C�
=@�G��(�þ�\)�~{C�`                                     Bxl�R  �          @��@��H�5���Ϳ���C�3@��H�+�����eC�]q                                    Bxl�  �          @���@�녿Y���L���/\)C��@�녿G��Ǯ��C���                                    Bxl�  �          @�z�@�녿B�\�����C��3@�녿333���
��Q�C�
                                    Bxl)D  T          @�(�@�=q�+��W
=�=p�C�S3@�=q�����p����\C���                                    Bxl7�  "          @�z�@�33�   �#�
�	��C�~�@�33��ff��=q�o\)C��R                                    BxlF�  T          @�@����33=�G�?���C���@����p�<��
>�=qC�|)                                    BxlU6  T          @�@�p��u>#�
@�RC���@�p���Q�>\)?�p�C�Z�                                    Bxlc�  T          @�p�@�z�>�=q>��
@��R@n�R@�z�>B�\>\@��R@$z�                                    Bxlr�  T          @�ff@�(�?5>L��@/\)A�
@�(�?#�
>�Q�@�ffAG�                                    Bxl�(  �          @�{@��\?@  ?�\@�G�A'�@��\?(�?.{A��Az�                                    Bxl��  �          @�@��>��H?   @��@�33@��>�Q�?��AG�@���                                    Bxl�t  T          @��
@���>�{?��Ap�@�=q@���>B�\?(��A�
@-p�                                    Bxl�  �          @�p�@���>Ǯ?8Q�A�
@���@���>W
=?J=qA0  @>�R                                    Bxl��  T          @�z�@���>�z�?Tz�A9��@��
@���=�Q�?^�RAC�
?�(�                                    Bxl�f  �          @��@���>�z�?5A
=@��H@���=�G�?B�\A)�?���                                    Bxl�  T          @��@�=q?0�׽��Ϳ�
=A=q@�=q?333=u?aG�A33                                    Bxl�  T          @��@�z�>��R�=p��!�@��H@�z�>��&ff���@��
                                    Bxl�X  �          @�G�@�  >�  ��
=��p�@XQ�@�  >�{��33���R@��
                                    Bxl�  �          @��R@�>��
������Q�@��
@�>Ǯ��  �W
=@��                                    Bxl�  �          @�z�@��>��R���
��(�@�  @��>�p��k��P  @��                                    Bxl"J  "          @�p�@��\?333���H��G�A�
@��\?L�;��R���A0��                                    Bxl0�  �          @��\@���?�R=��
?���A��@���?
=>k�@A�@��                                    Bxl?�  �          @�z�@���?��>���@���@�G�@���>�G�?�@��H@��H                                    BxlN<  "          @��H@���=#�
?�@�  ?�@������
?�@�C�q�                                    Bxl\�  �          @�=q@�  ��33?�\@ٙ�C��@�  ��>��@��C��                                    Bxlk�  
�          @��@�Q쾊=q?��@��
C�*=@�Q����>��@ə�C�W
                                    Bxlz.  �          @�G�@�  ����?�\@�33C�T{@�  �aG�>��@��HC���                                    Bxl��  "          @�=q@�Q��(�>��@�=qC�%@�Q��>�Q�@�G�C�z�                                    Bxl�z  �          @��@��s33>Ǯ@���C��3@����\>��?��RC�(�                                    Bxl�   T          @��@z=q���?aG�AD��C�p�@z=q���R?(�A�HC�:�                                    Bxl��  �          @�  @B�\�˅@   B��C�/\@B�\���@�
A�C�n                                    Bxl�l  �          @�Q�@S�
���H@\)A�(�C�{@S�
����?�=qA�C��)                                    Bxl�  
�          @�\)@]p�����?���A��
C���@]p����R?��A�33C�\                                    Bxl�  T          @��R@c�
��=q?�{A�33C��@c�
��?�
=A���C��
                                    Bxl�^  �          @�@Tz�ٙ�?��A֣�C�y�@Tz��?�
=A�  C��H                                    Bxl�  T          @�ff@g���G�?ٙ�A�\)C�e@g��У�?���A�z�C��                                    Bxl�  �          @�@e����?��
A�\)C��@e�˅?�Q�A�p�C�{                                    BxlP  �          @�p�@c33����?���A�33C�*=@c33���?��
A�Q�C�H�                                    Bxl)�  	          @�@aG���Q�?�z�A��
C��\@aG���{?�=qA��C���                                    Bxl8�  �          @�ff@c�
��  ?�\A�p�C�XR@c�
�У�?�
=A�{C���                                    BxlGB  T          @�  @c33��ff?�Q�A���C�0�@c33��z�?��
A��\C���                                    BxlU�  "          @���@a����?��Ax(�C�\)@a��ff?��AG�C�#�                                    Bxld�  �          @�=q@c33��?���A�C��R@c33�?(��AC�J=                                    Bxls4  �          @��\@l(�� ��?c�
A?
=C��q@l(��
=q>\@�z�C���                                    Bxl��  �          @��H@r�\��?�G�AW�
C�@ @r�\�   ?��@���C�q                                    Bxl��  �          @�z�@vff����?333Az�C���@vff�z�>aG�@7�C��H                                    Bxl�&  
�          @�(�@{���{>�(�@�ffC�Z�@{���zἣ�
��z�C�\                                    Bxl��  T          @�33@y����?Y��A4z�C�o\@y������>�
=@��
C�}q                                    Bxl�r  �          @��\@��׿��R>�@�C��
@��׿Ǯ=���?��\C�k�                                    Bxl�  S          @�  @~�R����?Q�A3�C��3@~�R��\)?�\@�ffC���                                    Bxlپ  
�          @���@x�ÿ�p�?��RA��
C�4{@x�ÿ��R?h��AF{C�|)                                    Bxl�d  T          @��\@^�R�{>�z�@vffC�W
@^�R�p����R��C�\)                                    Bxl�
  T          @�=q@aG��\)��\)�h��C�c�@aG�����(���G�C��                                     Bxl�  T          @��@l���p����
���C���@l����ÿ�����C���                                    BxlV  T          @���@n�R��p�?   @׮C��@n�R��\<#�
>�C��f                                    Bxl"�  "          @�@e���=q?xQ�AW�
C�w
@e�� ��?   @��C�\)                                    Bxl1�  �          @�{@[����?��HA�
=C��{@[��޸R?˅A��
C��\                                    Bxl@H  
�          @�\)@dz��\@��A�\)C��@dzῃ�
@ ��A�C���                                    BxlN�  
�          @�\)@\(�=�@��B�\@�\@\(���@
=B�
C�4{                                    Bxl]�  �          @�(�@e�B�\@�RB
ffC�}q@e�G�@
=B�C��q                                    Bxll:  �          @���@z�H�B�\?�
=Aљ�C��H@z�H�&ff?�=qA�C�P�                                    Bxlz�  �          @�z�@x�ÿ�{?��A�\)C�Y�@x�ÿ��?s33AJ{C��{                                    Bxl��  �          @�p�@q녿�{?�ffA��C��@q���?Q�A*�RC�Z�                                    Bxl�,  "          @�{@��Ϳ�Q�=�?ǮC�j=@��Ϳ�
=�k��AG�C��                                     Bxl��  �          @��@�G���Q쿞�R���C�Ф@�G��Y�����R���C��                                    Bxl�x  :          @��
@�녿�33�\)��z�C�~�@�녿�(��aG��;33C���                                    Bxl�  �          @�z�@l�ͿB�\������C�(�@l�;W
=�z�� G�C�aH                                    Bxl��  
�          @�(�@j�H��  ��{����C���@j�H��G��
=q��33C�O\                                    Bxl�j  
�          @�p�@s�
��\)��ff��=qC�s3@s�
������33��\)C�@                                     Bxl�  "          @��@aG����R��p����HC�G�@aG���G��
=q����C�e                                    Bxl��  �          @�
=@h�ÿ����=q���C���@h�ÿ���p����
C�8R                                    Bxl\  "          @�ff@r�\��Q�E��$  C�q�@r�\�ٙ���(�����C��3                                    Bxl  
�          @�z�@�  ���
>�{@�(�C�  @�  ����G���33C��{                                    Bxl*�  "          @��
@p�����>��@���C�+�@p���(���\)�fffC���                                    Bxl9N  �          @��@hQ��\)?��@��
C�.@hQ��33�#�
��C���                                    BxlG�  "          @�(�@h�ÿ�p�?�=qA��RC���@h���  ?O\)A,  C�.                                    BxlV�  T          @�
=@xQ���\?E�A�
C��@xQ��
�H>�  @N�RC�ff                                    Bxle@  "          @��R@h��� ��>��
@�{C���@h��� �׾����w
=C���                                    Bxls�  
�          @��@a��'
==#�
>�C���@a��"�\�������C�"�                                    Bxl��  �          @��@hQ��
=�k��@��C���@hQ���R�L���)p�C�<)                                    Bxl�2  �          @��
@\(��z�.{�Q�C��@\(�����(���G�C�q�                                    Bxl��  �          @�(�@n{�	���O\)�*=qC��)@n{��녿����\)C���                                    Bxl�~  T          @��\@g
=�  >��@b�\C��@g
=�  ������  C��                                    Bxl�$  �          @�z�@j�H��R?.{A
=C�U�@j�H��=�G�?�z�C���                                    Bxl��  �          @�(�@g
=�?�G�A��HC���@g
=�?5AffC��\                                    Bxl�p  T          @��@c�
��?��HA�Q�C�'�@c�
��?h��AAp�C�l�                                   Bxl�  �          @��@e��33?��HA���C�q@e���?+�A33C��)                                   Bxl��  �          @�{@j�H�{?��A^�\C�e@j�H��H>�@�Q�C�P�                                    Bxlb  �          @�ff@tz��G�>#�
@z�C���@tz��\)�������RC��                                    Bxl  �          @��R@`���&ff?B�\Ap�C��
@`���-p�=�Q�?�
=C�*=                                    Bxl#�  <          @�z�@C33�C33?0��A{C���@C33�HQ콣�
����C�'�                                    Bxl2T  :          @�ff@2�\�O\)?�G�AU��C�XR@2�\�X��>8Q�@
=C��{                                    Bxl@�  "          @�ff@L(��6ff?��A^=qC��@L(��AG�>��
@���C�B�                                    BxlO�  �          @���@N{�1녽L�Ϳ�RC��
@N{�+��:�H��HC�R                                    Bxl^F  �          @�(�@P���7
==��
?�{C�\)@P���333�!G��p�C��{                                    Bxll�  "          @��
@S33�.�R?��@��RC�1�@S33�2�\���Ϳ�ffC��                                     Bxl{�  4          @�z�@s33��
=?�\A�\)C�j=@s33��=q?�A�{C��q                                    Bxl�8  �          @�{@~{��ff?���A{\)C�U�@~{��ff?E�A!G�C��                                    Bxl��  �          @�ff@�����\?^�RA4Q�C�~�@������?�@�Q�C�ff                                    Bxl��  �          @�ff@�G��z�H?+�A\)C��3@�G���{>��@�Q�C��
                                    Bxl�*  �          @�\)@��ͿO\)>L��@!�C�Ф@��ͿTz�#�
��C��=                                    Bxl��  �          @��R@��H��G�>W
=@(��C�k�@��H���
�L�Ϳ333C�N                                    Bxl�v  �          @�@��\�k�=��
?��C��@��\�h�þ����(�C��                                    Bxl�  �          @���@����c�
=�?�  C�.@����c�
��G���Q�C�,�                                    Bxl��  �          @��@��
��ff����p�C��\@��
��(�����  C�1�                                    Bxl�h  �          @��\@y����ff��\)�l��C��@y�����8Q���C�l�                                    Bxl  �          @��@B�\��  ?ǮA˅C�Ǯ@B�\��{?���A��C�\                                    Bxl�  �          @�
=@=p�=���@Tz�BA  ?��H@=p��=p�@N�RB:��C�H                                    Bxl+Z  �          @�{@3�
��@VffBFp�C�` @3�
���@G�B533C��{                                    Bxl:   �          @�\)@N{�0��@B�\B*��C���@N{��
=@1G�B�\C��                                    BxlH�  �          @�
=@H�ÿG�@C�
B-��C�\@H�ÿ\@0��B�C�                                    BxlWL  T          @��R@<(��u@J�HB6�C���@<(���p�@4z�BG�C��H                                    Bxle�  �          @��H@j�H?=p�?\A�(�A7\)@j�H>�{?�33A�\)@�\)                                    Bxlt�  �          @���@`�׿���?�{A�
=C�R@`�����?W
=A9��C�XR                                    Bxl�>  �          @��@
=�i��?�A��C�  @
=�x��>�ff@�G�C�8R                                    Bxl��  �          @���@
=q�n{?���A��\C�{@
=q�{�>���@�\)C�e                                    Bxl��  
�          @���@E��>�R?�(�A��C��R@E��P  ?+�A33C��3                                    Bxl�0  �          @�=q@3�
�U?���A��RC��@3�
�c�
>�(�@��C�q                                    Bxl��  �          @��H?�������?(��A�C���?������H�Ǯ��\)C�                                    Bxl�|  �          @�33@z�����?=p�A��C��H@z����\���R�x��C�xR                                    Bxl�"  �          @�=q@*=q�_\)?��Ai��C���@*=q�j=q>W
=@,(�C���                                    Bxl��  �          @�=q@z��u�?k�A:ffC��
@z��|(���Q쿎{C�<)                                    Bxl�n  �          @���@�H�r�\>��H@�{C�AH@�H�r�\�   ���
C�B�                                    Bxl  �          @���@p��p  ?8Q�A�C���@p��s�
��=q�W�C�n                                    Bxl�  �          @�  @G��u�?(�@���C�O\@G��w
=��������C�8R                                    Bxl$`  �          @�G�@"�\�QG�?(��A��C���@"�\�U��L���2�\C���                                    Bxl3  �          @�(�@@  �&ff@��A�(�C��@@  �HQ�?У�A�\)C��                                    BxlA�  T          @��@E�;�?��HA��RC�E@E�QG�?c�
A3�C���                                    BxlPR  �          @�33@;��G�?�\)A�  C���@;��\(�?B�\A��C�=q                                    Bxl^�  �          @��\@N{��\@(�Bz�C�˅@N{�'
=?�=qA�z�C��H                                    Bxlm�  �          @�33@>�R�/\)@
=qA���C���@>�R�Mp�?���A��
C�k�                                    Bxl|D  �          @�(�@?\)�QG�?�\)A��C�>�@?\)�`��>�@�(�C�8R                                    Bxl��  �          @�33@J=q�333?�\A��\C�<)@J=q�J=q?}p�AG33C�|)                                    Bxl��  �          @�=q@_\)��\@z�A�G�C��R@_\)� ��?�(�A�33C�#�                                    Bxl�6  �          @�G�@p�׿��@�AۅC��q@p�׿��?��A��C��3                                    Bxl��  �          @��@Y���z�@�
A�
=C�P�@Y���&ff?�Q�A�=qC�Ff                                    Bxlł  �          @�(�@^�R�
=@�A�G�C�^�@^�R�'
=?�ffA�
=C���                                    Bxl�(  �          @�z�@dz��?�  A���C�\)@dz��.{?�=qAXQ�C�W
                                    Bxl��  �          @�=q@B�\�$z�@�RA�\C��@B�\�Dz�?�p�A��C�]q                                    Bxl�t  T          @�=q@L(���@Q�A�p�C�T{@L(��9��?�z�A�
=C��{                                    Bxl   �          @�33@<���3�
@A��HC�4{@<���P��?��
A�33C�{                                    Bxl�  �          @�z�?�z����\?�Q�A���C��f?�z���(�?��@��HC��=                                    Bxlf  �          @��H?�=q�u�?ٙ�A�Q�C��{?�=q��z�?!G�Az�C��f                                    Bxl,  �          @�=q@���n�R?��A��C��{@����  ?   @�G�C�
                                    Bxl:�  
�          @��\?�{���H?W
=A)�C�4{?�{�����=q�XQ�C�H                                    BxlIX  �          @���@
�H�n�R?��A��HC��@
�H�|(�>�=q@^�RC�l�                                    BxlW�  T          @��\?ٙ���
=>��H@��
C��?ٙ���ff�&ff��\C��q                                    Bxlf�  T          @�33?��H���R?\)@��C��?��H���R�z�����C�                                    BxluJ  �          @�(�?������
?�ffAV{C�k�?�����  ���
����C�                                      Bxl��  �          @���?�Q��z�H?��
A��RC�'�?�Q���>�(�@�z�C�k�                                    Bxl��  �          @���?��R�33@e�BRG�C�j=?��R�K�@5B�C��=                                    Bxl�<  �          @���?\�{@^�RBH�\C��f?\�S�
@,(�B�C�P�                                    Bxl��  �          @���?����1G�@Y��BB�RC�#�?����e�@"�\B=qC�˅                                    Bxl��  �          @��\?��5�@aG�BJ�
C�C�?��j�H@(Q�Bp�C�{                                    Bxl�.  �          @��?���  @~{Bn��C���?���N�R@N{B1C�H�                                    Bxl��  �          @��\>����ff@���Bx=qC�h�>����Fff@S�
B:��C��                                    Bxl�z  �          @�G�?h���Fff@HQ�B0�C�5�?h���u�@�A�(�C���                                    Bxl�   T          @�G�>�{�c�
@0��B��C���>�{��?ٙ�A��\C�U�                                    Bxl�  �          @��>�\)�fff@+�B33C�0�>�\)��{?���A�G�C��                                    Bxll  �          @���>B�\��  @ ��AՅC�]q>B�\��z�?\(�A0(�C�=q                                    Bxl%  �          @�\)>�\)��Q�?���A���C�>�\)��z�?J=qA"�\C�ٚ                                    Bxl3�  �          @��>�
=�y��@
=qA���C��>�
=���H?��\AR�HC���                                    BxlB^  �          @�Q�?���J=q@G�B0C�c�?���x��@	��A��C�e                                    BxlQ  �          @���?���^�R@%�B�C��H?������?��
A�Q�C��H                                    Bxl_�  �          @�=q>B�\�H��@S33B9p�C��>B�\�{�@z�A�z�C�k�                                    BxlnP  �          @��>���_\)@=p�B ffC�Q�>����p�?�33A¸RC���                                    Bxl|�  �          @�z��G��u@$z�B�C�'���G�����?�A��HC�C�                                    Bxl��  �          @�z���z�H@��A��C�!H����{?��
A��RC�:�                                    Bxl�B  �          @�z������@��A�=qC�
����  ?���AU��C�/\                                    Bxl��  �          @�(�>�=q��Q�?Tz�A'�C��q>�=q��=q��G����RC��R                                    Bxl��  �          @�z�?�\���H>�33@���C�(�?�\��Q�n{�9C�8R                                    Bxl�4  �          @�z�>��H��=q>�@�
=C�3>��H���׿Q��#33C�)                                    Bxl��  �          @��>�33��\)?z�HAD��C�@ >�33���\�����l(�C�4{                                    Bxl�  T          @���>�����?L��A$Q�C�%>���ff��ff��ffC�)                                    Bxl�&  �          @��>aG���G�@ffA�
=C���>aG����R?c�
A4��C�g�                                    Bxl �  �          @���?�\��G�@G�A�z�C��3?�\��{?O\)A$��C�AH                                    Bxlr  �          @�Q�>��
���
?��A�p�C�7
>��
��ff?�@�=qC��                                    Bxl  �          @���?���?ǮA�z�C���?���ff>���@�  C�S3                                    Bxl,�  �          @�=q�u���?Q�A%�C����u��G������(�C���                                    Bxl;d  S          @��ý����>u@B�\C�8R����(����
�R�HC�33                                    BxlJ
  �          @��\��Q���Q�?\)@��C��쾸Q���\)�8Q��=qC��\                                    BxlX�  �          @�=q��Q���?��A]C�e��Q������B�\��HC�h�                                    BxlgV  �          @�{?L���g
=@�B(�C�7
?L������?���A�z�C�n                                    Bxlu�  �          @�33@#�
?�33@G
=B;(�A�G�@#�
>��@W�BQffA&{                                    Bxl��  �          @�ff@
�H�h��@k�Bep�C��@
�H��z�@QG�BB{C�Y�                                    Bxl�H  �          @��@녿L��@p��Bd��C�b�@녿�=q@XQ�BD��C��                                    Bxl��  �          @��?ٙ����R@g�BXp�C�E?ٙ��;�@:=qB"z�C��                                    Bxl��  �          @��R?�Q��p�@a�BZ\)C�*=?�Q��G�@0��BQ�C�s3                                    Bxl�:  �          @�z�@
=��  @aG�B\ffC���@
=���@AG�B2�HC��                                    Bxl��  �          @�33@	��>�Q�@�B9�Ap�@	������@�\B:p�C�                                      Bxl܆  �          @��H@HQ�@�@   Aޣ�B�@HQ�?�(�@!G�BQ�A�
=                                    Bxl�,  �          @�=q@>�R@)��?���A�\)B&�@>�R@@�\B{B�                                    Bxl��  �          @��
@.�R?���@<(�B+��A��H@.�R?0��@Q�BE\)Ac33                                    Bxlx  �          @�=q?���@@N{BAB;��?���?�=q@l(�Bk\)A�p�                                    Bxl  �          @�33@33>�@g�Bd{A:�R@33�z�@g
=Bb�\C��                                    Bxl%�  �          @��
@'�?
=q@X��BO
=A:=q@'���@Y��BP{C�
=                                    Bxl4j  �          @��
@,(�>\@XQ�BM�@��R@,(��(�@VffBJ33C���                                    BxlC  �          @�G�@�Ϳ��@H��B=(�C���@���-p�@p�B�RC���                                    BxlQ�  �          @���@   �#�
@P  BNz�C��f@   �Ǯ@;�B3�\C��                                    Bxl`\  �          @��@*=q���
@W
=BN��C���@*=q�}p�@Mp�BB(�C�                                    Bxlo  �          @�33@AG���=q@;�B)C�.@AG����@   B�C�
=                                    Bxl}�  �          @�33@!녿�\)@=p�B-{C��H@!��)��@�\B �C���                                    Bxl�N  �          @��@G���\@5B%��C�W
@G��AG�@33A�ffC�q�                                    Bxl��  �          @��H?�p��>�R@&ffB�C�
?�p��g
=?�\)A�Q�C��3                                    Bxl��  �          @���?��R�mp�?��HA��C�7
?��R��(�?J=qA'33C�Y�                                    Bxl�@  �          @��
?W
=��  >�ff@�(�C���?W
=��{�Q��,��C���                                    Bxl��  �          @�(�?�33�h��?�=qA�{C���?�33�|(�>�G�@���C���                                    BxlՌ  �          @��
?s33���?.{A\)C���?s33���\�
=q����C���                                    Bxl�2  �          @��?��R�z=q?˅A���C��{?��R���R>�p�@�33C�>�                                    Bxl��  �          @��?��R�u?Tz�A-G�C��\?��R�z=q��{��p�C�w
                                    Bxl~  �          @��
?�G��A�@:�HB)��C�/\?�G��p��?��Aϙ�C��                                     Bxl$  �          @��?����\(�@	��A�  C���?����z�H?��Aa�C�P�                                    Bxl�  �          @��H?�{�{�?��At��C���?�{���H���
��  C�1�                                    Bxl-p  �          @�33?�G���Q�?@  A   C�L�?�G����������p�C�1�                                    Bxl<  �          @��?���n�R?ٙ�A��RC�>�?����=q?�\@��C�w
                                    BxlJ�  �          @��
?�=q�p  ?��HA��C�� ?�=q���H?�@ڏ\C���                                    BxlYb  �          @�33?����g
=?���A���C�?�����  ?0��A��C��)                                    Bxlh  �          @�z�?Ǯ�K�@�B	�RC�?Ǯ�p��?�\)A��\C�B�                                    Bxlv�  �          @�33?�(��]p�?�A�ffC�8R?�(��i��=��
?�{C���                                    Bxl�T  �          @�z�>�33���?=p�AC�aH>�33���׿����C�]q                                    Bxl��  �          @�z�>Ǯ���׿@  ��C��q>Ǯ�vff�   ����C��f                                    Bxl��  �          @�(�?�  ����8Q���
C��?�  �p  ��
=�ӮC�y�                                    Bxl�F  �          @�z�?��H����B�\�p�C���?��H�w
=�����C�\)                                    Bxl��  �          @�p�?У���G�>��@]p�C��)?У��{��xQ��M�C�B�                                    BxlΒ  �          @���?�G���Q쾙���x��C��{?�G��n�R��  ���C���                                    Bxl�8  �          @�p�?�����=q��{��Q�C���?����qG��Ǯ��33C�Ff                                    Bxl��  �          @�
=@�
�g
=?�\)A�{C��q@�
�{�>�ff@�\)C��
                                    Bxl��  �          @��R@�
�tz�?0��AG�C�+�@�
�vff�   �љ�C�3                                    Bxl	*  �          @�ff?�Q��}p��k��@  C��?�Q��mp���
=����C�Ф                                    Bxl�  �          @�@��r�\?.{A\)C�Z�@��tz��\��p�C�E                                    Bxl&v  �          @�ff@�\�p��>���@�p�C��@�\�l�ͿE�� ��C��H                                    Bxl5  �          @�?ٙ��~�R���
��
=C��3?ٙ��l(��\��ffC�`                                     BxlC�  �          @�?�G��~�R�8Q���\C���?�G��dz��z�����C�&f                                    BxlRh  �          @�?����{��J=q�%G�C�� ?����`  ���H����C��R                                    Bxla  �          @��R?��R����
=��C�˅?��R�u��Q���{C��3                                    Bxlo�  �          @�ff?�  ��{��G��R�RC��3?�  �j�H�  ����C���                                    Bxl~Z  �          @�
=?��R��ff�\)����C�AH?��R�tz��=q���
C�                                    Bxl�   �          @�ff>L��������R���\C�XR>L���i���   �	\)C��\                                    Bxl��  �          @�{?�ff�����R� ��C���?�ff�u��33��ffC���                                    Bxl�L  �          @��R?����(������C�8R?���q녿޸R��ffC��                                    Bxl��  �          @��?�\)��33�Y���-�C�?�\)�hQ�����
=C���                                    Bxlǘ  �          @�ff?������R�W
=�-C�K�?����o\)�
=��33C�*=                                    Bxl�>  �          @�ff?�����p�����d(�C�1�?����g������C�<)                                    Bxl��  �          @�{?c�
����z��r{C�H?c�
�g
=�=q��RC���                                    Bxl�  �          @�p�?��z=q>��@^�RC��{?��s33�z�H�T��C��                                    Bxl0  �          @���@�\�vff?
=q@��
C��@�\�u��0���p�C��                                    Bxl�  �          @�p�?���}p�?@  A  C�%?����  �
=q��Q�C�                                    Bxl|  �          @��?:�H��  ����(�C��?:�H���׿�\)��Q�C�*=                                    Bxl."  �          @�
=?G��XQ��3�
�  C�w
?G��z��o\)�c{C�@                                     Bxl<�  �          @�\)?:�H�\(��1G��
=C��?:�H�Q��n{�`��C���                                    BxlKn  �          @�\)?5�K��E��-��C�G�?5��\�{��t��C��{                                    BxlZ  �          @�
=?&ff�Q��n{�b{C���?&ff��������u�C��H                                    Bxlh�  �          @�{?����R�e�YQ�C���?���xQ����
(�C�p�                                    Bxlw`  �          @��
?Tz����R>\)?���C��R?Tz����ÿ��H��\)C���                                    Bxl�  �          @�p�?�=q�^�R�Q���RC�>�?�=q�'
=�G��6�C���                                    Bxl��  �          @��?�p�����33����C�9�?�p��vff����G�C���                                    Bxl�R  �          @�p�?�����H?333A(�C���?����33�&ff�	��C��
                                    Bxl��  �          @��H?�33�o\)?ٙ�A��\C�8R?�33���H>��@�z�C�j=                                    Bxl��  �          @�z�?k����@,��Bn�\C���?k��@��B,�
C���                                    Bxl�D  �          @��\=u?�33@tz�B���B��f=u=���@�=qB�ǮBc�                                    Bxl��  �          @�ff>Ǯ?�  @x��B��B���>Ǯ>#�
@��B�k�A�G�                                    Bxl�  �          @��\?��>�@�33B�  A��?���c�
@���B�u�C��R                                    Bxl�6  �          @�33?��?�@�B��HA��H?���\(�@��
B���C��q                                    Bxl	�  �          @�(�?G�?k�@�ffB��BG�
?G����H@���B��qC��
                                    Bxl�  �          @��?5?�33@�(�B�Bh�H?5���@���B��C�                                    Bxl'(  o          @�=q?�z��C33@+�B
=C�l�?�z��p��?��A�\)C�N                                    Bxl5�  �          @��\?����U@Q�Bz�C�@ ?����|(�?�z�Axz�C��\                                    BxlDt  �          @��
?��R�C�
@3�
B!�C�
=?��R�s�
?�z�A��C��                                    BxlS  �          @��
?L���)��@P  BF��C�k�?L���dz�@p�A��HC�T{                                    Bxla�  �          @�z�>8Q���
@uBv��C��>8Q��Mp�@<��B*�C��                                    Bxlpf  �          @�p�?#�
�_\)@%B\)C�:�?#�
����?��A�
=C�j=                                    Bxl  �          @�(��\� ��@X��BTC����\�_\)@Q�BQ�C��=                                    Bxl��  �          @�33�5�\@�Q�B���Ct�H�5�0  @Q�BD=qC��                                    Bxl�X  �          @�p����ÿ�  @���B�.C��f�����>�R@Mp�B;�
C���                                    Bxl��  �          @�Q����p�@���B=qC�!H���L��@H��B1�HC��R                                    Bxl��  �          @��ý�\)�
=@�  Bx�HC�׽�\)�Tz�@E�B+Q�C�`                                     Bxl�J  �          @�  ���
���@��\B��3C�����
�H��@N{B6Q�C��                                    Bxl��  �          @�G�>�
=�(�@p��Bbp�C��{>�
=�c33@.�RBG�C�U�                                    Bxl�  �          @�Q�?}p�����@{�Bw�\C�?}p��C�
@E�B/Q�C���                                    Bxl�<  �          @�G�?\)�=q@p  Bb��C��f?\)�aG�@.�RB��C�~�                                    Bxl�  �          @�33�u�J=q@S�
B8��C��
�u���H@�
A�33C�T{                                    Bxl�  �          @��;���fff@;�BG�C���������
?�ffA��C�G�                                    Bxl .  �          @��H������Q�@��A�=qC�E������Q�?8Q�A�C�Y�                                    Bxl.�  �          @��
��\)���@	��A޸RC��H��\)����?&ffA=qC��\                                    Bxl=z  �          @�G�?aG��fff@%�B�C��q?aG���Q�?��HAy��C��{                                    BxlL   �          @�=q?xQ��c�
@*�HBffC��H?xQ���  ?�ffA��C�o\                                    BxlZ�  �          @�33?�
=�W�@)��B�C�@ ?�
=���?���A�ffC�8R                                    Bxlil  �          @��?�
=�:=q@8��B�C�� ?�
=�n{?�p�A��C���                                    Bxlx  �          @���?��H�R�\@9��Bp�C��)?��H���\?�{A���C��)                                    Bxl��  �          @��?\�qG�@��A�\C��q?\��=q?W
=A&=qC��
                                    Bxl�^  �          @���?�Q��O\)@6ffB�C�?�Q�����?���A���C�e                                    Bxl�  �          @��?�\)�L��@5�BffC�%?�\)�~{?ǮA���C��
                                    Bxl��  �          @�(�?�33�W�@�B �\C���?�33��  ?���Ad��C��R                                    Bxl�P  �          @�z�?�p���33>�
=@�{C�]q?�p���\)�����]G�C��q                                    Bxl��  �          @��\?�G���=q=�?\C��H?�G����H��\)���C�)                                    Bxlޜ  T          @���?����=q>�=q@^{C��3?�����Ϳ�p��{33C�J=                                    Bxl�B  T          @�{?��\����>\)?���C���?��\��녿�����z�C��                                    Bxl��  �          @�{?@  ���H�#�
��C��q?@  ���׿�z���ffC�@                                     Bxl
�  �          @�{?B�\��33>u@HQ�C��R?B�\��p����
���RC�0�                                    Bxl4  �          @��>�(����=L��?5C�Ф>�(�������H��{C��)                                    Bxl'�  �          @�G���33���R>��@��C�Ǯ��33���H��{�d��C��R                                    Bxl6�  �          @�=q�k�����>�
=@�Q�C��\�k���zῘQ��qC���                                    BxlE&  �          @��H��  ��G�?��@���C�h���  ���R���\�N{C�aH                                    BxlS�  �          @��H�����>�@�Q�C�5þ�����z��k\)C�0�                                    Bxlbr  �          @��?W
=����=��
?s33C�<)?W
=��Q��G����C���                                    Bxlq  �          @�33?�=q��ff�u�?\)C���?�=q���\�����\)C�g�                                    Bxl�  �          @�z�?У�����=�\)?Y��C�b�?У�������������C��
                                    Bxl�d  �          @���?�����\�\)��
=C�^�?����Q��33��33C�"�                                   Bxl�
  �          @�z�?�  ��{�B�\���C�ٚ?�  ���H��  ���C�}q                                   Bxl��  �          @�z�?����=q>�?˅C���?����=q��  ��p�C���                                    Bxl�V  T          @��?n{��  �
=q��C�ٚ?n{�����ff�׮C���                                    Bxl��  �          @�(�?(�����ÿ�����
C��?(����G�����\)C��)                                    Bxlע  �          @�p�?&ff���\�����
C��?&ff���ÿ�33��p�C�Q�                                    Bxl�H  �          @�{?aG����H=#�
>�C�h�?aG�����������(�C�                                    Bxl��  �          @�  @
=�aG��Q����HC�z�@
=�(��^�R�<�C�p�                                    Bxl �  �          @�  ?���l����
���HC�p�?���(Q��^�R�<�C���                                    Bxl :  �          @���?�p��hQ�����Q�C�S3?�p��"�\�`���=C��)                                    Bxl  �  �          @�=q@
=�AG��E��z�C�n@
=��(��}p��^{C�e                                    Bxl /�  �          @���?�p��'
=�\(��9��C���?�p����������s\)C�k�                                    Bxl >,  S          @��R@(��#33�R�\�1�C�\)@(���Q��\)�g��C���                                    Bxl L�  �          @�Q�@'�����G��$Q�C�t{@'���z��r�\�T  C�{                                    Bxl [x  �          @�
=@(���
=q�N{�-��C�Z�@(�ÿY���q��W33C�R                                    Bxl j  �          @�{@,(�����^�R�?z�C�T{@,(���=q�u��[ffC�,�                                    Bxl x�  �          @�
=@8�ÿfff�hQ��HC�U�@8��>��mp��N�HA�                                    Bxl �j  �          @�p�@(��\(������n��C�@ @(�?(����=q�rp�A��                                    Bxl �  �          @�Q�@��\(���z��q�C�L�@�?5��p��t��A�(�                                    Bxl ��  �          @�  @�Ϳu���H�np�C�C�@��?������t��Atz�                                    Bxl �\  �          @���?�33?   ��p�(�A�Q�?�33?�p��l���\z�BIQ�                                    Bxl �  �          @���?�\)���
�:=q�[{C��?�\)��Q��QG�{C��=                                    Bxl Ш  �          @�33��=q���R�^�R�0z�C�=q��=q�u�������C��{                                    Bxl �N  �          @�G���\)���R��z���\)C�쾏\)�S�
�E�+��C���                                    Bxl ��  �          @�=q��
=���R������p�C�  ��
=�XQ��9���!�C�n                                    Bxl ��  �          @�zᾮ{�|(�����G�C�����{�2�\�j=q�Q�C��f                                    Bxl!@  �          @���L���vff�&ff��
C�z�L���'��u�^Q�C��                                    Bxl!�  �          @��Ϳ+����
�G��Ώ\C�ff�+��E��Y���<��C���                                    Bxl!(�  �          @�(�����\)��\��
=C�w
���QG��Mp��0��C�q�                                    Bxl!72  �          @�33��{�|����
��33C��3��{�4z��g
=�OQ�C���                                    Bxl!E�  �          @��H>��a��:=q�
=C�R>��(���Q��u=qC�                                    Bxl!T~  �          @��H�#�
�j=q�0  ��C����#�
���z�H�k  C��                                    Bxl!c$  �          @��
����mp�� ���(�C}�{����   �mp��VCvh�                                    Bxl!q�  �          @�{���H�n{�$z��Q�C|  ���H�\)�qG��W
=Ct)                                    Bxl!�p  �          @�p��^�R�i���.�R���C�T{�^�R���y���dffCy�f                                    Bxl!�  �          @��H�p���Dz��Mp��433C|��p�׿����(��
Cp0�                                    Bxl!��  �          @���z��dz��5���C�]q�z��  �}p��nffCxR                                    Bxl!�b  �          @��\)�~�R��H��z�C��\)�1��p  �U��C���                                    Bxl!�  �          @�{�B�\�z=q�$z��
=C�� �B�\�*=q�vff�]p�C���                                    Bxl!ɮ  �          @�p���Q��}p��{��p�C�^���Q��.�R�q��XffC�
                                    Bxl!�T  �          @�z�>��b�\�<����HC���>��
�H��=q�u�C�0�                                    Bxl!��  �          @��>��QG��Fff�-��C�#�>���{���G�C��                                    Bxl!��  �          @�(�=���ff���R�C�P�=�?!G�����¢��B�.                                    Bxl"F  �          @�(�=u=L����ff±�BG�=u?���w
=  B���                                    Bxl"�  �          @�(�>��?�����{ffB��>��@(���\(��Q��B�aH                                    Bxl"!�  �          @�G�>�  >�z���  ©ǮBE��>�  ?���tz��}��B��                                    Bxl"08  �          @�G�>.{?\(���z�#�B�>.{@��`  �_
=B���                                    Bxl">�  �          @���?���E��XQ��<��C��?�ͿǮ����
=C���                                    Bxl"M�  �          @�����P���Z�H�933C��{���ٙ�����3C�p�                                    Bxl"\*  �          @���<��
�@���k��K�C�33<��
��{�����qC�p�                                    Bxl"j�  �          @��þL���W��W
=�3�\C�W
�L�Ϳ��������fC��                                    Bxl"yv  �          @��ÿ(��L(��`  �<��C��Ϳ(��������R�Cx�                                    Bxl"�  �          @������R�\�]p��8��C�⏾��ٙ����R�)C~\)                                    Bxl"��  �          @�Q����C�
�g
=�E�HC�#׾�녿�
=��Q���C}Ǯ                                    Bxl"�h  �          @�����z��E��i���FC�Ff��zῷ
=����=C�/\                                    Bxl"�  �          @��H�.{�U�`  �9=qC���.{��(�����(�C�"�                                    Bxl"´  �          @��<#�
�Y���]p��6
=C�)<#�
�����Q�C�4{                                    Bxl"�Z  �          @��\�+��k��B�\�ffC��\�+��p���\)�u��C}�                                    Bxl"�   �          @��
�&ff�~{�.{�G�C�W
�&ff�%�����bQ�C��                                    Bxl"�  �          @�(��aG���=q�z���p�C�Ff�aG��J�H�fff�>G�C~��                                    Bxl"�L  �          @��\�G����R����ٮC���G��@���j�H�F�HCk�                                    Bxl#�  �          @��\�&ff�p���<(���C�  �&ff��
���p�RC~Y�                                    Bxl#�  �          @�33�#�
�p  �@���Q�C�#׿#�
�G���\)�s�C~.                                    Bxl#)>  T          @�녿k��vff�+��Q�C�AH�k��\)�~�R�a(�Cy��                                    Bxl#7�  �          @����n{�l(��:�H�
=C�׿n{�\)��z��nz�Cw�                                    Bxl#F�  �          @�G���
=�p���-p��
(�C|�Ϳ�
=����~�R�`��Cs��                                    Bxl#U0  �          @�G������o\)�)����\Cz0���������z=q�[�RCp�
                                    Bxl#c�  �          @������
�g��/\)�p�Cw{���
�\)�|���]��Ck��                                    Bxl#r|  �          @�G�����j�H�+���RCwLͿ����
�z=q�Z�Cl^�                                    Bxl#�"  �          @�G��У��l���"�\� �Cv+��У�����s33�S
=Ck�{                                    Bxl#��  �          @��׿�{�^{�1��  Cu
=��{�z��{��`��Ch&f                                    Bxl#�n  �          @�{��G��XQ��L���-�\C�)��G���=q�����3C�Y�                                    Bxl#�  �          @��R?��Z=q�K��*�\C���?녿�{����.C���                                    Bxl#��  �          @�?���Fff�\(��>
=C�  ?�Ϳ�(���z��3C�5�                                    Bxl#�`  �          @�ff<#�
�C33�@  �2(�C��<#�
�˅�~{G�C�1�                                    Bxl#�  �          @��H��Q��o\)�#33��
=Cu�f��Q�����u�RG�Cj�
                                    Bxl#�  �          @�����  �p�������Cu
��  �\)�l���JffCj�f                                    Bxl#�R  �          @��\�k��\)�!���p�C���k��(���z�H�Z�Cz��                                    Bxl$�  �          @��\��{�|(�� �����RC~G���{�%�x���XG�Cv�
                                    Bxl$�  �          @�=q���s�
�%��Cy�����(��y���X�Co�                                    Bxl$"D  �          @��\�����\)���R�¸RCu#׿����5�[��5Q�Cl�q                                    Bxl$0�  �          @��H��\�n�R��
��  Cq^���\��R�g��A�HCf��                                    Bxl$?�  �          @�G��z��W
=�,���	��Cnn�z��p��u��U(�C_�q                                    Bxl$N6  �          @�=q�p��33�dz��A
=Cb+��p��&ff��ff�u{CDk�                                    Bxl$\�  �          @��\�   �7��;���Cd���   ��
=�u�Tp�CQ�
                                    Bxl$k�  �          @��\�
�H�!G��^{�8�HCeB��
�H�c�
��{�s33CJ=q                                    Bxl$z(  �          @��ÿ����,(��mp��K�Csk����Ϳp����\)ǮCV��                                    Bxl$��  �          @�  ��
=�   �q��W�Ct���
=�=p���
=ffCT)                                    Bxl$�t  �          @�  �����5�e��E  Cw�����������8RC_33                                    Bxl$�  �          @�?�  �n{�#�
��C���?�  ��w
=�bffC���                                    Bxl$��  �          @�p�@���Dz��p����HC���@�ÿ���Q��<\)C�˅                                    Bxl$�f  �          @�{@�ff���R>�\)@Z�HC�S3@�ff��
=����\)C���                                    Bxl$�  �          @�@����ff�\��(�C���@��Ϳ�G����H�pz�C�}q                                    Bxl$�  �          @�z�@xQ����^�R�,��C�J=@xQ��ff��\���C���                                    Bxl$�X  �          @��@}p��(������TQ�C���@}p��Ǯ������\C�G�                                    Bxl$��  �          @���@\)��ff������z�C���@\)�z�H�����\)C�q                                    Bxl%�  �          @�(�@S33�8Q쿴z�����C�s3@S33��\��R�=qC�*=                                    Bxl%J  �          @���@u���  � ����ffC�K�@u����p���\C��                                    Bxl%)�  �          @�z�@e����*�H���C��R@e���
�>{��\C�Y�                                    Bxl%8�  T          @��@�ff�QG��8�\C�Ǯ@�G��|���r�C��=                                    Bxl%G<  �          @�(�?�G��У��x���i33C��q?�G�>\)���Ru�@�                                      Bxl%U�  �          @�z�?����(���P���5ffC��H?��Ϳ������y�C�`                                     Bxl%d�  �          @�z�@*=q�E�����C�e@*=q���L(��0�
C��                                    Bxl%s.  �          @��@#33�1G��%�
33C�L�@#33��
=�aG��I
=C�Y�                                    Bxl%��  �          @�p�?�  �R�\�333��HC��?�  ���z�H�e=qC�{                                    Bxl%�z  �          @��?��
�7
=�`���D�C��f?��
��\)��z�ǮC�O\                                    Bxl%�   �          @�ff?�Q��P  �8���{C��{?�Q�޸R�\)�k{C�R                                    Bxl%��  
�          @��?�{�Z=q�*�H�	��C�L�?�{��(��w
=�[�RC��\                                    Bxl%�l  �          @�
=?�p��c�
�333�ffC���?�p��33��G��m�\C��                                    Bxl%�  �          @�\)=��{��&ff�  C��R=��p������i�C�XR                                    Bxl%ٸ  �          @��<��
�Y���L(��,��C�/\<��
��p����\z�C�\)                                    Bxl%�^  �          @���?u�U�N{�+33C�
=?u��z����H33C�                                    Bxl%�  �          @��þL���w��,(��(�C����L���
=���\�op�C��R                                    Bxl&�  �          @�  �c�
�mp��2�\���C�8R�c�
�
�H��33�pCw��                                    Bxl&P  �          @�\)�z�H�Z�H�Dz��#Q�C}��z�H��������Cq0�                                    Bxl&"�  �          @��fff�Fff�W��8��C}��fff��\)���
��Cl��                                    Bxl&1�  �          @��G��1��E�'�Ci���G���Q��~�R�m��CR�\                                    Bxl&@B  �          @����Ǯ���^�R�I(�CmE�Ǯ�5��k�CLn                                    Bxl&N�  �          @�p��.{�]p���
��=qC���.{�\)�U�`Q�C��                                    Bxl&]�  �          @����aG���\)��
��Q�C�!H�aG��<(��k��H��C}T{                                    Bxl&l4  �          @�G����R������\)���C�쾞�R�a��L(��(33C��H                                    Bxl&z�  �          @����ff�qG��
�H��{C��R��ff�{�e��[�
C��
                                    Bxl&��  �          @�녿У��K��1���\Cr�)�У׿��xQ��k��Ca��                                    Bxl&�&  �          @�����aG��.{��HC{�Ϳ��   �~{�n�Co��                                    Bxl&��  �          @�(������e��&ff�	  Cy� �����
=�x���e��Cm��                                    Bxl&�r  �          @��Ϳ�ff�QG��8Q���Ct�)��ff�ٙ���Q��p��Cc��                                    Bxl&�  �          @�(��p���p���!��C��p����\�z=q�f�RCw�R                                    Bxl&Ҿ  �          @��
�=p������G���(�C�Ǯ�=p��0  �dz��M�C~�f                                    Bxl&�d  �          @�{=��
����>��H@�ffC���=��
��녿��
��Q�C��=                                    Bxl&�
  �          @��?Tz���{?\(�A,z�C�O\?Tz����
��z��i�C�c�                                    Bxl&��  �          @��
?��R��\)�k��?\)C�(�?��R�mp��33��\)C�=q                                    Bxl'V  �          @�33>�����=q������C�
>����Fff�;��-
=C��q                                    Bxl'�  �          @�G��#�
�U��5�"{C��H�#�
��  ��Q��
C��f                                    Bxl'*�  �          @�녿   �k��#33�
  C�  �   �(��y���p33C���                                    Bxl'9H  �          @��?�=q�:�H���\�\C�R?�=q?����\)�v��A�=q                                    Bxl'G�  �          @�=q?�녿����vff�f(�C�{?��=u��Q�@                                       Bxl'V�  �          @�G�?��Ϳ�p��n{�^z�C�~�?��;\)���R33C�n                                    Bxl'e:  �          @�G�>���N�R�G��.C�.>�׿��
��
=u�C��                                     Bxl's�  �          @�G�>�ff�k��%���C�� >�ff�
�H�|���r�HC��f                                    Bxl'��  �          @�G�?B�\�n{��R��RC��)?B�\�  �w��j�C�L�                                    Bxl'�,  �          @���?�p���{��{�c\)C�,�?�p��QG��5�z�C�L�                                    Bxl'��  �          @��\?0����p��333��RC�p�?0���i���%��(�C�Y�                                    Bxl'�x  �          @�p�?��~�R�s33�UC�?��H���'��z�C���                                    Bxl'�  �          @�33>��]p�?.{A1�C��{>��Y���k��p  C�                                    Bxl'��  �          @��?+��ٙ�@w�B�ffC��
?+��O\)@-p�B=qC���                                    Bxl'�j  �          @��
>���333@���B�  C��H>���!G�@`��BY
=C��f                                    Bxl'�  �          @�
=>��
�:�H@��B��C��
>��
��R@W
=BU33C��                                     Bxl'��  �          @���?E���=q@���B�W
C��?E��0��@G�B=z�C��{                                    Bxl(\  �          @�33?�����@q�Bu�C�,�?��0��@6ffB%��C��)                                    Bxl(  �          @���@Q��X��?��AhQ�C�
=@Q��`  �
=��HC��                                    Bxl(#�  �          @��@�H�I�������C���@�H�333��Q���C�l�                                    Bxl(2N  �          @�@��P�׾�����RC�` @��.�R������  C�ٚ                                    Bxl(@�  �          @�{?�z��:=q����Q�C��f?�z��{�QG��j��C��f                                    Bxl(O�  �          @�\)@�
��
=@S33B>�
C�
@�
�L��@A��HC��                                    Bxl(^@  �          @�G�@ ���{@UB@z�C��@ ���_\)?�p�A�  C���                                    Bxl(l�  �          @��R@���-p�@.�RBQ�C�#�@���h��?��HA�ffC�0�                                    Bxl({�  �          @�\)@-p����@*�HBC��@-p��N{?�\)A�Q�C�
=                                    Bxl(�2  �          @��@
=�(�@9��B"  C�  @
=�^�R?�p�A�\)C��                                    Bxl(��  �          @�{@�\�[�?���A�ffC�^�@�\�y��=�Q�?�p�C��3                                    Bxl(�~  �          @�{@���L(�@   A�Q�C�N@���p  >�{@�
=C�5�                                    Bxl(�$  �          @�?����fff?���A�Q�C�c�?������ü#�
��C�                                      Bxl(��  �          @��?�׿�z�@hQ�B]Q�C�9�?���HQ�@�RB�C�z�                                    Bxl(�p  �          @���?���p�@j�HB�.C�Ф?����R@FffBHQ�C��\                                    Bxl(�  �          @�ff?�
=���@��HB�ffC��?�
=�AG�@C33B,C���                                    Bxl(�  �          @��R?��ÿ��\@���B�Q�C�{?����>{@A�B+�RC���                                    Bxl(�b  �          @�{>���?�(�@��B�  B�>����J=q@��RB�Q�C���                                    Bxl)  �          @�p�?B�\���@�Q�B�k�C���?B�\��R@aG�BVffC��                                     Bxl)�  �          @�p�?��R��33@���B�u�C�5�?��R���@a�BY�C��3                                    Bxl)+T  �          @�p�@z����@xQ�Bv�C��@z��Q�@QG�B?
=C��                                    Bxl)9�  �          @�(�@��L��@\(�BX  C���@���\@,��B(�C�f                                    Bxl)H�  �          @��H@C33��=q@0��B�RC�@ @C33���?�A�  C��                                    Bxl)WF  �          @���@z��\)@3�
B$p�C��@z��Q�?�(�A�
=C���                                    Bxl)e�  �          @��
@����@5�B#{C�5�@���N�R?�G�A�G�C��R                                    Bxl)t�  �          @�G�@��!G�@   Bp�C���@��XQ�?��Af{C�~�                                    Bxl)�8  �          @�33?Q녿�@S33Bk�C�1�?Q��HQ�@Bp�C�\)                                    Bxl)��  
�          @���Q��=q@���B�p�C�e��Q��[�@J�HB+  C�B�                                    Bxl)��  �          @�{�\)��z�@���B�{C�t{�\)�G�@_\)B@ffC��                                    Bxl)�*  �          @��R���Y��@��B�  C������8Q�@l��BP�C��                                    Bxl)��  �          @�p�>��Ϳ�p�@�\)B���C��q>����J=q@Y��B;G�C��R                                    Bxl)�v  �          @��R��<��
@�G�B��\C2&f���@���Bw\)C�f                                    Bxl)�  �          @�\)?p�׿��@�=qB�C��
?p���]p�@Dz�B"G�C��                                     Bxl)��  �          @�  �\)�L��@��B��Ck&f�\)�z�@\)Bo
=C�Ff                                    Bxl)�h  �          @��H��  ?O\)@��B�=qC&f��  ��
=@�ffB���Cd�H                                    Bxl*  �          @����?h��@��B�ǮC� �����@�z�B���C^&f                                    Bxl*�  �          @����p�?���@���B�aHC�쿝p����@��B�{C\k�                                    Bxl*$Z  �          @�(���  ?��@�
=B�B�C�)��  ��33@�ffB��)CYn                                    Bxl*3   T          @��ͿǮ?���@��B��=C
=�Ǯ����@�  B�CW&f                                    Bxl*A�  �          @����?��H@�33Bz��CO\��녿xQ�@��B��CO!H                                    Bxl*PL  �          @��R��?Y��@�Q�B�(�C�=����{@�z�Bz�RCX�                                    Bxl*^�  �          @�\)���
>�@�{B�33C"�)���
���
@��Bw
=Ce5�                                    Bxl*m�  �          @�Q쿈��?��@��\B�u�C�����ÿ޸R@�G�B���CnW
                                    Bxl*|>  �          @�ff����u@��B���C@k�����{@�(�Bc33Cv(�                                    Bxl*��  �          @�p������@W�BWC�����p  ?���A��C��                                    Bxl*��  �          @�p�>aG��|(�@2�\B{C���>aG���G�?��@�{C�J=                                    Bxl*�0  �          @�z�G��33@�
=Bpp�C{:�G�����@(��B=qC��                                     Bxl*��  �          @���aG��:�H@uBN�C}G��aG���(�@   A�
=C�XR                                    Bxl*�|  �          @�ff��(��R�\@j=qB?�C�Ff��(���(�?�A��
C�XR                                    Bxl*�"  �          @�{�����p  @L(�B!z�C�C׽�����=q?��AB=qC�l�                                    Bxl*��  
�          @�녿@  �1G�@u�BT�RC~�Ϳ@  ��  @�
A�Q�C��)                                    Bxl*�n  
�          @��\�p���HQ�@c33B=ffC}LͿp����{?��A��HC��                                    Bxl+   T          @��ÿ�{�W�@K�B'Q�C{��{��\)?���Ah��C��                                    Bxl+�  �          @���=�\)�W�@VffB3ffC���=�\)����?���A��
C�j=                                    Bxl+`  �          @�33>��5@x��BW\)C�XR>���33@z�A˙�C��H                                    Bxl+,  �          @���?J=q�\(�@��RB�L�C�J=?J=q�@  @o\)BI\)C�Y�                                    Bxl+:�  �          @���>�{�#�
@��B��HC��3>�{�6ff@z�HBW  C�Y�                                    Bxl+IR  �          @��>8Q�Tz�@�Q�B��RC��>8Q��@  @r�\BN�C���                                    Bxl+W�  �          @���>\)��@�  B��=C�h�>\)�33@��BtQ�C�                                    Bxl+f�  �          @�\)�8Q�#�
@�p�B�u�C7xR�8Q��\)@��HBt�
C��                                    Bxl+uD  �          @�?�{�3�
@^{B@=qC��3?�{��(�?��HA�\)C�#�                                    Bxl+��  �          @��?˅�L(�@G�B$�C�>�?˅����?���Ak�C�*=                                    Bxl+��  �          @��?�(��a�@%�BffC��3?�(����\?�@��
C�4{                                    Bxl+�6  �          @��\@���  ?��AS�
C��@����׿}p��D(�C��H                                    Bxl+��  �          @�
=?����33��33����C�K�?���:�H�L(��*ffC�W
                                    Bxl+��  T          @�\)@  �x�ÿ��H��=qC��@  �-p��J=q�'z�C��)                                    Bxl+�(  �          @��?У�����B�\���C�B�?У��\���0����C���                                    Bxl+��  �          @���?�ff��ff��
=���C��)?�ff�l��� ��� 33C�b�                                    Bxl+�t  �          @�  @����
�B�\���C��@��O\)�)����C��
                                    Bxl+�  �          @���>����\)>�=q@Mp�C�y�>�������33�̏\C��q                                    Bxl,�  �          @�G�>�
=��  >�=q@QG�C���>�
=��G���
��Q�C��f                                    Bxl,f  �          @���>�{��  �8Q����C��>�{��=q�p���{C�b�                                    Bxl,%  �          @��?�  ��p�?(��A   C�b�?�  ��{������p�C��                                     Bxl,3�  T          @��?�\)��
=?
=q@�
=C��\?�\)��p��ٙ���=qC���                                    Bxl,BX  �          @�Q�?�p����R?8Q�Ap�C���?�p���  ���
��33C�                                    Bxl,P�  �          @�(�?���z�H@�\A�(�C�{?����p�����ffC��                                    Bxl,_�  �          @�  ?��
�xQ�@G�A�{C�� ?��
���=�Q�?�{C�e                                    Bxl,nJ  �          @�?���p��@33A�z�C�%?������>8Q�@\)C���                                    Bxl,|�  �          @�33?�ff�xQ�?��A�(�C��q?�ff��녾�  �E�C��H                                    Bxl,��  �          @�\)?�����p�?�z�A��C�&f?�����ff�
=��p�C��)                                    Bxl,�<  �          @�{?c�
��Q���\C��q?c�
�z�H��R���C�n                                    Bxl,��  �          @�ff?333��=q?5A�
C�XR?333���\��{��p�C��3                                    Bxl,��  �          @�(�?Tz�����>\)?�G�C�+�?Tz������ff��{C��                                    Bxl,�.  �          @��?}p����=#�
>�C�<)?}p��{��
=q����C��                                    Bxl,��  �          @��?\(���  ��z��h��C�k�?\(��q��p���\C�k�                                    Bxl,�z  �          @�z�@p��1G�@Q�A�RC�@p��]p�>��H@�33C��\                                    Bxl,�   �          @�@N�R���H@-p�B��C�b�@N�R�B�\?�33A�z�C�`                                     Bxl- �  �          @��@#33��@Mp�B.�C��q@#33�^{?�(�A�ffC�"�                                    Bxl-l  �          @��H@'
=���@4z�B{C�h�@'
=�aG�?�  A�C�E                                    Bxl-  �          @�z�@  �A�@  A���C�E@  �p  >��@�(�C�xR                                    Bxl-,�  �          @�  @��k�?�{A�33C�S3@��w��!G�� ��C���                                    Bxl-;^  �          @�?��R��ff>�(�@���C��=?��R�w
=�ٙ����RC��)                                    Bxl-J  �          @�ff?�\��33<�>��RC�R?�\�r�\�����C��f                                    Bxl-X�  �          @���?�z����R����G
=C�1�?�z��p  �����C��                                    Bxl-gP  �          @��?�\)��{�#�
�\)C���?�\)�w
=�p��ޏ\C�^�                                    Bxl-u�  �          @�p�?У����\��{����C�L�?У��fff�(���G�C�,�                                    Bxl-��  �          @�  ?�G��X��@(�A���C���?�G�����>k�@;�C��                                    Bxl-�B  �          @�33@�R�@W�B?�C�y�@�R�^�R?��Aď\C�W
                                    Bxl-��  �          @��R@.�R���
@XQ�B7�HC�u�@.�R�N{@33A���C�(�                                    Bxl-��  T          @��
@&ff�Ǯ@^�RBC�C���@&ff�E�@  A��HC�
                                    Bxl-�4  �          @�=q@�
��\@W�B>
=C�AH@�
�\��?�z�A�\)C��                                    Bxl-��  �          @���@��(�@N�RB6p�C��@��`  ?��HA�(�C��f                                    Bxl-܀  T          @�G�@p��\)@9��B*�C�W
@p��X��?�33A��RC��R                                    Bxl-�&  �          @��\@�
�.{@!G�B(�C��@�
�g
=?Tz�A5C�Ф                                    Bxl-��  �          @�z�@	���w
=?��A�33C��@	�����׿B�\��
C�{                                    Bxl.r  �          @��@Q��l��?�\A�=qC���@Q����H���R�w
=C���                                    Bxl.  �          @��
?��R�Y��@�HA���C�&f?��R���>���@�
=C��f                                    Bxl.%�  �          @���@���X��@\)A��C��@�����>�  @FffC��                                     Bxl.4d  �          @�(�@��AG�@'
=B	�C�˅@��{�?@  A��C�T{                                    Bxl.C
  �          @��R@ ���E�@!G�B�\C���@ ���{�?#�
@�=qC�L�                                    Bxl.Q�  �          @�\)@+��:�H@$z�B  C�AH@+��s�
?B�\A��C��                                    Bxl.`V  �          @�\)@'
=�?\)@"�\BG�C��{@'
=�w
=?333A�
C�                                    Bxl.n�  �          @�{@.�R�:�H@(�A�33C���@.�R�p  ?#�
@��C��                                    Bxl.}�  �          @��@:�H�8Q�@�RA�z�C��R@:�H�g
=>��H@�
=C��                                     Bxl.�H  
�          @�
=@N�R�/\)@33AθRC���@N�R�Y��>���@�=qC�                                    Bxl.��  �          @�Q�@W��8Q�?�  A�=qC���@W��W�=u?@  C�~�                                    Bxl.��  �          @��@S33�H��?�  At��C�:�@S33�U�   �ÅC�XR                                    Bxl.�:  �          @��@X���J�H?k�A3�C�p�@X���L�ͿL���(�C�K�                                    Bxl.��  �          @�
=@\(��E?aG�A,��C��@\(��G
=�L����
C��f                                    Bxl.Ն  �          @�ff@mp��(��?���AT(�C�L�@mp��4z������ffC�e                                    Bxl.�,  �          @��@~�R���?0��AQ�C��@~�R�=q������C�c�                                    Bxl.��  �          @��R@�{�����
�}p�C���@�{�������`��C�q�                                    Bxl/x  �          @�\)?L���z�H@(�A�33C��f?L�����
=�?\C��                                    Bxl/  �          @�
=?���{�@�RA�p�C�ff?�����׽�\)�\(�C�@                                     Bxl/�  �          @��?p����(�@ffA�G�C�l�?p����(���\)�VffC��H                                    Bxl/-j  �          @�
=?������?��A�
=C��3?�����p���=q�Tz�C���                                    Bxl/<  T          @��@��tz�?���A]p�C�E@��u��G��QG�C�8R                                    Bxl/J�  �          @��@#�
�g�?�  A|��C��=@#�
�p  �@  ��RC�+�                                    Bxl/Y\  �          @�p�@/\)�fff?��AY�C���@/\)�i���c�
�/�C�s3                                    Bxl/h  �          @��R?fff��33���Ϳ��HC��?fff�z=q��H���HC��H                                    Bxl/v�  �          @�
=?�{��Q�>u@8��C�` ?�{��  ���=qC�]q                                    Bxl/�N  �          @��R?У���(��L�Ϳz�C�:�?У��p  �G���Q�C�Ǯ                                    Bxl/��  T          @�z�>����Z�H������  C�]q>����   �U�j{C��{                                    Bxl/��  �          @�z����:�H�aG��H\)C����녿8Q�����G�Cp\                                    Bxl/�@  �          @��?B�\�w�������C���?B�\�
=��G��s�C���                                    Bxl/��  �          @�ff>����U�QG��0�RC�s3>��Ϳ��H�����\C�E                                    Bxl/Ό  �          @�{>B�\�qG��0  �Q�C�w
>B�\����G�ffC���                                    Bxl/�2  �          @�p�<���ff��Q���{C�/\<��'��s�
�]��C�K�                                    Bxl/��  �          @�z������p���p��xz�C�aH�����J�H�Tz��9{C�#�                                    Bxl/�~  T          @��ͽL����=q��
=��Q�C��׽L���6ff�j=q�P33C�t{                                    Bxl0	$  T          @��
���
��  �\)��{C�k����
��\�~{�p(�C��q                                    Bxl0�  �          @�(�����`  �>{���C��H��Ϳ�p����H�HCy��                                    Bxl0&p  �          @���u�c�
�=p��{C����u������
.C��q                                    Bxl05  �          @��?�R�s�
�!G���\C���?�R��p����
�}{C���                                    Bxl0C�  �          @��R?5�x���!G����C�&f?5�33����y��C��f                                    Bxl0Rb  �          @�?�Q��p  �(����\C�#�?�Q���s�
�[{C�}q                                    Bxl0a  �          @��;����w
=��� �\C��׾�����
��=q�{�\C���                                    Bxl0o�  �          @�����g
=�8����C�ý���{���H(�C��{                                    Bxl0~T  �          @�p�>#�
�s33�)����C�7
>#�
��33��\)�{C�o\                                    Bxl0��  �          @�
=>B�\�6ff�]p��IC��)>B�\�(����¢Q�C��=                                    Bxl0��  �          @����\)���R������C�����\)>��H��{§�HB�#�                                    Bxl0�F  T          @�\)>�\)�Fff�aG��A�
C���>�\)�Y�����#�C�
=                                    Bxl0��  T          @��R>L���H���_\)�@{C���>L�Ϳc�
����C�w
                                    Bxl0ǒ  T          @�?Q녿�Q���=q�z�C���?Q�>�G����z�A��                                    Bxl0�8  �          @�?Ǯ�����(��x(�C��R?Ǯ?O\)���\�RA�z�                                    Bxl0��  �          @�?�
=���w
=�^��C��?�
=>����33ffA��                                    Bxl0�  �          @�?���{�p  �Z
=C��3?�>�33���HA#33                                    Bxl1*  �          @�p�@����
�aG��O33C�H@�>����{��u�@��R                                    Bxl1�  �          @�p�?�33�:=q�\(��>�HC���?�33�5��ff(�C�<)                                    Bxl1v  �          @�(�?Ǯ�&ff�_\)�D=qC�xR?Ǯ������\33C���                                    Bxl1.  T          @�(�?�p��5�Z�H�?��C��)?�p��&ff������C��                                    Bxl1<�  �          @�(�@\)�XQ��{����C�8R@\)��
�HQ��0(�C�AH                                    Bxl1Kh  �          @��H?˅���
��\)�c\)C���?˅�<(��E�+  C�5�                                    Bxl1Z  �          @�33?��\��=q�Tz��'�C�.?��\�P���<��� �C��                                     Bxl1h�  �          @��?�����
=��\)�b�RC��?����AG��I���-C�H�                                    Bxl1wZ  �          @��\?�����ÿ�33��=qC��q?���%�`���L�\C���                                    Bxl1�   �          @�G�?=p������G��Z�\C�3?=p��>�R�@  �1Q�C���                                    Bxl1��  �          @�33?s33�l������ �C�4{?s33��z��~{�w{C�8R                                    Bxl1�L  �          @��\?W
=�w�������  C�R?W
=��\�k��a��C��                                    Bxl1��  �          @�z�?У�����=�?\C�?У��fff��
���
C�4{                                    Bxl1��  �          @��?�����ff>�ff@�z�C��?����tz��ff��C���                                    Bxl1�>  �          @�z�?�������>k�@<(�C�Ǯ?����qG������C�޸                                    Bxl1��  �          @�z�?�\)��33?G�A33C���?�\)���������(�C�/\                                    Bxl1�  �          @�(�?���{?B�\A  C�Y�?��~{���
��(�C��3                                    Bxl1�0  �          @���?�=q�~�R?�  A��HC�T{?�=q��p��L���\)C�Ф                                    Bxl2	�  �          @��?n{�p��@{B�C��?n{��  >\)?�ffC���                                    Bxl2|  �          @�  @��[�������C��
@���R�:�H�.�HC�p�                                    Bxl2'"  �          @�z�@��!��'
=�p�C��
@��J=q�b�\�X��C��)                                    Bxl25�  �          @�(�?�{�s33������G�C�t{?�{��R�N�R�>{C�w
                                    Bxl2Dn  �          @��?���Vff� �����HC���?�׿�ff�]p��T=qC��                                    Bxl2S  �          @��H@�
�G����مC��H@�
��33�QG��L�C��=                                    Bxl2a�  �          @��
@Q��333��R��p�C�,�@Q쿜(��W
=�M��C�g�                                    Bxl2p`  �          @��H@1G��K�?Q�A/�C���@1G��H�ÿxQ��P(�C��q                                    Bxl2  �          @�p�@,���:=q?�{A�z�C�o\@,���]p�=u?@  C�f                                    Bxl2��  �          @���@S�
�.�R?�\)A��C�AH@S�
�B�\��=q�W�C��R                                    Bxl2�R  �          @�G�@QG����ÿ�ff��C�y�@QG�����\)�(�C���                                    Bxl2��  �          @���@.{�aG��Z�H�H��C�\@.{?�ff�W��DA���                                    Bxl2��  �          @���@=q�1G��)���=qC���@=q�xQ��mp��[��C�\                                   Bxl2�D  �          @��@1G���\)�K��.z�C��q@1G�=��
�l(��Tp�?�
=                                   Bxl2��  �          @��\@'����H�U�;ffC�e@'�>�z��o\)�[\)@�                                    Bxl2�  �          @��
@<(���R�p����C��@<(��O\)�Y���@ffC�B�                                    Bxl2�6  �          @�=q@	���?\)�#33�
�
C��R@	�������o\)�b��C�`                                     Bxl3�  �          @��@ �׿����p���d�RC�{@ ��?Tz��z=q�sz�A���                                    Bxl3�  �          @�  @6ff��p��3�
�\)C��q@6ff�L���[��I{C��q                                    Bxl3 (  �          @��R@P  ?��H������
A��
@P  @\)�W
=�@Q�B
Q�                                    Bxl3.�  �          @��@c33?�����H�хA�ff@c33@#33�8Q��p�B��                                    Bxl3=t  �          @�Q�@r�\>�  �{��{@r�\@r�\?��׿�  ��A�{                                    Bxl3L  T          @�@0  ?�\�HQ��A�A'
=@0  @�����=qB�\                                    Bxl3Z�  �          @�p�?����\�Q��b�C��R?�?\(��U��f��A�                                      Bxl3if  �          @��R?�������u��m�C�/\?��>�����R�3A��R                                    Bxl3x  �          @��?�\�����(�C���?�\?n{��ff�Bt                                    Bxl3��  �          @��?�(��.�R�H���9z�C��?�(��+������C���                                    Bxl3�X  
�          @��
?����H�������C�AH?��ÿ\�`���h�RC���                                    Bxl3��  �          @�G�@333���z����C�{@333��R�G��=p�C��R                                    Bxl3��  �          @���@
=q���\�[��WG�C�@
=q?:�H�e�f{A��                                    Bxl3�J  @          @��H@!G���33�-p��"C�~�@!G��8Q��S33�R�C��
                                    Bxl3��  �          @�(�@��333�����C��@���33�]p��S{C�
                                    Bxl3ޖ  �          @���?�  ���R�\(��QffC���?�  >��~{L�@�z�                                    Bxl3�<  T          @�ff?��Ϳ������3C���?���?�  ���W
BBQ�                                    Bxl3��  �          @��R?�����\)�+�
C�@ ?���
=q�>�R��C��R                                    Bxl4
�  �          @��@\)�s�
����W
=C�33@\)�*�H�:=q��C��3                                    Bxl4.  �          @�G�@
=q�mp�������C�"�@
=q����J�H�2
=C��                                    Bxl4'�  �          @�  @{�^{������ffC��)@{�
�H�Dz��+�HC�T{                                    Bxl46z  �          @�G�@*�H�aG����
�R�RC���@*�H�(��/\)�C��f                                    Bxl4E   
�          @���@3�
�aG���{�]�C�T{@3�
����333��HC��                                    Bxl4S�  
�          @��R@>�R�_\)�s33�:=qC�:�@>�R�p��)���	�HC�8R                                    Bxl4bl  T          @�\)@:�H�b�\�����W�
C���@:�H�=q�3�
�(�C�0�                                    Bxl4q  �          @��H@�\�aG��%���C��@�\��{����e��C��{                                    Bxl4�  �          @�(�@���tz��G���Q�C��@���33�`���:=qC�
                                    Bxl4�^  �          @�  @<(��k������C�L�@<(��\)�W��)ffC�]q                                    Bxl4�  �          @�\)@@���vff�G���C���@@���6ff�-p���RC�H�                                    Bxl4��  �          @��R@E�j=q����D��C��@E�"�\�5���
C�J=                                    Bxl4�P  �          @�@Vff�X�ÿ�  �j�RC�U�@Vff�p��6ff�C�L�                                    Bxl4��  �          @�{@\(��W
=��ff�D��C��q@\(���\�*�H��C�4{                                    Bxl4ל  �          @���@^�R�c33�(����C�/\@^�R�*=q�����  C�L�                                    Bxl4�B  �          @���@Q��l(��c�
�"�\C���@Q��*=q�-p����C��H                                    Bxl4��  T          @�Q�@p���H�ý�Q쿊=qC��@p���'
=��  ����C��H                                    Bxl5�  �          @���@�(��\)?�G�A9p�C�u�@�(�����{�uC�y�                                    Bxl54  �          @��R@����?��A��C�t{@���0��?   @�  C��H                                    Bxl5 �  T          @��@s33��
=@8��B�HC�c�@s33�!�?���A�C�'�                                    Bxl5/�  �          @�  @Y���ff@A�B�C�,�@Y���X��?���A�(�C���                                    Bxl5>&  �          @�33@P  �33@7�B�C���@P  �P��?��A�C�xR                                    Bxl5L�  �          @�33@L���p�@9��B�HC���@L���Z�H?��
AtQ�C���                                    Bxl5[r  �          @��H@333�\)@B�\B{C�/\@333�n{?�  Ao�C�y�                                    Bxl5j  �          @�{?��
��Q�@w�Bf=qC�=q?��
�_\)@�A�p�C��f                                    Bxl5x�  �          @���?���(�@}p�BtffC��?��H��@,��B�C��)                                    Bxl5�d  �          @���?�p���p�@j=qBY(�C�aH?�p��Z�H@
�HA�RC��                                    Bxl5�
  �          @��?c�
��Q�@�p�B��)C��q?c�
�\(�@0��B�\C�@                                     Bxl5��  T          @��R?fff�   @�(�Bx�C�"�?fff�y��@�A�z�C��                                    Bxl5�V  �          @���?���Y��@C�
B"ffC��R?����Q�?G�A(�C���                                    Bxl5��  �          @�G�?Ǯ��H@hQ�BNz�C�` ?Ǯ�~�R?��
A��C���                                    Bxl5Т  T          @���?Q녿�Q�@���B
=C�e?Q��z�H@%�B33C��H                                    Bxl5�H  �          @��R?�
=�z�@��Bn33C���?�
=�z�H@�A�
=C�^�                                    Bxl5��  �          @��?Y���H��@Y��B9  C���?Y����
=?���Al��C�b�                                    Bxl5��  �          @�=q?c�
�B�\@eBB\)C��?c�
���?�A��C���                                    Bxl6:  �          @��
?�G��$z�@\)B]\)C���?�G�����@   A�\)C��
                                    Bxl6�  �          @��\?L���
=@�(�BkC�e?L����ff@\)A�33C�g�                                    Bxl6(�  �          @��\?W
=��R@�ffBq��C�C�?W
=��(�@
=A�  C���                                    Bxl67,  �          @���?aG�� ��@��RBy�
C���?aG��|��@\)A���C�K�                                    Bxl6E�  �          @���?�ff�}p�@��RB�ǮC�h�?�ff�Mp�@O\)B,ffC��                                    Bxl6Tx  �          @�33?��
��(�@�(�B���C��?��
�r�\@2�\BffC��H                                    Bxl6c  �          @���=L���6ff@{�BW�HC���=L����Q�?�ffA�C�S3                                    Bxl6q�  �          @�ff=u�J=q@r�\BH�RC���=u��ff?\A�\)C�e                                    Bxl6�j  �          @�p�>aG��g
=@Tz�B*p�C�� >aG���=q?c�
A&=qC�O\                                    Bxl6�  
�          @��>�{�?\)@l��BK\)C�7
>�{��Q�?��
A���C�#�                                    Bxl6��  �          @�p�?+��1G�@{�BX{C���?+���{?�A�Q�C�E                                    Bxl6�\  �          @�ff?���O\)@dz�B9�\C��H?����z�?��ArffC�K�                                    Bxl6�  �          @�ff?333�Z=q@`��B4�
C��=?333��Q�?��AV=qC�.                                    Bxl6ɨ  �          @�ff?=p��`  @Y��B.=qC��?=p�����?�G�A<(�C�aH                                    Bxl6�N  �          @���?+��a�@U�B+33C�Y�?+�����?n{A.�RC��q                                    Bxl6��  �          @��?.{�"�\@\)Bbz�C�w
?.{����@G�AǮC��                                     Bxl6��  �          @�=q?�33��
=@��BxQ�C�T{?�33�y��@#33B �C�0�                                    Bxl7@  �          @���>��W
=@��
B���C�:�>��K�@]p�B<p�C�{                                    Bxl7�  �          @�z�=��
��\@���B�B�C�*==��
��=q@(Q�BffC���                                    Bxl7!�  �          @�\)�z��e�@W
=B+(�C�g��z���=q?k�A,z�C��3                                    Bxl702  �          @����������@*�HB��C~Ǯ������33=���?���C���                                    Bxl7>�  �          @�
=��p��z�H@1G�B(�C|�쿝p���G�>�  @:=qC�)                                    Bxl7M~  
�          @�\)��
=���
@
=A��
Cz޸��
=����L����C}@                                     Bxl7\$  �          @�{��
=�|��@(Q�B��C}ff��
=���=�G�?��\C�f                                    Bxl7j�  �          @�
=�}p�����?��A���C��Ϳ}p����ÿh���(Q�C��                                    Bxl7yp  �          @��ÿ:�H��G�?��RA�
=C�q�:�H���Ϳ@  ���C��f                                    Bxl7�  �          @�Q��R���?�(�A�z�C�AH��R��G�����o�C�K�                                    Bxl7��  �          @��R�\)�{�@%�B��C��þ\)��ff=��
?h��C�!H                                    Bxl7�b  �          @�  �8Q��^�R@W
=B0{C���8Q���\)?}p�A<  C���                                    Bxl7�  �          @��H�\)����?ٙ�A��\C�#׾\)��ff��\)�K�C�+�                                    Bxl7®  �          @��ý������R?ٙ�A���C�]q������zῊ=q�G�C�c�                                    Bxl7�T  �          @��R<#�
���
?�(�A�G�C��<#�
��=q���\�>=qC��                                    Bxl7��  �          @�  ?!G�����?��HAb=qC���?!G���{���
���C���                                    Bxl7�  �          @��þ��
���>��
@j�HC�+����
��z������z�C��                                    Bxl7�F  �          @�G�������  ��p���
=C�g����������>�R�
=C�B�                                    Bxl8�  @          @�\)���R���
�B�\��RC�*=���R��Q��0���	�
C��f                                    Bxl8�  6          @�ff�0����33?
=q@�\)C���0����z�������
C��                                    Bxl8)8  "          @�\)�.{���
�\)��G�C�
=�.{�tz��E���
C��
                                    Bxl87�  "          @�  ����G��:�H�ffC�#׿��j=q�J�H�G�C|G�                                    Bxl8F�  S          @�p������p��z�H�8��C��ÿ���Z�H�Tz��+Q�C|�                                     Bxl8U*  
�          @�z�
=q���þ��
�vffC�� �
=q�w��5����C���                                    Bxl8c�  T          @��(���Q�}p��:�RC�W
�(��`  �XQ��.�C�
=                                    Bxl8rv  �          @�
=�����Ϳ�
=���C�/\���A��y���O33C��{                                    Bxl8�  T          @��>\)��=q>W
=@(Q�C�޸>\)�~�R������C�                                      Bxl8��  
Z          @��
?��
��p�@ffA���C���?��
�����p���ffC���                                    Bxl8�h  A          @�33?E���{@Q�A�=qC��?E����׾�Q���33C�^�                                    Bxl8�  	A          @���>L����33?�
=AY�C�1�>L����\)������
=C�9�                                    Bxl8��  �          @�G�<#�
����?���AJ{C��<#�
��\)��Q����C��                                    Bxl8�Z  �          @��\?   ��(�?�G�Af�\C��?   �����������C���                                    Bxl8�   �          @��R?J=q���
@.�RA���C��?J=q���ͼ��
���C�W
                                    Bxl8�  T          @�33?fff�u@L��Bz�C���?fff��ff?(��@�C�&f                                    Bxl8�L  "          @�33?�G��G�@y��BG�C���?�G���\)?У�A�\)C�\                                    Bxl9�  "          @�=q?���O\)@g�B7��C�:�?����?�=qAt��C�                                      Bxl9�  "          @���?�z��I��@qG�BA\)C�#�?�z���p�?�G�A�G�C��)                                    Bxl9">  �          @�33?�G��[�@h��B633C�.?�G����H?�  Ac33C�޸                                    Bxl90�  �          @��R?�G��%�@���B^z�C���?�G����H@33A�p�C���                                    Bxl9?�  �          @�G�?���C33@`��B1p�C���?����ff?��Aw33C�l�                                    Bxl9N0  �          @��?.{�Q�@�{Bm�C�H?.{��  @�A��HC���                                    Bxl9\�  �          @�Q�?^�R�33@�
=B~=qC�}q?^�R���
@,(�Bp�C��{                                    Bxl9k|  "          @���?E��33@��B�
=C�U�?E���(�@-p�BG�C�L�                                    Bxl9z"  
�          @���?��
=@���B�p�C��=?���ff@,��B33C���                                    Bxl9��  
�          @���?���(Q�@z�HBP=qC�.?����=q?�z�A��HC��
                                    Bxl9�n  �          @���@%�0��@G�B(�C��3@%��  ?�A\��C�s3                                    Bxl9�  "          @�33?�z��[�@^{B/Q�C�U�?�z���  ?�{AM�C�ٚ                                    Bxl9��  �          @�  ?���{�@O\)B�C�R?������?&ff@�z�C�]q                                    Bxl9�`  T          @�Q�@ff���H?�A�33C�4{@ff��z�G��z�C�l�                                    Bxl9�  
(          @�ff@ff���R?��HAX��C��@ff��p������p��C��)                                    Bxl9�  "          @�{@���Q�?�Q�A�
=C�]q@����H��Q��T(�C�1�                                    Bxl9�R  T          @��@����>��?�z�C�|)@���tz���R��\)C�S3                                    Bxl9��  
Z          @���@*=q���R�����H  C�+�@*=q�>{�J�H���C��                                    Bxl:�  T          @��@�R���\�(���{C���@�R�U�4z��p�C�Ff                                    Bxl:D  �          @��H@   �vff�%��
=C��)@   �����  �dp�C�ff                                    Bxl:)�  "          @�33?����xQ�?k�AL(�C�!H?����qG��������C�Y�                                    Bxl:8�  "          @��?���z=q@Mp�B�C�H�?����Q�?#�
@�G�C�T{                                    Bxl:G6  T          @�Q�?��\���H@FffB{C�  ?��\���>�ff@�(�C��                                    Bxl:U�  
�          @�Q�?�33��Q�@0  A�Q�C�q?�33��=q=�\)?:�HC���                                    Bxl:d�  �          @���?У���=q@ ��A�ffC�T{?У���  �.{��C��                                    Bxl:s(  �          @�G�@����\)@�
A�  C�t{@����녾�z��G�C�                                    Bxl:��  T          @��H@2�\��{?ٙ�A��C��@2�\���R�G���C�H                                    Bxl:�t  �          @��H@z���p�?�G�A�  C���@z���Q쿙���P  C���                                    Bxl:�  s          @�33@(������?��\A0  C�"�@(����(������(�C��\                                    