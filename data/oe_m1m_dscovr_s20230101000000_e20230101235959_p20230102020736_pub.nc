CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230101000000_e20230101235959_p20230102020736_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-02T02:07:36.597Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-01T00:00:00.000Z   time_coverage_end         2023-01-01T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill         �   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxV��  
�          @���,��@��\�\�\)B�(��,��@�녾�{�b�\B�3                                    BxV�&  T          @���*�H@��׿�=q�3\)B�ff�*�H@�z�>�?�=qB�33                                    BxV�)�  T          @�  ��
=@���1���Q�B�uÿ�
=@�33������=qB�k�                                    BxV�8r  
�          @�{�%�@��������\B�ff�%�@���?=p�@���B�                                    BxV�G  �          @�p��5@��H�aG��p�B����5@��?z�HA!p�B��
                                    BxV�U�  
�          @��\�z�@�
=?��A_
=B�Q��z�@�Q�@�A�
=B�.                                    BxV�dd  T          @�
=���@�33�8Q����B����@��
?\)@���B��                                    BxV�s
  �          @���?���@��R�dz���B��=?���@��R�����B�                                    BxV���  �          @��Ϳ���@��>��
@O\)B��H����@�  ?���A�z�Bޮ                                    BxV��V  )          @�
=��G�@�z��  �x  B�=q��G�@�33������B��f                                    BxV���  M          @��>��@���)���ܣ�B�8R>��@��������U�B��q                                    BxV���  
�          @�{�W
=@��\��{���HB�녿W
=@�녾�  �#�
B�.                                    BxV��H  
�          @��?\@�Q�=��
?B�\B��?\@�=q?�
=Ag
=B�\                                    BxV���  �          @��\?�@����z����B���?�@�z�Tz��
ffB�L�                                    BxV�ٔ  T          @���?�@���n{�ffB�u�?�@�p�>��@�
=B��\                                    BxV��:  �          @��H����@�33��p��m��B�#׽���@�G���\)�0��B�{                                    BxV���  
�          @�G�>�
=@+��,(��3{B�G�>�
=@P�׿��H��{B�aH                                    BxV��  
�          @����fff@�zῳ33�eB�
=�fff@�=q�#�
����B�p�                                    BxV�,  T          @��
��R@�ff�x���)�
B���R@����/\)�㙚B�\                                    BxV�"�  �          @���>k�@E�dz��DG�B�k�>k�@w��-p����B�#�                                    BxV�1x  
Z          @��R?
=@�
=�333�陚B���?
=@�\)?&ff@�
=B���                                    BxV�@  "          @��Ϳ   @�33��\)�333B�B��   @�ff?�ffAUG�B��=                                    BxV�N�  
�          @�{�333@�������9�B�p��333@�33>u@��B�(�                                    BxV�]j  
Z          @�{���@�p�� ����Q�B�녿��@�  �0������B�u�                                    BxV�l  
(          @�Q�
=@�������B�z�
=@���ff���RB��)                                    BxV�z�  
�          @��þǮ@��R��ff�}B�G��Ǯ@�p��8Q����B���                                    BxV��\  
Z          @�  ����@�z�˅��
=B�\����@��
�k��Q�B�Ǯ                                    BxV��  
�          @�=q?�=q@�ff�����Tz�B���?�=q@��<�>�33B�ff                                    BxV���  
�          @��H?�z�@�G����R����B�=q?�z�@���&ff��Q�B���                                    BxV��N  	`          @���?W
=@��H��  �H  B��)?W
=@�
=>�?�=qB�G�                                    BxV���  
�          @�=q?s33@�{�����RB���?s33@���?\(�A��B��
                                    BxV�Қ  
�          @��R?!G�@��Tz��	�B��3?!G�@�
=?�\@��B���                                    BxV��@  "          @��
���\@5�?˅A���CE���\@
=@�RAʏ\C�3                                    BxV���  T          @�p��e�G�@-p�B\)CTW
�e�7
=@A��
CZ�=                                    BxV���  T          @��R�O\)�1G�@<��B
��C\��O\)�X��@��A��Cb^�                                    BxV�2  T          @�  �����c�
@5�B\)C?�f�������@"�\A�z�CG�R                                    BxV��  T          @��R��=q��@I��B{C;aH��=q���\@;�B
�CEL�                                    BxV�*~  
�          @�����=�\)@,(�A�G�C3{���Ϳz�@(Q�A���C;p�                                    BxV�9$  �          @�
=��\)?   @�A�ffC.���\)��@�A��C4Q�                                    BxV�G�  T          @��R���?��?�z�A�G�C$:����?fff?�
=A��HC)8R                                    BxV�Vp  "          @�Q����@2�\�#�
����C�)���@-p�?(��@�ffC\)                                    BxV�e  
�          @�{�r�\>�?�\)A��RC-#��r�\>8Q�?���A��C1Y�                                    BxV�s�  T          @������\���@.�RA�(�CLٚ���\��R@��A�  CSY�                                    BxV��b  "          @�Q�����s33@*=qA�(�C@W
����Ǯ@
=Aי�CG��                                    BxV��  )          @�\)��33�#�
@G�B�
C<����33��{@8Q�B��CFY�                                    BxV���  
�          @�������Q�@4z�B�
C5.���B�\@.{A�33C=��                                    BxV��T  �          @�����녾u@#�
A�{C6�q��녿W
=@�A�33C>h�                                    BxV���  "          @���w
=��G�@J=qB�\CL���w
=�p�@)��A��\CT�                                     BxV�ˠ  �          @�ff�|�Ϳ�  @333BCK��|���
=@�
A��HCR��                                    BxV��F  �          @���{�c�
@
=A�Q�C?aH��{��
=@A���CE�\                                    BxV���  
�          @�ff����@<�ͼ#�
�\)C�3����@7�?.{@�(�Cs3                                    BxV���  �          @������\@L�ͿE���RC����\@R�\<#�
=�\)C&f                                    BxV�8  
�          @������?�33?#�
@���CW
����?�Q�?���AF�RC�\                                    BxV��  "          @�(��i��@S�
��=q�p��C	�
�i��@a녿�\��  C�                                    BxV�#�  
(          @�33�>{@w
=����ffB���>{@��׿��H�Qp�B��R                                    BxV�2*  �          @���*�H@xQ��*=q��B�{�*�H@�z����ffB�{                                    BxV�@�  �          @�ff�G�@~{�B�\��\B�{�G�@�=q�����B��)                                    BxV�Ov  �          @��R���H@����<(����B�z���H@�
=�������B�                                      BxV�^  T          @�zῚ�H@~{�Vff��B����H@�z�����33B�Q�                                    BxV�l�  �          @���   @���(�����B�z�   @�33���R��  B�L�                                    BxV�{h  T          @�Q쾨��@��
>�(�@�\)B�Q쾨��@�(�?���A�33B��                                    BxV��  �          @�  �z�H@�G��Q����B�{�z�H@��H>�p�@��\B��
                                    BxV���  �          @��H�G�@�G���\)�q�Bř��G�@�
=�.{����B�                                      BxV��Z  "          @�=q�z�@��Ϳ������B�k��z�@�ff�0������B���                                    BxV��   
(          @�녾��@�zῥ��`z�B�Ǯ���@����u�.{B���                                    BxV�Ħ  "          @�(��8Q�@�z�J=q�
{B�녿8Q�@�>�
=@��HB���                                    BxV��L  �          @���	��@���?+�@��
B�=�	��@���?ٙ�A���B�\)                                    BxV���  T          @����g�@$z�@G�A�=qC���g�?�p�@333Bz�CW
                                    BxV��  �          @���XQ�?333@.�RB��C(W
�XQ�=L��@4z�Bp�C3#�                                    BxV��>  )          @���n�R?��
@>{B(�C$�=�n�R>��
@HQ�B��C/\                                    BxV��  
�          @�Q��qG�>��@L��B ��C,�
�qG���=q@N�RB"{C8#�                                    BxV��  �          @�\)�e��\@ffA�\)CK��e����H?ٙ�A�
=CP��                                    BxV�+0  �          @����P���XQ�?޸RA���Cb
=�P���k�?k�A*=qCd��                                    BxV�9�  T          @�  �dz��Q�@(�A���CU���dz��7�?�=qA�ffCZ޸                                    BxV�H|  T          @�\)�z=q�33@G�A��
CRz��z=q�,��?�Q�A�
=CV�H                                    BxV�W"  "          @��������?��HA�ffCR:�����,(�?���AQ��CU�\                                    BxV�e�  )          @����l���Fff?��As\)C\
=�l���Tz�?�@��HC]�3                                    BxV�tn  �          @����a��QG�?��Ayp�C^���a��_\)?�@���C`��                                    BxV��  �          @�G��^{�L��?�{A�Q�C^�q�^{�^�R?W
=A\)Ca!H                                    BxV���  �          @�Q������\)?��\Ak�CO�)�����{?:�HA�HCQ��                                    BxV��`  �          @�
=������?��HA��CR������*�H?aG�A$��CUY�                                    BxV��  �          @���vff�7
=?��At��CX���vff�E�?#�
@�CZ��                                    BxV���  �          @����`  �c33�����s33Caff�`  �XQ쿏\)�N�HC`
=                                    BxV��R  T          @��H�aG��W���Q��\Q�C_��aG��AG������
C\��                                    BxV���  
�          @�G����H�����
�c�
CP33���H�  �
=��COk�                                    BxV��  �          @�z���Q�\?}p�A;�CF�H��Q�ٙ�?�R@��
CH�                                    BxV��D  
�          @��\��  ��Q�?�  A��C8aH��  �=p�?У�A�C<��                                    BxV��  �          @�33���>�\)@p�AΣ�C0� ��녾aG�@{A�G�C6�                                     BxV��  "          @�{��(�?�ff@��A�  C%����(�>��H@#�
A��RC-8R                                    BxV�$6  �          @��
�z�H?Tz�@,(�B�\C()�z�H>W
=@3�
B=qC0�                                    BxV�2�  "          @��
��Q�?h��?�(�A�=qC(����Q�?\)?��A��RC,�)                                    BxV�A�  �          @��\���R?��H?��AHz�C!�����R?���?��A�p�C$��                                    BxV�P(  
�          @�=q��{?Ǯ?xQ�A:�RC ����{?��?��Ay��C#�=                                    BxV�^�  �          @�����H@p��.{� ��C�����H@��>��
@p  C(�                                    BxV�mt  �          @�ff���?���=q�FffC ����?�Q�=�G�?��RC�f                                    BxV�|  S          @�����
=?�\)�8Q���C&�H��
=?���=�\)?^�RC&�                                     BxV���  �          @�33��G�?�{�Ǯ����C&@ ��G�?��\)�ٙ�C%��                                    BxV��f  �          @�z��vff@<��<�>�33C�{�vff@7�?+�@�{CO\                                    BxV��  �          @��\��{?Y��>���@�Q�C)Ǯ��{?@  ?�\@�(�C*�H                                    BxV���  �          @���y��@!녿333�	C��y��@'��\)�ٙ�C�                                    BxV��X  �          @�{�w�?�33��G���=qC�{�w�@�����W
=C��                                    BxV���  �          @���dz�@>�R?�\@ǮC0��dz�@333?��Aap�C�                                    BxV��  �          @�p����\@)�����
�B�\C�����\@%?��@�G�C��                                    BxV��J  �          @��R���?У׿L����HC Y����?�G���G����HC�)                                    BxV���  �          @������?h�ÿ�{��C(�R����?�
=�����T(�C%�\                                    BxV��  �          @����=q?.{�p���C*� ��=q?��� �����C$aH                                    BxV�<  "          @�z�?�
=@.�R�g��A�RBiff?�
=@\(��<���z�B�R                                    BxV�+�  �          @�zῳ33?\��z���C����33@�R���R�_�B�{                                    BxV�:�  "          @�{�<�ͽu��\)�\\)C5��<��?Y�������U�C$
=                                    BxV�I.  �          @�\)�333��R�s33�:��CZz��333��\)���U�CN
                                    BxV�W�  �          @����7��
=q�u�;��CX�q�7���ff���R�T�RCLc�                                    BxV�fz  �          @�Q��{���  �Fp�C_���{���������c��CRh�                                    BxV�u   �          @��������R����UC[Y����������offCK�                                     BxV���  �          @�Q��
=��
��\)�U��C]��
=��\)����p��CMp�                                    BxV��l  �          @�G��
=��(���p��cQ�C_)�
=��G���\)�~�CM�                                     BxV��  �          @�ff�:=q�Q��j=q�5p�CX.�:=q���������M��CLc�                                    BxV���  �          @���O\)�W
=��\)���Cb
=�O\)�8���!����C]�=                                    BxV��^  T          @�z��O\)�:=q�'
=���C]��O\)�33�J=q��CWff                                    BxV��  �          @���R�\���H�S33�"�
CR�=�R�\���R�hQ��7�RCH��                                    BxV�۪  T          @�G��p  >L���:�H��C0���p  ?L���4z��=qC'��                                    BxV��P  �          @������?�p��(Q����\C#+����?޸R�z���C�                                     BxV���  �          @��H�z=q>�Q��K��  C.�z=q?�  �B�\��
C%�
                                    BxV��  �          @�Q���Q쾀  �;��ffC7����Q�>�Q��:�H�C.��                                    BxV�B  T          @�����  =#�
�����z�C3� ��  >��H�	������C-��                                    BxV�$�  �          @�p����ü��
�5�(�C4@ ����?\)�1��(�C,�=                                    BxV�3�  T          @�����  <#�
�����C3ٚ��  >������=qC.8R                                    BxV�B4  
�          @������?#�
�ff��
=C,�����?����
=��(�C'0�                                    BxV�P�  �          @�z���=q?Tz�����
=C)� ��=q?�  ��(���Q�C$�3                                    BxV�_�  T          @�\)��@~�R������RB�k���@�p���\��33B�                                    BxV�n&  �          @���@33@�{��33�xQ�Bff@33@�p�?��@�G�B~�                                    BxV�|�  �          @��
?�  @�
=��ff��(�B�  ?�  @�����B�                                    BxV��r  �          @�z�#�
@��ÿ��H��(�B��#�
@�\)����\)B�Ǯ                                    BxV��  �          @�����
@�녽#�
��\B��
���
@�
=?k�A)B�aH                                    BxV���  �          @�녿�33@��>8Q�@�\B�uÿ�33@���?���AJffBᙚ                                    BxV��d  T          @�=q���@�{��G���
=Bݳ3���@�?   @�\)B�                                    BxV��
  �          @�33����@����O\)��
Bӊ=����@��>��?�  B�#�                                    BxV�԰  �          @�p��I��@4z��G���G�C
!H�I��@C�
�k��=�CǮ                                    BxV��V  T          @���z῜(���Q���33CD^���z�\(���33��G�C?��                                    BxV���  �          @�(���{��p�����ffCM:���{��z���
��z�CI�{                                    BxV� �  �          @�z��~�R�-p��W
=�{CV5��~�R�\)������z�CT�                                    BxV�H  �          @��
�\)���?���AO
=CR�q�\)�#�
?z�@߮CT�                                    BxV��  T          @�z������ÿ���  CNs3��녿��R�xQ��9CL��                                    BxV�,�  T          @���p������
����CB�=��p��Y����(����C>޸                                    BxV�;:  T          @�{�}p�?���#33��  C!��}p�?�G��  ���C                                    BxV�I�  �          @�33�dz�?Ǯ�0���p�Cff�dz�@�
�=q���C��                                    BxV�X�  �          @�녿�(�@Q��w��M��C����(�@E��U�)�RB�=q                                    BxV�g,  �          @�ff�;�?^�R�l���J  C#xR�;�?Ǯ�^{�9{C��                                    BxV�u�  T          @�ff�\)��G������pffCLJ=�\)�8Q���z��{�\C8��                                    BxV��x  �          @���\�:�H�����r�RCE����\=������H�x��C1}q                                    BxV��  T          @�z���?���
=�s\)C%޸��?�{�����bffC!H                                    BxV���  �          @�(����u���H�iz�CJ^����.{��ff�s��C8#�                                    BxV��j  �          @�z��7
=�����c�
�Dz�CI�)�7
=��G��mp��O��C<��                                    BxV��  �          @�z��c�
�\�Mp��'ffC:#��c�
>k��N{�(\)C0\)                                    BxV�Ͷ  �          @�(��j�H�u�@����
CB�
�j�H�\�H���!�C9�{                                    BxV��\  �          @�
=�=q@s�
?��HAp��B����=q@aG�?�33A�Q�B��)                                    BxV��  �          @����(��@q�?h��A2�RB����(��@b�\?���A���B�u�                                    BxV���  �          @�\)���@l(�?��RA�ffB�(����@Vff@	��Aڏ\B�W
                                    BxV�N  �          @����H@P  @ ��A��B�aH��H@4z�@$z�B
�
C�H                                    BxV��  �          @����S�
@<��?8Q�A33C
O\�S�
@1G�?�  A��C!H                                    BxV�%�  T          @��ÿ�(�@{�?�  AMp�B�G���(�@j�H?ٙ�A��
B�\)                                    BxV�4@  �          @�33�5�@4z��ff��33C)�5�@7
==��
?�{C�                                    BxV�B�  �          @�p��{�>��R�=q����C/z��{�?=p���
��ffC)E                                    BxV�Q�  �          @��
��\)?��
���ǮC'{��\)?��\>#�
?�Q�C'8R                                    BxV�`2  T          @�����33?�p��!G�����C!8R��33?��þ�33���HC +�                                    BxV�n�  �          @�  ����?�녿��
���HC!�����?�Q��  ��  C��                                    BxV�}~  �          @�
=�z�H@녿�\)����C���z�H@�\��p��q��C�q                                    BxV��$  �          @��H�z�H@���Q���C�{�z�H@�ÿ��
�y�C�f                                    BxV���  �          @�=q���\>�������z�C0Ǯ���\?   �����\)C-�                                    BxV��p  T          @�p����>����
���C2^����>��Ϳ��R��
=C/\                                    BxV��  T          @��R���H��G��p���/�C5G����H=u�p���0Q�C3L�                                    BxV�Ƽ  T          @��
��Q�>k��+���\)C15���Q�>�33��R��C/�
                                    BxV��b  "          @�G���\)?(�þ\)��
=C,{��\)?+����
��=qC+��                                    BxV��  �          @�=q��ff?333?5A�
C+���ff?�?Q�Ap�C-!H                                    BxV��  �          @������?�z�?��HA��HC�H����?��?�(�A��HC!��                                    BxV�T  �          @�����  ?�z�?�  Aup�C����  ?�?\A�  C!��                                    BxV��  T          @�\)��z�?��H?�z�A�  C����z�?�Q�?�Q�A�ffC �=                                    BxV��  
�          @��R��=q?��?k�A4��C s3��=q?�{?�Ag
=C"��                                    BxV�-F  �          @��R��?���?Tz�A"ffC#
=��?�Q�?�ffAN{C$��                                    BxV�;�  �          @��x��@��?}p�AG�C�f�x��@33?�\)A��C0�                                    BxV�J�  
�          @��H��Q�@(Q�?�@�
=C�=��Q�@\)?�  A@  C&f                                    BxV�Y8  T          @�(��XQ�@Vff?Tz�A�RCG��XQ�@J=q?��A���C�3                                    BxV�g�  \          @�(��`  @W
=>�G�@��RC@ �`  @N�R?�  A>{C	Q�                                    BxV�v�  ~          @�Q��mp�@6ff?W
=A"=qCk��mp�@*�H?�ffA~�HC8R                                    BxV��*  T          @��\�n�R@=p�?5A33C���n�R@333?�Q�AeC�                                    BxV���  �          @��\�n{@>�R?+�A ��CQ��n{@4z�?�z�A_\)C��                                    BxV��v  
�          @�������@!�>�\)@VffC� ����@��?333AffC�{                                    BxV��  �          @�p��:=q@.�R@�A���C�{�:=q@�@&ffBz�CJ=                                    BxV���  �          @�{��(�?�G�?��HApz�C���(�?��?�p�A�ffC��                                    BxV��h  
Z          @����R?��>�Q�@�(�C#B����R?�G�?�@���C$+�                                    BxV��  
�          @�����ff?���=�Q�?���C#���ff?��
>���@o\)C#�f                                    BxV��  �          @�������?��\�!G����C#޸����?�{�����(�C"ٚ                                    BxV��Z  T          @�33����?�=#�
>�C�����?��>��R@vffC�                                    BxV�	   T          @�����(�?��R�aG��/
=C$(���(�?�\)�+��z�C"�f                                    BxV��  �          @����33?�\)�Y���&�HC�)��33?�p��z���33CE                                    BxV�&L  
�          @�G���ff?��
��{��z�C&���ff?�p����dz�C$z�                                    BxV�4�  T          @�����H?k��n{�2�RC(�����H?���E���
C'�                                    BxV�C�  �          @���33?Q녿�p���=qC)����33?����=q�{�C'=q                                    BxV�R>  "          @��R��G�?�ff������RC&���G�?�{��z��fffC&33                                    BxV�`�  �          @�  ��=q?�p��}p��B�RC!���=q?�\)�=p��Cs3                                    BxV�o�  "          @�p��(��?�(��tz��N�RC�f�(��@33�c33�:�
C0�                                    BxV�~0  T          @��H�R�\?���p���C0��R�\@\)������
C�3                                    BxV���  	�          @�(��z�H?�{�$z�� ��C$&f�z�H?�  ����
=C�                                    BxV��|  �          @�z��{�?k��&ff�
=C&���{�?����H��RC!��                                    BxV��"  T          @�(����\?����Q���
=C�{���\?�z��{���C�                                    BxV���  
�          @�����?�{��{��\)CT{���@�����w�C�
                                    BxV��n  T          @���q�@(���\��=qC0��q�@+�������
=C�                                    BxV��  
�          @���w�@!녿�  �r�\C�
�w�@,(��\(��$��C5�                                    BxV��  "          @���\��@2�\����]p�C��\��@:�H�(����C��                                    BxV��`  �          @��G�@Mp�����dQ�C8R�G�@U�&ff� ��C
=                                    BxV�  
�          @�{�%@s33�8Q��{B����%@w
=������B��q                                    BxV��  
�          @�  ���@~�R�\)��Q�B�3���@���<�>�ffB�G�                                    BxV�R  T          @�=q�
�H�^�R��z��rffCI�{�
�H��  ��
=�z��C:��                                    BxV�-�  
�          @��
�녿�\�y���V
=CY�\�녿�
=���
�hG�COQ�                                    BxV�<�  �          @��R��(��(���]p��>�Cl����(��
=�s�
�Y�
Cf�                                    BxV�KD  �          @��H�z��N{�0  � 
=C�箿z��2�\�L(��@��C�!H                                    BxV�Y�  T          @���?�ff�#33�<���733C�y�?�ff�ff�Q��T33C�ٚ                                    BxV�h�  
Z          @���@�ÿG���\)�m�C��@�þ.{��G��s�\C���                                    BxV�w6  T          @�z�}p��,���u��Tp�Cy�\�}p������r��Ct��                                    BxV���  �          @�
=?�p�����O\)�}��C��\?�p��
=�W
=�HC�%                                    BxV���  T          @�G�@&ff��33��(��]C��@&ff�\)��  �g�
C��R                                    BxV��(  
�          @�Q�@
=>.{���|@��@
=?B�\���
�u�
A�Q�                                    BxV���  
�          @�=q?�G�>�ff��p��qA��?�G�?������BAG�                                    BxV��t  
�          @��R��(�@	���s�
�X��C ����(�@*=q�^�R�>��B�Ǯ                                    BxV��  T          @�(��J=q@#33��G����C{�J=q@1G�������\C
                                    BxV���  
�          @�p��r�\?�
=?�z�Atz�C��r�\?�  ?�A��C5�                                    BxV��f  
�          @����Q�?���?��RA���C.��Q�?У�?�p�A���Cٚ                                    BxV��            @�{�hQ�?�{@  A���C��hQ�?��
@�RB �RC)                                    BxV�	�  T          @�  �S33@33@'�B(�C8R�S33?�@8Q�BffC#�                                    BxV�X  "          @��R�E�?��@@  B  C�)�E�?��H@N{B-�RC��                                    BxV�&�  
�          @�  �\)?��@g�BF�
CB��\)?��@u�BW  C��                                    BxV�5�  �          @�\)�>�R?��R@W
=B4��C� �>�R?��
@a�B@��C!                                      BxV�DJ  
�          @����\��?��H@A�B�\C ���\��?L��@J=qB'
=C&�                                    BxV�R�  
Z          @����5�@{@8��B=qC
���5�@z�@L(�B)�C�{                                    BxV�a�  
�          @����:=q@,(�@1�B=qC	.�:=q@�
@G
=B��C�\                                    BxV�p<  T          @��H�L(�?
=@c�
B>C)���L(�=�@g
=BB
=C1�\                                    BxV�~�  T          @����C�
@z�@>�RB��C  �C�
?�z�@N{B+
=Cz�                                    BxV���  "          @�(��8��@�@AG�Bz�C�R�8��@G�@S�
B,p�C�                                    BxV��.  
�          @�p��&ff@'
=@N�RB$�RCٚ�&ff@�@a�B8�\C�                                    BxV���  T          @�p��*�H@1�@B�\B�
C�H�*�H@�@W
=B-
=C
h�                                    BxV��z  
�          @�
=���@Z�H@5BffB�(����@A�@O\)B$z�B�u�                                    BxV��   
(          @��
�z=q?(��@,(�Bz�C*u��z=q>��
@0  B
=C/aH                                    BxV���  �          @���s33?u@5�B\)C%ٚ�s33?��@;�B{C+
=                                    BxV��l  �          @��
��z�?z�H@z�A�z�C&����z�?0��@�HA�C*��                                    BxV��  "          @�����?�?ٙ�A�p�C����?˅?�z�A���C5�                                    BxV��  "          @�p���@	��?���A���C� ��?��H?ٙ�A�C޸                                    BxV�^  	`          @�{��?�(�?^�RA#33C����?���?���AO\)CJ=                                    BxV�   �          @�p���  @��?���A�33Cu���  @��?�p�A�G�C�                                     BxV�.�  �          @��H�j�H@G�@�A֏\CB��j�H?��R@(�A�(�C�                                     BxV�=P  	�          @�33�o\)@
�H@	��A�G�C���o\)?�33@��A�p�C�                                    BxV�K�  K          @����J�H@<(�@  A�  C	��J�H@*=q@%�A��
C                                      BxV�Z�  �          @�  �<��@R�\@�
Aܣ�C��<��@?\)@+�B�C��                                    BxV�iB  
�          @�{�8��@X��@
=Aʏ\Cs3�8��@HQ�@\)A�\C��                                    BxV�w�  T          @���>{@P��@�
A݅Cff�>{@>{@+�B�C\                                    BxV���  �          @����L(�@W�@   A��RCff�L(�@G�@�A�Q�C��                                    BxV��4  �          @����C33@[�?�z�A���C���C33@L(�@�\A�p�C�                                     BxV���  �          @��H�7
=@x��?��
A�(�B����7
=@l��?��HA��B��                                    BxV���  T          @��\���
@�{?�@��
Bݔ{���
@��?��A>ffB�L�                                    BxV��&  
�          @����=q@�33?�@���Bң׿�=q@�Q�?���AC33B�33                                    BxV���  �          @������@��?��HA]G�B��Ϳ���@�  ?�(�A���B�\                                    BxV��r  
�          @����\)@��R?�z�A�=qB�uÿ�\)@���?��A���B�.                                    BxV��  T          @��\�8Q�@�G�?z�HA3
=B����8Q�@y��?�A�(�B�                                    BxV���  �          @�(�����@�ff?J=qA��B��\����@��H?�G�Ap��B�{                                    BxV�
d  �          @��R��Q�@�(������\B�\��Q�@��<�>���B�                                    BxV�
  �          @��R��\@��\�!G�����B�uÿ�\@��
���
�n{B�aH                                    BxV�'�  "          @�G�?u@�=���?���B�\?u@�z�?&ff@�33B��f                                    BxV�6V  
�          @���h��@��R?:�HAz�BȮ�h��@��?�  Ac\)B��                                    BxV�D�  
�          @�p����@��R?Tz�AG�B�8R���@�33?���Ar�RB�                                    BxV�S�  �          @����(�@���?�@�z�Bۀ ��(�@�
=?��\A7�B��                                    BxV�bH  "          @�p����H@���?O\)A��B�aH���H@��?��Ai��B�=q                                    BxV�p�  
�          @�ff��  @�=q>�@���B�p���  @��?}p�A0  B�                                    BxV��  
�          @��R�u@�=q��\��z�B�\)�u@��H<#�
=uB�=q                                    BxV��:  
�          @����R@��������B��H��R@�z����33B���                                    BxV���  T          @��R�#�
@��ÿ�ff�g\)B��ý#�
@�(��J=q�(�B��                                    BxV���  �          @�
=>�@�\)���H����B�>�@�33�xQ��+�B��                                    BxV��,  
�          @��׿��@��+�����B����@�
=�W
=�=qB�                                    BxV���  �          @�=q�AG�@c�
?@  A=qCO\�AG�@^{?��AUC�                                    BxV��x  �          @��H�S�
?xQ�@8Q�Bz�C#��S�
?0��@=p�B$�
C(B�                                    BxV��  T          @�p��N{?�=q@2�\BffC��N{?��@;�B Q�C�3                                    BxV���  
�          @��R�hQ�?�(�@p�B �
C�R�hQ�?�p�@%BffC!L�                                    BxV�j  
�          @��R�QG�?���@C�
B#�C
=�QG�?��
@J�HB+(�C"�{                                    BxV�  "          @����Fff?�\)@E�B!G�C�)�Fff?�=q@O\)B+��C�q                                    BxV� �  �          @��\�XQ�@�R@��A�z�C�XQ�@  @'�BQ�Cc�                                    BxV�/\  "          @��\�`  @\)@
�HA�=qC���`  @�@Q�A��
C�
                                    BxV�>  T          @��
�U�@.�R@  A�=qC���U�@!G�@�RA��C�                                    BxV�L�  �          @��\�j�H@z�@ffAθRC���j�H@�@33A�  C�                                    BxV�[N  
�          @�
=�HQ�@Z�H?�ffA��C���HQ�@O\)@ffAǅC�q                                    BxV�i�  "          @�ff�;�@n�R?��A���C (��;�@e?��HA�(�C33                                    BxV�x�  
Z          @���3�
@g
=?�\)A��
B��)�3�
@\��?�
=A��C&f                                    BxV��@  "          @�ff�'�@c�
?c�
A6�\B����'�@^{?�Q�Au�B�=q                                    BxV���  �          @�(�����?��H?�R@��
C�����?�33?G�A�
C�3                                    BxV���  �          @�  �xQ�?��R?�Q�A�G�C"B��xQ�?�\)?��A��\C#��                                    BxV��2  
�          @�Q��w
=?�  ?�
=A�G�C"��w
=?��@G�A�ffC$G�                                    BxV���  �          @�p���{?��?^�RA-p�C&ٚ��{?u?uA>=qC'�=                                    BxV��~  T          @��H���?G�?uAB�RC*  ���?0��?��
AO
=C+
=                                    BxV��$  �          @�33���
?��?���A�  C%�����
?p��?��
A���C'0�                                    BxV���  T          @�ff�|(�@Q�?�{A��C���|(�@ ��?��
A���C�                                    BxV��p  �          @�����?��>��
@y��C&�H����?��
>���@�p�C'8R                                    BxV�  T          @������R?��?�R@�\)C%�����R?��?5AG�C&B�                                    BxV��  
�          @�ff����?�{>�\)@X��C&:�����?��>�Q�@�{C&��                                    BxV�(b  
Z          @���=q?n{>k�@2�\C(s3��=q?h��>���@j=qC(��                                    BxV�7  �          @�ff���
?E�>�33@��C*�=���
?=p�>��@���C*��                                    BxV�E�  �          @�ff����>�ff>���@�p�C.u�����>�
=>�(�@�=qC.�
                                    BxV�TT  
�          @��R��>���>8Q�@  C/���>Ǯ>W
=@'
=C/G�                                    BxV�b�  
�          @�
=��{=�G�?�\@�{C2����{=�\)?�\@ȣ�C3�                                    BxV�q�  T          @������R���
>�ff@�{C7�H���R��33>�(�@�z�C8B�                                    BxV��F  
�          @������?p��?�33A]G�C(\)���?Y��?��HAj{C)c�                                    BxV���  �          @�Q�����>\)?�  AG�
C2+�����=�\)?�G�AIp�C3{                                    BxV���  T          @����33��?xQ�A>{C9����33���?p��A6�HC:�=                                    BxV��8  
Z          @�\)��녾k�?�
=Ah(�C6���녾��
?�z�AdQ�C7�3                                    BxV���  "          @�
=����>�33?�{AZ�\C/�\����>�\)?���A_
=C0��                                    BxV�Ʉ  T          @�����>�=q?E�AffC0������>aG�?J=qAC18R                                    BxV��*  �          @�{��=q����?L��A33C5B���=q���?J=qAp�C5�                                    BxV���  T          @�
=���
=u?p��A733C3J=���
    ?p��A7�C4\                                    BxV��v  T          @��R����>�=q?!G�@�\)C0�3����>k�?&ff@�C15�                                    BxV�  T          @��R��>�{��  �Dz�C/ٚ��>�33�k��333C/��                                    BxV��  �          @�\)��ff>Ǯ=�Q�?��C/J=��ff>\=�G�?�=qC/\)                                    BxV�!h  T          @�����  ��  ��\)�[�C6�q��  �k������g
=C6�                                    BxV�0  �          @�
=���R����u�G�C5�H���R�����\)�c�
C5��                                    BxV�>�  �          @�����
��z�>���@�p�C7�\���
���R>\@�{C7�)                                    BxV�MZ  T          @�(�����<��
?.{A��C3�3���ü��
?.{A��C48R                                    BxV�\   �          @��R���H?�R?J=qAffC,O\���H?�?Q�A!p�C,�f                                    BxV�j�  
�          @�Q�����?�Q�����HC�����?�Q켣�
��  C��                                    BxV�yL  
�          @�Q���z�?�(�>�(�@�\)C����z�?�Q�?�@�C�                                    BxV���  J          @�����?�?W
=A#�
C%\)���?�\)?fffA0��C%��                                    BxV���  T          @�\)����?�(�<#�
=��
C$������?�(�=��
?k�C%                                      BxV��>  �          @�ff��G�?�Q콸Q쿋�C%W
��G�?�Q켣�
��\)C%L�                                    BxV���  T          @�{���?�\)����P  C(����?�녾#�
��(�C�                                    BxV�  T          @�  �z�H@#�
�&ff����Cٚ�z�H@%��\��{C��                                    BxV��0  �          @�  �vff@#�
�aG��,  CO\�vff@'
=�=p����C�)                                    BxV���  J          @�\)�xQ�@&ff��G����
C0��xQ�@'������l(�C�q                                    BxV��|  
�          @���dz�@8Q�?&ffA�C(��dz�@5?L��A�C��                                    BxV��"  
�          @�G��n�R@8Q�?O\)AQ�CY��n�R@5�?uA8��C�=                                    BxV��  "          @���vff@3�
=#�
>��C�H�vff@333>8Q�@
=qC�                                    BxV�n  
Z          @�=q�e�@@�׿p���6{C���e�@C33�J=q���C�\                                    BxV�)  T          @����<��@1��p����C���<��@8Q�����C��                                    BxV�7�  
�          @�z��'�?��\�z�H�Yz�C��'�?��H�w��U33C=q                                    BxV�F`  �          @�z���?p����G��vp�C(���?������q�RC�\                                    BxV�U  �          @��H�HQ�?�33�^�R�8�HC�f�HQ�?�ff�Z�H�5
=CaH                                    BxV�c�  "          @���?�(����jC�)��?�z�����d�CQ�                                    BxV�rR  
�          @�p���?�
=��  �Z�
C�\��?�{�z�H�T��C��                                    BxV���  
�          @��� ��?��R�z�H�U=qCJ=� ��?�z��w
=�O�C�                                    BxV���  T          @�ff�   ?޸R�����e��C
���   ?���=q�_(�C+�                                    BxV��D  �          @�p���R?���u�S�C����R@�\�p���L��C	�{                                    BxV���  �          @�p��˅@�H�|(��V�RB���˅@%��u�NB�8R                                    BxV���  �          @���
�H@#�
�dz��;
=C5��
�H@-p��^{�3�C �3                                    BxV��6  �          @���z�@@  �@���B�p��z�@G��8����\B�W
                                    BxV���  T          @��H�G�@J=q�5���B�u��G�@P���.{�	ffB���                                    BxV��  ,          @�G��G�@#�
� ����G�C�=�G�@*=q�=q��{C�=                                    BxV��(  
�          @����|(�?Q��=q���C(B��|(�?h���Q����
C'                                    BxV��  T          @�  �@  ?5�(Q��!��C&���@  ?O\)�&ff�C$��                                    BxV�t  
�          @������G����ϮCM�H����   �����(�CM�f                                    BxV�"  ^          @�\)��(���Q�p���8Q�CF5���(���z�}p��B{CE�\                                    BxV�0�            @����p  ��\�������CSn�p  ��R��\)����CR�q                                    BxV�?f  T          @���\�Ϳ�(���{��\)CG� �\�Ϳ���33��Q�CF�3                                    BxV�N  �          @������aG���(����C6�f���.{��p���Q�C6.                                    BxV�\�  T          @�  ��(����R��
=���
C8���(���  ��Q����RC7O\                                    BxV�kX  �          @��R���;�
=�������C9u����;�p���ff���HC8ٚ                                    BxV�y�  T          @�ff�w
=�   � ����C;W
�w
=��
=�!G���C:=q                                    BxV���  �          @�p��dz�s33�,����RCB�{�dz�^�R�.�R�\)CA��                                    BxV��J  
�          @�����
?\)��ff���C,L����
?(�������
C+��                                    BxV���  �          @�������?��H�=q��  C#Q�����?��\����Q�C"u�                                    BxV���  T          @�����?���
=���HC%�H����?�{�����
C%�                                    BxV��<  �          @��
��\)?��z���z�C,�q��\)?z��33����C,0�                                    BxV���  T          @��\��z�?�\�����Q�C-\)��z�?\)��{����C,�                                     BxV���  
�          @�����{?333�����z�C*� ��{?@  �Q��ԏ\C)�\                                    BxV��.  �          @����q�?�������33C�3�q�?�33��R��ffCO\                                    BxV���  "          @�Q��s�
?�{��H����C ff�s�
?�z������  C��                                    BxV�z  
�          @�33��\?��
�}p��b
=C����\?�\)�{��_z�C0�                                    BxV�   T          @��\�޸R?��
��z��y�
C�{�޸R?�\)����v�HC�                                     BxV�)�  
Z          @��H��p�?�33��p��w�HC�{��p�?޸R��z��tQ�Ch�                                    BxV�8l  "          @��ÿ�=q?�����z�\)B��
��=q?�z����z�B��)                                    BxV�G  
�          @�  �#�
?�\)�����B�Q�#�
?��H����B�\                                    BxV�U�  
�          @�  �^�R?�=q��Q��{C �^�R?�z�����B�B�                                    BxV�d^  _          @�=q��G�>�{����«�HB�
=��G�>�(�����©��B�33                                    BxV�s            @�ff�33?��
�`  �]p�C�f�33?���^�R�[ffC�                                     BxV���  
�          @��\�h��@
=��
��33C�h��@���G���G�C��                                    BxV��P  
�          @��H�i��@	���{����Ck��i��@(�����p�C�                                    BxV���  T          @�ff�g�=u�N�R�&�
C3��g�=�G��N{�&C2L�                                    BxV���  
Z          @�{�aG�����C�
�  CK�
�aG���  �E��=qCK                                    BxV��B  "          @�=q�Y�����XQ��'
=CN@ �Y���У��Y���(Q�CM��                                    BxV���  "          @�=q�^�R����I����HCP�{�^�R��{�J�H�33CP�                                    BxV�َ  
(          @�  ����G���G���p�CQ=q����  ���
��  CQ                                    BxV��4  
Z          @�
=��z����G��l��CK����z��33���
�p��CKn                                    BxV���  �          @��\�S�
@�/\)�ffCǮ�S�
@
=�.{��Cz�                                    BxV��  �          @����   @E��L(��=qC&f�   @Fff�J=q���C ��                                    BxV�&  
�          @���'
=@#�
�dz��0��C�\�'
=@%�b�\�/��C:�                                    BxV�"�  
�          @�=q�Vff@�<(���
C
�Vff@
=�:�H��
C�{                                    BxV�1r  T          @�z��^�R@�\�@  �
=C�)�^�R@�
�>�R�(�CaH                                    BxV�@  
�          @����<(�@+��HQ����C	�H�<(�@,���G
=��RC	n                                    BxV�N�  �          @����\(�@8Q��G���ffC��\(�@8���  ���HC�3                                    BxV�]d  T          @���'
=@;��U�!��C� �'
=@<(��U�� �
C��                                    BxV�l
  �          @��R��?�����
=�zffCc׿�?��H���R�y�
C\                                    BxV�z�  
Z          @�����H?����{�pG�C#׿��H?�33���o�
C��                                    BxV��V  �          @�����{?�Q��33��\)C(���{?�Q��33����C)                                    BxV���  �          @�33���@Q��z����RC�����@Q��z���ffC��                                    BxV���  �          @�G���(�@	�����
��p�C���(�@	�����
��33C�                                     BxV��H  "          @����(�@
�H�#�
���
C����(�@
�H�#�
��\)C��                                    BxV���  T          @�33���
@�
�L�����C�����
@�
�L���=qC��                                    BxV�Ҕ  T          @�{�|(�@(�ÿ�
=�_
=C.�|(�@(�ÿ�
=�_�C0�                                    BxV��:  �          @��R�\(�@N�R��z���\)C���\(�@N�R����  Cٚ                                    BxV���  �          @�
=�?\)@g
=��=q�~=qC�H�?\)@fff�����{C��                                    BxV���  �          @�(��8��@Tz��  ��C  �8��@S�
��G�����C�                                    BxV�,  �          @���z�@N�R�J=q� z�B�{��z�@N{�K��!Q�B�G�                                    BxV��  "          @�
=��ff@vff�.{��B�B���ff@u�/\)�33B�aH                                    BxV�*x  �          @��R��  @hQ��1��
ffB�k���  @g
=�333��\B瞸                                    BxV�9  "          @�p��G�@l(�����(�B� �G�@j�H�����RB��3                                    BxV�G�  
�          @�  �(�@y�����
�IB�{�(�@x�ÿ���O\)B�.                                    BxV�Vj  
�          @�\)�j=q@<(��#�
��C8R�j=q@<(������C8R                                    BxV�e  
�          @�G��K�@c�
����33C�=�K�@c�
�u�5C�=                                    BxV�s�  
�          @��\�I��@e��
=��33C\)�I��@e��!G���  Cff                                    BxV��\  T          @���a�@P  >��?�C	W
�a�@P  =�?�
=C	T{                                    BxV��  
�          @����b�\@P��=��
?fffC	Q��b�\@P��=#�
>��HC	Q�                                    BxV���  
�          @�ff��Q�@\)>��@��C�{��Q�@\)>Ǯ@��HCǮ                                    BxV��N  T          @��R��Q�?�׾\���C.��Q�?�׾�����{C=q                                    BxV���  "          @�G���  >Ǯ��33��(�C/���  >�p���z�����C/B�                                    BxV�˚  	�          @�\)�����(����H��z�CD}q������R��Q��¸RCD�{                                    BxV��@  
�          @��\���׾\�#33�{C9W
���׾���"�\� ��C9ٚ                                    BxV���  {          @�����(�����z����
CCn��(����33��  CC��                                    BxV���  
�          @������\��z�xQ��=�CH�����\��
=�s33�8��CI&f                                    BxV�2  "          @�=q��  �xQ��Q���{C@�H��  ��  ��
=��ffCAG�                                    BxV��  �          @������
���H� �����HCDc����
���R���R�ƣ�CD��                                    BxV�#~  �          @�\)�����=q����33CB:������{��33��33CB��                                    BxV�2$  �          @�\)��G�>��R��ff����C/����G�>�\)��ff��(�C0B�                                    BxV�@�  T          @�
=���?��n{�7\)C-aH���?�\�p���9G�C-��                                    BxV�Op  
�          @�Q����>W
=�����HC10����>8Q����G�C1��                                    BxV�^  
�          @�  ��
=@������~{CO\��
=@���p����Cc�                                    BxV�l�  "          @�z��]p�@AG���z��hQ�C
�
�]p�@@�׾�33���C
�f                                    BxV�{b  
�          @�{�QG�@R�\�L�Ϳ&ffC�{�QG�@R�\���\Cٚ                                    BxV  {          @�\)����@��
�O\)�&�\Bޔ{����@�33�fff�9p�B�                                    BxV�  
�          @�Q��   @vff?�G�At��B���   @xQ�?�Ac\)B��=                                    BxV§T  �          @�  �1�@l(�?�\)AZ�\B���1�@mp�?��
AIp�B��\                                    BxVµ�  �          @�p��*=q@b�\?��A�(�B��
�*=q@dz�?��HAr�\B�aH                                    BxV�Ġ  �          @�z��C33@<(�?\A�(�C
�C33@>{?���A�=qC��                                    BxV��F  
�          @�녿�Q�@�G��L���(�B���Q�@��þ��R�qG�B�q                                    BxV���  �          @�����G�@��
������{BҀ ��G�@��H��  ���Bҽq                                   BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�+*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�9�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�Hv              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�W              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�e�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�th              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVÃ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVÑ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVàZ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVï               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVý�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�$0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�2�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�A|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�P"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�mn              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVĊ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVę`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVĨ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVĶ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��R              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�:�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�I(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�W�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�ft              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�u              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVŃ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVŒf              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVš              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVů�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVžX              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�ۤ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�$�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�3�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�B.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�_z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�n               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�|�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVƋl              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVƚ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVƨ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVƷ^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�Ԫ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��P              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���  �          @�����׿�
=?�@���CM�����׿��?(��A=qCM0�                                    BxV� �  	�          @��\���H���
?�\@��CDu����H��  ?
=@�
=CD                                    BxV�B  "          @�(�����B�\?��@�ffC=�f����8Q�?#�
A�C=#�                                    BxV��  �          @�
=����ff>���@l(�C9xR����(�>���@�G�C98R                                    BxV�,�  �          @���z�u�������C4����zὣ�
��������C5                                      BxV�;4  
�          @�\)���=����\C2�����=��#�
��C2�\                                    BxV�I�  
�          @����  ���?���A�Q�CBW
��  �}p�?\A�CA\                                    BxV�X�  T          @�����{>�?�  AzffC.&f��{?
=q?�(�As\)C-{                                    BxV�g&  T          @�p��s�
?���?�\)A�p�C#z��s�
?�  ?��
A�  C!                                    BxV�u�  �          @����g
=�"�\��33����CW��g
=�#�
�.{�  CWJ=                                    BxVǄr  
Z          @�=q��������#�
���
CCff�������=��
?xQ�CC^�                                    BxVǓ  T          @��
����>.{?&ffA=qC1�\����>aG�?#�
@�
=C1@                                     BxVǡ�  �          @�����?
=q>�z�@a�C-W
���?\)>�  @C�
C-)                                    BxVǰd  "          @�p���=q?(��?
=@��C+���=q?333?
=q@�p�C+G�                                    BxVǿ
  "          @�(����>���>���@n�RC08R���>��
>�\)@\��C/�R                                    BxV�Ͱ  T          @�33��33?�p�?0��A\)C$=q��33?��\?��@�=qC#��                                    BxV��V  �          @�(�����?�(�>�=q@R�\C)����?޸R>\)?�  C�                                    BxV���  T          @��\��  ?��>�  @G
=C�3��  ?�33>�?�{C�=                                    BxV���  
�          @�33��G�@�=���?�ffC����G�@��u�0��C�{                                    BxV�H  
�          @��R�x��@(Q�=#�
>�(�C�R�x��@'��#�
��C�                                    BxV��  �          @�����H@�>aG�@'�C����H@Q�=#�
?   C��                                    BxV�%�  "          @�  ���?�
=�L����C޸���?�zᾮ{���C
                                    BxV�4:  
�          @���\)@Q�=p��z�C#��\)@z�h���3
=C��                                    BxV�B�  �          @�z��r�\?�׿�������C���r�\?޸R��p��˙�CaH                                    BxV�Q�  T          @����vff?�=q��R��
=C �3�vff?�z��z��C#@                                     BxV�`,  
�          @����p��?��H�
=q��p�C".�p��?���  ��\)C$��                                    BxV�n�  
�          @��e�@333�L����
C���e�@/\)��G��JffC�f                                    BxV�}x  �          @�\)�:�H@[���33��=qC}q�:�H@S�
��z�����CxR                                    BxVȌ  "          @�ff���\?�p�����C&f���\?�Q�(���G�C�H                                    BxVȚ�  
Z          @���z=q@	���O\)�%�CE�z=q@��z�H�F�\C�                                    BxVȩj  �          @�=q�L(�@R�\����Qp�C)�L(�@L(�������  C�                                    BxVȸ  T          @����L��@`�׾#�
���HCaH�L��@^�R��(���  C��                                    BxV�ƶ  
(          @�  ����@논��\C�{����@G��B�\��\C�                                    BxV��\  T          @�\)��(�?�(���=q�R�\C�H��(�?�Q�������C�                                    BxV��  "          @�ff���
?�녾�
=��Ck����
?��Ϳ����  C�)                                    BxV��  T          @����j�H@%���
�~ffCǮ�j�H@�R��p���C�                                    BxV�N  
�          @����U@8Q������Q�C:��U@.{�z��ͮC�=                                    BxV��  
�          @���e@2�\���
����C#��e@*=q��  ��
=Cu�                                    BxV��  �          @��\�aG�@=p���  ��Q�C�)�aG�@5��p�����C)                                    BxV�-@  
�          @����s33@p���(�����C޸�s33@�
�������C��                                    BxV�;�  
Y          @�Q��[�@
=�\)��Cn�[�?�33�)���(�C�                                    BxV�J�  �          @�\)�@  @QG���p��w�C���@  @J=q���R���HC��                                    BxV�Y2  �          @�G��3�
@hQ�����p�B�z��3�
@e��5�p�C 
                                    BxV�g�  
�          @���i��@0�׿����^�\C�H�i��@*=q�������C�f                                    BxV�v~  
�          @��
�'
=@vff������Q�B�8R�'
=@n{��Q���33B�                                      BxVɅ$  �          @�(��=p�@,�����ffC	���=p�@   �#�
���C��                                    BxVɓ�  
�          @��\�33@��R����H��B��H�33@�������Q�B���                                    BxVɢp  T          @�{�Ǯ@�ff��z����B��Ǯ@����G��߅B�W
                                    BxVɱ  �          @��R�B�\@��\�ٙ�����B���B�\@�p��z���Q�Bǣ�                                    BxVɿ�  "          @�\)���@����3�
�
=Bã׿��@qG��H�����Bģ�                                    BxV��b  
�          @��׽��
@���z����B����
@����*�H��B��                                    BxV��  �          @��\?��\@�����=qB��?��\@�p�������B��                                    BxV��  T          @��H?�@���� �����RB�aH?�@��
����=qB��\                                    BxV��T  �          @��?�@�{����~=qB��?�@�녿���z�B��3                                    BxV��  �          @�  @{@�>8Q�@33B}��@{@��L����\B}��                                    BxV��  T          @���?��
@�  ��
=���HB�B�?��
@�{�Q���B��H                                    BxV�&F  
�          @�ff>W
=@�������z�B�\)>W
=@��H�����B��                                    BxV�4�  T          @����@hQ���H����B�����@\���G����B�ff                                    BxV�C�  
�          @�33��
@��\�����33B� ��
@{������(�B�33                                    BxV�R8  
�          @��Ϳ\@z�H������B�LͿ\@l���.�R�	(�B��{                                    BxV�`�  
�          @��R�xQ�@�33����~�HB�녿xQ�@�
=��p���{B̞�                                    BxV�o�  �          @�33��  @�녿�33�h��B�\��  @�{��G�����Bνq                                    BxV�~*  
          @����G�@�Q�����z�B���G�@|(��p���;33B��                                    BxVʌ�  �          @��H�33@y������\)B�\�33@o\)�   �Ù�B�.                                    BxVʛv  �          @��\�K�@Fff�����{C�K�@;���
�˙�C	Y�                                    BxVʪ  /          @�(��=p�@c�
��z����
C�\�=p�@Z�H��(���z�C�                                    BxVʸ�  	�          @����A�@o\)�W
=�C
=�A�@i����z��[�
C�3                                    BxV��h  T          @���I��@k��!G���{C���I��@g
=�s33�4  C�                                    BxV��  �          @�{�Vff@`  �L����C���Vff@Z�H�����P(�Cc�                                    BxV��  �          @�{�j�H@P  �\)��
=C
}q�j�H@N{��(�����C
��                                    BxV��Z  T          @�p��i��@L(���R��Q�C
�
�i��@G��fff�)G�Cs3                                    BxV�   T          @�z��G
=@l(����Ϳ�z�C{�G
=@j�H��
=����CB�                                    BxV��  	�          @�{�HQ�@s33>B�\@	��Cu��HQ�@s33������Cs3                                    BxV�L  "          @��N�R@j=q>��@�{Cs3�N�R@l(�>\)?���C=q                                    BxV�-�  "          @�z��l(�@HQ�=���?�z�C�R�l(�@G��8Q��	��C                                    BxV�<�  
(          @�z����\@Q�����RC�
���\@��&ff��ffC^�                                    BxV�K>  "          @�ff����@{?��\A@  C������@33?Q�Ap�C�3                                    BxV�Y�  
�          @�ff�{�@,��?��
AB=qCu��{�@1�?G�A
=C��                                    BxV�h�  T          @�(���ff?�=q���R�mp�C'
=��ff?�ff�����33C'k�                                    BxV�w0  
�          @�����
=>B�\?(��@�
=C1���
=>�  ?#�
@�\)C0��                                    BxV˅�  �          @��
��33?��\?��HA���C#���33?�?�=qA���C!��                                    BxV˔|  T          @��
��G�?��
?��HA��C&�\��G�?���?�{A��C$c�                                    BxVˣ"  �          @����w�@!G�?�z�A���C���w�@*=q?�
=A�p�C�                                    BxV˱�  "          @�p��}p�@{?��HA�33C
�}p�@%?�p�Aip�C�{                                    BxV��n  
Z          @�{�n{@3�
?��HA�33C��n{@<(�?���Ac33C�                                     BxV��  
�          @�
=�w�@4z�?�(�Ad  C�f�w�@:�H?s33A2{C�3                                    BxV�ݺ  �          @��{�@�R?��Aw\)CǮ�{�@%?�ffAIC��                                    BxV��`  
�          @�p��\)?�G�@(�AՅC=q�\)?���@G�A�C��                                    BxV��  "          @�ff�}p�@$z�?���A~�RC
�}p�@+�?���AP(�C�3                                    BxV�	�  
�          @�Q��N{@`  ?�
=A�Q�C���N{@g�?�{AM�C��                                    BxV�R  	�          @�\)�XQ�@S33?�p�Ai�C� �XQ�@X��?k�A.�\C�f                                    BxV�&�  T          @��R�U@L(�?��HA�z�CL��U@S�
?�z�A]�C:�                                    BxV�5�  
(          @�  �p�@�\)>���@z=qB�k��p�@���u�.{B�B�                                    BxV�DD  "          @������@��R������HB�q���@�p��\)��{B�{                                    BxV�R�  �          @�����@�G�����
=B�\)���@�\)�fff�$z�B���                                    BxV�a�  /          @��
�#�
@��
�(����
=B�q�#�
@�G������@��B��                                    BxV�p6            @�����@�p��+���B�  ���@��\��=q�DQ�B��                                    BxV�~�  
g          @�33�&ff@���   ����B�녿&ff@��u�/33B�#�                                    BxV̍�  
          @���333@�33�\)��Q�B�aH�333@��׿�G��<��Bģ�                                    BxV̜(  
�          @����G�@�\)�p���)B�
=��G�@��
��33�33B�G�                                    BxV̪�  
�          @��#�
@��ÿ&ff����B����#�
@�{��\)�L(�B���                                    BxV̹t  �          @��\���@�
=?\(�A��B�aH���@�G�>�
=@�\)B��                                    BxV��  �          @�=q��(�@���>�  @1�B��ÿ�(�@���B�\�
�HB��                                    BxV���  �          @�=q��z�@�\)�W
=��HB���z�@�{�(����  B�{                                    BxV��f  
�          @��׿�@��R?�33AV�RB����@���?8Q�A�RB�Q�                                    BxV��  �          @���ff@��H?�33A`��B�k��ff@�?B�\A  B�k�                                    BxV��  T          @��Ϳ�z�@��?
=q@љ�B�3��z�@��H>�?�{B�ff                                    BxV�X  "          @��
�@��@g
=?�p�Ak33C���@��@mp�?c�
A(z�C�                                    BxV��  T          @��H�U�@W
=?s33A6�\C���U�@\(�?!G�@�G�C\                                    BxV�.�  
�          @��H�U�@^{=L��?��C�\�U�@\�;�\)�W
=C�f                                    BxV�=J  �          @��H�h��@K�=�G�?��C
���h��@J�H�B�\��RC
�                                    BxV�K�  
�          @��H���@��=��
?k�C����@(������=qC�\                                    BxV�Z�  T          @�=q���?�33>��?�  C����?�33���ǮC��                                    BxV�i<  �          @�33���
?�����Q쿇�C"�H���
?�
=�k��*�HC"��                                    BxV�w�  
�          @�����33?!G��u�&ffC,�\��33?!G�������C,�f                                    BxV͆�  
�          @�z����H?���
=���C-�����H>�������RC.aH                                    BxV͕.  T          @��R���?\)�=p��
ffC-ff���>��H�J=q��
C.@                                     BxVͣ�  �          @�p���=q?=p��\)���C+O\��=q?.{�!G���33C+�R                                    BxVͲz  �          @�p���33>8Q�E����C1����33=��ͿJ=q�\)C2�                                    BxV��   �          @�p���=q��ͿO\)�Q�C:���=q�!G��B�\�C;ff                                    BxV���  
�          @��H��p�����z��ۅC@�=��p���������z�CA�                                    BxV��l  
�          @�����\��ff?\)@���C@޸���\�}p�?&ff@��RC@.                                    BxV��  �          @������\)?���A��CB0����}p�?�A�G�C@�
                                    BxV���  �          @���p�׾�
=@=p�B33C:n�p�׾\)@?\)B��C6!H                                    BxV�
^  �          @�(��c33��\@5�B=qCN}q�c33���R@?\)B\)CJǮ                                    BxV�  T          @��7
=�
=q@UB+�HCY��7
=��=q@a�B8ffCT�H                                    BxV�'�  "          @�ff�*=q�{@`  B5�C[�f�*=q���@l��BBz�CW@                                     BxV�6P  �          @����   �-p�@mp�B?G�Ci�)�   �@|��BPp�Ce��                                    BxV�D�  �          @�\)�ff�(Q�@i��B=
=CgaH�ff�G�@xQ�BM�Cc8R                                    BxV�S�  T          @��2�\���
@g�BC
=CP��2�\��
=@p  BL��CJ�f                                    BxV�bB  �          @�z��=p�����@h��BC�RCH�3�=p��G�@n�RBJp�CBǮ                                    BxV�p�  T          @�p��S33?��\@W
=B-��C��S33?�=q@N{B%G�Ck�                                    BxV��  
�          @�p��dz�?�R@Dz�B!33C*!H�dz�?h��@@  B��C%�                                    BxVΎ4  �          @�\)�+����\@\)BU�\CMp��+��c�
@�33B]CFQ�                                    BxVΜ�  
�          @�z��5�����@o\)BH��CM
=�5��u@vffBP�
CF��                                    BxVΫ�  �          @���O\)���@Z�HB4=qCF#��O\)�:�H@`  B:{C@�3                                    BxVκ&  
(          @���j=q�fff@AG�B�
CA�{�j=q�(�@EBG�C=p�                                    BxV���  
�          @�=q�s33�+�@6ffB��C>��s33����@9��B�
C9�q                                    BxV��r  	�          @����z���@ffA��C9����z�8Q�@Q�A�
=C6�                                     BxV��  
�          @��\�s�
���@:=qB�C<���s�
���
@<��B�RC8�                                     BxV���  T          @�33�e�!G�@L��B%p�C=�R�e���
@P  B(Q�C9!H                                    BxV�d  �          @�33�\(��:�H@P��B+33C@��\(���
=@S�
B.�
C:�R                                    BxV�
  
�          @���\(���@S�
B/��C6��\(�>B�\@S�
B/ffC0�=                                    BxV� �  �          @��\��\)?���?k�A4(�C%���\)?��
?L��A��C$�                                    BxV�/V  �          @��\��(�?��H?�@ə�C%aH��(�?�  >��@���C$޸                                    BxV�=�  "          @�����33?��?���Ah��C-(���33?(��?��A]G�C+�=                                    BxV�L�  "          @��
��33>��H@�RA�C-(���33?:�H@�HA�\)C)��                                    BxV�[H  �          @���}p��B�\@1�B33C6���}p�=��
@1�Bz�C2ٚ                                    BxV�i�  
�          @�33�q녿.{@<(�B��C>8R�q녾���@?\)B�RC:�                                    BxV�x�            @�G��aG���=q@@  B�CD�q�aG��J=q@EB"=qC@�{                                    BxVχ:  
�          @����aG��+�@EB#33C>Ǯ�aG���p�@H��B&ffC:�                                    BxVϕ�  �          @�  �g���@@��B�
C<:��g��k�@C33B �C7�R                                    BxVϤ�  T          @�  �Fff���\@U�B3G�CJQ��Fff�s33@[�B:��CE�                                    BxVϳ,  �          @�  �;���
=@Z�HB9p�CN��;�����@b�\BB
=CH�\                                    BxV���  �          @�����Ϳ��H@g
=BD{CZ�f��Ϳ�{@q�BP�HCUE                                    BxV��x  T          @���{�G�@g�BC�Ca���{��z�@tz�BR
=C\��                                    BxV��  �          @�G��0  ��@X��B5=qCV���0  �˅@c33B@�CR�                                    BxV���  
�          @�  �S�
�.{@N�RB/(�C?�\�S�
��p�@Q�B2�\C:T{                                    BxV��j  	.          @���HQ�\@Dz�B%CM�)�HQ쿜(�@L��B.p�CIJ=                                    BxV�  �          @�=q����@�(����
�r�\B��H����@��\�:�H�z�B�\)                                    BxV��  �          @����@��@c�
?O\)A�\C&f�@��@hQ�>��@�  C��                                    BxV�(\  "          @�{�fff@ ��?��A��C&f�fff@)��?��A�Q�C��                                    BxV�7  
�          @��R�u�@�?��A�\)C
=�u�@(�?˅A�z�C5�                                    BxV�E�  �          @�{�o\)?Ǯ@A��
Ck��o\)?�\@�A�Q�C��                                    BxV�TN  �          @�{�N�R?��
@G�B'�RCW
�N�R?���@?\)B  C�                                    BxV�b�  �          @��h��?��@
=A�\C���h��?���@��A��C\                                    BxV�q�  
(          @��H�^�R?��R@)��B�RC Y��^�R?��R@!�B�HC�H                                    BxVЀ@  T          @���U�@#�
@G�A�\)C��U�@.�R?�\A��\C��                                    BxVЎ�  	�          @�(��c�
?�p�@�A�(�C���c�
@
�H?�
=AƏ\C��                                    BxVН�  T          @�=q�aG�?��@*�HB�
C#}q�aG�?��@#�
B	�C�f                                    BxVЬ2  �          @��
�E��=q@1G�Bp�CR���E����@;�B CN��                                    BxVк�  �          @��{�5�@2�\BffCd�f�{�#�
@B�\B"(�Cb�                                    BxV��~  �          @�p��(Q��ff@?\)B!=qC]���(Q���
@L��B.��CZ�                                    BxV��$  T          @�p���
�AG�@4z�B�\Ck����
�/\)@EB'�Ci
=                                    BxV���  
�          @�p��ff�\)@O\)B3��C_���ff��
=@[�BAC[ff                                    BxV��p  
�          @�
=�(��z�@G
=B0��C\5��(���\@R�\B=�CW�H                                    BxV�  �          @�{�
�H�z�@eBHz�C_���
�H��p�@p��BVz�CZ�\                                    BxV��  �          @��R�	����=q@n{BSQ�C\p��	����p�@w�B`=qCV�=                                    BxV�!b  "          @��R�(���p�@fffBIp�CWT{�(����@p  BT�
CQ�f                                    BxV�0  �          @�\)�#�
���@tz�BXz�CJxR�#�
�333@z=qB_\)CCQ�                                    BxV�>�  "          @����p��n{@}p�Ba��CH���p��\)@���Bg�RC@�=                                    BxV�MT  
�          @����z�J=q@��\Bk�RCFٚ�z��
=@�(�Bq  C>!H                                    BxV�[�  
�          @���\)����@uB[(�CKaH�\)�8Q�@z�HBbG�CD)                                    BxV�j�  F          @�ff�%��G�@p  BU��CIE�%�(��@u�B\ffCBJ=                                    BxV�yF  �          @��R�L(�?��
@J=qB*ffC��L(�?���@A�B!��C��                                    BxVч�  
�          @��N�R>�G�@S�
B5�HC,E�N�R?=p�@P��B2�C')                                    BxVі�  x          @��[�>�
=@G�B(p�C-��[�?333@Dz�B%{C(�                                     BxVѥ8  T          @��C�
���
@K�B8z�C5�=�C�
>W
=@K�B8{C0)                                    BxVѳ�  �          @�z��fff@��?�p�A��CL��fff@�\?��
A�{C�{                                    BxV�  
�          @���^�R���
@,��BQ�C5@ �^�R>.{@,��B
=C18R                                    BxV��*  "          @�ff�X�ÿ�  @<��B�CHL��X�ÿz�H@C33B#ffCD&f                                    BxV���  "          @�{�Vff�L��@J�HB*(�CAz��Vff��@N{B.(�C<��                                    BxV��v  "          @��R�Z�H���@I��B(��C=��Z�H���R@L(�B+�\C98R                                    BxV��  
�          @�Q��&ff��\)@g
=BF��CS��&ff���@o\)BP�RCNaH                                    BxV��  
Z          @�
=�J�H��{@QG�B1  CGT{�J�H�O\)@W
=B6��CBff                                    BxV�h  "          @�{�^�R�\)@Dz�B$ffC=.�^�R��\)@G
=B&�HC8�                                    BxV�)  �          @����]p�����@C�
B%��C8��]p����
@Dz�B&�\C4aH                                    BxV�7�  
�          @����|(�>��@��A��RC1���|(�>�Q�@Q�A�{C.��                                    BxV�FZ  �          @�z����=u@�A�C3����>��@
�HA�{C0c�                                    BxV�U   
�          @�ff������=q?���A�33C7����������?�A���C5^�                                    BxV�c�  
�          @��|�;��@�A�C6=q�|��=L��@�A�=qC3=q                                    BxV�rL  
�          @��|��>�p�@=qA��C.�H�|��?z�@
=A���C+�)                                    BxVҀ�  T          @��R���?Y��?��
A��C(Ǯ���?}p�?���A�G�C')                                    BxVҏ�  
�          @��
���H?z�?��RA�Q�C+�q���H?@  ?�Q�A��C)��                                    BxVҞ>  
�          @�(���(�>�@G�AυC-T{��(�?(��?�(�A�  C*�                                    BxVҬ�  	�          @�{�^{?�G�@%B�
C\�^{?�p�@�A���C=q                                    BxVһ�  �          @���n�R>��@p�B
=C-�n�R?�R@�HBQ�C*��                                    BxV��0   �          @�ff�p�׿�\@,��B��C;�q�p�׾�=q@/\)B�HC8+�                                    BxV���   2          @�  �?\)���H@HQ�B$��CUB��?\)��
=@R�\B/(�CQY�                                    BxV��|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�n              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�0�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�?`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�\�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�kR              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVӈ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVӗD              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVӥ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxVӴ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��6  �          @�=q�b�\@�?��A���C�=�b�\@{?�{A��HC޸                                    BxV���  ]          @��\��ff?k�?�ffA��\C'����ff?��?�(�A�  C&\                                    BxV���  �          @��H��?�  ?�33A�  C#T{��?�{?��A��RC!�3                                    BxV��(  �          @�G��x��?˅?�A���C�R�x��?�(�?��A��HC{                                    BxV���  
�          @�33��  ?�=q?�  A�=qC!����  ?���?���A��C )                                    BxV�t  "          @�33���?��H?��RAz�HC �����?�ff?�\)Aap�C��                                    BxV�  �          @�(�����?�\)?���A��HC!����?�p�?�=qA���C \)                                    BxV�)�  
�          @�33��?   ?��HA���C-:���?(�?�A�{C+��                                    BxV�8f  
�          @�p���(�>�{<�>�Q�C/����(�>�{    <��
C/��                                    BxV�G  "          @������=L�;�  �HQ�C3Y�����<�����K�C3�)                                    BxV�U�  	�          @�����>�=�\)?k�C.\)����>�=#�
>��C.L�                                    BxV�dX  T          @��R���H?z�H>�  @B�\C(  ���H?}p�>.{@�C'��                                    BxV�r�  �          @�p���Q�?h��?#�
@��
C(����Q�?s33?\)@�ffC(
=                                    BxVԁ�  
�          @������?+�?k�A7�C+}q���?=p�?^�RA,(�C*�)                                    BxVԐJ  �          @����?0��?���A��\C+����?L��?��A��\C)Ǯ                                    BxVԞ�  �          @����(�?}p�?��
A�C&����(�?�\)?ٙ�A��HC$�=                                    BxVԭ�  
�          @�=q�g
=@ ��?��A�33C��g
=@��?У�A�p�CT{                                    BxVԼ<  "          @��H�qG�?���@
=A�z�CT{�qG�?�p�?�p�A�(�CL�                                    BxV���  T          @�p��i��?�G�@33A���C@ �i��?�
=@
=qA���C�                                    BxV�و  �          @�{����?�p�?�z�A��CǮ����?���?\A��CT{                                    BxV��.  
�          @������@��>�?�\)C@ ���@�ü���(�C33                                    BxV���  �          @����z�@���
=����C�)��z�@�ÿ(���Q�C                                    BxV�z  T          @�=q���@{��\�\C�����@�H�0���G�C33                                    BxV�   
�          @�����H?�=q���
�E�C.���H?޸R��z��_�C)                                    BxV�"�  
�          @���g
=@B�\���
�E��C��g
=@<�Ϳ�G��q��C�                                     BxV�1l  "          @�G��|(�@������G�CO\�|(�@�R�\���\Cz�                                    BxV�@  "          @�
=��?�\)����QG�C���?�������Q�C5�                                    BxV�N�  
(          @����}p�@"�\>.{@Q�CQ��}p�@#33�#�
�8Q�CB�                                    BxV�]^  T          @�{�z=q@"�\�W
=�(��C��z=q@!G�������{CE                                    BxV�l  �          @�{�{�@�������\CY��{�@33�z���
=C�q                                    BxV�z�  �          @�  �x��@{�xQ��?
=C���x��@�ÿ�33�b�RCc�                                    BxVՉP  
�          @������
@���p��n�\C5����
?�p�������{CE                                    BxV՗�  �          @�����?�녿�  �t��C�H��?�ff������C��                                    BxVզ�  "          @�Q����?�����R�n�RC#�����?��þ����z�C#ٚ                                    BxVյB  T          @�\)����?����(����C$�����?�G����˅C$xR                                    BxV���  T          @�
=��(�?У׿z���G�C�\��(�?˅�0���33C 
                                    BxV�Ҏ  �          @�  �|��@�R��33�c\)C���|��@	����ff��p�Cz�                                    BxV��4  "          @����o\)@33������Cn�o\)@
�H���H��p�C�                                    BxV���  �          @�G��^�R@9����{�\��C=q�^�R@3�
�������C{                                    BxV���  �          @��H���?˅�����\C�����?��H��Q����
C 33                                    BxV�&  
�          @��
��p�?Tz��
=��  C(���p�?(���=q�홚C*��                                    BxV��  �          @�=q�U�?�=q�Fff� \)C�)�U�?�{�L���&C�q                                    BxV�*r  �          @�=q�`��?���C33���C"���`��?aG��G��"�HC&                                      BxV�9  �          @����p  >�\)�7
=�  C/���p  =��
�7��C2�R                                    BxV�G�  �          @�\)�\(�?�{�C�
�!{C"33�\(�?c�
�HQ��%p�C%n                                    BxV�Vd  T          @�\)�\(�?=p��G��&ffC'��\(�?��J�H�)(�C+\)                                    BxV�e
  T          @��H�vff?L���G����
C(J=�vff?#�
�z����C*�                                     BxV�s�  �          @�=q�{�>��R����C/s3�{�>#�
��\��C1��                                    BxVւV  �          @�
=�i��>�
=�<(���C-aH�i��>k��=p���HC0k�                                    BxV֐�  T          @�p��u�>��
�'
=��
C/(��u�>���(Q��	��C1��                                    BxV֟�  �          @���vff=�\)�%��C2���vff�����%���C5n                                    BxV֮H  �          @�=q��=q���R�G�����C8J=��=q��G��   ��{C:�                                    BxVּ�  �          @�����
�\)�u�C�C;W
���
��R�k��;�C<�                                    BxV�˔  �          @����=q�aG��z�H�H��C?�\��=q�p�׿k��<��C@W
                                    BxV��:  �          @������?�\�����c33C-:����>�G������i��C.+�                                    BxV���  �          @�z���Q�?��fff�3�C-p���Q�>��n{�9�C.+�                                    BxV���  �          @�������\)�Tz��"=qC5� ����B�\�Q�� (�C6c�                                    BxV�,  �          @������Ϳ#�
�p���5G�C;�H���Ϳ333�fff�,��C<��                                    BxV��  T          @�����33��ff��ff�{�
CG����33�У׿����h(�CH��                                    BxV�#x  �          @�\)���ÿ\�����z�CG}q���ÿ˅���H�mCHh�                                    BxV�2  �          @������Ϳ\��  �B{CG����Ϳ�=q�h���/\)CG�                                    BxV�@�  �          @�����p���\)����^ffCE=q��p���Q쿇��M��CF�                                    BxV�Oj  T          @����Q�
=q�W
=�(  C:ٚ��Q�
=�O\)�!�C;z�                                    BxV�^  
�          @��\���R��p��=p����C8����R��
=�5���C9Q�                                    BxV�l�  �          @�����p�?J=q��ff��\)C)����p�?B�\�   ��=qC*=q                                    BxV�{\  T          @�Q���?5��
=��G�C*�3��?.{����=qC+E                                    BxV׊  �          @�  ����?��
�����Q�C#L�����?�G����H�ƸRC#��                                    BxVט�  �          @�\)��\)?�33����z�C!�f��\)?�\)�
=q��C"�                                    BxVקN  �          @�33���?�
=���Ϳ��
CY����?�
=�L���   Cs3                                    BxV׵�  T          @�
=���?��>.{@�
C\)���?�33=u?=p�CG�                                    BxV�Ě  �          @��R���R?�p����Ϳ�G�C!�H���R?�(��B�\��C!�R                                    BxV��@  T          @��R���R?��þ��H����C#}q���R?������ffC#�
                                    BxV���  �          @�ff��\)?Ǯ�����[\)C����\)?�  ��
=�l��C s3                                    BxV���  �          @���|��@�R=�G�?�{C���|��@\)���
�L��C�                                     BxV��2  �          @�33���\@ ��>8Q�@{C�R���\@G�=�\)?Tz�C�f                                    BxV��  T          @����|(�@Q�>�=q@\��C���|(�@��>��?�(�CxR                                    BxV�~  �          @���z�H@G�>\@�p�C�3�z�H@�\>�=q@`  C�                                     BxV�+$  �          @�\)�}p�?�(�?�\@�=qC���}p�@   >���@�ffC=q                                    BxV�9�  T          @�G�����@�<��
>aG�C0�����@���Q쿘Q�C5�                                    BxV�Hp  �          @�\)�z=q@
=>�{@�=qC���z=q@�>aG�@8��C}q                                    BxV�W  �          @����P��@>�R?s33AAC	���P��@AG�?J=qA!p�C	+�                                    BxV�e�  �          @����j=q@!�?.{A
ffCY��j=q@#�
?��@�
=C                                    BxV�tb  �          @�Q��z�H@z�?5A��C��z�H@
=?��@�\)C��                                    BxV؃  �          @�����
=?�33?L��A#33C!����
=?�
=?:�HAz�C!8R                                    BxVؑ�  �          @���~�R?�z�?&ffAffC^��~�R?�Q�?\)@���C                                    BxVؠT  �          @�����p�?�ff?n{A<  C�)��p�?���?Y��A+�
C{                                    BxVخ�  �          @��H��{?�Q�?5A=qC
=��{?�(�?�R@��\C��                                    BxVؽ�  �          @�(���p�?�z�>���@��RCs3��p�?�>��R@s33C@                                     BxV��F  �          @���}p�@���
=���
Ch��}p�@=q�����C�f                                    BxV���  �          @�33����?.{?�G�A_33C*xR����?8Q�?z�HAW�
C)�\                                    BxV��  �          @���l��?��?��HA�  C"��l��?�(�?�33A��C!��                                    BxV��8  �          @��R�~{?���?E�A)C$��~{?���?8Q�A�\C$xR                                    BxV��  �          @�G��w
=?333?��A�{C)���w
=?B�\?�{A�(�C(�{                                    BxV��  �          @�G��dz�?޸R?��A�(�C\�dz�?��?��HA���CT{                                    BxV�$*  �          @��R�c�
?\(�?��\A���C&ff�c�
?k�?�p�A�  C%��                                    BxV�2�  �          @����l(�>L��?�(�A��C0��l(�>�z�?��HA�G�C/�{                                    BxV�Av  �          @��
�w�?��?�A�C+��w�?0��?���A�ffC)��                                    BxV�P  �          @�z���Q�?���?��A�
=C$8R��Q�?�Q�?��A��C#xR                                    BxV�^�  �          @�z����?�p�>���@�=qC ����?��R>�{@�\)C�\                                    BxV�mh  T          @���s33@33>��@�C�)�s33@z�>\@���Cff                                    BxV�|  �          @����l(�@�R?E�A!�CǮ�l(�@��?.{A�Cp�                                    BxVي�  �          @��H�^�R@%��\)�c�
CW
�^�R@%�.{��
Cc�                                    BxVٙZ  T          @����W�@��aG��E�C��W�@
=���
��p�C
=                                    BxV٨   �          @���L��@(Q�.{�=qC��L��@(Q쾏\)�w�C��                                    BxVٶ�  �          @���6ff@J�H>�ff@�  C�R�6ff@L(�>��
@���C�
                                    BxV��L  �          @�z��Mp�@>�R>���@�G�C	��Mp�@?\)>W
=@0��C	                                      BxV���  �          @����L(�@0��>��@ffC(��L(�@0��=L��?&ffC�                                    BxV��  �          @����z�@��H������B�{��z�@�=q�(��G�B�B�                                    BxV��>  �          @��H���H@�  �B�\�!G�B�.���H@�
=�k��C�B�G�                                    BxV���  �          @�{����@�\)�xQ��C\)B�G�����@�{����e�B�\)                                    BxV��  �          @�ff?�  @�  ��Q��p(�B��?�  @�ff�����{B�Ǯ                                    BxV�0  �          @���?c�
@����33����B��\?c�
@�\)��\��p�B�z�                                    BxV�+�  �          @��
��Q�?���?
=qAb{CͿ�Q�?�\)>��HAI�C��                                    BxV�:|  �          @��R����>���?8Q�A#33C/�R����>���?5A ��C/\)                                    BxV�I"  T          @�33�xQ�>�
=?�Q�A�z�C-�
�xQ�>��?�
=A��RC-!H                                    BxV�W�  T          @�=q��
@G����
��p�B�{��
@Dz�У����
B�                                    BxV�fn  �          @��R�\(�@�
��ff����C!H�\(�@�\�����
CT{                                    BxV�u  �          @��R��33@Z=q�.{�$z�B癚��33@Y����\)��  B�                                    BxVڃ�  �          @����z�@��?ǮA��B���z�@��?�
=A��
B�ff                                    BxVڒ`  �          @�p���G�@��?&ffAG�B�녿�G�@�?�\@��HB���                                    BxVڡ  �          @�ff��
=@��
?L��A�B�(���
=@�z�?(��A
=B�                                      BxVگ�  �          @�����
=@n�R�n{�V�HB���
=@l�Ϳ�ff�qG�B�L�                                    BxVھR  �          @��\?��@��H�:�H��B�Ǯ?��@�  �B�\��HB�u�                                    BxV���  �          @���>�
=@�\)�)�����
B���>�
=@����1G���\B�p�                                    BxV�۞  �          @�Q�?�(�@�p���
=����B��?�(�@���33��G�B��R                                    BxV��D  �          @�\)?�{@�(��&ff��ffB�8R?�{@���G���
B�{                                    BxV���  �          @�p�?���@��
���R�mp�B�G�?���@����G����RB�33                                    BxV��  
�          @���@�@���!G���33B~�@�@�z�@  ��B~�                                    BxV�6  T          @�G�?�ff@�33�p���,  B�=q?�ff@�=q����B�\B�                                    BxV�$�  �          @���@,(�@xQ��������B]33@,(�@u�����Q�B[�H                                    BxV�3�  �          @��\@7
=@b�\�����BLff@7
=@^�R����{BJ�                                    BxV�B(  �          @��@)��@p  �33��ffB[{@)��@l������ɅBY��                                    BxV�P�  �          @��@\)@mp��33���B`��@\)@j=q�����
=B_(�                                    BxV�_t  �          @�
=?��\@�33�����w�
B�(�?��\@�=q��
=��=qB���                                    BxV�n  �          @���?��@�Q��Q���{B�L�?��@��R��\��B��)                                    BxV�|�  �          @�
=?�\)@|(��Q�����B�.?�\)@xQ��p���  B��{                                    BxVۋf  �          @�{?��\@�{��=q��
=B��?��\@��Ϳ����RB��H                                    BxVۚ  �          @�>Ǯ@�=q��  ��33B�8R>Ǯ@��ÿ�������B�#�                                    BxVۨ�  �          @�
=?Tz�@�\)�����S\)B�\?Tz�@��R��(��eB���                                    BxV۷X  �          @�z�>aG�@���z�H�:{B���>aG�@�
=�����L(�B�Ǯ                                    BxV���  T          @��ÿ��H@�녿���{B�=q���H@��׿���z�BՅ                                    BxV�Ԥ  �          @�\)�Q�@����\��G�B�Ǯ�Q�@��H�
=��B��f                                    BxV��J  �          @�����
@��;�33��
=B�8R��
@�z��
=���
B�L�                                    BxV���  �          @�=q�E�@��
��  �AB�녿E�@�33��=q�Qp�B�                                      BxV� �  �          @�G���  @��׿�����Bڮ��  @�  ���R��=qB��H                                    BxV�<  �          @��R��  @�p����H�p  Bᙚ��  @��Ϳ��
�}B���                                    BxV��  �          @��R��33@�p���\)�`(�B�B���33@��Ϳ�
=�mp�B�k�                                    BxV�,�  �          @��R��=q@�G��Tz��$��B܀ ��=q@��ÿfff�1��Bܞ�                                    BxV�;.  �          @��R���@�{��z���B�Ǯ���@�p���(���  B���                                    BxV�I�  �          @�Q쿣�
@��R������B��Ϳ��
@�������B�                                      BxV�Xz  �          @��R� ��@s33��=q��B��� ��@q녿У���
=B��                                    BxV�g   �          @��(�@r�\��\)����B���(�@qG������B�ff                                    BxV�u�  �          @�ff���@���z���33B�k����@���!G���
=B�                                     BxV܄l  
�          @��H�У�@��þ�\)�W
=Bۣ׿У�@��þ����~{Bۮ                                    BxVܓ  �          @�녿޸R@�p��.{�33B�
=�޸R@���:�H�(�B��                                    BxVܡ�  �          @��׿�  @����  ��
=B�׿�  @�z�����G�B���                                    BxVܰ^  �          @�=q��(�@��H��ff���\Bᙚ��(�@��\����z�B�                                    BxVܿ  �          @�33�˅@����Q��c�
B۞��˅@��Ϳ�p��k\)B۳3                                    BxV�ͪ  �          @�33���
@��H������{Bνq���
@��\������B��
                                    BxV��P  �          @�����@p�׿�33��{B�G����@o\)��
=��
=B�u�                                    BxV���  �          @�Q��ff@�33��G���
=B�LͿ�ff@��H������B�k�                                    BxV���  �          @�=q��(�@��\������G�B��3��(�@��\��z���{B��R                                    BxV�B  �          @�G����@�=q�\����B�zΎ�@�녿�ff��33B֊=                                    BxV��  �          @��H�Y��@�������|z�B�8R�Y��@�����=q��ffB�=q                                    BxV�%�  �          @�=q�O\)@��
��=q�P(�B���O\)@�������T  B��
                                    BxV�44  �          @���L��@���z�H�9B�녾L��@���}p��=�B��                                    BxV�B�  T          @����333@��H��ff�K�B�W
�333@��\����NffB�\)                                    BxV�Q�  �          @��R�ff@�33�Y���'�
B�(��ff@�33�\(��)B�.                                    BxV�`&  �          @�ff���@\)�Ǯ��{B����@\)��������B�                                    BxV�n�  �          @��\�"�\@n�R��33��{B�z��"�\@n�R��33���B�z�                                    BxV�}r  �          @���p�@o\)�8Q��
=B�L��p�@o\)�8Q��\)B�L�                                    BxV݌  �          @���'�@X�ÿL���(��B�Q��'�@X�ÿL���(z�B�Q�                                    BxVݚ�  �          @��\��33@�\)������z�Bнq��33@�\)������{Bнq                                    BxVݩd  �          @�Q쿰��@�������UB��Ϳ���@������Tz�B�Ǯ                                    BxVݸ
  �          @�ff�
�H@�(���{��ffB�{�
�H@�(��������HB�{                                    BxV�ư  �          @�33�G�@}p���ff����B�G�@}p���G���Q�B                                    BxV��V  �          @�p����R@�33�fff�3\)B�Ǯ���R@�33�c�
�0z�B�q                                    BxV���  �          @�{��ff@���aG��-p�B�{��ff@���\(��)�B�\                                    BxV��  �          @��\��p�@����G��r�\B�\��p�@�p����R�nffB�                                    BxV�H  T          @��ÿ��
@�33��=q�S�
B����
@�33����O33BڸR                                    BxV��  T          @��(��@��׿������BĨ��(��@�G���ff���BĞ�                                    BxV��  �          @����\@�ff��p�����B�
=��\@�
=������B�                                      BxV�-:  �          @���z�@���˅����B����z�@�p��Ǯ��Q�B�
=                                    BxV�;�  �          @�����G�?+�?�Aȣ�C*����G�?&ff?�A�G�C*�)                                    BxV�J�  �          @�G���z�?�\)?�A��HC$޸��z�?�{?�
=A��C%�                                    BxV�Y,  �          @��\����?�\)?�\A�(�C$������?�{?��
A�\)C$��                                    BxV�g�  �          @�33�W�@Dz�>�p�@�{C	�R�W�@C�
>���@��\C	                                    BxV�vx  �          @�(��^{@B�\>��?�=qC
���^{@B�\>8Q�@\)C
��                                    BxVޅ  �          @��:�H@c�
��R��{CW
�:�H@dz�z���CG�                                    BxVޓ�  �          @����@tz��z���(�B���@u��{��G�B�                                    BxVޢj  �          @�G����H@{��33�̏\B�����H@}p��   ���B���                                    BxVޱ  �          @����@�녿����S\)B���@�=q��ff�H��B�p�                                    BxV޿�  �          @��
��
@z�H��p����RB�W
��
@|�Ϳ�����B�                                    BxV��\  �          @�=q��@~�R��G���Q�B���@�Q�ٙ���=qB�R                                    BxV��  �          @��H��\)@��R��Q���z�B�{��\)@���У����
B��)                                    BxV��  �          @��H�B�\@�(���G�����Bǽq�B�\@����Q���\)BǞ�                                    BxV��N  �          @���?\)@hQ�z�H�<(�CxR�?\)@i���k��0Q�CW
                                    BxV��  �          @�\)�A�@dz�!G����
CL��A�@e��\)��33C8R                                    BxV��  �          @�=q�q�@\)���R�|(�C���q�@\)��=q�W
=C�H                                    BxV�&@  �          @�33�k�@'
=�E��
=C�{�k�@(Q�8Q����Cn                                    BxV�4�  �          @�p��a�@?\)�.{��C���a�@?\)��G���33C��                                    BxV�C�  �          @����mp�@1G�=���?��
C.�mp�@1G�>#�
?�(�C5�                                    BxV�R2  T          @��H�u�@��?5A33C�u�@�?B�\AG�C0�                                    BxV�`�  �          @���|��@��>�
=@��C�R�|��@(�>��@��RC�                                    BxV�o~  T          @��\�n�R@(Q�B�\�C�
�n�R@(Q������C�\                                    BxV�~$  �          @�33�Vff@G��8Q��  C	��Vff@G���G�����C	                                    BxVߌ�  �          @���Y��@>{>.{@(�C
ٚ�Y��@=p�>u@E�C
�f                                    BxVߛp  �          @����p��@ff�B�\��C
=�p��@
=�333��C�
                                    BxVߪ  �          @���i��@{��=q�^{C���i��@{�W
=�*=qC��                                    BxV߸�  �          @�=q�@  @Tz�!G����C(��@  @U��
=q��z�C�                                    BxV��b  T          @����.�R@\�Ϳ^�R�2=qC ff�.�R@^{�E��33C :�                                    BxV��  �          @����{?��ü#�
��C"����{?���<�>�33C"��                                    BxV��  �          @�ff�o\)@�Ǯ��=qC��o\)@ff������
=C�f                                    BxV��T  �          @���w
=@>W
=@2�\C�=�w
=@>�=q@dz�C��                                    BxV��  �          @�{�u@p��u�J=qC\�u@p�    =L��C�                                    BxV��  �          @�\)�w
=@��=��
?�G�C�f�w
=@��>��?�\)C��                                    BxV�F  �          @��R��Q�?�
=�#�
�ffCO\��Q�?�
=���Ϳ��CB�                                    BxV�-�  �          @��~{?�{��
=��Q�C��~{?�\)��Q���  C�                                    BxV�<�  �          @���w�@�׿8Q��G�C�=�w�@녿#�
�ffC��                                    BxV�K8  �          @��\�]p�@"�\��  ��=qC���]p�@%����r�RCL�                                    BxV�Y�  �          @�{��@dz��33��ffB�aH��@hQ���
���B�z�                                    BxV�h�  �          @�Q���@�Q�ٙ�����B�33���@�=q�Ǯ��\)B�{                                    BxV�w*  T          @�z��e@p������ffCY��e@�׿������C��                                    BxV���  T          @���(��@q녿�{���B��f�(��@u���p��lQ�B�=q                                    BxV��v  �          @��
�&ff@x�ÿ�p��j�HB����&ff@{���=q�O33B��                                    BxV�  �          @�ff?��@�Q�?E�A��B���?��@�\)?p��A5p�B�p�                                    BxV��  �          @�
=?z�H@����B�\��B���?z�H@����#�
��\)B�                                      BxV��h  �          @��
�L��@��þ�ff���B��3�L��@�����=q�Mp�B��                                    BxV��  �          @�����=q@�=q��\)�X��B�Q쿪=q@��\���Ϳ��RB�G�                                    BxV�ݴ  �          @��Ϳ&ff@��R����D��Býq�&ff@���Y��� ��BÞ�                                    BxV��Z  �          @�Q��(�@��ÿ����B�p��(�@�����(���Q�B�.                                    BxV��   �          @�G���@����
=��=qB�{��@�Q쾀  �=p�B��                                    BxV�	�  �          @�p�����@�{��=q���RB��)����@�  ��33��=qB�k�                                    BxV�L  �          @�z��  @�33����Z�\B��
��  @��Ϳs33�6=qB�p�                                    BxV�&�  �          @��ÿ�{@��R�0����
B�q��{@�\)�   �ƸRB�z�                                    BxV�5�  �          @�=q��(�@��Ϳ^�R�'\)Bޙ���(�@��.{���B�L�                                    BxV�D>  �          @��H��z�@�Q쿝p��lz�B��쿔z�@�녿��
�D��BЅ                                    BxV�R�  �          @��ÿ���@�  ��Q���=qBր ����@��\���R��=qB���                                    BxV�a�  �          @�������@�  ���H��B֊=����@��\��G����B���                                    BxV�p0  �          @��H���@�z��p����BҸR���@�\)� ����33B�                                    BxV�~�  �          @��H��z�@���������  B�#׿�z�@����
=�i�Bѽq                                    BxV�|  �          @�(�����@����(��hz�B�\)����@�G���  �>{B�                                      BxV�"  
�          @�zᾮ{@��Ϳ����}G�B�G���{@��R����PQ�B�(�                                    BxV��  �          @���<��
@����\)���\B�k�<��
@�������B�p�                                    BxV�n  �          @�=q���R@��Ϳ����{B�z῞�R@��R����S�B�{                                    BxV��  �          @�p�?�ff@��Ϳ������HB�#�?�ff@�\)�����Q�B���                                    BxV�ֺ  �          @��@�@��׿aG��&�HB�{@�@�녿&ff��  B��                                     BxV��`  �          @��R?�(�@�{��
=���
B�#�?�(�@��R�.{���RB�B�                                    BxV��  �          @�G�?�ff@��׿n{�+\)B�� ?�ff@�녿.{��Q�B�                                    BxV��  �          @��?�ff@�
=�33��=qB�� ?�ff@��\����ffB�33                                    BxV�R  e          @��?��@��
������B�\?��@�\)�������B��q                                    BxV��  �          @�Q�>�ff@�z�������HB�>�ff@�\)�Ǯ���B�=q                                    BxV�.�  �          @�
=�#�
@�33��=q��G�B�W
�#�
@�{�Ǯ��p�B�W
                                    BxV�=D  �          @�
=�p��@ff=���?��C� �p��@>aG�@;�Cٚ                                    BxV�K�  �          @�z�����?+�?��\A�G�C*�\����?��?�ffA�p�C+�{                                    BxV�Z�  T          @�{�u�    ?���A׮C4\�u���?���A�G�C5                                    BxV�i6  �          @�{�e�>�
=@%�BG�C-L��e�>�=q@&ffB�\C/��                                    BxV�w�  �          @�ff�o\)�\)@��B=qC6#��o\)��\)@�Bz�C8Q�                                    BxVↂ  �          @��c�
>�z�@$z�B�C/ff�c�
>�@%�B�C1��                                    BxV�(  �          @�33�\(�@5��   ��p�C���\(�@7
=������CL�                                    BxV��  �          @�z��^�R@.�R�����\)C޸�^�R@3�
���H�s33C�                                    BxV�t  �          @����G
=@7���p���=qC	^��G
=@>{��ff���\C\)                                    BxV��  �          @�z��P  @QG��B�\�33C�H�P  @QG�<��
>�\)C�
                                    BxV���  �          @�z��i��@5���
���\C��i��@5=���?�  C)                                    BxV��f  �          @��\�:�H@`  �����33C޸�:�H@`  =��
?uCٚ                                    BxV��  �          @�����@}p���\�У�B�B���@\)��  �L��B���                                    BxV���  �          @�33�k�?c�
?�\)A��
C&u��k�?O\)?�
=A���C'�
                                    BxV�
X  �          @�=q�/\)@g
=>�33@�ffB�L��/\)@e�?
=@�Q�B���                                    BxV��  �          @�(����@Y���˅���\B�=q���@`  ������{B�                                    BxV�'�  �          @�ff�   @h�ÿ�(�����B����   @qG���(���33B�                                    BxV�6J  �          @�{����@|�Ϳ����  B�B�����@�=q��{��ffB�{                                    BxV�D�  �          @�Q��=q@�Q����ģ�Bܔ{��=q@��Ϳ��
��z�B�k�                                    BxV�S�  �          @�Q쿾�R@�{����
=B�#׿��R@��H�G���ffB��)                                    BxV�b<  �          @�  ���@�����\��p�B�.���@��Ϳ�  ��ffBꞸ                                    BxV�p�  �          @��׿�@�������
=B�{��@�
=���H����B�=                                    BxV��  �          @�G���\)@�33�,�����BҔ{��\)@��������G�B�Q�                                    BxV�.  �          @��H�\)@�{�4z��
=B�(��\)@�(�� �����B�z�                                    BxV��  �          @�G���  @��&ff��\)B��῀  @������(�B��H                                    BxV�z  �          @�녿G�@�
=�)����p�B���G�@���z��ۙ�B��                                    BxV�   �          @�G��z�H@�{�%����B�W
�z�H@�(��G��ָRB�B�                                    BxV���  �          @�  ���@��
�/\)�p�Bď\���@�=q��H��{B���                                    BxV��l  �          @�  ����@�p��p���=qB�#׿���@�33�Q���{B��f                                    BxV��  �          @�����@|(��1���B�B����@�z��{��33B֏\                                    BxV���  �          @����ff@z�H�8Q��ffB��ÿ�ff@�z��#�
���
B�u�                                    BxV�^  �          @�
=�E�@xQ��?\)�p�B�Q�E�@���+���B��                                    BxV�  �          @�\)��@j=q�U�(z�B�B���@z=q�B�\�G�B�8R                                    BxV� �  �          @����\@j=q�Vff�(��Býq��\@z�H�B�\�ffB¸R                                    BxV�/P  �          @��׿�@vff�J=q�=qBĳ3��@�33�5�	�RBøR                                    BxV�=�  �          @��׾aG�@g
=�\���.ffB��
�aG�@xQ��H����B�\)                                    BxV�L�  �          @��׿�\@\)�>{�B�uÿ�\@�
=�(Q����B��                                    BxV�[B  �          @�녿   @�G��>{��B�\�   @����'����HB�G�                                    BxV�i�  �          @�G����@`  �`  �3
=B�8R���@r�\�L(��B�=q                                    BxV�x�  �          @��>��@W
=�o\)�@33B���>��@j=q�\(��,��B�=q                                    BxV�4  �          @�Q쿀  @{��z��d��B��ÿ�  @4z��z=q�R�B���                                    BxV��  �          @��H>�  @Fff�u��K�B���>�  @Z�H�c33�7�B���                                    BxV䤀  �          @�(�?�=q@R�\�n�R�<p�B��=?�=q@fff�[��)G�B��=                                    BxV�&  �          @���>\@8Q���  �^Q�B���>\@P  �~�R�J=qB���                                    BxV���  �          @�z�>���@1G�����d
=B�aH>���@H�������O�
B�W
                                    BxV��r  �          @�33>�z�@+�����g�HB��>�z�@C33�����SffB�
=                                    BxV��  �          @��H>8Q�@L(��x���J=qB�aH>8Q�@a��e��5�B�                                      BxV���  �          @��>�z�@W
=�u�BB�
=>�z�@l(��`���-�B��                                    BxV��d  �          @��?n{@W
=�qG��<�B�
=?n{@l(��\���(�\B��3                                    BxV�
  �          @�z�?fff@Z�H�n{�9��B�z�?fff@p  �X���%\)B���                                    BxV��  �          @���?Tz�@g��dz��/Q�B�G�?Tz�@|(��Mp��ffB�G�                                    BxV�(V  �          @��?��@p  �XQ��#�
B�z�?��@����AG��
=B��R                                    BxV�6�  �          @��?xQ�@^{�l(��6��B���?xQ�@s33�Vff�!��B�=q                                    BxV�E�  �          @��H?�Q�@^�R�`���.z�B�\)?�Q�@s33�J=q���B�Q�                                    BxV�TH  �          @�(�?��H@`���\���(�HB��?��H@tz��E�ffB�{                                    BxV�b�  �          @�33?�p�@e��J�H�Q�B�aH?�p�@w��3�
�(�B���                                    BxV�q�  �          @��
?��
@^�R�aG��0��B�  ?��
@s33�J�H�{B��3                                    BxV�:  �          @�Q��R@@  �|���P��B�33��R@XQ��hQ��:p�Bȳ3                                    BxV��  �          @��\��33@Tz��s�
�C  B�𤾳33@k��]p��,\)B�                                    BxV坆  �          @�G���
=@P  �mp��B33B¸R��
=@g
=�W��+z�B�L�                                    BxV�,  �          @��?�  @B�\�vff�I=qB���?�  @Z=q�a��333B�W
                                    BxV��  �          @��ÿQ�@P���j=q�=��B�33�Q�@g��S�
�'
=B͊=                                    BxV��x  �          @�  >8Q�@I���c33�Ap�B�� >8Q�@_\)�Mp��*
=B�#�                                    BxV��  �          @�����@n�R����z�B�aH���@{����
���
B�R                                    BxV���  f          @���@q��
=��z�B��f�@~�R���H����B�u�                                    BxV��j  �          @����C33@I����
=��Q�C\�C33@K���Q쿚�HC��                                    BxV�  �          @����h��@G�?Y��A7�C���h��?�?�ffAeG�C.                                    BxV��  �          @�z��U@�
?���Aә�CO\�U?���@
=qA�{C
                                    BxV�!\  �          @���a�@�?�{Aj�HC
�a�@\)?�{A�(�C�
                                    BxV�0  �          @�  �j�H@$z�>�ff@�{C���j�H@ ��?8Q�A
=C��                                    BxV�>�  �          @����h��@*=q>���@�(�C���h��@'
=?(�@�Q�CQ�                                    BxV�MN  �          @�33�I��@9���!G��p�C	c��I��@<�;��R���C�f                                    BxV�[�  �          @����;�@>{�n{�I�C���;�@C33�����C��                                    BxV�j�  �          @�G��6ff@8Q쿱����C���6ff@AG������fffCk�                                    BxV�y@  �          @�  ��
@<(���
=��(�C 8R��
@HQ�˅��B��H                                    BxV��  �          @�Q����@Dz��(�����B�(����@QG���\)����B�                                    BxV斌  �          @�{�ff@C33�	���陚B�W
�ff@QG���ff��Q�B�z�                                    BxV�2  
�          @���p�@A�����홚B�8R�p�@P  ���
��G�B�ff                                    BxV��  �          @�\)�(��@ff�33�Q�C
L��(��@&ff�G�����Cu�                                    BxV��~  �          @�Q��*=q@33�Q��p�C!H�*=q@#�
�ff��C)                                    BxV��$  �          @�33��p�?���i���W33C	�)��p�@���Z�H�Dz�C                                      BxV���  �          @�G��33?��\���Fz�C�=�33@���Mp��5
=CQ�                                    BxV��p  �          @����#�
?���^�R�HffC�{�#�
?��
�S33�:\)C.                                    BxV��  �          @�=q�9��?��R�K��1p�C�)�9��?����?\)�$
=CxR                                    BxV��  �          @����G�?�=q�@  �%�
C��G�?��4z���C��                                    BxV�b  �          @����?\)?��H�N{�3�C��?\)?˅�C33�((�C\                                    BxV�)  �          @�  �J�H?\�.{���CaH�J�H?�=q�!G��
  C��                                    BxV�7�  �          @����B�\?���<���"{C�\�B�\?��R�.�R���C�\                                    BxV�FT  �          @�=q�>�R?���L���1��C�\�>�R?��H�@���$��C
                                    BxV�T�  T          @��R�8��?�(��G
=�2�
C
=�8��?˅�<(��&��C&f                                    BxV�c�  
�          @����(�?����QG��@�HC޸�(�?�p��C33�0\)C��                                    BxV�rF  �          @�\)��?�ff�{��v�\C  ��?��
�o\)�c�C8R                                    BxV��  �          @������@ff�u��_�B��
����@3�
�`���EBݨ�                                    BxV珒  �          @�(����H@��r�\�\��B�����H@2�\�^{�C(�B���                                    BxV�8  �          @�ff��@���G��pQ�B���@0���n�R�T(�BǮ                                    BxV��  �          @��ÿ�Q�@  �z=q�c�HB뙚��Q�@.{�fff�I�HB�{                                    BxV组  �          @�
=�5�>#�
�Dz��=�C0ٚ�5�?
=q�AG��9��C)(�                                    BxV��*  �          @��n{�   �   ��33CPL��n{���/\)��CL(�                                    BxV���  �          @��R�N{�QG�?L��ACah��N{�U>�p�@�33Cb\                                    BxV��v  �          @��H��{?�=q�=q�Q=qB����{@��	���433B�=q                                    BxV��  �          @���?��
@fff�9���G�B�?��
@|(��=q�홚B���                                    BxV��  �          @�33?�33@W
=�J=q�#B�?�33@p  �,���
=B���                                    BxV�h  T          @��?��R@hQ��1���B�aH?��R@}p���\��G�B�Ǯ                                    BxV�"  T          @�?p��@]p��W��,�B���?p��@w��8����RB��R                                    BxV�0�  �          @��?��
@���j{B��=?��
@7��u��M�HB��\                                    BxV�?Z  �          @��H>��@=q�����n�B��f>��@<(��r�\�P\)B�\                                    BxV�N   �          @���?E�@#�
���\�c
=B�ff?E�@E��mp��E�B��
                                    BxV�\�  �          @�  ?�\@`  �aG��3G�B�\)?�\@|(��AG��B�33                                    BxV�kL  �          @��?#�
@L���{��I=qB�u�?#�
@l���]p��*�
B�p�                                    BxV�y�  �          @�?��@2�\�z�H�R��B�B�?��@R�\�`  �5p�B�8R                                    BxV舘  �          @�p�>�G�@.�R�����^��B���>�G�@P  �h���?�
B��R                                    BxV�>  �          @�\)?@  @:=q�{��R�\B�{?@  @Z�H�`  �3��B�=q                                    BxV��  �          @�G�?��@
�H�����u�B��)?��@0������X
=B��                                    BxV贊  �          @�33?��?�33�����z\)BWG�?��@   ��\)�_�\Bt{                                    BxV��0  �          @�=q?��
?��H���H33BU  ?��
@�����i�Bu(�                                    BxV���  �          @���?��?����33ǮB�
?��?����
�r�BL��                                    BxV��|  
�          @���?��?�Q�������B?\)?��@ff�����uz�BhQ�                                    BxV��"  �          @��?���?�ff�����
B[{?���@p���G��u��B~��                                    BxV���  �          @�33?�ff?�\)��33Br��?�ff@   ��G��fB��\                                    BxV�n  �          @�33?k�@�H����m=qB�Q�?k�@@���~{�NQ�B�                                    BxV�  �          @��\?5@�
������B�33?5@,(����R�b33B���                                    BxV�)�  �          @��R>��?�����
=(�B�p�>��@33��\)�
B��R                                    BxV�8`  �          @���?!G�?�z����R=qBu{?!G�?�{���  B�\)                                    BxV�G  �          @�33?5?�������{Bf��?5?�=q��{��B�k�                                    BxV�U�  �          @���?5?�=q��
=�3B�Q�?5@  ��{�x{B�                                    BxV�dR  �          @�Q�>.{@�����
�u{B�Ǯ>.{@@  �~�R�S�B�u�                                    BxV�r�  �          @�
=��@:�H��  �Wp�B�  ��@^�R�a��5��B��
                                    BxV遞  �          @������@r�\�E�=qB��þ���@��R�\)���RB�{                                    BxV�D  �          @��;�z�@����0���	�HB�W
��z�@�(�����ϮB���                                    BxV��  �          @�{<��
@~{�:=q�
=B��<��
@����\�݅B��\                                    BxV魐  �          @����
@����6ff�{B�����
@����p���\)B�u�                                    BxV�6  �          @�(���@���%���p�B�k���@�ff��
=����B�Q�                                    BxV���  �          @�33��G�@S33�b�\�;��B��)��G�@s33�@  ��B�W
                                    BxV�ق  �          @��H�.{@aG��S33�,�B�ff�.{@~�R�.�R�	�B�Ǯ                                    BxV��(  �          @��>L��@�{�����=qB�u�>L��@�
=��p���Q�B���                                    BxV���  �          @��
>k�@��Q���\B���>k�@�  ���H��=qB��                                    BxV�t  �          @�Q쾔z�@Y���e�9��B�����z�@z=q�A���RB�ff                                    BxV�  �          @��=#�
@`���e�6�\B�=#�
@����@���
=B��                                    BxV�"�  �          @���?:�H@%���
=�f33B���?:�H@Mp��qG��C��B��                                     BxV�1f  T          @�{?��R@�\���H�g33Bc��?��R@<(��{��H�B|�                                    BxV�@  �          @��?�
=@'�����R  Be=q?�
=@N{�fff�2�
By                                    BxV�N�  �          @�{@�@<(��j�H�533BX�@�@^�R�J=q�{Bj��                                    BxV�]X  �          @��?�z�@1��{��E\)B]�H?�z�@W��\(��&��Bq�                                    BxV�k�  �          @�?�33@J�H�n{�8p�BzQ�?�33@n{�J�H��\B�B�                                    BxV�z�  �          @�{@�
@�\)�Q�����B��@�
@�Q쿷
=����B�                                      BxV�J  �          @�z�@33@�\)��z��\)B��@33@��Ϳ.{��\B�z�                                    BxV��  �          @��
?У�@��ÿ�(���G�B�G�?У�@���xQ��/�B���                                    BxVꦖ  �          @���?�(�@�  �Q���  B��?�(�@��ÿ�\)�x��B�ff                                    BxV�<  �          @���?xQ�@p  �Tz��"33B�{?xQ�@���*�H��(�B�G�                                    BxV���  �          @�ff>u@~�R�P  ���B���>u@�ff�#�
���B��R                                    BxV�҈  �          @���p��@`���i���4�B��H�p��@���A��(�B�                                      BxV��.  �          @�����@j=q�c33�/(�B�33���@�{�9���	�B�\                                    BxV���  �          @�\)��@g��n{�6
=Bą��@��E����B�Q�                                    BxV��z  �          @�p����@G
=�w��Bz�B�33���@mp��S�
�ffB�(�                                    BxV�   T          @�33����@:�H�vff�D�HB�aH����@`���Tz��"��B��)                                    BxV��  �          @�Q�\@8�����
�N��B�uÿ\@a��e�,�B�                                    BxV�*l  �          @��ÿ���@8�����R�S�B�#׿���@c33�j�H�033B���                                    BxV�9  �          @�33��  @1���z��\�B� ��  @^{�w
=�933Bۣ�                                    BxV�G�  �          @�33���@&ff��G��g��B�B����@Tz���G��C�HB�                                    BxV�V^  �          @�녿�Q�@���p��e�HB�k���Q�@H���|���Cz�B�\)                                    BxV�e  �          @�����@   ���
�a  C����@-p��~{�D�C �                                    BxV�s�  �          @������?�����
B�k�����@p����R�i��B��                                    BxV�P  �          @�  ���?������CO\���@p����
�v�B�                                    BxV��  �          @��
��ff?�  ��G��z��CG���ff@  ��
=�^�C �                                    BxV럜  �          @��
���?�������C�3���@���33�i��B�aH                                    BxV�B  g          @��H��(�?�����W
C���(�?�=q����qp�C0�                                    BxV��  A          @�33�0��?z�H��ff�qB��׿0��?������R
=Bݽq                                    BxV�ˎ  
�          @������H>����
#�C)���H?�ff���R��C�                                    BxV��4  �          @�p���  ?�G����HffC LͿ�  @�\�����x��B�                                    BxV���  �          @��׿���?����33��C�\����?�������C k�                                    BxV���  �          @�p���(�?!G���Q�\Cٚ��(�?�z����\�3C+�                                    BxV�&  �          @����?�������}�\B�=��@%�~{�YB�\                                    BxV��  �          @��H����?�33�����~��Cff����@Q��|���]  B�33                                    BxV�#r  �          @�(��c�
?��\����� C��c�
?�=q��z�Q�B��)                                    BxV�2  �          @��H���>k����)C(����?�ff��=q�3C@                                     BxV�@�  �          @��\�n{?fff����C�n{?��H��(�#�B�G�                                    BxV�Od  �          @��;�{@�����R�p33B�33��{@G��mp��GG�B��                                    BxV�^
  �          @�Q��G�@<(���G��W�B�Q��G�@g��\���.\)B��                                    BxV�l�  
�          @�����@(Q����H�cB��)���@U��c33�:��B�.                                    BxV�{V  �          @�{�xQ�@I���h���?z�B�G��xQ�@p  �@���ffB���                                    BxV��  �          @�{��p�@)���w
=�O=qB�W
��p�@S33�S�
�)�
B�8R                                    BxV옢  �          @���Ǯ@�R����yG�B�Ǯ�Ǯ@>{�u��O�
B��H                                    BxV�H  �          @�G�>L��@33�����y=qB��)>L��@Dz��z=q�OG�B�aH                                    BxV��  �          @��?n{?�z������Bs  ?n{@ ����z��j�\B�G�                                    BxV�Ĕ  �          @��
?�G�?�(���
==qB0�?�G�@�
����u{Bjff                                    BxV��:  �          @���?�ff?fff���\��Aԣ�?�ff?�p����H�pp�B/�                                    BxV���  �          @�(�?�\)?�=q��=q{B\)?�\)?��H�����x��B[�H                                    BxV���  �          @���?fff?}p����R�
B?�
?fff?�33��{�
B�p�                                    BxV��,  �          @�z�?���?(����{W
A�ff?���?Ǯ��\)��BXQ�                                    BxV��  �          @��?�>�����z��
AR{?�?��R����3B$z�                                    BxV�x            @�p�@z�?��
��=q�f�
B"��@z�@#�
�z=q�G�RBK�                                    BxV�+  �          @�{@{@��vff�Az�B.�@{@C33�U�!�BK                                    BxV�9�  "          @��H?�G�@   ��\)�x�
Bf�
?�G�@3�
�����Rp�B��{                                    BxV�Hj  �          @�33?�=q@ff��ff�sQ�Bf?�=q@9���~{�L�HB��
                                    BxV�W  "          @�(�?���@
�H����tQ�Btp�?���@>�R��  �L�RB�8R                                    BxV�e�  �          @��?@  @
�H��  �{��B���?@  @?\)��  �Q�RB��H                                    BxV�t\  �          @�p�?У�?�\��  �w�\B=ff?У�@%���\�Tp�Bg�                                    BxV�  �          @�p�?��H?����G��y�RB.��?��H@�R�����W��B]=q                                    BxV푨  �          @��R@   ?�z���z��XG�B  @   @�H�p  �<p�B0(�                                    BxV��N  �          @�{?޸R?У����H�y��B,�R?޸R@\)��{�X33B[��                                    BxV���  	�          @���?�
=?�Q����H��B\=q?�
=@&ff���dB�                                      BxV���  T          @��?Tz�?����R��B�k�?Tz�@1G���Q��e\)B��\                                    BxV��@  �          @�33?.{?�z�������B�L�?.{@&ff���
�o{B���                                    BxV���  T          @��?��?������� Bt=q?��@1G�����^�B��f                                    BxV��  
�          @�Q�?�ff?޸R��33u�Bk�?�ff@)������dffB��
                                    BxV��2  �          @��\?fff?���\)aHBv��?fff@'
=����k=qB�
=                                    BxV��  
�          @�33>Ǯ?Ǯ���\�fB�.>Ǯ@!����u\)B���                                    BxV�~  �          @���=���?#�
��
=¥��B�B�=���?�z����33B�{                                    BxV�$$  �          @�
=?fff?�p����\��BzQ�?fff@)����z��e�HB��=                                    BxV�2�  �          @�(�?�
=?�ff���B�BSp�?�
=@p����\�gG�B��H                                    BxV�Ap  A          @�z�?�33?�
=��33�B6(�?�33@������w�
Bu��                                    BxV�P  "          @�p�?�
=?�������B??�
=@33����u�B{(�                                    BxV�^�  
�          @���?�?�{���

=B-��?�@
=�����}33Bs�
                                    BxV�mb  �          @�33?c�
?���z�BRp�?c�
@��������B�L�                                    BxV�|  "          @�Q�>�Q�@�H�����y(�B���>�Q�@S33��=q�K=qB�aH                                    BxV  
�          @��\<��
?�����\)Q�B��H<��
@3�
��Q��h33B�G�                                    BxV�T  �          @���<#�
?�{��(��B��H<#�
@Q�����=qB�k�                                    BxV��  
�          @�\)��\?Tz����Hk�B�B���\?���=q��B��                                    BxV  
�          @��׾��>�\)��
=©=qC�=���?�=q����8RB��f                                    BxV��F  
(          @��
����>�(���G�ª\B�W
����?�Q���33ǮB���                                    BxV���  
�          @��=�\)��33��(�«�C�p�=�\)?333��33¤Q�B��)                                    BxV��  
�          @���>����k����H«�\C�+�>���?Q����� k�B��                                    BxV��8  
�          @��?�(�?�(����3BH�?�(�@�������k�B}�                                    BxV���  
�          @��?E�?��\����=qBj�R?E�@����{33B��{                                    BxV��  
�          @���<#�
?�ff��z��)B��<#�
@���{B�u�                                    BxV�*  �          @��>�Q�?�33��(�k�B�\>�Q�@���\)�z{B�\                                    BxV�+�  �          @�33�B�\?����L�B��B�\@+���33�o
=B�{                                    BxV�:v  T          @�33?�?����{� B�p�?�@7���{�bp�B��{                                    BxV�I  �          @�z�@hQ�?����`  �%A�{@hQ�@�\�E�{B
=                                    BxV�W�  
�          @���@'
=?�z���Q��S{B@'
=@0���qG��3{B:33                                    BxV�fh  "          @��@�\)?˅�4z����A�33@�\)@	����H��ffA�\)                                    BxV�u  �          @�{@Mp�?����J�A���@Mp�@G��u�5ffB �H                                    BxV  �          @�@   ?�����{�\�B�@   @.{�}p��<  B=�                                    BxV�Z  5          @�(�?�p�@\)�����e=qBm
=?�p�@X���y���9�HB��q                                    BxV�   @          @���?
=@9����z��`�B�\?
=@p���k��0(�B�8R                                    BxVﯦ  �          @�����
@-p��K��EffB�p����
@Tz��!���
B���                                    BxV�L  "          @��H�8Q�>k�@7
=B2��C/\)�8Q�\@5B1�\C;xR                                    BxV���  "          @�z��J=q>�{@)��B(�C-�)�J=q�k�@*=qB 
=C8(�                                    BxV�ۘ  "          @���L(�>�Q�@{BffC-�H�L(��8Q�@\)BC75�                                    BxV��>  
�          @�33�C33>L��@-p�B&
=C0+��C33�\@,(�B$��C;�                                    BxV���  
�          @��R�QG�=u@(��BffC2���QG��   @%B{C<�R                                    BxV��  �          @�z��Z=q>�(�@�B(�C,Ǯ�Z=q��\)@z�B��C5)                                    BxV�0  �          @�33�j�H?�=q?��A��HC#�f�j�H?E�?��RA�33C({                                    BxV�$�  
(          @�(��\(�?=p�@��A�
=C'��\(�>�=q@  BC/�{                                    BxV�3|  T          @�  �e�?��@�A���C*���e�=�@��B �RC2(�                                    BxV�B"  "          @��R�j=q>�G�?��HA�(�C-5��j=q    @ ��A�(�C3��                                    BxV�P�  
Z          @��Q녿z�@9��B$
=C=���Q녿�Q�@-p�B
=CG�3                                    BxV�_n  �          @�z��^{�\@*=qB�\C:T{�^{�s33@!G�B
=CCc�                                    BxV�n  "          @�(��dz�8Q�@!G�B  C6���dz�:�H@�HB��C?}q                                    BxV�|�  �          @���`  �333@ffB�C?B��`  ��Q�@��A���CF�{                                    BxV��`  �          @��\�g
=?B�\?�G�A�=qC(��g
=>�Q�?��A��
C.L�                                    BxV�  "          @�33�g
=@�>��@��
C���g
=@{?k�AB�HC^�                                    BxV�  
�          @��
�l(�@p�?�\@�  C��l(�@33?z�HAQC��                                    BxV�R  "          @�Q��a�@�>���@�{C�)�a�@�?fffAD��CJ=                                    BxV���  
Z          @�33�Q�@(�>.{@=qCO\�Q�@ff?5A�HCk�                                    BxV�Ԟ  6          @���J=q@(�þ����ffC)�J=q@*=q>8Q�@!G�C�
                                    BxV��D  �          @�p��G
=@+��(��ffC33�G
=@0      =#�
C
}q                                    BxV���  
Z          @��
�L��@{�J=q�0Q�CO\�L��@%�aG��C�
C\                                    BxV� �  T          @��
�AG�@'��aG��F�HC{�AG�@0  ��\)�xQ�C	�3                                    BxV�6  6          @�33�.�R@;��\(��@��C��.�R@C33�8Q��#33C��                                    BxV��  
�          @��H�=p�@*=q�c�
�J{C
)�=p�@2�\��\)�x��C�R                                    BxV�,�  T          @�(�� ��@L(��333�p�C 0�� ��@QG�<�>�p�B�\                                    BxV�;(  �          @������@H�ÿ
=q��\)B������@K�>8Q�@ ��B�B�                                    BxV�I�  "          @��\�"�\@G��.{�p�C+��"�\@L(�=#�
?�C �                                    BxV�Xt  
�          @y���(�@*�H���Σ�Cs3�(�@>{���
�x��B��
                                    BxV�g  T          @\)���@U��Q����B�ff���@U�>���@�
=B�u�                                    BxV�u�  �          @�Q��ff@QG��c�
�O33B��=�ff@X�þ���z�B�R                                    BxV�f  
(          @~{�G�@Vff�������B�L��G�@W�>���@��B�                                      BxV�  �          @�=q��@W��&ff�ffB�����@Z�H>�?��B��                                    BxV�  
(          @�G�����@e��\)��p�B�\����@s�
����{B��H                                    BxV�X  	B          @y����Q�@\�Ϳ�ff�{�B�=q��Q�@fff�u�c33Bߊ=                                    BxV��  �          @���(�@s33��  ���HB���(�@~�R��33����B�33                                    BxV�ͤ  
<          @����G�@�G����R����B֙���G�@��R��z��n�RB�=q                                    BxV��J  
�          @��׾aG�@��׿�Q����RB��aG�@�\)��
=��z�B��q                                   BxV���  T          @�G�����@�=q��ff��G�B��R����@�  ��=q�]p�B�\)                                    BxV���  T          @��׿:�H@��ͽ��Ϳ��
B��:�H@��?h��A<��B�.                                    BxV�<  
�          @�녾�
=@�p�?n{AI�B�k���
=@u?�\)A�=qB�aH                                    BxV��  
�          @�33�\(�@�(�?Q�A/�B�uÿ\(�@u�?�  A��
B�=q                                    BxV�%�            @�Q쿠  @�(�?.{A\)Bծ��  @w
=?�\)A��B��)                                    BxV�4.  
�          @{��AG�?��H������  C5��AG�?���Q����RC�f                                    BxV�B�  
�          @}p��&ff@$zῷ
=���RC@ �&ff@5��J=q�:�RC}q                                    BxV�Qz  T          @���C33�Tz��%�(�CC@ �C33�W
=�-p��&33C7�                                    BxV�`   �          @�  �HQ�5�I���2(�C@���HQ�=#�
�N�R�7�HC3E                                    BxV�n�  	�          @�\)�J=q��p��:=q�"�CIG��J=q���G
=�0�HC=\)                                    BxV�}l  "          @�G��1G���=q�E�8
=CIW
�1G���{�P  �EQ�C:�f                                    BxV�  
�          @���S�
�k��(���CC�{�S�
�����%��RC9��                                    BxV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�ƪ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��P              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�-4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�;�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�J�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�Y&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�g�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�vr              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��V              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�&:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�4�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�C�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�R,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�`�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�ox              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�-�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�<�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�K2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�Y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�h~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�w$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�ݮ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��T              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�	�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�&�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�5�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�D8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�R�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�a�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�p*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�~�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�ִ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxV�.�            @�G���>�ff�E���C.xR��?!G��!G���C,G�                                    BxV�=>  �          @�����
=?�녿^�R�'�C"�q��
=?Ǯ�����\C �R                                    BxV�K�  	�          @�����(�?8Q�O\)���C+!H��(�?h�ÿ������C(�                                    BxV�Z�  T          @�Q����\?@  ����JffC*�3���\?}p��Q��{C'�=                                    BxV�i0  �          @�33��\)?.{�L����
C+�
��\)?\(������C)�3                                    BxV�w�  
�          @������?E��
=q��(�C*�����?aG������w
=C)�{                                    BxV��|  �          @��\��p�?xQ������C(=q��p�?���W
=�\)C'5�                                    BxV��"  
�          @�\)���H?�\�xQ��>ffC-�����H?=p��O\)�ffC*޸                                    BxV���  "          @��R���\?z�Y���'�C,�����\?G��.{���C*c�                                    BxV��n  i          @�  ���>�׿E��{C.33���?&ff�!G����HC,                                      BxV��            @���=q?�\�J=q��C-�H��=q?0�׿!G�����C+Y�                                    BxV�Ϻ  
�          @�G���?s33�u�AG�C'�)��?z�H<#�
>#�
C'�                                     BxV��`  
�          @�  ��p�?G��L���\)C)����p�?L��<#�
>\)C)�                                    BxV��            @�=q����?zὸQ쿘Q�C,������?
==u?:�HC,�{                                    BxV���  
�          @��H���>�Q쾞�R�uC/� ���>�(��L���#�
C.��                                    BxV�
R  
�          @������>�(��L�Ϳ(��C.�)����>�(�=u?=p�C.��                                    BxV��  	�          @�\)��
=>u=�Q�?�\)C0����
=>W
=>��?��C1G�                                    BxV�'�  �          @�p�����>�\)>�=q@p  C0(�����>L��>���@��C1@                                     BxV�6D  �          @��R���
���ÿ0����
C8�����
�\)�@  �%G�C5�                                    BxV�D�  
(          @��R���\�G��
=q��33C>�\���\��R�8Q��C<�{                                    BxV�S�  �          @���~{��>�\)@s�
CDc��~{��������CD�\                                    BxV�b6  T          @�\)��G��(�?z�HAX��C<����G��W
=?L��A.ffC?                                    BxV�p�  �          @��vff�B�\������(�C6� �vff>.{������ffC1��                                    BxV��  �          @�p��7�?����-p��$\)C^��7�?�����
�ffC&f                                    BxV��(  "          @��
�Q�?�ff�:=q�6�Cٚ�Q�@  �=q��RC�
                                    BxV���  �          @��Ϳ�(�?����\(��`z�C����(�@33�AG��:��C��                                    BxV��t  �          @r�\�,�;8Q����"�RC7��,��>�
=�33� �C++�                                    BxV��  
�          @�G��E�>k���R�  C/��E�?\(����\C$\)                                    BxV���  "          @�(��QG�?n{�z��	z�C$.�QG�?��R� ����p�Cs3                                    BxV��f  T          @���^�R?W
=�����ffC&aH�^�R?�ff��{���C��                                    BxV��  "          @|(��`  >����H�Ώ\C,�=�`  ?aG����
��  C%ٚ                                    BxV���  �          @z=q�U�!G������Q�C>���U�����R���C6�                                    BxV�X  T          @p���]p��5���H��\)C?���]p��\������  C:8R                                    BxV��  T          @s33�^�R�Q녿�Q���33CAL��^�R���H��{��\)C;�q                                    BxV� �  "          @����p  ?B�\�z�H�f�RC(���p  ?z�H�@  �0z�C%E                                    BxV�/J  "          @y���g�>�(����\�y�C-O\�g�?+��aG��U�C)xR                                    BxV�=�  T          @|(��Tz�+���=q��33C?xR�Tz�8Q������z�C7
                                    BxV�L�  T          @�  �Y��?�=q��
=��33C"T{�Y��?�p��������C��                                    BxV�[<  
�          @|(��@  ?h�����(�C#33�@  ?��R�33��{C��                                    BxV�i�  
Z          @�(��C33?�\)���G�C�C33?�Q��
=��p�C��                                    BxV�x�  T          @��R�4z�@z���R�홚C���4z�@ �׿����  C
E                                    BxV��.  �          @�Q��(Q�@ ���{���CaH�(Q�@@  ��  ����CL�                                    BxV���  �          @��\��@'���
=��
=C  ��@A녿�Q����
B��)                                    BxV��z  �          @\�����?���{��(�C�����@(���=q���C	�
                                    BxV��   T          @\���?\)?���>��@�G�C�3�?\)?���?=p�AN�HC#�                                    BxV���  
�          @_\)�>�R�L��?�Q�A�Q�C4��>�R��?�\)A�C=0�                                    BxV��l  T          @g
=�J�H�^�R?�  A���CCE�J�H���?=p�AF�RCG�                                    BxV��  T          @o\)�g�?�?+�A'33C+�R�g�>���?J=qAC�C.��                                    BxV���  
�          @p���g�?0��>�ff@߮C)33�g�?��?�RAQ�C+Y�                                    BxV��^  T          @s�
�Tz�?�(��(���33C��Tz�?�=q�B�\�8Q�C0�                                    BxV�  T          @y���A�@	����G���z�C�
�A�@���+����C��                                    BxV��  
Z          @z=q�<��?�p���\)��Q�C!H�<��@�����{33C��                                    BxV�(P  T          @u��.{>����(���0{C-�R�.{?xQ���R�"�C \)                                    BxV�6�  �          @�\)�-p�?��
�)���ffC�3�-p�@���
=��C
h�                                    BxV�E�  "          @�ff�/\)@��J�H�*�HCٚ�/\)@4z��!��
=C#�                                    BxV�TB  
�          @w
=�p��?�  �U��x{B�G��p��@�
�5��B�B�k�                                    BxV�b�  �          @|(����?s33�R�\�}�
C�3���?�\�;��S�B�p�                                    BxV�q�  T          @\)���?��
�-p��.��CG����@���R�
(�C	�
                                    BxV��4  �          @p����?aG��3�
�@
=C�R��?����\)�$�RC}q                                    BxV���  T          @~{�<(����
�!G��"\)C5�{�<(�?��p��ffC)                                    BxV���  
�          @�Q��   ?����{C�3�   ?��Ϳ�
=���\C�                                     BxV��&  �          @�����>�ff����p�C$)����?�p��s33�p��C	�                                    BxV���  
Z          @�Q쿑�?.{����� C����?�G��|���xQ�B��)                                    BxV��r  "          @�G���
=?����
=
=C
����
=@�s�
�g  B�                                    BxV��  
�          @��\���
�8Q��8���nCM+����
=L���>�R�{=qC2&f                                    BxV��  
�          @��ÿJ=q@:=q�mp��K�\B�B��J=q@qG��5��B˙�                                    BxV��d  "          @��ͿaG�@��z=q�nG�B�
=�aG�@C�
�O\)�5�HB��                                    BxV�
  
�          @�33�\>�33���Rz�C'{�\?��~�R�y�
C�H                                    BxV��  	�          @�  ���
>����\)�3C%�����
?������L�B�(�                                    BxV�!V  T          @�(��aG�?�\)��G�z�B���aG�@��s33�_33B�                                      BxV�/�  
�          @��ÿp��@���{�v  B䙚�p��@E��aG��>{B���                                    BxV�>�  T          @��׿h��@!�����f�\B�p��h��@a��\���-�\B�Ǯ                                    BxV�MH  T          @�  ���@   �����k�B�{���@aG��_\)�1�\B�p�                                    BxV�[�  
�          @�
=��p�@%���R�hz�B�\��p�@e�Y���-ffB���                                    BxV�j�  8          @�����@��\)�sp�B��f���@Vff�_\)�833B�
=                                    BxV�y:  �          @�p�>�\)@����H�x�B��>�\)@S�
�g
=�=�B�k�                                    BxV���  
Z          @�=q�(��@G���G��~�HB��(��@C33�hQ��EffB�B�                                    BxV���  "          @�33�B�\@C�
�O\)�733B�{�B�\@q��ff��ffB��
                                    BxV��,  T          @���Y��@O\)�S�
�2�B�W
�Y��@~�R����(�B�
=                                    BxV���  �          @����(�@.�R�j=q�K{B�33��(�@dz��5��RBٳ3                                    BxV��x  �          @��
��33@Q��b�\�LffC�=��33@>{�8Q���HB�ff                                    BxV��  	�          @�Q쿏\)@r�\�{��
=B��)��\)@�\)��
=�qp�Bє{                                    BxV���  T          @�����R?\(��W
=�Z=qC���R?�Q��A��=
=C�                                    BxV��j  T          @�z��\)?����>{�-{C���\)@(���Q���HCJ=                                    BxV��  �          @���!�@A��(����
C޸�!�@c33��=q��B��                                    BxV��  T          @�G���@tz�����B�G���@�G�����z=qBޏ\                                    BxV�\  �          @��H�xQ�@��ÿ�(�����Bͣ׿xQ�@�z�O\)��
B˳3                                    BxV�)  
�          @��ÿs33@�(�������
B��s33@�����=q�Pz�B˔{                                    BxV�7�  �          @�G���33@�(�����B��ΐ33@��ÿxQ��;33B�aH                                    BxV�FN  �          @��׿�{@�Q��(���=qB�ff��{@�{����S\)B�
=                                    BxV�T�  "          @�33��33@�z��\)��  B�𤿓33@�=q��{�S�B��                                    BxV�c�  "          @�{���
@�G��p���z�B��f���
@�
=����B�RB̔{                                    