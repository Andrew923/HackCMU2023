CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230517000000_e20230517235959_p20230518021847_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-05-18T02:18:47.581Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-05-17T00:00:00.000Z   time_coverage_end         2023-05-17T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�p�  �          @���tz��
=@p  B'CK�3�tz��@��@333A�CZ+�                                    Bx�p)&  T          @���5��@�A�ffCn�q�5���
=���?uCp��                                    Bx�p7�  �          @�33�Fff���\@�HA��Cjh��Fff��?��@�p�Cm�)                                    Bx�pFr  �          @��
�Fff��p�@��A��Cj��Fff��ff>�Q�@e�Cm�                                    Bx�pU  �          @�=q����Q�?�
=A���Ct\�����׾Ǯ�vffCu0�                                    Bx�pc�  T          @�������  @p�Aʣ�Cu������\>�(�@���CxL�                                    Bx�prd  �          @����
=��{@  A��CsG��
=��{>��@(Q�Cu��                                    Bx�p�
  �          @���Q���p�@
=A�  Cg�\�Q���Q�?\)@�=qCkk�                                    Bx�p��  �          @�Q��8Q����?�33A��Cn��8Q���\)����{Co�q                                    Bx�p�V  
u          @���9����  ?k�A��Coٚ�9����
=��=q�*�\Co�                                    Bx�p��  �          @��
�Tz�����
=q��33Ck  �Tz��������33Cgn                                   Bx�p��  
�          @�
=�^{��Q�����Chs3�^{�~�R�
�H���HCd�3                                   Bx�p�H  y          @�p��aG����Ϳ.{��33CgL��aG��r�\�z����Cc!H                                    Bx�p��  �          @���W
=���R>���@E�Ch�q�W
=��\)��Q��o
=Cg��                                    Bx�p�  �          @����*=q���ÿ}p���Cr(��*=q��Q��6ff��RCn�                                    Bx�p�:  T          @�  �@  ���ÿ�=q�-�Cm��@  ��  �5��RCi&f                                    Bx�q�  �          @�
=�4z����ÿ����Z�HCop��4z��x���E��Cj�                                    Bx�q�  �          @��R�8Q���Q쿝p��G
=Cn�\�8Q��z�H�=p�����Ci�3                                    Bx�q",  �          @�33�L����G��\)���Cj��L���~{������HCg!H                                    Bx�q0�  �          @����X�����=�?�Q�Ch��X����=q�������
Cf0�                                    Bx�q?x  �          @��H�XQ���ff>\)?�(�Ch�XQ���p���=q���HCf��                                    Bx�qN  �          @�(��U�����>�ff@��Ci���U���33�����X  Ch�
                                    Bx�q\�  �          @����P����ff�u���Ci�
�P�����H��G�����Cg�                                     Bx�qkj  �          @����9����{��\)�<(�Cn@ �9����
=��
����Ck��                                    Bx�qz  T          @�33�_\)��z�?�p�AMG�Ce���_\)������G����HCf�                                    Bx�q��  �          @��H�n{�|��?�  AO\)Cb�3�n{��(���Q��j=qCc��                                    Bx�q�\  �          @��\�|���fff?��HAs�C^W
�|���x�ü����
C`�                                     Bx�q�  �          @����tz��R�\@G�A��
C\���tz��y��?W
=AQ�Ca��                                    Bx�q��  �          @�=q�u��,(�@>�RB{CW��u��hQ�?�p�A��C_n                                    Bx�q�N  �          @����z�H�$z�@<��B ��CU:��z�H�`  ?�  A�p�C]�                                     Bx�q��  "          @���(Q��=p�@�  B5G�Cd^��(Q���Q�@$z�A��
CnO\                                    Bx�q��  "          @�p��*=q�P  @Z=qBp�Cf�R�*=q����?�A�ffCnL�                                    Bx�q�@  �          @����tz���
@I��B�CS+��tz��U�@�A�z�C]
                                    Bx�q��  T          @����ÿ�z�@_\)B33CMT{�����E@   A�{CYz�                                    Bx�r�  �          @�Q��w
=����@��B6�HCC�3�w
=�%�@X��B�CUǮ                                    Bx�r2  "          @�
=�?���@�  Bp�HCz���8Q�@�=qBw33CE                                      Bx�r)�  y          @�\)�AG�>���@���B`�C-�
�AG���@��\BQz�CM(�                                    Bx�r8~  
�          @����HQ쾞�R@��\B^�C9�
�HQ�� ��@��
B?�CT�=                                    Bx�rG$  �          @���Q녿У�@��B?��CN���Q��C33@H��B=qC^�                                    Bx�rU�  y          @����S�
��(�@��B;=qCR��S�
�XQ�@C�
B�Ca�\                                    Bx�rdp  
�          @���A��(�@�
=BACW���A��g�@E�Bp�Cf�                                    Bx�rs  �          @�p��=p��Q�@�=qBG=qCW�R�=p��fff@L��B	Cf��                                    Bx�r��  
�          @��\�X���p�@�p�B7\)CU!H�X���g�@A�A�G�Cb�H                                    Bx�r�b  
�          @�\)�Tz��
=@�B;  CTff�Tz��aG�@E�B�
Cb��                                    Bx�r�  "          @�p��P�׿�\)@��BA��CQ�=�P���Tz�@N�RB
�HCa�=                                    Bx�r��  T          @�p��U����H@�Q�BB��CO33�U��K�@S33B��C_�q                                    Bx�r�T  �          @���dz��33@|(�B1
=CP
=�dz��O\)@<(�A���C^=q                                    Bx�r��  "          @�  �p  �#33@e�BQ�CV:��p  �mp�@
=A�\)C`��                                    Bx�r٠  �          @����dz��(Q�@g
=Bz�CXff�dz��r�\@
=A��HCb�R                                    Bx�r�F  T          @�=q�g
=��@z=qB(��CUY��g
=�j=q@/\)A�(�Cas3                                    Bx�r��  �          @�{�u�%@n�RB�\CU�R�u�r�\@   AƏ\C`��                                    Bx�s�  �          @��
�|(���@�33B(33CR{�|(��i��@<��A�RC^��                                    Bx�s8  �          @��H������?��
AB=qCc�������׾�G����
Cd{                                    Bx�s"�  
�          @���y�����H?c�
A
=qCd
=�y����33�J=q���Cd&f                                    Bx�s1�  �          @��
���H���?0��@�=qCa&f���H���\�c�
��C`�H                                    Bx�s@*  �          @�33��{�`��?�=qAS
=CZJ=��{�p  ��\)�8Q�C\+�                                    Bx�sN�  �          @�{�y������=�G�?�ffCe8R�y�������=q�uG�CcaH                                    Bx�s]v  �          @�
=�o\)����=��
?:�HCg&f�o\)���H��33���RCe@                                     Bx�sl  "          @�33�B�\���Ϳ333��
=Cn0��B�\��G��{��p�Cj�                                    Bx�sz�  �          @�(��C�
��Q쾞�R�Dz�Cn�)�C�
��G��	�����Cl�                                    Bx�s�h  
�          @�(��.�R��ff�\)��{CrQ��.�R�����33���CpG�                                    Bx�s�  
�          @���*=q���>��
@J=qCr�*=q�����{����Cq��                                    Bx�s��  �          @�
=�,�������G���\)Cq�=�,�����H��Q���Q�Co��                                    Bx�s�Z  T          @�{��R��  �n{��Cs����R����-p�����Cp�                                    Bx�s�   "          @�z��=p����ý��Ϳz�HCnB��=p����������Cl#�                                    Bx�sҦ  T          @�Q��[����
=�Q�?p��Cik��[����\��{��=qCg�H                                    Bx�s�L  T          @��R�Vff��=q?!G�@�33Ci�R�Vff��\)����0��Ci0�                                    Bx�s��  T          @������P�׾���/\)CYaH�����<(���Q��w�
CV�=                                    Bx�s��  �          @��R��\)�[���\)�333C[  ��\)�J�H�����`(�CXǮ                                    Bx�t>  T          @�Q��dz���(�����  Ce��dz��i���   ��33Ca�
                                    Bx�t�  �          @��`����(�>L��@�Ce�3�`���z�H��=q�bffCd0�                                    Bx�t*�  T          @��R�I����ff<�>�z�Cj���I�����Ϳ�������Ch�=                                    Bx�t90  "          @���U���  >\)?\Cg��U���Q쿵�s�CfO\                                    Bx�tG�  �          @���^�R����?B�\AG�CeJ=�^�R�����E���CeE                                    Bx�tV|  T          @�Q�����X�ÿ0����{CZ������:=q�����(�CVs3                                    Bx�te"  �          @���g��e?�\)A���C`���g��{�>L��@��CcW
                                    Bx�ts�  T          @����\���P  @!G�A�  C_B��\���|(�?�AH��Cd�=                                    Bx�t�n  �          @���~{�Tz�?޸RA�=qC[��~{�n�R>��@��C_.                                    Bx�t�  �          @���333�J=q@`  B�RCd� �333���R@�A���Clp�                                    Bx�t��  y          @����R�G�@�z�B<��Cjff��R���@.�RA��Cs!H                                    Bx�t�`  �          @���*=q�E�@j�HB'�Ce+��*=q��ff@33AƸRCm��                                    Bx�t�  �          @�\)�C�
�B�\@VffBG�C`�
�C�
����@G�A�  Ch��                                    Bx�tˬ  �          @���)���<(�@}p�B3�
Cd��)����p�@(Q�A��
Cm�
                                    Bx�t�R  �          @��
�/\)�>�R@x��B/Q�Cck��/\)��{@#33A�=qCl�                                    Bx�t��  T          @���"�\�<(�@}p�B6(�Ce.�"�\��p�@(Q�A��HCn��                                    Bx�t��  "          @�{�   �?\)@r�\B0�\Cf{�   ����@p�A���Cn�                                    Bx�uD  �          @��
��\�AG�@p��B2�HChǮ��\��p�@�HA׮Cq+�                                    Bx�u�  �          @�p��
�H�+�@���BH��Cg�
�H�\)@:=qB�Cqu�                                    Bx�u#�  �          @�z��!G��-p�@x��B9Cc!H�!G��{�@*=qA홚CmT{                                    Bx�u26  �          @���8Q��;�@[�Bz�Cap��8Q��}p�@
=qA�{Ci��                                    Bx�u@�  �          @���7��,(�@mp�B-G�C_8R�7��u@   A�z�CiE                                    Bx�uO�            @���0���+�@h��B-�\C`
�0���s33@(�A�Ci��                                    Bx�u^(  a          @����1��(��@h��B.(�C_�\�1��qG�@p�A��Ci�
                                    Bx�ul�  �          @�Q��q��xQ�8Q����
Ca���q��W
=��
����C]��                                    Bx�u{t  �          @�=q�qG����\�aG��\)Cc5��qG��n�R��33��ffC`��                                    Bx�u�  �          @���n�R����\)��(�Cc���n�R�r�\�˅��p�Ca}q                                    Bx�u��  T          @��\�y���~�R�u�!G�Ca���y���hQ�������C^�R                                    Bx�u�f  �          @�  �p  �j=q�aG���C`E�p  �Vff��  ����C]�3                                    Bx�u�  �          @�G��`���fff=#�
>�Ca�q�`���XQ쿝p��c\)C_�3                                    Bx�uĲ  �          @����k��p�׽�Q쿀  Ca�f�k��_\)��z��y��C_z�                                    Bx�u�X  �          @�33�tz��p�׼��
�aG�C`���tz��`�׿���h��C^�)                                    Bx�u��  �          @�  �{��\��=L��?��C]:��{��O\)��z��M��C[�                                    Bx�u�  T          @�
=�i���j�H?��@��Ca(��i���i���:�H�   C`�q                                    Bx�u�J  �          @���l(��k�>��@�  C`��l(��e�h��� (�C`5�                                    Bx�v�  T          @�
=�g��mp�?\)@�(�Ca���g��j�H�G��	Cah�                                    Bx�v�  �          @�{�h���G
=?�z�AYG�C\xR�h���Tz�#�
�uC^O\                                    Bx�v+<  �          @�{�Q녿�  @>�RBp�CL���Q����@G�A�  CX�{                                    Bx�v9�  �          @�G��S�
��33@J=qB%=qCJ޸�S�
�=q@{A���CX�                                    Bx�vH�  
�          @��H�Dz��
=@K�B$�RCT��Dz��:�H@33A�(�C_}q                                    Bx�vW.  �          @��\�S�
��
=@;�B��CR0��S�
�5�@z�A��
C\z�                                    Bx�ve�  
�          @���J=q��33@>{B�CS��J=q�4z�@Q�Aՙ�C]�q                                    Bx�vtz  
�          @��_\)�;�?�\A�C[�R�_\)�W
=?#�
@�C_��                                    Bx�v�   y          @����e�O\)?�ffAqG�C^\�e�_\)=���?�\)C`0�                                    Bx�v��  �          @�z��w
=�L��?fffA#�C[���w
=�Tzᾅ��<(�C\��                                    Bx�v�l  T          @�ff�Z=q��\)>��@���Cg
�Z=q�����=q�733CfG�                                    Bx�v�  �          @���N{���ÿL�����Ci��N{�o\)��R����CeJ=                                    Bx�v��  �          @��?\)���O\)�	G�Ck�R�?\)�xQ���\��{Ch^�                                    Bx�v�^  
�          @�z��S�
��ff>��H@�  Cg� �S�
����z�H�)�Cg(�                                    Bx�v�  T          @��
�S�
����>�@��Cgu��S�
��녿}p��+�
Cf��                                    Bx�v�  �          @���I�����
��G��Z�RCh�
�I���\(��&ff���Ccz�                                    Bx�v�P  �          @�(��Y����p�=��
?\(�Cf���Y���|(������i�Ce@                                     Bx�w�  
�          @��
�Tz���ff�\)���
Cg���Tz��y���������\Ce��                                    Bx�w�  �          @��H�J=q��G���{�j=qCi��J=q�z�H�����z�Cg&f                                    Bx�w$B  T          @��\�E���\�L���	��Cjz��E��  �����
ChQ�                                    Bx�w2�  �          @�p��333���\��R�ӅCn�H�333��33�	����G�Ck��                                    Bx�wA�  �          @����(Q�����=p����HCp�
�(Q���(����ǮCm��                                    Bx�wP4  
�          @�p��,(��������Cp)�,(���������
Cm��                                    Bx�w^�  �          @��R�   ��=q����  Cr�{�   ���H�
�H���Cp\                                    Bx�wm�  T          @���H���<��
>W
=Cs����H���\�У����Cr!H                                    Bx�w|&  T          @��R�   ���H<#�
>��Cr���   ��녿�\)����Cq@                                     Bx�w��  T          @��
�*=q��(�>\@�=qCp
�*=q������H�P(�CoO\                                    Bx�w�r  �          @�(��:�H��\)?�@�33Cl�)�:�H����xQ��&�\Cls3                                    Bx�w�  T          @���Fff��Q�?xQ�A'�Ci��Fff���\������Cjp�                                    Bx�w��  �          @�33�@  ����>.{?�Ck�q�@  ��ff�����e�Cj�                                     Bx�w�d  
�          @��\�>�R���>8Q�?��RCk���>�R��
=����c
=CjǮ                                    Bx�w�
  "          @����=p���z�    ��Cl  �=p���zῼ(���33Cjff                                    Bx�w�  �          @�G��E����׾���\)Cj&f�E��w���{����Cgp�                                    Bx�w�V  T          @��H�C�
���
�W
=�{Cj��C�
������33��Q�Ch޸                                    Bx�w��  
�          @��H�I����녽��
�W
=Ci�{�I�����ÿ�G���p�Ch�                                    Bx�x�  "          @�\)�/\)��=q?��A;�Cm���/\)����G���G�CnE                                    Bx�xH  T          @����0������?L��A  Cm
�0����G��(����33Cm=q                                    Bx�x+�  
�          @����0����\)?k�A%�Cl���0����������{CmQ�                                    Bx�x:�  �          @����5���p�?��A;33Ckٚ�5����þ������Cl��                                    Bx�xI:  �          @�{�2�\��Q�?�  A2ffCl� �2�\��33�����HCmO\                                    Bx�xW�  "          @���.{��Q�?���Ao�
Cms3�.{��
=����33Cn�                                    Bx�xf�  �          @�ff�.�R��?��HA�Q�Cl�f�.�R��        Cnc�                                    Bx�xu,  "          @��1G���33?��
A��Ck���1G����
=���?�=qCm�)                                    Bx�x��  
�          @�{�0  ����?�ffA�(�Ck���0  ��z�>\@�{Cm�                                    Bx�x�x  �          @��>{�x��?�
=A�  Ch�f�>{��\)>��
@`��Cj�                                    Bx�x�  �          @�p��Dz��r�\?�Q�A�
=Cg��Dz���z�>�Q�@~�RCiu�                                    Bx�x��  �          @�=q�@  �u�?�Q�A��Cg޸�@  ���H=�G�?�p�Ci�                                    Bx�x�j  �          @����@���s33?�ffAo�
Cg���@�����׼#�
��Ci+�                                    Bx�x�  �          @�p��>{�l(�?�=qA|  Cg(��>{�z�H=u?:�HCh޸                                    Bx�x۶  �          @��\�@  �^{?��HA��Ce5��@  �p��>��@C�
Cgs3                                    Bx�x�\  �          @�{�AG��h��?�Q�A���Cf\)�AG��z�H>8Q�@�Ch\)                                    Bx�x�  "          @�  �Dz��Q�?\A�G�Cb�H�Dz��e>�p�@���Ce}q                                    Bx�y�  �          @���>�R�Y��?�A��
CdǮ�>�R�j�H>�  @@  Cf�q                                    Bx�yN  �          @�z��'��a�?��HA��\Ciff�'��s�
>u@=p�Ckz�                                    Bx�y$�  �          @�ff�\)�P��?�z�A�z�Ch���\)�g�?�@ٙ�Ck�                                    Bx�y3�  �          @���У�?�=q@Mp�BR�\C��У�?333@hQ�B�C�                                    Bx�yB@  �          @�Q��?���@O\)BY\)Cn��>���@a�By�\C)��                                    Bx�yP�  �          @�����p�?�\)@L(�B^{Cs3��p�?�\@b�\B�Q�C �R                                    Bx�y_�  �          @l(��
�H>\)@>{BW33C0:��
�H�Q�@7
=BK��CH��                                    Bx�yn2  �          @�(���  ?z�H@@��Ba�Cٚ��  ��@J=qBs��C5{                                    Bx�y|�  �          @�����R@S33@@  B!��B�����R@
�H@y��Be=qB�L�                                    Bx�y�~  �          @�Q쿑�@c33@VffB'�HB׮���@33@��Bl�
B��
                                    Bx�y�$  �          @�{���H@U�@[�B0p�B�\���H@z�@��BsB��
                                    Bx�y��  �          @��\����@G
=@Y��B4G�B�  ����?��@�
=Bt��B���                                    Bx�y�p  �          @��R��G�@_\)@J�HB�B����G�@�
@��B`ffB�aH                                    Bx�y�  �          @�\)���R@g�@Dz�Bz�B�Ǯ���R@{@��BZz�B�Q�                                    Bx�yԼ  �          @��R�ٙ�@Q�@~�RBV�\B�{�ٙ�?�G�@���B�ǮCW
                                    Bx�y�b  �          @�z�ٙ�@Q�@��RB\�B��ٙ�?p��@�  B��\C#�                                    Bx�y�  �          @�G���\)@#33@��HB\�RB��Ϳ�\)?��@�p�B��C��                                    Bx�z �  "          @���   @�@�(�Bc�
C���   ?\)@���B�8RC$n                                    Bx�zT  �          @�(��!G�?���@��BY�C���!G�>aG�@�ffBqC/                                    Bx�z�  �          @��
�AG�?�(�@mp�B;\)CO\�AG�>�@��BT
=C*�R                                    Bx�z,�  �          @��\�J�H?�(�@j�HB9�C
�J�H>��@|��BL��C/c�                                    Bx�z;F  T          @��\�j=q?�p�@R�\B!�RC!z��j=q>\)@`��B/  C1�                                     Bx�zI�  �          @�=q�Z=q?�Q�@a�B1Q�C ���Z=q<�@n�RB>  C3k�                                    Bx�zX�  �          @����N{?�z�@i��B;�C 
�N{���
@u�BH
=C4ff                                    Bx�zg8  T          @�z��G
=?�=q@w
=BC=qC�)�G
==�\)@��\BR�HC2��                                    Bx�zu�  �          @�G��.�R?^�R@�=qB[z�C"^��.�R��(�@�z�Ba�C<�                                    Bx�z��  "          @�G��<(�?h��@h��BG  C"�<(���  @p  BN��C8޸                                    Bx�z�*  �          @�Q��1�>��@u�BV��C*s3�1녿:�H@r�\BSG�CB��                                    Bx�z��  �          @�33�1�?+�@z=qBW=qC&p��1녿\)@z�HBX�C?^�                                    Bx�z�v  "          @�(��A�>��
@]p�BB�C-�H�A녿@  @X��B=��CA�H                                    Bx�z�  T          @�{�QG�>���@S�
B4=qC-)�QG��!G�@QG�B1�C>�                                    Bx�z��  �          @���5����@h��BN�
C=Q��5���ff@Tz�B7=qCP��                                    Bx�z�h  T          @��H�˅��  @�Q�B{ffC_c׿˅�.�R@g
=BC33Co                                    Bx�z�  �          @�
=����  @��Bu��C[E���0  @l��B@�\ClE                                    Bx�z��  �          @��R��p�@xQ�BOCb�f��Q�@A�BCm��                                    Bx�{Z  �          @�(���Q��{@��
B~��Cd=q��Q��6ff@k�BD(�CsG�                                    Bx�{   �          @�녿�\)�L��@��RB��CR.��\)�ff@�Q�Bg��Cl�3                                    Bx�{%�  �          @���Ǯ�#�
@��B��HCJz�Ǯ�   @��RBk�Ch!H                                    Bx�{4L  �          @�������z�@\)BS�C[^����AG�@O\)B!=qCh^�                                    Bx�{B�  "          @��H���p�@��B\{CX�\��9��@\��B+33Cg�                                    Bx�{Q�  �          @�ff�%��Q�@��BM�\CX�{�%�Dz�@R�\B\)Ce�                                    Bx�{`>  �          @�(��,���ff@r�\B?��CY�H�,���H��@?\)BffCeB�                                    Bx�{n�  �          @�z��   �)��@g
=B2�RCb���   �fff@)��A�G�Ck8R                                    Bx�{}�  
�          @�=q�(���)��@W
=B'�RCa��(���a�@�HA�=qCi33                                    Bx�{�0  �          @��2�\��@dz�B;��CUn�2�\�4z�@7�Bz�CaG�                                    Bx�{��  �          @����z��L��@7
=B��Cm��z��z=q?��A�=qCr{                                    Bx�{�|  �          @����)����R@g�B933C\\�)���L(�@333B�CfY�                                    Bx�{�"  �          @����1G�����@o\)BA��CUO\�1G��5@B�\B�Ca��                                    Bx�{��  �          @��R�AG��33@Mp�B!
=CYW
�AG��H��@��A�z�Cb.                                    Bx�{�n  
�          @���U��У�@G�B p�CN!H�U��{@ ��A�33CX��                                    Bx�{�  
�          @��H�^�R�u@C�
B �HCCz��^�R����@(��B��CO�=                                    Bx�{�  T          @��
�`  ?�Q�@G�B �
C!J=�`  >aG�@U�B.33C0^�                                    Bx�|`  �          @�
=�dz�>���@qG�B9��C-��dz�8Q�@n{B6�C?ff                                    Bx�|  �          @�Q��dz᾽p�@uB;�C9��dzῼ(�@dz�B*�HCJ^�                                    Bx�|�  �          @�ff�dzῳ33@`  B)��CI^��dz��ff@<��B
Q�CUaH                                    Bx�|-R  
�          @����\�Z�H@
=A�G�Co&f��\�|��?��\A
=Cr��                                    Bx�|;�  �          @��ff��z�?�z�A|z�Cq�)�ff���=��
?c�
Cs                                      Bx�|J�  T          @���������?�
=A�(�Cqp������H>\@�{Cs\                                    Bx�|YD  T          @�G��'
=�w�?��A�=qCk�R�'
=��  ?333@�
=Cnh�                                    Bx�|g�  T          @����!��|(�?�A��CmQ��!�����?#�
@��Co��                                    Bx�|v�  �          @�{�   ����@A�ffCn+��   ���R?Y��A�Cp��                                    Bx�|�6  �          @�G��ff��p�@ffAծCsG��ff��p�?��A;�CuǮ                                    Bx�|��  "          @��R�'����?�A��CmxR�'����R?
=@�Q�Co��                                    Bx�|��  �          @��\�����?�AV�\Cq.����
=���Ϳ���Cr�                                    Bx�|�(  
�          @��\�����p�?�(�A`  Ct)������\���
�^�RCt�                                    Bx�|��  T          @�(��ff���H?��A���Cr�H�ff��z�>��@�33CtxR                                    Bx�|�t  T          @�������H?�
=A�G�Cs�f������>�ff@��CuB�                                    Bx�|�  
�          @�33�ff�~�R?��A�G�Cr.�ff��=q?��@��Ct{                                    Bx�|��  �          @�����y��@AɅCqJ=������\?h��A+\)Cs�q                                    Bx�|�f  �          @����o\)@�A�  Co������  ?�Q�A_�Cr�                                    Bx�}	  �          @�����w
=@�A�
=Cq�\����?�G�A>�HCt#�                                    Bx�}�  �          @����\�h��@\)A�
=CpǮ��\��{?���A�{Ct
                                    Bx�}&X  �          @��R�33�K�@!G�B
=Cj��33�p  ?��A�ffCnz�                                    Bx�}4�  
�          @������^{@  A��
Cm������}p�?���AmCp��                                    Bx�}C�  T          @�G���33��=q?���A�ffCt녿�33��33>Ǯ@��CvT{                                    Bx�}RJ  �          @��
�{�^{?�ffA��HCmh��{�p��?�@�Coz�                                    Bx�}`�  "          @���>�R��\@G�B(
=CR���>�R�#�
@ ��Bz�C\��                                    Bx�}o�  "          @��R�;���@!�B
��CX���;��2�\?�=qA�G�C_�
                                    Bx�}~<  
�          @���Y���7
=?�A�G�C\
�Y���P  ?uA8  C_��                                    Bx�}��  "          @���I���O\)@ffA�
=Ca���I���l(�?���AQCe�                                    Bx�}��  �          @��\�@  ����@1G�B =qCL!H�@  �33@33B{CVL�                                    Bx�}�.  �          @���&ff>��
@�=qBk=qC-{�&ff�^�R@�  Bd��CF��                                    Bx�}��  "          @���1�>Ǯ@�{B`�C+�q�1녿B�\@�z�B\��CCW
                                    Bx�}�z  
�          @��H�9���aG�@��B\�C8Y��9�����@|(�BL  CL�                                    Bx�}�   �          @��
�2�\��\@�Q�BaQ�C>W
�2�\���@~{BK
=CRn                                    Bx�}��  �          @��
�*�H��ff@�=qB[\)CIn�*�H�
=@h��B;��CZB�                                    Bx�}�l  
�          @�=q�s33����?�p�A�G�C�ff�s33��(�?!G�@ۅC�Ǯ                                    Bx�~  
�          @��H��\)���R@A�z�C}�Ϳ�\)��33?@  Ap�C~�R                                    Bx�~�  �          @����z���33@\)A���C|�R��z����H?�AG\)C~��                                    Bx�~^  T          @�p������Q�@)��A�Q�C|�{�������?���Ag33C~�{                                    Bx�~.  �          @�(���=q��{@.�RA�C}W
��=q��  ?���Ay�C!H                                    Bx�~<�  T          @�=q�����\)@QG�B��C~�Ὲ�����R@ffA�C���                                    Bx�~KP  
�          @��R��ff�\)@>{B  Cx�Ϳ�ff��(�?�A�z�C{��                                    Bx�~Y�  �          @��R�^�R�e�@j=qB3\)C�33�^�R��p�@%A��C�t{                                    Bx�~h�  
�          @�  ��=q�q�@QG�B��CwLͿ�=q��  @
=qA�33Cz�                                    Bx�~wB  �          @��\��{���H@A�B\)C{�\��{���?���A�Q�C}�3                                    Bx�~��  �          @��H�ff�w�@?\)B�
Cqz��ff����?��A��Cu
=                                    Bx�~��  �          @����G��XQ�@`  B&��Co{�G���p�@   A�z�Ct#�                                    Bx�~�4  �          @����
�U@c�
B)  Cnc���
����@$z�A�{Cs��                                    Bx�~��  �          @�  ����K�@e�B,�Cl������  @(Q�A�RCq��                                    Bx�~��  �          @�  ���R�\@b�\B*
=CnY������H@$z�A���Cs��                                    Bx�~�&  �          @����ff�K�@j=qB/=qCl���ff����@-p�A��Cr^�                                    Bx�~��  T          @��ÿ��n{@Mp�BCr�q����@��A�\)Cv��                                    Bx�~�r  T          @��׿��H��G�@9��B��Cw����H����?�\A�
=CyǮ                                    Bx�~�  
�          @������H�\)@A�B��Cv�{���H��(�?�z�A��Cy�                                    Bx�	�  �          @�����ff�b�\@`  B%=qCs���ff��=q@{A���CwW
                                    Bx�d  T          @����Ǯ�O\)@x��B=(�Ct8R�Ǯ���
@<(�B��Cy=q                                    Bx�'
  �          @��H����P  @{�B>{Ct�3�������@>{B�RCy��                                    Bx�5�  T          @��H��ff�E�@}p�B?�HCo�ÿ�ff�\)@C33B�\Cu��                                    Bx�DV  T          @��\�����<��@\)BA�ClxR�����w
=@G
=B�HCs(�                                    Bx�R�  
�          @��H��
=�J=q@|(�B?  CrͿ�
=���@AG�B

=Cw�                                    Bx�a�  �          @�33��{�C33@�=qBG  Cr+���{�~{@J�HB
=Cw�3                                    Bx�pH  T          @�z�����+�@�z�BZz�Co5ÿ����l(�@e�B&�\Cv�H                                    Bx�~�  T          @�p��˅�-p�@��BZ=qCo�׿˅�n�R@fffB&Q�Cv�                                    Bx���  T          @��Ϳ���R@��\Bh=qCpO\���c33@tz�B4
=Cx@                                     Bx��:  T          @�����   @��Bd�Cnh�����c�
@r�\B1ffCv��                                    Bx���  T          @�p���Q���H@�(�Bj�
Co5ÿ�Q��_\)@x��B7G�Cw��                                    Bx���  
�          @�z��=q�\)@�ffBb(�Cm�H��=q�`��@l��B/�\Cu��                                    Bx��,  �          @���\�,��@��
BZ�RCp���\�l(�@dz�B'(�Cw��                                    Bx���  T          @������J=q@I��BCh�R����u�@�Aՙ�Cm�                                    Bx��x  �          @�녿���"�\@�BS�HCi������_\)@[�B#�RCq�R                                    Bx��  �          @����˅��H@�{Bcp�Cl�q�˅�[�@mp�B1�RCu+�                                    Bx���  �          @����
�
=@�33BY{Cd�{��
�Vff@h��B+ffCnaH                                    Bx��j  T          @�������@~�RBD��C`����Tz�@Q�B��Ci�H                                    Bx��   �          @�33����  @��\BlffC[�=���3�
@���BCz�Ch��                                    Bx��.�  �          @�녿�p��3�
@���BE�
Cj����p��l��@N{B  Cq�f                                    Bx��=\  T          @��H�1녿�(�@�
=BP��CS���1��+�@k�B.ffC`�                                    Bx��L  �          @����!G���(�@�33B`��CR5��!G��{@w�B>�C`h�                                    Bx��Z�  T          @����H���@�{B\�\Cf� ���H�X��@o\)B.�Cp                                      Bx��iN  T          @�p����
�?\)@�  BN�Cr�=���
�z=q@Z=qB(�Cx�
                                    Bx��w�  �          @��Ϳ�p��J=q@~{B?33CqE��p���Q�@FffB�\Cv��                                    Bx����  T          @��
���O\)@{�B@
=CvQ쿵���\@C33B��Cz��                                    Bx���@  T          @��H�#�
�ff@���BD��C^�=�#�
�O\)@VffB=qCg��                                    Bx����  
�          @��\�,�Ϳ���@�(�B^33CN(��,���z�@|(�B?�RC\�R                                    Bx����  �          @����,(���p�@���BY{CP�=�,(��(�@s33B9�C^:�                                    Bx���2  �          @�
=��
���H@���ByQ�CR� ��
�\)@�z�BV�Cc��                                    Bx����  T          @���vff��Q�@P��BQ�CK�vff�=q@1�A���CT�                                    Bx���~  T          @����G���{@E�B=qCB�=��G���ff@/\)A�CJǮ                                    Bx���$  �          @�(���녿W
=@\��B�
C?����녿�\)@J=qB=qCI��                                    Bx����  
�          @�\)���@���B}{C4�R���{@���Bp��CMO\                                    Bx��
p  �          @�G��!논�@�z�Bu�C4�
�!녿���@�Q�Bj=qCK}q                                    Bx��  �          @��R�{�8Q�@�ffB�Q�C8���{��G�@�G�Br��CQ�                                     Bx��'�  T          @�Q��녾u@�
=B�B�C9��녿��@�G�Bo�CR�                                    Bx��6b  �          @�\)�;��G�@�\)BYQ�CB�
�;���(�@|(�BD��CRxR                                    Bx��E  T          @�
=�'���  @�  BZp�CQ���'���@r�\B:�HC^�\                                    Bx��S�  �          @����I���}p�@�33BL��CEc��I����33@qG�B6��CS                                    Bx��bT  �          @�{�=p���ff@�z�BS�CG��=p����H@s33B;��CU��                                    Bx��p�  �          @�\)�HQ�Tz�@�33BN��CB��HQ�޸R@s33B:��CQ\                                    Bx���  �          @�ff�`  ��G�@k�B5{CD\�`  ��@W
=B!��CO\)                                    Bx���F  T          @�
=�@�׿s33@���BS(�CE�
�@�׿�{@u�B=(�CS�=                                    Bx����  T          @�  �>{���
@�BT(�CG
=�>{��Q�@uB=�CU
                                    Bx����  �          @�  �=p��#�
@���BZ�
C@(��=p���=q@���BH�
CP�                                    Bx���8  �          @�G��G
=���H@�Q�BV  C<��G
=��
=@�G�BF�CL��                                    Bx����  T          @��
�QG��Ǯ@�\)BPz�C:���QG�����@�G�BC�\CI�q                                    Bx��ׄ  �          @���b�\>�@uB<��C1���b�\�8Q�@qG�B8�RC?�                                    Bx���*  
�          @�33�g
=>���@e�B2�\C-� �g
=���@e�B2z�C:k�                                    Bx����  z          @�G��h��>\@\��B-{C.��h�þǮ@\(�B-{C:�                                    Bx��v  �          @�p��\��?��@r�\B=�C*���\�;�z�@tz�B?
=C8�\                                    Bx��  �          @�33�\(�?#�
@l(�B:
=C)c��\(��8Q�@p  B=p�C6�R                                    Bx�� �  T          @����#�
=�\)@���Bm�C2u��#�
�^�R@��RBeffCF��                                    Bx��/h  T          @���.{=�@�  Bep�C1��.{�O\)@�p�B_33CD�\                                    Bx��>  �          @�=q�0��?(�@�ffB`(�C'}q�0�׾��R@�\)Bb�
C:p�                                    Bx��L�  �          @�G��C33?�@}p�BOC*E�C33��{@~�RBQ=qC:h�                                    Bx��[Z  �          @��
�>{>�Q�@�(�BX{C-#��>{�
=q@��BV��C>E                                    Bx��j   �          @����@  >B�\@�Q�BT�\C0aH�@  �.{@}p�BP��C@�                                    Bx��x�  T          @�  �>�R>�G�@\)BT{C+���>�R��(�@�  BT(�C<(�                                    Bx���L  �          @����:=q?��@���BV�RC(J=�:=q��=q@��\BYz�C9Y�                                    Bx����  T          @�Q��>�R?�@vffBI  C�\�>�R>���@�Q�BT�HC-                                    Bx����  �          @�\)�K�?��H@h��B;�RC��K�>��@s�
BG��C,��                                    Bx���>  T          @�G��Fff?�G�@p  B@�
C�f�Fff>�G�@{�BM�C+��                                    Bx����  �          @�Q��7
=?��H@uBH�C���7
=?�R@��BY  C'��                                    Bx��Њ  �          @�Q��*�H?�@x��BL\)C  �*�H?O\)@��B`��C#�                                    Bx���0  �          @��p�?�{@vffBM=qC���p�?�G�@���Bep�C�)                                    Bx����  �          @��R�3�
?�p�@dz�B8�C�
�3�
?���@y��BO�C��                                    Bx���|  �          @�z��Q�?��H@q�BK\)Cn�Q�?���@��BeffCxR                                    Bx��"  �          @��
��33?��@��B�u�C���33>\@�G�B��C'�                                    Bx���  �          @��
�xQ�?��@���B{B�Ǯ�xQ�?s33@��HB��C�)                                    Bx��(n  
�          @�
=�8Q�@@��Bv�
B���8Q�?��@�  B�B�                                    Bx��7  
�          @��R���
@��@�{B~�\B�Lͽ��
?��R@�G�B��fB��{                                    Bx��E�  �          @�(�=#�
@�@��B}�
B�{=#�
?��R@�
=B�z�B���                                    Bx��T`  T          @�
=�B�\?��@��HB�B�B��)�B�\=��
@��RB�33C�R                                    Bx��c  �          @����=q?xQ�@��B���B�zᾊ=q�#�
@�ffB���C<&f                                    Bx��q�  �          @�{����?333@��
B��HB�.���þ���@��B��RC]�
                                    Bx���R  �          @�ff=�Q�?:�H@��B�B��{=�Q쾅�@��B��C���                                    Bx����  �          @�z�\(�?B�\@�B�p�C
E�\(��B�\@��B�C@��                                    Bx����  T          @��ͽ�>��
@��B�  B�(����(��@�=qB�(�C��H                                    Bx���D  
�          @���B�\?333@�Q�B��=C	ͿB�\���@���B��CG                                      Bx����  z          @�
=��?Y��@�  B�#�C33����G�@�=qB��=C9�{                                    Bx��ɐ  `          @�p���{?�Q�@�=qB��B�B���{?�\@�Q�B�.C^�                                    Bx���6  "          @�����?�z�@��\B�� B�33���>�@�Q�B�C.                                    Bx����  �          @�{��
=?�Q�@�=qB��RCO\��
=?�\@���B���C}q                                    Bx����  �          @�G���p�?���@��B���B�=q��p�?+�@��HB���C�)                                    Bx��(  �          @��׿��\?Ǯ@�33B���CE���\?!G�@��B�.C��                                    Bx���  z          @�G���33?�  @�33B�C���33?z�@���B��)C�                                     Bx��!t  .          @��׿��?��@�Q�B�aHC
=���?#�
@�
=B�\C��                                    Bx��0  �          @�Q��\)?���@�G�B�z�C�
��\)>�
=@��RB���C%z�                                    Bx��>�  �          @������
?�@��HB�{C	����
?�\@���B��3C!�=                                    Bx��Mf  T          @�33����?�33@�{B���C�׿���>��@��
B��RC!+�                                    Bx��\  T          @�(���{?�z�@��B��C c׿�{>�@��B��)C��                                    Bx��j�  �          @�녿�p�?���@�
=B��qCQ쿝p�>�(�@�z�B�=qC �                                    Bx��yX  �          @��H���R?���@��B�C�H���R>�
=@��B��C!(�                                    Bx����  �          @�33��z�?��@���B�k�C#׿�z�>\)@�z�B���C.u�                                    Bx����  
�          @�33��\)?\(�@��\B�{C�ÿ�\)��\)@���B���C6�                                    Bx���J  �          @��\��
=?�G�@�ffB��)C
����
=>�Q�@�33B�B�C%�                                    Bx����  �          @�33��Q�?�
=@�  B�#�C����Q�>�=q@�(�B���C)@                                     Bx��  T          @�=q��>���@�(�B��3C��녿�@��
B���C^�                                    Bx���<  �          @����#�
���@�G�B�  C�
�#�
��Q�@���B���C���                                    Bx����  �          @��\<���z�@�=qB�Q�C�
<���p�@�p�B�C���                                    Bx���  �          @�p�=�G���z�@���B��C�j==�G���p�@�Q�B�33C��                                    Bx���.  �          @�p�=u���@��B��{C��{=u���
@��B�u�C�)                                    Bx���  �          @��H    >B�\@�=qB���B��    �@  @���B��C�%                                    Bx��z  �          @�=q<�>Ǯ@���B�\B�z�<���@���B��fC�c�                                    Bx��)   T          @�33��p�>���@�=qB�ǮC	�{��p��
=@���B�B�Cn{                                    Bx��7�  �          @�(��8Q�>�p�@��B�ǮB��8Q�\)@��HB�ǮC|�                                    Bx��Fl  T          @��ͽ���=�G�@��
B��3C:���ͿQ�@��B��=C��f                                    Bx��U  T          @���z�<#�
@�p�B���C2�q��z�k�@��HB�u�C|�                                    Bx��c�  �          @��R��Q�>W
=@�{B��C쾸Q�:�H@�z�B��Cs�f                                    Bx��r^  T          @�{�h��=�\)@�33B���C/���h�ÿW
=@���B�B�C^�\                                    Bx���  �          @����\>\@��B���C�����\��@�G�B�L�CO�                                    Bx����  T          @�p���G���\)@��
B��C<���G��xQ�@���B�  Cu�=                                    Bx���P  �          @��Ϳ�=q>B�\@���B�Q�C*���=q�333@�\)B���CT�\                                    Bx����  �          @��ͿY��>�
=@���B��C��Y�����@���B�#�CQ�                                    Bx����  �          @�(��^�R>�  @�G�B���C$!H�^�R�#�
@�Q�B���CXE                                    Bx���B  T          @�\)���?:�H@��B���CO\������@�G�B�\C:L�                                    Bx����  �          @�Q쿑�?z�@��HB��RC#׿�녾��R@��
B��HCCW
                                    Bx���  �          @�  ��G�>.{@��HB�C,33��G��5@�G�B�
=CQ��                                    Bx���4  T          @���u��Q�@���B��RC9���u�z�H@���B���Ca}q                                    Bx���  "          @�����
�\@��B��)CH33���
��  @�
=B���Cf�)                                    Bx���  �          @�����\��Q�@�=qB�CC�H���\��(�@�{B��3C_�q                                    Bx��"&  �          @��׿��þ�G�@��\B�\)CF8R���ÿ�ff@�B��HC`ff                                    Bx��0�  "          @�����\���@��B�33CL�쿂�\��=q@�ffB���Ch�\                                    Bx��?r  "          @��R�Q녾�\)@�z�B�ǮCF޸�Q녿�33@���B��{CjQ�                                    Bx��N  �          @��R���þ���@���B��=CB#׿��ÿ�@�z�B�33C]��                                    Bx��\�  �          @�ff��z��(�@��B�#�CD�
��zῡG�@��HB�p�C]                                    Bx��kd  T          @�p���33��\@�ffB�� CG����33��=q@���B�(�C_u�                                    Bx��z
  �          @��R��Q�!G�@��RB��fCK�f��Q쿹��@�G�B��)Ca0�                                    Bx����  �          @�{���H�J=q@�p�B��qCPzῺ�H�˅@�
=B�ǮCc�{                                    Bx���V  �          @�{��33���\@�(�B��CW�q��33��@���B{ChB�                                    Bx����  �          @���G���
=@��B���CY�3��G�����@���Bv{ChJ=                                    Bx����  T          @�p���������@��B�\)C[� ������(�@���Bw{Ci�H                                    Bx���H  �          @�Q�˅�Y��@�{B���CP��˅���@��B��Ca�                                    Bx����  �          @�ff��
=���\@��B�p�CS8R��
=���@��\Bw{Cb�R                                    Bx����  �          @������  @���B��=CP\)���޸R@�Bq
=C_G�                                    Bx���:  �          @����(��^�R@���B�k�CK����(��У�@�=qBs{C[��                                    Bx����  �          @�\)��{�J=q@��\B��\CK
��{�Ǯ@�z�By�\C[�R                                    Bx���  �          @�Q���
�L��@�(�B�k�CLLͿ��
����@�{B|�\C]��                                    Bx��,  �          @�33��p��5@��\B�u�CM����p���(�@���B��C`޸                                    Bx��)�  �          @�zῊ=q�+�@�
=B�B�CS�q��=q����@���B���Cik�                                    Bx��8x  �          @����Q�
=q@��B��
C�g���Q쿫�@��RB��RC�#�                                    Bx��G  �          @����׿�G�@���B�{Ct��׿��@��B���C=q                                    Bx��U�  �          @�{�=p����@���B��Ck޸�=p���{@���B���Cxff                                    Bx��dj  �          @��R��  ��G�@�Q�B�L�Ca{��  ���
@�G�B��
Cp�=                                    Bx��s  �          @�\)�z�H��G�@�
=B�k�Ch\�z�H�G�@�
=B��Ct�                                    Bx����  T          @��H�Q녿}p�@�p�B���CfO\�Q녿޸R@��RB���Ct�                                    Bx���\  �          @�33�Y���h��@�{B��Cc\�Y����z�@��B���Cr�R                                    Bx���  �          @�  �u�}p�@���B�  Caٚ�u���H@��HB��CpǮ                                    Bx����  T          @��R�\(��c�
@���B�(�Cb!H�\(���\)@��B��qCr
                                    Bx���N  �          @�Q�8Q쾙��@�ffB�  CJ�q�8Q쿆ff@��HB�p�Ck��                                    Bx����  �          @��
�z�>#�
@��\B�u�C%��z��R@�G�B��HCb��                                    Bx��ٚ  �          @�p���?\)@��HB��HC�����aG�@��B�=qCJ�3                                    Bx���@  �          @���Ǯ<#�
@�=qB�� C2E�Ǯ�J=q@�Q�B��Css3                                    Bx����  �          @�ff�녿^�R@�=qB���Cl��녿�\)@�(�B�8RCz��                                    Bx���  �          @���aG��E�@�\)B��C]33�aG���G�@��B�{Co�                                    Bx��2  �          @�{��=q�:�H@���B��fCV5ÿ�=q��(�@��B���Ci�=                                    Bx��"�  T          @�\)�\(��}p�@���B���Cd�
�\(���(�@�33B��Csp�                                    Bx��1~  �          @�ff�.{��\)@�
=B���Cs���.{�@��RB�\C{��                                    Bx��@$  �          @��R�E��z�H@���B�B�Ch
=�E����H@�33B���Cu�f                                    Bx��N�  �          @�G��Y������@��HB��fChc׿Y����ff@�(�B�p�Ct��                                    Bx��]p  T          @�G��z�H��\)@�  B���CjW
�z�H��\@�  B}  Ct^�                                    Bx��l  �          @��
�=p����@�{B�W
Cm\�=p���@�
=B�p�Cx5�                                    Bx��z�  �          @���W
=��{@�z�B�G�Ch�=�W
=��ff@�B���Cu                                      Bx���b  
Z          @�������
=@���B�\ChaH�����@��Bj\)Cp��                                    Bx���  "          @�zΉ���=q@�Q�B�33Ck^�����\)@��Bs�
Ct�                                    Bx����  "          @�33�����p�@��HB��Ce녿����z�@��B�u�Cq}q                                    Bx���T  �          @�33��G����
@��B�k�Cg����G�����@��\B���Cr��                                    Bx����  �          @�z῎{��
=@�
=B���Cl�R��{�@�{Bo(�Ct�                                    Bx��Ҡ  �          @�(����\����@��RB���Cg.���\�{@�ffBp\)CpT{                                    Bx���F  
Z          @��Ϳ��R��(�@���B�Ce�Ϳ��R�Q�@�Q�Bu�\Co�                                    Bx����  �          @��H���Ϳ�33@�\)B��fC\xR���Ϳ�ff@���B~33Ci+�                                    Bx����  "          @�����  ��R@�33B���CN=q��  ���
@�
=B�ffCa�\                                    Bx��8  "          @�������Q�@�=qB�
=Cc�쿋���@��B�#�Co��                                    Bx���  T          @�G����þ�(�@��\B�
=CF
���ÿ��@�\)B�=qC[�R                                    Bx��*�  �          @��\��
=�=p�@�B��COxR��
=����@���B��{C_�3                                    Bx��9*  �          @�z��33�=p�@���B���CL!H��33���@���B�\)C\!H                                    Bx��G�  T          @��\��\)�xQ�@�{B�
=CR�ÿ�\)����@�Q�B|�RC`��                                    Bx��Vv  �          @��ÿ�Q��\)@�G�B��CdQ쿸Q��{@���BiQ�Cm�                                    Bx��e  �          @��ÿ��׿���@�ffB���Cf!H���׿�(�@�\)B|Q�Cp@                                     Bx��s�  �          @��ÿ������@�ffB��{C\�f�����  @�Q�B�33Ci)                                    Bx���h  �          @�  ��(��u@�p�B�CU���(���=q@�  B�ǮCb��                                    Bx���  �          @������H��@��B���CVG����H��\@��BsffCb                                      Bx����  �          @�=q��33��(�@��
B�
=CXk���33����@�p�Bsz�Cc�=                                    Bx���Z  �          @�{���!G�@�33BQ=qCh����B�\@n�RB8=qCm�R                                    Bx���   �          @�  �Q���
@���B]�HC`��Q��'
=@\)BG�Cf��                                    Bx��˦  T          @�  �Q쿢�\@���Bu�CR���Q����@��HBcC\�3                                    Bx���L  �          @����ý�G�@�Q�BxQ�C6����ÿ0��@��RBsffCD0�                                    Bx����  �          @�\)��R�c�
@���Bz�RCI�R��R���R@��Bl�RCU��                                    Bx����  T          @����ÿ#�
@��\Bv�\CC
=��ÿ�p�@��RBkCO=q                                    Bx��>  �          @����׿���@�  Bs�CMp���׿��@�=qBdp�CX�                                    Bx���  
�          @�����z�@���B]��C`@ ���&ff@�  BG��Cf�=                                    Bx��#�  �          @�{�G��޸R@�ffBlQ�C\�
�G��33@�{BW�\Cd��                                    Bx��20  �          @�ff�녿ٙ�@�
=BmffC\������@��RBY  Cd�                                    Bx��@�  �          @�p���Q�У�@�  Br33C\��Q��(�@�  B]��Cdn                                    Bx��O|  �          @��R���G�@�Bi�Cd(����$z�@�z�BS33CjǮ                                    Bx��^"  �          @�p���
=�J=q@a�B9(�Cy�{��
=�e�@FffB�C{��                                    Bx��l�  "          @�Q쾸Q���(�?�z�A��RC���Q����\?��Ar�HC���                                    Bx��{n  T          @��R�=p�����@33A�{C�ÿ=p�����?޸RA��C�Z�                                    Bx���  �          @�z�fff���
@
�HAمC��\�fff���?��A��
C�"�                                    Bx����  �          @�G��4z�@q�?W
=A!��B�aH�4z�@i��?�ffA|  B�W
                                    Bx���`  �          @�  �(Q�@u�?}p�A?�B����(Q�@k�?���A�(�B��                                    Bx���  �          @�G��
=@�33?�@�=qB�R�
=@�  ?��ANffB��                                    Bx��Ĭ  �          @��H��p�@�z�?�\@�Q�B���p�@�G�?��AE�B�u�                                    Bx���R  �          @�Q��\)@��>��@��B�  ��\)@�
=?}p�AA�B��H                                    Bx����  �          @��׿�Q�@�33?uA:�\Bޔ{��Q�@�ff?�p�A��HB��                                    Bx���  �          @��׿�=q@�  ?���AU�B���=q@��\?���A�33B�Q�                                    Bx���D  �          @�����\@L(�@7
=BQ�B�#���\@5�@N{B+{B���                                    Bx���  �          @�  �
=q@]p�@�HA���B�
=�
=q@I��@3�
BffB�                                    Bx���  �          @���
=@a�?�  A�p�B��=�
=@S33@
=qA�Q�B�33                                    Bx��+6  T          @�33�!�@X��?�G�A�G�B��\�!�@I��@	��A�(�C �                                     Bx��9�  
Z          @���>�R@E�?ǮA��C��>�R@7�?�z�A�{C�                                    Bx��H�  T          @��\�z�@c�
?�z�A�Q�B�B��z�@U@�
A���B���                                    Bx��W(  �          @�33��@e@�A���B�W
��@U�@\)B(�B���                                    Bx��e�  T          @�33�Ǯ@qG�?�{A�B��)�Ǯ@a�@�A���B�{                                    Bx��tt  �          @�ff�0��@��H?���A�B�{�0��@�(�@�
A�{B���                                    Bx���  T          @�Q��G�@�{?�A��\B۞���G�@~{@��A�(�Bݣ�                                    Bx����  �          @�\)��G�@��H?�=qA�(�B�B���G�@�(�@z�A�\)Bϊ=                                    Bx���f  �          @�p����
@�=q@ ��A�(�B�p����
@tz�@p�A��HB�B�                                    Bx���  �          @��Ϳs33@��H?��RAʣ�B�=q�s33@u�@(�A�\)B��H                                    Bx����  �          @��
��
=@�  ?�G�A�ffB�8R��
=@q�@p�A�33B�aH                                    Bx���X  �          @���!�@4z�@�B 33C�f�!�@!�@*�HB��C�3                                    Bx����  �          @��
��
=@dz�@Q�A���B�p���
=@Q�@0��BffB�B�                                    Bx���  T          @�=q���H@P��@1G�B33B�(����H@;�@G�B.ffB��)                                    Bx���J  �          @�p���@\��@EB&�\B����@E@]p�B?��B��f                                    Bx���  �          @���:�H@j=q@+�B��Bʙ��:�H@U@Dz�B'�B̮                                    Bx���  "          @�\)��ff@p  @(�A�RB�aH��ff@_\)@%B��B՞�                                    Bx��$<  "          @�z��=q@aG�?˅A���B����=q@Tz�?��HA���B��3                                    Bx��2�  T          @�  �C33@G
=?��A[33CxR�C33@>{?��A���C��                                    Bx��A�  �          @��H��@�?��
A�G�C=q�H��@{@�A���C��                                    Bx��P.  �          @�=q��
@b�\?��RA�p�B�aH��
@S33@
=A�  B��)                                    Bx��^�  �          @�33��\)@aG�@G�A�B����\)@P  @(��BffB�Ǯ                                    Bx��mz  �          @�(�����@l(�@
=Aٙ�B�uÿ���@\(�@   B�RB�                                    Bx��|   "          @����@dz�@  A�G�B��Ϳ�@S33@'�BQ�B�ff                                    Bx����  �          @�(��6ff@333@�A홚C� �6ff@"�\@#�
Bz�C
B�                                    Bx���l  �          @�G��QG�@,��@�RA��C^��QG�@��@   A�  C
                                    Bx���  �          @��R�E�@/\)@{A�z�C
aH�E�@\)@   B �HC�                                    Bx����  �          @�  �J=q@3�
@
=qA�  C
Y��J=q@$z�@(�A�G�C��                                    Bx���^  "          @�33�L��@@��@�
A���C�q�L��@1�@
=A���C                                    Bx���  
�          @��H�HQ�@H��?�p�A��RC޸�HQ�@:�H@33A��C��                                    Bx���  �          @�\)�l��?���@#�
B��C�)�l��?���@,(�BQ�C#c�                                    Bx���P  "          @�Q��i��?��H@A�C�{�i��?ٙ�@!�B z�C��                                    Bx����  �          @���n{@�@\)Aۙ�C�{�n{?�Q�@��A�z�Ck�                                    Bx���  �          @�z��s�
@	��@(�A��HC���s�
?�z�@��A��HC^�                                    Bx��B  T          @���u�@z�@  A���C�f�u�?�=q@��A�  C}q                                    Bx��+�  
�          @���C�
@>�R@��Aԣ�C��C�
@0  @�A�ffC
{                                    Bx��:�  "          @����Tz�@1G�?�{A�C&f�Tz�@$z�@Q�A���C@                                     Bx��I4  �          @���
=q@fff?�A�=qB����
=q@X��@G�A�\)B�\                                    Bx��W�  �          @����=q@w�?�A��RB枸��=q@l(�@33A�  B��
                                    Bx��f�  T          @����  @w
=?�\A�B��f��  @j=q@	��A��B�.                                    Bx��u&  �          @�z��G�@|��?�\)A�=qB��f��G�@p  @  A�\B���                                    Bx����  T          @�����@���?�=qA�
=B֞����@��H@\)A�(�B�33                                    Bx���r  �          @�{�{�@{@G�A���C���{�@ ��@�RA�(�C��                                    Bx���  �          @�{��(�@ ��?�A��C
=��(�?�=q?��RA��\C(�                                    Bx����  �          @�z���(�?��?�A�  C� ��(�?ٙ�?��RA�C��                                    Bx���d  T          @��R���?�\)?�  A�\)C&f���?ٙ�?�A���C33                                    Bx���
  T          @�����p�?�z�?��HA��Ch���p�?��R?�\)A�ffC!ff                                    Bx��۰  �          @�  ��  ?�z�?��A�G�C"�)��  ?�  ?�\A�C$��                                    Bx���V  "          @�  ��  ?�{?ٙ�A�33C#(���  ?�Q�?�=qA�33C%&f                                    Bx����  T          @�\)��z�?�
=?��AyC%���z�?�ff?�Q�A��C':�                                    Bx���  �          @�����{?��?�G�Ah  C$����{?�?�\)A~{C&�                                    Bx��H  �          @�����?���?�
=A��C&z���?}p�?��
A�p�C(�                                    Bx��$�  �          @��R���?��?�\)A�
C's3���?fff?��HA��RC)�                                    Bx��3�  �          @�����\)?�G�?���As\)C'���\)?c�
?�33A�(�C)ff                                    Bx��B:  �          @����Q�?\(�?��HAa�C)�=��Q�?=p�?��Ao\)C+(�                                    Bx��P�  �          @������?fff?�  A9�C)aH����?L��?�=qAH��C*��                                    Bx��_�  T          @�G���  ?n{?�z�A�Q�C(����  ?J=q?��RA��
C*��                                    Bx��n,  �          @��R��33?!G�?޸RA�{C,@ ��33>�?��A�
=C.@                                     Bx��|�  
�          @����Q�?(��?��At��C,���Q�?
=q?�\)A33C-��                                    Bx���x  T          @�Q���33?:�H?�  A7\)C+}q��33?!G�?��AC
=C,�{                                    Bx���  �          @�G���33?!G�?�(�A`z�C,�\��33?�?��
AjffC-�H                                    Bx����  
�          @�  ��  >�?�(�A��\C.� ��  >��
?�  A�  C0)                                    Bx���j  "          @�����>���?��A��C0k����>#�
?ǮA��C2)                                    Bx���  T          @����
=>8Q�?�{A���C1޸��
=<�?�\)A��C3��                                    Bx��Զ  T          @�G���
=��?�  A��HC4O\��
=�8Q�?޸RA�  C65�                                    Bx���\  T          @�Q���Q�?
=q?��HA��C-����Q�>��?�  A��C/�                                    Bx���  �          @�  ��Q�?G�?��Axz�C*���Q�?(��?�33A�Q�C,�                                    Bx�� �  �          @������?=p�?�G�Aj�HC+!H����?!G�?�=qAvffC,xR                                    Bx��N  
Z          @�{��Q�?0��?��AT��C+�3��Q�?
=?�Q�A_�C,�                                    Bx���  �          @�\)���?B�\?��
A?�C*�����?+�?���AK�C,�                                    Bx��,�  T          @�����?^�R?}p�A6�\C)���?G�?��AD  C*��                                    Bx��;@  
�          @������H?h��?uA0(�C)ff���H?Q�?��
A>=qC*c�                                    Bx��I�  �          @�G�����?333?h��A&ffC+�����?(�?xQ�A0��C,�)                                    Bx��X�  �          @�(�������
?�G�A?�C7޸������?z�HA9��C8�f                                    Bx��g2  T          @����
=�\(�?@  A
=C>L���
=�k�?.{A z�C?                                    Bx��u�  T          @��
��{�L��?aG�A(  C=�R��{�aG�?O\)A{C>��                                    Bx���~  �          @�p�����O\)?s33A3
=C=�q����c�
?aG�A%G�C>�f                                    Bx���$  �          @����G���?h��A*ffC:E��G����?\(�A!G�C;(�                                    Bx����  T          @��
��{��z�?�p�Ak�C7�{��{����?���Ae�C8�{                                    Bx���p  �          @�����
=�#�
?��Ax��C4�=��
=���?�ffAw
=C5�H                                    Bx���  T          @�����׾�?���AO�C5�{���׾aG�?��AL��C6�                                    Bx��ͼ  �          @������
�.{?�ffA|  C6
���
��\)?��AxQ�C7n                                    Bx���b  �          @�G���=q��\)?�
=A���C4���=q�B�\?�A��C6h�                                    Bx���  �          @�z���\)���R?�z�A\��C7����\)���?���AV�HC8�                                    Bx����  "          @�33���R��?�33A[
=C5�H���R�k�?���AX(�C6�                                    Bx��T  
�          @������=���?�p�A�(�C2���논��
?��RA�z�C4O\                                    Bx���  T          @�ff���
>k�?�(�A��C0�3���
=�Q�?�p�A�G�C2��                                    Bx��%�  �          @�=q���>��?�{A�p�C-����>�{?��A���C/s3                                    Bx��4F  {          @�{�s33?���?��A�Q�C"z��s33?��?�(�A��HC$�f                                    Bx��B�  �          @�(��n{?���?�A��C
�n{?�
=?��A�\)C�                                    Bx��Q�  �          @�p��hQ�?5@ffBC)  �hQ�?�@��B�
C+�                                    Bx��`8  �          @�  �S33?���@&ffB�RC�=�S33?�{@-p�B33C�H                                    Bx��n�  T          @�\)�5�?�@<��B%  CW
�5�?���@E�B.ffC�                                    Bx��}�  T          @�G��:�H@�@4z�B�C8R�:�H?�ff@>�RB#�Cc�                                    Bx���*  �          @����AG�@��@�B�C���AG�@(�@'
=B�C�                                    Bx����  �          @�G��5@	��@4z�BG�C�=�5?�
=@>�RB#�
C�
                                    Bx���v  �          @�z���
@0  @:=qB�RC���
@!G�@G
=B)G�C�)                                    Bx���  T          @�=q�@>{@.{Bp�B�L��@/\)@<��B!��B��\                                    Bx����  "          @�{���@Z�H@$z�BffB�����@Mp�@5�B  B���                                    Bx���h  �          @�����33@fff@!G�A�  B���33@Y��@2�\B�B�                                    Bx���  �          @����p�@aG�@ffA�B��p�@U�@'
=B=qB�k�                                    Bx���  �          @��H���@]p�@*�HB��B�G����@P  @;�B=qB��
                                    Bx��Z  �          @�=q�\@g�@A�=qB�׿\@[�@'
=BQ�B��f                                    Bx��   T          @�33���H@s�
@(�A�B�.���H@hQ�@�RB�B���                                    Bx���  
�          @�z���\@-p�@#�
B�B�Ǯ��\@ ��@0��B!C�                                    Bx��-L  �          @��
��p�@3�
@*�HBG�B�k���p�@&ff@7�B%\)B��3                                    Bx��;�  T          @��H���@B�\@
=Bz�B�����@6ff@%�B�B���                                    Bx��J�  �          @�=q��@7�@(�B

=B�  ��@+�@)��B�B���                                    Bx��Y>  �          @���ff@Dz�@�B G�B��H�ff@8Q�@#�
BffB�=q                                    Bx��g�  �          @��H���H@H��@�HB�RB��Ὼ�H@<��@)��B=qB�Ǯ                                    Bx��v�  T          @�{�ff?�Q�?ٙ�A���Ch��ff?�?�A�=qCc�                                    Bx���0  �          @�{�^{?��?�(�A�(�C���^{?ٙ�?���A���C�R                                    Bx����  �          @���6ff@!G�?���A�{C
� �6ff@Q�@   A�Q�C#�                                    Bx���|  �          @�
=�<(�?�Q�@\)B�
C�{�<(�?�\@Q�B
�C�R                                    Bx���"  T          @�{�<(�@Q�@�\A�z�C
�<(�?�(�@(�A��C&f                                    Bx����  T          @��\�b�\?Ǯ?fffAR=qC=q�b�\?��R?�G�Al��C33                                    Bx���n  �          @~{�n�R��(�?�
=A��C:��n�R��\?��A��HC;�\                                    Bx���  T          @����p  >�p�?��
A���C.O\�p  >�\)?�ffA��\C/�q                                    Bx���  �          @��R����Tz�>L��@*�HC?p�����W
=>�?�C?��                                    Bx���`  �          @����Q�   >�33@�z�C:�3��Q��>��R@�z�C;                                    Bx��	  �          @��\�������R>\@�\)C8{��������>�33@�p�C8n                                    Bx���  �          @�=q����<�>��@`��C3�f����<#�
>�=q@a�C3�f                                    Bx��&R  T          @�=q��=q=#�
<�>���C3z���=q=#�
<�>�G�C3��                                    Bx��4�  T          @�����Q�>L�ͽ�G�����C1G���Q�>W
=��Q쿠  C10�                                    Bx��C�  �          @������>�(��aG��@  C.(����>�ff�B�\�$z�C-��                                    Bx��RD  �          @�Q���\)��(�>�  @X��C9޸��\)��ff>aG�@=p�C:
                                    Bx��`�  T          @�\)���R�Ǯ=�Q�?��HC9W
���R����=u?Tz�C9h�                                    Bx��o�  �          @�ff����{<�>�Q�C8�3����{    <��
C8��                                    Bx��~6  T          @�  ����B�\�u�Q�C6������.{����]p�C6T{                                    Bx����  T          @�����\)>�33�����Q�C/O\��\)>\��G���p�C.�H                                    Bx����  �          @������?��!G��C-
����?\)������C,�                                    Bx���(  �          @�����ff?E���G��U��C)����ff?W
=�u�IG�C(��                                    Bx����  T          @�(���  ?k������ffC'����  ?s33������HC'ff                                    Bx���t  �          @�33��\)?aG��   ���
C(B���\)?h�þ�G���G�C'�{                                    Bx���  �          @��\��
=?W
=�Ǯ��ffC(�q��
=?\(��������C(h�                                    Bx����  
e          @������?�ff�����C%�����?�����
���C%�
                                    Bx���f  �          @�����?E�<��
>��RC)�{���?B�\=��
?��
C)�H                                    Bx��  �          @�Q���{?=p�>B�\@ ��C*���{?8Q�>u@L(�C*=q                                    Bx���  �          @�����ff?O\)�L�Ϳ(��C)
��ff?O\)<#�
=�Q�C){                                    Bx��X  �          @�G���ff?\(���G���Q�C(u���ff?\(��L�Ϳ&ffC(c�                                    Bx��-�  �          @����z�?u��R���C&���z�?�  �������C&c�                                    Bx��<�  �          @�G����H?��
�E��&�\C%�)���H?�=q�333��HC%0�                                    Bx��KJ  �          @������H?������R��
=C#�3���H?�(��k��G�C#u�                                    Bx��Y�  �          @�����?�\)�����HC$�����?�33���H�љ�C$T{                                    Bx��h�  �          @�(���=q?�{��ff�`(�C$��=q?�
=�xQ��O33C#ٚ                                    Bx��w<  �          @�{���
?��׿���n�RC$�����
?�������]��C#��                                    Bx����  �          @�
=��z�?����  �O�
C"\��z�?�z�h���<(�C!@                                     Bx����  �          @�  ��33?�z῕�r{C �R��33?��R�����]G�C �                                    Bx���.  �          @��R��z�?�=q�k��?\)C":���z�?�녿Tz��,  C!}q                                    Bx����  �          @�p�����?�{�(���	�C!�H����?�33�����C!\)                                    Bx���z  �          @�
=��=q?��R��=q���HC#{��=q?��ÿ��R����C!�R                                    Bx���   �          @�p����?�=q��ff�Z�HC%u����?�33�xQ��K
=C$�
                                    Bx����  �          @�ff��{?�녿s33�F=qC$�q��{?����^�R�5��C#�R                                    Bx���l  �          @�{�}p�?G�������ffC(�f�}p�?fff��\��{C'=q                                    Bx���  �          @�{�|(�?}p�������C%ٚ�|(�?�{��(���{C$B�                                    Bx��	�  �          @�ff�z=q?aG���z��̏\C'\)�z=q?�  ������p�C%��                                    Bx��^  �          @�ff��=q?�G���z�����C&���=q?��Ϳ�����C$��                                    Bx��'  �          @�p���{?��ͿO\)�)��C%=q��{?�z�=p��C$�
                                    Bx��5�  �          @���?�\)�h���=C%���?�
=�Tz��-��C$O\                                    Bx��DP  �          @�z�����?�
=���R���
C#�f����?�G���z��u�C"�)                                    Bx��R�  �          @���33?�p���33�q��C#G���33?�ff�����_�C"W
                                    Bx��a�  �          @�33��  ?�ff��\)�o
=C!����  ?�\)���
�[�
C!
=                                    Bx��pB  �          @���|��?��׿��\�[�C �q�|��?�Q�n{�G\)C��                                    Bx��~�  �          @�=q�y��?��
���
�]p�C�)�y��?˅�n{�G
=C��                                    Bx����  �          @�=q��33?�{?0��A33C$����33?��?B�\A"�\C%��                                    Bx���4  �          @�Q�����?��
?aG�A?
=C%������?xQ�?s33AMp�C&��                                    Bx����  �          @�Q��~�R?���?�  AYp�C$�)�~�R?��
?���Ah��C%�                                    Bx����  �          @����\)?��?W
=A5�C"�\)?��R?n{AH  C"Ǯ                                    Bx���&  �          @��\�z=q?�(�?���AfffCff�z=q?�33?�A{
=C Y�                                    Bx����  �          @���s�
?��?Y��A733C�\�s�
?�p�?xQ�APz�C��                                    Bx���r  �          @�G��s33?�G�?@  A#\)C5��s33?ٙ�?^�RA<(�C޸                                    Bx���  �          @��\�e�?�ff?�  A���C�H�e�?�
=?���A�z�C:�                                    Bx���  �          @���j�H?�ff?��A�\)C
=�j�H?�Q�?޸RA���C�                                    Bx��d  �          @�=q��Q�?��
?h��AC33C"B���Q�?�(�?}p�AU�C#�                                    Bx�� 
  �          @���y��?�z�?��A�z�C#xR�y��?���?�{A��RC$��                                    Bx��.�  �          @�33�~�R?z�?���A�
=C+�~�R>��?У�A���C-=q                                    Bx��=V  �          @��\�~{>��?��A���C.��~{>���?�z�A�p�C/�)                                    Bx��K�  �          @��H������G�?\A�
=C5�\�����W
=?�G�A��
C6�3                                    Bx��Z�  �          @�33�p��?fff?��
A�C&���p��?G�?�=qA�=qC(E                                    Bx��iH  �          @���r�\?fff?��AΣ�C&���r�\?G�?�Q�A�
=C(p�                                    Bx��w�  �          @�(��e?��@G�A��
C� �e?��H@ffA��
C!aH                                    Bx����  �          @��H�{�?333?У�A��C)ٚ�{�?��?�A�Q�C+^�                                    Bx���:  �          @�(���Q�>��?�A�(�C.!H��Q�>���?ٙ�A���C/��                                    Bx����  �          @��
���>8Q�?�Q�A��\C1}q���=�Q�?���A��C2�=                                    Bx����  �          @�z���z�?\)?�ffA�  C,B���z�>�?�=qA��C-k�                                    Bx���,  �          @����{�?�p�?��
A�z�C"�f�{�?��?�{A���C#��                                    Bx����  
�          @��
�g�@33?��
A�33Cu��g�?��H?�z�A���C�                                    Bx���x  �          @��H�dz�@33?�{A��\C#��dz�?��H?��RA��CB�                                    Bx���  �          @�(��|(�?�33?Y��A3�CE�|(�?˅?s33AI�C�q                                    Bx����  �          @�33�|(�?�  ?\A��
C%�=�|(�?fff?�=qA���C'+�                                    Bx��
j  �          @�33��Q�>aG�?�{A�
=C0޸��Q�=�G�?�\)A�=qC2\)                                    Bx��  �          @����Q켣�
?�33A�33C4G���Q��?�33A���C5��                                    Bx��'�  �          @��\���ý��
?�ffA�ffC5+����þ8Q�?��A�p�C6�
                                    Bx��6\  �          @�=q�}p��.{?�G�A�z�C=� �}p��G�?��HA��C?
                                    Bx��E  �          @���mp�?�\)?��HA��C#��mp�?�  @�A�33C$�R                                    Bx��S�  �          @��
�r�\?��?���A��C$ff�r�\?p��?��AΣ�C&
                                    Bx��bN  �          @�(��`��?���?�\)A�{CB��`��?�(�?��RA�p�C޸                                    Bx��p�  �          @�z��Z�H?�(�?�33A�G�C\�Z�H?���@G�A�C�                                    Bx���  �          @���Mp�@
=q?��A��C��Mp�@�@G�A�(�C�H                                    Bx���@  �          @�Q��E@
=q?��HA�\)C
=�E@�@A�  C�3                                    Bx����  �          @�G��J�H@�R?�=qA��C�
�J�H@
=?�(�Aڣ�CY�                                    Bx����  �          @��H�[�?��?�{A˅C8R�[�?�G�?�p�A�\)C�{                                    Bx���2  �          @��H�P��?�(�@\)A�(�C(��P��?���@B��C=q                                    Bx����  �          @�(��^{?�ff@	��A�(�C��^{?�z�@  A�C�                                    Bx���~  �          @����b�\?���@p�A�G�C�3�b�\?�p�@�\A���C Ǯ                                    Bx���$  �          @��H�Q�?�z�@Z�HBQ  C\�Q�?n{@_\)BW  C��                                    Bx����  �          @��\�7�?^�R@G
=B8(�C##��7�?+�@J=qB<  C&�                                    Bx��p  �          @��
�J=q?�@<��B*C*���J=q>���@>�RB,C-�R                                    Bx��  �          @�33�2�\?z�H@J=qB;C ���2�\?E�@N{B@33C$��                                    Bx�� �  �          @�=q�:�H>8Q�@I��B<�C0k��:�H���
@J=qB=�C4s3                                    Bx��/b  T          @���?\)�#�
@A�B5z�C4�q�?\)�u@AG�B4��C8��                                    Bx��>  �          @����Vff=���@(Q�Bp�C2Q��Vff��\)@(Q�Bz�C5@                                     Bx��L�  �          @���I��?Y��@0��B �C$�
�I��?+�@3�
B$p�C'��                                    Bx��[T  �          @��H�\(�?�(�@z�Bp�C � �\(�?���@��B(�C"��                                    Bx��i�  �          @��H�S�
?\@�
B�Cc��S�
?�{@=qB�C�H                                    Bx��x�  �          @�33�a�@1녿���ffC��a�@3�
�\���
Cu�                                    Bx���F  �          @���]p�@>{�\(��*=qCO\�]p�@AG��(����HC
�
                                    Bx����  �          @��b�\@?\)����
=C���b�\@AG���p���G�C��                                    Bx����  �          @����\(�@8Q�>�?��C{�\(�@7
=>��
@���C:�                                    Bx���8  �          @��H�]p�@Q�?�ffAaC���]p�@33?��HA�C\)                                    Bx����  �          @�33�i��@\)?(�A�C� �i��@(�?B�\A!�C                                    Bx��Є  T          @�{�e@ ��?333A{C��e@p�?^�RA4z�C�)                                    Bx���*  
�          @�ff�R�\@1�?�RA�C���R�\@.�R?L��A*�\CO\                                    Bx����  �          @�=q�HQ�@@��?�33A�ffC#��HQ�@:=q?˅A�33C	�                                    Bx���v  �          @���`  @AG�>�G�@�C=q�`  @>�R?#�
@���C��                                    Bx��  �          @��^�R@G�=��
?��
C
#��^�R@G
=>�z�@dz�C
@                                     Bx���  �          @���Mp�@^{��Q쿋�C�q�Mp�@^{>\)?��HC�                                     Bx��(h  �          @�Q��@  @k���=q�Q�C5��@  @l(������
C!H                                    Bx��7  �          @���.�R@s33�����\B�ff�.�R@u��aG��.{B�{                                    Bx��E�  �          @��x��?�ff?�G�A�{CW
�x��?��H?�{A��Cu�                                    Bx��TZ  �          @��
�mp�?5@
�HA�\C)!H�mp�?�@p�A�C+T{                                    Bx��c   �          @�=q�tz�?�=q?���Ao33Cz��tz�?�G�?��HA��Cz�                                    Bx��q�  �          @����xQ�?�{?�p�A��
C �3�xQ�?��\?���A�C!�
                                    Bx���L  T          @��H�qG�?��?У�A���C n�qG�?�p�?��HA���C!��                                    Bx����  T          @�33�~�R?��R?��HA�p�C"��~�R?�z�?��A�ffC#��                                    Bx����  T          @�z��z�H?��?�A��C!}q�z�H?��H?�  A���C"�=                                    Bx���>  �          @�����\?��H?uAJ�HC#� ���\?��?��A\(�C$aH                                    Bx����  �          @�=q���\?��?k�AD��C%�����\?xQ�?}p�AS�C&�{                                    Bx��Ɋ  �          @��H�o\)?�?�{A�33C"���o\)?��?�Q�A�  C$u�                                    Bx���0  �          @����o\)?��?�A�p�C$+��o\)?p��?��AυC%�R                                    Bx����  �          @�G��dz�?�
=?��A��
C=q�dz�?�ff?�(�A���C {                                    Bx���|  �          @���mp�?��?�\A�=qC ���mp�?�?�{A�{C"��                                    Bx��"  �          @���o\)?��R?�(�A��HC!�q�o\)?�{?�ffA�Q�C#n                                    Bx���  �          @�G��w
=?h��?�\)A�C&�=�w
=?J=q?�
=A��\C(ff                                    Bx��!n  �          @���n�R?��H?�\)A�33C���n�R?�{?�(�A�ffC�                                    Bx��0  �          @����q�?�=q?��AyG�C^��q�?��R?�  A��\Cp�                                    Bx��>�  
�          @���u�?�Q�>�@�\)C8R�u�?�33?�@�=qC��                                    Bx��M`  �          @����tz�?�p�?333A��C�)�tz�?�
=?Q�A2�\CE                                    Bx��\  �          @����E�@8��>��@ʏ\C�
�E�@6ff?+�A��C	:�                                    Bx��j�  �          @�G��^�R@�H?z�@�33C��^�R@Q�?@  A!�C��                                    Bx��yR  �          @�Q��`��@�>��H@�C^��`��@�\?(��A=qC�{                                    Bx����  �          @��H�p��@�\?�\@���C���p��@   ?&ffA
�HC�                                    Bx����  �          @�=q�n�R@�>�G�@�z�C�)�n�R@33?
=@��HCG�                                    Bx���D  �          @��H����?��
>�z�@u�C5�����?�G�>���@�Q�C��                                    Bx����  �          @�33����?���>�@���C$�\����?��?\)@�{C%J=                                    Bx��  �          @�(����?}p�>�G�@�=qC&�����?u?�\@׮C'=q                                    Bx���6  �          @�=q��p�?��>��@ ��C%T{��p�?�=q>k�@B�\C%��                                    Bx����  �          @��\���\?��>���@�  C!=q���\?�{>���@��C!�\                                    Bx���  �          @��
���R>��?��Aa�C-�����R>Ǯ?���Ah��C.��                                    Bx���(  �          @��
���\?s33?uAN=qC&�H���\?aG�?��\A\��C'�
                                    Bx���  �          @��H�{�?�  ?�  AU�C{�{�?�
=?�{Al��C �                                    Bx��t  �          @��R��  ?G�?O\)A*ffC)����  ?8Q�?^�RA5C*p�                                    Bx��)  �          @�
=���\�.{?��
AVffC6@ ���\�u?��\AS\)C7:�                                    Bx��7�  �          @�����{��\)?B�\AQ�C4�f��{��?B�\A�HC5�)                                    Bx��Ff  �          @�{��\)>�z�?c�
A=p�C0{��\)>aG�?h��AAp�C0��                                    Bx��U  �          @�{��G��#�
?�G�AT(�C6
��G��k�?�  AQ�C7{                                    Bx��c�  �          @�{��z�>���?�\@�(�C.ٚ��z�>�Q�?
=q@߮C/Y�                                    Bx��rX  �          @�  ��z�?h��>�33@�C(T{��z�?aG�>��@���C(�                                    Bx����  �          @�����R?�(����
��  C�\���R?��H=�G�?���Cٚ                                    Bx����  �          @��R���
?�33��=q�`  C.���
?������z�C�q                                    Bx���J  �          @�Q����\?��=�G�?�
=CQ����\?�{>�  @Mp�Cz�                                    Bx����  �          @�  ���?�{=L��?!G�Cp����?���>B�\@=qC�=                                    Bx����  �          @�Q���ff?�=q>�\)@`��C^���ff?Ǯ>Ǯ@���C�                                    Bx���<  �          @�  ���
?���>�G�@�p�C�����
?Ǯ?\)@�RCG�                                    Bx����  �          @�{�s�
@
=?J=qA$��C��s�
@�\?s33AEC��                                    Bx���  �          @�ff�i��@��?aG�A7�C�=�i��@�
?��A]�C�)                                    Bx���.  �          @��I��@3�
?�Q�A{\)C
8R�I��@-p�?�33A���C:�                                    Bx���  �          @���j=q@�R?s33AG33C�f�j=q@	��?�{Ak33C�\                                    Bx��z  �          @�ff��?�ff?G�A ��C"����?�p�?^�RA5�C#��                                    Bx��"   �          @�����?xQ�?
=q@�=qC'J=����?k�?�RA Q�C'�)                                    Bx��0�  �          @�\)��{>��>L��@%C.�q��{>Ǯ>k�@?\)C.�3                                    Bx��?l  �          @�ff���
?333>Ǯ@�Q�C*�f���
?+�>�G�@�{C+O\                                    Bx��N  �          @�ff��>��=�?�G�C0����>�  >\)?�G�C0�=                                    Bx��\�  �          @�
=��ff>�=q=��
?��
C0xR��ff>�=q=���?�ffC0�\                                    Bx��k^  �          @�z����
>��R>L��@*�HC/�����
>���>k�@>�RC0!H                                    Bx��z  
�          @�p���G�?��?J=qA%�C,���G�?�?W
=A/\)C,�                                    Bx����  �          @����G�>�G�?!G�Ap�C.!H��G�>Ǯ?(��AQ�C.�\                                    Bx���P  �          @����33>�(�>�ff@��C.k���33>Ǯ>��H@��HC.��                                    Bx����  �          @�(����H>�Q�=�G�?���C/5����H>�33>\)?�=qC/Y�                                    Bx����  �          @�33���?zὣ�
���C,c����?z����(�C,T{                                    Bx���B  �          @�����R?
=q>�=q@hQ�C,�f���R?�>��R@��RC,��                                    Bx����  �          @�33���H���.{�{C5�����H���;8Q��C5W
                                    Bx����  �          @�33����>��;�{��G�C.������>�(����R���HC.J=                                    Bx���4  �          @�\)���
<��k��>ffC3�����
=��Ϳk��=G�C2��                                    Bx����  �          @�G���
=�B�\�5��C6xR��
=���8Q��=qC5��                                    Bx���  T          @�Q����?�Ϳ
=��33C,�3���?
=�����  C,\)                                    Bx��&  �          @�
=��z�?�R��Q����C+����z�?&ff�����{�C+�\                                    Bx��)�  
�          @����s�
��\)�ff�ޣ�CDT{�s�
�p������\CAٚ                                    Bx��8r  T          @��R�~{�O\)����G�C?���~{�(�ÿ�33��=qC=p�                                    Bx��G  �          @�{�~�R��Q��z���Q�C9��~�R�L�Ϳ�
=���HC6�\                                    Bx��U�  �          @�(��xQ��녿�����{C:�xQ쾊=q��\)��p�C8
=                                    Bx��dd  �          @��
�z=q�
=��z����C<���z=q��ff�ٙ�����C:�)                                    Bx��s
  �          @�{�z=q�fff��G�����C@���z=q�@  ��=q�ř�C>Ǯ                                    Bx����  �          @�
=�{�����p����
C4xR�{�>\)��(���G�C2�                                    Bx���V  �          @�{�w
==�G������C2c��w
=>�\)��
���C/�=                                    Bx����  �          @��
�R�\�����$z��=qCH��R�\�z�H�*�H�p�CD�)                                    Bx����  �          @���Q녾�{�-p���\C9�)�Q녽����.�R��HC5�\                                    Bx���H  �          @����u?��
����\)C@ �u?��<��
>W
=C0�                                    Bx����  
�          @�{����?��
?:�HA(�C"������?��H?W
=A/
=C#�q                                    Bx��ٔ  �          @��R�}p�?��
?��A��C�)�}p�?�z�?�A�p�C \)                                    Bx���:  T          @�{�QG�?�z�@G�A��C�R�QG�?��H@�B�CxR                                    Bx����  �          @��@  @Q�@p�B��C���@  ?�z�@(Q�B  C��                                    Bx���  �          @����C�
@\)@��A�33C�R�C�
@�\@��B33CE                                    Bx��,  �          @��@��@z�@\)A�(�CY��@��@�@(�B{C��                                    Bx��"�  �          @��L��@z�?�p�A��HC{�L��@Q�@�A�{CO\                                    Bx��1x  �          @�\)�HQ�@��@
=A�\)C���HQ�@��@z�A���C�f                                    Bx��@  �          @�ff�HQ�@=q@z�A�G�Cn�HQ�@p�@�A��C�q                                    Bx��N�  �          @�{�5�@\)@�
A�\)C
���5�@�@!G�B
��C5�                                    Bx��]j  �          @�{�C33@
=@
�HA�\)CT{�C33@	��@Q�B  C�{                                    Bx��l  �          @�ff�C33@�@
=A��Cn�C33@�R@�A�33C��                                    Bx��z�  �          @���K�@&ff?�ffA��HC���K�@��?��
A��HCc�                                    Bx���\  �          @�\)�Vff@�
?�Q�A��
Ck��Vff@
�H?��A�33C
                                    Bx���  T          @��\)?�
=?�G�A�  C B��\)?��?���A�C!��                                    Bx����  �          @�  �u@�?��Aj=qC0��u?�z�?���A�z�C��                                    Bx���N  �          @�  �\)?�Q�?�ArffC�\)?�=q?�=qA�\)Cs3                                    Bx����  �          @����y��?�(�?�Ao\)C8R�y��?���?���A�z�C�)                                    Bx��Қ  �          @���s33@Q�?�{A���C���s33@   ?ǮA�p�CE                                    Bx���@  �          @���\)?���?��HAv�RC��\)?ٙ�?���A��RC��                                    Bx����  �          @�G�����?�{?���At��C!������?��R?���A�p�C#aH                                    Bx����  �          @����  ?=p�?�(�A|Q�C*���  ?�R?��
A��C+�H                                    Bx��2  �          @�
=��(�?xQ�?�=qA��C&���(�?W
=?�A��HC(��                                    Bx���  �          @�\)���?Tz�?ǮA�p�C(�{���?.{?У�A�\)C*�f                                    Bx��*~  �          @�{��G�?�Q�?�z�A��\C#����G�?�ff?\A�Q�C%xR                                    Bx��9$  �          @�ff��Q�?��R?�  A��RC"�)��Q�?�{?�{A�33C$��                                    Bx��G�  �          @�ff�z�H?��
?�Q�A��HC%B��z�H?\(�?��
A�\)C'�
                                    Bx��Vp  T          @�\)���\?}p�?�z�A��RC&L����\?Y��?�  A���C(33                                    Bx��e  �          @�{��z�?k�?�ffA��RC'z���z�?J=q?��A��
C)=q                                    Bx��s�  �          @��R��p�?��
?^�RA6�\C&)��p�?p��?xQ�AK\)C'G�                                    Bx���b  �          @��R��  ?��\?W
=A.�\C&s3��  ?p��?p��AC
=C'�{                                    Bx���  �          @�\)���\?+�?k�A>�\C+0����\?z�?}p�AK�C,n                                    Bx����  �          @�  ��33?\)?uAEG�C,����33>�?�G�AP  C-��                                    Bx���T  T          @����p�>W
=?!G�A�C1B���p�>��?&ffA��C2
                                    Bx����  �          @������=��
?O\)A&{C2������    ?O\)A&�HC4
=                                    Bx��ˠ  �          @����=q�z�H?#�
A
=C@��=q���?�@��
CA��                                    Bx���F  �          @�����{��33?aG�A<��C8�R��{��(�?W
=A4(�C9��                                    Bx����  �          @��H��(�?�  ?�ffA�ffC#(���(�?�=q?�A�33C%J=                                    Bx����  T          @�p���?��\?�G�A�Q�C&8R��?W
=?�{A���C(��                                    Bx��8  
�          @��R��(�?�p�?��A�C#xR��(�?��\@ ��A�=qC&{                                    Bx���  �          @�p����
?�z�?�
=A�Q�C!!H���
?�p�?�A���C#h�                                    Bx��#�  �          @���z�?���?��RA���C.��z�?�z�?�33A�33C!0�                                    Bx��2*  �          @�33���H?���?��A���C�����H?���?�(�A�\)C z�                                    Bx��@�  �          @���~{?��?�\)A�(�C�
�~{?�{?�\A��RC!)                                    Bx��Ov  �          @��H��  ?޸R?�33A�{Ck���  ?˅?���A���CY�                                    Bx��^  �          @��\�}p�?�(�?��
A��C�\�}p�?�ff?ٙ�A��\C��                                    Bx��l�  �          @�����?�=q?�33AhQ�C�3���?ٙ�?��A��CJ=                                    Bx��{h  �          @��
����?��H?�=qAW�
C�\����?˅?�  A|Q�C�                                    Bx���  �          @�����?��R?��AW\)C =q���?�\)?��HAw�C!�                                     Bx����  T          @�Q��\)?�\)?uAC�
C�
�\)?�G�?�33Am�C33                                    Bx���Z  �          @���z�H@�?@  A
=C���z�H?�
=?xQ�AH(�C�                                    Bx���   �          @�����?�
=?fffA9��C�����?���?�=qA_
=C޸                                    Bx��Ħ  
�          @������?˅?��
AR�\C�)���?�(�?���AuC \)                                    Bx���L  �          @�����
=?��\?��A��C&s3��
=?^�R?�z�A���C(ff                                    Bx����  �          @��H���?���?��AR�HC%O\���?�G�?�z�Ak�C&ٚ                                    Bx���  �          @��R����?�33>B�\@(�CǮ����?�{>���@���C(�                                    Bx���>  �          @��
����?�z�=�?�ffC������?У�>���@~�RC�                                    Bx���  �          @�G���z�?�>�33@�
=C�R��z�?��?\)@�=qC�\                                    Bx���  �          @�������?�ff>u@C�
C�
����?�G�>�G�@��
C
=                                    Bx��+0  �          @�33��G�?˅>L��@(��Cp���G�?Ǯ>\@�Q�Cٚ                                    Bx��9�  �          @�G���33?�z�>��H@љ�C$E��33?��?�RA��C%�                                    Bx��H|  �          @��H���?�(�?!G�A��C#�����?��?B�\A"�\C$�=                                    Bx��W"  �          @����33?��?=p�A�HC%{��33?�  ?\(�A9�C&J=                                    Bx��e�  �          @��
��?�G�?@  A{C&u���?h��?\(�A5�C'�                                    Bx��tn  �          @�\)���?���>�ff@��C%O\���?���?z�@�z�C&�                                    Bx���  �          @�\)���
?Q�?�\@�G�C)aH���
?@  ?��@�\)C*5�                                    Bx����  �          @�\)��(�>aG�?k�A=�C1\��(�=�?p��AAG�C2}q                                    Bx���`  �          @�G���?�R?.{A
�HC,
��?�?@  A��C-+�                                    Bx���  �          @�33��z�>Ǯ?��
A�C.�f��z�>u?���A��C0��                                    Bx����  �          @�\)��  ?��
?n{A@��C&h���  ?h��?�ffAYG�C'�                                    Bx���R  �          @�\)��ff?���>�{@��C �H��ff?�33?�\@�(�C!��                                    Bx����  �          @�ff��=q?�z�>��H@�(�C�\��=q?˅?0��A33C��                                    Bx���  �          @���=q?�z�?   @�C�\��=q?˅?0��A(�C�                                    Bx���D  �          @�p����
?�  >��@�(�C ����
?�
=?&ffA�C ��                                    Bx���  �          @�\)��\)?�>L��@$z�C!z���\)?���>�p�@���C!��                                    Bx���  �          @�  ��\)?\=���?�G�C G���\)?��R>�\)@g�C ��                                    Bx��$6  �          @������?��
=u?@  C�=����?�G�>�\)@\��CǮ                                    Bx��2�  
�          @����33?��
�u�333C �H��33?\>�?�z�C ��                                    Bx��A�  �          @�����(�?���>k�@<(�C%}q��(�?���>�p�@��RC%��                                    Bx��P(  �          @�
=���\?:�H?=p�A��C*p����\?!G�?Q�A*�RC+�3                                    Bx��^�  �          @�
=���?u>�ff@��C'ff���?fff?�@�z�C(8R                                    Bx��mt  �          @�(���
=?��>�@�G�C%����
=?}p�?(�A   C&ٚ                                    Bx��|  �          @�{���?�33?
=@�(�C$޸���?��?:�HA�C%�                                    Bx����  �          @��R����?�\)?(�@��HC%aH����?��
?@  A�C&xR                                    Bx���f  �          @�����Q�?xQ�>��H@�p�C'5���Q�?fff?(�@��C(�                                    Bx���  �          @�\)��?���?�=qA^�RC#�q��?�ff?�(�A}p�C%��                                    Bx����  T          @�����p�?���?��A��C#�R��p�?��\?�p�A�ffC&Q�                                    Bx���X  T          @����Q�?@  ?z�HAK�C*
=��Q�?�R?���A^�\C+�=                                    Bx����  �          @�����33?   ?n{A@  C-z���33>�p�?}p�AK�
C/�                                    Bx���  �          @�p���=q>�(�?B�\A�\C.Q���=q>���?O\)A)�C/�                                    Bx���J  �          @������H?(�?}p�AK�
C+�R���H>�?��AZ�HC-��                                    Bx����  �          @��R��p�?#�
?��
A�
=C+L���p�>�?���A�
=C-��                                    Bx���  �          @�  ��p�����?:�HA��C7ٚ��p��Ǯ?.{A(�C9�                                    Bx��<  �          @����(��aG�?aG�A5�C6�f��(���{?W
=A.{C8h�                                    Bx��+�  T          @��\��  �#�
?Q�A%C4����  ���?O\)A#
=C5�3                                    Bx��:�  �          @������?5?Tz�A)�C*�
���?
=?k�A;33C,\)                                    Bx��I.  �          @����\)>�(�?5A�HC.}q��\)>���?B�\Ap�C/                                    Bx��W�  �          @�z���(��#�
�L�Ϳ.{C4.��(��#�
�u�0��C4{                                    Bx��fz  �          @�(������G�=���?��\C5W
�����=�Q�?�=qC5��                                    Bx��u   �          @�{���
��{?333A	�C8+����
��(�?&ff@�
=C9Q�                                    Bx����  �          @���=q>aG�?p��A8��C10���=q=�Q�?uA=p�C2�{                                    Bx���l  
�          @�p���33?h��?�33A��C((���33?5?�G�A�33C*��                                    Bx���  T          @�\)��G���{?!G�A33CI��G���
=>���@�CJ�H                                    Bx����  �          @�
=��Q쿅�?=p�A��CA���Q쿑�?
=@��CB�                                    Bx���^  �          @�����
=�aG�?W
=A1�C?�=��
=�}p�?5A�CA5�                                    Bx���  T          @�ff���H��  ?W
=A/\)CH����H����?�RAG�CIaH                                    Bx��۪  �          @�
=��\)����?5A(�C8���\)����?(��Ap�C9k�                                    Bx���P  �          @���\)?fff?\(�A3�
C'����\)?E�?z�HAM�C)��                                    Bx����  �          @�p���\)>u?�  A��C0����\)=u?��\A�p�C3=q                                    Bx���  �          @�  ���>���?�z�ApQ�C.�����>k�?��HAz{C0�                                    Bx��B  �          @�\)����?=p�?���Ad��C*(�����?�?���Ax��C,Y�                                    Bx��$�  T          @�  ����?.{?�
=As33C*������?   ?�G�A��RC-J=                                    Bx��3�  �          @�����
=?�p�?�\)Ad��C#�\��
=?�ff?��A�  C&�                                    Bx��B4  �          @�
=���?�33?z�HAN=qC!����?��R?�
=Aw�
C#\                                    Bx��P�  �          @�=q��
=�u?fffA5�C4����
=�B�\?aG�A1G�C6h�                                    Bx��_�  �          @�����{��Q�?Y��A,z�C5)��{�W
=?Q�A(  C6��                                    Bx��n&  �          @�����z�=u?���A[�C3E��z����?���A[
=C5Y�                                    Bx��|�  �          @�{�����W
=?8Q�A{C?�����p��?
=@���C@G�                                    Bx���r  �          @��������>�@ə�CE�
�����33>�\)@fffCF��                                    Bx���  �          @�
=��(���z�>��R@�  CI����(���Q�=u?@  CJ8R                                    Bx����  �          @�  ���
���
>�Q�@�=qCKs3���
����=��
?���CK�
                                    Bx���d  �          @�ff��p���G���Q쿓33CG�)��p���(����
��33CG}q                                    Bx���
  �          @�(��w����H?�RA��CN�{�w���>��R@��\CO��                                    Bx��԰  �          @����u��ff>�ff@��CP�3�u����>�?��CQ5�                                    Bx���V  �          @�
=���\��>.{@p�CK�H���\����G����CK�                                    Bx����  �          @�
=��  ����>L��@!�CF���  ��33���
�uCF33                                    Bx�� �  �          @���������?W
=A,��CH�������
=?
=@��CJ:�                                    Bx��H  T          @��\�Y���  @�AӮCUp��Y���!�?�33A�ffCX�3                                    Bx���  �          @���Z=q�p�?�p�A�Q�CW���Z=q�,��?���A��RCZaH                                    Bx��,�  �          @�  �n�R�
�H?�G�A�Q�CR.�n�R�?fffA9��CT�                                    Bx��;:  �          @��\�n�R�p�?�Q�A���CR���n�R�=q?���AY�CT�3                                    Bx��I�  �          @��\�o\)��
?��A�=qCP޸�o\)�33?��A�p�CS��                                    Bx��X�  �          @�=q�|(����?�ffA���CM���|(���
?z�HAG�CO��                                    Bx��g,  �          @��\��z�˅?���A_�CH����z�޸R?W
=A(��CJ��                                    Bx��u�  �          @��\���R��Q�?�ffAS�CF����R�˅?L��A!G�CH�q                                    Bx���x  �          @�=q���\�У�?�G�A�z�CI�=���\��?z�HAF�HCK�R                                    Bx���  �          @����\)��(�?�p�A}CKT{�\)��33?p��A@  CMp�                                    Bx����  �          @����|(���{?\A�
=CJG��|(���?�p�A}CM
=                                    Bx���j  �          @����~{���?�z�A�
=CJk��~{����?�\)Aep�CL�                                    Bx���  �          @��������{?�ffA��CI�H������ff?��\AO�CK�                                    Bx��Ͷ  T          @����~�R�ٙ�?�  A���CK��~�R���?s33AC33CMO\                                    Bx���\  �          @����\)��ff?k�A8z�CH&f��\)��
=?&ffACI��                                    Bx���  �          @��
�����33?�Q�An=qCFT{�����=q?p��A;33CHxR                                    Bx����  �          @����xQ��ff?�
=A�p�CI�R�xQ��?�33A��HCL�3                                    Bx��N  �          @��R�tzῳ33?�  A�CH(��tz��
=?��RA���CK��                                    Bx���  �          @�ff��zῙ��?���Ad��CD)��z΅{?aG�A7
=CF:�                                    Bx��%�  �          @�
=��33��Q�?��A�  CD&f��33���?��Aa�CF�q                                    Bx��4@  �          @�  ����z�?���Ax(�CCs3�����?}p�AK33CE�\                                    Bx��B�  �          @����G���\)?0��A��CB����G���(�>�@ƸRCC��                                    Bx��Q�  �          @�G����Ϳ��?��\AP(�CHff���Ϳ�Q�?=p�A�\CJ:�                                    Bx��`2  �          @�
=��녿�
=?@  A33CJxR��녿��>�ff@���CK�R                                    Bx��n�  �          @�Q���{��(�?@  A��CGaH��{��=q>�@�z�CH��                                    Bx��}~  T          @�\)��G���=q?8Q�A�
CB���G���Q�?�@�CCxR                                    Bx���$  �          @����������?G�A�CH�3�����
=?   @��CJ
=                                    Bx����  �          @�\)���H���?�A�z�CB����H����?��HA{33CE�                                    Bx���p  �          @�{���ÿ�ff?���A�Q�CE����ÿ�G�?���A^�HCH��                                    Bx���  �          @�{��G�����?��HA�CB�
��G����?�  A��RCE��                                    Bx��Ƽ  �          @�����
�L��?��A���C>�����
���\?�33ArffCA�H                                    Bx���b  �          @�����Ϳp��?���Ao
=C@�=���Ϳ���?s33AG33CC8R                                    Bx���  �          @��H�~{�s33?�
=A�Q�CA���~{��Q�?��RA���CD�q                                    Bx���  �          @�\)�z=q�h��?��A�{CA&f�z=q����?�\)Av{CD(�                                    Bx��T  �          @����\)�\(�?�@���C?� ��\)�p��>�Q�@�\)C@�=                                    Bx���  �          @�z���녿!G�?
=q@�33C<G���녿8Q�>�
=@��C=k�                                    Bx���  "          @��������?�@�p�C9@ ����   >��H@�33C:��                                    Bx��-F  �          @�{���Ϳ�ff��G����
CB+����Ϳ�G�������=qCA��                                    Bx��;�  �          @�{�Y���%���z��s33CY@ �Y������{��33CVh�                                    Bx��J�  �          @��o\)��\�����ffCSz��o\)�	���xQ��J{CQ�H                                    Bx��Y8  �          @�Q��u�33�����w�CR��u��Ϳ8Q��Q�CQ�)                                    Bx��g�  �          @�
=�vff��׼#�
��G�CRu��vff�{��(���  CR�                                    Bx��v�  �          @��s33�G�>��?�p�CR�H�s33��׾�=q�a�CR                                    Bx���*  �          @�p��c33�&ff=��
?�=qCX(��c33�$z��������CW�
                                    Bx����  �          @�z��hQ���>��@XQ�CU�{�hQ��(��L���*�HCU��                                    Bx���v  �          @�{�p���p�?8Q�Ap�CRp��p����
>�z�@n�RCS��                                    Bx���  �          @�p��W
=�-p�?L��A'33CZ��W
=�3�
>�=q@`  C[��                                    Bx����  �          @��
�z=q���
>�
=@�=qCLk��z=q����=�\)?z�HCL�q                                    Bx���h  T          @�(��w
=��z�?��A���CH��w
=�У�?�  AVffCJ�                                    Bx���  T          @����\���  ?�=qAj=qCU!H�\����H?!G�A
=CW�                                    Bx���  �          @�  �s33��
=?J=qA,Q�CK�
�s33��ff>�ff@���CMY�                                    Bx���Z  �          @�����  ���?�\@�CG5���  ��(�>u@J�HCH#�                                    Bx��	   �          @���o\)��p�?E�A&{CO�)�o\)�>\@���CQ0�                                    Bx���  �          @��\��녿�z�>��@eCC�f��녿�Q�<�>��CDG�                                    Bx��&L  �          @�(�����>\)�8Q��
=C2�����>�\)�+��{C0G�                                    Bx��4�  �          @�����33��p��\��
=C8�\��33��\)��G���=qC7�q                                    Bx��C�  �          @�p���=q�^�R����VffC?n��=q�O\)��
=����C>�\                                    Bx��R>  �          @�����33�녾u�EC;����33����33����C:�                                    Bx��`�  �          @������׿G�>8Q�@
=C>W
���׿L��<��
>�=qC>��                                    Bx��o�  �          @��\���þ�����
=C:k����þ�ff�aG��;�C9��                                    Bx��~0  �          @��
�����G�����{CAaH����z�H�k��>�RCA\                                    Bx����  �          @��\������=q�O\)�+�CF(�������녿���a�CC��                                    Bx���|  �          @�����������u�E�CE�3������  ���
����CE5�                                    Bx���"  �          @������h�ÿ
=q��  C@Y�����G��5�G�C>�f                                    Bx����  �          @������J=q��\���C>������+��&ff�p�C=�                                    Bx���n  �          @�=q��ff��\)�G��(��C7�H��ff���Tz��1�C5��                                    Bx���  �          @�=q��  �0�׾W
=�1�C=B���  �#�
��33���C<�                                     Bx���  �          @�����Q�������~�RC;)��Q��������33C:#�                                    Bx���`  �          @������þ�33�����C8�����þ��
�aG��=p�C88R                                    Bx��  �          @�\)���R��33�u�G�C8�����R��zᾞ�R�\)C7�                                    Bx���  �          @�
=��������\)�j=qC7�{���k���{��(�C6�q                                    Bx��R  �          @����ff���
�fff�@(�C8ff��ff���p���J�\C5                                    Bx��-�  �          @�=q���=�G��L���+
=C2�����>�=q�B�\�"=qC0L�                                    Bx��<�  �          @��R���=��Ϳ!G���C2�3���>k��
=��C1�                                    Bx��KD  �          @����33>��#�
��C2J=��33>�������=qC0�\                                    Bx��Y�  T          @�33��G�>L�Ϳ+���C1^���G�>��ÿ(�� ��C/��                                    Bx��h�  �          @�����>�  �������C0�����>�p�����z�C/)                                    Bx��w6  �          @�(���=q>W
=�+����C1+���=q>�33�(�� (�C/Y�                                    Bx����  �          @����=q>��#�
�
=qC-����=q?�\�u�@  C-O\                                    Bx����  �          @�(���=q?333>B�\@\)C*� ��=q?&ff>�{@��C+}q                                    Bx���(  �          @��H��Q�?J=q>L��@,��C)����Q�?:�H>�p�@�z�C*Q�                                    Bx����  �          @�����p�?h��>��@G�C'����p�?\(�>�{@��
C(^�                                    Bx���t  �          @�����?^�R>�{@��
C(E��?E�?�@�G�C)��                                    Bx���  �          @������?Tz�=#�
?\)C)
=����?L��>aG�@:�HC)k�                                    Bx����  �          @�����R?Y��>aG�@9��C(�����R?G�>���@��C)u�                                    Bx���f  �          @������R?J=q>��
@���C)W
���R?333>��H@�Q�C*�                                    Bx���  �          @����\)?=p�>u@P  C*#���\)?(��>���@��HC+
                                    Bx��	�  �          @����Q�?+�=���?��C*�q��Q�?!G�>�  @UC+�                                    Bx��X  �          @��\����?z�>�33@��C,L�����>�>��@�  C-�                                    Bx��&�  �          @��
��=q>�33?�@ۅC/W
��=q>k�?
=@�
=C0�q                                    Bx��5�  �          @������?
=>�G�@�G�C,&f����>�?\)@�z�C-�H                                    Bx��DJ  �          @��H��  ?�\?.{A�C-&f��  >�33?E�A$��C/W
                                    Bx��R�  �          @��R���\>\?aG�A7�
C/  ���\>.{?p��AD��C1�q                                    Bx��a�  �          @�(�����?(�?&ffA	��C+�H����>�ff?E�A#\)C.�                                    Bx��p<  �          @����>��?�
=Ax(�C.p����>�?�  A��HC2.                                    Bx��~�  �          @�z�����>��R?z�@�z�C/�f����>.{?!G�A�C1�                                     Bx����  �          @�{��녽L��?�ffA[�C4����녾�z�?�G�AS\)C7޸                                    Bx���.  �          @�z����\>.{?(�A   C1Ǯ���\<��
?!G�A��C3�                                    Bx����  �          @�(����
�#�
>��@[�C4.���
��\)>�  @S33C4��                                    Bx���z  �          @�z���(�    >�z�@o\)C4���(���\)>�\)@hQ�C4޸                                    Bx���   �          @�33���>#�
>��H@�Q�C1�����=#�
?�@�=qC3xR                                    Bx����  �          @��\����>�(���Q쿡G�C.B�����>�G�<#�
=���C.�                                    Bx���l  �          @������?
=��33���C,8R����?&ff�L���(Q�C+aH                                    Bx���  �          @�z�����?�\�+��Q�C-+�����?(�ÿ���{C+Q�                                    Bx���  �          @�=q���R>��Q��/�C-����R?#�
�0���\)C+^�                                    Bx��^  �          @������>�G��8Q��(�C.0�����?
=�����z�C,)                                    Bx��   �          @��\���R>��ÿn{�F�RC/�����R?
=q�Tz��0z�C,�R                                    Bx��.�  �          @�33��ff>��ÿ���b{C/����ff?녿s33�J�RC,L�                                    Bx��=P  �          @�p�����?�ͿW
=�/�C,�q����?:�H�0���
=C*Y�                                    Bx��K�  �          @�����?(�ÿn{�B=qC+(�����?^�R�@  �\)C(��                                    Bx��Z�  �          @�������>����h���>ffC0�����?�\�O\)�)��C-G�                                    Bx��iB  �          @��
����>�33���\���
C/8R����?#�
��33�up�C+8R                                    Bx��w�  T          @����z�H>B�\������C1B��z�H?z��ff���HC+��                                    Bx����  �          @�ff��33���\(��=G�C4^���33>8Q�W
=�9�C1xR                                    Bx���4  �          @�\)��>��
��\��(�C/����>�(�������
C.)                                    Bx����  �          @��R��p�>.{��\��\)C1����p�>�z��ff��
=C0\                                    Bx����  �          @����G�?O\)���R���C(�3��G�?\(���G���  C'�q                                    Bx���&  �          @���u�?���c�
�J{C$�)�u�?��R�(����C"\                                    Bx����  �          @��
����?�R����\)C++�����?@  �������HC)�                                     Bx���r  T          @��\�vff?W
=�s33�X��C'�f�vff?�ff�5�"ffC$�q                                    Bx���  �          @����  ?333�.{��
C*���  ?W
=���H����C(
                                    Bx����  �          @�33����?@  �����C)k�����?B�\=#�
?z�C):�                                    Bx��
d  �          @��H�|��=u?W
=A?�C3��|�;��?Tz�A=�C6)                                    Bx��
  �          @�z��  @[�=�G�?�  B��\�  @S33?uAW�B���                                    Bx��'�  �          @�=q�,(�@S33?+�A�
C(��,(�@A�?�(�A�(�C�
                                    Bx��6V  �          @�33�]p�@%�?
=q@�C:��]p�@�?�Ay�C�)                                    Bx��D�  �          @����Z=q@#33>�33@���C@ �Z=q@Q�?xQ�AR�HC)                                    Bx��S�  �          @����Q�@(�?.{A�
CY��Q�@(�?��\A�p�CG�                                    Bx��bH  T          @��H�a�@�
?xQ�AO�CǮ�a�?��R?��
A���C�{                                    Bx��p�  �          @���C33@B�\?aG�A7�
C�C33@.{?�\)A���C
5�                                    Bx���  �          @�=q��R@W�?��Ab�\B��
��R@?\)?��A�(�C��                                    Bx���:  �          @�=q�?\)@8Q�?�33Av�\C��?\)@\)?�A�Q�C33                                    Bx����  T          @�z��G�@2�\?��\A���C
8R�G�@�?�Q�A�ffC�                                    Bx����  �          @��\�(��@_\)?�=qA���B�(��(��@A�@�A�p�C�                                    Bx���,  �          @���*=q@`��?�\)AeG�B�W
�*=q@Fff?��RAϙ�C��                                    Bx����  �          @�  �J=q@-p�?�p�A�33Cp��J=q@33?��AͅC��                                    Bx���x  �          @�Q��n�R@p�?p��AB�RCO\�n�R?�33?�p�A�p�C�                                    Bx���  �          @���Q�@?\)?(�@�=qC	���Q�@.�R?�{A�CB�                                    Bx����  �          @�Q��k�@#33>��H@�
=CO\�k�@�?���AiG�C�H                                    Bx��j  �          @�Q��\)@G�>�33@�{C(��\)?�{?\(�A1�C�                                    Bx��  �          @����?�ff>�(�@��C���?���?Q�A)��C!�R                                    Bx�� �  �          @�ff���?�=q?#�
A��C%� ���?^�R?fffA;�C(c�                                    Bx��/\  �          @�Q�����?��\>��H@�ffC#������?���?L��A$z�C%��                                    Bx��>  �          @�Q�����?��>��
@��C"����?�  ?.{A
ffC#��                                    Bx��L�  T          @�����ff?���=u?8Q�C!H��ff?��
>�@ÅC�q                                    Bx��[N  �          @��\��=q?�p�=#�
?   C!���=q?�>�(�@�C!�\                                    Bx��i�  �          @�G���
=?�<�>�(�Cc���
=?���>�@�G�C5�                                    Bx��x�  �          @�=q��  ?��u�B�\C����  ?�\)>Ǯ@�33C�                                    Bx���@  �          @�=q��?���33���\C�=��?�>��?�z�C=q                                    Bx����  �          @�G����?�p������
Cz����?�ff�#�
��C�{                                    Bx����  �          @��\��  ?���333�G�C ���  ?�����R�\C�\                                    Bx���2  �          @��R��(�?�=q������C
��(�?У�<�>ǮCu�                                    Bx����  �          @�{����?�\)�G��!G�C!������?��
�\���
C�                                    Bx���~  �          @�����  ?�=q�333��RC"���  ?�(����R�~�RC ��                                    Bx���$  �          @����G�?xQ�aG��5��C':���G�?�
=�
=��  C$��                                    Bx����  �          @������?���G��
=C%�\���?�G���ff��  C#��                                    Bx���p  �          @�����?�����]p�C!@ ��?�z�(����CO\                                    Bx��  �          @������?�p���\)�c�
C#�
���?��R�=p��C �f                                    Bx���  �          @�  ��\)?+����
���RC*�3��\)?�  ����W�C&�R                                    Bx��(b  �          @�(����
?333��������C*޸���
?����=q�W33C&��                                    Bx��7  �          @������>�G����
���C.!H����?Y����������C(�
                                    Bx��E�  �          @�����?333�����z�C*�����?5=L��?333C*�f                                    Bx��TT  T          @�(����?��
?5A�RC&����?J=q?xQ�AN=qC)=q                                    Bx��b�  �          @�����  ?�������|��C%����  ?���<#�
>.{C%{                                    Bx��q�  �          @��H���?n{�#�
�8Q�C'�����?c�
>�  @O\)C()                                    Bx���F  �          @��\��z�?�  �#�
�\)C#@ ��z�?��H>��R@�33C#��                                    Bx����  �          @�33���?��\���Ϳ���C#����?��R>��@[�C#\)                                    Bx����  �          @�(���ff?�>#�
@�C$ff��ff?�=q>��H@�(�C%�{                                    Bx���8  �          @�33��{?�{>��
@�ffC%���{?z�H?�RA33C&�)                                    Bx����  �          @����p�?�ff>�Q�@�33C%����p�?h��?&ffA
=qC'�R                                    Bx��Ʉ  �          @����ff?n{>k�@A�C'����ff?Tz�>�@�p�C(޸                                    Bx���*  �          @������R?L�;W
=�0��C):����R?Tz�<�>ǮC(��                                    Bx����  �          @�����z�?J=q�333�
=C)33��z�?u��ff��  C&��                                    Bx���v  �          @�ff��(�?&ff��{��33C+���(�?8Q�\)��{C*�                                    Bx��  �          @��
�~{?E��5�   C(���~{?s33���θRC&�)                                    Bx���  �          @�����Q�>�����ffC-=q��Q�?       ��\)C,�                                    Bx��!h  �          @�33����=#�
�(����C3h�����>aG���� Q�C0�3                                    Bx��0  �          @�����
=��׾W
=�4z�C:\)��
=�Ǯ��{����C9G�                                    Bx��>�  �          @����
=��>B�\@(Q�C5����
=�.{>��@33C6O\                                    Bx��MZ  �          @�  ���=�>B�\@!G�C2n���=u>W
=@7�C3:�                                    Bx��\   �          @�(����H�#�
��ff�ʏ\C6.���H���
����{C4=q                                    Bx��j�  �          @�z�����k����ٙ�C7:�����B�\�8Q��$z�C6�
                                    Bx��yL  �          @�����(���{=u?aG�C8����(���{�#�
�
=qC8Ǯ                                    Bx����  �          @���z�Ǯ>�  @`��C9s3��z��ff>�?�G�C::�                                    Bx����  �          @�������(��k��K�C9�������{��{����C8�R                                    Bx���>  �          @�{���;\������Q�C9:����;�=q��(���33C7��                                    Bx����  �          @�p������=q=�Q�?��HC7������z�    =L��C7��                                    Bx��  �          @�ff����Q�<#�
>8Q�C8�H����33��Q쿢�\C8�R                                    Bx���0  �          @�ff��p����>L��@1�C9����p���ff=�\)?z�HC:5�                                    Bx����  �          @�ff����   =#�
?
=C:�
������H��G���(�C:��                                    Bx���|  �          @�\)��
=�.{=u?Q�C6T{��
=�8Q�<#�
=��
C6xR                                    Bx���"  �          @��R�|��=��
�O\)�:ffC2���|��>��
�@  �,(�C/Y�                                    Bx���  �          @�G���Q�>�zῺ�H��\)C/�\��Q�?8Q쿦ff���
C)޸                                    Bx��n  �          @�����
=>u���H�ӅC0����
=>\�Ǯ���C.�H                                    Bx��)  �          @�������>��5�p�C-h�����?+�����p�C*�{                                    Bx��7�  �          @�\)���H?+��J=q�,��C*�����H?aG������C'ٚ                                    Bx��F`  �          @�ff����?aG��J=q�-��C'�3����?�=q���H��C%
=                                    Bx��U  �          @�ff�}p�?�ff�k��K�C%+��}p�?��
�\)���RC"�                                    Bx��c�  �          @���}p�?p�׾����z�C&��}p�?���.{�C%T{                                    Bx��rR  �          @�33��Q�?L��=�Q�?�p�C(���Q�?=p�>���@��HC)��                                    Bx����  T          @�(���=q?.{>�?޸RC*s3��=q?(�>���@���C+k�                                    Bx����  �          @�����Q�?k�>���@��C'!H��Q�?@  ?+�A  C)k�                                    Bx���D  �          @�=q�y��?�{�\)���RC$(��y��?���>L��@8��C$L�                                    Bx����  �          @�=q�|(�?z�H=�G�?ǮC&�|(�?fff>��@��C'#�                                    Bx����  "          @�G��{�?fff>#�
@\)C'��{�?O\)>�(�@�33C(Q�                                    Bx���6  �          @����z=q?p��<#�
=��
C&p��z=q?c�
>���@��RC'�                                    Bx����  �          @����r�\?���W
=�>{C ޸�r�\?��>W
=@>{C ޸                                    Bx���  
�          @\)�x��?Tz�<#�
=�Q�C'��x��?J=q>��@s33C(��                                    Bx���(  T          @��\�q�?��������C��q�?���u�W
=C�{                                    Bx���  
�          @��H�r�\?�\)�+���
C 33�r�\?�G��aG��B�\CT{                                    Bx��t  �          @�z��s33?����  �_33C ��s33?Ǯ�
=q��C��                                    Bx��"  �          @���l(�?�(���G��c�CO\�l(�?��H�   �߮C&f                                    Bx��0�  �          @��H�_\)?�G���(�����C���_\)?�녿n{�R�HC��                                    Bx��?f  T          @�=q�a�?Ǯ���\��  C0��a�?�׿8Q��#\)C�q                                    Bx��N  �          @�G��a�?�
=�xQ��_�C���a�?�33������(�C��                                    Bx��\�  �          @�(��a�?�  ��p����HC s3�a�?�p���  ��G�C��                                    Bx��kX  "          @��
�G
=?�(�������C�f�G
=?�z������Cu�                                    Bx��y�  "          @��H�W
=?�������p�C ٚ�W
=?�(���p�����C޸                                    Bx����  "          @��
�Vff?���   ��
=C���Vff?�{���R���C�R                                    Bx���J  "          @��H�g
=?s33����z�C%J=�g
=?���B�\�8��C#��                                    Bx����  
�          @��H�|(�>B�\?���As
=C1.�|(��#�
?�=qAtz�C6E                                    Bx����  T          @�33�tz�?��?=p�A(��C$#��tz�?J=q?�ffApz�C(E                                    Bx���<  
Z          @���u�?��?(�A	G�C!!H�u�?�ff?�  Aap�C$��                                    Bx����  �          @�33�r�\?��\?n{AP��C!���r�\?fff?��
A��C&�{                                    Bx����  �          @��H�l��?�\)?   @�=qC\)�l��?�\)?}p�Ab{C�H                                    Bx���.  "          @�(��b�\?�33?O\)A5p�C��b�\?��?�\)A���CxR                                    Bx����  �          @��H�c33?�33?(�A
=qC�H�c33?���?�Q�A��
C�                                     Bx��z  �          @�33�u?���>aG�@G
=C G��u?�p�?&ffA\)C"8R                                    Bx��   �          @��
�xQ�?��
?�\@�\C!���xQ�?�ff?fffAH��C$��                                    Bx��)�  �          @�z��x��?��>�p�@��\C!��x��?�33?G�A.{C#��                                    Bx��8l  �          @��u�?˅>��@θRCz��u�?���?uATz�C �
                                    Bx��G  �          @�p��tz�?���?Q�A7
=C ��tz�?��?�p�A��C$                                    Bx��U�  �          @�{�2�\���@G�B@ffC9B��2�\��G�@7�B,��CLJ=                                    Bx��d^  �          @��R�I��=u@1G�B%33C2���I���Y��@(��B��CC�                                    Bx��s  �          @��S33>��@!G�B=qC+���S33�\@!�BQ�C:��                                    Bx����  �          @�ff�l(�?@  ?�{A���C(xR�l(�=���@ ��A�z�C2n                                    Bx���P  T          @����W
=?Y��@  BC%�q�W
==�\)@��B=qC2��                                    Bx����  �          @���B�\?Q�@*�HB!
=C$ٚ�B�\����@2�\B*{C5�)                                    Bx����  �          @����O\)?�@33B��C +��O\)>�33@#�
Bz�C-�f                                    Bx���B  �          @��@��?��@+�B ffC ���@��=���@7�B.��C2(�                                    Bx����  �          @�\)�:�H?�\)@4z�B(ffC��:�H=�@A�B8\)C1��                                    Bx��َ  �          @�\)�Mp�?�Q�@p�B��C���Mp�>��R@-p�B 33C.��                                    Bx���4  �          @�
=�W�?�=q@	��A��C� �W�?
=q@{B�C*�                                    Bx����  �          @��Fff?���@Q�B
=C��Fff?z�@.{B#33C)h�                                    Bx���  �          @��?\)?���@%B�RC.�?\)>\@8Q�B.��C,��                                    Bx��&  �          @��:=q?�z�@ ��B�\C@ �:=q?:�H@:�HB1(�C%��                                    Bx��"�  �          @���7
=?�  @(�B
=C�)�7
=?Tz�@8Q�B0
=C#�                                    Bx��1r  �          @�  �HQ�?�33@ffB33C&f�HQ�?E�@1G�B"z�C&)                                    Bx��@  �          @����J=q?�ff@
=BQ�C�3�J=q?+�@/\)B!(�C(�                                    Bx��N�  �          @�\)�R�\?�33?��RA���CO\�R�\?c�
@�HB��C$�{                                    Bx��]d  �          @����L(�?���@�B��C�L(�?.{@0��B!  C'��                                    Bx��l
  �          @�G��E�?�
=@  A�(�C�3�E�?��@1G�B!p�C!�                                    Bx��z�  �          @�Q��I��@?�G�A�Ck��I��?�\)@
=B

=Ck�                                    Bx���V  �          @���J�H?�p�@��A�=qCh��J�H?c�
@)��BG�C$c�                                    Bx����  �          @�  �Fff?�G�@&ffB(�C���Fff>��R@7�B*�C.T{                                    Bx����  �          @�\)�O\)?�  @\)B �\C(��O\)?&ff@'�B�\C(��                                    Bx���H  �          @��R�Vff?���@�A���C���Vff?(��@��BffC(�)                                    Bx����  �          @��R�HQ�?�p�@Bp�C���HQ�>�Q�@'�B  C-s3                                    Bx��Ҕ  �          @��7
=?5@<��B4p�C&��7
=��{@AG�B9�\C:�                                    Bx���:  �          @�{�E�?@  @.{B"33C&T{�E��aG�@3�
B)  C8�                                    Bx����  �          @�{�J=q?Y��@&ffB��C%
=�J=q��Q�@.�RB#{C5�                                    Bx����  �          @����H��?Tz�@2�\B"�\C%0��H�þ.{@:=qB*�HC7(�                                    Bx��,  �          @��H�s33?c�
?�{A�=qC&�H�s33>B�\@33A��HC1{                                    Bx���  �          @��
�n�R?Q�@�A�Q�C'�
�n�R=u@\)A�33C3)                                    Bx��*x  �          @�p��]p�?�(�?��HAׅC���]p�?p��@�BC$��                                    Bx��9  �          @�p��[�@(�?�Q�A���CaH�[�?�p�@B �C��                                    Bx��G�  �          @��R�Tz�@  ?�\)A��
C���Tz�?�(�@!�Bp�C+�                                    Bx��Vj  �          @����I��?�
=@&ffBffC��I��?h��@FffB-��C#�{                                    Bx��e  �          @�Q��@��?��
@1�B��CaH�@��?5@N{B8�C&��                                    Bx��s�  �          @��\�I��@#�
@ ��Aң�C���I��?��H@1G�B  C}q                                    Bx���\  �          @��\�G�@P��?(��A�C�\�G�@4z�?�  A���C	��                                    Bx���  �          @�G��C�
@Dz�?�G�A�{Cٚ�C�
@(�@  A�p�CaH                                    Bx����  �          @��\�?\)@J�H?���A��RCQ��?\)@\)@��A��C33                                    Bx���N  �          @���@  @B�\?��
A�G�C���@  @�
@\)B{CaH                                    Bx����  �          @����C33@J=q?��A^�\C�3�C33@%�@�A��
C�R                                    Bx��˚  �          @�  �G
=@L��?
=@�  C(��G
=@2�\?�A�(�C
\                                    Bx���@  �          @����B�\@P  ?G�A\)C)�B�\@1G�?�\)AÅC	��                                    Bx����  �          @����7�@S�
?���Af�RC�f�7�@,��@p�A�\C��                                    Bx����  �          @�
=�*=q@X��?���Aj�HC !H�*=q@1�@  A�Q�C�                                    Bx��2  �          @��
�#33@R�\?�ffA��HB�z��#33@'�@Q�B  C&f                                    Bx���  �          @��\�  @dz�?s33AK�B� �  @@  @	��A�{B��                                    Bx��#~  �          @��
�@fff?Tz�A.ffB���@Dz�@�\A޸RB��\                                    Bx��2$  �          @���	��@k�?#�
A��B��\�	��@Mp�?�33A�
=B�u�                                    Bx��@�  �          @��
��
=@~{?!G�A�B��Ϳ�
=@_\)@   A�=qB�W
                                    Bx��Op  �          @�{��@{�?s33AEp�B�W
��@U@�A��B�                                    Bx��^  �          @���G�@z�H?z�HAL��B�8R��G�@Tz�@33A�  B�3                                    Bx��l�  �          @��Dz�@@  ?uAI�C�H�Dz�@{?��HA�{C8R                                    Bx��{b  T          @��\�(Q�@L��?��Af{Ck��(Q�@'
=@Q�A���C.                                    Bx���  �          @������@k�?�G�A�Q�B��H����@?\)@\)B=qB�=q                                    Bx����  �          @�=q� ��@dz�?�ffA�Q�B�� ��@8Q�@�RB�B��                                    Bx���T  �          @��H�2�\@QG�?
=q@�
=Cc��2�\@7
=?�
=A�{C33                                    Bx����  �          @�z��X��@.�R>�p�@�G�C=q�X��@�H?��A�{C��                                    Bx��Ġ  �          @����z�@p  >�  @S�
B��q�z�@Z�H?���A�{B�ff                                    Bx���F  �          @���'
=@S33>�Q�@�(�C W
�'
=@<��?\A�ffCxR                                    Bx����  �          @�(��5�@S�
>�33@�(�C� �5�@=p�?\A�z�C�f                                    Bx���  �          @�z��+�@\(�>���@�\)B���+�@Dz�?�{A�=qC0�                                    Bx���8  �          @�z��.{@Z=q>�33@��
C ���.{@C�
?�ffA��\C�f                                    Bx���  �          @�=q�QG�@/\)?��@��C��QG�@
=?�  A�C0�                                    Bx���  �          @�z��Z=q@-p�>���@�{C��Z=q@��?���A�p�C                                    Bx��+*  T          @�ff�?\)@R�\��Q쿓33CE�?\)@Fff?���Ae�C��                                    Bx��9�  �          @��0��@[�>Ǯ@�  C ٚ�0��@C�
?���A���C
                                    Bx��Hv  �          @���>�R@Mp�>�=q@^�RC��>�R@8��?�z�A�C�H                                    Bx��W  �          @���E�@H�þW
=�0  Cz��E�@@  ?p��AE�C�                                     Bx��e�  �          @����C�
@Fff�(���	G�C���C�
@G�?�\@�p�C^�                                    Bx��th  �          @��Dz�@J=q������\)C.�Dz�@E?E�A ��C�\                                    Bx���  �          @��I��@G
=�����u�CW
�I��@@  ?Y��A1CT{                                    Bx����  �          @�Q��%@fff>���@�B��\�%@L��?�
=A��C ��                                    Bx���Z  �          @���p�@q�?�@�RB�(��p�@S�
?�z�AƏ\B�=q                                    Bx���   �          @�=q�Q�@~�R?\)@�(�B�Q��Q�@`  ?�p�A�{B��                                    Bx����  �          @�G���@���?333AffB�#׿�@_\)@�A�  B�                                    Bx���L  �          @��\���@��H?�  AH��B�\)���@\(�@�HA���B�                                    Bx����  �          @��H��\)@���?��A]�B��)��\)@Vff@\)B��B�ff                                    Bx���  �          @�Q��
=@z=q?�\)Ae�B�\��
=@N�R@{B33B�                                    Bx���>  �          @�=q���@~�R?�(�Ax  B��ÿ��@P  @%B
\)B�u�                                    Bx���  �          @�33��
=@~�R?��\A�Q�B�׿�
=@O\)@(��BQ�B�                                    Bx���  �          @����@��?�@�B����@dz�@G�A���B�W
                                    Bx��$0  �          @����
=q@|(�?�\@�ffB�=�
=q@^�R?�A��B�Ǯ                                    Bx��2�  �          @�Q����@w�?Q�A'�B��
���@S33@(�A�RB��                                    Bx��A|  �          @���(�@p  ?���AjffB�{�(�@Dz�@(�B��B�\                                    Bx��P"  �          @�  ��@u?\(�A0  B�L���@P  @p�A��B���                                    Bx��^�  �          @�\)��
@p  ?J=qA#33B�8R��
@L(�@�A�z�B��q                                    Bx��mn  �          @�{�9��@HQ�?���A`��C���9��@!G�@	��A�=qC                                      Bx��|  �          @�Q��5�@Q�?��RA�
C�\�5�@%�@
=A�p�C	�
                                    Bx����  �          @����"�\@e?fffA8��B��{�"�\@@  @
�HA�RCB�                                    Bx���`  T          @���3�
@]p�?�@���C
�3�
@AG�?�G�A�\)C�                                    Bx���  �          @���� ��@p��=�?��
B�p�� ��@\��?�G�A��B��                                    Bx����  �          @����0  @c33>�(�@�{B�z��0  @H��?�(�A���C5�                                    Bx���R  �          @���<��@\(�    ��C���<��@L��?��
A�
=C��                                    Bx����  �          @�=q�5@dz������C ���5@W�?�Q�AqC(�                                    Bx���  �          @����Fff@Q녾�����HCs3�Fff@Fff?�=qA]�C�                                    Bx���D  �          @�G��Mp�@J�H>\)?�G�CT{�Mp�@8��?��A�33C	��                                    Bx����  �          @���J=q@Q�=�?��
C�3�J=q@?\)?��A��\C��                                    Bx���  �          @��\�O\)@K�>�@��\C�=�O\)@1�?�\)A�(�Ch�                                    Bx��6  �          @����5@_\)?#�
A=qC)�5@@  ?�33A�  Cs3                                    Bx��+�  �          @��\�N�R@J�H?&ffA�HC���N�R@,��?��A�z�C!H                                    Bx��:�  �          @�  �QG�@AG�?+�A��C	Q��QG�@#33?�  A�G�C)                                    Bx��I(  �          @��R�S33@<��?�@�z�C
+��S33@"�\?���A���Cp�                                    Bx��W�  �          @�\)�U@7
=?:�HA
=Cz��U@�?�  A���C��                                    Bx��ft  �          @�G��g
=@+�?��@߮CaH�g
=@�?\A��RC�q                                    Bx��u  �          @�G��N{@I��?
=@�\)C�{�N{@,��?�p�A��C�                                    Bx����  �          @����L��@L(�>�Q�@�Q�C
=�L��@4z�?��
A�{C
��                                    Bx���f  �          @����^�R@7
=>\@���C�)�^�R@ ��?�
=A�
=C=q                                    Bx���  �          @���XQ�@>{?(�@�ffC
� �XQ�@!G�?�
=A��CW
                                    Bx����  �          @���5@e�=�\)?Tz�C \)�5@R�\?�z�A�(�C��                                    Bx���X  �          @����Fff@\(����R�uC{�Fff@S33?�G�AIG�C=q                                    Bx����  �          @����=p�@^{�
=q����Cz��=p�@Z�H?O\)A"ffC��                                    Bx��ۤ  �          @��H��@z=q�z�H�FffB�=q��@\)?�@�(�B�=q                                    Bx���J  �          @����@|(��c�
�5�B��@~�R?(��AffB�z�                                    Bx����  �          @�
=� ��@j�H��(���  B���� ��@c�
?�  AN{B�G�                                    Bx���  �          @�\)���@k��0����B�33���@j=q?B�\A��B�k�                                    Bx��<  �          @�ff��
@l(��^�R�5��B���
@p  ?
=@�p�B�B�                                    Bx��$�  �          @��R�	��@qG����\�T��B��	��@w�>�@ƸRB�#�                                    Bx��3�  �          @�{�p�@mp����\�Up�B��p�@tz�>�@�\)B�33                                    Bx��B.  �          @�{��@g
=�����ep�B��{��@p��>�Q�@�z�B�=                                    Bx��P�  �          @�{��@l�Ϳ�(���B���@x��>�\)@h��B��                                    Bx��_z  �          @�{�	��@k����H�}�B��=�	��@w
=>�\)@j=qB�#�                                    Bx��n   �          @���@mp����R��{B�{��@y��>�=q@]p�B�#�                                    Bx��|�  �          @�  ��@u���\)���B�#׿�@���#�
��B�Ǯ                                    Bx���l  �          @��
��  @{��z��ՙ�B�k���  @�p���
=��BӨ�                                    Bx���  �          @����  @j�H�
=��Q�Bٔ{��  @�ff�����33B�#�                                    Bx����  �          @�\)���H@��ÿ��R��ffBՊ=���H@�G�>#�
@�
BӔ{                                    Bx���^  �          @�����
=@��
��  ��{B��ÿ�
=@�(�>B�\@(�B�(�                                    Bx���  �          @�(����\@z=q�����G�B�\���\@��>�z�@tz�B�aH                                    Bx��Ԫ  �          @�=q��Q�@q녿��ˮB��쾸Q�@��aG��AG�B�Ǯ                                    Bx���P  �          @�G��n{@p�׿�G���
=B�Ǯ�n{@��;8Q���HB�G�                                    Bx����  �          @�녿z�H@��H�^�R�:�HB��z�H@��
?E�A$z�B���                                    Bx�� �  �          @����fff@�녿�  �X��B���fff@�z�?!G�A(�B̨�                                    Bx��B  �          @�=q��33@����@  �!p�BӨ���33@���?\(�A:�\B���                                    Bx���  
�          @�(��fff@�  ����ƸRB��Ϳfff@��?���A~�\B̞�                                    Bx��,�  
�          @��
�J=q@�G��.{�
�HB�Ǯ�J=q@���?�  A��\B��                                    Bx��;4  T          @�33��ff@��>L��@&ffB��;�ff@y��?�A��HB�                                    Bx��I�  T          @�33�#�
@��H�#�
�
�HB��ý#�
@�=q?\A��
B�                                    Bx��X�  �          @�녿
=q@�Q�8Q��ffB�
=q@�  ?�p�A�
=B�u�                                    Bx��g&  �          @��þ���@�p��E��'
=B�
=����@���?fffAB�RB��                                    Bx��u�  �          @�
=���
@��ÿ�33�~�HB�\���
@��>��H@�ffB�                                    Bx���r  �          @��Ϳ�=q@xQ쾏\)�~{B�#׿�=q@l(�?��RA�
=BԮ                                    Bx���  �          @��
�'�@�@	��A���C
���'�?��
@7
=B1��C�R                                    Bx����  �          @�33�Ǯ@(�ÿ:�H�V=qB�=q�Ǯ@-p�>�33@��B���                                    Bx���d  �          @��
����@aG���
=��z�B�
=����@Z=q?xQ�AmG�B�G�                                    Bx���
  �          @����\@tz�O\)�5B��H���\@u�?:�HA$(�Bؽq                                    Bx��Ͱ  �          @�  �O\)@|��=#�
?z�B�#׿O\)@hQ�?���A��HB�#�                                    Bx���V  T          @��R�5@�G�>\@�Q�B���5@e�?�z�A�\)B�L�                                    Bx����  
�          @�  ����@_\)��z�����B�W
����@j=q>�z�@�33B�G�                                    Bx����  
�          @�Q쿴z�@i���Tz��@Q�B�B���z�@k�?&ffA��B��H                                    Bx��H  S          @��ÿ�ff@j=q�\)�   B�  ��ff@e?fffAPQ�B���                                    Bx���  �          @�(��xQ�@z=q�G��0(�B��f�xQ�@z=q?J=qA2�\B��                                    Bx��%�  �          @�  �W
=@��ÿp���NffBˏ\�W
=@��\?333A�B�G�                                    Bx��4:  �          @�z��@���c�
�<  B�aH��@�  ?Tz�A/�B�W
                                    Bx��B�  �          @��\�aG�@���z�H�Q��B�G��aG�@��?333Az�B���                                    Bx��Q�  �          @��
>�@��ÿУ����B�=q>�@�33=�\)?fffB��                                     Bx��`,  �          @�\)��33@��H�}p��X��B���33@���?.{A  B���                                    Bx��n�  T          @��333@z=q���\��B�W
�333@��H>�33@�33B�p�                                    Bx��}x  �          @�(��W
=@z=q�O\)�8(�B���W
=@z�H?E�A.�\B�\                                    Bx���  �          @�ff>.{@~{�}p��`Q�B���>.{@�G�?!G�A{B�\                                    Bx����  �          @�z�\)@�녿#�
�Q�B��\)@\)?}p�A^ffB�                                    Bx���j  �          @��;���@�G��W
=�;\)B�Q쾨��@���?L��A2ffB�L�                                    Bx���  �          @����@��H�G��,z�B�.���@�=q?aG�AB{B�8R                                    Bx��ƶ  �          @�=q���@��þ�{����B����@�=q?���A��\B��{                                    Bx���\  T          @�녿333@�{�#�
�	G�B��333@��?��Aa�B�ff                                    Bx���  �          @�\)�Ǯ@�(��\(��1�B�G��Ǯ@��?k�A=�B�L�                                    Bx���  �          @�  �B�\@���aG��3�B���B�\@���?h��A;�
B��
                                    Bx��N  �          @�  ��@�(���  �N{B�(���@�?J=qA!B��                                    Bx���  �          @�Q�>W
=@�z�W
=�-�B��=>W
=@��
?p��AAB��                                    Bx���  �          @���>�
=@���fff�733B�33>�
=@��?fffA7�B�33                                    Bx��-@  �          @���>��@�{�=p���RB�\>��@��
?��AYG�B�                                      Bx��;�  T          @���>�
=@�
=����z�B�.>�
=@��?��\A�=qB���                                    Bx��J�  �          @�  ?&ff@�(��0����B�{?&ff@���?�=qA_�B�Ǯ                                    Bx��Y2  T          @�(�?fff@�ff�333���B��3?fff@�z�?�  AS�
B�\)                                    Bx��g�  �          @�=q?�@�{����B���?�@��?�z�A|Q�B��=                                    Bx��v~  �          @����@��\����\)B����@}p�?�\)AyB�L�                                    Bx���$  �          @�?��@~�R�aG��8��B�B�?��@�Q�?@  AG�B��                                    Bx����  T          @��?�@�33�&ff���B��q?�@���?�G�AQp�B�                                      Bx���p  �          @�z�?Ǯ@��׿\)��z�B���?Ǯ@z�H?��Ab=qB���                                    Bx���  �          @��R?���@}p��h���<(�B��?���@\)?8Q�A�B�\)                                    Bx����  �          @�Q�?У�@�33�aG��4��B��?У�@��
?L��A#�B��
                                    Bx���b  �          @�{?��@��׾�(���33B�B�?��@�33?��
A�(�B��                                    Bx���  �          @��R?��@�  �@  ��B�8R?��@�ff?z�HAJ�\B��                                    Bx���  �          @���?�(�@�p��}p��Ip�B�.?�(�@�
=?:�HA(�B���                                    Bx���T  �          @���?�z�@��ͿB�\�
=B�L�?�z�@��?n{A<��B���                                    Bx���  �          @��?�\)@�(��0���p�B�33?�\)@�=q?}p�AK\)B���                                    Bx���  T          @�Q�?˅@�(��c�
�5B���?˅@�z�?L��A$Q�B���                                    Bx��&F  �          @���?�\@|(���33��(�B���?�\@�p�>��@N�RB�\                                    Bx��4�  �          @�p�?u@�\)�(�� Q�B�Q�?u@�(�?���Ah��B��R                                    Bx��C�  �          @��?�
=@�33�=p��33B��=?�
=@��?n{AB�RB�33                                    Bx��R8  T          @��?�G�@w�����\(�B�{?�G�@~{?\)@��B�(�                                    Bx��`�  �          @��>�G�@�Q��R��B�W
>�G�@��?�{Al��B�\                                    Bx��o�  �          @��\>��@��R�fff�@��B�
=>��@�
=?Tz�A0��B�{                                    Bx��~*  �          @�Q�?J=q@�(�����  B�k�?J=q@���?���Ap  B��)                                    Bx����  �          @��=�\)@�{��(����B�=�\)@���?�  A�ffB��                                    Bx���v  �          @�\)��\@�z�L�Ϳ�RB����\@u�?˅A��HB�=q                                    Bx���  �          @������@��׾��H����B�(�����@xQ�?���Ax��B�8R                                    Bx����  �          @����
=@g
=�#�
��\B�LͿ�
=@U�?��A�ffB�.                                    Bx���h  �          @�p��.�R@I��=�?ٙ�C�f�.�R@6ff?���A���CǮ                                    Bx���  �          @���(Q�@L(����
���RC���(Q�@C�
?uAUC�                                     Bx���  �          @�{�J�H@-p����
��  C}q�J�H@   ?�ffAiG�CǮ                                    Bx���Z  �          @�G��Mp�@5�=���?�\)C
�)�Mp�@#�
?��HA��Cn                                    Bx��   �          @���Q�@1G�>u@J=qC�\�Q�@p�?��A�C33                                    Bx���  �          @����W
=@(��>\@��
C���W
=@G�?��A�33C��                                    Bx��L  �          @�(��b�\@!�>�(�@�(�Ck��b�\@
=q?��A�z�C��                                    Bx��-�  �          @��
�e�@\)>�\)@g�C#��e�@�?�p�A�\)C��                                    Bx��<�  �          @���g�@��>���@�  Cu��g�@�?�(�A��RC�                                    Bx��K>  �          @�z��w
=@z�>�  @R�\C���w
=?�ff?�ffA\��C��                                    Bx��Y�  �          @�z��z�H?��H>L��@+�Ch��z�H?�(�?uAJffCG�                                    Bx��h�  �          @�z��s33@�=�Q�?�p�C+��s33?�(�?s33AG33C�f                                    Bx��w0  �          @�z��s33@(�>8Q�@�\C��s33?�Q�?��\AW�
C��                                    Bx����  �          @�(��o\)@��>�z�@o\)C���o\)?�(�?�33AtQ�C@                                     Bx���|  �          @�(��k�@>\@�p�C���k�@   ?��\A���Ck�                                    Bx���"  �          @��
�e@�H?�@���C�3�e@G�?�Q�A���C��                                    Bx����  �          @��
�b�\@   ?�@�33C�b�\@?��HA��RCff                                    Bx���n  T          @��\�X��@!�?n{AE�CB��X��?��H?���A�=qC��                                    Bx���  �          @����aG�@{?��\A\  C��aG�?��?�A�  C
=                                    Bx��ݺ  �          @����e�@��?8Q�A
=Cp��e�?޸R?��
A�C�                                    Bx���`  �          @����hQ�@��?.{A��C�R�hQ�?�G�?��RA�=qC!H                                    Bx���  �          @�=q�a�@��>���@�{C=q�a�@�?��\A�G�C�                                    Bx��	�  �          @�G��\(�@!�=��
?�=qC���\(�@�\?���Ah(�CT{                                    Bx��R  �          @���E@+��:�H�"�RC��E@0  >\@�Q�C
L�                                    Bx��&�  �          @�{�E@)���#�
�\)C^��E@,(�>�ff@ʏ\C
��                                    Bx��5�  �          @��Fff@2�\�aG��AG�C	���Fff@)��?fffAF�\Cn                                    Bx��DD  �          @����@��@4zᾀ  �\��C�H�@��@,(�?aG�AE�C
:�                                    Bx��R�  �          @�p��E@.�R����Q�C
���E@/\)?�@�
=C
c�                                    Bx��a�  �          @�Q��0  @C�
�8Q��33C���0  @Fff?
=q@�RC��                                    Bx��p6  �          @����	��@e��h���D  B���	��@i��?z�@�  B���                                    Bx��~�  �          @��׿��H@xQ쿑��z{B֙����H@���>�@��BՅ                                    Bx����  �          @���{@K���33��G�B𙚿�{@W�>B�\@4z�Bힸ                                    Bx���(  �          @�33�qG�?�{���
���HC 0��qG�?��
>�@�G�C!8R                                    Bx����  �          @��H�  @+�=L��?^�RC���  @(�?���A��C�{                                    Bx���t  �          @�p���
=@p  �u�Tz�B�G���
=@^{?�A�B��                                    Bx���  �          @��
�;�@6ff��(���=qC���;�@2�\?8Q�A!��Ch�                                    Bx����  �          @����@Z�H�#�
��B�  ���@J=q?��A�Q�B�Q�                                    Bx���f  �          @���(��@L��>8Q�@�C�=�(��@7�?�A�  C��                                    Bx���  �          @�z��333@@��>�\)@vffC���333@*=q?�Q�A�p�Cu�                                    Bx���  �          @�z��#33@=p�?�G�A���CǮ�#33@�R@z�B	�RC
��                                    Bx��X  �          @�33�8Q�@-p�?�  Ac�CǮ�8Q�@?�p�A�
=C��                                    Bx���  T          @�{�?\)@2�\?n{AM��C���?\)@��?���A�(�C��                                    Bx��.�  T          @��R�C�
@.�R?fffAF=qC
T{�C�
@	��?�33A�G�C�f                                    Bx��=J  T          @�\)�H��@.{?B�\A&�RC!H�H��@��?�\A�=qC                                      Bx��K�  �          @���S33@'
=>�(�@���C�H�S33@\)?�A���C��                                    Bx��Z�  T          @�ff�U�@!G�>#�
@\)Cٚ�U�@  ?��A}p�C��                                    Bx��i<  �          @�  �^�R@�=L��?0��C�q�^�R@{?�  AZ�RCp�                                    Bx��w�  T          @�\)�_\)@��aG��=p�C��_\)@G�?:�HA\)C�R                                    Bx����  T          @����c33@G���R�=qCW
�c33@>��
@�=qC�f                                    Bx���.  �          @�Q��Z�H@ �׾�����C� �Z�H@p�?(�Az�C8R                                    Bx����  �          @�  �G
=@8Q쾔z��w�C	.�G
=@0��?^�RA<Q�C
aH                                    Bx���z  �          @�Q��N{@1G����
����C5��N{@+�?L��A,(�C8R                                    Bx���   
c          @���S�
@'
=�����p�C�S�
@%�?
=@�ffC�                                    Bx����  �          @�G��N{@2�\����Q�C.�N{@1G�?(�A�\C\)                                    Bx���l  �          @����R�\@&ff�J=q�*�RC�H�R�\@,��>�z�@|(�C�\                                    Bx���  �          @����7
=@I������(�C+��7
=@G�?5Az�Cp�                                    Bx����  �          @���-p�@S33���H��33Cff�-p�@N�R?W
=A3�
C                                      Bx��
^  �          @�=q�0��@QG���R�\)C0��0��@P  ?5A�C\)                                    Bx��  T          @�=q�6ff@L(����H�ҏ\C�\�6ff@HQ�?J=qA)�C\)                                    Bx��'�  T          @����6ff@E�k��F�\C�R�6ff@L��>�Q�@�Q�C��                                    Bx��6P  T          @�G��>�R@;��u�O
=C��>�R@Dz�>��@_\)C(�                                    Bx��D�  �          @��R�>�R@+����\����C
��>�R@=p�����33C(�                                    Bx��S�  �          @�33�E@�R������ffCB��E@'
=���У�C�=                                    Bx��bB  �          @�  �@��?��H�
=q��C\�@��@(�����33C��                                    Bx��p�  �          @�  �J=q?��Ϳ�����ffC(��J=q@(���=q�z{C:�                                    Bx���  �          @����Z�H?�  ��G���C�f�Z�H@ff���ڏ\Cc�                                    Bx���4  T          @���G
=@*�H�
=��\CQ��G
=@,(�?   @�ffC#�                                    Bx����  �          @�z��Tz�@����   C� �Tz�@��>���@���C
                                    Bx����  T          @�p��XQ�@=q�������\Cu��XQ�@?+�A�
CQ�                                    Bx���&  
�          @���E@.{���˅C
���E@+�?#�
AG�C)                                    Bx����  "          @���<��@:�H�����C8R�<��@0  ?�  A_�C�R                                    Bx���r  �          @����@  @5��\����C�)�@  @0��?@  A'�C	aH                                    Bx���  T          @�G��W
=@(��>�z�@uC�{�W
=@z�?��A�z�C^�                                    Bx����  �          @����^{@p�>�
=@�ffC���^{@ff?���A��C��                                    Bx��d  �          @���N{@(�?�
=A�  C޸�N{?��@�A��C�3                                    Bx��
  �          @����S�
@p�?�Q�A��HCc��S�
?�ff@33A�RCn                                    Bx�� �  �          @�
=�QG�@z�?�ffA�p�C�H�QG�?У�@ffA���Cp�                                    Bx��/V  �          @����>{@�ÿ�Q���Q�C=q�>{@*=q�.{�Q�C
+�                                    Bx��=�  �          @��H�Dz�@Q���H��(�C=q�Dz�@Q�>�@߮C8R                                    Bx��L�  �          @�\)�C33@0��?s33AO\)C	�{�C33@�?�Q�A�\)Cu�                                    Bx��[H  �          @�
=�K�@'
=?h��AG
=C���K�@33?���A�=qC#�                                    Bx��i�  �          @�\)�P  @%�?E�A(Q�C��P  @�?��HA�p�CY�                                    Bx��x�  �          @�\)�J�H@*=q?Y��A9G�C\�J�H@�?�A��C=q                                    Bx���:  T          @�
=�J=q@*�H?G�A)C���J=q@
=q?�  A�ffC�f                                    Bx����  �          @�ff�E@0��?(��AQ�C
33�E@�\?�
=A�Q�Cn                                    Bx����  T          @�
=�N{@*�H?�@�z�C@ �N{@��?\A��HC�)                                    Bx���,  �          @�ff�Fff@3�
>\@�
=C	�\�Fff@(�?�Q�A�Q�C�                                     Bx����  �          @��R�R�\@%>��
@�p�C�{�R�\@��?�ffA�(�C�                                    Bx���x  
�          @��R�fff@Q�>��@��
Ch��fff?��
?�  A���C��                                    Bx���  �          @�
=�\��@Q�>�@�G�CW
�\��@G�?���A���C�H                                    Bx����  "          @�{�^{@33>��@��\Cp��^{?�(�?�G�A��RCu�                                    Bx���j  �          @�{�`  @G�>�
=@�
=C��`  ?�
=?�G�A�(�C�                                    Bx��  
Z          @��\�p  ?��
��������C�\�p  ?��
>��R@�z�C�
                                    Bx���  
�          @�Q��n{?�=q����33C G��n{?��H�#�
�#�
C�                                    Bx��(\  �          @�  �r�\?��
�:�H�(��C$�
�r�\?�p���\)����C"�                                    Bx��7  �          @\)�qG�?��Ϳ333�"=qC#��qG�?��
�k��P  C!33                                    Bx��E�  �          @����qG�?��R�E��/�
C!��qG�?�Q�k��S�
C\                                    Bx��TN  �          @��\�s�
?��;�G���ffC }q�s�
?�=�G�?���C�H                                    Bx��b�  "          @���s�
?��
�   ����C!� �s�
?�\)<��
>�{C 8R                                    Bx��q�  K          @�  �qG�?�Q�
=�33C"z��qG�?��ý�Q쿜(�C ��                                    