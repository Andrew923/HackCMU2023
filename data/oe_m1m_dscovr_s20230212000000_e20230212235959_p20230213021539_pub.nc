CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230212000000_e20230212235959_p20230213021539_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-13T02:15:39.779Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-12T00:00:00.000Z   time_coverage_end         2023-02-12T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxd.�   �          @�ff?�Q쿢�\@��\B���C��q?�Q��C�
@>�RB(�C���                                    Bxd.̦  �          @��\?(�ÿ��@��B�Q�C�� ?(���g
=@.{B  C�&f                                    Bxd.�L  T          @��?�{� ��@\)Bp\)C���?�{�j�H@#33B�\C�t{                                    Bxd.��  
(          @��?��ff@}p�Bj��C�c�?��|��@
=A�\)C���                                    Bxd.��  
�          @���}p��Tz�@EB&�C}aH�}p�����?��Ab=qC���                                    Bxd/>  "          @�Q��{�z=q@{A홚Ct�{��{���
>��R@b�\Cx\                                    Bxd/�  r          @�\)�(��|(�@   A�Cp��(���p���G���=qCs�H                                    Bxd/$�  �          @����H�z=q@(�A���Css3���H���=u?=p�Cvp�                                    Bxd/30  �          @��H��H���
?�{A��RCo�)��H��z�����Q�Cq(�                                    Bxd/A�  �          @��
�$z���ff?\A�Cp�$z���z�G��=qCq�                                    Bxd/P|  �          @�z��G����R?���AK33Cs\�G����R�����K�
Cs\                                    Bxd/_"  
Z          @�z���H��\)?�@ʏ\Cq�H��H���R�У�����Cp)                                    Bxd/m�  �          @�ff�����?k�A$Q�Cu�=����ÿ�33�z�\Cu:�                                    Bxd/|n  �          @�=q�z�����?�33AG\)Cs� �z���(������P��Csp�                                    Bxd/�  �          @��
�A���ff�Ǯ��  Cm(��A��|���%�����Chz�                                    Bxd/��  �          @��\�7���
=�����=qCmL��7��w��  �ȣ�Ciff                                    Bxd/�`  T          @��׿�������33�k�
CzΎ��xQ��l(��'��Cz��                                    Bxd/�  T          @�  �����Ϳ�G��~�RC��3���tz��qG��-ffC|�f                                    Bxd/Ŭ  
�          @��ÿ�  ���
�����r=qC|���  �h���a��'��Cw��                                    Bxd/�R  
�          @�  ��=q��=q���
�y�Cy�׿�=q�L(��K��&�HCs��                                    Bxd/��  �          @�G��  ��녿�=q�tQ�Crh��  �J=q�Mp��ffCj�\                                    Bxd/�  �          @�z��
=��\)�O\)��\Czk���
=�q��=p���Cv�                                    Bxd0 D  
�          @���������\��R��Cx������H���
�Z��Ck�)                                    Bxd0�  "          @�
=��p����\�8�����Cz쿽p��{��\)�l�RCl^�                                    Bxd0�  T          @�녿�  �b�\�HQ��Cs�f��  �У����v�\C^��                                    Bxd0,6  "          @�
=��
=�N�R�^�R�2p�Cv쿷
=��Q���33�
C[��                                    Bxd0:�  
Z          @��H��R�Fff�,(��!\)C�Uÿ�R���H�uCv��                                    Bxd0I�  
�          @�{�����'
=�a��UffC��R�����!G����H¡ǮCt}q                                    Bxd0X(  
�          @��R�\)���y���tz�C~��\)<#�
��p�¥�{C2��                                    Bxd0f�  �          @���(��e�2�\���C�/\�(���������)C{��                                    Bxd0ut  �          @�p��#�
�P  �\)�  C��\�#�
��Q��n�RaHC�e                                    Bxd0�  �          @N�R>�
=��(��(Q��oz�C��>�
=�L���@��¤{C�q�                                    Bxd0��  �          @G�?�\)��R��33����C�?�\)������>�C���                                    Bxd0�f  �          @E�?���(���ff��z�C��?���\)���8
=C���                                    Bxd0�  
�          @W
=@0�׿�p�������  C���@0�׾�ff��\�G�C�T{                                    Bxd0��  T          @N{?��������G�C�w
?����׿�{�V33C��3                                    Bxd0�X  
�          @k�@\�ͽ#�
�����{C���@\��?녿�z���z�A
=                                    Bxd0��  �          @j=q@]p�=�Q쿊=q���\?��@]p�?
=�h���h(�A
=                                    Bxd0�  T          @O\)@H��?����=q���
A.{@H��?(��=#�
?G�A>{                                    Bxd0�J  �          @L��@E>����(���
=@7
=@E>�{���R��{@���                                    Bxd1�  T          @5@{�&ff>��HA-p�C���@{�L��>�@3�
C�f                                    Bxd1�  "          ?��R�k��k�?=p�B=qCǮ�k���z�>aG�A&�HC�O\                                    Bxd1%<  T          ?�ff?Q녿�ff?�A��C�� ?Q녿�����\)C�b�                                    Bxd13�  	�          @]p�>���3�
?�B	�\C���>���XQ�>\@��C�AH                                    Bxd1B�  �          @!G�?����?���A��C���?��(���Q��ffC��                                    Bxd1Q.  
�          @�?^�R��(�?z�A�\)C�^�?^�R��������(�C��                                    Bxd1_�  �          @Q�?\(���(�?G�A�G�C�K�?\(���녽�G��J=qC�C�                                    Bxd1nz  
(          @�
?n{�޸R?��A�(�C��?n{�z�>B�\@���C��                                    Bxd1}   "          @333?У׿���?��B��C��?У��
=?�RARffC���                                    Bxd1��  T          @1�?h����H?W
=A�p�C�Q�?h���!녾Ǯ��C��                                    Bxd1�l  �          @^�R?Y���7�?�
=A��HC�4{?Y���Tz�>\)@�HC�!H                                    Bxd1�  
�          @mp�>��[�?�=qA��
C��
>��i����(����C���                                    Bxd1��  	�          @p��<��dz�?���A���C�<)<��j�H�333�,z�C�:�                                    Bxd1�^  h          @h��?���9��?��
A�(�C�  ?���Y��>aG�@dz�C��H                                    Bxd1�  �          @o\)?��H�>{?���A�G�C�q?��H�_\)>�  @s�
C��R                                    Bxd1�  
�          @j�H?c�
�U?��A�C�w
?c�
�_\)�����C�+�                                    Bxd1�P  
�          @mp�@%� ��?L��AHz�C���@%�&ff����{C�t{                                    Bxd2 �  �          @k�?�=q�G
==�\)?��C�<)?�=q�2�\�������C��f                                    Bxd2�  T          @z=q��\)�k�?.{A\)C}  ��\)�dz῎{���RC|��                                    Bxd2B  �          @��
��Q�����>�(�@��RC}�3��Q��w���ff��\)C|޸                                    Bxd2,�  �          @��
�
=q��ff?(�A33C�Z�
=q��  ��z���ffC�,�                                    Bxd2;�  �          @�{��\�xQ�?���A��C�<)��\��녿(��Q�C�ff                                    Bxd2J4  "          @�Q쾅����?L��A#�C�S3������ÿ����G�C�E                                    Bxd2X�  "          @�G��
=��p�?@  AG�C�0��
=��Q쿱����RC�                                    Bxd2g�  �          @��R��������>�@�z�C�.������  ��
=���Cs3                                    Bxd2v&  T          @�p��8Q���G�?0��A��C��׿8Q����H���R���C�P�                                    Bxd2��  "          @�  �8Q���Q�?�ffAYp�C�/\�8Q���Q쿆ff�Z=qC�.                                    Bxd2�r  "          @��Ϳ�\��G�?�p�A~ffC��H��\���
�fff�8(�C���                                    Bxd2�  
�          @���&ff�Z�H@Q�B	p�C����&ff��z�?�@�p�C��\                                    Bxd2��  �          @�  ��G��Q�@�A�33CuO\��G��z=q>�
=@�Cx�                                    Bxd2�d  
�          @�Q�8Q��X��@Q�B	��C���8Q���33?
=q@�C��                                    Bxd2�
  T          @�
=����z=q?8Q�A�RCm������s33��z��f�HCm.                                    Bxd2ܰ  	�          @�������u?���A�{CsǮ������ÿz���Ct�)                                    Bxd2�V  
�          @��\�^�R�K�@{B{C~��^�R�}p�?8Q�A ��C�Ф                                    Bxd2��  �          @�ff    �Fff@'
=B \)C���    �}p�?c�
AJ=qC���                                    Bxd3�  
�          @��?�R�@��@'�B!�\C��\?�R�xQ�?n{AV{C��=                                    Bxd3H  
�          @�G�>�p��,��@5�B8Q�C��>�p��l��?��\A���C��                                     Bxd3%�  r          @��R�aG��6ff@@��B9��C�Ф�aG��z=q?�{A�33C�h�                                    Bxd34�  
�          @�Q쿠  �Z�H@
�HA��\Cyٚ��  ����>�Q�@��HC|��                                    Bxd3C:  �          @�z�˅�g�?�AǙ�CvY��˅��논��ǮCx��                                    Bxd3Q�  
Z          @�Q��
=�aG�?�G�A��Cl&f�
=�tzᾊ=q�^{CnJ=                                    Bxd3`�  
�          @��
�Y���;�?#�
A�RC\� �Y���9���E����C\s3                                    Bxd3o,  
�          @�(��l���%>�ff@�
=CW
=�l��� �׿J=q��CV0�                                    Bxd3}�  �          @�Q�@o\)�z�?�\)Aң�C��q@o\)��\)?�33A���C��                                    Bxd3�x  "          @��@���=L��?���A�G�?B�\@��Ϳz�?�  A�33C�                                      Bxd3�  
(          @�=q@dzῘQ�@�A�C��f@dz�� ��?��A�p�C�L�                                    Bxd3��  �          @�(�@K�����?�\)Aי�C��@K��   ?Q�A7\)C��\                                    Bxd3�j  "          @�z�@1G��'
=?�G�A���C�T{@1G��@��>��@C�J=                                    Bxd3�  �          @�\)@=p��"�\?���A�p�C��\@=p��>�R>u@P  C�k�                                    Bxd3ն  
�          @���@C�
���?�A�{C�  @C�
�=p�?�@�Q�C�H                                    Bxd3�\  T          @��R@N�R�p�@
=A�=qC���@N�R�;�?^�RA4  C��                                    Bxd3�  
�          @�p�@E���H@(��B=qC���@E�.�R?�=qA�\)C�L�                                    Bxd4�  "          @��R@L�Ϳ��R@.{B=qC���@L���$z�?޸RA��C���                                    Bxd4N  T          @��H@QG��Ǯ@333B�HC�=q@QG��*�H?��A��C�j=                                    Bxd4�  �          @��R@2�\��p�@)��B��C��@2�\�0  ?�=qA��C���                                    Bxd4-�  "          @���@녿�{@:=qB:ffC���@��>�R?�  A�
=C�)                                    Bxd4<@  �          @���@!����@
�HB�C�q@!��@  ?h��AQG�C�3                                    Bxd4J�  T          @z=q?�ff�,��@��B�HC��{?�ff�XQ�?0��A#33C��                                    Bxd4Y�  
�          @���@���\)@0  B'p�C�,�@��:=q?���A�G�C��                                    Bxd4h2  �          @��@Dz��
=@*=qB"=qC�
@Dz��=q@�B �HC�j=                                    Bxd4v�  "          @�Q�@<�ͿQ�@%B ��C�5�@<�Ϳ�z�?�Q�A��C��
                                    Bxd4�~  "          @�Q�@8Q��(�@*=qB)z�C���@8Q�˅@�B��C���                                    Bxd4�$  �          @y��@8Q쿂�\@�HBG�C�B�@8Q��   ?ٙ�A�z�C��                                     Bxd4��  
�          @z=q@&ff��
=@*�HB,�C���@&ff���?���A�=qC�y�                                    Bxd4�p  
�          @}p�@�\�O\)@E�BO  C�H�@�\�
=@Q�B�C��\                                    Bxd4�  T          @mp�@��G�@;�BQ��C��@�� ��@��B(�C�7
                                    Bxd4μ  "          @���?�(��p�@P  B>C��
?�(��[�?�z�A�(�C��                                    Bxd4�b  �          @��\?�(��(�@aG�BHQ�C���?�(��a�@
=qA���C��                                    Bxd4�  T          @�{@
=q��H@n�RBD(�C��@
=q�u�@  A��HC��{                                    Bxd4��  �          @�=q@33�'�@tz�BD{C�
=@33����@  Aҏ\C�h�                                    Bxd5	T  �          @�ff?����`  @U�B G�C��?������
?�z�A}�C��=                                    Bxd5�  
�          @�  >�33��Q�?�=qA�(�C�@ >�33��33�\��\)C�R                                    Bxd5&�  "          @���B�\���H?O\)A��C��B�\��z��z����C���                                    Bxd55F  �          @�녿�����׽�G���\)C�'������z��#�
��  C��=                                    Bxd5C�  �          @���W
=���;�����C��f�W
=��Q��#33���HC��H                                    Bxd5R�  T          @�
=<#�
��ff>�?��HC�\<#�
��
=�����{C��                                    Bxd5a8  �          @�ff?5���
>�ff@�ffC��?5��G�������
C�33                                    Bxd5o�  T          @�  �u���H>8Q�@
=qC���u��p����R��C�j=                                    Bxd5~�  
�          @�z�k���\)>\)?ٙ�C���k�������(���\)C��q                                    Bxd5�*  "          @��\?L�����R>�=q@EC��)?L����녿��H��G�C�                                      Bxd5��  �          @�?\(����������C��?\(�����'���C�H                                    Bxd5�v  "          @��?}p���\)��G���  C���?}p������&ff���C���                                    Bxd5�  "          @��?�\)���
��=q��  C�'�?�\)�Fff�e��6�\C��H                                    Bxd5��  T          @�33@5��.{�L���ffC��@5����
��=q�U�HC�H                                    Bxd5�h  �          @�ff@Fff�@  �9����\C���@Fff��z��z�H�DG�C��q                                    Bxd5�  
Z          @�z�@U��(��C�
�(�C�ٚ@U��W
=�tz��@\)C���                                    Bxd5�  "          @�ff@%�(���J�H�"�\C�B�@%�xQ���Q��]��C��                                    Bxd6Z  �          @�(�@1G��{�P  �$�HC�)@1G��J=q��Q��YG�C�                                    Bxd6   �          @�
=@7
=�{�A���C��@7
=�(���l(��N
=C�}q                                    Bxd6�  "          @��R@�
�G
=�&ff��\C�O\@�
��33�mp��R�C�>�                                    Bxd6.L  �          @��\@G��c33��
��z�C�J=@G��
�H�h���D�
C�                                      Bxd6<�  �          @�z�@������[��:(�C���@�ÿ0�������v\)C���                                    Bxd6K�  
<          @���?����=q��=q��C��q?��?333���R�AÅ                                    Bxd6Z>  
�          @���?��\�������ǮC��?��\?�R��33�fAљ�                                    Bxd6h�  T          @��\?�Q��Q���\)�g\)C�>�?�Q쾔z���33�3C�R                                    Bxd6w�  "          @��
?
=�B�\����R��C��?
=�u��\)� C��)                                    Bxd6�0  
�          @�=q?@  �E��|���L�HC�ٚ?@  ��ff����p�C���                                    Bxd6��  T          @���?k��7���Q��T=qC��R?k��W
=���W
C��f                                    Bxd6�|  "          @��?���2�\�~�R�SG�C�
?�녿G������
C��)                                    Bxd6�"  �          @�G�?L���C33�z�H�L��C�XR?L�Ϳ�����C�                                    Bxd6��  �          @���?����C33�s33�D��C���?���������  G�C���                                    Bxd6�n  
�          @�G�@E����P  �2Q�C��q@E>\�Z�H�>@�Q�                                    Bxd6�  �          @���@�\)�L���/\)���C���@�\)?���\)��33Aqp�                                    Bxd6�  �          @��R@�
=�����A��p�C���@�
=?���7
=�ffA]�                                    Bxd6�`  T          @��@vff��Q��J�H�
=C�N@vff?���9���G�A�{                                    Bxd7
  
�          @��
@�(�?����(���
=AT��@�(�?��ÿaG��+�A��                                    Bxd7�  T          @�
=@��H?�(��s33�>�HAz�R@��H?�G��������A�p�                                    Bxd7'R  �          @�
=@��\?h�ÿ�(���33A>=q@��\?��H�n{�9�A���                                    Bxd75�  
(          @�@w�?=p�������HA-@w�?�(������z�A�(�                                    Bxd7D�  �          @�(�@R�\?O\)�;��#{A]�@R�\?��R��\��(�A�G�                                    Bxd7SD  
�          @�(�@~�R>�{�G���z�@�p�@~�R?��
��33��\)A��H                                    Bxd7a�  "          @�G�@�Q�(�ÿ���c�C��)@�Q�����\����C�&f                                    Bxd7p�  �          @�ff@Fff�u�/\)�&  C�g�@Fff?�\)�   �
=A�z�                                    Bxd76  
�          @�z�@\��=�\)�+��z�?�\)@\��?��H������A�z�                                    Bxd7��  "          @�  @L�;8Q��)����C�Y�@L��?xQ��{�33A��\                                    Bxd7��  �          @�{?�33>�p��hQ��w�A1�?�33?���G��Ep�B3
=                                    Bxd7�(  T          @�{@�\>W
=�g��r
=@�\)@�\?޸R�L(��G��B!��                                    Bxd7��  
Z          @���@33�u�qG��u�HC��@33?�ff�\(��U  B=q                                    Bxd7�t  T          @��R@�R�����b�\�f
=C���@�R?��H�U�S33A�                                    Bxd7�  "          @�z�?�{�G��mp���C��?�{?^�R�l(��~�\A�                                    Bxd7��  
�          @��?�p���z��|(��RC�Y�?�p�?!G�����=A�=q                                    Bxd7�f  
�          @�z�@��Ǯ�P  �Sp�C�y�@�?��
�G
=�F�A��                                    Bxd8  T          @��
@`��?!G��33���A"�R@`��?�Q�˅��  A�(�                                    Bxd8�  �          @�ff@dz�?�{��33��(�A��\@dz�?���G����Aׅ                                    Bxd8 X  �          @�{@�H���
�333�,=qC�ٚ@�H��(��R�\�U�C��q                                    Bxd8.�  �          @�  @�����W
=�QC��q@�<��
�l���r�?(�                                    Bxd8=�  �          @�(�?���
=�E��B(�C�n?���ff�g
=�uQ�C�]q                                    Bxd8LJ  T          @�(�?���p��J�H�J(�C�!H?�녿+��s33ǮC�*=                                    Bxd8Z�  
�          @�ff?�{�{�R�\�NffC��H?�{�#�
�z�HC�|)                                    Bxd8i�  T          @�=q?���ff�?\)�>\)C�N?�녿aG��mp�
=C��{                                    Bxd8x<  �          @��H?��\�p��L(��Mp�C�� ?��\�+��tz��HC��
                                    Bxd8��  
�          @�?=p��;��3�
�+��C��?=p����R�q��\C�'�                                    Bxd8��            @�(�?�{��R�O\)�Q�C�:�?�{�+��w�C�k�                                    Bxd8�.  
(          @�G�?�
=��Q��E�I(�C�y�?�
=����g�p�C�'�                                    Bxd8��            @��H@   ���
�@���A33C�<)@   ��Q��^{�n\)C��                                     Bxd8�z  T          @���?�Q�����5�6  C�� ?�Q�}p��e���HC���                                    Bxd8�   �          @�G�@Q���6ff�5�C���@Q���H�W
=�c�RC��                                    Bxd8��  �          @\)?�33��(��8Q��9z�C��)?�33����\(��o\)C�9�                                    Bxd8�l  �          @�=q?�  �1G��333�.G�C��f?�  ��{�l(��C��                                    Bxd8�  "          @�  ?ٙ������J=q�U��C�aH?ٙ�����b�\k�C�t{                                    Bxd9
�  �          @��H?�
=��\)�\���gffC�Y�?�
=>.{�mp�{@�{                                    Bxd9^  "          @�=q?�\)��{�^{�jC���?�\)>B�\�n{@��
                                    Bxd9(  �          @�G�?˅���
�c�
�w�HC��3?˅?��j�H��A�\)                                    Bxd96�  
�          @��?�p��+��g
=�{ffC�y�?�p�?aG��dz��u�
A�                                    Bxd9EP  �          @s33?��\��Q��Dz��g�
C���?��\�L���X���fC��\                                    Bxd9S�  T          @s33=��
�&ff�#�
�2
=C���=��
�����Y��k�C���                                    Bxd9b�  T          @}p�?J=q���X���m��C��3?J=q��z��u8RC��                                    Bxd9qB  T          @���=�\)��H�L���SQ�C�� =�\)�fff�z=q�fC�W
                                    Bxd9�  T          @s33����#�
�#�
�1�C����Ϳ���XQ���Cv�                                     Bxd9��  	�          @xQ�>�  �33�E�T��C�&f>�  �W
=�p  �RC�h�                                    Bxd9�4  T          @u��u�p��:=q�GffC�W
�u����j=q�qC�w
                                    Bxd9��  �          @G��W
=�����BCv��W
=�E��9��W
C^p�                                    Bxd9��  �          @Y����R�����,(��S
=C|uÿ�R�0���P  =qCd33                                    Bxd9�&  �          @^{>�p��G��1G��U�HC�!H>�p��8Q��Vff�
C�j=                                    Bxd9��  T          @a�>�{��H�   �6�
C���>�{�����QG�Q�C�Ф                                    Bxd9�r  "          @vff>�Q��'
=�1��9�C��f>�Q쿢�\�e��C���                                    Bxd9�  �          @xQ�?#�
��  �W��t
=C���?#�
�u�r�\�\C��                                     Bxd:�  
G          @s33>8Q��   �5��A�C�q>8Q쿑��e=qC���                                    Bxd:d            @tz�?5�ff�8���F�\C�j=?5�}p��e�=C��)                                    Bxd:!
  
�          @l�;�z�� ���*�H�:{C��쾔z῝p��\����C�T{                                    Bxd:/�  
Z          @j�H?�G��xQ��S33�C�\?�G�>�G��Z=qQ�A�                                    Bxd:>V  �          @qG�?�{�޸R�H���b�
C�P�?�{�����dz�  C���                                    Bxd:L�  T          @o\)?}p���
=�Vff�qC�f?}p�>���b�\ǮAn�\                                    Bxd:[�  T          @j�H?�������Mp��t�RC�p�?���=����]p��{@�=q                                    Bxd:jH  
�          @e�?h�ÿL���Vff33C�q�?h��?(��X��.B                                      Bxd:x�  �          @G
=?�33�\)�z�H����C�.?�33������G�C�Y�                                    Bxd:��  
�          @H��?�  �{���
��ffC��f?�  ��{���ffC�xR                                    Bxd:�:  
�          @P��?�G���Ϳ�����p�C�` ?�G������
�7z�C���                                    Bxd:��  �          @AG�?�  �녿Y����=qC���?�  ��  ��Q����C�xR                                    Bxd:��  �          @C33?�G��z�Q��z�RC��q?�G���ff����HC�&f                                    Bxd:�,  "          @@��?��R���:�H�b=qC��)?��R��z��G����C�q                                    Bxd:��  �          @C�
@33���(��8��C���@33��\����  C���                                    Bxd:�x  
�          @O\)@��33�:�H�R�RC�U�@���=q��=q��p�C��H                                    Bxd:�  
(          @HQ�?��z�L���n{C���?����ÿ�33���C�Ff                                    Bxd:��  "          @Q�@z��
=�xQ���(�C�޸@z��ff��(���p�C�3                                    Bxd;j  "          @P  @$z�����
����C���@$z��녿�ff��=qC��
                                    Bxd;  T          @P��@ ����\�8Q��QG�C�y�@ �׿�ff�xQ���
=C�*=                                    Bxd;(�  "          @S�
@=q�p����
��p�C��R@=q��z῔z����
C���                                    Bxd;7\  T          @QG�@�{��=q���\C��@�
�H��(���C���                                    Bxd;F  �          @N{@
=q�Q콣�
��Q�C�q@
=q�
=q��  ���C��                                     Bxd;T�  
�          @O\)@��   �:�H�Q��C�AH@��Ǯ�������
C��f                                    Bxd;cN  T          @L(�@333��p�����C��@333���Ϳ���.�HC�"�                                    Bxd;q�  
�          @J=q@7
=���>�=q@�(�C��3@7
=��ff��  ��z�C��                                    Bxd;��  T          @I��@'���(�<�?#�
C�b�@'����Ϳ!G��:=qC�S3                                    Bxd;�@            @J�H@*�H��=L��?fffC�@*�H�Ǯ����/33C��                                    Bxd;��  5          @@  @333�^�R>�{@�=qC�` @333�n{���
��ffC�˅                                    Bxd;��  
�          @H��@#�
�У�?333AO�C�@#�
��\<��
>���C��f                                    Bxd;�2  "          @G
=@1G����>�z�@�p�C�@1G����;�  ��=qC���                                    Bxd;��  T          @G
=@'����H?+�AHQ�C�o\@'���{=u?��
C�B�                                    Bxd;�~  
�          @I��@$zῺ�H?z�HA�z�C�.@$z��p�>�{@��HC�\                                    Bxd;�$  
�          @G
=@7��L��?Tz�Ax��C�<)@7����>�A	G�C��                                    Bxd;��  
�          @J=q@6ff�5?
=qA%�C��q@6ff�\(�>aG�@���C���                                    Bxd<p  
Z          @<(�?��H��
��Q���\C�7
?��H��\)�\(���p�C�p�                                    Bxd<  �          @6ff���*�H�+��_\)C�q���{�����RC�!H                                    Bxd<!�  �          @9�����5�������G�C��{���   ��������C��=                                    Bxd<0b  "          @3�
>�=q�0�׾�Q���(�C��
>�=q��H������=qC�<)                                    Bxd<?  �          @C33?.{�:=q�
=�3�C���?.{�{��33�(�C���                                    Bxd<M�  �          @AG�?n{�1G��8Q��\  C�>�?n{�33���H�
p�C��q                                    Bxd<\T  T          @E�?p���3�
�L���pQ�C�E?p����
��ff�Q�C�                                      Bxd<j�  �          @<(�?fff�%��ff��\)C��=?fff�   ��Q��&�C��                                    Bxd<y�  
�          @Mp�?�z��*=q��{����C��3?�z��33�G���C�@                                     Bxd<�F  �          @Dz�?���$z῜(����RC�}q?����
=��-�C��H                                    Bxd<��  �          @>{?xQ��p��������C��q?xQ���
����:  C�7
                                    Bxd<��  T          @J�H?p���*=q����\)C��q?p�׿�Q���
�;�C��3                                    Bxd<�8  T          @J=q?J=q�,(����H�܏\C�0�?J=q�����
=�A33C��                                    Bxd<��  �          @G
=>�ff�8�ÿ������C�h�>�ff��\��\�$z�C���                                    Bxd<ф  T          @*=q>�ff��Ϳh�����RC���>�ff�ٙ���z��-ffC�l�                                    Bxd<�*  
�          @7
=@녿z�H������{C�g�@녾�녿�{�p�C��\                                    Bxd<��  
�          @*=q?�z��\)��{��ffC�~�?�zῪ=q��(��&�C�Z�                                    Bxd<�v  
�          @/\)@{=���?�33A�33@*=q@{����?�{A�Q�C��f                                    Bxd=  �          @��?�Q�?
=q?���B{A�  ?�Q��?�ffB)��C�s3                                    Bxd=�  T          @%�?�(��=p�������\C�XR?�(��8Q�Ǯ�'�C�H                                    Bxd=)h  
�          @�?�G���33>��HAg33C�\)?�G���  <�?p��C�4{                                    Bxd=8  T          @33?�Q쿕�L����p�C�ff?�Q쿁G���R�}��C�(�                                    Bxd=F�  
�          @n{@���@��=�G�?�\C��
@���5��  �z{C�~�                                    Bxd=UZ  "          @n{?����J=q���Ϳ�  C�ٚ?����9�����R���RC���                                    Bxd=d   "          @n�R@���9��������C��)@����R��\)���
C�0�                                    Bxd=r�  �          @xQ�@�\�=p��}p��mp�C�� @�\����   ����C��q                                    Bxd=�L  T          @y��@\)�*�H������=qC�� @\)���R�{���C��H                                    Bxd=��  T          @e@  �������ffC�h�@  ��������C��
                                    Bxd=��  
(          @P  ?�{�녿�����C���?�{����z��=qC�K�                                    Bxd=�>  �          @H��?�
=���Ϳ������C��?�
=��33�
=q�/��C��q                                    Bxd=��  "          @Fff@ff�޸R��\)��G�C�1�@ff���׿�33�p�C��                                    Bxd=ʊ  �          @_\)@p���H>W
=@Y��C���@p����.{�5��C�=q                                    Bxd=�0  �          @s�
@,���*�H=�Q�?��
C��{@,���!G��aG��U�C��                                    Bxd=��  T          @|��@0���.�R�
=q���C��@0���
=��p����C���                                    Bxd=�|  T          @���@X���p��W
=�5G�C�H@X���   ��z���p�C��q                                    Bxd>"  T          @��@HQ��"�\���\���C�s3@HQ���ff���C�9�                                    Bxd>�  �          @��@:�H�'������z�C��@:�H���'
=��HC�q                                    Bxd>"n  
�          @u?����z��(���1Q�C�k�?��׿�p��R�\�r�\C�5�                                    Bxd>1  �          @tz�?�  �=q�z���RC�  ?�  ��
=�AG��T��C�e                                    Bxd>?�  "          @w
=?���33�=q�%��C��R?������>�R�\
=C���                                    Bxd>N`  T          @j=q?h�ÿ�
=�HQ��j��C�5�?h�þ��aG�C��f                                    Bxd>]  �          @n{?�ff��
=�<(��P��C�Q�?�ff���U��C�xR                                    Bxd>k�  T          @s�
?��p���K��c�C�w
?�>u�S�
�r��@�                                    Bxd>zR  �          @w�?�Q쿦ff�N�R�b{C�33?�Q콣�
�^�R�C���                                    Bxd>��  T          @�(�@��ff�,���$Q�C�� @녿��\�QG��R��C��                                    Bxd>��  
�          @���@	����z��1G��/p�C�1�@	���Q��P���[  C���                                    Bxd>�D  T          @q�?�G��333�\���C�t{?�G�?���^�R�fA��\                                    Bxd>��  �          @e?��\�^�R�Mp���C�Ǯ?��\>��R�Tz��AZ�\                                    Bxd>Ð  "          @c33?�(���{�QG��C�S3?�(�?Q��K�W
B{                                    Bxd>�6  �          @^�R?h�þ�Q��S33�fC�
?h��?L���N{�B%��                                    Bxd>��  "          @^�R?�p��u�L(��C�ff?�p�?aG��E��ffB
=                                    Bxd>�  �          @[�?s33�����P���qC�ff?s33?Q��J�H�B"��                                    Bxd>�(  
�          @Z�H>k���ff�Vff¢�)C��
>k�?=p��S33��B���                                    Bxd?�  �          @XQ��=�Q��Vff¯C���?��R�G��=B�                                    Bxd?t  �          @S�
���?h���EB�B��\���?�z��%�P��B֔{                                    Bxd?*  "          @L��>\>�=q�3�
¡\)B>\?��H�#33�}��B��                                     Bxd?8�  �          @N�R����=q�J�H©p�CuT{��?Tz��Dz�B�BĮ                                    Bxd?Gf  �          @8�ý�G��aG��*�HW
C�xR��G�=�Q��3�
®Q�CG�                                    Bxd?V  "          @'�>.{���H���\�C�e>.{�(���\)�C�s3                                    Bxd?d�  �          @?\)?��=q��p���p�C�q?���\�p��H�HC�@                                     Bxd?sX  T          ?�p�>W
=��
=�u����C��>W
=�޸R�aG���{C�^�                                    Bxd?��  �          ?���>�33��G��E��ӮC�s3>�33��z῞�R�6�C�N                                    Bxd?��  �          @z�?0�׿����p��D\)C��=?0�׿+���p�C��=                                    Bxd?�J  �          @{?fff��(���ff��\)C���?fff���\�����533C��                                     Bxd?��  T          @  ?�{��
=��  ��z�C�Ф?�{���R��  �'�
C���                                    Bxd?��  T          @ff?Tz���Ϳ���C��f?Tzΐ33��G��:��C��                                    Bxd?�<  
�          ?���?���z�H�.{��
=C���?���0�׿xQ��;G�C�b�                                    Bxd?��  �          @z�?^�R���H���
���C��=?^�R��
=���
�I��C�:�                                    Bxd?�  �          @%?��
����\��C���?��
��
=���Q{C���                                    Bxd?�.  
�          @%?�G���\�����ffC���?�G�����ff�X33C���                                    Bxd@�  
�          @'�?����ff������C���?����Q��33�O�C�=q                                    Bxd@z  �          @�?�G���33��ff�33C��?�G����׿�\�E��C���                                    Bxd@#   
�          ?�33?=p����ÿ�\)��C��3?=p��^�R���R�Q��C�+�                                    Bxd@1�  f          ?��H?����33�Tz���
=C���?���O\)��z��G�C�]q                                    Bxd@@l  
�          @��?\���H�J=q��ffC�
?\��\)��p��{C��R                                    Bxd@O  
�          @*�H?�z��{�s33��=qC���?�zΌ����G��	p�C�t{                                    Bxd@]�  
�          @0  ?�(����Ϳ����p�C�` ?�(���z��{���C�S3                                    Bxd@l^  �          @5�?�녿�(���
=��C�޸?�녿�p���G��\)C��q                                    Bxd@{  
�          @:�H?�33�zῚ�H��
=C�H�?�33�Ǯ�����
=C�O\                                    Bxd@��  �          @5�?���������\)C�� ?�녿�����+
=C�
=                                    Bxd@�P  �          @/\)?ٙ�����������C�U�?ٙ����׿����C�l�                                    Bxd@��  �          @#�
?\������\���C�'�?\��{��ff�ffC��                                    Bxd@��  T          @%?�33��33���\��p�C�"�?�33��(��˅�  C��                                    Bxd@�B  
�          @(Q�?�\)��Q쿋�����C���?�\)��p���z��\)C�aH                                    Bxd@��  T          @'�?�G���׿�  ���
C�ff?�G����H��ff�C��                                    Bxd@�  T          @'
=?��Ϳ�{�Q���(�C�b�?��Ϳ��R����� p�C�}q                                    Bxd@�4  
�          @5�?���\�}p���(�C���?���\)�˅�	p�C���                                    Bxd@��  �          @/\)?�p�� �׿�����=qC�(�?�p�������H���C�޸                                    BxdA�  	�          @'�?�33��\��ff��\)C�u�?�33���ͿǮ�  C�e                                    BxdA&  �          @J=q?�p��
�H��z�����C�(�?�p���Q����
�C���                                    BxdA*�  T          @G
=@ff��׿:�H�Z{C��
@ff��ff����ĸRC�W
                                    BxdA9r  
�          @8Q�?�G��
=q��\)��(�C�n?�G���Q��  �p�C��\                                    BxdAH  T          @B�\?˅�Q쿂�\��p�C��)?˅��Q��p��
=qC��3                                    BxdAV�  �          @Fff?��R�!G���G���C�C�?��R�z��  �	��C�Ф                                    BxdAed  
�          @A�?�p��\)�c�
��ffC�b�?�p���У����C��R                                    BxdAt
  "          @<��?��\�\)�n{��C�z�?��\�z���

=C���                                    BxdA��  
�          @?\)?�{� �׿aG���33C�5�?�{�
=��\)��C�g�                                    BxdA�V  
�          @?\)?�p����h����=qC��{?�p��G��У���C�"�                                    BxdA��  
�          @@��?����!녿h����G�C��?�������z����C�G�                                    BxdA��  �          @@  ?�G��!G����
���C�K�?�G��z��\�=qC��=                                    BxdA�H  T          @Fff?�ff�'
=��ff�ƣ�C�  ?�ff���33�%\)C�u�                                    BxdA��  �          @J�H?�G��!G���\)���C��\?�G�����z��==qC�\                                    BxdAڔ  T          @J�H?(�����������C��3?(�ÿ�33�%��]��C���                                    BxdA�:  
�          @P  ?L�������H��HC�{?L�Ϳ��H�'��X�C���                                    BxdA��  
�          @W
=?�  �:�H������C�y�?�  �
=�
�H� �
C�z�                                    BxdB�  	�          @C33?=p������p���(�C��f?=p���=q�	���=��C��{                                    BxdB,  
�          @Fff?����
��ff���C�*=?�������H�L�C�0�                                    BxdB#�  	.          @C�
?˅�  �G��{�
C���?˅��zῷ
=��33C��\                                    BxdB2x  T          @>�R?��H�{��{��{C��?��H��\�޸R�p�C��q                                    BxdBA  "          @B�\?�\)�33���\�ʏ\C�c�?�\)����z�� G�C��R                                    BxdBO�  T          @Mp�?��'�����G�C�33?���ÿ�z��z�C���                                    BxdB^j  
�          @P��?���$zῐ����(�C�B�?���
=�����
�C���                                    BxdBm  T          @U�?����R�������C��?���G����p�C��=                                    BxdB{�  
�          @Q�?��H�녿����
=C�W
?��H������(�C���                                    BxdB�\  
�          @QG�?��
�
=��\)��z�C��?��
��� �����C�3                                    BxdB�  �          @W
=?�z��%�������Q�C�o\?�z��33��
��C��f                                    BxdB��  �          @\��?�z��/\)������C��
?�z��{�z���C�]q                                    BxdB�N  "          @b�\?޸R�333���
����C��{?޸R��\��
�33C���                                    BxdB��  
�          @aG�?�z��1G���33����C�y�?�z���R�
=q�G�C�Y�                                    BxdBӚ  T          @\��?�ff�+������\)C��?�ff��� ���>�
C��H                                    BxdB�@  �          @[�?(���)���z���C��3?(�ÿ��0���VC�p�                                    BxdB��  �          @\(�?#�
�,(��33�z�C��f?#�
���H�0���T��C���                                    BxdB��  
�          @g
=?���6ff��
=��RC�:�?���	���,(��@�HC�#�                                    BxdC2  
�          @c�
?����=q���'��C�B�?��Ϳ����>�R�c�\C�7
                                    BxdC�  �          @`��?k��!G����p�C�3?k���\�4z��[{C���                                    BxdC+~  t          @c�
?+��<�Ϳ�{��\)C�g�?+��G��)���@��C�7
                                    BxdC:$  
�          @e�?.{�3�
�ff���C�� ?.{���5��Qp�C��                                    BxdCH�  �          @c�
>���2�\�	����C��>����\�7��Wp�C�w
                                    BxdCWp  �          @h�þ�33�J�H�޸R��33C��
��33�!��%�5�RC��                                    BxdCf  
Z          @\��?Ǯ��  �>{�w�RC�aH?Ǯ?�R�:�H�p(�A�z�                                    BxdCt�  
�          @U?u��=q�J�H(�C�)?u?(���G�{B

=                                    BxdC�b  "          @Tz�?E���G��L����C�C�?E�?   �L(��3B�                                    BxdC�  
�          @Y��>��Ϳ^�R�N�R��C�aH>���=�Q��U¥�fAP                                      BxdC��  �          @aG������\)����033CtaH���ÿ��H�<���iCi��                                    BxdC�T  
�          @`  �����=q������RClk����Ϳ�  �#�
�<z�Cc�
                                    BxdC��  "          @\�Ϳ
=�2�\��\)�G�C�3�
=����%�E�HC~��                                    BxdC̠  �          @Z�H=L���.{���
=C���=L��� ���1G��X=qC�                                    BxdC�F  "          @^{>aG��,������(�C�K�>aG���p��4z��Z��C�                                      BxdC��  
�          @c�
?z��!G��=q�+�
C���?z�޸R�A��kffC�L�                                    BxdC��  �          @`  >���
�$z��>�C���>���p��G��}��C��                                     BxdD8  �          @g
=>�{�'
=�(��+G�C���>�{�����E��l{C�T{                                    BxdD�  "          @hQ�>�(��2�\�  �ffC�b�>�(���\�<���Z��C��3                                    BxdD$�  T          @q녿#�
�(��.{�<z�CLͿ#�
��=q�S33�z��Cw�                                    BxdD3*  
�          @e����33�'��A
=C�N����(��J=q�C|�                                    BxdDA�  
(          @qG���G�����B�\�X�
Cj�׿�G��^�R�Z�H�RCV��                                    BxdDPv  	�          @n{�c�
���L(��l��Cq޸�c�
�8Q��a���CZ�                                     BxdD_  �          @w����  �G
=�V\)C�:����ff�g�aHCzxR                                    BxdDm�  
�          @l�Ϳ�
=��Q��7��Qp�C\���
=�
=�J=q�rffCG^�                                    BxdD|h  
�          @b�\��(���(��>{�eQ�CfW
��(��
=�P��CMǮ                                    BxdD�  
�          @b�\��ff���(Q��D{Cr�{��ff��p��E�y��CeO\                                    BxdD��  
�          @~{��(��ff�=p��@�RCruÿ�(���Q��_\)�v33Ce��                                    BxdD�Z  "          @|(���33�Q��3�
�6
=CoxR��33��G��Vff�i��Cc0�                                    BxdD�   
Z          @y������=q�%��&\)Ck�ÿ�녿�{�H���Wz�C`aH                                    BxdDŦ  �          @{���Q����(���(CdJ=��Q쿯\)�H���Sp�CWB�                                    BxdD�L  �          @�Q���H��\�)���%�Ce�����H��p��K��QQ�CY!H                                    BxdD��  
�          @z=q���
���@  �I�Cnٚ���
��(��]p��{�RC_��                                    BxdD�  
�          @z�H�����R�C33�L��Cj��������^�R�z�CY��                                    BxdE >  "          @|���	�����H�333�6�CZu��	���aG��J=q�VCJ\)                                    BxdE�  �          @|(���
=�˅�B�\�JG�C[�{��
=�5�W
=�j�CHO\                                    BxdE�  
�          @}p��޸R����J=q�S�C_:�޸R�:�H�^�R�v33CJ��                                    BxdE,0  T          @}p����Ϳ�p��N{�]�CU�3���;��
�\(��uQ�C=�                                    BxdE:�  �          @Tz�=�Q��E�������C��)=�Q��+�����{C��q                                    BxdEI|  T          @P  ���
����\�"�HC�����
�޸R�&ff�`�HC���                                    BxdEX"  T          @b�\�L�����5��XC�,;L�Ϳ����QG�
=C�C�                                    BxdEf�  �          @��׾�z�� ���G
=�K�C��ᾔz�����j�HG�C��                                    BxdEun  
�          @��;B�\�<(��:=q�2��C�)�B�\��
�e��o�C�O\                                    BxdE�  
�          @�    �\(��
=�	�C��q    �,���L(��G
=C��q                                    BxdE��  �          @��R�B�\�Dz��.�R�#33C�  �B�\�\)�\(��]��C{#�                                    BxdE�`  
�          @�\)�}p��HQ��'��=qC|ff�}p����Vff�S�\Cv�                                    BxdE�  "          @�
=���
�333�:�H�1ffCy޸���
��Q��c33�i(�Cr{                                    BxdE��  "          @�zΐ33�   �C�
�?�HCuW
��33��\)�g
=�tCj�{                                    BxdE�R  �          @��Ϳ��9���.{�$�Cx쿕��XQ��Z�HCp��                                    BxdE��  "          @��ͿQ��.{�AG��;33C}:�Q녿��g��s��Cu��                                    BxdE�  
(          @��H��ff�*�H�+��(ffCt���ff����Q��\��CkQ�                                    BxdE�D  �          @��������*�H�%  Ca������z��H���K�CU�{                                    BxdF�  T          @|������%����
��=qCc�������Q��C\O\                                    BxdF�  T          @|��=#�
�%�<(��BffC�~�=#�
�޸R�`  �~(�C��)                                    BxdF%6  �          @|(�?.{��N�R�^�HC��?.{��Q��i���)C��                                    BxdF3�  �          @|��?!G����S�
�d�
C��q?!G���\)�n{�fC��
                                    BxdFB�  
�          @��\>u���dz��vQ�C���>u�p���|(�\)C�<)                                    BxdFQ(  
�          @��>�G��Q��S�
�W33C�H�>�G����H�s�
aHC�y�                                    BxdF_�  T          @��
>����:�H�8���233C�C�>������b�\�l�C���                                    BxdFnt  
�          @�z�?(��J=q�%��C�~�?(�����S�
�T��C�+�                                    BxdF}  
�          @�p�?O\)�^{�����C��?O\)�4z��;��3�
C���                                    BxdF��  "          @�z�?��fff��
=�ޏ\C���?��@  �1G��(��C�k�                                    BxdF�f  
Z          @�(�=��vff��(�����C��=��Vff�����
C�
=                                    BxdF�  T          @�녾Ǯ�b�\��\)�ݮC��R�Ǯ�=p��,���(��C�:�                                    BxdF��  �          @|(��=p��b�\���
���C��=p��B�\���p�C�"�                                    BxdF�X  T          @qG��\)�c33��G��|(�C���\)�K���\)��ffC��                                    BxdF��  "          @e��8Q��b�\>\)@z�C��׾8Q��]p��=p��AC�z�                                    BxdF�  �          @fff�u�aG�=�Q�?�z�C���u�[��J=q�O\)C���                                    BxdF�J  �          @\(��\�?\)?��A�C�Y��\�S33?W
=Ac\)C���                                    BxdG �  �          @Z=q�0���>{?�G�A�=qC�o\�0���P  ?8Q�ADz�C���                                    BxdG�  �          @s33����^�R?s33Aj{C}�����g
==u?n{C}�{                                    BxdG<  "          @|(�����e?uAb�\Cy�ÿ���n{=L��?:�HCz�
                                    BxdG,�  �          @��
�s33�|(��@  �&�HC�>��s33�hQ��Q����Ch�                                    BxdG;�  
Z          @�ff��ff��G���
=��Cuÿ�ff�tz῱����C~�f                                    BxdGJ.  "          @��\�k���ff��������C���k���  �������RC��H                                    BxdGX�  �          @�33�s33��G����\��=qC�aH�s33�g
=�{���
C=q                                    BxdGgz  �          @��
�Q�����������  C�O\�Q��e���{C��)                                    BxdGv   �          @��������ÿ�Q���z�C������`  �'
=���C�33                                    BxdG��  
�          @�녾�
=�I���8���(�HC�(���
=�
=�c�
�`=qC��                                    BxdG�l  �          @�G���\�O\)�0  ��C��H��\��R�\���V�HC�(�                                    BxdG�  
�          @��þ#�
�/\)�Mp��E��C�` �#�
����qG��}�\C���                                    BxdG��  
Z          @��H>����{�Z�H�b��C�@ >��ÿ�=q�vff�3C���                                    BxdG�^  �          @��H>u�����z�H
=C���>u�\(���  G�C�                                    BxdG�  �          @�
==u��z�����ffC��R=u�c�
���
C�{                                    BxdGܪ  �          @���>�p��Q��~�R�u�C���>�p���\)���
ǮC�:�                                    BxdG�P  
�          @�ff>��R����u��o�C���>��R��p���  ��C�
=                                    BxdG��  "          @�
=>\)�&ff�g��Y
=C��{>\)����(��fC�u�                                    BxdH�  "          @�>�=q�{�u��n��C�� >�=q��  ���k�C�%                                    BxdHB  �          @�z�>L���G��xQ��yp�C���>L�Ϳ�ff��  C�c�                                    BxdH%�  �          @�(�>u��
=�{��~ffC�z�>u�u����C��                                    BxdH4�  T          @�>�  ��=q�o\)�~C��3>�  �fff��=qC��q                                    BxdHC4  
�          @�  >�������p���yG�C�>�����\���L�C�R                                    BxdHQ�  
�          @�=q?
=q�33�p���r33C�h�?
=q��\)��(��\C���                                    BxdH`�  
�          @�G�>�33�   �qG��w
=C��>�33�����(��qC��                                    BxdHo&  "          @��R>#�
��h���p=qC�33>#�
��Q������)C��)                                    BxdH}�  �          @w��������@  �KffC�8R��녿���^{���C�{                                    BxdH�r  �          @j�H���!G��'
=�5�C�*=�������G��i�C~E                                    BxdH�  
�          @y��>8Q���\�I���WQ�C�L�>8Q��G��e33C�}q                                    BxdH��  T          @n{=#�
���7
=�Ip�C�|)=#�
��33�U��~��C��3                                    BxdH�d  T          @l�;u�z��6ff�J��C�  �u�����S�
���C���                                    BxdH�
  �          @l��<��ff�2�\�G�RC�ff<���33�P���|��C���                                    BxdHհ  T          @h�ý��
��\�3�
�K33C��q���
�˅�P�����C��=                                    BxdH�V  �          @j�H?fff�����P  �}��C�ٚ?fff�z��^�RǮC���                                    BxdH��  �          @n{@#33?c�
��H�'p�A�=q@#33?��	���A�R                                    BxdI�  �          @s�
@zῈ���=p��N�
C�T{@z᾽p��HQ��_��C���                                    BxdIH  "          @n{@p���z��.�R�>��C���@p�>����.{�>p�@�z�                                    BxdI�  �          @l��@;�>������  @��@;�?O\)�33�  Ax(�                                    BxdI-�  
�          @p��@!G���=q�'��8
=C��3@!G�>����'��7p�@�p�                                    BxdI<:  �          @w�?�׿����H���ZQ�C��?�׾��U��o
=C�R                                    BxdIJ�  �          @|(�@녿��B�\�K=qC��f@녿333�Q��c{C�s3                                    BxdIY�  T          @z�H@.{�#�
�/\)�1��C�h�@.{�#�
�3�
�7�RC��f                                    BxdIh,  
Z          @\)@3�
=u�(Q��,p�?�z�@3�
?#�
�#33�&�AM                                    BxdIv�  �          @��\@>�R?���#33��A��H@>�R?У��\)�A�p�                                    BxdI�x  �          @~{@@  >�z��$z��"  @�\)@@  ?Y���(��p�A}p�                                    BxdI�  
Z          @u@B�\>��R�z��z�@�(�@B�\?Q��(���Ap��                                    BxdI��  T          @tz�@8Q�?�R��H��AA�@8Q�?����R�
=A��\                                    BxdI�j  
�          @z=q@J�H?J=q�	�����A_33@J�H?��R��
=��z�A�G�                                    BxdI�  T          @c33@G������*�H�E�\C���@G�>�=q�+��F33@�                                      BxdIζ  �          @c�
@�R�#�
����2��C��=@�R?   ����-��A5��                                    BxdI�\  �          @`��@/\)?������ڣ�AڸR@/\)?�p���  ��=qB�                                    BxdI�  
�          @\(�@#�
?�G���\�Q�A��\@#�
?�
=��\��33A��
                                    BxdI��  "          @Y��@.�R?333������Af�R@.�R?����
=��A��\                                    BxdJ	N  T          @Vff@1�>�33��ff��@�@1�?:�H��
=���Aj�R                                    BxdJ�  T          @U@2�\?8Q��z���Ag�@2�\?����(���33A��H                                    BxdJ&�  
�          @L��@(Q�?+������Ac�
@(Q�?�G����R�޸RA�z�                                    BxdJ5@  �          @K�@)��?!G���\)��ffAV{@)��?u�����ٮA�{                                    BxdJC�  
�          @R�\@7
=?
=q��G��ۙ�A+33@7
=?Y����{��{A���                                    BxdJR�  �          @S�
@A�?녿�Q����A,  @A�?Q녿�����HAq��                                    BxdJa2  	�          @\��?��������;\)C��f?�����R�'��O�C�!H                                    BxdJo�  �          @[�?����Q��*=q�L��C�޸?����=q�=p��p�C�/\                                    BxdJ~~  
(          @U?�=q��=q�$z��Ip�C�@ ?�=q��  �9���q�C�y�                                    BxdJ�$  
(          @W�?��׿�p��(���O�C���?��׿����<���v33C���                                    BxdJ��  
�          @N�R?�  ��z��%�W\)C��?�  �
=�1��o�HC�G�                                    BxdJ�p  �          @P  ?�ff�У��%�T�
C�l�?�ff����7��{
=C���                                    BxdJ�  �          @S�
>���33�*�H�W\)C��=>�����@��\C��                                    BxdJǼ  �          @U�?:�H��{�@\)C���?:�H���
�6ff�mQ�C��                                    BxdJ�b  "          @S33?=p��ff���%�HC���?=p����'��S�C���                                    BxdJ�  T          @X��    ����7
=�gC��    ��33�J�H\C�                                    BxdJ�  "          @`  �#�
��
�'��Bz�C��ü#�
���H�B�\�r�C��f                                    BxdKT  T          @e    �%��R�/=qC�      �G��>{�_Q�C�                                      BxdK�  �          @l��<��
�+��{�*�C�,�<��
�
=�>{�Z�C�8R                                    BxdK�  "          @^�R�\)�AG���Q���Q�C����\)�&ff���%  C�k�                                    BxdK.F  T          @p  ��Q��Y����Q�����C�׾�Q��A����
��C��f                                    BxdK<�  �          @u�?@  �R�\���
�îC�o\?@  �9�����(�C�B�                                    BxdKK�  "          @��>u��녿�z��q�C���>u�~�R���ׅC��                                     BxdKZ8  �          @��?�\�~�R����l(�C���?�\�j�H������{C��3                                    BxdKh�  �          @�?E����ÿ5���C�g�?E��s33�\��G�C���                                    BxdKw�  �          @�(�>�  �\)�E��/
=C���>�  �o\)������C��                                    BxdK�*  "          @�z�?z�H�|(��Q��6�HC���?z�H�l(���{���RC�n                                    BxdK��  
�          @��\?�R��{�Y���5p�C�<)?�R�z�H��
=���C���                                    BxdK�v  
�          @�p�?E���  �\(��4z�C�)?E��\)���H��z�C�p�                                    BxdK�  �          @�\)?(���G�����n�HC�H�?(��n�R������C��H                                    BxdK��  T          @��\?!G����Ϳz�H�R�\C�T{?!G��w
=��ff�ř�C���                                    BxdK�h  T          @�G�?Y���\)��p���=qC�H?Y���j=q� ����G�C���                                    BxdK�  "          @�?������
������HC�Ff?����z�H�����RC���                                    BxdK�  
�          @�p�?��\���R��������C�e?��\������p����
C��
                                    BxdK�Z  �          @�  ?\���R�������\C��3?\��G���p��
=C�S3                                    BxdL
   "          @���?�����ÿ.{�
�RC��3?���s�
��(����\C�1�                                    BxdL�  �          @�  @Q��n�R�B�\��C�B�@Q��`  ���R���HC��                                    BxdL'L  �          @���?�\)���H>���@u�C�@ ?�\)��=q����(�C�O\                                    BxdL5�  �          @~{@G��Y����\)���C�]q@G��Tz�@  �.ffC��                                    BxdLD�  
�          @\)@�������\C�U�@��Ǯ�333�2p�C�Q�                                    BxdLS>  
�          @~�R@#�
��
=�
=�C�|)@#�
���,���*��C�t{                                    BxdLa�  
�          @vff@#33��׿ٙ��ԏ\C�E@#33��������C���                                    BxdLp�  "          @l(�@33�p�������33C���@33��ÿ��C���                                    BxdL0  �          @��׿��
�1G�@,(�B)�Cy�����
�O\)@ffA��C|s3                                    BxdL��  �          @j�H�k��'�@Q�B"p�Cz�׿k��B�\?���A���C}!H                                    BxdL�|  "          @u�O\)�Mp�?�G�A�Q�C�׿O\)�_\)?���A�ffC�~�                                    BxdL�"  
�          @|�Ϳ�  �j=q?�G�AmC~����  �r�\>�=q@~{C&f                                    BxdL��  
�          @p�׽����i��?��A	�C�.�����l(��.{�(Q�C�0�                                    BxdL�n  T          @S33?W
=��R@z�B#�
C�aH?W
=�%�?˅A���C��                                    BxdL�  C          @A�>�p��/\)?���A�Q�C��R>�p��:=q?z�A2�HC��                                     BxdL�  3          @QG�>aG��1�?�=qA�=qC�>�>aG��A�?�G�A�
=C�\                                    BxdL�`  "          @��׿����=q�fff�E��C������tz����
=C���                                    BxdM  T          @�\)�xQ����ÿO\)�0  C�E�xQ��s33�������\C�                                     BxdM�  f          @�33���\�y���@  �(��CG����\�k����R��{C~s3                                    BxdM R  �          @�33�^�R�z�H�@  �(��C��
�^�R�l�Ϳ��R���HC�\)                                    BxdM.�  
�          @��H�p���xQ�Q��9p�C�,Ϳp���i����ff���\C�                                    BxdM=�  T          @��
��\)�tz�������Cz:`\)�\(��33����CxB�                                    BxdMLD  "          @�{���������������C}s3�����n{�ff��(�C|�                                    BxdMZ�  T          @�Q쾊=q�n�R��\)���HC��{��=q�fff�}p��t��C��                                    BxdMi�  �          @{�=�Q��y��>#�
@�HC��=�Q��w
=�z����C��f                                    BxdMx6  �          @xQ�=��u>��@�p�C���=��vff���
��G�C���                                    BxdM��  "          @tz�>�p��n�R?(��A�RC���>�p��r�\�u�uC��=                                    BxdM��  "          @w
=>���u�=���?��
C��>���q녿!G��{C�q                                    BxdM�(  
�          @|(�?   �xQ�>�@�=qC��?   �y������o\)C��                                     BxdM��  
�          @z=q?��r�\?+�A{C��R?��vff�u�c�
C���                                    BxdM�t  
�          @~{�u�qG�?�A���C���u�z�H>�G�@�33C���                                    BxdM�  T          @xQ�?k��Q�?�AУ�C��{?k��a�?��\Ay�C�J=                                    BxdM��  �          @qG�?�ff�;�?�ffA�C���?�ff�N{?�(�A�z�C���                                    BxdM�f  T          @xQ�?�G��W
=?�G�A���C�Q�?�G��e�?\(�AK�C���                                    BxdM�  "          @k�?����H��?\Aď\C�޸?����W�?h��Af=qC�>�                                    BxdN
�  �          @n{?aG��Tz�?���A��RC�k�?aG��a�?L��AF�HC���                                    BxdNX  
�          @r�\?�
=�[�?z�HAr�RC���?�
=�c33>���@�  C�5�                                    BxdN'�  T          @q�?�\)�_\)>#�
@ ��C�� ?�\)�]p����H��RC���                                    BxdN6�  �          @w
=?��
�dz�?��A�C��f?��
�g���\)���C��f                                    BxdNEJ  T          @dz�?�G��Tzᾊ=q����C�ff?�G��Mp��aG��c�
C��
                                    BxdNS�  �          @]p�?�(�����p��G�C��{?�(���(��Q��7p�C��                                    BxdNb�  T          @b�\?�ff�(����+��C�W
?�ff��(��/\)�O
=C���                                    BxdNq<  T          @h��?��H�>�R��p���{C�{?��H�'
=�  �  C�w
                                    BxdN�  T          @j=q?�
=�N{�������C��?�
=�:�H�����(�C��R                                    BxdN��  T          @y��?�33�Z�H���H���C�&f?�33�H�ÿ�������C�
=                                    BxdN�.  T          @w�?�=q�O\)������{C���?�=q�;�������G�C�#�                                    BxdN��  T          @~{?��
�XQ쿱���ffC�"�?��
�Dz���R��RC�4{                                    BxdN�z  �          @q�@���5?Q�AK�
C�Ff@���<(�>�z�@�{C���                                    BxdN�   
�          @hQ�?�\)�@  ?@  A?�
C��)?�\)�E>B�\@A�C���                                    BxdN��  �          @z�H@ ���U>�G�@�z�C���@ ���W
=�8Q��'
=C�q�                                    BxdN�l  �          @vff?�{�Q�?:�HA.=qC���?�{�W
==�?޸RC�u�                                    BxdN�  �          @l��?�\�H��?E�A?�
C���?�\�N{>8Q�@6ffC�e                                    BxdO�  �          @_\)?�\�<(�?(�A ��C�|)?�\�@  =u?��C�9�                                    BxdO^  �          @mp�@�\�A�?(�A�\C��{@�\�E=L��?8Q�C���                                    BxdO!  
Z          @e�@z��p�>�p�@�(�C�4{@z��\)���z�C�H                                    BxdO/�  �          @r�\?��R�,��?�\A���C�.?��R�>{?�  A���C�޸                                    BxdO>P  �          @n{?�\)�4z�?�ffAĸRC�˅?�\)�C33?��\A}p�C���                                    BxdOL�  T          @j�H?�  �:�H?��A�
=C��)?�  �L��?��RA�Q�C���                                    BxdO[�  
�          @fff@
�H�*=q?�=qA��C���@
�H�3�
?z�AQ�C��\                                    BxdOjB  "          @c�
?�(��#33?�\)A�  C��)?�(��2�\?���A�ffC��f                                    BxdOx�  T          @_\)?�z��(�@
=qBp�C��=?�z��2�\?�Q�A�p�C�AH                                    BxdO��  T          @e�?����1�?�(�Bp�C���?����E�?�Q�A��C���                                    BxdO�4  �          @n�R?�ff�)��@�
B��C�8R?�ff�>{?ǮA�33C��\                                    BxdO��  �          @w�?�=q�+�@z�B  C�H�?�=q�C33?�A�{C���                                    BxdO��  �          @l��?��B�\?�p�A��C��?��S33?�z�A�(�C��=                                    BxdO�&  "          @h��?h���P  ?��A�=qC�ٚ?h���[�?8Q�A6ffC�s3                                    BxdO��  T          @dz�?s33�Q�?W
=A\  C��?s33�XQ�>u@z�HC�ٚ                                    BxdO�r  "          @>�R?L���Q쿣�
��z�C�H�?L������
=��RC�Y�                                    BxdO�  
Z          @U�?�=q�<�;�  ��(�C�?�=q�7
=�E��c
=C�Z�                                    BxdO��  
�          @e?���C33?}p�A�C�^�?���K�>�G�@�\C��f                                    BxdPd  "          @X��?˅��R?���A�=qC�]q?˅�+�?c�
A}�C�aH                                    BxdP
  S          @W�?�=q��H?���B ��C���?�=q�,��?�\)A���C�&f                                    BxdP(�  
�          @S�
?У���?��A���C�H�?У��&ff?���A��C��                                    BxdP7V  �          @z=q@�׿�(��;��C33C���@�׿0���Fff�R�C���                                    BxdPE�  �          @�(�@G���{�P  �P�C��{@G����X���^Q�C��                                    BxdPT�  T          @�G�@p�����`  �[=qC���@p����hQ��h�\C��                                    BxdPcH  �          @�ff@��  �r�\�jG�C�(�@���
�z=q�vffC��
                                    BxdPq�  
�          @�=q@ �׿Y���mp��n�C��3@ �׾B�\�s33�x
=C�]q                                    BxdP��  
j          @'
=?�\�Ǯ���K�C��?�\��Q��G��qffC���                                    BxdP�:  �          @xQ�?�33����Fff�W�RC��?�33�xQ��Tz��o�C�Ǯ                                    BxdP��  �          @e�?�zῴz��%�=��C�� ?�z�p���333�R�
C���                                    BxdP��  "          @�(�?�zῠ  �X���_��C�j=?�z�#�
�c33�qz�C���                                    BxdP�,  T          @�(�@  ���H�n{�h�HC��=@  >.{�o\)�k�@���                                    BxdP��  �          @���@�Ϳ!G��z�H�n�
C��R@��=u�~{�t  ?��R                                    BxdP�x  T          @��?�p��Q�����xz�C��?�p�������z��3C��=                                    BxdP�  "          @�ff?�녿8Q��|���{p�C��H?�논��
��Q�z�C���                                    BxdP��  T          @xQ�?���Tz��a��C��3?���W
=�g�B�C�b�                                    BxdQj  "          @l��?�  �\�S�
C�޸?�  >L���Tz���@�                                    BxdQ  
�          @Y��?�
=�aG��C�
W
C��?�
=>��R�C33�3AE�                                    BxdQ!�  "          @Fff?��
���8Q�Q�C���?��
���
�:�HB�C�|)                                    BxdQ0\  T          @4z�?\(���R�'
=��C��?\(��#�
�+��RC���                                    BxdQ?  
�          @E?���\)�5���C��?����\)�8��z�C�9�                                    BxdQM�  
�          @I��?������7��C��3?���=#�
�:=q  @�                                    BxdQ\N  �          @C�
?�z���4z�C��?�z�>�p��2�\�A��H                                    BxdQj�  T          @o\)@녿�G��=q�%C�C�@녿����(Q��933C�l�                                    BxdQy�  
�          @�p�?��p  ?��A�Q�C���?��z�H?+�A��C�c�                                    BxdQ�@  �          @�{?�(��dz�?��HA�33C��q?�(��n�R?�@�
=C�ff                                    BxdQ��  T          @���?��
�g�?�z�A��HC�o\?��
�vff?�G�A[�C�Ф                                    BxdQ��  
�          @�p�?���g
=?�A�z�C�?���u?��
AX(�C�
=                                    BxdQ�2  �          @}p�?�33�P  ?޸RA�p�C���?�33�`  ?�z�A���C��                                    BxdQ��  �          @���?�
=�U?�\A��
C���?�
=�fff?�A�
=C��R                                    BxdQ�~  �          @���?�\)�C�
@�\A�\C���?�\)�W
=?��RA�Q�C��q                                    BxdQ�$  �          @�Q�?�=q�5�@
�HB��C�g�?�=q�J=q?�33A���C��                                    BxdQ��  
�          @x��?��%�@  B��C�Ǯ?��:�H?��
A�  C�)                                    BxdQ�p  �          @���@��\)@'
=B!��C�u�@��)��@(�B�C�3                                    BxdR  �          @w
=?�\���@.{B:
=C��
?�\�z�@
=B
=C���                                    BxdR�  
�          @c�
����'
=@=qB*(�C�4{����>�R?�
=B(�C��                                    BxdR)b  �          @Y���#�
�.�R?�33B�
C�h��#�
�@��?�z�A�33C��                                    BxdR8  �          @J=q�Ǯ� ��?�  B
\)C��H�Ǯ�1G�?�ffA�G�C��                                    BxdRF�  �          @A녾�z��4z�?�ffA�
=C�þ�z��<��?\)A*ffC�8R                                    BxdRUT  �          @?\)�.{�5�?k�A�  C�N�.{�<(�>�(�A  C�^�                                    BxdRc�  T          @:=q�333�"�\?��HA��HC~�)�333�-p�?B�\Ar�HC�                                    BxdRr�  �          @K�����
?���B#�RC~�����ff?�p�A��
C�1�                                    BxdR�F  �          @Dzᾅ��2�\?�A�G�C�O\����<��?.{AO33C�t{                                    BxdR��  �          @H�þ\)�/\)?�  A�C��׾\)�<��?��\A��
C��                                     BxdR��  �          @E���33�\)?�  B=qC��)��33�0��?�ffA�G�C�\)                                    BxdR�8  T          @L��    �0��?���A��HC���    �?\)?�{A�
=C���                                    BxdR��  
�          @XQ�?+��G
=?�=qA���C�R?+��P  ?
=qA�HC��R                                    BxdRʄ  
Z          @X��?�(��HQ콣�
����C���?�(��Dz�(��&ffC��R                                    BxdR�*  �          @\��?���,�Ϳ��H��C��f?���������C�E                                    BxdR��  T          @]p�?�z��Dz῁G����HC�` ?�z��6ff��G���33C�{                                    BxdR�v  
�          @XQ�?u�N{>�  @��\C�O\?u�Mp���z���\)C�Q�                                    BxdS  �          @^�R?�Q��P  =��
?�=qC��?�Q��N{�������C�(�                                    BxdS�  
�          @S33?���>{>B�\@QG�C�%?���=p����
��=qC�/\                                    BxdS"h  
�          @-p�����(�@   B?��C|�׿���\?�
=Bz�Cc�                                    BxdS1  "          @%�}p���\)@ffBZffCd��}p����H?��B;��Ck�3                                    BxdS?�  �          @(�ÿ�����H@ffBS\)CeT{������?�\)B4p�Cl�                                    BxdSNZ  "          @,(���{����@�\BE�HCf���{��
=?��B&�\Cl�H                                    BxdS]   "          @/\)��
=��{?�\)B,Q�Ci�
��
=��z�?���B(�Cn=q                                    BxdSk�  �          @4zῂ�\��p�?�p�B2��Co� ���\��?�z�B�\Csp�                                    BxdSzL  
�          @@�׿�  ��{@Q�B4Cq�쿀  �(�?��
B
=Cu�=                                    BxdS��  
�          @@  �aG��p�?���B��CxT{�aG��\)?�
=A�p�Cz�=                                    BxdS��  T          @:=q�8Q��
=q?�\B��C{��8Q���?���A�\)C}�{                                    BxdS�>  �          @5����
���?z�HA���C�Ǯ���
���?z�A[\)C�˅                                    BxdS��  T          @K����
��z�?�B�ClͿ��
���?���A�p�Co                                    BxdSÊ  �          @Vff�Ǯ���?���B(�Ckff�Ǯ�#�
?��A�p�Cn��                                    BxdS�0  �          @>{��ff�?��A�\)Cp�f��ff�!G�?^�RA�33Cr�3                                    BxdS��  �          @I����  ���?�=qA�(�Cm�ÿ�  �%�?h��A��Co��                                    BxdS�|  "          @:=q��  �!G�>u@�(�Cs�{��  �!녾B�\�q�Cs��                                    BxdS�"  T          @5�
=�(�>#�
@{�C~�ÿ
=�(��W
=��z�C~�                                    BxdT�  �          @0  ���H�˅?��
A�
=C[
���H�޸R?=p�A}�C]�H                                    BxdTn  �          @,(��˅���R?\(�A�CgLͿ˅�ff?�A1p�Ch�H                                    BxdT*  
�          @*=q��녿�=q?p��A�\)Cd���녿�(�?�RAY�Cf
                                    BxdT8�  
(          @.{��\)��ff?�  A�(�Cd��\)���R?n{A�\)Cf�\                                    BxdTG`  �          @-p���(���?��RAۙ�C`0���(�����?s33A�Q�Cc33                                    BxdTV  
�          @�H��\)�\?aG�A�z�C_@ ��\)��33?(�Al��Ca�                                    BxdTd�  �          @.{������H?ٙ�B��C_�)�����p�?�
=A���CdY�                                    BxdTsR  �          @6ff��\)��  ?�BQ�C^ٚ��\)���
?��
B�
Cc�                                    BxdT��  "          @aG��z��ff@�B{C\��z����?���A���Ca�H                                    BxdT��  
�          @c33�G����
@�
B"p�C]n�G����?�(�B�Cb�R                                    BxdT�D  �          @*=q�Ǯ��ff?��Ạ�Cd�q�Ǯ��(�?Q�A�33Cgz�                                    BxdT��            @.�R����z�?�A�ffC^p�����=q?aG�A���CaJ=                                    BxdT��  2          @q��#�
���?�z�A�  C\�
�#�
�(�?��\A�{C_��                                    BxdT�6  
�          @N{��R��?�  A�z�C[���R��\?�A��C^s3                                    BxdT��  
�          @s33�'
=�z�?�A�(�CZxR�'
=�ff?�Q�A�(�C]�R                                    BxdT�  �          @z=q�/\)�G�?�
=A���CXW
�/\)��
?���A��HC\&f                                    BxdT�(  
�          @z=q�5���@ ��A���CT0��5�ff?�
=A���CXu�                                    BxdU�  �          @tz��<�Ϳ�
=?�Q�A�\CM޸�<�Ϳ޸R?�A�  CRxR                                    BxdUt  
�          @u�J�H��33?ǮA���CK�H�J�H��33?�ffA�{COn                                    BxdU#  �          @L�Ϳ\����?��A�{Cf��\�33?�G�A�
=Cih�                                    BxdU1�  T          @+�>u�#33?8Q�AyC���>u�(��>�\)@�\)C���                                    BxdU@f  �          @*=q��� ��?aG�A�z�C��\���'�>�ffA=qC��3                                    BxdUO  �          @;��Q��&ff?�G�A��C|z�Q��.�R?\)A2=qC}G�                                    BxdU]�  �          @.�R�^�R�z�?��
A��Cy^��^�R�p�?�RAVffCzp�                                    BxdUlX  �          @'
=��=q��
=?��
A��HCk� ��=q��?0��Axz�Cmh�                                    BxdUz�  �          @�H��ff���H?���A�p�C_:��ff��\)?W
=A���CbL�                                    BxdU��  T          @'���G���
=?��HA�p�C[��G���{?uA��RC^�                                     BxdU�J  �          @4z��\)��33?�G�A֣�C]aH��\)��?xQ�A�C`z�                                    BxdU��  	�          @U��"�\���?�G�A�33CSL��"�\���
?�p�A��RCV��                                    BxdU��  "          @Tz��"�\��\)?�{A£�CT���"�\��=q?���A���CW�                                     BxdU�<  "          @�H�����\?:�HA��HCr� ����Q�>\A33Cs�=                                    BxdU��  T          @��B�\�
=?&ffA�G�Cz5ÿB�\�(�>�z�@�Q�Cz޸                                    BxdU�  "          @p���R�z�>8Q�@�  C~�R��R��
�aG���(�C~�3                                    BxdU�.  	`          @�׿J=q��(�?:�HA�  Cx)�J=q�z�>ǮA33Cy�                                    BxdU��  �          @�=u�33?z�Az�\C��=u��>aG�@�{C���                                    BxdVz  �          @ff>�����?0��A�ffC��)>���\>�33Ap�C�Ǯ                                    BxdV   
�          @�����33?.{A��C�t{���Q�>�\)@׮C���                                    BxdV*�  �          @���{�  ?�RAt(�C��=��{�z�>k�@�(�C���                                    BxdV9l  �          @�R�����?+�A�Q�C��3�����
�H>���@�(�C��                                    BxdVH  
�          @�þ�z����?��\A�ffC�q쾔z��(�?333A�\)C�Ǯ                                    BxdVV�  T          @{��\)��G�?��
A�ffCi}q��\)��?B�\A��\Cl�                                    BxdVe^  "          @  ��ff�޸R?n{A��Cn��ff��\)?�RA�z�Cp�
                                    BxdVt  
Z          @�׿s33��z�?8Q�A��Cs}q�s33� ��>ǮAffCt�{                                    BxdV��  T          @33<��
��?��A�=qC�5�<��
��?333A�  C�1�                                    BxdV�P  �          @���=q��=q?k�A�G�C�޸��=q���H?
=A�C�#�                                    BxdV��  "          @
=�   ��(�?���A��C}���   ���?J=qA��\C~��                                    BxdV��  "          ?�z�B�\��
=?}p�A�Q�Cq��B�\��=q?=p�A�ffCtB�                                    BxdV�B  
�          @$zῡG���
=?�(�B��Ci(���G���33?�z�A׮Cl��                                    BxdV��  
�          @0�׿�G���ff?�A���Cf  ��G��G�?��A�G�Ci&f                                    BxdVڎ  �          @6ff��녿���?��A�{CdW
�����
?�ffA�  Cg\)                                    BxdV�4  
�          @'���ff�z�?
=qA@  Cu����ff�Q�>\)@A�Cv0�                                    BxdV��  T          @'
=�����33?5A
=ClͿ����	��>�33@�
=Cm0�                                    BxdW�  
(          @\)����
=>B�\@�Q�Cn������
=�.{�y��Cn�H                                    BxdW&  
�          @5���\�
=q>\@���Cf����\�(�<#�
>#�
Cg
=                                    BxdW#�  
�          @,�Ϳ޸R��\)?\(�A�z�Cc
�޸R���R?�A3�Cdٚ                                    BxdW2r  �          @2�\�Ǯ��=q?���A�Q�Cek��Ǯ��?��A���Chu�                                    BxdWA  
�          @#33��=q��33?��B!�RCb�H��=q��z�?���B�
CgxR                                    BxdWO�  
(          @/\)�ٙ��˅?�\)A�p�C_��ٙ���ff?�=qA���Cb�
                                    BxdW^d  T          @'
=��
=���?���AîCV�H��
=��G�?Q�A��\CZ                                    BxdWm
  �          @@  �z��\?�
=A���C\���z����?Y��A��C_O\                                    BxdW{�  T          @?\)��z���H?���B(�CoT{��z��  ?�p�A��HCr��                                    BxdW�V  "          @?\)��Q��G�?��HB
=Cb33��Q���?���A���Cf:�                                    BxdW��  �          @7
=��(����H?�  A�ffC`��(���Q�?�Q�Aď\Cdk�                                    BxdW��  	�          @"�\��
=��=q?p��A��CV�)��
=��(�?0��A~=qCY\)                                    BxdW�H  �          @ �׿h�ÿ���?���B)�RCn  �h�ÿٙ�?��B33Cq�                                    BxdW��            @G�?O\)��33?c�
A��C���?O\)��?
=qA`  C���                                    BxdWӔ             @(�?n{���H?n{A˅C�9�?n{����?�RA���C�U�                                    BxdW�:  
Z          @!G�?����(�?E�A��RC��f?����>�
=A33C��                                    BxdW��  �          @��?�ff�   ?O\)A�  C���?�ff�
=>�ffA-��C�C�                                    BxdW��  T          @(Q�?�{��?�=qA�C��?�{��R?5A}G�C�8R                                    BxdX,  
�          @/\)?�G�� ��?���A�p�C�3?�G��p�?z�HA��RC���                                    BxdX�  
�          @333?����   ?�{A�  C�P�?����
=q?@  Ax��C�:�                                    BxdX+x  
Z          @/\)?����(�?��A�z�C��?���Q�?:�HAxz�C��\                                    BxdX:  
(          @(�?�(����H?�(�A�G�C�� ?�(����?fffA���C�`                                     BxdXH�  
�          @'�?�\)��  ?�  B��C�W
?�\)��p�?�A�=qC��q                                    BxdXWj  v          @,��?�\)��33?�G�A�RC�޸?�\)�?h��A�(�C��
                                    BxdXf  d          @0��?c�
�?ǮB	ffC���?c�
��?�z�A��C�p�                                    BxdXt�  �          @8��?n{��?��HB�C��H?n{�Q�?��AظRC��                                    BxdX�\  
�          @7�?���Q�?�{B(�C�/\?���Q�?��HA��C��
                                    BxdX�             @>{?�=q�G�?�  A�(�C���?�=q�  ?�\)A���C���                                    BxdX��  �          @>{?�\��(�?��A�  C���?�\�
�H?p��A�Q�C���                                    BxdX�N  �          @>{@녿�?���A��RC��R@��   ?G�AuC���                                    BxdX��  �          @g
=@.{��;\��  C���@.{�ff�E��H��C�,�                                    BxdX̚  �          @|(�@*�H�p�?�
=A�ffC��\@*�H�*�H?z�HAg�
C�}q                                    BxdX�@  
�          @��
@XQ��Q�?}p�A_�C��@XQ��G�?��A�\C��                                    BxdX��  
K          @��@e���(�>\)?�C�� @e����H�W
=�?\)C��=                                    BxdX��  �          @��@XQ��  �����p�C�(�@XQ����p���S�C��{                                    BxdY2  T          @\)@L���
=q�u�^�\C��)@L�Ϳ��H��=q����C�G�                                    BxdY�  T          @z=q@K��
=�Q��AC�:�@K���
=��
=���HC�aH                                    BxdY$~  �          @|(�@Fff��\�333�$Q�C�@Fff�Q쿌�����
C���                                    BxdY3$  T          @q�@R�\�У׿B�\�:{C��\@R�\��(������(�C��{                                    BxdYA�  
�          @mp�@>{�   ����C�AH@>{������R���C��                                    BxdYPp  
�          @n�R@C�
�\����C�q�@C�
    �Q��33C��)                                    BxdY_  T          @l��@Mp��z��p���\)C��H@Mp��������{C���                                    BxdYm�  "          @s33@U��0�׿�Q��ә�C�&f@U���p������RC��=                                    BxdY|b  �          @p��@Y�������
���C���@Y���u������
=C��                                    BxdY�  �          @qG�@Tz��ff��(���p�C�'�@Tz�����
��RC��q                                    BxdY��  �          @q�@QG���׿�����\C��=@QG�����\)��(�C���                                    BxdY�T  �          @q�@E�+���\�ffC��@E��z��Q��	C�S3                                    BxdY��  
Z          @q�@C�
�h�������C���@C�
���
=q�{C�#�                                    BxdYŠ  
�          @qG�@P  �aG���
=��z�C�q�@P  �\)����z�C�                                      BxdY�F  T          @tz�@Z�H�!G���=q��  C���@Z�H���ÿ��ϮC�4{                                    BxdY��  �          @u@^{��\��ff��Q�C��R@^{�aG���{��
=C�33                                    BxdY�  "          @mp�@W
=��z���
��\)C���@W
=    �Ǯ����C���                                    BxdZ 8  �          @P  @.�R���Ϳ���\)C���@.�R��Q��(�� p�C��                                    BxdZ�  �          @\(�@7
=�u��{��\C��H@7
==�G���\)��@ff                                    BxdZ�  
�          @]p�@%�>�(����R�=qA(�@%�?J=q�����A�                                      BxdZ,*  �          @g
=@333���H��33���
C�O\@333��\)�(���8(�C��                                    BxdZ:�  	�          @p��@�
�<(�����C�)@�
�8Q�����C�`                                     BxdZIv  �          @mp�@{�2�\�fff�c33C�=q@{�%���33����C�Z�                                    BxdZX  T          @l(�@33�<(��:�H�8��C�t{@33�0  ��G���33C�W
                                    BxdZf�  �          @j�H?�  �N{��p���ffC�w
?�  �Fff�u�x��C��f                                    BxdZuh  T          @c�
>����aG��.{�.{C�H�>����[��O\)�Tz�C�^�                                    BxdZ�  �          @c�
>\�a�<�?�C��>\�^{��R�!�C��                                    BxdZ��  �          @e=�\)�c�
���Q�C���=�\)�^�R�J=q�L��C��{                                    BxdZ�Z  �          @^�R���
�\��>�\)@�(�C�P����
�\(���Q�����C�P�                                    BxdZ�   �          @\�;u�Z�H��\)��C�f�u�S�
�h���u��C���                                    BxdZ��  	�          @k��L���g������C�s3�L���a녿Q��R=qC�h�                                    BxdZ�L  �          @o\)<#�
�[�?8Q�A=G�C��<#�
�`  =�\)?��C��                                    BxdZ��  �          @l(���Q��e�?��Az�C�AH��Q��hQ콣�
��p�C�C�                                    BxdZ�  
�          @l(�=��j=q�#�
�$z�C���=��c�
�W
=�Up�C��R                                    BxdZ�>  T          @s�
>W
=�q�>k�@]p�C���>W
=�p�׾���=qC��3                                    Bxd[�  �          @u�#�
�p  �#�
���C�33�#�
�j=q�\(��Q�C�3                                    Bxd[�  �          @r�\���n{���Ϳ��HC�箿��hQ�J=q�C
=C�Ф                                    Bxd[%0  �          @q�    �qG���G��ٙ�C��    �k��Q��H��C��                                    Bxd[3�  �          @s�
>��R�q녾#�
���C�` >��R�k��^�R�S�C�p�                                    Bxd[B|  
Z          @r�\?��n�R=���?�ffC��{?��k��(���
C�H                                    Bxd[Q"  �          @w�����N�R���
��(�C{T{����>{��\)�؏\Cy޸                                    Bxd[_�  
�          @s33�Y���Tz��ff���C�3�Y���>{�
=q���C~�                                    Bxd[nn  
�          @o\)�:�H�\(���p���{C�  �:�H�H�ÿ�{��C�t{                                    Bxd[}  T          @n{�#�
�fff�c�
�]G�C��\�#�
�W
=��ff��ffC��=                                    Bxd[��  "          @n{>aG��b�\��{��\)C���>aG��P�׿�\��33C��{                                    Bxd[�`  
K          @l(�=��a녿�ff��ffC��q=��P�׿ٙ���z�C��                                    Bxd[�  	�          @o\)>����e��Q��MG�C���>����W
=���R���\C��{                                    Bxd[��  T          @j�H?#�
�K��˅��G�C��\?#�
�3�
�
�H��C�ff                                    Bxd[�R  �          @p  ?�Q��A녿�����G�C�� ?�Q��(Q����z�C�9�                                    Bxd[��  �          @n�R?B�\�W�������C�O\?B�\�C33�   �{C��
                                    Bxd[�  �          @j�H>����U�������HC�k�>����@���   ��\C��=                                    Bxd[�D  �          @c33=�Q��Q녿������RC��==�Q��>{��
=��C��                                     Bxd\ �  �          @e���\�W���=q���C��=��\�E���H��\)C�H�                                    Bxd\�            @n�R=�Q��U������C�Ф=�Q��?\)�z��
��C���                                    Bxd\6  �          @p  ?�ff�5��
(�C�&f?�ff�Q��&ff�3C���                                    Bxd\,�  �          @p�׾�=q�e���ff��z�C��R��=q�S33��p���{C��=                                    Bxd\;�  �          @tz῁G��c33�h���]C~&f��G��S�
�˅��\)C}�                                    Bxd\J(  �          @qG��+��`�׿xQ��t  C����+��P  ����ҏ\C�4{                                    Bxd\X�  �          @p�׿aG��dz�(���#�
C�{�aG��W��������CaH                                    Bxd\gt  �          @s33�:�H�hQ�G��>=qC�W
�:�H�Y����p����
C���                                    Bxd\v  �          @p�׾�=q�`�׿��\���\C��{��=q�L(���
=��
=C��q                                    Bxd\��  �          @vff?n{�Mp������
C�\?n{�2�\�=q�ffC�33                                    Bxd\�f  �          @~{?��B�\���z�C���?��#�
�+��*�C��=                                    Bxd\�  �          @w�?��H�@  ����C��
?��H�!��*�H�.�C�˅                                    Bxd\��  �          @u?�(��%����=qC�˅?�(��ff�*�H�2�C��f                                    Bxd\�X  �          @u�@���\)�C��=@����&ff�+�\C�O\                                    Bxd\��  �          @r�\?�33�   �	�����C��R?�33�G��'
=�3�C���                                    Bxd\ܤ  �          @w
=@�\�G�����p�C��H@�\����#33�%
=C��3                                    Bxd\�J  �          @tz�@G��ff��p���C�H@G���33�=q�z�C��q                                    Bxd\��  �          @o\)@�\����p�C���@�\��{�!G��)�\C���                                    Bxd]�  �          @p��?�
=�*�H������C���?�
=�  �z��33C�^�                                    Bxd]<  �          @u@
=q� �׿�����(�C�Z�@
=q���z���C�H                                    Bxd]%�  �          @z=q@*=q�����H����C���@*=q��33��33C��                                    Bxd]4�  �          @{�@333���R��z���=qC�Ff@333�����G��(�C�b�                                    Bxd]C.  �          @�  @8�ÿ���G���RC�ff@8�ÿ�Q��
=�C��)                                    Bxd]Q�  �          @}p�@4z������p���  C���@4z��G����C��                                    Bxd]`z  �          @|(�@2�\��
=�   ��(�C��@2�\��p��ff��\C�f                                    Bxd]o   �          @~{@=p��������ffC�c�@=p���{�  �
\)C��                                     Bxd]}�  �          @|��@9��������R��=qC�1�@9�������
�z�C��
                                    Bxd]�l  �          @qG�@\)�z�����p�C�q@\)����G��  C�K�                                    Bxd]�  �          @r�\@!���ÿ�=q����C�ٚ@!녿�(���R�G�C��)                                    Bxd]��  �          @tz�@-p��33��p���(�C�y�@-p���33�����C�]q                                    Bxd]�^  T          @p��@*�H��\��
=�ԏ\C�U�@*�H��33�z��Q�C�+�                                    Bxd]�  �          @qG�@%��33�����C���@%���\)��R���C��                                    Bxd]ժ  �          @r�\@p��(�������\C�(�@p���G�����=qC�5�                                    Bxd]�P  �          @w�@��'
=��z���{C�C�@��
=q�=q�(�C��
                                    Bxd]��  �          @x��@
=�&ff��(���C��=@
=�Q��p���RC�Y�                                    Bxd^�  �          @tz�?�p��9�����
�ݮC�h�?�p��{��G�C��f                                    Bxd^B  T          @vff?�33�>{��\���C���?�33�!��ff�G�C��                                    Bxd^�  �          @q�?�(��P  �������HC�H�?�(��8Q��ff�\)C�~�                                    Bxd^-�  �          @q�?���Z=q�����Q�C�=q?���G
=���
��C�3                                    Bxd^<4  �          @n�R>�ff�dz�n{�h��C��q>�ff�S33��z��ԏ\C���                                    Bxd^J�  �          @qG�?&ff�`  ��Q����
C�AH?&ff�J�H��33��G�C���                                    Bxd^Y�  �          @z=q?@  �c33��z���33C��)?@  �K����(�C���                                    Bxd^h&  
�          @x��?��H�S�
��33��=qC��?��H�8����
���C�\)                                    Bxd^v�  �          @p��?@  �c�
�u�m�C��\?@  �QG��ٙ���=qC�o\                                    Bxd^�r  �          @vff?����Mp���
=���RC�l�?����5����33C���                                    Bxd^�  �          @y��?��R�XQ쿚�H��
=C���?��R�B�\��33��Q�C��                                    Bxd^��  �          @�Q�?�{�Y��������C��?�{�B�\��\��\)C��)                                    Bxd^�d  �          @~{?�\)�R�\�\���C�3?�\)�8���(����C���                                    Bxd^�
  �          @~�R?�=q�H�ÿ\��G�C�!H?�=q�/\)�
=q��RC��                                     Bxd^ΰ  �          @\)@  �2�\��  ���
C�}q@  �ff�33�=qC��                                    Bxd^�V  �          @}p�@p��3�
����(�C��@p�����\)�
  C�aH                                    Bxd^��  �          @}p�@33�9����(�����C���@33�p���
��C��                                    Bxd^��  �          @~�R@�H�!G�����z�C��@�H��\��H��C��=                                    Bxd_	H  �          @�Q�@.{���G����HC���@.{��33�{�=qC��=                                    Bxd_�  �          @���@p��(��G��
33C�*=@p�����,(��){C��                                    Bxd_&�  �          @|(�@Q�ٙ��(Q��'��C�4{@Q쿋��<(��A33C��{                                    Bxd_5:  �          @|(�@�Ϳ�ff�\)�\)C���@�Ϳ�(��5��7��C��                                    Bxd_C�  �          @}p�@{�����RC�E@{��(��.{�.  C���                                    Bxd_R�  �          @|(�@p��"�\� �����HC��\@p����!G���
C��
                                    Bxd_a,  �          @|(�@	���,(���33����C�U�@	���p������C�&f                                    Bxd_o�  �          @{�@{�   ��\��\)C���@{���R�#33�!�HC�
                                    Bxd_~x  �          @{�@�$z���� ��C��{@��\�&ff�&ffC��3                                    Bxd_�  �          @w
=?�=q�HQ��\)�뙚C�� ?�=q�(���!G��%�RC�)                                    Bxd_��  �          @l��?k��+��
=��
C�~�?k���9���N�C��f                                    Bxd_�j  �          @k�?���'���p�C�  ?����\�7��M=qC���                                    Bxd_�  �          @i��?��HQ���
��Q�C���?��)�����'�HC�~�                                    Bxd_Ƕ  �          @n{�#�
�`�׿��
���HC�޸�#�
�L(������ffC�^�                                    Bxd_�\  �          @n�R���`  ����ffC�����I�������HC���                                    Bxd_�  �          @p�׿5�`�׿�\)����C�Ff�5�J�H�����z�C��                                    Bxd_�  �          @n�R�333�_\)���
��G�C�]q�333�J�H�����33C��\                                    Bxd`N  �          @n�R�Ǯ�`  ��
=���C�˅�Ǯ�I����Q����C�o\                                    Bxd`�  �          @o\)���^�R��(�����C��3���G���(����
C�4{                                    Bxd`�  �          @u�=#�
�R�\����(�C�]q=#�
�1G��'��-p�C�o\                                    Bxd`.@  �          @n�R�#�
�X�ÿ�ff��{C����#�
�<�����z�C�~�                                    Bxd`<�  �          @mp��u�Tz��{����C���u�7
=�����C��\                                    Bxd`K�  �          @n�R��=q�P�׿�G���Q�C�����=q�1G��p��%�RC�8R                                    Bxd`Z2  �          @l(�����P  ����ԣ�C�k�����2�\���RC��
                                    Bxd`h�  �          @s33���Q녿�33���HC��
���0���'
=�-p�C��=                                    Bxd`w~  �          @p�׼#�
�QG���=q��C����#�
�0���!��*
=C��                                    Bxd`�$  T          @s33=��
�X�ÿٙ���
=C���=��
�9�������C��\                                    Bxd`��  �          @u������b�\������p�C��=�����HQ��
=q�	�C�7
                                    Bxd`�p  �          @s33�k��`  �����RC�  �k��Dz��(��C��q                                    Bxd`�  �          @tz�>�33�b�\��{����C��=>�33�HQ������
C�'�                                    Bxd`��  �          @u���G��b�\��  ����C�l;�G��J=q��\��RC��q                                    Bxd`�b  �          @w������dz῵��G�C�Y������H���p��G�C��)                                    Bxd`�  �          @y������fff��\)��\)C�J=����K��
�H�=qC��\                                    Bxd`�  �          @w�?z��]p����
��z�C��f?z��@���33��\C�z�                                    Bxd`�T  �          @vff?���AG�������G�C���?���&ff�����C��)                                    Bxda	�  �          @|(�?�z��O\)��p����C��H?�z��/\)�����C��                                     Bxda�  �          @z�H?!G��U��z����C�S3?!G��2�\�(���*C�U�                                    Bxda'F  �          @~{?\)�U���
���\C�Ǯ?\)�0  �2�\�3{C���                                    Bxda5�  �          @���?O\)�P���
�H��\C���?O\)�)���8Q��8p�C�|)                                    BxdaD�  �          @���?���W
=��z���\C��\?���3�
�*=q�&ffC�q                                    BxdaS8  T          @~{?ٙ��C�
��=q��ffC���?ٙ��!��   �G�C���                                    Bxdaa�  T          @~{?���E������{C��?���"�\�%��!C�l�                                    Bxdap�  �          @~�R?����@  ��=q��Q�C���?�����R�   �Q�C�#�                                    Bxda*  �          @���?�(��:�H����Q�C�E?�(��z��1��/�RC�K�                                    Bxda��  �          @���?�ff�1��Q��  C���?�ff�Q��>�R�A��C�R                                    Bxda�v  �          @}p�?�\�2�\���C�/\?�\���2�\�3=qC���                                    Bxda�  �          @{�?޸R�)����
�\)C��\?޸R� ���8Q��=�C�w
                                    Bxda��  �          @z=q?�  �;������  C�n?�  ����#33�"�
C�%                                    Bxda�h  �          @{�?�33�-p��
=��C��f?�33���,���.=qC��
                                    Bxda�  �          @~�R@
=�   ����
C��@
=����3�
�3\)C�&f                                    Bxda�  �          @|��@=q�G��ff��\C�\)@=q��Q��%�%ffC�|)                                    Bxda�Z  �          @~{@   �z�����=qC��{@   ��  �!���HC�~�                                    Bxdb   �          @\)@(���33��33���C�xR@(�ÿ�G�������C�                                      Bxdb�  T          @�  @3�
�������(�C�xR@3�
������z�C�1�                                    Bxdb L  T          @���@'
=���G���{C�ff@'
=��(��!G���RC�U�                                    Bxdb.�  �          @}p�@,��������\)C���@,�Ϳ�=q�Q��
=C��\                                    Bxdb=�  �          @���@333� �׿���  C�\@333���   ����C��3                                    BxdbL>  �          @���@,����\��(���\)C�޸@,�Ϳ�p���R���C���                                    BxdbZ�  �          @���@'��p����ծC�ff@'���
=�
=��HC��=                                    Bxdbi�  �          @�G�@.{�Q�����ffC�c�@.{��{���C���                                    Bxdbx0  �          @���@6ff��
��33���
C�w
@6ff��=q�����C��=                                    Bxdb��  �          @�G�@4z��{����G�C��)@4z��Q���C���                                    Bxdb�|  �          @���@333��R��{��G�C���@333��Q��
=�  C�o\                                    Bxdb�"  �          @�=q@@  ��(����مC�XR@@  ��Q����	��C�&f                                    Bxdb��  T          @��\@K���G�����p�C���@K���  �(��C�H�                                    Bxdb�n  T          @���@'
=� �׿�ff��{C��@'
=��(��Q��=qC�q�                                    Bxdb�  �          @�G�@<���Q�У����C�3@<�Ϳ�33��� G�C�ff                                    Bxdb޺  �          @}p�@h�ÿ�R��\)��
=C�&f@h�þ�  ��p����C�H                                    Bxdb�`  �          @|��@o\)�.{��G����\C���@o\)>#�
��G����R@=q                                    Bxdb�  �          @\)@qG���Q쿥�����C�XR@qG�>�  ���\��z�@n{                                    Bxdc
�  �          @�  @w
=��{��  �hQ�C�s3@w
=���
����uG�C�o\                                    BxdcR  �          @�Q�@z�H�.{�O\)�;
=C��q@z�H=#�
�Tz��?
=?��                                    Bxdc'�  �          @�Q�@{�=�Q�J=q�5?���@{�>�z�=p��*{@�Q�                                    Bxdc6�  �          @���@}p�>aG��W
=�?�@L(�@}p�>�(��B�\�,  @�ff                                    BxdcED  �          @��@~�R=�\)�Q��:=q?��@~�R>�\)�E��/�@��                                    BxdcS�  �          @��\@\)�.{�L���4Q�C��H@\)=#�
�O\)�8z�?��                                    Bxdcb�  �          @��\@��׾8Q��R��C���@��׼#�
�#�
�p�C��=                                    Bxdcq6  �          @�G�@~�R���
�0���p�C�e@~�R=��Ϳ0�����?�z�                                    Bxdc�  �          @���@~{    �0���p�=u@~{>8Q�(����@(��                                    Bxdc��  �          @�G�@}p��\)�:�H�'�C��)@}p�=u�@  �*=q?O\)                                    Bxdc�(  �          @~{@z=q�W
=�.{���C�y�@z=q���
�5�$(�C��3                                    Bxdc��  �          @~�R@z�H����!G��p�C��@z�H��Q�.{�Q�C�Z�                                    Bxdc�t  �          @���@~{���
�����=qC��=@~{�.{�(��Q�C�Ǯ                                    Bxdc�  �          @���@~�R��������=qC���@~�R����
=��HC��                                    Bxdc��  �          @���@}p���p������C�U�@}p��W
=�#�
�\)C��H                                    Bxdc�f  �          @~�R@z=q��녿
=�Q�C��@z=q�u�.{���C�AH                                    Bxdc�  �          @�Q�@{���׿&ff���C���@{���\)�@  �,  C��R                                    Bxdd�  �          @�Q�@z�H���+���C���@z�H��=q�E��0��C�                                    BxddX  T          @���@{��&ff��ff��C�J=@{���\�(��\)C�P�                                    Bxdd �  �          @�  @y���O\)���
���
C�  @y���333����\C���                                    Bxdd/�  �          @}p�@vff�\(����
���C���@vff�@  �
=q��Q�C���                                    Bxdd>J  �          @~{@u�fff�\���C�h�@u�B�\�(����C�^�                                    BxddL�  �          @~{@w
=�E����H����C�\)@w
=�(��.{���C��H                                    Bxdd[�  �          @�  @y���5���H����C���@y����Ϳ(�����C��R                                    Bxddj<  �          @���@|(��0�׾�
=����C��@|(���Ϳ
=��C�                                    Bxddx�  �          @�  @z�H�!G������G�C�o\@z�H���!G��  C��f                                    Bxdd��  �          @~�R@x�ÿ(��\)� ��C���@x�þ�G��333�"�\C��=                                    Bxdd�.  �          @\)@y���&ff�
=q��Q�C�>�@y�����333� ��C�z�                                    Bxdd��  �          @���@x�ÿTz�\)� ��C���@x�ÿ&ff�E��0Q�C�J=                                    Bxdd�z  �          @�G�@y���\(��\)���C��@y���+��E��1�C�                                      Bxdd�   �          @���@y���^�R�
=q��p�C��3@y���0�׿B�\�-�C���                                    Bxdd��  �          @���@xQ�h�ÿ�����C�g�@xQ�:�H�B�\�/\)C���                                    Bxdd�l  �          @���@w���G����H��  C���@w��W
=�@  �+�
C��f                                    Bxdd�  
�          @�G�@z=q�\(���\��G�C��f@z=q�0�׿:�H�&�HC��                                    Bxdd��  �          @��@x�ÿ��
����ָRC���@x�ÿ\(��=p��(Q�C��f                                    Bxde^  �          @�=q@x�ÿ��׾\��p�C��@x�ÿz�H�.{��RC���                                    Bxde  �          @���@u����{���
C��f@u���
�&ff���C��                                     Bxde(�  �          @�=q@z=q����\��(�C�k�@z=q�h�ÿ(���C�q�                                    Bxde7P  �          @�Q�@u��ff��(���{C�e@u�aG��5�"�\C���                                    BxdeE�  �          @~�R@tzῃ�
��G���z�C�xR@tz�\(��5�%G�C���                                    BxdeT�  �          @~�R@u��}p���ff��
=C�� @u��Q녿5�$  C���                                    BxdecB  �          @~{@vff�Y���   ���C���@vff�.{�8Q��&{C��q                                    Bxdeq�  �          @~�R@vff�aG����޸RC��\@vff�5�5�$��C��=                                    Bxde��  �          @~�R@s33�z�H��R�z�C�Ǯ@s33�B�\�aG��K�C�T{                                    Bxde�4  T          @���@w��n{�#�
�
=C�B�@w��5�aG��I�C���                                    Bxde��  �          @�G�@|(��:�H�����HC�� @|(���Ϳ333� (�C�                                      Bxde��  �          @�=q@~�R�(��
=�G�C�� @~�R��
=�=p��'�C��q                                    Bxde�&  �          @��\@{��B�\�:�H�%�C���@{����k��O�
C�=q                                    Bxde��  �          @�=q@w��h�ÿ\(��C�C�` @w��!G�����w�C�ff                                    Bxde�r  �          @�G�@y���5�G��1��C�� @y����ff�s33�Y�C���                                    Bxde�  �          @���@xQ�:�H�\(��E�C���@xQ��ff����mp�C��3                                    Bxde��  �          @�G�@vff�=p��z�H�`  C��
@vff��
=��33��{C��q                                    Bxdfd  �          @���@s33�O\)���
�n�\C��R@s33����(����
C�h�                                    Bxdf
  �          @\)@n{�k���{���C�f@n{�\)��=q����C���                                    Bxdf!�  �          @~�R@l(��}p���{���C�u�@l(��!G���������C�(�                                    Bxdf0V  "          @~�R@j�H��G���33����C�H�@j�H�!G������ffC�R                                    Bxdf>�  �          @���@l(������ff��G�C�*=@l(��(��������C�G�                                    BxdfM�  T          @~�R@aG�������\����C��q@aG��h�ÿ�����=qC��f                                    Bxdf\H  �          @}p�@aG�����������
C��\@aG��n{��ff���C���                                    Bxdfj�  
�          @~�R@`�׿�녿�  ��
=C�'�@`�׿xQ��{����C�N                                    Bxdfy�  �          @~�R@^�R��\)������C�>�@^�R�k���Q����HC��                                     Bxdf�:  �          @�  @^{���׿�z�����C�!H@^{�h�ÿ�G�����C��                                    Bxdf��  �          @z�H@\(���G���z�����C��@\(��J=q��(���  C��\                                    Bxdf��  �          @���@`�׿�����H���C��@`�׿L�Ϳ��
�ң�C��3                                    Bxdf�,  �          @���@]p���33��p�����C���@]p��fff��=q�؏\C���                                    Bxdf��  T          @�G�@`  ��33�������C��@`  �h�ÿ�ff��p�C��q                                    Bxdf�x  �          @���@U��{������C�q@U��=q������  C��                                    Bxdf�  �          @���@XQ��
=��33��33C�Ǯ@XQ쿗
=���ٮC�W
                                    Bxdf��  �          @}p�@O\)�޸R��(���Q�C��@O\)��(�����ffC���                                    Bxdf�j  �          @|(�@QG���\������C��H@QG���ff����ׅC�,�                                    Bxdg  �          @|(�@Q녿��
���\����C��R@Q녿��ÿ�  ����C�
=                                    Bxdg�  T          @|(�@W
=�˅�������C�Q�@W
=��\)��(����
C���                                    Bxdg)\  T          @\)@\�Ϳ�녿�Q���33C�H@\�Ϳc�
����ՙ�C��q                                    Bxdg8  �          @\)@Y�������\��{C��@Y�����H���H��G�C�:�                                    BxdgF�  �          @}p�@Tz��Q쿪=q��33C���@TzῙ�����
��C��                                    BxdgUN  �          @{�@Q녿�\��  ��  C��
@Q녿����p���ffC�'�                                    Bxdgc�  �          @|(�@U��(��������C�e@U���\��z��ǮC���                                    Bxdgr�  �          @|��@Vff��\��\)��p�C��@Vff�����������C�3                                    Bxdg�@  �          @z=q@U�����z���C��=@U����R��{���C�˅                                    Bxdg��  �          @xQ�@U���\)��33���\C�f@U���Q������Q�C�.                                    Bxdg��  �          @z=q@Vff����\)��33C�� @Vff���R��=q����C��                                    Bxdg�2  �          @z=q@XQ���Ϳ�����C�S3@XQ쿕��=q��Q�C�n                                    Bxdg��  �          @|(�@Y���\�������C��R@Y��������H��C�~�                                    Bxdg�~  �          @\)@\�Ϳ�녿�z����RC�O\@\�Ϳ�����{��C�j=                                    Bxdg�$  �          @���@^{��\)��  ��z�C�z�@^{��z��Q���ffC���                                    Bxdg��  T          @�  @Z=q�޸R��33��C�|)@Z=q��ff�������C��
                                    Bxdg�p  �          @~�R@QG���p����H���C�3@QG���Q����
=C��                                    Bxdh  �          @���@S33�޸R��  ���RC��@S33��
=���H���
C�!H                                    Bxdh�  �          @���@U��  ��33��ffC�0�@U��p���\)���C��\                                    Bxdh"b  �          @�G�@U��ٙ���G����
C�z�@U���녿��H��p�C��=                                    Bxdh1  �          @��@P  ���R��������C�C�@P  ���H��
=���
C��\                                    Bxdh?�  �          @�G�@N{�����H���C�Q�@N{�У׿���\)C��\                                    BxdhNT  �          @���@S33��
=���R��33C���@S33��Q���
��ffC�5�                                    Bxdh\�  �          @���@W�����p����C��3@W����Ϳ޸R��\)C��                                    Bxdhk�  �          @\)@Vff������  C��\@Vff���Ϳ�
=��G�C�
=                                    BxdhzF  �          @���@X�ÿ��Ϳ�����C��@X�ÿ�33�����=qC���                                    Bxdh��  �          @���@Tz��Q쿕��p�C��@TzῺ�H��p���\)C�                                      Bxdh��  �          @���@N{� �׿�=q���C��@N{��p���33���C��f                                    Bxdh�8  �          @�G�@P����\��p���  C��@P�׿���������C�c�                                    Bxdh��  �          @���@QG���Q쿫���C���@QG���z�����(�C�Y�                                    BxdhÄ  �          @�  @P�׿�Q쿡G�����C���@P�׿�
=�����C�+�                                    Bxdh�*  �          @}p�@Mp���(���(���p�C�7
@Mp���(��������C���                                    Bxdh��  �          @}p�@J�H�   ��G���=qC�ٚ@J�H���R�����C�e                                    Bxdh�v  �          @z�H@N{��녿�����{C��3@N{��33�޸R���HC�B�                                    Bxdh�  �          @~{@S33��z῎{���C�H@S33��Q���ƣ�C�0�                                    Bxdi�  �          @���@^{��׿5�"ffC���@^{��ff���
����C���                                    Bxdih  �          @�G�@c�
����(����C���@c�
��(����H��\)C��                                     Bxdi*  �          @�G�@_\)�����������C�k�@_\)��zΐ33���RC�B�                                    Bxdi8�  �          @~�R@Z=q��\�k��Tz�C���@Z=q���ͿaG��MG�C��
                                    BxdiGZ  �          @�Q�@Y���
=�B�\�/\)C��@Y����
=�^�R�Hz�C�:�                                    BxdiV   �          @�  @Tz���;�����=qC�E@Tz��p���  �h��C��                                     Bxdid�  �          @��\@XQ쿦ff�������C�� @XQ�(�����\)C��H                                    BxdisL  �          @�=q@XQ쿯\)��ff��(�C���@XQ�333�	����G�C�%                                    Bxdi��  �          @~�R@W
=��p��\���
C��@W
=�fff����C�z�                                    Bxdi��  �          @y��@N�R�����\�s
=C��)@N�R��p��������
C���                                    Bxdi�>  �          @{�@QG���z῀  �l(�C��3@QG���(���������C��R                                    Bxdi��  �          @xQ�@Q녿�(����H��\)C��@Q녿h�ÿ�\)��z�C�G�                                    Bxdi��  �          @�33@^�R��ff��G���\)C���@^�R�u��Q���=qC�J=                                    Bxdi�0  �          @��
@b�\���
��
=��(�C�XR@b�\�xQ������{C�`                                     Bxdi��  �          @�z�@_\)�޸R�����33C�� @_\)��
=����ٙ�C��\                                    Bxdi�|  �          @�(�@Z�H��\��p����\C�G�@Z�H��
=���R����C��                                     Bxdi�"  �          @���@QG����ÿ�G����C�z�@QG����H����C��                                     Bxdj�  T          @y��@I����
=��G���33C�<)@I����녿����HC��                                    Bxdjn  �          @w
=@Dz��zΉ����C��)@Dz��=q��p���Q�C�XR                                    Bxdj#  �          @u@E��Q쿘Q����\C��)@E����\��{C���                                    Bxdj1�  �          @w
=@H�ÿ�(���������C���@H�ÿ��R����\)C�K�                                    Bxdj@`  �          @x��@G
=�녿������C�l�@G
=���
��G��֣�C��=                                    BxdjO  �          @w�@Dz���
������{C�3@Dz�Ǯ��G�����C��3                                    Bxdj]�  �          @s33@5��׿�=q��Q�C��q@5��G����
��z�C��                                    BxdjlR  �          @qG�@\)�+��c�
�[33C�t{@\)�{��  �ݙ�C�*=                                    Bxdjz�  �          @r�\@/\)����=q��G�C��
@/\)��{��=q����C���                                    Bxdj��  �          @u�@E����������G�C��
@E������p����HC�j=                                    Bxdj�D  �          @w�@;������
���RC��f@;���{������(�C��{                                    Bxdj��  �          @z�H@@  ��(���{��33C�\)@@  ��ff�(��{C�L�                                    Bxdj��  
�          @�  @A녿���G���Q�C��@A녿�����
��C�+�                                    Bxdj�6  �          @~{@C�
��ff��\�ԏ\C��
@C�
��=q��\���C�:�                                    Bxdj��  �          @z=q@Fff�\�����C���@Fff�J=q�  �=qC���                                    Bxdj�  �          @z=q@J�H����������C��3@J�H�(�����	(�C��f                                    Bxdj�(  �          @~�R@L�Ϳ��R���ܣ�C��=@L�ͿB�\�\)�	=qC�XR                                    Bxdj��  �          @x��@G
=��
=����Q�C��q@G
=�5�{��C��R                                    Bxdkt  �          @y��@J�H���׿���ڣ�C�<)@J�H�+��	���z�C�                                      Bxdk  �          @{�@7
=����
=��Q�C��q@7
=��33��
�  C��{                                    Bxdk*�  /          @z�H@8������33��
=C�%@8�ÿ�\)�G��z�C�N                                    Bxdk9f  y          @z=q@8������\)��p�C��@8�ÿ���\)���C�.                                    BxdkH  �          @}p�@K���{��Q��ʣ�C���@K��k�����C���                                    BxdkV�  
�          @z=q@U����33��p�C�Y�@U���������C���                                    BxdkeX  T          @{�@XQ쿓33�У���z�C��)@XQ�   ��
=��C��                                    Bxdks�  �          @z=q@N{��33�������C�"�@N{��
=�
�H�{C�O\                                    Bxdk��  �          @z=q@U���G���=q��G�C���@U���R����z�C��{                                    Bxdk�J  �          @}p�@\�Ϳ�zῬ����C�޸@\�ͿW
=��G��ӅC�"�                                    Bxdk��  �          @z�H@XQ쿽p������G�C�*=@XQ�h�ÿ�\��
=C�q�                                    Bxdk��  �          @{�@Vff���R��z���Q�C�f@Vff�c�
������Q�C���                                    Bxdk�<  �          @|(�@[�������G���G�C��@[��h�ÿ�Q���C��R                                    Bxdk��  �          @xQ�@\�Ϳ��Ϳ����ffC�J=@\�Ϳ\(�������
C�                                      Bxdkڈ  �          @vff@Z�H��{������ffC�/\@Z�H�\(������z�C��                                    Bxdk�.  �          @z=q@\(���녿�p�����C��@\(��\(�������HC��                                    Bxdk��  T          @|��@c33���
��33��G�C��@c33�J=q���
����C���                                    Bxdlz  T          @z�H@b�\��  ��{��{C�Ff@b�\�E����R��z�C��H                                    Bxdl   T          @y��@e���\)����}�C�H�@e��+������C���                                    Bxdl#�  �          @w�@c�
���ÿ�=q��C��f@c�
�(�������RC�(�                                    Bxdl2l  T          @xQ�@aG���\)��33���\C�'�@aG��!G���p���C���                                    BxdlA  �          @z�H@l(��c�
�z�H�g�C�33@l(�����p���
=C�N                                    BxdlO�  �          @~�R@q녿J=q�xQ��a��C�3@q녾Ǯ��Q���G�C��                                    Bxdl^^  �          @w
=@j=q�L�Ϳp���`��C�˅@j=q��녿����\C���                                    Bxdlm  �          @u@k��B�\�G��;�
C�1�@k���
=��  �r{C��)                                    Bxdl{�  �          @tz�@k��+��L���@z�C��q@k����ÿz�H�n�HC�n                                    Bxdl�P  �          @s33@l�Ϳ
=q�+��"=qC��q@l�;���Q��G33C��                                    Bxdl��  �          @z�H@p�׿녿u�c33C��
@p�׾8Q쿌������C���                                    Bxdl��  �          @�Q�@tz�E��n{�V�HC�L�@tz�\��33���C�.                                    Bxdl�B  �          @�G�@tz�fff�u�\��C�XR@tz���H��p�����C�]q                                    Bxdl��  �          @y��@h�ÿ�=q�h���Xz�C��@h�ÿ+����R����C���                                    Bxdlӎ  �          @\)@o\)���ÿY���D��C���@o\)�0�׿�
=��z�C���                                    Bxdl�4  �          @�Q�@tz�0�׿h���S33C��@tzᾙ��������(�C��R                                    Bxdl��  T          @�G�@vff�=p��u�\��C���@vff���ÿ���Q�C���                                    Bxdl��  �          @��H@|(��녿u�Y�C��@|(��.{�����y�C��H                                    Bxdm&  �          @��
@�  ��\�Y���?\)C�T{@�  �#�
�z�H�\��C��
                                    Bxdm�  �          @��@�녾��8Q��
=C��q@�녾8Q�W
=�;�C��
                                    Bxdm+r  �          @~�R@p  �u�fff�PQ�C��R@p  �\)��Q�����C���                                    Bxdm:  �          @}p�@r�\�aG��@  �.�HC�w
@r�\�
=q���\�o�C��R                                    BxdmH�  �          @|(�@i����33�fff�S�C�:�@i���=p����\��(�C�O\                                    BxdmWd  �          @xQ�@^{���Ϳ���
=C�ff@^{�Y�����R����C��                                    Bxdmf
  �          @xQ�@_\)���
��=q���C��=@_\)�J=q��p���p�C��f                                    Bxdmt�  �          @y��@S�
��(�������ffC�޸@S�
�
=q����{C�W
                                    Bxdm�V  �          @~�R@1��"�\��  ���RC��\@1녿���
=� z�C��                                    Bxdm��  �          @\)@+��%���33���C��@+���\)�G��(�C���                                    Bxdm��  
�          @}p�@Y�����R�������
C��q@Y���\)����G�C�Z�                                    Bxdm�H  �          @��@\(���z`\)��C�!H@\(����\��33�ޏ\C��q                                    Bxdm��  �          @~{@Z=q�У׿�Q���  C�4{@Z=q�����(���C�Z�                                    Bxdm̔  �          @���@b�\�˅����t��C���@b�\���ÿ�=q��p�C��)                                    Bxdm�:  �          @�G�@aG���33�����u�C�u�@aG���\)��{���RC�.                                    Bxdm��  �          @���@\(���G�����ffC�s3@\(���Q��  ��G�C�~�                                    Bxdm��  �          @�  @`�׿Ǯ�p���[
=C���@`�׿�����H��G�C�^�                                    Bxdn,  �          @���@qG���p��!G���C��{@qG��fff��ff�r=qC�H�                                    Bxdn�  �          @�Q�@p  ������{C���@p  �xQ쿁G��j�HC���                                    Bxdn$x  �          @\)@b�\��33��G��nffC�1�@b�\�h�ÿ�(���p�C���                                    Bxdn3  �          @\)@J�H��(���
=��G�C��)@J�H�s33�p����C��f                                    BxdnA�  �          @���@Q녿�ff��p����\C�� @Q녿�����
���
C���                                    BxdnPj  �          @�G�@XQ�\�Ǯ��ffC��@XQ�O\)�G���C�J=                                    Bxdn_  �          @�  @:=q�p���=q��
=C�aH@:=q��
=��
�C��H                                    Bxdnm�  �          @���@<(����\��ffC�Ф@<(��Ǯ�33�
p�C�                                      Bxdn|\  �          @���@H����ÿ�����(�C���@H�ÿ�G���
=���C�,�                                    Bxdn�  �          @�=q@c�
��{�(��	G�C�<)@c�
��p����
��z�C���                                    Bxdn��  �          @��\@hQ����������HC�� @hQ��  ����qG�C��=                                    Bxdn�N  �          @�G�@g���G�����p�C��@g���=q�J=q�5�C�<)                                    Bxdn��  �          @~{@P���  ��  �c33C���@P�׿�p��������RC�`                                     BxdnŚ  �          @vff@(��3�
�������C���@(��33�����C�o\                                    Bxdn�@  �          @s33@33�0  �������C�W
@33��z��"�\�(�C�~�                                    Bxdn��  �          @r�\@��1녿�����C��
@����R�=q���C�k�                                    Bxdn�  �          @tz�?�z��0�׿���=qC�W
?�z���/\)�7�HC��                                    Bxdo 2  �          @u?�\)�?\)��G����
C��?�\)�
=q�#�
�'�C�xR                                    Bxdo�  �          @s�
?�Q��H�ÿ����G�C�0�?�Q��
=�{�!Q�C��R                                    Bxdo~  �          @vff?�\)�H�ÿ\��{C��f?�\)��\�(Q��,z�C���                                    Bxdo,$  �          @tz�?�  �A녿��H���\C�<)?�  �G��?\)�N�C��                                    Bxdo:�  �          @u�?����<������\C�)?�������G
=�Y\)C�Y�                                    BxdoIp  �          @r�\?��
�B�\� ��� (�C�ff?��
�   �B�\�U��C��=                                    BxdoX  �          @u?��
�:�H�G����C�� ?��
����N�R�e��C��                                    Bxdof�  �          @u?Y���9���
=�(�C�(�?Y����p��Tz��op�C�3                                    Bxdoub  �          @w
=?˅�3�
�ff���C���?˅��  �B�\�Pp�C�
                                    Bxdo�  	m          @vff?�=q�%�����  C���?�=q���Mp��b
=C���                                    Bxdo��  �          @u�?ٙ��%������C��R?ٙ���(��E�X
=C���                                    Bxdo�T  �          @vff?��'
=��R��RC�K�?���G��E��W\)C���                                    Bxdo��  �          @w�?\�2�\��(���=qC�K�?\��\�:�H�M(�C�J=                                   Bxdo��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxdo�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxdo��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxdo�             @xQ�?��*=q���G�C�\?��\�J=q�Y�
C��
                                    Bxdo�8  T          @vff?�(��"�\��
��C�3?�(���33�H���Z�
C�t{                                    Bxdp�  �          @w
=?�p��33����33C�T{?�p���33�G��W�C���                                    Bxdp�  	`          @u@���H���C��q@녿����.�R�5=qC���                                    Bxdp%*  T          @w�@����33� �C��R@���33�7
=�?�C���                                    Bxdp3�  T          @z=q@(�����������C�� @(������*=q�,��C���                                    BxdpBv  T          @w
=?��R�(�ÿ�����Q�C�~�?��R�У��6ff�?�HC�Z�                                    BxdpQ  �          @u?�
=�;������C�� ?�
=�����5�?G�C�U�                                    Bxdp_�  T          @tz�?�Q��<(����H��\)C��?�Q��33�>�R�M�
C��3                                    Bxdpnh  �          @u@�
� �׿�Q���z�C�E@�
��{�#�
�)=qC��                                    Bxdp}  "          @s33@4z��녿�33��G�C��@4zῇ����G�C���                                    Bxdp��  �          @r�\@Q��Q��
=���C�޸@Q쿱��.�R�<G�C�ff                                    Bxdp�Z  �          @u�?���/\)�����{C�W
?�녿�p��5��?ffC��f                                    Bxdp�   
�          @s�
?Y���Dz���
��Q�C��R?Y���z��8Q��P��C�q                                    Bxdp��  �          @x��?=p��E�(��
p�C���?=p����P���f�HC��                                     Bxdp�L  �          @w�>��I�����	�RC�,�>���p��Q��hC���                                    Bxdp��  T          @vff>����I�����
ffC�>�����p��Q��jz�C�^�                                    Bxdp�  
�          @vff>B�\�H���Q���C��{>B�\���R�N�R�hC��\                                    Bxdp�>  T          @vff�.{�G
=�	���G�C�⏿.{�����O\)�e��Cz�                                    Bxdq �  �          @u�8Q��E�Q��p�C�h��8Q��Q��Mp��d�RCy�\                                    Bxdq�  T          @z=q���\�:�H�\)�z�CvxR���\�޸R�P  �a�Ci��                                    Bxdq0  T          @~{��=q�+��&ff�$
=Cs����=q��\)�^�R�t��Ca�f                                    Bxdq,�  G          @|(���G��
=q�7
=�=ffCk)��G��J=q�`  #�CO��                                    Bxdq;|  /          @z=q���
�
=q�@  �H\)Cok����
�:�H�hQ��fCQ�H                                    BxdqJ"  y          @u���33�33�5��>��Cs}q��33�k��a���CZ�)                                    BxdqX�  
g          @s�
���R�Q��6ff�GQ�C�@ ���R�}p��e��3C|^�                                    Bxdqgn  �          @r�\?��H��R�G��  C�C�?��H�����Fff�\��C�1�                                    Bxdqv  
�          @s33?�(��"�\��R�33C��?�(������E��Y�RC���                                    Bxdq��  
�          @tz�?����\)�  �(�C��?��ÿ�=q�E��W
=C���                                    Bxdq�`  T          @q�?�
=�$z��ff���C��H?�
=��{�Mp��i�C�4{                                    Bxdq�  T          @tz�?�z�����p���C��R?�zῦff�A��R��C���                                    Bxdq��  �          @u�@
=q�(����Q�C�L�@
=q���
�>�R�M(�C�/\                                    Bxdq�R  
�          @w
=@\)�)����Q���p�C�  @\)�ٙ��(���,�RC�k�                                    Bxdq��  �          @s33@33�-p���\)��C��
@33���
�&ff�/{C���                                    Bxdqܞ  
�          @s�
@���9����\)����C��H@���
=q��R���C��=                                    Bxdq�D  "          @tz�@�\�9���h���\Q�C�(�@�\�  ��\���C��H                                    Bxdq��  
�          @u�@ff�2�\���
��C��@ff����#�
�(��C�3                                    Bxdr�  T          @q�@33�!녿�\)��C���@33��  �0���=�RC��q                                    Bxdr6  �          @r�\@  �(Q���
���HC�B�@  �޸R�   �%�C�                                      Bxdr%�  �          @q�@�\����(���(�C���@�\��p��%�.�C��                                    Bxdr4�  �          @s�
@�
�ff��33���C�G�@�
��=q�-p��5�
C��                                    BxdrC(  �          @s�
@���\)��{��C��3@�Ϳ�(��/\)�7��C��                                    BxdrQ�  �          @q�@���   �������C�<)@�ÿ��R�/\)�9C���                                    Bxdr`t  �          @p  @(���p��   ��C�t{@(��p���)���5�RC�j=                                    Bxdro  �          @qG�@(���
���H��(�C��\@(����
�(���3�
C���                                    Bxdr}�  T          @mp�@$z������\����C��f@$z���������C�L�                                    Bxdr�f  T          @p  @��'���p���=qC��@녿У��+��6��C���                                    Bxdr�  
�          @k�?�=q�3�
��p���  C��H?�=q���1G��DQ�C���                                    Bxdr��  �          @mp�@�)�����
��
=C��@��  � ���*�HC��                                    Bxdr�X  T          @j=q@�
�{��  ��ffC���@�
���R�(���8G�C�R                                    Bxdr��  �          @g�@	���p��\��Q�C��3@	���������)��C��=                                    Bxdrդ  
�          @l��?�
=�)�����H��=qC��?�
=���+��9��C��R                                    Bxdr�J  
�          @l(�?�\�,�Ϳ��
��p�C���?�\���1G��B�C�Z�                                    Bxdr��  T          @j�H?����1녿����
=C���?��Ϳ޸R�333�Gp�C�C�                                    Bxds�  �          @l��?�\)�0  ��{��C�E?�\)��
=�7
=�J�C��
                                    Bxds<  �          @mp�?����5���
  C��)?��Ϳ�z��Fff�eQ�C��                                    Bxds�  
�          @k�?���333��Q�� z�C�e?����Q��<���XC��                                    Bxds-�  �          @j=q?�z��7�����{C���?�z��ff�9���Vp�C�e                                    Bxds<.  
�          @n{?�\�2�\�
=���C�9�?�\��  �U��C�xR                                    BxdsJ�  �          @p��?��8���z��G�C�%?������U�|�
C�\                                    BxdsYz  "          @l��?���'
=��=q��{C�1�?���Ǯ�1��E�HC�h�                                    Bxdsh   �          @l(�@G���ÿ�z����C���@G���Q��!��,�C�˅                                    Bxdsv�  �          @l(�?��p���(��{C�H?���{�5�IffC�W
                                    Bxds�l  
�          @mp�?�{���(��p�C�H�?�{��33�>�R�W\)C�&f                                    Bxds�  
Z          @l��?���p�� ���Q�C���?�׿���8Q��M\)C�B�                                    Bxds��  
�          @qG�?�(��5��  �ޏ\C���?�(�����3�
�B  C��                                    Bxds�^  �          @q�?���%�� ��� C�?�׿����;��L
=C�7
                                    Bxds�  
�          @q�?��%���p����\C�]q?����H�9���I  C�h�                                    BxdsΪ  �          @p��?��(Q�����RC��?�����6ff�DffC��H                                    Bxds�P  �          @qG�@z���R��z����C���@z῱��3�
�AG�C��                                    Bxds��  T          @qG�@�
�����C�q�@�
��(��:�H�J��C���                                    Bxds��  �          @q�@33�p���p���ffC��@33����7
=�E�\C�n                                    Bxdt	B  T          @o\)@\)��
��z����C�
@\)���R�.�R�;G�C��                                    Bxdt�  T          @p��@�R�p���
�=qC��)@�R��=q�4z��B��C��                                    Bxdt&�  T          @qG�@
�H�G��ff��C��@
�H��\)�8Q��F�C�j=                                    Bxdt54  �          @r�\@���G���ff����C��H@�Ϳ��R�'��.G�C��\                                    BxdtC�  �          @s�
@Q��(��ٙ���C�%@Q쿸Q��&ff�,
=C�`                                     BxdtR�  �          @s�
@Q����   ��z�C�� @Q쿊=q�0���:=qC���                                    Bxdta&  	�          @q�@ff����   ��p�C�n@ff����1G��;�RC��=                                    Bxdto�  �          @tz�@9����\��(���ffC�p�@9����
=�{�ffC��                                    Bxdt~r  �          @s33@8�ÿ�z�Ǯ��z�C�K�@8�ÿ��\�\)��C�H�                                    Bxdt�  T          @s�
@1G�� �׿�z���\)C��)@1G������Q��  C�n                                    Bxdt��  �          @tz�@<�Ϳ�G�����G�C��{@<�ͿW
=�G��Q�C��                                    Bxdt�d  �          @u@8Q��ff���
�ܣ�C���@8Q�Tz�����33C���                                    Bxdt�
  �          @tz�@,�Ϳ�����\)��z�C��@,�Ϳk��"�\�&�RC��                                     Bxdtǰ  �          @s�
@-p����
�   ��(�C�\)@-p��333�%��+  C���                                    Bxdt�V  �          @r�\@?\)��������C�J=@?\)��ff��\��C��
                                    Bxdt��  �          @s�
@E����ÿ�Q����C�o\@E������p���C��                                    Bxdt�  �          @s33@<(��u�
�H��C��@<(�=������z�@                                       BxduH  �          @s33@)����ff�{��C��)@)�������+��3�HC��                                     Bxdu�  �          @q�@$z��������C��f@$zᾸQ��.�R�9��C���                                    Bxdu�  �          @vff@33�ff��H��C�,�@33�L���Fff�Zz�C�]q                                    Bxdu.:  �          @s�
@7��p����\�z�C��f@7�>8Q��{�"�@b�\                                    Bxdu<�  �          @tz�@<(������33��
C��H@<(���=q�����C�h�                                    BxduK�  �          @tz�@)����  �33�(�C�H�@)�����R�.�R�6Q�C���                                    BxduZ,  �          @s33@ �׿�\)�/\)�;{C��
@ �׾aG��K��f(�C��\                                    Bxduh�  �          @u@8Q��ff���R��=qC��@8Q���H�{� C�+�                                    Bxduwx  �          @w
=@  ����H��33C���@  ���H�333�>��C�޸                                    Bxdu�  T          @w�@���
��
=C��q@��fff�2�\�<z�C��{                                    Bxdu��  �          @x��@A녿��R�����C��@A녾������p�C��{                                    Bxdu�j  �          @w�@R�\��(������ď\C��)@R�\��Q��p���
=C��R                                    Bxdu�  �          @{�@Q녿��Ϳ�
=��{C��@Q녾�ff�ff��C�3                                    Bxdu��  �          @z�H@Tzῥ���33���HC�Y�@Tz�����
���RC�xR                                    Bxdu�\  �          @z=q@K�������  ��33C��{@K���\�{�
z�C�y�                                    Bxdu�  �          @y��@@�׿�{������C���@@�׿
=����G�C�}q                                    Bxdu�  �          @y��@>{��녿�z�����C���@>{���������C�J=                                    Bxdu�N  �          @|(�@N{���׿���ظRC�` @N{��
=�{�	��C�C�                                    Bxdv	�  �          @|��@H�ÿ����p���{C��
@H�þ�  �ff��
C��{                                    Bxdv�  �          @~{@QG���z����(�C�8R@QG��#�
�\)�	��C���                                    Bxdv'@  �          @z=q@R�\���������{C�Ff@R�\�������C�#�                                    Bxdv5�  �          @z�H@P  ��z��{��z�C�&f@P  �B�\�(��z�C�N                                    BxdvD�  �          @z�H@U������\)��\)C�h�@U��������p�C�xR                                    BxdvS2  �          @|��@Vff���׿Ǯ��C��R@Vff��� ����33C���                                    Bxdva�  �          @{�@U����
��\)��G�C�}q@U�����������C���                                    Bxdvp~  �          @w�@Q녿�{���H��z�C���@Q녾W
=�����C�/\                                    Bxdv$  �          @w�@L(��\��ff��p�C�>�@L(��&ff�����C�AH                                    Bxdv��  �          @w
=@K���{��p����C���@K��B�\�33� 33C�K�                                    Bxdv�p  �          @xQ�@L(���=q��ff��{C���@L(��333�ff�
=C�˅                                    Bxdv�  �          @xQ�@Fff��  ��ff��  C�G�@Fff�Y�����ffC�W
                                    Bxdv��  �          @w
=@J�H�ٙ���z���
=C���@J�H�^�R����p�C�^�                                    Bxdv�b  �          @w
=@L(���Q쿮{��33C��@L(��aG���p���p�C�J=                                    Bxdv�  �          @x��@A녿�
=������C�W
@A녽��ff��
C��R                                    Bxdv�  �          @y��@H�ÿ�녿��ޣ�C�
@H�þ����  �ffC�Z�                                    Bxdv�T  �          @{�@U��z��(���p�C�p�@U�u��
����C���                                    Bxdw�  �          @}p�@U���33�����G�C�w
@U��L�������C�C�                                    Bxdw�  �          @\)@U���33��\)���
C�o\@U��.{�(��\)C��3                                    Bxdw F  �          @}p�@S33��Q��=q��(�C�q@S33�W
=�
�H�  C�&f                                    Bxdw.�  �          @x��@J=q��Q��
=��(�C��=@J=q�.{����Q�C�u�                                    Bxdw=�  �          @u@J�H����=q��C��H@J�H�B�\�
=q�	  C�=q                                    BxdwL8  �          @vff@J�H����������Q�C��@J�H�aG��
�H�	\)C��)                                    BxdwZ�  �          @w�@Fff��������p�C�T{@Fff���
��\�=qC�3                                    Bxdwi�  �          @vff@G����������
=C�� @G�������R��
C�J=                                    Bxdwx*  �          @vff@C�
��\)�����G�C��3@C�
��{��
�p�C�Ф                                    Bxdw��  
�          @vff@8�ÿ�Q����C�� @8�þ��R�!G��#�C��                                    Bxdw�v  T          @w
=@-p���33�
�H�	��C�Z�@-p���ff�,(��1��C�=q                                    Bxdw�  
Z          @x��@7
=�Ǯ�ff�G�C��
@7
=����%��'33C��                                    Bxdw��  
�          @y��@=p�����Q��Q�C�o\@=p��u�!��!C���                                    Bxdw�h  T          @xQ�@=p���=q�����
C��@=p���\����C�3                                    Bxdw�  �          @x��@G
=��p��������C�w
@G
=�J=q�{�
��C��q                                    Bxdw޴  �          @y��@C�
���
������C�� @C�
�Q����(�C���                                    Bxdw�Z  T          @w
=@1G��녿�G���z�C��)@1G��}p��   �!ffC�4{                                    Bxdw�   �          @w
=@.�R�   �������
C�޸@.�R�n{�"�\�%z�C���                                    Bxdx
�  
�          @xQ�@1G���\���
�ڸRC��3@1G��z�H�!G��"�\C�Ff                                    BxdxL  
�          @u�@-p��z�޸R��(�C�J=@-p����\�   �#\)C��H                                    Bxdx'�  T          @u@)���Q����݅C�� @)����ff�$z��(
=C�,�                                    Bxdx6�  �          @u�@.{������{����C�7
@.{�Y���#�
�'p�C�H�                                    BxdxE>  �          @x��@&ff�
=��(���\)C�l�@&ff�s33�.�R�2G�C��=                                    BxdxS�  �          @xQ�@(��� ���   ��
=C�Z�@(�ÿW
=�-p��1p�C�#�                                    Bxdxb�  �          @x��@0�׿�33��(���C��H@0�׿B�\�(Q��*(�C�P�                                    Bxdxq0  �          @x��@Fff��G��������C�  @Fff�   ��\�{C�b�                                    Bxdx�  �          @z=q@B�\�\����C���@B�\��ff�=q�p�C��=                                    Bxdx�|  
�          @z=q@B�\��ff��z���Q�C��@B�\����=q�z�C��{                                    Bxdx�"  �          @z=q@]p����R��G��o33C�^�@]p��W
=��=q���C�%                                    Bxdx��  
�          @z=q@c33��{�c�
�S33C��=@c33�G�������C���                                    Bxdx�n  "          @xQ�@fff��G��.{� ��C�Y�@fff�L�Ϳ�Q����RC��                                     Bxdx�  �          @|��@g
=��{�L���;\)C���@g
=�Q녿����G�C��3                                    Bxdx׺  �          @{�@^�R��33����Q�C��@^�R�0�׿�Q���p�C�ff                                    Bxdx�`  T          @|(�@a녿�\)�����{�C�g�@a녿5��=q���C�T{                                    Bxdx�  �          @z�H@`�׿�33��G��o�
C�R@`�׿E���ff��
=C�ٚ                                    Bxdy�  T          @z=q@fff���
�W
=�FffC�33@fff�=p�������RC�5�                                    BxdyR  "          @x��@a녿�{�h���Yp�C�t{@a녿E��������\C��{                                    Bxdy �  "          @x��@c�
��=q�\(��K�C��H@c�
�E�������ffC��                                    Bxdy/�  T          @z�H@e���=q�fff�S\)C���@e��@  ����33C�
                                    Bxdy>D  �          @|(�@b�\��  ������C�C�@b�\��ͿУ���
=C���                                    BxdyL�  T          @}p�@dzῦff�������RC��q@dz�!G���=q��Q�C��)                                    Bxdy[�  T          @|��@c33��Q쿠  ��C��{@c33�����Ǚ�C�Ff                                    Bxdyj6  
�          @~{@c33���\��G���{C�.@c33�����H���
C�Ф                                    Bxdyx�  "          @}p�@e���
=��������C���@e���׿�{��=qC�=q                                    Bxdy��  �          @y��@`  ��(���Q���Q�C�g�@`  ��\��\)��33C��{                                    Bxdy�(  �          @z=q@`�׿�p���Q���{C�W
@`�׿��У��ŅC��                                    Bxdy��  �          @�Q�@j=q��{������\)C���@j=q��
=��=q��=qC�                                    Bxdy�t  �          @���@n�R�h�ÿ�����C��@n�R��  ���H��z�C��                                    Bxdy�  �          @�Q�@j�H��33��\)��=qC�S3@j�H�����
���
C�G�                                    Bxdy��  �          @\)@^�R��(������
=C���@^�R�.{��=q�ڣ�C�s3                                    Bxdy�f  �          @~�R@^�R��
=������
C��=@^�R�&ff�����ٮC���                                    Bxdy�  �          @|��@fff��\)������\C�]q@fff��ff���
��\)C�xR                                    Bxdy��  "          @xQ�@c�
�����\)��{C�ٚ@c�
�Ǯ��p���G�C���                                    BxdzX  "          @xQ�@a녿����������C��
@a녿
=q���
��C���                                    Bxdz�  �          @w�@Tzῑ녿����Q�C���@Tz�k���p���ffC�                                    Bxdz(�  	�          @tz�@Mp���
=��Q����C��@Mp��u��\��C��f                                    Bxdz7J  �          @o\)@H�ÿ�����\)��(�C���@H�þ�z��   �z�C�c�                                    BxdzE�  T          @h��@=p����\��Q���
=C�n@=p����R�����C���                                    BxdzT�  T          @c33@;���{��
=��p�C��=@;��8Q��   �	(�C�G�                                    Bxdzc<  "          @e@Fff��{��33��=qC�%@Fff���R��G��뙚C�                                      Bxdzq�  
�          @c33@Dz�c�
���
�̏\C��H@Dz�u��\���
C�q�                                    Bxdz��  "          @hQ�@Fff�\(���
=�ܸRC�@ @Fff=#�
�������?L��                                    Bxdz�.  �          @b�\@>{�}p���z���
=C���@>{��Q��
=�
=C�+�                                    Bxdz��  "          @g
=@@�׿�ff���ݙ�C�W
@@�׾���(��  C��H                                    Bxdz�z  "          @l��@1G��}p���R���C�.@1G�>B�\���$@}p�                                    Bxdz�   �          @j=q@(Q쿓33�����RC�:�@(Q�=�\)�!��/z�?��H                                    Bxdz��  �          @e�@%�������p�C�3@%=#�
��R�/{?p��                                    Bxdz�l  
(          @hQ�@1G��k��
=q��C��\@1G�>aG���� 
=@�                                      Bxdz�  �          @dz�@,�Ϳ�p���Q��p�C��H@,�;#�
��\� �HC�S3                                    Bxdz��  
�          @l��@-p���\)����z�C��@-p���\�p��'
=C���                                    Bxd{^  
�          @n�R@+���\)��\�\)C�o\@+���ff�#�
�-ffC�@                                     Bxd{  
�          @n�R@$z΅{�\)��C��@$z���'
=�5�C��3                                    Bxd{!�  T          @n�R@5�&ff�z����C��R@5?
=q���A+�
                                    Bxd{0P            @j�H@,�ͿTz����33C�s3@,��>\����'�\A                                    Bxd{>�  
�          @qG�@{��  �"�\�)�
C���@{=�G��5��C=q@#�
                                    Bxd{M�  
�          @s�
@!녿�\)�'
=�-(�C�R@!�>�=q�4z��?�H@�\)                                    Bxd{\B  
�          @vff@�R�����0���833C�~�@�R=L���G��Y��?��H                                    Bxd{j�  �          @}p�@녿�\)�;��?G�C�y�@�>L���N�R�Z�@�Q�                                    Bxd{y�  "          @�  @�����C33�FC�c�@>��H�L���U{A=��                                    Bxd{�4  
�          @w
=@�ÿ�G��C33�PffC�^�@��?��J=q�[�An{                                    Bxd{��  �          @u�@p���ff�<���I(�C�J=@p�>��H�E�W{AG�                                    Bxd{��  
�          @q�@G���z��1G��=�\C�o\@G�>��R�?\)�R33@�ff                                    Bxd{�&  �          @qG�@+���Q��ff��
C��@+�=��
�(Q��1�
?��
                                    Bxd{��  �          @j=q@4z῕��
=�33C��\@4z��G������RC���                                    Bxd{�r  
�          @]p�@P�׿c�
?B�\AIC�T{@P�׿�33>k�@xQ�C�N                                    Bxd{�  T          @^{@S33�aG�?��A�\C��=@S33���=���?�
=C�q                                    Bxd{�  "          @`  @U�c�
?��A��C���@U���=u?s33C�U�                                    Bxd{�d  
Z          @c33@XQ�xQ�?�\A  C��@XQ쿋����
���C�                                    Bxd|
  "          @a�@U���\>�@�G�C���@U��{���
���C�Ǯ                                    Bxd|�  �          @Z=q@O\)�xQ�>��@ڏ\C��{@O\)�����G�����C�
                                    Bxd|)V  
�          @R�\@G��^�R?
=qA�\C�9�@G����\=L��?p��C��{                                    Bxd|7�  "          @Vff@K��z�H>�33@�C�c�@K����
�.{�:=qC�                                    Bxd|F�  �          @]p�@R�\��ff=�\)?�z�C�.@R�\�s33��G����HC��\                                    Bxd|UH  T          @Z=q@S33�^�R��Q쿺�HC�� @S33�8Q���H��C��R                                    Bxd|c�  �          @Y��@N�R��G�>u@\)C�W
@N�R��  ��\)��C�l�                                    Bxd|r�  .          @Y��@QG��^�R>���@��HC��H@QG��h�þ#�
�(��C�4{                                    Bxd|�:  
�          @[�@R�\�Tz�>.{@<(�C��H@R�\�O\)��  ��ffC�
=                                    Bxd|��  z          @_\)@O\)����>�=q@���C��\@O\)��Q쾮{���C��                                    Bxd|��  
          @\��@N�R���>W
=@b�\C�N@N�R���;�Q���G�C��)                                    Bxd|�,  �          @[�@N{����>���@���C�P�@N{��녾�\)���RC�G�                                    Bxd|��  �          @W�@G����H>�{@�\)C�o\@G���(���z���(�C�U�                                    Bxd|�x  "          @Vff@H�ÿ�ff>�@��HC���@H�ÿ�녽�G���C�                                      Bxd|�  "          @Vff@E��
=?   A	p�C���@E���
�����C�                                    Bxd|��  �          @U�@E����>�A�C��@E��p����ffC�+�                                    Bxd|�j  �          @Y��@L(���{>�
=@�(�C�]q@L(���
=�.{�333C��R                                    Bxd}  
�          @X��@K���{>��R@��C�\)@K����׾����C�AH                                    Bxd}�  �          @X��@P�׿c�
=�G�?�33C�ff@P�׿Tzᾨ�����\C���                                    Bxd}"\  �          @Z�H@R�\�k�        C�1�@R�\�O\)��G���ffC�                                      Bxd}1  "          @[�@S33�n{��\)��
=C�q@S33�G���\�	�C�T{                                    Bxd}?�  "          @XQ�@O\)�s33������C���@O\)�G��\)�Q�C�C�                                    Bxd}NN  �          @W
=@L(����
���
���
C�@L(��^�R�\)�
=C�`                                     Bxd}\�  T          @U�@J�H�z�H�#�
�1G�C�h�@J�H�G��(��*=qC�3                                    Bxd}k�  �          @\(�@O\)��33�u���C�Ff@O\)�xQ�(��"�HC���                                    Bxd}z@  �          @Z�H@N�R��\)��Q��G�C�o\@N�R�p�׿�R�&�HC��                                    Bxd}��  "          @[�@O\)��=q�aG��p��C���@O\)�W
=�8Q��A��C��H                                    Bxd}��  �          @Z=q@O\)���\��=q��33C�H�@O\)�B�\�:�H�E�C�h�                                    Bxd}�2  �          @[�@P  ���
��z����C�5�@P  �B�\�@  �J�RC�ff                                    Bxd}��  T          @Z=q@O\)�}p��������C�|)@O\)�5�E��Q�C��{                                    Bxd}�~  �          @[�@O\)��=q���
��z�C��)@O\)�J=q�O\)�Yp�C��                                    Bxd}�$  �          @U@H�ÿ�=q��\)����C���@H�ÿO\)�B�\�S�C���                                    Bxd}��  
�          @R�\@Fff���\��p���p�C���@Fff�8Q�O\)�e��C�~�                                    Bxd}�p  "          @O\)@C33���
���
���C���@C33�@  �G��_\)C�!H                                    Bxd}�  �          @N�R@B�\��  ��G���  C�� @B�\�+��^�R�x��C��                                    Bxd~�  T          @Q�@G
=�\(����!p�C�N@G
=��׿k����
C��                                    Bxd~b  �          @U@K��!G��B�\�R�RC�]q@K��B�\�xQ����C�L�                                    Bxd~*  "          @Y��@P�׿z�=p��J�\C���@P�׾���n{�
=C��                                    Bxd~8�  �          @U�@K��!G��@  �Q��C�j=@K��B�\�u����C�T{                                    Bxd~GT  
�          @P��@Fff�(��E��[33C�h�@Fff�#�
�xQ����HC�z�                                    Bxd~U�  �          @N{@AG��5�W
=�r�HC�c�@AG��aG���=q����C��
                                    Bxd~d�  �          @J�H@<�ͿB�\�^�R�33C�� @<�;�  ������G�C���                                    Bxd~sF  T          @G�@;��L�Ϳ333�Pz�C�g�@;���Q쿀  ���C��                                    Bxd~��  �          @@��@5��aG��
=q�$(�C�]q@5���\�c�
��=qC��H                                    Bxd~��  
�          @B�\@8Q�O\)��\��C�(�@8Q���W
=��z�C�u�                                    Bxd~�8  �          @@��@6ff�\(���ff�Q�C��H@6ff�
=q�L���x  C��                                    Bxd~��  �          @>�R@6ff�(�ÿ
=q�((�C�u�@6ff���
�L���w�
C���                                    Bxd~��  T          @AG�@:=q��Ϳz��2{C���@:=q�L�ͿG��n�\C�                                    Bxd~�*  �          @B�\@:=q�(�ÿ
=q�$��C��{@:=q���
�L���s
=C��                                     Bxd~��  �          @G
=@?\)�z��R�8��C���@?\)�W
=�Q��v{C��                                    Bxd~�v  �          @G�@@  �������2�HC�\)@@  �k��Q��s�C��=                                    Bxd~�  �          @G�@@  �
=�!G��;33C�k�@@  �W
=�W
=�yC���                                    Bxd�  �          @G
=@>{��R�333�O
=C�  @>{�L�Ϳh����C��                                    Bxdh  �          @G�@?\)���=p��[�C���@?\)��G��fff���RC��)                                    Bxd#  �          @K�@>�R�\(��0���H��C��
@>�R��
=���\����C�                                      Bxd1�  �          @P��@Fff�&ff�O\)�d��C�
@Fff�8Q쿂�\���C�Z�                                    Bxd@Z  �          @O\)@Dz�(��L���f=qC�h�@Dz�\)��  ���C���                                    BxdO   "          @N�R@A녿&ff�aG��|(�C��\@A녾����=q����C���                                    Bxd]�  �          @P��@Dz�!G��^�R�w�C�+�@Dz�\)������
=C���                                    BxdlL  �          @Q�@E�(��k����C�j=@E��Q쿌����z�C�&f                                    Bxdz�  
�          @Q�@Dz�
=�z�H��33C���@Dz�#�
������C��)                                    Bxd��  T          @J=q@<�Ϳ\)��  ���C��3@<�ͼ#�
��33���\C��                                    Bxd�>  �          @L(�@>�R����G���{C��@>�R<������G�?!G�                                    Bxd��  �          @K�@>{���H��G����
C�L�@>{=L�Ϳ�\)��
=?��                                    Bxd��  �          @G�@;���(���  ��  C���@;�=�G������
=@�                                    Bxd�0  �          @G�@:�H��
=��G�����C��f@:�H=�������@z�                                    Bxd��  �          @B�\@3�
��p������(�C�Ff@3�
>L�Ϳ�����
=@�z�                                    Bxd�|  �          @>{@-p���������z�C���@-p�>�������(�@�p�                                    Bxd�"  �          @:=q@,�;�33���\����C�T{@,��>8Q쿇�����@z=q                                    Bxd��  �          @,(�@�;W
=�������C���@��>��ÿ�����@��
                                    Bxd�n  �          @!G�@33�����\����C�\)@33>\�u��
=AG�                                    Bxd�  �          @@
=q�k��\(����\C��3@
=q>W
=�^�R����@�33                                    Bxd�*�  �          @z�?��ͽu�^�R����C��?���>�p��J=q��G�A4Q�                                    Bxd�9`  �          ?�
=?��>Ǯ���
�;��A�?��?u�p����B 33                                    Bxd�H  T          ?�\�\?p�׿�Q��[�RB��f�\?\�O\)��G�B�                                      Bxd�V�  �          ?��H����?k�����\
=B؀ ����?�p��E�����B�G�                                    Bxd�eR  �          ?�����?\(���{�[�HB�녾���?�z�E��ޣ�B���                                    Bxd�s�  �          ?�=q�
=q?O\)��p��N�B��Ϳ
=q?�ff�0����=qB�W
                                    Bxd���  �          ?��þL��?������1�B�\)�L��?�Q��
=�33B��
                                    Bxd��D  �          ?�zᾀ  ?�\)�����2Q�B��H��  ?��
��ff��=qB�ff                                    Bxd���  �          ?ٙ��.{?p�׿�33�^ffB�33�.{?�  �E��ظRB�Ǯ                                    Bxd���  T          ?��
���?p�׿�  �e��B�{���?�ff�\(���p�B�{                                    Bxd��6  
�          ?�
=�#�
>�G���(��w�
Cc׿#�
?��Ϳ���� B��
                                    Bxd���  �          ?�{���?녿�(���B����?�=q��p��)  B�                                    Bxd�ڂ  �          ?������>�33��\)�fC����?�����(��D�B��f                                    Bxd��(  "          @�\��=q>\��p�p�B����=q?����ff�F��Bˮ                                    Bxd���  T          @�\���>��H���H��B�
=���?�\)��p��<ffB�{                                    Bxd�t  T          ?�(�<#�
?L�Ϳ�p�G�B��R<#�
?���������B�Q�                                    Bxd�  
�          ?��=�?�G���=q�d�RB�=q=�?�녿fff���B��                                    