CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230205000000_e20230205235959_p20230206021331_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-06T02:13:31.454Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-05T00:00:00.000Z   time_coverage_end         2023-02-05T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        x   records_fill         (   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxa���  
�          @s33���e�?���A�
=C�z���l�Ϳ���C��H                                    Bxa�f  �          @p  �Ǯ�]p�?��\A�(�C�Ф�Ǯ�j=q��(����
C��)                                    Bxa�  T          @|(�>����c�
?�z�A���C���>����z�H������C�h�                                    Bxa�!�  �          @���?(���j=q?�A�G�C��?(�����H        C��{                                    Bxa�0X  �          @�p�?���p��?�A��C�&f?������L���3�
C���                                    Bxa�>�  
�          @|��?z��j�H?���A���C�~�?z��w�������HC�B�                                    Bxa�M�  T          @�  ?�R�mp�?�{A��C��f?�R�z�H��ff�θRC���                                    Bxa�\J  "          @y��=��p  ?xQ�Ag�C���=��q녿Q��D  C��\                                    Bxa�j�  �          @xQ�u�s33?.{A#\)C��ͽu�l�Ϳ�����=qC��=                                    Bxa�y�  �          @y���aG��w
=>�
=@�Q�C�]q�aG��h�ÿ�{��\)C�E                                    Bxa�<  �          @w
=�\)�u�>�G�@��C���\)�g���=q��33C�޸                                    Bxa��  
�          @j=q=�\)�hQ�>��
@��RC��\=�\)�X�ÿ������C���                                    Bxa  T          @`��>B�\�^{>��
@�Q�C���>B�\�O\)���\��33C���                                    Bxa�.  �          @Z=q<#�
�XQ�>���@�
=C��<#�
�L�Ϳ�z���C��                                    Bxa���  �          @9��=�Q��8Q�>�=q@�=qC���=�Q��,(���ff����C��R                                    Bxa��z  �          @Q���Q�=u?�{C��R���	�����\��z�C���                                    Bxa��   �          @P��=#�
�P��=L��?k�C�P�=#�
�;���z����C�Y�                                    Bxa���  "          @qG�=�Q��p  �����C��==�Q��U���H��ffC��                                     Bxa��l  T          @���=����s�
?h��AUC���=����s�
�fff�TQ�C���                                    Bxa�  .          @��������l��?��HA�{C�t{�����~{��33����C���                                    Bxa��  H          @��
=q��  ?�\)AjffC�b��
=q��녿fff�<Q�C�l�                                    Bxa�)^  T          @���k����
?��A�p�C�b��k����
�����z�C�z�                                    Bxa�8  �          @���u�z�H?���A�  C��ýu���H���Ϳ��C��                                     Bxa�F�  T          @���?8Q��E@1�B$��C��?8Q���  ?���Ar{C�
                                    Bxa�UP  
Z          @�z�?�\)�.{@VffB:�\C�ff?�\)�{�?��
A��
C�4{                                    Bxa�c�  
�          @�G�?�녿���@�Bu=qC��H?���S33@A�B��C�7
                                    Bxa�r�  T          @���?xQ��2�\@Z=qBD33C���?xQ�����?�ffA�  C�                                    Bxa�B  T          @���?}p��1�@Z�HBD�C��?}p���Q�?���A��C��f                                    Bxa��  T          @��H?�p��p�@n{BY��C��?�p��h��@A�z�C�\                                    Bxa  �          @��?�����@h��BXC���?����mp�@p�A�\C��
                                    Bxa�4  �          @��?�  �"�\@c33BQ��C���?�  �w
=@�\A���C�>�                                    Bxa��  
�          @���?^�R�AG�@[�B>{C��?^�R��\)?�p�A�{C��\                                    Bxa�ʀ  T          @��
?fff�Z=q@U�B-�C�n?fff��G�?��HA�ffC���                                    Bxa��&  �          @�=q?\(��l(�@=p�B=qC���?\(����
?�G�AB=qC�L�                                    Bxa���  �          @�?8Q��u@&ffBz�C�U�?8Q���33?�R@�C�y�                                    Bxa��r  
�          @��
?�\)����@{A�C���?�\)��
=>��@�=qC��=                                    Bxa�  "          @�=q?���mp�@  A�
=C�=q?�����\>�p�@��C��H                                    Bxa��  "          @�Q�?��H�u�?�\)A��C�s3?��H��Q���ǮC�xR                                    Bxa�"d  �          @�ff?�������?�p�A�z�C��?�����  �
=��{C�w
                                    Bxa�1
  �          @�  ?����A�@"�\B�
C�9�?����u?n{AQ�C�/\                                    Bxa�?�  
�          @��
?��Ϳ�\@b�\BXffC�/\?����I��@Q�BffC�8R                                    Bxa�NV  
�          @�  ?�(��.{@K�B7ffC�9�?�(��vff?�33A�Q�C�w
                                    Bxa�\�  �          @�(�?�{�W�@>{BC��?�{���H?�AiC�+�                                    Bxa�k�  �          @���>�ff����?�
=A�  C�+�>�ff��33��\)�j�HC��\                                    Bxa�zH  �          @�\)��z����?!G�A=qC�#׾�z�����������C��                                    Bxa���  
�          @�(������?�\@ʏ\C�q����=q�Ǯ��  C�=q                                    Bxa�  
�          @�=q?�R�|(�?���A�
=C�z�?�R��33��Q쿎{C�\                                    Bxa�:  "          @�ff?c�
�u@G�AظRC��\?c�
���\=�G�?�C���                                    Bxa��  
�          @�ff?O\)���?�z�A�Q�C���?O\)��(����R�\)C�B�                                    Bxa�Æ  �          @�=q?Q���p�?Tz�A'33C�C�?Q����\���H�up�C�`                                     Bxa��,  "          @���?(�����?aG�A2�RC��?(����\��z��l��C���                                    Bxa���  T          @�=q?!G����?��@�C��?!G���G��������C�1�                                    Bxa��x  �          @�33?.{���?:�HAp�C�G�?.{��33��=q��ffC�j=                                    Bxa��  �          @�33>�
=��p�?���A^{C��q>�
=��ff�u�B�RC��R                                    Bxa��  T          @��?+���(�?��A��RC��H?+���  �aG��0  C�C�                                    Bxa�j  �          @�Q�?�����?�Aʏ\C��?����R�L�Ϳ��C�g�                                    Bxa�*  T          @�G�?��\)@z�A�p�C�R?����=���?�ffC���                                    Bxa�8�  "          @�ff?�����?�G�A�z�C�u�?�����z���33C�L�                                    Bxa�G\  
�          @�\)���
���H?��AV=qC��׽��
�������J�HC���                                    Bxa�V  T          @�>W
=��  ?�
=Ak\)C�P�>W
=��녿h���4��C�L�                                    Bxa�d�  �          @���?�����?�\)A��C�s3?����ff�E���C�Y�                                    Bxa�sN  �          @���?���=q?��RAp��C���?����Ϳc�
�,Q�C�|)                                    Bxa��  "          @�(��L����\)?���AV�\C��R�L����  ����Ip�C��R                                    Bxa�  �          @���   ��?   @�\)C��   ���Ϳ��H���HC��\                                    Bxa�@  �          @�=q�5��Q�<#�
>�C����5��Q�����z�C��\                                    Bxa��  �          @�z�z�H���׾B�\�Q�C�y��z�H������ffC���                                    Bxa�  �          @�\)�����>8Q�@�C�XR�����G�������\C��                                    Bxa��2  T          @����
=���
>��
@o\)C��\��
=���׿���z�C�`                                     Bxa���  "          @�������n{�0z�C��H���w
=�:=q��C�,�                                    Bxa��~  
�          @���   ��
=���
���RC��=�   �P  �c�
�<��C���                                    Bxa��$  �          @�G���33���
��
=��  C��쾳33�E�h���E�C��                                     Bxa��  "          @�{�\�������(�C�o\�\�>�R�e��G��C�\)                                    Bxa�p  "          @��׾�ff�}p��'��G�C�����ff��H���\�kp�C��\                                    Bxa�#  
�          @�33�:�H�i���^�R�,(�C�\)�:�H���
��
=�qCw��                                    Bxa�1�  
�          @��
����p���\���)z�C�ff��׿������ Cٚ                                    Bxa�@b  �          @�(������aG��mp��9ffC��H���ͿǮ���
u�C�{                                    Bxa�O  �          @��R�
=q�k��2�\�\)C��=�
=q�
=����xffC��                                    Bxa�]�  
�          @��׿��
���\<#�
=�Q�C�{���
�k���G���=qC~Y�                                    Bxa�lT  �          @�=q�k���ff>��@��C�s3�k��������(�C��                                    Bxa�z�  �          @��ͿTz����>.{@ ��C���Tz������
=��  C��=                                    Bxa�  �          @���:�H����=�Q�?�33C�p��:�H���
������C���                                    Bxa�F  .          @��R�&ff�����\��  C�n�&ff�_\)���  C���                                    Bxa��            @�=q�^�R�(Q��Mp��D��C{�Ϳ^�R�z�H����  Cd��                                    Bxa�  T          @�{�J=q���R�fff�l�
Cx:�J=q�u��33�CE                                      Bxa��8  `          @��R�xQ��2�\�Fff�9G�CzǮ�xQ쿗
=��  \Cf��                                    Bxa���  z          @�����
�i���-p��G�C~B����
�������nG�Ct+�                                    Bxa��  
�          @�(������ff����C�Ff����A��~{�K��Cz��                                    Bxa��*  H          @�G�������R��ff�p��C��=����l���L����C~                                      Bxa���  �          @��׿�=q�Fff�W��733Cz�=��=q������(���CgJ=                                    Bxa�v  `          @�
=����1��e��E�\Cu0�����z�H��p��HCY:�                                    Bxa�            @�Q쿗
=�E��Y���7\)Cy���
=��=q��z�Cdh�                                    Bxa�*�  
�          @�33��=q�G��e�=�RCz�쿊=q���
���\�)Ce�H                                    Bxa�9h  
�          @�G����R�j�H�G���C{T{���R���R��z��w�Cm��                                    Bxa�H  �          @�녿^�R��=q��Q��d��C����^�R�g��A��ffC�<)                                    Bxa�V�  "          @��ͿW
=�����p��u��C��׿W
=�]p��@  � �C�7
                                    Bxa�eZ  
Z          @�{�h����p���G��yC�.�h���]p��A��!
=C@                                     Bxa�t   "          @�ff�h����{��G��yG�C�5ÿh���^{�B�\� �
CT{                                    Bxa�  "          @��׿}p����׿�z��`z�C�ٚ�}p��fff�>{�G�C~��                                    Bxa�L  �          @�Q������\��{�W\)C�J=����k��<����C�b�                                    Bxa��  �          @�G��.{����aG��(��C��3�.{�u�1��C��3                                    Bxa�  �          @��׿���������F=qC~�H����q��<�����C{8R                                    Bxa�>  �          @�(��ٙ����
��
���Cx� �ٙ��E��n{�:ffCq�                                    Bxa���  
(          @�����H���H�z���
=Cx�\���H�C�
�mp��:��Cp�
                                    Bxa�ڊ  �          @����(�����
=��Cxs3��(��C33�p���;��Cp�
                                    Bxa��0  	.          @�33��G���Q쿽p���Q�Cx�H��G��]p��P  ���Cr�3                                    Bxa���  �          @�p������
��  ���C|����\(��b�\�.\)Cw��                                    Bxa�|  �          @�������\)�   ���HCx� ����1G������N  CoY�                                    Bxa�"  
�          @�33�����\)�P�����Cy�Ό���\)��(��p=qCm                                    Bxa�#�            @��׿��H��33�E�ffC}�����H�=q�����l�
CsW
                                    Bxa�2n  �          @��
���\�x���Dz���C{녿��\��R��p��op�Cp^�                                    Bxa�A  "          @��R�������R��p��k�C|�����a��?\)��Cx�
                                    Bxa�O�  �          @�{�Ǯ���H?=p�A
{C���Ǯ��ff��\)��\)C��)                                    Bxa�^`  "          @��?z���p�?�Q�A�=qC���?z����\���Ϳ��C�j=                                    Bxa�m  
�          @�=q?\(����H@6ffB�C��\?\(����?\(�AG�C��R                                    Bxa�{�  �          @�>��H���?�
=A�\)C�33>��H��p����R�p  C���                                    Bxa�R  �          @�p������z��33��  C�9�����9���e�K��C�y�                                    Bxa���  	�          @�\)?��
�c�
@C�
BG�C���?��
����?��Ar=qC�S3                                    Bxa���  
�          @���?���Z=q@O\)B =qC�� ?����ff?\A�Q�C���                                    Bxa��D  �          @�\)?�
=�P��@Q�B*�C��\?�
=���\?�\)A�Q�C��                                    Bxa���  "          @�  ?����@{�BR�\C��
?���p  @&ffA��C�e                                    Bxa�Ӑ  �          @�{?�
=�;�@q�BH��C���?�
=���@{A�  C�Ǯ                                    Bxa��6  "          @�  >��^�R@I��B'�\C�� >���
=?�z�A��C���                                    Bxa���  �          @����\)�dz�@C�
B"�\C�q콏\)����?�ffA��\C���                                    Bxa���  T          @�Q�?\)�+�@p  BW(�C��?\)��  @�\A�RC�                                    Bxa�(  T          @�����
�\(�@p�BG�C�˅���
���?O\)A0��C��{                                    Bxa��  /          @�33��G���(�?�p�Ab=qC�k���G���ff�fff�$(�C�t{                                    Bxa�+t  
          @�33>#�
��
=?���A�\)C���>#�
��녾�=q�A�C��                                    Bxa�:  �          @�ff>����G�@4z�B(�C���>����=q?J=qA=qC�u�                                    Bxa�H�  
g          @�=q�Q����>��@�C�o\�Q���33��Q���G�C�1�                                    Bxa�Wf  G          @�녿�����33>��@���CzῨ����녿޸R��p�C~��                                    Bxa�f  
�          @��ÿ��\��33����=qC�\)���\���R�!G���33C���                                    Bxa�t�  T          @�  ?�����
�#�
�޸RC�  ?����33�  ���
C���                                    Bxa��X  T          @��?O\)��zῆff�6ffC�|)?O\)��  �Dz��Q�C�ff                                    Bxa���  T          @���?=p����
�\���C�  ?=p���G��_\)�!\)C�33                                    Bxa���  "          @���>��H���H��p���\)C��f>��H�{��j=q�+  C���                                    Bxa��J  �          @�\)>B�\��(������p�C�'�>B�\�O\)���H�NffC��f                                    Bxa���  "          @���=L�������(�����C�Q�=L������$z�� {C�`                                     Bxa�̖  
�          @���?#�
������
����C���?#�
�l���U��&G�C��                                     Bxa��<  T          @�p�>�ff���ÿ�����\)C��>�ff�n{�X���(�\C�p�                                    Bxa���  
�          @�=��
��33�8Q�� {C���=��
�A���  �`Q�C��f                                    Bxa���  T          @��>�������\)�ۙ�C��>���W
=��\)�M�HC�H�                                    Bxa�.  T          @�(��Ǯ����������HC����Ǯ�W���{�L
=C��\                                    Bxa��  "          @�(����
��Q������
C�.���
�tz��p  �1��C��)                                    Bxa�$z  �          @��R�����H�fff�   C�Ф����G��8���C�Ǯ                                    Bxa�3   "          @��\?����R��=q����C�#�?��c�
�e��3=qC�"�                                    Bxa�A�  	.          @�=q?W
=����p���4��C�0�?W
=�u�.{�
��C�5�                                    Bxa�Pl  "          @��\������?G�A�
C��ü���녿��e��C��{                                    Bxa�_  "          @�z�����<��
>L��Cw�q�����G���\)��Q�Cv33                                    Bxa�m�  T          @��ͿxQ���  >��@�33C��׿xQ����������C�8R                                    Bxa�|^  
�          @�Q�>�����
=�L���  C���>�����
=���ϙ�C��                                    Bxa��  T          @�>k�����5���C�Y�>k�����&ff���C��3                                    Bxa���  �          @�
=?
=q�����(���G�C�Ǯ?
=q�)����G��`��C���                                    Bxa��P  "          @�
=>���=q�333�	��C�W
>��&ff��ff�g=qC�0�                                    Bxa���  �          @���>�(���������(�C���>�(��C�
�~�R�P�C���                                    Bxa�Ŝ  �          @��R=��\)�8Q��\)C��=��   ����m�C�l�                                    Bxa��B  "          @���?�G���=q��Q���z�C��\?�G��u�P  �  C�b�                                    Bxa���  	�          @�z�?h����\)�@  ��\C�/\?h�������*�H��  C�f                                    Bxa��  "          @�
=?Q���=q�1��ffC�Z�?Q��6ff�����]  C��q                                    Bxa� 4  	�          @�?&ff�����8���33C�S3?&ff�1G���33�cp�C���                                    Bxa��  "          @��?@  �����C33��C�H?@  �-p���  �h
=C�Ǯ                                    Bxa��  "          @�z�?0����33�޸R��  C�H�?0���a��Z�H�.=qC��                                    Bxa�,&  "          @���>�z����(�����C���>�z������ �����C��3                                    Bxa�:�  T          @�������Y��@!�B
=Cz��������
?xQ�ALz�C}�q                                    Bxa�Ir  T          @������)��@x��B<\)Cc@ ����~{@!G�A���CnQ�                                    Bxa�X  �          @�=q�+��G
=@Y��BffCe8R�+���ff?��A�z�Cmk�                                    Bxa�f�  T          @��H��R�`��@UB=qCm�H��R��G�?�Q�A��Cs��                                    Bxa�ud  
�          @�ff��z����?���A�\)Cyk���z����׽u�+�Cz�)                                    Bxa��
  �          @���޸R��
=?fffA!��Cy�R�޸R��{���
�:=qCy�)                                    Bxa���  
Z          @�33���p�?�{Ay�Ct�����33�����RCu�)                                    Bxa��V  
�          @����	����33?�Q�A�=qCrO\�	������>8Q�@�Ct��                                    Bxa���  
M          @�  ����p�?�33A�Cs�����\)��G����\Ct�                                    Bxa���  T          @����
�H��ff?��@�Ct�
�H���H��Q��\(�Csff                                    Bxa��H  �          @�=q��p���\)?0��@���C|����p����
���H�`z�C|=q                                    Bxa���  �          @������H����?��
Aj�RC|�H���H���ÿ#�
���C}�                                    Bxa��  "          @��������?�  Af�RCuG�����������\)Cu�                                    Bxa��:  �          @��
�������?�ffAn�HCxuÿ�����\)��p�Cy
                                    Bxa��  �          @��
�z���\)?�ffAk33Cu33�z���z�����(�Cu��                                    Bxa��  
�          @�z��&ff��z�?�ffA���Cm�{�&ff��p��.{��z�Coz�                                    Bxa�%,  "          @��R�9���xQ�?�33A�G�CiB��9�����>�  @0��Cl)                                    Bxa�3�  
�          @��J�H�g�?��A���Cd�=�J�H��=q>�{@o\)Ch\                                    Bxa�Bx  "          @�(��XQ��Q�@G�A�\)C`+��XQ��s33?
=@��Cdff                                    Bxa�Q  "          @���hQ��n{?�  AYG�Ca�R�hQ��z=q��=q�<(�Cc0�                                    Bxa�_�  
�          @�
=�G���  ?�(�AyCi���G������=q�7�Ck33                                    Bxa�nj  �          @�
=�L(���(�?�\)A�Q�ChW
�L(���{��Q�fffCjO\                                    Bxa�}  
�          @�  �8����p�?�ffA��RCl�{�8����p�����0  CnG�                                    Bxa���  
�          @��-p���ff?�ffA�z�Cn���-p���ff��=q�7�Cp\                                    Bxa��\  �          @�(��.�R����?��HA���Cm���.�R���
�u�&ffCos3                                    Bxa��  �          @��
�,(����?޸RA�
=Cn{�,(����ͽ#�
��
=Co��                                    Bxa���  
�          @�G���
���
?�\A��
Cr!H��
��
=���
��=qCs��                                    Bxa��N  
�          @���   ���?���A�
=Co��   ���>B�\@�Cqٚ                                    Bxa���  "          @���$z���
=?��A���Cn���$z����
>��?�\)Cp޸                                    Bxa��  �          @����$z����@(�A�(�Cm���$z����\>��H@�=qCp�q                                    Bxa��@  "          @�(��*=q��(�@(�A�(�Cm8R�*=q����>�@�{Cp=q                                    Bxa� �  
�          @�z��0  ��p�@G�A�Q�Cl�H�0  ��(�>���@I��CoJ=                                    Bxa��  �          @�p��5���
=?���A�=qCl��5����=�?���Cnh�                                    Bxa�2  �          @���7
=���\?��A��Ck�7
=���ͼ��
�k�Cm                                      Bxa�,�  T          @��
�0  ���H?���Ap��Cl(��0  ��G����R�`  Cmc�                                    Bxa�;~  "          @������u?�A��Cn\)����  >�=q@G�Cp��                                    Bxa�J$  T          @����
�H��=q?޸RA�z�Cs^��
�H����#�
�uCu\                                    Bxa�X�  �          @�{�У�����?�A���CzLͿУ����H�����
=C{k�                                    Bxa�gp  �          @�{��ff���R?�(�A��C~�H��ff���׾#�
����C��                                    Bxa�v  �          @�p���{��{@z�A�G�CzͿ�{��z�>��@<��C{                                    Bxa���  �          @�(���33����@��A��
Cy���33���>�G�@��C{\                                    Bxa��b  T          @�z��z���
=@z�A�(�Cs�
�z���{>�33@|��Cv&f                                    Bxa��  
�          @��H�0���|��?�\)A�z�Ck��0����Q�=#�
>�
=Cm�                                    Bxa���  
�          @���Fff�p��?�z�A��RCf� �Fff���>��?�Q�Ch��                                    Bxa��T  
�          @�ff�3�
�}p�?���A��Cj���3�
���
>��@5Cm@                                     Bxa���  T          @��
�8���z�H?���A�\)Ci���8����
=<��
>B�\Ck��                                    Bxa�ܠ  T          @�33�5��~{?�
=A��RCj���5���
=����  Cl!H                                    Bxa��F  �          @����������?�ffA�p�C|�H������׾aG�� ��C}�q                                    Bxa���  �          @�  ��
��G�?�\)A���CtY���
��������Z�HCuaH                                    Bxa��  T          @����'����
?�Q�AZ�HCm���'����׾������Cnp�                                    Bxa�8  T          @����R���?�\)A�G�Cq�{��R���
��  �:=qCs                                      Bxa�%�  
�          @��H��
=��
=��\)�aG�Cy^���
=��z�ٙ����HCw�)                                    Bxa�4�  T          @�ff�!G�����?�\)A��RC�q�!G����\�����33C�B�                                    Bxa�C*  T          @���{����>8Q�@Cw���{���ÿ�p����HCv�=                                    Bxa�Q�  
�          @���������?O\)Ap�Cs�)�����ÿL���z�Cs�)                                    Bxa�`v  
(          @�Q��
=����?�A�{CvW
��
=���
��=q�Dz�CwW
                                    Bxa�o  "          @����������\?�A���Cu�ÿ�����z�#�
���Cw(�                                    Bxa�}�  "          @�(���\����?J=qA�Cs(���\��  �fff�!Cs�                                    Bxa��h  	�          @��\�ff��=q?(��@�  CuG��ff��  ���
�;\)Ct�3                                    Bxa��  �          @��
�����?&ff@�G�CuQ������ÿ�ff�>=qCt��                                    Bxa���  �          @�z�����p�?���ALQ�Cx�R�����Q��R��{Cy\)                                    Bxa��Z  �          @�33�ff���
?�z�AR�HCq� �ff��  ����{Crk�                                    Bxa��   T          @�Q��"�\��?��AC33Cn���"�\���ÿ�\���HCoW
                                    Bxa�զ  "          @���'
=��(�?�  Ae��Cm�R�'
=��녾��R�`  CnǮ                                    Bxa��L  "          @����
��\)>��H@�{CuO\��
�������Tz�Ct�R                                    Bxa���  "          @��
��Q���ff�����l(�C�9���Q���Q��z���\)C.                                    Bxa��  
�          @��H��33���þ��R�_\)Cz���33������R���RCy=q                                    Bxa�>  
�          @�33������R=L��?\)Cx@ �����p���\)��\)Cv��                                    Bxa��  T          @�33��z���{>8Q�@z�C�b���z���{��=q���RC��                                    Bxa�-�  T          @�33�aG���(�>��H@�(�C��H�aG�������\�k�C���                                    Bxa�<0  �          @�{���G��L����\CuB���������p�Csh�                                    Bxa�J�  
�          @��\�����{=�?��Cy#׿����p���\)��33Cx�                                    Bxa�Y|  
�          @��H�����33>�@��C=q�����{����j=qC~��                                    Bxa�h"  �          @��H������G������ZffC��H��������E�\)C
                                    Bxa�v�  �          @�Q�������Ϳ��
�-��C|� �������R�1G���Q�CzG�                                    Bxa��n  
�          @��\������33�W
=�	�C�
������
=�)����33C~n                                    Bxa��  �          @������H��녿   ���C~�)���H�����z���(�C}{                                    Bxa���  
�          @��ÿ�����(��+��߮Cz� �������\�=q�Џ\Cx\)                                    Bxa��`  "          @��\�\)��?�A�33Cq�=�\)��  =�?���Cs�                                    Bxa��  T          @�
=���|(�?��RA�G�Co������(�?�\@��HCr��                                    Bxa�ά  �          @��R��
=��Q�?�z�AX��Cy����
=��(������ffCz�                                    Bxa��R  
�          @�{��{���?�z�A�\)C�0���{���׾�  �>�RC�xR                                    Bxa���  �          @�녿������?\(�A�C~LͿ����G��W
=���C~Q�                                    Bxa���  
�          @�ff��
=��z�?�AR=qC�7
��
=��  �z���p�C�Z�                                    Bxa�	D  
�          @�p�?:�H�c�
@j=qB4��C�Ǯ?:�H���H@{A��C�~�                                    Bxa��  
�          @�Q�?fff��Q�@QG�B��C�U�?fff��z�?ٙ�A���C�9�                                    Bxa�&�  T          @�ff?�  �n{@VffB"
=C�C�?�  ��z�?��A��HC��                                    Bxa�56  "          @�{?�
=�fff@c33B,�C��?�
=���H@
=A�C�.                                    Bxa�C�  �          @�ff?B�\���@>�RB�HC�0�?B�\��p�?��AyC�e                                    Bxa�R�  T          @��R?���ff@'
=A�  C�]q?���=q?s33A(Q�C��{                                    Bxa�a(  
�          @��
>k���
=@"�\A��
C�aH>k����?J=qA33C�:�                                    Bxa�o�  "          @�논���=q@{AŅC��R����G�>�@�C��)                                    Bxa�~t  �          @�����  ��(�@�\A�33C����  ����>�=q@<��C��=                                    Bxa��  	�          @���>�����H@�A�=qC�޸>����  >���@g
=C���                                    Bxa���  
�          @���>�����?�A�{C��f>�����=��
?h��C�l�                                    Bxa��f            @�  >k���?#�
@�  C�H�>k���33����I�C�N                                    Bxa��  T          @�
=>aG�����?���AUG�C�@ >aG����Ϳ\)��{C�9�                                    Bxa�ǲ  "          @�  ?333��?ǮA��HC�
=?333��p��8Q���HC��)                                    Bxa��X  �          @��R?������@��A��C�Z�?����z�?z�@���C�
=                                    Bxa���  
�          @��
�����\?n{A1��C�׿���(��&ff��
=CǮ                                    Bxa��  
�          @�33=����@
=A��C��==�����?=p�A�C��{                                    Bxa�J  �          @�p�?!G�����@%�A��HC�\?!G���  ?z�HA/33C��3                                    Bxa��  T          @����
=����@&ffA�z�C�\)��
=���
?uA)��C���                                    Bxa��  �          @�  ��=q��p�@4z�B�HC�G���=q��33?���AT(�C���                                    Bxa�.<  "          @���>����33@:�HB��C��=>����=q?�=qAi�C�J=                                    Bxa�<�  �          @�=q?�ff����@1�A��RC��\?�ff��=q?�
=AN{C��{                                    Bxa�K�  T          @���?��:=q@Z=qB133C���?��w�@��A�G�C�/\                                    Bxa�Z.  T          @��\@U�\@vffB9Q�C��H@U�,��@H��B�C��                                    Bxa�h�  �          @�ff@AG�����@p  B8�
C���@AG��Dz�@:=qB=qC�G�                                    Bxa�wz  "          @���@&ff�3�
@k�B/�C�b�@&ff�vff@#�
A�33C���                                    Bxa��   "          @��@�
�K�@a�B'��C��q@�
����@�
A�\)C���                                    Bxa���  "          @��@�K�@aG�B&�C�.@����@33A�ffC��R                                    Bxa��l  T          @��
@0  �+�@p  B1G�C�� @0  �p  @+�A�  C�#�                                    Bxa��  "          @��H@,���'
=@s33B5\)C��q@,���l(�@0  A�z�C�{                                    Bxa���  T          @�=q@@  �p�@o\)B4�\C�ٚ@@  �R�\@5B�HC�.                                    Bxa��^  T          @�
=@W���{@h��B0��C�9�@W��,(�@;�B�C��\                                    Bxa��  �          @�Q�?����\)@?\)B��C��R?������?ǮA�p�C�~�                                    Bxa��  "          @�
=?������@"�\A���C�xR?�����H?uA)p�C�{                                    Bxa��P  �          @���>�����R@�A�Q�C�u�>�����R?5@���C�:�                                    Bxa�	�  
6          @�=q?��H����@<(�B
=C�k�?��H����?�  A���C��{                                    Bxa��  
          @��\?�{���R@:�HB�\C��\?�{��p�?�
=AzffC��
                                    Bxa�'B  
Z          @����B�\��p�?�=qA�33C��{�B�\��Q�>��@=p�C���                                    Bxa�5�  T          @�  �������R?���A��C�C׿������H>�Q�@�  C���                                    Bxa�D�  T          @�  ������R?�p�A��
C{�������Q�>#�
?��HC|�                                    Bxa�S4  
�          @�\)�У�����?��Am��C{��У����R�k��!�C{�                                     Bxa�a�  �          @��
� ����@	��A���CtaH� ������?:�HA�Cv�f                                    Bxa�p�  �          @�p������p�@�\A��\Cqz������33?#�
@�33CsǮ                                    Bxa�&  T          @�(��J�H�r�\?�{AJ�RCf
�J�H�|�;���Q�Cg=q                                    Bxa���  
Z          @�(��fff���@\)A�=qC�!H�fff���
?��
A;�C��=                                    Bxa��r  T          @����{���R?��HAYCs�=�{���
�u�*�HCtO\                                    Bxa��  �          @�z��7����
?xQ�A.{Ck��7���
=��Q�����Ck�R                                    Bxa���  
(          @�ff�Q�����?�ffA�p�Csz��Q���(�>Ǯ@�p�Cu:�                                    Bxa��d  T          @��R�������H@�A�G�Cv녿�������?+�@�  Cxٚ                                    Bxa��
  T          @���Q����?��A�  Ct�
�Q�����>8Q�?���Cu�                                    Bxa��  �          @�������?&ff@�Cq�����G��G��	Cq��                                    Bxa��V  x          @��R�%���z�?z�HA-�Co�=�%���
=�����=qCp
=                                    Bxa��  �          @�\)� ����
=?z�HA-p�Cp��� ��������(���\)Cq+�                                    Bxa��  	�          @�  �(���=q?J=qA
�HCq��(����H�#�
��Q�Cr�                                    Bxa� H  �          @�  ����?fffA��Cs{���������
=Cs\)                                    Bxa�.�  T          @�\)�u��<��?�A�
=CY�)�u��X��?n{A%�C]�=                                    Bxa�=�  �          @��R�s33�!G�@{A�  CU�
�s33�I��?���A�p�C[�3                                    Bxa�L:  T          @���2�\�Z�H@$z�A�\)Cfٚ�2�\��G�?�z�A��\Ckc�                                    Bxa�Z�  "          @�  �,���g
=@)��A�Ci33�,�����?�Q�A�
Cm��                                    Bxa�i�  
�          @��R�!G��k�@'�A�p�Ck���!G�����?��Az{Co�f                                    Bxa�x,  �          @�p���
�|(�@(Q�A��RCrn��
��G�?�=qAn=qCu��                                    Bxa���  "          @����  �xQ�@@��B�Cx�)��  ��33?��HA��
C{�                                    Bxa��x  T          @�\)�;��K�@9��Bp�CcL��;��y��?���A��
Ci�                                    Bxa��  
�          @��
�  �aG�@=p�BG�Cmn�  ���?��
A��RCr�                                    Bxa���  �          @�����H��\)?�(�A�\)Cu0����H��(�?!G�@��
Cw{                                    Bxa��j  
�          @��
��������@,(�A�(�CxY���������?�\)Ay�Cz�q                                    Bxa��  �          @����33�~{@7
=B�Cz�f��33��z�?ǮA��RC}B�                                    Bxa�޶  
�          @��H��{�xQ�@@  B��Cz�H��{���H?�p�A��RC}u�                                    Bxa��\  
�          @��
��Q��Tz�@W
=B$\)Co�q��Q����@�RA�\)Cu                                    Bxa��  
�          @����Vff@_\)B*�Cq�Ϳ���
=@
=A�\)Cv                                    Bxb 
�  "          @�
=���R�j�H@_\)B(
=C{E���R����@G�A�{C~��                                    Bxb N  0          @�
=�����s�
@Y��B"C~E������z�@��A�{C�|)                                    Bxb '�  x          @�
=���H�q�@Y��B"33C|LͿ��H���@	��A�z�CO\                                    Bxb 6�  �          @��R���H�h��@a�B*�RC{�׿��H��  @�A�(�C                                      Bxb E@  
�          @��R��\)�s�
@U�B�C}�׿�\)���@A��
C�.                                    Bxb S�  
�          @����G��j=q@`��B(�HC{Ϳ�G�����@�
AхC~s3                                    Bxb b�  �          @�  �z�H��  @O\)B��C�  �z�H��Q�?��HA�\)C�1�                                    Bxb q2  T          @�녿h����Q�@U�B�C���h������@�\A��C���                                    Bxb �  �          @�=q������H@W�B��C�'������(�@33A���C��f                                    Bxb �~  
(          @��H��\)����@J=qB�C�"���\)����?�ffA�\)C�ff                                    Bxb �$  "          @��H���
���H@G�B��C�|)���
����?�  A���C��\                                    Bxb ��  T          @�G���{���H@@  B
33C��H��{����?��A���C�                                    Bxb �p  
(          @�Q�+����@A�B�C�� �+���p�?��HA�  C��                                    Bxb �  T          @����8Q��{�@^�RB$(�C��
�8Q���  @�RA���C��R                                    Bxb ׼  �          @�녿aG��hQ�@p��B5  C�0��aG���G�@%A�  C��                                    Bxb �b  T          @��\�s33�S�
@�G�BF
=C}�s33��=q@<��B�RC��                                    Bxb �  �          @�=q�����`��@r�\B6��C{
������@*=qA��\C~��                                    Bxb�  �          @��H�W
=�n�R@l��B0{C��\�W
=���
@ ��A���C��q                                    BxbT  �          @���+��o\)@uB5ffC��Ϳ+����@)��A��HC��=                                    Bxb �  �          @�ff�W
=�tz�@r�\B0��C��3�W
=��
=@%�A�z�C���                                    Bxb/�  �          @�(������w�@Z�HB�RC{!H������p�@{A�Q�C~.                                    Bxb>F  �          @��s33�a�@|(�B<��C~�3�s33���@3�
A�=qC�                                    BxbL�  T          @�(�����j=q@qG�B2�HC~&f������@'
=A�C��3                                    Bxb[�  "          @�=q�u�hQ�@p  B3�HC8R�u����@'
=A�p�C�                                    Bxbj8  �          @�p������A�@���BL�HCy�������Q�@B�\B=qC~E                                    Bxbx�  "          @��>�Q��fff@J�HB$�
C�޸>�Q����\@�
A��C�c�                                    Bxb��  �          @�=q@  ��p�>aG�@"�\C�xR@  ��녿}p��8��C�                                    Bxb�*  "          @�=q?��R��=q�5�C��?��R��ff��
=��{C���                                    Bxb��  
(          @��\?��R��������C��
?��R��{����{C��3                                    Bxb�v  
Z          @�\)@5�}p�?�33A�{C��
@5���\?@  Az�C���                                    Bxb�  �          @�G�@U��w�?�\)Ap��C�]q@U���33>�\)@?\)C���                                    Bxb��  "          @���@E�}p�?�z�A�Q�C��)@E��Q�?�@���C���                                    Bxb�h  �          @���@:=q�z=q@�
A��C�W
@:=q��=q?n{A"�\C���                                    Bxb�  �          @��\@6ff���
?��A��C�P�@6ff��ff?�R@�\)C�L�                                    Bxb��  
�          @�(�@7���=q?�{Aj�\C�˅@7�����>��?�33C�0�                                    BxbZ  "          @�z�@+�����?�Q�AK�
C�Z�@+���p��u��RC��                                    Bxb   
�          @�=q@7
=��=q?�z�AI�C�� @7
=��
=�����
C�K�                                    Bxb(�  
�          @��H@N�R�a�@
=A���C�@ @N�R�}p�?���AB�HC���                                    Bxb7L  �          @��\@G
=�N�R@7�B�C��3@G
=�xQ�?�A��RC�`                                     BxbE�  
(          @�
=@_\)�>�R@=p�B33C���@_\)�j=q@z�A�  C���                                    BxbT�  �          @���@]p��S33@0  A���C�&f@]p��z=q?��
A��HC���                                    Bxbc>  
�          @���@c33�aG�@�AǙ�C��@c33��Q�?���A^=qC��=                                    Bxbq�  T          @���@k��o\)?�\A��HC�K�@k���=q?:�H@�=qC�3                                    Bxb��  
Z          @�G�@p���{�?�G�A(��C��
@p����녽L�Ϳ   C�b�                                    Bxb�0  
(          @���@6ff��p�?��A�=qC�l�@6ff��{>�(�@�  C��f                                    Bxb��  	�          @�=q@E��p�?�\)Ad��C�w
@E���
>8Q�?��C��)                                    Bxb�|  �          @�=q@g
=���\?Y��AG�C���@g
=����u�{C�s3                                    Bxb�"  T          @���@w��xQ�>�  @%C�q�@w��tz�B�\��
=C���                                    Bxb��  "          @�
=@l���|��?5@�C��{@l����  �����_\)C�g�                                    Bxb�n  	�          @��@Z�H���?B�\Ap�C�� @Z�H��p����
�XQ�C��                                    Bxb�  �          @��?�����33@�\A�{C���?�������?�z�A^�HC��
                                    Bxb��  �          @���>�����(�@G
=B�C���>�����G�?��HA�p�C�g�                                    Bxb`  "          @��=�G���(�@*�HA�ffC���=�G���p�?�(�A��RC��                                    Bxb  T          @��>�����\)@�A�G�C�e>������\?E�Az�C�9�                                    Bxb!�  
(          @�(�?&ff��G�?У�A��C�ٚ?&ff����>�p�@�(�C���                                    Bxb0R  T          @��>�ff��\)@
�HA͙�C��
>�ff��(�?xQ�A2�HC��)                                    Bxb>�  T          @�\)?��H��@33A�{C�
?��H���
?�{AE�C�S3                                    BxbM�  T          @�\)?�z���p�?��HA��HC��?�z�����?J=qA\)C���                                    Bxb\D  T          @��@\)��=q?���A�\)C�3@\)���>\@��C�|)                                    Bxbj�  "          @�p�@&ff��?=p�@��HC��@&ff��
=��ff����C�n                                    Bxby�  "          @�p�@�����?��@��HC�e@������(���Q�C�ff                                    Bxb�6  �          @�z�@!G����>���@�\)C��q@!G���{�J=q��HC�q                                    Bxb��  "          @��
@  ���?E�A��C���@  ��33��G���C�k�                                    Bxb��  �          @�z�@(Q���{>Ǯ@�{C���@(Q���z�E���C�Ǯ                                    Bxb�(  
�          @��@$z���z�>�?�C�~�@$z����ÿ�G��/�
C��=                                    Bxb��  "          @�p�@0����(�?z�@��HC�aH@0����(��z���33C�aH                                    Bxb�t  �          @�
=@G
=���\?�z�ADz�C��3@G
=���=�\)?5C�\)                                    Bxb�  T          @�{@XQ����R>�(�@��C�]q@XQ���{�(���{C�p�                                    Bxb��  
�          @�(�@?\)��p�>k�@   C��@?\)��33�Y����C�B�                                    Bxb�f  �          @�@R�\��  >��R@UC�޸@R�\��ff�8Q���\)C��                                    Bxb  "          @�p�@O\)���R�������C�Ф@O\)�~{��(���{C��                                     Bxb�  �          @�ff@c33�o\)�����
C��@c33�N�R�!����C��H                                    Bxb)X  �          @��R@^�R�w�������  C���@^�R�XQ������=qC���                                    Bxb7�  �          @�
=@N{�{���p���(�C��=@N{�Vff�7
=���C��
                                    BxbF�  
�          @�
=@*�H���8Q���(�C��@*�H���
������{C��)                                    BxbUJ  �          @��@b�\��Q쾨���\��C��@b�\��녿���_
=C��3                                    Bxbc�  "          @�G�@=p���(�����Q�C�L�@=p����
��{���
C�                                    Bxbr�  
Z          @�
=@�R���\��\)�9��C���@�R��(���z��pQ�C�)                                    Bxb�<  �          @�G�@*=q��33>�z�@@��C�` @*=q���ÿ\(���HC���                                    Bxb��  
�          @�G�?�\)��  ����(�C��
?�\)��\)�޸R���RC�f                                    Bxb��  	�          @���?�����������C��?����G����H���
C���                                    Bxb�.  �          @�33@����������ffC���@����
=�ٙ����\C�q�                                    Bxb��  �          @�=q@(�����;8Q��=qC�#�@(����\)�����Z�\C��{                                    Bxb�z  �          @�=q@9����  >B�\?��RC��{@9������h���33C��\                                    Bxb�   
�          @�G�@(Q����
�\)���HC�33@(Q���ff��G��S\)C��q                                    Bxb��  �          @�\)@�������p��w�C�
=@����{���R�|��C��\                                    Bxb�l  x          @���@   ��33����Q�C��H@   ��33�У���=qC�B�                                    Bxb  �          @���@&ff��  �L�Ϳ��RC���@&ff��=q����[�C�,�                                    Bxb�  T          @�33@*=q��p��L�Ϳ\)C�+�@*=q���ÿ�
=�B�HC��f                                    Bxb"^  �          @��@6ff���\>8Q�?�=qC�AH@6ff����n{�z�C�z�                                    Bxb1  �          @��@$z����\>k�@��C�f@$z���  �^�R�ffC�5�                                    Bxb?�  
�          @�@8Q����?uA"ffC�S3@8Q����H��Q�p��C��                                    BxbNP  �          @�
=@E��  ?fffAQ�C��@E���H���
�\(�C���                                    Bxb\�  �          @�ff@������?�G�A+33C�L�@����(���Q�}p�C�                                    Bxbk�  "          @�p�@(Q����?�z�AG�C��q@(Q����R=�G�?�z�C���                                    BxbzB  �          @�{@%���?���AL��C��=@%����>\)?��HC�E                                    Bxb��  T          @�
=@U�u�?��Atz�C���@U��G�?   @�p�C���                                    Bxb��  �          @��R@o\)�dz�?�33A�G�C�&f@o\)�vff?O\)A	�C�{                                    Bxb�4  T          @�G�@\(���z�?���A`��C��{@\(����H>Ǯ@��C�/\                                    Bxb��  �          @�G�@>{��  ?�33A@  C���@>{����=�?�  C�H�                                    BxbÀ  �          @��@:=q����?�  A'�
C�]q@:=q��(�����z�C�\                                    Bxb�&  T          @�G�@Mp���=q?�  A(Q�C�S3@Mp���<��
>�  C��
                                    Bxb��  "          @��\@Z=q���>B�\@ ��C��@Z=q�����E���
=C�7
                                    Bxb�r  "          @�=q@n�R����?=p�@�z�C�c�@n�R���H�#�
��Q�C�*=                                    Bxb�  
(          @�  @Tz����
?:�H@���C�g�@Tz���{�8Q��(�C�33                                    Bxb�  1          @���?�p���������i�C�{?�p���{�p�����C��                                    Bxbd  �          @���?�ff��G��\)���C�,�?�ff�����ٙ����\C��                                    Bxb*
  T          @���?����=q�u�!G�C�?����{���H�I�C�B�                                    Bxb8�  T          @���?ٙ����ý��
�Y��C��=?ٙ���z῜(��K
=C�%                                    BxbGV  �          @��@���\)    <�C�� @���33��\)�8z�C���                                    BxbU�  
�          @�(�@z����
<�>�33C�0�@z���Q쿇��.�\C�n                                    Bxbd�  �          @�(�@!G�����<#�
=���C�Ff@!G���p������.�RC��=                                    BxbsH  �          @��@
=���\<��
>uC�w
@
=��
=����.ffC���                                    Bxb��  
�          @�@<����\)?�G�A#�
C���@<�����H���
�W
=C��                                    Bxb��  �          @��@1���z�?!G�@�p�C���@1������
=���C���                                    Bxb�:  T          @��R@8����(�>�G�@��C�O\@8�����
������HC�Z�                                    Bxb��  	�          @�\)@z���
=>W
=@
=C��@z���z�c�
��
C�(�                                    Bxb��  
�          @�Q�@
�H���>�{@W�C�"�@
�H��Q�G����HC�9�                                    Bxb�,  T          @�\)@���{?   @���C�B�@����
=��ffC�G�                                    Bxb��  
�          @��R@�
��?&ff@�  C��@�
��ff��ff��Q�C��q                                    Bxb�x  T          @���@0������?
=q@�p�C�c�@0����G���\���
C�aH                                    Bxb�  
�          @���@.{��33?   @��RC��@.{��33�����C��                                    Bxb�  
�          @���@/\)����?�@�p�C�>�@/\)��녾��H���\C�9�                                    Bxbj  
�          @�=q@L�����>��
@G
=C��\@L����=q�0���ڏ\C��=                                    Bxb#  
�          @��\@3�
���\>�@�=qC�}q@3�
��=q�
=��=qC��f                                    Bxb1�  
K          @�=q@7���G�?��@�p�C���@7���G����H���
C���                                    Bxb@\  
�          @���@7���\)>��H@���C��R@7���\)�
=q���\C���                                    BxbO  T          @�@.{��ff?��@���C�e@.{��
=��(�����C�Z�                                    Bxb]�  "          @��@4z���=q?�ffA���C���@4z���G�?
=@���C�:�                                    BxblN  �          @��
@�H��?��A;33C�3@�H���=�G�?�C���                                    Bxbz�  T          @��
@Q����?�\@�p�C���@Q���녿���(�C��
                                    Bxb��  T          @�  @L(���\)?У�A�ffC���@L(����R?333@�  C�
=                                    Bxb�@  �          @�G�@C�
��Q�?�A;\)C�aH@C�
����>B�\?�C�H                                    Bxb��  "          @�Q�@H����\)?z�HAz�C��f@H�����\<#�
=�G�C��                                     Bxb��  �          @�Q�@E���=q?0��@ۅC�K�@E������z��7�C�/\                                    Bxb�2  
�          @�{@7����\?J=q@�
=C�Y�@7���z�B�\��C�1�                                    Bxb��  
�          @��R@J=q��{?B�\@�p�C���@J=q��  �8Q����C���                                    Bxb�~  4          @�ff@Mp���p�?z�@���C�>�@Mp���ff��Q��g�C�/\                                    Bxb�$            @�=q@Dz���>�
=@�(�C���@Dz�����\)����C��)                                    Bxb��   �          @�=q@\����p�?\)@���C�4{@\����{��p��hQ�C�&f                                    Bxbp              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb*�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb9b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxbH              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxbV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxbeT              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxbs�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxbڄ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb�*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	#�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	2h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	A              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	O�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	^Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	m               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	{�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	ӊ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	�0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb	��  �          @���@����p  ��Q��_\)C�1�@����hQ쿂�\��C���                                    Bxb	�|  �          @�z����@�A�ffC�S3����
=?�A@��C�\)                                    Bxb
"  �          @��;�=q��  @%�A�z�C�s3��=q���
?�z�A���C��\                                    Bxb
�  "          @�=q?c�
���\@��A�(�C�5�?c�
��?�G�A�
C��q                                    Bxb
+n  
�          @�33>.{����@%�A�C��R>.{����?�A��\C��f                                    Bxb
:  
�          @�(��B�\����@@��B �C��þB�\��\)@Q�A��HC��\                                    Bxb
H�  
�          @���G����R@	��A�ffC����G���Q�?���AC�
C���                                    Bxb
W`  �          @��
�������@(��A�C~쿬����(�?�\A��
C:�                                    Bxb
f  
�          @�(�������@-p�A�=qCtT{�����?�33A�G�Cv@                                     Bxb
t�  "          @��?#�
��(�@��A���C���?#�
��?���Ab�RC��H                                    Bxb
�R  T          @��?�����?���A��C��?�����?uA�HC��\                                    Bxb
��  
�          @�=q?Ǯ��33@�A�C��?Ǯ��z�?��HAK\)C�s3                                    Bxb
��  T          @���>\���@p�A���C�q>\��p�?�ffAV�HC���                                    Bxb
�D  
�          @�Q�>�(�����?�A��
C�C�>�(�����?uAC�*=                                    Bxb
��            @��E����R@   A�z�C����E���
=?�=qA0z�C��)                                    Bxb
̐  T          @���ff��{?��A���C�f��ff��{?uA��C�AH                                    Bxb
�6  �          @���G���=q@   A���C�ٚ�G����\?���A+�C��                                    Bxb
��  �          @���z�����@�A�G�C�� �z����\?�Q�A>�RC��                                    Bxb
��  "          @�\)�.{��G�@A���C�\)�.{��=q?�z�A:�\C��=                                    Bxb(  T          @�{�����\@�RAϮC�H�����p�?˅A�{C�t{                                    Bxb�  �          @�(�������?��
A���C�j=����  ?\(�AffC��                                    Bxb$t  �          @�z�aG���  @ ��A�\)C�Ǯ�aG�����?���A3�
C���                                    Bxb3  "          @�p��333����@(�A�C�\�333���?���A��C�N                                    BxbA�  �          @�
=�k�����@*=qA�z�C��\�k�����?��
A�p�C��f                                    BxbPf  "          @���?   ��=q@'�A��
C��3?   ��{?޸RA�G�C���                                    Bxb_  
�          @���>�G�����@G�Bz�C��>�G���33@��A��\C�W
                                    Bxbm�  �          @�ff�xQ����H@0��A陚C�Z�xQ���\)?�A�z�C���                                    Bxb|X  
�          @��������
@ ��AӮC~�������
=?�A��
C޸                                    Bxb��  �          @�z���
��(�@A�B��Cw�
���
��=q@G�A���Cy��                                    Bxb��  T          @��Ϳ����Q�@7
=A�ffCx^������p�@�A�p�Cz�                                    Bxb�J  	�          @�zῡG����?�
=A�G�C�o\��G����?�G�A�HC���                                    Bxb��  �          @�p���Q���?�A��RC���Q����?k�A\)C��                                    BxbŖ  "          @����=q��G�@#33A�z�C|�
��=q��z�?�Q�A�ffC}��                                    Bxb�<  "          @�(����H���R@
=A��\C{�{���H���?��RAB{C|�                                    Bxb��  �          @��\��
=��?���A�  Cy�{��
=���?z�HA33Czc�                                    Bxb�  �          @���������
?���A{�Cv������?8Q�@�Cw}q                                    Bxb .  T          @�33��\���@{A���Cz�)��\����?�\)AX��C{�{                                    Bxb�  
�          @��
���H���R@�RA���C~J=���H��  ?�{AVffC{                                    Bxbz  	�          @��ͿaG���{@{A��
C��{�aG�����?��A��RC�E                                    Bxb,   �          @��R?   ��
=@0��A��C��R?   ���H?��A�=qC��=                                    Bxb:�  "          @�  �u��=q@{A�C��3�u��33?�ffAG�
C��
                                    BxbIl  "          @�ff�����@�HA��HC�\�����  ?ǮAv�RC�L�                                    BxbX  
�          @��z����@L(�B�\Cr33�z����@��AÅCts3                                    Bxbf�  "          @����'����\@C33A�Q�Cn�)�'���Q�@z�A���Cq33                                    Bxbu^  "          @�p��2�\���H@:=qA�G�Cm33�2�\���@�A�
=Co}q                                    Bxb�  T          @�p��>�R��33@A�A���Cj�>�R����@A��HCl�                                    Bxb��  �          @���	������@q�B  Cr
=�	����33@EA��HCt�                                    Bxb�P  "          @�Q�У���Q�@�p�B/�Cw��У����
@_\)B��Cz�)                                    Bxb��  
�          @��׿8Q����@`  B  C��8Q�����@,��A�Q�C�&f                                    Bxb��  �          @��ÿ����=q@z=qB#�\C|�����(�@L(�B �HC~��                                    Bxb�B  
�          @�Q���q�@�Q�B4�Cs�������@g�B{Cvc�                                    Bxb��  
Z          @�  �G��l(�@��B4��CqL��G����@g�B�HCt�H                                    Bxb�  �          @��R��33�z�H@�=qB,G�Ct���33��  @Z=qB��Cw�                                    Bxb�4  
Z          @����G��}p�@�Q�B+Q�Cu�q��G�����@VffB
=qCx�                                     Bxb�  T          @��
��{�u�@���B3�Cw:��{��@`��B�Cz
=                                    Bxb�  	�          @��Ϳ�Q����H@\)B*\)Cz�f��Q����@S33B�C|�)                                    Bxb%&  
�          @\��=q����@�A�{C�4{��=q���H?�=qAx��C��                                    Bxb3�  T          @Ǯ�Tz��\?���A#\)C���Tz���p�>#�
?�
=C�.                                    BxbBr  "          @���������H?���AO�
C�|)������\)>��@��
C���                                    BxbQ  �          @�p��������?��A"ffC  �������R>.{?�\)C=q                                    Bxb_�  �          @�=q��\)����?���A]��C�����\)��p�?��@��RC��                                    Bxbnd  "          @�=q�+�����?��HA8z�C����+���Q�>���@333C��                                    Bxb}
  "          @�녿@  ��{?G�@�RC�e�@  ��������RC�l�                                    Bxb��  �          @�녾�p�����?�
=A7�
C�+���p����>�z�@0��C�4{                                    Bxb�V  "          @��
?(�����?�z�A��C�f?(����?�G�Az�C���                                    Bxb��  �          @�=q?@  ���@Q�A�G�C��=?@  ���?�G�A@Q�C��                                     Bxb��  
�          @�G��������?���AL��C��{�����>�(�@�z�C��H                                    Bxb�H  �          @���p����R?
=q@�
=CxR��p���\)�����J�HC��                                    Bxb��  
Z          @�\)��=q���=���?�  C��Ϳ�=q��Q�E���{C�~�                                    Bxb�  �          @�\)���R��Q�>�(�@��C� ���R��Q��G���ffC�                                     Bxb�:  
Z          @�33�޸R���\>L��?�z�C}k��޸R�����+���33C}Q�                                    Bxb �  
7          @��׿�(���  �u�\)C}ff��(����k����C}0�                                    Bxb�  v          @\������p�>k�@��C��{������z�(����\)C���                                    Bxb,  �          @�
=��=q����=�?�C�����=q��  �=p���C�~�                                    Bxb,�  "          @�
=�������
?h��AG�C�\������{<�>�z�C�{                                    Bxb;x  �          @�ff�u��  ?�{A,��C�Ff�u��33>�  @��C�Z�                                    BxbJ  �          @�  �(����=q?��\AC�C��f�(����{>Ǯ@p  C��
                                    BxbX�  
�          @�33>��R��G�>��
@@��C�w
>��R���׿z���\)C�xR                                    Bxbgj  T          @���>���\?+�@�  C�9�>���Å�u�p�C�7
                                    Bxbv  T          @��
?���G�?B�\@��C��?����H�\)���
C��                                     Bxb��  
�          @�(�>�p����R?��A!G�C��>�p�����>.{?��C��q                                    Bxb�\  �          @�{>����H?J=q@�C�.>���z��G���ffC�*=                                    Bxb�  T          @�p�?333����?�@��C�K�?333��녾�p��\��C�H�                                    Bxb��  �          @�z�?�����
=?\)@���C�` ?�����\)���
�>�RC�\)                                    Bxb�N  
�          @��H?�����@p  B p�C��R?�����@E�A��
C��{                                    Bxb��  
�          @Å>�=q�+�@�z�B}�
C��>�=q�\(�@�ffB\z�C�H�                                    Bxbܚ  
Z          @Ǯ?}p���@~{B�C���?}p���ff@O\)A�C�^�                                    Bxb�@  �          @ȣ�?�=q��@)��Aȣ�C��?�=q��Q�?�=qA�ffC��                                    Bxb��  �          @ʏ\?�=q��p�@33A��C��{?�=q���?���A/�
C���                                    Bxb�  
�          @�33?��
��?�ffA=G�C�T{?��
���>�(�@vffC�+�                                    Bxb2  "          @��?�33�Å?���A�
C�XR?�33��ff>.{?���C�E                                    Bxb%�  �          @Ǯ?Y����z�?�\@�ffC��\?Y�����;\�`��C��                                    Bxb4~  "          @�
=?&ff�������{C�H?&ff�\��G��p�C��                                    BxbC$  T          @�z�>�33���\@%A�G�C��>�33��z�?�\A�G�C���                                    BxbQ�  "          @Å��z����R@G�A��C��쾔z���{?�
=A3�C��                                     Bxb`p  
�          @�33��z���ff?�G�A?\)C��H��z���=q>���@n�RC���                                    Bxbo  
�          @Å��33��
=?�=qA#33C�T{��33���>W
=?��RC�Z�                                    Bxb}�  �          @Å=���=q?�\@��HC��=��\��Q��W�C��                                    Bxb�b  �          @Å<#�
���H>Ǯ@j=qC�\<#�
�\����G�C�\                                    Bxb�  �          @�33�k����������5C�箾k���ff��z��0Q�C��                                    Bxb��  �          @��
���
���ÿaG����C��H���
��33��p����C�t{                                    Bxb�T  v          @�33����=q��  ��C�` ����
=��{�(z�C�]q                                    Bxb��  "          @�z�>�=q���
>L��?�\)C�AH>�=q���H�+�����C�C�                                    Bxbՠ  �          @�zὣ�
��{?�(�A_
=C������
���H?�R@���C��)                                    Bxb�F  "          @�
=>�G���p�>�33@P  C�>�G���������C�f                                    Bxb��  
�          @���>�p�����?��\A��C��q>�p����
>#�
?���C��
                                    Bxb�  �          @�
=?����ff?ǮAj�\C�޸?���Å?5@��HC�˅                                    Bxb8  
�          @�  >�{��p�?�
=A��\C��f>�{��(�?�=qA�
C��R                                    Bxb�  T          @��
������Q�Tz���
=C�f�������H�У��|��CxR                                    Bxb-�  T          @ə��%�������RCs���%����C33��(�Cq��                                    Bxb<*  
�          @���?�ff��33@%�AƏ\C�{?�ff���?�ffA�z�C���                                    BxbJ�  	�          @�ff?�����G�@=qA���C��)?�����=q?�{Ap��C���                                    BxbYv  T          @Ǯ?��
���@��A��C�%?��
��{?�=qAk�C���                                    Bxbh  �          @�33?�\)��  @!G�A�p�C�T{?�\)����?�G�A��C��3                                    Bxbv�  
|          @�?�\��G�@>�RA�=qC�'�?�\���@�A�C�z�                                    Bxb�h  T          @�ff?�p����@.{AУ�C���?�p���{?���A��RC�L�                                    Bxb�  
j          @�(�?=p���(�@A�z�C��\?=p����
?���AQ�C��                                    Bxb��  
�          @�ff?�����@\)A��\C�  ?���=q?ٙ�A�  C��{                                    Bxb�Z  
�          @�\)?\����@��A�p�C��?\����?���Ao�C�T{                                    Bxb�   "          @�?���=q?�A�p�C��?�����?���A)G�C��{                                    BxbΦ  �          @���@
=��(�@Y��B
=C�E@
=���@.�RAӮC�W
                                    Bxb�L  v          @��?�=q����@33A��C���?�=q���?�G�Ac\)C�o\                                    Bxb��  "          @��
@���G�@E�A�C��@���p�@��A��C�'�                                    Bxb��  �          @��@��ff@333A؏\C���@����@A���C���                                    Bxb	>  "          @�p�@Dz�����@.{A�C�@Dz����@z�A��\C�                                      Bxb�  
�          @�@#33����@z�A�ffC�k�@#33��=q?���As
=C��3                                    Bxb&�  v          @�ff@'
=��
=@�\A�z�C�C�@'
=��ff?��AC
=C��f                                    Bxb50  
�          @�
=@3�
��p�@   A�ffC�@ @3�
��z�?��\A=�C��q                                    BxbC�  "          @�G�@)����{?�  A�
=C��@)����(�?}p�A  C��                                     BxbR|  T          @�  @!G�����?��HAX(�C�Ff@!G���p�?0��@��C��q                                    Bxba"  
�          @�ff@ �����?=p�@��C�{@ �����ͼ�����C��)                                    Bxbo�  "          @�
=@&ff���\?=p�@��HC�y�@&ff��(������
C�aH                                    Bxb~n  
�          @�
=@G���
=?���AE�C��
@G���33?�@�(�C��                                    Bxb�  
�          @�p�?�
=��=q?�(�A�(�C��3?�
=��  ?p��A�C�J=                                    Bxb��  "          @ȣ�@33��
=?��A   C���@33��=q>�z�@,(�C���                                    Bxb�`  T          @ə�@ff���R?�G�A��C�.@ff��G�>aG�@G�C��                                    Bxb�  �          @��H@�����?uA�
C��3@���(�>#�
?�Q�C���                                    BxbǬ  �          @��
@$z�����>���@?\)C��\@$z���G�����{C��3                                    Bxb�R  �          @θR@`  ����G���p�C��{@`  ���R��  �W\)C��R                                    Bxb��  �          @�p�@G����H�B�\���HC��3@G����׿s33���C���                                    Bxb�  �          @��H@%���?��
A�
=C�� @%��?��\AQ�C�@                                     BxbD  T          @�  @  ���?h��A�C��{@  ���=�?���C��
                                    Bxb�  T          @Ǯ?�����?5@љ�C�q�?����ff���Ϳp��C�c�                                    Bxb�  "          @���@3�
���H>L��?�\)C�U�@3�
��=q�\)���
C�b�                                    Bxb.6  �          @�(�@ ������?J=q@��C���@ ����=q    �#�
C���                                    Bxb<�  2          @ə�?���
==u?�C�Ф?���p��B�\��  C��                                     BxbK�            @���?��\��G���G��{C��\?��\���H�������\C��
                                    BxbZ(  
�          @�33@H����
=@��A�p�C�#�@H����  ?ٙ�Axz�C�p�                                    Bxbh�  �          @ƸR@AG�����@��A��C���@AG����?�=qAlQ�C�,�                                    Bxbwt  "          @�ff@~�R��Q�@   A��HC�c�@~�R���?�
=A�C�Z�                                    Bxb�  "          @�Q�@R�\���R@p�A���C�z�@R�\��  ?�ffA��C��\                                    Bxb��  �          @���@QG���p�@A�A�\C�Ff@QG���G�@��A�C�0�                                    Bxb�f  �          @�  @^{��Q�@=p�A��HC��
@^{���
@
=A�C�t{                                    Bxb�  �          @��
@k��{�@Tz�A�33C��\@k����H@0  AΣ�C�"�                                    Bxb��            @�p�@a���p�@XQ�A�=qC�'�@a����H@2�\A��
C��\                                    Bxb�X  
(          @�@tz��w�@Z�HB z�C�L�@tz�����@7
=A�p�C��\                                    Bxb��             @�p�@��:=q@�Q�B\)C�XR@��[�@e�BG�C�"�                                    Bxb�  v          @�p�@u�~�R@n{BC���@u��ff@I��A�C�e                                    Bxb�J  	�          @Ӆ@j�H���H@i��Bp�C��@j�H����@Dz�A�p�C�o\                                    Bxb	�  "          @��
@�Q��z�H@L(�A��C���@�Q���=q@(Q�A��
C�H�                                    Bxb�  "          @�ff@�p����H@QG�A�C�@�p���  @+�A�
=C�h�                                    Bxb'<  "          @�{@�=q��\)@7
=A��C�"�@�=q���\@{A��C��                                    Bxb5�  �          @�
=>\���?�@�33C���>\��p���33�I��C��\                                    BxbD�  
�          @θR<��
��(�>��@33C��<��
��33�(���
=C��                                    BxbS.  "          @θR��\)����8Q���{C��3��\)��  ��=q�c\)C���                                    Bxba�  �          @��#�
���
��{��\)C�E�#�
��=q�+��ŅC�:�                                    Bxbpz  �          @�z`\)�����������C�T{��\)����AG����C��                                    Bxb   �          @�녿�(���
=��\)��p�C}=q��(�����(Q�����C|Y�                                    Bxb��  
�          @�G������������{C�\)��������%��z�C�                                    Bxb�l  �          @�G��Q����
�����!�C�33�Q���p������\C��                                    Bxb�  �          @�녿@  �������C����@  ��\)��{��p�C�j=                                    Bxb��  �          @�(��0���Ǯ��\)�"{C��3�0�����ÿ�������C��
                                    Bxb�^  �          @�z�h���ƸR���)p�C��\�h������   ��=qC���                                    Bxb�  �          @�(��z�H����(��0��C�|)�z�H���R�33���
C�P�                                    Bxb�  �          @�{���\��  �����C�b����\���������C�:�                                    Bxb�P  T          @�Q쿢�\�ȣ׿��%�C�G����\����� �����C�{                                    Bxb�  T          @�Q��ff��  �s33��RC�f��ff��녿���\)C�H                                    Bxb�  
�          @�{��ff��ff�J=q��G�C�ÿ�ff���ÿ�\)�iG�C�
                                    Bxb B  �          @ҏ\@p�����?p��A\)C�y�@p����
>\)?��C�XR                                    Bxb.�  �          @Ӆ@w
=���
?\AUG�C�� @w
=����?O\)@�\C��                                    Bxb=�  �          @θR@U����?��A#\)C�� @U��  >���@dz�C��H                                    BxbL4  �          @�p�@U���Q�>�=q@ffC�  @U���  ��\��z�C��                                    BxbZ�  	�          @�=q@~{��Q�?���A<��C���@~{��z�?&ff@��C�.                                    Bxbi�  �          @��
@�{����?��RAH��C��q@�{��?L��@�C�N                                    Bxbx&  �          @ָR@�G����?���A%C��@�G���33?��@�ffC���                                    Bxb��  
�          @��H@���z�?�ffA6�HC�C�@�����?!G�@��RC���                                    Bxb�r  D          @��
@�33��ff?��@��C���@�33��\)������C��3                                    Bxb�  
�          @��@�Q���z�>���@#�
C�Q�@�Q���z᾽p��L(�C�U�                                    Bxb��  �          @���@��
���>��
@2�\C��{@��
��녾�p��QG�C��
                                    Bxb�d  T          @ҏ\@�G����>�z�@#�
C�Y�@�G���\)�Ǯ�W
=C�]q                                    Bxb�
  2          @љ�@�����>\@S33C���@����׾��R�-p�C��                                    Bxbް  v          @�G�@��
����>�p�@R�\C��)@��
��녾��
�1G�C��R                                    Bxb�V  
(          @�z�@�  ��z�>�G�@p��C���@�  ���;L�Ϳ��HC���                                    Bxb��  
�          @ָR@�=q����>���@Z=qC��@�=q����u��\C���                                    Bxb
�  T          @�z�@��
��Q�>��R@(��C��@��
��Q쾳33�>�RC�{                                    BxbH  
�          @׮@�����ff>�p�@J�HC�L�@������R���R�&ffC�H�                                    Bxb'�  �          @��@�����\>��@{C�'�@����녾�G��p��C�0�                                    Bxb6�  "          @��@�����p�?
=@��C�aH@�����ff�����\C�K�                                    BxbE:  "          @��H@}p���  ?�@�\)C��H@}p����þ����ffC�n                                    BxbS�  �          @ҏ\@s�
��33>���@:=qC���@s�
��33�����_\)C��q                                    Bxbb�  �          @�
=@o\)���?&ff@�G�C���@o\)���ý�\)�!G�C���                                    Bxbq,  "          @�@��H����?��HA.=qC�9�@��H��p�?
=q@��C��                                     Bxb�  �          @�p�@�\)��{?���A��C�  @�\)����>�G�@}p�C��\                                    Bxb�x  �          @�p�@��H��{>���@*=qC��\@��H���\�Z=qC��3                                    Bxb�  �          @�\)@�p���{>\)?�(�C�{@�p�����
=q���C�(�                                    Bxb��  T          @��H@�����p������G�C���@���������  �0  C�H�                                    Bxb�j  T          @�33@�33������
�0��C�=q@�33��녿��\��HC�~�                                    Bxb�  �          @�=q@hQ����R>��@33C�Ф@hQ���{���H��G�C���                                    Bxb׶  �          @��H@l����p�>��
@4z�C�'�@l����p���(��n{C�,�                                    Bxb�\  �          @У�@Z�H��������:=qC���@Z�H��zῊ=q�{C�0�                                    Bxb�  T          @�33@g
=���R>k�@�\C�W
@g
=��{�   ���C�c�                                    Bxb�  �          @ə�@^{���>��
@<(�C�@^{��\)�����k�C��f                                    BxbN  �          @�Q�@]p���ff>u@�C�Ф@]p�������{C��)                                    Bxb �  
�          @�  @Y����\)>�\)@&ffC���@Y����
=��G���=qC���                                    Bxb/�  �          @�Q�@Z�H��
=>�=q@�RC��H@Z�H���R����{C��=                                    Bxb>@  �          @�p�@L����  >�Q�@UC���@L����  ��p��^{C��=                                    BxbL�             @�p�@E����H?&ff@��\C�l�@E���(�����C�XR                                    Bxb[�            @�ff@G���=q?n{Az�C��q@G�����>��?��C�t{                                    Bxbj2  T          @�{@:�H��ff?.{@��HC���@:�H��������C�|)                                    Bxbx�  T          @�p�@C�
���H?G�@��C�\)@C�
��z�    ���
C�=q                                    Bxb�~  �          @ʏ\@<(���G�?E�@߮C���@<(���33�#�
��G�C�ٚ                                    Bxb�$  
�          @ə�@1���(�>�z�@'�C�%@1����
�   ���\C�,�                                    Bxb��  !          @ʏ\@(Q����>�z�@'
=C�K�@(Q���\)����ffC�S3                                    Bxb�p  T          @ʏ\@333��>�p�@UC��@333��p���(��xQ�C�q                                    Bxb�  "          @�(�@;�����>���@<(�C��3@;���z�����
=C��R                                    Bxbм  �          @�@E��(�>aG�?�C�aH@E����z�����C�o\                                    Bxb�b  T          @��@G
=���H>�Q�@O\)C���@G
=���H��(��tz�C��                                    Bxb�  T          @θR@HQ���=q?n{A�C��f@HQ�����>\)?�  C�|)                                    Bxb��  �          @�p�@333���?!G�@��HC��@333���׾B�\��Q�C��{                                    BxbT  "          @�{@3�
���?Tz�@�z�C�f@3�
����<#�
=�Q�C���                                    Bxb�  �          @θR@:�H��?uA��C���@:�H��Q�>��?���C�p�                                    Bxb(�  �          @�ff@K�����?n{AG�C��{@K����>\)?��\C���                                    Bxb7F  �          @�@C�
���H?L��@���C�Y�@C�
����        C�:�                                    BxbE�  �          @θR@8����?�  A
=C�}q@8������>B�\?�z�C�Q�                                    BxbT�  �          @�{@Fff��=q?s33A
=C��\@Fff��z�>\)?��C�c�                                    Bxbc8  �          @��@l����Q�?��A (�C��@l�����
>�p�@R�\C�G�                                    Bxbq�  "          @���@`������?�=qAQ�C�~�@`����  >���@(Q�C�B�                                    Bxb��  �          @���@i����p�?�(�A)p�C���@i����G�>�G�@p��C��\                                    Bxb�*  �          @���@~{���?��HA$��C�B�@~{��\)>�(�@j=qC���                                    Bxb��  
�          @�Q�@�������?��HA$��C���@�������>�G�@n{C�]q                                    Bxb�v  �          @�
=@j=q����?�{AQ�C���@j=q���
>��R@(Q�C���                                    Bxb�  �          @�
=@h����Q�?�
=A"{C��{@h�����
>\@N{C�p�                                    Bxb��  �          @�@dz����?�  A,��C���@dz����>�ff@uC�9�                                    Bxb�h  u          @�z�@W����?���A��C��H@W����R>��R@*�HC�Ff                                    Bxb�  T          @�z�@s33���\?���A��C���@s33��{>�33@A�C�y�                                    Bxb��  �          @�p�@~�R��33?���A^ffC���@~�R����?Y��@陚C��                                    BxbZ  �          @��
@~�R��Q�?�(�Ap��C�7
@~�R���R?xQ�A�C��\                                    Bxb   �          @��
@|(�����?�33Ag�C��q@|(����?fff@��C�}q                                    Bxb!�  �          @ҏ\@|(���  ?�=qA^�RC�!H@|(���?W
=@�=qC��f                                    Bxb0L  �          @Ӆ@p����Q�?��\A2=qC�Ǯ@p����z�>��H@���C�s3                                    Bxb>�  T          @�  @�33���?�
=Ao�C�g�@�33��{?xQ�A	�C��
                                    BxbM�  �          @�{@dz���ff?�  A3�C�AH@dz����\>�@���C��\                                    Bxb\>  �          @�
=@}p����
?�ffA^{C��3@}p���G�?Q�@��C��                                    Bxbj�  "          @���@}p���ff?�ffA[�
C�Y�@}p����
?O\)@��
C�޸                                    Bxby�            @���@|(����?��
AX(�C�%@|(����?G�@�33C��                                    Bxb�0  
I          @�  @w
=����?�(�AQC���@w
=��?8Q�@��
C�XR                                    Bxb��  "          @�  @e����?�z�AmG�C�n@e���H?c�
@��HC��3                                    Bxb�|  �          @�G�@q����?ٙ�Ap��C�c�@q���  ?n{A33C��                                     Bxb�"  T          @�G�@|����z�?�A��HC�u�@|����33?���A�HC��)                                    Bxb��  �          @��
@�������?��HA�p�C�}q@�����Q�?�p�A*�HC��                                    Bxb�n  �          @�(�@��H����@ffA��C�S3@��H����?�\)A>�RC��{                                    Bxb�  T          @�@~�R��z�@��A��
C���@~�R���?���AHQ�C��\                                    Bxb�  �          @�\)@������@A��RC�<)@������
?��A8  C���                                    Bxb�`  �          @�(�@x����z�?���Au�C��f@x����33?�  A��C�aH                                    Bxb  �          @�p�@y�����?���AtQ�C��f@y�����
?}p�A  C�b�                                    Bxb�  
�          @�\)@�G����H?���A���C���@�G���=q?�\)Az�C��q                                    Bxb)R  �          @��@}p���  @�A�G�C���@}p���Q�?��A/33C�ٚ                                    Bxb7�  T          @޸R@��H���
@6ffA�G�C���@��H���@A�p�C���                                    BxbF�  �          @��H@�z���
=?�  Ae��C��\@�z���p�?h��@�z�C��                                    BxbUD  "          @�(�@}p�����?�R@��C��@}p���p���\)�{C��H                                    Bxbc�  �          @��@z�H��33?�{AT  C�~�@z�H����?=p�@��HC�3                                    Bxbr�  "          @���@z�H���?�\)AT��C�u�@z�H��G�?@  @ÅC��                                    Bxb�6  �          @��@y������?У�AV=qC�N@y�����\?B�\@��C��                                    Bxb��  �          @�\)@p  ���H?�ApQ�C���@p  ����?p��@��RC�l�                                    Bxb��  
�          @�Q�@u����H?�ffAmp�C�.@u�����?k�@��C���                                    Bxb�(  �          @߮@s�
��=q?�Atz�C�/\@s�
����?xQ�@�\)C��                                    Bxb��  
�          @�\@s�
��z�?���A�(�C�  @s�
���
?���A
�RC�w
                                    Bxb�t  �          @��@s33���H@   A�  C��@s33���\?�\)AffC���                                    Bxb�  �          @�z�@z�H���\@�A�{C���@z�H���\?���A�C��R                                    Bxb��  �          @�@o\)�Ǯ?c�
@޸RC�w
@o\)�ə����
�(��C�W
                                    Bxb�f  �          @��@`  �У�>���@33C�)@`  �Ϯ�5����C�+�                                    Bxb  �          @�z�@`���Ϯ>�p�@6ffC�8R@`����
=�#�
��\)C�B�                                    Bxb�  �          @���@Z=q����?8Q�@�33C�Ǯ@Z=q��녾�����
C��R                                    Bxb"X  �          @�@i����\)?J=q@���C���@i�����þk���\C��q                                    Bxb0�  �          @�
=@o\)��z�?�G�@�Q�C�(�@o\)��
=    <#�
C��                                    Bxb?�  �          @�@c�
��?�{A	�C�|)@c�
�У�=���?@  C�N                                    BxbNJ  �          @�@^�R��\)?��\@��
C�!H@^�R��논#�
��\)C��)                                    Bxb\�  �          @�\)@e��Q�?@  @�  C�n@e�љ���\)�	��C�]q                                    Bxbk�  �          @�@X�����?aG�@���C���@X���Ӆ�#�
��G�C��                                    Bxbz<  �          @���@N{�Ӆ?O\)@�G�C�H@N{���;u��{C��                                    Bxb��  �          @�  @P  ��  ?ǮAG\)C�� @P  ��p�?��@�(�C�q�                                    Bxb��  "          @޸R@N{����@<(�Aȣ�C�l�@N{��G�@33A��C���                                    Bxb�.  �          @ڏ\@L�����R@@��A�C�˅@L�����
@	��A���C��
                                    Bxb��  �          @���@J=q��
=@8��A�ffC��q@J=q���@G�A�G�C���                                    Bxb�z  �          @�ff@X������@/\)A�Q�C���@X����p�?�z�A��RC��                                    Bxb�   �          @��H@N{��{@;�AӅC��=@N{���H@
=A�33C��f                                    Bxb��  �          @�z�@7
=����@P��A�\)C���@7
=��\)@=qA��RC��                                    Bxb�l  �          @Ӆ@=p���@P  A��
C��H@=p���(�@�HA��
C�g�                                    Bxb�  �          @У�@<(�����@HQ�A�C�� @<(����H@33A��C�p�                                    Bxb�  �          @�  @(Q���\)@g
=B��C���@(Q���  @333A��C�P�                                    Bxb^  �          @�{@/\)����@`  B��C�9�@/\)���@,��Aƣ�C��R                                    Bxb*  �          @�  @$z�����@aG�BffC��@$z����@,(�A�ffC���                                    Bxb8�  T          @Ϯ@,(����H@X��A�{C��@,(����\@#�
A�  C�b�                                    BxbGP  �          @��H@7���  @L��A�(�C��@7���ff@A��\C��                                    BxbU�  "          @���@L(���ff?���AG�C���@L(�����>�?�\)C�h�                                    Bxbd�            @���@U���  ?�z�Ag�C���@U���ff?=p�@�(�C�#�                                    BxbsB  �          @�\)@QG����H?0��@�33C��q@QG����
��\)���C���                                    Bxb��  �          @׮@N{����>�G�@qG�C�U�@N{��z������C�Y�                                    Bxb��  �          @�p�@N�R��  ?fff@��C���@N�R��=q�u�
=qC���                                    Bxb�4  �          @׮@\����G�?���A\  C��)@\����
=?(��@���C��\                                    Bxb��  �          @ڏ\@r�\��{@33A��
C��@r�\��  ?�z�A?33C�E                                    Bxb��  �          @ٙ�@z�H���@p�A�p�C���@z�H���?�=qA4Q�C��{                                    Bxb�&  �          @�\)@b�\���R?��HAk�
C�w
@b�\��p�?G�@��C���                                    Bxb��  T          @�@XQ����H?�z�ABffC��3@XQ���  >�@z=qC�9�                                    Bxb�r  �          @ٙ�@a�����?���A7�
C��@a�����>Ǯ@S33C��                                    Bxb�  �          @��@�p���\)@0��A�=qC���@�p����
?��AxQ�C��                                    Bxb�  �          @�G�@xQ���(�@�A���C���@xQ���ff?�33A733C�                                      Bxbd  �          @��@z�H��p�@Q�A�
=C��@z�H��ff?�
=A33C�B�                                    Bxb#
  �          @�(�@|����=q@G�A�C���@|�����\?�ffA�C��                                    Bxb1�  �          @�  @p����{?���AS\)C��{@p����(�?(�@���C�G�                                    Bxb@V  �          @޸R@W
=���R?�  A%�C��R@W
=�\>aG�?�C�w
                                    BxbN�  �          @��
@S�
����?���A ��C���@S�
����>8Q�?�p�C�k�                                    Bxb]�  �          @�{@J�H��=q?��A�C���@J�H��p�=#�
>�33C���                                    BxblH  �          @�G�@E��\)?���A�C�0�@E��=q�#�
����C�f                                    Bxbz�  �          @��@@���ȣ�?h��@�C��{@@�����H�8Q쿽p�C��R                                    Bxb��  �          @���@=p��ȣ�?�z�A�
C��f@=p���(�=u?�C�w
                                    Bxb�:  �          @�=q@B�\��  ?��\A%G�C���@B�\��(�>.{?��C��H                                    Bxb��  �          @�G�@E��{?�G�A$��C�Ff@E��=q>.{?�33C��                                    Bxb��  �          @�G�@R�\����?�z�A8Q�C�Ff@R�\�ƸR>�{@/\)C���                                    Bxb�,  T          @�33@Z=q����?��HA=C���@Z=q�ƸR>Ǯ@G�C�^�                                    Bxb��  �          @��@X����33?��Ay�C��@X�����H?W
=@���C���                                    Bxb�x  �          @�G�@Y����z�?��Al(�C�H@Y���Å?=p�@�Q�C���                                    Bxb�  �          @��@W
=��{?���Ao
=C�@W
=���?@  @�(�C�L�                                    Bxb��  T          @��H@S�
���H@��A�(�C�O\@S�
��(�?���AQ�C��\                                    Bxbj  �          @�@_\)��G�@$z�A��\C���@_\)���?���AXz�C��R                                    Bxb  �          @ۅ@l(���@hQ�B �HC��@l(���  @2�\A�C�5�                                    Bxb*�  �          @�Q�@g��w�@�ffB�C���@g�����@\��A���C�C�                                    Bxb9\  T          @���@w���Q�@\)A�33C�/\@w���=q?��\A)p�C�ff                                    BxbH  �          @�33@r�\��p�@7�A���C���@r�\��33?��HA��HC��f                                    BxbV�  T          @ڏ\@e��p�@EA�\)C��@e��z�@
�HA�  C���                                    BxbeN  �          @׮@`����=q@\(�A���C��H@`�����@$z�A�{C�:�                                    Bxbs�  �          @�\)@\����@\)B��C�@\�����\@K�A�C��                                    Bxb��  �          @�=q@`  ����@n{B=qC�޸@`  ���@6ffAŅC�0�                                    Bxb�@  �          @�{@HQ���p�@g�B(�C��=@HQ����@.{A�{C�j=                                    Bxb��  �          @��@.{����@W
=A��HC�!H@.{��=q@��A��RC��)                                    Bxb��  �          @�(�@6ff����@p  B
�C��q@6ff��  @6ffA��C�:�                                    Bxb�2  �          @Ӆ@HQ����@c33B  C�
@HQ���@)��A��C��{                                    Bxb��  �          @Ӆ@W����
@g�B(�C���@W����R@1G�A�=qC�
                                    Bxb�~  �          @У�@Tz���=q@P  A�
=C��)@Tz����H@
=A��C��                                    Bxb�$  T          @Ϯ@U�����@C�
A�\)C�˅@U���(�@
=qA���C�z�                                    Bxb��  �          @ٙ�@fff��p�@@��A�{C�3@fff��z�@33A���C���                                    Bxbp  �          @�\)@n�R���@UA�RC���@n�R��z�@��A�{C�T{                                    Bxb  �          @߮@o\)��\)@aG�A�=qC�+�@o\)����@%A�  C��)                                    Bxb#�  �          @��@n�R����@EA�p�C��H@n�R��z�@Q�A�Q�C�U�                                    Bxb2b  T          @�
=@�����@#�
A���C�T{@���p�?��A9G�C�Ǯ                                    BxbA  �          @��H@(Q���z�@ffA�33C��3@(Q���
=?�A  C�
=                                    BxbO�  �          @��H@1�����@	��A�G�C�'�@1���{?u@�=qC��=                                    Bxb^T  �          @�@(������@�
A�ffC�e@(���љ�?Y��@ڏ\C��
                                    Bxbl�  �          @�  ?�=q��33?�@�C���?�=q�ڏ\�8Q���p�C��                                    Bxb{�  �          @��?�  ��z�>���@{C�%?�  �ڏ\��  ��C�4{                                    Bxb�F  �          @�ff?��
��
=>�
=@VffC�/\?��
��p��fff��p�C�:�                                    Bxb��  �          @�p�?�(���ff>���@(��C��3?�(���z�}p���{C��                                    Bxb��  �          @��
?Tz���ff�(���  C�c�?Tz��ָR���}p�C���                                    Bxb�8  �          @�=q?\)���Ϳ���/
=C�S3?\)�У��(Q����HC�u�                                    Bxb��  �          @�G�?W
=��ff>��@C�o\?W
=���
������C�y�                                    Bxbӄ  �          @���?(��߮=�?��\C�}q?(���zῚ�H�\)C���                                    Bxb�*  �          @�G�?fff��
==#�
>�Q�C��=?fff��33����)p�C���                                    Bxb��  �          @�G�?�����p��333��C��H?������� ������C��                                    Bxb�v  �          @�
=?u���
�+����\C��{?u�ۅ�����RC���                                    Bxb  �          @�?s33��R��\)�z�C�?s33�ۅ�   ���\C��{                                    Bxb�  �          @�ff?�����׿���33C��3?�����p�������C��                                    Bxb+h  �          @�z�?�(����Ϳ���G�C��=?�(�����=q��ffC��                                    Bxb:  �          @�
=?�����׿!G���33C��?�����Q���\�}�C�N                                    BxbH�  �          @��?ٙ���녾����33C���?ٙ�����(��TQ�C��{                                    BxbWZ  �          @�
=?�
=���=���?E�C��H?�
=���Ϳ�=q�#�C��q                                    Bxbf   �          @�z�?�����
�\)���C�/\?�����   �s33C�n                                    Bxbt�  �          @�(�?�{��z�.{��  C�?�{��
=��\)�D(�C�8R                                    Bxb�L  �          @��?��H����=u>�C�k�?��H��׿�33�(z�C���                                    Bxb��  �          @�\@
=q��Q�>W
=?У�C�G�@
=q�����p���C�c�                                    Bxb��  �          @�?�p���{?��A��C���?�p����þ��R�
=C���                                    Bxb�>  �          @陚?�
=�߮?��A#33C�� ?�
=�����z�HC��                                    Bxb��  �          @�G�?������?�\)A,��C���?������u�   C�|)                                    Bxb̊  �          @��?�  ��p�?�  A>{C��?�  ��\=��
?+�C���                                    Bxb�0  �          @�@  ���?�\)A��C�f@  �߮��z��\)C���                                    Bxb��  �          @��H@#33��ff?��HA8Q�C�n@#33��33=��
?#�
C�5�                                    Bxb�|  �          @�(�@1���{?�z�A0  C�G�@1��ڏ\<�>aG�C�\                                    Bxb "  �          @�p�@333��
=?�A0��C�N@333���
<�>�  C��                                    Bxb �  �          @�R@1G���G�?��A&{C�3@1G����u���HC��                                    Bxb $n  T          @�=q@0  �޸R?���Ap�C�Ǯ@0  ��G���{�&ffC���                                    Bxb 3  �          @�  @(Q���33?�  A9p�C�~�@(Q���Q�=�\)?�C�E                                    Bxb A�  �          @�Q�@(Q���
=@A�33C��\@(Q���  ?+�@��C�J=                                    Bxb P`  �          @��?�����p�@  A��
C��q?����߮?Tz�@�G�C�ff                                    Bxb _  �          @�{?���ٙ�@��A���C�^�?�����
?O\)@ʏ\C�                                    Bxb m�  �          @��@ff�Ӆ@8��A���C��\@ff��=q?��HA3�C�Ff                                    Bxb |R  �          @�  @p���
=@(�A�=qC�  @p���=q?�  @���C���                                    Bxb ��  �          @�\@Q���  ?��RAs�
C�w
@Q���Q�?   @p  C�,�                                    Bxb ��  �          @�@���z�@8��A�  C��{@���33?�Q�A0(�C�K�                                    Bxb �D  �          @��H@�\�Ӆ@>{A��C���@�\��\?\A8��C��{                                    Bxb ��  �          @�R@
=���
@\)A��
C��@
=�߮?��A ��C�Q�                                    Bxb Ő  T          @�R@z���p�?��A>�HC�E@z����H=L��>�
=C��                                    Bxb �6  �          @�{@33��p�?�p�A7�C�5�@33��=q�#�
��\)C�                                    Bxb ��  T          @��H@$z��ᙚ?���A��C���@$z���(���
=�I��C��f                                    Bxb �  �          @�@  ��\)@�HA���C�:�@  ���H?p��@�
=C��=                                    Bxb! (  �          @�@(����@o\)A�  C���@(��޸R@z�A��HC���                                    Bxb!�  �          @�@������@HQ�A�
=C��@����G�?�AK�C��                                    Bxb!t  �          @�@���z�@K�A�  C��)@���p�?�Q�AK�C�>�                                    Bxb!,  T          @�p�@
�H��
=@>�RA�33C��@
�H��ff?�p�A1��C�`                                     Bxb!:�  �          @�p�@�\���@,(�A�=qC�L�@�\��\)?�A�
C�˅                                    Bxb!If  �          @�p�@��ۅ@Q�A���C���@���
=?Y��@��
C�Q�                                    Bxb!X  �          @�@2�\��G�@{A���C�.@2�\��?5@���C��R                                    Bxb!f�  �          @���@;���  @ffA~�\C���@;���G�?
=@��C�N                                    Bxb!uX  �          @�z�@@  �ָR@�
Az�HC��@@  �߮?\)@�{C��
                                    Bxb!��  �          @�
=@ ���˅@?\)A��C�@ ���ۅ?��A>{C��                                    Bxb!��  �          @�=q@"�\���
@Dz�Aƣ�C�C�@"�\����?�z�AQC�u�                                    Bxb!�J  T          @�33@8���Å@3�
A��RC���@8���ҏ\?�z�A1G�C��
                                    Bxb!��  �          @�(�@C33��p�@-p�A���C���@C33�ۅ?��RA��C���                                    Bxb!��  �          @��H@<����Q�@$z�A�(�C�/\@<�����?���A{C���                                    Bxb!�<  �          @��
@N{�Å@L��A���C�� @N{��p�?��
AW�C��                                     Bxb!��  �          @��@AG���33@x��A���C��=@AG���=q@   A���C�\)                                    Bxb!�  �          @�p�@0  ��Q�@��Bz�C�� @0  ���@7
=A�G�C�]q                                    Bxb!�.  �          @�\@:=q��z�@n{A�(�C�(�@:=q��=q@z�A���C���                                    Bxb"�  �          @�@P  ��p�@7�A�{C�ff@P  ���?�p�A9C�t{                                    Bxb"z  �          @���@U����@Z�HA�  C�P�@U�ȣ�@z�A�C�
=                                    Bxb"%   �          @��@l�����\@l��A�p�C��@l���ȣ�@
=A�\)C�Ff                                    Bxb"3�  �          @��@b�\��{@Y��A�
=C��@b�\���@�Az�RC��                                     Bxb"Bl  �          @��@l(���z�@U�A�(�C��{@l(���  ?�(�As
=C�Ff                                    Bxb"Q  �          @�\@xQ���33@U�AЏ\C�W
@xQ��ƸR?�p�Ar{C��)                                    Bxb"_�  �          @�{@u���
@e�A�  C�/\@u����@p�A��C��R                                    Bxb"n^  �          @�p�@~{��@k�A�G�C�R@~{���
@ffA��\C�|)                                    Bxb"}  �          @�  @x����ff@z=qA�{C���@x����ff@#�
A�Q�C��                                    Bxb"��  �          @�33@�����@�  A�
=C�o\@����@*=qA�ffC���                                    Bxb"�P  �          @�Q�@i������@���A�{C�� @i���ə�@+�A��
C��                                    Bxb"��  �          @�\)@7
=��ff@n{A�p�C�޸@7
=���@�A�{C�y�                                    Bxb"��  �          @�33@\)����@{�B�\C�  @\)��G�@#33A�\)C���                                    Bxb"�B  
�          @�@Tz���p�@��B��C��q@Tz��Ǯ@2�\A���C��                                    Bxb"��  �          @�@X����{@���B
�HC���@X����G�@>�RA��C��f                                    Bxb"�  �          @�
=@N�R��z�?�Apz�C�h�@N�R���>Ǯ@>�RC��\                                    Bxb"�4  �          @���@AG��Ӆ?��Ahz�C�G�@AG����
>�z�@�RC��                                     Bxb# �  �          @�z�@O\)�Ӆ?�p�Ap��C��@O\)��z�>\@7
=C��
                                    Bxb#�  �          @�@R�\��Q�?�
=A{C��)@R�\��33��G��Tz�C��
                                    Bxb#&  T          @�\)@X���љ�@z�A�  C��@X�����?8Q�@��C�\                                    Bxb#,�  �          @�\)@Y����ff?�ffAW�
C�y�@Y����>.{?��C�{                                    Bxb#;r  �          @�  @`���ҏ\?�p�An{C�
=@`���ۅ>�p�@0��C���                                    Bxb#J  �          @��@]p���G�@`  AծC���@]p���ff?�(�Aj�\C��                                    Bxb#X�  �          @�G�@U���@�  BQ�C�g�@U��ff@333A�z�C���                                    Bxb#gd  �          @�z�@��H���\@��B��C���@��H��  @L��A�ffC���                                    Bxb#v
  �          @��H@�����33@q�A�p�C�J=@�����33@��A��RC�}q                                    Bxb#��  �          @��
@J�H��G�@{�A�C���@J�H�ٙ�@�A�{C�z�                                    Bxb#�V  �          @��\@$z����@�B	�
C���@$z��ٙ�@8Q�A���C�]q                                    Bxb#��  �          @�(�@3�
��{@�G�BffC�#�@3�
�Ӆ@B�\A�p�C��H                                    Bxb#��  �          @���@���p�@�33A���C�Z�@���\)@\)A���C�>�                                    Bxb#�H  �          @�(�@8Q���\)@�B �C�޸@8Q����@'
=A��C�w
                                    Bxb#��  �          @�(�@U���H@�{B	(�C�p�@U��  @<��A�  C���                                    Bxb#ܔ  �          @��
@P  ����@r�\A�33C��@P  ��G�@��A���C���                                    Bxb#�:  �          @�G�@QG���33@c33A�\)C��@QG�����?��HAiG�C��                                     Bxb#��  �          @�33@HQ���G�@]p�A��C�=q@HQ���{?���AW
=C�#�                                    Bxb$�  �          @�=q@Fff���H@��
A��C���@Fff��p�@$z�A��C�u�                                    Bxb$,  �          @��@E���R@��Bz�C�33@E�Ӆ@4z�A�33C��f                                    Bxb$%�  �          @�\)@AG���=q?�\)Ah��C�]q@AG��ڏ\>L��?\C��3                                    Bxb$4x  �          @�@+���(������B�\C�J=@+�����	����33C��)                                    Bxb$C  �          @�{@!���{?&ff@���C�@!���zΉ���RC�R                                    Bxb$Q�  �          @���@9���ٙ�?���A(��C���@9������33�,��C�W
                                    Bxb$`j  �          @��@3�
��
=@ ��Aw33C�Z�@3�
��Q�>�\)@ffC��                                    Bxb$o  T          @��@p����@��HB��C���@p����@%�A��\C�~�                                    Bxb$}�  
�          @�\@   ��@�z�Bp�C�aH@   ���@H��A�  C���                                    Bxb$�\  �          @�\?���z�@�G�BJG�C�Z�?����R@�  B�C��                                    Bxb$�  �          @���@%��:�H@�  Bc�
C��R@%����H@�=qB4p�C�`                                     Bxb$��  �          @�  @
�H?�33@�B�aHB  @
�H��@陚B��C�\                                    Bxb$�N  �          @�(�@7
=@   @�  B}��B��@7
=>#�
@���B�  @R�\                                    Bxb$��  �          @��?���@Q�@��
B��BP��?���?   @�B�W
Ap                                      Bxb$՚  �          @�Q�@ ��@;�@�(�BzB^p�@ ��?�=q@���B��RA�G�                                    Bxb$�@  �          @�\)?��
@H��@���B\)B�?��
?��
@�
=B��
B4
=                                    Bxb$��  �          @�Q�?�G�@U@�=qBv�B�� ?�G�?��R@�ffB�B�B2�                                    Bxb%�  �          A�
?5@a�@�z�B�  B�33?5?\A ��B�k�B���                                    Bxb%2  �          A��?s33@]p�@��B�\)B�\)?s33?�Q�A ��B��)Bbp�                                    Bxb%�  �          A��?�33@8Q�@��B��B�33?�33?=p�A{B���A�{                                    Bxb%-~  �          A
ff?��\@7�Ap�B�B���?��\?.{A��B��RBp�                                    Bxb%<$  T          A�?��@'�A�B��qB��
?��>���A
{B�
=A�33                                    Bxb%J�  �          Az�?�33@(�A��B��{Bp33?�33>L��A
�\B�ff@��                                    Bxb%Yp  �          Az�?��R@
=qAz�B��B=33?��R��\)A��B�ǮC��q                                    Bxb%h  �          A	�?�G�@'
=A ��B��fBo��?�G�>�
=A
=B��Aw�
                                    Bxb%v�  �          A�\?}p�@P��@�z�B��=B�33?}p�?��H@��B��)BJ�H                                    Bxb%�b  �          @�z�?E�@O\)@���B�\B�8R?E�?�  @�  B��=Bi\)                                    Bxb%�  �          A�?@  @Tz�@��B�
=B�p�?@  ?��
@�B�BnG�                                    Bxb%��  �          @�ff=L��@Vff@�{B�  B��=L��?��@�=qB��\B�L�                                    Bxb%�T  �          A z�>�33@P��@�=qB��RB��R>�33?��H@�p�B��fB��                                    Bxb%��  �          @���\)@�G�@��BI�B垸��\)@�R@ǮB~=qB�
=                                    Bxb%Π  �          @�녽�\)@|��@���BU�B�{��\)@��@��B�\)B�k�                                    Bxb%�F  �          @�\)���
@7
=@ÅBt��B����
?���@���B��)C��                                    Bxb%��  �          @�33�G�@N�R@���BY�HB���G�?��@�{B�\)C&f                                    Bxb%��  T          @�=q�0��@4z�@�ffBS\)Cff�0��?���@���Bx33Cff                                    Bxb&	8  T          @��þ�@}p�@�\)Ba33B�W
��@p�@ڏ\B�\)B˨�                                    Bxb&�  �          @��R��Q�@x��@���BnG�B��\��Q�?��H@�ffB��B��
                                    Bxb&&�  �          A\)�#�
@�=q@��Bm�B�.�#�
@��A�\B��
B�\)                                    Bxb&5*  �          A=q�.{@y��@��
Bu
=B�#׾.{?�=q@��B���B��f                                    Bxb&C�  �          @�(��
=q@n{@�p�Bv  BĔ{�
=q?ٙ�@�p�B�B�B�                                    Bxb&Rv  �          @�\)�
=@c�
@��HBx�
B��f�
=?Ǯ@�B�B�u�                                    Bxb&a  �          @�  �L��@I��@׮B��B�uÿL��?���@��HB��
B�z�                                    Bxb&o�  �          @�=q?�  >��H@�Q�B���A{33?�  ��ff@���B�Q�C��f                                    Bxb&~h  �          @�z�@aG��u@���Bg  C�\)@aG�� ��@�G�BK33C�C�                                    Bxb&�  �          @�  @�  �H��@;�A�  C�Ff@�  �tz�?��HA�z�C��
                                    Bxb&��  �          @���@���  ?���AG�C���@����ͽ�Q�=p�C�                                      Bxb&�Z  �          @�{@<(����>���@W
=C�\)@<(�����aG��z�C���                                    Bxb&�   T          @�33@Y�����R����_\)C�xR@Y���j�H�p���
=C�e                                    Bxb&Ǧ  �          @�=q@�\)����O\)��
=C�w
@�\)�vff���H����C�ٚ                                    Bxb&�L  �          @�=q@����p��}p��z�C�q�@���~{�����C���                                    Bxb&��  �          @�ff@�p����
��Q��s�C��@�p��}p��:=q�ָRC�<)                                    Bxb&�  �          @�=q@u�����=q���HC��
@u�~{�C�
��  C��                                    Bxb'>  �          @�p�@s�
����  ��33C��f@s�
�s33�]p���HC���                                    Bxb'�  �          @�33@p�����
�=p��͙�C�Ф@p���vff���R�33C�(�                                    Bxb'�  �          @���@W���
=�7
=��ffC���@W��_\)��  �{C�                                    Bxb'.0  �          @Ǯ@K����R�C33��
=C��@K��Z�H���'p�C�~�                                    Bxb'<�  �          @ڏ\@^{����J�H��ffC�� @^{�s�
���"��C�&f                                    Bxb'K|  �          @��@\����33��Q���
C���@\���aG���
=�:��C�9�                                    Bxb'Z"  �          @�Q�@vff�����z�H���C���@vff�n�R��ff�0p�C���                                    Bxb'h�  T          @�@������b�\���C�p�@���|(�����!(�C�G�                                    Bxb'wn  
�          @��@|����ff�E��ffC�K�@|���w
=��(����C��3                                    Bxb'�  �          @�z�@w
=�����
=��  C�3@w
=�Z�H�^�R���C�AH                                    Bxb'��  �          @���@�Q���녿xQ��C��=@�Q����H�\)����C�8R                                    Bxb'�`  �          @��@�G��n{���\�J=qC���@�G��L(���\���C��                                    Bxb'�  �          @�G�@�
=��@���B�\C��@�
=��@aG�B�C��                                    Bxb'��  �          @�\)@����<(�?�Ak�
C���@����N{?�@���C�Y�                                    Bxb'�R  �          @�(�@������?�33AC
=C���@�����G�>.{?�
=C��3                                    Bxb'��  �          @��@��H��H@?\)AхC�N@��H�K�@
�HA��HC�,�                                    Bxb'�  �          @׮@��þB�\@h��B
\)C���@��ÿ��R@[�BG�C�e                                    Bxb'�D  �          @��@���?}p�@�(�B,33AN�R@������@�\)B1�C�J=                                    Bxb(	�  �          @���@���u@��B,(�C���@����  @�33BC�
                                    Bxb(�  T          @��H@�녿=p�@�B)
=C���@��� ��@���B\)C���                                    Bxb('6  T          @ʏ\@s33?J=q@���BKz�A;\)@s33�&ff@�G�BLC�+�                                    Bxb(5�  T          @�(�@fff@33@���BFffB�\@fff?:�H@�
=B`��A6�R                                    Bxb(D�  �          @�=q@���~�R?��AS�C�j=@����  >���@%�C�xR                                    Bxb(S(  �          @߮@�Q���\)��\)�33C��3@�Q����Ϳ���{�
C��\                                    Bxb(a�  �          @޸R@�
=���\=��
?0��C���@�
=�����  �G
=C��3                                    Bxb(pt  �          @޸R@����ff���H��Q�C���@�����H��{�w�
C��{                                    Bxb(  �          @��@�
=����
=�^{C�3@�
=���H��ff�m�C�#�                                    Bxb(��  �          @�ff@�  ����E����
C�p�@�  �o\)��(���Q�C��H                                    Bxb(�f  �          @�=q@��
������  �
=C�xR@��
�dz��
=q��z�C�/\                                    Bxb(�  �          @�p�@�{�x�ÿ�p��1�C���@�{�U���G�C��{                                    Bxb(��  �          @�=q@����|(���Q��K33C���@����U��#�
���C�8R                                    Bxb(�X  �          @�
=@�p���(�����f�RC��f@�p��h���A���
=C�\                                    Bxb(��  �          @��H@����
=?�Q�AffC�u�@�������z���C��                                    Bxb(�  �          @��@�������?�G�A Q�C�b�@�����{�#�
���
C��q                                    Bxb(�J  �          @�G�@�������=�\)?\)C�*=@������\�����)��C��                                    Bxb)�  �          @���@�ff���>.{?�G�C���@�ff��\)��=q�#33C�+�                                    Bxb)�  �          @�\)@��\���=u>��HC�� @��\��Q���
�=p�C�b�                                    Bxb) <  �          @��
@�{����>�?�  C�� @�{���\��=q� z�C�.                                    Bxb).�  �          @��@�ff��z�>���@33C��@�ff��Q쿐���	C�g�                                    Bxb)=�  �          @�@��
��ff?8Q�@��RC�H@��
��ff�+����\C���                                    Bxb)L.  �          @��@������?�33A*�RC�f@������\�#�
�L��C�Z�                                    Bxb)Z�  �          @�z�@�G���33?�@�=qC�Y�@�G������^�R���C�}q                                    Bxb)iz  T          @��
@Å��  ��z���C�˅@Å��{��z��JffC��H                                    Bxb)x   �          @�=q@������þ�33�(��C���@�����ff��p��S�C��)                                    Bxb)��  �          @��H@�(����׾����{C�}q@�(�������Z{C��H                                    Bxb)�l  T          @���@����  ��Q�0��C��@����\)�Ǯ�<z�C��R                                    Bxb)�  T          @�R@��\��G�=L��>ǮC�@��\���\��{�*{C��{                                    Bxb)��  �          @��H@�
=��  >�  @   C�K�@�
=��33����C�                                    Bxb)�^  �          @�33@��\��zἣ�
�L��C���@��\�������4��C���                                    Bxb)�  �          @�{@�
=���>L��?���C�XR@�
=��ff��
=��RC�ٚ                                    Bxb)ު  �          @�(�@�{��?   @�=qC��{@�{��(��W
=����C��                                    Bxb)�P  �          @陚@�������?aG�@��C���@�����=q�
=��(�C��f                                    Bxb)��  �          @�G�@�������?�Q�A�\C�>�@�����{���R�(�C��\                                    Bxb*
�  �          @��@�������?(��@��HC�� @������׿c�
��RC��)                                    Bxb*B  �          @�G�@��R@p�@
=qA���A��@��R?�z�@0  A�ffA~{                                    Bxb*'�  �          @��
@�?���@
=A�\)AQG�@�>Ǯ@'�A�  @�33                                    Bxb*6�  �          @��@�\)?�33@\(�Bp�A{\)@�\)>u@mp�Bz�@-p�                                    Bxb*E4  �          @�@��=L��?�33Ao�?\)@������?���AeC��3                                    Bxb*S�  
�          @�  @�=q�fff��\)�p��C��H@�=q�9���+����C���                                    Bxb*b�  �          @ۅ@����p��^�R��=qC�� @��Ϳz�H��  �Q�C��R                                    Bxb*q&  �          @�{@�G���{�8Q�����C�
@�G��W
=�U�����C��                                     Bxb*�  �          @��
@����
�H�s�
�	�C��@��ͿW
=�����(�C�c�                                    Bxb*�r  �          @��@�{�AG���G��ffC���@�{��(������<33C���                                    Bxb*�  �          @�G�@c�
�:�H����4  C�K�@c�
���
�����Z  C�
                                    Bxb*��  �          @Ӆ@8Q��_\)���7�
C��f@8Q�������j\)C��                                    Bxb*�d  T          @�(�@k��Z�H�x����C��f@k���p���G��C�\C�ٚ                                    Bxb*�
  �          @�z�@r�\�>�R��(��"p�C��@r�\��(���(��H��C�aH                                    Bxb*װ  �          @�{@b�\�[����'C���@b�\������T�C�}q                                    Bxb*�V  T          @��@���Q��u�G�C���@�녿�����ff�9��C�Ǯ                                    Bxb*��  �          @�(�@J�H�hQ��Z�H�p�C���@J�H�z���p��A�\C���                                    Bxb+�  T          @�z�@{���(��qG��*{C�aH@{�=��
�~{�4��?�Q�                                    Bxb+H  �          @���?�
=���?�@��C���?�
=��zΐ33�N{C���                                    Bxb+ �  �          @�p�?8Q��@  ��Q��Q�\C���?8Q�\����(�C��
                                    Bxb+/�  �          @��R��
�(Q��7
=���Cd�3��
��ff�c�
�P(�CU�                                     Bxb+>:  �          @��\?h���W
=�hQ��8z�C��{?h�ÿ�p�������C�Z�                                    Bxb+L�  �          @��>����Vff���
�Q��C��3>�����(���\)�C���                                    Bxb+[�  �          @�33?��\�����l(��%�
C�\?��\�&ff��=q�o�RC��3                                    Bxb+j,  �          @��?�������P���ffC��
?����E���G��XC��f                                    Bxb+x�  �          @��H@$z��2�\�|(��8p�C�XR@$zῨ����z��hp�C�l�                                    Bxb+�x  �          @�z�@B�\?��R��\)���A�z�@B�\?�{�J=q�H��A�\)                                    Bxb+�  �          @�{@l��?��
@�G�B>�A���@l�;aG�@�
=BIp�C�B�                                    Bxb+��  �          @�{@W�@:�H@�(�BNp�B#\)@W�?^�R@���Bs�Af�\                                    Bxb+�j  �          @���@�R@�@��B8z�Bm�@�R@  @�p�Bv{B(��                                    Bxb+�  �          @��R@�@��@�p�BH�HB�L�@�@\)@�\B��RB?G�                                    Bxb+ж  �          @��H@.{@��@�ffBz�Bo(�@.{@>�R@���BWp�B>�                                    Bxb+�\  �          @У�@y��@o\)@c33B\)B/33@y��@z�@�33B5�A�Q�                                    Bxb+�  �          @�\)?���@`��>��@z=qB�\?���@N{?�
=A�{B~Q�                                    Bxb+��  �          @ə��'
=?����G��j��C.�'
=@dz����H�1�B�z�                                    Bxb,N  �          @��?��@��
?h��AB��H?��@��@+�A�B�8R                                    Bxb,�  �          A�H?�G�A\)��ff���B�  ?�G�AG�?�ffAG
=B�Ǯ                                    Bxb,(�  �          A	p�?�=qA��?�\)A�B���?�=q@��H@��A���B�u�                                    Bxb,7@  �          A  ?��A(�?�z�@�z�B��?��@�@\)A�B�#�                                    Bxb,E�  �          A	�?��AG�?�33A{B��R?��@�33@��A�p�B��
                                    Bxb,T�  �          A
=@'
=@�?u@У�B��\@'
=@�(�@g
=A˙�B�                                    Bxb,c2  �          A  @\(�@��@�Ay��B�33@\(�@�@�B{Bs��                                    Bxb,q�  �          A\)@�=q@љ�@{�A�  Bh��@�=q@���@�{B-\)BF�H                                    Bxb,�~  �          A��@���?��@�33BQ�AQ�@��þ�(�@�\)B�\C�0�                                    Bxb,�$  �          A��@��>u@}p�A�z�?�33@�׿��@s33A�z�C���                                    Bxb,��  �          A
{@��@{�@�p�B{B
=@��?��R@�
=B3p�A���                                    Bxb,�p  �          A	p�@��@��R@���B=qB%ff@��@�@���B@�A�p�                                    Bxb,�  �          A�@��@S�
@��HB�A��H@��?��@�B9�A[�                                    Bxb,ɼ  �          @�  @��@(Q�@�G�B=qA��
@��?L��@���B7��AG�                                    Bxb,�b  �          @�
=@��\?333@��RB&=q@��
@��\��33@�(�B#�C�g�                                    Bxb,�  �          A��@�{��z�@�A�p�C��H@�{���H@�RA�(�C�*=                                    Bxb,��  �          @�(�@����E@~�RB �RC�^�@�����G�@)��A�{C�{                                    Bxb-T  �          Az�@�G��~�R@tz�A�\)C�Ф@�G���=q@�Ar{C��\                                    Bxb-�  �          A��@�p���Q�@5�A��C��H@�p���z�?!G�@�p�C�ff                                    Bxb-!�  �          A=q@أ����\@C33A���C�<)@أ���33?�z�@�C�5�                                    Bxb-0F  �          AG�@�Q���{@?\)A�  C���@�Q���{?�33@��C��3                                    Bxb->�  �          A�Aff?��?�Q�AN�H@�p�Aff>���@(�Ai@z�                                    Bxb-M�  �          A�A�Ϳ�z�@\)Aa�C�p�A���
=?���A Q�C�@                                     Bxb-\8  �          A�RA	�O\)@33AT��C�O\A	��p�?У�A(��C�q                                    Bxb-j�  �          A��A���G�?s33@�=qC���A���G�>�G�@4z�C�#�                                    Bxb-y�  �          AQ�A{� ��>�@>{C��A{� �׾�G��/\)C��                                   Bxb-�*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb-��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb-�v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb-�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb-��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb-�h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb-�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxb-�            AffA
�R��z�8Q�����C�<)A
�R�g
=�
�H�O�C��
                                    Bxb-�Z  	`          AffA{�p  ��=q��\)C��\A{�Y����{�G�C���                                    Bxb.   	`          A�HA�
�0  �E���G�C��
A�
��\���H�"�\C��                                    Bxb.�  T          A  AG��-p��@  ����C��AG���׿�
=��\C�1�                                    Bxb.)L  
�          AG�AQ��\)�p����  C�aHAQ��  ��Q��ffC��                                    Bxb.7�  T          A��A녿�=q���\���RC�� A녿�=q��{�
=C�\                                    Bxb.F�  �          Ap�Ap���\)������{C�z�Ap�������H�   C��                                    Bxb.U>  T          A��A�\�޸R�h�����C���A�\�����p��
�\C�1�                                    Bxb.c�  "          A Q�A������ff����C�ФA�xQ��G��
ffC�33                                    Bxb.r�  "          A\)A�׿�p���\)��p�C�h�A�׿&ff��p���
C�q                                    Bxb.�0  "          A�HA\)��\)���H��Q�C�C�A\)���ÿ�(���C�޸                                    Bxb.��  T          A!G�A�R���H��  ��ffC���A�R��  ��(���C�!H                                    Bxb.�|  
�          A!��A\)��=q�k����C�/\A\)�fff������p�C�j=                                    Bxb.�"  T          A"�\A zῢ�\��  ���C�aHA z�Q녿�33��p�C���                                    Bxb.��  
�          A#\)A �Ϳ��
�Q����HC��A �Ϳ�\)������ffC���                                    Bxb.�n  �          A"�HA   ��\)�aG����RC�ffA   ��
=��z��   C��H                                    Bxb.�  T          A!�A{������(��C�ٚA{��  ����ǮC���                                    Bxb.�  �          AQ�A  ��<��
>�C�^�A  �녿J=q��  C���                                    Bxb.�`  
�          A(�A�׿�z�����!�C��)A�׿У׿�=q��(�C��                                     Bxb/  
�          A
=A{�\)�fff���C��A{��  ���#
=C���                                    Bxb/�  �          A
=Ap�����fff���HC���Ap���33��p��$��C�C�                                    Bxb/"R  �          A   A  ��p��fff��z�C�C�A  ��  �����z�C���                                    Bxb/0�  
�          A#�A   �����ӅC�C�A   ��{���H��\C���                                    Bxb/?�  "          A#�A\)��Q쿌�����C�z�A\)��녿޸R�G�C�                                    Bxb/ND  
�          A$z�A���;#�
�k�C��A��p�������\)C��                                     Bxb/\�  T          A#�A�H������R��  C�33A�H�
=��Q���C�                                      Bxb/k�  
Z          A$(�A!p���G����333C�\A!p���=q�E����HC���                                    Bxb/z6  
�          A!�A��33=�Q�?�C�&fA녿�
=�.{�xQ�C�xR                                    Bxb/��  �          A"=qA��R>���?�Q�C��HA���\)�L��C��f                                    Bxb/��  
�          A%��A%p��������(��C�'�A%p��u�k���  C�W
                                    Bxb/�(  �          A%p�A$��?:�H��  ����@��A$��?E�=L��>�  @���                                    Bxb/��  
�          A&=qA$��?�=q�Ǯ�
=q@��RA$��?��=L��>�  @ʏ\                                    Bxb/�t  T          A%�A$(�?�(��u��=qA{A$(�?���>���?�A z�                                    Bxb/�  �          A%A$z�?��>k�?�  @�=qA$z�?n{?�R@]p�@���                                    Bxb/��  �          A%�A$  ?=p�?(��@l(�@�33A$  >�(�?c�
@�
=@��                                    Bxb/�f  "          A%G�A$Q�?�G�>��R?�p�@�33A$Q�?O\)?+�@n�R@���                                    Bxb/�  
�          A&�\A%�?n{>�=q?�(�@�(�A%�?B�\?��@S�
@�{                                    Bxb0�  �          A$��A$z�>���>Ǯ@�?���A$z�>��>��H@-p�?Y��                                    Bxb0X  T          A$��A$  ?+�?5@}p�@o\)A$  >�33?h��@��H?�Q�                                    Bxb0)�  �          A%p�A#�
?�Q�?333@x��@��
A#�
?Tz�?���@��
@�z�                                    Bxb08�  "          A&ffA$��?Q�?�z�@�z�@���A$��>��R?���@�z�?�p�                                    Bxb0GJ  
�          A'�A%p�?h��?�33@�ff@��A%p�>��R?�33A��?�Q�                                    Bxb0U�  �          A'33A"�\?c�
@�\AJ�\@���A"�\�u@p�AY��C���                                    Bxb0d�  �          A)p�A'\)?�G�?
=q@:=qA�
A'\)?�Q�?�=q@�33@Ϯ                                    Bxb0s<  �          A&ffA"�H?���?���@�\)A�HA"�H?��?�z�A
=@��H                                    Bxb0��  �          A)G�A$��?�@��AQp�@K�A$�׾���@�AT��C���                                    Bxb0��  
�          A*�HA%�>�33@ ��AY?�A%녿&ff@��ATQ�C�33                                    Bxb0�.  T          A*ffA$��?#�
@"�\A]G�@e�A$�þǮ@%AaC��=                                    Bxb0��  
�          A'�A$��=�Q�?�ffAff>�A$�Ϳ�R?�Q�A��C�G�                                    Bxb0�z  T          A'�A#�>��@
�HA?�@�A#����@
=qA>ffC���                                    Bxb0�   �          A*�\A&{?:�H@33AG\)@���A&{�aG�@��APz�C�c�                                    Bxb0��  
�          A+
=A%G�?��
@#33A\��@��RA%G��#�
@0  An�RC��                                    Bxb0�l  T          A*ffA"ff?�\)@A�A�Q�@�=qA"ff���@N�RA�G�C��{                                    Bxb0�  
�          A*=qA!��?��@Dz�A�(�@���A!���#�
@U�A��C��R                                    Bxb1�  
�          A*ffA!?�
=@AG�A��A�A!>\)@UA�Q�?G�                                    Bxb1^  
�          A&�HA33?:�H@@  A��@�
=A33�   @C33A�(�C��\                                    Bxb1#  �          A'\)A ��=L��@6ffA}G�>�=qA �׿���@(��Aj=qC��3                                    Bxb11�  �          A)�A�?�@P��A��\@@  A��L��@L��A�  C���                                    Bxb1@P  "          A&�HA{@`  @��HA�p�A�(�A{?�{@�z�A��A$��                                    Bxb1N�  h          A(  A
=q@�33@���A���A�G�A
=q@@���B   AY�                                    Bxb1]�  @          A'\)A Q�@���@�z�AӅA�=qA Q�@#�
@���B��A��                                    Bxb1lB  
Z          A,z�A@S�
@�(�A�  A�A?�G�@���B(�@�Q�                                    Bxb1z�  �          A+
=A	�@[�@��HA�p�A��A	�?��@�Q�B=q@��
                                    Bxb1��  
Z          A%p�@�{@p��@���A���A���@�{?�@�
=B�A!�                                    Bxb1�4  T          A%p�@��
@��@��A���Bff@��
@1G�@�33BffA��
                                    Bxb1��  T          A&=qAff@��@��A�Q�Av�\Aff>�
=@��RA�Q�@,��                                    Bxb1��  
�          A&�RAQ�?p��@�G�A�G�@��
AQ�@  @��\A��C��H                                    Bxb1�&  T          A"=qA?�33@���A�G�@�RA�#�
@��
Ȁ\C��q                                    Bxb1��  �          A!�A  ?�ff@tz�A��R@�  A  �\)@{�A�C�Ff                                    Bxb1�r  T          A!p�A\)?�G�@tz�A��A��A\)�\)@�33A�  C��                                    Bxb1�  
�          A ��AG�?�ff@y��A���A3\)AG�=�Q�@�G�A�z�?�                                    Bxb1��  T          A#�Aff@
=q@\)A�{AT��Aff>\@���A�Q�@��                                    Bxb2d  T          A"=qA(�@\)@��A��Aep�A(�>�  @�Q�A�  ?�z�                                    Bxb2
  "          A
=@���?��@�ffBG�AW\)@��Ϳ\)@�\)B=qC��q                                    Bxb2*�  T          A�@��@�H@��B
A�p�@�녽u@���B�C��f                                    Bxb29V  �          A\)@��
@��H@FffA��B
=@��
@w�@��\A��Aׅ                                    Bxb2G�  �          A*=qA   @�(�@�  A��Bz�A   @b�\@�Bp�A���                                    Bxb2V�  
(          A*ff@�\)@��@��RAң�B
=@�\)@L(�@��BffA���                                    Bxb2eH  �          A'�@陚@�ff@��RA�{B  @陚@8Q�@޸RB&\)A�z�                                    Bxb2s�  �          A&{@�(�@��@�  B G�A��
@�(�?�G�@ٙ�B#�
AO�
                                    Bxb2��  �          A�@��H@��@�z�A�  Bz�@��H@.{@�\)B�A��
                                    Bxb2�:  �          AA33@��R@H��A��\A�\A33@3�
@�A�RA�\)                                    Bxb2��  �          A�@��R@��@6ffA�A�33@��R@U�@�33A�z�A�p�                                    Bxb2��  �          A!�A�@R�\@�\)AǅA��A�?��@�ffA��
A
{                                    Bxb2�,  �          AG�@��@G
=@�G�A�
=A��H@��?J=q@�(�BQ�@��                                    Bxb2��  �          A�@���?��@�=qB  A=q@��ÿ�
=@�Q�B
=C�o\                                    Bxb2�x  �          A�@ٙ�=�@�  B.?}p�@ٙ��"�\@�\)B��C��H                                    Bxb2�  �          AQ�@�(�?L��@��
BD�@�G�@�(��z�@�33B:�C�O\                                    Bxb2��  �          A�R@���?�\)@�BN��A�{@��׿�ff@�\)BSz�C�aH                                    Bxb3j  T          A�@��@:=q@��RB"Q�A�=q@��>aG�@�(�B9z�@G�                                    Bxb3  �          A�\@�
=@�Q�@<��A�p�A��@�
=@��@��HA�33A�{                                    Bxb3#�  �          A�
@ʏ\@P  @�G�Bz�Aم@ʏ\?
=@ӅB8�@�33                                    Bxb32\  �          A33@���@h��@��A�=qAޏ\@���?�\)@��BQ�A4Q�                                    Bxb3A  �          A�R@�
=��=q@���A�  C�(�@�
=�/\)@]p�A�ffC��                                    Bxb3O�  �          A�R@θR�@��@�Q�B�RC�~�@θR���@c�
A�\)C�aH                                    Bxb3^N  �          Ap�@����1�@θRB<��C�(�@�����{@��B\)C���                                    Bxb3l�  �          AQ�@�\)�*�H@�z�B\  C��
@�\)��(�@���B33C�3                                    Bxb3{�  �          A�@g
=�n{@߮BU��C�@g
=�ȣ�@��\B�C���                                    Bxb3�@  �          Aff@�R?
=@��B��
AV=q@�R��H@�Q�B{�C��{                                    Bxb3��  �          AG�@���?���@�\)Bp�A��
@��Ϳ�\@�BnQ�C�y�                                    Bxb3��  �          A�\@h��>�\)A33B�@��@h���H��@�\Bfz�C���                                    Bxb3�2  �          Aff@2�\�p�A�RB�(�C�@2�\��  @׮B>�C�l�                                    Bxb3��  T          A33@#�
�&ffA33B�G�C�Q�@#�
���
@ӅB7��C���                                    Bxb3�~  T          A=q@���%�A=qB�C���@�����\@��B8�C�e                                    Bxb3�$  �          Aff@hQ��"�\A   Bt
=C�|)@hQ���z�@ƸRB+G�C�`                                     Bxb3��  �          A�
@u��7
=@�z�BkG�C�� @u���(�@�\)B!�
C��                                    Bxb3�p  �          A33@dz��b�\@�Bc{C���@dz���p�@�Q�B\)C���                                    Bxb4  �          A�\@�=q�*�H@��BK
=C�
@�=q���@�Q�B  C��{                                    Bxb4�  �          AG�@��R����@�p�Bl=qC�!H@��R��(�@�{B/ffC�`                                     Bxb4+b  �          Az�@���
=@���BbffC���@����G�@�33B!C��H                                    Bxb4:  �          A=q@�\)�w�@�33BG�C���@�\)����@@��A��\C�q�                                    Bxb4H�  �          A��@�ff��  @���A�ffC���@�ff���\@#33A�ffC��                                    Bxb4WT  �          A��@�  �L��@���B!�
C���@�  ���H@z�HA�Q�C�+�                                    Bxb4e�  �          A��@��ÿ�\)@�\Bd�C�@�����(�@��HB)�HC�j=                                    Bxb4t�  �          A�R@����@��HBV
=C��
@����G�@�z�B��C�Ff                                    Bxb4�F  �          A33@���J�H@׮B=(�C���@�����R@���A�\)C��{                                    Bxb4��  �          A
=@�z��,(�@�
=B3�
C��
@�z���p�@�\)A�
=C�^�                                    Bxb4��  �          A�\@�33�hQ�@��B$(�C��@�33��=q@z=qAʸRC���                                    Bxb4�8  �          A
=@��H�K�@�B\)C�q@��H��p�@e�A�33C���                                    Bxb4��  �          A�@���6ff@�p�B$p�C���@�����H@�p�A�Q�C�9�                                    Bxb4̄  �          A�@�ff�?\)@�G�BZ��C�G�@�ff��G�@��HB  C���                                    Bxb4�*  �          Ap�@�����\)@�{Bi�C��R@�����  @�z�B3C���                                    Bxb4��  �          A�\@�=q=�\)@��B`?G�@�=q�H��@�(�BDffC�R                                    Bxb4�v  T          A  @��
@.{@�=qB5A�33@��
���
@�33BI��C�q�                                    Bxb5  �          Ap�@�33@�z�@��A�=qB��@�33?��@��B*ffA�{                                    Bxb5�  �          A�\@�@�@��
Aə�A�G�@�@��@��B��A�\)                                    Bxb5$h  �          A\)A�@��@8��A���A�A�@8��@��A��
A�                                    Bxb53  �          AQ�A��@�p�@�AN�RA�33A��@S33@}p�A��A���                                    Bxb5A�  T          A\)A
=@��\?�{@��HA�Q�A
=@���@X��A��A˙�                                    Bxb5PZ  |          A(��A\)@��\?&ff@c33A�
=A\)@�=q@0��Ar�\A���                                    Bxb5_   �          A+�A=q@�33?\(�@��HA�{A=q@�  @<��A�
A£�                                    Bxb5m�  �          A.=qA�H@�?(�@L(�A�{A�H@�ff@)��Ab{A��                                    Bxb5|L  @          A1�Az�@�ff=�?(�A�{Az�@���@�AB=qA˅                                    Bxb5��  
�          A2=qA�
@�����:�HA�\)A�
@�(�?���Ap�A���                                    Bxb5��  �          A0��A"{@��
���
�ٙ�A���A"{@w
=?�(�@�\)A��H                                    Bxb5�>  �          A2�\A-G�@%�#�
�L��AW\)A-G�@
=?�=q@���ADz�                                    Bxb5��  T          A3�
A1��?���z῾�RAz�A1��?У�>�ff@33A��                                    Bxb5Ŋ  �          A4��A3��+�>���@33C�J=A3��G�=��
>\C��                                    Bxb5�0  
�          A2�\A+�
�@
=A.ffC�}qA+�
�2�\?�  @��C��{                                    Bxb5��  �          A-A=q�Q�@Mp�A�C��
A=q���H?�p�@�
=C�޸                                    Bxb5�|  
�          A.=qAp���  @���A��C��RAp��Z=q@FffA��RC�n                                    Bxb6 "  �          A-G�AG����@�G�A��HC���AG��Fff@QG�A��HC�>�                                    Bxb6�  
�          A,Q�A33��  @�Q�A�  C�Q�A33�Dz�@a�A���C�4{                                    Bxb6n  �          A(��Az��8��@��HAΏ\C�"�Az����\@8��A~�HC���                                    Bxb6,  �          A*�HA��.{@�p�A�p�C�ǮA���
=@B�\A�(�C�\                                    Bxb6:�  
�          A+�A=q����@�p�A�{C�` A=q�w�@xQ�A��\C���                                    Bxb6I`  
�          A,Q�A��Y��@�=qA���C�t{A��6ff@~{A��C��3                                    Bxb6X  �          A-G�A��333@���A��C��{A��%@qG�A��RC���                                    Bxb6f�  
�          A-A녿xQ�@��
A�=qC�1�A��1G�@aG�A��C�+�                                    Bxb6uR  �          A0(�A{>u@�  A�(�?���A{��@�G�A�C�(�                                    Bxb6��  �          A4(�AG�>��@ȣ�B�?p��AG��%�@��RA�Q�C�C�                                    Bxb6��  T          A2�HAzῡG�@��BffC��Az��z�H@�\)A߮C�Ff                                    Bxb6�D  h          A3
=A�
�E�@���B�C��\A�
�a�@���A��HC�H�                                    Bxb6��  �          A1p�A�?L��@�=qB  @���A���@���B  C��3                                    Bxb6��  �          A1G�A���33@�=qBQ�C�~�A����
@���A��C�z�                                    Bxb6�6  �          A0��A33��p�@�z�B�HC�+�A33��\)@��\A���C��                                    Bxb6��  T          A/�
A
=q���R@�z�B�\C�RA
=q���@�=qA癚C��                                    Bxb6�  T          A/�
Aff��ff@�33BG�C���Aff���@�(�A�ffC��q                                    Bxb6�(  �          A0  A�
��\@߮B  C��A�
��z�@���A�\)C���                                    Bxb7�  �          A,��@����G�@���B%�C���@�����p�@�{A�C�1�                                    Bxb7t  �          A$(�@��H>�33A33BY�H@Z�H@��H�QG�@��BAQ�C�`                                     Bxb7%  "          A�H@�  >\)A�BZ�?�z�@�  �Y��@�=qB>ffC��                                    Bxb73�  �          Aff@�p�?\A��Bp�A�{@�p��Q�ABg��C�w
                                    Bxb7Bf  �          A@���?�
=Az�Bp{A^�\@����+�A�HB`�HC�^�                                    Bxb7Q  T          A@�G�?���A�RBm��A{�
@�G���RAffBb{C�Q�                                    Bxb7_�  T          A�@���@;�ABa\)B�\@��ÿn{A	�Bv�\C�:�                                    Bxb7nX  �          A
=@mp�@��@���BR�BL�\@mp�?8Q�AQ�B�u�A0(�                                    Bxb7|�  
�          A ��@(��@Q�A�B�p�BL��@(�ÿ��A{B�#�C�
=                                    Bxb7��  
�          A!��@G�@.{A
=B�� B$p�@G�����A33B�L�C�p�                                    Bxb7�J  �          A ��@+�@u�A{Bx�HB\(�@+����A�\B��=C�\                                    Bxb7��  �          A ��@QG�@��
A(�Bi(�BN�@QG��L��A33B���C��=                                    Bxb7��  
�          A"�\?�\)@�z�@�p�B`�B��?�\)?�\)A�B���B�H                                    Bxb7�<  �          A$  @��@�z�A��Bx��Bop�@����\)A33B��C��=                                    Bxb7��  "          A$��@2�\@vffA{By��BX=q@2�\��A=qB���C�<)                                    Bxb7�  T          A$��@*�H@x��A�\Bz��B^Q�@*�H���A
=B�W
C�&f                                    Bxb7�.  T          A"�\@�@��
A��B{p�Bz��@�����A�HB���C��R                                    Bxb8 �  T          A#33?У�@w
=A��B�p�B��?У׿&ffA z�B�#�C�/\                                    Bxb8z  �          A$Q�@   @�33A�RB~�B�H@   �\A ��B�
=C��f                                    Bxb8   �          A"{?�@{�A�B�(�B��R?���A�RB�(�C�                                      Bxb8,�  T          A�
>���@Mp�A��B���B��R>��Ϳ��RA(�B�C��R                                    Bxb8;l  
�          AQ��  @{AG�B�#�B�� ��  ��A�B��Cb
=                                    Bxb8J  @          A��� ��@J�HA�B��\B�Ǯ� �׿��A33B�� CQu�                                    Bxb8X�  "          A�\�
�H@Dz�A�\B�aHB����
�H����A�B�B�CS@                                     Bxb8g^  �          A���+�@  A33B�
=B�B��+���A\)B��C|�\                                    Bxb8v  r          A\)=�G�@��\A�\B�B���=�G�����A��B�8RC�G�                                    Bxb8��  �          A ��>�p�@��RA33Bp�B�z�>�p�?z�A   B�(�BeQ�                                    Bxb8�P  �          A��?   @�(�A��B}��B��H?       AG�B�#�>��                                    Bxb8��  �          A�
?s33@��A�Br�B���?s33>�
=A33B�{A���                                    Bxb8��  �          A&�\?�G�@��A�HBX33B��=?�G�?��RA"�\B�p�B ��                                    Bxb8�B  h          A+�
?���@�=qA��BRz�B���?���?���A'\)B���BE
=                                    Bxb8��            A*{?�{@��A��BW(�B��\?�{?ٙ�A&�\B��fBcff                                    Bxb8܎  �          A*�R?��@�Q�A
=BQ  B��{?��?�=qA%��B��B1=q                                    Bxb8�4  
n          A(Q�@(�@�p�A	p�BZ�HB�Ǯ@(�?��RA#�B�W
A�{                                    Bxb8��  �          A$z�>��H@��\AB|B���>��H<�A#
=B�� @A�                                    Bxb9�  
�          A)�?z�H@�p�A	�BW�B���?z�H?�Q�A&�HB���Bo��                                    Bxb9&  "          A+�
?���@��HAffBE�RB��?���@{A%��B�\)Bs�                                    Bxb9%�  T          A,��?�
=@�\)A{BK{B�\?�
=@�A(  B��=Bz
=                                    Bxb94r  T          A,z�?(�@˅A
�\BVp�B�{?(�?�A)p�B���B���                                    Bxb9C  "          A*{>�G�@љ�Ap�BOQ�B�ff>�G�@�A&=qB��B��q                                    Bxb9Q�  �          A'33>��@��AQ�BP��B��=>��@G�A$(�B��B���                                    Bxb9`d  �          A$�þ��R@�G�A
�\Bez�B�ff���R?���A#�B���B�Q�                                    Bxb9o
  T          A$�׿O\)@�G�A
{Bd33BĞ��O\)?�=qA#33B���B��)                                    Bxb9}�  �          A$�ÿ�=q@�=qAG�BY�B��쿪=q?��HA ��B��qCff                                    Bxb9�V  "          A$�Ϳ�z�@�z�A�BN�B�B���z�?�\)A�B��
C��                                    Bxb9��  
�          A&=q��p�@���@θRB\)BЀ ��p�@���AQ�B�(�B�
=                                    Bxb9��  
(          A&�R�}p�@�
=@�p�B.
=BÏ\�}p�@UA�HB�k�B��                                    Bxb9�H  
(          A)G��.{@�G�A�
Bi�B����.{?fffA(Q�B�B���                                    Bxb9��  �          A-p��VffA�@��HA�\)Bۊ=�Vff@��
A�BA{B�u�                                    Bxb9Ք  
�          A*=q��=qA�R@�A�(�Bď\��=q@�
=Az�BYffB�33                                    Bxb9�:  �          A*�R�&ffA�@�p�AυB�z��&ff@���A�BM�B��f                                    Bxb9��  T          A+33?��H@�G�@�z�B1  B��H?��H@=p�A\)B���BbQ�                                    Bxb:�  
�          A0z�@޸R>�\)A  BGG�@�@޸R�^{@�B/�C��H                                    Bxb:,  "          A0Q�@�{?��RA��BYG�A5�@�{�7
=A�\BK�
C��q                                    Bxb:�  7          A.�\@�G�?Y��A��BE�
@ۅ@�G��>�R@���B6
=C��                                    Bxb:-x  
w          A-��@�p�>�z�@��B:��@\)@�p��R�\@���B%�\C��                                    Bxb:<  �          A,(�@�ff?s33A�BKffAG�@�ff�9��@�33B<=qC�J=                                    Bxb:J�  �          A,z�@�Q�uA��BR��C���@�Q����\@�G�B1��C��R                                    Bxb:Yj  �          A,��@�z�>�AQ�BH�R?�{@�z��g
=@�{B.��C�(�                                    Bxb:h  �          A1��@�33?&ffA��BAff@��H@�33�J=q@�{B/�\C�Y�                                    Bxb:v�  �          A4z�@���#�
A
ffBI�\C��{@���z�H@��RB-z�C��f                                    Bxb:�\  �          A4z�@�׿J=qA��BL��C�˅@����@�B&ffC�&f                                    Bxb:�  �          A4  @��H����A��BO�RC�L�@��H��Q�@�(�B$(�C��H                                    Bxb:��  �          A4��@߮�ٙ�A
=BI��C�,�@߮��ff@�\)B��C��                                    Bxb:�N  T          A2ff@�G��,��AB[�HC���@�G�����@���BffC�Ǯ                                    Bxb:��  T          A0z�@�ff�)��Ap�Bg��C�~�@�ff���H@�(�B!G�C�"�                                    Bxb:Κ  T          A0��@�z��6ffAG�Bg�C�� @�z��أ�@���BQ�C��q                                    Bxb:�@  T          A1�@�Q����@��B
=C��@�Q����@vffA�p�C��\                                    Bxb:��  T          A0��AG�����@��HA�\)C�Z�AG���\)@P��A�33C��                                    Bxb:��  
�          A0z�A�
���@��A��HC�:�A�
��
=@QG�A�(�C��                                    Bxb;	2  �          A0��A33�}p�@�BC�t{A33�ƸR@_\)A�G�C��{                                    Bxb;�  �          A/
=AG����@�z�B��C���AG����@hQ�A�
=C���                                    Bxb;&~  �          A/33@��H���@��B��C���@��H��p�@S33A�ffC�W
                                    Bxb;5$  T          A*�\@�(����@���B�RC��@�(���(�@n{A�{C��                                    Bxb;C�  T          A((�@�G���p�@ָRB��C���@�G����@Z=qA�33C�Ǯ                                    Bxb;Rp  "          A)�@�{����@ǮBz�C��@�{��R@-p�AnffC�H                                    Bxb;a  
�          A(��@����Q�@�
=Bz�C�O\@���ff@��A;
=C��H                                    Bxb;o�  �          A'33@�33��@��RB�C�(�@�33��@
=qA@  C���                                    Bxb;~b  �          A&�\@��H���H@1G�Ay��C���@��H��
�����ÅC�8R                                    Bxb;�  �          A#
=@љ����@UA���C�j=@љ���녾u��{C��q                                    Bxb;��  �          A!�@ۅ���H@X��A�
=C��@ۅ���<��
>�C�b�                                    Bxb;�T  �          A!��@�p��ə�@��A��HC�P�@�p���\)?J=q@�Q�C��)                                    Bxb;��  �          A!�@��
���@��BG�C���@��
��  @$z�Al��C��                                    Bxb;Ǡ  T          A ��@�33��Q�@���A�p�C��{@�33��33?���@���C��                                    Bxb;�F  �          A�R@�z��ȣ�@qG�A�\)C�T{@�z��陚?   @9��C�'�                                    Bxb;��  "          A�R@�Q���Q�@  ARffC�T{@�Q����R���Q�C��
                                    Bxb;�  �          A�R@��R�p�?��
A�\C��@��R�����z��Yp�C�U�                                    Bxb<8  T          A=q@�  �ff?�p�@�C�e@�  � ���.�R���RC��{                                    Bxb<�  T          A@aG����H@��B��C��f@aG���
=?��
AB�\C���                                    Bxb<�  �          A�H?��
�{A33B��\C��?��
����@���B7�HC��
                                    Bxb<.*  �          A�ÿ��
@r�\@Q�A㙚B҅���
@��@g�BX��B�\                                    Bxb<<�  �          A$z��dz�@���=q�z�B�\)�dz�Ap��
�H�H  B��H                                    Bxb<Kv  �          A*=q�N{@�\)����B�HB���N{A�H������\)Bڞ�                                    Bxb<Z  
�          A'�
�<(�@�=q����Zp�B����<(�A
{��{����Bٔ{                                    Bxb<h�  �          A#�
�*=q@�{�G��n��B���*=q@�\)���H�
=B��f                                    Bxb<wh  �          A"=q�\)@fff�33p�B����\)@�����G�� �
BոR                                    Bxb<�  �          A!p��\)@*=q��H�\C��\)@�p����6G�B�
=                                    Bxb<��  
�          Aff���
@���(�� B�\)���
@ə���ff�C�B�8R                                    Bxb<�Z  �          A33���?�ff��H�{B�p����@�����G��N\)B�                                    Bxb<�   7          A!녿���?��R����B�\����@�  ��=q�K{B���                                    Bxb<��  q          A��!G����\�G�¢8RCs�׿!G�@@���
�\�
B˞�                                    Bxb<�L  �          @�G�������H�����NffC�{�����Q���Q�u�C�3                                    Bxb<��  �          @�p�=�\)��ff��p��-�C�g�=�\)�Q���  .C��H                                    Bxb<�  �          @��>�=q�����#�
C�Y�>�=q�333��\)(�C��)                                    Bxb<�>  �          @�=q>�p���  �����z�C���>�p��[��߮�~�C�
=                                    Bxb=	�  �          @��\?�G��������C�B�?�G��:�H��G�  C���                                    Bxb=�  �          A ��?^�R���
���R�{C��q?^�R�vff�߮�rQ�C�]q                                    Bxb='0  T          @��R?��\���
��=q�p�C���?��\�e���
=�u�C�˅                                    Bxb=5�  T          @�
=?�  ��33���R����C�k�?�  �|(���  �h  C�o\                                    Bxb=D|  
�          @�Q��G���ff���
��RC�}q��G��Y����ff�z�C��                                    Bxb=S"  T          @�    ��Q����R�33C���    �S�
�ָR�  C��R                                    Bxb=a�  �          @񙚿�  ���
�����p�C��ÿ�  �333���H�3Cu��                                    Bxb=pn  �          @�p��O\)����qG���C�y��O\)����˅�b\)C�k�                                    Bxb=  "          @�>�������S�
��
=C��=>�����������T{C�~�                                    Bxb=��  �          @���>��H����(��p�C�.>��H�u��Ӆ�nC��                                    Bxb=�`  �          Ap������
��Q��@�C��)���G������HC��H                                    Bxb=�  �          A	녿L���k�����33C��
�L��>k��  ¨C$5�                                    Bxb=��  �          A33��
=�1G��  �{Ck)��
=?����{C                                    Bxb=�R  �          A=q�)��������C[��)��@�\�(�B�Cu�                                    Bxb=��  �          A(����
��(����l{C�ff���
����
�R¥(�CL�                                     Bxb=�  �          A(��!G���33���sG�C����!G���  ��ª�fCI�H                                    Bxb=�D  T          A(���
=�e�����~G�CxG���
=>�Q��	� p�C%                                    Bxb>�  "          A{�z��!G��Q��Cf� �z�?�z���#�CW
                                    Bxb>�  T          @�R�@  ��  ���R�^�HC��\�@  �Tz���(�¢8RCc                                    Bxb> 6  T          @Ӆ�#�
��p��fff��
C���#�
�J�H���qz�C�Q�                                    Bxb>.�  T          @�{�*�H�����B�\��  CtE�*�H�c33����F�
Ch��                                    Bxb>=�  �          @�����ff��z��
=Cw{������p��z��Cd��                                    Bxb>L(  �          @ڏ\��{�}p���Q��X�RC��=��{��  �ָR¢
=C{:�                                    Bxb>Z�  �          @��R>�z�@)����
=.B���>�z�@�G�����'��B�\                                    Bxb>it  "          Az�>��@Vff�{G�B�8R>��@����p��#�HB��                                    Bxb>x  T          A�H=�G�?޸R�  p�B�z�=�G�@����߮�N�B���                                    Bxb>��  
�          A{�B�\@4z��
�R�fB����B�\@�
=��p��3
=B�Q�                                    Bxb>�f  �          A녾��H@X����\�RB�ff���H@ۅ���#�B�#�                                    Bxb>�  T          A33��{@3�
��H�fB�����{@ҏ\����5G�B��
                                    Bxb>��  T          A'��W
=@�\)��R�u�HB�Ǯ�W
=A
{��(��	�B�                                    Bxb>�X  T          A%p�>�  @�z��
{�c�B��H>�  A\)���H��RB�Ǯ                                    Bxb>��  
�          A!녾\)@\�G��T(�B��R�\)A������θRB���                                    Bxb>ޤ  T          A33>�p�@�����H�HB�\)>�p�AG��w
=��(�B�aH                                    Bxb>�J  �          A=q�xQ�@�\)���H�B\)B�LͿxQ�A�O\)��G�B��3                                    Bxb>��  T          A�ý��
��  ��
=�QG�C������
���R� ��  C���                                    Bxb?
�  T          AG�>��ff�G��3C���>�?�Q�����B�(�                                    Bxb?<  �          A=q?5?Q���¦Q�BD\)?5@�(���ff�ep�B�aH                                    Bxb?'�  
�          Az�?xQ�@G��(���B��?xQ�@�������H�B���                                    Bxb?6�            A�\?�(�@-p���R�B��3?�(�@�33��p��7�B�#�                                    Bxb?E.  �          A
=?��@��  �v�B��?��@�=q��G���B�ff                                    Bxb?S�  �          A�R�8Q�@�����nB�(��8Q�A�����B�z�                                    Bxb?bz  �          A  �k�@�Q�� ���hG�B�=q�k�A{���
��\)B�8R                                    Bxb?q   T          A zῸQ�@�
=���\�Q�B�(���Q�Az��z��z�B�8R                                    Bxb?�  �          A!����AQ���G���ffBΨ����A�?�@=p�B�p�                                    Bxb?�l  �          A!��\AG���
=��{B�����\A=q=L��>�z�B�B�                                    Bxb?�  �          A  ����A{��Q���\)B��3����A�>�@3�
B��                                     Bxb?��  "          A�\Aff�tz�����B�uÿ\A�>�(�@*=qBƮ                                    Bxb?�^  T          A�Ϳ���AG������{B�k�����A\)�L�Ϳ�B�B�                                    Bxb?�  �          AQ쿘Q�A�R��=q��=qB�#׿�Q�A
=��ff�'�B�
=                                    Bxb?ת  T          A   ���RA�������	B��)���RAzῪ=q��p�B�Q�                                    Bxb?�P  �          A$Q��AG���  �p�Bʣ׿�A �׿�����p�B��H                                    Bxb?��  
�          A#\)��\)A Q���(��
=B�.��\)A������33BɅ                                    Bxb@�  T          A�׿��H@ҏ\����;��B��f���HA��W�����B�(�                                    Bxb@B  �          A�H�(�@�������A�B��׿(�A
=�g����B��\                                    Bxb@ �  �          A�\�aG�@��R�	p��t�HB�ff�aG�A��  ��\B�#�                                    Bxb@/�  "          A�ͽ�Q�@���z��f�\B��)��Q�A
=��\)��ffB�(�                                    Bxb@>4  
�          A\)>��H@�ff��R�e��B�W
>��HA��������B�W
                                    Bxb@L�  �          Aff>8Q�@�33�{�z�B�B�>8Q�@�p����=qB�aH                                    