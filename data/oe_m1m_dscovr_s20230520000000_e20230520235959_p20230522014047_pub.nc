CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230520000000_e20230520235959_p20230522014047_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-05-22T01:40:47.682Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-05-20T00:00:00.000Z   time_coverage_end         2023-05-20T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�gK�  �          A�p�@��H@Y���i��s��Aƣ�@��HA	p��D  �=33BE�                                    Bx�gZf  �          A��R@���33�#��6  C���@���r�\�H���t33C�l�                                    Bx�gi  T          A�{@����X���{�	z�C�(�@����=q�c��YQ�C��R                                    Bx�gw�  "          A�z�@�=q�M�+33�=qC�=q@�=q��{�k��c��C��                                    Bx�g�X  
�          A�{@���I�.�R�ffC�k�@����z��mG��g��C�h�                                    Bx�g��  �          A�33@�z��2=q�F�\�6G�C�ٚ@�z���
=�z�\�C�AH                                    Bx�g��  �          A��@����2�R�L���<��C���@��������=qaHC�ٚ                                    Bx�g�J  �          A���@����,���Mp��=�C��@�����Q��
=�C�%                                    Bx�g��  �          A�p�@�(��'��Q��Cp�C��=@�(����������C�˅                                    Bx�gϖ  �          A�\)@��\�&ff�Tz��F��C�J=@��\��\)�����C�7
                                    Bx�g�<  T          A�=q@P����H�[�
�S  C��@P�����
����C�,�                                    Bx�g��  �          A��\?�p��{�aG��e�C��H?�p��A������C��
                                    Bx�g��  �          A�G�@R�\�*ff�P���F  C��{@R�\������\��C��H                                    Bx�h
.  �          A�@Fff�+��Q�F\)C�3@Fff�����33W
C���                                    Bx�h�  T          A��@!G��6�H�J�\�<��C�9�@!G���{��
C��H                                    Bx�h'z  �          A�(�@B�\�5��I��;�C�}q@B�\����~�R�3C��                                     Bx�h6   �          A��H@xQ��-p��PQ��B  C��R@xQ����������qC�,�                                    Bx�hD�  �          A���@!���H�_
=�VC�&f@!����H�����qC���                                    Bx�hSl  �          A�  @&ff�*{�U��JC��q@&ff���R���R33C��{                                    Bx�hb  �          A�z�@U�3\)�K�
�=C�L�@U��\)���HC�                                      Bx�hp�  �          A���@P  �4���K��=  C��@P  ���H��  �3C���                                    Bx�h^  �          A���@Tz��4���Lz��=p�C�1�@Tz�������=q�qC��                                    Bx�h�  �          A�
=@dz��2ff�Mp��>��C�޸@dz�������Q�C�                                      Bx�h��  �          A���@e��0���M��?�
C��R@e�������=q=qC�ff                                    Bx�h�P  �          A�ff@a��6=q�Hz��9�C��)@a���  �}p��C�:�                                    Bx�h��  �          A�(�@���<  �?��/��C��f@����G��w33�}C��                                    Bx�hȜ  T          A�  @q��$���R�R�H�C�@q�������=qffC�+�                                    Bx�h�B  �          A��\@��H���l���f(�C�z�@��H�ff����C�H�                                    Bx�h��  
�          A�@�p��أ��v�R�y\)C���@�p��(������B�C��)                                    Bx�h�  �          A��\@�p����Zff�LQ�C���@�p���������=C��                                    Bx�i4  T          A���@�  ��H�e�[ffC�j=@�  �S�
��Q�C��                                    Bx�i�  �          A�{@l���33�i��a��C�9�@l���AG�����W
C�`                                     Bx�i �  T          A���@R�\�{�b�R�Y33C�o\@R�\�s�
����p�C�k�                                    Bx�i/&  T          A�33@��R�=q�d  �Wz�C�>�@��R�U�����ffC��                                    Bx�i=�  �          A�{@�Q��Q��^�H�N��C�f@�Q��s�
����C��                                    Bx�iLr  �          A�Q�@�� ���[�
�J\)C��f@��������B�C���                                    Bx�i[  �          A�=q@��\�\)�_�
�O�C�\)@��\��ff��33\C��                                     Bx�ii�  �          A��
@�{� ���\���L�HC��H@�{���
��(��{C���                                    Bx�ixd  �          A��@�(��=q�bff�S�C�P�@�(��g����R�\C�]q                                    Bx�i�
  "          A��@�z�� z��[�
�K�RC���@�z����
���� C�
                                    Bx�i��  �          A�  @�33��R�_��P�C��@�33�n�R����RC��3                                    Bx�i�V  "          A�Q�@\��ff�q��g�C�R@\���\��G�{C��                                    Bx�i��  T          A���@��H�����z�\�u=qC��H@��H?W
=����=@��                                    Bx�i��  �          A���@�ff����x���u��C��@�ff?���
={A3
=                                    Bx�i�H  �          A��@�����w��mC�s3@�������  �C���                                    Bx�i��  T          A�  @�\)��
=�u���j(�C���@�\)��z�����=C��                                     Bx�i�  �          A��@�  ����s�
�g��C���@�  ����=qp�C��                                     Bx�i�:  
�          A�G�@�����Q��v{�l��C��@��ÿ��R��ffQ�C�                                      Bx�j
�  �          A��H@�33� Q��pz��e=qC���@�33�ff��G�z�C�G�                                    Bx�j�  �          A�
=@��
�	��l���_�\C�\)@��
�/\)��G�{C��                                    Bx�j(,  �          A���@���\)�l���_�C�.@���&ff����\C��                                    Bx�j6�  �          A���@�{�
�H�k
=�`z�C�/\@�{�6ff���\�)C�U�                                    Bx�jEx  T          A�(�@�ff����i�]��C�R@�ff�N�R�����C�                                    Bx�jT  T          A��H@�����^=q�O��C��\@�������Q�{C��H                                    Bx�jb�  T          A�@��H�<(��Lz��5p�C���@��H����
=��C��                                     Bx�jqj  "          A�=q@�G�� z��V�\�G�C���@�G���33���H�C���                                    Bx�j�  "          A�
=@����'�
�@z��7�HC�Ǯ@��������o��~  C��                                    Bx�j��  
Y          A�z�@��\�(���,���)�RC�g�@��\�����]���op�C���                                    Bx�j�\  
�          A{33@@  �B�\�z��33C��\@@  ��\�L(��^�C�R                                    Bx�j�  T          Arff��{�ep���Q����C}�R��{�K�������{C{Ǯ                                    Bx�j��  
�          Av=q�s�
�h  �C�
�8��CG��s�
�G��   ��(�C}                                    Bx�j�N  �          Atz�����h(���ff�أ�C}p�����N�H�����z�C{��                                    Bx�j��  T          A���33�o�
�������\C���33�B=q�$(��(�C�5�                                    Bx�j�  
�          A�=q>�33�e���R���C���>�33�((��K33�I�C��                                    Bx�j�@  �          A��?@  �b�R�
�H����C���?@  �#33�Q��P=qC�R                                    Bx�k�  �          A��\?�ff�O
=���(�C�� ?�ff�
ff�\  �f{C�G�                                    Bx�k�  
�          A��\?���M��$���G�C�l�?���{�b=q�k��C�4{                                    Bx�k!2  
�          A��
?}p��i�	����p�C��?}p��*�\�S
=�KC��                                    Bx�k/�  �          A�p�?�z��\z��(���HC�h�?�z����_��^��C��                                     Bx�k>~  T          A�{?��H�P(��!���C��)?��H�
=q�`���g��C��
                                    Bx�kM$  �          A���?�z��<���1��,�C��?�z������h���|�
C���                                    Bx�k[�  
�          A��
?����TQ����33C��=?����G��Y���_  C��                                    Bx�kjp  !          A�\)?�p��a���G��(�C��q?�p�� ���V�H�R�
C���                                    Bx�ky  U          A�\)?�z��`���������C�޸?�z��!G��Rff�P(�C�Z�                                    Bx�k��  
�          A�?�
=�t(�������{C�8R?�
=�C33�/��'Q�C��                                    Bx�k�b  
�          A��R?�z��{�
��p����\C��?�z��P���!p���C��=                                    Bx�k�  T          A�(�?�\)�x  ��  ��G�C��?�\)�E�4  �(C��
                                    Bx�k��  �          A�=q?�33�yG���33���\C�q?�33�H  �2{�&G�C���                                    Bx�k�T  �          A�p�@��xQ���  ���
C��R@��G��0Q��$=qC��q                                    Bx�k��  T          A�{@��q���
=���
C���@��?�
�0���)  C�H                                    Bx�kߠ  "          A���@\)�O
=�1�� z�C���@\)����m�o��C��=                                    Bx�k�F  �          A�p�@��Y��%����C�B�@���R�e��c\)C�>�                                    Bx�k��  "          A��\?���a����\���C��\?����
�[33�U�RC�Y�                                    Bx�l�  �          A��H@
�H�u����R�ǅC��@
�H�>�H�@���3\)C�%                                    Bx�l8  �          A�  @��t����������C��@��>ff�?��2�HC�/\                                    Bx�l(�  �          A�?��H�ip��	���ffC�ٚ?��H�,(��P���I{C���                                    Bx�l7�  �          A�(�?���j�H�������C�3?���-���P���Hz�C���                                    Bx�lF*  �          A��R?��R�e���\)�\)C�w
?��R�%��Y��R(�C�k�                                    Bx�lT�  �          A��?��R�g\)�33��33C���?��R�)�Q��K{C�                                      Bx�lcv  T          A��?��UG����ffC��?���R�\���_33C��                                    Bx�lr  
�          A��?�(��^�R�ff��C���?�(�� ���Q�P�\C�޸                                    Bx�l��  "          A�>��H�l�������ݮC��3>��H�4  �E���>�C�>�                                    Bx�l�h  
�          A��\�333�p(���{��Q�C����333�<  �:{�2�C�N                                    Bx�l�  �          A�G��8Q��t(���
=��(�C����8Q��@���8(��.�\C���                                    Bx�l��  �          A��ý�G��s
=�׮���C�˅��G��?�
�8  �/(�C��q                                    Bx�l�Z  �          A�\)�k��y��
=����C��3�k��L  �*�H�C�z�                                    Bx�l�   
�          A�\)>.{�yG���\)��ffC�O\>.{�N=q�#33�\)C�`                                     Bx�lئ  T          A��?&ff�s\)��\)��C�:�?&ff�D���,Q��$�
C��                                    Bx�l�L  �          A��@	���c���H��
=C�J=@	���(���K�
�GG�C��H                                    Bx�l��  T          A�G�@�b=q�������C�33@�&�H�L���I�C���                                    Bx�m�  �          A�  @(��Z�R���{C��@(��\)�V�H�U�C�\                                    Bx�m>  �          A�p�?�33�b=q������C���?�33�$Q��Tz��OG�C�:�                                    Bx�m!�  T          A��\@#�
�j�R���أ�C��{@#�
�4(��A���9G�C�h�                                    Bx�m0�  "          A�\)?�  �f�R�
�\���RC���?�  �+
=�O�
�I(�C�H                                    Bx�m?0  
�          A�\)?����i������HC�3?����/��L(��D{C��                                    Bx�mM�  �          A��@J=q�g\)����\C�(�@J=q�.{�Ip��@{C�R                                    Bx�m\|  �          A���@7��dQ��
ff���
C��@7��)��N�R�F�HC��
                                    Bx�mk"  T          A�@)���\�����  C�o\@)����X���S��C���                                    Bx�my�  T          A���@8Q��Z=q��R�=qC��{@8Q����V�H�S�RC�<)                                    Bx�m�n  �          A�ff@�b�\����ffC���@�)��G\)�C�C�7
                                    Bx�m�  �          A��
?˅�f{�����{C�*=?˅�/\)�B�H�>��C�#�                                    Bx�m��  �          A��@J�H�`Q���{��p�C�^�@J�H�)p��B=q�>�C�S3                                    Bx�m�`  �          A�  @0  �c����
��  C�z�@0  �-��B{�=z�C�#�                                    Bx�m�  �          A�=q@P  �b�R�����G�C�t{@P  �,z��A���<G�C�b�                                    Bx�mѬ  �          A���@4z��`Q����\C��@4z��(���Dz��A\)C�xR                                    Bx�m�R  �          A�{@��d(��ۅ��33C��
@��333�2�H�1=qC��
                                    Bx�m��  �          A�z�@9���^�\��{��
=C��@9���+
=�9���9=qC��R                                    Bx�m��  �          A�ff@Dz��Yp����R���
C�^�@Dz��#��?��A�C�]q                                    Bx�nD  T          A�Q�@<(��Zff���
��G�C��@<(��%��>�\�@  C��                                    Bx�n�  T          A�(�@.�R�c���G����HC�p�@.�R�3��1��/�C��R                                    Bx�n)�  
�          A~�H@��hz����H���
C���@��;�
�(  �%
=C���                                    Bx�n86  T          A�?�G��j�H��G����C�l�?�G��>�\�(  �$\)C�7
                                    Bx�nF�  �          A|��@��j�R������C���@��A��{�C�s3                                    Bx�nU�  T          A|��@<���e��������
C��3@<���:{�#�
�!�C�                                      Bx�nd(  �          A{
=@%�`z���Q���(�C�<)@%�2�\�+\)�,G�C��=                                    Bx�nr�  �          A|  @�dz��ʏ\���C�'�@�733�)�)ffC�(�                                    Bx�n�t  �          A{
=@p��eG���
=��
=C�b�@p��:{�$���#��C�b�                                    Bx�n�  �          Ay�?�\�dz����R���C���?�\�9p��$  �$��C�XR                                    Bx�n��  �          Au�?���_\)�Å��  C�%?���3�
�$z��(�RC��f                                    Bx�n�f  �          Au�?��a������{C��H?��8Q���\�!Q�C��)                                    Bx�n�  �          Amp�?����[���  ��ffC�Q�?����4���{���C�f                                    Bx�nʲ  T          Arff?����_���{��ffC�C�?����6ff���"z�C���                                    Bx�n�X  �          Aq��?�
=�^{��(���33C���?�
=�5p��z��!z�C��{                                    Bx�n��  �          Ak�?����W���{��(�C�b�?����/
=�33�%
=C�(�                                    Bx�n��  �          An{?�=q�W�
��  ����C�U�?�=q�-���
�)(�C�!H                                    Bx�oJ  
�          AmG�?��S33��33��\)C��?��&�\�'\)�3z�C��                                    Bx�o�  �          An{?�33�Q��z��ܸRC�
=?�33�#�
�+33�8
=C��                                    Bx�o"�  T          Ar=q?����U��p����
C�Z�?����'��,���6\)C�E                                    Bx�o1<  T          Ar=q?�33�Vff�ڏ\�֣�C��H?�33�(���+��4��C�q�                                    Bx�o?�  �          Avff?����]���  ���
C��q?����1���(���-=qC���                                    Bx�oN�  �          Aw�?��
�b�H����Q�C��3?��
�9G��!���#�C�]q                                    Bx�o].  T          Ar�\@��^=q���H��33C��
@��6�\�
=��HC��)                                    Bx�ok�  �          At  @:�H�\����Q����C���@:�H�4Q����� p�C�C�                                    Bx�ozz  T          Apz�@AG��V�\�\����C�Y�@AG��-����&�C���                                    Bx�o�   �          Ah  @Q��S\)������p�C��@Q��-���Q��z�C�/\                                    Bx�o��  �          Ar=q?�=q�L  �����C�n?�=q�  �2{�Bz�C�,�                                    Bx�o�l  �          AxQ�?�=q�P(���R� (�C�aH?�=q����=G��H�C�&f                                    Bx�o�  
�          Ap  @0  �]����Q���=qC���@0  �9���G��33C���                                    Bx�oø  �          An�R@333�W33���\��  C��@333�/����"�RC�+�                                    Bx�o�^  T          Ag
=@�ff�S�
�5�6�HC�Ǯ@�ff�;33������C��3                                    Bx�o�  T          Ak\)@����W33� �����C�'�@����B�\�\���
C�33                                    Bx�o�  �          AaG�@��R�33@�33A��
C��3@��R�&ff@A�APQ�C��
                                    Bx�o�P  T          Ah(�A���33A  B)�HC��A����@��
A���C�XR                                    Bx�p�  �          AuA���(�A��B\)C���A�� ��@���A�G�C�                                      Bx�p�  �          A~{A���\)A*�\B)�C��
A���R@��A�\)C���                                    Bx�p*B  �          A�\)Ap����HAK33BH  C���Ap���G�A-�B#�C���                                    Bx�p8�  T          A���A"ff��(�AD��B;�C��A"ff�ffA!�B�HC�c�                                    Bx�pG�  "          A��HA#���33ADz�B;33C�+�A#��A ��B�C���                                    Bx�pV4  �          A���A���G�AN�HBIffC���A��  A,z�B��C��H                                    Bx�pd�  T          A��A ����ffAK33BBQ�C���A ���	p�A)p�B�RC���                                    Bx�ps�  �          A�z�A33��z�AN�\BHC��)A33��A+�BC�|)                                    Bx�p�&  #          A��HA33���HAU�BL��C�ٚA33�=qA2ffB"�C�\)                                    Bx�p��  �          A�p�A{���RAXQ�BTQ�C��)A{���A8z�B+��C��f                                    Bx�p�r  �          A��A=q��(�A[\)BS�C�� A=q�G�A;�
B,ffC�8R                                    Bx�p�  T          A�Q�A
=���AS�
BUz�C���A
=��p�A6{B.p�C�=q                                    Bx�p��  
�          A��A
=q�A[33Bc\)C�nA
=q�˅AE��BD�C��3                                    Bx�p�d  �          A�ffA
=���AU�BS
=C��HA
=� ��A6�HB,Q�C�ff                                    Bx�p�
  �          A�G�Ap����A\  BP��C���Ap����A9�B&��C��3                                    Bx�p�  �          A�G�A(���ffA_�BU�C���A(��
�HA>�HB,�HC�n                                    Bx�p�V  �          A�A(�����A\��BQG�C�%A(���RA9p�B&��C���                                    Bx�q�  �          A�ffA33��ffAd  BT�C�.A33��AA�B+��C�=q                                    Bx�q�  �          A���A�H�ָRAZ�\BD�C���A�H�'�A0��B  C�^�                                    Bx�q#H  �          A��Aff��G�AR{B8�RC��=Aff�5p�A#�B�RC���                                    Bx�q1�  �          A�=qA�
��
=AQ�B6��C��{A�
�7�
A!�Bz�C��                                     Bx�q@�  �          A���A�
��ffAR�\B:�C���A�
�4  A$��B
��C�p�                                    Bx�qO:  �          A�=qA�R���
ATQ�B=�C�(�A�R�+�A)�B��C�b�                                    Bx�q]�  �          A�p�A���AD��B-(�C���A��F�HAz�A��C�K�                                    Bx�ql�  �          A�\)Az��,(�A4��B�C�b�Az��Y@�(�A��HC�%                                    Bx�q{,  
�          A��\A(��*{AJ{B+\)C�Y�A(��]�A33A�z�C��f                                    Bx�q��  
�          A��A���0Q�A6{B��C�ФA���^{@���A�{C��q                                    Bx�q�x  �          A���A ���/33A5G�BG�C�C�A ���\��@�(�A���C��                                    Bx�q�  "          A�=qA"=q�.=qA733BG�C�z�A"=q�\(�@���AÙ�C�33                                    Bx�q��  �          A�33A!��-p�A;\)B�C���A!��\z�A ��AɅC�&f                                    Bx�q�j  !          A��A"=q�;�A5G�B�C�nA"=q�h(�@�{A�Q�C�y�                                    Bx�q�  #          A��A2ff�3�A5�B\)C�j=A2ff�`Q�@�\A��C�@                                     Bx�q�  
�          A�\)A-G��:=qA3
=B��C�w
A-G��f{@�33A���C�}q                                    Bx�q�\  
�          A���A8(��"�HA<Q�B��C�AHA8(��RffAG�A�  C���                                    Bx�q�  �          A���A3\)�)��A=��B{C�NA3\)�X��A��A��C��=                                    Bx�r�  �          A�ffA9���*=qA9G�BQ�C���A9���XQ�A z�A�{C�O\                                    Bx�rN  �          A��A<���%��A>�HB��C�]qA<���UG�A\)A�\)C��q                                    Bx�r*�  T          A�Q�A>ff�(�AP  B!�
C��\A>ff�Mp�A�A�\C�k�                                    Bx�r9�  
�          A��AB�R���APz�B"�RC��3AB�R�F=qA�\A��C�@                                     Bx�rH@  U          A��RA;\)����Aj�\B;�\C�NA;\)�0z�A@(�B  C�]q                                    Bx�rV�  "          A�G�A;�
��RAj�HB;G�C�:�A;�
�1G�A@Q�B�RC�T{                                    Bx�re�  �          A���A4  �*ffA?33B��C�K�A4  �Y��A
=A�ffC��                                    Bx�rt2  �          A�ffA:ff��AO�B#�HC��RA:ff�J�\A��A�\)C�O\                                    Bx�r��  �          A�G�A=�(�AO�B%p�C�ǮA=�A�A�A�Q�C�AH                                    Bx�r�~  T          A�\)AAG��{AT(�B$��C�t{AAG��H  A"ffA�=qC��                                    Bx�r�$  T          A�{AE��  AZ�\B'\)C��AE��G�A)G�A��HC�P�                                    Bx�r��  "          A�(�AD����RAZ{B&�\C���AD���J{A((�A���C�q                                    Bx�r�p  �          A��RAHQ����\An�RB<�
C��fAHQ��Q�AK
=B�\C�                                    Bx�r�  �          A���A[
=��Q�A_�B+ffC��
A[
=��\A:�RB
p�C�                                    Bx�rڼ  
�          A�G�AW���{Ao
=B933C�ffAW��
�RAP��B�C��H                                    Bx�r�b  �          A��HAX���y��AuB=��C��RAX��� Q�AZ�HB#�HC���                                    Bx�r�  �          A���AT(����
Ax(�B@�C�aHAT(��  A\(�B%�C��                                    Bx�s�  �          A�
=AR�H���
At��B?�C�P�AR�H�
=AX��B$��C��                                    Bx�sT  �          A��A?
=�
{A[�B,  C�3A?
=�AG�A-�B  C�T{                                    Bx�s#�  
�          A�p�A
=�R=qA�\A��C�}qA
=�rff@�{A���C��                                     Bx�s2�  �          AS�@\���EG����R����C��3@\���8(����\���HC�Y�                                    Bx�sAF  ,          AHQ�@����/33@N{AuG�C���@����6�\=���>��C�Ff                                    Bx�sO�  �          A2�H@o\)�=q@b�\A�  C�Z�@o\)�'�?333@fffC���                                    Bx�s^�            A4(�@y���
=@��A�\C��q@y���p�@:�HA��HC�W
                                    Bx�sm8  r          A'�@�33��=q@�Q�B(C�� @�33��  @�p�A�G�C��                                    Bx�s{�  �          A)�@�\)��@�
=B'��C�C�@�\)��(�@��
A�{C��                                    Bx�s��  |          A6�\@�R��{@�G�B
=C�=q@�R��@�{A�Q�C��                                    Bx�s�*  T          AG�@�����@��B�
C��3@���z�@��Aޣ�C��                                    Bx�s��  �          A=@ҏ\����@�Q�B33C���@ҏ\��@�G�A�33C�`                                     Bx�s�v  
�          A6=q@�R��  @�p�A�
=C�q�@�R��z�@J�HA�  C�5�                                    Bx�s�  �          A*ff@ٙ���p�@���A�{C���@ٙ���p�@,(�Ak\)C���                                    Bx�s��  �          AM@Ǯ��\?��R@�G�C�)@Ǯ�{�������HC�&f                                    Bx�s�h  �          Au��@�z��[�
�o\)�a�C�'�@�z��D  �����\C�\)                                    Bx�s�  �          As�@���Z�H�Z=q�O33C�` @���Dz����؏\C��                                    Bx�s��  T          Av=q@�  �_��@���3
=C�� @�  �J�R��33���
C���                                    Bx�tZ  �          AxQ�@p���i�fff�V�RC�7
@p���R=q���H���C��)                                    Bx�t   �          Ap��@����Z�H�U��L��C���@����E���=q���C��{                                    Bx�t+�  T          Al��@��\�V=q�P  �K�C��)@��\�@�������z�C��                                    Bx�t:L  �          Al��@�\)�Z{�E�A��C��{@�\)�Ep������\)C�~�                                    Bx�tH�  �          A~ff@~�R�p  �XQ��D��C�l�@~�R�Yp���{��p�C�'�                                    Bx�tW�  T          Az�H@~�R�j�H�qG��^�RC���@~�R�R�H��
=���C�e                                    Bx�tf>  �          A��@p  �p(���G�����C��@p  �T  �����ffC���                                    Bx�tt�  T          Ar�\@]p��[33��������C��@]p��=�
ff�  C�!H                                    Bx�t��  �          Am�@0  �Q���θR��(�C��@0  �/��=q�!�C��                                    Bx�t�0  
�          Ahz�@tz��Rff��ff���C�
@tz��7�
��{��C�/\                                    Bx�t��  �          AqG�@]p��[
=�����C�R@]p��?\)�����C�3                                    Bx�t�|  �          Alz�@O\)�J{��p��߅C�0�@O\)�&�H��H�)33C��                                     Bx�t�"  �          Am�@C�
�C
=��G���Q�C��@C�
����*�\�8p�C���                                    Bx�t��  "          Al��@#33�D����  ����C�ٚ@#33�ff�*=q�8��C�7
                                    Bx�t�n  �          Ajff@��8z���
��
C�33@���R�6ff�LC���                                    Bx�t�  �          Ah(�?�p��5�����C�Q�?�p��  �6�R�P
=C��{                                    Bx�t��  
�          Ah  ?�ff�<  �  �Q�C���?�ff�Q��/\)�E�C��H                                    Bx�u`  �          AG
=>.{�!���
=��\C�}q>.{��p�����I=qC��                                     Bx�u  
Z          AM�>�
=�ff��\���C�7
>�
=����&=q�X=qC���                                    Bx�u$�  T          AQ��?��,z���{��C�h�?��	���R�@�C���                                    Bx�u3R  �          AM?.{�-���
=����C���?.{�G���
�8�
C�7
                                    Bx�uA�  �          AJ=q@�
��z��ff�DQ�C�O\@�
�����1G��{��C��                                    Bx�uP�  �          AJff@33��(�� z��R�C���@33��z��8���{C��                                    Bx�u_D  �          AG\)?�33�
=�p��<�C���?�33��p��-���t�\C�>�                                    Bx�um�  �          A@(�>�������\�WQ�C��q>����{�1k�C���                                    Bx�u|�  �          A<  >.{���H�"�R�t�HC��>.{�8���3�
�C���                                    Bx�u�6  �          AIp������(��  �H�C�� �����G��2ff=qC��f                                    Bx�u��  �          A_�
�!����R�+{C���!����8���aCz�                                     Bx�u��  �          Ax  ?u�Dz�����=qC�<)?u���@z��L�HC�ٚ                                    Bx�u�(  T          A���?�G��c\)�ff���HC���?�G��9���A�7��C�P�                                    Bx�u��  �          A~=q?�G��9��*�\�*33C�|)?�G��	�R�\�bz�C�U�                                    Bx�u�t  T          Ahz�@��H�AG�������Q�C�\)@��H�!p��=q�#{C��                                    Bx�u�  �          As33@;��d���|���r=qC���@;��N�\�陚��
=C�ff                                    Bx�u��  �          Ao�
@��f�H�]p��Up�C�K�@��R�\��33�مC��3                                    Bx�v f  �          AXQ�?Tz��G�
������C��?Tz��.=q��\)���C�+�                                    Bx�v  �          AP�ÿ�\)���������C���\)���H�!�R�C��H                                    Bx�v�  �          AS
=��{�����(���V�HCU�R��{����4(��m�CBG�                                    Bx�v,X  @          ANff��
=?
=�(��J��C/�=��
=@:=q�p��?  C��                                    Bx�v:�  �          AW\)�
=q@�{����/  C�=�
=q@�G���
=�\)C�f                                    Bx�vI�  T          AR�H��ff@k�����A
=C+���ff@���
�\�$��CaH                                    Bx�vXJ  �          A>�\��\@������=qC�q��\@��������=qC�                                    Bx�vf�  �          AS
=� ��>B�\�����C2�� ��@	��� ���Q�C'�f                                    Bx�vu�  �          Aa���3
=������ff� =qCG�\�3
=�
=q�
=��
C>�                                    Bx�v�<  �          A\  �5G���(�������
=CLY��5G��hQ������Q�CE�                                     Bx�v��  �          AX���+��\��(���z�CQ���+����H��������CK0�                                    Bx�v��  �          AZ{�8Q���=q���
����CN�{�8Q���z������z�CI�3                                    Bx�v�.  |          Abff�
=����=q�*\)CS��
=�A��&{�ACG0�                                    Bx�v��  �          A\z�� �����H�=q�"33C[W
� �����H�"�H�@ffCP^�                                    Bx�v�z  T          AG���
���
��  ����CN�\��
�U��
=�\)CF�3                                    Bx�v�   �          APQ��33��=q��{� CUp��33����ff�
=CM
                                    Bx�v��            A\������(��G��*{CoB������{�,���U�Cf��                                    Bx�v�l  "          AN�H�xQ��G��	G��&�\CvE�xQ����
�&{�V  Co�H                                    Bx�w  �          A=��2�\��
=��H�0\)Cz�R�2�\��(�����ap�Ct�f                                    Bx�w�  h          AI�����������\)�ѮCq�q���������=q�z�Cm�\                                    Bx�w%^  �          AM��Q�����  �D�\C��
�Q�����(z��x��C��)                                    Bx�w4  �          AJ=q?�����\�4���C��?���(Q��C\)�)C��=                                    Bx�wB�  T          AT(�?�����\)�2  C���?��ڏ\�,z��f(�C�`                                     Bx�wQP  "          Ah(�>���:�R��
�  C��>���\)�.ff�D{C�AH                                    Bx�w_�  �          Adz�@  �D(���
=�噚C�>��@  �&�H����&��C��\                                    Bx�wn�  �          Ab�\�B�\�5G��ff��C�R�B�\���'�
�BQ�C���                                    Bx�w}B  �          AW
=�Tz��;
=���\��p�C~!H�Tz��"=q� ����C{޸                                    Bx�w��  �          Aj=q�z��X(�������G�C����z��@�������z�C��                                    Bx�w��  �          Aq�W
=�@Q���R�{C~ff�W
=����.{�9�C{.                                    Bx�w�4  �          Apz��:=q�>{�
ff�  C�)�:=q��R�0���>ffC}:�                                    Bx�w��  �          Arff���B�H��
��C�*=���33�3\)�?=qC�)                                    Bx�wƀ  �          Ap���Z�H�:{�p��\)C}���Z�H�ff�2�H�@�Cz                                    Bx�w�&  �          Alz��=p��@������33C�
�=p�� (��%�3�C}��                                    Bx�w��  T          Ar�\�]p��U���������C� �]p��;�����{C}��                                    Bx�w�r  �          AyG��Mp��c�������{C����Mp��J�H��
�  CǮ                                    Bx�x  �          A{
=�j=q�_��\��z�CQ��j=q�E������C}u�                                    Bx�x�  �          Au��~{�T(���z�����C}T{�~{�8(�����{Cz�R                                    Bx�xd  �          As�����J�\����ԸRCx� ����.=q�{�G�Cuff                                    Bx�x-
  �          Ax�������K�
��(��ڸRCw�=�����.�\�33�Ct:�                                    Bx�x;�  �          Av{����TQ���z��ĸRC{�=����9p��G��
=Cy\)                                    Bx�xJV  �          Ax(���=q�X����\)��(�Cx����=q�A��   ��G�CvG�                                    Bx�xX�  �          Av�H��(��V�R���\��Cx#���(��?�� ����Q�Cu��                                    Bx�xg�  �          AuG�����Rff�������RCv  ����;���{��p�Csz�                                    Bx�xvH  �          Au�����O�
�����Cv�������7\)���=qCs�                                    Bx�x��  �          At����Q��U����{����Cw�{��Q��@(���z���Q�Cu^�                                    Bx�x��  �          Aw������Up��fff�W�Cr�������D����=q��33Cp��                                    Bx�x�:  "          Ay��(��[��U��EG�Ct:���(��K���(����Cr}q                                    Bx�x��  �          A{���z��g\)�����mp�C|5���z��T����ff��(�Cz�                                    Bx�x��  T          Aw\)���\�f=q�k��\��C~(����\�T�����H�ʣ�C|�                                    Bx�x�,  �          At����Q��a����33�z{C~���Q��O
=������C|Ǯ                                    Bx�x��  �          AxQ������W
=��Q����RCzY������>ff�
�\�  Cx�                                    Bx�x�x  T          Az=q��p��Zff�����ffCz+���p��B�\��
���Cw�R                                    Bx�x�  �          A{
=����[������\)Cz�3����C33�
�R�(�Cx�                                    Bx�y�  T          A��R����h��������C|:�����P����� �CzT{                                    Bx�yj  �          A\)���
�e���  ��C|)���
�O33� �����HCzY�                                    Bx�y&  �          A}���(��g�
��  ���C|L���(��Tz���33���HCz�=                                    Bx�y4�  �          A|z���  �iG��}p��h��C|ٚ��  �W��ڏ\��\)C{��                                    Bx�yC\  T          A{�
�j=q�g33��{����C���j=q�R�\��  ���C~u�                                    Bx�yR  T          A\)�Z=q�hz���z�����C�e�Z=q�Q���
=����Ch�                                    Bx�y`�  �          A}�����e���R���C~8R����PQ���\)��\C|�3                                    Bx�yoN  �          Ax����Q��^{���R��=qC{
��Q��J�R��(�����Cyk�                                    Bx�y}�  T          At(����\�Z�H��\)����C}aH���\�E����H��ffC{��                                    Bx�y��  �          AqG��}p��Z=q���\��Q�C}���}p��F�\��{��C|O\                                    Bx�y�@  T          A]G������;
=��G�����C�q�����!��\)�(ffC��                                     Bx�y��  �          AZ�H��{�-G�� �����C�lͿ�{���� Q��>�\C��R                                    Bx�y��            Al���o\)�L����z��ŮC}�3�o\)�5���z��
=C{�q                                    Bx�y�2  �          Aw��'
=�Fff����
��C�\�'
=�'33�0���6�HC�R                                    Bx�y��  r          Af�R�
=�>�H����C�h��
=� ���*{�7ffC�ff                                    Bx�y�~  T          AF�\?�\)�@  �'��D  C�AH?�\)�4Q��������HC�w
                                    Bx�y�$            AR�H>����J�\����+�
C��\>����?\)������
=C���                                    Bx�z�  �          ALzῷ
=�6{��p���Q�C�j=��
=�!���Q��33C���                                    Bx�zp  �          Ab=q?Tz��Qp���{����C��3?Tz��=������\C��                                    Bx�z  �          Ap�ÿfff�d(���z���C�/\�fff�Q������C��                                    Bx�z-�  �          AuG���p��h(�������  C��ÿ�p��V{��ff��p�C�b�                                    Bx�z<b  �          Ah��?�G��S�
������  C�/\?�G��>ff��H�	��C�n                                    Bx�zK  �          Ao�
����f�\�~�R�v�RC������Vff��(���z�C��
                                    Bx�zY�  �          As��aG��i�������}��C�E�aG��X���ڏ\�ՅC�#�                                    Bx�zhT  T          Aj�\��Q��]���G���{C����Q��M��ҏ\�أ�C��=                                    Bx�zv�  �          Ah(������_\)�fff�g\)C�` �����P���������
C�T{                                    Bx�z��  �          Ab{��p��S\)��p���33C�0���p��A���G���
=C�q                                    Bx�z�F  �          Ae�?#�
�X(���{����C�\)?#�
�F�\���
��ffC�z�                                    Bx�z��  �          Abff?����V{�n�R�w�C��?����G\)��z��Џ\C�S3                                    Bx�z��  T          Ap�;L���c\)��=q���RC���L���R{���㙚C���                                    Bx�z�8  T          Aw
=��  �m��n{�`��C��{��  �^=q��(��ď\C�k�                                    Bx�z��  �          Ax(����j=q���R��  C��3���YG������33C�<)                                    Bx�z݄  �          Aw���Q��k�
��G����HC��콸Q��Z�\�������HC��                                    Bx�z�*  �          A~�H?��\�s����
��
C���?��\�b�H�����\)C�                                    Bx�z��  �          A��R?�{�y��[��EC��?�{�l  ��ff���C�'�                                    Bx�{	v  T          A�33@(Q����l(��M��C���@(Q��p����Q�����C��3                                    Bx�{  �          A��
@�=q�w��(���p�C�˅@�=q�l  ��(���=qC�'�                                    Bx�{&�  �          A�{@���i��e�O33C�s3@���[33��(���=qC�                                    Bx�{5h  T          A��
@�z��^�R���H��p�C�7
@�z��HQ��\)�p�C�(�                                    Bx�{D  �          A�{?�33�t  ��=q��z�C��H?�33�a��{���
C��
                                    Bx�{R�  T          A��\?���v�R��33��{C�L�?���e���  ����C��=                                    Bx�{aZ  �          A��@5��|Q��k��O33C��@5��m�������G�C�b�                                    Bx�{p   T          A�z�@�q����
�q�C�g�@�b�\��ff��Q�C��3                                    Bx�{~�  
�          As�?У��i��e��[\)C�/\?У��[��������C�aH                                    Bx�{�L  T          A{\)?\�k�
��G���Q�C���?\�Z=q������  C�.                                    Bx�{��  
�          Az�H?��t���
=q��\)C��3?��k33��������C��R                                    Bx�{��  T          Ar=q?���n{��Q����C���?���e���\)��z�C��f                                    Bx�{�>  "          A[�@��H�N�\��Q��{C��=@��H�F=q��(���ffC�#�                                    Bx�{��  �          Ak�@�(��Yp���  ����C�q�@�(��Qp���������C��                                     Bx�{֊  	�          Aa�@#�
�Y���Ǯ���C�T{@#�
�R=q�vff��ffC���                                    Bx�{�0  �          An{?���a��c33�`Q�C�N?���U��������
C��H                                    Bx�{��  T          AtQ�?
=q�^�\��(����HC��?
=q�J{�
=q�	\)C�:�                                    Bx�||  �          Ai@ ���c\)���
���HC��@ ���]p��Z=q�Z{C�%                                    Bx�|"  �          A_\)@�33�M녿�G���C�O\@�33�HQ��I���S\)C��3                                    Bx�|�  
�          A_�@����S\)>��?!G�C�8R@����P����
��
C�S3                                    Bx�|.n  "          Ah  @��
�[�
���C���@��
�W\)�3�
�3�C��q                                    Bx�|=  "          Aip�@E�b�H�����p�C�#�@E�\���X���W�
C�L�                                    Bx�|K�  �          Af�H@���V=q���G�C��q@���M���Q���(�C�#�                                    Bx�|Z`  
�          AeG�@�33�J�R�`  �b�\C���@�33�>�H������=qC���                                    Bx�|i  �          Ae�@���H���HQ��K�C���@���=��z����C�>�                                    Bx�|w�  "          Ai�@����Q�������C�� @����C���z���G�C�&f                                    Bx�|�R  �          Ak�@��R�U�l���h��C�0�@��R�IG���=q��=qC�                                    Bx�|��  "          Ar�H@��R�[��l���bffC�c�@��R�O33���
���\C��{                                    Bx�|��  "          Ak33@��\�Z�\�L���IG�C�˅@��\�O\)��33��33C�:�                                    Bx�|�D  "          An{@�\)�Zff�4z��/�C��@�\)�PQ���
=����C�y�                                    Bx�|��  �          An�H��  �d��=L��>aG�C�H��  �b=q��R��RC��
                                    Bx�|ϐ  �          Aq>�ff�o�
��=q����C��)>�ff�h���|���r�HC��                                    Bx�|�6  "          As����
�p������
C��
���
�h  ��������C���                                    Bx�|��  
�          Au녿xQ��tQ쿥�����C�0��xQ��m��k��]C�%                                    Bx�|��  
�          Ayp��J=q�s\)�h���VffC�#��J=q�n{�R�\�C
=C��                                    Bx�}
(  
�          A~�R�^{�v�\=���>�Q�C����^{�s���
��RC���                                    Bx�}�  
�          A}����lz�?���@���Cy�)����m���8Q��'�Cy�3                                    Bx�}'t  �          A~=q�陚�[33@N{A;�Cq�3�陚�`z�?��@tz�Cr�                                    Bx�}6  T          A{
=�����i@p�A=qCz������lz὏\)���C{�                                    Bx�}D�  
�          Az�H����k33?ٙ�@�\)C{�����l�׿z��Q�C{)                                    Bx�}Sf  �          Axz������ip�?�
=@ǮC{�H�����j�H���Q�C{�R                                    Bx�}b  �          Axz��Q��o�
����33C��{�Q��g�
��(���G�C��                                     Bx�}p�  �          Ar�R��H�d���A��=�C�33��H�Z�\��p���\)C��)                                    Bx�}X  �          A��������v{������C}.�����n�R���o�C|��                                    Bx�}��  T          An=q�Mp��`��?�ff@��C����Mp��`�Ϳ�=q����C���                                    Bx�}��  �          Aj=q��Q��X  ��{��=qCx�R��Q��Q��w
=�uG�Cx
                                    Bx�}�J  T          Ad(����\�W
=�L���QG�C}����\�R�R�333�7�C|�=                                    Bx�}��  �          Ae��u�Z�H�(���(��C~O\�u�V�H�+��-C~
=                                    Bx�}Ȗ  �          Ai녿�  �W33?Q�@\��C�Y���  �V�H��
=��Q�C�XR                                    Bx�}�<  �          Ao�@���T��@�Q�A�z�C��@���\��@   A=qC�(�                                    Bx�}��  �          Amp�@��\�O�@��A�\)C�,�@��\�Xz�@6ffA1��C��H                                    Bx�}�  �          Ap  @����L��@�
=A�p�C�{@����T��@#�
AC���                                    Bx�~.  �          An=q@ƸR�G�@�p�A�p�C�9�@ƸR�Q�@C33A>�RC���                                    Bx�~�  T          Ap(�@c33�X  @���A�\)C�^�@c33�a�@P��AI�C�\                                    Bx�~ z  �          Ao\)@"�\�V�R@��RA���C�Z�@"�\�a��@n{Ah��C��                                    Bx�~/   �          Arff@��H�;\)@���A��
C�"�@��H�HQ�@�G�A�ffC�O\                                    Bx�~=�  �          Aq��@�=q�9p�@���A�C��
@�=q�Ip�@��A�(�C���                                    Bx�~Ll  
�          As\)@�ff�:�\A�HB��C��@�ff�K\)@�{A��C�!H                                    Bx�~[  �          At��@���9G�@�{A�Q�C�Ф@���Ip�@�\)A�(�C��)                                    Bx�~i�  
�          Ap��@��
�Bff@أ�A�  C���@��
�O�@��A�z�C�=q                                    Bx�~x^  "          As\)@�(��=��@�z�A�G�C���@�(��K33@��A��\C�˅                                    Bx�~�  �          As�@{��Ip�@�z�A��C���@{��X  @��A�z�C��                                    Bx�~��  �          Apz�@�Q��Jff@���A��
C�K�@�Q��U@��RA���C��                                     Bx�~�P  �          Aw\)@���L(�@�z�A�p�C�c�@���W�
@�=qA��\C���                                    Bx�~��  T          Az=q@p  �i��@y��Ah(�C�33@p  �o�
?���@��HC��                                    Bx�~��  �          Ay@@���j{@�
=A|��C���@@���q�@�AG�C��f                                    Bx�~�B  �          Ax(�@?\)�h��@eAYC��@?\)�n=q?�=q@�C���                                    Bx�~��  "          Ax  @E�r=q��Q쾮{C���@E�o�
�	����\)C���                                    Bx�~�            A<��?���qG�>Ǯ@�G�C���?���r�\����\)C���                                    Bx�~�4  |          AJ=q=u�^{�=��aHC��=u���C\)¢��C���                                    Bx�
�  �          AC33?
=q��\����M�HC��?
=q��\)�'��l�\C�xR                                    Bx��  "          AIp���
=���R�4  Q�C��\��
=�Z=q�<���=C�w
                                    Bx�(&            AB�R��
@p��7��=C�R��
@mp��1G��B�                                      Bx�6�  |          A>ff=���p��
=�L\)C�q�=���z��!p��j�C��=                                    Bx�Er  �          A>�H����	���G��(�C��H�����
=����G{C�n                                    Bx�T  �          A<(�@�=q� z���ff� 33C�  @�=q��{��ff�33C��R                                    Bx�b�  �          A@�ͿG����H���4�HC�{�G��У��	���R��C��)                                    Bx�qd  �          AJ�\�   �p��ff�&�HC~���   ��p��(��Cp�C|xR                                    Bx��
  h          AI��a��=q����)\)Cw33�a���  ����D\)Ct\                                    Bx���  �          AJ=q��G���������Ch���G���p�����1G�Cc�)                                    Bx��V  �          A_
=@���S33>��R?��
C���@���R=q������
=C��                                    Bx���  T          Ac�
@L���\  ?�@�  C���@L���]���{��{C���                                    Bx���  T          Aa�@%��]G�?s33@xQ�C�G�@%��]p��L���QG�C�G�                                    Bx��H  T          A`��@>{�Z=q��Q쾽p�C�%@>{�XQ����{C�1�                                    Bx���  T          A[\)@��IG�?\(�@hQ�C�/\@��Ip��8Q��AG�C�,�                                    Bx��  �          Aa@�\)�Pz�?�=q@�p�C��\@�\)�Qp�������p�C��                                    Bx��:  "          A`z�@�\)�K�@Dz�AL(�C�5�@�\)�P  ?�(�@�=qC�H                                    Bx���  �          A_
=@�33�Ep�@�A�p�C���@�33�M�@G�AO�C�^�                                    Bx���  �          A^�H@����EG�@���A���C�f@����M@^�RAg�C��                                    Bx��!,  T          AZ�H@�=q�:=q@��RA��C�>�@�=q�C
=@p  A
=C���                                    Bx��/�  "          A[\)@�  �6ff@n�RA}�C��
@�  �<Q�@�
AQ�C�s3                                    Bx��>x  
�          A[
=@�p��@��@N�RA[�
C�!H@�p��Ep�?�  @�z�C��q                                    Bx��M  �          Aa�@��F�R@P  AVffC�K�@��K�?�p�@��HC��                                    Bx��[�  �          A^�H@�ff�@��@^�RAh  C�#�@�ff�F=q@ ��A��C���                                    Bx��jj  �          A]@�(��D��@�G�A�  C�G�@�(��Lz�@Q�A[�
C��3                                    Bx��y  �          AXz�@����?�?�\@�p�C�!H@����A>���?�Q�C�                                    Bx����  �          AZ�\@���9녿�Q���Q�C��{@���6=q�#33�,��C��3                                    Bx���\  T          A<  @u��'�
�J=q����C��@u��%G��G��&=qC�*=                                    Bx���  ,          A6�R@�ff��33���^�C���@�ff�B�\��
�n�C��3                                    Bx����  �          A^�R@�=q�9@��HA�\)C�G�@�=q�C�@�ffA�C��f                                    Bx���N  �          Ag33@�p��E@g
=Ah��C���@�p��K
=@	��A	�C�k�                                    Bx����  �          AS�@�{�1��?�p�@��
C�  @�{�2�R�u��  C��                                    Bx��ߚ  �          Aq@��Q@
=A
=C��@��T��?W
=@P��C��                                    Bx���@  �          Aw�@�33�_\)�k��\(�C�9�@�33�]p������ڏ\C�P�                                    Bx����  
�          Aup�@��Ep�@U�APQ�C���@��J=q?�z�@��C�:�                                    Bx���  �          A{�
@�\�?
=@��
Aי�C���@�\�J�R@�\)A���C��                                    Bx��2  �          AyG�@�  �K�@��\A�\)C���@�  �TQ�@��Aw
=C�|)                                    Bx��(�  �          A���@�z��dz�@�AG�C��\@�z��g33?+�@�C���                                    Bx��7~  
�          A���@���Qp�@�Q�Ař�C�y�@���\Q�@�Q�A��
C��                                    Bx��F$  "          A�{@�z��a��@ʏ\A���C��@�z��k�@�
=A��C��=                                    Bx��T�  �          A���@��H�b�\@޸RA��
C�K�@��H�mp�@��A�ffC���                                    Bx��cp  �          A��@�
=�j�H@�G�A�=qC��@�
=�t  @���AiC���                                    Bx��r  �          A���@����@?\)Az�C�(�@���\)?�33@i��C��                                    Bx����  T          A��@�����
@c�
A5�C��3@�����=q?�ff@��RC��                                     Bx���b  �          A�  A
=�}p�@�
=A_�
C�A
=���
@.�RA  C��)                                    Bx���  �          A�{@�Q���ff@��RA��RC��=@�Q�����@�ffAO�C�,�                                    Bx����  �          A�\)@�{�~�H@�z�A��C�b�@�{����@�z�A�{C��R                                    Bx���T  
�          A���@����q@��\A�  C�C�@����}@�A�\)C��H                                    Bx����  T          A���A�
�N�HAG�A���C���A�
�]p�@��A�C��                                     Bx��ؠ  �          A��HA��P��A�HB{C��A��`Q�A  A���C�0�                                    Bx���F  T          A�(�A���[�
A33A�{C��fA���j{@��RAģ�C��q                                    Bx����  �          A�{A��S\)A$��B�C�Z�A��c\)A�A�C�k�                                    Bx���  �          A���@����o�A
=A�z�C�Z�@����~=qAp�AŮC��                                    Bx��8  
�          A��R@�  ���RA�A���C���@�  ����@�RA��RC�5�                                    Bx��!�  h          AS33?�  ���������
C���?�  �Q���
=��Q�C��)                                    Bx��0�  T          A��H?c�
�@���Q��=G�C��?c�
�*�\�dz��T��C�e                                    Bx��?*  T          A�=q>���5G��_�
�K�C�f>������q��c33C�,�                                    Bx��M�  �          A����R����|Q��o�
C�
=��R��R���Hz�C��q                                    Bx��\v  �          A�=q?޸R�;33�@z��5��C�9�?޸R�&�H�R=q�L�RC���                                    Bx��k  �          A�?����*�\�W�
�N
=C�f?����(��g�
�d�
C�z�                                    Bx��y�  �          A�Q쿡G������  ¡��Cp𤿡G��(����33©�CO�R                                    Bx���h  �          A��R��R��  ����¨��Cwh���R>����\)¯�{C&�3                                    Bx���  �          A����>���z�°(�C%�H��?޸R��§8RB��f                                    Bx����  �          Ax  >L��>8Q��w\)²B#�R>L��?�Q��u�§aHB�W
                                    Bx���Z  �          A�Q쿥�@2�\��{ǮB�\���@��R��G�G�B�=q                                    Bx���   �          A��׿�@�G��\)��BظR��@���up�(�B�33                                    Bx��Ѧ  T          A������@��\�r�H�xG�B΅����A(��e�b�Bʏ\                                    Bx���L  T          A�(��   A��[��`�\Bӊ=�   A"=q�M��KG�BϽq                                    Bx����  
�          A����\)A'��F�H�F�Bř���\)A9��5��0z�B��f                                    Bx����  �          A�ff����A%���R�R�N{B�=q����A8���B{�8�\B�u�                                    Bx��>  �          A�33��
=A&�R�D  �D��B�W
��
=A8z��3\)�/�\Bę�                                    Bx���  �          A|(��p�A-���2�H�4�\B͏\�p�A=���!���B�u�                                    Bx��)�  �          Atz�� ��A#\)�2{�;B�L�� ��A333�"{�&�RB�\)                                    Bx��80  �          Ao
=���HA\)�=p��Sp�B�L;��HA Q��/\)�>(�B���                                    Bx��F�  T          An�\�B�\A���9���M33B�\)�B�\A%��+
=�8  B��                                    Bx��U|  
�          Ak33>aG�A)p��"{�.��B���>aG�A7�����HB��
                                    Bx��d"  �          At��>��AU�������B�>��A_
=������RB��)                                    Bx��r�  "          A�=q>��RA\(��  �
p�B�ff>��RAi��\)��33B��=                                    Bx���n  "          A��Ϳ��Ar=q��ff��
=B�33���A{��������B��f                                    Bx���  �          A��
��A`  �������B�8R��Aj=q��ff��{B�ff                                    Bx����  T          A�(����AX������p�B��q���Adz�����  B�#�                                    Bx���`  
�          Ax��@�AX�������=qB��@�Aa����\���
B���                                    Bx���  �          Ap��?��AK�
��{����B��?��AU������Σ�B��H                                    Bx��ʬ  �          Ar�\@!�AD(����ffB��3@!�AO\)���
��33B��                                    Bx���R  T          Alz�@z�A:ff�Q���RB�u�@z�AF{��
=��  B��q                                    Bx����  �          Aj�\?��RA;33�
{�{B��?��RAF�\��\���RB�                                    Bx����  �          Aj�R?�33A=��{��B��
?�33AH����=q��B���                                    Bx��D  �          Ak\)?�p�A?
=�33�ffB���?�p�AI���z���B��{                                    Bx���  T          Am�?�Q�A9p��=q�=qB��
?�Q�AEG����RB��                                    Bx��"�  �          Ajff?xQ�A9p��  ��HB�p�?xQ�AD����\)�ffB�                                      Bx��16  �          AhQ�?��Az��/33�E�
B�8R?��A"�R�!��2�\B���                                    Bx��?�  
�          Ab�\?�z�A z��6�H�Y\)B���?�z�A��+��FffB�                                    Bx��N�  T          Ad��?\Az��%G��8�HB�L�?\A)���%�
B���                                    Bx��](  T          A`��?�33A�%�>�B��?�33A#33����+��B�{                                    Bx��k�  �          An{?��
A,���"ff�,=qB��?��
A9����G�B��)                                    Bx��zt  "          Ap��>�(�A1G��#
=�*G�B��=>�(�A>{��
�\)B��
                                    Bx���  �          Al�þ��A9p��\)��HB����AD�����
=B�Ǯ                                    Bx����  T          Aap��uAB�\��Q���
=B�  �uAK33��  �ɮB���                                    Bx���f  �          AlQ�
=qA+33�#�
�.�HB��
=qA7�
�p��=qB�\)                                    Bx���  �          Adz�@��A-���R��\B�p�@��A8  � (��	ffB���                                    Bx��ò  T          Alz��5�A333���z�B�aH�5�A>ff������B�                                    Bx���X  �          Ar�R�U�A0���G��!��BՊ=�U�A<����R�
=BӅ                                    Bx����  �          AxQ�����A&ff�*=q�-B������A3\)�z����Bݮ                                    Bx���  �          Au���z�HA%��'�
�-B�ff�z�HA2�\�=q��RBڮ                                    Bx���J  T          Ax����G�A��5���<�B����G�A)p��(���+B�Ǯ                                    Bx���  �          A����Aff�?�
�C�B��)���A ���3�
�3Q�B�Q�                                    Bx���  �          A}�����AG��;��?ffB�u�����A��0(��0\)B�                                     Bx��*<  �          Ax����=qAG��2�R�8�B�Q���=qA�\�'
=�(��B���                                    Bx��8�  �          Ak
=��p�AQ����!G�B�
=��p�A'��33��B�q                                    Bx��G�  
�          A[�
�@  A2�\����(�B��R�@  A;\)����G�B�\)                                    Bx��V.  "          A^ff@4z�A<�������HB��@4z�AD(���Q����HB��                                    Bx��d�  T          AZ�\@z�HA0  �������RB��q@z�HA7�
�������B�L�                                    Bx��sz  �          AQp�@K�A/����H��Q�B��R@K�A6�\������B��H                                    Bx���             AW
=@{A:{���H��Q�B�ff@{A?����R���\B�                                    Bx����  T          AQG�@XQ�A/�������\)B���@XQ�A6�\��=q����B�                                    Bx���l  �          ALz�@P��A'
=��33���HB�\)@P��A.�\�����ffB��q                                    Bx���  
�          AT(�@L��A)G�������B�Q�@L��A1p���Q���33B���                                    Bx����  T          A_33@Dz�A.�\����B��{@Dz�A7\)��33��\)B�                                    Bx���^  �          Adz�@+�A4Q��=q�ffB�=q@+�A=p��陚��(�B�z�                                    Bx���  
�          A��Ϳ=p�Abff��33��B��=p�Aj�\������\B���                                    Bx���  �          A��H�aG�Ah  ��R��p�B�녿aG�Ap  ��z���Q�B��                                    Bx���P  �          A��Ϳ���Ap  ��33��=qB�� ����Ax(���Q���p�B�(�                                    Bx���  |          A��Ϳ��A�����ff���B��3���A����G����B�ff                                    Bx���  T          A�녿�G�A�  ��=q�T  B����G�A�  �5���HB��{                                    Bx��#B  �          A�����
A��H���
���
B��R���
A��\�����p�B��=                                    Bx��1�  
�          A�G�@�
�Ǯ����¢  C��R@�
���
���¤��C�                                    Bx��@�  �          A�33@tz��z���ff.C�� @tz�n{��G�{C�.                                    Bx��O4  �          A�p�@I���+��}G�  C���@I��>�(��}p�L�@���                                    Bx��]�  �          A\)@J�H?(���x��p�A=G�@J�H?޸R�w���A��
                                    Bx��l�  �          Ayp�@'
=>�
=�r�\ =qAff@'
=?��H�q����A�Q�                                    Bx��{&  T          AUG��>{@�33�5p��i
=B�#��>{@��
�.{�Z��B�W
                                    Bx����  
�          AQ��6ff@��
�3�
�k��B���6ff@��
�,���]��B�\                                    Bx���r  
�          AP(��S�
@�p��33�B
=B�aH�S�
A���=q�3��B�L�                                    Bx���  �          AP(���\)@�=q�-��g�BϽq��\)@�G��%���X�
B��                                    Bx����  �          AXz��@�33�@z��)B�B���@�z��:=q�xffB���                                    Bx���d  "          AR�H>B�\@qG��F�\33B�8R>B�\@��\�B{�=B�=q                                    Bx���
  "          AIG�?�z�@@���@��#�Bt�R?�z�@s�
�=G�L�B��                                    Bx���  r          A\��@�@(Q��V=q��BLp�@�@`���R�Hu�Bkff                                    Bx���V  J          A<  ?�(�@qG��.�\k�B��
?�(�@�\)�*=qu�B��f                                    Bx����  �          A/�
?G�@qG��$  ��B�?G�@���p�B�(�                                    Bx���  
�          A`Q���@�ff�N�H\B�(���@����I�B�B���                                    Bx��H  
�          A^�R���@K��Lz�8RCٚ���@�  �H��� C	k�                                    Bx��*�  |          Ac�@Q녿����X���{C��@Q녿�\�Y��C��
                                    Bx��9�  T          Ak33@�ff�N{�PQ�8RC�xR@�ff�Q��S33�
C�(�                                    Bx��H:  J          Aa��Q�?E��>ff©33C� �Q�?��
�=G�£L�B�{                                    Bx��V�  T          Ah��@�HA���,���AG�B��H@�HA��#
=�3(�B��q                                    Bx��e�  
�          Am�@���A=G��Ӆ��33B��@���AC���33��=qB�aH                                    Bx��t,  
(          Anff@�ffA:�R��\)���HBbG�@�ffA?\)������RBd�
                                    Bx����  �          Ak�
@��HAK�
�w
=�v=qB��R@��HAO\)�C�
�B{B�p�                                    Bx���x  
�          At��@�
=AK�
��\)�ٙ�B�Q�@�
=ARff������B��                                     Bx���  
�          Aw�@��AA�����G�B��\@��AH��������{B��                                    Bx����  T          At��@��
A<������	�\B��=@��
AE���=q��ffB�                                      Bx���j  �          At�þ\)A&�R�2�H�<
=B���\)A1p��(Q��.  B�k�                                    Bx���  T          Aep����\A  �1�J�B�{���\A�\�(���={B�                                      Bx��ڶ  "          A_����A���,���K  Bѣ����A�H�$Q��=z�Bϣ�                                    Bx���\  T          Ai����(�A{�>=q�\(�B�녿�(�Ap��6{�N�\B�\                                    Bx���  T          Ab�H���
Aff�4���T=qB�B����
A��,z��F��B�                                    Bx���  T          A`�Ϳ8Q�A��,���J�B�LͿ8Q�A  �$  �<z�B��                                    Bx��N  �          A_�
�%A\)�   �6(�BҮ�%A ����R�(��B���                                    Bx��#�  �          Ad(��q�Aff��H�&��B����q�A'
=�G���B���                                    Bx��2�  
�          Ap����=qA8����
=��=qB�z���=qA?�
�����B��
                                    Bx��A@  "          Au����HA$Q���
�  B�G����HA-����33B�q                                    Bx��O�  T          A{33���A�R�A��G�RB��
���A�8(��;�B�3                                    Bx��^�  T          Ap���k�Az��@Q��T  B���k�A\)�8Q��GB��                                    Bx��m2  �          Alz��@��A{�2�R�E(�B�p��@��A(��*{�8�B�=q                                    Bx��{�  
�          Ao33�xQ�A ���>{�TQ�B�=�xQ�A\)�6=q�HQ�B�                                    Bx���~  T          A`�����A4(�����RB̀ ���A;
=�����Bˣ�                                    Bx���$  
�          AXz��^{A���z��z�B��^{A!����R��B�                                    Bx����  
Z          Aep���Aff�,  �A��B�L���A  �$Q��6�\B�                                    Bx���p  
�          Aq���p�A{�;33�M\)B����p�Az��3\)�A�B噚                                    Bx���  "          Aq����Q�@�{�>�R�Q�RB����Q�A���7��F�RB��H                                    Bx��Ӽ  T          AY<#�
A��ff�>  B��
<#�
A�{�1�B��
                                    Bx���b  T          Ap���n�RA{�>�R�Q�RB����n�RAz��7
=�E�B��)                                    Bx���  
�          Al����  @Ϯ�:�R�O(�C���  @�(��4���E�C\)                                    Bx����  "          AR�\�G�@_\)����.��Ch��G�@\)����)��C�                                    Bx��T  
Z          AQ���@�����\�733C����@�G���R�1\)C
                                    Bx���  �          Avff�33@����0���7��C�\�33@�  �+��1Q�C޸                                    Bx��+�  
�          A���
=@���8(��5�C���
=@�\)�2�H�/33C33                                    Bx��:F  !          A�{��R@��\�8���8�RC8R��R@�ff�3\)�1�HC�{                                    Bx��H�  �          Aw
=��@���5��<�HC���@�Q��0Q��6z�C�R                                    Bx��W�  �          AyG��@�z��4���:
=C:��@Ǯ�/��3G�C��                                    Bx��f8  
�          Ay��	p�@��H�7��BffC!H�	p�@�ff�2�R�;��CJ=                                    Bx��t�  "          AW�
?�z�A#���\��\B��?�z�A)����Q�B��{                                    Bx����  "          A^ff@��A<Q������{B���@��AA�������
=B��R                                    Bx���*  �          Ac\)@ȣ�A<Q���{��
=Bw��@ȣ�A@(�������By��                                    Bx����  �          Ac\)@߮A4Q���������Bh��@߮A8Q���ff��(�Bk{                                    Bx���v  �          A^{@�A&{���
��{BX�
@�A*ff��=q��ffB[��                                    Bx���            AQ�A��A�\��  ��p�B?\)A��A�`���z�HBA��                                    Bx����  �          AYp�A2ff@i��@���A�ffA�33A2ff@S33@ڏ\A�A��                                    Bx���h  U          AW�
A((�@hQ�@�RBffA�z�A((�@O\)@�z�B�A��H                                    Bx���  �          AW�A5��@�G�@�A�HA�G�A5��@�p�@�A'\)A�                                    Bx����  �          AL��A0(�@�33@.�RAEp�A��
A0(�@�ff@A�A[�A���                                    Bx��Z  �          AS
=A0z�@�\)@z�A"�HA���A0z�@�33@*�HA;\)A�
=                                    Bx��   �          A[\)A9�@ڏ\?J=q@W�A�A9�@���?�33@��A�(�                                    Bx��$�  �          A^�RA9@�=q�����
=A���A9@�����
����AԸR                                    Bx��3L  T          Aa�AO\)@���>�{?�z�A�  AO\)@�Q�?z�@�HA�33                                    Bx��A�  T          AeAO33@W�@��\A�(�AiG�AO33@G
=@��A��
AW�
                                    Bx��P�  T          Ah��AO
=?�=q@�z�A���@޸RAO
=?�  @θRA�\)@���                                    Bx��_>  "          Ai��AF=q�u@�{A�z�C�s3AF=q�!G�@�p�A��C��                                    Bx��m�  T          Alz�AI�>�ff@��RA�  @z�AI�=u@�
=A�z�>�z�                                    Bx��|�  �          Ah��A=G���p�Az�B�C�k�A=G���33A
=B	z�C�p�                                    Bx���0  �          Amp�A.{�L(�A�\B�RC��3A.{�j=qA�BG�C��{                                    Bx����  �          Ao�A���ffA#�
B,�C�
=A����RA33B&��C��                                    Bx���|  T          Ar=qA  ��z�A�B%�C��A  ��(�A�\BG�C��=                                    Bx���"  �          Ap(�AG�����A\)B!��C��AG���(�A=qBz�C��=                                    Bx����  �          AlQ�A   ��=qA{B#C�#�A   ��G�A{B��C��                                    Bx���n  �          Ai�A4�ÿ�\)A�B�C��A4����
A=qB�HC��
                                    Bx���  �          Ak
=A4�ÿ�(�A�B�C�NA4�ÿ�z�AffBG�C�5�                                    Bx���  �          Aj�RA4���	��AffB  C���A4���%�Az�B�RC��\                                    Bx�� `  �          Ag\)A�����
A�
BQ�C��
A������A�B�HC���                                    Bx��  �          Aep�AG���  AG�B#  C���AG���ffA��B�C���                                    Bx���  �          Af=q@�z���=qA��B-��C��)@�z���G�A�
B%�C��\                                    Bx��,R  �          AeG�@����z�A(�B/�C�^�@�����
A��B%�C���                                    Bx��:�  �          Ad��@�(���A�RB�C�)@�(��  A33Bp�C���                                    Bx��I�  T          A^=qA#
=���@�z�A�(�C�3A#
=��@���A�G�C�H�                                    Bx��XD  T          A\  AR�R@N�R?���@�(�A\Q�AR�R@H��?�(�Az�AV=q                                    Bx��f�  T          AZ{AG\)@�(�?�\@	��A�
=AG\)@�33?E�@O\)A�{                                    Bx��u�  T          AY��AJ{?�Q�@�  A��@��
AJ{?��H@�=qA�=q@��H                                    Bx���6  "          AVffAD(�?h��@�{A��@�  AD(�?(��@�\)A�z�@E                                    Bx����  �          AW�AG33?�p�@���A��@���AG33?�  @��\A��@�33                                    Bx����  �          AT��AIG�?��@���A��@�=qAIG�?\(�@�{A��@z=q                                    Bx���(  �          ATQ�AO33?���@!G�A.�H@\AO33?�G�@%�A3\)@���                                    Bx����  T          AU�AQp�?��?�
=@��@���AQp�?Ǯ?�G�@�33@ٙ�                                    Bx���t  �          AUp�AO�?��R��
=���A\)AO�@���=q��A��                                    Bx���  �          AV{AM��@�\�8Q��G
=A!AM��@�H�1G��?33A+
=                                    Bx����  �          AV{A5��@e������\)A�ffA5��@w����
�Џ\A��R                                    Bx���f  T          AW�A8��@�\)�����(�A���A8��@�
=���R���\A��                                    Bx��  �          AZffAS\)?J=q?�@���@[�AS\)?5?�@�{@C�
                                    Bx���  
�          A\  AS�
�aG�@hQ�AuG�C���AS�
�Ǯ@g�AtQ�C�%                                    Bx��%X  �          A[33ALz���@�\)A��C���ALz��  @���A���C�)                                    Bx��3�  �          AYp�A<���,��@��A�z�C���A<���?\)@�p�A�p�C��                                    Bx��B�  �          AXz�A7�
�H��@�(�A�33C�]qA7�
�[�@�
=A�\)C��                                    Bx��QJ  �          AW
=A?�
��R@�Q�AԸRC��fA?�
�fff@�
=A�\)C��)                                    Bx��_�  �          AW\)AD(���(�@�  A��C�  AD(��0��@�
=A�(�C�ff                                    Bx��n�  �          A[
=AG��(�@�p�A��C���AG��\(�@�z�A�ffC�f                                    Bx��}<  �          AY��A0(���p�@�(�A���C�� A0(���\)@�p�A�33C��                                    Bx����  	.          AYp�A{��z�@�\B(�C�ffA{�Ϯ@���B33C���                                    Bx����  �          AT��@�����  A Q�B{C�E@������
@�B��C���                                    Bx���.  T          AUG�A����=q@�=qB
G�C��HA����p�@�Q�B��C��                                    Bx����  �          AUG�A=q�љ�@���B��C�1�A=q��(�@��HA�
=C���                                    Bx���z  �          AS\)A	��33@߮B =qC�<)A	��p�@���A�RC��q                                    Bx���   �          AP��@������@�(�A�=qC�T{@���\)@�Q�A���C��f                                    Bx����  T          AO�@��H��@�{A�  C���@��H�Q�@ə�A�  C�aH                                    Bx���l  �          AM@�Q����@�  A���C�R@�Q��	��@˅A�z�C��
                                    Bx��  �          AQ�@����H@�(�A���C���@����@ǮA�p�C�S3                                    Bx���  �          AO�
@�G��z�@љ�A�\C�'�@�G��	G�@��A�RC���                                    Bx��^  �          AP  @���{@���A�p�C�5�@���
�\@�(�A׮C��q                                    Bx��-  �          AT��Aff���@�{A�C�8RAff�	��@���A�ffC���                                    Bx��;�  �          ATQ�@��H�ff@��A�  C�Ff@��H�ff@�{A�\)C���                                    Bx��JP  �          AP��@�ff�33@ٙ�A�=qC�#�@�ff�  @��A�ffC���                                    Bx��X�  �          AQ�@�=q�G�@�(�A�{C�f@�=q�{@�  A�RC���                                    Bx��g�  �          AR{A�
��(�@�\)A�C��)A�
�{@��A�p�C���                                    Bx��vB  �          AO33@�z����@��AƏ\C��R@�z����@�\)A�  C�:�                                    Bx����  �          AR�\@��  @�G�A�33C�\@���@��\A�  C��)                                    Bx����  T          AW�@��
�  @޸RA��C��@��
��@��A��C�h�                                    Bx���4  �          AX��@�ff��@ʏ\A��HC�@�ff�(�@��A���C�Q�                                    Bx����  �          AV�H@�G���R@��RA��C���@�G���R@�  A�
=C��H                                    Bx����  �          AV�\@�{�Q�@��HA�ffC��@�{�z�@�z�A���C���                                    Bx���&  �          AV�\@�p��\)@ÅA�z�C�l�@�p���
@�p�A��
C��                                    Bx����  �          AU�@�z��\)@�(�A�z�C�h�@�z���@�ffA�\)C��                                    Bx���r  �          AV=qA�	��@�p�A�  C�
A��@�Q�A£�C���                                    Bx���  �          AT  @�����@�z�A�Q�C�{@���� z�@�A��
C���                                    Bx���  �          AV�R@�p��   @�
=B��C���@�p��%��@�  B33C���                                    Bx��d  �          AW
=@��R� ��@��B�C��@��R�&=q@�{A���C��                                     Bx��&
  T          AW\)@�\)�"{@���A�(�C�E@�\)�&�R@���A�C��=                                    Bx��4�  T          AU@���%p�@�  A��
C��{@���*=q@���AָRC�B�                                    Bx��CV  �          A[\)@�=q�/�@��
A�ffC�ff@�=q�4  @�33A��HC��                                    Bx��Q�  �          A]�@�ff�3�
@�  AθRC�k�@�ff�8(�@�
=A�G�C�*=                                    Bx��`�  �          AS�
@�  �-@��\A�p�C��\@�  �1�@��A�C��\                                    Bx��oH  �          AW33@z�H�8Q�@�ffA��HC�c�@z�H�<z�@��A�z�C�33                                    Bx��}�  �          AX  @r�\�@��@�(�A�
=C���@r�\�D(�@��A�z�C��R                                    Bx����  �          AS�@1G��DQ�@vffA�=qC�]q@1G��G
=@QG�AfffC�H�                                    Bx���:  �          AU��@���A��@��A���C��@���EG�@��A��C��=                                    Bx����  �          AY@+��C�
@�Q�A�(�C�,�@+��G�@�p�A�
=C��                                    Bx����  �          AW�@�(��;33@�G�A��
C�AH@�(��>�H@�\)A��C�3                                    Bx���,  �          AQp�@��R�9p�@�(�A�
=C��f@��R�<Q�@dz�A}��C�`                                     Bx����  �          AM�@�z��4��@@��A]��C�(�@�z��6�R@{A5p�C��                                    Bx���x  �          ANff@��
�{@��RA�Q�C�z�@��
��\@���A���C�3                                    Bx���  �          AQp�@���1p�@�G�A�33C��R@���5�@�Q�A�33C���                                    Bx���  T          AP  @����/�@�  A��\C��
@����2�H@�
=A���C��                                     Bx��j  �          AI�A  �ᙚ@\A�  C���A  ���H@��AۮC�+�                                    Bx��  �          AP  A����R@�33A�z�C�l�A���
=@�\)A��C��{                                    Bx��-�  �          AO�@�����@�z�A���C�Ff@���p�@�
=A�  C�˅                                    Bx��<\  �          AJ�\@�\)�%�@��A��C��@�\)�)G�@��A�(�C��\                                    Bx��K  �          AL��@����z�@�=qA�=qC��
@������@�z�A��C�j=                                    Bx��Y�  �          AW
=A���أ�@׮A��C��RA����\@��A��C�W
                                    Bx��hN  �          AN�R@6ff�<Q�@�=qA��HC��\@6ff�?\)@o\)A�\)C��{                                    Bx��v�  �          AJ�H@����!�@�p�A�(�C�@ @����%p�@�A�=qC���                                    Bx����  �          ALz�@�\)���@\A�z�C���@�\)�=q@�(�A�=qC�O\                                    Bx���@  �          AJ�H@�33��p�@�B
=C�Ф@�33�z�@���B�C�1�                                    Bx����  �          AK�@���	G�@�Q�B�\C�J=@����R@��HBG�C��                                    Bx����  �          AMp�@�\)��
@�G�A�z�C���@�\)�z�@��\A�G�C��H                                    Bx���2  �          AR�HA��z�@{�A���C�4{A��\)@]p�At��C��                                    Bx����  �          AL��@�{�
=@���A�C���@�{�ff@��\A���C�w
                                    Bx���~  �          AHQ�@�(���z�@�G�A�\C�@�(���H@�z�A�{C���                                    Bx���$  
�          AHQ�@�Q�����@�A�G�C���@�Q��G�@�G�A�\C���                                    Bx����  �          AJffA��33@(��AB�\C���A���@�A Q�C�t{                                    Bx��	p  �          AJ=qA{��p�@[�A|��C�]qA{���\@B�\A_�C��                                    Bx��  �          AL(�A!����z�@a�A��HC���A!�����@K�Ag�C���                                    Bx��&�  �          AL��A
=��@��RA��HC�u�A
=���@���A�
=C��                                    Bx��5b  �          AK�@�33�
{@�\)A���C�&f@�33��@�G�A���C�                                    Bx��D  �          AIG�A(���  @��A�  C�t{A(���@�ffA��C�                                      Bx��R�  �          AK\)A��\)@���AծC��A��  @�Q�A�{C���                                    Bx��aT  
�          AN=qA�H���@��HB �RC��3A�H����@���A��RC�{                                    Bx��o�  �          AMG�A�\����@�ffB	�C�A�\��(�@�B�C�8R                                    Bx��~�  �          AI�Ap����@�\B��C�T{Ap���{@�33BC�]q                                    Bx���F  �          AO�
A�
��{@�B
�C�>�A�
��=q@�33B�C�`                                     Bx����  �          AO\)A{��(�@��
B
\)C�+�A{��  @��
B{C�G�                                    Bx����  �          AN�\A\)��=q@��B��C���A\)���R@���B(�C��                                    Bx���8  �          AI�A��P��@��B#z�C�~�A��j=q@�B��C�T{                                    Bx����  �          AF�HA
ff�z�H@���BG�C��\A
ff��=q@�=qB(�C��)                                    Bx��ք  �          AG33A���Q�@���A�  C�H�A���G�@���A�  C��                                     Bx���*  �          AEG�A�\��R@�G�A�
=C�|)A�\�=q@w
=A��
C�q                                    Bx����  T          AE@�ff�@���A�\)C�T{@�ff�	G�@}p�A��C��{                                    Bx��v  �          AC�
A(���
=@�33B ��C�w
A(���(�@��
B33C�]q                                    Bx��  �          AH  A�\���@�ffBz�C���A�\��Q�@�
=B(�C���                                    Bx���  T          AD��@�ff���@�G�B�RC���@�ff�Ǯ@�
=B
Q�C��\                                    Bx��.h  �          AEG�@�  ��@��
B�RC��f@�  ��G�@θRA�\)C���                                    Bx��=  �          AC�
A{��Q�@�p�B�HC��\A{��(�@�(�B(�C�                                    Bx��K�  �          AC�@�  � ��@�{A��C��R@�  �=q@�  A���C�j=                                    Bx��ZZ  �          AAG�@�����@�z�A��HC�:�@������R@��RA���C��
                                    Bx��i   �          A?�A-�?˅@�G�A��\A{A-�?��@��
A��@�\                                    Bx��w�  �          A=��A{��(�@���A�G�C��\A{���
@���A���C�/\                                    Bx���L  �          A:{A�\�У�@��A�G�C��A�\�أ�@�G�A���C���                                    Bx����  T          A5�A
�H���@�
=A�ffC��{A
�H�ƸR@xQ�A��
C�5�                                    Bx����  �          A2�\AQ����H@�(�A�z�C�ǮAQ����@s33A��C�H�                                    Bx���>  �          A0  @��޸R@��HA��\C��@���p�@l��A�
=C�q                                    Bx����  h          A*=q@�ff����@�p�A�p�C��q@�ff����@�(�A��HC�U�                                    Bx��ϊ  �          A%��@�G��\)�(��Z�HC��q@�G���R��=q�ÅC��\                                    Bx���0  �          A�@�=q��Q�@�=qB-{C�ٚ@�=q��(�@ə�B$�C��                                     Bx����  �          A!p�@������
@hQ�A��C���@�����=q@L(�A��\C�P�                                    Bx���|  �          A�H@�33��p�@7
=A�\)C�.@�33��=q@��Af�HC��                                    Bx��
"  �          Ap�@�\)�ᙚ@�  A�G�C��\@�\)��Q�@eA�z�C���                                    Bx���  �          A\)@�����{?��HA�C�s3@�����Q�?��@���C�O\                                    Bx��'n  �          A�@}p���ff@#33A��HC�Ff@}p����H@(�A��\C���                                    Bx��6  �          AQ�@�
=�˅@�=qA�Q�C���@�
=��(�@�{A�ffC��                                    Bx��D�  �          A(�@���ȣ�@uA�z�C��{@���Ϯ@^{A��C�=q                                    Bx��S`  �          A�R@�Q���  @���Bz�C��3@�Q�����@�A�{C�.                                    Bx��b  �          A�R@�Q���z�@�33BffC�Q�@�Q���{@�=qBQ�C�g�                                    Bx��p�  �          A{?Tz��\)@�RA^�RC���?Tz��G�?ٙ�A(��C���                                    Bx��R  �          A�?�{�
=@S�
A�33C�u�?�{��@2�\A�=qC�^�                                    Bx����  �          A33>�{���@Ag33C��>�{��H?��A0Q�C��                                    Bx����  �          A�H>���R?�R@w�C�j=>��33=�\)>�(�C�j=                                    Bx���D  �          AQ�����@=qAs33C�q����?�\)A;�
C�xR                                    Bx����  �          A�׾���
�H@;�A�G�C�~�����p�@��As
=C��H                                    Bx��Ȑ  �          A��?E��	G�@8��A�z�C���?E���
@
=AqG�C���                                    Bx���6  �          AQ�@G����@z�Aj�HC�s3@G��
�R?��A4��C�XR                                    Bx����  �          A33@���\)���{C���@����H�p���=qC�                                    Bx���  �          AG�?fff�>{�p���I�C�xR?fff�.�R�|(��Wp�C�(�                                    Bx��(  �          A�@fff���@(�Ap��C��3@fff�33?�A<z�C���                                    Bx���  �          A�R@r�\���@B�\A���C�5�@r�\�Q�@ ��AqG�C��q                                    Bx�� t  �          A\)@Z�H�G�@0  A��C�.@Z�H��@{AY�C���                                    Bx��/  T          A�?��	��@�Ah��C��3?���?�p�A0(�C�޸                                    Bx��=�  T          A�
>W
=�=q?z�H@���C���>W
=��H>��@(��C���                                    Bx��Lf  T          A=q@&ff���?��AD��C��@&ff��(�?��\A�C��                                    Bx��[  �          AG�@���\@�\)B��C�N@��33@��Bp�C�%                                    Bx��i�  �          A��@��1�@xQ�A�  C�\@��A�@l��AЏ\C�33                                    Bx��xX  |          A33A{�|(�@&ffA�Q�C�3A{��33@Af=qC���                                    Bx����  �          Ap�@����
=>�?fffC��@����
=�L�Ϳ�z�C��f                                    Bx����  �          @�Q�@Y������#�
��(�C�N@Y������<(����C���                                    Bx���J  �          @��@"�\��33��
=�33C���@"�\��G�����!C�aH                                    Bx����  �          A(�@����������n=qC���@��������z���z�C��                                    Bx����  T          A@����녿ٙ��AC�0�@��������t  C�t{                                    Bx���<  �          A  @�
=�xQ쿰�����C�p�@�
=�q녿���:�\C�                                    Bx����  �          A=q@�����z����r�HC���@����~�R�p����C�                                      Bx���  �          A�@�  ������H�C��@�  �w����� �C�f                                    Bx���.  �          A��@�\���
�h����C��@�\���
�p  ���HC��f                                    Bx��
�  �          A�H?�Q��陚�1���C�o\?�Q���33�Q���{C��                                    Bx��z  �          A	�@`�����ÿ�����\)C�~�@`��������4��C���                                    Bx��(   �          AQ�@�{����?�=qA;
=C�'�@�{��(�?��
A
=C���                                    Bx��6�  �          A�@��H��{@ ��AN�HC��@��H���?\A  C��\                                    Bx��El  �          A�@.{�z�?�Q�A�\C��@.{�?\(�@��HC�                                      Bx��T  �          A�H?���	?���AJ�\C�?����?��A
�\C���                                    Bx��b�  �          A\)?fff��@8��A���C�{?fff���@�\An�\C��                                    Bx��q^  �          A�?�\)��H@Dz�A��
C��)?�\)�{@�RA�C�z�                                    Bx���  �          A�\?�\)�=q@<(�A��
C���?�\)�G�@ffAw33C���                                    Bx����  �          A=q@�ff��\)?�  A8Q�C��@�ff���H?���@��C�y�                                    Bx���P  �          AQ�@���@���Bz�C�y�@��@�z�B�RC��                                    Bx����  �          AQ�?�\)���@q�A�
=C��)?�\)��
@L(�A�Q�C���                                    Bx����  �          A(�?�Q����@l��A�ffC�(�?�Q����@FffA�33C�
=                                    Bx���B  �          A�?�=q�@5A�ffC���?�=q���@{Af{C�o\                                    Bx����  �          A=q?�z��
�H?�
=AQ�C��\?�z��(�?G�@��C��f                                    Bx���  �          A?�\)�
{?���A	�C�� ?�\)�33?.{@���C�u�                                    Bx���4  �          Aff?�\)�33>�?Tz�C�w
?�\)�
=���Z�HC�y�                                    Bx���  �          A�H@�Q���=q��(����C�Z�@�Q���R��\�AG�C���                                    Bx���  �          A�?=p��p�@s�
A���C�� ?=p��@K�A�(�C��=                                    Bx��!&  �          A  =������H@���A�{C�Z�=����=q@l(�A�33C�W
                                    Bx��/�  �          A=q?fff�   @k�A�z�C�1�?fff�(�@C�
A��C�R                                    Bx��>r  �          A녽u��=q@z=qA�Q�C��׽u���@R�\A���C��f                                    Bx��M  �          A@K���
=@��B p�C���@K�����@i��AᙚC��                                    Bx��[�  �          A@��H����@�(�A�ffC��)@��H���
@��A��C�,�                                    Bx��jd  �          A{@�����@���B{C�0�@�����=q@��HB	(�C�&f                                    Bx��y
  �          A@����aG�@�33B7�C�}q@�����Q�@��B-{C��{                                    Bx����  �          A��@�{��{@���B/\)C��@�{����@��B"(�C��                                    Bx���V  �          A=q@�
=�u@ÅB0��C�!H@�
=��=q@�G�B%\)C�~�                                    Bx����  �          A�@��H�У�@�Q�A癚C��@��H���H@n{A�(�C�s3                                    Bx����  �          A  @��
��{@�Q�A��C��
@��
����@\)Aڣ�C�"�                                    Bx���H  T          Az�@������@��HBffC�` @������@�(�A�
=C�k�                                    Bx����  "          A33@�������@�z�A�  C�+�@������@}p�AٮC�@                                     Bx��ߔ  �          A\)@�Q���Q�@�ffA�=qC��@�Q����H@uA�z�C��H                                    Bx���:  �          A(�@�p���G�@�  B�\C��)@�p���{@���B �\C���                                    Bx����  T          A\)@��\��\)@���B33C�H@��\���@��A�\)C�Ф                                    Bx���  �          A
ff@ƸR�w
=@��A�
=C�
@ƸR��{@tz�A�=qC�H                                    Bx��,  "          A
�\@�Q��p  @Y��A�G�C��@�Q�����@Dz�A��\C��                                    Bx��(�  �          A
�R@�\)�G�@y��A�z�C���@�\)�\(�@hQ�A�  C��                                     Bx��7x  �          A
=q@��
�j�H@���A��RC�@��
��G�@�{A�z�C���                                    Bx��F  �          A��@�G����\@��A��
C��@�G���{@x��A��HC���                                    Bx��T�  �          A��@��H���@�=qB\)C��)@��H��@��BQ�C���                                    Bx��cj  �          A�@�=q�z�H@ƸRB0\)C�!H@�=q��{@��HB#�RC�`                                     Bx��r  �          A�@��
�W�@�ffB(��C���@��
�w�@�(�B(�C��R                                    Bx����  �          Aff@�G���@��B5�C��@�G��8Q�@\B,�C���                                    Bx���\  �          A�
@�(��Q�@��
B4�C�  @�(��;�@�(�B,33C���                                    Bx���  �          A
=@�����@陚B\Q�C�aH@����>{@��BR{C�aH                                    Bx����  �          A�
@����\)@�
=BA��C��3@����5�@ϮB9{C�l�                                    Bx���N  �          A��@�G��ff@�=qBD�C�� @�G��,��@�33B;��C�H                                    Bx����  �          A�@�\)���@أ�BA��C�aH@�\)��z�@�{B>�RC���                                    Bx��ؚ  �          AQ�@��H�W
=@�ffBB�\C���@��H��Q�@�33B>��C�3                                    Bx���@  �          A��@���@��@�=qBz�C�XR@���Z=q@���BC�                                    Bx����  �          A�@���{@{A�{C���@���z�?�AN�HC�`                                     Bx���  �          AQ�@�������@W
=A��\C��@�����@6ffA��C�\)                                    Bx��2  �          A
{@��R���\@�\)A��\C�|)@��R��
=@}p�A���C�y�                                    Bx��!�  �          A	p�@�ff���@�33A�\C��q@�ff�Ǯ@q�A�{C�                                    Bx��0~  �          A��@����@a�A���C�9�@���Ǯ@=p�A�C��                                    Bx��?$  �          A	p�@�ff��Q�@G�A\��C�j=@�ff��p�?�\)AG�C��                                    Bx��M�  �          A
ff@�p����@}p�A�p�C��R@�p����H@`��A��C��                                    Bx��\p  �          A
�\@�\)��{@a�A�ffC�C�@�\)��  @?\)A��
C�xR                                    Bx��k  �          A�@��\��(�?�\)A4(�C�~�@��\��Q�?��\@��HC�4{                                    Bx��y�  �          AQ�@\�w
=@[�A��C��=@\��@C33A��C���                                    Bx���b  T          A�
@�ff�J=q@��B  C���@�ff��ff@��RBp�C�L�                                    Bx���  �          A��@����ff@�G�B4��C�q�@����;�@���B+�C�H                                    Bx����  �          Az�@���33@�(�B8�
C�|)@���8��@�(�B/{C��                                    Bx���T  �          A(�@�(��
=@��B9G�C�+�@�(��<��@�z�B/G�C���                                    Bx����  �          A�R@��
�(��@���B{C�S3@��
�HQ�@��B
\)C�w
                                    Bx��Ѡ  �          A�
@�ff�K�@�{B�RC�޸@�ff�k�@�33B�C���                                    Bx���F  �          A	�@�Q���{@���B)��C�L�@�Q���Q�@�33B
=C��H                                    Bx����  �          A	@��R���\@�  Bz�C�p�@��R���
@�G�BC�Ǯ                                    Bx����  �          Az�@�{�c�
@�BQ�C�e@�{��33@�G�B�HC��                                    Bx��8  �          A��@��
�W
=@�{B(�C���@��
�u@�=qB=qC��                                    Bx���  �          A	p�@��H���H@��A��HC�=q@��H��  @l��Ạ�C�                                    Bx��)�  �          A=q@Dz���Q�\)�p��C���@Dz���ff�p����ffC���                                    Bx��8*  �          A{@����׮��Q�+�C��\@�����{�J=q���C��f                                    Bx��F�  �          A{@�p���=q?�z�AQ�C���@�p����>�G�@H��C���                                    Bx��Uv  �          A\)@����z�k����C��{@�����\�W
=��z�C��R                                    Bx��d  �          A��@���>{�\�,��C�R@���2�\��=q�O�
C���                                    Bx��r�  �          A  ��\���?E�@��
C�� ��\� Q콣�
�
=C��f                                    Bx���h  �          A�R�.{�
=?��@w
=C|!H�.{�\)���R��(�C|+�                                    Bx���  �          A33��
=�33��(��:=qC�b���
=������\C�P�                                    Bx����  �          A
=���R���J=q��p�C��H���R�����
=�8Q�C�˅                                    Bx���Z  �          A	�����
��H��ff�(�C������
��
�p��o�
C��                                    Bx���   �          A  �\)�	����
�"ffC����\)����p���\)C��                                    Bx��ʦ  �          A=q�����Q쿞�R��C��=�����	G�����e�C���                                    Bx���L  �          A{>�  �
�H���H�2{C�Ф>�  �
=�*=q���C��
                                    Bx����  �          A��>����(��&ff���C��>����
{��\)�)C��                                    Bx����  �          Az�}p���?���@���C���}p��33>L��?��
C���                                    Bx��>  �          Az�@5���R���H��C�K�@5��33����v�RC��f                                    Bx���  �          A33@3�
����ff�9��C�Q�@3�
� ���.�R���
C��R                                    Bx��"�  �          A�H@�G��$z���(��C��@�G��G������\)C�]q                                    Bx��10  �          A
{@���0������
�C�+�@���p�������RC�]q                                    Bx��?�  �          A\)@�p������l(���ffC��\@�p���{��33��C��                                    Bx��N|  �          A��@�{��
=�����=qC��@�{��(������ffC���                                    Bx��]"  �          A�R@�  ���R��33��\C��H@�  �����\)��\C�u�                                    Bx��k�  �          A33@��H��{�J=q��=qC�B�@��H�����u���C�H�                                    Bx��zn  �          Az�@�  ��=q�W����C���@�  ��(������{C�˅                                    Bx���  �          AG�@�33��G��j�H��p�C��@�33��=q������33C�e                                    Bx����  �          A�@�p������  �2�\C���@�p����H�*�H��33C�b�                                    Bx���`  �          A�@�����Q��:�H���C��@������
�g
=��  C���                                    Bx���  �          A=q@�����H��(���33C�@ @�����������
=C��f                                    Bx��ì  �          A=q@�����
=�����0G�C���@����Y���߮�@z�C��3                                    Bx���R  �          Aff@�33��(����
�2��C�(�@�33�r�\��(��D�HC���                                    Bx����  �          Ap�@�
=��{��{��C��R@�
=�����  � p�C���                                    Bx���  �          A��@�����\)����!\)C���@����l������2
=C�#�                                    Bx���D  �          Az�@����  ��{���C��=@�����������33C��)                                    Bx���  �          A�?�33�	G�@%A�=qC�n?�33�G�?�G�A��C�G�                                    Bx���  �          Az�@>{��@�AZ{C��=@>{��H?�=q@�z�C���                                    Bx��*6  �          A�@�33?@  �L����z�@�@�33?���E�����A(�                                    Bx��8�  �          AG�@��R���H?�p�AW�C�y�@��R��G�?���@�p�C�3                                    Bx��G�  �          A��@E�
=@	��A\��C�U�@E�ff?���@�  C��                                    Bx��V(  T          A�H@\)� ��@(Q�A��\C���@\)�G�?˅A%p�C�S3                                    Bx��d�  �          A��@θR��z�@�33B��C��{@θR�p��@���B(�C��)                                    Bx��st  T          A�H@?\)�7
=@@  B��C�!H@?\)�N{@'
=A�C�k�                                    Bx���  �          A33���\�	p���(��MG�C��ῂ�\�(��C�
���C�y�                                    Bx����  �          A
=>u�Q쿙����\C�>u����
=�p��C�Ǯ                                    Bx���f  �          A�>��G�?.{@���C�h�>��p���\�Mp�C�h�                                    Bx���  �          A=q>���G�@��Ae�C�W
>�����?�=q@ٙ�C�O\                                    Bx����  �          A=q���33@R�\A�33C��������@(�Ae�C���                                    Bx���X  T          A	녾aG���{@>�RA�z�C�5þaG��Q�?�33AO33C�>�                                    Bx����  �          A=q@\)��G�?�\)AO�C�@\)��
=?Q�@��
C�ٚ                                    Bx���  �          A
�\@����У�@b�\A�  C�w
@�����{@(��A�
=C��                                    Bx���J  �          Aff@�  ��(�@Q�Al  C��R@�  ��33?�
=A=qC�o\                                    Bx���  �          A�?����׿�����\C�g�?��� ����
�}p�C���                                    Bx���  �          AG�����(���p��<��C�������ff�5���RC�ٚ                                    Bx��#<  �          A	��?\)��H���H��HC��?\)��H�ff��{C���                                    Bx��1�  �          A
�\?���� z�?E�@�33C�c�?���� �þ�{�C�^�                                    Bx��@�  �          A
�H=��
�
ff���R��
C�C�=��
�z��  ��
C�E                                    Bx��O.  T          A  ��z��
=�p���ƸRC��)��z����ff�_�C�s3                                    Bx��]�  T          A=q�s�
��������Q�Cs��s�
�����=q�33Cs#�                                    Bx��lz  �          A��@�p���\)@Ab�HC�0�@�p���ff?��@�33C�Ǯ                                    Bx��{   �          A\)@�
=��\)?�p�A!��C�O\@�
=���
>�@P��C�                                    Bx����  �          Aff@�����?���A��C��f@����ٙ�>�Q�@p�C��=                                    Bx���l  �          A  @�(���33@ffAi��C�� @�(���33?�
=A��C��f                                    Bx���  �          AG�@Q��녾�  ��33C���@Q��   ��z��(�C�%                                    Bx����  �          A
=@U�=q�����  C�#�@U� �׿����
{C�Ff                                    Bx���^  �          Aff@g
=� (���  ����C��@g
=�����	���aG�C�o\                                    Bx���  �          A�@%��p��mp���ffC�� @%�ٙ���  ��C�l�                                    Bx���  �          A�
@p��ff��  ��  C���@p���p����mG�C���                                    Bx���P  �          A�?��H�z��(��5C��q?��H�{���2=qC��3                                    Bx����  �          A  @����׿�\�X��C�f@���{��p��8(�C�.                                    Bx���  �          A  ?޸R�����G
=����C�N?޸R��
=��\)��33C���                                    Bx��B  �          A
ff@,(���\)�-p���=qC��f@,(���  �r�\��Q�C���                                    Bx��*�  �          A��.{��(��hQ����C�]q�.{��  �����Q�C�O\                                    Bx��9�  �          A��@��z�?��A=qC��\@��=q=�Q�?�C��
                                    Bx��H4  �          A�
@3�
�\)?�ff@�  C�33@3�
�Q�k����RC�#�                                    Bx��V�  �          A(�@����
<��
>��C��@���
=q��ff���C��R                                    Bx��e�  �          A����
=�33��{���C�� ��
=�=q�*�H���RC���                                    Bx��t&  �          A
=������Q��a�C�#׿��� ���Y������C��                                    Bx����  �          A�>�=q�=q>��?��\C��=>�=q��ÿ�����C���                                    Bx���r  �          Az�@XQ�����@J=qA��C�H�@XQ�����?�p�AQp�C���                                    Bx���  �          A
=@)���33?���AL(�C��{@)���ff?(��@�Q�C��                                     Bx����  �          A  @���H?��@أ�C���@���
��z��\)C���                                    Bx���d  �          A��?(���G���z���RC�  ?(������!����C�1�                                    Bx���
  �          AG�@�����(�@8��A��C���@�����  ?�\A>=qC�G�                                    Bx��ڰ  �          A(�@Ӆ�6ff@��B
=C�Y�@Ӆ�dz�@���A�=qC��                                    Bx���V  �          A��@�(�����@��A݅C��R@�(����@Mp�A�ffC�*=                                    Bx����  �          A�@�ff����@XQ�A�  C��@�ff����@��A���C��)                                    Bx���  �          A33@���{@*�HA�
=C���@�����?\A��C�\                                    Bx��H  �          A33@�{����@�RA��C�� @�{��ff?�ffAffC�{                                    Bx��#�  �          A{@�ff�˅@��A�p�C�~�@�ff��ff@E�A�p�C�S3                                    Bx��2�  �          A
�H@����@�
=A�  C�:�@�����H@g
=Aď\C�n                                    Bx��A:  �          A
�H@�
=���@��A㙚C�y�@�
=���H@O\)A��\C��
                                    Bx��O�  �          A  @�33���
@EA�(�C�W
@�33��G�@ ��AT��C�s3                                    Bx��^�  �          A@��\��p�@��HAݮC���@��\�ȣ�@FffA�
=C�33                                    Bx��m,  �          A33@�ff��G�@�p�A��HC�C�@�ff��ff@\(�A���C���                                    Bx��{�  �          AQ�@�z���ff@�(�A�C��\@�z����@E�A��C�XR                                    Bx���x  �          AG�@�{����@8��A�C�P�@�{��G�?޸RA6�RC���                                    Bx���  �          A�@�  �أ�@��Aj=qC�:�@�  �ᙚ?��
@��C���                                    Bx����  �          A=q@��
��(�@�G�B�C��)@��
���
@|(�Aң�C���                                    Bx���j  �          A�@�������@��HBp�C��q@�������@~�RAծC�}q                                    Bx���  �          A�\@�ff�e�@��RBG�C���@�ff��p�@���A�  C�AH                                    Bx��Ӷ  �          A�\@����G�@�p�A�\)C���@����Q�@l��A�z�C���                                    Bx���\  �          A�@����33@�\)A߮C�` @���Ϯ@J=qA�(�C��f                                    Bx���  �          A�\@z��Q�?�\)@��C�e@z��G�����%�C�XR                                    Bx����  �          Ap�@^{��?���AG
=C���@^{�G�?�\@P  C�P�                                    Bx��N  T          Az�@a����@!G�A���C��@a���?���@���C��                                     Bx���  �          Az�@�ff����@#33A��C���@�ff���?�Q�@�C��                                    Bx��+�  �          A��@��R�陚@L(�A��C���@��R��  ?�{A>=qC�E                                    Bx��:@  �          A\)@6ff����@P��A�G�C�@6ff�  ?�=qA=��C���                                    Bx��H�  �          A�
@�=q��(�@�
=A�G�C�Y�@�=q����@Dz�A���C��q                                    Bx��W�  �          A\)@��
��@��
B(�C�� @��
����@��A�z�C�l�                                    Bx��f2  �          AG�@r�\��R@XQ�A�ffC�ٚ@r�\��ff@�\AV{C�R                                    Bx��t�  �          A
ff?z�H���R@,(�A��RC�� ?z�H��?�G�Ap�C�w
                                    Bx���~  �          A(�?����@{Alz�C�"�?����?333@�(�C��3                                    Bx���$  �          A��u�
=��ff�J=qC�%�u��
=���Y�C�                                      Bx����  �          AQ�@Dz����@Mp�A��\C�R@Dz���\)?��A=p�C���                                    Bx���p  �          A  @�\)��p�@��HB��C���@�\)����@�  A���C�aH                                    Bx���  �          A\)@��R����@���B�C�/\@��R����@r�\A�
=C��)                                    Bx��̼  �          A�@�
=��z�@=p�A�
=C��q@�
=��=q?��A0Q�C��)                                    Bx���b  �          AQ�@G������333C�+�@G���
�
=q�S
=C�T{                                    Bx���  �          AQ�@5�z�?(��@�  C��q@5��
���\��{C��                                    Bx����  �          A(�@�\�{?�p�@�p�C�@�\�
=�   �?\)C�H                                    Bx��T  �          AG�?��H�\)?E�@�\)C�� ?��H�
=�xQ����RC�                                    Bx���  �          A��?\(��Q�?5@��
C��f?\(���
�������C���                                    Bx��$�  �          A��?Y����=�\)>���C���?Y���G���
=�&�RC��                                    Bx��3F  �          A{�z�H����G��,��C�  �z�H�������X��C��                                    Bx��A�  �          A녿�33��׼#�
��\)C�xR��33�녿��
�/33C�g�                                    Bx��P�  �          Ap�=����Q�>��H@@��C�O\=����
=������C�O\                                    Bx��_8  �          A  ?��H��\?���A\)C�5�?��H�  ��
=�%�C�&f                                    Bx��m�  �          A�@2�\��G�@x��AθRC�(�@2�\�ff@�Av{C�t{                                    Bx��|�  �          A�@����@1G�A�(�C��
@����?��
A  C�
                                    Bx���*  �          AQ�@�p���
=@2�\A�ffC��
@�p����
?��A\)C��
                                    Bx����  �          A��@l��� Q�?�(�A0Q�C�c�@l���33=�?L��C�&f                                    Bx���v  �          A��@p  ��@Y��A��\C�O\@p  � (�?�\)A>�RC��                                    Bx���  �          Az�@u��Q�@l��A��
C���@u��33@��Ab{C��                                    Bx����  �          A\)@�ff��p�@~{A��HC��R@�ff��=q@%�A�{C���                                    Bx���h  �          A
=@y�����H@�33A�=qC��3@y�����@*�HA��C���                                    Bx���  �          A=q@l(���33@p��A�G�C���@l(����R@G�AnffC��=                                    Bx���  �          A@��
��=q@�p�A�  C�f@��
���@1G�A���C��H                                    Bx�� Z  �          A�R@�  ��\@j=qA�(�C���@�  ��p�@
�HAaC�                                    Bx��   �          A�R@�  ��@G�A��C�޸@�  �G�?��
AC�+�                                    Bx���  �          A
=@�\)��ff@%A��C�c�@�\)� ��?}p�@�C��
                                    Bx��,L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��:�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��I�   �          A�R@qG��=q@�\AG�C�)@qG��
{>�\)?ٙ�C��\                                    Bx��X>  �          Aff@��
���R?�ffA1G�C�3@��
��\>�?O\)C���                                    Bx��f�  �          A�@��\����?�\)A8��C��@��\� Q�>k�?�33C��=                                    Bx��u�  �          A=q@������?�(�A*{C�  @����
==��
>�C���                                    Bx���0  �          A@�33���
?��RAD��C�+�@�33�>��R?�C�˅                                    Bx����  �          A�@�G���z�@A�A�C�&f@�G����?�
=A
=C�W
                                    Bx���|  �          A�R@����G�@}p�A�Q�C���@�����R@��Ar�\C��                                     Bx���"  �          A�@`  ���@�\)A��C�0�@`  ���H@9��A��C��                                    Bx����  �          A�@�{�׮@�33A�C��@�{��  @5�A��
C��3                                    Bx���n  �          A  @�p��ʏ\@��HA��C�4{@�p���33@9��A���C���                                    Bx���  �          Aff@�p�����@�=qA�C���@�p��ۅ@L(�A�ffC�                                      Bx���  �          A��@�����
=@��B�C���@������
@b�\A��HC���                                    Bx���`  �          AG�@�����{@��\Bz�C���@�������@s�
A�{C�W
                                    Bx��  �          A��@����ff@��B
p�C��H@����z�@l��AǙ�C�G�                                    Bx���  �          Aff@�����@w
=AΣ�C��{@������
@(�A�C��                                    Bx��%R  �          A��@��R���
@�ffA��C�h�@��R��ff@HQ�A��C�w
                                    Bx��3�  �          A�@����Q�@�
=B �\C��3@����z�@W
=A�  C��
                                    Bx��B�  �          Ap�@�
=����@��HA�{C�Q�@�
=��Q�@L(�A�p�C�u�                                    Bx��QD  �          A��@��\���@��\A��C�
@��\�ə�@AG�A�
=C�                                      Bx��_�  |          A��A	G���ff?�Q�AG�
C��A	G��(�?�  A�C���                                    Bx��n�  �          A��A����p�@(�A�C�"�A����R@   APz�C���                                    Bx��}6  �          AQ�@��\�w
=@G
=A���C��H@��\��ff@�AYp�C�1�                                    Bx����  �          A�R@�p����R@*�HA�  C��@�p���?�  A�C�k�                                    Bx����  �          Aff@ᙚ��Q�@<(�A�  C�L�@ᙚ��G�?�  A0��C���                                    Bx���(  �          A��@Ϯ��Q�@1G�A��C�5�@Ϯ��\)?�z�A
=C��                                    Bx����  �          Az�@\��{@Dz�A�33C��{@\��
=?�A*ffC��
                                    Bx���t  �          Az�@��H����?J=q@�
=C�R@��H�����z����RC�
=                                    Bx���  �          A�H@�=q���\?�{AO�
C�4{@�=q���>��H@xQ�C���                                    Bx����  �          @��R@��@�HA�{C���@��4z�?�=qAUG�C�B�                                    Bx���f  �          A�@�G�@�\)>��H@`  Bz�@�G�@�  ?�ffA1G�B	��                                    Bx��  �          Ap�@�\)@���?p��@�  B �
@�\)@�=q?�AUp�A��
                                    Bx���  �          Ap�@�=q@qG�>�Q�@!G�A��
@�=q@e?��HA�AиR                                    Bx��X  �          AQ�@�  ?�ff���
��AQ@�  ?�\>�{@�AN=q                                    Bx��,�  �          A�@�?Tz�\�,��@�{@�?c�
�8Q쿡G�@���                                    Bx��;�  �          A�\@����L������C�l�@���������\C�Ff                                    Bx��JJ  �          A
=@����33?��Ap�C��@����?}p�@�ffC���                                    Bx��X�  �          Az�Ap��z�H?n{@���C��Ap���?.{@�Q�C��                                    Bx��g�  �          A
=A?+�=���?333@��A?!G�>�  ?޸R@���                                    Bx��v<  �          A\)Aff?J=q?&ff@�p�@�33Aff?(�?Q�@���@�p�                                    Bx����  �          A�Az�?\�B�\���
A&�\Az�?\>.{?�
=A&�R                                    Bx����  |          A�A
=?���?��AG33AffA
=?W
=@��Ab�H@�{                                    Bx���.  �          A�RA�H?��@%�A�
=@�(�A�H>�ff@1G�A�
=@Dz�                                    Bx����  �          A�A=q�^�R@'�A�  C��A=q��(�@Au��C�
=                                    Bx���z  �          A\)A	p��ff@#�
A�\)C�%A	p��(��?��RAH  C�p�                                    Bx���   �          A\)A	p�����@#�
A�{C��A	p��=q@33AN�RC�*=                                    Bx����  �          AG�A�;�=q?��HA+
=C�&fA�Ϳ+�?˅A�C��                                    Bx���l  �          AQ�Az�˅@L��A��
C��RAz���
@.{A���C�k�                                    Bx���  T          A��A���QG�?�ffA5C�P�A���fff?u@�G�C�Q�                                    Bx���  �          A  @�33��{?���A
=C���@�33��33�#�
�k�C�Y�                                    Bx��^  �          A��@�33�[�@N{A�
=C�ٚ@�33���@�\An{C���                                    Bx��&  �          A��A����@7
=A���C�� A���3�
@  Ae�C��H                                    Bx��4�  �          A�
A{��{@<(�A�z�C�eA{��@p�A�{C�*=                                    Bx��CP  �          A�A���@N{A�(�C�!HA����@?\)A���C��                                    Bx��Q�  �          Ap�Ap����@?\)A���C�AHAp���@#33A�Q�C��)                                    Bx��`�  �          AffA
�H�
=?�{A;�C�(�A
�H� ��?��
A ��C��3                                    Bx��oB  �          A��A���@5A��\C���A��C�
@	��A\z�C��                                    Bx��}�  �          A�@�ff�dz�@G�AQ�C��@�ff�}p�?�=q@�p�C��f                                    Bx����  �          A��Aff�j=q?��A�C��fAff�w�>�{@
�HC�N                                    Bx���4  �          A��@�Q���?��@��C�!H@�Q����\        C���                                    Bx����  �          A��@�
=���?&ff@��
C��@�
=��녿!G�����C��\                                    Bx����  �          A�@�{����?:�H@��C��@�{�����@  ��
=C��                                    Bx���&  �          Ap�@�G��ə�>\)?h��C��@�G���(���Q��ffC�j=                                    Bx����  �          A��@θR�ʏ\>�=q?��HC��=@θR��{�����
C��                                    Bx���r  �          A
�H@�����
=��ff�>�RC�8R@�����(�����j�HC���                                    Bx���  �          A�\@~�R��  ?��
@��HC�e@~�R���ÿW
=���C�Z�                                    Bx���  �          AG�@vff���?�  @�{C��@vff���ÿaG��ÅC��=                                    Bx��d  �          A@B�\�������C��
@B�\���E���  C��R                                    Bx��
  �          A�
@S�
���333��z�C��@S�
����%���C���                                    Bx��-�  T          A=q@]p���Q쿇���\)C���@]p��׮�8Q�����C���                                    Bx��<V  T          AQ�@��\�����#33���C�@��\������H��C�                                      Bx��J�  �          A\)@�=q���
�P����ffC�Q�@�=q���\��33�
C�9�                                    Bx��Y�  �          Aff@�p����
��ff�M�C�)@�p���{�X���ŮC��                                    Bx��hH  T          A33@������H��
=�!��C���@�����  �AG����\C�8R                                    Bx��v�  �          A��@�\)�У׿:�H��p�C���@�\)�Å�����=qC��q                                    Bx����  �          @�
=@���������C�f@���z��
=�`��C���                                    Bx���:  �          A Q�@�����33>aG�?ǮC�7
@������\�.�HC���                                    Bx����  �          @�\)@\)��=q?0��@��C�,�@\)��Q쿏\)�ffC�G�                                    Bx����  �          @�ff@G���{?��\@�C���@G���R�h�����HC���                                    Bx���,  �          @�z�@�����?O\)@�ffC���@�������
{C���                                    Bx����  T          @���@S�
��z�?0��@�Q�C�t{@S�
��녿�(���RC���                                    Bx���x  �          @��@s�
���
?.{@�C��H@s�
�ٙ������C��                                     Bx���  T          @�ff@�  ��\)?��
A(�C�Z�@�  �ڏ\�\)���C�,�                                    Bx����  T          @�\)@�ff�ƸR��G��E�C�K�@�ff���R��G��L��C��H                                    Bx��	j  �          @�
=@��\��p�?�(�A*{C��q@��\��=q��p��*�HC�u�                                    Bx��  �          A�@�33��  @�\Aip�C��@�33���>aG�?��
C�K�                                    Bx��&�  T          A��@�����@,��A��
C�Q�@����G�?E�@�C�s3                                    Bx��5\  �          Az�@����ڏ\?�
=A5��C���@������þk���ffC�&f                                    Bx��D  �          A�R@�����녿=p����C��@�����z����|��C�{                                    Bx��R�  �          A�@��
��(��#�
��C�}q@��
��\)�{�v�RC�|)                                    Bx��aN  �          A�@�������    �#�
C�o\@������\��=q�2{C��)                                    Bx��o�  �          A@��H���R?z�H@�ffC��
@��H���þ��Z�HC���                                    Bx��~�  T          A
=@�=q��\)?���@�p�C�  @�=q����W
=��p�C���                                    Bx���@  �          A(�@��H��=q�.{���HC��@��H��33���33C��f                                    Bx����  �          A33@�p��\)�����1��C�  @�p��Z=q�&ff��G�C��                                    Bx����  T          A{@���s�
��=q�
=C���@���S33������C�j=                                    Bx���2  �          A@ƸR��{>��?�ffC�
=@ƸR���ÿ���
=C�|)                                    Bx����  �          A�H@������?���@��C��R@�����(���R���C�o\                                    Bx���~  T          A�R@������?�{@��\C�
@����33�
=q�tz�C��                                    Bx���$  �          Aff@�������?�z�A ��C�H�@�����  �L�;�p�C��H                                    Bx����  T          A�R@љ����?У�A:�HC�.@љ���z�>�{@p�C�XR                                    Bx��p  �          A�
@�  ��?�=qAP��C��@�  ���R>.{?���C�XR                                    Bx��  �          Az�A���H�#�
��\)C��fA��녾�ff�G
=C�#�                                    Bx���  �          A
=A����>�  ?�p�C���A�����Q���C��H                                    Bx��.b  �          A\)A z�Ǯ?�R@�33C�~�A z��>#�
?��C�)                                    Bx��=  T          A�@�ff�Y��?�=qA/33C��q@�ff�mp�?\)@uC��                                    Bx��K�  �          A  @޸R��  ?��@��
C�K�@޸R��z�#�
��{C��                                    Bx��ZT  �          A��@�
=���?�33AUG�C��@�
=��>���@�C�9�                                    Bx��h�  T          AQ�@ۅ���
?��
A�RC���@ۅ�������
�\)C�:�                                    Bx��w�  T          A\)@�����(�?�ff@�33C�c�@������R���O\)C�'�                                    Bx���F  T          Az�@�  ����?���A33C��@�  ���u��z�C���                                    Bx����  �          A  @�Q����\?�  @�G�C��
@�Q������(��?\)C�y�                                    Bx����  �          A�
@�33��33@'�A�G�C��{@�33���
?c�
@���C��3                                    Bx���8  �          A
=@��\����@0  A�\)C�G�@��\��33?�{@���C���                                    Bx����  �          A�H@������?��HAB�\C��=@�����(�>#�
?�\)C�,�                                    Bx��τ  �          A ��@�������@Q�Ax  C���@�����p�>�Q�@&ffC��                                    Bx���*  �          A z�@�G���Q�?��\A�\C���@�G��˅�����=qC�}q                                    Bx����  �          A (�@��ƸR@�
Au��C���@���G�>#�
?�
=C�K�                                    Bx���v  �          A ��@������?�ffA=qC���@����ff������C�g�                                    Bx��
  �          A   @�=q��G�?�@x��C�޸@�=q���R���
��G�C��                                    Bx���  �          @�{@��H���@0  A��\C���@��H��\)?�ff@��HC�5�                                    Bx��'h  �          @�{@������@ffA�Q�C�b�@�����\)>��@Z=qC�w
                                    Bx��6  �          @���@��R����@8Q�A�p�C�z�@��R�Ӆ?z�H@��C�B�                                    Bx��D�  �          @��H@����z�?�
=Adz�C�<)@����p��#�
���
C���                                    Bx��SZ  �          @�\)@L����
=?��RAo�C���@L����Q콣�
�!G�C�G�                                    Bx��b   �          @��@1���
=@\)A��HC�=q@1����H>��?���C��
                                    Bx��p�  �          @�\)@���޸R@ffA��
C�z�@�����H>8Q�?��C��                                    Bx��L  �          @��R@/\)�ٙ�@ffA�(�C���@/\)��{>aG�?�\)C�j=                                    Bx����  �          @�
=@%��ۅ@ ��A��C�N@%��陚>�33@$z�C��)                                    Bx����  �          @�G�?�Q���
=@ ��ArffC��3?�Q����u��C�XR                                    Bx���>  �          @�p�?����  ?�z�Ad��C�AH?������33�%C��                                    Bx����  �          @��R@���љ�@i��A���C�C�@����33?\A2�HC�S3                                    Bx��Ȋ  �          @��H>����@�=qB,��C�q�>���  @W�AͅC���                                    Bx���0  �          @��\=#�
����@��\B��C�4{=#�
��=q@0  A��HC�+�                                    Bx����  �          @���?333����@�
=B%G�C�]q?333��R@L��A���C�                                    Bx���|  �          @�(�?����R@��
B,33C�+�?��ڏ\@^{A��C�k�                                    Bx��"  �          @��?�(�����@���B'(�C�u�?�(��߮@Tz�A��C���                                    Bx���  �          A ��?����=q@��HB{C��R?����\@��Aw�C��q                                    Bx�� n  �          A Q�?�
=�߮@n{A�p�C��?�
=��G�?�z�A"�\C�Q�                                    Bx��/  �          @�p�?��R��33@qG�A�p�C�q?��R��?�  A.=qC��3                                    Bx��=�  �          @��\?����߮@VffAʸRC�}q?�����?�ff@�
=C�
                                    Bx��L`  �          @�  ?E�����@j=qA�C�@ ?E���=q?�33A'
=C��=                                    Bx��[  �          @�z�>W
=���
@q�A�p�C��>W
=��R?ǮA<��C��                                    Bx��i�  �          @�녾��R�أ�@U�A�33C��3���R��
=?�=qA�
C���                                    Bx��xR  �          @�33>W
=����@��HB33C��>W
=��33?�z�Aip�C�Ф                                    Bx����  �          @��>B�\��\)@tz�A��HC��=>B�\��\?ǮA9C��3                                    Bx����  �          @��u�У�@�G�A��
C���u��ff?���A[33C�{                                    Bx���D  �          @��>���33@c�
A�(�C�\>���(�?�=qA#�C��R                                    Bx����  �          @��?=p���{@�z�B=qC�l�?=p���@ ��Ax��C���                                    Bx����  �          @��?�=q��=q@��RBffC�0�?�=q��\@
=A�z�C�T{                                    Bx���6  �          @�
=?�\)��p�@j=qA�=qC��{?�\)��?�(�A5G�C�g�                                    Bx����  �          @�ff?��Ϯ@c33A���C�T{?�����?���A'33C�{                                    Bx���  �          @���=q�ڏ\@'�A��C�޸��=q�陚>�33@/\)C���                                    Bx���(  �          @��
>�����Q�@\(�A�z�C�W
>�����Q�?�(�Az�C�33                                    Bx��
�  �          @�{�L����{@N�RA�{C�#׾L�����
?u@�C�7
                                    Bx��t  �          @��;����z�@Mp�A�{C�9�������?s33@�p�C�c�                                    Bx��(  �          @�\�������@a�A�Q�C�W
�����?�=qA'\)C��H                                    Bx��6�  �          @�=q>�z�����@P  AӮC�Ff>�z���
=?��\A z�C�'�                                    Bx��Ef  �          @陚�
=q��Q�@'�A�G�C��
�
=q��>�33@/\)C�޸                                    Bx��T  �          @�\?�\)��(�@5�A�z�C���?�\)��?z�@���C�o\                                    Bx��b�  �          @��H?�����@Q�A�33C��?����Q�?��\@�  C�N                                    Bx��qX  �          @�?�����@K�A���C��=?����\?c�
@أ�C�N                                    Bx���  �          @��H?�Q���{@@��A��RC�
?�Q��陚?5@�z�C�q�                                    Bx����  T          @�\@   ��\)@5�A��C�Ff@   ����?�@�Q�C��                                    Bx���J  �          @�=q?�������@*�HA��C�f?������>�33@,(�C��                                     Bx����  T          @��?�=q��Q�@4z�A���C��
?�=q���?   @uC�)                                    Bx����  T          @�  ?�����=q@�RA���C�s3?�����Q�>��?�
=C�f                                    Bx���<  �          @�@   ����@:�HA�
=C��q@   ��(�?=p�@���C��q                                    Bx����  �          @�(�@����
@b�\A�33C���@���?�\)A+�C�|)                                    Bx���  �          @�{?��R��33@Dz�A���C�XR?��R��?E�@�C��                                    Bx���.  �          @�(�?�\��G�@7�A��C��{?�\��?
=@�z�C��q                                    Bx���  �          @��
@Fff���@4z�A�=qC��=@Fff��z�?333@��C�~�                                    Bx��z  T          @陚@<(����@'�A�33C�<)@<(���{?\)@�Q�C�G�                                    Bx��!   �          @��@?\)�ə�?ǮAJ=qC��{@?\)��
=�����HC�j=                                    Bx��/�  �          @��@\)����?�
=A8��C�s3@\)��z�G��ȣ�C�J=                                    Bx��>l  �          @��@@  ��
=?�\@��C�t{@@  ���ÿУ��R�\C�Ǯ                                    Bx��M  �          @�G�@����
=@	��A�33C��f@������>���@S33C���                                    Bx��[�  
�          @�(�@�G�����@��A�p�C�p�@�G���(�?!G�@�p�C��R                                    Bx��j^  T          @�R@�Q���ff@
=A��C�*=@�Q���>�G�@a�C��                                    Bx��y  
�          @�ff@s33���?��AE��C�\)@s33���H���H�z=qC���                                    Bx����  �          @�@h����?���A;�
C��f@h���\�
=��\)C�xR                                    Bx���P  �          @�Q�@�33����>\@E�C���@�33���R��G��G\)C�q�                                    Bx����  �          @޸R@�����G�?�  A  C�{@�����녿c�
��(�C�
=                                    Bx����  
�          @�ff@���{?�=qA0��C�@ @����H���s33C�ٚ                                    Bx���B  �          @���@{�����?��\A*{C���@{���z�!G���  C�l�                                    Bx����  
�          @��@qG����?�(�A"�\C��3@qG���  �:�H���C��H                                    Bx��ߎ  �          @�33@qG���{?   @�p�C��f@qG����ÿ��?�
C�&f                                    Bx���4  �          @��H@{����?
=q@�=qC���@{�������4��C���                                    Bx����  �          @��@������?fff@�Q�C���@�����׿h����(�C��H                                    Bx���  �          @��
@w
=����>���@#33C�,�@w
=�������X  C��
                                    Bx��&  �          @���@j=q���?
=@�\)C�p�@j=q���ÿ���6�RC���                                    Bx��(�  �          @���@�33���\?J=q@�
=C��@�33��G�����G�C��                                    Bx��7r  �          @�33@�����
    =#�
C�%@����녿�ff�s�C���                                    Bx��F  �          @���@1G����\��Q��-��C�33@1G���33�@  ��C���                                    Bx��T�  �          @�
=?#�
��ff�����Y�C��?#�
���
�W����C���                                    Bx��cd  �          @�Q�?���G���
=�}��C���?���z��a��Q�C���                                    Bx��r
  T          @�  ?�(���ff����G�C�/\?�(���(��0  ��G�C��f                                    Bx����  �          @ۅ>�G���G��0������C�ٚ>�G�����<(���  C�
=                                    Bx���V  
�          @�33���=q�
�H���\Cx����  ��Q���Cs��                                    Bx����  �          @�z�fff��녿�ff�RffC�
�fff�����k��  C�xR                                    Bx����  �          @ᙚ>k���33��33�  C��R>k������Y����\C�R                                    Bx���H  �          @�@������H?�33A;�
C��3@��������#�
����C�AH                                    Bx����  �          @�z�@�=q���?��A]�C��f@�=q���\���ͿO\)C���                                    Bx��ؔ  �          @�
=@vff��Q�?B�\@�Q�C��@vff��p���  �$��C�R                                    Bx���:  �          @��@j=q��\)        C��R@j=q���
��\��p�C��                                    Bx����  �          @�Q�@E����ü��
�8Q�C�@E������
�H��Q�C��=                                    Bx���  �          @���@�  ���@
=A�\)C�w
@�  ����?��@���C���                                    Bx��,  �          @�=q@�{�aG�@33A��\C���@�{��Q�?8Q�@��HC��                                    Bx��!�  �          @��@����\)?z�HAQ�C���@������H��ff�uC�n                                    Bx��0x  �          @�=q@�G���  ?(�@�z�C���@�G���ff�c�
��\)C�Ǯ                                    Bx��?  �          @�  @�
=���?���AffC�� @�
=��\)����`  C�|)                                    Bx��M�  �          @��@����y��?���A^{C�n@������R>#�
?�=qC�\)                                    Bx��\j  �          @���@q�����?fff@�z�C�33@q���  ������C�Ff                                    Bx��k  �          @ڏ\@l�����R>8Q�?�G�C�w
@l�������u�C�'�                                    Bx��y�  �          @���@��
��z�L�Ϳ�
=C�/\@��
��Q���H��G�C�J=                                    Bx���\  �          @�Q�@�=q��p���p��FffC��{@�=q��
=�Q���\)C�<)                                    Bx���  �          @�p�@p����{�@  ��  C�T{@p�����H�%��\)C��                                    Bx����  �          @���@X����Q쿴z��Dz�C�˅@X�����N�R���C��{                                    Bx���N  �          @љ�@g
=���H�u��C�f@g
=��p��0����(�C���                                    Bx����  �          @�\)@@�����׿��H�P��C�S3@@����p��R�\����C�o\                                    Bx��њ  �          @�=q@#33���
��(��2=qC�33@#33��33�Fff��  C��H                                    Bx���@  T          @��R@(Q������R��
=C�f@(Q��z�H�e��
C��f                                    Bx����  �          @���?�����׾�����RC��?������p����\C���                                    Bx����  �          @��
�����  @�A�p�Cr���������?(�@�  Cu(�                                    Bx��2  �          @�Q쿬����?�ffA��\C��������
=�aG��
=C�Y�                                    Bx���  �          @�?^�R���>#�
?��RC��?^�R��\)��(����\C�L�                                    Bx��)~  �          @�
=>�ff��=q>B�\?�=qC�<)>�ff��Q��\)��33C�\)                                    Bx��8$  �          @�
=?(���\�����2�\C��?(�����H��H���
C�aH                                    Bx��F�  �          @\?����G��G����
C��?����z��0������C��q                                    Bx��Up  �          @���?���{�W
=�\)C�K�?������,������C�9�                                    Bx��d  �          @��\?�33�����ff�w\)C�@ ?�33�����Vff�\)C���                                    Bx��r�  �          @���?��R�����\)�4  C��{?��R��p��=p�����C���                                    Bx���b  T          @���?������>�z�@>{C���?����G���Q����\C�˅                                    Bx���  T          @��\>���
=?�A���C��{>���G��\)��\)C��=                                    Bx����  �          @���B�\����@Q�A��RC��׿B�\����>�Q�@c33C�>�                                    Bx���T  �          @�33�Y����p�@#33A��C�XR�Y�����?\)@�
=C���                                    Bx����  �          @���aG���Q�@8��A��HC�
=�aG����R?n{A��C���                                    Bx��ʠ  �          @��H���\��@N�RB��C~ٚ���\����?�{AW\)C���                                    Bx���F  �          @�z��z���ff@@��A���Cz�
��z���
=?��A4(�C}!H                                    Bx����  �          @��Ϳ����33?�
=A:=qC�~��������c�
�
=C��                                    Bx����  �          @�z῝p���G�?�(�Ag�
C��)��p���ff������C��                                    Bx��8  �          @�33��Q���
=?���Ae�CJ=��Q���(������p�C�                                    Bx���  �          @��ÿ�p����?
=@�{C�Ϳ�p���z�\�lQ�C��                                     Bx��"�  �          @�z�333���
?ǮAu��C�t{�333��녿\)��C���                                    Bx��1*  �          @��
�\(���G�@  A�Q�C�b��\(����>W
=@z�C���                                    Bx��?�  �          @��\�G����H@*�HA�
=C��׿G����R?+�@ӅC��                                    Bx��Nv  T          @��
��G�����@?\)A��HC�}q��G�����?}p�A�HC��\                                    Bx��]  
�          @�(��333��
=@C�
A��C����333���?�=qA(��C��f                                    Bx��k�  �          @��\����@E�A���C��
�\��=q?��A%G�C�                                      Bx��zh  �          @�zᾮ{���@j�HB��C����{���\?��
A��C�Ff                                    Bx���  �          @��;���c�
@�ffBSQ�C�Ǯ�����Q�@FffA�{C�!H                                    Bx����  �          @�z��R�hQ�@��HBL�HC�.��R��G�@>{A��C���                                    Bx���Z  �          @�G�<��
�7
=@���Bqz�C�<)<��
��\)@j=qB��C�&f                                    Bx���   �          @���?������
��33�Y�C�޸?�������U�G�C�
                                    Bx��æ  �          @���?8Q����\�C33���
C��?8Q��h�����Nz�C��R                                    Bx���L  
(          @��
?�33�����3�
���C�)?�33�z�H�����?��C�&f                                    Bx����  
�          @�  ?�33������
��z�C�  ?�33�~{�����)��C�˅                                    Bx���  "          @�{@x���s33�����  C��
@x���/\)�Z�H��
C�e                                    Bx���>  �          @�  @#�
��{�*=q�ә�C�P�@#�
�Y����p��1�C�t{                                    Bx���  "          @�Q�?����R�  ��33C��R?���G��~�R�'�C�<)                                    Bx���  T          @�Q�?��
�����=p���z�C��
?��
�`  ��G��D��C�z�                                    Bx��*0  "          @��@���p�����33C���@������vff�!��C�K�                                    Bx��8�  �          @��?����(���G��A�C��
?�������N{�  C�G�                                    Bx��G|  �          @���@   ���\�������C�@   �vff��(��.ffC���                                    Bx��V"  
�          @�{@����*�H��=qC�"�@�\(����R�5Q�C��                                    Bx��d�  �          @�p�@�����  ��p�C��@��x���|(��'��C���                                    Bx��sn  �          @�ff?�p���{��z���C�l�?�p���z��j�H�C��                                    Bx���  �          @�
=?�(�����=q����C�N?�(��xQ�����0z�C��R                                    Bx����  �          @�\)?�Q����R�E���
=C�` ?�Q�����-p�����C�T{                                    Bx���`  �          @��?����
=��p��qG�C���?����ff��ģ�C�ff                                    Bx���  �          @�ff?���Q�
=��C��)?�����#�
���C���                                    Bx����  �          @�{?��
���׿O\)�\)C�H�?��
���H�1���=qC��                                    Bx���R  �          @�{>�(���
=��{�^�RC�:�>�(����H�QG��\)C���                                    Bx����  �          @�{?Tz���{��33�d(�C�T{?Tz������S33��C�*=                                    Bx���  �          @�{?5��Q쿞�R�Ip�C��f?5��p��K����C�K�                                    Bx���D  �          @��R>���z�z���33C���>���G��'
=���C��)                                    Bx���  �          @�{�����(�?�@��
C��f�����������
=C��3                                    Bx���  �          @���?\(���녾�p��qG�C�^�?\(���G�����ȣ�C���                                    Bx��#6  �          @�ff?����\)>�?��
C�ٚ?����z��33��\)C�H�                                    Bx��1�  �          @�G�?��\��(�>�{@Y��C�XR?��\�����  ��\)C���                                    Bx��@�  �          @��?u��
=<#�
>\)C�Ǯ?u��=q���\)C�!H                                    Bx��O(  T          @�?E���33�Ǯ�|��C��f?E��������
=C�O\                                    Bx��]�  �          @�p����
��33��ff���\C��ὣ�
��G���R���
C��\                                    Bx��lt  T          @���?+����Ϳ�  �NffC��=?+�����I���	�C�/\                                    Bx��{  �          @�?�����  �5��
=C�+�?�����33�,������C��R                                    Bx����  �          @�ff?��
�����  �#33C�S3?��
��\)�=p���C�'�                                    Bx���f  T          @��?G���\)���H�l(�C��?G������XQ���
C���                                    Bx���  �          @�Q�=�G��������H�up�C��{=�G���z��S�
�=qC��3                                    Bx����  �          @�\)��ff���R�}p��'�
C�R��ff��
=�6ff���C}                                    Bx���X  �          @��������33����@z�C�=q������G��B�\���C��\                                    Bx����  T          @��׾8Q���p��h����C���8Q���ff�7
=���\C��H                                    Bx���  �          @�\)�W
=���H���\�-p�C�⏾W
=���\�;��ffC��3                                    Bx���J  T          @��\?#�
���R�B�\���
C�XR?#�
��G��.�R���C��\                                    Bx����  �          @���?&ff����G��ffC�h�?&ff��\)�.�R��=qC��                                    Bx���  T          @���?����=q�J=q��C��?�������-p���  C�޸                                    Bx��<  �          @��R?
=��z�ٙ���33C�K�?
=��z��_\)�33C�{                                    Bx��*�  T          @�  ?0����ff�У���  C���?0����
=�\���p�C��                                    Bx��9�  �          @��?Q���Q쿣�
�Xz�C�p�?Q�����I���Q�C�H�                                    Bx��H.  T          @��?����R��33�k�
C�W
?�����O\)��HC���                                    Bx��V�  T          @�p�?c�
��\)�fff�\)C�� ?c�
��  �8Q���(�C�Q�                                    Bx��ez  T          @�  ?k���녿���'�
C���?k������B�\� (�C�xR                                    Bx��t   T          @�p�?��H��p���{�4��C�Ff?��H����C33�ffC�S3                                    Bx����  T          @�{?��
�����
=�8  C���?��
�����L(���C��=                                    Bx���l  �          @�z�?��
��녿^�R�{C��H?��
����>{���C���                                    Bx���  �          @�=q?�{��p��^�R���C�
?�{��{�;���=qC�W
                                    Bx����  �          @���@���(������9��C�&f@���33�Q����C�
                                    Bx���^  �          @�G�@=q��ff�:�H��\)C��@=q�����-p�����C�`                                     Bx���  �          @��H@.�R����^�R�33C���@.�R��z��3�
��(�C�<)                                    Bx��ڪ  �          @�G�@G����R���\�  C�N@G����>�R���C��3                                    Bx���P  �          @�p�@(����
�z����
C���@(�����(�����C�H                                    Bx����  �          @�  ?�33��G����\�J{C�ٚ?�33����P  �\)C���                                    Bx���  
�          @�ff=��
���ff��p�C�u�=��
�z�H���
�9�C���                                    Bx��B  
�          @���=�����=q��\���\C��==�����=q���
�5G�C��{                                    Bx��#�  �          @���?
=���׿�p�����C��?
=��(��y���%�HC��)                                    Bx��2�  
�          @�\)?�R��{��Q����HC�<)?�R����fff��HC���                                    Bx��A4  �          @�=q@=q�������:�\C��@=q��ff�;����C��                                    Bx��O�  $          @�G�@e���G��Q���G�C�^�@e���(��#�
��z�C�o\                                    Bx��^�  R          @�{@(Q���{�����:�\C�q�@(Q�����C�
��(�C��\                                    Bx��m&  "          @���?�
=��녿�ff�o
=C�h�?�
=��=q�aG����C�R                                    Bx��{�  
�          @���?�33��
=��
=���C��\?�33��{�g��\)C��3                                    Bx���r  �          @���@7
=��  �fff�(�C���@7
=��Q��9���ܣ�C���                                    Bx���  
�          @��@G
=���@  ��33C��@G
=��  �/\)��  C��
                                    Bx����  "          @�@7
=���8Q���C���@7
=��Q��-p����C��                                    Bx���d  �          @�z�@.�R��ff�+��ə�C�L�@.�R�����+�����C�Ф                                    Bx���
  �          @���?�{���R�B�\���C��?�{��
=�z����
C��{                                    Bx��Ӱ  �          @�\)?޸R����xQ��p�C���?޸R���H�@����RC��                                    Bx���V  "          @�=q@����;�����\C��H@����H����=qC��                                    Bx����  �          @�33?�������H��
=C��?����z��hQ��C�AH                                    Bx����  �          @�(�?G���ff�   ����C�>�?G�����s�
�*�RC�h�                                    Bx��H  T          @�Q�?0����ff�G��Ù�C��{?0���n�R�~{�9G�C�9�                                    Bx���  �          @��\?.{��  ���ɅC�ٚ?.{�n�R���\�<G�C�!H                                    Bx��+�  �          @�{?333���
��\��C�q?333�i���}p��;G�C�w
                                    Bx��::  �          @���?8Q���z��
=q���HC�*=?8Q��mp��u�5�
C�t{                                    Bx��H�  �          @��?fff���R�
=��
=C�&f?fff�l������;Q�C��R                                    Bx��W�  �          @��>�(���p��
�H����C���>�(��o\)�w
=�7
=C�K�                                    Bx��f,  �          @��R=#�
���
�p���z�C�AH=#�
�e�����C�C�Y�                                    Bx��t�  �          @��\?.{��Q��  ��{C��H?.{�r�\�~�R�7�C��                                    Bx���x  "          @��
@;����
���H�f=qC��@;��}p��L(����C�AH                                    Bx���  T          @���@I����33����#33C�}q@I����33�333��33C���                                    Bx����  
�          @��@E���ff�����0��C��H@E��|(��2�\��\)C�f                                    Bx���j  	b          @�{@K�������p��G�
C�y�@K��p  �6ff��G�C�'�                                    Bx���  	�          @�ff@A���{�����1C�u�@A��{��2�\��z�C���                                    Bx��̶  T          @�p�@4z���녿�G��#\)C�1�@4z����\�0  ��  C�Y�                                    Bx���\  �          @��R@=p���\)��\)�6=qC�
=@=p��|���5���=qC�n                                    Bx���  �          @�
=@H�����
��(��D��C�
@H���tz��7���\)C��{                                    Bx����  �          @���@p  ���׿s33�Q�C���@p  �e�   ��p�C��                                    Bx��N  �          @�G�@XQ����H�u���C�0�@XQ��x���'���C���                                    Bx���  �          @�
=@C33���ÿ5��C�G�@C33����{�̣�C�&f                                    Bx��$�  �          @�\)@HQ�����8Q���\)C��R@HQ�����p���p�C��H                                    Bx��3@  T          @���@?\)���
�������C��@?\)�����ff����C�g�                                    Bx��A�  �          @�ff@(Q���Q�   ��G�C��{@(Q���ff�
=��{C�Ff                                    Bx��P�  �          @��R@'
=��G�<�>��RC��{@'
=�������z�C��R                                    Bx��_2  �          @�p�@5��33�:�H���
C�*=@5���R� ����(�C��R                                    Bx��m�  �          @�{@7
=���׿��>ffC�~�@7
=�~{�8����Q�C��f                                    Bx��|~  �          @��
@33��������RC�n@33��=q�����Q�C���                                    Bx���$  �          @�
=@%�����h�����C�E@%���G��(Q���{C�AH                                    Bx����  �          @��H@2�\��\)�c�
���C��
@2�\�s�
� ����Q�C�)                                    Bx���p  �          @�(�@@  ��Q쿂�\�8��C�e@@  �Tz��{��RC�f                                    Bx���  T          @�  @Mp����\�:�H� (�C��@Mp��`  �{��Q�C�H�                                    Bx��ż  T          @��@%�����Ǯ��ffC�\@%��Mp��@  �{C�`                                     Bx���b  T          @�
=?�Q���z��8���\)C��?�Q��/\)����[Q�C��)                                    Bx���  �          @��H@��q��G����HC��@��(���a��:��C���                                    Bx���  �          @��
@���9��?���AZffC��R@���L��>8Q�?�ffC�k�                                    Bx�� T  �          @��@����G�?E�@��C�� @����K������=qC��H                                    Bx���  �          @��@����S33?
=@�  C���@����QG��333��p�C��
                                    Bx���  �          @�33@���@��?\)@��C�t{@���@  ��R�\C��H                                    Bx��,F  �          @��@��H�L(�����=qC�O\@��H�9����=q�TQ�C���                                    Bx��:�  �          @�\)@����I��=u?
=C�N@����;�����7\)C�4{                                    Bx��I�  �          @�Q�@��
�C�
<��
>#�
C��{@��
�5�����733C��                                    Bx��X8  �          @�\)@�Q��+�>��@��
C��@�Q��(�ÿ!G���z�C�!H                                    Bx��f�  �          @�=q@��R�
=?��A-G�C�˅@��R�&ff=�?�C���                                    Bx��u�  �          @���@�z��=q?��HAAG�C�o\@�z��,(�>W
=@ffC�/\                                    Bx���*  �          @��
@���)��?�ffA-��C���@���7
=������C��f                                    Bx����  �          @�(�@�  �=p�>��H@�  C��@�  �:�H�(����G�C�7
                                    Bx���v  �          @�p�@�G��>�R>�33@eC�\@�G��8�ÿL���{C�p�                                    Bx���  �          @��@����q�<��
>uC�� @����`  ��z��k\)C��\                                    Bx����  �          @��H@���b�\���Ϳ�ffC�Ff@���O\)��
=�n=qC���                                    Bx���h  �          @��@��\�vff������C�T{@��\�aG��Ǯ��p�C���                                    Bx���  �          @��@�{�g��#�
�У�C��
@�{�R�\��G��}��C���                                    Bx���  �          @��
@����`  �L���G�C���@����J=q��  �xQ�C�%                                    Bx���Z  �          @�@�(��dzᾸQ��h��C�k�@�(��J�H��
=��(�C��                                    Bx��   �          @�ff@�{�a녾��H��C��@�{�E����
��=qC��                                     Bx���  �          @�@�\)�j�H�Y���	p�C��@�\)�E��
�H��(�C��
                                    Bx��%L  �          @��@���^{�&ff���C��=@���>{��z����
C��                                    Bx��3�  T          @�ff@�p��(�ÿTz��33C�� @�p��Q��\���
C�E                                    Bx��B�  �          @��@��R��p�<�>�\)C�0�@��R��\)�!G��ə�C��H                                    Bx��Q>  �          @�z�@��H���>L��@G�C�e@��H�33�8Q���33C�޸                                    Bx��_�  �          @��
@�(��\)�W
=��C�0�@�(��   ����*�HC�Z�                                    Bx��n�  �          @���@����"�\�   ��G�C��
@����
=q���hz�C�XR                                    Bx��}0  �          @�  @�33� �׿������C���@�33�У׿����S�C���                                    Bx����  �          @��@�p���H���H��(�C�t{@�p���
��{�\z�C�%                                    Bx���|  �          @�
=@��\�#�
>���@@��C��f@��\��R�0����p�C���                                    Bx���"  �          @�@�33�<(���Q�aG�C�j=@�33�,(���Q��A�C���                                    Bx����  �          @�@�z��H�ÿaG���C��{@�z��$z��   ��\)C�z�                                    Bx���n  �          @�  @�{�QG�>�p�@k�C���@�{�J=q�c�
�
=C��)                                    Bx���  Q          @�p�@�z��Mp�>��H@�\)C��=@�z��J=q�@  ��z�C��H                                    Bx���  �          @�
=@�p��[�?���A1G�C��@�p��e��\)�2�\C�o\                                    Bx���`  T          @�  @�  �?\)?\(�A
�HC��@�  �E�����U�C�y�                                    Bx��  �          @�ff@�Q��E=�?�G�C���@�Q��:=q��ff�*{C�J=                                    Bx���  �          @�{@���G�>�@�=qC�Q�@���Dz�:�H���
C���                                    Bx��R  �          @�@�Q��Dz�>u@{C���@�Q��;��n{��\C�.                                    Bx��,�  �          @�@�
=�E����
�8Q�C�p�@�
=�6ff��
=�@z�C�t{                                    Bx��;�  �          @�ff@�  �U��fff�{C���@�  �0  ������C�Ff                                    Bx��JD  �          @�z�@�z��HQ콣�
�W
=C��q@�z��7���  �N�HC�                                      Bx��X�  
�          @�(�@���J=q�}p��#�C�j=@���#�
�ff��\)C�'�                                    Bx��g�  
�          @�=q@���K��L�;��C��3@���;���p��MG�C���                                    Bx��v6  "          @�\)@�=q�Q�?�\)Ai��C��@�=q�\)>�@�  C�^�                                    Bx����  �          @�\)@�{�33?�Q�AHz�C��R@�{�ff>���@^{C�H�                                    Bx����  �          @�  @��p�?}p�A%C��=@���H=�Q�?fffC��                                    Bx���(  �          @�(�@�{�#33?L��A=qC�Z�@�{�*=q�aG���RC��R                                    Bx����  "          @���@��\�7
=>�G�@�\)C��\@��\�4z�(���أ�C��H                                    Bx���t  T          @��\@��\�*�H?G�A ��C��H@��\�1G���\)�7�C��                                    Bx���  
�          @��
@���+�?W
=A
=qC��@���333�aG��G�C��                                    Bx����  
�          @�(�@��>�R?c�
A��C��H@��Fff��z��>{C�AH                                    Bx���f  �          @�{@�ff�E?Q�A�C�Z�@�ff�J�H�Ǯ�}p�C�                                      Bx���  
�          @�(�@���=p�?333@�{C�@���@�׾�ff��33C��\                                    Bx���  T          @�G�@�G��Dz�?+�@�ffC��q@�G��E����\)C�޸                                    Bx��X  �          @���@�p����?��A.�RC���@�p���R>�?��C���                                    Bx��%�  T          @�G�@����33?z�HA#�C�Ǯ@���Q�>L��@C��=                                    Bx��4�  �          @���@�z���?�\)AD��C���@�z��Q�>�(�@�ffC�0�                                    Bx��CJ  �          @�z�@����\?W
=A
=C���@���p�<#�
>��C��f                                    Bx��Q�  
�          @���@���(�?k�A��C�"�@��
�H=�?�  C�%                                    Bx��`�  %          @�  @�G��ff?ٙ�A��
C���@�G��3�
?5@��HC�Z�                                    Bx��o<  Q          @��@��
=?�Q�A�33C���@��4z�?0��@��C�                                    Bx��}�  T          @�(�@�Q���
?��HA�
=C�y�@�Q��1�?:�H@�=qC�4{                                    Bx����  �          @��@|���Z=q?�  AV�\C���@|���hQ��G���p�C��3                                    Bx���.  T          @��@w��n�R>�(�@���C��@w��g��}p��)��C�y�                                    Bx����  �          @�ff@�{�G�?�@���C�n@�{�E�+���C��                                    Bx���z  �          @�
=@����@��>\@|(�C�1�@����<(��E��{C���                                    Bx���   �          @�Q�@���,(�=���?���C�e@���!녿fff��HC��                                    Bx����  T          @���@�=q�p�>W
=@
=qC�=q@�=q�Q�!G��ϮC��                                     Bx���l  T          @�G�@��\�G
=    ��C��f@��\�8�ÿ�z��@��C��H                                    Bx���  �          @�=q@�
=�:�H>B�\?�p�C�%@�
=�1녿fff��C��H                                    Bx���  T          @���@��:�H>�?���C�H@��0�׿xQ��!G�C���                                    Bx��^  �          @�ff@�z��5��L����C�P�@�z��#33��p��Qp�C��{                                    Bx��  �          @�
=@�(��7
=������\C�'�@�(��   ��Q��t��C��                                    Bx��-�  �          @��@��H�H�ÿ:�H��C��
@��H�(�ÿ�=q����C��                                    Bx��<P  T          @�p�@�z��H�ÿY���	G�C��R@�z��&ff�������RC�\)                                    Bx��J�  �          @��@�{�J�H���\�)�C�=q@�{�$z�����C��)                                    Bx��Y�  �          @��@�ff�HQ쿓33�>�RC�w
@�ff��R�{��Q�C�p�                                    Bx��hB  �          @��@��
�vff�u���C�Q�@��
�_\)��33��G�C��
                                    Bx��v�  �          @��H@�z��o\)�:�H��{C��@�z��Mp���
��C��                                    Bx����  "          @�=q@�\)�w��Y����C���@�\)�Q��{��=qC��                                    Bx���4  �          @���@�
=�Q녾�z��6ffC��q@�
=�<(���(��mG�C��                                    Bx����  �          @���@�{�j�H�k��  C�:�@�{�U��Ǯ�{33C���                                    