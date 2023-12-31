CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230324000000_e20230324235959_p20230325021552_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-25T02:15:52.992Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-24T00:00:00.000Z   time_coverage_end         2023-03-24T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxq�   "          A�p��K�
�&=q�-��p�C[5��K�
�>�R�k\)�A\)CA.                                    Bxq��  �          A��R�EG��)���.=q�33C\���EG��HQ��m��E��CB:�                                    Bxq�L  �          A��\�D���,(��,  �Q�C](��D���U��mp��EQ�CC(�                                    Bxq��  �          A�z��EG��-p��*=q�C]T{�EG��^{�lz��Dp�CC��                                    Bxqޘ  "          A��\�K��$���+���HCZ�3�K��=p��h���@\)CA�                                    Bxq�>  �          A����Lz��%��*�\���CZ�f�Lz��AG��hz��?�CAE                                    Bxq��  �          A��R�L(��)p��&�H� �RC[���L(��XQ��g��>��CB��                                    Bxq
�  �          A�ff�TQ��!�#
=��\)CYO\�TQ��Fff�`Q��7=qCA&f                                    Bxq0  "          A���P���&=q�((�� ��CZ���P���J�H�f�H�<33CA�f                                    Bxq'�  
�          A�ff�L���)��+��=qC[���L���P  �k�
�@�CB@                                     Bxq6|  
�          A��R�L(��/33�(��� z�C\���L(��hQ��lQ��@Q�CC�)                                    BxqE"  �          A�
=�S\)�$���(��� CY���S\)�E�f�R�;  CA.                                    BxqS�  
�          A�
=�_����+
=�(�CUk��_��ff�_\)�2�\C<��                                    Bxqbn  �          A����[���
�'�����CW\)�[��*=q�`���4�\C>��                                    Bxqq  "          A����\z��)p������ffCY�=�\z��w��[��/=qCC��                                    Bxq�  
�          A����_�����"ff���
CW��_��8Q��\���033C?�f                                    Bxq�`  T          A����_
=��H�(��� p�CV��_
=�ff�_\)�2�\C=�{                                    Bxq�  
�          A����b�H�{�((����CT�=�b�H�Q��\(��/Q�C<�\                                    Bxq��  �          A�G��i�	p��'33��33CRu��i���H�V�R�)z�C:�                                    Bxq�R  "          A���i��\)�%����
=CR���i���\)�Vff�)�C;O\                                    Bxq��  "          A�z��dQ��33�&�R��  CT)�dQ���\�Yp��-G�C<!H                                    Bxqמ  "          A��R�[33���*�H�ffCV��[33��H�b=q�5�HC>
=                                    Bxq�D  "          A���W
=���.{���CX+��W
=�"�\�g
=�:33C>��                                    Bxq��  �          A���[
=�(��*{�33CW� �[
=�'��c
=�6{C>�)                                    Bxq�  "          A���a���  �%G���=qCU���a���#�
�\���/��C>L�                                    Bxq6  "          A�G��c�����"{���RCU�H�c��.{�Z�\�-=qC>��                                    Bxq �  �          A���f=q�������
CU���f=q�7��V�H�)�C?G�                                    Bxq/�  T          A�
=�d����
� z���ffCU�
�d���.�R�X���+�C>�\                                    Bxq>(  �          A�33�h���Q��
=���CU:��h���<(��T(��'33C?n                                    BxqL�  �          A�G��l���z������z�CTǮ�l���J=q�N�R�"  C@\                                    Bxq[t  T          A�p��iG������R��  CU=q�iG��?\)�T(��&C?��                                    Bxqj  T          A���j�\��H� z���=qCS\)�j�\�G��S��&��C<��                                    Bxqx�  �          A����lz���� z���Q�CR���lz���Q��%=qC<�                                    Bxq�f  T          A�33�k33�33�#\)��
=CR�)�k33��p��TQ��'=qC;��                                    Bxq�  �          A���e�Q��%�����CT#��e�
�H�Y��,G�C<��                                    Bxq��  �          A�G��g
=����$Q�����CT)�g
=����X(��+
=C<�f                                    Bxq�X  
�          A�\)�f�R�\)�&ff��{CSٚ�f�R�
=�Y��+��C<Q�                                    Bxq��  T          A���b�H����*�H���CT���b�H�33�]���0�C<5�                                    BxqФ  �          A����bff��
�0  ��HCS�3�bff��\)�_33�1��C:��                                    Bxq�J  �          A�{�b=q�(��'����\CU��b=q�#�
�^�\�0Q�C>=q                                    Bxq��  �          A�(��]���"�R�$����\)CXG��]���Mp��a�3{CA�                                    Bxq��  �          A�{�R=q�-��((�����C[z��R=q�i���j=q�<�CC��                                    Bxq<  
�          A��W33�,Q��"=q���
CZ�3�W33�s�
�d���6�\CC�
                                    Bxq�  �          A���[\)�%p��"{��Q�CY
=�[\)�]p��`���3\)CB(�                                    Bxq(�  �          A�G��[\)�!G��%����ffCXT{�[\)�HQ��a�4\)C@ٚ                                    Bxq7.  �          A�33�X���"�H�&�R��z�CX���X���K��c\)�6Q�CA0�                                    BxqE�  �          A����P(��ff�9���ffCW�
�P(���33�l���A�\C<J=                                    BxqTz  �          A��\�U�����2{��RCW���U��G��g�
�;��C=�                                    Bxqc   �          A����W��33�.�R�Q�CW���W��!G��f{�9z�C>��                                    Bxqq�  �          A�
=�W
=�(��/
=��\CW�q�W
=�#�
�g
=�:33C>�=                                    Bxq�l  �          A�\)�Y��"ff�'\)���CX�=�Y��I���c��6p�CA�                                    Bxq�  �          A���\  �%p��!����G�CX���\  �`���`Q��2��CBW
                                    Bxq��  
�          A����Q��%�-���CZJ=�Q��H���j�H�=��CAp�                                    Bxq�^  �          A����PQ��*�R�+��{C[W
�PQ��^{�k��>=qCB�                                    Bxq�  "          A���O��+\)�+��{C[���O��`  �l  �>�\CC�                                    Bxqɪ  �          A���M���,(��+\)�G�C[��M���dz��l(��?�\CC�                                    Bxq�P  �          A����S�
�333�\)��  C\=q�S�
��
=�b{�5Q�CF��                                    Bxq��  �          A�{�W��4(������(�C[���W����
�Y���.{CG޸                                    Bxq��  
�          A���X(��8���  ��z�C\�=�X(�����Tz��)��CI��                                    BxqB  �          A�{�@���\)�Jff�Q�CYk��@�׿�G��y���P�RC9�3                                    Bxq�  �          A�=q�XQ��)��陚CZ)�XQ���G��[��0�
CD��                                    Bxq!�  
�          A�z��N�H�)��)��C[G��N�H�_\)�h���=��CC�                                    Bxq04  
�          A�ff�H���ff�A��RCX�
�H�Ϳ��H�s��I{C;�                                    Bxq>�  T          A�Q��Jff�G��9��RCY�
�Jff��pz��E�RC>z�                                    BxqM�  T          A�(��G��$���5p��
=C[�=�G��8���p���FQ�CA�                                    Bxq\&  �          A����=����HQ����CZ�{�=녿�=q�y��R{C;�\                                    Bxqj�  �          A�  �C�
�p��B�R�(�CZ��C�
��{�v{�L��C<�f                                    Bxqyr  
�          A�  �>�R��
�I���CZ��>�R�����z�R�R  C;�                                    Bxq�  T          A�  �:�\�ff�K33� {C[�)�:�\��33�}��U�C<�                                    Bxq��  �          A�(��=�$���@(��z�C]  �=�%��y��P�C@G�                                    Bxq�d  T          A�{�>ff�+��8���
=C^��>ff�L(��v�H�M��CB�q                                    Bxq�
  
(          A�33�?�
�"{�<z���\C\0��?�
�#�
�u��M\)C@�                                    Bxq°  
�          A�33�?\)�\)�?��G�C[��?\)��
�vff�N��C>�3                                    Bxq�V  �          A��R�=p���H�D  ���C[E�=p���Q��w�
�Q  C=L�                                    Bxq��  T          A����<���33�D���G�C[aH�<�ÿ�Q��x���Q�\C=O\                                    Bxq�  �          A�33�I��\�8���
=CYp��I�33�n{�E  C>Q�                                    Bxq�H  "          A���O
=��H�6{���CX��O
=�(��i�@G�C=�)                                    Bxq�  
�          A����O�
����/33���CY
=�O�
�/\)�g
=�=��C?�f                                    Bxq�  
�          A��P���33�.�H���CYL��P���8Q��h  �==qC@p�                                    Bxq):  �          A����Tz�� (��-G��G�CY  �Tz��?\)�g
=�:C@��                                    Bxq7�  "          A���V�\�!�*�\��CY��V�\�J�H�e���8�CAL�                                    BxqF�  �          A���Jff�!G��:�R�G�CZ���Jff�(Q��s
=�F�C?�q                                    BxqU,  �          A���L���33�:{���CYٚ�L���"�\�qp��D�\C?=q                                    Bxqc�  �          A�  �Q��!p��4���	��CY���Q��5�n{�@=qC@@                                     Bxqrx  �          A�33�J�H�!p��?\)�CZ���J�H� ���w33�HQ�C?.                                    Bxq�  �          A���I���D�����CZO\�I��R�z�R�J�C>�                                    Bxq��  
�          A��\�J{� ���EG���\CZ��J{��\�{�
�KG�C>L�                                    Bxq�j  "          A�Q��F�\��
�H���  CZ�
�F�\����~�\�N�C=�                                    Bxq�  "          A�z��Hz��33�H  ���CZp��Hz�����}G��L�C=�                                    Bxq��  �          A�Q��H  �\)�G�
�
=CZ�=�H  �	���}G��M(�C=Ǯ                                    Bxq�\  �          A�ff�G
=�p��J�R�ffCZW
�G
=��(��~�R�N�\C=                                      Bxq�  �          A��\�H(�� z��G\)�p�CZ���H(��  �}p��M
=C>.                                    Bxq�  �          A���IG��$  �E���
=C[+��IG��   �}���L{C?=q                                    Bxq�N  T          A����I��$(��EG����C[8R�I��!G��}p��L{C?W
                                    Bxq�  
�          A����Ip��#�
�D  �33C[��Ip��#33�|(��K=qC?p�                                    Bxq�  
�          A��R�LQ��"�H�A��\)CZ���LQ��$z��y��H�RC?ff                                    Bxq"@  �          A����H���"�R�EG��Q�C[  �H���{�|���K�HC?!H                                    Bxq0�  �          A�{�G33�$  �DQ��  C[z��G33�%��|z��L�\C?��                                    Bxq?�  �          A�  �F�R�$���C���\C[���F�R�)���|(��L�\C@
=                                    BxqN2  �          A���G��%�B{�33C[���G��0  �{\)�K�\C@s3                                    Bxq\�  �          A���H���&{�@z����C[�
�H���5��z{�J33C@�3                                    Bxqk~  T          A�z��J�\�&{�@  �  C[W
�J�\�6ff�y�I
=C@��                                    Bxqz$  T          A��\�K��'33�>�\��\C[ff�K��=p��x���H  CA)                                    Bxq��  "          A��\�LQ��'33�=���C[J=�LQ��@  �x(��G�CA8R                                    Bxq�p  �          A��R�M��&ff�>=q��C[��M��<(��xQ��F�C@�                                    Bxq�  T          A����MG��&�H�=��=qC[)�MG��AG��w��FG�CA:�                                    Bxq��  �          A�z��MG��%��>=q�Q�CZ���MG��8���w��F�\C@��                                    Bxq�b  
�          A�z��M��#��>�R���CZu��M��333�w\)�F33C@E                                    Bxq�  T          A�z��M���%��=��{CZ�M���:=q�w\)�F=qC@�                                    Bxq�  T          A�ff�N�H�(Q��9p��=qC[��N�H�N�R�t���C�
CB
=                                    Bxq�T  T          A�=q�M��'��:�R��C[)�M��J=q�u�D�
CA�                                    Bxq��  �          A�ff�O\)�&=q�:�R�ffCZ���O\)�E�t���C�HCAn                                    Bxq�  �          A�  �PQ��%��9��ffCZh��PQ��Fff�s33�B�CAaH                                    BxqF  �          A��P���'��5����\CZ���P���U�q��@��CB^�                                    Bxq)�  
�          A����O\)�)G��5��\)C[8R�O\)�]p��q���A�\CB�3                                    Bxq8�  �          A���P(��(���4z���HC[��P(��]p��p���@��CB�H                                    BxqG8  �          A����N�R�+
=�4  �p�C[���N�R�fff�q���A�CC�
                                    BxqU�  �          A���Nff�-p��2�\�
=C\
=�Nff�r�\�qp��A(�CD^�                                    Bxqd�  T          A��M���-��333��C\5��M���s33�r{�A�HCDxR                                    Bxqs*  �          A��
�N�H�+��4z���C[��N�H�h���r{�A��CC��                                    Bxq��  T          A��P���+33�2ff��C[aH�P���l(��p(��?�RCC�\                                    Bxq�v  �          A���S��*�R�0  ��CZ��S��p  �m�=  CC�)                                    Bxq�  
�          A��PQ��+
=�2�R��C[c��PQ��l���p(��?�
CC�
                                    Bxq��  "          A����MG��*�R�6=q�	=qC[�MG��dz��s
=�C{CC��                                    Bxq�h  
�          A�\)�K
=�)��8����C[��K
=�]p��t���E(�CCB�                                    Bxq�  
�          A���N{�+��4  �z�C[�=�N{�mp��qp��A��CD�                                    Bxqٴ  �          A����R=q�1���(����(�C\0��R=q��(��j�H�:��CFs3                                    Bxq�Z  �          A�\)�N�\�,  �2�R��C[�=�N�\�q��p���@CDQ�                                    Bxq�   "          A�p��N�\�+��3\)���C[���N�\�p  �p���@��CD33                                    Bxq�  �          A�\)�N�\�+�
�2�R��\C[�q�N�\�q��pz��@��CDQ�                                    BxqL  �          A�\)�M��(���6=q�	�C[\)�M��a��q��BG�CCQ�                                    Bxq"�  X          A����P(��,���0���C[���P(��y���o33�>��CD�3                                    Bxq1�  
�          A��
�Q��,���/\)�33C[z��Q��~�R�m��=G�CD޸                                    Bxq@>  T          A���Q���-��.=q�p�C[���Q����=q�mp��=  CEE                                    BxqN�  
�          A�G��Pz��0(��+�
� ��C\5��Pz������lz��<�\CF+�                                    Bxq]�  "          A�\)�P���0���+
=� {C\B��P�����H�l  �<
=CFff                                    Bxql0  �          A�p��P���1p��*ff���HC\W
�P������k��;��CF��                                    Bxqz�  X          A�33�P  �333�)���
=C\� �P  �����k\)�;�\CGJ=                                    Bxq�|  
�          A�33�P  �3��(����(�C\�{�P  ��33�k33�;Q�CG}q                                    Bxq�"  
�          A�\)�Q��3\)�(  ���HC\�H�Q�����jff�:\)CGn                                    Bxq��  T          A��H�P(��5p��%G���G�C]��P(�����i��9��CHE                                    Bxq�n  �          A����O��6�\�$�����RC]T{�O���z��iG��9��CH��                                    Bxq�  
�          A����Q�4z��$Q����C\���Q����g��8�CH!H                                    BxqҺ  �          A�Q��Pz��4(��#�
��  C\���Pz������g
=�8z�CH:�                                    Bxq�`  "          A����T(��-��$����{C[8R�T(�����d(��6\)CFc�                                    Bxq�  �          A�
=�]p��33�'33���
CW��]p��W
=�]�0�
CA�f                                    Bxq��  �          A����W��%��$  ���RCY�
�W������`(��3�CD��                                    BxqR  �          A����W��$z��$����Q�CYT{�W��|���`(��3�CDW
                                    Bxq�  �          A��R�[33� ���$  ����CXJ=�[33�r�\�]���1
=CCxR                                    Bxq*�  "          A����[���H�%G�����CW޸�[��i���]�1�CB�H                                    Bxq9D  T          A��H�\���{�%�����CW�H�\���g��]��0�CB�3                                    BxqG�  �          A����`Q��Q��%����CV.�`Q��R�\�[
=�.33CA33                                    BxqV�  �          A����g���H�)G���CR8R�g��\)�Up��)Q�C<��                                    Bxqe6  �          A����l  ���
�*ff���CP
�l  ��p��R{�%�C:�                                    Bxqs�  �          A�z��]G�����'����CV�)�]G��Q��\���0�CAW
                                    Bxq��  �          A��R�]������(Q�� {CV�
�]���QG��]G��0�CAL�                                    Bxq�(  T          A��H�^{�(��(��� G�CVn�^{�O\)�]G��0�CA�                                    Bxq��  T          A����_�
��\�+33�z�CU5��_�
�7
=�\���0  C?�=                                    Bxq�t  �          A����`�����,Q���CT�\�`���*�H�\(��/��C>�q                                    Bxq�  �          A�
=�^�R���0  �\)CT���^�R�$z��_\)�2ffC>s3                                    Bxq��  
�          A��H�ep��33�0  ��CQǮ�ep���33�Yp��,�HC;�=                                    Bxq�f  �          A��R�ep���{�2�\�	  CP�3�ep������Y���-=qC:Y�                                    Bxq�  �          A��\�d(����0z��=qCR��d(���
=�Z{�-C;��                                    Bxq��  T          A�=q�d������*�\�z�CR��d������W\)�+�\C=�=                                    BxqX  �          A�=q�d  �33�)�����CSh��d  �#�
�W��+C>.                                    Bxq�  �          A�=q�c�
��
�%���
=CTB��c�
�;��V�\�*C?�H                                    Bxq#�  T          A�{�ep��G��-G��p�CQc��ep����V{�+
=C;�)                                    Bxq2J  T          A����b�R�   �1G��	  CQn�b�R��p��X���.
=C:��                                    Bxq@�  "          A��H�b=q��p��0���	
=CQ@ �b=q�ٙ��W��-�C:�
                                    BxqO�  
Z          A�ff�c
=��ff�0���	p�CP}q�c
=��G��U��,��C:{                                    Bxq^<  �          A��\�c���  �2=q�
��CO���c���ff�U�,\)C933                                    Bxql�  �          A�{�dz�����1p��
�CO
�dzΐ33�S��*�HC8�)                                    Bxq{�  �          A��
�f�H��\�/�
�	p�CN��f�H��G��Pz��(
=C8                                      Bxq�.  "          A��
�f�H��ff�0���
z�CM���f�H�^�R�Pz��({C7u�                                    Bxq��  �          A���g����H�0���
�\CMG��g��G��O��'\)C7{                                    Bxq�z  T          A���d����{�1p���CM�H�d�ÿ^�R�P���)=qC7xR                                    Bxq�   T          A�
=�d  ��p��0  �
Q�CN�3�d  ��\)�QG��)�
C8}q                                    Bxq��  �          A��H�f�\��{�)G��p�COQ��f�\�����Mp��&�C:33                                    Bxq�l  �          A�G��h����33�%�  CO�
�h�Ϳ����K��#C;�                                    Bxq�  T          A����h(���z��#���p�CO��h(���
=�I��#  C;�\                                    Bxq�  �          A�33�k
=� �������ffCP���k
=�   �F�H�G�C=��                                    Bxq�^  �          A�z��f�H���H�"=q���CP���f�H�	���J=q�#�C<}q                                    Bxq  �          A�Q��h(���z��"ff��  CO��h(���p��H���"\)C;�                                    Bxq�  �          A�  �g33����"ff���\CO���g33�   �H���"�
C;��                                    Bxq+P  �          A��
�d�������$  � ��CP�=�d���z��K33�%=qC<=q                                    Bxq9�  �          A���b�\���)�{CO�H�b�\��z��N{�(p�C:��                                    BxqH�  �          A���f=q��{�-G��	Q�CL��f=q�Y���K33�%��C7c�                                    BxqWB  T          A�{�`(���(��/\)�
CO��`(���z��R{�,
=C9�                                     Bxqe�  �          A�=q�a����5����CL��a�   �Q��+��C6
=                                    Bxqt�  �          A�ff�^�R�����5�=qCN���^�R�n{�U��.�RC7�
                                    Bxq�4  �          A����Y�޸R�7����CO��Y�Q��Vff�2  C7p�                                    Bxq��  T          A�z��X(�����0����RCQ���X(���33�Up��1�C:�                                    Bxq��  
�          A�Q��^�R�أ��0����CM��^�R�\(��N�H�+Q�C7��                                    Bxq�&  �          A����N�H�
�R�0z��33CU�
�N�H�%��\���9�C?B�                                    Bxq��  �          A���N�H���*�H���CW��N�H�O\)�[�
�7z�CB�                                    Bxq�r  
�          A�ff�W�
��'�
��
CS���W�
�$z��R�R�/33C>�                                    Bxq�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxq�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxq�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxq
  �          A��
�G��&{����p�C[� �G���ff�X  �5ffCH��                                    Bxq�  
(          A���H  �"�R� Q�����C[&f�H  ���Y��6�CG��                                    Bxq$V  T          A��
�P���#
=�����ffCY�q�P����G��N�H�+�RCH+�                                    Bxq2�  �          A�33�H(��(  ����\)C\��H(���Q��Z�H�6�CHٚ                                    BxqA�  �          A��R�F=q�$(��$����\C[�f�F=q����]�:
=CG��                                    BxqPH  T          A��R�H���!G��$  ��HCZ�q�H�������[��7�
CF�\                                    Bxq^�  T          A�{�T(��\)��
���CX33�T(�����M��*�
CE�q                                    Bxqm�  �          A�=q�E��%��"{� �C\��E�����[�
�8��CH�                                     Bxq|:  T          A�ff�B�\�((��#����C\�{�B�\��ff�^{�;33CI!H                                    Bxq��  T          A����Rff�G���
��z�CX
�Rff�\)�T(��/�CD޸                                    Bxq��  "          A����R=q��������
=CX���R=q�����R�H�.�\CE��                                    Bxq�,  T          A��R�Zff�G���
��CU�H�Zff�j�H�L���(�CC�                                    Bxq��  �          A����Z{�p��  ��(�CU���Z{�k��L���(�HCC)                                    Bxq�x  �          A��\�P���{�(����\CY&f�P����z��R�H�/  CF��                                    Bxq�  �          A�ff�R�R��\�����CX���R�R��Q��P(��,G�CF�                                    Bxq��  �          A�(��W\)�ff���CW���W\)��z��K\)�'��CF�                                    Bxq�j  T          A����]�� (����У�CW���]�����?�
�  CHh�                                    Bxq   �          A�=q�[\)�!���R�ң�CXO\�[\)���R�@����CH�=                                    Bxq�  
�          A����T������(���  CXh��T����33�J�H�(
=CG
                                    Bxq\  
�          A���S�
���{��33CX���S�
����L���)��CG                                      Bxq,  T          A�\)�Qp����\)���CY��Qp����\�N{�+�\CGG�                                    Bxq:�  T          A����K�
�!G������p�CZ^��K�
����P���/Q�CHn                                    BxqIN  T          A�ff�D  �)G��z����C\���D  ��ff�T(��3�\CK�                                    BxqW�  T          A�ff�@���+�
�{����C]���@������V�R�6(�CK�                                    Bxqf�  "          A�=q�HQ��'\)�z����C[���HQ���
=�O��.�CJ�f                                    Bxqu@  �          A����b�\�"�H���H��
=CW���b�\��ff�.{�=qCJ��                                    Bxq��  �          A����i��!G��У����CV���i����
�%����CJ��                                    Bxq��  "          A����i�#���p���Q�CW  �i����$�����CKaH                                    Bxq�2  �          A�{�g��$����ff��G�CWn�g���Q��)��\)CKaH                                    Bxq��  �          A�p��eG��ff��z����CV���eG���33�0(��\)CIT{                                    Bxq�~  T          A�G��d(���H��ff����CV�
�d(�����1��ffCIu�                                    Bxq�$  �          A����a��"�\�����
=CW�{�a����H�2=q��CJ�=                                    Bxq��  T          A��H�]��&�\�������HCX�H�]��\�3��=qCK�f                                    Bxq�p  �          A��R�`z��"�\�����G�CW�f�`z���33�2{���CJ�f                                    Bxq�  �          A����`���   ��z����CWp��`�����
�4(����CI��                                    Bxq�  �          A��\�`z������G���CV�3�`z���z��4����
CI                                      Bxqb  "          A�{�`�����	����ffCT�)�`�������:�R��CE�                                    Bxq%  �          A�\)�TQ��.ff��G����C[aH�TQ���33�5��z�CNp�                                    Bxq3�  �          A�
=�W��%��z�����CY�\�W���
=�6ff��CK��                                    BxqBT  "          A�z��Rff�3�
��  ��ffC\��Rff����/��  CP�{                                    BxqP�  T          A�  �Qp��/���p���z�C[���Qp���33�0(����CO��                                    Bxq_�  �          A�G��X(�������\)CTL��X(��n{�>�R�!�CCc�                                    BxqnF  �          A�{�M���ff�!�  CS���M��0���H���.��C@
                                    Bxq|�  �          A���Jff��G��*=q�Q�CRǮ�Jff�
�H�M���4�C=�R                                    Bxq��  �          A�p��J�\�\)�\)�{CV��J�\�h���HQ��.(�CD�                                    Bxq�8  "          A�p��H����H�!���RCU�H�H���N�R�K��1��CBp�                                    Bxq��  �          A�{�M���
�
�H��{CXff�M������>�H�#��CHs3                                    Bxq��  �          A����:=q� z��2�\�{CV�
�:=q��H�Xz��BC?�                                     Bxq�*  �          A����@����H� z��(�CX�=�@���n�R�Nff�6��CE0�                                    Bxq��  T          A��
�G
=��H�$(��	G�CV��G
=�L(��N=q�4p�CBaH                                    Bxq�v  �          A��
�Hz�����=q��ffCX�\�Hz���33�G��,��CG&f                                    Bxq�  T          A����H���(������{CW���H�������H  �.  CE��                                    Bxq �  �          A���Mp��
=������CUO\�Mp��[��G\)�,��CB�                                    Bxqh  �          A��
�H(��$(���R��  C[\)�H(���p��@(��$��CL^�                                    Bxq  �          A�Q��D���"�\�ff��Q�C[���D�����H�?
=�%��CLs3                                    Bxq,�  �          A����G
=��R��\���CY!H�G
=����D���+Q�CHT{                                    Bxq;Z  T          A�
=�=p���R�\)��z�C[=q�=p���=q�G\)�0��CJ&f                                    BxqJ   �          A���F�R�ff�  ��  CY!H�F�R���B=q�)��CH��                                    BxqX�  
�          A���O
=� (��������CY���O
=���R�-p��{CL�q                                    BxqgL  �          A�
=�X  �"{��z����CX�f�X  ���=q�p�CNO\                                    Bxqu�  �          A�{�M�0�������p�C\�H�M���������HCSE                                    Bxq��  �          A����;\)�-��H��
=C^�
�;\)���
�@(��(
=CP�\                                    Bxq�>  
�          A���D���8�������=qC_0��D�����2=q��RCS��                                    Bxq��  T          A��R�J�R�5p���33��C]�
�J�R��=q�!p��p�CS�                                    Bxq��  T          A��H�PQ��<(��U�,��C^{�PQ��  �����z�CWff                                    Bxq�0  �          A��
�Q�4z��n{�BffC\�R�Q�
�R�{�ڸRCUz�                                    Bxq��  �          A����C\)�=����R�j�HC`33�C\)�ff�����
=CX�                                    Bxq�|  �          A�
=�D���AG��j=q�?�C`��D���
=�=q��p�CY��                                    Bxq�"  �          A�33�Ap��C\)�xQ��K�
CaG��Ap��\)�
=q����CZ
=                                    Bxq��  T          A�p��;��<����z���  Ca0��;��(��$�����CW&f                                    Bxqn  �          A�p��?��8Q���\)��Q�C_�f�?���
=�#�
���CU��                                    Bxq  "          A��\�I���3����
���RC]���I����R�����CT�R                                    Bxq%�  "          A��H�Hz��9���
=�]�C^ٚ�Hz�����
�R��(�CW)                                    Bxq4`  �          A��\�K33�9����p��f�RC^ff�K33�\)�p�����CVs3                                    BxqC  �          A�33�K��9p������~�\C^Y��K��(��Q����CU��                                    BxqQ�  "          A�33�A�<�����H��z�C`G��A�G��#\)�33CV�                                    Bxq`R  T          A�p��HQ��9p���G���{C^���HQ��(��p���CUk�                                    Bxqn�  �          A�Q��IG��6�\��  ���C^8R�IG����  �(�CU.                                    Bxq}�  �          A���Mp��2�R���H�}C]��Mp��
=�(�����CT��                                    Bxq�D  �          A�\)�W��/33�QG��*�RC[
�W��
{���ȣ�CT��                                    Bxq��  �          A�p��X���)��Q��QCZ��X��� Q�� (��׮CR�)                                    Bxq��  �          A�
=�Y��$����G��nffCY5��Y���G�����\CQ
=                                    Bxq�6  �          A�ff�S33�&ff��G���p�CZ@ �S33�����\CQ^�                                    Bxq��  
�          A��R�Nff�%p���
=��(�CZ���Nff�陚��
��  CQ�                                     BxqՂ  
�          A�33�N�H� ����=q���CY���N�H���
��
����CP��                                    Bxq�(  "          A�{�J�R�   ��=q���
CZG��J�R�޸R��H���CP�=                                    Bxq��  �          A�(��G��(��u��]G�CX�{�G�����������CP�                                    Bxqt  �          A����?\)��H������=qCZ���?\)���H��
�p�CP�)                                    Bxq  �          A����5������
��p�C\���5��������CQ��                                    Bxq�  �          A�G��8����H��
=��  C[���8����\)�p���CPW
                                    Bxq-f  �          A��R�7\)���������HC]
=�7\)��{�����HCRB�                                    Bxq<  
�          A�p��8(��#�����{C]���8(�����G��
��CS�=                                    BxqJ�  �          A����DQ��
=������p�C[  �DQ�����	��z�CQ�                                    BxqYX  �          A�=q�D  �33��G���{CX��D  ������ffCN�                                    Bxqg�  T          A����733��R�أ�����C\���733�˅�#
=��\CQ�                                    Bxqv�  T          A�G��"�R�#
=��\��=qCa\�"�R���R�<���4  CRaH                                    Bxq�J  �          A����-���\)��C^@ �-��  �7��,�CO�f                                    Bxq��  T          A�G��>{�!G���p��ŮC\Q��>{���
�)����RCP5�                                    Bxq��  �          A�
=�C�
�!G���\)���C[u��C�
�����\)�=qCP��                                    Bxq�<  �          A����EG��$  ��z���p�C[�q�EG��ᙚ��
�  CQ                                    Bxq��  �          A�ff�F�\�&�H������=qC\
=�F�\��
=��
����CS
=                                    BxqΈ  �          A��R�E��(z������{C\ff�E��������� ��CSW
                                    Bxq�.  "          A�=q�G�
�(Q�����\)C\��G�
�����{�z�CR�f                                    Bxq��  T          A��\�Ap��"ff��  ��
=C\�Ap���  ��
�G�CQ+�                                    Bxq�z  T          A�(��?\)� (��ٙ���Q�C[��?\)��Q��#33�Q�CP��                                    Bxq 	   "          A����>ff�+\)�������
C]���>ff��Q��ff�	�RCTB�                                    Bxq �  T          A�
=�B�\�)������p�C])�B�\��\)��
�p�CS�
                                    Bxq &l  
�          A�G��B=q�$  �������RC\.�B=q��ff���CR��                                    Bxq 5  "          A~ff�9���   ���\�|z�C\�9���������ffCT�)                                    Bxq C�  �          A�ff�<���#�
��  ��ffC\���<�������ff���CT�                                    Bxq R^  "          A�
=�9p���H������RC\�
�9p��������� ffCS��                                    Bxq a  T          A��=G�� ���l���V�\C\T{�=G�������\����CUc�                                    Bxq o�  �          A�=q�8  �����{��=qC[�
�8  ��=q��H���CP�=                                    Bxq ~P  "          A���.=q��
��{��G�C]�=�.=q��Q��"�H��CQ�f                                    Bxq ��  T          Az�H�33�����
=���
C_5��33����,  �-  CQk�                                    Bxq ��  "          AyG���R��
��� �Ca����R�����5�:CS                                    Bxq �B  �          Av�R��
��R��\���C`���
��33�&�\�)��CR��                                    Bxq ��  (          Au�������p���z�Cb�����p��/��6p�CS��                                    Bxq ǎ            At�������\�����  Cb�������ff�,���3=qCU.                                    Bxq �4  
�          Aup��
=��������{Cb��
=��33�$(��(�CU��                                    Bxq ��  �          Au���"{����(���z�C`=q�"{���H���{CV
=                                    Bxq �  �          AuG��%��#������(�C`���%����H���G�CX5�                                    Bxq!&  
(          AuG��#\)�%�������\)Cas3�#\)��  �=q��HCY0�                                    Bxq!�  T          At����R�*{�����ffCb�q��R� (��33�(�CZ��                                    Bxq!r  �          As��\)�-��qG��eG�Cc��\)�	��z���
=C\�
                                    Bxq!.  �          At���"�H�,z��mp��`��Cb���"�H�����G���z�C\                                      Bxq!<�  �          AuG��  �1������v{Cd�R�  �������\)C]��                                    Bxq!Kd  �          Au� ���2�\�E��9p�Cc��� ���ff��33��z�C^G�                                    Bxq!Z
  �          Au�� ���333�.�R�$z�Cd�� ���G�������ffC^��                                    Bxq!h�  �          At�����5G��@���5�Ce{���p���\����C_��                                    Bxq!wV  "          Ao\)�$z��+���\��G�Cb5��$z��(���ff��
=C^                                      Bxq!��  
�          Amp��%��(z´p����Ca�{�%��
=��33��\)C]��                                    Bxq!��  "          Al(��#�
�(zῧ����\Ca���#�
�(������\C^�                                    Bxq!�H  
�          Ak
=�!��)��������G�Cbs3�!���������\)C^�q                                    Bxq!��  T          Aip��$  �%G�������Ca8R�$  ��R�����C]�{                                    Bxq!��  
�          Ak��#\)�'33��=q���Ca���#\)�z���p����C^G�                                    Bxq!�:  �          Ar�\�2=q�\)����	C]���2=q�=q���\��p�CX�R                                    Bxq!��  �          At���6�\����(Q���HC\���6�\�����33��z�CWY�                                    Bxq!�  �          Ax  �7
=�Q���G���{C[
=�7
=��=q�������CR��                                    Bxq!�,  T          Ax���4����R�y���h��C]G��4����  �������CVs3                                    Bxq"	�  T          Ay��3
=��
���R���
C]��3
=��(���Q���z�CUh�                                    Bxq"x  T          Az�\�+�
�,���c�
�RffCa#��+�
�33�����
=C[�                                    Bxq"'  T          A}p��*�H�'\)�����z�C`c��*�H������
���CX
=                                    Bxq"5�  T          A{��/
=��\�������C^(��/
=��
=�
=q���CUn                                    Bxq"Dj  �          A|Q��7
=�+33�Ǯ��ffC_
�7
=����������C[Q�                                    Bxq"S  T          A}���;��)p���33��C^��;���R��\)��z�CZǮ                                    Bxq"a�  �          Ax���6{�(��>�\)?��\C^ٚ�6{��R�hQ��X(�C]�                                    Bxq"p\  �          Aq�,Q��)G����
���
C`z��,Q��G��z=q�p  C^aH                                    Bxq"  "          Aqp��'
=�,  ��p��ҏ\Ca�
�'
=����������C]�f                                    Bxq"��  �          Ah(��p��,(��������CdG��p��z���p���\)C`��                                    Bxq"�N  �          A]��{�&=q�����ffCg���{��R���
��Ca(�                                    Bxq"��  �          A\����\)�{��  ����Cg���\)������
���C^ff                                    Bxq"��  
�          A_���R�$����p����Cf�3��R��
=������C_n                                    Bxq"�@  �          A_�
�  �)p��vff��{Ch��  �\)��{� �RCa��                                    Bxq"��  "          A_�
�z�����L���R�RCa�{�z���\)��G���{C[�                                    Bxq"�  �          A_33�z��\)�<���D  Ca��z���\)�ȣ���{C[�                                    Bxq"�2  T          A_33�z���H�Fff�MCb���z��=q�Ϯ���
C]E                                    Bxq#�  �          A`z�����p��;��@��Ca���������������C\W
                                    Bxq#~  
�          Ad(���� ���Fff�IG�CbE���(��У��ٮC\�f                                    Bxq# $  �          Ac��(  ��H��(��޸RC]��(  ��H��  ���RCY�                                    Bxq#.�  �          Ab�\�.{�  ��z���{C[�
�.{��\)�����CXE                                    Bxq#=p  "          Aa��,z���
�Ǯ�ʏ\C[�\�,z����\��p�����CW�q                                    Bxq#L  �          Abff�.�\�녿�\)��=qC[��.�\��ff������CW5�                                    Bxq#Z�  "          Aa�-��=q��Q����HC[E�-���G���Q���G�CW��                                    Bxq#ib  "          Aa�-������=q��p�C[+��-����R��(���G�CWY�                                    Bxq#x  �          Aa���.�H����G����CZ���.�H������  ���CV�{                                    Bxq#��  �          Aa���-���Ϳ������CZ�q�-���=q��33����CVٚ                                    Bxq#�T  T          A_�
�+��(���\��C[=q�+���
=��Q����
CVٚ                                    Bxq#��  
�          A_�
�,���33��\)��p�CZٚ�,����\)��=q��\)CV��                                    Bxq#��  "          AaG��.�R����(����CZ���.�R����{���CV�f                                    Bxq#�F  "          A_33�)p�����
=q�ffC[� �)p���\)��(����\CW:�                                    Bxq#��  �          A\���&�R��
�G���C[���&�R��(���ff��33CWL�                                    Bxq#ޒ  "          AZ�R� (��\)�"�\�+�C]�
� (���  ��������CX�
                                    Bxq#�8  
�          AYG��   �����!�C]���   ��
=���\��
=CX�                                    Bxq#��  �          AX(��p��
=�
=�!�C^B��p������H��z�CY}q                                    Bxq$
�  
�          AX����\�\)�
=�   C^!H��\��\���H��p�CYc�                                    Bxq$*  "          A\z��#
=�z��	����C]�\�#
=���R��p���(�CY#�                                    Bxq$'�  �          A]��$z���Ϳ�33����C]\)�$z���=q����33CYE                                    Bxq$6v  "          AZ�H�!��p����
��C^��!�������\���CZ&f                                    Bxq$E  �          AX  � (��=q��(����C]��� (������{����CY��                                    Bxq$S�  "          AX���   �������\)C]��   �����������CZ(�                                    Bxq$bh  �          AX�����녿���C^�R������33��CZ�                                    Bxq$q  T          AXz�������  ��C_� ��� �����\��G�C[��                                    Bxq$�  "          AW33���ff�ٙ����C`�����
���H���\C]                                      Bxq$�Z  
�          AV=q�Q���ÿ�33����C`^��Q��(��������
C\�3                                    Bxq$�   �          AU���p��33���
��  C_�\�p��33�����z�C\��                                    Bxq$��  �          AT(��=q��������
=C^�q�=q�����=q��=qC[�f                                    Bxq$�L  �          AS�����Q쿹���ə�C_^������\)��ff��\)C[��                                    Bxq$��  "          AR=q���(����H��z�Ca!H���33��G����C]��                                    Bxq$ט  �          AP���G��Q��G���(�Ca�{�G��33���\��{C^{                                    Bxq$�>  �          AQ�G��(�������Ca�\�G�� ��������C]�)                                    Bxq$��  
�          AT����
��H�`���w33Ce�{��
������H���
C_��                                    Bxq%�  T          AX������=q����  Cb\����33��ff��C^��                                    Bxq%0  �          AXQ������H�p����  C`}q��������=q���RC]��                                    Bxq% �  �          AK������������
=Cf������33�������C]�3                                    Bxq%/|  �          AS33� ���{��(���=qCeT{� ���������
�C^�                                    Bxq%>"  "          AY����H�  �qG���
=CbǮ��H����  ��z�C\�3                                    Bxq%L�  "          AZ{����33�j=q�y�Cc� �������ָR��Q�C]�f                                    Bxq%[n  
�          AXQ��\)�
=�7��Dz�C`+��\)��ff��G���
=C[�                                    Bxq%j  �          AV=q�(������Q�C`��(���ff�������CY�f                                    Bxq%x�  �          AXQ���G����  ��=qCiǮ��G���
=�=q�{Ca��                                    Bxq%�`  �          AU������@p�A�C_������׿\(��r�\C`h�                                    Bxq%�  �          AV=q�z�����i���~ffC\T{�z��Ӆ�ƸR���CV�                                    Bxq%��  �          AV=q�   �	���R�C\�   ��z���
=��Q�CXs3                                    Bxq%�R  �          AUp������\�_\)�u��C]W
����أ����H�ۙ�CWT{                                    Bxq%��  �          ANff�	����\��\)��{Cb�	�����
��
=��  C^z�                                    Bxq%О  �          AL���ff�	����ff��C^n�ff���R�QG��m��C\B�                                    Bxq%�D  �          AK
=��\�{�Ǯ��G�C\���\���C33�_\)CZ�                                    Bxq%��  T          AE��z����1G��PQ�C^��z���Q�������{CY�
                                    Bxq%��  �          A)p���z����
���R�1�CZ����z�����~�R����CV0�                                    Bxq&6  T          A=q�(����
�C<.�(�@6ff��H.B��                                    Bxq&�  �          A!��G��k�����CJ!H�G�@�
��
G�C��                                    Bxq&(�  �          AG����H�G�����e�RCO�)���H>�Q���\)�u��C/@                                     Bxq&7(  T          A��AG���z��	��CM
�AG�?�(��	��Q�C                                      Bxq&E�  �          A(��'
=��ff�Q�=qCN� �'
=?У��\)#�C
=                                    Bxq&Tt  �          A���!G���ff�	p��RCJ�f�!G�?�ff����C�                                     Bxq&c  �          Aff�S33��{�p�CJk��S33?�����\C�H                                    Bxq&q�  �          A��C33��{���CH�C33?�z��Q�z�Cu�                                    Bxq&�f  �          A(��;������\)�CH�)�;�?�\�	����C�                                     Bxq&�  �          A
�\�0�׿L����\.CD8R�0��?�\)��\)��C�
                                    Bxq&��  T          Ap���\��ff�(�Q�CTu���\?�G�����C\)                                    Bxq&�X  �          A(����ÿ�33��
.C]�{����?�33��
.C
.                                    Bxq&��  �          A
=�%��z��
�\k�CP�=�%?����
ff.C�=                                    Bxq&ɤ  �          A��������33(�C\Q����?(����H� C%�                                    Bxq&�J  �          A
=�I����G��33Q�CE�R�I��?�=q�	��HC��                                    Bxq&��  �          A���]p��������
=C5�R�]p�@-p��{�y33C�                                    Bxq&��  �          A Q��z�fff��\#�CI(��z�@�R�
=ffC!H                                    Bxq'<  �          A!�Z�H����=qC6+��Z�H@8���z��~\)C��                                    Bxq'�  �          A$z��b�\�W
=�=q�\C7Y��b�\@7
=�33�~�C�                                    Bxq'!�  �          A)����H?����R�}�RC'����H@\)�	���Z�HC�                                    Bxq'0.  T          A#��Vff?0���=q�C(h��Vff@l����H�s�C&f                                    Bxq'>�  "          A$  �Mp������R33C8���Mp�@333�(���C
�{                                    Bxq'Mz  �          A%p��H�ÿ�(��ff��CT!H�H��?��Q�(�C�\                                    Bxq'\   �          A.�\�W
=��\)�"�R�CM� �W
=?�z��"�\z�C��                                    Bxq'j�  T          A.�R�Y���^�R�%�u�CBh��Y��@=q�!�
=C�)                                    Bxq'yl  
�          A2�H�n�R������Yp�Cj�{�n�R�
=q�"�\CR#�                                    Bxq'�            A;33�������\)�?�Cjٚ�����XQ��$z��t�HCX�q                                    Bxq'��  T          AFff�����޸R����;�Ckk������r�\�,(��q��CZG�                                    Bxq'�^  
�          AHQ���\)�������J��Cm���\)�Tz��3��qCZ�                                    Bxq'�  �          AG�����
=����IffCn����Z=q�2�Rk�C[:�                                    Bxq'ª  T          ADQ��8���ff�33�.Q�C{��8����33�*=q�q  Cq�f                                    Bxq'�P  
�          AA��R�
=�33�,��C33��R��\)�&�R�q�\Cw�)                                    Bxq'��  T          A9G���
�����\�:�RC~���
���%��
=Cv0�                                    Bxq'�  �          A2�\�\(���G���R�q�CgO\�\(�����(Q�.CE�)                                    Bxq'�B  �          A3
=�(���ff��w  Cr�\�(���
=�-G��RCR\)                                    Bxq(�  �          A1������z��{�g�Cv�f����R�)p��C_�)                                    Bxq(�  "          A*�R�.�R���R���u�Cl�q�.�R��z��!�#�CK�                                    Bxq()4  "          A �ÿ������\�Q��v33Cu�q������
=��H�CX8R                                    Bxq(7�  "          A(��ٙ��k���L�Cu.�ٙ��p���p��)CP�H                                    Bxq(F�  "          A=q�N�R�+����x�C[���N�R������C60�                                    Bxq(U&  
�          A��ff�����
�Q�CM+���ff�#�
���b33C4xR                                    Bxq(c�  �          A
=�H���.�R�����u�HC]{�H�þ�=q����C8�                                    Bxq(rr  "          A
�\�HQ���R����x��CZT{�HQ�u���R  C5
                                    Bxq(�  �          A  �n{�ٙ��	G���Cqff�n{?}p��
=¢G�C�                                    Bxq(��  �          A  ���
���
�HaHC��q���
?fff��¨=qB�\                                    Bxq(�d  �          A����p��*�H��p��P�\CQ� ��p���ff�����f�\C9�                                     Bxq(�
  
�          A���j�H�c�
�Ϯ�N��C`��j�H��
=��Q��u��CIO\                                    Bxq(��  �          Az�������@w
=A���C`0������z�?���AECdW
                                    Bxq(�V  �          A	G��q���\)@��B5��Cgh��q����
@�ffA�  CoT{                                    Bxq(��  "          @޸R���k�@��\BI�Cl�)�����@~{BG�Ct�)                                    Bxq(�  
�          @��
�AG���G�@���B:ffCj�
�AG���Q�@�Q�A�
=CrO\                                    Bxq(�H  M          A  �w����\@�(�B%�\Ch���w���G�@vffA���CoY�                                    Bxq)�  �          A33�~{����@�\)B�
ChT{�~{����@Mp�A��HCn+�                                    Bxq)�  "          A��  ���@�  B=
=CdO\��  ��\)@��
B\)CmJ=                                    Bxq)":  �          A=q�\����  @�(�BN{Ch�=�\����p�@�
=B�Cq                                    Bxq)0�  
�          A
=���
��\@�G�AƸRCj+����
��
=?�  A'�
CmL�                                    Bxq)?�  
�          A\)��(��߮@��RA�RChs3��(���@�HA`  ClY�                                    Bxq)N,  T          AQ������33@��
B=qChE�����\@@��A��Cm�                                    Bxq)\�  T          A=q���R�Å@��A��Ce�{���R����@5�A��\Cjk�                                    Bxq)kx  �          A=q��\)��{@�  A�33Ce�\��\)���@-p�A�=qCju�                                    Bxq)z  T          A�
�����33@�
=B�Ci�q�����@G
=A��Cn�                                    Bxq)��  �          A(��\)���
@��HBffCm�f�\)���@mp�A�\)Cr�f                                    Bxq)�j  
�          A�\���\����@���B�Ck�q���\���@`  A��Cq#�                                    Bxq)�  
�          A(��^�R��=q@�(�BffCr#��^�R� (�@l(�A�p�Cv��                                    Bxq)��  �          A�������
=@y��A��\Cj���������?��AffCmc�                                    Bxq)�\  
�          A$(������Q�@W�A���Ci0������\?u@�33Ck^�                                    Bxq)�  "          A"�H��
=��(�@b�\A��Ci&f��
=�G�?�z�@љ�Ck�=                                    Bxq)�  �          A33���\����@.�RA~ffCjL����\��>�{?�(�Ck�
                                    Bxq)�N  �          A$Q���  ��@A�A�p�Ck޸��  �
ff?
=q@@��Cm��                                    Bxq)��  T          A#������ ��@P  A�z�Cl������
=q?G�@�33Cm�q                                    Bxq*�  �          A#\)���\��\)@^{A���Cl@ ���\�
=q?�  @���CnT{                                    Bxq*@  T          A&=q��(�����@~{A�Cj{��(��	p�?�ffA	�Cl�                                     Bxq*)�  "          A-����(�� ��@�(�A�  Ck���(����@
=A3
=Cn�                                    Bxq*8�  �          A/�
��
=��@�{A�ffCj�{��
=�{@	��A4��Cm�                                    Bxq*G2  T          A.�\������@�=qA�{Ck�)������@�\AB=qCn�=                                    Bxq*U�  T          A,�����H���@�z�AˮCk����H��
@	��A7�Cn!H                                    Bxq*d~  �          A+����H��\)@�ffA�=qCk  ���H��R?�(�A)G�Cm�                                    Bxq*s$  T          A-����R�@��A�{Cl\���R�p�@A2�\Co�                                    Bxq*��  T          A/33���\��
@�(�A�Ck�����\��@�A.�RCn�
                                    Bxq*�p  �          A2�\��p���@��A��HCk�H��p���@	��A1G�Cn�{                                    Bxq*�  T          A1���z��z�@��\A�Ck� ��z���@��A;�
Cn�\                                    Bxq*��  T          A0�������@��\A���Ck������(�@�A>�\Cn�                                    Bxq*�b  T          A0����=q�33@��A���Ck�{��=q�(�@Q�AF�\Cn�q                                    Bxq*�  �          A/33�����@�z�A�Cm(�����  @'�A]G�Cpp�                                    Bxq*ٮ  �          A+�����=q@�{A�Cm�=�����@?\)A���Cqh�                                    Bxq*�T  �          A:�R���R�
ff@��\A���Cm�R���R��@9��Af�HCq�                                    Bxq*��  �          A:=q��
=�Q�@��A�G�Cl���
=��@7
=Ac�Co��                                    Bxq+�  �          A9����H�Q�@�=qA�G�Cl�R���H��
@<(�Aj�\Cp&f                                    Bxq+F  �          A;�
�����@�{A��Cn
����!p�?�z�AffCpaH                                    Bxq+"�  �          A?����H�  @��A���Cl�)���H�"ff?�\A(�Co�                                    Bxq+1�  �          A8�������@��A�z�ClO\������?�(�@���Cn��                                    Bxq+@8  "          A5������@~�RA�Q�Cl������?��@�Q�Cn�f                                    Bxq+N�  �          A7�
�����=q@�  A��Cn)�����@z�A%Cp�R                                    Bxq+]�  T          AAp��������@���A�\)Cm)�����$z�@ ��A�Co�f                                    Bxq+l*  �          A>=q����H@�
=AŮCn=q���#33@p�A+\)Cp�H                                    Bxq+z�  �          A3\)��  �  @J�HA�p�Cjs3��  �z�?(�@FffCl�                                    Bxq+�v  �          A.�R�У���\?�(�A�RCh=q�У���ÿ0���k�Ch�R                                    Bxq+�  T          A5����{��
@��A�{Ck�{��{���?\@�ffCn)                                    Bxq+��  �          A;
=�У����@Tz�A�{Cj^��У���\?(��@P  Ck��                                    Bxq+�h  �          A1���Q���\?��H@�  Ciٚ��Q��=q����أ�Ci��                                    Bxq+�  T          A.�H��  ��\?h��@���Cj����  ��ÿ˅�Cj��                                    Bxq+Ҵ  T          A0z�����?�33@��RCk�����zῳ33���Ck��                                    Bxq+�Z  T          A1G��Å�{?�
=@�p�Cl:��Å��R�����ClW
                                    Bxq+�   T          A2{�ƸR�33?Q�@�  Ck���ƸR���޸R�  Ck��                                    Bxq+��  �          A2ff��z��\)?��@�p�ClJ=��z��\)�����z�ClL�                                    Bxq,L  "          A4z����
��?��\@�\)CkY����
�\)������CkW
                                    Bxq,�  �          A4  �ʏ\�\)?��\@ϮCk���ʏ\�33��ff���Ck�                                     Bxq,*�  �          A5G���Q���R?���@ۅCj����Q��
=���H��z�Cj��                                    Bxq,9>  �          A6ff���
�=q?��@�  Cj����
�����\���CjO\                                    Bxq,G�  �          A5���Q��\)?�Q�@��Cj��Q��  �������RCj�H                                    Bxq,V�  T          A3�
��
=�z�?�p�A�CjaH��
=��\�J=q����Cj                                    Bxq,e0  "          A1����G�?޸RA��Ci������:�H�q�Cj\)                                    Bxq,s�  T          A1p���ff�G�?�=qA(�Ci�)��ff��
�#�
�S33Cj^�                                    Bxq,�|  �          A0����  ��?�Q�A!CiG���  ��R�   �&ffCi�f                                    Bxq,�"  �          A1���أ��	G�?��RA%G�Cg���أ���;���
=Chh�                                    Bxq,��  �          A:ff����z�?�\)@�  Cg+�����  ��\)��Q�Cg\                                    Bxq,�n  "          A6�\��Q���
?�Q�@�  Cj�{��Q��z῎{���Cj��                                    Bxq,�  �          A;
=��G��\)?�p�@��Ch�)��G��(�������RChǮ                                    Bxq,˺  
�          A<����{��H?�=q@θRCi����{�
=���\��p�Ci��                                    Bxq,�`  �          A:ff��33�?�\)@���Ci�\��33����Q����HCi��                                    Bxq,�  �          A;
=���
��H?��@�p�Ch5����
�
=���H��{Ch@                                     Bxq,��  
�          A7
=��
=�Q�?��@�
=Cf���
=��Ϳ������HCf��                                    Bxq-R  S          A<���߮�=q�B�\�n{CiT{�߮�  �,���Up�Ch&f                                    Bxq-�  "          AF�R�����"�\���
�Q�Ck�����������{���Ciu�                                    Bxq-#�  
�          AEp���ff�!G���  �߮Ckn��ff����z���
=CiG�                                    Bxq-2D  T          AE���p��"ff��
=����Ck�R��p��ff���H���\Ci�f                                    Bxq-@�  T          AE���ۅ�#
=���H���Cl��ۅ�  �xQ�����Cj#�                                    Bxq-O�  "          AD(���p�� �Ϳ����ffCkp���p���\�l�����\Ci�f                                    Bxq-^6  �          AE�ᙚ�!�����ƸRCj�q�ᙚ����~{��Ch�R                                    Bxq-l�  T          AF�\��  �!녿�����
=CkW
��  �����H��(�CiB�                                    Bxq-{�  �          AIG�� (��
=��R�#33Ce��� (��(��������Cb�q                                    Bxq-�(  "          AJ=q��(��ff���
���ChaH��(������H���HCe�                                    Bxq-��  "          AF�R��\)�\)��p��
=Chh���\)�p���\)��\)CeǮ                                    Bxq-�t  T          AF�H��R�(�����Ch����R�ff������Cf
=                                    Bxq-�  T          AEG����33������\)Ch�=����H�����Cf:�                                    Bxq-��  
�          A9��{����������Ci
=��{����mp�����Cf�                                    Bxq-�f  �          A6=q��{�z�� ���#33Ceٚ��{��ff����33Cb�                                    Bxq-�  �          A8���ᙚ�\)������33Cg�=�ᙚ���w���\)Ceff                                    Bxq-�  
�          A8z��޸R�ff�}p���ffCh� �޸R����W
=����Cf��                                    Bxq-�X  "          AA���������8Q��Y��Ci�������
=�O\)�x��Ch!H                                    Bxq.�  T          A6�\��{�Q�p�����Chp���{�\)�Q���\)Cf��                                    Bxq.�  "          A<z���ff��׿fff����Ch:���ff���S33��=qCf}q                                    Bxq.+J  �          A;������Q쿂�\��  Cf�f�����
=�U���RCd�                                    Bxq.9�  T          A6{��ff�(��xQ���(�Cf�
��ff�\)�N{����Cd�q                                    Bxq.H�  T          A4������
�h�����
Cf�H���33�J=q���
Ce{                                    Bxq.W<  "          A3�
��\�
�H�Q����Cf�=��\��R�C33�|  Ce�                                    Bxq.e�  T          A3
=��ff�z�B�\�z=qCe�
��ff� ���=p��uG�Cd(�                                    Bxq.t�  T          A2�R����
=�=p��s33Ce@ ������R�:=q�q��Cc�{                                    Bxq.�.  �          A1�����Ϳ5�i��CfxR������9���rffCd�
                                    Bxq.��  "          A2�\���	��E��|(�CfL������=p��v=qCd�H                                    Bxq.�z  �          A0(���z���ͿG����Cg(���z�� ���>{�zffCe}q                                    Bxq.�   �          A.�\��Q���ÿfff��  Ceٚ��Q���G��AG����RCd�                                    Bxq.��  �          A/�
��G��녿�������Ce���G���G��P  ��Cc��                                    Bxq.�l  "          AA��������p���(�Cj&f����R�9���\z�Ch�                                    Bxq.�  
(          AMG��陚�((��\��(�Ck33�陚� ���E��^�HCj�                                    Bxq.�  
�          AL����Q��(  ��  ����CkY���Q��!G��<���U��Cj@                                     Bxq.�^  T          AR�R���.{��\)���RCk����(  �7��IG�Cj��                                    Bxq/  �          AP�����,(����
��\)Ck�f���&ff�1��E�Cj�R                                    Bxq/�  
�          AS33���0  >�z�?�  Cl�����+\)�#33�2ffCk�                                    Bxq/$P  �          AN�\�陚�*=q��33�ǮCk�\�陚�#33�Dz��\(�Cjff                                    Bxq/2�  �          AK
=��(��'������Ck��(���
�L���i��Cjz�                                    Bxq/A�  T          AL(�����-��?fff@��CnxR����+���z��	p�Cn(�                                    Bxq/PB  "          AXQ���  �8z�?�G�@���Cn�R��  �6ff��(��Cns3                                    Bxq/^�  �          AMp����(���Q����CZٚ�����p����CT�                                    Bxq/m�  �          AO�
������
��p���
CR  ����W�����   CG�H                                    Bxq/|4  
�          A[�
�z����
��ff��z�CX��z����� ����
CP.                                    Bxq/��  
�          A\(��(����H��  ��33CY�H�(��������	�CRǮ                                    Bxq/��  �          A\(��{���G���{C\��{��{�����CU                                    Bxq/�&  T          A_�
��
ff���H����C^  ���Q����
��(�CX)                                    Bxq/��  �          A`  ��
�33������HC^���
�������{CX�                                     Bxq/�r  T          A^�H�������ff��\)C`�����������33CZff                                    Bxq/�  
�          A^�\�{��
��ff����C_��{��{��=q��z�CZh�                                    Bxq/�  �          A`Q��
=��
��ff���HC`ff�
=��  ��z���C[aH                                    Bxq/�d  �          A`���Q���R��ff���\C_�3�Q���{���
��z�CZ�                                    Bxq0 
  
(          A`z��ff�(���  ��ffC_��ff�����(�����CY�                                    Bxq0�  �          A_���R�����
����C_�f��R��(��������C[33                                    Bxq0V  �          A_\)����������C`���� (���z���{C\:�                                    Bxq0+�  �          A^�H�ff�
=�����
=Ca)�ff�p���33����C\��                                    Bxq0:�  �          A^ff���������C_���������  ���
C[��                                    Bxq0IH  T          A^�\��H��\�qG��|z�C`���H��\������  C\޸                                    Bxq0W�  �          A^�H����#\)�0���6�\Cdp�����
=������RCan                                    Bxq0f�  "          A^�R�=q�"ff�,(��2�\Cc���=q�=q�����  Ca�                                    Bxq0u:  
�          A]��=q�%�(��"=qCe^��=q��R���R���
Cb��                                    Bxq0��  T          A^{����'33�
=�z�Ce޸����Q���������Cc:�                                    Bxq0��  �          A]p���
�%��:�H�C
=Ce�q��
�(���p���Q�Cb�f                                    Bxq0�,  �          A\���
�\�$���C33�K�
Ce�R�
�\����������CbǮ                                    Bxq0��  
�          A]���
�"{�.�R�6�\Cdk���
������=qCan                                    Bxq0�x  �          A\z��p�����R�\�]p�Ca���p���H���\��G�C^{                                    Bxq0�  
�          A[���
�\)�>{�H  C`�f��
��R�������C]�\                                    Bxq0��  "          A[��G���=p��G33C`W
�G��G���ff��Q�C\�q                                    Bxq0�j  
�          A[�
���  �<(��E�Ca����\)��
=���RC]�                                    Bxq0�  �          A\z�������>{�G\)Ca=q���(���Q����C]�                                    Bxq1�  �          A\����\��\�8���ACa���\�
=q���R����C^�\                                    Bxq1\  �          AZff��R�ff�ff���C`33��R�	G���(�����C]�
                                    Bxq1%  "          AZ{�(��(��ff�=qC_}q�(��33���H����C\޸                                    Bxq13�  T          AX�����p���z���C`(����	������C]��                                    Bxq1BN  �          AYG������R��
��Cb!H����������33C_�)                                    Bxq1P�  �          AW
=�p�����\)�*�HCc��p������H���Ca{                                    Bxq1_�  �          AXz��
=q�!�����
=Ceu��
=q�\)������p�Cb��                                    Bxq1n@  �          AYp��p��)���У����Cg�=�p�����R��z�Ce�                                    Bxq1|�  �          AX���p��(�ÿ�Q���(�Cg���p���������(�Ce��                                    Bxq1��  "          AYG��p��)����G�����Cg�\�p��ff���H��ffCe�H                                    Bxq1�2  "          AX��� (��.{�(��%Ci�H� (��&=q�R�\�a�Ch\)                                    Bxq1��  T          AX(���  �0�þ�녿�  Ck���  �)��H���V�HCi��                                    Bxq1�~  "          AX  ��=q�2�H���\)Ck���=q�,���8���E�Ck                                      Bxq1�$  �          AXQ���
=�1p�>.{?5Ck#���
=�,���%��0  Cjc�                                    Bxq1��  
�          AX������/�=�\)>�\)Cj33����*ff�)���4z�CiaH                                    Bxq1�p  T          AY������1p�>Ǯ?�z�Cj�������-p��
=��Ci�                                    Bxq1�  
�          AY����0��?��@�Cjs3����-����z�Ci�                                    Bxq2 �  
�          AY����z��0z�?s33@���Cjk���z��.�H��ff��33Cj(�                                    Bxq2b  �          AY���-�?�=q@��Ci(���,  ��\)��(�Ch�q                                    Bxq2  T          AX��� (��.�\?z�@(�Ci��� (��+��Q����Ci5�                                    Bxq2,�  
�          AXz���*=q>\)?�Cg�)��%���\)�)Cg�                                    Bxq2;T  
�          AV�\��(��-�����Ci����(��'��,(��9��Ci
=                                    Bxq2I�  T          AU�����R�*�R?   @Q�CiE���R�'��	���Q�Ch��                                    Bxq2X�  �          AS33��  �&�H�33� ��Ci^���  ���������z�Cf�f                                   Bxq2gF  �          APz���{�"�\�'��:{Ch�)��{�33��G���Q�Cf�                                    Bxq2u�             APz����H��
�~�R��Q�Ci�3���H�33�����  Ce޸                                   Bxq2��  �          AQG�����!��8���LQ�Ch�{�������������HCe�)                                    Bxq2�8  T          ATQ��	p��!p��B�\�S33Ce�{�	p��p��N{�`��Cd&f                                    Bxq2��  
�          AT���\)�$Q��(���Cf��\)����<(��K�
CeY�                                    Bxq2��  :          AT���Q��#�=#�
>.{Cf33�Q���R�{�+33CeY�                                   Bxq2�*  �          AT����R�$Q�B�\�Tz�Cf����R��\�-p��<(�Ce��                                   Bxq2��  T          AR�H�	��\��녿�Ce�	�  �5��G33Cc�{                                   Bxq2�v  "          AO������R�}p����Cf
����{�X���r=qCd}q                                   Bxq2�  "          AO����R�"ff���\���
Cg�����R�z��n{��\)Cf�                                   Bxq2��  �          AP������.=q���Cl�����&�H�K��b{Ck��                                    Bxq3h  
�          AR{��=q�3
=�0���@��Cn����=q�*�H�Z=q�p��Cmp�                                    Bxq3  �          AU����=q�0z�\��G�Clh���=q�%���p���  Cj��                                    Bxq3%�  �          AXz����.�\�7
=�C\)Ck� ���{��{��  Ci\                                    Bxq34Z  	�          AZ=q�33�ff��p���  Cd#��33��p���\)�Q�C^(�                                   Bxq3C   �          AUG���=q�=q�������HCg���=q��
=������Ca�\                                    Bxq3Q�  :          AQG���=q��
��������Cf�3��=q��p���(��	��Cac�                                   Bxq3`L  T          AJ�R��\)�  �\(��}p�Cin��\)�	��
=��Ce��                                    Bxq3n�  
          AL(���� Q��4z��M��Cj(����z������CgG�                                   Bxq3}�  
           AM�����G��xQ���Q�Cg\)����p���33��z�Ccs3                                    Bxq3�>  �          AM���33�
=�u��
=Cf@ ��33����G���CbL�                                    Bxq3��  �          AMp���=q��H�j�H���HCg�R��=q��
��p����CdG�                                   Bxq3��  T          ALz������R�dz����Ch  ����(���=q����Cdc�                                   Bxq3�0  �          AM���G��
=���\��{ChT{��G�� ����  ��
=Cc��                                    Bxq3��  T          AMG��������=q��Q�Cg=q���� z���\)��  Cb޸                                    Bxq3�|  �          AIG���\)�����z���Ce����\)���H�����ffC`u�                                    Bxq3�"  �          AK\)������Dz��`��Ccff��� Q���{��p�C_��                                    Bxq3��  �          AK33��\����Z�H�{
=Cb=q��\��{���R����C^h�                                    Bxq4n  �          AK���33�
=�Tz��r=qCf@ ��33���������  Cb�q                                   Bxq4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxq4�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxq4-`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxq4<  �          AL  ��\)����^{�}�CeY���\)��R��z���(�Ca��                                   Bxq4J�  �          AK���z��Dz��`z�Cc8R�� (�����
=C_�                                    Bxq4YR  @          AK\)���z��[��{
=Cb�������
=��ffC^+�                                   Bxq4g�  �          AK33������?\)�ZffCc@ ��� �������(�C_��                                   Bxq4v�  "          AIp�������,(��E��Cb8R������Q���z�C_�                                    Bxq4�D  �          AK�����\)�Z�H�yp�C_�3������
��(���(�C[Ǯ                                    Bxq4��  �          ALQ��Q��\)�r�\��ffCa���Q���������z�C]n                                    Bxq4��  �          AMp��\)�
�\�l(���C`ٚ�\)��  ��{�ҸRC\�q                                    Bxq4�6  �          AN�R�33�ff�|(���(�C]���33��{�����  CY
=                                    Bxq4��  �          AMG�����R�z�H���\C^����
=�������CY�=                                    Bxq4΂  T          ANff����Q��~{���C_�����陚���ڣ�C[��                                    Bxq4�(  T          AO
=����
=��Q���{C_�����������{��33C[�                                    Bxq4��  "          AN=q���33��=q��(�C^�����ڏ\���CYǮ                                    Bxq4�t  
Z          AP  �������������
C[������{�׮��CU�
                                    Bxq5	  T          AO\)�{��ff��Q���Q�C]�)�{�Ϯ�ٙ����
CX+�                                    Bxq5�  "          AL���G���p���z����C[��G��������-=qCQ�R                                    Bxq5&f  "          AMp��
�H��
=�У����
CY��
�H����
=�z�CQ�{                                    Bxq55  �          AI���   ��=q��(����HCSG��   ��Q���\)���HCMs3                                    Bxq5C�  �          AJ{��H��=q��Q����CSs3��H��\)��33��{CMs3                                    Bxq5RX  T          AH(����陚��{����CZ�{����  ���H���
CUxR                                    Bxq5`�  
�          AH����\�33�Mp��mCc�H��\��z�������=qC`�                                    Bxq5o�  �          AG\)��
=����g���33Cc�q��
=��{��p�����C`                                      Bxq5~J  T          AF�H��=q����Tz��xQ�Ce.��=q��
=��{�ϮCa��                                    Bxq5��  T          AFff� (��\)�n{���Ccff� (���G���\)��=qC_B�                                    Bxq5��  �          AF�R��{�(��mp���
=Ce���{���\��������Ca�                                    Bxq5�<  
�          AF�R����
=�n�R��Cap�����������C].                                    Bxq5��  T          AE������p�������
C^޸�����{��(���CZ{                                    Bxq5ǈ  �          AC�
�z���\�{����C\���z���(���p��ܣ�CX�                                    Bxq5�.  T          AE���
=�33�q���z�C`#��
=�����p���z�C[�q                                    Bxq5��  �          AF�R��=q�\)�{����Cg����=q��
=��=q��Q�Cch�                                    Bxq5�z  "          AI���=q��
�z=q���Cf����=q�   ������z�Cb�{                                    Bxq6   T          AIp�������|����Q�Cg(���� ���Å��ffCc�                                    Bxq6�  T          AI����33�p��u����Ce.��33���
��ff���Ca�                                    Bxq6l  "          AG������R�j�H����CdxR�����  ��  �ۅC`p�                                    Bxq6.  �          AEp���p��G��Z�H��z�Cd!H��p���
=��\)����C`L�                                    Bxq6<�  �          AF�H�����R�Tz��w�
Cd&f������\�����Q�C`p�                                    Bxq6K^  �          AG���(��G��W��z�RCe
=��(���
=��  ����CaT{                                    Bxq6Z  �          AB�R��(���R�]p���=qCnE��(��Q���G���\)Ck                                    Bxq6h�  �          AB�R�θR���XQ����Clc��θR�	����p���=qCi\                                    Bxq6wP  �          AC
=��p�����N�R�u��Cl����p��  �����؏\Ci��                                    Bxq6��  "          AC
=�У�����G��m�Clff�У���
�����CiJ=                                    Bxq6��  �          AD���ə��!��B�\�dz�Cn��ə�������љ�Ck0�                                    Bxq6�B  "          AC\)�ʏ\�!��.{�M�Cmٚ�ʏ\��������(�Ck&f                                    Bxq6��  "          AC\)����#
=���,z�Cn=q��������
=��(�Ck��                                    Bxq6��  �          A=���
�z��=q�;�Ck0����
�
=q����(�Ch�                                    Bxq6�4  4          A<����ff�z��P����ffCdٚ��ff��R��Q���{Ca
=                                    Bxq6��  �          A<(��p����������C]���p��������33CX:�                                    Bxq6�  �          A9������Q���ff��(�CZ�q�����ff��\)���CU#�                                    Bxq6�&  �          A:{�\)������\��ffC]{�\)����
=���
CW�{                                    Bxq7	�  �          A;33��
�����\��G�C]Q���
��Q�������CX�                                    Bxq7r  �          A;�� ����
=������  C^��� ���Ǯ������=qCY��                                    Bxq7'  �          A7���ff�	G��"�\�LQ�Cf  ��ff��{���\��{Cb�H                                    Bxq75�  T          A5���Q���
�+��[�Cd�H��Q���=q��z���Ca@                                     Bxq7Dd  
�          A4�����
� ���8���mp�Cc�����
���H������
=C_�H                                    Bxq7S
  �          A2ff��33���>{�w33Cd�f��33��\��(��υC`�                                    Bxq7a�  �          A1���(�� Q��<(��up�CdW
��(���G����H��(�C`��                                    Bxq7pV  �          A2ff��  ����@���z�RCc� ��  ��p���(��ϮC_��                                    Bxq7~�  �          A2�H��R� (��@  �y�Cc����R��Q���z��ϙ�C`.                                    Bxq7��  �          A1p��������'
=�Z=qCbc��������
=��(�C^��                                    Bxq7�H  �          A0�����H����Q��G\)CaE���H�ڏ\���R��\)C]��                                    Bxq7��  �          A0����Q���p��$z��W\)Cc� ��Q��������R��z�C`�                                    Bxq7��  �          A1p���  � �����Ep�Cc����  ��R������ffC`�\                                    Bxq7�:  �          A1����
�p��!��S33Cd�����
��ff���R��(�CaT{                                    Bxq7��  �          A0��������
�H�5G�Ce��������������RCb��                                    Bxq7�  �          A0��������\��
�+�
CdǮ������z��������Ca�3                                    Bxq7�,  �          A/���(����Q��"�HCd����(����
�y�����\Ca�                                    Bxq8�  �          A1�����\��{��\Cdk�����{�u��=qCa�                                    Bxq8x  �          A.ff�����{��
=���Cc���������g
=����Ca�                                    Bxq8   T          A1�������\��
=��Cd&f�����\)�j�H���\Ca��                                    Bxq8.�  �          A2�\��� �׿��	p�Cb�������
�hQ����C`Q�                                    Bxq8=j  T          A3���(��G��������CdxR��(���ff�_\)����Cb:�                                    Bxq8L  �          A4(���(��ff������Cd����(�����Vff��z�Cb�)                                    Bxq8Z�  �          A0Q����H�����ff� z�Ce�����H��{�e���CcW
                                    Bxq8i\  �          A4(���G������R��(�Cc����G����H�`����p�Ca33                                    Bxq8x  �          A2�\��33� z��G���G�Cb����33��z��^�R��\)C`.                                    Bxq8��  �          A0����p�� z��33�	��CcB���p����g����HC`                                    Bxq8�N  �          A1����(���������1Ca�\��(��߮������(�C^�                                     Bxq8��  �          A/\)��33�����=q�J�HC^�)��33��{��p���ffC[aH                                    Bxq8��  �          A.ff��\)������ (�C]�q��\)��
=�h����(�C[�                                    Bxq8�@  �          A/��p��޸R������C[�{�p����H�QG�����CY8R                                    Bxq8��  �          A0(���R��  �������C]����R���G
=��\)C[B�                                    Bxq8ތ  �          A/\)��ff��p���G�����C`�H��ff��G��X����=qC^n                                    Bxq8�2  �          A2�\����(���G��p�Ca�R�����l(����HC_�                                    Bxq8��  T          A3���z���G���(��G�C`�H��z���33�h����\)C]��                                    Bxq9
~  �          A/\)�����������MG�Cb�����������\��Q�C_��                                    Bxq9$  �          A0Q���=q��p��6ff�pQ�Cd:���=q��{��Q���p�C`p�                                    Bxq9'�  �          A/�
�ָR��R�?\)�|z�Cf�{�ָR��z����R���HCb�                                    Bxq96p  �          A.{�����Ϳ���Ca�H����{�n{��=qC^�\                                    Bxq9E  �          A/
=����Q쿴z�����CZ�{�����E����HCX�                                    Bxq9S�  �          A/��p���ff�ٙ��ffC]��p������^{��CZ�                                    Bxq9bb  �          A/���\)���H��{�\)C^�{��\)���[����
C[�                                    Bxq9q  �          A/���Q���G����
�p�Cb���Q���\�mp���{C_O\                                    Bxq9�  �          A/���������Q��"�RCb�������xQ�����C_Ǯ                                    Bxq9�T  �          A0���陚� �׿�
=� ��Cc�=�陚�����z�H��ffC`�3                                    Bxq9��  �          A0z����H�{�33�@Q�Cd����H�����������Ca��                                    Bxq9��  �          A0z��ff��\)��G��33C]�\�ff�����c�
��Q�CZ�R                                    Bxq9�F  �          A0z���
��33�У��Q�CZ����
��ff�U����RCX&f                                    Bxq9��  �          A.ff����ƸR��z��Q�CW@ ������\�L������CTff                                    Bxq9ג  �          A,������p�������CUk�����{�%��\��CS=q                                    Bxq9�8  �          A-G�����녿���CW���������0  �k\)CU��                                    Bxq9��  �          A-�  �ʏ\��{���
CW�)�  ���\�,���fffCU��                                    Bxq:�  �          A-���
�R��p�������CX� �
�R���,(��e�CVW
                                    Bxq:*  �          A+
=�  ��=q�^�R��CV� �  ��z��=q�PQ�CT��                                    Bxq: �  �          A*�R����ȣ׿����{CX:������G��(���ep�CV\                                    Bxq:/v  �          A*=q�����(��}p����CU�3�����p��{�V�HCS�)                                    Bxq:>  �          A)p��	����
��ff��
=CW��	���z��&ff�c�CUT{                                    Bxq:L�  �          A)���Q���\)�fff����CX&f�Q������\)�Y�CV#�                                    Bxq:[h  �          A)�����녿xQ���
=CX�f�����H�$z��`(�CV��                                    Bxq:j  �          A+�
�(���33������CV�)�(�����(Q��c33CT��                                    Bxq:x�  �          A+��{������\�ڏ\CUk��{���\�0  �n=qCR��                                    Bxq:�Z  �          A,Q���R��=q�z�H��ffCSG���R��(�����MG�CQ33                                    Bxq:�   �          A,(��p���p���  ��CP�R�p���p��#�
�\  CN��                                    Bxq:��  �          A.{�!�y���k����
CI
�!�aG���z��!CG8R                                    Bxq:�L  �          A-���H��  ���\���
CK+���H�u��1CI!H                                    Bxq:��  �          A.{��
���R�p����\)CP�q��
�����G��A�CN�R                                    Bxq:И  �          A.�\�G���ff�p����ffCU:��G���  �p��Q�CS.                                    Bxq:�>  �          A-���\)��(�����HCSn�\)��녿��#�CQ�R                                    Bxq:��  �          A*{�G�������\���CR���G����R����Q�CP�H                                    Bxq:��  �          A)���Q���
=�����\)CS8R�Q���  �p��V�\CP��                                    Bxq;0  �          A*ff�Q�����
=�J�HCQ!H�Q����\��
=�'
=CO�                                    Bxq;�  �          A)G����Ǯ���\����CX\)����  �(Q��f{CV(�                                    Bxq;(|  �          A)��������H���R��Q�CY0������G��7��|Q�CV��                                    Bxq;7"  �          A&�R��\�ڏ\��(��/
=C^���\��=q�l�����CZ��                                    Bxq;E�  �          A((�� ����33��ff��
C[aH� ����ff�O\)����CX�                                    Bxq;Tn  �          A(  �   ��������ffC[���   �\�A���Q�CY:�                                    Bxq;c  �          A'�
�   ��zῪ=q���C[���   �����C33��33CY)                                    Bxq;q�  �          A'�� z���33��p��أ�C[h�� z������<(����\CX�                                    Bxq;�`  �          A'
=�   �У׿�{��C[.�   ��33�Q�����CX5�                                    Bxq;�  �          A&�R���Ǯ��G���CY0�����{�7���(�CV�H                                    Bxq;��  �          A'��	G���p��\(���\)CV�)�	G���\)�=q�T��CT��                                    Bxq;�R  �          A((�����ə���
=��G�CY5������{�C33���CVs3                                    Bxq;��  �          A(z�����p�����Q�CY�3������E���=qCW33                                    Bxq;ɞ  �          A(  �{��  �k�����CX���{�����#�
�aCV��                                    Bxq;�D  �          A(Q���
����^�R��  CW�3��
��ff�\)�Z�HCU޸                                    Bxq;��  �          A&=q�=q��33���G�CX
=�=q��\)���A�CVQ�                                    Bxq;��  �          A&{�
=���þ��
��  CW���
=��
=���+
=CV&f                                    Bxq<6  �          A'33�
{���ͽu���
CV^��
{��p�������CUL�                                    Bxq<�  �          A&{��
��  �.{�o\)CY+���
���H���P(�CWL�                                    Bxq<!�  �          A#\)�=q��G��#�
�eCV���=q�������D��CT�=                                    Bxq<0(  �          A�R��z����R�#�
�l(�CY���z���=q�{�O�CW33                                    Bxq<>�  �          Ap���ff��
=��33��RCYǮ��ff���
�=p���  CV�H                                    Bxq<Mt  �          Ap���R�љ���Q��=qC^@ ��R����I�����C[^�                                    Bxq<\  �          A!G���  �أ׿�ff�$��C_��  �����dz����
C[�R                                    Bxq<j�  T          A!���������ٙ���\Cb&f������ff�e���(�C_{                                    Bxq<yf  T          A!����  ��ff�
=�@z�Cd�f��  ���H�������Can                                    Bxq<�  
�          Aff��G�����{�O33CeG���G������
��
=Ca��                                    Bxq<��  �          A������p�����_
=Ch�)�����
=��z����Cd�                                    Bxq<�X  �          A#�������
=�,���uG�Cj�������������
=Cf0�                                    Bxq<��  �          A�R��\)���R��ff��=qCjB���\)�陚�[���Q�Cg�H                                    Bxq<¤  �          Az����\�����(��Q�Ck�����\������  �ϮChT{                                    Bxq<�J  �          Az���  ���,(���Q�Cm���  ��z�������  Cj                                    Bxq<��  �          A�����
��G�����c�Ckh����
�ڏ\��ff����Cg��                                    Bxq<�  �          AQ��j�H���\(���Cu(��j�H������  Cq33                                    Bxq<�<  �          A33�E������T  Cy���E��z���G����CwJ=                                    Bxq=�  �          A{�!G��
�R����mC}�!G���p���{��\)C{Ǯ                                    Bxq=�  �          A�R�ff�\)��H�v�RC~}q�ff��R�����z�C|�=                                    Bxq=).  �          A33�Z=q�(���
�O\)Cw���Z=q���
��Q��ݙ�Cu+�                                    Bxq=7�  �          A�׿�(��   �dz����\C��Ϳ�(���ff�������C��                                    Bxq=Fz  T          A������G��p����(�C��{����{�����#�
CxR                                    Bxq=U   �          A  ������p��\(�����C�&f����������Q��Q�C�>�                                    Bxq=c�  �          A33������HQ���p�C~�
����(���{���C|J=                                    Bxq=rl            A
�H�mp���{����ffCs��mp���
=�������Cp.                                    Bxq=�  �          A
=q�s33��Q��.�R���Cr\)�s33�ƸR������Cn�                                    Bxq=��  �          A	p�������\)�%��z�Cn��������\)��{���Cjp�                                    Bxq=�^  �          A��Z�H��  �E��Q�Cuz��Z�H�ʏ\�����Cq�H                                    Bxq=�  �          AQ��QG������HQ�����Cw33�QG����H��{��
Cs��                                    Bxq=��  �          A(��'
=��\)�N�R��ffC{���'
=��  ���H���Cx��                                    Bxq=�P  �          A���{��\)������z�C�
=��{�����Å�*�HC�+�                                    Bxq=��  �          A�Ϳ�  ����\����G�C�c׿�  �Ӆ��G���C���                                    Bxq=�  �          Aff������G��p  ���HC�R���������G��%�C��                                    Bxq=�B  �          A��������U����\C~G�����љ�������C{s3                                    Bxq>�  �          A���%�����/\)���C{��%�������
��Cy=q                                    Bxq>�  �          A\)�\����G��(����
=Cun�\���Ϯ��p��Cr                                      Bxq>"4  �          Aff��ff���
�������Ci�
��ff�У��C33���\Cgk�                                    Bxq>0�  �          Az���p���>��
@   Cf:���p�������H�*�RCeh�                                    Bxq>?�  �          A(�����ᙚ�.{����Ce������ָR���YCdG�                                    Bxq>N&  �          A�������33>���@p�Cd���������=q���Cc^�                                    Bxq>\�  �          A=q������33>���@z�Cd�=�������Ϳ���%��Cc��                                    Bxq>kr  �          Aff������H���H��RCg�f�����(��[���Q�Cd�                                    Bxq>z  �          A����(���(����
��CdJ=��(��ҏ\���R�H  Cc
=                                    Bxq>��  �          A����33��G�>8Q�?�=qCb���33��녿��
�0��Ca�                                    Bxq>�d  �          Az���p���(�?&ff@�G�C]�{��p���녿����=qC]=q                                    Bxq>�
  T          A��P  ��33���\)Cun�P  ��(���G���Cr�                                    Bxq>��  �          A���У���\�hQ�����C�XR�У��أ����H��C�7
                                    Bxq>�V  T          A\)�R�\�����\�m��Cw���R�\�߮��Q���{Ct��                                    Bxq>��  �          A����33��(������Cr�=��33��33�p������Cp�                                    Bxq>�  �          AG�������(���{�
=qCq}q���������e����Co{                                    Bxq>�H  �          A=q�p  ���
=�B�RCu0��p  ��ff��ff����Cr�                                     Bxq>��  �          A�R�P���G����H�ECx�)�P������������{Cv33                                    Bxq?�  �          A���� z��]p�����CǮ����p������33C}
=                                    Bxq?:  �          A(��������p��Q�Cpu������\)�g
=��
=Cm�                                    Bxq?)�  �          AG���(����ÿ�����=qCo����(���33�Vff��
=Cmh�                                    Bxq?8�  �          A����(����H��\)�ٙ�Cj�f��(���p��\)�x(�Ci�                                    Bxq?G,  �          A����{��33=L��>���ChE��{�ᙚ�
=�Q��Cg{                                    Bxq?U�  T          A�������=�\)>�(�Cf������{��
�L  Ce�{                                    Bxq?dx  �          A=q���H��z�>���?�Ce�=���H��������2�HCd��                                    Bxq?s  �          A�\��
=��=q>���?��HCg� ��
=��\���7�Cfٚ                                    Bxq?��  T          A�H��z�����@�\Apz�C_�H��z����>�G�@7
=Ca�\                                    Bxq?�j  �          Az���33����@:�HA�C\����33��p�?�ffA��C`#�                                    Bxq?�  �          @��H������R@hQ�A��HC\������\@��A~{Ca!H                                    Bxq?��  �          A z���ff��Q�@'�A�z�CeW
��ff����?J=q@��Cg                                    Bxq?�\  �          A�\��Q���p�@%�A��Cc#���Q���?E�@�ffCe�
                                    Bxq?�  �          A�R��(��ə�@
=qAnffCf�\��(�����>B�\?��\ChY�                                    Bxq?٨  �          A
=�����ƸR@33AaG�Ce��������G�=���?+�Cf��                                    Bxq?�N  �          A(���(����?���AIG�Ce���(���=q���Tz�Cf�                                    Bxq?��  �          A	������ٙ�>�@K�Ch+�������(���=q�)�CgxR                                    Bxq@�  �          Ap����������  �;�Cb��������33�aG����RC^Ǯ                                    Bxq@@  �          AQ���G���(���=q��=qCh�=��G���33���
�Q�Ca^�                                    Bxq@"�  �          A�R������=q�Q���  Cl������Q���  �ffCf^�                                    Bxq@1�  �          A ���s33��z��e���
=CnE�s33���������#G�Cg�                                     Bxq@@2  T          A��tz��ȣ��Tz�����Cn��tz���
=��=q��RCh�                                     Bxq@N�  �          @�  �g���
=�,����33Cn��g����
��(��Q�CiaH                                    Bxq@]~  �          @���  ���H��Q��7�
Ck����  ��(��H����Q�Ch                                    Bxq@l$  �          @�
=�j=q������\���Clk��j=q��(��g���(�Cg�                                    Bxq@z�  �          @�  �tz���  �
=q��33Ck5��tz���=q�(���{Ci                                    Bxq@�p  �          @�=q�j�H��=q?��
A��CkaH�j�H��(��#�
���
Ck�                                    Bxq@�  �          @���fff��G��G��޸RCk��fff��G��
=���
Ci�                                    Bxq@��  �          @����N�R��z��Q��ip�Cp0��N�R��33�U��Q�ClT{                                    Bxq@�b  �          @�����������G��mp�CgT{�������33���Ce+�                                    Bxq@�  �          @�  ��=q��\)��33�(��Cg���=q���\�
�H��\)Ce+�                                    Bxq@Ү  �          @�\�����>Ǯ@:�HCeQ������\��33�*=qCd�                                    Bxq@�T  �          A�\��\)���������\Cl���\)��  �Dz���(�CiB�                                    Bxq@��  �          A��������G�������Ck�������Å�K���  Ch                                    Bxq@��  T          A
�\�����33�5��\)CkL���������7���
=Ci�                                    BxqAF  �          AG������\)�5��z�ClG�������
�Dz���{Cj�                                    BxqA�  �          A33��
=��33����)��Cn!H��
=���H�0  ��33ClW
                                    BxqA*�  �          @��mp���33���\)Cp�f�mp��Å�#33��\)Cn��                                    BxqA98  �          @أ��g����׿��H�G�Cl�q�g���G��G
=���HCh�                                    BxqAG�  �          @��
�XQ���
=���H�L��Cn@ �XQ�����E��  CjxR                                    BxqAV�  T          @�(��c�
���\��(��U�Ck  �c�
���
�>�R��G�Cf�{                                    BxqAe*  �          @��
�i����
=��ff�D��ChL��i�����\�-p����Cd(�                                    BxqAs�  �          @�G��qG��������
Cg.�qG������ ����\)Cd��                                    BxqA�v  �          @��
��G����H>�=q@ ��Cd����G���{��Q��3
=Cc��                                    BxqA�  �          @�(����R���>�ff@�{CbY����R���\�xQ��p�Ca��                                    BxqA��  �          @���(����\>�Q�@XQ�C]^���(��\)�p���(�C\��                                    BxqA�h  T          @�G����R��z�>W
=?��C]Q����R��  ��{�#�C\J=                                    BxqA�  �          @����ff�l��>.{?�{CZ5���ff�dz῁G���
CY(�                                    BxqA˴  �          @�
=������>�p�@Z=qCaG������G���ff�\)C`��                                    BxqA�Z  �          @˅��33���>�G�@\)Cb�f��33��Q쿆ff��Cb�                                    BxqA�   �          @���z���>�
=@qG�Cb޸��z���=q����(�Cb0�                                    BxqA��  �          @��H�����33>aG�?���C_Q������{��
=�+�
C^B�                                    BxqBL  T          @�G����H���׾W
=��\)C^�f���H�~�R�Ǯ�f{C\�f                                    BxqB�  �          @��������
��\)�*=qCbB����������\)C`{                                    BxqB#�  �          @�\)���������G���C`5������s33��  ���C]�H                                    BxqB2>  �          @�����(������Q��XQ�Ccff��(���(����
���
Ca�                                    BxqB@�  �          @�\)��������H���Ce�������׿�(����Cbk�                                    BxqBO�  T          @���mp����ÿ^�R��p�Ci�)�mp���\)�(���ffCfff                                    BxqB^0  �          @�  �l����
=��33�P��CiO\�l�����H��Q����Cg
                                    BxqBl�  �          @��H�qG���
==L��>��Cg\)�qG���\)��p��b�HCe��                                    BxqB{|  �          @���6ff��G�?\(�A�Co@ �6ff��녿=p���\CoY�                                    BxqB�"  �          @�=q�j=q����?333@�Ch��j=q����aG��=qChY�                                    BxqB��  �          @���R�\��=q?L��@��Coz��R�\���ÿ����CoG�                                    BxqB�n  �          @������\@G�A�Cx�H����Q�>��
@9��Cz
=                                    BxqB�  �          @��׿�
=�g�?�p�A�p�C{녿�
=�z=q>W
=@>�RC}&f                                    BxqBĺ  �          @��
�����
=?���AF�\Cs�������\��(�����CtT{                                    BxqB�`  �          @�\)�7
=�I���z�H�O�
CcǮ�7
=�)�����H��p�C^�H                                    BxqB�  �          @�33�O\)��ff��z��H��ChT{�O\)�w���33���Cf\                                    BxqB�  �          @�\)�?\)����=�?��Ck��?\)���\����e�Ci�                                    BxqB�R  �          @��H�?\)��p�=�G�?���Ck��?\)��ff��{�k�
Cj��                                    BxqC�  
Z          @��
�P�����>Ǯ@�{Ch���P����zῃ�
�1p�Cg�=                                    BxqC�  �          @�p��J�H��(�>\@��Cj#��J�H��Q쿋��9G�CiaH                                    BxqC+D  �          @�
=�?\)��G�>\@���Cl�)�?\)�������ACk�
                                    BxqC9�  �          @Ǯ�`  ����>aG�@ ��Ck�=�`  ����  �_\)Cj��                                    BxqCH�  �          @��H�j�H��z����Cjz��j�H������=q���HCh�H                                    BxqCW6  �          @���\)��Q�<��
>��Ce\)��\)��ff�����h  Cc�R                                    BxqCe�  �          @��������=�G�?s33Cc������녿�=q�U�Cb8R                                    BxqCt�  �          A���z���G�>���@
=qCh����z���  ��\�T��CgaH                                    BxqC�(  �          AQ������(�>���?�p�Cj^������=q�ff�]Ci&f                                    BxqC��  �          A(������p�>�=q?��HCks3�����\�\)�f{Cj.                                    BxqC�t  �          A������Q�>��?�z�Cj.���������c33Ch�H                                    BxqC�  �          Az���Q�����>k�?�ffCi�q��Q���=q�
=�d��Ch��                                    BxqC��  �          A	p���p���G�>��
@Q�Ck���p���\)���_�Ci޸                                    BxqC�f  �          A
{��z���\>�p�@p�Ck\)��z���G��33�\  Cj:�                                    BxqC�  �          AQ����H��>��
@�Cl33���H��p��	���d��Cj�q                                    BxqC�  �          A
=��Q���  ?333@��
CpaH��Q���G�����I�Co�                                    BxqC�X  �          A�
���
��?\)@k�Co����
��G���\�W�
Co�                                    BxqD�  �          AG��G����
?�(�A
=Cw���G���\����� (�Cw�)                                    BxqD�  �          A�
�`  ��z�?�@��Cuk��`  ��\��G��$  Cu=q                                    BxqD$J  �          A	��E���33?�A��Cx���E����
�����p�Cx�H                                    BxqD2�  �          Aff�O\)���?�@��Cw��O\)��33���
�'�Cv�f                                    BxqDA�  �          A{���H��  >�
=@8��Cp�����H��ff��h��Co�=                                    BxqDP<  �          Ap�����z�>�z�?��RCo�����ٙ����t��Cnp�                                    BxqD^�  �          A z���p��ʏ\�
=q�vffCh!H��p���  �,(���=qCeu�                                    BxqDm�  �          A   ��33��(��0����\)Cf=q��33��Q��1G����
Cc=q                                    BxqD|.  �          A��
=��zῆff��
=C`�H��
=���<(���(�C\Ǯ                                    BxqD��  T          Aff���\��=q���h��Cm�
���\�˅��R��33Ck�R                                    BxqD�z  �          A\)��������0�����\CbE���������,�����C_+�                                    BxqD�   �          A����ff��Q�u�У�Cd�R��ff��G������=qCb��                                    BxqD��  �          A�����H��z�<��
=�G�Cc�����H�������l��Ca޸                                    BxqD�l  �          A��p���z�=#�
>uCcG���p������
=�iCaz�                                    BxqD�  �          A�\���H���?�@h��Cb�����H��
=��=q�3
=Ca��                                    BxqD�  �          A33��{�ƸR?333@�Q�Cc����{��녿\�%��Cb�
                                    BxqD�^  �          A�\�������
?.{@�Ce�R�����ƸR�˅�/\)Cd�R                                    BxqE   �          A���(��ə����H�a�Ci�f��(����R�������CdT{                                    BxqE�  �          A�H��\)��Q�?
=q@q�Cf&f��\)�����ٙ��@  Ce.                                    BxqEP  �          A����p���  >��@5�Ce\��p���������K33Cc�)                                    BxqE+�  �          A  ��p��ƸR>\@)��Cd�)��p���{���M��Cc��                                    BxqE:�  �          A��������=�Q�?��Ce������������q��Cd\                                    BxqEIB  T          A��  ���L�;��
Ce����  ��Q���R��Cc�H                                    BxqEW�  �          A33����\�����Cc�H������H�����=qCa}q                                    BxqEf�  �          A���������z�����Cb�������{�,������C_��                                    BxqEu4  �          A
=���H��ff������CeG����H���\)���
Cb�\                                    BxqE��  �          A\)���\��zῑ�� Q�Ce
=���\��=q�P����G�C`��                                    BxqE��  �          A=q��z��ȣ׿��>{Ch���z���Q��s�
��\)Cc�                                    BxqE�&  �          A�\���
����(��&=qCfff���
��  �e���HCa�R                                    BxqE��  �          A�R���\��33�ff��z�Cg�H���\��33��(���\Ca�                                    BxqE�r  �          AG����\�������Cg&f���\��  ���H��RC`�=                                    BxqE�  �          A �������=q�E���(�Cg�������=q��
=���C_.                                    BxqE۾  �          Ap���33���R�:�H��\)ChT{��33��Q����
�ffC`ff                                    BxqE�d  �          A ������������Ch������p�����  Cb��                                    BxqE�
  �          A{��z��˅�   �d  Ci����z����R��p�����CdG�                                    BxqF�  �          Aff��=q��G���z��;�Ck���=q����z�H���
Cf8R                                    BxqFV  �          A���
=���ÿ��
�J�RCk����
=��{������\Cf�)                                    BxqF$�  �          AQ���z����
��p��AG�Cj����z���G���������Cf�                                    BxqF3�  �          A���  �Ӆ��(����CjL���  ���R�c33��ffCf=q                                    BxqFBH  �          A���ff�ҏ\�����p�Ci\��ff��ff�^{��p�Ce�                                    BxqFP�  "          A��z��У׿Tz���CgǮ��z������J=q���
CdO\                                    BxqF_�  �          AG���(����
��\)��33C`�R��(�����=q���HC^^�                                    BxqFn:  �          A����\)��\)���
�{Cb:���\)��{�   ��=qC_�=                                    BxqF|�  �          Ap�����ff���\���HC_޸����p��C33���RC[�3                                    BxqF��  �          Ap���=q��{�У��4Q�CeL���=q�����r�\����C`\                                    BxqF�,  �          A�\����녿=p����
CkxR�������L�����ChB�                                    BxqF��  �          A��p�����?�\)@��Co���p���������:=qCn��                                    BxqF�x  �          A33���R� ��?���A�\Cm����R�   ���H�"�\Cl�                                    BxqF�  �          AQ����\�p�?���@�Q�Cl�)���\�   ���
�'�ClQ�                                    BxqF��  �          A��G��\)?�  @�G�Cm.��G��p���33�1p�ClǮ                                    BxqF�j  �          A���p�� (�?Tz�@��Cj�q��p���  �{�Pz�Ci�\                                    BxqF�  �          Aff��
=�(�?��
A�RCm����
=��
��z���RCm�f                                    BxqG �  T          A
=���Q�?�{@�33Co�R����\��
=�3
=Co��                                    BxqG\  "          A33���R��?˅A
=Co�����R�33�ٙ����Co�
                                    BxqG  "          A (���p��	p�?�A�\Cp8R��p��  ��z��0(�Co�3                                    BxqG,�  �          A Q���\)�	?���@��
Co�3��\)��H�Q��D  Cok�                                    BxqG;N  �          A��������?s33@�\)Co�����������YG�Cn�                                     BxqGI�  �          A������
?k�@��Cn������\)�ff�YCm޸                                    BxqGX�  �          A
=������
?�\@:�HCn� ����� ���/\)��  CmaH                                    BxqGg@  �          A(����R�
=?c�
@�\)Cm�����R��p����XQ�Cl��                                    BxqGu�  T          A����
��33?�p�A�\Ci�=���
���H��G��p�Ci�                                    BxqG��  �          A��������
@ ��AAp�Cg������\�p����(�Ch�\                                    BxqG�2  �          A  �Å��G�@�AEp�Cf��Å��G��Y����Q�Cf�R                                    BxqG��  �          A������>��
?�\)Cj!H����z��-p���p�Chn                                    BxqG�~  �          A����
���>B�\?�\)Cj�{���
��(��5��Q�Ch�3                                    BxqG�$  �          A����ff��=q�������Ck#���ff��(��N�R���\Ch��                                    BxqG��  �          A=q��  ���
��p���33Ch�R��  ��=q����ǅCd��                                    BxqG�p  �          Aff���
��{��Q��9Ce�����
��z�������\)C_�f                                    BxqG�  �          A����H���Ϳ0����Cg�R���H����^�R����CdY�                                    BxqG��  �          Ap����
��ff���
���HCh�R���
��  �J=q��z�Cf8R                                    BxqHb  �          A  ������
=��\)���Cjc��������H�B�\��p�Ch{                                    BxqH  �          A�H��{���?@  @��Cj�H��{���G��^�HCi�                                    BxqH%�  �          Aff��(�����@	��AT  Ck����(����ÿp�����Cl�{                                    BxqH4T  �          A���=q��녿���"�\Cf� ��=q��(���  �ٙ�Ca@                                     BxqHB�  �          AQ������p�>u?��RCh��������*=q��ffCf�                                    BxqHQ�  �          AG���\)����@vffA��
Co5���\)��H?�R@mp�Cq�q                                    BxqH`F  �          A�������@l(�A�G�Cp!H������
>��@��Cr�)                                    BxqHn�  �          A\)��(��׮�\)�p��Cj
��(����\��{���Cc�                                    BxqH}�  �          A����\)��G����R��Cg���\)�1G�����]��CS�q                                    BxqH�8  �          A������������	�Ch0������e��33�K33CX�\                                    BxqH��  �          A�������녽�G��333Cn@ ������{�>{��p�Ck�f                                    BxqH��  �          A���w
=��\>��@(�Cv���w
=��G����Cu:�                                    BxqH�*  �          Az����Q�>��?��
Cq5������G����Cos3                                    BxqH��  �          A�R��(���ff�333���Cnu���(�����q�����Ck:�                                    BxqH�v  
�          A  �����녿�Q���
Cr���������
��(���p�Co\                                    BxqH�  �          A����Q��G���=q�p�Co����Q��ٙ�������Q�Ck                                      BxqH��  �          Ap���
=������H�j�\Ck�3��
=�����  �=qCeG�                                    BxqIh  �          A����H��\)��ap�Ci=q���H��{���
� �\Cb��                                    BxqI  �          A����
��=q�����5Chu����
��\)������Cb�                                     BxqI�  �          A�R�������\)�]�Cn&f����ƸR��z���
Ch:�                                    BxqI-Z  �          A���(������QG�����CjT{��(���  ��33�(�Ca��                                    BxqI<   �          A\)���R��������
=Cg�3���R�Y������Mz�CW�=                                    BxqIJ�  �          Ap�����������
���C`������������S��CL�                                    BxqIYL  �          AQ������  �����:�RCg�=�������  ��=qCa��                                    BxqIg�  �          A���������� ���{33Cg�
������p����
���C`�                                     BxqIv�  �          A	����o\)����HCW������������;�
CD��                                    BxqI�>  �          A{���=p����G{C<#���?����R�=p�C ��                                    BxqI��  �          A���z��HQ���Q�� ��CR.��z�=p��\�@�C;�)                                    BxqI��  �          AG���  �mp��������CW@ ��  ��  ���H�Fz�CAh�                                    BxqI�0  �          AG���33�S33���R�#�CS�f��33�O\)�ʏ\�E�HC<��                                    BxqI��  �          A������U�����CRp�����s33��=q�:ffC=}q                                    BxqI�|  �          A=q���\�����
�\)CS:���������=q�9��C>��                                    BxqI�"  �          A{��z��B�\��p��)�CQp���z�����p��GffC9�                                    BxqI��  �          A	����L(���(��#z�CQO\���(���{�A��C:.                                    BxqI�n  �          A	���Q��=p����$��CO.��Q�\��z��?��C7��                                    BxqJ	  �          A  �\�Mp���{�(  CO�{�\�����޸R�C=qC7�                                     BxqJ�  �          A���(����љ��1p�CE����(�?Tz�����:p�C,��                                    BxqJ&`  �          A�ʏ\������>�
C?z��ʏ\?޸R��Q��;ffC$�H                                    BxqJ5  �          A���G���  ��z��>p�C6G���G�@&ff��z��,C��                                    BxqJC�  �          A
=��ff>��������>(�C0�R��ff@E�����$�C�                                     BxqJRR  �          A�
����?L���޸R�<C,�{����@fff��Q��=qC��                                    BxqJ`�  �          A�
����?�p������C�RC&:�����@���������HC�
                                    BxqJo�  �          A Q����
?�������D(�C%�)���
@��H��
=��C:�                                    BxqJ~D  �          A�\��  ?�33��z��J{C"h���  @�z���p��(�C�=                                    BxqJ��  �          A(���\)?�����
�DffC's3��\)@�=q��(���
CE                                    BxqJ��  �          A�����H?��R�����IQ�C aH���H@�������Cp�                                    BxqJ�6  �          A=q���H?����R�I
=C'
=���H@�
=��  �{C#�                                    BxqJ��  �          A���(�?�������I=qC&���(�@�G���G��C�f                                    BxqJǂ  �          A\)����?J=q��p��FffC,������@mp���{�$G�Cu�                                    BxqJ�(  �          A\)��{��{���>Q�C@k���{?�z����
�;��C$�q                                    BxqJ��  �          A�\�����R���
�5\)CG+����?Y����p��?��C,s3                                    BxqJ�t  �          A\)��
=�����ff�7z�CDO\��
=?�Q���33�<C)�=                                    BxqK  �          A���33�aG���Q��?Q�C<����33?�=q�����6��C!��                                    BxqK�  �          AQ����H�����p��9�
C9\)���H@ff���\�,�HC =q                                    BxqKf  �          A
�H���\��(��ə��:p�C?�
���\?�ff��\)�7�RC%(�                                    BxqK.  �          A���Ϳ��H��33�@CD�H����?�p��ָR�D��C'��                                    BxqK<�  �          A\)������
��p��933CF�f����?s33����BG�C+
                                    BxqKKX  �          A�\����\�ҏ\�7��CI&f��?5���E
=C-8R                                    BxqKY�  T          A33���>{���H�.CN�\��    �߮�F��C4                                    BxqKh�  �          A������*�H�ʏ\�%CI�)����>�=q���
�7z�C1��                                    BxqKwJ  �          A���
=�l�����R�p�CP�)��
=�Q���
=�7ffC:�                                    BxqK��  �          A�
��{�Y�������\CN
��{�\)��G��1(�C8�H                                    BxqK��  �          A=q��33�y����=q��HCR����33������8�C<��                                    BxqK�<  �          AQ���=q�QG���z��Q�CNs3��=q�
=q��  �2ffC8�q                                    BxqK��  �          A����=q�Mp���z��CN���=q�5�����-�C:^�                                    BxqK��  �          A
�H��  �?\)����	�
CL����  �����
=�$��C9=q                                    BxqK�.  �          A33����A���=q��G�CK33��녿}p���
=���C;�R                                    BxqK��  �          A\)���H�dz������=qCO�\���H��p���ff��C@33                                    BxqK�z  �          A
{���
�z�H�\����=qCQ�����
�33�����  CD��                                    BxqK�   �          Az�������\�!G���33CR�����&ff������\)CH5�                                    BxqL	�  �          A�R��  ��Q��{�u�CU�q��  �G
=�|�����CL�R                                    BxqLl  �          A Q���{��33�]p���33Cd�q��{�U����-=qCWaH                                    BxqL'  �          @�33�l(����H�}p���33CkY��l(��Fff��(��J��C\�                                    BxqL5�  �          @�p��i�����H�U��
=Cm�R�i���tz���G��9z�Cb@                                     BxqLD^  �          @���{���33���z{Cm33�{����
��\)�Ce�{                                    BxqLS  �          @���y����p��/\)��ffCn���y�����H��Q��$ffCe��                                    BxqLa�  �          @�
=�vff��z��333��z�Cn��vff���������&Ce�                                    BxqLpP  �          @��
�u���p��U���ffCm\�u��xQ����\�6��CaO\                                    BxqL~�  �          @�  �Z=q��z��~{�
\)Ci���Z=q�(����H�T��CW�H                                    BxqL��  �          @�z��b�\��  �h������Cj��b�\�:=q��ff�G�HC[aH                                    BxqL�B  �          @�(��E��(��`����Cj�\�E�����=q�Qp�CY޸                                    BxqL��  �          @�  �A����H�]p�� �Cl���A��'
=���
�P{C\�=                                    BxqL��  �          @�\)�P�������e��Cj���P���&ff��Q��NffCZ��                                    BxqL�4  �          @�p��,(���{�Vff����Cs���,(��Z=q��z��Lz�Cg�R                                    BxqL��  �          @�G��B�\��33�=p���G�Cq���B�\�n�R���
�;33Cf��                                    BxqL�  �          @�=q�Dz�������
=�~�RCs)�Dz����\�����z�Cl+�                                    BxqL�&  �          @��*=q��ff�%��ffCv�\�*=q��p�����1��Cn�q                                    BxqM�  �          @�ff�'
=��=q�z���
=CxO\�'
=���
��\)�({Cq�=                                    BxqMr  �          @�{�0  ���Ϳ��R�o�CxJ=�0  ������\��
Cr��                                    BxqM   �          @��.{��p��!����Cw�.{��(�����.G�Co��                                    BxqM.�  �          @�\)�(����
=�(Q���33Cw�=�(����(���\)�1�
CpJ=                                    BxqM=d  �          @�{�
�H��Q�������RC|�{�
�H������z��'��Cw��                                    BxqML
  T          @�
=���=q�   �p��C{�����{��{�Q�Cv�3                                    BxqMZ�  �          @��\�z���(��������C|��z���33�����ffCy(�                                    BxqMiV  �          @����
�H���H�\)���\C}� �
�H��Q��X���ң�C{��                                    BxqMw�  �          @�Q������\)���C}
=����z��N{����C{                                      BxqM��  �          @�ff��G���?�@�Q�C33��G���\)�{��33C~0�                                    BxqM�H  �          @�R��
�޸R�0����(�C{����
��p��o\)��Cx��                                    BxqM��  �          @����Q���Q��Q����HCo5��Q��\������Bp�Cbp�                                    BxqM��  �          @�z��z=q��(���  �0(�Cb�{�z=q��(��أ��kQ�CE\)                                    BxqM�:  �          @�=q�����ff��  �Q�CcG������(���
=�]�
CJ�                                     BxqM��  �          @��33��33��\)�G�Cv�=�33�)����(��x�Ce�                                    BxqMކ  �          @�G��C�
�Å�n{����Csh��C�
�qG��\�Mz�Cf��                                    BxqM�,  �          @����Fff��=q�1G����HCt�R�Fff������1p�Cl{                                    BxqM��  �          @�p��<���׮������
Cvff�<���������R�!G�Co��                                    BxqN
x  �          @�(��;����
��H���
Cv��;���������)�
Cn��                                    BxqN  �          @����G����
�'
=��33Ct� �G���
=���\�-(�Cl}q                                    BxqN'�  �          @���O\)���HQ���Q�CsB��O\)������R�:�Ci�                                    BxqN6j  �          Az��U�޸R�:�H��p�CtW
�U��(���Q��1  Ck�\                                    BxqNE  �          @�Q��AG���p��Fff���RCt�{�AG�������<��Ck                                      BxqNS�  �          @��
�?\)�ƸR�QG�����CtB��?\)��G���\)�C�Cis3                                    BxqNb\  �          @�p��G
=���H�>{��Q�Cs�
�G
=��G������8�Cj�                                    BxqNq  �          @��L����p��-p�����Cs��L����\)��33�0�Cjp�                                    BxqN�  �          @���>{�����
���RCu�q�>{���
��33�'Q�Cn��                                    BxqN�N  �          @�G��P  �Ϯ�z��~�RCsaH�P  ��=q����ffCl                                      BxqN��  �          @����AG�������C
=Ct�{�AG���Q������\Cn��                                    BxqN��  �          @�\)�?\)�����H� ��Ct+��?\)���z�H��Cn�                                    BxqN�@  �          @�  �,(���=q���\)Cv)�,(�����u��
Q�Cq�                                    BxqN��  �          @����R���
��{�<��Cy�3��R���������Cu
                                    BxqN׌  �          @�z�����33��z��I�C|+�������������Cw�{                                    BxqN�2  �          @��0  ��=q���H���Cx�q�0  ���R���\�	Q�CtO\                                    BxqN��  �          @��
�3�
���Ϳ˅�@z�Cw���3�
��(����H�ffCrk�                                    BxqO~  �          @�\�!G���{�ٙ��O�
Cz\�!G������
=��Ct�)                                    BxqO$  �          @��������z��\�r=qCy��������
����"p�Cs��                                    BxqO �  �          @�33�3�
��33��  ���HCrG��3�
�{������33Cjn                                    BxqO/p  �          @˅�8Q����\>���@>�RC�R�8Q�����(���G�C�H                                    BxqO>  �          @��@���@EA�33C�s3@�����>���@$z�C�aH                                    BxqOL�  �          @���@�\)���@��HB �\C��R@�\)�dz�@/\)A�\)C��                                    BxqO[b  �          @�\)@�>�
=@{�A��@e@����H@j�HA�C��R                                    BxqOj  �          A(�@��
?�33@��BG�A`��@��
�0��@�33BffC��                                    BxqOx�  �          A��@У�?���@�  BG�Ay�@У׿(�@��B�C�W
                                    BxqO�T  �          A�@��
@p�@�(�B �A��@��
����@��B33C���                                    BxqO��  �          @��R@�  @dz�@>�RA�{A�  @�  ?�G�@��B=qA{�
                                    BxqO��  �          A�@�Q�?�=q@<(�A��Ab=q@�Q�>u@\��A�p�?��                                    BxqO�F  �          A Q�@�{@p�@J=qA���A�  @�{?0��@|(�A�@�                                    BxqO��  �          @���@�{@��@C33A�z�A���@�{>�ff@l��A�{@o\)                                    BxqOВ  �          A ��@�R?�@'
=A��
A,  @�R<��
@>{A��>.{                                    BxqO�8  �          A (�@�\?�G�@333A��A:ff@�\<��
@K�A���>�                                    BxqO��  �          A Q�@�R?�{@%A���A%��@�R���
@;�A�G�C���                                    BxqO��  �          @�\)@�?�G�@%A���A{@���G�@8Q�A�G�C��{                                    BxqP*  �          @�ff@�\)?�{@7
=A��AIG�@�\)=�\)@Q�AÅ?
=q                                    BxqP�  �          @�p�@�p�?���@0��A�p�A-�@�p���Q�@E�A��C���                                    BxqP(v  �          @��@�=q@p�@>{A�
=A��R@�=q?
=q@j�HAڣ�@��\                                    BxqP7  �          A Q�@���>�33@ffA���@%@��ͿO\)@�RA���C���                                    BxqPE�  �          @�(�@�?���@A��\A
ff@���@&ffA�=qC���                                    BxqPTh  �          @�(�@�33?�33@ ��A���A{@�33�8Q�@0  A�(�C�L�                                    BxqPc  T          @�33@�{���@��A��C���@�{��\?�(�A,��C�W
                                    BxqPq�  �          A ��@�
==���@3�
A���?:�H@�
=��  @!G�A���C�=q                                    BxqP�Z  �          A@�\�B�\@0  A��HC��@�\��p�@33Ak
=C���                                    BxqP�   �          A z�@�Q��
=?�=qAS�C�o\@�Q쿞�R?�33A!�C�q�                                    BxqP��  �          AG�@񙚾�=q@3�
A�G�C���@񙚿�=q@A�
=C�
                                    BxqP�L  �          A=q@��H��ff@��A�Q�C�5�@��H�"�\?�z�A�C�                                    BxqP��  �          @�
=@��7
=?���AW\)C�!H@��Y��>�  ?�=qC�U�                                    BxqPɘ  �          @�ff@���Tz�?\A0z�C�ff@���i���������C�XR                                    BxqP�>  �          A@����q�?��A��C�ٚ@����~�R������C�8R                                    BxqP��  �          AG�@ٙ�����?\A-��C�� @ٙ����ÿ(����C��=                                    BxqP��  �          @��R@ٙ��u�?��
A1�C�O\@ٙ����H�   �g
=C�|)                                    BxqQ0  �          @�=q@��s�
?�(�AQ�C�#�@��{��B�\��z�C���                                    BxqQ�  �          @�ff@�(��y��?h��@�G�C�>�@�(��vff��=q��=qC�c�                                    BxqQ!|  �          Ap�@�  ����?�p�A�C���@�  ���H�}p���Q�C��f                                    BxqQ0"  �          A z�@�33�{�?�Q�A%��C�R@�33��z�!G���Q�C�s3                                    BxqQ>�  �          A�H@�G��q�?��HAAp�C��@�G���(������33C��=                                    BxqQMn  �          A��@�\�AG�?�z�A�C�!H@�\�U��k��ǮC�%                                    BxqQ\  �          A�@��2�\�(�����HC�H@��Q��
=�Y��C�B�                                    BxqQj�  �          A�H@�
=�:�H��� ��C�N@�
=��\)�*=q����C���                                    BxqQy`  �          A�@�p��1G�������C�q�@�p����\�Z=q���HC��R                                    BxqQ�  �          @�{@�ff�E�
�H�
=C��@�ff��33�Y���ˮC�U�                                    BxqQ��  T          @�z�@�
=�\(������ ��C��q@�
=�{�(Q����RC�<)                                    BxqQ�R  �          @��@�Q������H��z�C���@�Q�k��Q����HC�Ff                                    BxqQ��  �          A   @��
�_\)����xz�C���@��
� ���dz���=qC�ٚ                                    BxqQ  �          A ��@׮�l(������C���@׮���u��RC�n                                    BxqQ�D  �          A@�{�u��\)��\)C�
@�{�Q�������C�,�                                    BxqQ��  �          A Q�@ٙ��C33�9����{C��@ٙ����R������\)C��3                                    BxqQ�  �          A ��@�p��H���(Q�����C���@�p����H�u���C��                                    BxqQ�6  �          A   @�\)�.�R�333��{C�P�@�\)��G��q��ᙚC��                                    BxqR�  �          @�
=@ٙ�����]p���=qC��3@ٙ���=q���
����C��)                                    BxqR�  �          @�33@�z���p  ���C�e@�z�>�=q�����   @�\                                    BxqR)(  �          @���@�p����H�fff���HC�
=@�p�?5�n�R��@��
                                    BxqR7�  �          @���@Ϯ�=p���ff��
C���@Ϯ?�z�������p�AD��                                    BxqRFt  �          A (�@��ÿ�{��{��C��@���?�  ��G��	�A�                                    BxqRU  �          @�\)@��ͿTz���(����C�K�@���?Ǯ��{��AZ�\                                    BxqRc�  �          @���@��Ϳ�=q����  C��\@���?��H��Q��G�A]��                                    BxqRrf  �          @�@�(��|(���Q��p�C�@ @�(���������@(�C�g�                                    BxqR�  �          @�p�@���Tz���{���C�8R@���333��  �5{C�U�                                    BxqR��  �          @�=q@�  �ٙ���ff��C�R@�  ?Y����p��Q�A (�                                    BxqR�X  �          @���@�G���
��  �\)C��=@�G�>aG�������@                                    BxqR��  �          @�@��R�*�H��=q��C�t{@��R=�\)��Q��0  ?(��                                    BxqR��  �          @�Q�@������(��%
=C��{@��>�(�����:�
@��                                    BxqR�J  �          @�(�@�=q�e�xQ�����C�(�@�=q��Q������#��C�8R                                    BxqR��  �          @��\@����H����z�C�� @��>���33�#�?��\                                    BxqR�  �          @�
=@���G���(���C�
@��?+�����+Q�@��                                    BxqR�<  �          @�{@�Q��z���  �"ffC�t{@�Q�?Tz���G��-�A	G�                                    BxqS�  �          @��@��ÿ��H��{�/��C�]q@���?c�
��\)�<{A Q�                                    BxqS�  T          @�\@���Q���(��L�\C��3@�?�(���{�OffA���                                    BxqS".  �          @�{@��\�`���333����C�H�@��\�����p��"z�C�!H                                    BxqS0�  �          @ٙ�@�Q����
�\�Mp�C�q�@�Q��A��`  ����C�g�                                    BxqS?z  �          @��H@�  �\�\�]�C�� @�  �\)����ffC�W
                                    BxqSN   �          @�\)@�33�zᾊ=q�\)C�P�@�33��\)��33�8(�C�{                                    BxqS\�  �          @���@�(���=�?uC�U�@�(��zῇ��33C�Q�                                    BxqSkl  �          @��H@Ӆ�z�?��
AC�XR@Ӆ�!G���=q��C���                                    BxqSz  �          @��H@����?�  ADz�C��3@�����>�ff@mp�C�޸                                    BxqS��  �          @�\@�33�`  ?(��@��\C��@�33�W
=��Q���HC��R                                    BxqS�^  �          @�G�@�{���
�����VffC��{@�{��Q��2�\��z�C��R                                    BxqS�  �          @ۅ@��R��>�  @z�C�W
@��R��p������{C��)                                    BxqS��  �          @��@�  ���?�R@���C��{@�  ��G�����W�C��)                                    BxqS�P  �          @�@�����p�?&ff@�\)C��@������Ϳ��X��C���                                    BxqS��  �          @��
@����Q�?z�H@�{C�b�@������ff�(Q�C���                                    BxqS��  �          @��@������?���A4��C�3@����{�Y����p�C���                                    BxqS�B  �          @�\@��R��G�>�@}p�C�w
@��R�������{�C���                                    BxqS��  �          @�Q�@�����ͼ��
�\)C���@����
=�\)��Q�C��                                    BxqT�  �          @�=q@�\)��\)=�G�?n{C�|)@�\)�����H���RC�ff                                    BxqT4  �          @ᙚ@�����
=�����0��C�u�@����tz��s33��
C��=                                    BxqT)�  �          @�\@�Q����H�s33���C��@�Q����
�aG���C��                                    BxqT8�  �          @�\@����G���
=�S�
C�G�@������J�H���
C��                                    BxqTG&  �          @��
@��R���?�=qA,(�C��@��R��������.�HC���                                    BxqTU�  T          @���@������@
�HA�
=C���@��������H�uC��H                                    BxqTdr  �          @�{@�{���\@{A�
=C���@�{��ff�L�Ϳ�=qC�Ǯ                                    BxqTs  �          @��@�\)��=q?��HA���C���@�\)����c�
����C��{                                    BxqT��  �          @��@�G����H@�A���C��
@�G����\�   ���\C�:�                                    BxqT�d  �          @��
@�  ����@ ��A�\)C���@�  ��G���=q�(�C�W
                                    BxqT�
  �          @�z�@�(���G�@�A��C���@�(���G�������C��H                                    BxqT��  �          @�@|�����@7�A�z�C���@|�����;#�
����C��                                    BxqT�V  �          @�=q@z�H���@333A�G�C��q@z�H���
�aG����
C��                                     BxqT��  �          @�@j�H����@>�RA�G�C�l�@j�H���������\C��)                                    BxqT٢  �          @�G�@�Q���G�@AG�A���C�{@�Q���(�<��
>#�
C��3                                    BxqT�H  �          @�\)@vff����@#33A�\)C�aH@vff��=q�&ff��z�C�1�                                    BxqT��  �          @�(�@����@Q�A��
C��@����(�>.{?��C��                                     BxqU�  �          @��
@�z���(�@h��A���C�*=@�z�����?!G�@�z�C�0�                                    BxqU:  �          @�Q�@�=q���@vffB ��C��R@�=q��z�?xQ�@�{C�L�                                    BxqU"�  �          @�z�@�Q����H?���A�
C�@�Q����
�p�����
C�H                                    BxqU1�  �          @��@�ff��
=?.{@��C��@�ff��=q�ff��Q�C�!H                                    BxqU@,  �          @�33@����=q@{A�ffC���@�����;�Q��8��C�:�                                    BxqUN�  �          @�\)@����33@Q�A��C�ff@����(���G��aG�C��f                                    BxqU]x  �          @�Q�@����H@W�A�z�C��{@���ff?(��@�
=C�Q�                                    BxqUl  �          @�@�  ��\)@�A��C���@�  ���׾���QG�C��                                    BxqUz�  �          @�
=@��\��Q�@��A��RC�>�@��\�������r�\C��f                                    BxqU�j  �          @�ff@�p���G�@�A�33C�o\@�p�����G��ƸRC�k�                                    BxqU�  �          @�@��H���H@33A�Q�C��)@��H���
�����I��C�n                                    BxqU��  �          @��@�(����@=qA�=qC���@�(������G��_\)C��                                    BxqU�\  �          @�\@���z�@W
=A�\)C��{@����R>�@uC��f                                    BxqU�  �          @�
=@�33��33@l��A�  C��)@�33��33?J=q@��HC���                                    BxqUҨ  �          @�@���@G
=AͮC�"�@����\=�G�?k�C�Ф                                    BxqU�N  T          @�z�@�  ����@1�A�
=C�"�@�  ��\)�aG���\C�H�                                    BxqU��  �          @�@��R����@�
A���C�@��R��  �!G����\C��H                                    BxqU��  �          @�ff@fff���H@(��A���C��@fff��������ffC��                                    BxqV@  �          @�(�@[���z�@4z�A�z�C���@[���z�����
C��                                    BxqV�  �          @�(�@e���?��A���C�S3@e������=q��
C��                                    BxqV*�  �          @��
@I����  >�Q�@J=qC�aH@I����33�*�H��(�C��q                                    BxqV92  �          @��H@3�
���
>W
=?�\)C���@3�
���
�7����HC�`                                     BxqVG�  �          @��@��ҏ\��=q��C��H@���\)�i�����C�`                                     BxqVV~  �          @�G�?����z�J=q�ə�C���?����\)��Q��=qC���                                    BxqVe$  �          @�R?�
=��p���\)���C���?�
=���H�r�\����C��f                                    BxqVs�  �          @���?����׾�p��8��C�AH?����Q���33�p�C�!H                                    BxqV�p  �          @�{?����G�=���?5C��q?����\)�u��\)C���                                    BxqV�  �          @�p�?�����Q�>�Q�@.�RC�s3?�����33�fff��z�C�.                                    BxqV��  �          A ��@����  @1G�A�ffC�7
@����z��\�z�HC��q                                    BxqV�b  �          A (�@4z���
=@,(�A�G�C�  @4z���33����33C�|)                                    BxqV�  �          A�@��R���@N{A��
C��H@��R��
=��{��C��f                                    BxqVˮ  �          A�R@����ƸR@#33A��C�h�@������
�������C��f                                    BxqV�T  �          AG�@a���?.{@�(�C�7
@a���p��H����p�C�k�                                    BxqV��  �          Aff@�H��{=�?aG�C�� @�H���
�z=q��Q�C�\                                    BxqV��  �          A\)?����>aG�?��
C���?�����������p�C���                                    BxqWF  �          A
=?�\)�p����W
=C��?�\)��z���z��
G�C���                                    BxqW�  �          A��?E��(������1G�C��f?E���=q��p����C�4{                                    BxqW#�  �          A(�>L���  ��  ��33C��>L����������C��\                                    BxqW28  �          A�
?h����H�8Q쿝p�C�3?h����\��=q��\C���                                    BxqW@�  �          A33?��R�=�G�?8Q�C�5�?��R��p���G���Q�C��                                    BxqWO�  �          A�?�{���L�;�33C���?�{��\)�����Q�C���                                    BxqW^*  T          Aff?�(���R>k�?�ffC���?�(���=q���H��G�C��                                    BxqWl�  �          A�\?��=q?z�@\)C���?����q���ffC�o\                                    BxqW{v  �          A�?�Q��Q�?�@xQ�C��=?�Q����H�w
=���C��H                                    BxqW�  �          AG�?�p��{>#�
?�=qC�H?�p���  ��z����C���                                    BxqW��  �          A�<����R��\)�(�C�)<�������33�&
=C�%                                    BxqW�h  �          A���k���ٙ��<(�C��f�k���  ���R�-{C���                                    BxqW�  �          A�W
=�G�����Q�C��W
=���
����2�C��=                                    BxqWĴ  �          A
=��\����=q�H  C�<)��\������(��0�HC��R                                    BxqW�Z  �          A�H?Tz���
���
�(  C�� ?Tz���{�����(p�C�Ф                                    BxqW�   �          A
=?G��녾���1�C���?G���z���Q��	��C�8R                                    BxqW�  �          A  ?�������{����C��q?�����\)���H��C��                                    BxqW�L  �          A��?�ff�Q��\)�D��C���?�ff��  �\�/ffC�                                    BxqX�  �          A�
?��R��H���K33C�.?��R�������H�0�\C��3                                    BxqX�  T          A
�H?���
=����>ffC���?����
=��\)�-��C��                                     BxqX+>  �          A(�?���{�	���c\)C�o\?����\)��  �6p�C�+�                                    BxqX9�  �          A{?�=q�(��
=q�a��C�q�?�=q�\���H�6�C�.                                    BxqXH�  �          A=q?�����
�����EC�޸?�����(������-C���                                    BxqXW0  �          A  @��(��G��P  C���@�������
=�/\)C�q                                    BxqXe�  �          A33@I���G����H�H��C�^�@I�������=q�  C�n                                    BxqXt|  �          AQ�@j=q��
����5C�\)@j=q��p������ 
=C���                                    BxqX�"  �          A{@hQ��������ƸRC�(�@hQ���ff��=q��\C��=                                    BxqX��  �          Ap�@dz��{���L��C���@dz���Q������\)C��{                                    BxqX�n  �          A�@]p���Y�����\C��=@]p��������
=qC��                                    BxqX�  �          A��@a���ÿJ=q��\)C���@a��ᙚ���H�z�C�J=                                    BxqX��  �          A33@z�H�	G�>�?B�\C�H�@z�H��\��ff��
=C��                                    BxqX�`  �          A��@w
=��
>8Q�?��C��=@w
=��  �����  C��q                                    BxqX�  �          A�\@u���8Q�����C�@u���H�����33C�7
                                    BxqX�  �          A=q@k��Q��=q��HC�^�@k����H��  ���C��{                                    BxqX�R  �          A�@`  �����33���C���@`  ��  ��z����C��
                                    BxqY�  �          Aff@mp��{����c33C�Z�@mp���ff��\)�Q�C��                                    BxqY�  �          A�\@�ff�
=>�  ?��HC���@�ff������ٙ�C���                                    BxqY$D  �          Ap�@hQ����W
=���HC�P�@hQ�����33�	z�C���                                    BxqY2�  �          A(�@N�R�Q��  �)�C��@N�R�Ϯ�����!G�C�8R                                    BxqYA�  �          A��@n�R�  �fff��(�C��=@n�R����p��
=qC�#�                                    BxqYP6  �          A��@l(���׾��H�<(�C�b�@l(�����33��C���                                    BxqY^�  �          A�H@N�R��ÿ��]p�C��@N�R��������C�"�                                    BxqYm�  �          A��@I����
�L�;��
C��@I�����
���R��p�C��\                                    BxqY|(  �          A�
@Vff��׿�(�����C�k�@Vff��G���
=�ffC�!H                                    BxqY��  �          A33@_\)�
=q��  ��RC���@_\)��  ��(��=qC��                                    BxqY�t  �          A�@QG����ٙ��%p�C�E@QG��θR���H� =qC�ff                                    BxqY�  �          A�@�  ��R�ٙ��%��C��3@�  �ƸR���(�C�b�                                    BxqY��  �          A  @e��(��	���QC�k�@e���G��˅�(��C�U�                                    BxqY�f  �          A\)@{��=q��=q�2�\C���@{����
�������C�\)                                    BxqY�  �          A�\@�33�33�Q��QG�C�G�@�33��G����$=qC���                                    BxqY�  �          A�R@��H�
=�p��Y�C�AH@��H�����  �&33C��R                                    BxqY�X  �          A
=@���{�33�IG�C��R@�������\� �RC�aH                                    BxqY��  �          Aff@
=q��R�\)��Q�C�h�@
=q��������_��C�33                                    BxqZ�  �          A�\@9������>�R��G�C���@9����ff�߮�B�C���                                    BxqZJ  �          A@qG��	�����R�?�
C��3@qG���ff��G��#�RC���                                    BxqZ+�  �          A�\@p  �=q�:�H����C�
=@p  �����޸R�8��C��                                    BxqZ:�  �          A�H@y�����C�
����C���@y����(���G��:��C���                                    BxqZI<  �          A\)@tz��Q��S�
���C�e@tz����R��\)�@�HC�                                      BxqZW�  �          AQ�@u�ff�o\)��z�C���@u���
�����I��C��                                    BxqZf�  �          AQ�@\���  �z�H��p�C�Z�@\���������Qp�C���                                    BxqZu.  �          A=q@�z��
=��\�D  C�\@�z������Ǯ�"  C�:�                                    BxqZ��  �          A�@��R�(��<�����C�� @��R��z���p��5G�C��q                                    BxqZ�z  �          AG�@�(��
=�\)�j�HC��R@�(���G���(��)��C���                                    BxqZ�   �          A(�@�
=�
=��z��5G�C���@�
=���H��z��ffC�!H                                    BxqZ��  �          A=q@�{��׿�����HC�b�@�{�˅����(�C�1�                                    BxqZ�l  �          A�
@�=q�
{�z��?�C��\@�=q��p���(���C�>�                                    BxqZ�  !          A   @�Q���
���0��C��H@�Q������
=��(�C���                                    BxqZ۸  T          A Q�@�  �{��\)��=qC��H@�  ��p���\)��\)C�:�                                    BxqZ�^  �          A!�@�������
��G�C�  @����
=������(�C���                                    BxqZ�  �          A   @��
�Q�=#�
>k�C�R@��
��  �����\)C�aH                                    Bxq[�  �          A�@�G���>�z�?�z�C���@�G���Q����
��33C��f                                    Bxq[P  �          A
=@�����?
=q@FffC��=@�����33�x����{C���                                    Bxq[$�  �          A�@��\����?xQ�@�(�C�4{@��\���[�����C���                                    Bxq[3�  �          A��@�
=���?�
=@�z�C�Ф@�
=����J�H��{C��f                                    Bxq[BB  �          A�@�=q��(�?n{@�C�9�@�=q��z��\����Q�C���                                    Bxq[P�  �          AG�@�p���p�?�\)A/�C��@�p���Q��=q�c33C�/\                                    Bxq[_�  �          Ap�@�����p�?�(�@�(�C��@������
�Fff��
=C�)                                    Bxq[n4  �          A Q�@�
=�(����
��\)C�+�@�
=��R��\)��ffC���                                    Bxq[|�  �          A"{@�Q��z�J=q���C�H�@�Q��������Q�C�{                                    Bxq[��  �          A!@�(����#�
�fffC���@�(���R���� {C�Y�                                    Bxq[�&  �          A Q�@��\)����
=C���@����
��G���C�W
                                    Bxq[��  �          A!G�@�ff�(��\)�Mp�C���@�ff��\��{��{C�y�                                    Bxq[�r  �          A ��@�z��Q�u��{C��{@�z���Q���p�����C��q                                    Bxq[�  �          A (�@�33�	��        C�T{@�33��  ��(�����C���                                    Bxq[Ծ  �          A�H@�G���;8Q쿇�C�@ @�G���33��Q���G�C��=                                    Bxq[�d  �          A!p�@�{��R����+�C�� @�{��Q���ff��Q�C�o\                                    Bxq[�
  �          A!�@��H��H�.{�xQ�C��
@��H�����z��ffC�S3                                    Bxq\ �  �          A ��@��p��L������C��@������{�33C��{                                    Bxq\V  �          A!�@���33�!G��dz�C��H@����ff��33� z�C�,�                                    Bxq\�  �          A!�@��\�G�=�Q�?�C�XR@��\������C�k�                                    Bxq\,�  �          A ��@����z�>#�
?h��C���@�����\)��33��C��R                                    Bxq\;H  �          A�@���	�>B�\?���C�
=@���������{C��                                    Bxq\I�  �          A
=@�ff�	��>��@
=C���@�ff��R��G����HC��f                                    Bxq\X�  �          A!�@����p��&ff�j�HC���@��������ff�G�C��f                                    Bxq\g:  �          A!�@�ff�ff�\�	��C�S3@�ff��  ��  ��p�C���                                    Bxq\u�  �          A ��@�  ��
����Tz�C�N@�  ����
=����C��H                                    Bxq\��  �          A�
@�{���u��Q�C��@�{��z������=qC�4{                                    Bxq\�,  �          A z�@��녾\)�J=qC��@���z�������  C�(�                                    Bxq\��  �          A   @�{��������p�C���@�{��Q���Q����HC�k�                                    Bxq\�x  �          A ��@����
=?z�@S�
C��@�����(���ff��\)C��
                                    Bxq\�  �          A!�@����
=?�R@c33C�{@��������p���\)C���                                    Bxq\��  �          A!��@�����?G�@�{C��@�������������G�C�k�                                    Bxq\�j  �          A"{@��
��?!G�@c33C�c�@��
������Q��ƣ�C��)                                    Bxq\�  �          A"=q@�  ��H?0��@z�HC�H@�  ��z���\)�ŅC��                                    Bxq\��  �          A"�H@�{�ff>aG�?�G�C��\@�{���\��  �޸RC�Ǯ                                    Bxq]\  �          Aff@���� ��?˅AC���@�����(��@������C�W
                                    Bxq]  �          A(�@�  ���H@0��A��RC��H@�  ��  ��\)���C�Ф                                    Bxq]%�  �          A�@����@<��A��HC���@����
�����p�C�}q                                    Bxq]4N  �          A�@�  ��@{Aqp�C�|)@�  ��=q��G��+�C��                                    Bxq]B�  �          A  @�z���=q@��AV=qC�˅@�z���33�ff�Lz�C��q                                    Bxq]Q�  �          A33@�p����@�AK\)C��R@�p����
�H�T��C�f                                    Bxq]`@  �          A�\@�z���  @z�AK�C��{@�z���
=�
�H�T��C�H                                    Bxq]n�  �          Ap�@����@�Ab{C�!H@����R���=C���                                    Bxq]}�  �          Az�@�  ��@z�Ag\)C��)@�  ��\)����;�C��q                                    Bxq]�2  �          A��@�  ��@�HAqG�C�ٚ@�  �陚��ff�2ffC��                                     Bxq]��  �          A��@��R��33@(�As�C��@��R�陚���
�1�C�g�                                    Bxq]�~  �          Az�@�p����@
=Al(�C���@�p��陚��\)�:=qC�K�                                    Bxq]�$  �          Aff@�����Q�@�A`(�C��@������H�   �D��C�~�                                    Bxq]��  �          A��@�����p�@�
AeG�C�Ф@�����G����>{C��R                                    Bxq]�p  �          A�@������@�Al  C�� @���陚��{�8Q�C�u�                                    Bxq]�  �          A=q@�������@ffAN�HC��R@�����Q��
=q�Up�C���                                    Bxq]�  �          A{@�����@'
=A�\)C��@������Ϳ���!G�C�\)                                    Bxq^b  �          A��@����
@3�
A�C��@��陚����Q�C���                                    Bxq^  �          A�@�{�Ӆ@S33A��C�Y�@�{���H�J=q��(�C��                                     Bxq^�  �          A��@�z���G�@dz�A��
C�b�@�z������L(�C��
                                    Bxq^-T  �          A��@�  ��  @y��Ař�C�N@�  ���
���
��C���                                    Bxq^;�  �          A�\@�33���
@��A�\)C��)@�33���
>#�
?}p�C�8R                                    Bxq^J�  �          A�@�����(�@��A���C��R@������?   @@  C�                                    Bxq^YF  �          Az�@�\)���@���A��
C�W
@�\)��Q�?�p�A�C�Ф                                    Bxq^g�  �          A@�Q��Ϯ@tz�A���C�e@�Q���׾�z���HC�N                                    Bxq^v�  �          A!@�ff��@|(�A���C��=@�ff���R��
=���C��
                                    Bxq^�8  �          A�@���ҏ\@��A�\)C��@�����\<��
=���C��                                    Bxq^��  �          A=q@�
=��z�?��A;\)C��3@�
=�أ��p��g�C���                                    Bxq^��  �          A��@����(�������RC�Z�@������H�����RC�L�                                    Bxq^�*  �          A�\@�p��Q����Z�\C�@�p������(��%�C���                                    Bxq^��  �          A@\)�
=�B�\���
C���@\)��Q���\)�:z�C��R                                    Bxq^�v  �          AQ�@qG���
�8Q�����C�L�@qG���z����
�8�
C�~�                                    Bxq^�  �          A�@y���33�7���ffC��R@y��������H�7��C�f                                    Bxq^��  �          A�R@%�����  ��{C��R@%��p�����m��C��                                    Bxq^�h  �          A�
@��߮�aG���=qC�s3@������}p���ffC���                                    Bxq_	  �          Aff@�����������1C��@�����\)������C�t{                                    Bxq_�  �          A(�@�  ��(�����7�
C���@�  ������H�p�C�j=                                    Bxq_&Z  �          A�@�33��\)�}p���{C�1�@�33��(����R���C���                                    Bxq_5   �          A��@�\)�ʏ\�Tz����C��@�\)��z������=qC���                                    Bxq_C�  �          A��@�(���=q=L��>���C��f@�(���z��N�R��C�'�                                    Bxq_RL  �          A ��@�{��z�=�Q�?+�C��@�{��\)�N�R��\)C�h�                                    Bxq_`�  �          A
=@��
�љ���G��EC�Q�@��
�����y�����
C�W
                                    Bxq_o�  �          A��@�z��љ��
=q�s33C�c�@�z���
=�~�R��RC���                                    Bxq_~>  �          A(�@�ff����>�{@�C��R@�ff���N�R��p�C���                                    Bxq_��  �          A33@�
=���@��HA��\C�"�@�
=��Q�?333@��C���                                    Bxq_��  �          A�
@�{���R@h��A���C�\@�{�أ׼#�
�#�
C��                                     Bxq_�0  b          A(�@�����@S33A�{C�k�@������þ���'�C��f                                    Bxq_��  	�          A  @������@j=qA�ffC���@�����H���.{C�*=                                    Bxq_�|  T          A�H@�������@G�A�(�C���@�����Q�(��z=qC�@                                     Bxq_�"  �          A��@�����33@J�HA��RC��@�����33�(��|(�C�l�                                    Bxq_��  �          A\)@�  �Å@x��A�Q�C�U�@�  ��    ���
C��
                                    Bxq_�n  
�          Az�@�����(�@tz�AΏ\C��@������=u>���C�u�                                    Bxq`  "          A(�@��H��z�@l(�A�33C��@��H��ff�L�;�Q�C��H                                    Bxq`�  T          A
�H@����ƸR@,��A��C�}q@�����{����ᙚC�s3                                    Bxq``  �          A
=q@�
=���@5�A�33C��@�
=��(��\(����C��                                     Bxq`.  "          A��@����
@8Q�A�C�:�@���p��}p���ffC��                                    Bxq`<�  T          A��@�����
=@B�\A�
=C�O\@������H�h������C�f                                    Bxq`KR  "          Ap�@�{���
@X��A�  C�Ф@�{��\)��
=�0  C��{                                    Bxq`Y�  
�          A33@�����
=@���A�33C��@������?��@}p�C�\)                                    Bxq`h�  T          A�\@��\��(�@.�RA��C��@��\���Y�����C��
                                    Bxq`wD  �          Az�@�z���p�@W�A�ffC���@�z����H�L�Ϳ�=qC���                                    Bxq`��  �          A��@�����@�
=A�p�C�O\@����z�?�R@���C�3                                    Bxq`��  �          A
=@�����@��RA���C�R@������H?&ff@���C��3                                    Bxq`�6  �          A��@�G����H@�=qA�ffC�f@�G���=q?G�@��HC���                                    Bxq`��  T          A=q@��\��{@���B
C��)@��\���?�  A5�C���                                    Bxq`��  
�          A{@�z���
=@��BG�C���@�z���33@ ��A���C��                                     Bxq`�(  "          A�@�{�p  @�{B<{C���@�{���
@[�A�\)C��f                                    Bxq`��  �          A
ff@��XQ�@�ffB@�RC�3@����@h��A���C�G�                                    Bxq`�t  "          A
=q@�ff�Mp�@�z�BIC�R@�ff��Q�@x��A��HC��\                                    Bxq`�  �          A@����<(�@��BN�
C���@�����  @~{A���C���                                    