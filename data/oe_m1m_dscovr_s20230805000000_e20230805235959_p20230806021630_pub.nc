CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230805000000_e20230805235959_p20230806021630_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-08-06T02:16:30.991Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-08-05T00:00:00.000Z   time_coverage_end         2023-08-05T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�/�  J          AG����@�ff�z��N��B�����@�{�*ff�k�B�L�                                    Bx�/�&  
�          AE>���@��#��_\)B�>���@��
�0  �{�\B��)                                    Bx�0�  "          A@  >\)@�Q��ff�^B���>\)@���*�\�{
=B�{                                    Bx�0r  
Z          A=>L��@�  �Q��V�\B���>L��@����$���r�B��                                    Bx�0!  �          AD��?�p�@����G��E�HB��f?�p�@�=q�#��a�B�u�                                    Bx�0/�  �          AHQ�?��HAp��{�;Q�B��?��H@�z��!���V�HB�L�                                    Bx�0>d  
Z          AH(�?��
A	��\)�7�B�\?��
@�{�\)�SG�B�p�                                    Bx�0M
  "          AL��?��A  ����3G�B��?��@����!p��Oz�B��{                                    Bx�0[�  �          AO33?�
=A��{�3ffB��R?�
=@����"�H�OG�B��                                    Bx�0jV  �          AO
=@z�A
=q����9�B��@z�@����%���T�B�8R                                    Bx�0x�  T          AJ�H@33AG�����<�B�(�@33@�33�$z��X{B�#�                                    Bx�0��  �          AQp�@A��{�733B�8R@@�=q�&�\�R�HB��{                                    Bx�0�H  �          AR=q@   A
ff��R�=�B��@   @��
�*�H�Y�B���                                    Bx�0��  "          AM�?�A{���@��B�z�?�@��(���]=qB��{                                    Bx�0��  h          AU�@.�RA
=��H�)�B�Ǯ@.�RA�
� ���D��B�aH                                    Bx�0�:  �          AW�@A�A����
�'�\B���@A�A���"{�C
=B�{                                    Bx�0��  "          AS�
@(Q�A{��R�)�HB���@(Q�A�R� z��EB�L�                                    Bx�0߆  
�          AS�?
=qAp��{�:\)B�33?
=q@����+\)�Wz�B�{                                    Bx�0�,  T          AN�R>�A=q����9��B�>�@�(��&�\�VB��q                                    Bx�0��  �          AI�?=p�A����\�;ffB�#�?=p�@�=q�"�H�X�B�z�                                    Bx�1x  �          AF�\=uA�R���C�B�(�=u@���$���`p�B�                                      Bx�1  
�          AH��?�ffA (��z��FG�B���?�ff@�\)�'��c=qB�33                                    Bx�1(�  
�          AO
=?�33A(���\�G��B��?�33@�p��.{�d�RB��                                    Bx�17j            AH��?��
@�ff�33�E  B�?��
@�p��&=q�a�B��                                    Bx�1F  T          A=�@J=q@�ff�	�;  B�{@J=q@����  �U�B}=q                                    Bx�1T�  |          A:=q?#�
@����\)�B\)B�p�?#�
@�
=�{�_�B��                                    Bx�1c\  
x          A<Q���@����OQ�B�����@�\)�!G��m  B��H                                    Bx�1r  J          A6�\�Q�@�G���\�T��B�Ǯ�Q�@����\)�rz�B��
                                    Bx�1��  T          AG��#�
@�33� ���WG�B���#�
@�\)�.�R�u(�B���                                    Bx�1�N  �          AJ�R��Q�@��
�%��[�B��
��Q�@��R�3\)�y�B�G�                                    Bx�1��  �          A:�H����@ə����bffBɞ�����@�\)�'��B�.                                    Bx�1��  
x          A4  ��p�@θR�(��U�B�{��p�@�
=�z��rz�Bؽq                                    Bx�1�@  �          A���AG�@�����\)���B�\�AG�@�33��z���B�.                                    Bx�1��  �          A8Q�����@�
AQ�BVC����@\(�A=qBIC=q                                    Bx�1،  T          AH����녿�ffA/
=Bt��CB����녽���A0��By
=C4��                                    Bx�1�2  
�          AN�\��\)?J=qA1G�Bm��C-���\)@  A.{Bg
=C ��                                    Bx�1��  T          A=����R?�\)A�RBnffC(����R@�HA�HBep�C�                                    Bx�2~  
�          A8(���
=@���@�Q�B'G�C5���
=@�\)@أ�B�C
W
                                    Bx�2$  
R          AD���EA6�R�ٙ��
=B�G��EA1p��Mp��y�B�#�                                    Bx�2!�            AE��@�G�@�B �B�  ��A(�@��
A�33B�                                    Bx�20p  
�          AK�
���
�
=qA5�B~p�C9s3���
?}p�A5p�B}33C*�                                    Bx�2?  
�          AO
=���H?��
A4��B}=qC)�\���H@"�\A0��Bs�C��                                    Bx�2M�  T          AUG���\)@���A=qB@C�q��\)@��A\)B)��C}q                                    Bx�2\b  "          AU���H@\A/�B\��B�.���H@�  A ��BCQ�B���                                    Bx�2k  T          AR=q��Q�A
{A33BB����Q�Aff@�ffB (�B�ff                                    Bx�2y�  �          AQp���\)@�\A	��B$�Cp���\)A
�R@�B33B�k�                                    Bx�2�T  h          AT������@˅A�BF33C�3����@�z�A�RB-�B���                                    Bx�2��  �          AT(����HA\)@�B	  B��R���HA"=q@�33A�ffB��)                                    Bx�2��  "          AXQ��˅A�R@�  A�p�B���˅A(z�@��
A�Q�B�=q                                    Bx�2�F  "          AXQ���33A�@�=qA�Q�B�����33A%��@�ffA�\)B�\                                    Bx�2��            ATz����
A$��@�{A޸RB�Q����
A0��@�
=A���B���                                    Bx�2ђ  T          AVff��ffA&�R@ÅAٙ�B�ff��ffA2�\@��A��B�#�                                    Bx�2�8  "          AV=q���A-@�(�A��B������A:=q@�=qA���B�=q                                    Bx�2��  T          AV=q��p�A&�\@�\)Aʣ�B�L���p�A1��@�\)A�p�B��                                    Bx�2��  h          AW�
��A�\@��RA�p�C���A!G�@�33A��C 33                                    Bx�3*  T          AW\)���Aff@��RA�C{���A ��@�33A�{C (�                                    Bx�3�  
�          AV�H��Q�A��@�=qA��C���Q�A#33@|(�A��RB��{                                    Bx�3)v  T          AYG��ڏ\A"�H@�G�A�Q�B��R�ڏ\A.=q@��A�ffB�33                                    Bx�38  "          AY�����HA!p�@��A��
B�����HA+�
@x��A�33B��R                                    Bx�3F�  �          AY�����HA#�@�ffA�G�B�u����HA.=q@}p�A���B��                                    Bx�3Uh  �          AY����A-�@�Q�A���B�p����A6�R@[�Ai��B�                                    Bx�3d  T          AZ=q��=qA5G�@���A�ffB�aH��=qA>=q@HQ�AT��B�(�                                    Bx�3r�  r          AX(���RAQ�@��RA�
=C	)��RA��@]p�Ao�C8R                                    Bx�3�Z  �          AN�R�\)A33@�\)A��C�f�\)A@���A��C�                                    Bx�3�   
�          AJ=q�z�A�@�ffA�{C�
�z�A	�@P��Apz�C�3                                    Bx�3��  
�          AL  ��\)A  @��HA�ffCs3��\)A=q@r�\A�ffCu�                                    Bx�3�L  "          AO\)�ÅA*=q@~{A�G�B�3�ÅA1��@��A+�B���                                    Bx�3��  �          AR=q��  A:ff@:�HAM��B� ��  A?33?��H@��B�k�                                    Bx�3ʘ  T          ATz�����AD��@$z�A2�RB�k�����AHz�?E�@VffBٽq                                    Bx�3�>  �          AU���  AB=q@A�B���  AE�>��
?���B�.                                    Bx�3��  T          AS�
��=qA?
=?W
=@j=qB�  ��=qA>�H�c�
�xQ�B�                                    Bx�3��  T          AC
=��Q�A1녿L���q�B�aH��Q�A.=q�=q�7
=B�G�                                    Bx�40  �          ADQ�?޸RA2�R@���A��
B�B�?޸RA:�H@'
=AG33B�                                    Bx�4�  �          AE�uAB�H?��H@�=qB�
=�uAD(����ÿ�ffB���                                    Bx�4"|  �          AF�H?���A>�H@333AQB�
=?���AC\)?��
@�Q�B�aH                                    Bx�41"  "          AIp��&ffA@�׾�33��{B�\)�&ffA=�
=�B̸R                                    Bx�4?�            APz���G�A%���G���\)B�B���G�A���ȣ���\B�B�                                    Bx�4Nn  |          AIp�����@�p���\�W=qC
������@L(��#�
�kffC��                                    Bx�4]  6          AF�\���
@�G��"�H�_�CG����
@\)�*�R�p(�C)                                    Bx�4k�  
�          AAp���p��0  �"�R�p=qCP���p���\)�ff�]33C[J=                                    Bx�4z`  
          AB{��
=@Tz��
=�4�\C�3��
=@�����?��C%E                                    Bx�4�  
�          AE��33@K������C p��33@ ���   �$\)C'\)                                    Bx�4��            AAp���z���G��[�C8L���z���R�=q�UffCD�                                    Bx�4�R  |          A>�R��ff<#�
�33�d33C3�3��ff���R�p��`=qCA�                                    Bx�4��  T          A<(���33?�������a�HC(B���33�.{�{�e  C5�=                                    Bx�4Þ  �          A<  ���\�aG��"�\�p�\C6)���\��  � (��j�
CD�3                                    Bx�4�D  �          A2{��\)@]p��{�kz�CJ=��\)@   ����}�C�                                    Bx�4��  �          A1G�����@Fff�  �n=qCQ�����?�\)��}��C �{                                    Bx�4�  T          A.�H�@��@�{��G���HB�  �@��@����4{B��                                    Bx�4�6  T          A1��Q�AG����H�p�B��Q�@�
=��G��1�HB��H                                    Bx�5�  J          A7�
�q�@��\���gC���q�@Dz��!�aHC�3                                    Bx�5�  �          A9G��Fff@w��'33(�C �3�Fff@{�.�Ru�Ch�                                    Bx�5*(  
�          A:�H�
�H@)���2ff�HCO\�
�H?c�
�6�R��C��                                    Bx�58�  T          A0(��1�@��&�\�CE�1�>�G��)��=qC+�                                    Bx�5Gt  
�          A3
=�p�@�G��33�S��B�Ǯ�p�@�ff��
�tffB��
                                    Bx�5V  
�          A/��9��@�����R�]�B��)�9��@j=q����{p�C ^�                                    Bx�5d�  �          A-��!G�@U�33W
B��!G�?�  �%����C:�                                    Bx�5sf  V          A+\)�'�@HQ��=q(�C�q�'�?���$  ��C�
                                    Bx�5�  H          AG��P��@>�R�(��}{C	���P��?Ǯ��ǮCT{                                    Bx�5��  
�          A\)�aG�@{���qC���aG�?L�����C':�                                    Bx�5�X  �          A�\��ff?(���H33C+���ff�O\)��RC>�3                                    Bx�5��  �          A!G���Q�>����{C.����Q쿆ff�G�ffCA޸                                    Bx�5��  T          A ������\)�{�yz�C:�������(��
�H�o�HCK�                                    Bx�5�J  �          A"�H��\)�������o�RC@O\��\)��R�Q��cffCNp�                                    Bx�5��  
Z          A�R���Ϳ������H�R�RCG������K���p��C�\CQaH                                    Bx�5�  "          A"�\��  ��33����q{CG&f��  �B�\��R�`��CT��                                    Bx�5�<  T          A,z������J=q�
=q�V{CQ��������Q�� Q��A{C[�                                    Bx�6�  
Z          A,�����\�z�H�����C@�q���\�{�Q��tQ�CQ�                                    Bx�6�  
�          A&ff�]p�?���  W
C#T{�]p���������C;�\                                    Bx�6#.  "          A)���?h������C'L���녿.{��
.C=s3                                    Bx�61�  �          A*=q�i��@<���33�~ffC��i��?�\)�����Cn                                    Bx�6@z  T          A9G���\)A�����2
=B͏\��\)@�Q��Q��V�B��)                                    Bx�6O   �          A?���33AQ��z��1(�B��f��33@���z��U�RB�=q                                    Bx�6]�  T          A#33�   @������p�RB�.�   @8����H�C�)                                    Bx�6ll  T          A$���?\)@X���  ���Ch��?\)?�����H=qC�                                    Bx�6{  
�          A ���Z�H@I����H�z  C	k��Z�H?У��G�ǮC��                                    Bx�6��  
�          A#�
�\(�@���\)C��\(�?Y�����u�C&�                                    Bx�6�^  
P          A*�H�)��?�z��"�H�C���)�������$Q��)C:}q                                    Bx�6�  T          A.ff�8Q�@(Q��#
=�C	�f�8Q�?c�
�'�
�fC"�
                                    Bx�6��  
�          A-��Mp�@1G��33�)C:��Mp�?���$Q��{C!��                                    Bx�6�P  "          A,���Tz�@AG������C	Ǯ�Tz�?�=q�"�R8RC:�                                    Bx�6��  �          A z�����@s33��33�P��CxR����@�H�{�e�RC{                                    Bx�6�  "          A �����
@s�
� ���Uz�C�����
@���	G��k{C�f                                    Bx�6�B  T          A#
=����@K��
�\�i�\C^�����?����|�C�R                                    Bx�6��  �          A$������@O\)�z��j33C�����?ٙ��33�}��Cc�                                    Bx�7�  
�          A%����\@J=q�
=�e�\Ch����\?У���w��C k�                                    Bx�74  T          A&�H��p�@Mp��  �d33Cz���p�?���R�vQ�C T{                                    Bx�7*�  
�          A)�����@^�R��\�o
=CL�����?�{��Q�CW
                                    Bx�79�  
�          A)�^�R@����
=�oC޸�^�R@Q���
8RC�H                                    Bx�7H&  
�          A�  ��
=@����n�\�C=q��
=?���v�R�C!��                                    Bx�7V�  T          A�ff��33�&ff���k�C8����33�dz���=q�|G�CM&f                                    Bx�7er  "          A��H�У�?����p��C*� �У׿�p�����#�CD�                                    Bx�7t  �          A������@Mp�����ffC0����=#�
���
��C3��                                    Bx�7��  	�          A�{��p�@����G��C(���p�?�����p���C"p�                                    Bx�7�d  �          A�\)����@��H����W
C�����@{��p���C�                                    Bx�7�
  "          A��
���R@\)��{��Cu����R�z���p�z�C:\)                                    Bx�7��  �          A�\)���R�
�H����aHCH�)���R��(���ff�{C]��                                    Bx�7�V  "          A����{��33����k�CX�)��{����{�jz�Cf�f                                    Bx�7��  	�          A�����z��z����\CEW
��z���Q����
Q�CZc�                                    Bx�7ڢ  T          A��� �����R����mp�CV�� ������s\)�R�Ca�{                                    Bx�7�H  
�          A�33��ff@�z���(���C���ff?E�����C.!H                                    Bx�7��  
�          A�
=���R@�G���  �|�CL����R@ff��p�G�C$��                                    Bx�8�  �          A��H��z�A ����Q��g{B�{��z�@Ӆ��ff��C}q                                    Bx�8:  
�          A������@޸R����{\)CǮ����@U����C��                                    Bx�8#�  	�          A�Q���Q�A
=��33�d�B�ff��Q�@˅���(�C	�=                                    Bx�82�  T          A�(��أ�AJ�\�����Ip�B�B��أ�A�����R�l��B�
=                                    Bx�8A,  T          A�p��	��A\)��
=�\�RCE�	��@��R��Q��x�\CG�                                    Bx�8O�  
�          A����p�A����]G��	�HBԸR��p�A��
�����1��B�                                      Bx�8^x  �          A�����33A�{�J�R��B�8R��33A��H���\�(�
B��                                    Bx�8m  
�          A�����A��H�$����Q�B˔{����A����^ff�z�B�\)                                    Bx�8{�  T          A����]p�A�z��	����BǏ\�]p�A�
=�B�R���Bə�                                    Bx�8�j  T          A�(�?��HAff��33�}33B�8R?��H@�������B��H                                    Bx�8�  �          A��@33A=q��=q���B��@33@�
=��Q�B��                                    Bx�8��  
�          A�\)@'�AMp�����T��B���@'�A����\�~G�B���                                    Bx�8�\  �          A��\@G�A�33�c�
�#�\B�#�@G�AS\)��{�M��B�p�                                    Bx�8�  
�          A��@hQ�A�33�Z�R�=qB�ff@hQ�Ah����p��@G�B��                                    Bx�8Ө  
X          A�ffA Q�AF{�h���)�RBL�A Q�A�H�����I�B-{                                    Bx�8�N  �          A��AlQ�AW\)�1p���Q�B)Q�AlQ�A2{�V�H��B��                                    Bx�8��  T          A��
A[
=A{
=����
=BC�A[
=A\���4z����HB4��                                    Bx�8��  
�          A��@AG�A���C\)��
B���@AG�Aq��u���3��B�aH                                    Bx�9@  "          A���A=qA��\�����  B|A=qA�33�<z���G�Br��                                    Bx�9�  
�          A���@�  A�������33B�Q�@�  A���P����\B��                                    Bx�9+�  �          A�G�A�RA�  �����B�Q�A�RA�  �.�H��=qB�p�                                    Bx�9:2  �          A�\)@s�
A�\)������{B�{@s�
A�=q�<����
=B�33                                    Bx�9H�  "          A�G�@�33A�=q�����z�B���@�33A�\)�#�
��{B�k�                                    Bx�9W~  
�          A��H@�  A����p��.�HB�33@�  A�  � ����\)B�
=                                    Bx�9f$  �          A�(�A�A��R���R�J�HB��{A�A�������S�
B�33                                    Bx�9t�  �          A�33@�=qA����(��l��B��@�=qA��G��̣�B�p�                                    Bx�9�p  
�          A�  ?ǮA�(��4����z�B��R?ǮA����mp��&  B�L�                                    Bx�9�  �          A��@	��A���,z���{B��3@	��A����g��B���                                    Bx�9��  T          A���@��
A����z�����B��=@��
A����8  ��B��q                                    Bx�9�b  �          A��
A(�A����z=q���B��A(�A������33B��                                    Bx�9�  
�          A�33@�p�A��\���.�HB���@�p�A�(��
ff����B�z�                                    Bx�9̮  V          A��A
=qA�
=�����$��B��A
=qA���  ���\B�
=                                    Bx�9�T  
�          A�z�@��RA��R��(��#�B��R@��RA��R�G����B��                                     Bx�9��  "          A�z�@��A�\)��p��X��B���@��A��
��H�ģ�B�                                    Bx�9��  "          A�@��A��R�����V�\B�  @��A�\)�z�����B�8R                                    Bx�:F  T          A��A��A�p��{��33B�
=A��A�Q��˅����B�{                                    Bx�:�  
�          A��R?޸RA��������G�B�
=?޸RA�ff�<����RB���                                    Bx�:$�  T          A�  @�A��H�����=qB�� @�A�  ��G����HB�{                                    Bx�:38  
(          A��R@�A�\)��33��=qB�Q�@�A��
�#
=���HB�#�                                    Bx�:A�  �          A����A�������(�B��ῥ�A����A���RB�                                      Bx�:P�  "          A��
�ٙ�A��R��H��=qB�녿ٙ�A|���:{�B�L�                                    Bx�:_*  �          A��ÿ���A��p���G�B�G�����Az=q�<Q���\B�=q                                    Bx�:m�  �          A��\���
A���33��z�B�G����
Ayp��B=q�G�B�k�                                    Bx�:|v  �          A��\����A��\�����
=B��
����A��R�\���B�B�                                    Bx�:�  
�          A�G����A�G��G�����B��Ϳ��A����Q��B�#�                                    Bx�:��  
�          A��R��HA�Q��IG��{BÅ��HAl�����;(�Bƙ�                                    Bx�:�h  "          A�녽�\)A����(����B�aH��\)A��
�8������B�k�                                    Bx�:�  T          A��?W
=A����\����B�z�?W
=A�Q��J=q�33B���                                    Bx�:Ŵ  
�          A��\?��A����Q���B��R?��A����K���B�W
                                    Bx�:�Z  T          A�ff���A�����\��B�{���A��
�T(��{B�z�                                    Bx�:�   
�          A��\�8Q�A�������(�B���8Q�A�=q�S\)�ffB��                                     Bx�:�  "          A�p���=qA�  �
=��Q�B�� ��=qA�{�\z����B��3                                    Bx�; L  "          A�=q>�p�A���\)��z�B��>�p�A��R�_���HB���                                    Bx�;�  
�          A�p�?�A�33�(����B��
?�A����N�R��RB�p�                                    Bx�;�  
Z          A��?��HA�ff�{����B�\)?��HA����E�(�B��{                                    Bx�;,>  T          A��?Tz�A�33�����
=B�u�?Tz�A�z��=G�� �B���                                    Bx�;:�  �          A�ff?��A�����\B��)?��A�(��Ep��ffB�                                      Bx�;I�  
�          A���?��HA�33��\)����B��?��HA�p��3���p�B�\                                    Bx�;X0  
�          A��?O\)A��
�ʏ\��\)B�u�?O\)A���)G���ffB�                                    Bx�;f�  "          A��\���A�=q��  �w33B����A�z��!G�����B�L�                                    Bx�;u|  "          A����=qA��H���H�w�B�  ��=qA�p�������B���                                    Bx�;�"  �          A�z��7
=A��
�e��
=B�{�7
=A����
��z�B��                                    Bx�;��  "          A��\�7�A�p��p�����BÙ��7�A����{��z�B�ff                                    Bx�;�n  
�          A�ff�vffA�G���G����HB�W
�vffA������
�X��B���                                    Bx�;�  
(          A����7
=A�{���H��ffBĨ��7
=A�G���p���\)B�ff                                    Bx�;��  �          A�ff@;�A��
������\B�33@;�Az{�5���{B��q                                    Bx�;�`  �          A�ff@�33A�
=����ۮB�=q@�33A`���C
=���B��{                                    Bx�;�  
�          A��A33AG��?\)��RBY\)A33Az��f�\�?33B;=q                                    Bx�;�  "          A��@W
=A{
=�����B���@W
=AZ{�-G����B�G�                                    Bx�;�R  
�          A�{?��RA����p�����B�.?��RAz{�2�\�=qB�z�                                    Bx�<�  �          A�ff@�  AhQ��(����B��=@�  A<���XQ��2=qBmp�                                    Bx�<�  
�          A��@�33Am����H����B��H@�33AD(��O��*  Bo�R                                    Bx�<%D  
�          A���@ָRAr�H�����HB�W
@ָRAI���O\)�(��Bw��                                    Bx�<3�  "          A��@�p�Av=q�{���
B�#�@�p�AN=q�Ip��#Bz��                                    Bx�<B�  
�          A�(�@ƸRAz�\������HB��q@ƸRAR�R�I��#(�B��                                    Bx�<Q6  T          A��
@ʏ\A}p���R���B�u�@ʏ\AT���K�
�#\)B�(�                                    Bx�<_�  �          A�  @�(�A��R�  ��  B���@�(�AW��R{�)\)B�G�                                    Bx�<n�  "          A���@���A\)� ����Q�B�
=@���ATQ��Vff�/z�B�Q�                                    Bx�<}(  T          A��@w
=A�p��!�����B�.@w
=AW33�X���033B�                                    Bx�<��  �          A��H@mp�A��������B�.@mp�AdQ��O
=�%{B��H                                    Bx�<�t  
�          A��
@W
=A�{�Q���{B�Q�@W
=AaG��U���*�RB�#�                                    Bx�<�  �          A����RA�����=q���HB�����RA�\)�'����Bĳ3                                    Bx�<��  
Z          A�(��L��A�33��\)��
=B�8R�L��A�����R��B�8R                                    Bx�<�f  �          A���(��A�Q���  ��z�B��(��A�33�"�H��  B�Ǯ                                    Bx�<�  "          A�Q��{A���{����B�
=�{A��R�!������Bĳ3                                    Bx�<�  T          A��R�333A�����G���  B���333A�
=�'
=��  B��                                    Bx�<�X  �          A����G�A�ff���
���B�{�G�A�z��((����B�Q�                                    Bx�= �  
�          A��G�A��\��33���\B�\)�G�A�(��+
=�\)Bɽq                                    Bx�=�  
Z          A�\)��
A�����=q��=qB��
A����*{�B�Q�                                    Bx�=J  �          A�  ��A��
�أ���z�B��f��A���-�Q�B�\                                    Bx�=,�  
�          A����^�RA���=q���B�k��^�RA����/��Q�B��                                    Bx�=;�  �          A���!G�A�G���=q���B��!G�A�(��A��G�B�L�                                    Bx�=J<  "          A������A�G�����z�B�ff����A�{�Rff��HB�p�                                    Bx�=X�  
�          A���Y��A���
�H����B�33�Y��A����N{�Q�B�
=                                    Bx�=g�  �          A��Ϳ333A�  ���£�B�B��333A����K33�Q�B��                                    Bx�=v.  �          A�G���RA�p��z���\)B��Ϳ�RA�Q��O���HB�k�                                    Bx�=��  �          A�(��B�\A�{�����B�Ǯ�B�\Ax(��Vff�#(�B���                                    Bx�=�z  	�          A�33>B�\A�z�������B���>B�\Ay��Q�   B���                                    Bx�=�   
�          A��@  A��\�z����B��@  Av�H�M��33B��\                                    Bx�=��  �          A�{�(�A��
=��  B���(�Avff�Hz��\)B�33                                    Bx�=�l  �          A�Q쿺�HA�G�������HB�(����HAy�G�
�=qB���                                    Bx�=�  �          A�=q��A��R�����\)B��R��An�H�L���!\)B�#�                                    Bx�=ܸ  T          A�Q��=qA���=q��=qB�=q�=qAc��K��%B�=q                                    Bx�=�^  T          A�z��-p�A�=q�  ��\)B���-p�AW��R�H�/=qB�Ǯ                                    Bx�=�  T          A��R�S33Azff��H��B����S33AO��G�
�+��BЀ                                     Bx�>�  T          A�G����
A�  �ff�ٮB�� ���
A[
=�B{�%=qB���                                    Bx�>P  "          A�p��/\)A�������\B�Q��/\)AU���K�
�,G�B�(�                                    Bx�>%�  
�          A������A{�
�
�\��{Bъ=����AQ��Dz��&�B��                                    Bx�>4�  �          A�\)�A�A�(�� Q����HB�{�A�A\z��<����\B�Ǯ                                    Bx�>CB  �          A���0��A��������{B��R�0��Ai��,Q��ffB�k�                                    Bx�>Q�  �          A���8Q�A�33�陚���B�8R�8Q�Ad���3\)�Q�B�k�                                    Bx�>`�  
�          A��Ϳ}p�A����Q����
B��}p�Ak
=����\B��                                    Bx�>o4  
�          A�33�E�A���������Q�B�8R�E�Amp���H�  B��                                    Bx�>}�  
�          A��׾�A�G��A����B�\��A�����p����HB�Q�                                    Bx�>��  T          A�Q쾨��A�{����Q�B�G�����Ab{�4z��\)B���                                    Bx�>�&  T          A��׿���Ac
=�z����B�{����A733�E��;��B�=q                                    Bx�>��  �          A��\���
AJ�R�R=q�7��B�� ���
A
=�~�R�qB�B�                                    Bx�>�r  �          A�33���RAr�\�p�����B����RAH���>�\�,��B�                                    Bx�>�  �          A��R��A�ff�У���G�B��H��Aq��+33��B�                                    Bx�>վ  "          A���
=AI���n=q�E�B��
=A=q���H�\)B�=q                                    Bx�>�d  �          A�(����A=��z�R�Q�\B�k����@���\)k�B؅                                    Bx�>�
  �          A�
=��33AhQ��^{�.z�B�\��33A'�����i�\B��                                    Bx�?�  T          A�33��A����@(��=qB�.��AI��|Q��MB�=q                                    Bx�?V  T          A�G���p�A����/�
��B�����p�AS��nff�@Q�B�\                                    Bx�?�  T          A������A�ff�%G���B�G�����A\���f{�3p�B֔{                                    Bx�?-�  �          A�  �o\)A���&ff��G�B��H�o\)A[
=�f�H�5��Bҏ\                                    Bx�?<H  "          A�p��.{Av�\�;33�33B��.{A=p��t���N33B��)                                    Bx�?J�  
�          A����0  A{��B�H�ffB��
�0  A@(��}�P��B���                                    Bx�?Y�  �          A�{�8Q�A��H�9����RB����8Q�AL(��w��G{B�p�                                    Bx�?h:  
�          A���!�A���&�\��  B����!�A]��hz��7�HB�Ǯ                                    Bx�?v�  T          A�33� ��A���5���	G�B�W
� ��AN=q�t(��E=qB�                                    Bx�?��  
�          A���33A����(Q���
=B���33A\  �jff�9��B���                                    Bx�?�,  �          A�Q��\)A��\�(������B�  ��\)AW
=�i���<��B��q                                    Bx�?��  
�          A�  �\)AV�R�S33�2
=B��Ϳ\)A33��=q�oG�B���                                    Bx�?�x  T          A�����A��H�  ��RB�� ��Aj=q�b=q�/��B�#�                                    Bx�?�  �          A�z�У�A�Q���R��G�B�z�У�Aip��`���.��B�Ǯ                                    Bx�?��  
�          A��\��A���'33��\)B�W
��AX���ip��:��B��                                    Bx�?�j  T          A���ffA�  �$Q���p�B�{�ffAZ=q�f�H�9�Bŀ                                     Bx�?�  T          A�zῈ��A�p��\)��ffB�8R����A[\)�^=q�5(�B��f                                    Bx�?��  �          A����A����33��p�B�{��A`z��[��0��B��\                                    Bx�@	\  T          A�33�\)A���&�R���HB��f�\)AU��h���<
=B�(�                                    Bx�@  T          A��\��A����#33���B�W
��AW��e��9��B��f                                    Bx�@&�  "          A��\�L��A��R�3��33B���L��AO
=�t���F�HB�\                                    Bx�@5N  �          A��R�#�
A�{�H���\)B�#׾#�
A@(���\)�WG�B��                                    Bx�@C�  T          A����\A�ff�C�
�G�B��R�\AA�����T\)B���                                    Bx�@R�  T          A�G���
=A�Q��33��\)B�G���
=AZ�H�c��7�\B�                                    Bx�@a@  T          A�z��(�A�z��#\)��z�B��H�(�AV{�f�\�:��Bƣ�                                    Bx�@o�  �          A�\)��\)A�=q�-���  B����\)AK
=�n=q�E
=B�Ǯ                                    Bx�@~�  T          A����\Av�\�:�H��HB�#׿�\A9�w\)�S{B�aH                                    Bx�@�2  �          A��z�HAip��Mp��%33B���z�HA'�
���H�d�HB��                                    Bx�@��  
�          A������APQ��a��<G�B��q���A
{���H�{�
Bţ�                                    Bx�@�~  "          A��þ�A:{�l���OQ�B��\��@�\�����qB�z�                                    Bx�@�$  T          A�{����A?33�g
=�H��B�(�����@�\)��p�B�B��H                                    Bx�@��  �          A���)��An�H�J=q�33B���)��A-G���Q��^p�B�z�                                    Bx�@�p  T          A���^{A~ff�;��
=B̞��^{A@(��{
=�M�
B�=q                                    Bx�@�  %          A����Z�HA���!���B����Z�HAW
=�g
=�8��BЅ                                    Bx�@�  
W          A��
��  A�Q��\)�ŮB��׿�  Alz��M��#��B��{                                    Bx�Ab  
�          A�����  A�  �=q��=qB΀ ��  AX���d(��533B��                                    Bx�A  
�          A�����A���G���B�  ���AZff�W��*�
B�p�                                    Bx�A�  T          A����  A���z�����B��H��  AP���`���2�B��                                    Bx�A.T  T          A����}p�A�\)�  ��\)B�\)�}p�AX���^ff�2G�BԞ�                                    Bx�A<�  T          A�����RAs��<����BԞ����RA4(��zff�Q{B�.                                    Bx�AK�  
Z          A�G���33Ao\)�3�
��B�#���33A2ff�pQ��L��B�Ǯ                                    Bx�AZF  T          A�
=��=qA{\)�(���ffB�p���=qADz��]G��:�B���                                    Bx�Ah�  	�          A�p��
�HAi���>ff�z�B��H�
�HA)���y��\�
B��                                    Bx�Aw�  
Z          A����G�A[\)�[��4�B��3��G�A33�����v��B��                                     Bx�A�8  
�          A��R�EAr�R�:{���B�  �EA333�w�
�T�\B���                                    Bx�A��  "          A�
=��{A|����������B�aH��{AM���A���B�k�                                    Bx�A��  "          A���p��A~ff�&�R��BΙ��p��AC��h���B�HB�33                                    Bx�A�*  
�          A�=q?�33Az=q�2=q�\)B�Ǯ?�33A<(��r�H�P33B�k�                                    Bx�A��  
�          A�{��=qA{33�����\)B�{��=qAEG��W33�3G�B䞸                                    Bx�A�v  �          A�=q��A�\)��33��33B�p���AV�\�7�
�z�B���                                    Bx�A�  
�          A�����p�A�����Ǚ�B���p�AP���G
=�!Q�B�.                                    Bx�A��  T          A�����z�A������H��ffB�k���z�A{\)�.ff��\Bۀ                                     Bx�A�h  �          A�(�����A��H���R�b�RBΣ�����A����%G���G�B��                                    Bx�B
  
(          A�����HA�(���z���p�B�(����HA~�\�4�����B��                                    Bx�B�  �          A�{���A��R����L(�B�
=���Ap�����܏\B�aH                                    Bx�B'Z  T          A�{��RA���?\)�	p�B�{��RAy�������HB�                                    Bx�B6   
]          A�������A��R���\�_�B�p�����Ay��{���B�33                                    Bx�BD�  
�          A�Q���\)A�{�[�� z�B�k���\)A�\)�  ���B���                                    Bx�BSL  W          A�(����A�{�L�;��B��H���A��H���\��33B��                                    Bx�Ba�  Q          A�Q���ffA���33����B�W
��ffA����p���G�B�=q                                    Bx�Bp�  �          A�����z�A�=q����u�B�p���z�A������
���B��                                    Bx�B>  "          A�G��r�\A�  >.{>��B�z��r�\A���������B�k�                                    Bx�B��  �          A�p��O\)A�G�@'
=@�
=BƳ3�O\)A�Q��X���33B���                                    Bx�B��  �          A�=q�6ffA��@�p�A�{B��f�6ffA�{?�R?�  B���                                    Bx�B�0  �          A���.{A�G�@�A��B�aH�.{A���?n{@(Q�B�k�                                    Bx�B��  "          A����1G�A�33?�\)@G
=BÀ �1G�A�Q���=q�d  B���                                    Bx�B�|  
�          A�  ���A�\)@eA\)B��ÿ��A�ff�/\)��\B��                                    Bx�B�"  �          A�ff�(�A�ff?�?�Q�B��(�A����G����\B��                                    Bx�B��  
�          A��H?
=A��R@s�
A)B��3?
=A�Q��!����B��R                                    Bx�B�n  "          A���?p��A��\?�{@p  B���?p��A�  �����^�RB��                                    Bx�C  
(          A�33@�ffA�@H��A	�B�u�@�ffA��J�H�
�\B�u�                                    Bx�C�  "          A��
?��A��@���A0��B���?��A������p�B�                                    Bx�C `  
�          A�����A�=#�
=���B��3����A�  ��\)����B�Ǯ                                    Bx�C/  T          A����
=A���?�\)@I��B�W
�
=A��
��p��j�HB�p�                                    Bx�C=�  �          A��H��z�A���?�ff@j=qB��Ϳ�z�A�����
=�bffB��                                    Bx�CLR  "          A�33��\)A��\@���As\)B��\��\)A�  �n{�)��B�=q                                    Bx�CZ�  �          A�33��=qA�\)@�An�HB��{��=qA��\����B�\B��                                    Bx�Ci�  
�          A��R��A�ff?:�H@B����A�Q���  �~�RB��3                                    Bx�CxD  �          A�  ��Q�A�ff@{@��HB��
��Q�A��R�r�\�.�HB��                                    Bx�C��  �          A�p���(�A��R?c�
@!G�B�p���(�A�����z��x��B��=                                    Bx�C��  "          A�\)��Q�A��@��@߮B��
��Q�A���xQ��1B��                                    Bx�C�6  �          A��R���A�{?�=q@uB�33���A�������e�B�L�                                    Bx�C��  T          A�\)���A��R�Q���B�����A��������=qB��)                                    Bx�C��  
�          A�z��(�A��R�&ff��B�uþ�(�A�(��(����HB��3                                    Bx�C�(  
�          A���\A�Q�?�\)@\��B�(���\A����Q��mG�B�B�                                    Bx�C��  �          A�녾�{A�
=@Q�@�p�B�B���{A�{�G��&�\B�B�                                    Bx�C�t            A����~�RAx��@�(�A���BиR�~�RA�{?�z�@w
=Bγ3                                    Bx�C�  �          A�  �{AB�RA1��B'��Bȳ3�{Ap��@�{A��
B���                                    Bx�D
�  �          A���O\)A/
=AB=qB?��B�� �O\)Ac�A ��A��B��=                                    Bx�Df  �          A�p��\(�A0��A9B9Q�B�녿\(�Ab�\@�  A��B���                                    Bx�D(            A�녿Y��A@z�A,Q�B'{B���Y��AmG�@�z�A�=qB��{                                    Bx�D6�  
�          A����A{A>�HBIp�B�z��AR�RA�\A�Q�B��                                    Bx�DEX  "          A�p��{A^�RAz�A���B���{A}G�@Y��A?�
BŽq                                    Bx�DS�  
�          A�����
=AS
=A��B�HB�u���
=Au�@�(�AiG�B�Ǯ                                    Bx�Db�  �          A�\)��
=A:�\A�B�HB�G���
=Ac\)@��A���B��H                                    Bx�DqJ  "          A�ff���A��A2�RB*
=C����AC\)@��A���B�(�                                    Bx�D�  "          A�G����\A��AC�BHz�B�8R���\A@z�A�B�B�q                                    Bx�D��  h          A����AffAK33BO�
B�Ǯ��A<��A=qB�B�ff                                    Bx�D�<  
�          A���{�@�\)A[�Bip�B�\)�{�A8(�A(��B#�HB٨�                                    Bx�D��  T          A����[�@�{A_�
Bk��B���[�A<��A,  B$��B�k�                                    Bx�D��  �          A�
=�L(�@���Ae�Bn=qB����L(�A@(�A0Q�B&G�B�Ǯ                                    Bx�D�.  "          A�ff�fff@�
=Ab�HBl
=B�
=�fffA>�\A.ffB$�Bը�                                    Bx�D��  �          A~{���\A ��AN�\BZ�
B����\A<��Ap�B=qB�8R                                    Bx�D�z  
�          A�  �Q�@�z�Ae��B�B�B�G��Q�A'�A8��B9�HB�                                    Bx�D�   �          A}��ff@��Ao33B��B�\�ffA��AI��BS
=B�ff                                    Bx�E�  
�          A|���<��@��Ao\)B��HB����<��A�
AL��BX�B�L�                                    Bx�El  "          AxQ��k�@s�
Ai�B�C�R�k�A{AH(�BW33B�aH                                    Bx�E!  �          Au���Q�@�
Af�RB�\Cz���Q�@�AO
=BhG�B�                                    Bx�E/�  
�          As33��G�@���A^ffB��)C
� ��G�A�A<��BL\)B��                                    Bx�E>^  
+          Atz��8��@���Aap�B�B�B����8��AffA:=qBGG�B�8R                                    Bx�EM  �          Aup�����@W
=Ac33B�\C������@��
ADz�BW��B�\                                    Bx�E[�  T          At�����@FffA\��B��CT{���@�  A@(�BO�
B��                                    Bx�EjP  �          Ar�R����?���AW�B|�HC'������@�33AC�BXp�C
��                                    Bx�Ex�  �          Aq��?��RAb=qB�u�C%����@��
AN�RBmG�C�                                    Bx�E��  T          At���(�ÿ�
=Aj=qB��
CPu��(��@a�Adz�B��3B��                                     Bx�E�B  T          Aw�����@b�\AUBr\)C�3����@���A6�\B@G�Cp�                                    Bx�E��  
(          AxQ���
=@���AH��BX��Ck���
=A��A!B"��C��                                    Bx�E��  �          AtQ��Ӆ@?\)AVffBv=qC�3�Ӆ@陚A9�BF��C(�                                    Bx�E�4  "          Apz���{>�=qA_�
B�u�C133��{@�ffAQp�Br�C	��                                    Bx�E��  �          Al����=q?�=qAX��B�ffC'���=q@��AD��B`��C\                                    Bx�E߀  
�          As
=��  @;�A`(�B�{C�f��  @�\)AC\)BVp�B��                                    Bx�E�&  �          Ay����R<�At��B�.C3E���R@�{Af{B�B��                                    Bx�E��  T          Av�R�8�ÿ�AqG�B���C>Y��8��@��Ae�B��\B�(�                                    Bx�Fr  
�          Aq���Mp�@33AfffB�ffCz��Mp�@ٙ�AM��Bn��B�z�                                    Bx�F  "          Ao���G�@���A733BK�\C�)��G�A�A��B�\C 
=                                    Bx�F(�            Ag\)��\A+
=@#�
A&�\C����\A.�R��p���  C8R                                    Bx�F7d  �          Ap  ���A/�@��\A\)C�f���A;\)=�Q�>�33B��3                                    Bx�FF
  �          Anff��ff@�p�A#
=B*�RC0���ffA.�H@��
Aٙ�B�\)                                    Bx�FT�  @          AuG�� ��@���A8Q�BC�CG�� ��A��AQ�B��C
=                                    Bx�FcV  
Z          AuG�� ��@��AD��BUz�C�
� ��A(�A!�B$p�CE                                    Bx�Fq�  "          Ar=q�?�\)AAp�BT�RC({�@��A+\)B4�RCL�                                    Bx�F��  T          Ap(���G�?�(�AR=qB{{C%E��G�@�(�A;�
BR��C�{                                    Bx�F�H  �          Ar�\�љ�?�\)AX  B}�C&)�љ�@�p�AABU�C�R                                    Bx�F��  
Z          Aq����\?���A^{B�u�C)!H���\@���AJffBf33C�                                    Bx�F��  T          Aj�\���H>��
A^=qB�\)C0����H@�33AN�RBx(�C�R                                    Bx�F�:  
�          Ae�Ǯ�n{AN{B��C<z��Ǯ@eAF�\Boz�C
                                    Bx�F��  �          Al�����\)AN=qBr\)C8ff��@|��ADz�B`p�Cff                                    Bx�F؆  �          Aup���33�ǮA^�HB��HC7���33@�
=AS
=Bn  C�
                                    Bx�F�,  �          Aw����>�z�Ad��B�p�C1#����@�Q�AT��Bm��C

                                    Bx�F��  �          Aw\)����?�
=Aa�B�#�C)#�����@���AM�B`\)C�{                                    Bx�Gx  "          Aw
=��?���Ac�B�C$�=��@�{ALz�B`ffCc�                                    Bx�G  	�          Avff���
@
=Ab�RB�\Cn���
@�{AH��BZ�C�                                    Bx�G!�  
Z          Atz�����@VffAM�Bg  C�H����@���A.{B6z�C
=                                    Bx�G0j  
�          As33��(�@�\)AO\)Bk  CxR��(�AQ�A*ffB2z�B��
                                    Bx�G?  T          Ar�\���@tz�AT��Bu��C(����A(�A1��B<�\B�k�                                    Bx�GM�  
�          As����@@��AYB~Q�C�=���@��
A:�HBHC z�                                    Bx�G\\  �          At�����
@A�AW33Bv=qCh����
@�\A8(�BCG�C!H                                    Bx�Gk  �          Ap����
=@�\AJ�\Bf��C%5���
=@�(�A1�B?�HCk�                                    Bx�Gy�  T          Ao\)���;��HAO
=Bt{C7�����@��
ADz�B`\)C                                    Bx�G�N  �          Ar�\��=q��Q�AS\)Bt{C4����=q@�33AF=qB\\)C�{                                    Bx�G��  
�          As
=�(�?k�A@��BTQ�C.+��(�@�z�A/
=B:{CG�                                    Bx�G��  T          Anff�
=@�z�A&ffB2ffC�R�
=A�\@�A�(�C�{                                    Bx�G�@  T          Al(��p���VffAO�B���C]��p��?�  AU�B�33C33                                    Bx�G��  �          A_
=�q��@��AO33B�{CZ���q�?���AR�HB��CY�                                    Bx�Gь  T          AZ�R��33�L(�AQ�B�ffCo&f��33?�(�AVffB�Q�C	��                                    Bx�G�2  T          A[���33��{AYB��CA����33@��AMG�B�8RB��                                    Bx�G��  �          AU���
=@Q�A��BY�HC0���
=@Ϯ@�\)B$�C�)                                    Bx�G�~  "          AUG�@W
=�~{AC�B��)C�  @W
=?0��AMG�B�
=A8��                                    Bx�H$  h          AH����\)A�
�z��6ffC ���\)A�\�w���  C��                                    Bx�H�  |          AMG�����A+
=?O\)@n{B�������A$(��Fff�e�B��f                                    Bx�H)p  
�          AP����Q�A8z�?aG�@{�B�{��Q�A1��Vff�r=qB��                                    Bx�H8  
�          ATQ��9��AJ�\?ٙ�@�{B��9��AF�R�@  �S�
B�B�                                    Bx�HF�  �          AO��VffAE��?�Q�@���B�Q��VffAB{�9���O�
B��)                                    Bx�HUb  
�          AS\)�-p�A4  @��A�(�B�{�-p�AK�?�33@�B�\                                    Bx�Hd  T          AQ��$z�A33@�p�B��B����$z�ABff@qG�A���B��H                                    Bx�Hr�  "          AHQ����A��@���BB������A9��@Z=qA�p�B��                                    Bx�H�T  "          AA��A,z�@g
=A���B�aH��A5�����5�B�.                                    Bx�H��  
�          AC
=��z�A:�R�Y�����B�8R��z�A(��������ffB��                                    Bx�H��  "          ABff��=qA=�?���A��B���=qA;��'��Hz�B��                                    Bx�H�F  
�          AG����AFff?aG�@�=qB�aH���A=�n{��\)B�z�                                    Bx�H��  T          AE녿\)AB{�5�W
=B�G��\)A0Q����
���HB���                                    Bx�Hʒ  
�          A@��>�A7��N{�z�HB�aH>�A���R��B�ff                                    Bx�H�8  
Z          AC\)�5�A:�\?E�@j�HB�L��5�A2{�e���ffBЏ\                                    Bx�H��  �          AD(��Y��A;33>���?���B�p��Y��A0  ��  ��p�B�aH                                    Bx�H��  �          AD�����\A2ff?�(�A��B������\A0  �!G��?\)B�33                                    Bx�I*  �          AD�����
A,z�?��@�=qB������
A)��p��<��B�u�                                    Bx�I�  
�          AB=q�aG�A7�?���@�  B���aG�A1��N�R�xQ�B�B�                                    Bx�I"v            A?�
�}p�A.=q?�\)@�  B��}p�A(z��@  �o
=B�G�                                    Bx�I1            A>�R����A(Q��-p��X  B������AQ���p���B噚                                    Bx�I?�  	�          A>{���HA{��=q����B�����H@��H����.�RC�                                    Bx�INh            A=��Tz�A,  �^{��G�B�Q��Tz�A
�H��\)�z�B��
                                    Bx�I]  
�          A;
=�dz�A)p�?�(�A(�B�=q�dz�A'��ff�@(�B٣�                                    Bx�Ik�            AB=q���A(�@��HB�Bγ3���A4z�@#�
AHz�Bʞ�                                    Bx�IzZ  T          ALz��AA�@�33A�p�B�G���ALQ�=p��S33B�B�                                    Bx�I�   
Z          AI��=��
AA�@S33As�B�G�=��
AG���p��أ�B�L�                                    Bx�I��  �          AHz�?��AEG�@ffA�\B�aH?��AC\)�+��E��B�W
                                    Bx�I�L  �          AH��?:�HAEG�@ffA{B�33?:�HAC\)�,(��FffB�#�                                    Bx�I��  
�          AH��?�AE�?�p�@���B��\?�A?��QG��r{B��                                    Bx�IØ  
�          AHz�aG�A333@��AÙ�B�  �aG�AD��?�\@�B�33                                    Bx�I�>  "          AI녿\)A+33@��
A��
B��ÿ\)AFff@ffAp�B�#�                                    Bx�I��  T          AJ{���A&�\@�\)BB�𤿋�ADQ�@!�A9G�B�#�                                    Bx�I�  "          AJ=q���A�A�RB'�\B��f���A<��@���A�=qB���                                    Bx�I�0  T          AM����A%�@�B(�Bŀ ����AF�\@<(�AS�B¨�                                    Bx�J�  
Z          AO33�B�\A8��@��HA�Q�B��{�B�\AN=q?��\@���B�Ǯ                                    Bx�J|  T          AK33<��
A;33@�\)A�Q�B�Ǯ<��
AK�>#�
?5B���                                    Bx�J*"  T          ADQ�?�G�A0��@���A£�B��?�G�AA�>�G�@B��                                    Bx�J8�  
(          AC33?L��A$  @ϮB{B��?L��A?33@ffA�HB�aH                                    Bx�JGn            A7
=��ff@�p�AffB>=qB��R��ffA$��@��HA�33B�                                    Bx�JV  
          A;���{@�A  BI�HB�B���{Aff@��RA�B�(�                                    Bx�Jd�  T          A8�Ϳٙ���
=A.ffB�  C\(��ٙ�@6ffA)�B�Q�B�                                    Bx�Js`  "          A<(�?!G�@hQ�A�B�� B�(�?!G�@�p�@��\B=�\B��f                                    Bx�J�  
�          AB{��z����A
=BP�C�s3��z��EA8  B��CyaH                                    Bx�J��  T          A7�?����p�A�
B>G�C��q?���k�A'33B�  C�                                    Bx�J�R  �          A6�\@�G����
@��HB"��C�#�@�G��aG�A�RBcp�C��                                    Bx�J��  �          A4(�@���G�@��BCG�C���@��@��Az�B��RC��)                                    Bx�J��  
�          A4z�@Q���  @�
=B9
=C�R@Q��g
=A"ffB��
C���                                    Bx�J�D  T          A7\)@����
=@���B+��C��=@����33A$  B�
=C��\                                    Bx�J��  "          A6=q?p�����HA�HB7�C�k�?p����=qA)�B��
C��H                                    Bx�J�  �          A5G�@
=q��@��Bp�C�@ @
=q���A�Br��C��
                                    Bx�J�6  
�          A4��@E�ff@�z�A��
C���@E����@���B,��C�*=                                    Bx�K�  
�          A/�?�p���\@�\B3�
C���?�p��s�
Ap�B���C��)                                    Bx�K�  T          A(  �\)�:=qA\)B��fC��\�\)?�Q�A$��B�p�B�.                                    Bx�K#(  T          A
=�=p��0��A��B���C_  �=p�@EAG�B�u�B��H                                    Bx�K1�  
�          A'�
�\@|(�A�\B�W
B��\@�G�@�(�B133B��                                    Bx�K@t  T          A(  =�G�@�ffA  Bc�RB���=�G�A
�\@��HB�B��\                                    Bx�KO  
n          A:�\?���@��A�B7�B���?���A'\)@�Q�A��B��H                                    Bx�K]�  
�          A?
=?aG�A&=q@�
=A�p�B�G�?aG�A;33?xQ�@��B�\)                                    Bx�Klf  
�          AA�@��A)G�@���A�ffB�u�@��A=��?Tz�@z�HB��                                    Bx�K{  "          AE��?�=qA(�Ap�B&�
B�aH?�=qA7�@z=qA�{B�G�                                    Bx�K��  
(          AQ�?��RAD(�@��A�=qB��?��RAN�\�z�H����B�p�                                    Bx�K�X  
�          AUp�@   AIp�@B�\AT��B��{@   ALQ��	���ffB��H                                    Bx�K��  
�          AS33?�33AN�\@ffA�HB���?�33AK
=�I���]��B�u�                                    Bx�K��  T          AXQ�@_\)AIp�@:�HAI��B�  @_\)AK��G����B�L�                                    Bx�K�J  "          Alz�?��HAb=q@z=qAuB�8R?��HAh�׿��R��  B��=                                    Bx�K��  T          As�?�(�Ag�@��A�33B�p�?�(�Apz���H��ffB��                                    Bx�K�  �          Ao�?�\)Adz�@���A��B�
=?�\)Al�Ϳ��H��=qB�\)                                    Bx�K�<  T          At��?�G�Ab�H@�(�A��RB��)?�G�At  ��ff��B�p�                                    Bx�K��  �          At��?aG�A[�@ӅA�G�B���?aG�As�?!G�@
=B�aH                                    Bx�L�  "          Aa��<#�
AX��@c33Ak
=B��f<#�
A]���
�HB��f                                    Bx�L.  
�          A\��=#�
AV�\@G�AQp�B���=#�
AX���p��$��B���                                    Bx�L*�  
W          An�H?p��A`��@���A�\)B�aH?p��Amp�������RB�Ǯ                                    Bx�L9z  �          A`��?�{AZ=q@P  AU��B���?�{A\�������
B�Ǯ                                    Bx�LH   �          A\��?0��AUG�@_\)AjffB��?0��AY��
=���B�8R                                    Bx�LV�  
�          A[
=?���AS\)@Z=qAg
=B�?���AW�����  B��                                    Bx�Lel  
�          AV�H��\)AD��@��\A�ffB�����\)AU������{B��R                                    Bx�Lt  �          AW33��p�AHz�@�A�G�B�.��p�AU�8Q��C33B��                                     Bx�L��  "          A\z�@  AO�
@�Q�A�
=B��׿@  A[\)�����p�B�L�                                    Bx�L�^  �          A_���\)AQG�@���A��
B��쿏\)A^ff�^�R�c�
B�=q                                    Bx�L�  �          A\�ÿ(�AP(�@��HA�G�B�\)�(�A\  ��G����RB�\                                    Bx�L��  "          AX�׿��\AF�R@�  A��HB�aH���\AW��.{�=p�B���                                    Bx�L�P  �          AZ�\�
=AD  @�p�A�  BǏ\�
=AX  >��?���B���                                    Bx�L��  T          A]녿˅ANff@��A�z�B��˅A[�
�J=q�P  B�(�                                    Bx�Lڜ  
(          A]p���Q�AT  @s33A\)B�=q��Q�AZ�\������ffB��                                    Bx�L�B  
(          A[\)��33AW
=@	��A��B���33AR�R�\���j�RB�{                                    Bx�L��  �          A`�׾���AZ�\@!G�A'\)B�zᾙ��AXQ��Mp��Up�B��                                     Bx�M�  
�          Ad�׾���A^=q@'�A*�HB��3����A\Q��N{�R=qB��q                                    Bx�M4  "          Aep���
=A[�
@vffAyB��)��
=Ab{��\��RB��{                                    Bx�M#�  �          Ab�\����AX��@`  Af=qB�������A]���\�{B�ff                                    Bx�M2�  "          AP  ��=qAH(�@+�A@Q�B����=qAHQ��'��<(�B��                                    Bx�MA&  �          AR{�=p�AQ�>�p�?�{B�� �=p�AAp���
=��=qB�                                    Bx�MO�  T          AF=q��
=A>{@N�RAp��B����
=AB�\��\)�
�RBýq                                    Bx�M^r  �          AT�Ϳ��AR{��\)���HB�G����A>{���H��  B�                                    Bx�Mm  
x          AC�
����A>{�   �<��B�  ����Ap����
�z�Bî                                    Bx�M{�  �          A&=q�h��@Y���  B�B�\�h�ÿ\(��$��¦\C_k�                                    Bx�M�d  
�          A(�׿���@��\�G��o�RBѸR����?p���%¢  C5�                                    Bx�M�
  �          A*�R�(�@��R�z��gBᙚ�(�?����%��C��                                    Bx�M��  �          A,�׿��@�{��R�k(�B����?�(��)p�¢�CE                                    Bx�M�V  
�          A)p��E�@������x�B�{�E�?(��(Q�©G�C�
                                    Bx�M��  �          A*ff?+�@����
=�o�HB�u�?+�?�ff�(��¦�\Be                                      Bx�MӢ  
�          A*�H>8Q�@�{��`�B��\>8Q�?���((� �{B���                                    Bx�M�H  �          A(��?�\)@Ϯ��\�LffB�z�?�\)@$z��!����B��                                    Bx�M��  �          A%�?z�@�����P(�B��=?z�@�H�   \B�(�                                    Bx�M��  �          A"ff=��
@����G��R�B���=��
@�\�=q��B�8R                                    Bx�N:  "          A�R�E�@�Q����f\)Bģ׿E�?������¡��B�L�                                    Bx�N�  
�          A�\�h��@�{��H�`�B��h��?˅�33��B�                                    Bx�N+�  T          Azῳ33@�G������J33B�
=��33@������B�k�                                    Bx�N:,  T          A!��G�@��
��G��#33B��
��G�@����  B�G�                                    Bx�NH�  �          A!p���@����ff�_=qB�� ��?����	�W
B�.                                    Bx�NWx  T          A5@�A'��`�����B��f@�AG�����'G�B���                                    Bx�Nf  �          A5�?�  A,���C�
�z�HB�8R?�  A����  �G�B�.                                    Bx�Nt�  �          A2�H?�=qA*�H�0���d��B��{?�=qA	G����G�B���                                    Bx�N�j  T          A0Q�?�A(�������
B��?�@������Q�\B�
=                                    Bx�N�  "          A2�R?У�A z�������B�� ?У�@�\���={B��)                                    Bx�N��  "          A3�
@%�A&�R�G
=��G�B�33@%�A
=����33B�\                                    Bx�N�\  
�          A3
=@8��A)p��ff�-p�B�� @8��A(�����	�B��                                     Bx�N�  
�          A5p�@;�A,(��������B��\@;�A���Q���G�B��                                    Bx�N̨  T          A7�@��A)���o\)��z�B��\@��AG�����,  B��{                                    Bx�N�N  
�          A6ff?���A   ��  ���
B���?���@�����H�O{B��H                                    Bx�N��  	�          A8��?z�HA(��ۅ�z�B��f?z�H@���!p��s�B���                                    Bx�N��  "          A;�
?J=qA���  �\)B�33?J=q@�33�&�H�z\)B�8R                                    Bx�O@  
(          A@  ?
=qA����\)�=qB�z�?
=q@���(���v��B�
=                                    Bx�O�  
Z          A?\)?�33A���G��\)B�?�33@�
=�(���w\)B�L�                                    Bx�O$�  �          A?\)?�Q�A�R��=q�(�B�=q?�Q�@�p��+\)�~��B��                                    Bx�O32  �          A>�\?�
=A����1�B���?�
=@��
�0���)B��q                                    Bx�OA�  �          A<��?�z�A�\�{�5Q�B�\)?�z�@y���0z��qB�8R                                    Bx�OP~  
�          A<Q�?��A�����*  B��?��@�{�,��u�B�L�                                    Bx�O_$  �          A;�
?�=q@��
�
=�?�B���?�=q@QG��1p���Br��                                    Bx�Om�  T          A:=q?���@������\p�B��f?���?����6�\ǮBr
=                                    Bx�O|p  T          A8��?5@�{���8��B���?5@mp��.=qB�aH                                    Bx�O�  
�          A7�
?k�@����33�E�\B�?k�@C�
�0(���B�\)                                    Bx�O��  r          A8��?��\@�����<
=B��?��\@^{�,Q��HB�aH                                    Bx�O�b  
�          A7
=?
=@�Q����:G�B�L�?
=@e��+���B�G�                                    Bx�O�  	�          A9p�?�p�@ڏ\��\�Q�B��{?�p�@�3
=�
BV                                      Bx�OŮ  |          A?�@$z�@�
=� z��d=qB�aH@$z�?�
=�9��A�p�                                    Bx�O�T  "          A=�@�@�G�����N�\B��=@�@���6�\�fBAQ�                                    Bx�O��  �          A=�?�z�@��
�  �G�B��?�z�@8Q��4Q��Ba                                    Bx�O�  T          A8  ?��@�
=�p��I�B�u�?��@3�
�0����B��                                    Bx�P F  "          A8(�?p��@��H����G��B�\)?p��@;��1G���B�L�                                    Bx�P�  "          A7�
?��@���z��W�B��=?��@
=�3��)BvG�                                    Bx�P�  "          A6�H?�{@�
=���Z�B�33?�{?�z��2�HL�BY��                                    Bx�P,8  �          A5G�?�33@�ff��R�j��B�ff?�33?����2�R¢�B{                                    Bx�P:�  
�          A4��@
=@�G��ff�O�HB�Q�@
=@
=q�-G��B)�                                    Bx�PI�  
Z          A3�
@p�@љ��{�PffB��q@p�@
=q�-��B1�                                    Bx�PX*  T          A3\)?��@�����Y��B��?��?��/
=��BAff                                    Bx�Pf�  �          A4z�@z�@��R�p��_�RB��@z�?�
=�/�
.B
�\                                    Bx�Puv  T          A3�?�Q�@�p����f��B�8R?�Q�?����/�\A��                                    Bx�P�  �          A2{?�33@�33�G��d��B�  ?�33?����.�H   B-33                                    Bx�P��  T          A1��?�\)@љ����P{B�\?�\)@p��*�H��BF�
                                    Bx�P�h  T          A0��?�\)@��R�\)�k�B��f?�\)?h���.{ �\A�33                                    Bx�P�  "          A/�?��@��\�\)�n�\B��
?��?G��-G�¡�A�Q�                                    Bx�P��  "          A/33?�G�@�=q����r��B�Ǯ?�G�?��,�� �qA��H                                    Bx�P�Z  "          A.�\?��@�Q��  �rz�B��f?��>��+��Ac\)                                    Bx�P�   "          A-��?˅@��H�G��x{B��\?˅>�\)�+�¢�)A (�                                    Bx�P�  �          A,(�?���@��\�
=�w  B��)?���>����)��¢A;�                                    Bx�P�L  T          A+33@�\@��R�p��`��B��@�\?�33�%��A�{                                    Bx�Q�  
Z          A+\)@%@��\���gQ�B|(�@%?8Q��%���Aw�                                    Bx�Q�  S          A,��@S�
@�Q����O�RBpG�@S�
?����"{.A��                                    Bx�Q%>  T          A+�
@�z�@�����{�7�BM\)@�z�?����
=�w��A�(�                                    Bx�Q3�  �          A+
=@��@�������LBR=q@��?�p��Q�\A�
=                                    Bx�QB�  
�          A*ff@�@�����p33B�@�>��&�HaHAT��                                    Bx�QQ0  �          A)p�?�
=@�Q��Q��t  B��f?�
=>��
�&�R�RAQ�                                    Bx�Q_�  �          A&�H@�@��
���m��B��@�?
=q�#33��Ad��                                    Bx�Qn|  "          A%�?�ff@�(��G�z�B��)?�ff��\)�#33�qC��                                    Bx�Q}"  
�          A$��?���@����G���B�L�?��׾���#\)¤\)C��{                                    Bx�Q��  �          A#�?�z�@s�
��\#�B�.?�z�(��"{¥L�C�+�                                    Bx�Q�n  �          A"�\?�33@R�\����
B��?�33����� z�¡�RC��R                                    Bx�Q�  
�          A�?���@^{��#�B�aH?��ÿW
=���z�C��\                                    Bx�Q��  T          A
=@�@tz����|\)Bjp�@���33���k�C��)                                    Bx�Q�`  T          A   @L��@����z��`z�B[\)@L��?���HA33                                    Bx�Q�  �          A!��@k�@�(��G��W(�BN  @k�?0�����
=A*�R                                    Bx�Q�  
�          A z�@4z�@`  �33�}z�BL��@4z�5�p�L�C��{                                    Bx�Q�R  �          A�@4z�@*�H�{�)B-��@4z���
�=q8RC���                                    Bx�R �  �          A(�@:=q@�Q���jQ�BX  @:=q=�G����G�@\)                                    Bx�R�  T          A33@33@5���(�BXQ�@33��{���=C���                                    Bx�RD  �          A�R?��
?�����B4��?��
����
�R(�C��                                    Bx�R,�  �          A�?��
?�z��Q���B>�?��
��H���z�C��q                                    Bx�R;�  T          Aff?
=?Q���§Q�BY�H?
=�K��z�p�C�7
                                    Bx�RJ6  �          A\)=�=����33²{B%  =��\)�����C�ٚ                                    Bx�RX�  "          A�
���
��33��¯�C�˅���
��p��=q�x�C��                                     Bx�Rg�  �          AQ쿷
=�\��HW
Cb���
=��p���
=�R�RC}��                                    Bx�Rv(  "          A������ ��  Cj5ÿ�����
��
=�K
=C�3                                    Bx�R��  "          A�׿�33�,(���  ��Cv�Ϳ�33������=q�2z�C�]q                                    Bx�R�t  �          A녿���
=q����p�CsQ쿋���=q��=q�@�RC�:�                                    Bx�R�  �          @�녿:�H�$z���\��C~.�:�H��  ����2�HC�:�                                    Bx�R��  �          @�=q���-p��陚�RC�׾���������/G�C��3                                    Bx�R�f  �          @�Q�^�R�ff���HQ�Cy��^�R�����33�8�\C�XR                                    Bx�R�  T          @�ff�G��#�
��ff33C|�q�G���{���
�1�C��f                                    Bx�Rܲ  �          @�z῔z��[���ff�v�\C{J=��z���  �������C��f                                    Bx�R�X  
�          @�G���{������(�8RC�e��{��Q������T=qC��R                                    Bx�R��  �          @�׾�p�������=q��C�z᾽p����
��z��O�C��{                                    Bx�S�  
�          @�R���þ������¬  Ce𤾨���e���  �t(�C�W
                                    Bx�SJ  
�          @�>�{?p����£�B��>�{�\)���#�C�Ff                                    Bx�S%�  �          @��?   ?Q���\¤z�Bi��?   ���\)��C�{                                    Bx�S4�  
�          @�=q>�{?���޸R �3B��>�{� ����Q�p�C�                                    Bx�SC<  "          @���?��=�\)��{«.@�{?���Dz���  ��C��                                    Bx�SQ�  �          @��H?�=q@@  ��ff�~��B��?�=q��G���¢�C�,�                                    Bx�S`�  �          @��H?��@��R����YffB��R?��?�����(��\B�                                    Bx�So.  "          @�{?��@����\�[��B�B�?��?}p���Bz�                                    Bx�S}�  
�          @أ�?Ǯ@��\�P  ��33B��H?Ǯ@J=q��z��^G�B~�
                                    Bx�S�z  
V          @�
=?�Q�@hQ�@�ffBEz�B��?�Q�@��R@�HA���B�=q                                    Bx�S�   
Z          A\)>�=q@K�@��B��=B�.>�=q@ȣ�@�z�B'�RB�                                      Bx�S��  "          Az�>B�\@+�AB�Q�B�  >B�\@���@ٙ�B=(�B��{                                    Bx�S�l  
�          A�?!G�@��A�HB��B��)?!G�@��@�G�BIB��f                                    Bx�S�  
�          A=q?8Q�?�  AQ�B���B���?8Q�@���@�BZ��B��H                                    Bx�Sո  
�          A33?&ff?�  A	�B�\)ByQ�?&ff@��
@��B^G�B��3                                    Bx�S�^  �          @�\)?�����H@�=qB�ǮC�  ?��?�  @�G�B�ffB                                    Bx�S�  
(          A
=q?�  ��z�A\)B��C��?�  @\(�@��B�=qB��                                    Bx�T�  
Z          A�?(��@�33A��Bx�B���?(��@�G�@�=qBffB��                                    Bx�TP  
�          A�\?0��@љ�@�
=B6\)B��?0��AG�@K�A��B�
=                                    Bx�T�  
Z          A��?^�R@�z�@��B&ffB���?^�RA=q@!�A|��B��
                                    Bx�T-�  "          A�R?5@�@�
=B+G�B���?5A�@*�HA�B���                                    Bx�T<B  
�          A�>�ff@�=q@�
=B-p�B��>�ffA
{@-p�A�p�B�                                      Bx�TJ�  
�          A��=�Q�@��@���B1Q�B�L�=�Q�@��H@&ffA��HB��R                                    Bx�TY�  T          A\)�8Q�@�=q@�Q�B<��B����8Q�@�z�@=p�A��B���                                    Bx�Th4  "          AG�>aG�@�G�@��B��B�  >aG�@��H?�p�AF�RB���                                    Bx�Tv�  
�          @�\)?Q�@�\)@�B,�\B���?Q�@���@
=A���B���                                    Bx�T��  
(          @�G�?.{@�
=@�\)B
=B�
=?.{@�p�?u@��
B��f                                    Bx�T�&  
�          @���?�@���@��B�RB��H?�@��?�{A  B��=                                    Bx�T��  
Z          @���?�R@У�@�{B�B��?�R@�ff?h��@ָRB�Ǯ                                    Bx�T�r  
�          @�Q�>���@�=q@�33A���B��q>���@��R?L��@��RB��                                    Bx�T�  	�          @�{�h��@�Q�@��\B	�\BĔ{�h��@���?��A	G�B���                                    Bx�Tξ  "          @��#�
@У�@\)A���B�=q�#�
@�?:�H@�B���                                    Bx�T�d  "          @�z�c�
@ҏ\@qG�A���B�aH�c�
@�=q?   @qG�B�aH                                    Bx�T�
  
�          @�(���R@�(�@�\)B�B��׿�R@�\)?��A!��B��=                                    Bx�T��  T          @��R��ff@��@�z�B��BƔ{��ff@�\?fff@�\)B�                                    Bx�U	V  �          @�녾#�
@ə�@��BG�B��)�#�
@�
=?k�@���B�ff                                    Bx�U�  
�          @�\)?��H@��@�Q�B!p�B�8R?��H@�ff@�
Aw
=B��{                                    Bx�U&�  �          @�{?�33@�  @��B&��B���?�33@�{@
�HA�p�B�#�                                    Bx�U5H  "          @�{?���@�p�@�  B,��B�
=?���@�@A�  B��3                                    Bx�UC�  	�          @�=q?��\@��@��RB$
=B�L�?��\@�
=?��RAtQ�B��
                                    Bx�UR�  "          @��?�ff@��@��B1��B�z�?�ff@ۅ@��A�G�B��\                                    Bx�Ua:  
�          @�z�>�p�@�Q�@�(�B;�B���>�p�@ۅ@(Q�A�33B��)                                    Bx�Uo�  �          @��@�@��@�=qB1�B���@�@Ϯ@{A�G�B�u�                                    Bx�U~�  �          @���@u�@��@��B33BN�
@u�@���?\A=�Bh=q                                    Bx�U�,  "          @�ff@]p�@�{@�z�B{BVp�@]p�@�  @ffA�G�Bt=q                                    Bx�U��  "          @�G�@W�@�ff@��RB BSp�@W�@��@�A�\)Bs�
                                    Bx�U�x  �          @�G�@Q�@�33@��
B2�RBz�@Q�@�(�@%�A��
B�{                                    Bx�U�  �          @�Q�@=q@�@��B@
=Bp
=@=q@Å@>�RA�
=B��                                    Bx�U��  "          @�=q@�@�  @��BB�HBwG�@�@�\)@C�
A�Q�B��)                                    Bx�U�j  
�          @��H@&ff@��@�Q�B,��Bs�@&ff@�(�@��A��\B���                                    Bx�U�  �          @��H@+�@�ff@�(�B	
=BQ�@+�@ָR?���A%G�B��                                    Bx�U�  
Z          @�Q�@ ��@��@\(�A���B���@ ��@���?�\@���B�aH                                    Bx�V\  
�          @�@33@�{@5A��B�#�@33@ٙ��8Q쿳33B��{                                    Bx�V  
Z          @�Q�@�H@�p�@A��\B��R@�H@�G��8Q���\)B���                                    Bx�V�  
�          @�{@(��@�(�@L(�A���B��=@(��@�>�\)@��B���                                    Bx�V.N  	d          @�R@1�@�Q�@VffA�p�B�z�@1�@�z�>�@s33B���                                   Bx�V<�  z          @�\)@Z=q@�=q@N�RA�z�Bn�R@Z=q@���>�{@$z�B{�                                    Bx�VK�  �          @�p�@R�\@�=q@L��Ȁ\Br�@R�\@�(�>��R@�HB~�                                    Bx�VZ@  �          @�p�@	��@��@G
=AӮB�Ǯ@	��@�(�>W
=?޸RB��                                    Bx�Vh�  �          @��H@
�H@�
=?�Aq�B��@
�H@��
��Q��
=B��3                                    Bx�Vw�  �          @�\@7
=@��?�=qA.ffB�8R@7
=@�  ��=q�N�RB��
                                    Bx�V�2  �          @�R@G�@�ff?z�@�(�B�\)@G�@�G���
��ffBz��                                    Bx�V��  �          @�ff@7
=@��H>�p�@=p�B��@7
=@\�$z����B���                                    Bx�V�~  �          @�\)@-p�@�  �p����z�B�33@-p�@�ff�fff��Q�Bz
=                                    Bx�V�$  �          @��@S�
@�
=�#�
����Bk\)@S�
@r�\���R�,z�BCp�                                    Bx�V��  �          @�Q�@{@����k��33B�@{@G���(��Z��BNQ�                                    Bx�V�p  �          @��@1�@����
=�/z�Bf  @1�?�p���(��{ffA�\)                                    Bx�V�  �          @�@ff@r�\����N
=Bh�@ff?p����ff��A��                                    Bx�V�  �          @�G�@�@e���p��Z�RBjp�@�?(���z��\Az�R                                    Bx�V�b  �          @�R@*�H@�z������&��Bpff@*�H@z���=q�w��B33                                    Bx�W
  �          @�z�@C33@��\�|���Bl  @C33@5������Y{B+(�                                    Bx�W�  �          @޸R@.{@��H�aG���\)B|
=@.{@P  ��Q��QBHG�                                    Bx�W'T  "          @��@8Q�@�{�:=q���B|�@8Q�@u�����;Q�BT=q                                    Bx�W5�  "          @���@Y��@����0�����Be{@Y��@b�\����1�\B8��                                    Bx�WD�  �          @��@j=q@���G����B_z�@j=q@~{��{�=qB=�                                    Bx�WSF  �          @�G�@u@�
=�n{����B[��@u@�  �O\)��G�BF(�                                    Bx�Wa�  �          @�33@i��@���p���  BZ  @i��@��E���HBC��                                    Bx�Wp�  �          @�33@~�R@��
���H�t  BJ��@~�R@g��k��	z�B)33                                    Bx�W8  �          @У�@qG�@�p������*=qBW�@qG�@��
�Vff���B>(�                                    Bx�W��  �          @��H@l��@�p���33�h��BY�@l��@z�H�p  �p�B:�                                    Bx�W��  �          @أ�@g
=@�녿s33��HBc��@g
=@�=q�S33���BN�H                                    Bx�W�*  �          @�Q�@8��@\>u@�B�#�@8��@�=q�p���
=Bz=q                                    Bx�W��  �          @��H@g�@�Q��n{���BFz�@g�@(���
=�K�RA�\)                                    Bx�W�v  �          @���@C�
@���@  ��
=Bk33@C�
@N�R��p��?Q�B:                                      Bx�W�  �          @�=q@e@����W����BG��@e@ff����CQ�B
=                                    Bx�W��  �          @��@i��@����
=q���BR��@i��@W
=������HB*��                                    Bx�W�h  �          @�@}p�@�ff�}p��{BMQ�@}p�@�Q��C�
��  B5z�                                    Bx�X  �          @�  @|(�@�p���  �  BG\)@|(�@p���<(����B.��                                    Bx�X�  �          @��@n{@��׿��\���BP{@n{@u�@������B7z�                                    Bx�X Z  �          @\@z�H@��>�\)@(Q�BF�@z�H@��׿�\��(�B=�
                                    Bx�X/   �          @�=q@aG�@.�R@1G�A��RB�@aG�@j=q?��Ac�
B8��                                    Bx�X=�  �          @�Q�@?\)?s33@��
Bj
=A�z�@?\)@Fff@�{B0�
B7��                                    Bx�XLL  �          @���@-p�>W
=@�33B|(�@�Q�@-p�@"�\@��RBO{B,\)                                    Bx�XZ�  �          @ȣ�@N�R?k�@���Be��A~=q@N�R@HQ�@��\B/�B0z�                                    Bx�Xi�  �          @�33@'�=���@�\)B���@
=@'�@&ff@��BX��B2��                                    Bx�Xx>  �          @���@H��?�z�@��BlA�=q@H��@`  @���B/�HB@p�                                    Bx�X��  �          @�p�@Vff?���@���B^33A���@Vff@c33@���B!Q�B:�R                                    Bx�X��  �          @�ff@<��?��@�z�Bt{A���@<��@Z=q@��
B6�HBD�                                    Bx�X�0  �          @��H@,��?c�
@��B}33A�=q@,��@Q�@�ffB?z�BJ
=                                    Bx�X��  �          @��H?�  >�ff@�=qB�AhQ�?�  @Dz�@�Q�B`z�Bp�
                                    Bx�X�|  �          @���?�Q�>���@���B���AW33?�Q�@@  @��Bb��BrQ�                                    Bx�X�"  �          @˅?�z�?�@�G�B��qAs
=?�z�@G
=@��RB[��Bi                                    Bx�X��  �          @�33?�33���R@\B�k�C��)?�33@Q�@�33B~�Bn{                                    Bx�X�n  �          @��?�ff��Q�@ÅB��=C�j=?�ff@ff@���By\)BR{                                    Bx�X�  �          @�Q�@C�
?�ff@��Bq��A��
@C�
@n�R@�ffB133BJ�                                    Bx�Y
�  �          @���@mp�@�
@�p�BJ\)A�  @mp�@��@p  BffB?�\                                    Bx�Y`  �          @�  @0  ?�
=@�
=Bz=qA�p�@0  @x��@�  B3�\BZ��                                    Bx�Y(  �          @�Q�@'
=?�@�  B�
=A���@'
=@j�H@�z�B=ffBZQ�                                    Bx�Y6�  �          @�ff@\)>�
=@�p�B��{Az�@\)@Dz�@��
BV�RBL                                      Bx�YER  �          @�  @��>\@�=qB��A\)@��@E@���B\�HBW\)                                    Bx�YS�  �          @�=q?�
=�B�\@�=qB�\C�Ф?�
=@%@���Bwz�Bd{                                    Bx�Yb�  �          @���?��
�h��@�G�B�
=C�@ ?��
?�
=@��B�G�Ba�H                                    Bx�YqD  �          @�(�?G��@ϮB��C�@ ?G�?s33@�  B�\)BJp�                                    Bx�Y�  �          @ᙚ?O\)�Q�@�p�B�Q�C�u�?O\)?}p�@�B�
=BJ�                                    Bx�Y��  �          @�G�?fff��@�G�B��RC�4{?fff?(��@�ffB��3B\)                                    Bx�Y�6  �          @��?8Q����@У�B�=qC�1�?8Q�?!G�@�{B�B�B%G�                                    Bx�Y��  �          @�{>Ǯ�'
=@���B�33C�5�>Ǯ>�G�@���B�\)BA                                    Bx�Y��  �          @�{>Ǯ�(�@љ�B�p�C�
>Ǯ?aG�@�33B�\B��{                                    Bx�Y�(  �          @ڏ\����1G�@�\)B���C�y����>W
=@�=qB���B��q                                    Bx�Y��  �          @�z�\)�2�\@�Q�B��C�J=�\)>W
=@�33B���C��                                    Bx�Y�t  �          @�33�L���N{@��RBsffC��L�;��R@أ�B�p�CI^�                                    Bx�Y�  �          @�Q�?�
=��R@���B�u�C���?�
=?+�@�Q�B��=A��                                    Bx�Z�  �          @�{?�Q���@�33B���C�T{?�Q�?.{@�{B��A��
                                    Bx�Zf  �          @ָR�}p��\��@���Bf(�C}�R�}p��(��@��HB�z�CU�f                                    Bx�Z!  �          @�
=�E��`  @��Bd�C��3�E��=p�@���B�z�C_��                                    Bx�Z/�  �          @ٙ�?��
��33@���B���C�H�?��
?���@ϮB�G�BK(�                                    Bx�Z>X  �          @�  ?���� ��@���B�#�C�e?���?h��@���B�\B
=                                    Bx�ZL�  �          @�{?#�
��\@�z�B�#�C��3?#�
?�@���B��HBu{                                    Bx�Z[�  �          @���?:�H���@�ffB�C�,�?:�H?333@�G�B�ffB/�                                    Bx�ZjJ  �          @�=q?��R��@�=qB��C�Ǯ?��R?(��@��B�W
A�
=                                    Bx�Zx�  �          @У׾aG��
�H@�z�B��fC�
�aG�?333@�\)B�\B�                                    Bx�Z��  �          @�G�>\����@ǮB�L�C�Ǯ>\?��
@�B���B��                                    Bx�Z�<  �          @�\)>�G���=q@�B�8RC���>�G�?��
@˅B�#�B��
                                    Bx�Z��  �          @�Q�?B�\��@��B���C�?B�\?n{@�z�B��qBKQ�                                    Bx�Z��  �          @�p�>���@�(�B�{C�p�>�?��\@��B���B�u�                                    Bx�Z�.  �          @�ff>��Ϳ��@�B���C�O\>���?�ff@��HB��)B�aH                                    Bx�Z��  �          @��=����@ÅB��3C��R=�?p��@�=qB�B�=q                                    Bx�Z�z  �          @�ff>���   @��
B��qC�{>��?Tz�@�z�B��HB�B�                                    Bx�Z�   �          @���>���33@�G�B��RC��=>��?@  @ʏ\B��Bt�                                    Bx�Z��  �          @�p�?���@���B�.C���?�?.{@˅B���BR\)                                    Bx�[l  �          @�{>B�\�33@�  B�C�Y�>B�\?�\@��B�G�B�u�                                    Bx�[  �          @���>�(���H@��B��RC�3>�(�>�33@��
B���B
=                                    Bx�[(�  T          @���?+��z�@���B�=qC��?+�?8Q�@�=qB�aHB<33                                    Bx�[7^  �          @˅?fff�G�@��
B���C��f?fff>��@ȣ�B���AܸR                                    Bx�[F  �          @�(�?�R�Q�@�(�B��\C�K�?�R>�p�@��HB���A�(�                                    Bx�[T�  �          @�z�>��H�Q�@�(�B�{C��H>��H>�p�@ʏ\B���B��                                    Bx�[cP  �          @�p�=����S33@�\)Bk�C��q=��Ϳ!G�@��
B�� C�z�                                    Bx�[q�  �          @�Q�?Q��1�@��B}��C�0�?Q논��
@�{B��=C�=q                                    Bx�[��  �          @�\)>�=q�9��@��By�C��>�=q�u@ȣ�B�z�C�Y�                                    Bx�[�B  �          @Ϯ��G�����@�ffBH  C�B���G��У�@�  B���C��                                    Bx�[��  �          @�\)>��
�Fff@�z�Btz�C��>��
�\@�B�#�C���                                    Bx�[��  �          @�{���
�ff@\B���C��ü��
?333@���B��=B�z�                                    Bx�[�4  �          @�
==#�
���R@�z�B8�C�B�=#�
�33@�33B��
C��\                                    Bx�[��  �          @�z�=������@��
B-��C�� =����33@�p�B�p�C�@                                     Bx�[؀  �          @�z�?B�\���@��HB��{C���?B�\>�\)@�=qB�u�A�=q                                    Bx�[�&  �          @˅?���S�
@��Bg�C�  ?���333@�Q�B�33C�'�                                    Bx�[��  �          @���>��
��ff@^{B�HC��q>��
�N{@��Bkz�C��                                    Bx�\r  �          @ʏ\=�\)��(�@�Q�B*ffC�k�=�\)�Q�@��\B���C��3                                    Bx�\  �          @�G�������G�@�
=BEC�Lͽ��Ϳ�33@��B�=qC�J=                                    Bx�\!�  �          @�33@�\����@{�B�HC�q�@�\�z�@�Bqz�C��
                                    Bx�\0d  �          @�Q�#�
�E@�Bqp�C��ͽ#�
���H@�
=B�  C��3                                    Bx�\?
  �          @�Q�>�z��H��@�(�Bn�C���>�z�\)@�ffB�p�C��                                    Bx�\M�  �          @�\)?��Dz�@��Bo{C�H�?녿�\@��B��
C�K�                                    Bx�\\V  �          @�>�
=�s�
@��BM�RC�%>�
=��z�@�
=B�G�C�J=                                    Bx�\j�  �          @ƸR>���U@�
=Bep�C�T{>���O\)@�z�B���C�ff                                    Bx�\y�  �          @�>�z��<��@���Bt�HC��3>�z�Ǯ@���B���C���                                    Bx�\�H  �          @��>.{�X��@��B_�C�w
>.{�u@��RB�k�C�"�                                    Bx�\��  �          @�{��=q�^�R@�\)BVz�C��f��=q��33@�Q�B��
C�^�                                    Bx�\��  �          @����J=q@�\)Bd��C�G����B�\@�33B�
=CkT{                                    Bx�\�:  �          @�
=�Tz��333@�ffBr�C}��Tzᾮ{@���B�ǮCJ33                                    Bx�\��  �          @�\)�G���@���B�\)C|0��G�=�\)@�p�B��fC.��                                    Bx�\ц  �          @�
=��ff�(�@�{B��Ct^���ff>��R@��B���C#��                                    Bx�\�,  �          @���0�׿��H@�33B�L�Cz�Ϳ0��?z�@��B��{Cٚ                                    Bx�\��  �          @���=�G��"�\@��HB�.C�7
=�G�=L��@�z�B�A�z�                                    Bx�\�x  T          @�{>�
=�>{@��
Bs=qC��>�
=��G�@��
B��RC��{                                    Bx�]  T          @�=q�u��p�@���B���C���u?^�R@��B�ffB���                                    Bx�]�  �          @\����=q@��RB�=qC�z��?\@��B���B�\                                    Bx�])j  T          @�Q�#�
��\@�Q�B�C�  �#�
>k�@�
=B��\B��)                                    Bx�]8  �          @��׾���	��@��B�ǮC��q���>���@��B��B��                                    Bx�]F�  �          @�G���33�\)@��HB�C��q��33>���@�Q�B�=qC
�\                                    Bx�]U\  �          @������(�@�33B�u�C����>�Q�@�  B��fCE                                    Bx�]d  T          @�Q�����\@���B��fC�G����>k�@�\)B���C��                                    Bx�]r�  �          @���u���@�z�B��Cx  �u=��
@���B�W
C.�q                                    Bx�]�N  �          @�Q�Y���)��@�G�Bx�HC|#׿Y���L��@��B��CA5�                                    Bx�]��  �          @�{�G��p�@��\B�\)C|O\�G��#�
@��
B�ǮC4�3                                    Bx�]��  �          @��R�E��!G�@��\B~��C}
=�E���\)@���B�\C9^�                                    Bx�]�@  �          @�\)�fff���@�z�B�B�Cyz�fff=�\)@���B��C/�H                                    Bx�]��  T          @�  �����  @�G�B�8RC��R����>��@�\)B��)C�f                                    Bx�]ʌ  �          @�Q�z���\@���B���C� �z�>W
=@�
=B�.C 8R                                    Bx�]�2  �          @�G���{��@���B�aHC��=��{>��@���B��C��                                    Bx�]��  
�          @��;B�\��@�B�G�C���B�\>8Q�@�z�B���C��                                    Bx�]�~  T          @��R�Ǯ���@�{B��
C�Z�Ǯ=�\)@�ffB�Q�C)��                                    Bx�^$  �          @�ff�(����@��B�(�C~k��(��=�Q�@���B�#�C,s3                                    Bx�^�  
�          @�  ��
=�%@�z�B�#�C�k���
=��@�\)B���CD��                                    Bx�^"p  �          @�\)�0���\)@�z�B��fC~��0�׼�@�{B���C6�3                                    Bx�^1  �          @�p��O\)��R@���B~��C{ٚ�O\)��\)@��\B�{C9&f                                    Bx�^?�  �          @���aG��;�@��RBrQ�C��R�aG���@��RB�.Cv�                                    Bx�^Nb  �          @���=L���4z�@��HBx��C���=L�;�Q�@���B�
=C��                                    Bx�^]  �          @�G�>8Q��%@�ffB��C��{>8Q��G�@���B�C��=                                    Bx�^k�  �          @�  �\)�\)@�{B��
C�b��\)��@�\)B�G�C>=q                                    Bx�^zT  �          @��ÿ��!G�@��RB�z�C�)���L��@�Q�B��C9�                                    Bx�^��  �          @��׾�ff���@�Q�B�z�C��
��ff=�Q�@�Q�B�\)C(��                                    Bx�^��  T          @��׿��4z�@�  BuG�C��=����
=@�ffB�(�C[
=                                    Bx�^�F  �          @���+��&ff@�33B}��C� �+��.{@�ffB��3CBp�                                    Bx�^��  �          @��;�Q��
=@�\)B�=qC����Q�>��R@��B���C8R                                    Bx�^Ò  �          @���ff�@���B���C��׾�ff>�33@�z�B��fC=q                                    Bx�^�8  �          @��R�L���ff@�=qB�� C�Y��L��>�Q�@�{B�#�B�(�                                    Bx�^��  �          @�ff�u���@��B��fC�q�u<��
@�p�B�L�C0@                                     Bx�^�  �          @�z����{@��B��C����ͽ��
@��B�{C<�q                                    Bx�^�*  �          @�33�.{�#33@��Bv�HC��.{����@���B�k�CM�3                                    Bx�_�  �          @��\�z�� ��@�p�By�\C�p��zᾔz�@���B��=CN!H                                    Bx�_v  �          @��
�
=���@���B�RC�)�
=�\)@��HB���CAJ=                                    Bx�_*  T          @��\�#�
�-p�@�=qBo�
C�Uÿ#�
��@�Q�B��fC[�                                    Bx�_8�  �          @�녿   �5�@�  Bk=qC���   �(��@��B��
Ci�                                    Bx�_Gh  �          @�����:�H@��\Biz�C��H��Ϳ5@�33B���Ch0�                                    Bx�_V  &          @�Q�#�
�9��@�BkC����#�
�(��@�B���Ca��                                    Bx�_d�            @�\)���HQ�@�  BaG�C�˅���p��@��
B�ǮCs޸                                    Bx�_sZ  �          @���!G��J�H@�\)B^�\C�j=�!G��}p�@��B�\)Cm�R                                    Bx�_�   �          @�=q�!G��J�H@��\B`��C�g��!G��s33@�ffB�W
Cl�f                                    Bx�_��  �          @�G��!G��K�@���B`
=C�g��!G��xQ�@�B��Cm)                                    Bx�_�L  �          @�G��
=�E�@��Bd�RC��{�
=�\(�@�{B�33Ck�H                                    Bx�_��  �          @�G��\�P  @���B^G�C��׾\���@�p�B���Cz{                                    Bx�_��  �          @�\)��(��Fff@���Bc�\C�\��(��h��@�z�B��qCt��                                    Bx�_�>  �          @��;���A�@�  Bd�HC�q��녿^�R@�=qB�W
Ct��                                    Bx�_��  �          @�G���{�-p�@��Bq�RC�l;�{���@�  B�� Cns3                                    Bx�_�  �          @��׾��\)@�p�B|ffC��������R@�Q�B��Cw�R                                    Bx�_�0  �          @��þ�=q�  @�G�B���C��H��=q�L��@�Q�B�k�C=�3                                    Bx�`�  �          @��׿�R���@�z�B�C{�q��R>��
@�
=B�aHC}q                                    Bx�`|  �          @��\��ff�	��@��
B���C����ff=�\)@���B��C*�f                                    Bx�`#"  �          @�p����
�
=@�z�B�=qC�1쾣�
��@��B�(�CHٚ                                    Bx�`1�  �          @�������@�p�B�33C�S3�������
@�p�B��qCBT{                                    Bx�`@n  �          @�{����ff@�p�B�\)C����녽�G�@�B�k�CB�=                                    Bx�`O  �          @��R�\��@�p�B�33C�y��\��@�{B�  CF�                                    Bx�`]�  �          @�p���G��{@�ffB�  C�g���G�<�@��B��C0�3                                    Bx�`l`  �          @����ff��\@�Q�B�C����ff>W
=@�z�B���C�\                                    Bx�`{  �          @��\��G�����@��B�\C�]q��G�>�p�@���B�33C��                                    Bx�`��  �          @��H����
@��B�{C~�)��>��@���B���C%L�                                    Bx�`�R  �          @�G�����z�@�33B�p�C
=���=�G�@�Q�B�L�C(Ǯ                                    Bx�`��  �          @���(����@��
B��C~}q�(��#�
@��\B�� C4�                                    Bx�`��  �          @�(��=p���@���B���Cz� �=p�=�\)@�=qB�C.L�                                    Bx�`�D  �          @�(��:�H��@���B�p�Cx)�:�H>��@��B��C��                                    Bx�`��  �          @�ff�+����
@�(�B�{Cy0��+�>�@�z�B�W
C�                                     Bx�`�  �          @�G���G���@�
=B��HC�K���G�>�ff@�  B�� C{                                    Bx�`�6  �          @��׾���ff@�G�B�Q�C|�)��?=p�@�ffB�Q�B�\)                                    Bx�`��  �          @�G���׿�Q�@��HB���C{�׾��?\(�@�ffB��=Bힸ                                    Bx�a�  �          @�G���녿�{@�G�B���C�\���?.{@�\)B�8RB�                                    Bx�a(  �          @�p��u�ٙ�@��HB�u�Cp���u?(�@��B��
CxR                                    Bx�a*�  �          @�����Ϳ�@�\)B���Cp@ ����>�Q�@�G�B�k�C!�                                     Bx�a9t  �          @�33�������@��
BtQ�Ce����<��
@�G�B�\C3T{                                    Bx�aH  �          @���33�
=@��
BpffCa���33=L��@���B�B�C2��                                    Bx�aV�  �          @�{�\)����@�z�Bo��C]��\)>B�\@��B�aHC/&f                                    Bx�aef  �          @�z��\��@�33B�C_O\��\?�@��\B�L�C#Y�                                    Bx�at  �          @��\�
�H�Ǯ@��Bz\)CW���
�H?\)@��B��{C%�H                                    Bx�a��  �          @�녿ٙ���33@���B��C`�ٙ�?�@���B�G�C"��                                    Bx�a�X  �          @�z῞�R����@��
B��
C_�῞�R?��@���B��qC�                                    Bx�a��  �          @�����H��  @�
=B��Ca𤿚�H?�ff@�Q�B��C0�                                    Bx�a��  �          @�  �fff�:�H@��
B��CZ�=�fff?˅@�ffB�u�B�B�                                    Bx�a�J  �          @��׿}p��Q�@��
B�aHC[��}p�?�  @��B�33B��R                                    Bx�a��  �          @�=q���H����@�p�B�W
Cw����H?�z�@�B���B��                                    Bx�aږ  �          @��
�0�׿+�@�  B�C`���0��?�=q@�=qB���B��                                    Bx�a�<  �          @��Ϳ�\�k�@�G�B�k�Cq)��\?�{@�ffB�aHB��)                                    Bx�a��  �          @�p�������\@�G�B�  Co�
���?��\@��B�� B�B�                                    Bx�b�  �          @�p���=q��=q@��B�W
CA𤿊=q?�(�@�
=B�k�B�z�                                    Bx�b.  �          @��H�W
=���
@���B�Q�CI+��W
=?�33@��RB�ffB㙚                                    Bx�b#�  �          @�Q�xQ�B�\@��B�#�C?&f�xQ�?�(�@��B���B�k�                                    Bx�b2z  �          @�Q�}p�����@��B�{C9�=�}p�@�
@���B�#�B�L�                                    Bx�bA   �          @��E��aG�@��B��RCD+��E�?�@���B�33B߽q                                    Bx�bO�  �          @�Q�zᾨ��@��RB��CQ�q�z�?�G�@�B���B���                                    Bx�b^l  �          @���<��
��ff@�z�B��)C��\<��
?��@�z�B��RB�                                    Bx�bm  �          @���=#�
���\@�G�B��C���=#�
?�ff@���B��=B��                                    Bx�b{�  �          @�z�=L�Ϳ�33@�Q�B�W
C�Q�=L��?k�@���B�W
B�ff                                    Bx�b�^  �          @�33>��R��G�@�B���C��>��R?G�@���B��B��=                                    Bx�b�  �          @�(����+�@��\B��\C�
��?��@�{B�B��                                     Bx�b��  �          @���
=q�Ǯ@�=qB���CW�
�
=q?У�@��\B���B��H                                    Bx�b�P  �          @�(��˅>��@�(�B�p�C.��˅@�
@�ffBt�B�B�                                    Bx�b��  �          @������=u@�(�B���C1aH���?�(�@��B~=qB�G�                                    Bx�bӜ  �          @�=q���\=L��@��RB��RC1����\?��R@�=qB�\B�aH                                    Bx�b�B  �          @���������@�Q�B��RCW5þ�?�z�@�  B�
=B�33                                    Bx�b��  �          @��ÿ�\�   @�\)B�Q�C`���\?�p�@�G�B��=B�W
                                    Bx�b��  T          @��׽u�\(�@�{B�{C�3�u?���@�z�B�=qB��f                                    Bx�c4  �          @�\)�#�
���
@��RB��C�g��#�
?У�@��RB��=B�z�                                    Bx�c�  �          @�z�=�\)���@�33B�G�C��=�\)?�(�@�p�B�#�B�                                    Bx�c+�  �          @�{>B�\�(�@�z�B��{C���>B�\?�
=@�\)B�.B���                                    Bx�c:&  T          @�z�>��@  @��B�#�C��q>�?�G�@��RB�.B�\                                    Bx�cH�  �          @��>���@�Q�B��C���>�?Y��@�=qB�u�Bv�                                    Bx�cWr  �          @�33>L�Ϳ�  @�  B�C��>L��?�  @��B��B�                                      Bx�cf  �          @�=q=��Ϳ0��@���B��C�{=���?��@��B�B�=q                                    Bx�ct�  �          @�33>�=q�z�H@��B�C��{>�=q?�G�@�\)B�z�B�ff                                    Bx�c�d  �          @�33>�
=��\)@�=qB�33C�޸>�
=?�
=@���B��B�B�                                    Bx�c�
  �          @���>B�\��(�@���B��C�  >B�\?��
@�=qB��B��                                    Bx�c��  T          @��þ#�
���@�  B�B�Cz�q�#�
?�p�@��B�33B�ff                                    Bx�c�V  �          @������@��RB�#�C�^���?�{@��B���B�B�                                    Bx�c��  �          @��R��  �c�
@��
B��C~B���  ?�\)@�=qB��{B�\                                    Bx�c̢  �          @��
�#�
��G�@��HB�
=C��
�#�
?�\)@��
B�\)B���                                    Bx�c�H  �          @�ff��=q���@���B�Cu���=q?��@�  B��qB�
=                                    Bx�c��  �          @�
=��z���@�B��fCj�;�z�?��R@�\)B�\B�(�                                    Bx�c��  �          @�
=��Q쿈��@��HB�  C{aH��Q�?W
=@�z�B���B�k�                                    Bx�d:  �          @��H����(�@���B�k�C�K����?�=q@���B�ǮB�z�                                    Bx�d�  �          @��ͽL�;Ǯ@��
B��C��R�L��?���@���B�\B�aH                                    Bx�d$�  �          @�G�=��   @�  B��RC��{=�?�z�@��\B�B��                                    Bx�d3,  �          @���>aG���Q�@��B�#�C��R>aG�?У�@�Q�B�G�B��{                                    Bx�dA�  �          @��>�(�>L��@��
B�L�A�=q>�(�@��@�B�G�B���                                    Bx�dPx  T          @���>��>�z�@�(�B��
Bff>��@�@���B���B��=                                    Bx�d_  �          @�>.{>8Q�@�p�B��fB;�>.{@(�@�\)B�\)B�                                    Bx�dm�  �          @�p�>��H��Q�@�z�B��)C��=>��H?�@��B�  B�L�                                    Bx�d|j  �          @�p�?   �u@�z�B��RC�J=?   ?��
@�33B��fB�u�                                    Bx�d�  �          @���>�33��@��
B�#�C��{>�33?Ǯ@�p�B��B�ff                                    Bx�d��  �          @��
?�\�.{@�G�B��=C�7
?�\?���@�B�B�.                                    Bx�d�\  �          @��
>�z���@��\B�aHC��3>�z�?�z�@�B�L�B��                                    Bx�d�  �          @�p�>u���@�(�B���C��>u?�@�\)B��{B�\                                    Bx�dŨ  �          @�>B�\����@���B�B�C�H�>B�\?�
=@��B�8RB���                                    Bx�d�N  �          @�=q>Ǯ����@�ffB�C�Ǯ>Ǯ?aG�@��B�
=B�#�                                    Bx�d��  �          @�G�?
=���
@�G�B���C��H?
=>�(�@��B���B�                                    Bx�d�  T          @�
=?�ͿǮ@�{B���C���?��>�33@��B�(�B�                                    Bx�e @  T          @��?���33@�ffB���C�Q�?�?=p�@���B�z�BY�\                                    Bx�e�  �          @�(�>�33�xQ�@�  B��3C��>�33?k�@�Q�B��
B�B�                                    Bx�e�  �          @��>u���\@�B��C�W
>u?�R@���B�B�                                    Bx�e,2  �          @���>�녿��@�(�B�\)C���>��?�@���B���BNz�                                    Bx�e:�  �          @���?0�׿�Q�@�33B�8RC��)?0��?=p�@�B���B<�                                    Bx�eI~  �          @���?(�����@��\B�C���?(�?W
=@��
B�(�BX�H                                    Bx�eX$  �          @�{?W
=��G�@��\B��C���?W
==���@��
B��@׮                                    Bx�ef�  �          @�{?�\)�{@��\B{  C�Z�?�\)��
=@���B��{C���                                    Bx�eup  �          @��?��
��@��By33C���?��
��@��B��qC��H                                    Bx�e�  �          @�z�?
=q�#�
@�ffBsp�C��
?
=q�J=q@�G�B�k�C�(�                                    Bx�e��  �          @�{?��\�:�H@�
=B]\)C��{?��\���H@�ffB�33C��)                                    Bx�e�b  �          @��
?���>{@��BVG�C���?�녿���@��B�C�}q                                    Bx�e�  T          @�p�?����N{@j�HB3Q�C���?��ÿ�\@���Bsp�C��H                                    Bx�e��  �          @�?���X��@|(�B<p�C��?�녿�@��B�=qC�xR                                    Bx�e�T  T          @�{?�G��7�@��B_z�C���?�G���@�ffB��
C�\)                                    Bx�e��  �          @�
=>�=q�?\)@���BbQ�C���>�=q���\@���B�u�C��                                    Bx�e�  �          @�(�>�
=�Q�@�\)BO��C��f>�
=��33@��HB���C�!H                                    Bx�e�F  �          @��H>�
=�J�H@���BT�RC�˅>�
=���@��\B�.C���                                    Bx�f�  �          @�z�?+��\(�@��\BEp�C�z�?+���{@�  B��HC�ٚ                                    Bx�f�  �          @��?0���Fff@���BV\)C�B�?0�׿�(�@��HB��C���                                    Bx�f%8  �          @��H?.{�6ff@�p�BaC���?.{����@��
B���C���                                    Bx�f3�  �          @��?�\)�1�@��B^  C���?�\)��z�@�G�B��\C��                                    Bx�fB�  �          @�G�>��6ff@�(�BbQ�C��>���(�@��HB�\C��
                                    Bx�fQ*  �          @��?5�B�\@�G�BW�C���?5��
=@��B���C�!H                                    Bx�f_�  �          @�p��#�
�C�
@�(�BU��C��ͼ#�
��G�@��B���C��R                                    Bx�fnv  �          @���    �Tz�@��
BLp�C�f    ��G�@��B�\C��                                    Bx�f}  �          @�  <#�
�P  @�(�BOp�C��<#�
��Q�@�\)B�z�C�<)                                    Bx�f��  �          @��=����A�@���BWG�C��=��Ϳ��R@�p�B�B�C�޸                                    Bx�f�h  �          @��=u�HQ�@�33BRp�C���=u����@��B�ǮC��                                    Bx�f�  �          @�ff<��
�B�\@�ffBX��C�1�<��
��p�@�
=B�ǮC�e                                    Bx�f��  �          @�G�=�\)�Z�H@���BF��C��=�\)��z�@�B��3C���                                    Bx�f�Z  �          @��R�#�
�Mp�@�33BO�
C��׽#�
��
=@�B�=qC�P�                                    Bx�f�   �          @�p��L���dz�@n{B8��C��{�L���
=q@�{B��\C�N                                    Bx�f�  �          @�(�>��
�U@w
=BC�C�>��
��33@��B���C��{                                    Bx�f�L  �          @���=�G��X��@x��BCC��
=�G���
=@���B��C��\                                    Bx�g �  �          @�����Q��HQ�@|��BNQ�C�!H��Q��@�  B��C�aH                                    Bx�g�  �          @�  >B�\�L(�@uBI�C��\>B�\��\@�p�B�W
C��                                    Bx�g>  �          @�\)���)��@��
Bd��C��3����
=@�  B�C��                                    Bx�g,�  �          @��R��\)�   @�Q�Bm��C��f��\)�}p�@��HB��C~�                                    Bx�g;�  �          @�
=����R@�(�Bz{C�0����333@��B�z�Cl�q                                    Bx�gJ0  �          @�ff�����p�@��
B��HCjff���=��
@�33B�8RC/��                                    Bx�gX�  �          @��H�z�H�ff@���B{�Ct�q�z�H�
=q@��RB���CP�                                    Bx�gg|  �          @��\�s33��ff@���B���Cr0��s33�aG�@�\)B��
C@��                                    Bx�gv"  �          @�(��s33��\@�
=B���Cq�q�s33�#�
@�G�B�W
C=��                                    Bx�g��  �          @�\)�5��{@�ffB���Cr�Ϳ5>W
=@�z�B��
C#��                                    Bx�g�n  �          @�z�=�Q��G�@�p�B���C�W
=�Q��\@��HB�ǮC�B�                                    Bx�g�  �          @���=�G���33@�Q�B�W
C��=�G���p�@�(�B�\C�K�                                    Bx�g��  �          @�=q>B�\��Q�@�  B���C�.>B�\�.{@���B��HC��=                                    Bx�g�`  �          @��\�����@�  B�C��H����.{@���B��3Cd                                    Bx�g�  �          @�(�    ��{@�B��=C��=    >8Q�@��
B��B��                                     Bx�gܬ  �          @�33>�(���=q@�ffB��C��>�(�>�ff@���B�(�B8(�                                    Bx�g�R  T          @�=q?z῀  @��B�C�{?z�?�\@��B�p�B$��                                    Bx�g��  �          @�
=�8Q쿡G�@�  B�W
C��)�8Q�>W
=@�p�B�B�Cp�                                    Bx�h�  �          @��W
=����@�B�Q�Cj�ÿW
=>u@��\B�W
C#��                                    Bx�hD  �          @��
�!G����@��B��HCw�
�!G��#�
@�=qB��C7��                                    Bx�h%�  �          @��\�^�R��@��RBs33CxW
�^�R�E�@�B�.C]��                                    Bx�h4�  �          @�zΐ33�ff@s�
B^(�Cs�3��33��ff@��B���C^xR                                    Bx�hC6  �          @��ͿУ��8��@J�HB.�HCp�)�У׿�G�@z=qBi��CcT{                                    Bx�hQ�  �          @��H��Q��'�@UB;��CmB���Q쿺�H@~�RBr�C\�3                                    Bx�h`�  �          @����ٙ��"�\@W�B?33ClO\�ٙ�����@\)Bu�C[(�                                    Bx�ho(  �          @�z��  ���@a�BE�
Cjs3��  ��  @��By\)CW�                                     Bx�h}�  �          @�{���
�%�@fffBH�RCo=q���
����@��RB�B�C]Y�                                    Bx�h�t  �          @�33���R��
@j�HBT��Cm
���R����@�ffB���CW�q                                    Bx�h�  �          @�����  ��@q�Ba��Cp&f��  �k�@���B�  CX\)                                    Bx�h��  �          @�ff��{��@x��Bt=qCn�H��{��@���B�\)CO#�                                    Bx�h�f  �          @�Q쿎{�z�@qG�B�L�COǮ��{?
=@qG�B�33C�3                                    Bx�h�  �          @\)�p�׾#�
@vffB�33C=�\�p��?�G�@mp�B�B�C�)                                    Bx�hղ  �          @s33�Q녿c�
@aG�B�  Cc��Q�>aG�@g�B�� C%5�                                    Bx�h�X  �          @xQ�=p���{@\(�B{  CuaH�=p��   @qG�B�ffCV�                                    Bx�h��  �          @�33�����(�@p��Bp�Ci{�����\@�33B���CI}q                                    Bx�i�  �          @�p���
=�y��@*�HB��Cy����
=�:�H@n{BC\)Cs��                                    Bx�iJ  �          @��R��(��o\)@ ��A��Cx�=��(��4z�@`��B?\)Cru�                                    Bx�i�  �          @��\���\�\)@?\)BE
=Cp�{���\��  @aG�B|�C`�)                                    Bx�i-�  �          @h�ÿ�G����@W�B�L�C?�쿡G�?B�\@R�\B��C�                                    Bx�i<<  �          @`  ��=q>���@QG�B�\C$Y���=q?�G�@A�Bu�C�H                                    Bx�iJ�  �          @^�R��=q��G�@S33B��C:���=q?^�R@K�B�C�                                    Bx�iY�  �          @\�Ϳ8Q���@P��B��\C?��8Q�?Q�@J=qB��CW
                                    Bx�ih.  �          @c�
�}p�=u@[�B���C0���}p�?���@P��B��C�{                                    Bx�iv�  �          @k��u>�p�@aG�B�aHC��u?��@P��Bz33B�8R                                    Bx�i�z  �          @j�H���\>���@_\)B�(�C#���\?��@P  B{�HB��H                                    Bx�i�   �          @r�\�z�H���
@h��B�ǮC5@ �z�H?�ff@_\)B�G�C=q                                    Bx�i��  �          @QG���녿z�@>{B�z�CO
=���>��R@@��B�8RC$��                                    Bx�i�l  �          @`  ��p����H@O\)B�k�CI�{��p�>��@O\)B��RC�                                    Bx�i�  �          @h�ÿ��;�ff@VffB��)CF�Ϳ���?
=q@U�B��)CJ=                                    Bx�iθ  �          @i����녽�@]p�B�p�C9�{���?c�
@UB�L�C��                                    Bx�i�^  �          @s�
�Tz�>���@l(�B��fCxR�Tz�?���@Z�HB�
B��)                                    Bx�i�  �          @w���{����@g�B��3C8:΅{?s33@`  B�B�C(�                                    Bx�i��  �          @��ÿ��\    @r�\B��C3�\���\?��@hQ�B�k�CE                                    Bx�j	P  �          @�
=�c�
?��@|(�B�ǮC   �c�
@{@\��B]�Bߣ�                                    Bx�j�  �          @�33�z�H?aG�@�z�B���C
��z�H@G�@n{Bk\)B�=                                    Bx�j&�  �          @����33?p��@�33B�33C�\��33@z�@j=qBd�B�(�                                    Bx�j5B  �          @�zῌ��?s33@��B�u�C(�����@@mp�Bf{B�k�                                    Bx�jC�  �          @�G����
�n{@eB�k�CW޸���
>�@mp�B���C.=q                                    Bx�jR�  �          @�녿�G����@mp�B�u�Cg�ÿ�G��8Q�@z�HB��qC>ff                                    Bx�ja4  �          @�z�z�H��@e�Bpz�Cq�ÿz�H�:�H@|��B��qCX�R                                    Bx�jo�  �          @�{��
=����@r�\B�Cd(���
=�L��@�Q�B���C=�                                     Bx�j~�  �          @����G���@k�Bx\)Cn�쿁G���@�  B��3CQ\)                                    Bx�j�&  �          @���G��c�
@xQ�B��3CW^���G�>k�@~�RB�aHC)�)                                    Bx�j��  �          @����
=��R@���B�  CKs3��
=?
=q@��B��HC^�                                    Bx�j�r  �          @�Q��p���=q@�p�B�ǮC<� ��p�?z�H@�=qB���C�)                                    Bx�j�  �          @��ÿ����
@�
=B�{C6����?�@�=qB��=C�q                                    Bx�jǾ  '          @�
=�˅���@�B�ffC98R�˅?��@���B��RC��                                    Bx�j�d  �          @��
�������@�Q�B�.C=�����?s33@�p�B��3C�                                    Bx�j�
  �          @�Q���Ϳ#�
@���B�ǮCG
����?\)@�G�B�p�C#�                                    Bx�j�  �          @�G������@w�Bkz�CO
=��=#�
@���BzG�C2�)                                    Bx�kV  �          @�=q�����@��\B~��CVs3��<��
@��B�\C3aH                                    Bx�k�  �          @�33���R��@�{B�W
CZ����R<��
@�33B�(�C3:�                                    Bx�k�  �          @�  ��Q��G�@�Q�B�RCb^���Q쾔z�@�Q�B���C?n                                    Bx�k.H  �          @�zΐ33�Y��@�z�B���CXJ=��33>��R@�
=B�k�C$�\                                    Bx�k<�  �          @����}p����@qG�B��fCb���}p�    @z�HB���C3�f                                    Bx�kK�  �          @HQ���=u@C�
B�ffC,����?aG�@;�B�B�B���                                    Bx�kZ:  �          @Tz�녿L��@G�B�.Cj�f��=�\)@N{B��fC,�3                                    Bx�kh�  �          @j=q�5���@U�B���Cr�=�5��G�@e�B���CS�R                                    Bx�kw�  �          @n{�.{�5@c�
B�.Cbff�.{>�=q@g�B�=qCG�                                    Bx�k�,  �          @�����{��33@y��B���Cg����{���R@�(�B��RCC}q                                    Bx�k��  �          @��H��ff��Q�@xQ�B}�
Cd���ff��33@��
B��fCCQ�                                    Bx�k�x  �          @��Ϳ��׿�(�@y��Bz��Cb���׾\@���B��fCCaH                                    Bx�k�  �          @�(���Q��
�H@z=qBf�RCqT{��Q쿇�@�33B���C]�                                    Bx�k��  �          @�\)���\�{@~{Bd�Cp.���\���@�p�B�\)C\��                                    Bx�k�j  �          @�p�������@x��Bbp�CuaH���ÿ�(�@��
B�33Cd�H                                    Bx�k�  �          @���
=���H@�Q�Bl�\Cauÿ�
=�
=@�=qB�#�CGxR                                    Bx�k�  �          @��
���z�@���B_Cn������Q�@��B�G�C\                                    Bx�k�\  �          @�p����\�{@���B^(�Cr� ���\��=q@��B�
=Cb33                                    Bx�l
  �          @�p���ff�&ff@�p�BX
=Co8R��ff��@�ffB�C^�)                                    Bx�l�  �          @����\��33@�{Bm��CZ���\��G�@�
=B�L�C@�                                    Bx�l'N  '          @�33�Q��  @�  Bc�HC[n�Q�
=@��B�(�CC��                                    Bx�l5�  �          @����׿��@�{B[�
C[�R��׿:�H@�G�By\)CE�                                    Bx�lD�  �          @����p��\@�z�B\G�CS�{�p����@�z�Bq�
C=h�                                    Bx�lS@  �          @�G��#�
���@��BbQ�CP�\�#�
�W
=@��
Bs�
C8�                                     Bx�la�  �          @�\)�
=q�G�@���B]�C_
�
=q�\(�@�p�B~�CI�{                                    Bx�lp�  �          @�\)��Q��@��RBf��C^���Q�.{@���B�(�CG.                                    Bx�l2  �          @�녿�
=��H@��BY��Ck@ ��
=���@��HB�� CY�
                                    Bx�l��  �          @�
=�����@���BYp�CkJ=��녿��\@��B�aHCY                                    Bx�l�~  �          @����>�R>Ǯ@@  B3�HC,��>�R?��@3�
B%Q�C
                                    Bx�l�$  �          @����,�;8Q�@Q�BJ{C7�H�,��?&ff@N{BE
=C&k�                                    Bx�l��  �          @��@  ?�G�@B�C!Y��@  ?�  ?�\A�(�CY�                                    Bx�l�p  �          @�G��G
=@:�H>L��@,��C�
�G
=@7��
=q��\)C	B�                                    Bx�l�  T          @�33�0  @W
=���Ϳ�{CB��0  @N�R�p���H(�Cff                                    Bx�l�  �          @����%@W�������HB�\�%@H�ÿ�ff��\)Cs3                                    Bx�l�b  �          @��׿���@�=q��  ��\)BУ׿���@p  �1G��G�B�Ǯ                                    Bx�m  �          @�
=����@�z�������B�G�����@����8���p�B�=q                                    Bx�m�  �          @�33��Q�@�Q쿼(���
B�W
��Q�@�  �&ff��33B�{                                    Bx�m T  �          @�(���{@��H��Q��x��B�\��{@��H�%��(�B�z�                                    Bx�m.�  �          @�33��{@�녿�
=�x(�B�G���{@���$z���B��                                    Bx�m=�  �          @��׿�(�@��R��\)��\)B�8R��(�@�p��.�R��33B��                                    Bx�mLF  �          @��\����@�녿�Q���33B��ÿ���@�{�C�
�G�B��f                                    Bx�mZ�  �          @�Q쿢�\@����ٙ���G�BѮ���\@���4z����B�aH                                    Bx�mi�  �          @�\)��@�Q쿓33�K\)B�Q��@��\�G����B�
=                                    Bx�mx8  �          @�Q��{@����  �/
=B��)�{@����ff��{B�q                                    Bx�m��  �          @�  ��Q�@�Q쿁G��1�B�=q��Q�@�(��Q����B�R                                    Bx�m��  �          @�����p�@�����ff�7\)B��
��p�@�������{B�ff                                    Bx�m�*  T          @���� ��@�Q�=p��  B��� ��@|�Ϳ޸R����B��
                                    Bx�m��  �          @���:�H@�  �(����(�B�=q�:�H@n{�������C {                                    Bx�m�v  �          @��\�)��@��׿0�����B�3�)��@~�R��Q����B�W
                                    Bx�m�  �          @���%�@�G��@  ��B���%�@\)�޸R��G�B��
                                    Bx�m��  �          @���
=@�\)�(����B�z��
=@��R��z���B�                                     Bx�m�h  �          @�(��"�\@��Ǯ���
B�\�"�\@�
=��
=��\)B�{                                    Bx�m�  �          @��(�@��ÿ����HB�\�(�@��ÿ�=q��(�B�W
                                    Bx�n
�  �          @��!�@�Q쾮{�n{B�\�!�@��������xQ�B��)                                    Bx�nZ  �          @�ff�-p�@�p���\)�Dz�B�\�-p�@�\)����h��B�L�                                    Bx�n(   �          @���7
=@��
��Q��~{B�p��7
=@����\)�s33B�                                      Bx�n6�  �          @���<(�@�녾�p�����B���<(�@����{�q��B�#�                                    Bx�nEL  �          @�Q��B�\@��þk�� ��B��R�B�\@����(��V=qB��                                    Bx�nS�  �          @�Q��A�@�G��aG���HB�u��A�@��
���H�T��B���                                    Bx�nb�  �          @����G�@�  �k����B���G�@��H�����R=qB��R                                    Bx�nq>  �          @��
�QG�@�Q�L���
=B���QG�@�33���I��C ��                                    Bx�n�  �          @�33�I��@�=q��=q�6ffB�8R�I��@�zῠ  �W�
B��                                     Bx�n��  �          @�(��O\)@����k���B����O\)@�(������N�RC {                                    Bx�n�0  �          @�p��Tz�@��þ����{B���Tz�@�(���\)�?�C ��                                    Bx�n��  �          @����U�@����\)�=p�C +��U�@��
���
�0(�C                                    Bx�n�|  �          @��QG�@��\�#�
���B����QG�@��R���\�-�B��\                                    Bx�n�"  �          @����W�@��R=L��?�\C ���W�@��
�fff�ffCL�                                    Bx�n��  �          @����XQ�@��R=�?�G�C �XQ�@�(��Tz����CO\                                    Bx�n�n  �          @����[�@��>u@#33C���[�@���333��{C�H                                    Bx�n�  �          @�p��H��@��=��
?L��B���H��@�녿h���
=B��                                    Bx�o�  T          @����@  @���#�
����B��\�@  @��
��G��,Q�B��                                    Bx�o`  �          @���HQ�@�p�>L��@ffB��{�HQ�@�33�J=q��B�ff                                    Bx�o!  �          @�ff�@��@�G�>.{?�B�
=�@��@��R�Tz����B��f                                    Bx�o/�  �          @���'�@�p��.{��B�u��'�@��׿����O
=B��                                    Bx�o>R  �          @�z��
�H@�p��Ǯ��(�B�\�
�H@��R���H�{�B�aH                                    Bx�oL�  �          @����p�@��;Ǯ����B䙚�p�@�ff���H�z�RB�u�                                    Bx�o[�  �          @����!G�@�Q���Ϳ��
B�q�!G�@��
����B�RB��                                    Bx�ojD  �          @��\�'�@�(�=�G�?�(�B���'�@����fff��B��H                                    Bx�ox�  �          @��\�,(�@��\>�?�Q�B��H�,(�@�  �^�R��B�                                    Bx�o��  �          @��
�=p�@�\)>u@#�
B��H�=p�@�p��=p����
B��=                                    Bx�o�6  �          @�p��C�
@�
=>�\)@<��B��R�C�
@��0�����HB�G�                                    Bx�o��  �          @��H�G
=@���?�@�ffB����G
=@�=q�������B��                                     Bx�o��  
�          @����Fff@�Q�>��H@��B�8R�Fff@�Q��(����B��                                    Bx�o�(  T          @���E�@��>Ǯ@�{B�z��E�@��H�������B��3                                    Bx�o��  �          @�(��<(�@�ff?��@�p�B����<(�@�
=��Q��w�B���                                    Bx�o�t  �          @����A�@��?#�
@��B�(��A�@��þ�=q�?\)B���                                    Bx�o�  �          @�
=�K�@��\?333@�B����K�@�(��8Q��   B�8R                                    Bx�o��  �          @�G��J�H@�?.{@�z�B�aH�J�H@�\)�aG��ffB��q                                    Bx�pf  �          @��\�Tz�@�G�?^�RA�HCp��Tz�@�(�    ���
C ��                                    Bx�p  �          @�{�U�@�\)?E�AffC 5��U�@�������z�B��                                    Bx�p(�  �          @����Z=q@���?s33A��C ���Z=q@�(�<�>�33B��R                                    Bx�p7X  �          @�Q��k�@w�?��\AV�RC��k�@���>�G�@��HC8R                                    Bx�pE�  �          @�\)�i��@tz�?�z�An�\C���i��@���?z�@�33C#�                                    Bx�pT�  �          @�  �
=@��\=#�
>�
=B���
=@�\)��G��)B��                                    Bx�pcJ  �          @�=q�(�@��
>8Q�?��B�ff�(�@�G��aG����B�                                      Bx�pq�  T          @�����@�
=>��R@Mp�B�(���@��8Q���ffB�=                                    Bx�p��  �          @�33�2�\@��R?J=qAz�B���2�\@��׾\)���B�\)                                    Bx�p�<  �          @�\)�?\)@�z�?���AG
=B��R�?\)@���>��@3�
B��H                                    Bx�p��  �          @����Dz�@��?�AN{B��\�Dz�@���>��R@XQ�B��{                                    Bx�p��  �          @�  �L��@���?�=qA=G�C � �L��@���>u@%�B�.                                    Bx�p�.  �          @����E�@�ff?W
=AffB�� �E�@��ü����
B�p�                                    Bx�p��  �          @�33�=p�@�>�G�@�  B�k��=p�@�����ffB�p�                                    Bx�p�z  �          @����H��@�G�?(�@�Q�B��q�H��@�=q�aG���B�33                                    Bx�p�   �          @�
=�7
=@��?&ff@�B�8R�7
=@�33�k��#33B��R                                    Bx�p��  �          @�Q��<(�@���?G�AQ�B�\�<(�@�33��Q쿁G�B�33                                    Bx�ql  �          @����   @��\?5@�G�B�Q��   @�(��W
=�z�B�Ǯ                                    Bx�q  �          @����   @��
?333@�33B��f�   @���k��\)B�k�                                    Bx�q!�  �          @�(��J�H@x��?uA-�C&f�J�H@�Q�>.{?�33C Y�                                    Bx�q0^  T          @���=p�@tz�?��AS�B�k��=p�@~{>�Q�@���B�Q�                                    Bx�q?  �          @����1�@qG�?���AY�B����1�@z�H>�p�@��
B��3                                    Bx�qM�  �          @��H�P��@O\)?��A�ffC5��P��@^�R?aG�A&�HC(�                                    Bx�q\P  �          @�
=�W
=@R�\?���A�(�C�{�W
=@b�\?k�A+�C�                                     Bx�qj�  �          @��\�e�@b�\?�(�A�\)CL��e�@s�
?�  A-C5�                                    Bx�qy�  �          @��H�q�@N�R?�z�A�Q�Cu��q�@b�\?�  AX��Cٚ                                    Bx�q�B  �          @���p��@L(�@z�A���C���p��@a�?�Av{CǮ                                    Bx�q��  �          @����n�R@AG�@�AîC�n�R@XQ�?ǮA���C	�=                                    Bx�q��  �          @�G��dz�@4z�@
=qA˅C���dz�@K�?�=qA�Q�C
J=                                    Bx�q�4  �          @�z��h��@7�@p�A�(�C��h��@O\)?�\)A�G�C
W
                                    Bx�q��  �          @����h��@4z�@33A��CG��h��@L��?��HA��C
��                                    Bx�qр  �          @����l(�@4z�@{A�ffC�H�l(�@L(�?��A��HC(�                                    Bx�q�&  �          @�  ��33@33@33AУ�C����33@,(�?���A���C�                                    Bx�q��  �          @��\�vff@"�\@!�A�  C���vff@>�R@   A�
=C@                                     Bx�q�r  �          @���e�@Tz�@33A��C	+��e�@l��?��A���C)                                    Bx�r  �          @���u�@Z=q@  A���C
L��u�@q�?���A�(�Ck�                                    Bx�r�  �          @�����(�@p�@ ��A�=qC���(�@8��@   A��C��                                    Bx�r)d  �          @����
?�p�@333A��HC� ���
@{@�A���CT{                                    Bx�r8
  �          @�z����@
=q@(Q�A�=qC�����@'
=@�A�{C�                                     Bx�rF�  �          @�33��ff@�@%A��C����ff@1G�@
=A�
=Cp�                                    Bx�rUV  �          @��H��{@Q�@#33A���C���{@3�
@�
A��C�                                    Bx�rc�  �          @����@#�
@"�\A�\)CL���@?\)@G�A��Cu�                                    Bx�rr�  �          @����  @,��@��A��C^���  @Fff?�z�A��HC��                                    Bx�r�H  �          @�����@3�
@ ��A�(�C�����@N{?��HA�=qCaH                                    Bx�r��  �          @�  ��@5@(�A��C����@O\)?��A�  CO\                                    Bx�r��  
�          @�\)���R@)��@!�A�=qC�����R@Dz�@   A���C�                                    Bx�r�:  �          @�{��ff@
=@1G�A�Q�CaH��ff@4z�@33A��C
=                                    Bx�r��  �          @�����
=?�=q@I��B�C!}q��
=@�@4z�A�\C                                    Bx�rʆ  �          @��
������@]p�B��C9ٚ���>#�
@_\)B\)C2�                                    Bx�r�,  �          @�����
��z�@b�\BC7�����
>�33@b�\B�C/�                                    Bx�r��  �          @�p�����>��@P  B�C1  ����?W
=@I��B�\C*T{                                    Bx�r�x  �          @�{�����
=@o\)B�
C;T{����=�Q�@r�\B
=C2�
                                    Bx�s  �          @�����  @s33B��C@� �����R@z�HB%��C8�                                    Bx�s�  �          @�  ��녿�
=@l��Bz�CB����녿�\@vffB�C:T{                                    Bx�s"j  �          @�����=q>8Q�@{�B"�
C1�R��=q?^�R@uB(�C)(�                                    Bx�s1  �          @�z���
=�.{@���B+ffC6!H��
=?�@��B)��C,                                    Bx�s?�  �          @�  ����>�z�@��B&��C0p�����?��\@���B!\)C'�                                    Bx�sN\  �          @�����>\)@���B'�C2Y�����?c�
@�{B#=qC)}q                                    Bx�s]  �          @љ���z�>���@��B%�C0{��z�?�=q@�{BC'��                                    Bx�sk�  �          @�Q�����?(�@�  B"�HC,޸����?��@��\B
=C$��                                    Bx�szN  �          @������?�@�{B*=qC-�R����?��@���B"�
C$��                                    Bx�s��  �          @�G����
?O\)@���B#p�C*�����
?��@�=qB{C"}q                                    Bx�s��  �          @�����z�?���@�33B�C$����z�@�\@s33B�Cff                                    Bx�s�@  �          @�G�����?���@{�B��C������@{@c�
B�CB�                                    Bx�s��  �          @љ���(�@   @w
=B�RC�R��(�@(Q�@]p�A���C��                                    Bx�sÌ  �          @��H��(�@��@uB�\C�q��(�@4z�@Y��A��C��                                    Bx�s�2  �          @Ӆ��z�@�
@{�B
=C&f��z�@,��@aG�B �HC�                                    Bx�s��  �          @��
���@{@y��Bp�CxR���@6ff@^{A��RC��                                    Bx�s�~  �          @�ff���
@"�\@tz�B=qCs3���
@I��@UA�33C�                                    Bx�s�$  �          @�Q����@(�@|��BCE���@5�@aG�A���Cu�                                    Bx�t�  T          @׮���@
=@w
=BC����@>�R@Z=qA�
=C33                                    Bx�tp  �          @�  ��p�@=q@z=qB33Cٚ��p�@B�\@\��A���CO\                                    Bx�t*  �          @����Q�?c�
@���B>(�C(����Q�?��H@���B3�C5�                                    Bx�t8�  �          @�=q��{?
=@�z�BC�RC,n��{?�Q�@�
=B;33C"�                                    Bx�tGb  �          @ڏ\���?0��@�33BAG�C+Q����?\@�p�B8(�C!E                                    Bx�tV  �          @أ����R?�{@��RB<�RC&����R?�@��RB0�C�                                    Bx�td�  �          @�  ����?�  @�z�B.C�����@\)@�G�B�C@                                     Bx�tsT  �          @��H��\)?�\)@���B&�Ck���\)@%@��B�\CL�                                    Bx�t��  �          @أ����@p�@�  B  C.���@E�@b�\A��RC�f                                    Bx�t��  �          @�Q����H@33@�Q�B'��C�R���H@0��@��B33C                                    Bx�t�F  �          @�����?�@��B4�\C����@#�
@�ffB#��C�                                    Bx�t��  �          @�ff���@P��@S33A�RC޸���@p  @.�RA���C
=                                    Bx�t��  �          @�  ���R@Z�H@Z�HA�C�q���R@{�@4z�Ař�C&f                                    Bx�t�8  �          @�G����@Tz�@mp�B��C@ ���@w�@HQ�A��
C�3                                    Bx�t��  �          @�ff��Q�@Tz�@r�\B�\C{��Q�@x��@L��A�Q�C�q                                    Bx�t�  �          @�����G�@^�R@q�B{C���G�@�G�@J�HA֏\C�f                                    Bx�t�*  �          @����@Z=q@~{B	�HC���@�  @XQ�A�{CxR                                    Bx�u�  �          @�Q���  @P  @�Q�B\)C�f��  @u@\(�A���C                                    Bx�uv  �          @�  ��p�@G
=@�p�B�C\)��p�@n�R@hQ�A�CaH                                    Bx�u#  �          @�\)����@G
=@�p�B�\CǮ����@qG�@xQ�B�Ck�                                    Bx�u1�  �          @�  ��G�@<(�@�z�B'{C���G�@h��@��
B�C��                                    Bx�u@h  �          @�����  @7
=@��B��C����  @`  @s�
Bp�C��                                    Bx�uO  �          @�����\@Mp�@�\)B  C}q���\@w�@z�HB�C
8R                                    Bx�u]�  �          @�Q���p�@5�@��
B  C)��p�@\(�@hQ�A�G�C�                                    Bx�ulZ  �          @�
=��{@=q@���BC���{@C33@w
=B\)CB�                                    Bx�u{   �          @�������@%@�Q�Bz�C������@N{@s33B�
C                                      Bx�u��  �          @޸R��(�@�@�
=B �\C����(�@0  @�33B�RC�)                                    Bx�u�L  �          @ᙚ���@33@�\)BQ�C�R���@=p�@�=qB�C�                                    Bx�u��  �          @ᙚ���@�@�\)B�
CaH���@6ff@��HB�
CJ=                                    Bx�u��  �          @��
��@G�@�33B!{CB���@<��@�{B�C
                                    Bx�u�>  �          @�=q���@�@��HB"(�C�{���@=p�@�{BffC��                                    Bx�u��  �          @������@%@�33B   C����@N{@y��B�HCW
                                    Bx�u�  �          @�p���Q�@,��@�=qB"G�C���Q�@Tz�@w
=B
=C\                                    Bx�u�0  �          @�  ��?�  @�33B �\C#
��@��@�=qBp�C�                                    Bx�u��  �          @����33?�=q@�p�B!�RCQ���33@�R@��HB�\C�                                    Bx�v|  �          @ָR���?���@�Q�B�Cp����@(�@{�B�\CJ=                                    Bx�v"  �          @�ff��=q?�z�@�\)B�HCff��=q@!G�@x��BffC\)                                    Bx�v*�  �          @׮���
?�Q�@��B'�C:����
@%@���BQ�C��                                    Bx�v9n  �          @�{���\?�(�@�ffB'  C�����\@'�@�33B�\C:�                                    Bx�vH  �          @�{��?�{@�\)B�HC!�)��@�R@|(�B\)C��                                    Bx�vV�  �          @�
=���?�ff@��
B�C .���@��@s�
B
=qCaH                                    Bx�ve`  �          @ָR��ff@
=q@~{BQ�Cff��ff@.�R@g
=B(�C!H                                    Bx�vt  �          @׮���\@   @}p�BQ�C�����\@C�
@c33A���C�f                                    Bx�v��  �          @�\)���H@&ff@w
=B=qC�=���H@H��@[�A��
C�                                    Bx�v�R  T          @�{��Q�@9��@j�HB�\C����Q�@Z=q@L��A�CaH                                    Bx�v��  �          @�ff����@<��@q�B	�C������@^�R@S�
A�\)C:�                                    Bx�v��  �          @�  ��Q�@J=q@xQ�BC����Q�@l(�@XQ�A�RC�f                                    Bx�v�D  �          @أ����H@=p�@�  B  C�����H@c33@q�B�
C��                                    Bx�v��  �          @�=q���@;�@�33B��C����@a�@xQ�B
�C��                                    Bx�vڐ  �          @׮���\@S�
@~{B(�C�)���\@vff@\(�A��C
Y�                                    Bx�v�6  �          @�=q���@W
=@�G�B(�Ch����@z=q@`��A��C
#�                                    Bx�v��  �          @�p����R@J�H@��BQ�C�����R@p  @o\)B��C�                                    Bx�w�  �          @��
���@:�H@�z�B��Cz����@aG�@z�HB=qC^�                                    Bx�w(  �          @�����\@e@z=qB��C�����\@��
@W
=A��C
\                                    Bx�w#�  �          @��H��33@u@r�\BG�C&f��33@��H@L��AָRC�f                                    Bx�w2t  �          @�33��  @{�@u�B��C
����  @�@N�RA���Ck�                                    Bx�wA  �          @�z���\)@�(�@n{A�G�C	W
��\)@��
@EA�p�C#�                                    Bx�wO�  	[          @�
=��
=@��@dz�A�\)C\)��
=@��
@:=qA�(�C�                                     Bx�w^f  T          @���@��\@\(�A�
=C���@���@0��A��Cp�                                    Bx�wm  �          @���33@�{@L��A�ffC�)��33@�33@   A�C�                                     Bx�w{�  �          @�{��G�@HQ�@�
=B Q�C�)��G�@n�R@~�RB
=C
�R                                    Bx�w�X  �          @�=q��
=@K�@��RB��C����
=@qG�@~{B�RC��                                    Bx�w��  �          @�33��
=@>{@�z�B#p�C^���
=@fff@�BQ�C.                                    Bx�w��  "          @�(���33@8��@�33B ��C�
��33@`��@���B�\C�                                    Bx�w�J  	�          @�(���Q�@1�@���BG�C� ��Q�@XQ�@��\B�
C�H                                    Bx�w��  
�          @�(����@;�@�{B(�CB����@aG�@\)B�Cc�                                    Bx�wӖ  �          @�p���(�@G
=@�  B�C���(�@mp�@���B�\CQ�                                    Bx�w�<  
�          @�p���33@|(�@�33B�C#���33@�  @p��A���CO\                                    Bx�w��  
�          @������@c�
@�B�\Cs3���@�z�@x��B\)C+�                                    Bx�w��  �          @��
���@P��@�G�B�\C�����@vff@���B
��C
��                                    Bx�x.  T          @�����@H��@�G�BQ�Cc����@o\)@��B
=qC�H                                    Bx�x�  "          @�ff��ff@U@��BG�C)��ff@{�@���B	Q�C
�                                    Bx�x+z  T          @�\)��
=@=p�@�33B33C����
=@dz�@��BG�C�                                    Bx�x:   �          @陚��Q�@:�H@�ffB ffCk���Q�@b�\@�Q�BCY�                                    Bx�xH�  T          @�G����H@?\)@��B�
CL����H@e�@��B	=qC}q                                    Bx�xWl  "          @�������@I��@�  B�
C�\����@o\)@���B�\C�3                                    Bx�xf  �          @�  ��G�@L(�@���B�RCQ���G�@p��@{�B\)C޸                                    Bx�xt�  T          @�R����@K�@�Q�B�HC�f����@p��@�G�BG�C
=                                    Bx�x�^  
�          @�
=��(�@G�@��\Bz�C���(�@mp�@��
B
=CG�                                    Bx�x�  "          @�{��  @K�@��B��C� ��  @q�@�z�B��C�                                    Bx�x��  �          @�
=��p�@Z�H@�=qB
=CG���p�@�  @�=qB	(�C	�
                                    Bx�x�P  
�          @�ff��Q�@h��@���BffCxR��Q�@�
=@\)Bz�CL�                                    Bx�x��  T          @��
���@H��@��BG�C����@l��@y��B{C��                                    Bx�x̜  �          @�33���@W�@�B�C�����@{�@|(�B�C
E                                    Bx�x�B  "          @�z���(�@Q�@��\B�RC(���(�@w
=@�33B\)C
�
                                    Bx�x��  
�          @�G���(�@]p�@�Q�Bp�C�f��(�@�  @p��B��C	�
                                    Bx�x��  �          @����\)@e�@�G�B�C^���\)@��H@a�A�\)C	�f                                    Bx�y4  	�          @�Q���p�@e�@��HB  C�q��p�@��H@e�A�{C	5�                                    Bx�y�  
(          @�G�����@Z=q@���Bz�C)����@{�@j=qA���C(�                                    Bx�y$�  T          @�=q��\)@_\)@�ffB��C���\)@���@mp�A���C
#�                                    Bx�y3&  	�          @����ff@]p�@�
=B=qC���ff@\)@n�RA��C
�                                    Bx�yA�  
�          @�����\@W�@�B(�C�f���\@x��@l��A���C��                                    Bx�yPr  O          @ᙚ��Q�@a�@��\B�
C�3��Q�@�G�@e�A���C
0�                                    Bx�y_  �          @ᙚ��{@o\)@�Q�B��C
8R��{@���@o\)B   CxR                                    Bx�ym�  �          @��H���R@qG�@�G�B�HC
0����R@�G�@p��B �Cp�                                    Bx�y|d  �          @���{@P  @�33B\)CW
��{@r�\@x��BffC�                                    Bx�y�
  
�          @�R����@aG�@�33Bz�CB�����@��@w
=B\)C
E                                    Bx�y��  �          @�p���
=@n{@�{B  C:���
=@�\)@j�HA�ffC�\                                    Bx�y�V  �          @�
=���@�ff@���B  CB����@�{@\��A���C!H                                    Bx�y��  
Z          @�\)���@|��@���B�C
!H���@��R@g
=A�G�C�3                                    Bx�yŢ  
�          @�������@�
=@�33B	ffCL�����@�
=@aG�A��C#�                                    Bx�y�H  �          @�
=��
=@���@�Q�B
=C����
=@�  @[�A��HC�)                                    Bx�y��  T          @�R���@��H@~�RB  C� ���@�=q@X��A�=qC�=                                    Bx�y�  T          @�=q�y��@�{@xQ�BG�CE�y��@���@Q�A�G�C }q                                    Bx�z :  �          @��
�n{@�z�@e�A�(�B�u��n{@��@;�A£�B���                                    Bx�z�  �          @�=q�j�H@��
@dz�A�\B�  �j�H@�G�@:�HA���B��=                                    Bx�z�            @�\�q�@��@j=qA���C �)�q�@�p�@A�A�ffB�p�                                    Bx�z,,  T          @�z��o\)@���@g
=A�RB���o\)@�=q@<��A�p�B�.                                    Bx�z:�  
(          @����g
=@�(�@[�A�B�G��g
=@���@0  A��B�Q�                                    Bx�zIx  �          @���vff@��\@g
=A�z�C ���vff@�  @>{A�(�B��=                                    Bx�zX  �          @�ff���\@�=q@q�A��C�=���\@�Q�@J�HA���C.                                    Bx�zf�  
�          @�����@�@q�A�z�C�����@�(�@L(�Aљ�C8R                                    Bx�zuj  �          @�
=��  @�=q@z=qB33C����  @���@U�A�  C�f                                    Bx�z�  "          @���  @���@���BC�
��  @�  @\��A�\)C�
                                    Bx�z��  "          @�Q����@�33@q�A���C(����@�G�@J�HA��C��                                    Bx�z�\  �          @���}p�@��H@l��A�
=CE�}p�@�Q�@C�
AǅB��
                                    Bx�z�  T          @�  �aG�@���@k�A�=qB��q�aG�@�=q@@��A�  B���                                    Bx�z��  �          @����:=q@��H@hQ�A�B���:=q@�  @9��A�=qB��                                    Bx�z�N  
�          @����8��@���@n�RA��B�B��8��@�ff@@��A�B��)                                    Bx�z��  �          @��H�AG�@�p�@{�BQ�B�G��AG�@��
@N{A�  B�                                     Bx�z�  
�          @�=q�S33@�z�@�G�B{B�ff�S33@�33@W�A�G�B���                                    Bx�z�@  "          @����X��@���@�Q�B��B��R�X��@�Q�@VffA�
=B�.                                    Bx�{�  �          @�z��Z�H@���@�G�B�B���Z�H@��
@W
=A���B��                                    Bx�{�  T          @�33�\(�@��
@�  BB��R�\(�@��\@U�AׅB�B�                                    Bx�{%2  �          @陚�Z=q@�Q�@�=qB��B��{�Z=q@�
=@Z�HA�p�B��                                    Bx�{3�  
�          @���`��@���@|(�B��B��\�`��@�  @R�\A�B�\                                    Bx�{B~  �          @��H�j�H@�  @{�B\)B����j�H@�{@Q�A�=qB�                                    Bx�{Q$  �          @�Q��p��@�33@w�B{B���p��@�G�@O\)Aԏ\B��q                                    Bx�{_�  
(          @��o\)@���@{�BffB��H�o\)@�  @S�
A�\)B���                                    Bx�{np  Y          @�R�w
=@���@g
=A�RC 0��w
=@��@>�RA�p�B���                                    Bx�{}            @陚�\)@�ff@c�
A�{C ޸�\)@�33@;�A�G�B�k�                                    Bx�{��  �          @���{@��H@qG�A�(�C�f��{@�Q�@I��Aȣ�C ��                                    Bx�{�b  �          @陚����@���@]p�A��HCٚ����@��@6ffA�{C��                                    Bx�{�  T          @�
=��Q�@��@N{A��
C@ ��Q�@�
=@&ffA���C=q                                    Bx�{��  "          @�Q����H@�G�@L��A��
C
���H@�z�@#�
A��B�ff                                    Bx�{�T  "          @�33��ff@���@|(�B  C����ff@�  @X��A��C!H                                    Bx�{��  
�          @�����(�@���@w�A��RC	T{��(�@��R@Tz�Aՙ�C�                                    Bx�{�            @�p�����@�{@\)B��C	�q����@���@\(�A�33C
=                                    Bx�{�F  �          @�(���33@���@w�A�G�C	���33@�
=@Tz�A�{CO\                                    Bx�| �  T          @�p�����@��H@��\B��C\)����@��@aG�A��Cp�                                    Bx�|�  T          @�
=���
@���@~{B ��C�����
@�  @X��A�Q�C!H                                    Bx�|8  �          @���z�@��@\)Bp�C33��z�@��@[�A�z�C\)                                    Bx�|,�  �          @��H��@�@��B�C����@�z�@dz�A���C��                                    Bx�|;�  �          @�z���
=@�=q@�  B�CY���
=@���@l��A�ffCO\                                    Bx�|J*  "          @���  @��@���B��C�f��  @�=q@\(�AۮC��                                    Bx�|X�  �          @���  @�(�@w�A�
=C���  @�=q@P  A�=qC ��                                    Bx�|gv  �          @�������@���@�\)B�C������@��@|��A��HCc�                                    Bx�|v  �          @����@��@�\)B�CQ����@�p�@z�HA�Q�C5�                                    Bx�|��  �          @��H���\@��@�\)B�Cu����\@�33@�p�B\)C�                                    Bx�|�h  �          @�\�~�R@�p�@��B!�RC�R�~�R@�\)@��
B
=C)                                    Bx�|�  
�          @�\��z�@~{@�ffB#  C5���z�@�G�@�B�CaH                                    Bx�|��  �          @�
=���R@���@�{B  C�)���R@���@x��A�p�Ck�                                    Bx�|�Z  �          @�\����@P  @�=qB2\)C�f����@w�@�z�B �C
                                    Bx�|�   �          @��H���@�33@�Q�B�C����@���@�\)BC\)                                    Bx�|ܦ  �          @�\�y��@U@�G�B<�Ch��y��@~�R@�33B)��CaH                                    Bx�|�L  �          @�33�e@:�H@���BR
=C�f�e@g�@�(�B?ffC�q                                    Bx�|��  �          @��
�Vff@{@�
=Bh��Cs3�Vff@?\)@��BW��C
O\                                    Bx�}�  �          @���W�@��@�G�Bb�RC���W�@H��@��RBQ33C	
                                    Bx�}>  �          @�
=�^{@*�H@�G�BXQ�Ck��^{@XQ�@�BF33C�                                    Bx�}%�  �          @�G��Z�H@'
=@�z�BWQ�C���Z�H@S33@���BEQ�C�                                    Bx�}4�  �          @�Q��S�
@Q�@�  B_
=C@ �S�
@E@�p�BMz�C	�                                    Bx�}C0  �          @�R�A�@  @�(�Bi��Cn�A�@>{@�=qBW�C�
                                    Bx�}Q�  
�          @��S�
@	��@�ffBf33C��S�
@8Q�@���BU�C
�                                    Bx�}`|  �          @���O\)@
=@ǮBe=qC���O\)@E@��BS�CQ�                                    Bx�}o"  �          @��U�@�R@��B`�CW
�U�@L��@�=qBN=qC#�                                    Bx�}}�  �          @�\)�]p�@,(�@��BXffC
�]p�@Y��@�{BF33C}q                                    Bx�}�n  �          @���J=q@z�@���Bh�C���J=q@C33@��RBVQ�C�                                    Bx�}�  �          @����C�
@��@ƸRBi��C���C�
@?\)@�z�BX
=C�f                                    Bx�}��  �          @�\�HQ�@p�@��Bdz�C��HQ�@K�@�=qBR
=C}q                                    Bx�}�`  �          @�(��?\)@��@˅Bn(�C��?\)@@  @���B\  Cٚ                                    Bx�}�  �          @���B�\@��@�G�Bm��C���B�\@8Q�@�\)B\{C�=                                    Bx�}լ  �          @���AG�@�@�  Bl��C&f�AG�@:�H@�{BZ�C�                                    Bx�}�R  �          @���G
=@p�@�{Bi�C���G
=@;�@�(�BX
=C�                                    Bx�}��  �          @���C33@  @\Bh=qC���C33@=p�@�Q�BVffC�\                                    Bx�~�  �          @���<��@��@��
Bj�HC���<��@>�R@��BX��C�                                     Bx�~D  �          @�\)�9��@33@ƸRBl��C���9��@A�@�z�BZ33C�                                     Bx�~�  �          @�
=�7�@��@ǮBo��C�{�7�@;�@�B]Q�Cs3                                    Bx�~-�  �          @�
=�1G�@��@�G�Br�
C���1G�@;�@��B`(�CO\                                    Bx�~<6  �          @�ff�!�?�G�@�  B�B�C{�!�@!�@ǮBpC�R                                    Bx�~J�  �          @����`��@��@��BX�C�3�`��@E@��\BH
=C
�3                                    Bx�~Y�  �          @�R�l��@-p�@��BI�HC�l��@W
=@�{B8p�C	�                                    Bx�~h(  �          @��
�qG�@5@�=qBA�C���qG�@]p�@�{B0
=C	u�                                    Bx�~v�  �          @��
��Q�@c�
@���B�C���Q�@���@fffA�z�C�f                                    Bx�~�t  �          @�  ��
=@�Q�?�{A�HCO\��
=@��?�@���C�                                    Bx�~�  �          @�  ��\)@�Q�
=��\)CǮ��\)@����33�p�Ch�                                    Bx�~��  �          @ۅ���@��?���A\��C�����@�\)?�{A\)Cn                                    Bx�~�f  �          @�\)��Q�@l(�@^{A�C33��Q�@��\@@  A��HCc�                                    Bx�~�  �          @�ff��{@C�
@��BCc���{@c�
@}p�BQ�C=q                                    Bx�~β  �          @߮���@>�R@�Q�B!33C����@`  @��
B�C�)                                    Bx�~�X  �          @����
=@>�R@��B*=qC�{��
=@aG�@�33Bp�C(�                                    Bx�~��  �          @�\��\)@J�H@��RB&��C!H��\)@n{@�G�B33C
�R                                    Bx�~��  �          @�G���Q�@;�@���B*�HC}q��Q�@^�R@�z�BQ�C�                                     Bx�	J  �          @�\��\)@/\)@��B(Q�C����\)@R�\@��
B�C��                                    Bx��  �          @��
��33@%@�  B'��C����33@I��@���B33C��                                    Bx�&�  �          @��H���?u@�
=BW�\C&�����?��@�=qBO(�C�                                    Bx�5<  �          @�33�}p�?�Q�@���BY��C#O\�}p�?�\)@��HBO�RC�                                    Bx�C�  �          @��H�|��?�ff@�\)BX=qC!�|��?�p�@���BM��CY�                                    Bx�R�  
�          @�\���?�
=@��
BR33C �����@ff@��BG33C�f                                    Bx�a.  �          @�p�����?��R@���BP33C .����@
=q@�BE
=Cp�                                    Bx�o�  �          @������?�@�z�BP{C$�
���?�@��RBF�
C��                                    Bx�~z  �          @��
����?5@��BV��C*E����?�33@��BP  C!c�                                    Bx��   �          @�Q���33?W
=@�(�BUffC(u���33?�G�@�\)BMC�=                                    Bx���  
Z          @�����?\)@��BW{C,E���?��R@���BQ33C#@                                     Bx��l  �          @�\���H>�  @���BZ�C0s3���H?s33@�ffBV�C&�                                    Bx��  �          @���(�>�
=@���BX��C.(���(�?�\)@�p�BS�HC$޸                                    Bx�Ǹ  "          @�\���H>��R@���BZ=qC/�����H?�  @�{BU�
C&5�                                    Bx��^  T          @������?��@�BJ�HC%����?޸R@�Q�BB33C��                                    Bx��  �          @�\)����?�(�@�  B7��C�����@z�@�Q�B,��C�
                                    Bx��  "          @����
?�G�@�33BD��C �����
@	��@�(�B9�HCٚ                                    Bx��P  
�          @����\)?��@��
BO=qC#���\)?��H@�p�BE=qC!H                                    Bx���  N          @�p����\?(��@��\BZ�
C*�
���\?�{@��RBT�C!��                                    Bx���  T          @�z�����?�G�@��BEz�C'p�����?�z�@�  B=��C�H                                    Bx��.B  T          @�p�����?�  @��BK�HC'@ ����?�@���BC��CB�                                    Bx��<�  T          @����
?�33@��BA��C&{���
?��@�B9�C�{                                    Bx��K�  	�          @��
��G�?fff@���BE�C(�q��G�?Ǯ@�  B>\)C!{                                    Bx��Z4  
�          @�33��
=?(�@�BTz�C+����
=?�ff@��BNQ�C"�                                    Bx��h�  
�          @�����>�G�@�ffBH�C.n����?�{@�33BCffC&W
                                    Bx��w�  "          @������H>��R@�
=BG��C0&f���H?z�H@�z�BC�HC(                                      Bx���&  �          @�p����>�33@�Q�BV33C/W
���?�ff@�BQ�C&�                                    Bx����  �          @�ff���
?5@��HBY�\C*+����
?�
=@��RBR�C �                                    Bx���r  
�          @�{��  ?�z�@��BZp�C#�{��  ?�\)@�p�BPz�C�3                                    Bx���  T          @�p�����>��R@��BTC/������?��\@��BPp�C&��                                    Bx����  "          @���p�?W
=@��
BXffC(����p�?Ǯ@�
=BP��C�                                     Bx���d  T          @�R����>�Q�@��RB_�C.�
����?��@��
BZ��C$�                                    Bx���
  T          @�ff��(�?8Q�@��
BY��C*���(�?�Q�@��BR�
C �\                                    Bx���  �          @�p�����?�
=@���BW�HC#Ǯ����?��@��BM�HC
=                                    Bx���V  
�          @����?}p�@�G�BW��C&u����?ٙ�@�(�BN��C��                                    Bx��	�  
�          @����
=?���@�{BR�C%aH��
=?��@�Q�BH�C�                                    Bx���  �          @�p���33?fff@�=qBXC'����33?�\)@��BPffCs3                                    Bx��'H  
�          @���{?�  @��BTp�C&����{?ٙ�@�=qBKC��                                    Bx��5�  
�          @���
=?p��@�\)BS�
C'n��
=?��@�=qBK�C�3                                    Bx��D�  
�          @�{���?p��@���BVQ�C'=q���?�33@��
BM�
CQ�                                    Bx��S:  �          @�R����?�Q�@�BO�C$s3����?��@�\)BF=qC0�                                    Bx��a�  
�          @�z��|��>�ff@�p�B`ffC-xR�|��?�Q�@�=qBZ��C#J=                                    Bx��p�  T          @�33��G�?#�
@���B[=qC+\��G�?�{@�BTffC!u�                                    Bx��,  
�          @����?��@���BZ��C+�
���?���@�BT33C!�q                                    Bx����            @�=q��=q?0��@��BY
=C*W
��=q?�z�@��BQ�HC �3                                    Bx���x  �          @�=q�s33����@�ffBe�RC5�\�s33?#�
@�p�Bc�RC*c�                                    Bx���  �          @�G��\)?
=@���B\(�C+�)�\)?��@���BUz�C!ٚ                                    Bx����  T          @����=q?aG�@��BV��C'����=q?�=q@�  BNQ�C�                                     Bx���j  T          @�  ����?p��@���BW\)C&������?�33@�\)BN�C�R                                    Bx���  "          @�Q���  ?p��@�BX�\C&Ǯ��  ?�33@���BO�C��                                    Bx���  �          @�����?Y��@��BW(�C(5����?�ff@�Q�BN��C�                                    Bx���\  �          @ᙚ��=q?\(�@�{BWffC(��=q?�=q@�G�BO
=C�\                                    Bx��  "          @�=q��{?E�@�z�BS��C)�f��{?�(�@�  BLG�C ��                                    Bx���  �          @�=q��{?:�H@���BT�C*#���{?�Q�@�Q�BL�
C!
=                                    Bx�� N  "          @�G���z�?W
=@�(�BTffC(�{��z�?�ff@�
=BL\)C��                                    Bx��.�  "          @ᙚ��z�?:�H@��BU�RC)����z�?���@���BNG�C ��                                    Bx��=�  �          @������?
=@�ffBYG�C+�q����?��@��\BR��C"�                                    