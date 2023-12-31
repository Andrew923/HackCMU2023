CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230506000000_e20230506235959_p20230507021808_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-05-07T02:18:08.308Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-05-06T00:00:00.000Z   time_coverage_end         2023-05-06T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx~�@  
�          @r�\����K��O\)�D��Coh����� �����z�Ci{                                    Bx~���  �          @p�׿�Q��A녿������RCmn��Q��  �33��RCeQ�                                    Bx~�،  "          @p�׿�33�P�׿W
=�N=qCs&f��33�#�
���p�Cm5�                                    Bx~��2  
�          @hQ��33�7
=�(���.�\Clk���33�녿������Cf5�                                    Bx~���  
�          @^{�C�
�˅�L���W
=CO� �C�
����aG��lQ�CK��                                   Bx~�~  �          @XQ��A녿����
���RCM!H�A녿�녿h���{\)CH��                                    Bx~�$  �          @Z�H�<(�����ff��33CQ�{�<(���ff�����CK��                                    Bx~�!�  T          @Vff�6ff��Q��ff��  CR�R�6ff���ÿ�33��z�CL�                                    Bx~�0p  �          @]p��=p���  ��\)���\CR�{�=p�������G����RCN!H                                    Bx~�?  "          @[��8�ÿ��H�#�
�,��CR�f�8�ÿ�G���=q��ffCK��                                    Bx~�M�  
�          @Y���5��(��333�?\)CSE�5���R������HCK�f                                    Bx~�\b  
�          @aG��%��(��h���o�C\Y��%���=q��\��
=CSh�                                    Bx~�k  �          @e�p��(�ÿxQ��|��Cf��p��������
=qC]z�                                    Bx~�y�  �          @i����
�*=q�h���h��Ce\��
���R��p��{C\�
                                    Bx~�T  
(          @a��\)�&ff�Q��W33Ce@ �\)��p���\)� =qC]}q                                    Bx~��  T          @dz����!G������=qCc޸�녿��33��CZh�                                    Bx~楠  
�          @i����\�%������
Cds3��\���
�H��
CZk�                                    Bx~�F  �          @hQ�����(�ÿ}p��~�HCek���׿�����\�	��C\�3                                    Bx~���  �          @l(�����(�ÿh���d��Cc����ÿ�p���(��\)C[��                                    Bx~�ђ  "          @c�
�/\)�(���p���  CZ���/\)�����
����CUh�                                    Bx~��8  T          @i���@��� �׾�\)��ffCU���@�׿�Q쿐������CQB�                                    Bx~���  T          @q��R�\���þ�����\)CP��R�\��  ��=q���CLp�                                    Bx~���  T          @���c33��Q��(��\CP���c33�Ǯ��p�����CK�3                                    Bx~�*  �          @|(��Z=q��p��Tz��DQ�CN��Z=q������  ��Q�CG^�                                    Bx~��  �          @��b�\��33����l(�CP:��b�\��  ���
�ʣ�CG�                                     Bx~�)v  T          @�33�O\)�\)��ff�o�CV���O\)�Ǯ�����CM�3                                    Bx~�8  T          @��P  ��Q���x(�Cp�)�P  ���������Ci�{                                    Bx~�F�  �          @�z��G���  �������Ct\)�G����\��
=�#p�Cm)                                    Bx~�Uh  �          @�=q�<(���ff�\)����Cu���<(���������(33Cn=q                                    Bx~�d  �          @���=p��ָR�����RCv.�=p���  ��33�$�HCo\)                                    Bx~�r�  �          @��R�X����p��p����
Cr+��X����\)����"�Cj^�                                    Bx~�Z  
�          @��a���33��R���HCp��a������\)�!�Ch�
                                    Bx~�   �          @����hQ��ə��1G���
=Cp��hQ������
=�(ffCf�q                                    Bx~瞦  �          A ���Y����
=�5�����Cs.�Y�����H��ff�*�\Cj�R                                    Bx~�L  
�          @�=q�]p���G��\)����Cr(��]p����H����!�Cjs3                                    Bx~��  
�          A
=�u���{�7���p�Cq��u��������\�%Ch�3                                    Bx~�ʘ  "          A��s�
��G��'�����Cp�3�s�
��  ��G�� =qCh�3                                    Bx~��>  
�          A{�HQ���=q�b�\��\)Ct���HQ������Q��@(�Cj��                                    Bx~���  �          A �����\���H�.�R��
=Cm=q���\������!�Cd(�                                    Bx~���  T          Az���Q��љ��E����Cn���Q����H���H�*33Cd޸                                    Bx~�0  
�          A�R�tz���Q��z=q���HCqh��tz������G��=ffCf�H                                    Bx~��  T          Az��HQ���\��  ��Q�Cv���HQ����H���T�Ck�R                                    Bx~�"|  �          AQ��%���G���z��{C{��%������G��i�Co�)                                    Bx~�1"  
�          A Q��������Q��C|.�������
=�q33Cph�                                    Bx~�?�  �          A��ff�����33�	G�C}Y��ff��
=��H�g�Cs}q                                    Bx~�Nn  "          A*�\��Q���p����'�RC�\��Q��y���33��Cs��                                    Bx~�]  T          A.{���������H�
=C�S3������Q��
=�|p�Cv��                                    Bx~�k�  "          A1�����
�	�������\C����
����=q�{�
C|\)                                    Bx~�z`  �          A0�ÿ��
�	���(��C�� ���
��=q���|C#�                                    Bx~�  T          A0�ÿ���	p���  ��RC��)��������(��x�C{��                                    Bx~藬  �          A9���z������\)C�W
�����#33�v�
CzǮ                                    Bx~�R  �          A>=q��G��	��p��+�C�ٚ��G����
�/�
{C}�                                    Bx~��  T          A>�\��������(��%�HC��=�����(��.{(�C}�{                                    Bx~�Þ  �          A>=q��z��z��
=�.\)C�#׿�z���  �0���C~�                                    Bx~��D  �          A>ff��Q��33� z��)�HC�ῘQ���
=�/\)=qC
                                    Bx~���  T          A<Q�z�H�
{���)��C��f�z�H���R�-p�W
C��{                                    Bx~��  �          A:=q�(��  � ���1
=C���(�����,��Q�C��                                     Bx~��6  �          A9���G���H�=q�2��C�J=�G��~{�-p�\C�xR                                    Bx~��  �          A9p���Q��������,{C����Q���\)�+
=��C{(�                                    Bx~��  �          A@  ���\�
=�
�H�9�C�s3���\�qG��5���C~�
                                    Bx~�*(  �          AD  �!G������<33C��3�!G��p���:=q�C�9�                                    Bx~�8�  �          AC
=��R��H�  �>��C��3��R�g��9�
C�"�                                    Bx~�Gt  �          AF=q����{����<
=C�y�����r�\�<  C~��                                    Bx~�V  �          AD�ÿ�p����ff�933C��q��p��vff�9p���Cy�                                    Bx~�d�  �          AC
=���
��ff����@  C�����
�Z=q�8��#�Crh�                                    Bx~�sf  �          AE녿�z��{����7\)C��{��z��x���8���)Cs�)                                    Bx~�  �          AE������33�:��C�lͿ��n{�9G�B�Cr��                                    Bx~鐲  �          AH(������z��"{�Y
=Cz������ ���AG�  C[�                                    Bx~�X  �          AK
=�>�R�׮�%��Y�
Cv(��>�R��ff�C
=�CS+�                                    Bx~��  �          AK\)�9����(��'
=�]  CvY��9����z��D  u�CQ��                                    Bx~鼤  �          AJff�6ff��{�(  �`z�Cv&f�6ff��(��C���COE                                    Bx~��J  �          AI���!G���G��,(��j��CwW
�!G��}p��D��(�CI�                                     Bx~���  �          AK
=�z�����2{�u��Cw���z��ff�G\)� C?\                                    Bx~��  �          AI��ff��  �4���}Cx0��ff�u�G\) �HC5�\                                    Bx~��<  �          AG�
������H�4  � Cy�3���=u�E��¢��C2E                                    Bx~��  �          AJ{�
=��z��5��\)Cw�f�
=<��F�H �3C3!H                                    Bx~��  �          AJ�\�{����5��}z�Ct&f�{=u�F�Ru�C2�f                                    Bx~�#.  T          AL  ������
�8z��\Cs�
���>�33�H(��C+��                                    Bx~�1�  �          AL���p���p��733�}�RCt���p�=#�
�I��HC3�                                    Bx~�@z  T          AL  ������\�5��z�
Cuٚ��þ#�
�H(�W
C7�{                                    Bx~�O   �          AD�Ϳ�p����\�0Q��~�RCx���p��u�B=q¡p�C5��                                    Bx~�]�  T          A9G���Q��w
=�,��\)CvW
��Q�?s33�6�R �qC�3                                    Bx~�ll  �          AC����
�����%�n�C|���
��G��=G� aHCQs3                                    Bx~�{  |          AV�H��p���\)�!��C�s3��p�����<Q��fC��3                                    Bx~ꉸ  �          A[\)���Dz�������z�C�E���(��'
=�G
=C�/\                                    Bx~�^  �          AZ=q�O\)�D(�������C�)�O\)�(��&=q�F��C�\)                                    Bx~�  �          A]G���R�K����
���HC�����R�{�"{�<p�C��                                    Bx~굪  �          Ac�
�:�H�MG���
=��p�C�aH�:�H��
�+��D�\C��H                                    Bx~��P  h          A]���(��(Q��=q�33C��3��(������?\)�{  C��q                                    Bx~���  �          Ah(���(��B�H������C�����(�� ���=G��\��C��{                                    Bx~��  �          Ap�����1p���\�%=qC��\������YG���C{u�                                    Bx~��B  �          Ao33���H����9��LQ�C�O\���H�n�R�f�\�=C|)                                    Bx~���  �          Al  ��Q���
=�Lz��pC�O\��Q쿯\)�i�¥�
Cd�                                    Bx~��  �          An�H���\��z��P���s�RC��{���\��Q��m�¦��C_�                                    Bx~�4  �          Ar{�����z��TQ��uQ�C��Ϳ�������pz�¦��C\!H                                    Bx~�*�  h          Ap�ÿ�������N�R�o{C��ÿ��Ϳ�  �m�¤� Cd                                      Bx~�9�  �          Aw�������
=�U���n��C��
���Ϳ�\)�uG�¥p�Ck��                                    Bx~�H&  �          Av=q�����
=�]��C�箾�����u��¯��Cs�                                    Bx~�V�  �          A{33?�R��p��k33L�C�G�?�R?��\�z{«B�BjG�                                    Bx~�er  �          A{
=��ff�G��S��fffC�ff��ff�  �w\)¡ǮCt�f                                    Bx~�t  �          Ap��?@  ����S�
�|�C�33?@  �:�H�l��«�C��H                                    Bx~낾  �          As�?�
=��\)�\���C���?�
=��G��rff§Q�C�4{                                    Bx~�d  T          At(�?�������\��\)C�H�?�����z��s\)ª�qC�'�                                    Bx~�
  �          Aup�?�G��ٙ��[��}p�C�<)?�G��0���t��ª�
C��3                                    Bx~뮰  �          At  ?�\��G��S��q{C��3?�\�Ǯ�rff§��C��                                    Bx~�V  �          Ar�\?aG���\�Ip��`z�C��q?aG��'��n�\��C�G�                                    Bx~���  �          Alz�>����p��I���j�C���>�녿����j{¤u�C�޸                                    Bx~�ڢ  �          Ah��>�=q���H�Lz��w=qC�  >�=q�����g33ª��C��3                                    Bx~��H  T          Amp�?333����G\)�e��C��?333�\)�ip�¡�3C��                                    Bx~���  T          Af�R=#�
��{�L���y�RC�+�=#�
�}p��f�R¬#�C�#�                                    Bx~��  �          AR�R��ff���H�@(�B�C}��ff>���P��¦ffC.p�                                    Bx~�:  �          AS
=��Q���z��:=q�{z�C{��Q��\�N�R¢\)CBǮ                                    Bx~�#�  �          AQ�Ǯ�ָR�2{�i  C�o\�Ǯ�ٙ��N{��Ccc�                                    Bx~�2�  �          AW�
�ff����4���cC}s3�ff���H�R�Hu�C_                                    Bx~�A,  �          AS�
�`����
=�8���u��Cl{�`�׾B�\�J�R��C7�                                    Bx~�O�  "          AU���33����?\)���C�쿳33��G��S�§\)CET{                                    Bx~�^x  �          AQG��X�������:{�}�Cj�f�X��>u�I�C/�f                                    Bx~�m  �          AP�׿5��p��;�\C�p��5�
=q�P(�¬#�CY��                                    Bx~�{�  �          AO\)���
���H�0z��h�C�����
��z��Mp�£\C���                                    Bx~�j  �          AZff����p��D��8RC�%�녾�{�Xz�®ffCS                                    Bx~�  �          AaG��5�����C33�sQ�C�  �5����^=q§#�Cr�3                                    Bx~짶  T          A`���5��G��D���v{Ct�)�5�0���[
=�CA�H                                    Bx~�\  T          A\����\��{�E��Cw0���\���X   �qC7                                      Bx~��  T          ATQ��L����p��A�W
Cj��L��?0���MG�aHC'�                                     Bx~�Ө  �          Aa�����{�L��{Cy
=��=#�
�^�\¢��C2�f                                    Bx~��N  T          Ad(���(����R�O�
(�C�=��(��\)�c
=¨�C9��                                    Bx~���  "          AaG���  ����N=q�
C��ÿ�  <��
�`(�©��C333                                    Bx~���  
�          AhQ�?�(����D(��e=qC�K�?�(�����d���\C��                                     Bx~�@  �          Ad��?�(���(��Bff�i�C���?�(����a� �HC�xR                                    Bx~��  �          A`z�?
=��Q��Dz��tG�C�z�?
=���_33§k�C�=q                                    Bx~�+�  �          Ab=q>�  ��
=�H(��v�HC�\>�  ��ff�b=q©G�C�g�                                    Bx~�:2  T          Abff?�����
�C33�n�C�k�?����ff�`Q�¤�{C�Ff                                    Bx~�H�  	�          AV=q?����Q��3\)�b�\C��?���z��Rff�C���                                    Bx~�W~  �          AK33?�z���G��-p��j{C�?�z��ff�Hz� �{C�W
                                    Bx~�f$  �          AS
=?������2ff�fp�C���?�����
�O\)\)C��f                                    Bx~�t�  �          AY�@G��   �+��Q��C�R@G��Mp��O�
z�C�                                      Bx~�p  �          AX(�@
=q�  �'��KG�C�S3@
=q�c33�M��#�C���                                    Bx~�  �          Ag�?�����z��@z��affC���?����'��bffp�C��                                     Bx~���  T          Ad��?�  �����C�
�k��C�޸?�  ��p��a��¡#�C�)                                    Bx~��b  �          Af�R?�R���H�C��h\)C�S3?�R���c�¡�C��3                                    Bx~��  �          Aj{    ����J�R�p
=C��    ��{�h  ¥\)C��                                    Bx~�̮  �          Ah�þ�{�׮�N�\�y��C�����{��  �h(�©ǮC~�                                    Bx~��T  "          Afff�!G�����L���yz�C�O\�!G����R�e�¨��Cs�                                    Bx~���  �          Ac
=��ff����N�R�fC�� ��ff�����b�\ª�CI�                                    Bx~���  �          AXQ쿪=q���\�G\)G�CG���=q=��
�W33¨��C0k�                                    Bx~�F  "          AaG����\��=q�K�
33C�׿��\��\�`  ª�CNE                                    Bx~��  	�          Ag33��G���G��T���C�ٚ��G��#�
�fff¬  C6�                                    Bx~�$�  T          Ak\)�5��=q�Y��{C�` �5=#�
�k33®u�C1�                                    Bx~�38  
�          AlQ쿦ff���\�Y��W
C�k���ff<��k33©�)C2��                                    Bx~�A�  
�          Ag�
>�����IG��q�HC���>�녿���ep�¥ffC�}q                                    Bx~�P�  �          At  @��H�%��  ���C��q@��H��(��O\)�i��C��                                    Bx~�_*  �          As
=@��
�,����z����HC���@��
��  �6=q�C{C��q                                    Bx~�m�  "          As
=@�=q�#\)�{���C�` @�=q����D���XC���                                    Bx~�|v  r          Ab=q@���H����#�C�  @����\�=p��lp�C�3                                    Bx~�  �          AT  @0  �ff���@�RC��@0  �����E��3C�{                                    Bx~��  �          AW�@���=q�(��'�C���@�����H�;33�q�C�˅                                    Bx~�h  �          AX  @�(��G���R��C�Ф@�(���
=�4���d(�C��                                     Bx~�  �          AZ{@�p��ff����RC���@�p����
�2�R�\(�C��q                                    Bx~�Ŵ  �          Aa��@����R��R�!  C��{@�������@���g�C��
                                    Bx~��Z  |          Ay@�z��$(��*ff�,�C���@�z������[��w  C��                                     Bx~��   
�          Ai@~�R�  �)��:{C��@~�R��33�T���HC�o\                                    Bx~��  �          AdQ�@�{������C�\@�{���
�@���f33C�Z�                                    Bx~� L  �          A_�@��R�����\�)�\C�Ф@��R����C33�s�RC��                                    Bx~��  
�          Ac�@�����R����33C�#�@�����p��=��dffC���                                    Bx~��  �          Ac\)@�
=��\��R�\)C�/\@�
=�����=��dG�C�]q                                    Bx~�,>  �          Ahz�@�ff�����{C�3@�ff��
=�A���a(�C�p�                                    Bx~�:�  �          Aip�@���=q��R�!  C��R@�����R�EG��f�HC���                                    Bx~�I�  �          Ae��@��R��p��\)C�^�@��R�����<Q��^�RC���                                    Bx~�X0  �          Aa�@�=q�=q�p����C�t{@�=q��{�<Q��b
=C���                                    Bx~�f�  �          AiG�@��H�������)p�C�n@��H�����G\)�k33C�\)                                    Bx~�u|  �          Ae�@�ff�G��"{�4p�C�1�@�ff��Q��G�
�s\)C�q�                                    Bx~�"  
�          Ad��@�����\�#33�6ffC�Y�@�������IG��w{C��                                    Bx~��  �          Af�\@������(  �;=qC�J=@���w��M��{�C��
                                    Bx~�n  �          Aj{@���H�,z��=��C�˅@��xQ��Q��~�C��{                                    Bx~�  
�          Ajff@�������/�
�BffC�5�@����p  �Tz�(�C�P�                                    Bx~ﾺ  �          An�\@s�
��ff�@z��W{C���@s�
�@  �a��{C��                                    Bx~��`  T          A~=q@~{���J=q�S\)C�8R@~{�`  �o\)
=C�E                                    Bx~��  �          A|  @����	p��Fff�O��C��{@����^�R�j�HaHC�N                                    Bx~��  �          A���@����=q�P  �Oz�C�.@����q��v�H#�C�\                                    Bx~��R  �          A�\)@����R�T(��\�C��@���333�u��C��                                     Bx~��  @          Aw�
@�=q��G��O��cp�C�T{@�=q���k�
�C�                                      Bx~��  T          A{�@�{�����X  �lp�C��q@�{��ff�qG�
=C��q                                    Bx~�%D  �          Au�@����H�U��r
=C�H@��\(��i��3C�y�                                    Bx~�3�  T          Ao�
@�{���S��w��C�4{@�{���f{G�C�s3                                    Bx~�B�  
�          Ao�@�p���=q�PQ��r{C�:�@�p��xQ��eG�ǮC�u�                                    Bx~�Q6  
Z          Ar�R@S33��ff�QG��n33C��{@S33��\�k33k�C�޸                                    Bx~�_�  T          Av=q@:=q���R�\���C��@:=q����pQ���C�,�                                    Bx~�n�  
�          Am@4z���
=�\���\C�k�@4z�?
=q�ip��RA.�R                                    Bx~�}(  �          Aw�@9�����
�d��L�C��@9��>�{�s
==q@�Q�                                    Bx~���  �          Ar�\@>�R�����[33=qC���@>�R�   �mC�<)                                    Bx~�t  
�          Af=q@=p���Q��@���c��C�{@=p��\)�]p�C��3                                    Bx~�  
�          Ai�?�p������Vff{C��f?�p���\)�g
=¦�C�p�                                    Bx~��  
�          A]���������X��aHC������@$z��XQ���B���                                    Bx~��f  "          Aq��?�=q���\�e���RC�t{?�=q?�ff�o\)£��A�{                                    Bx~��  �          Azff@�H��=q�lz�p�C���@�H?xQ��w
= ��A�(�                                    Bx~��  
�          AyG�?�=q��G��hQ���C���?�=q>B�\�w33¦k�@�=q                                    Bx~��X  "          Ae��@z���z��V�R�C�
=@z�>����c
=£{A-                                    Bx~� �  T          Ad��?�ff�\)�YQ�C�(�?�ff?���b=q£(�A�                                      Bx~��  �          Ac�?�(��n�R�ZffB�C�Ǯ?�(�?���ap�¤\B&�                                    Bx~�J  �          Ae��?�(��hQ��]�C�O\?�(�?��H�c\)¤BG�
                                    Bx~�,�  "          Ad  ?�\)�e��[\)C�t{?�\)?����ap�£�HB;{                                    Bx~�;�  �          A_\)?Tz��(���Z�H�C���?Tz�@�
�[�
�RB�z�                                    Bx~�J<  
Z          A`z�?\(���R�\z�k�C���?\(�@\)�\Q�W
B�                                    Bx~�X�  
�          A@��?!G�����<(� ǮC�B�?!G�@��:�H��B��                                    Bx~�g�  T          A8��>�녿�
=�4��§ffC�}q>��@5��0(��fB���                                    Bx~�v.  �          ADQ�.{>�G��D  ¯��B�Q�.{@�p��5p�33B�\                                    Bx~��  T          AT��?   ��G��Tz�®G�C�aH?   @�ff�I���B�u�                                    Bx~�z  �          AT(�>.{�����S\)±  C��R>.{@�G��H  \B��                                     Bx~�   �          AS���G�<��
�S�³
=C+Ǯ��G�@��\�FffffB��q                                    Bx~��  �          A[
=>8Q쿆ff�Z=q«{C��>8Q�@n{�R�\ffB��=                                    Bx~�l  
Z          Ad(�>�p��˅�b�R¦�)C��>�p�@X���]��Q�B�aH                                    Bx~��  T          Ab�R>W
=���H�a�¦
=C���>W
=@N�R�\���{B�p�                                    Bx~�ܸ  
�          A[�>k������Zff©C�w
>k�@c�
�S�
�
B��=                                    Bx~��^  
�          AH  >�p��z��G�­�C�R>�p�@n�R�>�H�B�                                    Bx~��  "          AA�=��0���@��­W
C�  =�@^{�8���=B���                                    Bx~��  
�          A<��?\)>����<Q�­��A�(�?\)@����/\)�
B�(�                                    Bx~�P  
�          A;�=��;�  �;\)±ffC�R=���@p���1p��\B��                                    Bx~�%�  
�          A:�\>aG��\)�:=q®�C��)>aG�@[��2=q��B��                                    Bx~�4�  T          A7\)��\)����7
=±W
C�
��\)@h���-���fB�{                                    Bx~�CB  T          A3��B�\���3\)±�3CT��B�\@k��)p���B��f                                    Bx~�Q�  
Z          A0z�#�
�(���0  ­�C��{�#�
@Fff�)G�aHB�\)                                    Bx~�`�  
�          A-p��0�׿+��,��©��C_�Ϳ0��@@���&{��B��                                    Bx~�o4  �          A1���Q�&ff�0��¬G�Cp�쾸Q�@G
=�)(�B�Q�                                    Bx~�}�  T          A$Q쿡G��   �"�\¤�qCI����G�@<(���
��B�                                    Bx~�  "          A�\��\)��
=��\)C\����\)@z���(�B�8R                                    Bx~�&  T          A��33��G�����CS���33?��
=��C�)                                    Bx~��  T          A{�˅��  ���
=C_\)�˅?У��G��HC=q                                    Bx~�r  T          AQ쿺�H��=q�z�
=C^Y����H?����
=Q�C B�                                    Bx~��  "          A�\�u��z��33�Co�H�u?˅���
B�p�                                    Bx~�վ  T          A9G���G����
�6ff L�CfxR��G�@ff�4(�aHB�z�                                    Bx~��d  
�          A4�׿
=����/�#�C�.�
=?�=q�2�H¥#�B�33                                    Bx~��
  
Z          A4  �����  �/�G�Cs�3����?�G��1�� �fB�33                                    Bx~��  �          A333������z��.�HL�Chٚ����?��/33�C �
                                    Bx~�V  "          A5����˅�0(���CX���@Q��.�RB�C�
                                    Bx~��  T          A3�
�
=��ff�.{�CU8R�
=@��,z�8RC

                                    Bx~�-�  
�          A3
=�=q���-B�CM��=q@p��*=q#�CxR                                    Bx~�<H  �          A4���;��\�,��  CO\)�;�@ff�+33�RCQ�                                    Bx~�J�  �          AH����
�8Q��>�R�{CgE��
?�
=�C33C                                    Bx~�Y�  �          AXQ��Z=q���L  #�CVY��Z=q?��H�L���HC{                                    Bx~�h:  T          A`Q���Q��\�T���\CK޸��Q�@*=q�R�\�)Cp�                                    Bx~�v�  �          A�{��G���(��nff�)C�޸��G��5��{®Cnu�                                    Bx~�  T          A�(�>�(��ᙚ�s33(�C���>�(��޸R��\)§�qC�                                    Bx~�,  �          Ad���C33�J=q�I�z�Ca���C33?�=q�N�\�
C xR                                    Bx~��  
�          Am��&=qA	G��������Cp��&=qA(������  C	�                                    Bx~�x  
           Ae����
A�RA  B$  B��)���
@��\A2�\BY�C��                                    Bx~��  
d          AZ=q���@��A(��BPp�B�����@I��AC33B�\Cs3                                    Bx~���  
�          A\Q��.�R@��HA>�RBs{B�W
�.�R?�Q�AT  B���C��                                    Bx~��j  	�          Ap����Q�A�A@  BQ  B����Q�@vffA_�B�  C�                                    Bx~��  |          A����W�A&=qA3�B733B���W�@�  A_�B|(�B��                                    Bx~���  T          A�
=�У�A&�HAg
=BWffB��ͿУ�@���A�(�B�\B�k�                                    Bx~�	\  
�          A��\�˅A2=qAb=qBN  B�=q�˅@���A���B�z�B�ff                                    Bx~�  "          A��
��ffA,(�Ad��BR�\B���ff@��A�B�ffB�Q�                                    Bx~�&�  "          A�=q��\A7\)A\��BG��Bŏ\��\@ϮA�B�#�B�z�                                    Bx~�5N  
�          A��H�ٙ�A<Q�AJ{B:�
B�uÿٙ�@�p�A{33B��3BθR                                    Bx~�C�  "          A�(���
=A6=qAIB>�RB�\)��
=@ڏ\Ax��B��3B˸R                                    Bx~�R�  
�          A��\��p�A6�RA]��BHQ�BǮ��p�@�
=A��B���B�
=                                    Bx~�a@  
�          A�����A333AX  BF��B�#����@�z�A��RB��B��                                    Bx~�o�  
�          A�33�B�\A+
=AV{BI(�BӸR�B�\@�ffA�z�B���B�\                                    Bx~�~�  �          A�(��VffA1�AR�RBB�Bծ�Vff@�(�A�B�k�B�\)                                    Bx~�2  �          A�  �g�A<z�AF�RB4�HB�(��g�@��Aw�Bx��B�                                    Bx~���  	�          A�=q�Tz�A1��AMG�B?��B�W
�Tz�@љ�AzffB��RB���                                    Bx~��~  �          A���_\)A3
=AG�
B;G�B֣��_\)@�  Au��B~�B�                                    Bx~��$  �          A�z�����A3
=A@z�B3�B������@���An�\Bt33B�\)                                    Bx~���  �          A�=q���\A+
=A=�B4\)B��)���\@ϮAi��Br=qC {                                    Bx~��p  T          A�G����RA(��A0��B)�B������R@ӅA\Q�Bdz�C
=                                    Bx~��  @          A�\)��A.�\A%��B��B��q��@�AS\)BQ��C                                      Bx~��  
�          A�����RA-G�A=G�B/  Bힸ���R@��Aip�Bj
=C�=                                    Bx~�b  �          A�(���Q�A&�\AX  BIB�(���Q�@�Q�A�ffB��qB��R                                    Bx~�  
�          A�G����A(Q�AMp�B>{B�����@��HAw33Bz�C��                                    Bx~��  
Z          A�����A&�\AM��B>��B� ���@��Av�RBy�HC�)                                    Bx~�.T  	`          A���a�A'�Ab�\BP{B�B��a�@��A�p�B�ffB��                                    Bx~�<�  �          A��R��  A(Q�A`(�BK
=B�Q���  @�Q�A�ffB�W
C                                     Bx~�K�  o          A��\��A,��AW�BA\)B�8R��@ƸRA���B}p�Cٚ                                    Bx~�ZF  
�          A�z��
=AK�
A&=qB�
B����
=A�
A\(�B>�C�                                    Bx~�h�  |          A�z��(�AI�A-p�B��B�u��(�A  AbffBFp�C^�                                    Bx~�w�  �          A�{��Q�A>�RA@��B%G�B�{��Q�@�\)Aqp�B_  C&f                                    Bx~��8  �          A�(���=qA=�AC�B(33B����=q@�(�As�
Bb�C�R                                    Bx~���  T          A���ǮA9�AO
=B1B�z��Ǯ@�ffA}G�Bk�
C�                                    Bx~���  T          A��\���
A2�\AT��B:{B����
@��A�=qBt
=Cff                                    Bx~��*  �          A�=q��\)A6�HAMp�B2p�B�(���\)@�=qAz�RBk�CaH                                    Bx~���  T          A�Q�����A8(�AI�B.�B�(�����@�
=Aw�Bg=qC!H                                    Bx~��v  �          A�������A9�AI�B.�B��)����@�G�Ax  Bf�HC�
                                    Bx~��  
�          A�G���=qA9�AJ�RB.
=B�����=q@�\Ax��Bf��C�)                                    Bx~���  �          A�(�����A7
=AM��B2�
B�=����@�Az�\Bl{C�{                                    Bx~��h  "          A����ffA1A\��BAp�B����ff@�G�A�B|p�C xR                                    Bx~�
  �          A������A5G�AJ�\B-�B�z�����@��HAw
=Bc�C5�                                    Bx~��  T          A�  ��{A6�\AHz�B*p�B�=q��{@�
=AuG�B_�RC�H                                    Bx~�'Z  "          A�����A1��AS�
B8�RB�.��@�\)A~�\Bp�\C�\                                    Bx~�6   �          A�Q����RA0  AU�B;�\B�����R@ӅA�{Bs�RC{                                    Bx~�D�  �          A��H��{A,  Aep�BI�B�Q���{@��HA���B��C�                                    Bx~�SL  
�          A�����z�A*=qAj{BO33B��)��z�@��A��RB�aHB�33                                    Bx~�a�  
�          A�=q��=qA*=qAf�\BL�B�=��=q@��A�
=B��fB�Ǯ                                    Bx~�p�  
�          A�����33A3�A_�B?G�B�\��33@�{A���Bwz�C+�                                    Bx~�>  �          A�{�\A5�AZ�RB;{B�z��\@�(�A��HBrCu�                                    Bx~���  
�          A�p����
A-A^ffB@�B������
@�(�A���Bv��C��                                    Bx~���  
�          A�G��љ�A*{A]�B@  B�G��љ�@�{A��RBt�C�H                                    Bx~��0  �          A��R����A$  A`(�BC��B�������@�G�A���Bv�C
��                                    Bx~���  
�          A������HA ��Ae�BIQ�B�z����H@���A���B|��C
�3                                    Bx~��|  
�          A�ff�ÅA�Ag�BMz�B����Å@���A�p�B�\)C:�                                    Bx~��"  �          A��R���HAAe�BM=qB�G����H@��
A�Q�B��RC	k�                                    Bx~���  �          A�����A�Ah��BT��B�u����@�A�33B��=C	��                                    Bx~��n  
�          A�Q����A(z�AS33B8B�ff���@�33AzffBk=qC	�H                                    Bx~�  �          A����A+�
A@  B'�
B��)��@���Ah��BX�C
��                                    Bx~��  
�          A��H��ffA)�AA��B*B��
��ff@أ�AiB[��C
��                                    Bx~� `  
�          A����Q�A+
=A@��B)��B����Q�@ۅAip�BZ�C
�                                    Bx~�/  �          A����  A��ATz�BB��B��=��  @�p�Av�HBr��C:�                                    Bx~�=�  
�          A�{��A'\)A<Q�B)�HB�8R��@�\)Ac�BZ�HC

=                                    Bx~�LR  
�          A�(���(�A�\AUBG
=B���(�@�G�Axz�Bz(�C��                                    Bx~�Z�  T          A�=q���A�HAX  BI��B�u����@���A{�B��C�=                                    Bx~�i�  
Z          A�(���G�A��AF�\B8�B�aH��G�@�
=AjffBiQ�C
�3                                    Bx~�xD  �          A�p���z�A (�A:�HB+ffB����z�@��
A`  BZp�C(�                                    Bx~���  �          A������A
=A?
=B1��B������@�  Ab�\B`33C�                                    Bx~���  �          A��\��\)A�\AC�B6��B��R��\)@���Af�RBf�C��                                    Bx~��6  "          A�
=��ffA
=AU�BI
=B�R��ff@�(�Axz�B|�C
                                    Bx~���  
�          A��\��A ��Af�RBT  B�33��@�A�
=B�
=B���                                    Bx~���  	�          A�����\)A"{Ac33BO=qB���\)@��HA��B��HC ��                                    Bx~��(  
�          A��
��G�A (�A_�BK�HB����G�@��A�\)B�8RCO\                                    Bx~���  �          A�����
=A=qAZ�HBH�\B����
=@���A}B{p�C�R                                    Bx~��t  T          A�p��ƸRA�AT��BC�B����ƸR@��\Aw�
BtC�{                                    Bx~��  �          A�����{A ��AH��B5�RB�#���{@ȣ�Al��Bd�
C
�                                    Bx~�
�  
�          A��\���Ap�AE�B3
=B������@��
Ai�B`=qCaH                                    Bx~�f  T          A��A��A?�B-
=C���@�ffAb�HBX��C��                                    Bx~�(  T          A�����G�A�AHz�B8C �{��G�@�z�Ai��Bd�RC33                                    Bx~�6�  T          A��R�33A=qA:�HB,�\C��33@���A\z�BW  C=q                                    Bx~�EX  
�          A�z���{A�A?
=B1z�C���{@�p�A`��B]\)Cff                                    Bx~�S�  �          A���	��A{A1p�B$\)C��	��@��AS33BMC��                                    Bx~�b�            A�33��A\)A+\)B�C����@�  AN�RBG��CJ=                                    Bx~�qJ  �          A����
A\)A0��B%ffC���
@�AR�\BO�
C!H                                    Bx~��  @          A�
=��A#
=A'�B33C���@�G�AL��BE{CJ=                                    Bx~���  �          A�ff�
=A"�RA)G�BffC� �
=@�  ANffBD�C�                                    Bx~��<  h          A�ff�G�A$��A+�B
=C ���G�@��HAQ�BH33C�{                                    Bx~���  6          A�Q����A  A((�B$ffB��f���@�z�AJ=qBQ
=C33                                    Bx~���  T          Ao
=��G�A+
=@���A�(�B�����G�A	��ABCL�                                    Bx~��.  �          Az�n{@S33��\��BӅ�n{@���ƸR�HQ�BȮ                                    Bx~���  "          A��?��ÿ\)�\C�q�?���?�z���H��B9�                                    Bx~��z  T          A0��@p  �Q�� ��#�C��{@p  ?�=q��\Q�AϮ                                    Bx~��             As�
@��z��;
=�N�C�Ǯ@����Z=q�C�~�                                    Bx~��  �          A�z�@�  �^�R��\)��  C�}q@�  �<Q�����C��\                                    Bx~�l  "          A|��>�\)�dz�?��@�\)C��>�\)�b{���Q�C���                                    Bx~�!  
Z          A)��?E��أ�A{BH�C�>�?E��
=q@ÅB��C���                                    Bx~�/�  �          AG��:�H����@�
=BY  C��{�:�H�ڏ\@���BC���                                    Bx~�>^  T          AG����1G�@���B���Cn޸�����\@�{BS��Cy�                                    Bx~�M  �          @�
=�tz�@��H@~{A���B��q�tz�@y��@�z�B-{C\)                                    Bx~�[�  
�          A��Z�H@�(�@�p�BE=qB�L��Z�H@HQ�A  Br��C	�                                     Bx~�jP  J          AW33���\@���A,z�BS�
C����\@L��A?�
BzffC
=                                    Bx~�x�            AU�����\@�Q�A ��BC��C�=���\@\)A7
=Bl=qCxR                                    Bx~���  
�          AY�����\@��A*�\BMC����\@`  A?
=Btp�C�f                                    Bx~��B  �          AZ�\��p�@��
AD(�B�C	��p�?aG�ANffB�� C(��                                    Bx~���  �          AW\)�\)@��AAG�BC� �\)?�AL��B���C#�                                    Bx~���  T          AU���\@�(�A;�
BvG�C}q���\?��AFffB��C'�\                                    Bx~��4  �          AX(����H@uA>{Bw{C����H?=p�AG�B�(�C+�3                                    Bx~���  "          AW\)��z�@�G�A=p�Bw�RC���z�?�Q�AI��B�C!�H                                    Bx~�߀  T          A]���=q@�\)AA��BsQ�C}q��=q?���ANffB��C!��                                    Bx~��&  T          An�R��\)@r�\AW�B�33CǮ��\)>��RA`  B��qC0h�                                    Bx~���  "          A|z����@��AeG�B���CO\���?h��Apz�B��fC(�                                    Bx~�r  �          Ax�����\@���A[�
BxC�=���\?�AjffB�#�C0�                                    Bx~�  T          A�
=��z�@�(�Al  B��C�\��z�>�Q�At��B�� C/�                                    Bx~�(�  T          A�  ����@�(�AqG�B�W
C޸����?��\A|��B�G�C'O\                                    Bx~�7d  T          A�\)���\@�G�Aq��B��)C�3���\?0��A|  B��)C*�                                    Bx~�F
  
�          A������@s�
A|(�B�(�C������#�
A��B��fC5��                                    Bx~�T�  �          A����p�@eA}�B��)C�H��p��\A�B�#�C8c�                                    Bx~�cV  "          A�  ��Q�@P  Az=qB�8RC���Q�(��A\)B��HC;xR                                    Bx~�q�  "          A�����=q@�z�Ao�B�8RC	����=q>\Ax��B��C/5�                                    Bx~���  "          A�(���
=@�{Ao�B�aHC(���
=?.{AyB��3C+L�                                    Bx~��H  T          A����
=@X��At��B�aHC:���
=���Az�HB�ffC8@                                     Bx~���  T          A�����z�@fffAzffB���C�H��zᾏ\)A�ffB�{C7:�                                    Bx~���  �          A�  ���@�
=Af�HBp33CB����?�{As33B���C(p�                                    Bx~��:  �          A������\@��Af�RBx(�C(����\?��Au�B�z�C �)                                    Bx~���  	�          A�z�����@X��AqB��{C�����þ��
Aw�B��3C8�=                                    Bx~�؆  T          A~�\��@p�Ay�B�  C@ ������A{
=B�#�CY
                                    Bx~��,  
(          A��@  @[�At(�B���C.�@  ����Az=qB�33C9��                                    Bx~���  
Z          A��N{@o\)A
=B���C�R�N{�\)A�
=B��qC6�\                                    Bx~�x  T          A������@c�
A�z�B��HC
�f��������A��B�B�C8�3                                    Bx~�  �          A�����Q�@q�A~ffB��fC�R��Q콏\)A���B��fC4�\                                    Bx~�!�  "          A�=q��G�@r�\A|  B�#�CaH��G����
A���B��)C433                                    Bx~�0j  T          A�����  @�=qAw�B��C+���  >�{A�
B���C0T{                                    Bx~�?  
�          A�����@�\)Av{B�\)C
Q����?�G�A�z�B�ǮC)G�                                    Bx~�M�  "          A������R@��RAu��B���C�=���R?�  A��B�� C!u�                                    Bx~�\\  @          A�����=q@�  Am�B�Q�Cn��=q?n{Ax(�B��C(�\                                    Bx~�k  T          AG�
��p�@�G�@�33A�ffC	aH��p�@��
@��B{C�f                                    Bx~�y�  
�          ADz��L��@G�A�Bw�\CǮ�L��?���AB�(�C!\                                    Bx~��N  �          AL��@#�
���
AE��B���C��f@#�
��=qA9B�ffC���                                    Bx~���  
�          AS�@1�� ��AL(�B��C��@1���z�A?\)B���C��                                    Bx~���  �          AXz�@;��%�AO
=B�p�C�XR@;���
=A@(�Bz��C��                                    Bx~��@  T          Af�R@W
=�eAX��B�{C��@W
=���HAE�Blp�C���                                    Bx~���  �          ATQ�@G���RAHz�B�ǮC�@G�����A:=qBy{C�L�                                    Bx~�ь  T          A&=q����@%@��BTp�C������?�G�@�\Bh��C&�                                     Bx~��2  �          A.�R�{@���@(Q�A`(�C���{@��@uA�C
                                    Bx~���  �          A'���\@�33?��@�=qC)��\@�@#33Ac
=C�                                    Bx~��~  "          @��
�ə�@�p��L����p�C�3�ə�@��>aG�?�\)C��                                    Bx~�$  �          @ᙚ���H@y���<(���
=C�\���H@��R���R��
=C
�=                                    Bx~��  "          A&�\���@�\)?^�R@���C�{���@��?�z�ALz�C^�                                    Bx~�)p  T          A1���H@Z�H@�{B1G�C8R���H?�\)A(�BB�\C'�                                    Bx~�8  
          A<  �xQ�?8Q�A*�\B�k�C)���xQ��A(��B�aHCK^�                                    Bx~�F�  	�          A4(���p���=qA1G�B�(�CS�H��p��`��A(��B�W
Cs�                                     Bx~�Ub  �          AI�˅@O\)A�
BQ=qC�R�˅?�\)A�Ba�RC*
=                                    Bx~�d  
�          AQ�� ��@�@�(�A�  Cp�� ��@��@�G�B�C�                                    Bx~�r�  �          AS
=��@�  @�ffA��
C=q��@�z�@��A��
C�                                    Bx~��T  �          AN�\���@�@tz�A�p�C�=���@�p�@���Aģ�C�
                                    Bx~���  T          AM��+33@��H>�\)?��\Cz��+33@ۅ?�=qA
=C\)                                    Bx~���  �          AN{�'
=@���@�HA-G�C���'
=@�ff@z�HA��
CL�                                    Bx~��F  �          AO�
� Q�@�@VffAo\)C�q� Q�@�z�@�p�A�  Cu�                                    Bx~���  �          AT(��p�@�G�@��RA�Q�Ck��p�@��@��B{C�=                                    Bx~�ʒ  �          A[\)���@��
@�p�A�RC�����@�  AB33CO\                                    Bx~��8  �          AY����H@��@�33A���CJ=��H@\@�=qA���C�=                                    Bx~���  �          AS33�{@�
=@�A�  C33�{@Ǯ@�p�A�C�                                    Bx~���  �          A\���)�@��
@�  A��RC33�)�@ҏ\@���A�(�C�                                    Bx~�*  �          AZ�H�#\)Ap���녿�(�C
O\�#\)A�?��@ϮC
�3                                    Bx~��  T          AXz��{A�@W�Al��C� �{@�p�@���A���C
=                                    Bx~�"v  
�          AV{���@�@�Q�A��\Cs3���@��@�Q�A�\C�                                    Bx~�1  
�          AX(��A{@w
=A��HC�f�@��
@���A�(�C:�                                    Bx~�?�  
�          AU��G�Az�?�33A\)C&f�G�A
�R@tz�A���C	�                                    Bx~�Nh  
�          A6{�%�@�33��{���RB�ff�%�A��J�H���B�                                      Bx~�]  �          A6=q�	��@����H�$p�B�
=�	��A\)��p���33B�u�                                    Bx~�k�  �          A:�R���
A	����{��=qB�\���
Ap��"�\�JffB���                                    Bx~�zZ  "          A4z���Q�@�p���=q��  CG���Q�@�\)��  ��33C8R                                    Bx~��   
�          A8����  @���(���33C 8R��  A�H�vff��B��                                    Bx~���  
�          A8(����
A=q��{����B�����
A���U����HB�\                                    Bx~��L  "          A<����{AQ������\B�z���{A���z�H���B�                                    Bx~���  �          AC33��(�A�\������B�����(�A�
��{���\B���                                    Bx~�Ø  T          A;���{Aff��(���B�����{A�
��Q���ffB��                                    Bx~��>  "          AL����33A���=q�㙚B�(���33A0(��x����=qB�R                                    Bx~���  "          AK\)��{A$  ��ff���\B�.��{A/33�  �#�
B���                                    Bx~��  
�          AH(����
A(����R��33B������
AG��AG��aB��H                                    Bx~��0  �          AT����=qA&{�Mp��`Q�B�8R��=qA,�ÿ��
��ffB�                                      Bx~��  "          ATz���
A#\)���   C ����
A%�>�=q?�z�C xR                                    Bx~�|  T          AW33���A&�\��������C �
���A)�>���?��\C +�                                    Bx~�*"  "          APQ��{A���ff��ffC8R�{A!�>�  ?�{C Ǯ                                    Bx~�8�  �          AN�R��A�
�G��aG�C�=��A33?�33@���C�f                                    Bx~�Gn  �          AMG����A&�R?+�@@��B��)���A!G�@0  AF�\B��3                                    Bx~�V  T          AM��RA'
=�k����\B�{��RA$Q�?�\)A�B���                                    Bx~�d�  
�          AL�����\A!G���Q���B������\A=q?�A
�RC Y�                                    Bx~�s`  
�          AM���{A Q�:�H�P  C aH��{A�?��
@�  C �=                                    Bx~��  T          AI��33A�׿����HC �q��33AG�?J=q@g
=C �H                                    Bx~���  T          AH���ffA녿����=qC  �ffA�
>�{?�=qC��                                    Bx~��R  "          A>�\� (�A	G�������Q�C�� (�A
=q?z�@5�C�{                                    Bx~���  �          A?\)�A�H��ff��
C�)�A	녽��Ϳ   C:�                                    Bx~���  "          A?�
�p�@�
=����p�C�\�p�@�ff�\����C��                                    Bx~��D  
�          A:=q�\)@޸R�g
=���
C�\�\)@������.{C
Q�                                    Bx~���  "          A8  ����A
=q�I���~ffC �
����Ap���33��Q�B�aH                                    Bx~��  T          A#�
?�R@����Q��b��B�W
?�R@�33�����6Q�B�{                                    Bx~��6  �          A*�H?���@�=q�
=��B��?���@Å�
�R�Yp�B��                                    Bx~��  "          A!�?�@e�  z�Bw�?�@���{�`�B�G�                                    Bx~��  �          A&�R@s33@ff���#�A��H@s33@w
=�Q��i�B5��                                    Bx~�#(  T          Az�@�
=?�
=���
�_��A��@�
=@Y����33�J�RB	�                                    Bx~�1�  "          A=q@�?n{���
�R�A
=@�@�H��G��F�A��
                                    Bx~�@t  	�          A$��@׮?\(����A�
@�G�@׮@�
���
�7A�p�                                    Bx~�O  |          A,��A���Q���5C�k�A���zῗ
=�˅C���                                    Bx~�]�  "          A.=qA�
�Å��ff�
=C���A�
��p������\)C�Q�                                    Bx~�lf  
�          A-G�AQ�����{�C�xRAQ����R�B�\��ffC��                                     Bx~�{  T          A.{A��������C�1�A�������
���C��                                     Bx~���  
�          A.�RA{����G���C���A{����333�r�RC��R                                    Bx~��X  @          A"�\>�\)@z���{
=B�  >�\)@g���(��u  B�
=                                    Bx~���  �          A�R���H@��R��G��$=qB��)���H@�����=q��\)B�L�                                    Bx~���  �          A\)�߮@�Q����H�Σ�C�߮@�{�HQ����
C�                                    Bx~��J  �          A/
=����@�{��G���=qC	�)����@��
�>{�~{C{                                    Bx~���  
(          A���\@�  ����� (�C
n��\@��;���\(�C	�\                                    Bx~��  
�          A9p���Q�@�33������p�C����Q�A���Dz��x��C#�                                    Bx~��<  �          AAG��ff@��H�j�H���
C�3�ffA=q�
=q�%G�C
=                                    Bx~���  T          AFff���@�=q�L(��o\)C(����AQ�ٙ�����C	�\                                    Bx �  "          A>ff�Q�@��*=q�O�C}q�Q�@�33���R��Q�C
(�                                    Bx .  �          A3��33@���Ǯ�   C�R�33@�\�aG�����C�                                    Bx *�  
�          A,(���
=@���33��
=Cc���
=@����8Q�C�H                                    Bx 9z  �          A9��
=q@��������ffC
k��
=q@�\)>��R?��
C
&f                                    Bx H   �          A8(��	G�@�\)>Ǯ?�Q�C	�q�	G�@���?��A=qC
��                                    Bx V�  T          A.�\� Q�@�=q>\)?B�\C	��� Q�@�?�(�@�Q�C
0�                                    Bx el  
Z          A/�
�ff@�z���Ϳ   C	�\�ff@���?��R@�ffC
8R                                    Bx t  
�          A,���z�@��þL�Ϳ��C���z�@θR?xQ�@�{C��                                    Bx ��  T          A���ᙚ@�p�>���@%Cs3�ᙚ@�Q�?�\)AG�CG�                                    Bx �^  	�          A
ff��(�@aG�����p�C� ��(�@x�ÿٙ��8��C8R                                    Bx �  
�          @��R��ff@��>.{?�ffC���ff@��?h��@߮C�q                                    Bx ��  
�          @�(�����@w
=��
��Q�C������@������(z�C��                                    Bx �P  
(          A����z�@�ff������C&f��z�@����G��333CO\                                    Bx ��  T          Az��љ�@޸R?�@G�CB��љ�@�Q�?޸RA'�
C�                                    Bx ڜ  
�          A����
=@ə��xQ�����C޸��
=@��
>#�
?k�C��                                    Bx �B  
�          A33��p�@��H>\@   C\)��p�@�?�Q�A�C!H                                    Bx ��  �          Ap���z�@љ�<��
>�C����z�@�ff?�@�Ck�                                    Bx�  �          A  ��  @Ǯ>�33@Q�C���  @\?�Q�A�
Ch�                                    Bx4  
�          @�p����@�(��u��CǮ���@�=q?(��@��C!H                                    Bx#�  T          @�\)���\@�\)=�?uC�q���\@�z�?c�
@�C��                                    Bx2�  �          @�(��vff@�{��\)�}�C���vff@��Ϳ^�R��HC!H                                    BxA&  T          @�ff��(�@��Z=q��(�CG���(�@�ff�%���C
                                    BxO�  T          A�
��
=@����������CB���
=@�p���Q�����C��                                    Bx^r  
�          A�����@��������HC�����@���w���p�C�)                                    Bxm  
�          @������@l(���33�
�C�����@��H�Z=q��C	
=                                    Bx{�  �          @�ff�S33@�(�� �����B��S33@�zῇ��33B�                                    Bx�d  �          @���@��@w
=�{��Q�B���@��@�\)���
��\)B���                                    Bx�
  �          @��H��@~�R�b�\��(�Ck���@����3�
��ffC                                    Bx��  	�          @ٙ���p�@0���{���
C� ��p�@XQ��Z=q��C��                                    Bx�V  
Z          A
ff�}p�@�R��Q��`Q�C�)�}p�@U�����J  C�)                                    Bx��  �          A����
=@�{�vff���C@ ��
=@�Q��:�H����C
J=                                    BxӢ  
�          Ap���z�@��\��  �%��C����z�@�G��@  ���C�                                    Bx�H  
(          A>�\�33A�<��
=�Q�CG��33A  ?�@ۅC��                                    Bx��  T          AQ��{A�R>��?��
C�f�{A�?�
=A�CE                                    Bx��  �          A`Q�� (�A�?��@\)C�\� (�A(�@
=qA=qC}q                                    Bx:  $          AW�
�'
=A�\?�{@��C&f�'
=A@\)A*�RC(�                                    Bx�  �          ADQ���@�z�?�33A33C�R��@�\)@C�
Ag�
CW
                                    Bx+�  �          A>ff��z�@�z�@��Bp�C����z�@�\)@�  B��C�                                    Bx:,  
�          A7���G�@�ff@���B
��B��3��G�@ȣ�@�p�B'��C�3                                    BxH�  �          A)���z�@�
=@�ffA�{B�z���z�@ȣ�@�=qB�\C�3                                    BxWx  �          A (����R@��RAp�Bb�B�Ǯ���R@dz�A��B��B�W
                                    Bxf  
�          A@z��"ff@Å���#�
C�3�"ff@���?^�R@�ffC33                                    Bxt�  
�          AG33�#33@�(�@�A)��C���#33@�p�@U�Ayp�C�\                                    Bx�j  
�          AA����@��
@G�An{C	����@�Q�@��A�{CB�                                    Bx�  �          A;
=��  @�\)@�p�A�=qC���  @�  @��B��C�                                    Bx��  T          A6{�  @�Q�@S33A�\)C��  @��@���A�Q�C�f                                    Bx�\  
�          A:ff���H@�ff@�\)A��
C	�����H@�{@׮B�
C:�                                    Bx�  �          A<���Q�@ʏ\@�p�A�C��Q�@��\@�(�A�z�CB�                                    Bx̨  "          A>�H�
=@�G�@P��A~�\C{�
=@�{@�A��HC�3                                    Bx�N  
�          A@���33@�
=@Z=qA��CQ��33@�33@��A�=qC�                                    Bx��  �          AFff�&{@�Q�@&ffAA��C�f�&{@���@c�
A���C�3                                    Bx��  �          AG33�(Q�@�{?�=q@��C�=�(Q�@Å@%�A?�
C�H                                    Bx@  �          AD���,  @�p��u��\)C&f�,  @��?aG�@��Ck�                                    Bx�  
�          AC33�%@�(�>��@(�Cc��%@�
=?�(�@�p�C�                                    Bx$�  
�          A>�H�
=@ƸR��Q��33C��
=@�z�0���UCB�                                    Bx32  �          A9���{@�����33���HCG��{@�z��^{��Q�Ck�                                    BxA�  �          A5��߮@�Q�����'=qC�=�߮@�(���
=�\)C�                                    BxP~  �          A7�����@��\��Q��+ffC�f����@�\)���
��C��                                    Bx_$  �          A8�����@�p���ff��C�����@����
���HCu�                                    Bxm�  "          A=���@����33����C	O\��@�Q���ff���HCk�                                    Bx|p  T          A9���\@O\)��ff���HC �q��\@�(����
���HCT{                                    Bx�  
�          A?\)�޸R@�ff����$(�C���޸R@�����  �CT{                                    Bx��  �          AF�\�@P  ������C ^��@��
��{�
  C\)                                    Bx�b  �          AE�ff@�\��  �ffC'Ǯ�ff@L�����H��C!.                                    Bx�  �          AC\)�(�@J=q���H� �C .�(�@�=q������C��                                    BxŮ  "          AD�����@-p��	��1�HC"����@�  ����$Cn                                    Bx�T  T          AE��ff@<�����:�\C����ff@�G���R�,  C��                                    Bx��  T          A?���@=q�=q�Hz�C!����@qG��
�\�:Cz�                                    Bx�  "          A@z��	��?��H����.�RC,  �	��@p�� (��'p�C$\                                    Bx F  "          A:�R��
@"�\��Q��'��C"�H��
@l(���G����C�f                                    Bx�  �          A@Q��
=?�z��ff�1=qC(�H�
=@:=q� ���((�C ��                                    Bx�  
�          A>{��?��H�=q�-��C*B���@*�H���\�%p�C"�=                                    Bx,8  
�          A=���=q?�R��R�T
=C.�q��=q@����M�C#�R                                    Bx:�  
�          A9����\@(Q����
���C"!H��\@\(������C)                                    BxI�  $          A8����@�=q��=q��(�Cٚ��@���:�H�hQ�C�                                    BxX*  "          A8  �"�\@����G��C�\�"�\@�ff��  ��  C�H                                    Bxf�  �          A<  ��ff?\�33�^�RC&�q��ff@<�����S�Cn                                    Bxuv  �          A=G���Q�@�H���KffC L���Q�@l����=Q�CG�                                    Bx�  "          AA�����@hQ��33�.��C�q����@�����=q��\CJ=                                    Bx��  �          A>ff��33@�����\�2
=C�3��33@���\�
=C��                                    Bx�h  �          A=����
@����(���p�C
�
���
@�ff���H��G�C��                                    Bx�  �          AD���3\)@������z�C(=q�3\)@;��vff����C%W
                                    Bx��  �          AB=q�*�H@`����ff���RC!�=�*�H@�(��xQ���  C�)                                    Bx�Z  �          AD  �.{@N�R�������HC#z��.{@w
=������  C xR                                    Bx�   T          AE��(z�@��H��p���z�CW
�(z�@�z��Z�H��
=C�H                                    Bx�  
�          AD  �,��@j�H������C!=q�,��@�Q��n�R��ffCz�                                    Bx�L  �          AB�\�(z�?�ff������  C*G��(z�@(Q���\)��(�C%�q                                    Bx�  T          ADQ�� ��@:=q��G���C#޸� ��@r�\�������CY�                                    Bx�  �          AG33�&�\@^�R������=qC!��&�\@�����
=����C�f                                    Bx%>  "          AC
=�*=q?��������C,� �*=q@
=q��p���Q�C(��                                    Bx3�  �          AHQ��4z�>\)������
C3E�4z�?h������ƸRC/c�                                    BxB�  T          AMp��/�@`����p��ə�C"@ �/�@�Q�������C��                                    BxQ0  
�          ALQ�� (�@����������C�� (�@�����z���p�C��                                    Bx_�  �          AIG��#\)@������H�ݙ�C���#\)@�ff��������C
=                                    Bxn|  "          AG
=�   @��H���
��=qC� �   @�p���
=��33C�=                                    Bx}"  �          AF�R�#
=@w������Q�C.�#
=@�{���
���HCJ=                                    Bx��  �          AN�\��\?�Q�� (��{C*L���\@3�
����
C$&f                                    Bx�n  "          AMG��/
=@�(��u��33C���/
=@��
�E��`��C�\                                    Bx�  T          AR=q�4��@�Q����R���HC{�4��@��H�l�����C��                                    Bx��  �          AUG�� (�@K���p���C"ff� (�@�\)��z��=qC�                                    Bx�`  �          AM����(�?���"ff�O�C,^���(�@�R�ff�H��C"�\                                    Bx�  �          A\Q����H���@Q��t��C8�����H?�z��?��s=qC*                                      Bx�  "          A]���>aG��<  �g�C2E��?�\)�9���c(�C%�                                     Bx�R  T          AZff��(�>W
=�4���_�
C2p���(�?���2�\�[�C&�\                                    Bx �  "          AO33��Q�>W
=�(���Z  C2c���Q�?�
=�&�\�V  C'aH                                    Bx�  �          AM�����?=p��%p��V�RC.h����@
�H�"=q�P�C#��                                    BxD  �          ALQ����?Y���*�H�c��C-����@��'\)�\�C!ff                                    Bx,�  �          AMp���p�?Y���.�H�i�C,�q��p�@
=�+\)�b=qC �                                     Bx;�  T          AW\)��Q�?333�<  �s�C-�{��Q�@��8���l\)C T{                                    BxJ6  "          AZff��ff?s33�;��lQ�C,33��ff@#�
�7��d�\C�                                     BxX�  T          AXz���Q�?J=q�7�
�i��C-�\��Q�@��4Q��b�
C!Q�                                    Bxg�  T          AT����=q?��\�333�eG�C)ٚ��=q@333�.�R�\�\Ck�                                    Bxv(  T          AP(���\@�\�33�A  C&aH��\@Vff�G��7\)CE                                    Bx��  �          AS\)�33@.�R�G��-��C#u��33@|���	�"��C�                                     Bx�t  
�          AR{�(�@Q��=q�=qC!k��(�@��
���C��                                    Bx�  
�          AP���"�R@e��33�	{C �\�"�R@����G����RCٚ                                    Bx��  �          A0��� ��?���Q��%{C&�
� ��@8������C =q                                    Bx�f  
�          A$Q��=q@s�
��G���C��=q@z=q���Dz�CW
                                    Bx�  �          A%���@���@G�A4��C�R���@��\@/\)Av{CL�                                    Bxܲ  "          A%����@o\)@�ffB9Cc����@5�@���BK=qC�                                     Bx�X  �          A#�
�Mp�@��HA
�RBhp�B���Mp�@HQ�A�HB�  C�q                                    Bx��  
�          A%��y��@�z�AG�BW�HC
=�y��@^{A{Bn=qC
Q�                                    Bx�  �          A=q�\@��H@���B�Cz��\@~�R@���B)�HC��                                    BxJ  "          A�H��
=@#33A ��BeCT{��
=?�Q�AG�BrQ�C"+�                                    Bx%�  T          A!���?
=A  B�\)C%�����:�HA�
B��CEff                                    Bx4�  �          A+
=����ff@�Q�B9p�CB)���5�@�ffB/��CI�                                    BxC<  
�          A ���񙚿�(�@��A�(�C@޸���z�@�G�A�=qCE
                                    BxQ�  "          A Q��
{=��
@��RA��HC3s3�
{�z�@�p�A�
=C7ٚ                                    Bx`�  "          A z���H@   ?�G�A!�C(W
��H?�  @ ��A8z�C)Ǯ                                    Bxo.  
�          A��@>{?У�Az�C"&f��@.�R@ ��A<��C#��                                    Bx}�  �          AQ���{@{�@I��A�z�C�f��{@^�R@i��A�ffCY�                                    Bx�z  "          A&{�
=q@�ff�#�
�g�C��
=q@��=u>���C�                                     Bx�   
Z          A�����
@�  �L�;��
C�����
@��R?(�@w
=C�f                                    Bx��  "          @�R���@��׿�=q�
�RC�����@�(��
=q���C��                                    Bx�l  "          A����R@���B�\����C�{���R@�
=?�@�G�C��                                    Bx�  T          Aff��p�@�p���p��(Q�C}q��p�@��H�n{��z�C��                                    Bxո  T          A$����@�(���
=��\Cs3��@����c�
���RC�                                    Bx�^  T          A/
=���@��
�HQ���ffCO\���@�
=����8��C	��                                    Bx�  �          A.�R�\)@��
�vff��G�C���\)@��H�Mp���z�Ch�                                    Bx�  "          A333�$��@dz�?J=q@��
C �f�$��@\(�?�  @љ�C!��                                    BxP  
�          A5���(�@�G���
=��33C	J=��(�@����QG����C\)                                    Bx�  �          A1��(�@�\)��\�=\)C
���(�@�  ��{�(\)C+�                                    Bx-�  
�          A.�\��  @�Q��\)�F(�C���  @������
�3�\C
�3                                    Bx<B  �          A)����R@^�R��=q�!C@ ��R@��H����p�C�R                                    BxJ�  �          A,������@��
�����(�C^�����@�
=�����\)CG�                                    BxY�  T          A.�\��\@�z������Q�C�\��\@�\)��(����HC�                                    Bxh4  T          A,z���@���\��C'���@.�R��Q���z�C#T{                                    Bxv�  �          A.=q�G�@���?.{@�z�C�3�G�@�?�p�@���CaH                                    Bx��  T          A-G��G�@���.{�k�C{�G�@�33>�@��C(�                                    Bx�&  �          A,����@�G��k���
=C�H��@��
���ÿ�G�C:�                                    Bx��  �          A,���  @�  �.{�k�C=q�  @�\)>�@"�\CW
                                    Bx�r  �          A,  �G�@�33���6ffCW
�G�@��
>L��?�ffC@                                     Bx�  �          A*=q�(�@�z�@'
=Ai�Cٚ�(�@�Q�@N{A��\C�                                    Bxξ  T          A(���Q�@���@`  A�p�CJ=�Q�@j�H@�Q�A���C��                                    Bx�d  �          A*{�z�@�=q@o\)A���CL��z�@�G�@���AŅC��                                    Bx�
  
�          A%G����@Z�H@љ�B(�Cz����@#33@�{B)�C h�                                    Bx��  "          A%����H@O\)@��
B0z�C�
���H@33@�\)B;�HC �=                                    Bx		V  "          A)p�� ��@���@�A�33CY�� ��@l��@�ffB�
CQ�                                    Bx	�  �          A(  �=q@��R@��HA��C��=q@�z�@�{A�
=C��                                    Bx	&�  
�          A(���z�@��@���A��
C5��z�@���@��
AݮC^�                                    Bx	5H  
Z          A+
=�
�H@�ff@��\A�p�C�)�
�H@x��@��A��HCٚ                                    Bx	C�  T          A.=q���@*�H@��\A�=qC$z����@�@�z�A�ffC'��                                    Bx	R�  f          A2ff��
@N�R@أ�B��C����
@@��
B
=C$�                                    Bx	a:  �          A5��\@i��@�
=BC�H��\@.�R@��
B"p�C!u�                                    Bx	o�  �          A2�R�
=@I��@��A��C �)�
=@Q�@��B��C%{                                    Bx	~�  �          A3�� (�@�����(��
=C��� (�@�������33Ch�                                    Bx	�,  �          A@(����@i������+��C8R���@�����=q�
=C��                                    Bx	��  �          A333��ff@���?n{@��C.��ff@��
?��HA  C�                                    Bx	�x  �          A4����ff@�  @�G�A�\C\��ff@�  @�Q�Bz�C��                                    Bx	�  �          A8z���\)@�(�@G
=A�  Cc���\)@�p�@��A�Q�C	(�                                    Bx	��  �          A;33����A�@N�RA�G�C�����@��@�  A�Q�C:�                                    Bx	�j  �          A>�\����@�33@��B�Cs3����@��
A�B/��C�)                                    Bx	�  �          A:=q���@�{@/\)Ac33C����@�G�@eA�Q�CǮ                                    Bx	�  �          A0  ��@�ff?W
=@��C��@�G�?��A��Cp�                                    Bx
\  �          A2�H�
=@�  ��p���33C)�
=@����l(���C�3                                    Bx
  �          A5G����@�����
=�G�Ck����@�=q��{����C	�f                                    Bx
�  �          A1��G�@�p���p��'=qC���G�@ᙚ�ҏ\�
=C (�                                    Bx
.N  �          A2=q���@��
���H��=qB�Ǯ���A���������B���                                    Bx
<�  �          A3�
����@�Q��l�����RC������A�R�,���_33C33                                    Bx
K�  �          A2=q��R@������
C����R@���\(���  CT{                                    Bx
Z@  �          A/��=q@�Q��2�\�mp�C��=q@�=q�
=�2ffC��                                    Bx
h�  �          A/�
�"�H@~�R��������C�f�"�H@�33�0���g�C�                                    Bx
w�  �          A0z��#33@vff?�{@�(�CJ=�#33@i��?�A�C J=                                    Bx
�2  �          A.=q�  @�  ��(��Ch��  @���>k�?�p�CW
                                    Bx
��  �          A0����
=@���?��R@��
C
{��
=@�{@Q�A6ffC
�                                    Bx
�~  �          A3\)���@��ÿ����C�R���@��R������z�C(�                                    Bx
�$  �          A0Q��\)@�=q�!��VffC  �\)@����(��&{C�)                                    Bx
��  �          A5���{@hQ������
Cٚ�{@�  ���ÅC�H                                    Bx
�p  �          A>�R���@)������(G�C"E���@g���R�{Cp�                                    Bx
�  �          AHz��G�@L(����#z�C !H�G�@�����ffC��                                    Bx
�  �          AF�H���@$z���Q���C$�3���@_\)��(��33C�                                    Bx
�b  �          AH�����?����
=�#33C(#����@=p���(���
C"Q�                                    Bx
  �          AK
=�ff@�����ff�
��C���ff@������H����C��                                    Bx�  �          AI��@333���
�ffC#^��@p����ff��HC�                                    Bx'T  �          AI�
=<�����%33C3�{�
=?������#�C-�\                                    Bx5�  �          AI��Q�\���
�G�C633�Q�?(�������C0k�                                    BxD�  �          AG
=�p�?����(�C+�\�p�@���(��C&��                                    BxSF  �          AM�� ��@$z����z�C%�)� ��@_\)���(�C ��                                    Bxa�  �          AQ��7
=@3�
���
��(�C&0��7
=@_\)��\)���HC#                                    Bxp�  �          ANff�*�H@��������Ǚ�C�*�H@�p���G����
C+�                                    Bx8  �          AO\)��R@�33��z��ffC����R@�ff��(���33C޸                                    Bx��  �          AO����@�  ���H��33C�3���@�������p�CJ=                                    Bx��  T          AO33� z�@�\)���H��  C��� z�@�\)�������C@                                     Bx�*  �          AS\)�+�@�33���\��33C���+�@�G���z����C��                                    Bx��  �          AO33��@����z����Cp���@�=q�������CE                                    Bx�v  �          AJ{��\@�����)\)C����\@�  ��ff��RC	\)                                    Bx�  �          AIG���\@������!(�C���\@��
������Cٚ                                    Bx��  �          AO�
��(�@�\)�G��,
=C
����(�@�G���\)�G�CY�                                    Bx�h  �          AO����@�����R���C�����@�z�������C�
                                    Bx  �          AO��
ff@���ʏ\��G�C�\�
ff@�G����ʣ�C
��                                    Bx�  �          AN�\��\)@������C�f��\)A����=q���C��                                    Bx Z  �          AO\)�'�
@�Q���{��\)C���'�
@�����
��Q�C�                                    Bx/   �          AP(��:�R@>�R��
=��
=C%���:�R@c33������C#
                                    Bx=�  �          AP���3
=@�\)���R��=qC  �3
=@�  �s�
��C��                                    BxLL  �          AP���'
=@ʏ\��ff��=qC��'
=@��H�g����CǮ                                    BxZ�  �          AO�
��@��\��G����HC����@�Q���Q����\C޸                                    Bxi�  �          AQ�ff@�(����ᙚCE�ff@Ӆ��z����HCB�                                    Bxx>  �          AP���@����Q���
=CE�@�33��\)��=qC33                                    Bx��  �          ALz��%��@��H��Q��ׅC{�%��@�����(����RC�q                                    Bx��  �          AP���#\)@��R��z����HC��#\)@�
=���R��(�C                                    Bx�0  �          AW33�"�R@�����Q��癚C�)�"�R@�p����R��33C��                                    Bx��  �          AUp�� ��@�(��ָR��C� � ��@�p���ff��  Ch�                                    Bx�|  �          AT��� Q�@������H�G�Cff� Q�@��
������
=C�f                                    Bx�"  �          AT  �"{@����������
C���"{@����{�ՙ�C)                                    Bx��  �          AQ�!G�@�ff���H�
=C+��!G�@����\)���HC:�                                    Bx�n  �          APQ��!�@�����ff�33C���!�@�z��˅��  C�                                    Bx�  �          AIp��"�H@����G����C��"�H@�G���ff����C�                                    Bx
�  �          A<����@W
=��G�����C �q��@����=q��(�C
                                    Bx`  �          A;33���@X���������C �f���@�G����H��ffC��                                    Bx(  �          AAG��-G�@#�
��33��C&�R�-G�@G�������C#�3                                    Bx6�  �          A8���!��@-p������33C$�q�!��@R�\������RC!��                                    BxER  �          A:�R� ��@L(�������ffC"n� ��@p����=q��p�C�                                     BxS�  �          A9p��#
=@�  �^�R��{C\)�#
=@����:�H�i��Cs3                                    Bxb�  �          A<���)p�@�G��*=q�Q�C���)p�@��H�ff�$��C�\                                    BxqD  �          A>�\�0Q�@�=q�����C���0Q�@�  �������C��                                    Bx�  �          A=�(��@��H��\)��=qC+��(��@��\>���?���C5�                                    Bx��  �          A8  �33@�=q�Tz����C��33@�(����z�C��                                    Bx�6  �          A8(�� z�@��
��
=���C�
� z�@�Q�=p��k�C:�                                    Bx��  �          A$����Q�@�
=@�\)Bz�C	�{��Q�@�\)@��Bp�C��                                    Bx��  �          A%p���G�@ҏ\@�(�A�C���G�@��@�Bp�C�                                    Bx�(  �          A%�����@�z�@��B��B������@�z�@ȣ�B�RB�                                      Bx��  �          A3\)����A	��@  ��33B������A	��>�33?�
=B���                                    Bx�t  �          A$����\)@hQ�@�Q�A��\C���\)@=p�@�B{Cc�                                    Bx�  �          A2�\�	�@ᙚ�.{�`  C�R�	�@�\>aG�?�33C��                                    Bx�  �          A<Q���Q�A33��Q��{C�{��Q�A	p�����<��C
                                    Bxf  �          AA����A�33��C�����A�ÿxQ���=qC�q                                    Bx!  �          AC��(�@�=q��ff��ffC�{�(�@�z�#�
�B�\C�{                                    Bx/�  
�          A=��Q�@�=q��33�Q�C���Q�@�  �p����33CE                                    Bx>X  �          A'
=��@љ��.{�rffCB���@�33�����$Q�C
�                                    BxL�  �          A*�R��  @��Tz���C@ ��  @ᙚ���QG�C	�R                                    Bx[�  �          A{��ff@�33�^�R���HC
�R��ff@�  �*�H�zffC�                                    BxjJ  �          A!p���(�@�
=�8����\)C�
��(�@�G���
�=G�C
k�                                    Bxx�  �          A���\)@7
=@(Q�A���C���\)@   @>{A��HC\)                                    Bx��  �          @�p����Ϳ�p�@�Q�B�{CcB������)��@�\)B�(�Cn�                                    Bx�<  �          @�녿���(��@��B��fCw�
����\��@�Q�Bk�HC|u�                                    Bx��  �          A  �P��@�@�\)Bv�
C�f�P��?�@�{B��C :�                                    Bx��  �          AG��u�@G�@��HBq�C��u�?��
@�G�B|{C%�                                    Bx�.  �          A�qG�?xQ�@��RB�RC%���qG���\)A Q�B���C5
=                                    Bx��  �          A�H����?��@�33Bw��C+�q���;�
=@��
BxG�C9��                                    Bx�z  �          A\)��=q>��@�\)Bdz�C/)��=q�\)@�
=Bd  C:�                                    Bx�   �          A  �x��    A ��B�Q�C4��x�ÿ�ff@�
=B|�CC\                                    Bx��  �          AG�����?L��@޸RBQ��C+^�������G�@�Q�BS�\C50�                                    Bxl  �          A(��<��?��A��B�G�C�{�<��=��
A�HB�G�C2�                                    Bx  �          A�R��{<#�
AffB�=qC3�ÿ�{���AG�B�L�CQ�=                                    Bx(�  �          A����<��
A z�B���C3� �녿��
@��RB��CO�                                    Bx7^  �          @���(���=q@�(�B��C:���(����\@��B���CR                                      BxF  �          A	G�����?�@J�HA��
C&xR����?�z�@X��A�ffC)�{                                    BxT�  �          A	p���=q@Y��?G�@�Q�C���=q@P��?�(�A�CW
                                    BxcP  �          A	����=q@c�
<��
>�C� ��=q@a�>��H@P��C�3                                    Bxq�  �          A33��@\�;�p��!G�C�=��@^{=�Q�?��C��                                    Bx��  �          @����Q�@N{��p��-��CY���Q�@XQ쿅���C:�                                    Bx�B  �          A z���@S33��(��(��CO\��@]p����\��=qC:�                                    Bx��  �          A���\)@XQ��G��+
=C�f��\)@c33�����z�C�\                                    Bx��  �          @��\��\@Mp��n{�ٙ�C�H��\@S33���H�g
=C�q                                    Bx�4  T          @�����Q�@QG��z���Q�C����Q�@Tz�\)��  C��                                    Bx��  �          @������@Tz�:�H��(�C�3����@X�þ�\)��C@                                     Bx؀  �          @�����33@C�
�B�\���HC�3��33@HQ쾮{�!G�C5�                                    Bx�&  
�          @�\)��@g�����(�C�
��@j=q�#�
��33CQ�                                    Bx��  T          @����@i�����w
=C^����@k�<#�
=��
C&f                                    Bxr  �          A���33@�Q쿋����HC����33@��
�
=q�p��C޸                                    Bx  �          @���Q�@]p���=q�
=C����Q�@g
=�\(��ƸRC�                                     Bx!�  �          @��
��Q�@Vff��33��HCu���Q�@^{�0����G�C��                                    Bx0d  T          @�(���33@j�H������C�{��33@s�
�O\)��(�C�f                                    Bx?
  �          @�z��У�@4z�\�Dz�C�{�У�@6ff    <�Cc�                                    BxM�  �          @ڏ\��(�@33?B�\@�C :���(�@
�H?���A  C!8R                                    Bx\V  �          @����\)@�?
=@��HC##���\)@G�?^�R@��C#�H                                    Bxj�  �          @�G�����@G����
�L��C"� ����@p����R�%C �R                                    Bxy�  �          A33��(�@-p���ff��\)C �q��(�@U���������C
                                    Bx�H  T          A"{���@K���=q��C\���@s�
���H��33CaH                                    Bx��  �          A���@2�\��33�陚C!E��@[�����\)CaH                                    Bx��  T          A\)����@������ԸRC$�����@333�p����p�C s3                                    Bx�:  T          A
�\��33?���g��Ə\C&#���33@
=�U�����C"�=                                    Bx��  �          A	���z�?��R�������
C(�{��z�@�\�s�
���C$��                                    Bxц  �          A\)��@�R������G�C"�H��@333�z�H��z�C�H                                    Bx�,  T          @�
=���@G��W
=��=qC#�����@p��C33����C c�                                    Bx��  �          A{�߮@\)�aG����C":��߮@-p��K����
C�{                                    Bx�x  �          A\)��\)@��%�����C#����\)@!G��  �
=C!aH                                    Bx  �          @�ff��(�@��4z���ffC#� ��(�@��� ������C!�                                    Bx�  
�          A=q���?�33�����Q�C%�����@{�
=q�v�RC#��                                    Bx)j  �          A{�
=@���z��s
=C#���
=@,�Ϳ��H�L��C!                                    Bx8  T          A�H�{@+���ff��HC"B��{@5�k����C!J=                                    BxF�  T          A	��R?޸R����FffC(Y���R?��
�aG�����C(
=                                    BxU\  �          A{�
=?�G�?fff@�z�C)�{�
=?�\)?���@�G�C*��                                    Bxd  �          A
=��@Q�>��@L��C#����@�\?L��@�ffC$#�                                    Bxr�  T          A
ff�33@/\)���
�
=qC!}q�33@.�R>��R@�C!�
                                    Bx�N  T          @�Q���  ?��?�
=Ac33C'�R��  ?���?�{A|(�C*)                                    Bx��  �          @�ff�ʏ\?�@333A��C%33�ʏ\?�G�@@��A�=qC(�                                    Bx��  T          @��ҏ\?���@1G�A�(�C&W
�ҏ\?�Q�@>{A��C)                                    Bx�@  �          @߮��=q?�Q�@�Q�Bz�C'�3��=q?(�@�(�B�C-�=                                    Bx��  T          @�33����?.{@`��A��C-������>8Q�@dz�A�\C2W
                                    Bxʌ  �          @��\��?��R?��A z�C%���?�?�\)A ��C&ff                                    Bx�2  �          @�\)��  ?Ǯ?��@�  C(G���  ?��?�G�Ap�C)�                                     Bx��  �          @�ff��G�?�G�?^�R@У�C*z���G�?���?�ff@�=qC+��                                    Bx�~  �          @��R���H?�p�?.{@�z�C+����H?�\)?Y��@ÅC+��                                    Bx$  �          @����
=?�\)?��@�G�C)���
=?��
?@  @�{C*��                                    Bx�  �          @�p���33?��
?^�R@���C$�f��33?У�?�\)A�C&
                                    Bx"p  �          @�\�ۅ@p�?0��@���C J=�ۅ@�?��AffC!:�                                    Bx1  �          A��(�@L��?.{@���CaH�(�@Dz�?��@��C +�                                    Bx?�  �          A(��	�@U?�@S33C���	�@N�R?�G�@�G�Cz�                                    BxNb  �          A��{@L(�>Ǯ@%C�\�{@Fff?\(�@�(�C!H                                    Bx]  T          Az���
@>�R?�@mp�C ���
@7
=?�  @���C �
                                    Bxk�  T          A{��G�@HQ���\(�C#���G�@G
=>���@�C=q                                    BxzT  
�          @������@   ?�@{�C!�����@��?c�
@�z�C!��                                    Bx��  �          @�
=�У�@Mp�?�\)AH��C�{�У�@;�@z�A�G�C��                                    Bx��  �          @���\)@^{>��@H��C����\)@W�?k�@�  Cff                                    Bx�F  �          @��H��
=@Tz�?�z�AN{C�)��
=@A�@��A�33C��                                    Bx��  �          @�
=�˅@AG�?��HA;33C���˅@1G�?��At  C}q                                    BxÒ  �          @�\)���H@O\)?�(�A  C����H@AG�?�Q�AX��C�                                    Bx�8  �          @����޸R@e�?�=q@��C�=�޸R@XQ�?���A;\)C�                                    Bx��  �          A�\��\@^{�.{��=qC����\@b�\�#�
���CB�                                    Bx�  �          Az����H@r�\�Ǯ�-p�C����H@s�
>8Q�?��
C�{                                    Bx�*  �          A Q��޸R@O\)@p�A��\C\�޸R@7�@*�HA�z�C��                                    Bx�  �          A\)���H@Y��@�=qBp�C�����H@*�H@�G�Bz�Cs3                                    Bxv  �          A�
���@�(�@��BQ�C	����@e�@���B*�\CQ�                                    Bx*  T          A{����@S�
@�z�A�
=C������@(��@��HB	��C�                                    Bx8�  �          A  ��Q�@QG�@3�
A���C����Q�@3�
@Q�A��C0�                                    BxGh  �          AG���{@6ff@}p�A�ffC����{@{@�33A�\)C!��                                    BxV  �          A	p���(�@3�
@���B*�Cz���(�?��@��B8=qC!xR                                    Bxd�  �          A	����@1�@�z�B-�
CaH����?���@ǮB;p�C!��                                    BxsZ  �          A(�����@p�@���B:Q�C������?�p�@θRBF�RC$J=                                    Bx�   �          A�\��=q@�R@��RB5��C  ��=q?��
@ȣ�BB(�C#�3                                    Bx��  �          A�����R@,��@�B533C�����R?�  @���BC33C!p�                                    Bx�L  �          A=q���
@8Q�@�\)B633C�)���
?�@˅BE\)Cu�                                    Bx��  �          A
=��\)@E@�=qB8  C(���\)@
=@�
=BH�\C�                                    Bx��  �          A\)����@L(�@���B,�\C������@  @ƸRB=
=C�f                                    Bx�>  �          A	p����@G�@�ffB8��CaH���@
=@ӅBIQ�Ch�                                    Bx��  T          AQ����@QG�@��BI�CǮ���@��@�  B]G�C�                                    Bx�  T          A	�����@]p�@ÅB6
=C�{����@��@�=qBH�RCs3                                    Bx�0  �          A
{��\)@K�@\��A��RCE��\)@%@z=qA�C E                                    Bx�  �          A	��� (�@5?n{@ǮC z�� (�@*=q?�\)A\)C!��                                    Bx|  �          A�
��@�Ϳ�G��
=C%0���@=q�����  C#�{                                    Bx#"  �          A	��\@���
=���C${��\@!녿����{C"��                                    Bx1�  �          A���@�H����+�C#����@)����p�� ��C"W
                                    Bx@n  �          A�R��@(�ÿ��;
=C"k���@8�ÿ�{�Q�C �{                                    BxO  �          A33�ff@<���33�g�
C �f�ff@Q녿�ff�4Q�C��                                    Bx]�  �          A  �33@=q�5��\)C$
�33@5��H�r�RC!s3                                    Bxl`  �          AQ����@G��Y������C$����@2�\�?\)���HC!ff                                    Bx{  �          A��  @a�>Ǯ@'�C}q��  @Z�H?u@�ffC+�                                    Bx��  �          Az���\@�p�@�HA�=qCh���\@n{@Dz�A��C!H                                    Bx�R  �          A���{@���@&ffA��\Cn��{@dz�@N�RA���CY�                                    Bx��  T          AG�����@~{@-p�A�33C�=����@^{@U�A��C�)                                    Bx��  �          AQ����H@���@H��A�(�CG����H@^�R@p��A˙�C޸                                    Bx�D  �          Ap���{@�\)@H��A�z�C&f��{@y��@u�AͮC��                                    Bx��  �          Az���G�@�  @Z=qA�
=C��G�@��
@���A�C�=                                    Bx�  �          A(���@��@[�A�=qC�H��@�\)@�A�Q�C�f                                    Bx�6  �          A(���=q@���@7
=A���C����=q@��@eA�{C�f                                    Bx��  �          A33��R@}p�@�
A[33C���R@c�
@,(�A�(�C}q                                    Bx�  �          A
=q��p�@w
=@��Ae��C���p�@\��@0��A��HC{                                    Bx(  �          A�����H@y��@   AXQ�C  ���H@`��@(Q�A�
=Cn                                    Bx*�  �          A  ��ff@�{?���A@Q�C����ff@tz�@   A��HC�
                                    Bx9t  �          A���p�@�  @Q�Aa��C�f��p�@�=q@6ffA��RCaH                                    BxH  �          A���\)@��\?�\)A+\)C����\)@��@�A|z�C��                                    BxV�  �          A
{��ff@�p�?��HA7
=Cp���ff@�=q@(�A�33C�                                    Bxef  �          A\)��G�@�?�(�A6�\C�R��G�@�=q@��A���C�\                                    Bxt  �          A��陚@�z�?�=qAB=qC���陚@�Q�@#33A�Q�C33                                    Bx��  T          A�
�陚@���?�=qA&ffC&f�陚@�ff@�Aw�C�                                    Bx�X  �          A���G�@���?�=qA&�\C\��G�@��R@Ax  C�                                    Bx��  �          A\)��Q�@�  ?��A-�C33��Q�@���@��A33C=q                                    Bx��  �          AG���R@��?��A+�CaH��R@��@Q�Az=qCh�                                    Bx�J  �          A33����@�Q�?�Q�A4z�C�H����@���@\)A��
C�R                                    Bx��  �          A
�R�?��?&ff@���C'O\�?޸R?u@�C(:�                                    Bxږ  �          A���33?���?   @P��C,�)�33?��
?0��@�  C-G�                                    Bx�<  T          AG����?��R?Q�@��HC&����?�?�z�@���C(
=                                    Bx��  �          A�z�@�?�  AC&�
�z�?��
?˅A%�C(0�                                    Bx�  �          A�\��?�{@ ��AP��C'�)��?�p�@�
Ap(�C*�                                    Bx.  �          A\)�33?�G�@
=At��C(5��33?���@(��A�
=C+!H                                    Bx#�  �          A�
�33@�@�AU�C%���33?��
@�A{33C(
                                    Bx2z  �          A���	G�@>�R?��H@�C ٚ�	G�@.{?�(�A-p�C"n                                    BxA   �          A�R�	G�@w
=>��
?�
=CǮ�	G�@o\)?�  @�33Cp�                                    BxO�  �          A����@��ͽu��Q�Cz���@�33?.{@��RC�=                                    Bx^l  �          Aff�ff@�{�   �C33C��ff@��R>��?���Cff                                    Bxm  
�          A���  @���W
=��G�C}q�  @�=q�u�\C�                                    Bx{�  �          A=q���@��\�^�R��p�C&f���@�p���\)��(�C�3                                    Bx�^  �          A33�
�\@������ȣ�C��
�\@����\)�У�CG�                                    Bx�  �          A{��
@�=q��{�=qC���
@���\)�Tz�C�                                    Bx��  �          A����@�{��� ��C޸���@�p��Tz����C�f                                    Bx�P  �          A�R���@�녿�=q�.ffC=q���@�=q�u��
=C�f                                    Bx��  T          Ap��ff@�  ����+�
C!H�ff@�Q�z�H��33C                                    BxӜ  �          A33�{@{������1C޸�{@�ff������\)Cc�                                    Bx�B  �          A  ���@s�
��33� Q�C�����@����k���=qC�H                                    Bx��  �          AG��z�@~�R�޸R�(  C�q�z�@���z�H���
C��                                    Bx��  �          A�R�z�@�33�����G�C  �z�@��׾��6ffC)                                    Bx4  T          A�\�z�@��׿�\)��\C��z�@�{���.�RC.                                    Bx�  �          Aff���@��׿��R�>=qCc����@�녿�{��33C�)                                    Bx+�  �          A��33@������Xz�C�)�33@�(����R�	��C��                                    Bx:&  �          A���@�=q�Q��`z�C�H��@���Q��ffC��                                    BxH�  �          A$���	�@��\�#33�eCQ��	�@�
=��ff�
�\Cn                                    BxWr  �          A%p���
@�{�p��\(�C���
@�=q��p���
C��                                    Bxf  �          A%�@�(��{�E�C&f�@��R��  ��ffC��                                    Bxt�  �          A&{��@������S\)C����@�Q쿳33��\)C8R                                    Bx�d  �          A#��  @����Q��Xz�C  �  @��׿����33C&f                                    Bx�
  �          A%���=q@���1��yC0��=q@�{��\)�&�HC�                                    Bx��  �          A'��Q�@��"�\�`  C�
�Q�@��\�˅��C�H                                    Bx�V  �          A%���  @���$z��f{C�f�  @�=q�����C��                                    Bx��  �          A$(��=q@�
=�:=q���
C�H�=q@�ff��\�7�C#�                                    Bx̢  �          A#��Q�@���6ff����CO\�Q�@�(���Q��/\)C�                                    Bx�H  �          A$���G�@�p��=p���p�Ck��G�@����\�6�HC�                                    Bx��  �          A$z���@�ff�Mp����C� ��@�  ���QC�                                    Bx��  �          A$���\)@vff�S�
��{CǮ�\)@�p��   �`z�C�                                     Bx:  �          A&{�
=@u��<(����CaH�
=@��\�Q��=p�C�                                    Bx�  �          A#
=��R@�=q����*�RC�H��R@�33�s33���CxR                                    Bx$�  �          A"=q�  @�z��	���D  CE�  @�������=qC�                                     Bx3,  
�          A$���z�@j=q��R�_33CxR�z�@��\�ٙ����CE                                    BxA�  �          A&ff��
@8Q��G����C#)��
@\(��\)�]C �                                    BxPx  �          A"�R�(�@333�HQ����C#&f�(�@W�� ���d��C�q                                    Bx_  �          A\)���@��X����{C&u����@3�
�8����p�C"��                                    Bxm�  �          A����?�33�w
=��(�C)����@���\����  C%8R                                    Bx|j  �          A"�H�z�@�\�i����{C'�)�z�@.�R�J=q���HC#��                                    Bx�  T          A%���(�?�  �l(���
=C)�
�(�@p��P�����HC%�                                    Bx��  �          A%��Q�?�\)�n�R���C+�\�Q�@ff�W����RC'�{                                    Bx�\  �          A��z�?�Q��N{���C+.�z�@�
�7
=��{C's3                                    Bx�  �          A!�����@Vff����-�C .���@j�H�����Cz�                                    BxŨ  �          A#�
���@s�
��\)���C�
���@�녿O\)����C�=                                    Bx�N  "          A#\)�	p�@��>��?��RC��	p�@�{?�{@�{C�)                                    Bx��  �          A!G����@�G�=�?333CQ����@���?��H@�z�C�q                                    Bx�  �          A=q�{@l�Ϳ(���tz�C��{@p��=�?0��C��                                    Bx @  �          A�H�z�@����\���HC%���z�@(�ÿ=p���Q�C$��                                    Bx�  �          A Q���R@
=�Ǯ�33C'����R@�ÿ����
=C&!H                                    Bx�  �          A�\��?\(������{C.����?�{��=q���C-!H                                    Bx,2  �          A33�\)?޸R��  ��\)C)޸�\)?��H�\(���
=C(��                                    Bx:�  �          A���?����n{��33C(.��@ff�   �AG�C'J=                                    BxI~  �          A=q�(�@�G���(�� ��C޸�(�@�G�>�(�@{C�)                                    BxX$  �          A33�  @{���
=�!�Cٚ�  @{�>��@{C�
                                    Bxf�  �          Az����@/\)�=p���ffC#.���@5��\)�Y��C"�H                                    Bxup  
�          A��G�@n�R���
��{C���G�@mp�>�@2�\CǮ                                    Bx�  �          A��p�@q�>\)?O\)C�
�p�@j�H?p��@�33CxR                                    Bx��  T          A\)��@8Q�?G�@�Q�C"33��@(��?���AffC#��                                    Bx�b  �          A���
@8��?�  @��C".��
@#33?�A2ffC$#�                                    Bx�  �          A����@�p�?�z�A�
C���@���@z�Aa�C:�                                    Bx��  �          A=q��ff@��?��\A Q�CaH��ff@�33@��AeC��                                    Bx�T  �          A\)��=q@�z�?c�
@��Cٚ��=q@�=q?�p�AG
=C�                                     Bx��  �          A{��z�@�=q?�=qAG�B�� ��z�@��
@.�RA��B�B�                                    Bx�  �          A��P  @�
=?�ffA((�Bី�P  @�{@L(�A�Q�B�\                                    Bx�F  �          A�
����@�(�>��
@�B�������@���?���AO
=B���                                    Bx�  �          @�{���@�=q?fff@�B��f���@θR@�A��HB�L�                                    Bx�  �          @�Q���@˅?&ff@�
=B����@��@�A��\B�                                    Bx%8  �          @�{�^{@Ǯ>��@Q�B���^{@��?�\AeB�{                                    Bx3�  T          @�=q�Tz�@��
<#�
=�\)B��f�Tz�@�
=?�{A4Q�B�#�                                    BxB�  �          @׮��@ҏ\�\)��B�G���@�ff?���A5p�B�Ǯ                                    BxQ*  �          @�Q��z�@���(��i��B�{��z�@��
?�G�A
�\B�ff                                    Bx_�  �          @�p��#33@Ӆ��  � ��B�8R�#33@�  ?�p�A   B��)                                    Bxnv  �          @�\)���@�\)�z���B�����@�ff?Tz�@�33B��                                    Bx}  �          @��H��\)@��ÿ0����33B�녿�\)@�Q�?Y��@���B�                                      Bx��  �          @�R���H@�=q�(����  B�k����H@���?s33@�=qBǀ                                     Bx�h  �          @�G��Y��@�p���G��o\)B{�Y��@��H?�=qA33B½q                                    Bx�  �          @�
=?0��@أ�?s33@�{B�G�?0��@�(�@p�A���B��\                                    Bx��  �          @���@   @�  @�A�  B�(�@   @�G�@n�RB�B���                                    Bx�Z  �          Aff�k�@�R�������B���k�@�\?��A�B��                                    Bx�   �          A�
��ff@ۅ�\(�����B�
=��ff@��
?:�H@�=qB��                                    Bx�  �          A{��p�@�ff�aG��ǮB�����p�@ʏ\?�G�A\)B��3                                    Bx�L  �          @�ff����@��;����\)CǮ����@���?���A Q�CB�                                    Bx �  �          A ������@���B�\���C	� ����@�{?�\@h��C	W
                                    Bx�  �          @�  ��p�@�=q���(��C
=q��p�@�Q�W
=���
C	0�                                    Bx>  �          @������@�33��z��
�RCff���@��    ���
C��                                    Bx,�  �          @�����(�@����\��\C	c���(�@��\���L��C��                                    Bx;�  �          @��H��Q�@�\)���
�33Ck���Q�@��;L�Ϳ�(�C^�                                    BxJ0  �          @����z�@����\)�#33Cu���z�@�G��L�Ϳ��HCff                                    BxX�  �          @�������@��R�8Q���{C  ����@�\)?z�@��C�                                    Bxg|  �          @�33���
@�Q쿡G��z�C�����
@�p��L�;���C
��                                    Bxv"  �          @��R��33@�=q����\)C(���33@�\)�����C�                                    Bx��  �          @����p�@��\�(�����C:���p�@�
=�����{C�3                                    Bx�n  �          @�
=��\)@��H� �����RC� ��\)@�=q���H�4z�Cs3                                    Bx�  �          @�ff��@x���8Q���C����@��R��\)�hz�Cٚ                                    Bx��  �          @�(���p�@e�Z�H��  C����p�@���p����\C                                    Bx�`  �          @���=q@u?�\)A�\C�)��=q@]p�@ ��Ak�
C)                                    Bx�  
�          A ����p�@��\��
=�
=C�R��p�@�����Ϳ:�HC
=                                    Bxܬ  �          @�
=�ҏ\@{�����yG�C&f�ҏ\@��\�����=qC��                                    Bx�R  �          @�(���(�@y���{��Q�C����(�@�(����H�*ffC��                                    Bx��  �          @��H��{@5�?��A��C����{@�\@!G�A�ffC^�                                    Bx�  �          @�=q�Z=q?�p�@׮Bq��C\�Z=q=#�
@޸RB�C3\)                                    BxD  �          A z��e?��H@��Bl  Cu��e>k�@�\B|(�C0J=                                    Bx%�  �          @��R�G�?E�@�\B�C!^��G���ff@�G�B��CL�
                                    Bx4�  �          @�\)��{=���@��RB��)C#��{��{@��B��\C�Ǯ                                    BxC6  �          AG����
���A ��B�(�C��\���
�!G�@�p�B���C��q                                    BxQ�  �          @�(�=L�Ϳ�\@���B���C���=L�����@ٙ�B�8RC���                                    Bx`�  �          @޸R�K�>�(�@�\)Bi��C,@ �K��aG�@�p�Be�HCCff                                    Bxo(  �          @�׿@  ��
=@�Q�B�.Cx��@  �j�H@��Bp�C�9�                                    Bx}�  T          @�\)�}p��#33@�=qB�Cx�}p����@�\)BVG�C�7
                                    Bx�t  �          A{�7
=@���@�p�BH(�B��H�7
=@Tz�A(�Bx33C��                                    Bx�  �          A��W�@��\@���B2z�B���W�@~�R@���Bb=qCO\                                    Bx��  �          A�\��  @�G�@���B �B�����  @��H@陚BL33C#�                                    Bx�f  �          Az�����@��\@�z�B�HC�{����@j�H@ᙚBE�Ck�                                    Bx�  �          A\)����@��\@�  B$=qCn����@Y��@��HBK�C�\                                    Bxղ  �          A	G���Q�@��
@~{A�=qB�u���Q�@���@�ffB\)B��
                                    Bx�X  �          @��
��(�@�
=?
=q@�  C�q��(�@�p�?���Az�RC�R                                    Bx��  �          A Q���ff@�=q@��A�Q�C\)��ff@�=q@g
=A�  CO\                                    Bx�  �          A33��z�@�=q@$z�A��\Cu���z�@�
=@~{A�\C��                                    BxJ  �          A ����(�@�G�@'
=A��\Cs3��(�@�ff@xQ�A�\Ch�                                    Bx�  �          @�
=���H@~{@:=qA�
=C�\���H@E@u�A��HC&f                                    Bx-�  
�          @�33��Q�@�=q@_\)A��C����Q�@QG�@��B��CL�                                    Bx<<  �          @������@�
=?�G�A5p�B��q���@�(�@@  A�ffCL�                                    BxJ�  �          AQ����@�ff@A�(�C c����@�(�@w�A�{Ch�                                    BxY�  �          @��
���\@�33=u>�G�B������\@Å?��HAMp�B��                                    Bxh.  �          @�ff���@��@�{BA33B�aH���@E@�\)Bx�B�#�                                    Bxv�  �          @�z��
�H@�@��RBG�Bߊ=�
�H@�=q@�
=BLp�B�
=                                    Bx�z  �          @����,(�@�(�@b�\A��HB��,(�@�
=@��HB#��B�                                     Bx�   �          @�(��33@���@\)A��B�k��33@�(�@�{B=qBޣ�                                    Bx��  �          @��H�L��@�{@���B��B�Q��L��@|��@�  B7C\                                    Bx�l  �          @�p����@���?�(�A��C  ���@�(�@'
=A�C޸                                    Bx�  �          @�Q���{@���?��
A<(�C� ��{@�=q@-p�A�33CaH                                    Bxθ  �          @�ff���\@���?xQ�@�=qC�����\@��\@G�A��CG�                                    Bx�^  T          @���p�@��H���
�=qB�8R��p�@�
=>���@!�B�                                    Bx�  �          @��\��ff@��H������RB�#���ff@�{?�z�A'33B�p�                                    Bx��  �          AG����R@У�<�>W
=B�����R@ȣ�?��
ALQ�B���                                    Bx	P  �          A=q����@�(�������C :�����@�?���A2�\C
                                    Bx�  �          A�����H@�=q�����333C5����H@�{?���A=qC�                                    Bx&�  �          A����H@�33�#�
��=qCǮ���H@��?�  A#33C��                                    Bx5B  �          A	��ƸR@�{>���?��RCG��ƸR@�z�?�33AL(�C	Ǯ                                    BxC�  �          A33��{@�ff?Y��@���C
xR��{@�  @�A~�RCǮ                                    BxR�  �          A���ڏ\@˅?��HA�C	
=�ڏ\@�
=@H��A��C�                                    Bxa4  �          A ����G�@�z�@%An�\C�
��G�@�ff@�p�A��
Ck�                                    Bxo�  �          A!��@��@5A���C����@�\)@��
A�\)C��                                    Bx~�  �          A!p�����@�G�@-p�Ay��Ch�����@�=q@�ffA��HCG�                                    Bx�&  �          A Q���@�  @*�HAw�CE��@���@��\A��\C(�                                    Bx��  �          AG���=q@��@<(�A��
C�=��=q@���@�{A�\)C�f                                    Bx�r  �          A��=q@���@AG�A��C
����=q@�=q@��A�p�CY�                                    Bx�  �          A=q��{@�ff@#�
Aq�C^���{@�  @��
A�\)C�                                    BxǾ  �          A�H��  @�(�?��@���C  ��  @���@3�
A���C�                                    Bx�d  �          A����@�{@��AL(�C�q���@�(�@`��A���C                                    Bx�
  �          A=q��p�@���@mp�A��C����p�@���@�G�A���C�
                                    Bx�  �          A{��
=@��@���A�G�Cp���
=@��@���B�C޸                                    BxV  �          A=q��Q�@�G�@��\A�p�C����Q�@{�@��B�C�
                                    Bx�  �          AG���  @�  @`��A�{C�q��  @�G�@��A�p�C�f                                    Bx�  �          A33��@�  @i��A���C{��@p��@�p�A���C+�                                    Bx.H  �          A����@�ff@p�AV�RC�f���@��
@aG�A��C5�                                    Bx<�  �          A����@���@�RAp��C������@xQ�@o\)A�Q�Cٚ                                    BxK�  �          A=q��@�Q�@FffA�(�CB���@\(�@�Q�A�(�CQ�                                    BxZ:  �          A����z�@��?�
=A
�RC{��z�@�\)@5A���Cc�                                    Bxh�  �          A\)���
@���?�\)Ap�CG����
@�ff@0��A�  C�                                    Bxw�  �          A�R��{@�G�?��H@�p�Cz���{@��R@+�A���Cz�                                    Bx�,  �          A(����@���?��@���C�R���@�@9��A�ffC\                                    Bx��  �          Aff���H@�{?}p�@�{C=q���H@�p�@!G�At(�C��                                    Bx�x  �          A\)�@{����R��C\�@vff?W
=@��Cz�                                    Bx�  �          A�R���R@���>.{?��\C�����R@���?У�A
=C�                                    Bx��  �          A\)��{@�z�>#�
?uC\��{@��
?�33A ��Cu�                                    Bx�j  �          Aff��p�@�G�>L��?�(�C�\��p�@�Q�?�z�A#33C�                                    Bx�  �          A{��\)@��
���O\)C����\)@�ff?�ffA   C�                                     Bx�  �          AQ���{@�\)�(����=qC33��{@�
=?B�\@�
=CJ=                                    Bx�\  �          A�H��{@��
�:�H���C(���{@�z�?
=@n{C
=                                    Bx 
  T          A���陚@�33@8��A�\)CB��陚@��@��A�Q�C�q                                    Bx �  �          A  ��R@�{?�z�A5G�CxR��R@��@e�A���C��                                    Bx 'N  �          A
=��{@�ff?B�\@��C� ��{@�\)@A`Q�C޸                                    Bx 5�  �          A=q��{@��
@�RAP(�C����{@�@|��A�(�C!H                                    Bx D�  �          AQ����@�G�?�z�AG�C�R���@��
@@  A�G�CaH                                    Bx S@  �          AG����@��R?��@N{C����@���@�AC�
C��                                    Bx a�  T          A=q�ff@�ff>�
=@��C=q�ff@��\?��HA7
=C{                                    Bx p�  �          A��p�@�  ?�p�@��C�=�p�@�(�@5�A�33C�                                    Bx 2  �          A���@�  ?�  @�RC5���@��
@6ffA��HC\)                                    Bx ��  �          A"{��\@���?���@��C����\@��R@,(�Av�RCs3                                    Bx �~  �          A#
=��@���?��@�
=C0���@�{@0��A{�C�                                    Bx �$  T          A#33�z�@�(�?^�R@���C���z�@��H@'
=Am��C.                                    Bx ��  �          A$���{@�{?0��@w�C���{@�{@{A]�C�q                                    Bx �p  T          A%��	�@��?(��@l(�C���	�@�z�@AR{C5�                                    Bx �  �          A&�R��@��?!G�@]p�Cٚ��@�(�@��A>=qC��                                    Bx �  �          A&�H��@���>u?��Cff��@��\?�A�C��                                    Bx �b  �          A'
=�	�@�33>�Q�?�p�C�
�	�@��R@��A=p�C��                                    Bx!  �          A'���H@�  �8Q�uCh���H@��?�\)@�CQ�                                    Bx!�  �          A(���@��\�B�\���
C�f�@�(�?��H@�
=C��                                    Bx! T  �          A)��\)@�z�
=q�<��C�
�\)@��?���@��\C=q                                    Bx!.�  �          A)G��(�@����0���qG�Cp��(�@�G�?@  @�=qCz�                                    Bx!=�  T          A)p���@�(��(��S33C����@�=q?��\@�Q�C8R                                    Bx!LF  �          A)����@��
�����(�CO\���@�?0��@n{C�                                    Bx!Z�  �          A(���
=@�p���
=��
C���
=@�Q�?���@���Cs3                                    Bx!i�  
�          A(����@��\�CW
��@���?�z�@�ffC�                                    Bx!x8  �          A(  ���@�(���\�33C�H���@�
=������33C=q                                    Bx!��  �          A'��  @�\)��G����C���  @�>�z�?���C�q                                    Bx!��  �          A)G�� Q�@�Q쿞�R�׮C��� Q�@ٙ�?s33@�C�3                                    Bx!�*  �          A(��� ��@�{���\����C@ � ��@�  ?h��@��RC�                                    Bx!��  �          A(  ��@��Ϳ�
=��
CǮ��@�33>�(�@ffC�f                                    Bx!�v  �          A((��	@��H���:ffC�	@�
=�k����
CB�                                    Bx!�  T          A'��ff@��\��9�C8R�ff@�ff���5C�{                                    Bx!��  T          A'�
�  @���=q�UG�C� �  @���#�
�`��C#�                                    Bx!�h  �          A'���\@����9�����\Cu���\@��\��G����C!H                                    Bx!�  �          A%��@�G��[����C� ��@����H�.�\C�                                    Bx"
�  �          A&=q�
�H@��R�
=�<z�C��
�H@�����ÿ���C�                                    Bx"Z  �          A$���Q�@��R��\�7�C}q�Q�@����{���C�{                                    Bx"(   �          A!�����@�G��:�H��
=C�f���@��?(�@a�C��                                    Bx"6�  �          A"ff��@���=p����C���@�p���Q��ָRC��                                    Bx"EL  �          A Q���Q�@���o\)��G�C����Q�@�\)�����)p�CG�                                    Bx"S�  �          Aff��ff@���0����ffC���ff@�z�:�H��  CY�                                    Bx"b�  �          AG���{@�(���{����B�Ǯ��{@�\?�Q�A�C 
                                    Bx"q>  �          AG����@��?O\)@�
=C����@��R@/\)A��C��                                    Bx"�  �          A���\)@S�
�p����p�C�f�\)@[�>B�\?�Q�C��                                    Bx"��  �          Aff��\@���R�fffC^���\@�33?�{@�CǮ                                    Bx"�0  �          A\)��\@��
����-p�C�H��\@���?}p�@��RCW
                                    Bx"��  �          A�R��R@�녿k���33C�q��R@��\?L��@�(�C�H                                    Bx"�|  �          A�H�G�@����xQ����
C� �G�@��\?Tz�@���CaH                                    Bx"�"  �          A!���@��
��(���
=C���@���>�z�?�33C=q                                    Bx"��  �          A�\��
@�zῴz��\)CW
��
@��=�?333C33                                    Bx"�n  �          A���	@����=q�,��CǮ�	@�����
���Cٚ                                    Bx"�  �          A��33@Z=q���
�0Q�C��33@s�
�
=q�U�C�q                                    Bx#�  �          A=q���@2�\�"�\��G�C!h����@^�R���H��
CE                                    Bx#`  �          A���   @,(��I�����C!p��   @e��z��T��C޸                                    Bx#!  �          A(���@�{����ə�C�H��@�G�?   @C33Cc�                                    Bx#/�  �          A�R�p�@x�þu��
=CB��p�@o\)?���@�z�C�                                    Bx#>R  �          A�����@�\)?^�R@��HC����@��@=qAc�C                                    Bx#L�  �          A���Q�@���@#33AvffCL���Q�@�\)@�
=A�z�C��                                    Bx#[�  �          Ap��@�33=#�
>W
=C���@�G�?�  A$��C33                                    Bx#jD  �          A
=��@�p��8Q���{C�{��@�z�?fff@�{C��                                    Bx#x�  �          A�R��\@����(����C�3��\@�
=?z�@W�C&f                                    Bx#��  �          A�
� z�@�녿������C�� z�@�ff?��@Mp�C�
                                    Bx#�6  �          A����G�@�ff@�AV�RC	z���G�@���@��HAѮC
=                                    Bx#��  T          A�H��=q@�
=?�@I��C
�f��=q@�p�@'
=A�ffC@                                     Bx#��  �          AQ����@�z�>�\)?��HC	8R���@��@��Ap��Cp�                                    Bx#�(  �          A�
�ᙚ@�p�>���?�
=C�)�ᙚ@���@%Aw33C
�f                                    Bx#��  �          A\)��p�@�
=?\(�@���C���p�@���@<(�A���C.                                    Bx#�t  �          A  ��@�Q�>��@0  C����@�
=@%Av�HCk�                                    Bx#�  �          Az���\@�(���=q�ǮC
=��\@�33?�{A/�
C\)                                    Bx#��  �          A����@��R>�\)?�33C�����@���@��AL��C��                                    Bx$f  �          A�
����@ȣ׿k���Q�C
�R����@�ff?��@��C
=                                    Bx$  �          A�R�أ�@Ӆ�����:=qC��أ�@��
>�@.�RC��                                    Bx$(�  T          A����  @�(��z�H��G�C����  @��H?���@�C��                                    Bx$7X  �          A����ff@��ÿ�\)���HC	���ff@У�?�(�@�z�C	ٚ                                    Bx$E�  �          A����z�@�G�>B�\?���C����z�@��H@�AX��C�                                    Bx$T�  T          A  ��@�{@+�A��
C�q��@{�@�=qA�(�C�R                                    Bx$cJ  �          A33���@�{@=p�A��RC�����@�=q@�A癚C:�                                    Bx$q�  �          Az����@��@ ��A@Q�Cc����@�{@s33A��
C�                                    Bx$��  �          Az���33@�Q�@�
AD(�C����33@���@�G�Aď\CJ=                                    Bx$�<  �          A����@�\)?�  A\)C
�����@��@eA��Cu�                                    Bx$��  �          A=q���
@���?��HA��C0����
@�p�@N�RA�p�C��                                    Bx$��  �          Aff�   @�\)?�\)@�G�C!H�   @�Q�@333A��C��                                    Bx$�.  �          A���ff@��
�\)�\(�C#��ff@��?˅A{C�                                    Bx$��  T          AQ���  @���?��@hQ�C(���  @�{@,��A��RC�                                    Bx$�z  �          A33�Q�@_\)����8��C� �Q�@{����L(�C@                                     Bx$�   �          AG���{@�G���Q��Q�CQ���{@�=q?�ffAG�Ck�                                    Bx$��  T          A�
���H@��
>B�\?���C���H@��H@-p�A�C��                                    Bx%l  T          AQ��߮@��ÿ�\�@��C�)�߮@У�?�Q�A7�C	                                      Bx%  �          A����33@�p���33�z�C����33@��@z�A[�
C                                      Bx%!�  �          Az��أ�@�׿@  ���
C�R�أ�@�=q?���A+\)C�=                                    Bx%0^  �          AG��ٙ�@�׿������C��ٙ�@�p�?�ffAG�C�                                    Bx%?  �          A{��@�  ��p����C����@׮?��
@��C�=                                    Bx%M�  �          A����G�@��þ.{���
B�� ��G�@�=q@(��A�ffB�Ǯ                                    Bx%\P  �          A�����Az����(�B������A��@,(�A�33B                                    Bx%j�  �          Az���Q�A
�\���
��=qB�\)��Q�A��?�
=A733B�                                      Bx%y�  �          A����\)A
�\�
=q�L(�B�q��\)AQ�@(Q�Ay��B��                                    Bx%�B  �          A����@�p�=�G�?(�C����@���@,��A}G�C�3                                    Bx%��  �          A �����H@�����Ϳ��C�f���H@�{@&ffAs\)C�                                    Bx%��  �          A"{� (�@Å��ff��ffC��� (�@\?�  @��C�\                                    Bx%�4  �          A"�\�@��׿k���{Cff�@�ff?��@��HC�                                    Bx%��  T          A"�\� Q�@���� ���6�\C��� Q�@�\)>�\)?���C(�                                    Bx%р  �          A"ff��@�녿�p��G�C@ ��@��>�
=@
=C{                                    Bx%�&  �          A!�����R@��׿�
=��\C�����R@��
?^�R@��RCn                                    Bx%��  �          A!G����@�33�����C���@�G�?0��@z=qC.                                    Bx%�r  �          A �����@�
=�	���F=qC�)���@�zὸQ���HC��                                    Bx&  �          A ����p�@��
���=qCB���p�@��?��H@��C
                                    Bx&�  �          A33�У�@ۅ����hz�C���У�@���>�  ?�(�C�{                                    Bx&)d  �          A=q����@�  ��33�=qC������@��?Tz�@��Cff                                    Bx&8
  �          A����H@��H�L�;�\)C����H@��
@\)Ai�C
\                                    Bx&F�  �          A (����RAp�>�G�@"�\B�p����R@�33@Z=qA���B��                                    Bx&UV  �          A z�����@�R@QG�A�
=B�z�����@��R@��BG�CO\                                    Bx&c�  �          Aff����@�  @1G�A���C ������@�ff@�Q�B�C}q                                    Bx&r�  �          A33���H@�  @��RB��Ch����H@S�
@�RB<=qCxR                                    Bx&�H  �          A�H��\)@�z�@:�HA�{C	�)��\)@��@�  A�  C�
                                    Bx&��  �          A�����@�{?��A�C	  ���@��@{�A���C0�                                    Bx&��  �          A�����@�p�@n�RA���B��=���@�  @���B=qC��                                    Bx&�:  �          AQ�����@�=q@
=A`Q�C�q����@�{@��A�\)C0�                                    Bx&��  �          A���G�@�=q@=qAc33C\)��G�@��@�p�A��
C@                                     Bx&ʆ  �          A��߮@���@%�As�C	�\�߮@��R@�G�A�C��                                    Bx&�,  �          A���p�@�R@S33A�(�B�����p�@�{@���Bp�C��                                    Bx&��  �          A��8��A�@�\)A܏\B�\)�8��@�G�@�{B@�B�
=                                    Bx&�x  �          A�
�p��@�R@�
=A�  B�u��p��@���@�p�BG
=B��                                    Bx'  �          A(����R@�p�@�(�A�z�B������R@���@�\)B?�C�q                                    Bx'�  �          AG���=q@׮@��BG�B�#���=q@�  @��BD33C
��                                    Bx'"j  �          A�R��\)@�33@��A�\)B��3��\)@��@�33B7Q�CT{                                    Bx'1  T          A�H��=q@ָR@�33A�B����=q@���@��B>�C޸                                    Bx'?�  
�          A!����G�@޸R@�  B(�B�  ��G�@�33@�\)BHp�C	                                    Bx'N\  �          A!�����@�  @���A�{C����@��@���B"z�CǮ                                    Bx']  �          A����@���@~{A�\)C@ ���@�  @�Q�B=qCY�                                    Bx'k�  �          A!����@�{?�
=A0(�C�
���@��R@�  Aƣ�Cp�                                    Bx'zN  �          A"{����@�?���@�z�C�R����@�{@g
=A�33C��                                    Bx'��  �          A#���@�=q@G�A8(�C����@��@�G�A�(�C�f                                    Bx'��  �          A!p���@��H@=qA]�C8R��@�33@���A���C	�3                                    Bx'�@  �          A!p���33@��?�33A.�\C��33@�  @��A�33Cu�                                    Bx'��  �          A z���G�@ۅ?�  A!C� ��G�@��@�Q�A�{C:�                                    Bx'Ì  �          A   ��G�@θR?Y��@���Cp���G�@�(�@Q�A��CG�                                    Bx'�2  �          A Q����H@���?�G�@�Q�C�����H@��@h��A�z�C�                                    Bx'��  �          A���@��>�z�?�z�C���@�G�@!�Ak\)C�q                                    Bx'�~  T          A33��@�
=>\@
�HC�{��@��
@#�
Ao33C��                                    Bx'�$  �          A   ��
=@��H?�@C�
C����
=@��@6ffA�z�C��                                    Bx(�  �          A (��(�@��R<�>L��C���(�@���@�
A>=qC�f                                    Bx(p  �          A�
�{@��?}p�@��RC(��{@��\@G
=A��
CJ=                                    Bx(*  �          A z���@�33?��@�  C�3��@��@L(�A��\C�                                    Bx(8�  �          A Q���{@����"�\�n�\CJ=��{@��
��=q��=qC��                                    Bx(Gb  �          Aff���
@�ff������CW
���
@�z��\)�/�
CB�                                    Bx(V  �          A\)�  @�����
=�4��C�=�  @���>L��?��C�=                                    Bx(d�  �          A z��(�@��Ϳ���HC�=�(�@���?O\)@�C33                                    Bx(sT  �          A
=���@�G�=�G�?!G�CxR���@�Q�@{Af�RC�R                                    Bx(��  �          A\)����@��ͽ��Ϳz�C����@�{@z�AXz�C�H                                    Bx(��  �          A
=��p�@�ff�@  ���C���p�@�Q�?�33AG�C�R                                    Bx(�F  �          A�H���@��\��{��z�Ch����@�p�?s33@�
=C                                      Bx(��  T          A�H���@���   �;�Ck����@���=���?\)Cu�                                    Bx(��  �          A���
@|���P  ���\C^���
@�����
=C.                                    Bx(�8  �          A�� ��@����*�H���C��� ��@�
=�(���|��C                                    Bx(��  �          A�R� ��@���.�R��p�C� � ��@��׿333��p�C�{                                    Bx(�  �          A�H��@��H�ff�ICxR��@�녾W
=���RC�3                                    Bx(�*  �          A�\��@��
�33�Dz�C����@����L�;���Ch�                                    Bx)�  �          A=q�ff@y���(Q��}��C��ff@�(��\(���z�C&f                                    Bx)v  �          A����@'
=�5����C#� ���@aG������=qC.                                    Bx)#  �          A=q���@0���%�y��C"�����@c�
���\��C
                                    Bx)1�  �          A{�=q@�{���Qp�Cu��=q@��R���ÿ�Q�C�                                    Bx)@h  �          Ap��\)@����G��B�HCk��\)@��׾u��C�H                                    Bx)O  �          A�����@w
=�(������C���@�33�^�R����C                                      Bx)]�  �          AG��ff@�����=q�z�C���ff@�=q>�=q?�\)Cp�                                    Bx)lZ  �          AG��\)@z=q�ff�K\)C0��\)@�p���Q��	��CaH                                    Bx){   �          Az���@l(��)����
=C5���@�{�p����p�C�                                    Bx)��  �          A��� z�@:=q�c33����C �� z�@��H���PQ�C�                                    Bx)�L  �          A���
=@<(��XQ���{C 8R�
=@�녿�z��=G�C�H                                    Bx)��  �          A�\�(�@;��Vff���C u��(�@��ÿ���9�C�q                                    Bx)��  �          A  �   @�H�z=q��ffC#+��   @r�\�&ff���HC�f                                    Bx)�>  �          Ap���ff@ ����33��33C%Y���ff@fff�J�H��\)C��                                    Bx)��  �          AG����@\)�y����  C$����@hQ��*=q���\C8R                                    Bx)�  �          A���{?�p�������=qC&���{@Mp��>�R���C��                                    Bx)�0  �          A(���33@�p���\�^�HC0���33@��H>���?���C
=q                                    Bx)��  �          A�
��  @θR���h  C(���  @��
>�@0��Ck�                                    Bx*|  
�          AG���@��H���
�+�C���@�ff?�  @�Q�C��                                    Bx*"  �          A\)����@��Ϳ�\)�733C�3����@��?���@�p�C�                                    Bx**�  �          A=q�˅@����
��p�C�)�˅@��
?�ffA�Cٚ                                    Bx*9n  �          A  �߮@�p���G��+
=C
�\�߮@�33?h��@��C	�q                                    Bx*H  �          Ap���@���#�
�w�
C+���@�\)���G�Cc�                                    Bx*V�  �          A�
���H@����z��b�\C�H���H@�녾#�
�z�HC�f                                    Bx*e`  T          A�����R@���p��U��C�{���R@��H���8Q�C!H                                    Bx*t  �          A����(�@�
=���E�C� ��(�@��
>8Q�?�=qC��                                    Bx*��  �          A  ���H@�  �z��P��C����H@�=L��>�{C�f                                    Bx*�R  �          A������@�z�aG���ffC:�����@�Q�?�G�A�\C��                                    Bx*��  �          A����
@ȣ׾L�Ϳ��RC	�H���
@���@��An=qC�
                                    Bx*��  �          A{��33@��H�+���33C\��33@���@ffAO�Cn                                    Bx*�D  T          A����@�(���  ���
C���@�33@-p�A�C�3                                    Bx*��  T          A33��\)@��
��=q��B�
=��\)@���@@  A�ffB�\                                    Bx*ڐ  �          A�� (�@�  ��z��  C�3� (�@�?�@_\)C��                                    Bx*�6  �          Ap���R@��R��ff�3�
C޸��R@�
=?�R@uC��                                    Bx*��  �          AG���(�@���33�<��Cs3��(�@��R?(�@q�C)                                    Bx+�  �          A���p�@��
����p�C�H�p�@���?�R@s�
C��                                    Bx+(  �          A  �  @XQ���
�O�
C� �  @{�����>{C��                                    Bx+#�  �          A
=��p�@��H>�=q?�
=C���p�@��@	��AX��C�
                                    Bx+2t  �          A(����@�  �k���p�C�q���@~{?�ffA33CaH                                    Bx+A  �          A�
���@j�H�z���  C�q���@e�?}p�@�G�C�=                                    Bx+O�  �          Ap���33@�Q�+�����C���33@���?�z�@��
CaH                                    Bx+^f  �          AG���G�@���?��A!��C���G�@�(�@l��A�{CJ=                                    Bx+m  �          Aff�ə�@���@uA�
=C	c��ə�@j�H@���B�C�                                    Bx+{�  
�          A����
@��\@AG�A���C����
@s�
@�=qBp�C�                                    Bx+�X  �          A����
@��@aG�A�\)C�=���
@\��@��RBp�C\)                                    Bx+��  �          A����@�G�@\)A�z�C����@h��@�z�A�  Cu�                                    Bx+��  �          A
ff���@�\)@�\)B�CxR���@)��@���B<�\Cc�                                    Bx+�J  �          Az���  @���@�=qBQ�C �
��  @�@ָRB\G�C��                                    Bx+��  �          AG���  @��\@�B
=C)��  @�@��BT�\C�)                                    Bx+Ӗ  �          Az��e@��
@��
B>ffB��e?��A ��B��Cc�                                    Bx+�<  �          A  �	��@S33@�\)B  B�G��	����Q�A
{B���C=n                                    Bx+��  �          A�R�)��@�G�@ᙚBLQ�B�ff�)��?�{AQ�B�C��                                    Bx+��  �          A���J�H@���@�  BZ�HC (��J�H?(�AQ�B�aHC)�                                    Bx,.  �          A�H���@��@�=qB-
=C&f���?���@�Bi�Ck�                                    Bx,�  �          A����33@���@��HA�=qC����33@N{@��HB>Q�C�                                    Bx,+z  �          A=q��ff@���@��B��C�3��ff@'�@�G�BU�C�                                    Bx,:   �          A=q����@��\@�{B�RCE����@*=q@�Q�BU��CxR                                    Bx,H�  �          A\)����@��@���B\)C�����@\)@��\B`�C�H                                    Bx,Wl  �          A����
@��@��BQ�Cz����
@!G�@��B`��C^�                                    Bx,f  �          A����=q@�33@���B
=B��=��=q@g�@�BL��C�)                                    Bx,t�  �          A���@��@��B1ffCc���?��
@�G�Bo  C�f                                    Bx,�^  �          A(����\@���@�Q�B"��C�R���\@�R@�Ba�
C\                                    Bx,�  T          A33���@��@�=qB%�
C  ���?�=q@��B^
=Cff                                    Bx,��  �          A�H���H@�@�Q�B$(�C����H?�z�@�G�B]p�C}q                                    Bx,�P  �          A�R���\@�(�@��BffC�����\@�R@��
BVz�C�                                    Bx,��  �          AQ�����@��\@�\)B\)C������@J=q@߮BF�C��                                    Bx,̜  �          A\)�Vff@�33@�p�B��B��f�Vff@j=q@��RBd�CxR                                    Bx,�B  �          Aff���@��@�  B/�\B������@A�A	��B�W
B�u�                                    Bx,��  �          A�H�>{@��
@�  B\)B�W
�>{@\(�A  Bt�C޸                                    Bx,��  �          A���E@أ�@�
=B
=B�\�E@e�A��Bq=qC��                                    Bx-4  �          A
=�6ff@��@�ffB&�RB�G��6ff@QG�A
�HB}�RC�                                    Bx-�  �          A�H��p�@�@���B�B�(���p�@��A�Bu�Bـ                                     Bx-$�  �          Aff���
@�(�@�  B�HBʳ3���
@���A�Bn�B�.                                    Bx-3&  �          A�R�^�R@���@�B
z�B�Q��^�R@r�\@��B^p�C�\                                    Bx-A�  �          A�\��
=@�@���B��B�����
=@a�@߮BO(�C�                                    Bx-Pr  �          A33��z�@��H@�33A�33C ���z�@h��@�G�B<�C\)                                    Bx-_  �          A����\@��@�z�A�z�B��H���\@z�H@�ffB8{C��                                    Bx-m�  �          A��l��@�Q�@��A陚B����l��@��@�\)BJ��C �                                     Bx-|d  �          A�Mp�@�z�@��HA�Q�B��Mp�@�=q@���BSz�B�33                                    Bx-�
  �          Az��K�@�\)@��A�z�B�k��K�@�{@�(�BB33B��\                                    Bx-��  T          A�H�Tz�@�33@�G�Aʣ�B��)�Tz�@��@��B@Q�B�                                    Bx-�V  T          A���^{@��@^{A�B��^{@��@�Q�B2z�B�\                                    Bx-��  �          A��`  AG�@*=qA��\B��
�`  @���@�(�B�HB�L�                                    Bx-Ţ  �          A��mp�@�{@��\A�\)B�=�mp�@�=q@�
=BC�HB�aH                                    Bx-�H  �          A�� ��@��@�z�B#(�B�G�� ��@W
=A�\B}�B�k�                                    Bx-��  �          A�R��@Ϯ@�
=B.  B׏\��@Dz�A
=qB��B�L�                                    Bx-�  �          A33�+�@�@��\Bp�B�G��+�@o\)A  BsffB�B�                                    Bx. :  �          A���-p�@�(�@�33B!�B�u��-p�@VffAB{
=C �q                                    Bx.�  �          A����@�{@ϮB1(�B�=q���@1�Az�B�
=Cs3                                    Bx.�  �          A�\�,(�@��@��B<B�R�,(�@  A  B�B�C{                                    Bx.,,  �          A�R���@�=q@s�
A�\)B�\)���@�
=@ϮB533C�R                                    Bx.:�  �          Ap����
@�p�@z�AS\)B�\���
@�=q@�
=B(�C�
                                    Bx.Ix  �          A
=�xQ�@��@L(�A���B��)�xQ�@�ff@�p�B-
=C )                                    Bx.X  �          A�׿�p�@�{@��BR
=B��f��p�?���@��B�Q�CxR                                    Bx.f�  �          AQ�J=q@���@��
Bs�
B��ÿJ=q>�A33B�G�C�f                                    Bx.uj  �          Aff���@��@�z�Bm�B�8R���?:�HAG�B��
C�=                                    Bx.�  �          A��?\)@�z�@��HBWz�B�� ?\)?ǮA	�B�=qB�33                                    Bx.��  �          A33��{@��@���B3p�Bҳ3��{@)��@��B��{B�z�                                    Bx.�\  �          A������@�z�@:=qA�ffB�������@�Q�@�G�BQ�C
G�                                    Bx.�  �          A  ��  @z=q@�\Bp��B�\��  >�{A��B���C'ff                                    Bx.��  �          A\)��33@���@�Bm�HB�aH��33?�\A  B�{C 33                                    Bx.�N  �          A  �#�
@�=q@��
B!�RB��#�
@AG�@�ffB{
=CG�                                    Bx.��  �          A��$z�@�\)@��B�HB��$z�@|��@陚Bd�RB��                                    Bx.�  �          A�H�Q�@�ff@���B)�RB�W
�Q�@5@�
=B��B�.                                    Bx.�@  �          A��@�  >�  @�p�B]�H@C33@�  �2�\@��HBC��C��                                    Bx/�  �          Ap�@l��>L��@�33B�z�@G
=@l���O\)@���B]�C�`                                     Bx/�  
(          Az�?xQ�@^{@���B��{B�Ǯ?xQ�aG�AffB��C��                                    Bx/%2  �          A�\>��H@�z�@�\)Bh33B���>��H?h��A(�B��=Bu��                                    Bx/3�  �          A�?8Q�@���@���B\33B��?8Q�?��
A��B���Br�R                                    Bx/B~  �          A
{����@ָR@�
A��B�=q����@���@�Q�Bz�C�                                     Bx/Q$  �          A����G�@��?��
@�ffB�p���G�@�(�@��
A�\)C �                                    Bx/_�  �          A�����@�z�?�  @ǮB�����@��@�33A�\)B���                                    Bx/np  �          A�
��A�#�
��G�B�W
��@�@Y��A���B��                                    Bx/}  �          Aff��ff@�p��#�
�uB�����ff@���@VffA�=qB�G�                                    Bx/��  �          A
{��@陚�#�
����B� ��@�33@(Q�A���B��
                                    Bx/�b  �          @�
=�6ff@�z�@8Q�A�B���6ff@s�
@��B:��B���                                    Bx/�  �          @��-p�@أ׾��
��RBߞ��-p�@�
=@+�A���B�
=                                    Bx/��  T          @��
�@��@��\@���B/��B����@��?�ff@�  Bz�C�                                    Bx/�T  �          A ����
@��@�
=B+�HB����
@"�\@�=qB��HC�                                    Bx/��  �          @��R�У�@�ff@��B'ffB��f�У�@2�\@��B�{B��\                                    Bx/�  �          @�p����
@�(�@�z�B
(�B�G����
@s33@׮Br\)B�z�                                    Bx/�F  �          @���/\)@�=q@���B \)B�=�/\)@#�
@ۅBu=qC�f                                    Bx0 �  �          @��H�Tz�@���@���B	p�B���Tz�@AG�@�(�B[p�C	��                                    Bx0�  �          @�G��c33@�  @w�A�\B�aH�c33@]p�@�  BI�
C��                                    Bx08  �          @�
=�b�\@Ǯ@1G�A�
=B�
=�b�\@�z�@��B+��C ޸                                    Bx0,�  �          @���+�@���@���B>p�B�B��+�?��
@���B�{C^�                                    Bx0;�  �          @�33�C�
@Q�@��Bh�\C33�C�
�:�H@�=qB~(�CAh�                                    Bx0J*  �          @�=q�>{?�(�@���Bi��Cs3�>{�J=q@\B|�RCBٚ                                    Bx0X�  �          @ٙ��Dz�?�  @�{Bh��C@ �Dz�n{@�(�Bu��CD�f                                    Bx0gv  T          @����QG�?W
=@�
=Bv  C%���QG���p�@��RBe(�CS:�                                    Bx0v  
�          @���(��@Mp�@���B^  CxR�(��>W
=@�33B���C/\)                                    Bx0��  �          A
=�5�@��H@�ffB
=B��H�5�@`��@�B_�C �                                    Bx0�h  �          A(��(Q�@�  @(��A��HBڣ��(Q�@��@�p�B*��B�=                                    Bx0�  �          A33��p�A�@�RAm�B���p�@�z�@�(�B!p�B�G�                                    Bx0��  �          AQ쿰��A�
?ٙ�A)�B�k�����@��@��B��B�Q�                                    Bx0�Z  �          A=q�h��A33?��@�G�B����h��@�(�@�33B=qB�{                                    Bx0�   �          A���z�A�
?�p�A=qB�p���z�@�33@�  B��B�G�                                    Bx0ܦ  �          A��L(�@�
=@n{A�=qB��L(�@��@�ffBG
=B�                                      Bx0�L  �          A��8Q�@�ff@,��A�p�Bߣ��8Q�@�Q�@�33B,
=B�ff                                    Bx0��  �          Aff�\)@�G�@!�A�\)Bٽq�\)@�p�@��B*z�B�\)                                    Bx1�  �          @��R�2�\@���?�
=AE�B޳3�2�\@�p�@�p�B�HB�z�                                    Bx1>  �          @�G�����@�G�����B��ÿ���@��H@\)A�(�B̅                                    Bx1%�  T          @��� ��@�Q�\�Tz�Bה{� ��@���@�A�
=B�Q�                                    Bx14�  T          @�\)�Z�H@��?˅AP(�B�z��Z�H@�p�@��\B�RB��                                     Bx1C0  �          @����h��@�33@l��A�
=B��q�h��@.{@�
=BI�CG�                                    Bx1Q�  �          @�33�\)@�G��z���33B�p��\)@�\)?�
=A��B���                                    Bx1`|  �          @�R��G�@�=q@.�RA�z�C33��G�@U@�{B#(�C\                                    Bx1o"  �          @���33@J=q@�{B.CaH��33?#�
@��\BZ=qC+
                                    Bx1}�  �          A (���Q�@L��@��B0z�C���Q�>���@���BU{C/#�                                    Bx1�n  �          @�G���p�@L��@���B-z�C\��p�?
=@�G�BVQ�C,n                                    Bx1�  �          @�ff��p�@J=q@�Bz�C�f��p�?^�R@��
BA��C)h�                                    Bx1��  �          @�z���ff@2�\@��B�HC�)��ff?�R@�G�B5{C,ٚ                                    Bx1�`  �          @������
@hQ�@W
=A�C�����
?�G�@��
B"  C!                                      Bx1�  �          @������R@���@�\)B  C
�����R?޸R@��BE��C�                                    Bx1լ  �          @����=q@���@U�A�(�C����=q@
=q@���B��C�{                                    Bx1�R  �          @����G�@�=q?�\)Aj�RC�{��G�@n{@{�B \)C�)                                    Bx1��  �          @�G����@�
=?�G�AY��C (����@��H@��Bp�C�R                                    Bx2�  �          @�\����@�?�@�\)B�\)����@��\@G
=A��HC h�                                    Bx2D  �          @��H�Q�@ٙ��aG���33B��Q�@�{@4z�A��B��)                                    Bx2�  �          @�{����@mp�@w
=A���C������?�{@�33B/��C"J=                                    Bx2-�  �          @�33��z�@�  @}p�A��C�H��z�@G�@���B<{C��                                    Bx2<6  �          A (���z�@�{@g
=A�Q�CxR��z�@QG�@���B3p�C��                                    Bx2J�  �          A{����@��H@|��A�ffB�� ����@^�R@�(�BCCW
                                    Bx2Y�  �          A  �I��@Ϯ@|(�A�B�q�I��@��@�p�BMG�B��                                    Bx2h(  �          @���Q�@�\)@Dz�A���Bٔ{�Q�@�z�@�33B<\)B���                                    Bx2v�  �          @����s33@�\)@u�A�33B�z��s33@Mp�@��
BF�Cٚ                                    Bx2�t  �          @�Q��S33@��@HQ�A���B�Q��S33@y��@�B;
=C0�                                    Bx2�  �          @�{�_\)@�(�@c33A�(�B�u��_\)@AG�@��RBG33C#�                                    Bx2��  �          @�ff����@1�@AG�A��C+�����?�33@|��B�C'�3                                    Bx2�f  T          @�{���@(�@�A��RCaH���?��\@<��A�33C'�\                                    Bx2�  �          @����@,��@(��A�Q�C�\��?�  @dz�B�C'
                                    Bx2β  �          @���  @#33@1G�A�ffC)��  ?�ff@g
=B\)C)#�                                    Bx2�X  �          @�z���@��@<��A�G�C&f��?W
=@mp�B{C+5�                                    Bx2��  �          @��
��=q?���@Z�HA���C����=q>k�@z�HB�\C1�\                                    Bx2��  �          @�p���?�z�@N{A�=qC(xR���Ǯ@Y��A��\C7��                                    Bx3	J  T          @У���z�?��
@   A��C*{��z�B�\@,��A��C5�)                                    Bx3�  �          @�ff����?(�@   A��
C.E������33@z�A�z�C7G�                                    Bx3&�  �          @��H��(�?��
@L��A�=qC${��(��L��@c33B�C4�                                    Bx35<  �          @�p��c33@Q�@�B)�HC	=q�c33?xQ�@�B_\)C$�                                    Bx3C�  �          @���?8Q�@�@c33A�ffB��?8Q�@`��@�ffBgz�B�                                    Bx3R�  T          @�(���\@�z�@�B%�HB��)��\?���@��HBx�C\                                    Bx3a.  �          @��n�R@g�@��RB.��C�H�n�R?��
@���Be�C$��                                    Bx3o�  �          @�=q����@'�@��B��C����?�@�33B,�\C.@                                     Bx3~z  �          @�33��Q�@Q�@�(�BG�C�\��Q�>B�\@��B3ffC1�H                                    Bx3�   �          @�{��Q�@��@�B,p�C����Q�#�
@�z�BB�
C6�                                    Bx3��  �          @��H�{�?��@�\)BU
=C#Ǯ�{���z�@��BQ�CG�=                                    Bx3�l  �          @�=q�z�H?��@�BJ�CB��z�H�Tz�@��
BU{C?��                                    Bx3�  �          @����dz�@�=q@\)A��B��
�dz�@i��@��HB'C\)                                    Bx3Ǹ  �          @����@o\)@z�HBp�C�H���?�{@�B;C ��                                    Bx3�^  �          @�����p�@�@��
BM��C\��p��5@�ffB^C=�                                    Bx3�  �          @������
@aG�@���B\)C�R���
?���@�z�BGp�C&J=                                    Bx3�  �          @�p���ff@`  @�G�B  C����ff?�p�@�z�B:ffC&
=                                    Bx4P  �          @�=q����@S33@��RB\)CW
����?xQ�@��RBA=qC(��                                    Bx4�  �          @��
����@Q�@~{B�RC.����?�
=@���B3ffC&}q                                    Bx4�  �          @陚����@\(�@�Q�Bp�CQ�����?�Q�@��HB=33C&{                                    Bx4.B  �          @�=q���@5@���B=C����>��@�33BaG�C1�3                                    Bx4<�  �          @�z����@H��@��
B"
=C�����?B�\@�G�BLG�C*B�                                    Bx4K�  �          @�ff��@\(�@`  A�C���?��
@��B(�C"Ǯ                                    Bx4Z4  �          @޸R���
@C�
@�{B{C�R���
?E�@��HBI�C*                                      Bx4h�  �          @����(�@\(�@���B33C����(�?��
@��BA��C#�                                     Bx4w�  �          @߮�O\)@%@��RBS�CO\�O\)����@���Bw=qC5��                                    Bx4�&  �          @�
=�AG�@   @���BdG�Ch��AG���ff@У�B��HC<xR                                    Bx4��  �          @�=q�:�H@Tz�@��BRCaH�:�H>���@�B�L�C,G�                                    Bx4�r  �          @�p��.{@P  @��RBU��C���.{>�p�@��B���C,8R                                    Bx4�  �          @��I��@~�R@��
B5{C O\�I��?��@˅BwffC��                                    Bx4��  
�          @�z��C33@vff@�G�B6�C ^��C33?��H@�
=Bx��CG�                                    Bx4�d  �          @�  �:=q@r�\@�z�BA�HB���:=q?�  @�Q�B�ffC!\                                    Bx4�
  �          @��
�(�@l��@��BG
=B��)�(�?}p�@��HB��
C�                                    Bx4�  �          @��
�2�\@\)@���B2ffB����2�\?�Q�@�=qBz�\C��                                    Bx4�V  �          @��H�*�H@~�R@��
B5B��R�*�H?�z�@�(�B�C5�                                    Bx5	�  �          @��H��R@���@�Q�B=�B��H��R?��@ȣ�B��\C�                                    Bx5�  �          @�  ��z�@~{@�z�BE33B����z�?��@���B�\)CxR                                    Bx5'H  �          @�녿�z�@}p�@���BH=qB�8R��z�?���@�  B���C�{                                    Bx55�  �          @��
�ff@n�R@���BF{B�R�ff?�\)@���B��3C�f                                    Bx5D�  �          @�33���
@l��@�z�BM��B�W
���
?��@�  B��=C�q                                    Bx5S:  �          @�p���@s33@��\BU{B�(���?�ff@θRB��qC
&f                                    Bx5a�  �          @��
�W
=@Z�H@�p�Bk�\B��W
=?��@�33B���B�\)                                    Bx5p�  �          @�=q��@AG�@\B~�B��=��<��
@�G�B��C*��                                    Bx5,  T          @أ׾���@W
=@�=qBo�B�{����>�(�@ָRB�\B��R                                    Bx5��  �          @љ���33@_\)@�33B^=qB؅��33?B�\@��HB��C�                                    Bx5�x  �          @Ϯ����@�z�@�33B(�B�녿���@ ��@��HB��RC�                                    Bx5�  �          @���)��@�(�@G�A���B�ff�)��@#33@��BP�C(�                                    Bx5��  �          @��H�n�R@���@�RA�{C��n�R@'�@��B&�HC��                                    Bx5�j  �          @�\)�b�\@��@#33A��HC��b�\@!G�@��HB-�C��                                    Bx5�  �          @�ff����@#�
@_\)B=qC� ����?@  @�Q�B7��C)��                                    Bx5�  �          @��R��=q?Tz�@`  B��C)���=q�O\)@`  B
=C=��                                    Bx5�\  �          @����p�����@O\)B�HCC(���p��   @�A�(�CQ�                                    Bx6  �          @��������8��@/\)A癚CW�R�����s�
?�
=A@z�C_k�                                    Bx6�  
Z          @��R�@  ����@�A���Cm�f�@  ��{��z��.�RCo�R                                    Bx6 N  
�          @�\)�`��>\@O\)B*{C-�)�`�׿��@Dz�B�CEB�                                    Bx6.�  "          @�G��{�@`��@[�B(�C
33�{�?��@�(�B=�C\)                                    Bx6=�  T          @������R@R�\@R�\A��HC���R?�  @���B233C ff                                    Bx6L@  �          @ȣ���\)@8��@S33A�{C#���\)?��@��B*{C%��                                    Bx6Z�  "          @ȣ���ff�L��@^{B�HC4�
��ff����@EA��CE�H                                    Bx6i�  
�          @�33��  �^�R@P��B�
C=����  �p�@"�\AǙ�CK�H                                    Bx6x2  T          @������?:�H@j�HB��C+�����u@g
=B33C?5�                                   Bx6��  
(          @���z�?��@g
=B33C(  ��z�(��@l��BQ�C;��                                   Bx6�~  
�          @Ǯ���\��\)@.{A��C4����\���\@=qA��RC@Ǯ                                    Bx6�$  
�          @�����(�?0��@)��A�=qC-��(����@+�A��HC9��                                    Bx6��  �          @�����?333@/\)A��
C-����׿z�@1G�A�{C9�3                                    Bx6�p  �          @�����\?��H@
=A���C(=q���\=L��@)��AîC3}q                                    Bx6�  �          @�z�����?��?��
A^ffC'�q����?�\?���A��RC/5�                                    Bx6޼  T          @�����?�z�?.{@�Q�C"0���?�z�?���AV=qC&��                                    Bx6�b  "          @Ǯ���\@/\)?ٙ�A}G�C� ���\?�(�@.�RA�(�C"!H                                    Bx6�  
�          @�{��p�?�p�?��AJ=qC�3��p�?��H@�A���C'n                                    Bx7
�  �          @�=q���H?��@(��A��
C&����H<��
@<��A���C3��                                    Bx7T  �          @�\)���?���@)��A��C'����<��
@=p�A�  C3                                    Bx7'�  �          @�G���G�@Q�@{A�z�C����G�?�ff@QG�A���C)5�                                    Bx76�  "          @�
=��
=@��@Q�A�=qC=q��
=?�{@L(�A��C)                                    Bx7EF  �          @ڏ\����?��@%�A��\C(#�����<�@8Q�A�
=C3�R                                    Bx7S�  
�          @�������@J=q?�=qA�ffCB�����@�\@A�A�CǮ                                    Bx7b�  "          @ڏ\���@+�@Q�A��C�����?�p�@E�A��HC%�3                                    Bx7q8  T          @�\)���H@#33@p�A�
=C^����H?�=q@G
=A�G�C'.                                    Bx7�  �          @�{����@G�@\)A�=qC�f����?s33@N�RA�Q�C*�3                                    Bx7��  T          @�33��\)@&ff@&ffA�33C����\)?���@^{A�C'�H                                    Bx7�*  "          @������\@!G�@  A��C�f���\?��@G�A��HC&�q                                    Bx7��  �          @˅����?��H@   A�
=C#������?(��@#33A�G�C-s3                                    Bx7�v  "          @ƸR��\)?��?�p�A�33C$
=��\)?8Q�@G�A���C,�
                                    Bx7�  �          @�z���p�?�z�@��A��C ����p�?G�@0��A��C+�\                                    Bx7��  "          @�ff��
=@p�@=p�A���C����
=?333@hQ�B�C,�                                    Bx7�h  
�          @�ff��33@ff@&ffA���C=q��33?z�H@W�B�HC))                                    Bx7�  �          @�
=��(�@
=@@��A�ffC:���(�?O\)@o\)B=qC*��                                    Bx8�  
�          @�  ��Q�@��@.�RA�p�C����Q�?u@`  B	�RC)#�                                    Bx8Z  T          @�G���\)@J=q@A�=qC�{��\)?�
=@P��A�33C�                                    