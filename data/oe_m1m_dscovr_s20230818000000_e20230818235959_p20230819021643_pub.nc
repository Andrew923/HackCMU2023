CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230818000000_e20230818235959_p20230819021643_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-08-19T02:16:43.064Z   date_calibration_data_updated         2023-08-08T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-08-18T00:00:00.000Z   time_coverage_end         2023-08-18T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�_@  "          @��@:�H��
=�Y����z�C�1�@:�H��Q������p�C��3                                    Bx�_�  �          @�z�@P�����H�C33���
C��\@P���ə���
=�t(�C��=                                    Bx�_.�  
�          @��H@#�
�����Q����HC���@#�
��Q��Q���\)C��                                     Bx�_=2  �          @��?��R��z��Q���C�}q?��R��(���
��\)C�)                                    Bx�_K�  "          @�33@���  �!G�����C���@���33��  ��C���                                    Bx�_Z~  �          @�z�@�ff��\)����;33C�T{@�ff�����(��N{C��                                    Bx�_i$  �          @��@|����33�(����z�C�t{@|���Ӆ?\)@���C�p�                                    Bx�_w�  �          @�@u���ff�\(���  C�޸@u���  >�z�@�RC�                                    Bx�_�p  "          @�
=@|�����\������
C�
=@|����p����
�ffC�K�                                    Bx�_�  
�          @��
@�z���\)�����W�
C��@�z��ƸR�0����=qC�]q                                    Bx�_��  
�          @�{@�
=��zΉ��   C�Ǯ@�
=�Ǯ=u>��C���                                    Bx�_�b  
(          @�33@�Q����������C�>�@�Q���{?L��@�p�C�^�                                    Bx�_�  
(          A�@ҏ\����?�AO�C�}q@ҏ\�w�@$z�A���C��                                    Bx�_Ϯ  
Z          A�H@N{�����#�
��Q�C�7
@N{��p�?�A�C�b�                                    Bx�_�T  �          AQ�@\(�� (�?G�@��\C�� @\(���\)@\)AmG�C�                                      Bx�_��  T          A	�@�{��33?}p�@�{C���@�{��G�@�A~ffC�^�                                    Bx�_��  �          Az�@�����
@p�ApQ�C�H@�����H@_\)A�\)C��)                                    Bx�`
F  
�          A�@Vff��=q@(�A��HC�P�@Vff�׮@p  A�  C�<)                                    Bx�`�  u          A	?���=q@z�A{�C���?����=q@r�\A�=qC�ٚ                                    Bx�`'�  �          A
�\?�ff��@   A���C�Ff?�ff��R@}p�A�=qC��                                    Bx�`68  	�          A	�?�  ��Q�@a�A��C���?�  �޸R@��B(�C�3                                    Bx�`D�  T          A
{?����33@U�A��C���?�����H@�\)B{C�%                                    Bx�`S�  �          A�@
=��Q�@{A�  C�H@
=�ָR@b�\AӅC��                                    Bx�`b*  T          @���@e���G��:�H��{C�q@e���33>�?�ffC��                                    Bx�`p�  �          A@�  ������z��6  C��)@�  �8����ff�%
=C�%                                    Bx�`v  "          A  @��׿���ff�a=qC��@��׿�{��\)�V�HC�`                                     Bx�`�  �          A�@���Y�������R�C�{@����
��  �F=qC��q                                    Bx�`��  T          @�
=@����+������{C�Ff@����\(��n{���C�%                                    Bx�`�h  
�          @��@��
�N{�fff���C�>�@��
�u�;���{C���                                    Bx�`�  �          @���@c33�333�;���\C�޸@c33�S�
�ff��ffC��=                                    Bx�`ȴ  "          @���33���\>�
=@���Cu�f�33��p�?��
Ak�Cu�                                    Bx�`�Z  "          @�  �(����33��H��C�Uÿ(����
=��z��fffC��{                                    Bx�`�   
�          @\>.{����B�\���C��3>.{��G��G���Q�C�޸                                    Bx�`��  �          @�33>�z��w�@XQ�B$33C�"�>�z��J=q@���BO�
C��)                                    Bx�aL  	�          A ��@W���@i��A�p�C�]q@W���(�@��B��C�T{                                    Bx�a�  	�          A=q@�����p�@�
A�=qC�3@�����33@b�\A�  C�C�                                    Bx�a �  C          @���@�{��=q?���AffC���@�{��  @�RA�(�C��R                                    Bx�a/>  
�          @�Q�@�����
��33�R�HC�xR@����33�:�H���\C�Ǯ                                    Bx�a=�  T          @��@�Q��o\)�b�\��z�C��@�Q���33�0�����C��                                    Bx�aL�  �          A��@���S�
�����C��f@�������\������C�Y�                                    Bx�a[0  �          A\)@߮�(����\���C�e@߮�J�H�a��ř�C�˅                                    Bx�ai�  
�          A�@���=q��Q���ffC���@���HQ��^�R���C�f                                    Bx�ax|  T          A	p�@�33�Vff�O\)���
C��
@�33�z=q�!���(�C�"�                                    Bx�a�"  
�          A	p�@����{��G��>{C��R@����z�?�  @׮C��\                                    Bx�a��  
�          A�\@����{�J=q���RC�f@����R?(�@�{C���                                    Bx�a�n  �          A��@��H����<����Q�C�#�@��H���\��z��^{C��q                                    Bx�a�  T          A
=@�(������|(����
C�8R@�(���ff�?\)���RC�AH                                    Bx�a��  
�          A(�@�����~�R��=qC�xR@���G��=p����
C���                                    Bx�a�`  "          A\)@�p���p��[���  C��=@�p�����=q�yG�C�B�                                    Bx�a�  �          AG�@أ������Q��t  C�j=@أ���(���G��   C���                                    Bx�a��  "          A�@ƸR��������?\)C��)@ƸR����(��|(�C�                                    Bx�a�R  
Z          Aff@�p��Å�+����C�33@�p����
?
=q@_\)C�+�                                    Bx�b
�  
Z          A�@�ff��33>#�
?���C���@�ff��{?�33A��C���                                    Bx�b�  	�          A33@�����z�h������C��@�����?�\@Y��C��R                                    Bx�b(D  �          A
�\@�p���ff�C�
��ffC�j=@�p���{��\)�IG�C�=q                                    Bx�b6�  "          A
=@�Q���p���ff�陚C�*=@�Q���  �l(���{C��                                    Bx�bE�  
8          A�@��
�e��������C��H@��
���R��  ��C��{                                    Bx�bT6  v          A
=@��
�1G������G�C�
=@��
�p��������\)C���                                    Bx�bb�  "          A��@θR>W
=�Ǯ�/�H?�\)@θR���
����-  C�y�                                    Bx�bq�  
�          A��@ڏ\?�\��{�#�R@���@ڏ\�.{��p��#(�C�*=                                    Bx�b�(  �          Ap�@�
=@����
�<33A��@�
=?k���{�H�A=q                                    Bx�b��  	�          Ap�@�ff��p��H�����C��@�ff���R�  �_�C�AH                                    Bx�b�t  d          Ap�@�=q���\�*�H��
=C��f@�=q���׿�  �-�C�N                                    Bx�b�            A  A�\�B�\�mp���Q�C�ǮA�\�n{�AG�����C���                                    Bx�b��  
�          A�RA�ÿ=p���33��
=C��fA�ÿ������
��z�C���                                    Bx�b�f  �          A��@�G�?J=q��  �G�@���@�G�������G����C���                                    Bx�b�  �          Az�@�R?�{��z���HAB=q@�R>�G���33�33@W
=                                    Bx�b�  �          A  @�z�@ff������RA~=q@�z�?s33���\��@��                                    Bx�b�X  
�          Aff@�
=?�(�����\)A1��@�
=>�  ������?�
=                                    Bx�c�  
(          A�
@�?����\)�(�Ai�@�?z���\)�&��@�33                                    Bx�c�  �          A�@���@x����ff�2��B��@���@����33�J33A�p�                                    Bx�c!J  "          A�@�p�@��H��\)�5{B&z�@�p�@E������P��A�=q                                    Bx�c/�  r          A  @���@h����\�J��B  @���@z���p��b��A��                                    Bx�c>�  �          A�@HQ�@������
�\ffBW�@HQ�@)���p��(�B �H                                    Bx�cM<  "          A�@E�@�����[{B\33@E�@0������~��B'\)                                    Bx�c[�  �          A\)@K�@n{��\)�c�BF  @K�@
=�G�ffBp�                                    Bx�cj�  �          Az�@�G�@>{����P�
A�(�@�G�?�z���\)�dffA�p�                                    Bx�cy.  �          A�
@�  @���ٙ��=�A�Q�@�  ?�  ����K=qA�                                    Bx�c��  �          A��@q�@P  ���_z�B"��@q�?���(��y33A�Q�                                    Bx�c�z  "          A�R?�{@\)��(��r\)B���?�{@����G�B](�                                    Bx�c�   
�          A33@.�R@G���(�  BG�@.�R>�����z�p�A=q                                    Bx�c��  "          A
=?�(�����p�k�C�/\?�(��=p���z�#�C��3                                    Bx�c�l  
�          A��?�G�@!���p�B�z�?�G�?B�\�	G�£=qB=q                                    Bx�c�  T          AQ�@C�
@J�H����n�\B833@C�
?��� (�AՅ                                    Bx�c߸  T          A
�R@�Q�?����z��c�Ac\)@�Q쾽p���
=�g�RC���                                    Bx�c�^  
�          A��@����L�������dG�C��R@�����=q��\�Z�
C�
=                                    Bx�c�  
Z          A#�
@�\)�z�H���\C�~�@�\)�8������oQ�C�ٚ                                    Bx�d�  
�          A$��@X�ÿ����G��fC�ff@X���[�����w�C�T{                                    Bx�dP  
�          A#\)?���p��� ��£�RC�� ?���A��=q  C��                                    Bx�d(�  "          A$��@J�H���Q�k�C�E@J�H��(�����i�\C���                                    Bx�d7�  �          A'\)@����	���=q�t�C�Y�@����������Yz�C�9�                                    Bx�dFB  
�          A'
=@�\)����
=�c  C�P�@�\)����� Q��I�HC�"�                                    Bx�dT�  T          A33@�
=�
=��[{C���@�
=������p��A��C�*=                                    Bx�dc�  �          A33@����G����P��C��@�������أ��2��C���                                    Bx�dr4  "          A�R@����J�H��(��L(�C��3@�����p������.{C���                                    Bx�d��  "          A
=@���o\)��\)�<
=C�%@����ff�ȣ��\)C���                                    Bx�d��  �          A$(�@�G������ff�<��C��
@�G���33�����p�C�Q�                                    Bx�d�&  �          A$��@\�c�
���H�8�RC���@\��������{C��                                    Bx�d��  �          A#33@�ff��������L
=C�@�ff�L����(��:ffC�Y�                                    Bx�d�r  "          A)G�@�=q���H� z��FQ�C��@�=q�ƸR��33�"�C���                                    Bx�d�  �          A)G�@�=q���
��H�J��C��R@�=q��G���G��'(�C�H                                    Bx�dؾ  "          A'\)@���������{�E�C�h�@��������=q�#  C��q                                    Bx�d�d  �          A'�@����\)���E33C�{@���������
�$p�C�Ff                                    Bx�d�
  �          A(Q�@�{�[��
�\�\C���@�{��\)��ff�;�
C���                                    Bx�e�  �          A(��@��R�J�H�	��ZG�C�\)@��R��ff���;z�C�9�                                    Bx�eV  �          A$z�@����K������9�RC�U�@�����{������C�W
                                    Bx�e!�  �          A$��@�33�E��  �+�HC�� @�33��G�����Q�C�:�                                    Bx�e0�  �          A'�@У��@����\�:G�C��)@У����H�׮� ��C�g�                                    Bx�e?H  �          A#33@�p��E��p��\�\C��q@�p����\��R�<�C���                                    Bx�eM�  T          A!�@S�
�O\)�p��yffC��@S�
�������QC�t{                                    Bx�e\�  �          AQ���H�����¤��CvǮ���H�I����
  C��3                                    Bx�ek:  T          A�Ϳ������(�C_����e�z�W
Cs�                                    Bx�ey�  T          Az��p�������CRG��p��Mp��p��HCk�                                    Bx�e��  �          A33�HQ쿈���Q�aHCFǮ�HQ��7����\�u��C^�                                    Bx�e�,  �          AQ�˅������
�{C>�׿˅��\�
=u�Ck8R                                    Bx�e��  �          A�
�^�R�����  ¢#�Cf޸�^�R�?\)����)C}��                                    Bx�e�x  �          A�@�=q��{@�z�B"\)C�  @�=q�-p�@�p�B<��C���                                    Bx�e�  �          Ah��A33��(�A (�B�
C��3A33��z�A�B$�HC��                                    Bx�e��  
�          AfffA�
��\)Ap�B�C�~�A�
��\)A��B&�C��=                                    Bx�e�j  �          Ac33A33��z�A	�B�HC��3A33��G�A=qB2ffC��                                    Bx�e�  |          Ag�A
=��(�A�B#Q�C�\A
=�7
=A&=qB7�C��f                                    Bx�e��  �          Ae�A�R��(�@�A��C�'�A�R���A\)Bp�C��                                    Bx�f\  �          Af�HA
=����@�  A���C�h�A
=����A��B  C�H                                    Bx�f  �          Ae�A
=����@�ffA�
=C�fA
=���A��Bp�C��q                                    Bx�f)�  T          A]G�Ap���(�@��HB (�C��fAp���G�A	p�B�C��)                                    Bx�f8N  �          A[\)Aff��=q@�B (�C�*=Aff��\)A
�\B�\C�                                      Bx�fF�  �          AY�A$  ��p�@��A�{C��
A$  �У�@���A�Q�C��                                    Bx�fU�  �          AZ{A ��� ��@��HA�
=C���A �����H@�=qA�(�C��                                    Bx�fd@  T          AXz�A��'�
@9��AEp�C��HA����@�\)A�G�C�4{                                    Bx�fr�  T          A]G�A
=�Q�@��A�\)C��
A
=���H@��A�33C���                                    Bx�f��  �          Ad(�AG���  @�
=A�=qC��
AG����\A�
B33C��                                    Bx�f�2  �          Av�HA'
=���A)G�B-\)C�]qA'
=��G�A3�B;�C��H                                    Bx�f��  �          Ax��A*{���\A'�
B)�RC��A*{���
A333B8z�C�=q                                    Bx�f�~  
�          Av=qA)���=qA&{B)�C���A)녿�ffA0��B7=qC��
                                    Bx�f�$  �          AqA*�\�P��A"ffB)Q�C��HA*�\�O\)A*{B3ffC���                                    Bx�f��  T          AnffA(�׿5A'�
B3(�C��A(��?�G�A%B0z�A�                                    Bx�f�p  �          Al��A-�.{A Q�B*�C�8RA-?�
=AffB(
=A��                                    Bx�f�  �          An�HA4Q��;�A
=B(�C��{A4Q�333A�B!C�8R                                    Bx�f��  �          Ag
=A+
=��33A
ffBffC��A+
=���A�HB#Q�C�.                                    Bx�gb  
Z          A`Q�A$(���
=A Q�Bp�C��fA$(��L(�AQ�B {C�`                                     Bx�g  �          A`��A!G���  @�
=Bp�C��qA!G���Q�A�RB�RC�&f                                    Bx�g"�  �          A_\)A(Q���=q@˅A�
=C�  A(Q���z�@�\)B�RC���                                    Bx�g1T  �          Aa��A*{��(�@�A�  C���A*{��
=@�(�B33C���                                    Bx�g?�  T          Aa�A*�R����@�z�A��
C�O\A*�R��{@�z�A�\)C��H                                    Bx�gN�  �          A]A)��Q�@���A�  C�Z�A)����@�{A�C���                                    Bx�g]F  �          A_\)A&=q��G�@�{A�(�C���A&=q�ʏ\@�z�A�Q�C�T{                                    Bx�gk�  �          AV=qA!G���p�@���A�  C��\A!G����@�{A�C�}q                                    Bx�gz�  T          AO\)A#���R@"�\A5p�C���A#���(�@�  A��
C���                                    Bx�g�8  �          AS
=A�R�
{��z�����C�|)A�R�
�\?s33@��
C�q�                                    Bx�g��  T          AYp�A	��	p�����z�C���A	��z��.{�?\)C��                                    Bx�g��  �          AV�\A�
��������G�C�'�A�
� ���9���I�C��                                    Bx�g�*  T          A]p�@�z��ff�����\C�,�@�z��"�R��=q��Q�C���                                    Bx�g��  T          A\z�@�\)������
��C��\@�\)�$����Q���Q�C�k�                                    Bx�g�v  T          AX��@�G��\)��\)� ffC�q�@�G��#33��z����RC�:�                                    Bx�g�  T          A_
=@������\���
C���@���'
=�����RC��
                                    Bx�g��  T          Aa��AQ���\���H��  C�nAQ��%���ff��{C�H�                                    Bx�g�h  �          A_\)A\)�p�������Q�C���A\)�)p��s33�}p�C��                                    Bx�h  T          A`Q�A=q�=q���
��G�C��A=q�)���hQ��pQ�C�,�                                    Bx�h�  T          A`��AQ��(����H���C��AQ��%���H���NffC���                                    Bx�h*Z  �          AX��A ���z���ff��C��A ����R��z����C���                                    Bx�h9   
�          AJ�R@�=q��{��H�6\)C��q@�=q��H��{��HC�e                                    Bx�hG�  "          AC�
@������+C���@��33������\C��)                                    Bx�hVL  �          AI�@У���
=��ff��C��@У��G�������C�y�                                    Bx�hd�  �          AN�\@��
��\�Å��ffC�
@��
��R�u���=qC�                                      Bx�hs�            AQ�A�\�33�aG��z=qC���A�\�
=q?���@�z�C���                                    Bx�h�>  �          AL��Aff��?��
@���C��=Aff��=q@b�\A��RC��\                                    Bx�h��  �          AI��A���{?��@��C�ǮA����@Q�Aqp�C���                                    Bx�h��  T          AI��A�H���?��\@��C�T{A�H����@Dz�AaC�@                                     Bx�h�0  T          AI�A33��R?E�@a�C��fA33��ff@7
=AS�
C���                                    Bx�h��  �          AC
=A   ����33��C���A   �(�>���?��C�k�                                    Bx�h�|  T          A<��@��
=���*ffC��)@��\)=�Q�>��C�L�                                    Bx�h�"  T          AF�\A	������
��G�C��A	���	�@{A%C���                                    Bx�h��  �          AK�
A�\�\)?��A33C�@ A�\��ff@~{A��HC���                                    Bx�h�n  �          AN{A���
?���@���C�EA��@a�A�(�C�W
                                    Bx�i  0          AN{A�Q�=�?�C�8RA�\)@�\A$  C���                                    Bx�i�  �          APQ�A�����@
�HA�C�A����
=@�G�A�{C���                                    Bx�i#`  �          AS
=A��
{@�\A   C�4{A���  @�A���C��
                                    Bx�i2  �          AT��Ap��	��@�\A�RC�k�Ap���\)@�p�A���C��                                    Bx�i@�  �          AX  A���Q�@�RAz�C���A���=q@�\)A��C�q                                    Bx�iOR  �          AX��A z��=q?���A��C�9�A z��G�@�A��
C���                                    Bx�i]�  T          AZ�\A����׽L�;B�\C�T{A����
@
=A z�C��                                    Bx�il�  �          A\��A Q���>L��?Y��C�Q�A Q��p�@)��A1G�C��f                                    Bx�i{D  �          A[\)A��׿�����HC�\)A�G�?�=q@���C�K�                                    Bx�i��  �          AV�\A
=�
ff��
=��{C�\)A
=�Q��z���C���                                    Bx�i��  �          AX��A{��������33C�.A{������
��=qC���                                    Bx�i�6  �          A[�A��{��(���=qC��A��ff����\)C�s3                                    Bx�i��  �          A\��A�����|(�����C�]qA��zῷ
=���RC�Ff                                    Bx�iĂ  �          A^�\A=q�z��p  �zffC���A=q�33��  ��z�C�Ǯ                                    Bx�i�(  T          A[�A"�R����N�R�ZffC���A"�R����\(��g
=C��                                    Bx�i��  �          AW�A��G�@��A���C�eA����@�B�C��\                                    Bx�i�t  �          AVffA  �\)@��A�Q�C���A  ��ff@�G�B	  C�.                                    Bx�i�  T          AV�RA{�	�@�{A�  C��A{��@�G�B��C�                                      Bx�j�  T          AV=qA�R�z�@�
=A�\)C���A�R��
=@�=qA��C�b�                                    Bx�jf  T          AU�A  ��
@E�AVffC�w
A  ���
@�\)A�p�C�u�                                    Bx�j+  
�          AS
=A  ���@;�AN�\C���A  ��ff@�G�A���C��f                                    Bx�j9�  
�          AW
=A����A�B�C�w
A��~{A=qB7Q�C�s3                                    Bx�jHX  �          AW\)@����G�AG�B#��C��@���|(�A$(�BFG�C��                                     Bx�jV�  �          AW
=@�=q���A�B�C�P�@�=q���A!p�BBC�U�                                    Bx�je�  T          AW33@���\)AG�B33C�` @����RA�B?�C�<)                                    Bx�jtJ  �          AW
=@�ff��p�A��BQ�C��@�ff��33A!BCQ�C�AH                                    Bx�j��  �          AN�\@�
=��G�A\)BC=qC�XR@�
=�K�A/�Bi\)C��R                                    Bx�j��  T          AJ�R@�(���{A*{BcG�C�Ǯ@�(��У�A8  B��)C�0�                                    Bx�j�<            AF�HA�����@��A���C��
A����@ȣ�A�\C�޸                                    Bx�j��  �          AK�
A��@��A�z�C��HA��\)@�  Bp�C�}q                                    Bx�j��  
�          AN�HAQ����
@��\A��C�H�AQ���Q�@�\)A��HC��                                    Bx�j�.  T          AN{A=q��@G
=A`(�C�A=q��G�@�(�A�(�C��
                                    Bx�j��  �          AC\)@�33� (�@��A���C�33@�33��(�@�=qB��C�n                                    Bx�j�z  �          AF�RA�R���\@�(�A���C�)A�R�Å@�\B
=C��
                                    Bx�j�   T          AM�A33�{@��RA��C���A33��@�\BG�C��3                                    Bx�k�  
�          AT��A33�@�
=A�Q�C��
A33���H@ҏ\A�\)C�t{                                    Bx�kl  �          ATQ�A33�(�@B�\AT  C�q�A33���@�(�A���C�\)                                    Bx�k$  �          AS�
A=q�{@&ffA5��C�,�A=q�Q�@��A��HC��                                     Bx�k2�  0          AS�A���{@C33AUG�C��=A���ff@��AȸRC���                                    Bx�kA^  Z          ARffA�R�
=?��H@�(�C��A�R�=q@�p�A�z�C�P�                                    Bx�kP  �          ATz�A
�\��H?�@�{C��A
�\�Q�@�G�A���C���                                    Bx�k^�  �          AP��A���?�33@�33C�!HA���=q@w�A��HC�4{                                    Bx�kmP  �          AJ{Ap��
=�:�H�U�C���Ap��Q�?���A{C��q                                    Bx�k{�  T          AN�\A����=�Q�>ǮC�{A���H@7
=AL��C��{                                    Bx�k��  "          AM��A{��>L��?aG�C�c�A{�Q�@;�AS33C��                                    Bx�k�B  "          AM��A  �p�>aG�?xQ�C���A  ��@?\)AW�C���                                    Bx�k��  T          AP��Aff�=q?�p�@���C�,�Aff�=q@~�RA�(�C�L�                                    Bx�k��  b          AO�A(��
=?���@�33C�޸A(���@w�A�
=C��\                                    Bx�k�4  F          AN{AQ���?�{@�\)C�/\AQ��(�@uA�\)C�Ff                                    Bx�k��  "          AF{AQ���
�\)�;33C���AQ��p�=�G�?   C�#�                                    Bx�k�  �          AC
=@�����ff�~{���RC��@����z����RC�u�                                    Bx�k�&  T          AMp�AG��ff@{A1G�C���AG����@�(�A��\C��3                                    Bx�k��  T          ALz�A���?���@���C��\A�G�@uA�  C���                                    Bx�lr  �          AP  Ap��ff�  ��
C���Ap��"{?0��@C�
C�P�                                    Bx�l  �          ARffA�p��Mp��b{C��qA�%���k����\C��                                    Bx�l+�            AT  A	G��=q�a��x  C�4{A	G�� (��(���7�C�L�                                    Bx�l:d  �          AT��A���=q�tz���C�aHA���%G��^�R�n�RC�ff                                    Bx�lI
  �          AT��AG��z��dz��xQ�C�7
AG��&=q���p�C�Z�                                    Bx�lW�  T          AV�RA�����Vff�i��C�l�A��{�#�
�1G�C�|)                                    Bx�lfV  U          AV�\Az���{��=q��ffC��RAz��ff�1G��@Q�C��3                                    Bx�lt�  
Z          AX(�A��G�������\)C�S3A��
=�L���[�
C�(�                                    Bx�l��  �          AW�A
=����ff��
=C��)A
=������G�C�4{                                    Bx�l�H  �          AV=qA���(���\)���\C���A���33���
���C�C�                                    Bx�l��  
�          AUp�A=q�Q���Q���33C�RA=q�����{C�\)                                    Bx�l��  �          AT��A������G����\C��)A������\�ffC���                                    Bx�l�:  T          AUp�A  ��p���=q�噚C��)A  ����  ��G�C��                                     Bx�l��  �          AUG�A���   �����ҏ\C��{A���=q�QG��c33C��=                                    Bx�lۆ  �          AV=qAff���
��=q��{C��Aff�G��a��uG�C��                                     Bx�l�,  
�          AV{A	��������
��p�C��3A	��ff��  ��\)C�@                                     Bx�l��  
Z          AUG�A(��߮��p��\)C��{A(������  ���C�k�                                    Bx�mx  �          AR�HA{��Q���z���RC��qA{����ff���C�
                                    Bx�m  "          AW33A�\��{����C���A�\�
=q��ff���
C���                                    Bx�m$�  
�          ATz�A�R��=q��(���C�s3A�R���������ffC�7
                                    Bx�m3j  
�          AS�
A\)������C���A\)���
��G��ᙚC��3                                    Bx�mB  �          AZffA���
=�����HC�T{A��   ��Q����
C�K�                                    Bx�mP�  �          AR�RA������
=��C�` A��	��s�
���\C��\                                    Bx�m_\  �          APz�A  �Ϯ��{��C�z�A  �{��p���C�W
                                    Bx�mn  
�          AO
=A  ��z���G���p�C��A  ���i�����C�                                    Bx�m|�  �          AR{A"�H�ҏ\��ff���
C���A"�H�����.{�?�C�Q�                                    Bx�m�N  �          AR�RAff������
��z�C�ǮAff�\)�z��z�C���                                    Bx�m��  �          ARffA�
��
��������C�&fA�
�ff�Ǯ���HC��                                    Bx�m��  �          ARffA���
=q�+��<Q�C��A�����=�Q�>���C�H�                                    Bx�m�@  �          AQG�A\)��R�\)�{C�NA\)��>�p�?��C��\                                    Bx�m��  "          AS�A!�p���Q���HC�=qA!�z�?8Q�@FffC��=                                    Bx�mԌ  �          AVffA+�
��(�����C�K�A+�
����>�p�?���C��3                                    Bx�m�2  "          AUp�A%p��G���z����C���A%p��z�?�(�@ʏ\C��f                                    Bx�m��  
�          AW�A�R��ÿQ��_\)C�˅A�R��@�A
ffC�R                                    Bx�n ~  T          AXz�A&�\�	���O\)�Z�HC�:�A&�\��R?�33A�C���                                    Bx�n$  �          AX��A+����^�R�l��C�B�A+��p�?�p�@�=qC�}q                                    Bx�n�  �          A\(�A�\�����{��{C��A�\��\������
C�`                                     Bx�n,p  �          A\��A��	��l(��z=qC��{A��p��O\)�Z=qC�o\                                    Bx�n;  �          A[�A-G�� z��33�
=C��
A-G����>���?�\)C�.                                    Bx�nI�  �          AZ�RA5���R��33�޸RC�L�A5���?E�@N�RC��                                    Bx�nXb  �          A\z�A(���
=�-p��5��C��A(����=��
>��
C��q                                    Bx�ng  T          AZffA$����K��X  C�u�A$���
=�Ǯ�У�C���                                    Bx�nu�  �          A\Q�A"=q�33�w�����C��A"=q��
��  ��{C��
                                    Bx�n�T  �          A^{A�\�33��  ��  C�33A�\��	����C�Q�                                    Bx�n��  T          AUA!G���ff�`���w33C�Q�A!G���\�c�
�w
=C�{                                    Bx�n��  �          ARffA�����33��33C��
A���?�{@�ffC��{                                    Bx�n�F  �          AQG�A ����
��\)����C�T{A �����H@#33A5�C��                                    Bx�n��  �          AS33A(������?�33@�33C���A(����p�@s33A�Q�C�\)                                    Bx�n͒  �          AU�A,����G�@z�A�RC��fA,����\)@��
A�
=C�~�                                    Bx�n�8  "          AW
=A+�
��G�@4z�AAC�w
A+�
�Ǯ@��\A�  C��                                    Bx�n��  �          AS�
A.{����?��H@��HC��A.{��G�@p��A�  C�}q                                    Bx�n��  �          AQ�A,  ��{?\)@{C���A,  �ۅ@;�AN�RC���                                    Bx�o*  �          AS�A.=q����?s33@��
C���A.=q��ff@Q�AfffC�0�                                    Bx�o�  �          AS�A)����
�k��~{C���A)���  ?�
=@�G�C��                                     Bx�o%v  "          AS�
A)��z�   ��C���A)��(�@�Az�C�"�                                    Bx�o4  T          AU�A+���=q���
��ffC���A+����?Ǯ@�\)C��                                    Bx�oB�  T          AU�A-G���{����)�C��HA-G����\=���>�
=C��                                    Bx�oQh  �          AT��Aff��\)��Q����C��Aff�33��G��У�C�Z�                                    Bx�o`  �          AT��Aff��H������Q�C�J=Aff����   �
=C�j=                                    Bx�on�  "          AV�RA����������z�C��A���p������(�C���                                    Bx�o}Z  �          AV�\A�\��������\)C�� A�\�������C���                                    Bx�o�   �          AT(�A
ff�ff������C��A
ff��
���)�C��\                                    Bx�o��  �          AV=q@�{������p�C�>�@�{�*�H��(����C���                                    Bx�o�L            AT  @���{�=q�(�C�*=@���)p���p���
=C�R                                    Bx�o��            AT��@�Q������(��{C��f@�Q��333�y�����C���                                    Bx�oƘ  T          AY�Az������
���C�9�Az����%�/33C���                                    Bx�o�>  U          AY@��
=��(����HC���@��,���p����ffC�"�                                    Bx�o��  T          AT��@����������\)C��R@����&=q��
=����C���                                    Bx�o�  @          AW
=@�(��
=��(��33C�� @�(��/���
=��33C��                                    Bx�p0  �          AT(�@�=q��
�\)�(�C���@�=q�-G���z���33C��q                                    Bx�p�  �          AX��@������{�/ffC�z�@����0Q���G���  C�>�                                    Bx�p|  �          AXQ�@�G���  �\)�>(�C��)@�G��*=q��G���p�C��                                    Bx�p-"  �          AW33@�������ff�2�C���@����0Q�������HC�q�                                    Bx�p;�  T          AQG�@��
�=q��{���C��R@��
�)���!G��6=qC�                                      Bx�pJn  �          AEG�@�����
=q�"{C��@���?�
=@���C���                                    Bx�pY  �          A>�H@�����׿��
��  C��\@�����H?�
=A  C��)                                    Bx�pg�  "          A<(�@���{��  ���\C�h�@���
��ff���HC��                                    Bx�pv`  �          A9@�{�
{���\��z�C��3@�{��R�����ۅC�P�                                    Bx�p�  T          AAG�@ۅ���~�R���
C�u�@ۅ�
=�&ff�Dz�C�N                                    Bx�p��  �          AE�@ۅ�p���Q���p�C�(�@ۅ�#��Tz��vffC��                                    Bx�p�R  �          AHQ�@�����j�H����C���@���33�Ǯ��ffC��                                    Bx�p��  �          AI��@���{�q���=qC�#�@���{��(���p�C��                                    Bx�p��  
�          AI�@���=q�z=q��ffC���@���
=�#�
�;�C�o\                                    Bx�p�D  �          AG�
@�=q����z��ٙ�C��R@�=q�%�{�&{C��3                                    Bx�p��  "          AF�\@����Q����R��ffC��R@����   ������HC��                                    Bx�p�  |          AJ=qA�
���j=q���HC��A�
�p��(���@��C��                                     Bx�p�6  �          AE��A�����5��U��C��qA�����=�?z�C���                                    Bx�q�  T          AB�RAff� �Ϳu���C��Aff��z�?��A  C�<)                                    Bx�q�  T          AF=qA&=q��\)@!�A<��C���A&=q��
=@���A��C�9�                                    Bx�q&(  �          AF�HA$  ��  ?���A	�C�O\A$  ��{@��\A�\)C�y�                                    Bx�q4�  �          AIG�A#
=��\)@5AQ��C�H�A#
=��=q@�  A�(�C�7
                                    Bx�qCt  
�          AL��A*{���@�(�A��
C���A*{�\��@�ffA�Q�C�H                                    Bx�qR  �          ALz�A!����33@�z�A�C�(�A!���=q@�ffB��C�J=                                    Bx�q`�  �          AL��A'����H@��A��C�� A'��L��@�ffA�C���                                    Bx�qof  T          AL(�A1p�����@AG�A[\)C��HA1p����
@���A���C��\                                    Bx�q~  T          ALQ�A����\�  �$��C�l�A���?#�
@9��C���                                    Bx�q��  �          AJ{@�����ff��\)��ffC�0�@����ff�HQ��h��C�\)                                    Bx�q�X  �          AG\)@���
=q��ff��
=C��@���G��˅���HC���                                    Bx�q��  T          AJff@�p��(�����ә�C�^�@�p��*=q��p����C��                                    Bx�q��  �          AJ{@ۅ�	��Ǯ����C�W
@ۅ�#��3�
�M��C��                                    Bx�q�J  "          AFff@���{������HC��f@���  �Q��ffC�q�                                    Bx�q��  �          AH  @�(��ff��p���\C��@�(����)���C�
C���                                    Bx�q�  T          AL��@�{�z��ȣ���RC��@�{�#33�5��Lz�C��
                                    Bx�q�<  h          AO�
@�  ���˅��G�C�Ff@�  �#
=�:�H�RffC���                                    Bx�r�  �          AP(�@�33���H��H���C��q@�33�z���������C�                                    Bx�r�  �          ANff@�  ��G��$���S=qC��@�  �G������C��                                    Bx�r.  
�          AO�@�=q��=q�
=�H�\C��3@�=q�
=���ffC��                                    Bx�r-�  �          AR{@���  ���633C�p�@���H�θR��33C��                                    Bx�r<z  �          ARff@e�qG��@��L�C�Ф@e��33�ff�C�C�K�                                    Bx�rK   
�          AC�@Ϯ�{��ff��Q�C�T{@Ϯ���?�
=A{C�n                                    Bx�rY�  
�          AB=q@�33�zῢ�\��Q�C���@�33��H?�ffA��C��R                                    Bx�rhl  
�          AB�\@��H��
@�\)A���C���@��H���
@�33B�HC�}q                                    Bx�rw  "          AH��@�
=���@tz�A�G�C�\@�
=���@�Q�B	�\C�h�                                    Bx�r��  �          AM�@�ff�)�>���?�  C�!H@�ff��@�z�A�33C�@                                     Bx�r�^  
�          AL��@����!�O\)�h��C��=@����\)@:=qAR�HC�XR                                    Bx�r�  T          AL  @�{�33�����ffC��
@�{��@A�C���                                    Bx�r��  "          AP��A  �Q�˅���C���A  ��R@z�AffC��                                    Bx�r�P  h          AP(�@�
=�$z�?\)@ ��C�s3@�
=��@���A�\)C���                                    Bx�r��  �          AO�@���+
=@3�
AJ{C��@����R@���A�  C�*=                                    Bx�rݜ  �          ALQ�@�33�%G�?޸R@���C�AH@�33��@���A�(�C�.                                    Bx�r�B  �          AK�
@����R@��A Q�C��f@����R@�{Aՙ�C��{                                    Bx�r��  
�          AQ�A�H�{@��\A�Q�C�P�A�H�Ϯ@�  Bp�C���                                    Bx�s	�  
�          AR�R@��z�@�A\C���@���z�A�B�C��3                                    Bx�s4  
Z          AQ��@�z����@�z�A߮C�\@�z���{A��B/33C���                                    Bx�s&�  `          AP��@�
=��
@�\)A�ffC��\@�
=���A�\B2�C��R                                    Bx�s5�  z          AJffA33�(�@�
=A���C�b�A33��@�ffBffC��                                    Bx�sD&  �          AI�A(��\)@q�A��
C��)A(���{@׮B�\C��                                     Bx�sR�  "          AF=qA�
�z�@b�\A��HC�HA�
�Ӆ@�ffA�p�C��                                    Bx�sar  �          AFffA=q�z�@7
=AV�RC�j=A=q���H@�p�A�  C�y�                                    Bx�sp  
�          AI@陚��@��AۮC�Z�@陚��ffA��B*�C�                                      Bx�s~�  T          AI�@���33@�p�A�Q�C���@����=q@�\B(�C��                                     Bx�s�d  
�          AJffAff��@�z�A�Q�C��3Aff���
@�
=B��C���                                    Bx�s�
  T          AJff@��
�  @�=qA���C�g�@��
��\)@�(�B=qC�b�                                    Bx�s��  
�          AL  A���@�=qAυC�ffA���{A\)B ��C�                                      Bx�s�V  �          AL(�A�H�
=@���A�\)C��A�H��@��HB33C��                                    Bx�s��  
Z          AJ�R@�z���@���A��\C��@�z��ʏ\@�33B��C��H                                    Bx�s֢  "          AL��@�33�  @�G�A�RC��\@�33��  Az�B4��C�q                                    Bx�s�H  �          AO�@�
=��  A=qB��C���@�
=�w�A%�BT�
C��                                    Bx�s��  �          AR{@ٙ����A33B ��C�y�@ٙ��dz�A)G�BX(�C�*=                                    Bx�t�  
�          AS\)@��
��(�@��\B��C�\@��
��=qA#�BK\)C�aH                                    Bx�t:  h          AU�A(���\@�
=B��C�4{A(���G�A\)B7=qC���                                    Bx�t�  �          AR�\@�=q�\)@�33A���C�^�@�=q���Ap�B.��C���                                    Bx�t.�  |          AR{@�{�
�\@ƸRA�Q�C�S3@�{����A��B0Q�C�g�                                    Bx�t=,  �          AV{A(  ��
=@�G�A�
=C�0�A(  �S�
@�(�B
��C�B�                                    Bx�tK�  �          AVffA%p���Q�@�  A�G�C��A%p��P��@�33B�\C�AH                                    Bx�tZx  �          AV�RA)����H@���A�\)C���A)��J�H@�33B
33C���                                    Bx�ti  "          AV�RA"�R���@�\)A�33C��A"�R�;�A�B�\C��{                                    Bx�tw�  �          AVffA"{��Q�@��HA���C�"�A"{�\(�A z�B��C��q                                    Bx�t�j  �          AV{A&�H���@�{A�C�7
A&�H�z�H@�  BG�C��{                                    Bx�t�  T          AUA!�陚@��HA��C��A!���
@���B��C�#�                                    Bx�t��  "          AW�A�\��
@��A�C�b�A�\��\)@��BG�C�S3                                    Bx�t�\  "          AV�RA���ff@���A���C��A����p�A��BC��                                    Bx�t�  �          AV=qA{�Q�@p  A�ffC��A{���H@ᙚA�z�C���                                    Bx�tϨ  �          AY�AQ��@�=qA�
=C�'�AQ��Ϯ@�p�B  C���                                    Bx�t�N  �          AVffA�\���H@�{A��
C��A�\��ff@��\B=qC��f                                    Bx�t��  "          AR�HA����@���A�{C���A����@�Q�B  C��                                    Bx�t��  
�          AR=qA"=q���@��A��\C�ٚA"=q��=q@�{A���C���                                    Bx�u
@  
�          AY��A�
��{@陚B  C��A�
��\)A�B6�RC��                                    Bx�u�  
�          A\  @������@��
B�C�K�@������RA#�B@�C���                                    Bx�u'�  �          AVff@��
���@�(�A���C�=q@��
���RA�
B;�
C��=                                    Bx�u62  
�          AT��A�H��33@�33A�p�C���A�H�q�@��RBffC��                                    Bx�uD�  �          AR�HA�R��
=@�Q�A�\)C�K�A�R�C�
A
�\B$��C�                                      Bx�uS~  �          AV�HA�
��ff@�z�B {C��A�
�VffA33B,\)C�
=                                    Bx�ub$  �          A_33A (����@�RA�C�'�A (���  A!p�B:
=C�aH                                    Bx�up�  	�          A`(�A
�R��{@��HB\)C�4{A
�R�~�RA#�
B<(�C��                                    Bx�up  �          Ad��A�
���A�HB�C��A�
�VffA%p�B9��C��                                    Bx�u�  
�          Ad��A  ��\)@���B  C�\)A  �`��A!�B3�HC��R                                    Bx�u��  "          AdQ�A�
� ��@�B�C��A�
���A'\)B<�\C��                                    Bx�u�b  �          Ae�A
ff��A Q�B��C�<)A
ff���A*{B>�\C��                                     Bx�u�  "          Aep�A
=��(�@�{B��C��3A
=��A"�HB5  C���                                    Bx�uȮ  T          Ad(�A�� ��@�Q�B��C��HA�����A%�B9\)C�3                                    Bx�u�T  "          A`z�A  ��Q�@�\B33C���A  �h��Ap�B2�C�@                                     Bx�u��  T          A\��Aff��Q�@�B 
=C�"�Aff�S�
Ap�B,�C�L�                                    Bx�u��  T          AB=qA!�w
=��
�@z�C��\A!���R���>{C��                                    Bx�vF  �          A?33Ap��33���
�(�C��RAp�������  �ң�C�\)                                    Bx�v�  �          AEp�A�
�L����\)���C�qA�
����������\)C���                                    Bx�v �  �          AK\)A{�tz���\)��C�o\A{�ə������(�C��q                                    Bx�v/8  "          AF=qA�R?����5��A�HA�R��\�{�/33C�+�                                    Bx�v=�  h          A@��@�\@7
=�	���:�HA��@�\�Y���Q��G(�C��                                    Bx�vL�  �          AC33A (�>����B�\@W
=A (��O\)�z��2z�C���                                    Bx�v[*  �          A>{@���@  ��
�EC�1�@����=q��p��(
=C�K�                                    Bx�vi�  @          A;\)A33�J=q����.�C�C�A33�y�����
��C�H�                                    Bx�vxv            A;\)@�
=?(������1=q@�\)@�
=�#�
����%��C��                                    Bx�v�  
Z          AE�A(�?����أ����A
�\A(��������C��q                                    Bx�v��  �          AH��A/\)�u��33��\)C�U�A/\)��\)�7��S
=C���                                    Bx�v�h  
n          AK�A��������R��Q�C��=A������p�� Q�C��q                                    Bx�v�  
�          AJ�HA���  ��\)��
=C�h�A���R�ff�,��C�<)                                    Bx�v��  T          AC�
A����33���H�ffC�ٚA���أ���
=��p�C�=q                                    Bx�v�Z  �          AAp�AG��������
�G�C��3AG���
=��  �Ù�C��q                                    Bx�v�   T          AF�\A�H��z������C��A�H��=q�qG����RC�U�                                    Bx�v��  "          AF{A�
�����z���33C�%A�
���\����2�HC���                                    Bx�v�L  �          AF�RAff��z���\)����C�"�Aff�����H�   C�w
                                    Bx�w
�  
�          AF�\Ap��
{�e����HC��{Ap��p�>�=q?��\C�s3                                    Bx�w�  h          AEG�A�R����
=�2=qC�FfA�R�(�?�{@���C��                                    Bx�w(>  �          A>{A�����
�J=q�w33C�G�A�����H����8Q�C��                                     Bx�w6�  �          AB�RA(���  ��
=��=qC���A(���\�ٙ�� z�C��                                    Bx�wE�  "          ADz�A)���I�����R��C��RA)�������\(����HC�S3                                    Bx�wT0  �          ADQ�A%��{��33��G�C�G�A%�������G���{C���                                    Bx�wb�  �          A@  A$���(���  �ۙ�C�Y�A$����  ��  ����C�7
                                    Bx�wq|  T          A@(�A!p�������
�  C��{A!p��@����(�����C��                                    Bx�w�"  T          A>{A����\)�e���\)C�*=A����ff�aG�����C�R                                    Bx�w��  �          AC
=A�����?^�R@�z�C�&fA���33@~{A�ffC�33                                    Bx�w�n  |          A@Q�@��
�Q�@Q�A9��C�ff@��
��p�@�(�A�\)C�e                                    Bx�w�            A?�@�p��G�?!G�@S�
C���@�p����@�33A��C�C�                                    Bx�w��  �          AB�\A����z��^�R��33C���A���	�>�?!G�C�`                                     Bx�w�`  �          AC�
A�������P���w\)C�Z�A����R>�z�?���C�4{                                    Bx�w�  T          A=��A(���
=������C��RA(����
��Q���=qC�7
                                    Bx�w�  �          A@��Az���
=�g���(�C�O\Az����G��o\)C�U�                                    Bx�w�R  �          A;\)A�
�����ff��Q�C��A�
��p�����(��C���                                    Bx�x�  "          A8z�A�H�|(�������C��=A�H��  �C33�v�\C�P�                                    Bx�x�  �          A;33AQ����R��G���G�C���AQ����R�\)�1G�C�O\                                    Bx�x!D  �          A=�A&�R�b�\��=q���C��HA&�R�������=C�                                      Bx�x/�  �          A=�A����p��I���w33C�J=A����=u>��RC���                                    Bx�x>�  �          A=A����������{C�33A����R�
=q�)G�C�޸                                    Bx�xM6  �          A:=qAG���{��{�ՅC��{AG�� z��p��{C���                                    Bx�x[�  �          A9�@����� z��A�\C�j=@��陚��ff��{C���                                    Bx�xj�  �          A:�H@3�
��Q��3���C��@3�
���\�!��tG�C��                                    Bx�xy(  �          A6{@L(�@\(��$��\B<��@L(���G��,���qC�<)                                    Bx�x��  �          A:=q@#�
@33�/�33B'�
@#�
�%��.�R.C�`                                     Bx�x�t  �          AAp�@8��?����:ffk�A�z�@8���\)�0  ǮC���                                    Bx�x�  �          AL  @�G��k��@  �C��q@�G���{�'��^33C�)                                    Bx�x��  ,          AE�@G
=?���9p��=A���@G
=�p���0��\)C��\                                    Bx�x�f  �          AN�R@/\)�(��G���C���@/\)��(��0z��n(�C��H                                    Bx�x�  �          AN�R@r�\�(��B{Q�C��3@r�\��p�� Q��L33C���                                    Bx�x߲  T          AFff@���1��4Q��=C�^�@�����H�  �:�
C��                                    Bx�x�X  �          AE@���tz��)���lp�C�s3@���   ��{�   C�'�                                    Bx�x��  �          AC�@\)�*�H�1p�(�C�)@\)��p����<��C��                                    Bx�y�  h          AM�@����
�H�)���vC��3@�����Q��
=q�7z�C��                                    Bx�yJ  �          ALz�@�33�:�H�7�.C�Ǯ@�33��=q����7{C���                                    Bx�y(�  �          AK33@aG��G��;�Q�C�4{@aG������
�<{C�
=                                    Bx�y7�  T          A<��@Z=q�8Q��,Q�C���@Z=q�����:�RC��)                                    Bx�yF<  T          A>�\@����\)��
�[�C�  @�������������C���                                    Bx�yT�  T          A?33@���{���\�`  C��@����=q��  �(�C�U�                                    Bx�yc�  �          AE��@�Q���
=�%��cQ�C���@�Q��Q���\)�p�C�=q                                    Bx�yr.  �          A<(�@�G��K��'��|  C�� @�G��������-\)C�\                                    Bx�y��  �          A;\)@����|��� ���o�RC���@�����p����H��C�*=                                    Bx�y�z  �          A>�R@�(������\)�L=qC�!H@�(��
=q��=q��z�C��
                                    Bx�y�   �          ABff@ȣ����R��JffC�H�@ȣ��G���  ��C��                                    Bx�y��  |          A=p�@�  �c�
��\�5ffC�K�@�  �ۅ������
=C��                                    Bx�y�l  �          A=�@�(���G��   �*
=C�` @�(�� Q���Q���=qC���                                    Bx�y�  "          AB�R@�=q��p������33C�` @�=q�33�p����{C���                                    Bx�yظ  �          AF�\@��H���H���R� C���@��H�(���\)��(�C��q                                    Bx�y�^  h          AC33A ��������G�C���A �����H��z����
C��                                    Bx�y�  h          AB�HA���*=q�  �0ffC�/\A����33��\)� =qC���                                    Bx�z�  �          AA��A�����R������
C��{A����{�
=�DQ�C�\                                    Bx�zP  �          A@(�A�R��G�������Q�C���A�R�
=�(��;�C���                                    Bx�z!�  �          A=G�A�R�˅���\��z�C�w
A�R� ���   �  C�"�                                    Bx�z0�  �          A=G�AQ���Q���G���C�H�AQ���H�E��p  C�R                                    Bx�z?B  �          A<z�A
ff���
��G�����C���A
ff��{�(��>{C���                                    Bx�zM�  �          A<��A����
=������C���A����׾�ff�(�C���                                    Bx�z\�  �          A?33A  ��z��)���M�C�q�A  � Q�?z�H@�
=C��)                                    Bx�zk4  �          A?33A�R�����]p����\C���A�R��{=L��>uC�'�                                    Bx�zy�            A;
=@���� Q��5��c
=C��@����33?�  @�\)C�\)                                    Bx�z��  h          A;�
@��\����&ff�M��C���@��\�	��?�=q@�G�C�(�                                    Bx�z�&  �          A>=qA���\)�L���zffC���A���ff=�G�?�C���                                    Bx�z��  "          A<��Ap������S33���HC��Ap���33�8Q�aG�C�b�                                    Bx�z�r  �          A:ffA���Ǯ�������C�T{A����G���
=��=qC���                                    Bx�z�  �          A9A������  ���C���A���������{C�˅                                    Bx�zѾ  
�          A7\)@�������p��
�C�aH@�����O\)��33C���                                    Bx�z�d  @          A4��@��
��=q��33�\)C���@��
�ff�xQ���=qC��                                    Bx�z�
  �          A3
=@�  ��z�� z��8�C�ff@�  ���R��Q��ՅC���                                    Bx�z��  "          A1G�@������H��Q��2{C��f@���� (�����{C�xR                                    Bx�{V  �          A/
=@�
=�XQ��p��[p�C��f@�
=��
=�����Q�C�j=                                    Bx�{�  �          A1�@�\)��
=��z�� Q�C��@�\)�G��|(�����C�Z�                                    Bx�{)�  �          A333@�=q��ff��
=��\)C���@�=q�G��c�
���HC���                                    Bx�{8H  
�          A.�R@����p��������C�@����ff�W
=��{C�h�                                    Bx�{F�  T          A)�@أ���\�8���}�C��@أ���?O\)@�z�C�H                                    Bx�{U�            A)p�Az���Q��
=�N�\C��qAz��ƸR>��@#�
C���                                    Bx�{d:  
�          A"�H@����  �E����HC��
@����@�AA��C���                                    Bx�{r�  �          A ��A	���n�R�X������C�FfA	����33��������C�G�                                    Bx�{��  T          A!A  �G��h����p�C��A  �j=q�\)�O\)C��                                    Bx�{�,  �          A�@����k������C���@�����
=�,(���=qC�                                    Bx�{��  �          A��@�\�\(���z���C�j=@�\��z��B�\���C��{                                    Bx�{�x  "          Ap�@��R���������HC��\@��R�b�\�s�
��33C��=                                    Bx�{�  �          A�@�(��W
=��z���C��q@�(��K�����
=C�S3                                    Bx�{��  T          A�@��H�ff��Q���Q�C�h�@��H�y���!G�����C�c�                                    Bx�{�j  
�          AG�@�Q��=q�������\C�*=@�Q��s33�]p���z�C��R                                    Bx�{�  K          Ap�@��
���\��G����C��@��
�p  ���H���C���                                    Bx�{��  
Z          AG�@�(��<(����5�\C�8R@�(����������C���                                    Bx�|\  �          A@�33�8Q�����D��C�%@�33�W
=�ƸR�&�C���                                    Bx�|  T          AG�@�?�\)��z��@�RAk
=@������33�?=qC���                                    Bx�|"�  �          A�R@�  @�����J�A���@�  ���
��z��X\)C�o\                                    Bx�|1N  �          A@�ff����%��zffC���@�ff����>��R?���C�s3                                    Bx�|?�  
�          A�@ᙚ��z��#�
�p��C��@ᙚ��33?!G�@i��C��f                                    Bx�|N�  
�          Az�@�ff�ٙ��2�\��ffC��H@�ff��=q?(�@eC���                                    Bx�|]@  
�          A��@أ���G��,(��
=C�  @أ��ᙚ?z�@Z�HC��                                    Bx�|k�  �          AQ�@ٙ���\)�,����(�C�.@ٙ���  ?��@O\)C��                                    Bx�|z�  T          AQ�@������<������C�ff@�����>�@5C�5�                                    Bx�|�2  +          A{@��
��p��.�R���C��H@��
����?:�H@��HC���                                    Bx�|��  �          A�\@���z��-p����RC�
=@���z�?�R@mp�C���                                    Bx�|�~  �          AG�@�����\)�G��V{C��R@�������?��@�C��)                                    Bx�|�$  �          A33@�=q��=q�/\)��p�C�k�@�=q��=q?.{@�G�C�j=                                    Bx�|��  �          A�H@�����AG���\)C�c�@���p�?G�@�33C�xR                                    Bx�|�p  �          A�@�33�����HQ���33C��)@�33�p�?E�@��HC��                                    Bx�|�  �          A�\@����(��Q��d  C�*=@����?�z�@���C��R                                    Bx�|�  
�          A{@׮���ÿ�z��7�
C���@׮��p�?�ff@��C���                                    Bx�|�b  �          A�@�{��p����5�C�7
@�{��\)@�\AZffC�Ff                                    Bx�}  "          A�@�\����k���
=C��q@�\��z�@G�AAC��q                                    Bx�}�  �          Aff@�=q��ff�Tz���C��)@�=q��z�@�AK\)C��R                                    Bx�}*T  T          Aff@�=q��33�c�
��G�C�4{@�=q��(�?��A*�RC���                                    Bx�}8�  �          A�@��
�����G��(z�C�.@��
���R?Y��@�33C��H                                    Bx�}G�  
�          A��A�������
=��C��RA���  ?W
=@�Q�C�u�                                    Bx�}VF  
�          Az�@���Ǯ��G��&ffC��H@����33?���@��C���                                    Bx�}d�  T          A��@��H�ȣ׿�Q���\C�@��H���H?�33A\)C���                                    Bx�}s�            A  @޸R�ҏ\����ƸRC�Q�@޸R���@�AK
=C��                                    Bx�}�8  �          A  @Ǯ���H�������
C�0�@Ǯ���@�A^{C��)                                    Bx�}��  U          A
=@�����33��G��'�C��@������@O\)A�z�C�%                                    Bx�}��  	�          A��@�z���
=�+���G�C�l�@�z����
@FffA�(�C�t{                                    Bx�}�*  �          A�@����p���
=�'�C�C�@�����?�(�A*�HC�G�                                    Bx�}��  |          A��@�\)��ff�,����33C��=@�\)��z���Ϳ��C��                                    Bx�}�v  �          Az�A	�<(��_\)��{C��A	��\)��p��#33C��=                                    Bx�}�  �          A�Ap���  �&ff�x��C��HAp���
=���ÿ���C���                                    Bx�}��  "          A��A  �@  �G����C���A  �����\)� ��C�n                                    Bx�}�h  �          AA��s33�>�R��=qC���A���  �^�R��33C�Z�                                    Bx�~  �          A�AQ��=q�{���\)C��AQ��}p��
=�aG�C���                                    Bx�~�  �          A
=A=q����p����C��\A=q��=q�@  ���C�^�                                    Bx�~#Z  �          A33A���33��ff�  C��A��c�
��G���ffC�Q�                                    Bx�~2   
�          A   Ap��u��(����C��RAp��W
=�������C��                                    Bx�~@�  
�          A"ffA
=?�33��=q��(�A=qA
=�fff��{�܏\C�#�                                    Bx�~OL  �          A"�\A
�H���������C��RA
�H�#�
���\�֣�C��=                                    Bx�~]�  
�          A ��A��?O\)�����@���A������������
C���                                    Bx�~l�  �          A!�A33?�=q��  ��A.=qA33��
=��33�=qC��                                    Bx�~{>  "          A"=qA�H?�z���(��33@��\A�H��  �������C���                                    Bx�~��  �          A�
@���@N{����
�\A�\)@���>L����Q��#\)?��R                                    Bx�~��  �          A�H@��@vff���\��
A�R@��?(����ff�1�
@��
                                    Bx�~�0  �          A&ff@���� ��@�\A4��C���@�������@��\B��C�˅                                    Bx�~��  �          A,(�@��H���@�
AF=qC���@��H��
=@���B	
=C���                                    Bx�~�|  �          A*ff@�
=���?�p�A+�C��)@�
=�ᙚ@��B	\)C�<)                                    Bx�~�"  �          A*�H@���@6ffAyp�C�3@���z�@�z�B"=qC��                                     Bx�~��  �          A((�@�Q��33@�A:�HC��q@�Q�����@\B��C���                                    Bx�~�n  �          A%�@J=q��
@   A`��C�5�@J=q��{@�p�B!\)C���                                    Bx�~�  �          A&�\@`����?���A-�C�{@`������@�\)Bp�C���                                    Bx��  T          A'
=@\)��R?�\)A&{C�|)@\)��p�@��HB\)C�"�                                    Bx�`  �          A(��@��H���?���@�p�C��f@��H��{@���A��C��q                                    Bx�+  �          A+�
@�ff�p�?c�
@�Q�C�@�ff� z�@��A�C�(�                                    Bx�9�  �          A-�@vff� z�>�(�@G�C���@vff�
=q@��A���C��                                    Bx�HR  �          A-G�@�����
?:�H@y��C�t{@�������@�Q�A�G�C���                                    Bx�V�  �          A1�@����%��>��H@!G�C���@����=q@�=qA���C�0�                                    Bx�e�  T          A2�R@`  �'�?:�H@q�C�AH@`  �{@��HA��HC���                                    Bx�tD  �          A.ff@�  ��Q��i�����RC�k�@�  ���>�@\)C��                                    Bx���  T          A0��@�{�
ff����I�C�XR@�{�  ?�(�A$��C�,�                                    Bx���  �          A-�@ə��\)�Fff��33C���@ə��33?�z�@�C��{                                    Bx��6  �          A.�R@Ӆ����I����{C��)@Ӆ�	�?��@�33C��)                                    Bx���  �          A/33@\)��{����
�C�U�@\)��\��z��G�C���                                    Bx���  �          A.ff@����
=��
=�Q�C��H@���\)�ٙ���C��                                    Bx��(  �          A-��@�z��	���(���z�C���@�z�����ff��C�E                                    Bx���  �          A-�@�����z������=qC��q@�����׿�����Q�C��                                    Bx��t  �          A/
=@�{�ᙚ���\�33C��@�{�����
�=qC���                                    Bx��  T          A-G�@ʏ\��{��{���\C�/\@ʏ\�
=���R��(�C��                                    Bx���  �          A+\)@����Q���{��HC�u�@��������7\)C��f                                    Bx��f  �          A*�\@�G�������=q��\C�\)@�G���\)�I�����C�q                                    Bx��$  �          A(��@�����=q�����=qC��@�����=q�U���\)C�e                                    Bx��2�  �          A(Q�@�����������HC�~�@���{�#33�c�C��{                                    Bx��AX  T          A*=q@����\�ҏ\�p�C��q@��(��@����C���                                    Bx��O�  �          A)p�@�����������
�RC�L�@����Q��1G��s
=C�N                                    Bx��^�  �          A'�
@�  ������  ��(�C��f@�  ��=q�#�
�c33C��q                                    Bx��mJ  �          A&ff@�Q����H�����RC���@�Q����   �_\)C�"�                                    Bx��{�  �          A(  @˅����z��{C���@˅��ff�333�zffC��                                    Bx����  �          A&�H@�ff��{���\��\C�s3@�ff����8Q����RC�R                                    Bx���<  |          A%��A�R�c33��
=��
=C�C�A�R����8����(�C�                                    Bx����  �          A%G�@���xQ���(��(�C���@����33�E��\)C��                                    Bx����  �          A%G�@�G�������p����C�
@�G���{�4z��~=qC��q                                    Bx���.  T          A%�A�
�`  ����C�}qA�
��G��7
=��ffC�
=                                    Bx����  �          A$��A33��  ��Q���Q�C�  A33��G�����K\)C�c�                                    Bx���z  �          A$��A Q���G������ޣ�C���A Q���=q�
=q�B{C�s3                                    Bx���   �          A ��@�33���
��Q���z�C���@�33��33�!G��hz�C���                                    Bx����  �          A"{@������R��\)����C�{@������G��7�
C��                                     Bx��l  �          A"=q@��������(���C���@�����H��R�K\)C��                                    Bx��  �          A#�
@�33���\��Q��ϙ�C�'�@�33���H��z����C�8R                                    Bx��+�  T          A%��@��޸R������C��@��أ�@�A=C���                                    Bx��:^  �          A#�@���=q�����G�C��)@���\)?�33A+�C��=                                    Bx��I  �          A"�HA�����ÿ���C���A����  ?Tz�@�{C�1�                                    Bx��W�  �          A$z�A�
�b�\��{�ǮC���A�
�h��?:�H@��C���                                    Bx��fP  T          A$z�A\)�^�R������G�C���A\)�mp�>�@'
=C�J=                                    Bx��t�  "          A&�\A�Ϳz��]p���G�C�P�A���
�H�0  �u��C�                                    Bx����  @          A!@�z����=�\)>�(�C��=@�z���{@UA�\)C��)                                    Bx���B            A�RA
{���H��=q���
C���A
{��=q?��@׮C��3                                    Bx����  T          A z�A���C�
��=q���C���A���\(�=�Q�?�C��                                    Bx����  �          A Q�A�
�]p��У��C���A�
�tz�>u?��C��f                                    Bx���4  �          A"�HAG��C33��z��   C�'�AG��W
=>k�?��C�W
                                    Bx����  �          A ��A�\�N{������p�C���A�\�^�R>�{?�
=C��
                                    Bx��ۀ  "          A#
=A\)�`�׿=p���
=C���A\)�[�?��@�z�C�f                                    Bx���&  �          A"�HA���?G�@�z�C�4{A����H?���A%�C��                                    Bx����  T          A%��A%��#�
>L��?�{C��HA%�=u>L��?��>���                                    Bx��r  �          A%�A#���  ?�z�@��C���A#��
=q?�{A�C�~�                                    Bx��  �          A%�A!����(�?�z�@��\C�(�A!���Tz�@z�A8Q�C��f                                    Bx��$�  T          A'33A�
��G�@�AN�\C���A�
��
=@8��A��HC��=                                    Bx��3d  �          A&ffA���Q�@,(�Ar�\C�ФA�<�@C33A�(�>L��                                    Bx��B
  �          A#�Aff��Q�?�\)@ʏ\C��Aff�hQ�@Q�A��RC���                                    Bx��P�  �          A"�\A����=q?��HA2�RC�L�A���C33@z�HA��\C�q�                                    Bx��_V  �          A$(�A���c�
@G�AV=qC��A���G�@mp�A��C���                                    Bx��m�  �          A$Q�A  ���
?��\@�ffC�FfA  ��@g
=A���C�@                                     Bx��|�  "          A#�AQ����\?��@�C��RAQ��x��@\��A�ffC��                                    Bx���H  �          A#\)A=q��ff@ ��A6�HC���A=q�;�@z=qA���C���                                    Bx����  T          A@����Q�?\)@R�\C��@������@Q�A��HC�aH                                    Bx����  
�          AQ�@�33���
�
=q�K�C�1�@�33��@  AW\)C�J=                                    Bx���:  r          A�H@��R�^{@�
=BK��C��@��R?z�A\)Bj�
@�{                                    Bx����  �          A��@����aG�@�BN�\C��@���?��A(�Bo=q@��                                    Bx��Ԇ  
�          Az�@��H�Mp�@�ffBc33C��@��H?��A(�B~p�Ac�
                                    Bx���,  �          A@��
�hQ�@�Q�BZ�HC�Q�@��
?z�A��B�{AG�                                    Bx����  �          Ap�@����z�@��B0  C�c�@�����A33Bk{C��q                                    Bx�� x  
�          A(�@�R���@�Q�B>�\C�0�@�R����A��B��C�.                                    Bx��  )          A(�@.�R���\@�z�B;C��\@.�R��G�AB���C��)                                    Bx���  M          A@�33����@�=qB-
=C�f@�33����A�HBj=qC��                                    Bx��,j  T          A�\@����=q@���B;�\C�~�@���L��A�
Bv��C��\                                    Bx��;  �          A��@�����\)@��HA�(�C��H@��ÿ�@�
=B�RC���                                    Bx��I�  "          AQ�@������R@�A���C��@����(Q�@У�B3=qC�7
                                    Bx��X\  
�          A��@�{�ָR@���B�C��)@�{�S�
@��B_G�C��{                                    Bx��g  "          A�?�\���HA�BlffC��=?�\��A�\B��3C��{                                    Bx��u�  �          A�@�����z�@=qAdQ�C��f@�����\)@�{BQ�C�W
                                    Bx���N  
�          A
=@��\��\)@~{A�\)C��@��\���
@�33B7��C��H                                    Bx����  
�          A33@�p���G�@�p�A��HC�/\@�p���=q@��BM�\C��                                    Bx����  �          A�H?��R��33A z�Bb�C�*=?��R�\)A�
B���C��=                                    Bx���@  
�          A
=@ff�ƸR@��B?  C�` @ff���A�\B��C���                                    Bx����  "          Aff?�p���\)@�z�BT�C�=q?�p���
=A�HB��fC���                                    Bx��͌  "          A{@x���ҏ\@�Q�B��C�K�@x���5�A33Bqp�C���                                    Bx���2  "          A  @�(���\)@�{B
=C��)@�(��333@���BTp�C��\                                    Bx����  �          A��@��H��@�p�B��C�f@��H��H@��HBT�RC��\                                    Bx���~  �          A�@��\��G�@��HB�C���@��\�   @���BS{C�#�                                    Bx��$  "          Az�@+���p�@��
B(�C��{@+��Dz�A\)B��HC���                                    Bx���  
�          A��@J=q��p�@��HB&�C��@J=q�!�A
�\B��)C��\                                    Bx��%p  �          Ap�@�z�����@�B,{C���@�z��Q�A	G�Bu{C�w
                                    Bx��4  T          A��@�33��  @���B/\)C��)@�33��p�A�B{�C��\                                    Bx��B�  �          Aff@�{��z�@���B5(�C�1�@�{����A
=qBs�\C��\                                    Bx��Qb  
�          A
=@�ff���
@���BG�C���@�ff���@��B>{C�@                                     Bx��`  
�          A33@�\)��{@�z�A��C���@�\)��@�p�B0p�C�O\                                    Bx��n�  #          A�H@�\)��=q@���B(�C�u�@�\)�<��@�{BL�C��
                                    Bx��}T  J          A
=@�����R@�{A�C���@����33@�z�B��C��
                                    Bx����  �          A%�AG��   ?�=qA%��C�B�AG��Y��@$z�Ai��C��                                    Bx����  "          A$��A�R�u�@*=qApQ�C���A�R�33@�{A�ffC���                                    Bx���F  T          A#\)A�R��
=@3�
A�(�C��RA�R�>�R@��
A��HC�B�                                    Bx����  �          A"=qAp����H@5�A�  C�K�Ap��R�\@�=qA��HC��                                    Bx��ƒ  �          A ��A z���Q�@8��A�{C�c�A z��L(�@��\A�C�+�                                    Bx���8  �          A#
=@���Q�@��
A�\)C��R@��,(�@ϮB�C���                                    Bx����  T          A Q�@�{��G�@��\B�C�޸@�{�@�{BI(�C�<)                                    Bx���  
�          A Q�@��
��\)@���B��C�@ @��
�!�@��BI��C�U�                                    Bx��*  T          A!�@�����Q�@�\)B�
C�L�@����A=qBVz�C��                                    Bx���  T          A   @����z�@���B��C�R@����Q�A�BYp�C�C�                                    Bx��v  �          A (�@��\��=q@ϮB!��C��@��\�	��A
{Bn\)C�p�                                    Bx��-  T          A ��@�=q�˅@��B(�C�H�@�=q�,��A�B\(�C���                                    Bx��;�  
Z          A ��@w���p�@�B\)C��@w��,��A�Bw�C��3                                    Bx��Jh  
�          A"�\@�(��љ�@�(�Bz�C�� @�(��/\)A��Bg(�C��                                    Bx��Y  �          A$��@��\��{@�p�B!��C���@��\�&ffAG�Bw�HC���                                    Bx��g�  �          A%��@i���ָR@޸RB)G�C�C�@i����RAp�B�u�C��                                    Bx��vZ  T          A%p�@�z�����@�=qB�HC��@�z��K�A�Bhp�C��                                    Bx���   
�          A%�@E��޸R@�B)=qC��{@E��-p�A
=B���C�\)                                    Bx����  �          A&ff@k���
=@�(�B\)C��@k��c33A�
Bqz�C�
=                                    Bx���L  
�          A((�@J=q��
=@�33B:(�C�@J=q���HA��B�ffC�!H                                    Bx����  �          A(��@
=��ffA  Ba
=C��{@
=�   A$��B�33C�Ff                                    Bx����  "          A&�H@Tz���=q@��RBG��C��R@TzῚ�HA��B�G�C��{                                    Bx���>  �          A&=q@8Q����HA�Bi��C�
=@8Q�>�\)A
=B��{@�33                                    Bx����  �          A%p�@2�\����A��BkffC��@2�\>��
A=qB�Q�@�(�                                    Bx���  T          A$��@���=qA(�Bu��C�(�@�?��A�B��)Ah��                                    Bx���0  
�          A$  @'
=�j�HA�RB\)C��{@'
=?�  A��B�{A���                                    Bx���  T          A"�R@���4z�Az�B�B�C�� @��@
�HA
=B�\B5��                                    Bx��|  
�          A"ff@{� ��A��B�B�C�� @{@��A��B��)B>�H                                    Bx��&"            A#
=@�����\@��B��C��R@���(�@��BH�C��R                                    Bx��4�            A'�@�p���@���A���C�4{@�p����H@���BM�\C�b�                                    Bx��Cn  �          A&{@��H��\)@�ffA�{C��H@��H���@���BB�
C���                                    Bx��R  T          A'\)@������@��
B�\C���@���_\)A33BkC�Ф                                    Bx��`�  T          A)G�@����z�@���A��\C�>�@������A�HBTz�C�q                                    Bx��o`  �          A(��@�
=��
=@�  A�G�C��@�
=��=qA��B[  C��R                                    Bx��~  
�          A+
=@�Q���Q�@��B5�HC��H@�Q�k�A�RBo
=C��                                    Bx����  T          A*{@��\���@��B8�
C�@ @��\�333AffBn\)C�G�                                    Bx���R  �          A(��@��H�Y��A��Bd�C��)@��H?��A��B{�
A~{                                    Bx����  �          A&ff@�����@�(�B'{C�4{@�����
ABlp�C�+�                                    Bx����  �          A%��@�\)��p�@�B6�C�c�@�\)�8Q�AG�BlffC�3                                    Bx���D  �          A$��@�ff��A��BY�RC��\@�ff>�p�Az�B��@���                                    Bx����  
�          A$Q�@{��s33A
ffBg  C��q@{�?aG�A�\B��AK
=                                    Bx���            A"�\@qG��n{A
ffBj33C��{@qG�?s33A�B��Ac33                                    Bx���6  T          A"�R@����[�A	�Bgz�C�5�@���?�z�A�\B���Az=q                                    Bx���  �          A"�H@S�
���@�p�BN�
C���@S�
�L��A  B���C�<)                                    Bx���  T          A!@l(���=qAp�B^��C�C�@l(�>�=qA{B��@�(�                                    Bx��(  �          A#�@/\)���A
=Ba\)C��3@/\)�L��AQ�B���C���                                    Bx��-�  �          A#�@
=���A	�Bh�C�@
=��A�B�#�C��                                    Bx��<t  �          A((�@�{��
=A (�BK�
C��@�{>L��A��Bu\)@�\                                    Bx��K  �          A/
=@��R��A�BL�C�S3@��R<#�
A�HB{�>.{                                    Bx��Y�  �          A/33@�����RAz�BK�C���@��<#�
A�
BzG�>��                                    Bx��hf  �          A/�@�����A�HBB{C�)@�����A�RBy(�C�Ǯ                                    Bx��w  �          A,z�@�{��{A33BF=qC�� @�{��33A��Bz�C���                                    Bx����  �          A+�@�
=����@��
B-�C�  @�
=��  A(�BnffC��)                                    Bx���X  �          A,  @\���|(�ABr��C��)@\��?�=qA!��B��\A�33                                    Bx����  �          A/\)@dz��l(�A�HBxG�C��@dz�?��HA$  B���A��                                    Bx����  �          A0z�@�33�y��A��Bh
=C�@�33?�{A!�B���Af{                                    Bx���J  �          A)�@
=q����A{B{�RC�ff@
=q?L��A$z�B��qA���                                    Bx����  
�          A(��?�{�^�RA��B�L�C�c�?�{?ٙ�A$  B�W
B:\)                                    Bx��ݖ  U          A)�?�
=�AG�A (�B��C���?�
=@��A#�B��\Bw\)                                    Bx���<  �          A(��@'��AG�A33B�33C�y�@'�@33A33B�
=B{                                    Bx����  �          A)p�@tz��HQ�AG�Bx\)C�S3@tz�?�\A�HB�  Aƣ�                                    Bx��	�  �          A*=q@�����G�Ap�BaffC�Q�@���?=p�A
=B��qA(�                                    Bx��.  �          A+\)@�
=��  A��B_C�e@�
=?\)A��B�{@�                                    Bx��&�  �          A*�R@��
���AffBc�C��R@��
?@  A  B��A$Q�                                    Bx��5z  �          A'�
@�{�j=qA(�B\=qC���@�{?fffA�Bz�\A.�R                                    Bx��D   T          A,(�@�(���ff@�ffB+�C�G�@�(��#�
A	�BY{C��                                    Bx��R�  �          A*�\@����p��A��BL�C��H@���?8Q�Ap�Bj�@�ff                                    Bx��al  �          A,��@�=q�~{A  BP  C��)@�=q?�RABp��@�z�                                    Bx��p  �          A,��@��
�y��A
�\BUz�C�\)@��
?B�\A�BuG�A33                                    Bx��~�  �          A+�
@�
=�c�
A\)BbG�C�� @�
=?���A��B|  Adz�                                    Bx���^  T          A,��@��^�RAG�Be
=C��3@�?�=qA{B|�A|��                                    Bx���  �          A,(�@�
=�W
=A�HBj��C���@�
=?�p�A�\B�A��\                                    Bx����  �          A+�@l(��L(�Az�B{�\C���@l(�?��A=qB��A�p�                                    Bx���P  �          A+33@Q��Z�HAp�B~C�޸@Q�?�\)A ��B�
=A�                                      Bx����  �          A*�R@y���\)A��Biz�C�/\@y��?^�RAB��HAH(�                                    Bx��֜  �          A*{@dz���p�Ap�Bkz�C�N@dz�?8Q�A�B��
A6�R                                    Bx���B  �          A*ff@,(��tz�A��B��C���@,(�?��RA#\)B���AŮ                                    Bx����  �          A*=q?�������AffB��C�ٚ?���?��A&�\B�ǮA�\                                    Bx���  �          A(��?���aG�Az�B��C��f?��?�\)A$Q�B��B2                                      Bx��4  �          A+\)>��p�A&�RB��\C��>�@Dz�A#33B�#�B��                                    Bx���  �          A.=q<��
��33A,��B��fC�k�<��
@�{A z�B��B��=                                    Bx��.�  �          A/33��=q��
=A-�B�=qC�}q��=q@�A!B��B�z�                                    Bx��=&  �          A.�H��{��(�A-p�B���C~c׾�{@�(�A!��B�\)B�p�                                    Bx��K�  
�          A0Q�>�(����RA.ffB�ǮC���>�(�@z�HA$z�B�
=B��=                                    Bx��Zr  �          A0z�?0�׿\A.ffB���C�'�?0��@y��A$��B���B�
=                                    Bx��i  �          A2=q?&ff��
A.�RB���C���?&ff@[�A)�B�p�B�u�                                    Bx��w�  �          A3\)?+��
=A.�HB�C��f?+�@J�HA+�B�aHB�B�                                    Bx���d  �          A1�?�  �8Q�A)G�B��
C��=?�  @"�\A*�\B�\Bm\)                                    Bx���
  �          A0��?��H�B�\A'�B�.C���?��H@ffA*ffB��BW                                    Bx����  �          A1?��\�&ffA+\)B�ǮC��?��\@5A*ffB�W
B���                                    Bx���V  �          A0  >�{� ��A*�HB�Q�C��>�{@9��A)G�B��B��=                                    Bx����  �          A0(�>���p�A+�B��3C�q�>��@>{A)p�B�W
B��H                                    Bx��Ϣ  �          A2ff>��!�A-��B�L�C�
>�@=p�A+�
B��
B��                                    Bx���H  �          A4��?�R�5A.�RB�33C�  ?�R@.{A/33B�ffB�u�                                    Bx����  �          A5p�>�ff�l��A+\)B���C�w
>�ff?�{A2�RB��=B��f                                    Bx����  
�          A733?(��xQ�A,(�B��fC�u�?(�?�p�A4��B���B��                                    Bx��
:  T          A7�?:�H��A*�RB���C���?:�H?�A5�B�
=Bz��                                    Bx���  �          A6�H?5��{A)B�Q�C���?5?�\)A5G�B�p�Bz��                                    Bx��'�  �          A5G�?��
����A'
=B�B�C���?��
?��HA3�B��
BF�                                    Bx��6,  �          A5p�?�����33A%p�B��qC�q?���?���A2�\B�8RB
ff                                    Bx��D�  �          A5�@ff��  A&{B��fC��)@ff?�A0z�B�{B
=                                    Bx��Sx  �          A4�Ϳ��Ϳ�(�A*=qB��\C^(�����@\)A
=B��B�u�                                    Bx��b  �          A9G��|(���\)A-�B���C5��|(�@��\A��B`(�B���                                    Bx��p�  �          A;33�p  ����A1�B�k�C9
=�p  @�
=AffBgG�B�\                                    Bx��j  �          A:=q�8�ÿ.{A3�
B�u�CAL��8��@��A#�Bu��B��                                    Bx���  
�          A9�'���RA4z�B��CA5��'�@��A#�Bw�B�33                                    Bx����  �          A:�H�c33��z�A1B�p�C8���c33@���A�RBh�HB��{                                    Bx���\  T          A:�R�P  �k�A3
=B��\C8
=�P  @��A\)Bj�HB��                                    Bx���  �          A;33�4z�<�A5��B�
=C3aH�4z�@��A�Bj=qB�(�                                    Bx��Ȩ  �          A=��$z�>��HA8Q�B�aHC)#��$z�@�z�A�\Bd{B�.                                    Bx���N  �          A=���?��A9p�B��C&  ���@�Q�A�RBc�B�u�                                    Bx����  �          A?
=���?&ffA;33B��
C#}q���@ÅA   Bc��Bۏ\                                    Bx����  �          A?
=���>��HA:=qB�\)C(\)���@�{A z�BeB�Ǯ                                    Bx��@  �          A;�
@����p�@��RB,��C��@�녿��RA
=BaC�q�                                    Bx���  �          A:�\@ƸR���
@�B+ffC�L�@ƸR��
=A(�Bcz�C���                                    Bx�� �  �          A8��@�G�����@��HB(�C���@�G�����A�BW��C��                                    Bx��/2  �          A:{@�����@���BC�` @�����ABNp�C�T{                                    Bx��=�  �          A9@��
�Ϯ@�  B=qC���@��
��HA�BR�\C��{                                    Bx��L~  �          A7\)@ٙ����@�(�B�C���@ٙ��(�A�BM�C�{                                    Bx��[$  T          A7�@���  @��B��C��=@���\A�RBL
=C�Ǯ                                    Bx��i�  T          A8  @ٙ����H@�Q�B
  C���@ٙ��0  Ap�BIQ�C��)                                    Bx��xp  �          A7�@�{�Ӆ@ƸRB�
C�.@�{�:=qA	��BC=qC��)                                    Bx���  �          A6ff@����{@��\Aȏ\C��@�����\@�z�B(z�C��{                                    Bx����  �          A5G�@�G���\)?��
@�ffC��{@�G���=q@�33A�33C�f                                    Bx���b  �          A5�@�����@�Q�A�(�C�,�@���Q�@�ffBp�C�l�                                    Bx���  �          A6ff@�=q�   @��ADz�C��
@�=q��{@��
A�
=C��3                                    Bx����  �          A6�\A�����aG���\)C��A����@aG�A���C�^�                                    Bx���T  �          A6�\Ap��   �333�a�C���Ap�����@HQ�A~�HC�Ǯ                                    Bx����  �          A6�RA��� z�\��
=C���A�����@Z�HA��C��                                    Bx����  �          A7�Ap���u���RC�y�Ap���{@o\)A�ffC�/\                                    Bx���F  
�          A733@�z���R?�Q�@��RC�)@�z���z�@��A���C���                                    Bx��
�  T          A6�H@�p��33?�@)��C�H@�p���  @���A�
=C�B�                                    Bx���  
�          A6�R@�Q���?��@�  C���@�Q����@�ffA���C��f                                    Bx��(8  �          A7
=@��R���?�{@��C�\)@��R���
@��
A�=qC�&f                                    Bx��6�  �          A6�H@�  ��R?�ffAp�C�ٚ@�  ��33@���A�\)C�P�                                    Bx��E�  �          A5�@�(�� ��@5Ai�C�@ @�(�����@�Q�B�HC��=                                    Bx��T*  �          A0z�@�G���33?�ff@��HC�t{@�G���ff@��\AɅC�@                                     Bx��b�  �          A0��@����\)@I��A�(�C���@�����
@�z�Bp�C��q                                    Bx��qv  �          A/�@���(�@�33B)=qC���@��uA�HBJ�C�H                                    Bx���  �          A1G�@�
=��  @�{B0
=C���@�
=�B�\Az�BR=qC�+�                                    Bx����  �          A1�@����z�@�ffB"z�C�<)@���
=qA�RBF33C��{                                    Bx���h  �          A3�@���У�@�(�A�(�C�"�@���Q�@�G�B0C��=                                    Bx���  �          A2�R@�33��G�@n�RA�  C�"�@�33��ff@ٙ�B��C�                                    Bx����  �          A2{@�p����
?��R@��C�!H@�p�����@�
=A�
=C�W
                                    Bx���Z  T          A0  @���� (�?ٙ�A
=C�%@������@�
=A�\)C���                                    Bx���   
�          A.�H@����\)@33AB�\C�� @����Q�@�\)A���C��=                                    Bx���  h          A1�A���
=?�(�@�ffC��A���  @���A£�C�W
                                    Bx���L  �          A0Q�@�33���R@.{Ad��C��)@�33���@��HBG�C�\)                                    Bx���  �          A0z�@У��  @K�A��\C�*=@У���33@�(�BffC�                                    Bx���  �          A1G�@�
=��@VffA�33C���@�
=���H@�=qB�C���                                    Bx��!>  �          A2�R@�  ��Q�@��HA�  C��@�  �b�\@�33B4{C���                                    Bx��/�  �          A3�@��H��
=@��A��
C�=q@��H�u@�{B �
C��{                                    Bx��>�  �          A3�
@ҏ\�	��@<(�Ar�HC���@ҏ\����@��B33C�.                                    Bx��M0  �          A333@ٙ����@FffA��HC���@ٙ����R@�=qB�C�b�                                    Bx��[�  �          A2=q@�����@.�RAc�C���@���ʏ\@�33BffC�H                                    Bx��j|  �          A1�@����
=?ٙ�AC�\)@�����(�@�\)A�C�S3                                    Bx��y"  �          A3\)@�\)����@���A���C��f@�\)���R@��\B1ffC��q                                    Bx����  T          A333@����33@�\)A�=qC���@�����@���B0�C���                                    Bx���n  T          A2�R@����(�@�A���C�@���aG�A\)B=�
C�3                                    Bx���  �          A2�\@ᙚ��\)@�ffA陚C�(�@ᙚ�`  @�p�B4��C�Ф                                    Bx����  �          A3�@�ff��{@O\)A��\C�)@�ff���\@�Q�B33C��                                    Bx���`  �          A3�@����\)?�\)AG�C��@�����\)@�G�A���C��                                    Bx���  �          A2�\@�=q�	�@   A%C�C�@�=q�ָR@�{A��
C��q                                    Bx��߬  �          A0Q�@���\)@H��A�  C�  @����(�@���B��C��                                     Bx���R  �          A.ff@�(���G�@��HB=qC�O\@�(��`  A
{BR  C��                                    Bx����  �          A/�@����=q@��HB/=qC�T{@����p�A�\BlQ�C��                                    Bx���  �          A0  @��
���@a�A�\)C���@��
��\)@���B#��C�^�                                    Bx��D  �          A0(�@����G�@p  A�(�C��@�����
=@��
B"ffC�>�                                    Bx��(�  �          A0(�@�33���@j�HA�Q�C��@�33����@��B (�C�Ff                                    Bx��7�  �          A0(�@�\)�Q�@��A9�C���@�\)��33@��\B ��C�AH                                    Bx��F6  �          A1p�@����\)?�A��C�R@�����
=@�{A�\)C�Y�                                    Bx��T�  �          A3�@��
�\)����F{C���@��
�?�ffA  C�e                                    Bx��c�  �          A3�
@��
�=q�L�;��C�o\@��
��R@��HA�(�C���                                    Bx��r(  �          A6�H@�ff��@&ffAR�\C��
@�ff�ᙚ@�\)B
p�C��=                                    Bx����  �          A5@�33�
=>�33?�  C��R@�33� ��@��RA�p�C��\                                    Bx���t  �          A3�@ڏ\���>aG�?�\)C��@ڏ\����@�p�A��C���                                    Bx���  �          A4��@�\)�G�?��@��C��q@�\)���@��
A�33C�h�                                    Bx����  �          A4(�@�33������Ϳ   C��R@�33��R@�Q�A�
=C�t{                                    Bx���f  �          A2�\@����p�����L��C�y�@�����\@��A��C�޸                                    Bx���  �          A3
=@�=q��þ�33��=qC���@�=q��@w
=A���C�˅                                    Bx��ز  �          A4(�@��
���=���?�C��R@��
���\@�G�A�C��H                                    Bx���X  �          A1�@�����H�ٙ��(�C�s3@����G�?��A�
C���                                    Bx����  �          A0Q�@�z������p���{C���@�z���׿J=q��z�C�\)                                    Bx���  �          A/�@���33�a����C�.@��ff>�?0��C��3                                    Bx��J  �          A0z�@ڏ\�{?B�\@�G�C��
@ڏ\���@�A��HC��{                                    Bx��!�  �          A0��@����Q�@�{A��HC�U�@�������A   B;��C�5�                                    Bx��0�  �          A0��@ȣ����@?\)A|Q�C�q@ȣ���=q@ϮBp�C�`                                     Bx��?<  �          A/
=@޸R�  ��=q���C�{@޸R���
@,��Af�HC��                                     Bx��M�  �          A0Q�@�����{�z�H���
C���@�����=q@"�\AV�\C�W
                                    Bx��\�  �          A0��@����Q�@(�AR=qC�� @�����=q@��BQ�C��                                    Bx��k.  �          A/�@љ��ff@G�A@Q�C���@љ��У�@�Q�A��C���                                    Bx��y�  �          A/
=@�R��?Tz�@�33C��)@�R�׮@�Q�A�C��\                                    Bx���z  �          A-@�=q���H�c33����C��3@�=q���R����Q�C�>�                                    Bx���   �          A.�R@�p����ͿW
=��p�C��f@�p���\)@(��A`��C�T{                                    Bx����  �          A/�@�����H����9G�C�ff@����?��\@�C��)                                    Bx���l  �          A.ff@�ff��=q�<(��{\)C�˅@�ff�p�?333@n{C��R                                    Bx���  �          A-G�@�z���������Q�C�z�@�z�� Q쿙����z�C��
                                    Bx��Ѹ  |          A,��A��Å�@�����HC���A���논#�
��C�.                                    Bx���^  �          A.�H@�ff���S33��33C��@�ff��p�=�Q�>��C���                                    Bx���  �          A,Q�@�G���z���G���Q�C��@�G����Q�����C���                                    Bx����  �          A+\)A �����H��33���C�~�A ���ҏ\����Tz�C�Z�                                    Bx��P  |          A-�A���  ��G����
C��=A�����1G��mG�C�(�                                    Bx���  �          A.�\@�������{��G�C���@���G��������HC�q�                                    Bx��)�  �          A/�@����θR��=q��33C��3@����=q���H�%G�C�ff                                    Bx��8B  �          A1@�����
��p��陚C�l�@�����
����6�\C��
                                    Bx��F�  �          A333@��
�ٙ����\��p�C�*=@��
�
=��=q�=qC�3                                    Bx��U�  �          A2{@�(�������p�� �C�f@�(�����:=q�r�RC��H                                    Bx��d4  �          A1G�@��������z����C��H@����ff�U��ffC���                                    Bx��r�  �          A0Q�@�33��������C�Y�@�33��p��XQ���33C�Q�                                    Bx����  �          A.�R@�p���ff��
=��33C��R@�p�����ff�F�HC�e                                    Bx���&  �          A.�R@�\)�������C�{@�\)��(��U����HC�<)                                    Bx����  �          A*�R@ָR�P�����7p�C�@ָR��Q���Q����
C�y�                                    Bx���r  �          A+�
@����33��\�5
=C��@�����33���H���C��R                                    Bx���  �          A1�@�  ���R��{�33C��{@�  ��(��\)���RC�8R                                    Bx��ʾ  �          A0  @�(���  ��z���p�C�)@�(������-p��d��C��3                                    Bx���d  �          A,(�@�R������{��{C��=@�R�����H�P��C���                                    Bx���
  �          A'
=@�\)�����G��  C��
@�\)���R�\���RC���                                    Bx����  �          A)�@�����  ��{���C��@����p�����Y�C�e                                    Bx��V  �          A'�@�ff���
���R���HC���@�ff�\)��{�#�C���                                    Bx���  �          A((�@��\��ff������z�C�@��\�
�H�c�
��33C��{                                    Bx��"�  T          A*ff@\���
��ff�(�C��{@\����(���f=qC��                                    Bx��1H  �          A)@�33���AG�����C�˅@�33��
?\)@E�C���                                    Bx��?�  �          A*=q@׮��p��(��=�C�1�@׮���?�z�@�z�C���                                    Bx��N�  �          A*ff@ȣ���\)�Z�RC��)@ȣ��ff?��H@��HC�`                                     Bx��]:  �          A*ff@�z��	G����5�C���@�z���\)@L(�A��\C��=                                    Bx��k�  �          A*=q@��
���R?
=q@?\)C��H@��
��
=@xQ�A��C���                                    Bx��z�  �          A)�@��H�޸R�h�����
C�@ @��H��33���ÿ�C���                                    Bx���,  �          A*{@�{��{?�@7
=C���@�{�޸R@vffA�33C�q�                                    Bx����  �          A)G�@�{���?�  @�p�C��R@�{��Q�@���A�p�C��=                                    Bx���x  |          A*=qA��У�?��H@��HC��A���z�@w
=A��C�b�                                    Bx���  �          A/�@����33@R�\A�(�C�xR@�����@�(�B(�C�C�                                    Bx����  
�          A,��@����z�@��A7�
C���@����ᙚ@���A�=qC���                                    Bx���j  T          A*=q@�=q�=q@��A:�HC��3@�=q��ff@�\)A��
C�R                                    Bx���  �          A(��@����?�Q�@ϮC���@����G�@�=qA�ffC�
=                                    Bx���  �          A'
=@����{@\)AF�RC�N@������@��B  C�}q                                    Bx���\  �          A((�@���ff?�@8��C�{@�����@�A�{C���                                    Bx��  �          A&�H@'���?�Q�A.�RC���@'��   @�33B
=C��                                    Bx���  �          A&=q@�\)�	G�@b�\A�z�C�"�@�\)�ə�@��B'��C���                                    Bx��*N  T          A$z�@xQ���@�
=A��C�J=@xQ���@�z�B8�
C��R                                    Bx��8�  �          A#33@�33��ff@W�A�
=C�  @�33��z�@�=qB(�C��3                                    Bx��G�  �          A%@H�����
@��
B�C���@H����z�A
=qBk�C��
                                    Bx��V@  �          A$z�@���@�\)A�p�C�e@���
=@�B:G�C��f                                    Bx��d�  �          A$��@)������@���B(�C���@)�����HAffBq  C��{                                    Bx��s�  �          A$Q�@��\��Q�@�  A��HC���@��\���H@�BB  C�4{                                    Bx���2  �          A$(�@�����H?�p�A��C�t{@������@���A�z�C��)                                    Bx����  �          A$(�@|���(�@$z�Ao�C�"�@|������@�
=B��C��                                    Bx���~  �          A$z�@�
=��@���A�33C�>�@�
=����@�{B2=qC�޸                                    Bx���$  �          A$(�@���
�\@4z�A�  C�%@���ָR@���B��C�aH                                    Bx����  �          A#�@�(���?�=q@�\C�5�@�(����@��RA�{C�t{                                    Bx���p  �          A#33@*=q�{������  C��@*=q���@1�A�\)C�5�                                    Bx���  �          A#�@n{�z�?�z�A
=C�33@n{����@�  A�C���                                    Bx���  �          A  @��
��
=@h��A��\C�aH@��
����@��HB)\)C���                                    Bx���b  �          A��@�z��ʏ\@�  A�\)C�` @�z��tz�@�BG��C�}q                                    Bx��  �          Az�@\)���
@���B�RC���@\)�g�@��
BO�C��)                                    Bx���  �          A��@h����@��B,��C�5�@h���	��@�Q�Bo=qC���                                    Bx��#T  �          A
=@�����R��\)�W�
C�K�@����\)>�33@7�C�t{                                    Bx��1�  �          A	G�@�p�� ����(��{C��
@�p��u��hQ���Q�C�                                    Bx��@�  �          A{@��H�7���(��p�C���@��H����R�\���C���                                    Bx��OF  �          A@��
�l(��s33���C�J=@��
��p���(��Mp�C��=                                    Bx��]�  �          A�@�ff��\)��\)�
=C��{@�ff���?z�H@�G�C��q                                    Bx��l�  �          AQ�@�(����\�E���p�C�9�@�(���z�?�A4��C��3                                    Bx��{8  �          A�
@߮����z�H��p�C�+�@߮���?}p�@ָRC�+�                                    Bx����  T          A��@�p�����   �c33C���@�p����?   @_\)C���                                    Bx����  �          @��@������33����C�AH@���  <��
>��C�H                                    Bx���*  T          A z�@�\)������R��ffC�b�@�\)����=q��C���                                    Bx����  �          A��@�33������\�\Q�C�L�@�33��{>.{?���C�AH                                    Bx���v  �          Az�@陚��p���p��QG�C�b�@陚��33�#�
�k�C�33                                    Bx���  �          A
�R@��qG�� ���V�RC��R@����׾�����C�E                                    Bx����  �          A	�@��l(��
=�c\)C���@���\)���C33C�G�                                    Bx���h  �          A	��@�(��tz����k33C�P�@�(���(�����G�C���                                    Bx���  �          A
ff@�p����ÿ���IG�C�@�p���{���J=qC���                                    Bx���  �          A33@�R������.=qC�]q@�R���>#�
?�ffC�z�                                    Bx��Z  �          A��@�33��ff�����\)C��=@�33��>�33@  C��                                    Bx��+   �          A	��@�{���Ϳ�  �p�C�j=@�{���?   @W
=C��R                                    Bx��9�  �          A	�@�����녿�����HC��R@�������?!G�@�
=C���                                    Bx��HL  T          A33@��H��z�}p���G�C��\@��H��ff?:�H@��\C���                                    Bx��V�  �          A
=q@���o\)�c�
����C�q@���q�?(��@��C���                                    Bx��e�  T          Az�@�  �`  �=p���\)C��
@�  �`��?333@�\)C�Ф                                    Bx��t>  �          A	��@���aG��B�\���HC�޸@���a�?0��@��HC��3                                    Bx����  �          A	�@�Q��b�\�\(�����C��H@�Q��e?��@���C��
                                    Bx����  �          A33@����~{������  C�E@�����33?�@[�C��                                    Bx���0  �          AQ�@����\)��ff���C�` @������>�@A�C��                                    Bx����  �          A=q@�p���p��У��(��C��
@�p���\)>\)?h��C���                                    Bx���|  �          A(�@�
=��
=���E�C���@�
=��z��G��:�HC��                                     Bx���"  �          A�\@��H������F�HC�ff@��H���ͽ�G��0��C�=q                                    Bx����  �          A\)@��
�����z��p  C�5�@��
��=q��p��ffC��R                                    Bx���n  �          A33@�=q���
�-p����C���@�=q���\�J=q���HC��)                                    Bx���  �          A�R@�  ��(��z��t��C�o\@�  ��{��(��5�C��H                                    Bx���  �          AG�@�����
��33�-��C��=@������>�=q?�\C�                                    Bx��`  �          AG�@�R���
@<��A�{C���@�R�G�@�Q�A��HC�P�                                    Bx��$  �          A  @�  ���H@Dz�A���C���@�  �4z�@�Q�A���C�b�                                    Bx��2�  �          A�\@�ff��  @%�A�\)C���@�ff�J=q@��
A�\)C�+�                                    Bx��AR  �          A�\@��H���
@G�AR{C���@��H�_\)@i��A��C�J=                                    Bx��O�  �          Aff@�p����@R�\A��
C��{@�p��N{@��B{C�)                                    Bx��^�  T          A�@�ff��p�@
=qA^=qC�Ф@�ff�n{@xQ�A��HC�XR                                    Bx��mD  �          A�
@�=q���@
=qA^{C��@�=q�x��@|��AУ�C��{                                    Bx��{�  �          A(�@߮��(�?�(�A0��C�4{@߮���@i��A�C�*=                                    Bx����  �          A@ᙚ���?�ffA!��C��@ᙚ��(�@X��A�{C��3                                    Bx���6  �          Aff@ᙚ��p�?���Az�C���@ᙚ��Q�@N{A��RC�o\                                    Bx����  �          A�R@�Q��j�H@5A�ffC���@�Q���
@���A��C�s3                                    Bx����  �          AQ�@�33��G��(Q���C���@�33��  >�{@��C���                                    Bx���(  �          A��@�
=�ٙ���{�)C��H@�
=����?��@�RC���                                    Bx����  �          A�\@�G���Q쾊=q��G�C�l�@�G���(�@p�Af�HC�J=                                    Bx���t  �          A�R@�Q���\)?�G�@У�C���@�Q���z�@R�\A�C��f                                    Bx���  �          A
=@θR��33?O\)@��RC�Q�@θR���H@C33A�{C�4{                                    Bx����  �          A(�@Ǯ����?L��@��C�%@Ǯ��(�@I��A�ffC��)                                    Bx��f  �          A
=@��R��(�=#�
>�=qC���@��R��(�@!�A�{C��                                    Bx��  �          A�R@�(��Å>8Q�?�z�C�!H@�(����H@p�A�G�C�b�                                    Bx��+�  �          A�\@�����>�@G�C�Q�@������@��A���C���                                    Bx��:X  �          A�@�(����?c�
@���C��@�(�����@L(�A���C�                                      Bx��H�  �          A�@����G��#�
��G�C��@�����
@33AqC��)                                    Bx��W�  �          AG�@�ff��Q�>�{@{C�\)@�ff��@)��A��C���                                    Bx��fJ  �          A�@�
=���
>��?�
=C�O\@�
=�׮@>�RA�33C�|)                                    Bx��t�  �          A��@�����>#�
?��C��\@�����G�@2�\A��C��3                                    Bx����  �          AQ�@�������?�(�A ��C�H�@������@u�A�z�C�33                                    Bx���<  �          A�@�{��\)?h��@���C�� @�{��(�@aG�A��C�,�                                    Bx����  �          A33@B�\��{@*�HA�33C��f@B�\���@�=qB{C�                                      Bx����  �          A�
@8����=q@ ��A�=qC�#�@8�����H@�
=BQ�C�@                                     Bx���.  �          A	p�@�ff��@�RA�(�C���@�ff��33@��A��C���                                    Bx����  �          A��@��\��G�@�\Ayp�C�Z�@��\����@�33A�G�C��{                                    Bx���z  �          Az�@����G�@(Q�A�{C��\@����(�@�Q�B�C�'�                                    Bx���   �          A��@�z���33@>{A���C��\@�z���=q@��HB(�C���                                    Bx����  �          A33@�����@E�A��HC�=q@����ff@���B�
C�y�                                    Bx��l  �          A
=@�����(�@^�RAď\C�33@�����{@�\)B#ffC��H                                    Bx��  �          A��@�  �ڏ\?�=qA1�C��@�  ��G�@}p�A��HC�#�                                    Bx��$�  �          A��@^{��
=�G�����C��{@^{��Q�?�33AZ=qC�(�                                    Bx��3^  �          A��@�G����@�
=A�C�5�@�G��C33@�
=B(�\C���                                    Bx��B  �          A�
@^{������
��C��@^{��
=@1�A��C��3                                    Bx��P�  T          A
�R@������>��R@�
C��H@�������@C33A�G�C��=                                    Bx��_P  �          A\)@�G���  ?�z�@�(�C�K�@�G����H@l��A�Q�C�f                                    Bx��m�  �          A�
@���Å@C�
A���C�\)@����33@��B�
C�\)                                    Bx��|�  �          A��@�33��=q@�33B(�C���@�33�Q�@�33B3��C�                                      Bx���B  �          A�H@�z���(�@`  A�z�C��@�z���
=@��RB'�C�g�                                    Bx����  �          Aff@h����z�@=p�A��\C�^�@h����z�@�z�B�C��                                    Bx����  �          A
{@����@�=qBO�RC�  @�33AG�B��HC�c�                                    Bx���4  �          A  @Y������@��
B/�C���@Y���'�@�G�Bm�HC�/\                                    Bx����  �          A	�@vff���@���B Q�C��{@vff�B�\@��HB]p�C���                                    Bx��Ԁ  �          A(�@5����@�BBG�C��)@5���
@�  B�u�C�aH                                    Bx���&  �          A(�@=q��Q�@�G�B=qC��@=q�x��@���Bi
=C��q                                    Bx����  �          A��@R�\����@�p�BffC��\@R�\����@��BV�C���                                    Bx�� r  �          A
�\@QG��ə�@���B
�\C���@QG���p�@��BP�\C��                                    Bx��  �          A	G�@n�R��=q@�ffB��C���@n�R����@���BG��C�k�                                    Bx���  �          A  @�\)��\)@`��A�33C��@�\)���\@��B#��C��{                                    Bx��,d  �          Az�@�����
=���H�a�C�޸@�����\)?�AQG�C�b�                                    Bx��;
  �          A�@�z����\?:�H@�G�C���@�z����R@.�RA�
=C�@                                     Bx��I�  �          Az�@�  ����@ ��A��
C��f@�  ����@�
=B=qC�R                                    Bx��XV  �          A\)@b�\���H@�
=A�\)C�
@b�\���@\B>�C��                                    Bx��f�  �          A�@�����@7
=A�\)C�+�@����Q�@��B(�C�~�                                    Bx��u�  �          A{@����33?���A6=qC���@������@mp�A�G�C�ٚ                                    Bx���H  �          A�R@�����?�  @�\C�/\@����@E�A�\)C���                                    Bx����  �          Aff@��
��z�?���A$Q�C��=@��
��Q�@^�RA���C�q                                    Bx����  �          A��@��R��
=>�G�@HQ�C�}q@��R���R@��A�33C��\                                    Bx���:  �          A�@��
��Q콸Q�!G�C�h�@��
���R?�AP(�C�9�                                    Bx����  �          A ��@�p���33�B�\��
=C��3@�p���G�?�\)A z�C��                                    Bx��͆  �          @��\@�ff���ÿ@  ����C���@�ff��
=?���A�C��q                                    Bx���,  �          @�\)@�G�����\)��z�C�@ @�G����
��
=�HQ�C���                                    Bx����  �          @��@�ff��\)�E��Q�C��
@�ff���ÿ��H�.ffC�H�                                    Bx���x  �          @�@�33����@$z�A�=qC�` @�33�w
=@�(�B 33C��                                    Bx��  �          A   @�ff���H@Dz�A��RC�h�@�ff�~�R@�ffB  C���                                    Bx���  �          @�@�����Q�@�Am�C���@�����  @q�A�Q�C�ٚ                                    Bx��%j  �          @�p�@�p���33?�  A��C���@�p����\@J=qA�Q�C���                                    Bx��4  �          @�=q@�z���{>8Q�?���C�"�@�z���  @ffA�=qC�R                                    Bx��B�  �          @�  @�G����H���Z�HC�@�G��˅?��AUp�C�7
                                    Bx��Q\  �          @��
@�  ���R����Q�C�p�@�  ����?0��@��
C�E                                    Bx��`  �          @��@�=q����>L��?�p�C��3@�=q����@Q�A���C���                                    Bx��n�  �          @�G�@��R�ƸR�s33��\C��R@��R��p�?�Q�A��C��                                    Bx��}N  �          @���@��R��G�?(�@��C���@��R����@A���C�aH                                    Bx����  �          @��H@�p����@
�HA�=qC�f@�p����H@�33A�C���                                    Bx����  �          @�33@�(���\)@%�A�=qC�u�@�(�����@��HB{C��H                                    Bx���@  �          @���@�����ff@p�A�Q�C���@������\@�(�B�C�H�                                    Bx����  X          @��\@�\)��@L��AÅC�� @�\)��=q@�=qB33C��                                    Bx��ƌ  �          @���@��\��{@�A��RC��@��\���H@�ffB{C�XR                                    Bx���2  �          @���@\)��33?�p�Al��C�{@\)���H@}p�A��
C�c�                                    Bx����  �          @�\)@QG���33?��A�C�@QG�����@^{A�ffC�4{                                    Bx���~  �          @�{@h����  ?���A�HC��@h������@K�AǮC�"�                                    Bx��$  �          @���@�=q���@2�\A�ffC��R@�=q��ff@��B
ffC�xR                                    Bx���  �          @�{@�Q���
=@ ��Ar�\C��)@�Q�����@fffA�  C�q�                                    Bx��p  �          @�p�@�
=��{?�(�Ao�C�/\@�
=��  @h��A��C��f                                    Bx��-  �          @�(�@�=q��
=?�Q�Amp�C�L�@�=q����@l��A�C���                                    Bx��;�  �          @�@�����@3�
A�z�C�xR@�����@�z�B��C�#�                                    Bx��Jb  �          @�@�
=���
?�\)AJ�\C�L�@�
=�g
=@<��A�ffC��q                                    Bx��Y  �          @�  @��
����@G�Az=qC��@��
���\@l��A�
=C���                                    Bx��g�  �          @��
@�����z�?��RA;
=C���@�������@?\)A�  C���                                    Bx��vT  �          @�Q�@��H��z�?}p�@��C�h�@��H���@ ��A�z�C�,�                                    Bx����  �          @�@�����R?�G�AVffC���@�����
@UA���C�3                                    Bx����  �          @��
@���  ?޸RAT(�C�N@���p�@U�AЏ\C�Ǯ                                    Bx���F  �          @�@�
=���R@�A�(�C�g�@�
=�^�R@hQ�A�=qC���                                    Bx����  �          @�\)@������H@Mp�A���C�l�@����6ff@�33B�HC��                                    Bx����  �          @��@��
��33��  �   C���@��
��z�?�ffAIp�C��                                    Bx���8  �          @��@e���{�/\)���
C�
=@e��׮�!G���  C�                                      Bx����  �          @��@i����p��=q����C�P�@i�����
�����\)C�s3                                    Bx���  �          @�Q�@dz�����
���C��@dz����H�u��C�9�                                    Bx���*  �          @�R@Z=q���
�!G���(�C���@Z=q��33���aG�C���                                    Bx���  �          @�R@g
=��33�33���
C�L�@g
=��Q쾅�� ��C�}q                                    Bx��v  �          @��@n�R��33���g33C���@n�R���
>.{?��C�/\                                    Bx��&  �          @��
@c�
��(�� ������C���@c�
���
�
=q��  C���                                    Bx��4�  �          @�@p  ��  �0������C��@p  ���?�p�A�HC�4{                                    Bx��Ch  �          @陚@L(���(��Q���
=C���@L(���\)��G��aG�C�)                                    Bx��R  �          @�\)@P  ��=q�n{��ffC���@P  �ə�?��A\)C��                                    Bx��`�  �          @�  @>�R�Ϯ����z�C�XR@>�R�У�?p��@�RC�J=                                    Bx��oZ  �          @�{@dz����
����Q�C�  @dz�����?c�
@��
C��                                    Bx��~   �          @�33@q���ff�����	p�C���@q���\)?^�R@��HC���                                    Bx����  �          @�p�@���������n�\C���@�������G��c�
C��                                    Bx���L  �          @�z�@x�������,������C�4{@x����33�p����C�Ф                                    Bx����  �          @�@Mp���z��Tz���G�C�ff@Mp����Ϳ�G��D  C��=                                    Bx����  �          @�\)@;���p�?:�H@�(�C��\@;�����@%A�33C���                                    Bx���>  �          @�\)@(���=q@%A���C�o\@(��s33@|(�B$z�C�`                                     Bx����  �          @�ff@���ff@8��A��C��\@����@�p�B(�C�/\                                    Bx���  �          @�z�@E���?�p�AjffC��H@E���@\(�A��HC���                                    Bx���0  �          @��H@l����(�?�z�A(�C��@l����Q�@3�
A�ffC�>�                                    Bx���  �          @�  @XQ���{?���A4��C�` @XQ�����@>�RA�  C���                                    Bx��|  �          @׮@aG�����?�  AQ�C��@aG����H@*=qA�(�C�S3                                    Bx��"  �          @���@l(����>��@^�RC���@l(���
=@�A�=qC���                                    Bx��-�  �          @��
@Y����?
=@�{C��\@Y����
=@ffA�  C��\                                    Bx��<n  �          @��@j=q���R���z�HC�S3@j=q��\)?���AZ�HC��)                                    Bx��K  �          @�  @tz�����5��G�C�e@tz���ff?xQ�A��C�~�                                    Bx��Y�  �          @�{@~�R���ÿ\)��33C���@~�R��ff?��A��C��{                                    Bx��h`  �          @ȣ�@l������>�{@QG�C�=q@l�����H?��HA���C�8R                                    Bx��w  �          @��?�=q���R�|(��"ffC�˅?�=q�������C���                                    Bx����  �          @�=q@>�R����333�ə�C��R@>�R��=q?�  A\)C��                                    Bx���R  �          @Ӆ@c�
��?�Q�A%�C��q@c�
���\@/\)A�ffC�,�                                    Bx����  �          @��
@
=q��
=��  �Q�C�U�@
=q��Q�?=p�@�=qC�E                                    Bx����  �          @���?���Ǯ�fff�{C���?����\)?z�HAQ�C��3                                    Bx���D  �          @���\)��
=��z��DQ�C�Ǯ��\)��33?\)@��C��                                    Bx����  �          @�Q�?�=q���Ϳ���
=C�33?�=q����?�ffA<��C�T{                                    Bx��ݐ  �          @��@��H��Q�?xQ�A	�C�1�@��H����@z�A��HC��H                                    Bx���6  �          @��@�33��z�?���A��C�H�@�33��33@!�A�ffC��R                                    Bx����  �          @ָR@p����ff?�G�A
=C�L�@p����p�@#33A�\)C���                                    Bx��	�  �          @�{@fff��\)?�G�A.�RC���@fff���
@333A�z�C�:�                                    Bx��(  �          @�
=@H�����>�G�@s33C�Q�@H�����
@�A�  C�#�                                    Bx��&�  �          @�{@333��  ?�G�Av�RC��)@333��  @VffA�RC���                                    Bx��5t  �          @׮@^{��
=?8Q�@��C��q@^{����@A��HC���                                    Bx��D  �          @�Q�@)�����׿fff���C�T{@)������?J=q@��
C�L�                                    Bx��R�  �          @���@:=q���\���m�C���@:=q���\    <#�
C�E                                    Bx��af  �          @Ӆ@\������p����
C�\@\������?+�@�z�C��R                                    Bx��p  �          @أ�@b�\���Ϳ�ff��C�@b�\��
=?(�@���C�޸                                    Bx��~�  �          @�Q�@dz����׿��]p�C��H@dz���  =#�
>�33C�]q                                    Bx���X  �          @��@^�R��=q��(��#
=C�p�@^�R��>��H@���C�5�                                    Bx����  �          @��@G
=��녿�����C�
@G
=����?\)@���C��                                    Bx����  �          @ڏ\@7���{�ff��C�]q@7����������[�C���                                    Bx���J  �          @�ff@33���\�U��C��@33�����ٙ��k�C�g�                                    Bx����  �          @�
=@�������7���G�C���@����33�����%�C���                                    Bx��֖  �          @�@&ff��G��K���=qC�Ff@&ff������H�{\)C���                                    Bx���<  �          @��H@>�R�(Q����
�P�RC�L�@>�R�|����p�� �\C�~�                                    Bx����  T          @���@:�H�L������z��C���@:�H�У���p��iQ�C�n                                    Bx���  �          @���@"�\��=q����t
=C�@"�\�A���  �H�C��                                    Bx��.  �          @��@8������!����C�]q@8���ƸR�Tz�����C�xR                                    Bx���  �          @��@������A�����C���@����G���z��C�C���                                    Bx��.z  �          @���@G������\�����C���@G����Ϳ������C�t{                                    Bx��=   �          @���?��R������{�Q�C�N?��R����-p�����C��=                                    Bx��K�  �          @��H?�
=�������H�0�HC�\?�
=�����N{����C���                                    Bx��Zl  �          @�{?�(��z�H��=q�A�C��3?�(���33�c�
��HC�O\                                    Bx��i  �          @�{@�����������RC�t{@�����?��A\)C���                                    Bx��w�  �          @ʏ\@�Q�����\)�$=qC���@�Q��Mp��N�R� �RC���                                    Bx���^  �          @θR@����g
=�����C�t{@������
��ff�:=qC���                                    Bx���  �          @�@�����R�
=���RC���@����ff�����!C��                                    Bx����  �          @�@QG��n{�w
=��
C��@QG���(��-p���ffC���                                    Bx���P  �          @�{@�R���H�J�H��C��3@�R���׿޸R��(�C�)                                    Bx����  �          @˅@e����H����(�C�e@e��33��33���
C�=q                                    Bx��Ϝ  �          @�(�@G
=����^{���C�"�@G
=��Q������G�C��                                    Bx���B  �          @��@*�H�e�����2{C�H�@*�H��z��QG���\)C��                                    Bx����  �          @�z�@x�������&ff��z�C���@x����ff����Mp�C��\                                    Bx����  �          @���@�R���H�Z=q� =qC��3@�R��=q���R��Q�C�                                      Bx��
4  �          @��H@z���ff�Vff��p�C��)@z��������\)C��=                                    Bx���  �          @�ff@`  �����{�b=qC���@`  ��33�����C��                                    Bx��'�  �          @Ӆ@33��ff�s33�33C�>�@33��G����33C�                                    Bx��6&  �          @�33@%����\�_\)�
=C��@%����H����
C�h�                                    Bx��D�  �          @У�@s33��{��(��w\)C��f@s33��
=��Q��I��C��                                    Bx��Sr  �          @ƸR@r�\�����z��x��C���@r�\��z�\�c33C���                                    Bx��b  �          @���@�Q������&ff��{C�Z�@�Q����H��Q��T��C���                                    Bx��p�  �          @�{@C33��
=�?\)��(�C�j=@C33���H��33�o�
C�ٚ                                    Bx��d  �          @Ϯ@1G���\)�U���RC�0�@1G���{��p���  C��H                                    Bx���
  �          @�  @h����33�;��ڏ\C��
@h�����R��Q��u�C�!H                                    Bx����  �          @љ�@:=q��ff�!G���33C���@:=q��p�������C��\                                    Bx���V  �          @���@G���������  C�f@G������Q��N�RC��f                                    Bx����  �          @�p�@8Q���(���ff�l(�C���@8Q�����.{�ǮC�q                                    Bx��Ȣ  �          @�=q@333��=q��33�~�RC�xR@333��=q��\)�,��C��H                                    Bx���H  �          @�{@l�������	�����
C�C�@l����ff����#
=C��)                                    Bx����  �          @�@AG���{��
��Q�C�%@AG���(�����,  C���                                    Bx����  �          @�?��R���dz��G�C���?��R��ff��
��33C��R                                    Bx��:  �          @�������
=�^{�{C�Ή���{� ������C���                                    Bx���  �          @���$z���\)�A���\)Cr���$z����H��z��q��CuL�                                    Bx�� �  �          @�\)?^�R��(��_\)���
C�c�?^�R�ʏ\��Q����HC���                                    Bx��/,  �          @׮�����z��a���\)C������33��p���{C�G�                                    Bx��=�  �          @�33��=q�����_\)� p�C��
��=q��  ���H��p�C��                                     Bx��Lx  �          @���?������H�Dz���\C���?�������ff�^ffC��                                    Bx��[  �          @�=q?&ff��{�tz���
C��)?&ff���������C�7
                                    Bx��i�  �          @�Q�?aG����s�
�C���?aG���
=�Q���=qC�4{                                    Bx��xj  �          @˅?�=q��z��aG��
=C�C�?�=q����
=����C�g�                                    Bx���  �          @��?�=q�����J=q��{C�O\?�=q��p���p��|z�C�y�                                    Bx����  �          @�=q?�p���G��>�R��C��?�p����
�����f{C�4{                                    Bx���\  �          @�  ?�(���  �$z�����C�H�?�(���\)��
=�0Q�C�~�                                    Bx���  �          @��?k���=q�2�\��ffC��?k���33��\)�QG�C�xR                                    Bx����  T          @�p�@�
���ÿ����aG�C��{@�
��
=��Q�^�RC�u�                                    Bx���N  �          @ȣ׾8Q����\�{���\C����8Q������'
=��z�C��                                    Bx����  �          @ȣ�@\)���H��G���=qC���@\)����?uA  C��                                    Bx���  �          @�  ?@  ��Q���R���C���?@  �\��\���C��H                                    Bx���@  �          @��?����\)�Ǯ�nffC�S3?����{��G��}p�C�)                                    Bx��
�  �          @θR@&ff��=q�n{��C��@&ff���
?�@��C��{                                    Bx���  �          @�\)@����
=�}p����C�Z�@������?   @��C�B�                                    Bx��(2  �          @�\)@P  ��G��=p����HC�7
@P  ����?!G�@��C�/\                                    Bx��6�  �          @�  @a�����Q��Mp�C���@a���33?xQ�A	��C���                                    Bx��E~  �          @�G�@G���
=>�@�  C�N@G���?�{A�=qC��\                                    Bx��T$  �          @��@S�
���\?:�H@��
C�XR@S�
���@�
A���C�&f                                    Bx��b�  �          @��@`����=q��z��$z�C���@`�����?�G�A��C��                                    Bx��qp  �          @��
@c33���R�L����Q�C�#�@c33��  >�@��C��                                    Bx���  �          @�=q@P  ����u���C�|)@P  ���?���A�C���                                    Bx����  �          @ʏ\@E���Q쾳33�HQ�C���@E���?}p�A�
C��f                                    Bx���b  �          @��
@W
=��z�����
C���@W
=���H?Y��@�z�C�
                                    Bx���  �          @ʏ\@Z=q��녾���(�C�XR@Z=q��{?��A&=qC��                                    Bx����  �          @�33@^�R������Q�G�C���@^�R��p�?�
=A+�C��{                                    Bx���T  �          @�33@_\)��G������.�RC��@_\)���R?xQ�A  C��H                                    Bx����  �          @��H@fff��ff��
=�r�\C�U�@fff����?Tz�@�  C�w
                                    Bx���  �          @�=q@x����
=�k���
C��@x����(�?uA\)C�B�                                    Bx���F  T          @�z�@�=q����.{���C��{@�=q���?}p�A33C�q                                    Bx���  �          @��
@vff��녾Ǯ�b�\C��q@vff����?O\)@��C��                                     Bx���  �          @�=q@������>.{?��
C�U�@�����?��A<z�C��)                                    Bx��!8  �          @�33@~�R���>�(�@y��C���@~�R��p�?���Aep�C�7
                                    Bx��/�  �          @��@�����ͼ#�
��G�C�H@����Q�?���A#\)C�e                                    Bx��>�  T          @θR@�Q����H?�G�A5�C��=@�Q��l(�@�A�
=C�!H                                    Bx��M*  �          @ҏ\@��
����@�A���C���@��
�e@W
=A�C�L�                                    Bx��[�  �          @�Q�@p����p�@!�A�G�C�e@p���}p�@c33B33C��H                                    Bx��jv  �          @��
@C33��p�?�z�A/�C�C�@C33��Q�@�A�33C�Q�                                    Bx��y  �          @�\)@E�����@VffBG�C�Ф@E��Z=q@�  B+
=C�                                    Bx����  �          @�@.�R����@j�HB�\C���@.�R�Fff@�  B=�C���                                    Bx���h  �          @���@#33���H?z�HA�\C���@#33��
=@�A��RC���                                    Bx���  �          @Å@(���{@Q�A��C�s3@(���  @`��B(�C���                                    Bx����  
�          @Å@   ����@G�A�33C�z�@   �k�@�=qB)�
C��                                    Bx���Z  T          @�33?�33��33@W�B��C�9�?�33�l��@��HB8�C���                                    Bx���   T          @\?�=q���H@-p�A���C��H?�=q���H@s33B�\C�n                                    Bx��ߦ  �          @\@G����
@�A��\C��\@G����R@[�B	�
C�y�                                    Bx���L  �          @�  @!����?ٙ�A��C�\@!���z�@4z�A���C�N                                    Bx����  T          @�
=@
�H���H?�A]�C��@
�H��(�@%A�\)C���                                    Bx���  T          @�=q@�����?uA=qC��@���ff@Q�A�  C���                                    Bx��>  �          @�p�?Y����{@uB�HC�޸?Y���|��@��BI
=C��                                    Bx��(�  �          @ȣ�?������@AG�A��C�!H?����
=@�(�B&{C�]q                                    Bx��7�  �          @��?��R���@Q�A��C���?��R���@a�B��C��H                                    Bx��F0  �          @�33>�������@A�A�C���>������R@�{B&z�C�Ф                                    Bx��T�  T          @�G�?˅��@�A�p�C���?˅��=q@S33A�ffC��{                                    Bx��c|  �          @\@-p�����=q�Up�C�e@-p���33�L���   C��)                                    Bx��r"  �          @�@{�����Ǯ�j{C��=@{��Q쾔z��*�HC��                                    Bx����  �          @���@�
��녿�G���HC�&f@�
����>�  @ffC��                                    Bx���n  �          @�ff?333��33��p��hQ�C�g�?333��G�?n{A�C�q�                                    Bx���  �          @Å?��R��Q쿢�\�A�C�=q?��R����<�>�z�C�3                                    Bx����  T          @�@/\)��=q��  �`Q�C��q@/\)���׾�z��)��C�0�                                    Bx���`  �          @���@>{��33��=q�D(�C���@>{��Q����{C�,�                                    Bx���  �          @��@/\)���\�L�����HC��@/\)���
>�G�@��C��q                                    Bx��ج  �          @��@2�\��녿�=q�lQ�C�� @2�\���þ\�a�C�h�                                    Bx���R  T          @�  @@�����\�]p��  C���@@����\)�Q����C��R                                    Bx����  �          @ҏ\@(���(��g
=���C�B�@(������p�����C�ٚ                                    Bx���  �          @�G�?�33���R�j=q��C��f?�33��(��(�����C��3                                    Bx��D  T          @�33?������R�r�\�Q�C�w
?�������$z�����C��R                                    Bx��!�  �          @׮?��H����S�
��\)C��?��H��z�����G�C�Ǯ                                    Bx��0�  T          @��@ ���w
=��33�K�C��H@ ����p���z��(�C��                                    Bx��?6  �          @�{@5���\)����\)C�k�@5���  �<(��˅C���                                    Bx��M�  �          @���@J=q��z�����\C��@J=q��p��+����C�                                    Bx��\�  �          @ٙ�@@����  >�@xQ�C�S3@@������?�p�An�RC�˅                                    Bx��k(  �          @ٙ�@E���\)��(��l(�C��H@E���{?\(�@��C��R                                    Bx��y�  �          @�p�?У���{����#�
C���?У���  �Mp���C�@                                     Bx���t  �          @��H�.{�HQ������oQ�C�uþ.{���R����>�C���                                    Bx���  �          @�Q�?���33����0p�C�P�?���ff�Z=q���C���                                    Bx����  �          @�p�?�{���H�hQ��
=C�^�?�{����=q���C��H                                    Bx���f  �          @��?�Q������Q��4G�C��H?�Q���p��b�\�Q�C��q                                    Bx���  �          @أ�?�ff��=q�����p�C�/\?�ff���
�HQ����
C�>�                                    Bx��Ѳ  �          @�{?Ǯ��=q�n{�z�C�,�?Ǯ��\)� ������C�N                                    Bx���X  �          @���?�=q��Q��g��{C���?�=q��������C���                                    Bx����  �          @�z�?������hQ��G�C��?����ff�(����HC�:�                                    Bx����  �          @�G�?�(���z���p��*��C�Y�?�(����R�U��
=C�G�                                    Bx��J  �          @��@~{��H@4z�A�=qC�O\@~{���
@QG�B�RC���                                    Bx���  �          @�{@�����@
=qA��C�j=@�����@   A��C���                                    Bx��)�  �          @�  @��׿�ff@Dz�A�z�C���@��׿�\)@XQ�B
Q�C�g�                                    Bx��8<  �          @�
=@�G��z�@@��A�\)C�Q�@�G���33@W�A�\)C���                                    Bx��F�  �          @��@�  ���@[�B{C�%@�  ���@hQ�B��C�n                                    Bx��U�  �          @ʏ\@��\�\)?��HAW
=C��
@��\���?���A�=qC���                                    Bx��d.  �          @�G�@����(Q�?��HAV�\C�=q@�����?�(�A�(�C�˅                                    Bx��r�  �          @��@��
�/\)?B�\@�ffC��@��
�!G�?��A@(�C��                                    Bx���z  �          @�p�@���P  ?aG�@�p�C��H@���@  ?��
A^{C��                                     Bx���   �          @У�@�{�^�R?W
=@�z�C���@�{�O\)?��AZ=qC��q                                    Bx����  �          @�@����g
=?Tz�@���C�t{@����W�?�ffAW�C�^�                                    Bx���l  �          @��@���u>��H@��C�P�@���j=q?��\A/�
C��R                                    Bx���  �          @ҏ\@���s33>B�\?�Q�C�XR@���k�?xQ�A33C�Ǯ                                    Bx��ʸ  T          @���@��\�`  ?�33AE�C�Z�@��\�H��@�A��HC��q                                    Bx���^  �          @У�@�  �l��?��A�\C�h�@�  �Y��?���A�{C���                                    Bx���  �          @Ϯ@�33�~{>�(�@tz�C��@�33�s33?�p�A.�HC��                                    Bx����  �          @У�@�p�������
�6ffC�� @�p���
=?
=q@�G�C��{                                    Bx��P  �          @�ff@���w
=>���@,��C��q@���mp�?�=qA�C�                                      Bx���  �          @�=q@�\)�mp���R����C�aH@�\)���H��Q��[�C��                                    Bx��"�  �          @�{@�  �qG���ff����C��f@�  �q�>��
@>{C��R                                    Bx��1B  �          @�(�@������>�(�@�=qC��R@���xQ�?��RA<z�C��{                                    Bx��?�  �          @�ff@��R��G�@�A��C�>�@��R�q�@;�A�\)C��                                    Bx��N�  �          @�@�p��W
=��
=�qG�C�q@�p��W�>��@C��                                    Bx��]4  �          @�\)@�\)�y���8Q��\)C���@�\)�vff?��@�\)C�"�                                    Bx��k�  �          @θR@�  ��������RC��R@�  ��ff>�33@I��C���                                    Bx��z�  �          @���@���l(�@�RA�=qC��@���G�@J�HA���C��R                                    Bx���&  �          @�@�Q��<(�@p�A���C�k�@�Q��(�@0  A���C��3                                    Bx����  �          @�{@�\)�l��@33A�
=C�  @�\)�Mp�@0  A�33C��f                                    Bx���r  �          @�\)@����a�?��RA��C�k�@����Dz�@)��A�Q�C�Ff                                    Bx���  �          @�33@���n{?�G�Ax  C�  @���R�\@p�A�p�C���                                    Bx��þ  �          @ҏ\@�Q��\��?uA{C��R@�Q��L��?�\)Adz�C��{                                    Bx���d  �          @�z�@���E?+�@���C��)@���9��?��\A0z�C��q                                    Bx���
  �          @���@�
=�z�����ffC�� @�
=��
>��R@+�C�Ф                                    Bx���  �          @��
@�Q��G������=p�C�9�@�Q�޸R���
��C�5�                                    Bx���V  �          @�ff@��
�;��k��G�C���@��
�:�H>�33@C33C��H                                    Bx���  �          @���@��Q녿   ���C��q@��Tz�>��?�=qC��)                                    Bx���  �          @�Q�@�(��S�
�u��\C�S3@�(��R�\>��@`  C�c�                                    Bx��*H  �          @�G�@���^{?�
=A!��C�9�@���K�?�A}�C�XR                                    Bx��8�  �          @�  @��\�g
=���
�8Q�C���@��\�c�
?�R@�(�C��q                                    Bx��G�  
�          @��H@�
=�h��>Ǯ@Y��C�/\@�
=�_\)?��A��C��R                                    Bx��V:  �          @��H@���녾�����C���@�����?�@��C�                                    Bx��d�  �          @��H@�p��~{�c�
���C�9�@�p���=q���ͿY��C���                                    Bx��s�  �          @���@��
�xQ쿏\)��C�k�@��
���þ�33�A�C��=                                    Bx���,  �          @�\)@��R�qG���33�l��C�]q@��R���׿aG�����C�|)                                    Bx����  T          @�  @�33���\�n{�=qC��R@�33������z�C��q                                    Bx���x  T          @�ff@�p��n�R��������C�p�@�p���Q쿈����C�l�                                    Bx���  �          @��@�33�@���(���
=C��R@�33�Z�H������C�\                                    Bx����  �          @�\)@�G��u����C���@�G����
�����{C���                                    Bx���j  T          @ʏ\@�  �L(���\��=qC��
@�  �aG�����M��C�ff                                    Bx���  T          @У�@������
=#�
>�{C��f@�������?^�R@���C��                                    Bx���  �          @���@����y����\���C�@�����p��}p��(�C�{                                    Bx���\  �          @�ff@�  ���þu���C��3@�  ��\)?.{@��HC��{                                    Bx��  �          @�ff@�\)��녿8Q����C�<)@�\)���
>.{?�(�C��                                    Bx���  �          @˅@�G���Q쿈���(�C�b�@�G���z�u�
�HC��R                                    Bx��#N  �          @�z�@6ff�������\�a��C��q@6ff�0  �����F=qC�H                                    Bx��1�  
�          @��@;��Q���z��[\)C�  @;��C�
��z��>\)C��R                                    Bx��@�  
�          @�(�@u��b�\��=q��C���@u���  �U���C��                                    Bx��O@  �          @��@o\)�l(��z=q��\C���@o\)���
�H����
=C�H�                                    Bx��]�  �          @��H@x����G��.{��C�G�@x����\)����=qC�H                                    Bx��l�  �          @У�@l���\)�c�
��HC�j=@l�����H�/\)��C�n                                    Bx��{2  
�          @ٙ�@�33��Q��<������C�(�@�33��\)�33����C��)                                    Bx����  �          @�ff@z�H���ÿ�{�iC��{@z�H����.{��z�C�{                                    Bx���~  �          @�{@w���33�Fff���C���@w��������C�                                    Bx���$  �          @�ff@��H�w
=�J=q��\C�T{@��H��(��Q���
=C��H                                    Bx����  �          @˅@��R��p��
�H���C��H@��R��Q쿬���F{C��f                                    Bx���p  �          @Ϯ@������Ϳ����C�@�����{��{�33C��                                    Bx���  �          @�=q@�33�xQ��ff���C�!H@�33��  ��=q�i�C��{                                    Bx���  �          @��\@���U��G��H��C��@���`�׿(���=qC�1�                                    Bx���b  T          @�@k��`�׿�  ��G�C�0�@k��n�R�Q���C�S3                                    Bx���  �          @���@i����=q�%����C���@i�����������C���                                    Bx���  �          @�{@�\)�p���   ��z�C�*=@�\)�����  ��C���                                    Bx��T  �          @�@QG��O\)�P  ��
C��)@QG��r�\�%���HC�b�                                    Bx��*�  �          @�  @\���<(��^�R�{C���@\���b�\�8Q���C�&f                                    Bx��9�  �          @�  @`���/\)���\�0��C�@`���_\)�o\)�p�C���                                    Bx��HF  �          @ƸR@8Q��p���p��c��C�~�@8Q��*=q��Q��J(�C��                                    Bx��V�  �          @��
@   �c�
�����C��@   ��=q�����t33C��)                                    Bx��e�  �          @�?�\)�u����� C�b�?�\)����Q��wQ�C�                                      Bx��t8  �          @�p�?��H�.{��\)#�C���?��H��=q����\)C�^�                                    Bx����  �          @�Q�?h��>�(����� ��A�z�?h�ÿ+���(�Q�C���                                    Bx����  �          @�?L��?����\)u�Bh��?L��>.{���
¤�qA?
=                                    Bx���*  �          @�33?�{?=p����G�Bz�?�{��33��z���C�J=                                    Bx����  �          @�p�?^�R?���\)�
A�  ?^�R�
=q��
=�3C�4{                                    Bx���v  �          @��?�{�s33��33k�C�l�?�{��\)���H�uz�C�ff                                    Bx���  �          @�ff@z��  ���oC�� @z��%������QQ�C�\)                                    Bx����  �          @�\)@ �׿�(����\�{(�C���@ ������\)�^Q�C�c�                                    Bx���h  �          @�  ?����R��{��C��f?���  ����{C��                                    Bx���  �          @��>Ǯ?fff��{z�B�>Ǯ��G���Q�«�\C�c�                                    Bx���  �          @���@3�
��\�s33�9�C�l�@3�
�<(��Tz��  C��{                                    Bx��Z  �          @�G�@\���vff�'����C��@\����Q��\)��
=C�}q                                    Bx��$   �          @��@aG��j�H�p���z�C���@aG���녿�  ��Q�C�z�                                    Bx��2�  �          @���@q��g
=�
=q���
C�+�@q��|(���(��p��C��                                    Bx��AL  �          @�G�@qG��o\)��=q����C��
@qG��~{�^�R��C��                                     Bx��O�  �          @��@^�R�h�ÿ�=q��C�ٚ@^�R�z=q����D(�C��3                                    Bx��^�  �          @��@^{�i��������
C�Ǯ@^{�x�ÿs33�$z�C��                                    Bx��m>  �          @���>�ff���R����
C�aH>�ff�1����
�^��C��R                                    Bx��{�  �          @�=q@J�H�ff�b�\�'��C��3@J�H�<���C�
���C�|)                                    Bx����  �          @�z�@c33�(Q��8����\C���@c33�G
=�
=�ԏ\C�c�                                    Bx���0  �          @�p�@C�
�\)�dz��-  C��@C�
�5�Fff�\)C��\                                    Bx����  �          @�G�@P�׿�G��xQ��<�C��=@P���(��b�\�(�C��                                    Bx���|  �          @�\)@Fff�0���c�
�"G�C�,�@Fff�W
=�@  �
=C�`                                     Bx���"  T          @�
=@P  �j�H�G���=qC�@P  ���׿Ǯ����C�z�                                    Bx����  �          @�Q�@c�
�l(���z���
=C���@c�
�~{���H�L(�C��                                    Bx���n  T          @���@N�R�\)�p���Q�C�xR@N�R���\�����qG�C�\)                                    Bx���  �          @�
=@|(��~�R�Q���ffC�` @|(���G������Tz�C�Ff                                    Bx����  �          @��@p  ��33��Q��5G�C��q@p  �����{�O\)C�1�                                    Bx��`  �          @�=q@w
=�����\)�!G�C���@w
=���H?E�@�\C��                                    Bx��  �          @��@n{���׿�ff�G�C��R@n{��(��.{��\)C���                                    Bx��+�  �          @�p�@j�H���
�&ff��(�C�� @j�H����>k�@	��C�c�                                    Bx��:R  �          @��H@XQ���=q>B�\?�C���@XQ���ff?���A&=qC�*=                                    Bx��H�  �          @�Q�@Z=q���k���RC�S3@Z=q����?(��@ȣ�C�n                                    Bx��W�  �          @�Q�@Tz���Q켣�
�W
=C�@Tz���?aG�A=qC��
                                    Bx��fD  T          @�  @b�\���H��G����\C�R@b�\����?B�\@陚C�C�                                    Bx��t�  �          @\@fff��33�����\)C�J=@fff���>���@tz�C�Ff                                    Bx����  T          @�=q@hQ���p��!G����C���@hQ���ff>aG�@ffC���                                    Bx���6  T          @���@�(��o\)������C���@�(��~�R�z�H�{C��                                    Bx����  �          @��@�ff�vff��p��i��C��@�ff�����E���G�C��                                    Bx����  �          @�\)@�33���R�z�H�\)C��@�33��=q�W
=��C���                                    Bx���(  �          @�{@X����녽L�;�C���@X����\)?Y��A�C��                                    Bx����  �          @��@Vff���H=�Q�?G�C�@Vff���?��A�C�N                                    Bx���t  �          @Ǯ@W���\)��=q�!G�C�ff@W���ff?(��@��
C�|)                                    Bx���  �          @˅@QG���z�8Q���  C��q@QG���>�  @33C��                                    Bx����  �          @���@Vff��\)�c�
�C�T{@Vff���=u?\)C�&f                                    Bx��f  �          @�@c33���?�@���C�c�@c33��{?�AR�RC��)                                    Bx��  �          @Ǯ@:=q���?xQ�A��C�AH@:=q��33?�z�A�=qC��)                                    Bx��$�  "          @��@qG���{?�p�Ac�C�33@qG���33@\)A�\)C�P�                                    Bx��3X  �          @�Q�@i����Q�?�Q�A��RC�K�@i���vff@*�HA�C��                                     Bx��A�  �          @�  @l�����H@�A�
=C��@l���g
=@AG�A�=qC��R                                    Bx��P�  �          @�p�@l���p��@*�HA�z�C�C�@l���N�R@R�\BffC�t{                                    Bx��_J  �          @�p�@����aG�?�(�Af{C��@����L��@z�A�  C�W
                                    Bx��m�  �          @�=q@��`  ?�33A^=qC��q@��L(�@   A��\C�q                                    Bx��|�  �          @�=q@���c33?��
A#\)C��3@���S�
?�33A��C��=                                    Bx���<  �          @��@��H�h��?J=q@�p�C���@��H�\(�?�A^�HC��\                                    Bx����  �          @��H@���e>��@�z�C��f@���\(�?���A.�\C�t{                                    Bx����  �          @��@��R�P��?E�@���C��@��R�Dz�?��AV{C�o\                                    Bx���.  �          @��\@�=q�]p�?O\)Ap�C�k�@�=q�P��?�Ab�\C�<)                                    Bx����  �          @��
@����O\)>��@���C�J=@����Fff?��
A*{C���                                    Bx���z  �          @��\@�G��4z�>�\)@6ffC�� @�G��.�R?E�@��C�+�                                    Bx���   �          @�
=@���,��>�33@g�C�'�@���&ff?Q�A	�C���                                    Bx����  �          @�  @�{�6ff>���@�p�C�` @�{�.�R?fffAffC��f                                    Bx�� l  �          @���@����A�?G�A�\C�/\@����5?��A[\)C��                                    Bx��  �          @���@����QG�?#�
@ҏ\C��@����G
=?��HAG�C��                                    Bx���  �          @�p�@�33�l��?��
Apz�C���@�33�W�@
�HA��
C�!H                                    Bx��,^  �          @�\)@��\�Q�>�p�@n{C�/\@��\�J=q?p��AC���                                    Bx��;              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��I�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��XP              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx��f�   �          @��H?�(���p�@ ��A��C��R?�(���\)@>�RA���C�Y�                                    Bx��u�  �          @��@����p�׾aG��z�C��\@����o\)>�ff@�(�C���                                    Bx���B  �          @��H@��
�'�����4z�C�<)@��
�'�>W
=@�RC�5�                                    Bx����  �          @�=q@�z��!�>��?��HC�f@�z��p�?��@أ�C�Z�                                    Bx����  �          @���@����Mp�?h��A��C�#�@����@  ?�(�A�C�{                                    Bx���4  T          @�(�@p  ��{?+�@�33C��@p  ��  ?���Ae��C��R                                    Bx����  �          @�(�@p����Q�>�z�@2�\C��f@p����z�?�=qA)��C�H�                                    Bx��̀  �          @��@�=q��ff=�?��HC��@�=q���?^�RAQ�C�^�                                    Bx���&  �          @��R@aG�����>��R@<(�C�/\@aG�����?�33A2ffC��\                                    Bx����  �          @�G�@j�H����>�?��
C���@j�H��p�?z�HAC�R                                    Bx���r  �          @�{@R�\��z�   ��p�C�L�@R�\����>�
=@|��C�H�                                    Bx��  �          @�ff@^�R��33�333����C�U�@^�R���=��
?^�RC�(�                                    Bx���  �          @�z�@���g
=?�33A=G�C�Z�@���Vff?�\A���C�j=                                    Bx��%d  �          @��@`���Q�@'
=A�=qC�y�@`���0��@I��B�
C��                                    Bx��4
  �          @b�\@{��33?��
A�\)C�1�@{�Ǯ@B(�C��                                    Bx��B�  �          @U�?�=q��@G�B-\)C��f?�=q����@#33BJp�C�j=                                    Bx��QV  T          @W
=?�=q��@
=B�\C�H?�=q��\)@(�B<z�C�0�                                    Bx��_�  �          @Q�?��Ϳ���?���B��C���?��Ϳ˅@
�HB/\)C���                                    Bx��n�  �          @I��@!녿�33?�A�  C�}q@!녿�?�33A�G�C��\                                    Bx��}H  �          @Mp�@%���\?���A�G�C���@%��G�?ǮA�z�C�O\                                    Bx����  �          @O\)@����?�  Aޣ�C�h�@�����?�\Bz�C��R                                    Bx����  �          @O\)@�Ϳ�Q�?�=qA�
=C�L�@�Ϳ�?�33A�33C�b�                                    Bx���:  �          @L��@�׿��?���B�C�s3@�׿0��@ffB&C�t{                                    Bx����  �          @R�\@���\)?�\)A�  C���@����?��B�HC�Y�                                    Bx��Ɔ  �          @U�?��H�
�H?���A��
C��?��H����?��HBQ�C�U�                                    Bx���,  �          @N�R?�{��\?�  A�G�C���?�{���R?��B�C�|)                                    Bx����  �          @L(�?�ff�8Q�>��@�C��q?�ff�0��?k�A�  C�b�                                    Bx���x  
Y          @N{?�\)�,��?���A�z�C�@ ?�\)��H?��
B�C�ff                                    Bx��  
�          @O\)?��p�?�ffA�ffC�H?����?��HB=qC���                                    Bx���  "          @QG�?0���-p�?��A�RC�(�?0�����@   B{C��                                    Bx��j  �          @O\)?Y���   ?�33B 33C�` ?Y���
�H@�B&��C��{                                    Bx��-  "          @K�?�\)���H?��A��C��\?�\)���?�(�B�\C�g�                                    Bx��;�  �          @I��@녿��?�(�A�p�C���@녿˅?��
B
��C��                                    Bx��J\  #          @Fff?�33�   ?�\)A��C��)?�33��p�?��HB�C�޸                                    Bx��Y  �          @J�H?���?��AѮC�P�?���ff?޸RBG�C�g�                                    Bx��g�  �          @I��?�{���?���A�G�C���?�{��\)?��HB�C�|)                                    Bx��vN  �          @H��?���33?���A���C���?���   ?�B  C��3                                    Bx����  �          @K�?�33��?�\B	Q�C�+�?�33��(�@
=qB-C�'�                                    Bx����  �          @J�H?@  �z�?��RB�\C���?@  ��
=@�BC��C���                                    Bx���@  �          @U��\(����@�B(G�Cy&f�\(���=q@#�
BNp�Ct�\                                    Bx����  "          @K�?&ff�Q�?��RB��C���?&ff��p�@��BC�C��                                    Bx����  �          @J�H?�����H@
=B133C�?����ff@Q�BPp�C��)                                    Bx���2  �          @O\)?�녿���@�HBK��C��\?�녿��@*�HBk\)C��q                                    Bx����  T          @XQ�?�R�ff@ffB/�HC�g�?�R���@/\)BWC��                                    Bx���~  T          @U�>��$z�@ ��B�C�O\>��
=q@(�B?=qC�Ff                                    Bx���$  �          @O\)>8Q��>�R?�  A�33C�� >8Q��-p�?�\B�C���                                    Bx���  �          @W
=?
=�<(�?�p�A��
C��f?
=�(Q�?��RB�C�O\                                    Bx��p  �          @XQ�?   �H��?���A��RC���?   �8Q�?�
=A���C��f                                    Bx��&  �          @S�
?&ff�:=q?�Q�A�\)C�P�?&ff�'
=?�Q�BG�C��                                    Bx��4�  T          @L(�?!G��@  ?=p�AZ�\C���?!G��3�
?��
A��HC�S3                                    Bx��Cb  �          @Q�?^�R�Fff>.{@B�\C���?^�R�@��?:�HAQG�C�\                                    Bx��R  �          @N{?J=q�C33�\)��C�Ff?J=q�Fff<#�
>��C�(�                                    Bx��`�  �          @L��?�\�Fff��\�z�C��{?�\�I��=u?��C��                                    Bx��oT  �          @Fff>u�9����ff��G�C�T{>u�B�\��\��C�8R                                    Bx��}�  �          @C�
>��5���\��Q�C���>��>�R���H�
=C�c�                                    Bx����  T          @E�>��1녿�  ��{C���>��:=q���ffC�p�                                    Bx���F  �          @C�
?aG��8Q�.{�S33C�}q?aG��7�>�33@׮C���                                    Bx����  �          @K�@
=���?�
=B��C�S3@
=��G�@	��B*ffC�.                                    Bx����  
�          @J�H?�(�����?��B33C�j=?�(���p�@��B*��C��
                                    Bx���8  �          @K�?��(�?�{A�\)C�  ?���z�?޸RB
=C��)                                    Bx����  �          @K�?����?�(�Aݙ�C�
=?�׿��
?�=qB  C�L�                                    Bx���  "          @G
=?����\?�\A33C�?���
=q?k�A�{C���                                    Bx���*  "          @J�H@�G�?+�AE�C�H�@�
=?�=qA��
C�XR                                    Bx���  
�          @I��@z�� ��?.{AF�HC���@z����?��
A��C���                                    Bx��v  �          @G�@
=��33?:�HAY�C��R@
=��p�?���A���C�޸                                    Bx��  T          @Dz�@����?!G�A>�HC�s3@��޸R?xQ�A��C���                                    Bx��-�  T          @>�R@\)��{?
=A7�C�%@\)��(�?k�A��HC�>�                                    Bx��<h  �          @Fff@Q����?#�
A?\)C�
=@Q�ٙ�?xQ�A��\C�1�                                    Bx��K  �          @@  @ �׿�
=?�{A�  C�` @ �׿u?��A�p�C���                                    Bx��Y�  
�          @G�@ff��{?��
A�p�C��\@ff���?�ffA���C��                                    Bx��hZ  T          @J�H@p����H?���A��C�AH@p���(�?�(�A�(�C�q                                    