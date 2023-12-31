CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230418000000_e20230418235959_p20230419021721_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-19T02:17:21.978Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-18T00:00:00.000Z   time_coverage_end         2023-04-18T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxy��  �          A����@�z�8Q쿠  C�\���@�ff@2�\A��RC
h�                                    Bxy�f  �          Az����
@�녾�{�ffCG����
@�ff@'
=A��C
p�                                    Bxy�  �          A����@�  >W
=?�z�C�����@�33@J�HA���C
�3                                    Bxy��  "          A�H����@�(��p����{C	�����@�33?��RA\  C
��                                    Bxy�X  
�          AG�����@�녿��}p�C�H����@�  @)��A�=qC��                                    Bxy��  	�          A���  @��?p��@ҏ\B�.��  @���@�Q�A�G�B�#�                                    Bxy�  T          A���Q�@�@G�A~ffC
=��Q�@�\)@���BG�C�
                                    Bxy�J  "          Ap����@أ�?���A!��B�\���@���@�(�Bp�C�                                    Bxy�  �          A����z�@�G�@�Ah(�B�����z�@�z�@��B�C�R                                    Bxy�  
Z          A  ����@�?fff@�B�R����@��\@�(�A�  B���                                    Bxy&<  
�          A(��N{@�  ?�
=A z�B���N{@�z�@�(�B�B�\)                                    Bxy4�  
�          A�
��{@љ�@G�A]�B���{@�@�Q�B��C��                                    BxyC�  T          A�����@�(�@��A��RC �R���@��H@�  B(�C��                                    BxyR.  
�          A	G����H@ȣ�?�p�A:{C�����H@��H@��
BG�C��                                    Bxy`�  T          A
=���@��8Q쿚�HB��=���@Å@R�\A�\)C @                                     Bxyoz  "          A
=��{@��<��
>#�
C)��{@��
@=p�A�33C5�                                    Bxy~   �          A\)���@��H���
=B�z����@�z�@
=qAl��B�
=                                    Bxy��  "          A33��p�@�׿�33���B�u���p�@�G�@G�Ay�B�33                                    Bxy�l  �          Az���p�@�  �z�H��z�C=q��p�@�p�@\)At��C�\                                    Bxy�  �          AQ����@�\)��z��3�B�=q���@�p�?�33ANffB�                                    Bxy��  "          A
=����@�  �   ����B�.����@�(�?�ff@�B�{                                    Bxy�^  �          A�\��Q�@�  �G��aG�B����Q�@���?��HA!B�G�                                    Bxy�  T          Ap��J=q@�(�����o33B�\)�J=q@�\)?��AF�RB��
                                    Bxy�  �          A33���@�녿�(�� ��B�aH���@��
@p�As\)B��
                                    Bxy�P  "          A
=�q�@�=q��(��<Q�B���q�@�
=@Q�Ai�B�W
                                    Bxy�  �          Aff�~{@�Q��2�\���HB����~{@�  ?h��@���B�W
                                    Bxy�  
�          A=q�z=q@ڏ\�7���=qB�\�z=q@�?aG�@�Q�B���                                    BxyB  �          A�R�z�H@׮�G����\B�aH�z�H@���?(�@��B���                                    Bxy-�  
�          A��u�@�R��=q��B�ff�u�@��@/\)A�
=B�B�                                    Bxy<�  �          A  �*=q@���h����p�B����*=q@���@I��A�ffBۅ                                    BxyK4  �          A�
�C33@�G������\B��C33@�
=@(Q�A��\B�u�                                    BxyY�  �          Aff�hQ�@��#�
��(�B�\�hQ�@�{?��\A�
B��                                    Bxyh�  "          A�H�`��@�\)��  �$��B�W
�`��@�  @�A�33B�R                                    Bxyw&  "          Az��y��@�Q��(Q�����B���y��@�?��@�B�                                    Bxy��  
�          A��w�@����33��
=B��w�@�\)��R���B�W
                                    Bxy�r  "          Ap��x��@����hQ��֣�B����x��@�G��L�Ϳ�B��H                                    Bxy�  T          A (��n{@���g
=�ڣ�B�\)�n{@��u��\B�                                     Bxy��  
�          A Q��S�
@�\)�xQ���z�B��S�
@���G��O\)B�                                      Bxy�d  "          A�\��
=@���  �,��C�H��
=@��5��Q�B��{                                    Bxy�
  "          A�
�hQ�@�=q��p����B����hQ�@���Q����B�Ǯ                                    Bxyݰ  "          A���w�@�{����(�B�
=�w�@�=q�Tz���33B�.                                    Bxy�V  �          A
{��(�@ȣ������홚B��R��(�@�녿+���p�B�B�                                    Bxy��  T          A	�H��@��e���z�B�G��H��A (�>Ǯ@%B���                                    Bxy	�  T          A
=q�U@��W���ffB����U@��?�@r�\B��)                                    BxyH  T          A
=�;�@ᙚ��33���
B�(��;�A�\���aG�Bۊ=                                    Bxy&�  �          A\)�@  @�(��Z=q����B�B��@  A��?�R@��
Bܞ�                                    Bxy5�  �          A�
�e�@�33�E���\)B���e�@�p�?c�
@�(�B䞸                                    BxyD:  "          A  �B�\@��H�ff�yp�B�p��B�\@�\)?�\A;�BݸR                                    BxyR�  T          Az��\Ap���G�� ��B���\A Q�@0  A��B���                                    Bxya�  T          A녿�=qA\)��{��ffB�(���=qAff@Q�A�B��                                    Bxyp,  �          A�R��ffA	���\���B�.��ffA=q@E�A�  B�(�                                    Bxy~�  
�          A{���A
ff    <#�
B�z���@��@��RA��B��f                                    Bxy�x  T          A  ��{A	G�?�@^{B�G���{@�@�(�A��B��                                    Bxy�  T          A
�R��p�A�\?�=q@�
=B��ÿ�p�@�=q@���B
=B�u�                                    Bxy��  T          A�
�W
=A=q��z����B�k��W
=@�
=@tz�AׅB���                                    Bxy�j  T          A33�!�A Q콸Q�(�B�{�!�@ᙚ@s�
Aי�Bۅ                                    Bxy�  S          A�
�Mp�@�녾��H�W
=B�q�Mp�@�=q@VffA���B��H                                    Bxyֶ  
�          Az῵A���  ���Bǅ��@���@8��A�(�Bȏ\                                    Bxy�\  "          A�
�+�Aff��
�b{B�ff�+�A{@��Ak�B�k�                                    Bxy�  "          A����@�(��,�����B����A33?���A=qB�B�                                    Bxy�  T          A�\��Q�@�z��Q���B���Q�Az�?Tz�@�z�B�k�                                    BxyN  
(          A�H��ff@��S33��33B�W
��ffA(�?J=q@�(�B��)                                    Bxy�  �          A	G��4z�@�{�=p���  B���4z�@��@L(�A�33B�33                                    Bxy.�  �          A(����A=q�p���˅B�G����@�\@FffA��B�ff                                    Bxy=@  �          A����A�R������
B�ff��@���@mp�AΏ\B��                                    BxyK�  {          A\)�+�A�׾�ff�Dz�B�8R�+�@�\)@fffA��B�8R                                    BxyZ�  
�          A�
�Y��A
=>#�
?���B�� �Y��@�=q@�ffA�B�=q                                    Bxyi2  "          Azῑ�A
=>\@#33B�\)���@�
=@�z�A�{B��                                    Bxyw�  T          A��h��Aff>�z�?�p�B�\)�h��@�
=@�G�A�=qB�W
                                    Bxy�~  �          A�
���
A�?   @\(�B�ff���
@�33@��RB �B�.                                    Bxy�$  �          A  ?�(�A (�@%�A�33B��3?�(�@�G�@ÅB7��B�=q                                    Bxy��  T          A��.{A��?�=q@�\B�\)�.{@��@�ffB��B��f                                    Bxy�p  T          AQ�8Q�A\)�Ǯ�(��B�Ǯ�8Q�@�@mp�A��B��)                                    Bxy�  T          A  �s33A�R�+����B��s33@��R@\��A�p�B��                                    Bxyϼ  
�          Aff��\)A�������B��f��\)@�G�@s33A��HB�B�                                    Bxy�b  �          AG��Tz�A\)?J=q@��RB����Tz�@ڏ\@��
Bp�B��                                    Bxy�  
�          Aff��Q�A   ?�(�A=�B��
��Q�@�ff@��\B��B�p�                                    Bxy��  
Z          A(���p�A�#�
����B�\��p�@�p�@vffA��
Bʊ=                                    Bxy
T  "          A���(�@�녿�����Bֽq�(�@�(�@1G�A���B؞�                                    Bxy�  
�          A�ÿ�\)A �Ϳ@  ��ffB�.��\)@�@J�HA�  B�B�                                    Bxy'�  �          A녿�Q�A{���Q�B����Q�@�R@qG�A�Q�B��                                    Bxy6F  
�          Ap���A������3�
B�LͿ�@�=q@`  A��B͸R                                    BxyD�  �          A�ÿ}p�A�����Q�B�  �}p�@�
=@,��A�G�B£�                                    BxyS�  "          A��>aG�@�{����=qB�Ǯ>aG�A�?�\AEG�B��
                                    Bxyb8  T          Az�=L��@�{����w�B�G�=L��A z�?�AN{B�G�                                    Bxyp�  �          A�ÿ
=A녿У��4��B�Q�
=@�p�@�HA��B��                                    Bxy�  
e          A�;�
=@�=q�,(����B�33��
=A�\?��A��B��                                    Bxy�*  �          Az�aG�@��H�'���B�B��aG�Aff?�z�A��B�#�                                    Bxy��            A��=�\)A (����m��B���=�\)A ��?�AV=qB�                                      Bxy�v  -          A(���@���33�f�RB�\)��A (�?�(�A\��B�\)                                    Bxy�  T          A33>��H@���5���\B���>��HA��?�{@��B�
=                                    Bxy��  T          AQ쿯\)A �Ϳ\(���G�B�aH��\)@�Q�@B�\A��BȽq                                    Bxy�h  
�          A\)?W
=@�z��4z���
=B�ff?W
=A�?�\)@�33B�\                                    Bxy�  T          A�?�ff@�\)�8Q���  B�L�?�ff@�ff?}p�@߮B�p�                                    Bxy��  �          A�R?�R@�
=�#33���
B�Ǯ?�RA (�?���AQ�B��                                    Bxy Z  
�          A녿�\A   ��z��  B�W
��\@�33@/\)A�  B��R                                    Bxy    {          A �ÿ���@�  ?aG�@�z�B�G�����@�@�p�B�HB�#�                                    Bxy  �  
�          @��R��\)@�  ?}p�@��B��ÿ�\)@�{@���B	��B�                                      Bxy /L  T          @��R�@���Q���p�Bը��@���>�33@\)Bҙ�                                    Bxy =�  �          AG��Q�@���;���G�B�{�Q�@�ff?E�@�  B��H                                    Bxy L�  T          A����
@�=q�Vff���
B�u���
@�=q>�33@p�Bр                                     Bxy [>  �          A����{@�(��vff��33B�8R��{@�(��B�\����B΅                                    Bxy i�  �          A��G�@ۅ��(���=qBĮ��G�@����G��J=qB�W
                                    Bxy x�  �          AG��s33@�  �z�H��=qBÅ�s33A z�L�Ϳ�z�B��\                                    Bxy �0  �          A �׿���@أ���(���33B��Ῠ��@�����H�`��B��)                                    Bxy ��  T          A Q쿙��@����
=��\)B�W
����@����R���B�L�                                    Bxy �|  T          A ���>{@�\�
=���\B�\�>{@�?��HA
�RB���                                    Bxy �"  T          A (��aG�@�����s33B�aH�aG�@��H@7
=A�B�B�                                    Bxy ��  �          A   �y��@�
=��p��*=qB�k��y��@��H@:=qA��HB��                                    Bxy �n  
�          @�\)�|��@�p�>��
@�B�k��|��@�\)@_\)A��B��)                                    Bxy �  "          @�
=��\)@�
=?333@���B�p���\)@��
@o\)A߮B��                                    Bxy ��  
�          @���vff@�
=?
=@�ffB����vff@��@p  A�{B�.                                    Bxy �`  
�          A z��r�\@ᙚ?�@��B�z��r�\@��@qG�A�{B���                                    Bxy!  �          A   �]p�@�{?��@{�B�ff�]p�@��
@s�
A�B��                                    Bxy!�  T          @�\)�S33@�\)?�R@�{B�{�S33@�(�@x��A陚B잸                                    Bxy!(R  T          A (��mp�@�{?�ff@��HB�=q�mp�@�p�@�(�A�
=B�\)                                    Bxy!6�  T          A   �~�R@أ�?�(�A�B��H�~�R@�{@�ffA��B�L�                                    Bxy!E�  �          @��
��(�@�
=@z�A��B����(�@��R@�p�B(�C��                                    Bxy!TD  
�          @��H����@<(�@�p�B&G�C������>k�@�{BF�
C1aH                                    Bxy!b�  _          @���G�@vff@���B��C+���G�?�{@�\)BN�HC&��                                    Bxy!q�  
�          @�(����\@��R@���B
=C	aH���\?�ff@�p�BO�
C!Q�                                    Bxy!�6  T          @����{@��@z�HAC���{@2�\@�33B@�C޸                                    Bxy!��  "          @�z���G�@�Q�@Av=qB��H��G�@��@�\)B��C�q                                    Bxy!��  T          A=q�Z�H@�(�=��
?�B�W
�Z�H@׮@dz�A�  B��
                                    Bxy!�(  
�          A  �u�@���G����B���u�@�
=@
�HAt  B홚                                    Bxy!��  
�          A33�hQ�@����QG�B��hQ�@߮@C33A�G�B��                                    Bxy!�t  �          A33�S33@��������B�8R�S33@��@@  A�ffB�                                     Bxy!�  T          A��Dz�@�(�<#�
=#�
Bޞ��Dz�@�Q�@fffA�p�B�W
                                    Bxy!��  
�          A	�<(�A Q�?�=q@�B�G��<(�@���@��B�B�q                                    Bxy!�f  
�          A	�G
=A (���(��6ffB�k��G
=@�\@P��A�ffB���                                    Bxy"  
Z          A	��=p�A ��=�Q�?�RB�u��=p�@��
@o\)A��HB�33                                    Bxy"�  
�          A	G��7
=A ��?5@�
=B�{�7
=@�33@���A��B�G�                                    Bxy"!X  �          AQ��"�\A ��?�\)@�\B���"�\@��@���Bz�B���                                    Bxy"/�  "          A�\��H@���?��RA#33B����H@��
@��B  Bݨ�                                    Bxy">�  �          A�R��RA Q�?��@�
=B�\��R@��@�33B
=B�                                    Bxy"MJ  T          A�Ϳ��HA   ?�  A(z�B�(����H@�ff@��RB=qB�33                                    Bxy"[�  
�          Ap��7�@�=q=L��>��RB�W
�7�@�ff@dz�A�33B��H                                    Bxy"j�  
�          A����\)@������Z�HB�#׾�\)@�Q�?�AR{B�#�                                    Bxy"y<  "          A33>�(�@�G��J=q���
B��>�(�A�\?�@���B�                                      Bxy"��  
Z          Ap�?&ff@���^�R�ˮB���?&ffA ��>.{?���B���                                    Bxy"��  "          A ��>��
@�33��\)��G�B��R>��
A Q�333��ffB�z�                                    Bxy"�.  T          @�{��\@�
=�����=qB�p���\@���?���Az�RB�z�                                    Bxy"��  "          @��H�`��@�
=>Ǯ@5B�=�`��@���@^{A��B�L�                                    Bxy"�z  
�          @��
�I��@������@  B�\�I��@ҏ\@5A�(�B�8R                                   Bxy"�   �          @��R��R@�\�޸R�P��Bڏ\��R@�?�\)AAB�ff                                   Bxy"��  �          @�
=��\)@��H�K���(�Bʨ���\)@�G�>W
=?˅Bș�                                    Bxy"�l  �          @�{����@ٙ��XQ�����B��f����@��H<��
>�B�\                                    Bxy"�  �          @�z�O\)@�p��C33���B�B��O\)@��>�Q�@-p�B�(�                                    Bxy#�  "          @�=q���@�(��>{���HB�����@�>��@EB�                                    Bxy#^  T          @��ÿ�33@���=p����
B����33@�p�>\@8Q�Bŏ\                                    Bxy#)  �          @�{����@���P����ffBȸR����@�\    ���
Bƙ�                                    Bxy#7�  �          @�
=��@�G���
=�R{Bʽq��@ᙚ?�\)AK�Bʳ3                                    Bxy#FP  �          @��H�ff@�
=��(��:�\Bڅ�ff@�p�?�z�AS�
B�                                    Bxy#T�  �          @�{�#33@��\?J=q@��B�8R�#33@�(�@R�\A���B�.                                    Bxy#c�  "          @��R��(�@�=q@��B7(�B�G���(�@@��B��qB��f                                    Bxy#rB  
�          A Q쾀  @��@��HB]
=B�=q��  ?�
=@�B��3B��
                                    Bxy#��  "          @�ff�#�
@��\@�=qBH\)B�Lͼ#�
?��@�{B��{B��
                                    Bxy#��  �          @�p�?J=q@��@�Q�B&B��{?J=q@<(�@��B���B�Ǯ                                    Bxy#�4  
�          @�
=?��R@��@�
=B8ffB���?��R@33@�  B���Bvz�                                    Bxy#��  
�          A Q�>�G�@��H@�p�B5(�B���>�G�@!�@�B�p�B�(�                                    Bxy#��  �          A �;�G�@��@�p�B=�B�8R��G�@��@�ffB��3B��
                                    Bxy#�&  T          A녾�@���@�p�B1�\B����@,��@�(�B�� B�L�                                    Bxy#��  
�          A\)�B�\@��
@�z�B$�\B�8R�B�\@H��@���B��=B�Q�                                    Bxy#�r  �          A�R�xQ�@ƸR@�
=B  B��ͿxQ�@R�\@��B�Q�B��H                                    Bxy#�  
�          Aff��G�@�33@�G�B=qBə���G�@|��@�
=Bm{B�\)                                    Bxy$�  �          A=q���@��@�\)A��HB�
=���@���@���Ba�
B���                                    Bxy$d  
�          A녿�{@�@z=qA�=qB����{@���@�G�BV�B�
=                                    Bxy$"
  
�          A(��޸R@�{@��A���B���޸R@�ff@�
=BZ=qBޔ{                                    Bxy$0�  �          A�\���R@��\@���B+B��ÿ��R@5@���B�Q�B��                                    Bxy$?V  �          A�\��{@�z�@�G�B4��B�Q쿎{@#�
@��B��B��f                                    Bxy$M�  
�          A
=����@�{@�\)B  BǮ����@S�
@���B���B��                                    Bxy$\�  
�          A �ÿs33@�  @�  Bz�B�8R�s33@]p�@�
=B~=qBҏ\                                    Bxy$kH  {          A����@�G�@��B
�\B��Ϳ���@y��@�\)Bm��B�W
                                    Bxy$y�  
�          A  ����@��
@�B
�Bˏ\����@|��@�\Bm�RBڔ{                                    Bxy$��  
Z          A  ��@���@��Bz�BȨ���@e@�\B{  B�                                    Bxy$�:  
�          A33��(�@�ff@�\)B�\B̳3��(�@��@�Bg�B�B�                                    Bxy$��  "          A���\)@ָR@�ffB=qB����\)@�z�@��Bd��B޳3                                    Bxy$��  T          A33���@�G�@�
=B  B�𤿱�@w�@�\Bo\)B�ff                                    Bxy$�,  �          A\)��G�@��
@���B��B��Ϳ�G�@���@�BfG�B�.                                    Bxy$��  T          A=q��@θR@�G�BB���@x��@��
Bf�B�ff                                    Bxy$�x  �          A�H��(�@�Q�@���B�B�B���(�@u@�Bq��B�W
                                    Bxy$�  �          A�H��\)@�(�@��Bz�BϏ\��\)@�=q@���BfQ�Bߏ\                                    Bxy$��  
�          A�H���@�@�z�B�B����@�p�@ڏ\BcBܮ                                    Bxy%j  
�          A\)���
@�p�@$z�A���B����
@��\@�33B,��B��
                                    Bxy%  T          A{��@�z�?�p�A'�B��Ϳ�@�Q�@�ffB�\B�L�                                    Bxy%)�  T          A\)�O\)A ��?��A33B�uÿO\)@�\)@��B��B��3                                    Bxy%8\  �          A  ��p�@���?�ffAJffB�B���p�@�z�@��B�B�#�                                    Bxy%G  
�          A�
���R@��@z�Ah��Bɏ\���R@�ff@�{B�\B���                                    Bxy%U�  �          A�����@��@�  A�p�B͙�����@��@љ�BVp�B���                                    Bxy%dN  �          A  ��=q@�\)@�{B��B����=q@\��@��HB}
=B�.                                    Bxy%r�            A�
��=q@�=q@z�HA�33B�#׿�=q@�  @�G�BR=qBس3                                    Bxy%��  
Z          A(���  @ָR@��B=qB�=q��  @��@�
=BfffB۽q                                    Bxy%�@  
�          A
=��G�@陚@X��A�ffBǣ׿�G�@�ff@���BC��B�W
                                    Bxy%��  
�          A=q����@�@*�HA��
Bπ ����@��@���B,�HB��H                                    Bxy%��  �          A�H��\@�\)@!G�A�=qB�z���\@�
=@�B'(�B�.                                    Bxy%�2  �          A��\)@���@U�A��
Bֽq�\)@��H@���B=33B�p�                                    Bxy%��  "          AQ��(�@�@8Q�A�
=B����(�@�G�@�\)B/�B��                                    Bxy%�~  �          A
=��\)@�@E�A�33B��f��\)@��R@��B9p�B�.                                    Bxy%�$  "          @�ff��G�@�ff>�{@#�
B�.��G�@�(�@Z�HA�33Bɀ                                     Bxy%��  
�          @�(���@׮�;�����BՏ\��@�33>��?�33B��                                    Bxy&p  
�          @�z���R@љ��P  ��G�Bٞ���R@�녾aG���
=B���                                    Bxy&  
�          @�33�C33@Ϯ�=q��z�B�L��C33@���?�\@xQ�B��                                    Bxy&"�  "          @����G
=@أ��  ��z�B�G��G
=@�33?E�@��B�L�                                    Bxy&1b  
�          @�p��R�\@�
=��33�F�\B���R�\@ٙ�?��A33B癚                                    Bxy&@  "          @��\(�@أ׿u��ffB���\(�@љ�?�
=Aj{B�ff                                    Bxy&N�  "          @�ff�QG�@��
�E���Q�B���QG�@ҏ\@	��A�=qB��                                    Bxy&]T  
�          @�{�z�H@˅�޸R�R=qB�W
�z�H@�Q�?��@�G�B��                                    Bxy&k�  "          @��#�
@�z�fff��\)Bۀ �#�
@��
@Q�A���B��f                                    Bxy&z�  T          @������@�ff�5���\Bُ\���@�33@z�A��B�Q�                                    Bxy&�F  
�          @����/\)@�녿�=q�B�W
�/\)@�(�?�Ah��B�\)                                    Bxy&��  
�          @��AG�@�Q�Q����B���AG�@�\)@Q�A���B�aH                                    Bxy&��  �          @���Mp�@ᙚ�p����{B���Mp�@��@�Aq��B�ff                                    Bxy&�8  �          @�33�Y��@�Q�}p���B��Y��@ٙ�?�(�Ah(�B�#�                                    Bxy&��  "          @�33�C33@�ff�B�\����B����C33@�(�@  A�p�B��H                                    Bxy&҄  "          @���Dz�@��Y���ǮB�=q�Dz�@��@	��A~�\B��f                                    Bxy&�*  T          @�z��_\)@�(��#33��=qB��_\)@��H>\@.�RB�ff                                    Bxy&��  	�          A Q��J�H@�Q��ff�s�B�{�J�H@�Q�?p��@�G�B��                                    Bxy&�v  �          Aff�L��@�p���{���HB➸�L��@�\)@   Ac33B�3                                    Bxy'  
�          AG��\(�@��������B�(��\(�@��
?�z�A=��B�=                                    Bxy'�  �          Ap��Q�@�녿����  B�aH�Q�@��
?�(�Aa�B�                                     Bxy'*h  �          Ap��K�@�z�Q����B♚�K�@��H@\)A���B�\)                                    Bxy'9  
�          A   �C33@��   �e�B�
=�C33@�@!G�A��B�                                    Bxy'G�  
�          A   �.{@�Q쾞�R�\)B��
�.{@�  @0  A���Bހ                                     Bxy'VZ  �          A��#�
@�p��#�
��  B����#�
@��@ ��A�{B���                                    Bxy'e   
�          A ���ff@���L�����RB�{�ff@�\@ffA��B׀                                     Bxy's�  "          A �����@��5���B׮���@�Q�@��A�\)B�Q�                                    Bxy'�L  �          @����!�@�\)��\)�33B�ff�!�@޸R@/\)A�ffB��                                    Bxy'��  T          @���(�@�\)>��
@�B�#��(�@�\)@R�\AŅB��f                                    Bxy'��  "          @����[�@��H���q�B�\�[�@ָR@
=A��B��                                    Bxy'�>  
�          @�G��W
=@�׾8Q쿮{B�33�W
=@�Q�@(Q�A�(�B��                                    Bxy'��  "          @�  �@��@ۅ>�{@&ffB�u��@��@���@C33A��B�33                                    Bxy'ˊ  
�          @�=q�E�@���?�@~�RB�.�E�@��
@N�RA��
B�                                     Bxy'�0  T          @�ff�c�
@�
=?�A(�B��f�c�
@�ff@o\)A���B�
=                                    Bxy'��  T          @�\)�B�\@��?p��@�  B����B�\@�=q@i��A��B�.                                    Bxy'�|  
�          @�\)�ff@�=�G�?O\)BҞ��ff@�Q�@I��A�(�B�ff                                    Bxy("  "          @�{���R@�>�  ?�=qB�
=���R@�ff@QG�A£�B��                                    Bxy(�  
(          @��R�
=@��>��@<(�B����
=@�(�@Y��A�Q�B��                                    Bxy(#n  T          A   ��\)@���>Ǯ@3�
B�\��\)@߮@Z�HAʣ�B���                                    Bxy(2  
�          @����33@���>��H@`��B��ÿ�33@�
=@aG�A�33Bή                                    Bxy(@�  
(          @���\)@��?z�@�ffB˳3��\)@�z�@e�A�ffBΊ=                                    Bxy(O`  {          @��R��{@��R?!G�@��B�33��{@��H@g
=Aי�B�z�                                    Bxy(^  
�          @�ff��Q�@�\)?0��@��RB̞���Q�@�33@j=qAۅBϳ3                                    Bxy(l�  
�          @��R��Q�@���?+�@��\B��ῸQ�@�p�@j�HA�p�B˙�                                    Bxy({R  
�          @�\)��
=@��?�R@�\)B�  ��
=@�  @h��AمB�\                                    Bxy(��  _          @���u@���?(��@�
=B��f�u@��@l(�A��
BÞ�                                    Bxy(��            @�ff�n{@�33?Q�@���B�� �n{@��@s�
A�
=B�L�                                    Bxy(�D  
�          @�p���  @��?Tz�@�Q�B¨���  @�(�@s33A�\)BĞ�                                    Bxy(��  �          @�p��^�R@�  ?�z�A�B��)�^�R@�{@��\A�
=B��)                                    Bxy(Đ            @�  >�G�@陚@!G�A�=qB�.>�G�@�  @���B'
=B�W
                                    Bxy(�6  
3          @���}p�@�(�?���AU��B��
�}p�@ʏ\@�(�Bz�B���                                    Bxy(��  	.          A �Ϳ333@�\)@Q�Au�B�LͿ333@��@�ffB�
B���                                    Bxy(��  "          A(���33@�ff@1G�A���B��ΐ33@���@���B'��Bɏ\                                    Bxy(�(  T          A=q��33@�@*=qA���B��ῳ33@�ff@��B%�B�u�                                    Bxy)�  	�          Aff�У�@�G�@'�A�{B�W
�У�@�ff@��B"�HBҏ\                                    Bxy)t  �          A=q��\)@�\)@
�HAv�HB����\)@��@�
=B=qB̊=                                    Bxy)+  �          A����{@�=q@�A�Q�B���{@\@��HB��BѨ�                                    Bxy)9�  	�          A�R��@�G�@A��HB�z���@�=q@�G�BG�Bڏ\                                    Bxy)Hf  �          A�H��@�G�@�Ak�B��H��@�@���B�RB���                                    Bxy)W  �          A�\�ff@�33?�AM��B�aH�ff@��H@�=qBG�Bܳ3                                    Bxy)e�  "          A�R�#33@�=q?�Q�A?
=B�L��#33@˅@�{BB�Ǯ                                    Bxy)tX  T          A
=�U�@��?�  AE�B�=q�U�@��@�(�B=qB�\                                    Bxy)��  �          A
=�Vff@��H?�A (�B�
=�Vff@�  @�33A�(�B�W
                                    Bxy)��  T          A\)��@�=q?�33A:�\B�\��@�{@��A�B���                                    Bxy)�J  T          A�H���@�=q?��
A-�B�Ǯ���@�\)@\)A���B��f                                    Bxy)��  
�          A��G�@ٙ�?�p�AEG�B�ff��G�@���@�p�A��B�#�                                    Bxy)��  
�          @��R�Dz�@ۅ?�AHQ�B�33�Dz�@�\)@�(�B�B�\)                                    Bxy)�<  
�          @�  �Q�@���?�Q�AJ�RB�3�Q�@�z�@��
B �HB�\)                                   Bxy)��  	�          @��R�;�@�p�?ǮA<  B���;�@��\@���A�Q�B�ff                                   Bxy)�  �          @�p���ff@�{?�G�A
=B�Ǯ��ff@�@|��A�z�B���                                    Bxy)�.  �          @�\��33@�\)?�=qA!�B�aH��33@ƸR@|(�A�  B���                                    Bxy*�  |          @�����R@陚?��A�B�����R@���@~{A�ffB�#�                                    Bxy*z  ,          @�=q��=q@���?s33@߮B�\)��=q@�(�@k�A�\)B��H                                    Bxy*$   �          @�=q��@�\?�(�A-B�
=��@�Q�@��HA�G�B�L�                                    Bxy*2�  �          @����(�@�?���Aj�\B�(��(�@�(�@�ffBG�B�                                    Bxy*Al  T          @��׿�
=@�����%B��)��
=@�
=?��RA2{B��f                                    Bxy*P  �          @�
=��ff@陚�
�H��
=B�k���ff@��H?0��@��B���                                    Bxy*^�  	�          @�ff�h��@�׿����B��q�h��@�
=?ǮA;
=B���                                    Bxy*m^  T          @�\)�^�R@�����
���B���^�R@�@!G�A��\B��R                                    Bxy*|  "          @�  �(��@�
=�\)���B�Ǯ�(��@�\)@-p�A��B�p�                                    Bxy*��  "          @�  �
=@���n{���
B��q�
=@�
=?���AiB���                                    Bxy*�P  �          @�ff�+�@�\�����
=B���+�@�R?�p�AO�B�G�                                    Bxy*��  "          @�
=�z�@�׿}p���B��f�z�@�?�=qA^�\B�{                                    Bxy*��  �          @���(��@�G����G\)B�  �(��@�(�?���A{B��H                                    Bxy*�B  �          @�ff�   @�{��Q��iB����   @�(�?fff@�
=B�u�                                    Bxy*��  
�          @�Q쾏\)@��   �o
=B�Q쾏\)@�ff?\(�@�(�B�33                                    Bxy*�  �          @����\)@�(�� ���t(�B�ff��\)@�33?L��@���B�B�                                    Bxy*�4  "          @��Ϳ�G�@�z�5��{B���G�@���?���AY��BҮ                                    Bxy*��  
�          @��H��ff@��H����B�RB��)��ff@��?�  AQ�Bƣ�                                    Bxy+�  
�          @����
@�
=��p��N�RB�� ���
@�=q?s33A  B�z�                                    Bxy+&  �          @�(�?\)@���33��
=B�?\)@��\�W
=�ffB���                                    Bxy++�  "          @��
>B�\@�p�������HB���>B�\@�33���Ϳ��\B�G�                                    Bxy+:r  �          @��=��
@�(���\)��ffB�aH=��
@��\?�\@�  B�p�                                    Bxy+I  �          @���>\)@���33���B��)>\)@��������XQ�B��                                    Bxy+W�  �          @�{?=p�@��R�
=q��Q�B���?=p�@�(������G�B�
=                                    Bxy+fd  �          @��>��@�Q��(Q��ә�B��H>��@�33�����\B��f                                    Bxy+u
  �          @��?�Q�?޸R�����v  B'z�?�Q�@U�w
=�4�BoG�                                    Bxy+��  "          @��?��\@�G��.�R��p�B�.?��\@��@  ��{B��3                                    Bxy+�V  �          @�{?��
@�Q��7���ffB���?��
@�
=�}p��"=qB��R                                    Bxy+��  �          @��H?+�@�
=�@  � �B��f?+�@�
=�����9�B��                                    Bxy+��  
�          @�>.{@����\)����B�G�>.{@�33>�z�@I��B�u�                                    Bxy+�H  |          @��R>B�\@�33�xQ��$��B���>B�\@�33?z�HA&=qB���                                    Bxy+��  
2          @�=q>��
@�
=�xQ�� ��B�B�>��
@��R?��\A(��B�=q                                    Bxy+۔  T          @��\>�Q�@�ff��\)�9��B��>�Q�@�  ?^�RA�B��{                                    Bxy+�:  �          @�
=?   @��
�^�R��B���?   @�=q?�z�A<(�B��R                                    Bxy+��  T          @�ff>\@���z�����B��>\@��?��
Ao
=B�u�                                    Bxy,�  "          @�\)>B�\@�{�!G���G�B�W
>B�\@���?��RAg
=B�=q                                    Bxy,,  �          @���=��
@��R�J=q��B�k�=��
@��?��AN=qB�ff                                    Bxy,$�  �          @��
?s33@�Q��   ��ffB�z�?s33@�(��#�
��G�B���                                    Bxy,3x  T          @���?=p�@�z�xQ����B��3?=p�@��?��A.�\B���                                    Bxy,B  �          @�=q?.{@��H��p��ap�B��?.{@��?#�
@ÅB�                                    Bxy,P�  
�          @ƸR?\)@�  ��p��]�B�ff?\)@���?0��@�(�B���                                    Bxy,_j  �          @ƸR>�z�@�33�����#�B���>�z�@Å?�=qA�
B���                                    Bxy,n  �          @Ǯ>L��@�ff�(����33B�ff>L��@���?\Aa��B�L�                                    Bxy,|�  
�          @�\)>u@�ff�����L��B�p�>u@�ff?޸RA���B�=q                                    Bxy,�\  T          @�Q�>�
=@���#�
����B���>�
=@���?�Q�A�p�B��                                    Bxy,�  �          @��>�  @�33�.{���HB�
=>�  @���?�A��HB�                                    Bxy,��  
(          @��H=��
@\=���?k�B�k�=��
@�@�A��B�L�                                    Bxy,�N  T          @Å>�
=@��?�@�\)B��>�
=@�Q�@$z�A�p�B�W
                                    Bxy,��  T          @�{�O\)@�
=�����(�Bƀ �O\)@�녽�\)�J=qB�Q�                                    Bxy,Ԛ  
�          @�
=�u@�G��N{�\)B�uþu@�z�������B�ff                                    Bxy,�@  �          @�ff�Y��@�  �:�H�33B�k��Y��@�\)��(��O\)Bƀ                                     Bxy,��  �          @��׿�  @�\)�A���RB�8R��  @����=q�_�
Bɣ�                                    Bxy- �  
�          @�(���@����Mp�����B����@�G���33��HBԞ�                                    Bxy-2  T          @���\)@�p��A���\)Bճ3��\)@ڏ\�B�\���Bҙ�                                    Bxy-�  "          @�(�����@���R��  Bє{����@�p������7�B�ff                                    Bxy-,~  �          @�녿�=q@����\)�d  BиR��=q@�33?\)@�33B��                                    Bxy-;$  
�          @�녿��R@������]B�.���R@�33?��@�Q�B�z�                                    Bxy-I�  �          @��H��  @�p���  �w33B�W
��  @���>�
=@hQ�B�aH                                    Bxy-Xp  �          @�z��Q�@�\)�1G����B��H��Q�@�=q�0����G�B��                                    Bxy-g  "          @�=q�p�@���`����\B���p�@��׿�\��ffB�                                      Bxy-u�  T          @׮���@�  ���R���HB߅���@�=q>�?���B݀                                     Bxy-�b  
�          @��4z�@�
=��ff���B晚�4z�@�\)?�  A	�B�=                                    Bxy-�  T          @���@  @���xQ��{B�#��@  @�33?��
A�HB�8R                                    Bxy-��  �          @ָR�QG�@�녿@  ��(�B�3�QG�@�\)?�(�A((�B�ff                                    Bxy-�T  �          @���]p�@�p��������B�3�]p�@�Q�?�Q�AG\)B�8R                                    Bxy-��  T          @���c�
@��
��{�>{B��R�c�
@��?��
ATz�B���                                    Bxy-͠  T          @���l(�@��þ8Q���
B�� �l(�@���?�33Af�\B���                                    Bxy-�F  
�          @�{�p��@��ý#�
����B��=�p��@�\)?��
Aw33B�z�                                    Bxy-��  �          @ָR�r�\@�G��L�;���B����r�\@�  ?�\Au�B��3                                    Bxy-��  |          @�(��}p�@�33�8Q���B��\�}p�@�33?�z�A`  B�
=                                    Bxy.8  
�          @أ����R@��þ��R�'
=C �����R@��H?�Q�AD(�C��                                    Bxy.�  
�          @�p���p�@�
=����
=C����p�@��?���A(�CE                                    Bxy.%�  �          @ָR���@�
=�u�33C 33���@�  ?L��@��C �                                    Bxy.4*  �          @ָR�_\)@�G���
=�E��B�k��_\)@��R>�ff@s�
B��                                    Bxy.B�  �          @Ϯ�(�@�33�����0  B�Q��(�@�z�?�{Ak
=B��H                                    Bxy.Qv  
P          @ٙ����@�33?W
=@�z�B�����@��@7
=A���B���                                    Bxy.`  �          @��
��p�@�Q�?aG�@�p�B�33��p�@�(�@7
=A�z�B���                                    Bxy.n�  
�          @��Ϳ�=q@�=q?fff@�G�B�=q��=q@�{@8��A�ffBי�                                    Bxy.}h            @�\)���@�p�?��HA&{B�ff���@�ff@HQ�A�  B�ff                                    Bxy.�  
�          @�p��Q�@�33?�ffA4Q�Bި��Q�@�33@L(�A�RB��                                    Bxy.��  "          @�p��,(�@�ff?�\)A>{B���,(�@�{@Mp�A��B�                                    Bxy.�Z  �          @�33�$z�@�\)?�ffA.�RB��
�$z�@��@N�RA�(�B�33                                    Bxy.�   �          @Ϯ��  @�  ?k�A{B���  @�(�@7�A�p�B��H                                    Bxy.Ʀ  T          @��H���@��
?�(�A$��B�����@���@L(�Aޏ\B�u�                                    Bxy.�L  �          @�ff�	��@�
=?���A/33B��	��@��R@S�
A㙚B�=q                                    Bxy.��  �          @��
��@�p�?���A5�B�#���@��@Tz�A��HBۀ                                     Bxy.�  "          @�\)�  @�\)?��A*�HB�W
�  @��@Q�A�{B��H                                    Bxy/>  T          @�����@�ff?��HA"{Bخ���@��@L(�A�(�B��f                                    Bxy/�  T          @�\)�(�@���?�
=A  B�\�(�@�=q@K�AظRB�33                                    Bxy/�  �          @�����
@׮?�z�AG�B�
=��
@���@N{A�ffB�                                    Bxy/-0  T          @�33��Q�@�\)?���A�RB�33��Q�@�G�@L(�AՅB׳3                                    Bxy/;�  �          @�  ��{@�z�?�A{B�G���{@�{@L(�A�G�B�                                    Bxy/J|  T          @�׿�\@�ff?�\)A�RBх��\@�Q�@J=qA��BԳ3                                    Bxy/Y"  T          @�33���@�  ?�
=A�HB��H���@���@I��A�33B�\)                                    Bxy/g�  �          @��ÿ��@�(�?n{@��
B�zῇ�@�  @A�A�Q�B�8R                                    Bxy/vn  �          @��ÿ�{@�p�?�@�z�B�8R��{@���@,(�A�  BǨ�                                    Bxy/�  �          @�p��s33@ڏ\>�p�@A�B��f�s33@�(�@{A�\)B�                                      Bxy/��  T          @�p��=p�@�(��L�;\B�LͿ=p�@��@A�Q�B��f                                    Bxy/�`  �          @�{�=p�@�z�\�AG�B��Ϳ=p�@���?���An{B�.                                    Bxy/�  
�          @��ÿ�=q@����k�B�\��=q@��
@ffA��RB���                                    Bxy/��  �          @�=q����@�z�?�R@�Q�B�𤿬��@�(�@(��A�Q�B���                                    Bxy/�R  
�          @��ÿ��@ҏ\?p��@�ffB�  ���@�
=@:=qA˅B�B�                                    Bxy/��  �          @��
��  @�=q?��A/�
Bͨ���  @��@P��A�RBЮ                                    Bxy/�  
�          @��H��(�@��H?�{A��B�LͿ�(�@�p�@I��A�G�B��
                                    Bxy/�D  
�          @�����@�G�?
=@�{B�#׿���@У�@-p�A�  B�\                                    Bxy0�  "          @����(�@陚���
�B�\B�  ��(�@޸R@��A�Q�B��                                    Bxy0�  �          @��Ϳ��@�Q�>��H@tz�B�����@أ�@*�HA��
BɊ=                                    Bxy0&6  �          @�=q��@�G�?&ff@�33Bνq��@�Q�@0  A�  B���                                    Bxy04�  
�          @��}p�@��H?
=q@��\B��f�}p�@�33@*=qA�(�B�{                                    Bxy0C�  �          @��H��@Ӆ?h��@��B�
=��@���@7
=A��B�k�                                    Bxy0R(  
�          @�{����@�{?.{@�p�B�녿���@�p�@*�HA��B�{                                    Bxy0`�  "          @�(����@׮�Ǯ�Mp�Bʊ=���@�G�?�z�A_
=B�33                                    Bxy0ot  �          @��Ϳ�(�@�{�5��(�B��
��(�@�33?�=qA2�RB�33                                    Bxy0~  "          @�Q쿮{@љ��Q���\)Bˀ ��{@�  ?�
=A!G�B˳3                                    Bxy0��  �          @�zῚ�H@׮�(���(�B�Q쿚�H@Ӆ?�
=A?�
BȮ                                    Bxy0�f  
�          @�Q�c�
@�{=L��>\B®�c�
@Ӆ@
=A��RB�ff                                    Bxy0�  
�          @�녿��@���\��B��f���@أ�?���AN�RB�Q�                                    Bxy0��  �          @�녿&ff@���33�p�B��3�&ff@޸R?u@��B���                                    Bxy0�X  �          @�
=����@�(�=u>�(�B�\)����@���@\)A�B�33                                    Bxy0��  �          @��ÿ�G�@�p���\)��B�B���G�@�33@Q�A�p�B�{                                    Bxy0�  �          @�G���Q�@�z�=���?:�HB�{��Q�@�G�@G�A��\B�.                                    Bxy0�J  T          @񙚿���@�R�u��B��Ϳ���@�ff?�(�Ar{B�ff                                    Bxy1�  �          @��ÿ�p�@�z��ff�[�B��쿝p�@�ff?޸RAV�HB�Q�                                    Bxy1�  T          @񙚿�p�@�ff=�Q�?5Bƽq��p�@��H@G�A�(�BǨ�                                    Bxy1<  T          @�녿��@�ff=��
?��BǙ����@�33@��A��HBȊ=                                    Bxy1-�  T          @�G���\)@�ff=�G�?Tz�B�#׿�\)@�33@�A��HB�                                      Bxy1<�  �          @�=q��33@�\)>�=q@ ��B�uÿ�33@�\@�A�  B�p�                                    Bxy1K.  �          @��
���@��?�@x��BĀ ���@ᙚ@+�A��HBř�                                    Bxy1Y�  
�          @��H�fff@��>�G�@W
=B���fff@��@'
=A���B=                                    Bxy1hz  "          @񙚿E�@�  >�z�@(�B��3�E�@�33@��A�p�B�\)                                    Bxy1w   "          @��þ�{@�׾B�\��B�=q��{@�  ?��RAuB�p�                                    Bxy1��  �          @񙚿L��@�{>��H@n{B�8R�L��@�\)@'
=A�
=B�                                      Bxy1�l  �          @���33@�\)?�
=AL��Bͮ��33@�{@l(�A�\Bг3                                    Bxy1�  �          @����(�@�(�?�  A;�B�aH��(�@���@^�RA�33BɊ=                                    Bxy1��  �          @�\����@�33?�ffA$  B�\����@�@Q�Aՙ�B���                                    Bxy1�^  "          @���?\)@�Q���n{B��?\)@�33?���AMp�B��R                                    Bxy1�  "          @���?xQ�@��ÿ5��\)B�L�?xQ�@�{?���A+33B��                                    Bxy1ݪ  �          @�R?��@�G������(�B��?��@ᙚ?z�H@�33B���                                    Bxy1�P  �          @�  ��z�@��?z�HA33B����z�@�G�@*=qA�  BЅ                                    Bxy1��  "          @�33�8��@�G�@��A��HB�k��8��@��@x��B�\B�Q�                                    Bxy2	�  �          @����
@�\)?޸RA^{B�����
@��R@dz�A�=qB�\                                    Bxy2B  �          @�녿�{@ٙ�@p�A��Bʨ���{@���@���B  B�                                    Bxy2&�  
�          @�\�@�ff@�A��HBָR�@��@u�A��B�W
                                    Bxy25�  �          @�\��
@��@�A��B�����
@���@�G�B�HB�u�                                    Bxy2D4  �          @�33���@Ϯ@%A��B�p����@�  @��B�\B�R                                    Bxy2R�  
�          @�Q����@˅@!�A�p�B�.���@�z�@��RB  B�                                    Bxy2a�  �          @�R����@޸R?�Q�A9�B��f����@���@U�AܸRB���                                    Bxy2p&  �          @��H��\)@�
=?���A33B��H��\)@��@o\)A�33B�k�                                    Bxy2~�  �          @�녿�Q�@���@p�A�33Bԅ��Q�@���@~{BQ�B�#�                                    Bxy2�r  �          @���,��@ʏ\@p�A���B�(��,��@�z�@��
B	ffB��                                    Bxy2�  �          @陚�#33@�p�@A�p�B�aH�#33@���@���B�B��                                    Bxy2��  
Z          @�����@Ϯ@�RA��RB�p����@��
@{�B�B�\                                    Bxy2�d  "          @�\)�
=@�  @ffA�p�B��)�
=@�p�@tz�A�
=B�(�                                    Bxy2�
  �          @�  �#33@˅@�A�Q�B߮�#33@�
=@\)B�B���                                    Bxy2ְ  �          @�  �)��@�Q�@�RA�z�B��
�)��@��H@��B
Q�B�                                    Bxy2�V  �          @���:=q@���@7
=A��B��:=q@�  @���B�HB�p�                                    Bxy2��  |          @�=q�>�R@�
=@;�A�\)B�{�>�R@�{@�ffBp�B�8R                                    Bxy3�  ,          @���R�\@��@[�Aߙ�B�=q�R�\@��@�=qB$�RB�k�                                    Bxy3H  �          @�R�G�@�z�@N�RA��B���G�@�G�@���B �B��                                    Bxy3�  �          @�R�N�R@��@dz�A홚B�8R�N�R@�{@�z�B+{B�aH                                    Bxy3.�  
�          @�p��N{@�(�@\��A�ffB����N{@��@���B'��B�ff                                    Bxy3=:  "          @�=q�3�
@�G�@p�A�  B����3�
@���@�  B	33B�Q�                                    Bxy3K�  �          @߮�(�@�Q�@'
=A�B�=q�(�@��\@�z�BG�B�\)                                    Bxy3Z�  �          @߮�=p�@�z�@7
=A���B�u��=p�@��@���B{B��                                    Bxy3i,  �          @�\)�J=q@��@)��A��B��H�J=q@��\@���BB�Q�                                    Bxy3w�  �          @޸R���@�Q�@u�B=qB������@���@�33B=��B�\)                                    Bxy3�x  T          @����B�\@��\@c�
A�(�B��B�\@|(�@�G�B/��B�G�                                    Bxy3�  T          @�33�QG�@�z�@`  A�(�B�� �QG�@q�@�B,��C޸                                    Bxy3��  �          @ָR��z�@g
=@s�
B  C
����z�@p�@�z�B/��C@                                     Bxy3�j  �          @�=q�J�H@�(�@AG�A��B�p��J�H@�z�@���B��B��f                                    Bxy3�  �          @ۅ�C33@�33@ ��A�  B�{�C33@�\)@z=qB(�B��=                                    Bxy3϶  T          @ڏ\�a�@�G�@W
=A�p�B��a�@n�R@�Q�B%�Cn                                    Bxy3�\  �          @ָR���@�G�@G�A��B܊=���@�\)@n{B	�
B�u�                                    Bxy3�  T          @�(����@���?O\)@��B�����@\@ ��A�z�B�L�                                    Bxy3��  �          @�p���G�@�
=?��RAO33B�Ǯ��G�@�33@H��A�B���                                    Bxy4
N  T          @�\)�
=@�=q?�p�A)G�B�B��
=@�Q�@:�HA�{B�33                                    Bxy4�  �          @�G���  @�{@�\A��B�G���  @�(�@u�B	�HB�=q                                    Bxy4'�  T          @ٙ����
@\@/\)A�z�B˽q���
@���@��Bz�B��H                                    Bxy46@  �          @�ff�(��@�G�@��A�\)B���(��@�  @qG�B�B��                                    Bxy4D�  �          @أ׿c�
@��@'
=A��B�\)�c�
@���@�(�B�RB�{                                    Bxy4S�  "          @�ff���@�G�@3�
A���B��þ��@�33@���B�B���                                    Bxy4b2  T          @�{�Y��@��\@�Q�B=qC޸�Y��@8��@�B?G�C�H                                    Bxy4p�  �          @ָR�Vff@�  @z=qBp�C 0��Vff@E@�(�B;�
C	=q                                    Bxy4~  �          @��]p�@AG�@��B.  C
���]p�?��@��BP�C��                                    Bxy4�$  �          @�  ���@ə�?���A��BȀ ���@��@Z�HA��RB���                                    Bxy4��  T          @��
�h��@�ff@%A���B�Q�h��@��\@�G�BG�B�8R                                    Bxy4�p  �          @�=q��\)@�
=@�(�BQ�B�8R��\)@`  @�ffBR�\B�8R                                    Bxy4�  �          @Ӆ�(�@�  @n{B	�B�aH�(�@xQ�@�(�B>p�B�                                    Bxy4ȼ  
�          @�=q��(�@��@��A���B����(�@�{@s33B�B�u�                                    Bxy4�b  �          @�(��
�H@�G�@A�{B�(��
�H@�  @p  B

=B�                                      Bxy4�  �          @����
=@��?�
=A��HB����
=@��@X��A��B�k�                                    Bxy4��  �          @��H��ff@���?���A�Bϳ3��ff@�G�@,(�A�G�B��                                    Bxy5T  �          @Ӆ�{@Å?���AB���{@�z�@%A��B���                                    Bxy5�  �          @�z��#33@�Q�?�\)A\)B�
=�#33@���@&ffA�33B�\                                    Bxy5 �  "          @�z���@��?�@�=qB����@�=q@z�A��B�Q�                                    Bxy5/F  �          @�\)�@�\)?\(�@���B�33�@��@=qA�Q�B��                                    Bxy5=�  �          @�
=� ��@��H?�(�A���B؅� ��@���@\��A�
=B��
                                    Bxy5L�  �          @����H@��H?�A��B׸R���H@��@X��A�{B��)                                    Bxy5[8  �          @�(���G�@���@�A�33B�p���G�@��@dz�B(�Bؙ�                                    Bxy5i�  �          @�G���
@���?��
AX(�B���
@�{@?\)A�\)B�p�                                    Bxy5x�  �          @���@���?���AG�B�aH��@��@$z�A��\B�#�                                    Bxy5�*  �          @�{��@\?�{A�B�{��@��@&ffA�=qB׳3                                    Bxy5��  �          @����@��?�  A�HB֔{���@��
@{A�33B��                                    Bxy5�v  �          @��H�(Q�@��ÿ��,��B���(Q�@�z�>u@�RB�                                      Bxy5�  ,          @���0  @�ff�p����HB�p��0  @�Q�?�\@�G�B�                                    Bxy5��  �          @θR�.{@��\����z�B�
=�.{@��R?�Q�A*�\B��                                    Bxy5�h  �          @׮��p�@�\)?�p�AT  Bأ׿�p�@�@:=qA�(�B�{                                    Bxy5�  �          @أ׿�z�@���@��A�{B�(���z�@�p�@i��B��B�                                      Bxy5��  �          @��H�˅@���@�A��HB�\)�˅@���@j=qBz�B��f                                    Bxy5�Z  �          @ڏ\���
@�Q�?aG�@�ffBҔ{���
@�33@p�A�
=Bԏ\                                    Bxy6   �          @أ׿޸R@��þaG���33B�녿޸R@�(�?���A<��Bҏ\                                    Bxy6�  "          @��H��@У׾���~{B��Ϳ�@�{?��AB�33                                    Bxy6(L  "          @��H��@љ����
�+�BԳ3��@��
?\AN{BՔ{                                    Bxy66�  �          @��ÿ�ff@˅?���A~�\B�
=��ff@�
=@UA��B͏\                                    Bxy6E�  "          @�33���@��@�A�Q�B�aH���@�ff@g
=A��BɸR                                    Bxy6T>  �          @�33��
=@�Q�@z�A�G�B�Ǯ��
=@�  @qG�BG�B�#�                                    Bxy6b�  T          @�33��  @��@ffA��HB�aH��  @���@s�
BQ�B�W
                                    Bxy6q�  T          @�p���ff@�  @/\)A��B�
=��ff@���@�p�B�B���                                    Bxy6�0  T          @�Q쾳33@���@�A���B�G���33@�
=@`��A�(�B�                                    Bxy6��  
Z          @ָR�B�\@���@A��B�k��B�\@���@r�\B	��B��H                                    Bxy6�|  
�          @�����@�G�@p�A��\B�\����@�=q@j=qB�
B���                                    Bxy6�"  �          @���\)@˅?�(�A��B����\)@�ff@\(�A��HB��R                                    Bxy6��  "          @�ff�.{@�(�?�33A�=qB�{�.{@��@XQ�A�z�B�k�                                    Bxy6�n  
�          @�{��\)@˅@A�\)B�Q콏\)@�p�@b�\B   B�z�                                    Bxy6�  �          @ָR���@���?�A�=qB�z���@���@Tz�A�ffB�aH                                    Bxy6�  T          @�Q�(�@θR?���A~{B�Ǯ�(�@��H@U�A�RB��                                    Bxy6�`  
�          @�{�k�@�  ?�  A-�B�\�k�@���@0��A�p�B�W
                                    Bxy7  
�          @�녿�p�@�ff?���A3�B���p�@�ff@4z�AÙ�B�ff                                    Bxy7�  �          @��ÿ�33@У�?B�\@θRB�aH��33@���@�A�B�                                      Bxy7!R  �          @�  ���@���?�A�
B����@�=q@+�A�B�k�                                    Bxy7/�  �          @׮�}p�@Ϯ?�\)A=�B�ff�}p�@��@7�A�ffB��)                                    Bxy7>�  T          @��ÿ�=q@�(�?�A{�
B�zῪ=q@���@R�\A�ffB��f                                    Bxy7MD  T          @׮�h��@�?��HAk�B�.�h��@�33@J�HA�\)B�                                    Bxy7[�  �          @�\)�\@θR?J=q@�=qB΅�\@��H@�\A�{B�
=                                    Bxy7j�  T          @�ff�>{@�{?(�@��B�#��>{@�z�?��RA�z�B랸                                    Bxy7y6  "          @���\)@�?��Az�B��Ϳ�\)@�Q�@ ��A���B�(�                                    Bxy7��  �          @���333@�\)?��\A0z�B�aH�333@�  @0  AÅB�\)                                    Bxy7��  T          @���8Q�@љ�?=p�@�B�z�8Q�@�ff@  A��\B�.                                    Bxy7�(  �          @�ff��=q@�\)?��\AQ�B�#׿�=q@��@   A�p�B̨�                                    Bxy7��  
�          @�
=�\(�@�
=?ǮAVffB�8R�\(�@�@@��Aՙ�BĔ{                                    Bxy7�t  T          @ָR�Y��@�\)?��HAIB��ÿY��@�
=@:�HA�
=B�8R                                    Bxy7�  �          @ָR��p�@�{?�=qA6�RBɨ���p�@��R@1�A�Q�B�aH                                    Bxy7��  
�          @�ff��(�@�?��\A  B��쿼(�@���@{A��Bπ                                     Bxy7�f  �          @�{�
=@ȣ�?fff@���B�.�
=@���@�A���B�aH                                    Bxy7�  T          @���/\)@�Q�?�Ak\)B��f�/\)@�
=@<��A���B�ff                                    Bxy8�  �          @��
� ��@�33?�  ARffB؏\� ��@��H@7
=A͙�B۞�                                    Bxy8X  �          @�33�  @Å?c�
@���B�u��  @��@G�A��B���                                    Bxy8(�  �          @ҏ\�
=q@�?�\@�ffBڔ{�
=q@���?�z�A�G�B�G�                                    Bxy87�  T          @��
��{@�G�?@  @���B����{@�ff@
�HA�\)Bֽq                                    Bxy8FJ  T          @��
���R@�p�>�33@C�
B�{���R@�p�?�A~=qB��                                    Bxy8T�  T          @��
��  @�p�>W
=?�\)B�W
��  @�ff?�Aj�RB�8R                                    Bxy8c�  "          @�(���@ə�?:�H@�G�B���@�
=@��A���B׮                                    Bxy8r<  T          @����@ʏ\?��@��B�Ǯ��@�G�?�(�A�(�B�L�                                    Bxy8��  "          @�(���@�Q�?uA=qBԳ3��@�(�@
=A�p�Bֳ3                                    Bxy8��  
�          @��
�'�@�  ?\(�@�ffB�.�'�@���@(�A�
=B�q                                    Bxy8�.  T          @���6ff@�\)?z�@�\)B����6ff@�ff?�z�A�B�.                                    Bxy8��  �          @Ӆ�5�@�ff?�@��\B��
�5�@�?���A�ffB��                                    Bxy8�z  �          @�33�3�
@�(�?aG�@��RB�\�3�
@���@�A��B��
                                    Bxy8�   �          @��
�&ff@��H?�z�Ah��B�
=�&ff@�=q@;�Aң�B�#�                                    Bxy8��  T          @���{@���?��A�(�B�G��{@�ff@H��A�RB�q                                    Bxy8�l  
�          @љ��
=@��
?��RA�G�B�u��
=@���@P  A�(�B߀                                     Bxy8�  "          @�=q�
=@�p�?�(�AO�B߅�
=@�{@0��A�G�B���                                    Bxy9�  
�          @љ���
=@��?ǮA\  B�W
��
=@��@7�A�  B�L�                                    Bxy9^  "          @�=q���@��?��A�B�{���@�p�@J�HA癚Bڀ                                     Bxy9"  
�          @љ����R@�=q@	��A�ffB��Ϳ��R@�{@X��A��B���                                    Bxy90�  
�          @У��33@�G�?���AC33B�p��33@��\@+�A�z�B�B�                                    Bxy9?P  �          @���	��@��
?n{A33Bڮ�	��@�Q�@G�A�{B��f                                    Bxy9M�  �          @����@��?8Q�@�G�B����@�33@�
A��HB���                                    Bxy9\�  
�          @��H��@���?Y��@��B���@��@(�A�\)B��                                    Bxy9kB  �          @����Q�@�z�?h��@�33B�Q��Q�@�G�@\)A���B��                                    Bxy9y�  T          @�z���@�ff?h��@�33B�����@��H@  A��\B���                                    Bxy9��  �          @���
=q@�?�A"{Bڊ=�
=q@���@   A���B�
=                                    Bxy9�4  �          @��
�{@��H?��A6�HB��{@���@'�A�z�B��
                                    Bxy9��  �          @���
=@��H?��\A0��B�L��
=@��@%�A�z�B�.                                    Bxy9��  T          @���@�  ?��Ac�B�#���@��@:=qA�
=B�                                    Bxy9�&  T          @�\)�%@�Q�?��RAL��B�{�%@�G�@1G�A���B�\                                    Bxy9��  
�          @�{� ��@Å?��\A��B��{� ��@��@�A��B�#�                                    Bxy9�r  �          @����<(�@���?z�@��B����<(�@�(�?�{A�=qB�#�                                    Bxy9�  �          @����*�H@�?�=qA8z�B�\�*�H@�  @%A�p�B��H                                    Bxy9��  T          @�z��AG�@�=q?xQ�A{B��
�AG�@�
=@p�A��B�Ǯ                                    Bxy:d  "          @����S�
@�{?
=@���B�aH�S�
@�?���A~�HB�Ǯ                                    Bxy:
  T          @��H�>�R@��H>�Q�@FffB��>�R@��
?�{Ac�B�Ǯ                                    Bxy:)�  
�          @љ����\@�=q>�@�C �\���\@�33?ǮA]��C{                                    Bxy:8V  
�          @����O\)@�  ?��
AW�B�  �O\)@�G�@+�A���B�z�                                    Bxy:F�  �          @�p��hQ�@��>�33@B�\B�B��hQ�@��?�ffAV�HB�B�                                    Bxy:U�  �          @�{�i��@�z��
=�mp�B�33�i��@�33?@  @�33B��\                                    Bxy:dH  
�          @�(��a�@�z�>���@.{B��f�a�@��R?�33AN�HB��)                                    Bxy:r�  �          @�����z�@����   ��Q�C���z�@�33�n{� Q�C)                                    Bxy:��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy:�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy:��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy:��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxy:�,            @أ�?�\@�\)@C�
A�{B�33?�\@�{@���B(�B��R                                    Bxy:��  �          @�(�?aG�@�
=@C33A�Q�B�� ?aG�@�@�ffBB�Ǯ                                    Bxy:�x  �          @�Q�u@��?�33A�z�B��ÿu@��@N{A�BƊ=                                    Bxy:�  T          @�  ��\)@�?\AP��B����\)@��R@6ffA�(�B���                                    Bxy:��  T          @�����@�=q?��HAm�B˨�����@��@AG�AָRBͨ�                                    Bxy;j  �          @�z῕@���?�A}B�#׿�@�  @FffA�
=B�
=                                    Bxy;  
(          @�p���  @�Q�?�A�{Bʞ���  @��R@L��A��
B̸R                                    Bxy;"�  T          @�����H@�G�?�ffA{33Bɳ3���H@���@EA�\)B˞�                                    Bxy;1\  T          @Ӆ��z�@�Q�?ǮAZ�HB�\)��z�@�G�@6ffA�ffB�\)                                    Bxy;@  
�          @Ӆ���
@ə�?��AW�B�녿��
@��\@5�A��B̳3                                    Bxy;N�  �          @Ӆ���\@�p�?��A33B�����\@���@{A���B�8R                                    Bxy;]N  �          @��!G�@�Q�?��A2�HB���!G�@��H@(Q�A�(�B��)                                    Bxy;k�  
�          @��Ϳ0��@�z�?У�Ac�
B�LͿ0��@��@<(�A�ffB�Q�                                    Bxy;z�  T          @�{>B�\@У�?��
A1�B��R>B�\@Å@'�A�p�B��                                     Bxy;�@  "          @��?��@θR>\@R�\B��?��@Ǯ?�p�Ar=qB��3                                    Bxy;��  
(          @�  ?c�
@�p�>8Q�?�ffB�Ǯ?c�
@�\)?���AXz�B�\)                                    Bxy;��  �          @أ�>��R@ָR>�@��\B���>��R@θR?�\)A��B�u�                                    Bxy;�2  �          @�=q>B�\@�
=?���A��B���>B�\@�33@��A���B���                                    Bxy;��  "          @�  ���R@�Q�?���A\Q�B�aH���R@���@:�HA�z�B���                                    Bxy;�~  T          @��
��z�@��@W�A�\)B��ᾔz�@�G�@��RB+�RB�                                    Bxy;�$  T          @�
=��{@��
?��Ao
=B�(���{@�@+�A���B�ff                                    Bxy;��  T          @Ӆ� ��@���?��
A\)B�33� ��@�@��A�  B��                                    Bxy;�p  
�          @�z��aG�@�ff?��A@��B��3�aG�@�G�@\)A�=qB���                                    Bxy<  T          @��?\)@�?�A|Q�B�z��?\)@�@<��A�=qB��                                    Bxy<�  �          @ָR�H��@�p�?�Q�AiB����H��@�ff@5�A��
B�G�                                    Bxy<*b  "          @�{�h��@��\?޸RAqG�B��3�h��@��@3�
A�z�B�                                    Bxy<9  �          @ָR�p  @��?�=qA[
=B�\)�p  @��
@)��A��\B��                                    Bxy<G�  �          @�ff�p��@�?��HA�(�B����p��@��@?\)A�{C �H                                    Bxy<VT  
Z          @�
=�qG�@�
=?�
=A�\)B����qG�@�ff@=p�A�p�C ��                                    Bxy<d�  "          @����aG�@�33@
=qA��B����aG�@�G�@N{A�p�B��{                                    Bxy<s�  T          @�33�P��@�=q?��HA�
=B��P��@���@Dz�A�=qB��R                                    Bxy<�F  
�          @�33?���@љ�>��?��\B�\)?���@��
?��RAK
=B��\                                    Bxy<��  �          @ڏ\?�@�Q�
=q���B�p�?�@Ϯ?O\)@ۅB�L�                                    Bxy<��  T          @���?��H@أ�>\@I��B���?��H@љ�?�G�Am�B�#�                                    Bxy<�8  T          @�(�?5@ڏ\>��@X��B�(�?5@�33?�ffAr�RB�                                    Bxy<��  
�          @���>�\)@�33?:�H@�=qB�W
>�\)@љ�@�A��B��                                    Bxy<˄  �          @ᙚ�k�@�?��\A%p�B���k�@���@*=qA�33B�\                                    Bxy<�*  
�          @��
�8Q�@׮?�=qA333B�  �8Q�@�=q@,(�A�{B�33                                    Bxy<��  T          @�
==L��@�(�?�\)Ap�B�33=L��@�Q�@��A��B�(�                                    Bxy<�v  �          @ָR�#�
@Ӆ?���A�B��R�#�
@�  @�A�{B�                                    Bxy=  
�          @�\)�5@�G�?�z�AA��B�W
�5@Å@-p�A�z�B�.                                    Bxy=�  
Z          @�
=���@�Q�?�(�A'\)B��Ή�@�(�@!G�A��\B�(�                                    Bxy=#h  �          @��Ϳ��\@�
=?�{A�B��ÿ��\@Å@=qA��
B�                                      Bxy=2  "          @��ͿTz�@�
=?�(�A)�B\�Tz�@��H@!G�A�=qB�z�                                    Bxy=@�  �          @�\)��
=@��?��HAHz�B�G���
=@��
@0��A�  B���                                    Bxy=OZ  �          @ٙ�>L��@��?�ffAup�B�z�>L��@���@EA؏\B�33                                    Bxy=^   
�          @��>L��@У�?��HA�{B��=>L��@�\)@P  A��
B�8R                                    Bxy=l�  T          @ٙ�<��
@�{@
�HA�\)B��3<��
@��@\(�A�
=B��                                    Bxy={L  �          @��=L��@�{@�A��B�\=L��@��
@\��A�\)B���                                    Bxy=��  �          @ٙ���@ʏ\@�\A�ffB���@��@aG�A��B���                                    Bxy=��  �          @ٙ���G�@�(�@�A�33B����G�@���@a�A�z�B���                                    Bxy=�>  �          @׮>.{@ƸR@'
=A�  B��
>.{@��@s�
B	�B�z�                                    Bxy=��  
�          @�p���=q@�@p�A�G�B���=q@��@j=qBG�B��{                                    Bxy=Ċ  "          @�{�8Q�@�
=@3�
A�G�B�� �8Q�@���@}p�B=qB���                                    Bxy=�0  �          @�z��
=@�{@�RA�(�B���
=@��@[�A�33B��\                                    Bxy=��  �          @��
���@�ff@33A���B�\)���@��@P��A�\B�.                                    Bxy=�|  T          @�zῚ�H@�z�@p�A�\)B�=q���H@�=q@Y��A���B�u�                                    Bxy=�"  T          @��Ϳ��@���@�A�\)B��ÿ��@�{@b�\B  BϞ�                                    Bxy>�  T          @�
=���H@��R@   A��
B����H@��H@i��Bp�B׏\                                    Bxy>n  �          @�Q쿳33@��H@#33A�\)B�𤿳33@�ff@n{B�HB��
                                    Bxy>+  �          @أ׿���@�z�?��A�B�  ����@�(�@HQ�A�G�Bș�                                    Bxy>9�  "          @��H���
@Ϯ?�ffAffB��R���
@�z�@�A��B�
=                                    Bxy>H`  �          @��
�333@�{?��A;
=B�k��333@�G�@'
=A�B�=q                                    Bxy>W  T          @�p��aG�@�ff?�Q�AG�BÙ��aG�@���@-p�A���BĮ                                    Bxy>e�  �          @�{�G�@θR?�G�AQG�B��3�G�@���@1�Aď\B®                                    Bxy>tR  �          @��;�Q�@љ�?��A�B�=q��Q�@ƸR@�A�Q�B��{                                    Bxy>��  T          @�z���@�z�?�33Ag33B�W
���@�@9��A�  B���                                    Bxy>��  T          @׮��\@�z�@�\A�33B�#׿�\@�33@Q�A�p�B���                                    Bxy>�D  "          @�\)��@�Q�?�ffAUB�
=��@�=q@4z�A��B���                                    Bxy>��  T          @�
=�8Q�@�
=?�z�Ad��B����8Q�@�  @:�HA�=qB���                                    Bxy>��  
�          @�{���H@˅?�{A;�
B��
���H@��R@&ffA��Bπ                                     Bxy>�6  T          @�
=���\@�z�?�\AuG�B�{���\@���@AG�A�BǊ=                                    Bxy>��  
�          @ٙ����@�G�?�(�AG33B��
���@Å@/\)A�z�B�.                                    Bxy>�  �          @ڏ\�k�@�p�?�\)A�Býq�k�@��@�A�33BĞ�                                    Bxy>�(  "          @�=q��
=@Ӆ?�Q�A!�B�8R��
=@Ǯ@\)A�G�B�ff                                    Bxy?�  
�          @أ׿W
=@�=q?�ffA2{B£׿W
=@�@%A�z�BÏ\                                    Bxy?t  T          @أ׿�@���?��AB��R��@��@ffA���B�=q                                    Bxy?$  
�          @�G��#�
@�ff?Tz�@ᙚB���#�
@���@
=qA�(�B�z�                                    Bxy?2�  
Z          @�  �&ff@�=q?��A7�B�\)�&ff@�p�@'�A�p�B��                                    Bxy?Af  �          @��
<#�
@�=q>��
@1G�B��q<#�
@�(�?�\)Ad��B��q                                    Bxy?P  �          @�G��n{@ҏ\?��A6{B�#׿n{@�@'�A�=qB�.                                    Bxy?^�  	`          @ڏ\���@љ�?�  AK\)B��Ϳ��@��
@1�A�Q�B�(�                                    Bxy?mX  
Z          @�(��p��@�ff@	��A�{Bģ׿p��@�z�@X��A��B�33                                    Bxy?{�  �          @���^�R@�=q?��
Ap��B�
=�^�R@��H@C�
Aә�B�8R                                    Bxy?��  T          @�
=����@���?��
AK\)B�z῰��@ƸR@5�A���B��                                    Bxy?�J  
�          @�\)�У�@�=q?�\)AW33B�녿У�@Å@9��AĸRB��f                                    Bxy?��  	          @�
=�
=@�z�?�ffAo�
Bؔ{�
=@���@B�\A��HB�aH                                    Bxy?��  �          @޸R��=q@У�?�=qA2�HB�LͿ�=q@�(�@&ffA���B�                                      Bxy?�<  �          @�ff�h��@أ�?��
A*=qB�=q�h��@�(�@'
=A�(�B�(�                                    Bxy?��  �          @߮���
@�
=?�\)A�
Bͮ���
@˅@(�A�\)B�{                                    Bxy?�  �          @�\)��ff@�?�p�AD(�B��ÿ�ff@�Q�@1�A�(�B�k�                                    Bxy?�.  �          @�G���
=@�{?z�H@��B�=q��
=@˅@�\A�  B���                                    Bxy?��  �          @�=q�У�@�  ?�G�A$  B���У�@��
@$z�A��BЮ                                    Bxy@z  �          @�=q��\)@�33?�ffA��Bʙ���\)@�Q�@��A�Q�B�Ǯ                                    Bxy@   �          @�=q�s33@�Q�?�AmB�
=�s33@�Q�@G�A��
B�G�                                    Bxy@+�  �          @߮��=q@Ӆ?��Am�B��쿪=q@��
@Dz�A��HB̙�                                    Bxy@:l  �          @��Ϳ��
@��
?��A9B��쿣�
@�
=@+�A���B�8R                                    Bxy@I  �          @ٙ��\)@У�?   @��\B��\)@�G�?��
A|(�B��                                    Bxy@W�  �          @ٙ����@��
@\)A��C�����@u@C�
A�(�CB�                                    Bxy@f^  �          @����{@�?�33A�G�CY���{@�ff@6ffA��
CB�                                    Bxy@u  �          @�ff���@���@A�{Cn���@�33@N{A�C

                                    Bxy@��  �          @�ff���@��\@�
A���C
J=���@q�@HQ�Aי�C
                                    Bxy@�P  �          @�p�����@�?�\)A8  C�
����@���@�A�C�R                                    Bxy@��  �          @�ff��Q�@��?�A��RC����Q�@�@7
=A��HC�                                     Bxy@��  �          @������\@h��@�A�\)C����\@E@C33A�z�C޸                                    Bxy@�B  �          @�ff����@p��@��A��C������@L��@I��A�{C�H                                    Bxy@��  �          @������@S�
@��A��RC(�����@4z�@0��A�G�C�                                    Bxy@ێ  �          @�  ���\@A�@  A�Q�C}q���\@!G�@3�
A�\)C�                                    Bxy@�4  �          @�����  @(��@(�A�z�Cff��  @ff@:=qA˅C��                                    Bxy@��  �          @�G���z�@,(�@)��A�  C����z�@
=@HQ�A�  C�                                    BxyA�  �          @�G���Q�@*=q@:�HA��C33��Q�@�@X��A�  C�=                                    BxyA&  �          @ڏ\����@'�@?\)A�{C������?�(�@]p�A�G�C s3                                    BxyA$�  �          @����z�@p�@K�A���Cn��z�?��
@g
=A�33C"��                                    BxyA3r  �          @�(����
@{@EA��CO\���
?�ff@aG�A���C"=q                                    BxyAB  �          @ڏ\���H@��@FffA�=qC޸���H?�(�@aG�A�  C"�H                                    BxyAP�  �          @�  ��  @'
=@@  A�
=C�{��  ?��H@]p�A��C!(�                                    BxyA_d  �          @��H��33@'�@AG�AɮC޸��33?�(�@^�RA�\)C!k�                                    BxyAn
  �          @�(����@9��@;�A�ffCu����@��@\��A��
C                                    BxyA|�  �          @���G�@E�@333A���C�3��G�@p�@W
=A��
C
                                    BxyA�V  �          @����  @J�H@.{A��C���  @$z�@S33A�=qC��                                    BxyA��  �          @�ff��{@HQ�@>�RA�33C)��{@{@b�\A�p�C�
                                    BxyA��  �          @ڏ\���H@G
=@9��A�33C����H@{@^{A�  C.                                    BxyA�H  �          @�(���@:=q@E�A�z�CǮ��@\)@fffA�{C�{                                    BxyA��  �          @���  @P  @O\)A�33Ch���  @"�\@u�B�C:�                                    BxyAԔ  �          @���  @^{@J�HA��CǮ��  @0��@s33A�p�CT{                                    BxyA�:  �          @�\)��(�@G�@N�RA�33C
=��(�@=q@r�\A��
Cٚ                                    BxyA��  �          @�{��=q@C�
@UA�C.��=q@�@xQ�B
=CE                                    BxyB �  �          @�{���@N{@^�RA�C@ ���@��@��B	\)C�
                                    BxyB,  �          @�
=��=q@R�\@h��A�\)CB���=q@   @�
=B��C�)                                    BxyB�  �          @�ff��(�@8��@Z=qA��C�{��(�@	��@z�HB=qC!H                                    BxyB,x  �          @�p�����@&ff@QG�A�  C�
����?��@n�RA��C!�                                    BxyB;  �          @�p���@   @I��A�z�C&f��?���@e�A�=qC"��                                    BxyBI�  T          @�ff���R@#33@G�AͅC�)���R?�\)@dz�A�C"�{                                    BxyBXj  �          @������@2�\@Y��A�C�\���@33@x��B33C�                                    BxyBg  �          @����ff@,��@dz�A�\)C����ff?�
=@�G�B��C ��                                    BxyBu�  �          @߮��(�@�R@_\)ACY���(�?��R@w�B\)C%33                                    BxyB�\  �          @�{��33@��@\(�A�{C�{��33?��H@s�
Bp�C%c�                                    BxyB�  �          @߮��=q@=q@X��A陚C����=q?�@s33B��C#@                                     BxyB��  �          @�\)��33@��@X��A���CǮ��33?�@s�
B\)C#ff                                    BxyB�N  �          @�
=��p�@@��@J�HAأ�C���p�@�
@mp�B �
C�f                                    BxyB��  �          @�{��ff@-p�@S33A�C����ff?��R@q�B33C��                                    BxyB͚  �          @�  ����@8Q�@J=qA�\)CxR����@�@k�A�(�Cp�                                    BxyB�@  �          @�
=���H@.�R@X��A�C�����H?��R@w�B�C��                                    BxyB��  �          @�p���
=@@��@UA�=qC
=��
=@G�@xQ�B	(�Cz�                                    BxyB��  �          @�z���  @R�\@.{A�p�C����  @+�@U�A뙚C��                                    BxyC2  �          @޸R��z�@5�@QG�A�CG���z�@
=@q�B\)C�
                                    BxyC�  �          @�(�����@B�\@VffA�Q�Cs3����@33@y��B
�C�3                                    BxyC%~  �          @����@*�H@h��B   C����?��@��B�
C 
=                                    BxyC4$  �          @��
���@{@S33A�{Cz����?�  @n�RB
=C"
                                    BxyCB�  �          @�z����@'
=@XQ�A�ffC����?�\)@uBG�C Ǯ                                    BxyCQp  �          @�
=��
=@:�H@a�A�z�C�=��
=@��@���B�C�3                                    BxyC`  �          @�p���\)@(�@z�HB�HC����\)?˅@��\Bz�C"@                                     BxyCn�  �          @������@A�@]p�A�{CT{���@��@�Q�B��C)                                    BxyC}b  �          @�ff���\@>{@w
=B	{Ck����\@�@�z�B  CJ=                                    BxyC�  �          @�{����@L(�@tz�Bz�C5�����@ff@�z�B{CǮ                                    BxyC��  �          @�  ��Q�@<(�@y��B�C�H��Q�@�@�p�B�RCxR                                    BxyC�T  �          @�Q���Q�@<(�@u�B�C����Q�@ff@�33B�
CB�                                    BxyC��  �          @�
=���@.�R@mp�B��C(����?�z�@�{B��C��                                    BxyCƠ  �          @�����z�@Fff@�  B��C8R��z�@{@���B&
=Cn                                    BxyC�F  �          @�����@7�@|(�B  C�H����@   @�{B   C�                                     BxyC��  �          @�ff����@L(�@k�B �\C�����@�@�  B�C.                                    BxyC�  �          @�\)��
=@H��@{�B
��C\)��
=@G�@��B"p�CQ�                                    BxyD8  �          @�����@0  @r�\B
=CW
����?�z�@���B��C.                                    BxyD�  �          @�����Q�@C�
@s33B��C�\��Q�@{@�33B�C)                                    BxyD�  �          @�\)��\)@c33@b�\A�G�C{��\)@0  @�{B��C�{                                    BxyD-*  �          @�(���@;�@n{B�CJ=��@ff@��Bz�C�                                    BxyD;�  �          @�ff����@hQ�@Mp�A�Q�Cn����@8��@x��B	�\Ck�                                    BxyDJv  �          @�p����
@c33@X��A���C����
@1G�@�G�BQ�CT{                                    BxyDY  �          @�{���@J=q@`��A��
C  ���@�@��HBC��                                    BxyDg�  �          @�
=���R@X��@]p�A�C����R@&ff@��HBz�CY�                                    BxyDvh  �          @�{��\)@H��@|��B=qCk���\)@��@�Q�B"�HC}q                                    BxyD�  T          @�
=��(�@\��@`��A��C��(�@)��@�z�B�HC��                                    BxyD��  �          @�p���G�@^{@q�B��C�)��G�@'
=@��B p�C{                                    BxyD�Z  �          @�  ���
@U@W
=A�RC�)���
@$z�@~�RBQ�CQ�                                    BxyD�   �          @�  ���R@��@eA�  C�=���R@Y��@�z�B=qC�                                    BxyD��  �          @߮�w
=@�(�@a�A�
=C���w
=@r�\@�p�B�C}q                                    BxyD�L  �          @������@�ff@q�BffC�����@Tz�@�=qB"��C\)                                    BxyD��  �          @�G����\@�(�@l��A��
C�q���\@`��@���B Q�CJ=                                    BxyD�  T          @߮����@�\)@e�A�{CE����@X��@�(�B{C�=                                    BxyD�>  �          @�  ���\@��\@S�
A��C
Y����\@S33@��HB�\CG�                                    BxyE�  �          @�\��ff@���@^�RA�33C+���ff@\��@�G�B=qC5�                                    BxyE�  T          @�����{@��H@O\)AۮC
�3��{@Tz�@���B��C�R                                    BxyE&0  T          @�G���p�@�\)@G�A�Q�C	ٚ��p�@_\)@{�B��CJ=                                    BxyE4�  �          @ᙚ����@�{@B�\A�(�C
�=����@]p�@vffBQ�C�                                    BxyEC|  �          @�  ���@\)@UA�ffCaH���@L��@�33B�
CxR                                    BxyER"  �          @�  ��{@���@;�A�z�C	�3��{@c�
@p��B\)C�{                                    BxyE`�  �          @�Q�����@�  @4z�A���C(�����@s33@mp�B \)C�3                                    BxyEon  
�          @�{���
@�33@1�A��C�{���
@z=q@l(�B �C
8R                                    BxyE~  �          @�����
@�G�@-p�A�{C����
@w�@g
=A��C
��                                    BxyE��  �          @߮���\@��H@A�\)Cp����\@�G�@Dz�A�Q�C�f                                    BxyE�`  T          @����  @��@$z�A��C�R��  @z=q@^�RA��\C	Y�                                    BxyE�  �          @�����G�@��\@
=A�Q�C�)��G�@�
=@Tz�A�\)Cu�                                    BxyE��  �          @�Q�����@��@Q�A��\C� ����@�33@Dz�A�z�Cff                                    BxyE�R  �          @ٙ���G�@�33@A��C8R��G�@p��@Mp�A�\Ch�                                    BxyE��  �          @ָR��{@���@�\A�Q�CE��{@�\)@@��A�Q�C��                                    BxyE�  �          @�ff���H@�z�@#�
A���C�����H@o\)@[�A�=qC	��                                    BxyE�D  �          @�
=��
=@N�R@���B!�RC����
=@(�@��
B<G�C��                                    BxyF�  �          @ٙ��u@C�
@�p�B.G�Ck��u?��R@�
=BI\)C��                                    BxyF�  �          @�33���@\(�@���BffC{���@(�@��B7�C@                                     BxyF6  �          @������@`��@��
Bp�C
�H����@\)@���B;  C8R                                    BxyF-�  �          @�\)��(�@k�@�Q�BffC
J=��(�@+�@�ffB4��C                                      BxyF<�  �          @�(�����@g
=@�Q�B�RC
!H����@'�@�{B7
=C                                    BxyFK(  �          @�p���p�@j�H@��B
=C
����p�@,��@��B0(�C\                                    BxyFY�  �          @�����H@a�@���B{C޸���H@%�@�{B+�RCB�                                    BxyFht  �          @�(����@N�R@�
=B�Cn���@  @�=qB2  C��                                    BxyFw  �          @љ��q�@)��@��B3��C�3�q�?���@��\BL(�C!H                                    BxyF��  �          @����`  @1G�@�
=B:\)C�H�`  ?�Q�@��RBU{CJ=                                    BxyF�f  �          @�Q��g
=@/\)@�33B5��C�{�g
=?�
=@��HBO��C�                                    BxyF�  �          @�=q�p��@:=q@��B-��CB��p��?�{@���BHffC��                                    BxyF��  
�          @�G��s33@QG�@��B��CE�s33@�\@���B<ffC�                                    BxyF�X  �          @љ��tz�@S33@��B  C+��tz�@�
@���B;C�                                    BxyF��  �          @���{�@S�
@�\)B  C���{�@�
@�33B;=qC��                                    BxyFݤ  �          @ָR�y��@mp�@~{B�Cs3�y��@0  @�B1�CǮ                                    BxyF�J  �          @ٙ��w�@a�@�Q�BG�C	�f�w�@!G�@�B;Q�C                                      BxyF��  �          @޸R���\@��@p��BffCxR���\@N�R@��\B%\)C�)                                    BxyG	�  �          @�����H@w
=@�Q�B��C�����H@8Q�@�  B.33C�=                                    BxyG<  �          @ۅ���\@��\@k�B
=C�R���\@J�H@�\)B#��C�                                    BxyG&�  �          @��H���@��H@eA��C(����@L��@���B �C�                                    BxyG5�  �          @أ���  @�(�@P  A�{CaH��  @c33@�z�B�C
W
                                    BxyGD.  �          @Ӆ�\(�@���@R�\A�ffB�aH�\(�@l��@��RB\)C�3                                    BxyGR�  �          @����U@�z�@@  A��HB��U@�33@�  Bp�C.                                    BxyGaz  �          @��
�N{@��@8��A��
B��3�N{@�
=@{�B�
B���                                    BxyGp   T          @�z��W�@�G�@qG�B
�C (��W�@U@�(�B1{C5�                                    BxyG~�  T          @љ��O\)@fff@��\B'  C�q�O\)@#33@���BJG�C��                                    BxyG�l  T          @Ӆ�Vff@z�H@�p�B��C� �Vff@8��@�B@p�C&f                                    BxyG�  �          @ۅ�j=q@�z�@�Q�B�Ck��j=q@I��@��HB4G�CJ=                                    BxyG��  �          @�ff�w�@�  @���B(�C\�w�@@  @��HB2�RCB�                                    BxyG�^  �          @����y��@tz�@�  B
=C���y��@.{@��B?=qC)                                    BxyG�  �          @�
=����@��@��B�\C������@L��@�Q�B0�\C��                                    BxyG֪  �          @��s�
@�\)@e�A��C �
�s�
@s�
@���B �C�                                    BxyG�P  �          @����r�\@���@c�
A�  C���r�\@_\)@��RB#z�C	Y�                                    BxyG��  �          @��
�u�@�=q@P��A噚C��u�@n{@��RB
=C��                                    BxyH�  �          @�Q��qG�@s�
@�z�BffC���qG�@.{@�z�B?z�C+�                                    BxyHB  �          @߮�o\)@�  @x��B

=CW
�o\)@P��@�  B/
=C
�f                                    BxyH�  �          @�p��fff@�Q�@�G�B�HC�f�fff@<(�@��\B>33C�                                    BxyH.�  T          @���Vff@�=q@���B�B����Vff@S33@��B8��C}q                                    BxyH=4  �          @��
�x��@�Q�@�\)B��C#��x��@8��@���B=Ck�                                    BxyHK�  ,          AIp���\)@��RA/�B�#�B�8R��\)@�A<��B���B���                                    BxyHZ�  �          AXz��{@��
AEB��B�Ǯ��{?�
=AQp�B�Q�C	��                                    BxyHi&  �          AZ=q�\@y��AMp�B��Bޣ׿\?\(�AV=qB��Cu�                                    BxyHw�  �          AX  ��@�ffAI�B�G�B�Q쿵?��HAT  B�  C��                                    BxyH�r  �          AMG���=q@�Q�A4  By�B����=q@A�ADQ�B���B�=                                    BxyH�  �          AD�ÿ��@�\)A�HBEp�Bʀ ���@�p�A+\)By{Bӳ3                                    BxyH��  �          A@Q�\AAffB4z�B�=q�\@�ffA!��Bi�\B�\                                    BxyH�d  �          AD  =�AA=qB'{B��==�@߮A Q�B\\)B��                                    BxyH�
  �          AB{>�33Az�@�\)B(�B���>�33@�A�RBTz�B�k�                                    BxyHϰ  T          AN�\?�A=qA�
B833B��?�@�Q�A0��Bm�B�                                    BxyH�V  �          A`  ?��A�RA=qB/(�B�G�?��@�p�A:�RBc��B�#�                                    BxyH��  �          A]�?���A�HAz�B4�B�
=?���@�z�A<(�Bi�RB��                                     BxyH��  �          Ac�?�\A=qA!�B4Q�B��R?�\@���AA��Bh��B��                                    BxyI
H  �          Ac\)?�\A"�HA�HB,�\B�8R?�\@�z�A<��BaffB��)                                    BxyI�  �          AO
=�L��AA{B:��B����L��@�p�A3
=Bp�\B��)                                    BxyI'�  �          AZff>uA(�AB;33B�  >u@�ffA<  Bq=qB��)                                    BxyI6:  �          Ak�@
=A5G�A33B\)B��@
=A�RA5��BLffB�ff                                    BxyID�  
�          A|Q�@`��A^{@�\)A�(�B���@`��A@z�A��Bz�B�p�                                    BxyIS�  �          A:�\@:=qA+\)@�\A&�RB���@:=qA�H@�  A�{B�aH                                    BxyIb,  �          A(  @e@|(��33�j�B>�@e@��
�����>=qBjG�                                    BxyIp�  |          A.=q?�Q�A@aG�A��B�8R?�Q�A  @��B�B��                                     BxyIx  T          AT(�?���A;\)@�Q�A�B��H?���A\)A	B"�B�k�                                    BxyI�  �          A[�
?z�HA4(�@��B33B�
=?z�HA=qA (�B=B�                                    BxyI��  �          AK\)��\A�RA�BAQ�B���\@�A3\)Bx�B��
                                    BxyI�j  �          AMG���AG�A�RBK33B����@��A8��B�
=B�                                    BxyI�  �          A`  ��{@�\)A@(�Bmp�B�
=��{@z�HAU�B�k�B�B�                                    BxyIȶ  �          Ae��p�@׮AI�Bu��Bȳ3��p�@b�\A]B�ǮB�ff                                    BxyI�\  �          Ad�׿У�@�ffAG�BtG�B�Q�У�@aG�A[�B�ffB垸                                    BxyI�  �          Abff����@��AR�\B��B�aH����?ǮA_33B�ǮC�)                                    BxyI��  �          A^�H�,(�@�G�AO\)B�\)B�33�,(�?�ffAYB�\C��                                    BxyJN  
�          AZff���
@��\AC
=B�=qBҞ����
@�RAR�RB�\B�B�                                    BxyJ�  �          AM��G�@�\)A8z�B�{B�aH�G�?�AEG�B�� C�=                                    BxyJ �  �          AO
=��ff@��\A>=qB�\B���ff?���AJffB���C�                                    BxyJ/@  �          AMG�����@�{A@Q�B�p�B�����?�33AJ�RB��
C��                                    BxyJ=�  T          AN�H�+�@�ffA5G�Bx�\B�녿+�@AG�AF�HB�B��                                    BxyJL�  T          Aj{�(��@�{AL(�Bx�HB�LͿ(��@X��A`(�B�Q�B��                                    BxyJ[2  �          A{\)�W
=@�{Ap(�B���B��)�W
=>���AyG�B���B��f                                    BxyJi�  �          A|(�=�G�@أ�Ac\)B�\B�{=�G�@FffAv�HB�L�B�Ǯ                                    BxyJx~  �          Ax��?k�A=qAEp�BRp�B��)?k�@��HAdQ�B�Q�B��H                                    BxyJ�$  �          Aq�@@  A/
=A�B$�B�W
@@  A
=AD  BZ=qB�                                    BxyJ��  �          Amp�@0  A4  A��B
=B�z�@0  A	�A<Q�BQ�
B��\                                    BxyJ�p  �          Ao�
@a�A9p�A
=B��B��@a�A��A4z�BD�B��\                                    BxyJ�  T          Ap��@b�\A8��A��Bz�B��H@b�\A(�A6{BFp�B�{                                    BxyJ��  �          Ak�
@�A#�A#\)B1  B�W
@�@�(�AF�RBhG�B�                                      BxyJ�b  �          A_�
@�(�A8(�@���A�(�B�.@�(�A�A��B!��B�\)                                    BxyJ�  �          AQ�@�\)A0Q�@�p�A�
=B�ff@�\)Aff@��B33Bx\)                                    BxyJ��  �          Aw
=@�  A5��A�HB33B�8R@�  A	p�AC33BP��B��                                    BxyJ�T  �          Axz�@��A�\A+�
B4  B�z�@��@�p�AM�Bg�Bl                                      BxyK
�  T          Ab{@�\)A,z�@�  B  B���@�\)A	�A\)B5�BtQ�                                    BxyK�  �          Ahz�@�=qA-�A�Bz�B�p�@�=qA�A*�HB?�HBv�                                    BxyK(F  �          A�
=@C33AG�Ab�RBX
=B��\@C33@��RA�p�B�
=B{                                    BxyK6�  �          A���@{A((�Ab�\BRB��{@{@ӅA���B�L�B�{                                    BxyKE�  �          A��
?�Q�@��A�G�B��
B��)?�Q�@.�RA�G�B��Bh�                                    BxyKT8  �          A�p�?\(�@���A�Q�B�B�\)?\(�@_\)A��B���B�8R                                    BxyKb�  �          A���>�p�A	��A�Bx(�B��>�p�@�
=A���B�L�B�\                                    BxyKq�  �          A���?�ffA�A��Bv33B�B�?�ff@��\A��B�
=B��3                                    BxyK�*  �          A��H?��A�Aw�
Bf��B�B�?��@�
=A���B�B�B�aH                                    BxyK��  �          A��?�Az�Aw�Be
=B��3?�@���A���B��B�#�                                    BxyK�v  �          A�?&ffA�\A�  Bp��B��H?&ff@���A�z�B��RB�z�                                    BxyK�  �          A�(�@ ��@�Q�A��B��RB�@ ��@4z�A�  B��HBZ                                      BxyK��  �          A���@$z�@��
A�z�B�� B��\@$z�@8��A��B���BAQ�                                    BxyK�h  �          A���@C�
Az�A~=qBq33B�z�@C�
@�p�A�Q�B�#�BV��                                    BxyK�  T          A�z�?�=q@���A�Q�B��HB�u�?�=q����A�Q�B���C���                                    BxyK�  �          A�33@�@��HA��B��{B�u�@�?@  A���B�p�A���                                    BxyK�Z  �          A�Q�@�A�\Ab�RBS33B�8R@�@��A�  B��BZ�H                                    BxyL   �          A�\)@�ffAQ�AeBT�Bz
=@�ff@�\)A���B�W
B?�                                    BxyL�  �          A���@�\)Az�Aep�BV�Bj�@�\)@�  A��B�B)p�                                    BxyL!L  �          A�  @�
=Az�A[�BJ�RBg�R@�
=@�(�A{�By��B-\)                                    BxyL/�  �          A�Q�@��A��APz�BDp�Bg��@��@��HAq�Bs�HB1�                                    BxyL>�  �          A~�\@�z�A"�RA$  B ��Bc�@�z�@�z�AH��BP��B<p�                                    BxyLM>  �          A{33@�=qA7
=@���A�(�Ba�@�=qA�A$(�B#�BHz�                                    BxyL[�  �          A�z�@��A.�RA�Bz�B_ff@��@��AC�
BAQ�B<��                                    BxyLj�  �          A�=q@�  A.�RA!�B33Ba�R@�  @�z�AIp�BFffB=�R                                    BxyLy0  �          A�z�@�A��AUBGG�Bi33@�@��Av�RBw\)B0\)                                    BxyL��  ?          A��@�ffAAr{Be33Bm�\@�ff@�=qA�ffB�B�B��                                    BxyL�|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxyL�"  s          A��
@�\)AffAp(�BYp�Bs�R@�\)@��HA��B�p�B0��                                    BxyL��  �          A��@�33AffAs33B]�Bs�@�33@�G�A���B�u�B,                                    BxyL�n  �          A�p�@�  @�Q�A��Bw�
BlG�@�  @7
=A���B��RB�H                                    BxyL�  �          A��R@��
@�\)A��Bw�Bi=q@��
@5A�Q�B��A���                                    BxyLߺ  �          A��@��@��HA�p�Bt33Bh  @��@?\)A���B��RB �
                                    BxyL�`  �          A��@�=q@���A�B}�
Bk��@�=q@=qA�=qB�W
A�p�                                    BxyL�  �          A��H@��@�(�A�(�B�ǮBk\)@��@�A�  B��fA�Q�                                    BxyM�  �          A���@x��@�
=A�33B���Bo��@x��?�
=A��RB�AҸR                                    BxyMR  �          A�\)@|��@�33A�B���Blp�@|��?���A�
=B���A�z�                                    BxyM(�  T          A�@�  @޸RA�Bu�BS@�  @Q�A�{B�
=A���                                    BxyM7�  �          A��\@�=q@�{A�z�B{G�BY
=@�=q@A�=qB�.A��                                    BxyMFD  �          A�33@���@�(�A�
=B���Bj�R@���@   A��\B��)A�33                                    BxyMT�  
�          A��
@n�R@��A�Q�B���Bh33@n�R?�G�A�Q�B��A�\)                                    BxyMc�  �          A�Q�@n{@�  A��B�p�B_��@n{?^�RA��RB�ǮAR�\                                    BxyMr6  �          A�z�@�{@�ffAw�
B~Q�Bd{@�{?�p�A�G�B��A�(�                                    BxyM��  �          A��H@��
A z�AX��BMQ�B�Q�@��
@�A|z�B�ǮB\�H                                    BxyM��  �          A���@��\A!�AQp�BG�RB��q@��\@�z�Au�BB[ff                                    BxyM�(  �          A��@n�R@�(�A}G�B~�RBv�@n�R@��A��HB�aHA��                                    BxyM��  �          A���@{�@���Aj=qBm  B|��@{�@[�A��B�L�B$p�                                    BxyM�t  �          A���@��A��AM�BF33Bz�@��@�z�Ap��B|��BJ�                                    BxyM�  �          A�33@��A ��AMp�BF�RB�(�@��@��HAq�BffB\�                                    BxyM��  �          A���@U�@���Aw�
B~�B�� @U�@�A�ffB�ǮB{                                    BxyM�f  �          A�33@�=q@�p�At��BqBx=q@�=q@G
=A���B�.B�\                                    BxyM�  �          A��@�33A�Ac�BZp�Bn@�33@��A��RB��B#�                                    BxyN�  
�          A��
@��HA(�AV�\BQ\)Br(�@��H@��Av=qB���B/                                    BxyNX  �          A��\@��RA4z�A)p�B$p�B���@��R@��
AUG�B_Q�Bq�R                                    BxyN!�  �          A�(�@Mp�AD��A%��B��B��R@Mp�AffAV=qBZ��B�L�                                    BxyN0�  �          A�G�@�  A{A@��BC�B�Q�@�  @�33Adz�B|�\BW�H                                    BxyN?J  �          A��
@��A�
Ao
=Ba33B_
=@��@l(�A�G�B��B�                                    BxyNM�  �          A��H@�=q@��A{
=Bp��BW=q@�=q@"�\A���B���A�Q�                                    BxyN\�  �          A�{@��@�\)A���Bz�RBb(�@��@
=A�{B��3Aď\                                    BxyNk<  �          A���@�33@�=qA�ffB��BW�
@�33?��A�G�B��A{33                                    BxyNy�  �          A��@�33@��A�G�B�(�BT@�33?0��A��HB���AQ�                                    BxyN��  �          A�G�@��R@���A�Q�B�\B;��@��R�#�
A�  B�\C��                                    BxyN�.  �          A��@��H@��
A���B���B:��@��H�\)A�Q�B���C�)                                    BxyN��  T          A�z�@��HA33Af�HBX�HBy{@��H@�{A��B�z�B0ff                                    BxyN�z  �          A�z�@vff@S33A�(�B�Q�B"Q�@vff��ffA��B��3C�xR                                    BxyN�   �          A�G�?�\)?�A�{B��A{�
?�\)��33A�G�B��RC��                                    BxyN��  �          A�=q?��
@��A�G�B�k�B�Ǯ?��
����A��RB��\C���                                    BxyN�l  �          A�ff@:=q@���A33B{p�B�\)@:=q@-p�A��
B�aHB,
=                                    BxyN�  �          A���@���@�
=A
=B}��Beff@���?���A���B���A���                                    BxyN��  �          A���@a�@��Az{Bw��B���@a�@,��A�33B��B�
                                    BxyO^  �          A���@S�
A�Ak�
Bm  B���@S�
@^�RA��B�u�B9�
                                    BxyO  �          A�  @�A`��@ᙚAǅB�
=@�A6ffA-p�B!�B~�                                    BxyO)�  �          A�
=@��A=�A!��Bz�B���@��A��AQBQ��B_�                                    BxyO8P  �          A�\)@?\)A
=AR�RBQ�\B��q@?\)@��Av�HB�Q�Btp�                                    BxyOF�  �          A��\@���A'\)A@  B9�HB�=q@���@�\)Ah��Bu�HB\ff                                    BxyOU�  �          A��\@�A<(�A&�\B{B�k�@�A�HAV�\BX�Bf�                                    BxyOdB  �          A|��@�  AC�A	��B33B��\@�  A�\A=�BBG�Bu�                                    BxyOr�  �          A�
A;�
A-p�?B�\@-p�B*�A;�
A"{@z�HAc33B#=q                                    BxyO��  �          A���AMp�A33@5�A�BQ�AMp�A�H@�=qA���B33                                    BxyO�4  �          A��A   AHz�?�p�@��BM��A   A733@��
A���BCz�                                    BxyO��  �          A�
=AG�AO�@��A
=qB\  AG�A:�R@���A��RBPp�                                    BxyO��  �          A~=q@�RA<��@�(�A�  Bf�@�RA\)A!G�B!�\BK�H                                    BxyO�&  �          A��H@�G�A�
A2�\B,��BDff@�G�@�  AUG�B[Q�B�                                    BxyO��  �          A�A ��A�\A*�\B"�BF
=A ��@�G�AO�BR��B�                                    BxyO�r  �          A�Ap�AK�@�\)A��Bc(�Ap�A)G�A��B�RBO
=                                    BxyO�  �          A~{@��\A8Q�@�{A���B_(�@��\A  A(��B'�
B@�                                    BxyO��  �          A}�@��
AF�H@�
=A��Bm\)@��
AffA�\B�BUff                                    BxyPd  �          At��@�
=A��@�=qA���BW�R@�
=@��H@�
=Bp�B=                                    BxyP
  h          Ab=q@�ffA(Q���\)���
Bi�H@�ffA9����  Bs�R                                    BxyP"�            ATz�@���A,  �У���G�Be33@���A+�
?ٙ�@��HBe�                                    BxyP1V  �          AN=qA��A�?�G�@�{BH�RA��A�@���A�z�B>p�                                    BxyP?�  �          APz�A=qA��@1G�AD  BCp�A=qA��@��A�B2�
                                    BxyPN�  �          AL��@���A(z�@J=qAeG�Bh�@���A��@�\)A�
=BX                                    BxyP]H  �          ANff@�p�A,(�@C33AZ�RBl��@�p�A��@�{A�B]                                    BxyPk�  �          AO
=@�=qA{@��\A�=qBb�
@�=q@�@���BBI�                                    BxyPz�  �          AJ�R@�(�A��@�A���BZ33@�(�@�\@��HB�RBCG�                                    BxyP�:  �          AH��@�ffA(Q�>\?�  Bf
=@�ffA=q@fffA�=qB_�\                                    BxyP��  �          AK33@љ�A,z῔z���  Bj��@љ�A)@{A!��Bi33                                    BxyP��  �          AZff@��A:{@
=A�\B{  @��A&�\@�33A�  BpQ�                                    BxyP�,  �          AtQ�@��HAP(��j=q�ap�B|��@��HAX(�?��@ffB�                                      BxyP��  �          Au@���A{�"=q�2��B{�@���A=��ٙ���G�B�
=                                    BxyP�x  �          Au�>�p��0���tQ�®#�C�>�>�p�@�33�k�ǮB���                                    BxyP�  �          A��?�33?G���ff«{B�H?�33@�G���\)  B�Q�                                    BxyP��  �          A��@��@�z����Bx��@��A�e�]�B�aH                                    BxyP�j  �          A�=q@�
=A���o
=�^G�Bmz�@�
=AP���6�R�p�B�ff                                    BxyQ  |          A�p�@��\A\)�J�H�9p�BF�\@��\AK��ff��Bi��                                    BxyQ�  �          A��A   A8  �
=q��ffBC�HA   AZ{���H�r�RBV�                                    BxyQ*\  �          A��
@��\A�R�yG��^p�Bc=q@��\AV{�?��ffB��)                                    BxyQ9  �          A�\)@��R@�{��{ffBa�@��RA;��`z��@��B�L�                                    BxyQG�  �          A���@H��@�\)������Bl=q@H��A#��ap��S=qB��H                                    BxyQVN  �          A�33@�{@������(�Bi�@�{A<���\���?�B��)                                    BxyQd�  �          A��
@q�@�����R
=BW=q@q�A*ff�x(��W�B��                                    BxyQs�  �          A��R@R�\@�Q������HBb�H@R�\A'
=�uG��Y�
B�                                    BxyQ�@  �          A�{@e@�p�����k�Bb�@eA.�\�vff�U{B���                                    BxyQ��  �          A�Q�?��@vff��(��{B=q?��A  �����kB�p�                                    BxyQ��  T          A�
=?��@QG���Q�ffB�
=?��A=q��Q��s\)B�aH                                    BxyQ�2  �          A���?�=q@�z���
=�=B��=?�=qA ���f�\�[\)B��f                                    BxyQ��  �          A����@N{������B�Lͽ�A���rff�r�B���                                    BxyQ�~  �          A�33��?����Q�©G�Bٳ3��@�G��}�\B��                                    BxyQ�$  �          A���}p�@Q���G�¤.B��}p�@��t���  B��                                    BxyQ��  �          A�녿��\��=q��¥�CgB����\@u����ǮBؔ{                                    BxyQ�p  �          A�p��|(��A��up�\CY���|(�@��w�
ǮC�=                                    BxyR  |          A���@(�@\)�q���{Bj  @(�A
=�O��\B��{                                    BxyR�  �          A��@���A{�s
=�P�HBd\)@���Ad���2{���B���                                    BxyR#b  �          A���@�G�A�\�����b(�Bk=q@�G�A_33�D���p�B���                                    BxyR2  �          A�z�@�p�A�����v�HBu�H@�p�AT���YG��0Q�B�G�                                    BxyR@�  h          A��H@�(�A\)��z��sG�Bzp�@�(�AYp��UG��,(�B�B�                                    BxyROT  �          A�A=qAZ{?�G�@��\BZ�A=qAG33@�A��\BQ
=                                    BxyR]�  �          A���A%�A`Q�@�@�
=BV�A%�AG�@���A��BI��                                    BxyRl�  �          A��A"�\Au������
=Bb
=A"�\At(�@7�AffBaff                                    BxyR{F  �          A�33AffAy������f=qBfp�AffA�Q�?0��@�Bj�H                                    BxyR��  �          A��RA{AqG���\)��G�Bc{A{A�Q�.{�   Bi�                                    BxyR��  �          A�
=Ap�Ag\)�ڏ\��Ba�
Ap�A~�\�У���Q�Bk�                                    BxyR�8  T          A��A#�Ab=q��=q��  BX�A#�A|Q��(���z�Bd(�                                    BxyR��  �          A��\A33A]G����R��(�B[�
A33Ay�(���
=Bh�\                                    BxyRĄ  �          A���@�G�A]��7��(�B�ff@�G�A��H��{��
=B�
=                                    BxyR�*  �          A���?��HAX  �Jff�+��B���?��HA����z����B��                                    BxyR��  �          A��
@b�\AG��IG��0�\B�L�@b�\A�����ȣ�B�                                      BxyR�v  �          A�Q�@uA9���M�9�HB�\)@uAt(��\)��ffB��q                                    BxyR�  �          A�ff@=p�A3��V�\�Dp�B�k�@=p�Aqp��p���
=B�Ǯ                                    BxyS�  �          A�
=?�A)�YG��NffB���?�Ai��
=� (�B��)                                    BxySh  �          A��R?��A3��W�
�Hz�B�=q?��Ar{�=q��
=B�                                      BxyS+  �          A�(�?��\A(���X(��O�\B��?��\Ah(��{� p�B��                                    BxyS9�  �          Au@��\AIG��������
B�G�@��\A]녿�����  B�p�                                    BxySHZ  |          Alz�@���AC\)�Ǯ�\Bdff@���A9@s33Ar=qB_{                                    BxySW   �          A?�@�\)@�@�  B��B.�
@�\)@x��AB?A�R                                    BxySe�  �          AD��@�z�@���@�RB�B2=q@�z�@�z�A�HB@33A���                                    BxyStL  �          AG
=@�  @�@ə�A�(�B5@�  @��Az�B-�RB��                                    BxyS��  @          AI@�>�33ABffB��RA�@��vffA8��B���C��\                                    BxyS��  |          Ac�
@x��?��
AW33B�� A��@x���L��ARffB��C�G�                                    BxyS�>  T          As
=@�z�@y��AV�HBy{B
=@�z�z�HA_33B��C�K�                                    BxyS��  �          Ah��@�p�@��A/33BC�\B%ff@�p�@��AG33Bl  A���                                    BxyS��  �          Ae�@�R@љ�A'
=B;��B(��@�R@!G�AA�Bf��A�Q�                                    BxyS�0  �          A`z�@�  @�
=A#
=B:p�BUQ�@�  @mp�AC�
Bs(�B33                                    BxyS��  T          Ab�\@��A  A��B0(�Bz��@��@���AE�Bs��B<�
                                    BxyS�|  �          A33A�@�(�A$(�B (�B�RA�@UAC
=BG\)A���                                    BxyS�"  �          A|Q�A33A��A{B33B(=qA33@��\ABffBHQ�Aə�                                    BxyT�  �          A|Q�Az�AG�A(�B
=B7��Az�@�A7�B<ffB ��                                    BxyTn  �          A�33A��A(�A!��B��B0z�A��@�p�AHz�BKA�z�                                    BxyT$  T          A�33A(�A�A'�B"��B)�A(�@���AK\)BP�A�z�                                    BxyT2�  �          A��A  A�A ��B��B/ffA  @�\)AK�BC�HA�                                    BxyTA`  �          A�=qA�A
�RA-�BQ�B&�
A�@���AS�BL�A��                                    BxyTP  �          A�Q�A��A�\A(Q�BG�B+�A��@��
AO�
BJ�A�(�                                    BxyT^�  �          A��\A�
A
�\A3�
B$��B)��A�
@��
AYG�BR�HA��                                    BxyTmR  �          A�33A�A��A/33B�B4�HA�@���A[\)BLp�A��
                                    BxyT{�  �          A��\A#�A(Q�A{B�B7�A#�@���AK�
B:33B (�                                    BxyT��  T          A�(�A,(�A'�
A33B ��B1{A,(�@�{AI�B4=qA�
=                                    BxyT�D  �          A�ffA0z�A*=qA��A�=qB/A0z�@ָRAC�
B-�HA�Q�                                    BxyT��  �          A���A.�HA7\)@�\A���B9ffA.�HA�RA+\)Bz�B{                                    BxyT��  �          A���A)�A0Q�A33A�  B8�HA)�@�33A9B(33BQ�                                    BxyT�6  �          A�  A%��A,Q�A{A���B8�\A%��@أ�AF=qB4=qB��                                    BxyT��  �          A�Q�A#\)A"�HA33B
z�B3��A#\)@�p�AO33B>��A��                                    BxyT�  �          A�33A(�A%p�A=qB=qB:��A(�@\AO33BA�\A�p�                                    BxyT�(  �          A��A#�
A"{A*=qBB2�A#�
@�33AYG�BE=qA�\)                                    BxyT��  T          A�ffA.ffA/�A��A�
=B4A.ff@陚A8Q�B%�B=q                                    BxyUt  �          Av{A��A)�@���A�33B?��A��@�Q�A��B��B��                                    BxyU  �          A���A
=A<z�@��HA�\)BG\)A
=A�Az�BB+\)                                    BxyU+�  
�          A��A7\)AK33@��
A��B?�RA7\)A��A%�B
z�B��                                    BxyU:f  T          A���A:�RAP��@��A�G�B@�RA:�RA"�HA��B33B$Q�                                    BxyUI  �          A���A8z�AR�R@��A���BC33A8z�A#�A�
B�RB&Q�                                    BxyUW�  �          A��
A7\)AP��@�=qA�Q�BB�A7\)A!A�Bp�B%�                                    BxyUfX  �          A��A2ffAO�@��A�ffBE=qA2ffA ��A�B33B(�                                    BxyUt�  �          A��HA5p�A<  @�  A��B8{A5p�A��A%Bp�B��                                    BxyU��  �          A�
=A3\)A<��@�G�A�33B9��A3\)A�A*ffB�RB33                                    BxyU�J  �          A���A,��A;\)@��
AƸRB=Q�A,��A�\A2�RB(�Bp�                                    BxyU��  �          A�33A-�A!��A�A��HB,{A-�@���A3�
B'�A��H                                    BxyU��  �          A��RA#�
A{AQ�B��B,�A#�
@�Q�AB�HB9G�A�Q�                                    BxyU�<  �          A�33A'
=A$��@�z�A�  B2�A'
=@ڏ\A+
=B"\)B                                    BxyU��  �          A�33A#33A[
=@Z�HA5�BU33A#33A7
=A(�A�\BA{                                    BxyUۈ  �          A��
A%�A[�@,(�A=qBSA%�A<  @�A�33BB33                                    BxyU�.  
�          A�G�Ap�AS�
@.�RAG�B[33Ap�A4(�@�
=A�z�BIQ�                                    BxyU��  �          A��
A��A<  @��A���BH��A��A
{A (�B��B%�                                    BxyVz  �          A~ffA  A33A=qB (�B3G�A  @��
A4��B6A���                                    BxyV   �          A~�HA'�A�A  A�BA'�@�Q�A-�B,�A�\)                                    BxyV$�  �          A�
=A,(�Aff@���A�Q�B*�A,(�@��HA�B
�B��                                    BxyV3l  �          A��RAC
=A/
=@y��AV=qB'�\AC
=A
�R@�\)Aڏ\B��                                    BxyVB  �          A�(�A[�Aff@;�A%G�A��A[�@�p�@�(�A��\A�G�                                    BxyVP�  �          A���As
=@��R@�\)A��HA��HAs
=@S�
@�  A��AD��                                    BxyV_^  �          A��\Ae��@�
=A
=qA��A��HAe��?�=qA{B	�R@���                                    BxyVn  �          A�p�Ab�H@��HAAޣ�A�z�Ab�H@X��A&�HBQ�AV�H                                    BxyV|�  �          A���Aa��@��A��A�=qAݮAa��@l��A'�
B  Ak33                                    BxyV�P  �          A���AW\)A@�A�G�Bp�AW\)@��A'33B�A��                                    BxyV��  T          A���AIG�A(��@޸RA��B �AIG�@�A'\)B��A��                                    BxyV��  �          A�Q�AJ=qA9@�G�A�33B*G�AJ=qA	A��B   B	�                                    BxyV�B  T          A�33A3
=A@��@�
=A�{B<��A3
=Ap�A�A�
=Bff                                    BxyV��  �          A�Q�A&=qA,  A�A�
=B7�HA&=q@ڏ\A9G�B+��BQ�                                    BxyVԎ  �          A�A,  A?\)@��A�\)B@=qA,  A  A!�B�\B�\                                    BxyV�4  �          A��A)AK�@�\)A�z�BH�RA)A{A33B�B)                                      BxyV��  �          A���A+\)A?
=@У�A���B@p�A+\)A(�A)�BG�B�H                                    BxyW �  �          A���A!p�A6�R@�p�AծBB�A!p�@�\A7�B)33B�\                                    BxyW&  �          A�G�A%�A0Q�A
=A��
B;�A%�@�  A<��B-��B�\                                    BxyW�  �          A�A&�HA#\)A
�RA���B1�A&�H@��A?
=B2�HA��                                    BxyW,r  �          A�\)A#33A,z�@�G�AׅB:ffA#33@�Q�A2=qB'�B

=                                    BxyW;  �          A���A*�\A�A�A���B'��A*�\@�z�A=p�B2��A֏\                                    BxyWI�  T          A�=qA2�HA#�@�Q�A���B)�
A2�H@��HA*�RB�A�Q�                                    BxyWXd  T          A���A,  A!�A{A��B-(�A,  @���A6�HB*�
A�ff                                    BxyWg
  
�          A�z�AH��A&�\?�p�@�B�HAH��A(�@�A���B��                                    BxyWu�  �          A��AL(�A(��@X��A9p�BQ�AL(�A{@�  A�G�B(�                                    BxyW�V  �          A���AI��A*{?�33@ҏ\B ��AI��Az�@�p�A��B�\                                    BxyW��  �          A�\)AL��A+�
@�RA33B�HAL��Aff@�  A���B(�                                    BxyW��  �          A�p�AO\)A*{@(�A��BffAO\)A��@�p�A�Q�B�
                                    BxyW�H  �          A�ffA:{A0��@�{A���B.
=A:{A33A=qA��B                                    BxyW��  �          A�ffAH  A!�@(�@�Q�B�AH  A�H@�Q�A�B��                                    BxyW͔  �          A���A^{A
=q�����˅A��A^{A
ff?޸R@�=qA��                                    BxyW�:  �          A�
AK\)A�����BffAK\)Az�@U�A@��B��                                    BxyW��  �          A{33AS
=A{��=q��  BAS
=@���@Dz�A5��A��H                                    BxyW��  �          A�
=AO�A�
@{@�G�B�AO�A ��@���A�
=A���                                    BxyX,  
�          A���AO�A�@)��AG�B=qAO�@���@�Q�A��A�p�                                    BxyX�  
�          A��HAPQ�A�
@7
=A ��BffAPQ�@�=q@��A�ffA�\)                                    BxyX%x  �          A�Q�AN�\Az�@1G�Az�BAN�\@�z�@��HA�\)A���                                    BxyX4  �          A�=qAK�
A��@8Q�A%�B\)AK�
@�@˅A���A�{                                    BxyXB�  �          A��HAL(�A��@
�H@�  BffAL(�@�z�@�=qA���A��                                    BxyXQj  �          A���AQG�A��@%A�
B
��AQG�@�  @�  A���A�(�                                    BxyX`  �          A���AFffAp�@=p�A)�B�RAFff@�=q@��A�p�B                                      BxyXn�  �          A33A?33A!��@EA2�HB �A?33A (�@���A���BQ�                                    BxyX}\  �          A�
ABffA�R@<(�A)B�ABff@�z�@�{A�=qB
=                                    BxyX�  �          A\)A;�A$  @`��AK\)B$�A;�@��R@�33A�p�B�R                                    BxyX��  �          A}G�A5�A$��@���Al(�B)  A5�@���@�33A��
B	�                                    BxyX�N  �          A�Q�A@(�A(Q�@@��A*�HB$�HA@(�Aff@�Q�Ạ�B�H                                    BxyX��  �          A��\AA��A/
=@*�HAz�B(ffAA��A�R@��
Aď\B�                                    BxyXƚ  T          A��HA@Q�A2{@,(�A�B+�A@Q�A�@�\)A��HB(�                                    BxyX�@  �          A���A@��A3�@,��A��B+��A@��Aff@���A�G�B�
                                    BxyX��  T          A�z�AB�HA,��@Dz�A+
=B&=qAB�HA	��@�ffA�Q�B                                      BxyX�  �          A���AF{A+�@2�\A�\B#�\AF{A
ff@�p�A�
=B                                    BxyY2  �          A�
=AH��A*ff@�HA�B!G�AH��A�@��A��BG�                                    BxyY�  �          A��AH  A*�H@��A�B"
=AH  A�
@ӅA�\)B�
                                    BxyY~  �          A�=qAH��A&{@7�A�
Bp�AH��A��@��
Aď\B�
                                    BxyY-$  �          A�p�AB�HA5�?�@��
B+��AB�HA�
@�{A��HB��                                    BxyY;�  �          A��AJ{A�
@��A   BffAJ{A�H@�p�A�z�B�                                    BxyYJp  �          A�G�AG\)A=q@�G�Ahz�B
=AG\)@�z�@�Aߙ�A癚                                    BxyYY  �          A��
A@(�A�@��A���B
=A@(�@�(�A��BG�A���                                    BxyYg�  �          A}G�A@��A�@�33A�p�B�\A@��@��@��RA�{A�{                                    BxyYvb  �          A|  AEp�Ap�@�  Az{Bz�AEp�@�G�@���A�(�A�{                                    BxyY�  �          A{�
AAG�A33@��
Ar�RB{AAG�@��@��A�A���                                    BxyY��  �          A|��A3
=A%p�@�  Ak\)B*�HA3
=@��@�A��
B
�                                    BxyY�T  
�          A{\)A*�HA1G�@G�A7\)B833A*�HAz�@�ffA�ffB�                                    BxyY��  �          A}�A,z�A3�
@1G�A!��B8��A,z�A��@�ffAظRB =q                                    BxyY��  �          A~�HA,  A8Q�@��A�\B;�A,  Az�@ۅA�(�B&=q                                    BxyY�F  �          AtQ�AG�A8z�>W
=?L��BI(�AG�A&=q@�Q�A�{B=Q�                                    BxyY��  h          Ak
=@�z�A=G��w��z{Bkz�@�z�AE?�Q�@�  Bo�                                    BxyY�  �          Ap��A�A>{�g��_�B[ffA�AD��?�Q�@ϮB_(�                                    BxyY�8  �          AR�H@�(�A1���p����
B���@�(�AE���\)���HB��
                                    BxyZ�  �          A4��@5A��������
B���@5A.�H������  B���                                    BxyZ�  �          AB=q@|��A#�
��ff���HB�Ǯ@|��A7���녿�Q�B�                                      BxyZ&*  �          AC�@���A#33��ff���B�W
@���A7
=��녿�
=B��
                                    BxyZ4�  �          A:�\@�z�A�
��Q����B��@�z�A+33������{B�W
                                    BxyZCv  �          A=@���A�\��33��33B�ff@���A0  ��=q��ffB���                                    BxyZR  �          AB�H@�\)A����H����B��@�\)A4z῕����B��                                    BxyZ`�  �          A;�@�  A33��=q��Bz��@�  A)G��k�����B��3                                    BxyZoh  �          AV�H@�z�A.{������  Bn=q@�z�A:�R?=p�@K�Bu�                                    BxyZ~  �          A]p�@��HA3���{��p�Bj��@��HA?33?z�H@��Bp��                                    BxyZ��  |          AZ{@��A.�\�s33���Bb�@��A7�
?���@��\BhQ�                                    BxyZ�Z  �          Ac33@�RA2�\��=q��Q�B`��@�RA@��?�R@!�Bh�H                                    BxyZ�   �          A_�
@��A)G���G����BUp�@��A8  >��?�Q�B^�                                    BxyZ��  �          AS33@�ffA!��1G��B�\BN��@�ffA$��?�ff@���BQ(�                                    BxyZ�L  �          AP��A��A���������RB7��A��A
=@>�RAS
=B2�\                                    BxyZ��  �          AG33A	�A�?�\)@�(�B5\)A	�@�G�@�\)A�Q�B �                                    BxyZ�  �          AI��AQ�Ap�?�R@6ffBA��AQ�A�R@�=qA��B2��                                    BxyZ�>  �          AP��@�A%녿�����BV�@�A"�R@0��AD(�BT��                                    Bxy[�  T          AV�\@��
A1��������
Bap�@��
A*�H@Tz�Aep�B]��                                    Bxy[�  �          AZ�R@�Q�A-�G��RffBY��@�Q�A2=q?�\)@�=qB\�\                                    Bxy[0  �          AQ��@�A  ���
��p�Bbz�@�A5G��������Bq                                    Bxy[-�  �          AQ�@���A z���z���  Bg��@���A7��B�\�S�
BuG�                                    Bxy[<|  �          AO\)@�G�A ���������
Bk�H@�G�A733�!G��3�
Bx��                                    Bxy[K"  �          AO
=@θRA!����p���z�Be�\@θRA3�
�u���Bp\)                                    Bxy[Y�  �          AS
=@��A  ��\)��Bhp�@��A7�������=qBx��                                    Bxy[hn  
�          AR{@�(�A �����Q�BR��@�(�A.=q��G���G�Br�                                    Bxy[w  �          AR�H@�z�A
=��
=��HB_z�@�z�A4���Y���o33By��                                    Bxy[��  �          AW�
@���A�������p�Bf��@���A<���;��H  B}�R                                    Bxy[�`  �          AU�@��Ap�������BSz�@��A1��tz���33Bq��                                    Bxy[�  �          AO�
@�{A!���z���{Be�@�{A3
=�#�
��BpQ�                                    Bxy[��  �          ANff@�{A���
=��\)Bdz�@�{A3���Q�����Bt�                                    Bxy[�R  �          APQ�@��A
=������Bx�@��A:�R����+�B��                                    Bxy[��  �          AT  @��
A33�����z�B{  @��
A?33�=p��N�\B��3                                    Bxy[ݞ  �          AV=q@��HA{��  �p�Bv�\@��HA@�����=qB�G�                                    Bxy[�D  �          AV�H@��A z���� ��By�R@��AB=q��p��z�B�Q�                                    Bxy[��  �          AR{@�
=A!���G���=qBq�@�
=A;
=�}p����B�                                    Bxy\	�  �          AU@�z�A$  ���H��33Bx��@�z�AA����33����B���                                    Bxy\6  �          AU@�p�A.�\�qG���p�BjG�@�p�A7\)?�
=@���Bo33                                    Bxy\&�  �          AXQ�@ȣ�A2{�������Brff@ȣ�A>�\?�  @�Q�Bx��                                    Bxy\5�  �          A=G�@��A���N{��\)Bs�R@��A   ?��@�
=Bx
=                                    Bxy\D(  �          A=p�@��HA�R����B�.@��HA*{>��R?��B�B�                                    Bxy\R�  �          A2ff@��HAQ���  ����B�p�@��HA$��?!G�@P��B���                                    Bxy\at  �          A6ff@���A�R��(���  Bz
=@���A$Q�>�@=qB�                                      Bxy\p  �          A;�@�Q�A�
�(���PQ�Btz�@�Q�A!�@z�A#33Bu                                    Bxy\~�  "          AF�\@�z�A,z��33��(�BuQ�@�z�A&=q@Tz�Ax��Bq�R                                    Bxy\�f  �          AC
=@�ffA�R������33B�B�@�ffA3�������
=B��=                                    Bxy\�  �          AG33@��RA#\)���R���HB�33@��RA5��=�G�?�B���                                    Bxy\��  �          AF�R@�
=A#\)�s�
����Bn��@�
=A-G�?�33@��Bt�                                    Bxy\�X  �          A>{@��HAp��0  �U��Bi
=@��HA z�?�Q�AffBk                                      Bxy\��  �          A;
=@�z�A���Mp��\)BiQ�@�z�A�?�33@��Bm                                    Bxy\֤  �          A>=q@�\)A(�������=qBd�@�\)A$  >W
=?��\Bn�                                    Bxy\�J  �          A?�
@�G�A
�H�ə���=qBj��@�G�A)G���p��p�B}�
                                    Bxy\��  �          A:ff@��@�����"\)Bb\)@��A��\����{B�\                                    Bxy]�  �          A5�@���@�z���p��+��BVQ�@���A{�w
=��z�By�                                    Bxy]<  �          A6=q@�  @��
�����
=BO(�@�  A\)�����{Bf�\                                    Bxy]�  �          A1G�@��\@�(���{�(�BF�
@��\A���)���]p�Bd                                    Bxy].�  �          A3�@�ffA�
�mp����B[��@�ffA  ?�@,(�Be{                                    Bxy]=.  "          A.ff@�z�@��H�����33BO��@�z�A�
������B^�R                                    Bxy]K�  �          A&�H@�
=@�(���\)��=qB-�@�
=@��\��=q���
BEQ�                                    Bxy]Zz  �          A"�\@�p�@�{�k���(�B4Q�@�p�@�(���  ��BC\)                                    Bxy]i   �          A"{@��
@��
�j=q����B<��@��
@��þ���W
=BJ��                                    Bxy]w�  �          A\)@���@�ff�J=q��ffB9ff@���@�=���?#�
BE�                                    Bxy]�l  �          Ap�@�
=@�ff�^{���BN{@�
=@��>.{?��\BY�\                                    Bxy]�  �          A�\@�Q�@�Q��w
=��p�BS33@�Q�A����E�B`��                                    Bxy]��  �          A (�@�\)@�=q�~{���RB]\)@�\)A�ͽ�Q��Bj{                                    Bxy]�^  �          A%��@�(�Ap��I�����B\�R@�(�A
{?aG�@���Bc��                                    Bxy]�  �          A(��@�A���Q��)�Bi�@�A�
@�\AI�Bh�                                    Bxy]Ϫ  �          A+\)@ҏ\A z��p��@��BJ��@ҏ\Aff?ٙ�A
=BLQ�                                    Bxy]�P  |          A/33@��@��Q��I�B.33@��@�z�?�  @�G�B2ff                                    Bxy]��  �          A4��A33@�z�>\@�B �A33@�ff@_\)A�  B�H                                    Bxy]��  �          A/
=A@ᙚ?�=qA�HB$
=A@�@��A�B=q                                    Bxy^
B  �          A6�RA\)@߮?h��@��B�
A\)@���@�Q�A���B��                                    Bxy^�  �          A4  A�@�ff�u���HB
��A�@��H@0  A`Q�B 33                                    Bxy^'�  �          A(z�@�ff@�녿�(���{B1�H@�ff@ᙚ@G�AG�B-��                                    Bxy^64  �          A%p�@�  @ٙ���z��+33B(�
@�  @�p�?�
=@�
=B*��                                    Bxy^D�  �          A"=q@�=q@�G������θRB#\)@�=q@ʏ\@ ��A7�
B��                                    Bxy^S�  �          A=q@�z�@�p���  ��B,33@�z�@Ӆ?޸RA"�RB+(�                                    Bxy^b&  �          A�H@�  @�33�c33��{B_(�@�  A�\>u?��HBj
=                                    Bxy^p�  �          A (�@s33A���p��ŮB�p�@s33A�=���?z�B�=q                                    Bxy^r  �          A-��@�\)A33�}p���Bd=q@�\)AG�>Ǯ@BnQ�                                    Bxy^�  �          A.�R@���A	���{���HBi@���A��>�  ?�=qBtff                                    Bxy^��  �          A/\)@��A����ff��ffBq�@��A�>�33?�B{p�                                    Bxy^�d  �          A2�R@�\)A	��x����p�B\ff@�\)A=q?�@,��Bf(�                                    Bxy^�
  �          A5@��
A  ��{��p�B`�\@��
A
=>�33?�\Bk33                                    Bxy^Ȱ  �          A7\)@���A
=�qG���=qB`G�@���A�R?Q�@�33Bhp�                                    Bxy^�V  �          A0Q�@���A���z��-�BY��@���A  @��A=BY33                                    Bxy^��  |          A�R@�(�@��?�p�@��B�@�(�@�(�@}p�A��HBff                                    Bxy^��  �          A�@���@�?�33A�HB�H@���@{�@i��A�A�p�                                    Bxy_H  �          A33@�(�@���?u@�BG�@�(�@�ff@[�A��A�                                    Bxy_�  �          A�@�ff@��R?�  @��B	�@�ff@��@[�A��
A�                                      Bxy_ �  �          A��A��@{�?�A*=qA�z�A��@(Q�@[�A�{A�
=                                    Bxy_/:  �          AA  @\(�@ ��A<z�A���A  @
=@X��A�
=AR�R                                    Bxy_=�  �          A{@�G�@�ff?�  A�B�
@�G�@��H@\)A���A��H                                    Bxy_L�  �          A��@�=qA�=L��>��
Bdp�@�=q@�  @y��A�BV��                                    Bxy_[,  �          A�\@�{A{=u>�Q�B}��@�{@��@�
=A��Bqz�                                    Bxy_i�  �          A
=@��RA\)>���@�
Baff@��R@��@�ffAɮBQ{                                    Bxy_xx  T          A�H@У�@�>�@0  BB�H@У�@��H@z=qA�(�B0��                                    Bxy_�  |          Ap�@�
=@���?W
=@�{Bz�@�
=@�Q�@VffA��HA��                                    Bxy_��  �          A%�A
�\@�Q�?\(�@�G�B��A
�\@�ff@VffA��Aٮ                                    Bxy_�j  �          A$(�A��@��?�ff@陚A�\)A��@vff@a�A��A�G�                                    Bxy_�  �          AQ�A��@Y��@!G�An�RA��A��?���@tz�A�Q�A:�\                                    Bxy_��  �          A
=A\)@�p�?�A3�
A���A\)@1G�@j=qA���A�G�                                    Bxy_�\  �          A�A
=@���?�p�A ��A��A
=@.{@[�A�33A���                                    Bxy_�  �          A!G�A=q@W
=@%�Am��A�p�A=q?޸R@w
=A�ffA,z�                                    Bxy_��  �          A   A(�@%@`��A�(�A�ffA(�?�R@��A�Q�@}p�                                    Bxy_�N  �          A�A	�@'�@8��A�
=A��
A	�?p��@r�\A��@�\)                                    Bxy`
�  �          A�@�ff@/\)?��@��A�  @�ff?�@p�ArffAX��                                    Bxy`�  �          A{@���@��;�\)��{A��
@���@r�\?�(�A6=qAҸR                                    Bxy`(@  �          AQ�@�@�  ��p��33B�@�@�Q�?�Ap�B��                                    Bxy`6�  T          Az�@ٙ�@�G����H�AG�B+
=@ٙ�@�Q�@#�
A�  B!(�                                    Bxy`E�  �          A{@Ϯ@�  �޸R�4��B&�@Ϯ@�(�?��H@�33B(�R                                    Bxy`T2  �          A�@�=q@����J�H���HB)
=@�=q@˅�#�
���B9Q�                                    Bxy`b�  �          @��
@���@�=q�N�R����B,ff@���@��ÿ����G�BA�                                    Bxy`q~  �          @��\@�Q�@��H���\���BC\)@�Q�@��H��
�uBf��                                    Bxy`�$  �          Ap�@�{@�p��љ��,�RB?
=@�{@�p��R�\����Bj�                                    Bxy`��  �          A!��@�p�@��}p���{Bn�@�p�A>��R?�\Bx�H                                    Bxy`�p  �          A!��@��\@��H���\����BP(�@��\A�H������33Bf                                    Bxy`�  �          A ��@�  @�33��{�Q�BN  @�  Aff��\)���Bg��                                    Bxy`��  �          A z�@��H@�  ����Q�BYz�@��HA	G���z����Brp�                                    Bxy`�b  �          A#�@��A���\(����HB�(�@��A�\?�=q@��
B�(�                                    Bxy`�  �          A�@�33@�G��p��yBI��@�33@�z�?���@�(�BO�\                                    Bxy`�  �          A�@�(�@ə������\)B@�
@�(�@�  �z��c33BT\)                                    Bxy`�T  |          A�H@�p�@�33�	���Z{B�R@�p�@�G�>�G�@1�B�H                                    Bxya�  �          A{AQ�@p�׿z�H��(�AÙ�AQ�@p��?s33@��RA��                                    Bxya�  �          A��@�  @�(����^�HA�z�@�  @�p�=�\)>��A��\                                    Bxya!F  �          A{@�
=@���=q�{�
A���@�
=@�녽�G��8Q�B{                                    Bxya/�  �          A��@��@\)�\(�����A���@��@��������{B33                                    Bxya>�  �          A�
@�@�
=�#�
���A�\)@�@��ͽ�Q�
=B\)                                    BxyaM8  
x          Az�@*�H@����z��&�
B��@*�H@�  ���n{B���                                    Bxya[�  
�          A  @��@Å���
���HB@\)@��@�\������ffBXp�                                    Bxyaj�  �          A=q@�{@�����\���BE�\@�{@�zῧ�� ��B_�                                    Bxyay*  �          A(�@�\)@�(������(�BF{@�\)@��׿�z����B`Q�                                    Bxya��  T          A��@��H@�  ��
=���B\��@��HA�
���R�@��Bx�                                    Bxya�v  �          Aff@z�H@ʏ\��Q���Bh��@z�HAG���p��BffB���                                    Bxya�  �          A@��R@����=q�BXp�@��RA\)�
=�L  Bv                                      Bxya��  �          A=q@�p�@�
=�������HBI
=@�p�@�  ����p�BaG�                                    Bxya�h  �          A�R@�=q@��
��=q��BJ�@�=q@��H�˅� z�Bfp�                                    Bxya�  �          A
=@��@��
��{���HBL��@��@�33��G���p�Bc��                                    Bxyaߴ  �          A  @�
=@�z������B`(�@�
=A��Y������Bt                                      Bxya�Z  �          A�@�Q�@�{��Q���  BU��@�Q�@�{�������Bl(�                                    Bxya�   �          A�@W
=@�p���=q���B���@W
=A��s33����B��                                    Bxyb�  T          Aff@z=q@�z��\)����Bu=q@z=qA�H�#�
���
B��                                    BxybL  �          A
=q@�G�@�����\)��{B>ff@�G�@�(���G���Q�BW�                                    Bxyb(�  �          A��@��@�z���33�(�B*�\@��@�Q����0��BK=q                                    Bxyb7�  �          Ap�@�
=@�����
��BR��@�
=@���33�{Bl�                                    BxybF>  T          Az�@�(�@�Q�������BU��@�(�@�{�p�����RBk                                    BxybT�  T          A\)@�Q�@ƸR���H��Q�BF�H@�Q�@��ÿB�\���RB\G�                                    Bxybc�  �          A�\@���@��H��{���
BA�R@���@��.{��G�BV�                                    Bxybr0  �          A�@�G�@��
�Vff��Bh(�@�G�@�(�>�(�@5�Br�R                                    Bxyb��  �          A��@�ff@θR�l(����HBc�
@�ff@�{��\)��ffBr33                                    Bxyb�|  �          Aff@��@��y����  Bap�@��@陚�Ǯ�(Q�Brp�                                    Bxyb�"  �          A�@xQ�@��������
=Bi33@xQ�@�Q����o\)Bz�R                                    Bxyb��  �          A�@x��@�33�������Bj  @x��@�׾�G��<(�Bz��                                    Bxyb�n  �          Ap�@~�R@˅�j=q��=qBg�H@~�R@��H���
���Bv(�                                    Bxyb�  �          A�H@x��@Ǯ�dz�����Bh=q@x��@�{�L�;�{BvG�                                    Bxybغ  �          A   @\)@�(��h������B_=q@\)@����z���
Bo�
                                    Bxyb�`  �          @��R@���@��\�\(���Q�BX�H@���@љ������z�Bi                                    Bxyb�  �          @�Q�@U@���U��p�Bj�H@U@�p��W
=��Bz33                                    Bxyc�  T          @ᙚ@C33@��\�\(����Bp�R@C33@ʏ\�\�E�B�z�                                    BxycR  �          @�{@\(�@�(��J=q��
=B`@\(�@��׾u��Q�Bq33                                    Bxyc!�  �          @�\)@^�R@�ff�C33��B`�H@^�R@��ý��ͿL��Bp                                      Bxyc0�  �          @���@[�@��
�=p���p�Be��@[�@�(�=u?   Bs{                                    Bxyc?D  �          @��@L��@����*�H��\)Bo�R@L��@�(�>���@S33By��                                    BxycM�  �          @�\)@j�H@�{�4z���p�BZ@j�H@���=���?Q�Bhp�                                    Bxyc\�  �          @޸R@G
=@���HQ���z�Boz�@G
=@ƸR���
�333B}�                                    Bxyck6  �          @ڏ\@:=q@��R�Q���\Bs�@:=q@��;�z��(�B�Q�                                    Bxycy�  �          @�ff@)��@���p  ���Bm�H@)��@�  ��ff�ffB���                                    Bxyc��  �          @�@,��@���Z�H��RBg�H@,��@�{�Y�����RB~z�                                    Bxyc�(  �          @���@)��@�
=�W��=qBgp�@)��@��\�Y���p�B~G�                                    Bxyc��  �          @��@@��@�33�?\)��G�BV@@��@�G������Bl�                                    Bxyc�t  
_          @�=q@H��@����Fff��z�BS\)@H��@�z�&ff����Bj33                                    Bxyc�  O          @�z�@P��@�\)�B�\����BQ�\@P��@������(�Bg=q                                    Bxyc��  �          @�\)@"�\@�z��G�����Bo��@"�\@�33�
=q���RB�G�                                    Bxyc�f  
Z          @�z�@C�
@w
=�L���
=BNp�@C�
@�{�^�R�z�Bh�                                    Bxyc�  "          @�ff@�
@�{�p���33B=q@�
@�p�>���@��\B��f                                    Bxyc��  
�          @�z�@@����޸R���HB�G�@@�  ?}p�AQ�B���                                    BxydX  
Z          @�  @P  @z=q�I����  BH�
@P  @�ff�L�����Bb��                                    Bxyd�            @�Q�@I��@|���N�R�Q�BM��@I��@��ÿY���Bg�H                                    Bxyd)�  	�          @���@AG�@�33�O\)��HBV��@AG�@�p��G���Bn�H                                    Bxyd8J  "          @��
@0��@���>{����Bkff@0��@�{���
�=p�B|\)                                    BxydF�  
Z          @�@>�R@���>�R��33Ba��@>�R@�(���Q��S�
Bs�                                    BxydU�  T          @�
=@4z�@����0  ��ffBn\)@4z�@�G�    �L��B|
=                                    Bxydd<  �          @�@>�R@�
=�6ff��G�BZ��@>�R@��\�\�l��BnQ�                                    Bxydr�  �          @��@P  @p���<������BD�@P  @�\)�0����z�B]�                                    Bxyd��  T          @�
=@:=q@|(��Q��=qBV
=@:=q@����c�
�
�RBp
=                                    Bxyd�.  
�          @�Q�@	��@����'�����B�p�@	��@��H=#�
>�Q�B�.                                    Bxyd��  �          @���@  @y���Mp���RBo�@  @�\)�W
=��
B�G�                                    Bxyd�z  T          @�(�@�H@@  �W��$�\BLQ�@�H@��ÿ�
=����Bq�                                    Bxyd�   	�          @��@,��?���L���0�B�@,��@N�R������BHp�                                    Bxyd��  �          @�z�@#33@'
=�.�R�
=B6�R@#33@hQ쿈���X(�B[p�                                    Bxyd�l  �          @�@U?�{�Fff�=qA��H@U@J=q��G����B-��                                    Bxyd�  
�          @�\)@~{?�Q��B�\��
A��@~{@0�׿����RBG�                                    Bxyd��  
(          @��
@��
�333�,����p�C�n@��
?=p��,(���z�A��                                    Bxye^  "          @���@����5��k��"ffC�K�@����   �p���=qC��=                                    Bxye  �          @���@�
=���H�����F{C�
=@�
=�n{������C�h�                                    Bxye"�  T          @�@��ÿ5��\����C�Ǯ@���>�=q�����{@Mp�                                    Bxye1P  
�          @�\)@��H?��R�ff���RAqG�@��H@������V{A�Q�                                    Bxye?�  "          @�=q@b�\@G���\���B%��@b�\@vff��(����B=�\                                    BxyeN�  
$          @�z�@8��@��H���z�Ba33@8��@���>�Q�@j�HBk�                                    Bxye]B  
�          @��@G�@~{�
=��\)BOQ�@G�@��>#�
?��B\�
                                    Bxyek�  �          @�Q�@�z�@!G����R�\(�A�
=@�z�@333>B�\@33BG�                                    Bxyez�  
�          @��@n{@9���8Q��
�\B�@n{@7�?Tz�A�B�\                                    Bxye�4  
�          @�{@�\@�
=�E���\B�G�@�\@�{?���A���B��H                                    Bxye��  �          @��@  @�(�������B��\@  @�Q�?���A�G�B~
=                                    Bxye��  "          @��R@3�
@�33����p�Bjp�@3�
@��R?�33A��\BaQ�                                    Bxye�&  	�          @���@*�H@�ff�޸R��z�Bk�H@*�H@�
=?8Q�@���Bq�
                                    Bxye��  �          @���@�@~�R�Q��ȣ�Bu{@�@�Q�>��?�\B�\)                                    Bxye�r  
�          @���@��@u�?���A��HBs  @��@(��@EB)  BK�\                                    Bxye�  
Z          @�G�@w
=@p���(���(�B(�@w
=@G
=�����p�B�R                                    Bxye�  
�          @���@@  @j=q�ٙ����\BJ�R@@  @���>�Q�@��\BU33                                    Bxye�d  
�          @�33?��@�녿���R=qB�33?��@���?���Ac�
B���                                    Bxyf
  �          @�(�?ٙ�@�ff�p���;�B��f?ٙ�@�33?��A���B�                                      Bxyf�  T          @�G�?���@�(��˅��z�B��H?���@��H?Q�A�RB��q                                    Bxyf*V  �          @�=q?���@~{���R��\)B�  ?���@�{>�=q@N�RB���                                    Bxyf8�  "          @��H?�(�@����{�ڣ�B�\?�(�@�33=�G�?�  B��                                    BxyfG�  
�          @�ff?�{@���p���{B�B�?�{@��>�{@o\)B��                                     BxyfVH  	�          @��?��@�=q�����p�B�Q�?��@��
?G�A�B��
                                    Bxyfd�  
�          @��?^�R@��H��=q��=qB�� ?^�R@�
=?��A>�\B�
=                                    Bxyfs�  �          @�  ?�@�녿�G���p�B�u�?�@��?\(�A ��B��                                    Bxyf�:  �          @�\)@{@h�����ߙ�B_\)@{@�=q�B�\�(�Bp��                                    Bxyf��  	`          @\)?��H@O\)��\)��p�Bx�?��H@_\)>�ff@�=qBQ�                                    Bxyf��  �          @�ff@\)@L(��������B[�\@\)@b�\>u@Tz�Bf�
                                    Bxyf�,  �          @u@�@z��\)����B6(�@�@<�;�ff���
BQ�                                    Bxyf��  "          @0��?�{@��p����G�B�  ?�{@\)>�Q�@�33B���                                    Bxyf�x  "          @+�?�ff@�\�������Bf33?�ff@33=�G�@ffBr33                                    Bxyf�  
Z          @��@(��?��R�  �	�B�@(��@6ff�s33�[33B<�
                                    Bxyf��  �          @u@\)@33���{B��@\)@4z�B�\�7�BB                                      Bxyf�j  T          @i��@  ?�33���=qB ��@  @-p��aG��`��BI(�                                    Bxyg  "          @n{@�H?����
=q�B��@�H@*=q�s33�l(�B>�R                                    Bxyg�  
�          @���@/\)@
=q�(���B  @/\)@Fff���\�^ffBA��                                    Bxyg#\  T          @���@C�
@���*=q���B@C�
@]p������O�
BA�                                    Bxyg2  T          @��\@AG�@�H�������B�
@AG�@S33�\(��.=qB>=q                                    Bxyg@�  
�          @���@L��@�R�!G���
=B��@L��@Z=q�n{�3�
B;{                                    BxygON  �          @��H@Z=q@��*�H�=qA�
=@Z=q@K����R�mp�B+��                                    Bxyg]�  
�          @���@^�R?˅�=q� ��A�  @^�R@%�����(�Bff                                    Bxygl�  �          @��@j�H?�G��"�\��Au�@j�H@	����Q����A�
=                                    Bxyg{@  �          @�z�@p  ?��������  A���@p  @ff��33��=qB 33                                    Bxyg��  "          @�ff@w�?�  ���G�A�@w�@�׿�����
A�\                                    Bxyg��  T          @�Q�@z�H?��(�����@�@z�H?��
� ���ȣ�AÅ                                    Bxyg�2  �          @��R@[���
=�H���(�
C��=@[�?�(��:�H��
A��\                                    Bxyg��  T          @���@Dz���$z��(�C�#�@Dz�?W
=�\)���AuG�                                    Bxyg�~  
�          @�=q@�� ���(Q��"
=C���@�����Q��X��C�                                    Bxyg�$  T          @���@����   �(�C��@��   �K��Pz�C�7
                                    Bxyg��  "          @�@B�\��p������C�33@B�\���R�8Q��-�C��                                    Bxyg�p  
�          @�33@7�����\���C�  @7��E��3�
�-�C�w
                                    Bxyg�  
�          ?�33�   ��
=<�@1G�C\ff�   �\�B�\��=qCY}q                                    Bxyh�  "          @,(���@
=?���A�ffB�B���?��\@ ��B^�B�u�                                    Bxyhb  "          @�33��@~{?@  A(z�B����@C�
@)��B!��B�.                                    Bxyh+  
(          @w����@o\)?^�RAP��B�aH���@2�\@(��B,\)Bą                                    Bxyh9�  T          @=p�>\)@<�ͽ#�
�8Q�B�>\)@#�
?���A�(�B���                                    BxyhHT  
Z          ?�{>�=q?�(���\)�Y��B�u�>�=q?��?��A��B���                                    BxyhV�  "          @p�?Tz�?��R��33��HB��)?Tz�?��?333A�=qB���                                    Bxyhe�  �          @���>L��@~{��(���\)B�B�>L��@i��?У�A�{B��q                                    BxyhtF  �          @Z�H?���@A녿Q��c
=B���?���@A�?Q�A`��B��
                                    Bxyh��  	�          @���@33@u��  ���RBw�\@33@|��?Y��A-p�Bz33                                    Bxyh��  	2          @�{@ff@_\)�h���J=qBk�
@ff@^�R?xQ�AW33Bk\)                                    Bxyh�8  �          @��?��R@n�R����(�Bw�?��R@_\)?�A�(�Bq                                      Bxyh��  �          @���?�@�p��fff�2�RB�  ?�@���?�=qA��B���                                    Bxyh��  �          @�z�?J=q@��>�Q�@�z�B�?J=q@g�@"�\B	�B��                                     Bxyh�*  �          @�=q>��
@�p�?�@���B���>��
@c33@*�HB\)B���                                    Bxyh��  
�          @���>�\)@�\)�8Q��B�>�\)@}p�@ffA�
=B���                                    Bxyh�v  �          @���=u@�  �L���'
=B��==u@\)@A�\)B�W
                                    Bxyh�  �          @�\)>���@�{�����tz�B�Ǯ>���@\)?�p�Aң�B���                                    Bxyi�  �          @��>8Q�@�(��L���%�B�\)>8Q�@xQ�@�\A�\)B�Ǯ                                    Bxyih  �          @��>.{@�Q�8Q����B��
>.{@�ff@  A���B�L�                                    Bxyi$  �          @���?�ff@��\    <�B�ff?�ff@}p�@�
A�ffB��3                                    Bxyi2�  �          @��R?��
@�33<�>���B���?��
@|��@A�ffB��f                                    BxyiAZ  �          @�(�?�  @���=��
?��\B�{?�  @w
=@ffA�  B�                                      BxyiP   �          @���?}p�@�p�=u?5B��?}p�@�  @��A�{B�G�                                    Bxyi^�  �          @��R?���@�  >L��@�B�  ?���@�  @$z�A��HB��\                                    BxyimL  �          @��\?��
@�ff>���@q�B��{?��
@��@1�BG�B��                                    Bxyi{�  �          @�G�>�{@��׾����HB�W
>�{@~{@
=qA�B�#�                                    Bxyi��  �          @�>���@����   ��ffB��>���@�?�\)A��B��H                                    Bxyi�>  
�          @�
=?=p�@�33��Q���(�B���?=p�@���@ ��A�B��3                                    Bxyi��  �          @�ff?
=@����Q쿇�B�aH?
=@���@  A�\)B�G�                                    Bxyi��  �          @�33�#�
@��\�u�B�\B�#׼#�
@�=q@ffAٮB�(�                                    Bxyi�0  �          @���5@�  �\)��p�B��f�5@}p�@	��A���B�L�                                    Bxyi��  �          @��H���H@�p��B�\��\B��=���H@�{?���A�G�B�33                                    Bxyi�|  �          @�\)���@����c�
�9B����@��?�33A�  B��                                    Bxyi�"  �          @��
�\@�  �^�R�-��B����\@��\?\A�Q�B�                                      Bxyi��  �          @���>���@���������B���>���@��\?�RA�HB�33                                    Bxyjn  �          @�33?:�H@��Ϳ���U�B�.?:�H@��\?�ffA�{B��f                                    Bxyj  �          @�ff?���@��
��=q�{
=B�L�?���@�p�?�A\(�B��=                                    Bxyj+�  �          @���?Q�@�녿��H����B��3?Q�@�z�?:�H@��B��                                    Bxyj:`  �          @���?�z�@�  �޸R��{B��
?�z�@�ff?�G�A+�
B���                                    BxyjI  T          @�G�@p  @�녾�p��u�B=
=@p  @l(�?�(�A���B2=q                                    BxyjW�  �          @�  @vff@y����33�i��B533@vff@b�\?�z�A��\B*Q�                                    BxyjfR  �          @���@c33@�  >\)?�BHz�@c33@g
=@��A�Q�B5�R                                    Bxyjt�  �          @�  @S33@�(�>#�
?ٙ�BT(�@S33@mp�@A�ffBAQ�                                    Bxyj��  �          @�{@!�@�  ����˅Bw�H@!�@��?�\)A���Bp��                                    Bxyj�D  �          @�p�@5�@��\�����
=BiG�@5�@�ff?�33A��\B`�                                    Bxyj��  �          @�{@U�@�����\�-G�BM(�@U�@��?��HAN{BK                                    Bxyj��  �          @�=q@[�@���8Q��=qBOG�@[�@w
=@�\A�ffBA�                                    Bxyj�6  �          @�33@��@P  ?(�@љ�B�@��@   @
=qA��A�=q                                    Bxyj��  �          @�\)@_\)@w�=���?��B?��@_\)@S33@�A�
=B-Q�                                    Bxyjۂ  �          @�p�@U�@{�>��R@[�BG(�@U�@P  @\)A�(�B1=q                                    Bxyj�(  �          @�{@^{@u�=��
?W
=B?G�@^{@Q�?��RA��B-Q�                                    Bxyj��  �          @�{@e�@mp���\)�Q�B8(�@e�@P  ?�ffA�p�B(�
                                    Bxykt  �          @�\)@hQ�@n�R��33�y��B733@hQ�@Y��?���A�B,�R                                    Bxyk  T          @���@qG�@g������\)B/z�@qG�@Z=q?��Am�B(z�                                    Bxyk$�  �          @�p�@�\)@2�\�k��$(�B��@�\)@8Q�?(�@أ�B                                    Bxyk3f  �          @�{@�=q@\)��{����A��@�=q@0  ��=q�G�B=q                                    BxykB  �          @�(�@8Q�@��R��=q�1Bj=q@8Q�@��
?�
=Aj�RBh=q                                    BxykP�  �          @��@1�@�\)�^�R�(�Bn�@1�@���?�\)A��Bi�R                                    Bxyk_X  �          @���@(�@�(��
=����B}�
@(�@���?�Q�A�p�Bv��                                    Bxykm�  �          @��@l(�@��ÿ�Q��k�B>{@l(�@��R?G�@��BC                                      Bxyk|�  
�          @��H@�ff@&ff�=q��Q�A�  @�ff@]p��J=q��B�H                                    Bxyk�J  
�          @��@�
=@��,�����HA�
=@�
=@U���Q��HQ�B��                                    Bxyk��  �          @�G�@��@,(��"�\�ܸRB=q@��@fff�Y���
=B&p�                                    Bxyk��  	\          @���@�z�@+���p���z�A��H@�z�@S�
��p��x��B��                                    Bxyk�<  
^          @���@��H@Vff���R�{
=B33@��H@h��>Ǯ@��\B&�R                                    Bxyk��  
�          @���@���@=p������{B
�@���@`�׾#�
��33Bz�                                    BxykԈ  
�          @�G�@�ff@HQ��  ��(�B��@�ff@e=L��>��B!�                                    Bxyk�.  T          @�@Q�@���>�z�@:=qB��@Q�@�p�@8��A��
B���                                    Bxyk��  T          @�{@��@�Q�>�\)@7�B���@��@��@8Q�A�33B�W
                                    Bxyl z  "          @���?�  @�Q�=�\)?0��B��
?�  @��@3�
A��B��=                                    Bxyl   
�          @���?��@�\)�L��� ��B�8R?��@�33@#�
A��B��R                                    Bxyl�  	�          @��R?��H@��;�33�a�B���?��H@�33@��AǮB��                                    Bxyl,l  
�          @��?�
=@�=q>��R@H��B�8R?�
=@��R@;�A���B�B�                                    Bxyl;  �          @�
=?��@�G���z��:�HB��q?��@�{@ ��AхB��3                                    BxylI�  "          @�{?�=q@�\)�&ff�ҏ\B��R?�=q@�=q@
�HA��
B���                                    BxylX^  T          @���?��@�(���p��u��B��=?��@�?��
AS�B��H                                    Bxylg  
�          @��
@�
@�=q�}p��#�
B��q@�
@�z�?�A��B�=q                                    Bxylu�  "          @���@@  @������?
=B_��@@  @��?�p�AO�B_                                      Bxyl�P  
�          @��@+�@�{>\)?�
=Bu�H@+�@��R@%�A���Be��                                    Bxyl��  
�          @��
@��H@tz�.{��  B,=q@��H@Y��?�  A�(�B�                                    Bxyl��  
�          @��R@q�@{��   ��\)B8Q�@q�@j=q?\A�33B0{                                    Bxyl�B  	�          @�Q�@c33@�
=�@  ����BN33@c33@�  ?���A}G�BH��                                    Bxyl��  
�          @�33@333@�Q�z�H��\Bs33@333@��H?�33A�\)Bo��                                    Bxyl͎  %          @��H@>�R@���8Q���z�Bk
=@>�R@��
?���A��\Bd�R                                    Bxyl�4  
�          @�=q@J=q@��\��(��l(�B]z�@J=q@�ff?�G�A ��B`G�                                    Bxyl��  �          @�  @R�\@�
=�   ��  BPQ�@R�\@��>�Q�@h��B[{                                    