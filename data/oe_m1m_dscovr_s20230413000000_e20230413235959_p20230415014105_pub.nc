CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230413000000_e20230413235959_p20230415014105_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-15T01:41:05.416Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-04-13T00:00:00.000Z   time_coverage_end         2023-04-13T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxw~�   T          A (�?��?\��
=BY�H?��@�Q�����>
=B���                                    Bxw~��  
�          Aff=u@xQ���p��vQ�B�=q=u@߮�����HB�                                    Bxw~�L  
�          @�\)<�@�������K{B�8R<�@�
=�.�R��z�B�u�                                    Bxw~��  T          Ap�?^�R@�{���H�N��B��?^�R@��;���  B�Ǯ                                    Bxw~ј  T          A   ��p�@q���(��m�B޸R��p�@�Q��\)��B̨�                                    Bxw~�>  T          A�\�\@}p���ff�j�B��f�\@�ff�}p����HB̙�                                    Bxw~��  �          Az�E�@fff����~G�B���E�@�(���G��
=B�                                    Bxw~��  �          A(��Ǯ@������m��B�\)�Ǯ@���33��RB�                                    Bxw0  T          A(��\)@�p�����r��B��;\)@���Q���=qB�#�                                    Bxw�  �          A���@|(���\)�w�B�
=��@�ff����{B�G�                                    Bxw)|  �          A�׿
=@��R��z��pQ�B�{�
=@�z���
=��33B�33                                    Bxw8"  �          A��O\)@��\��Q��]�
B��ÿO\)@�(��_\)��33B�u�                                    BxwF�  T          @��Ϳ��@�������GQ�Bя\���@�z��ff��\)B�(�                                    BxwUn  �          @�ff�Q�@��
�����H(�B�=q�Q�@θR�ff���\B�u�                                    Bxwd  �          @��þ��H@�
=���>{B����H@�ff��
���RB�k�                                    Bxwr�  �          @�\)�!G�@����G��1z�B�#׿!G�@޸R�޸R�`  B�ff                                    Bxw�`  T          @�ff��\)@����=q�"ffB��)��\)@�׿��H�5B��)                                    Bxw�  �          @�ff��
=@�������%�
B��)��
=@�p���
=�=�B�B�                                    Bxw��  �          @��H��ff@�����=q�(33B�33��ff@�p������:ffB��q                                    Bxw�R  �          @��H��{@��H���
�#\)B�
=��{@�\)��33�6�\B��H                                    Bxw��  �          @陚���\@�
=�����8\)B�����\@�ff�G���z�BĞ�                                    Bxwʞ  �          @��
�333@����G��,��B���333@�(���33�O�
B�.                                    Bxw�D  T          @�{���@�����
�+
=B����@�p��˅�LQ�B�{                                    Bxw��  T          @�G����@�����(��AB�G����@׮����=qBŞ�                                    Bxw��  
�          @�녿�33@�G����\�?
=B��ÿ�33@�ff������
BΨ�                                    Bxw�6  T          @���y��@���y����33B�8R�y��@��H�L����p�B�.                                    Bxw��  �          @����Å@�33�����@Q�C�
�Å@��?Q�@�(�CL�                                    Bxw�"�  �          @�z���(�@����@  ���HCG���(�@�=q=��
?
=B��R                                    Bxw�1(  "          @�
=�j�H@��\�x����p�B�W
�j�H@�  ����>�RB�W
                                    Bxw�?�  T          @�ff���@�Q��7�����C ff���@�p�>�(�@EB���                                    Bxw�Nt  "          A�����
@�녿����4  C	�)���
@�(�?���A�\C	}q                                    Bxw�]  �          A
=���H?�G�@@��A�G�C(c����H�aG�@W
=A���C5��                                    Bxw�k�  T          A������?޸R@�\A�z�C'^�����>�33@7
=A�\)C1n                                    Bxw�zf  
�          A�H����?�
=@j�HA���C*�3���ͿTz�@qG�AָRC:k�                                    Bxw��  T          A{��
=?u@`  A�\)C,����
=�n{@`��A��
C;�                                    Bxw���  �          A����  @8��@j=qA��
C�q��  ?(��@��
B{C.33                                    Bxw��X  �          A33���@0��@2�\A��C�f���?z�H@s33A�  C,+�                                    Bxw���  �          AG���R@33@EA�{C$�)��R>aG�@l��A�
=C2G�                                    Bxw�ä  "          A����H@?\)@(�Aw�C�
���H?�p�@Y��A�33C(�{                                    Bxw��J  
�          A���G�@�(�?\)@|��C&f��G�@`��@+�A��RC��                                    Bxw���  T          @�z���=q@˅��\)�^=qB�G���=q@�{?���A:�RB���                                    Bxw��  T          @�ff�U�@��H�HQ���=qB�aH�U�@��H>�{@%B���                                    Bxw��<  �          A\)?�\@�  �h����ffB�� ?�\@�p�@EA�ffB��H                                    Bxw��  
�          A z�#�
@���;�����B�z�#�
@���?�\)A33B�W
                                    Bxw��  �          A ����@�׿У��<z�B�k���@�\@
=A�G�B�33                                    Bxw�*.  �          @���>k�@��ͿG����\B�z�>k�@�Q�@I��A�\)B�(�                                    Bxw�8�  �          @�녿�{@�\��
�w
=B�
=��{@��
?��Aa�B��                                    Bxw�Gz  �          @��
�@��@�녿�33�BffB�.�@��@�{@Aw�B��f                                    Bxw�V   T          @���N�R@޸R��  �1G�B����N�R@���@
�HA��B���                                    Bxw�d�  
�          @�������@�=q���z�HCn����@��H@z�A���C                                    Bxw�sl             @����ə�@��׾Ǯ�7�CO\�ə�@��?�Ae�C�
                                    Bxw��  �          @�����(�@�Q켣�
�8Q�C0���(�@�z�@ffA���C�q                                    Bxw���  
�          @����{@�  =L��>�Q�C����{@j=q@
=qA��Cff                                    Bxw��^  
�          @�
=�Ϯ@�(�>#�
?�(�C��Ϯ@`  @��A���C�f                                    Bxw��  T          @����(�@}p�>��?�z�C+���(�@S33@(�A��
C��                                    Bxw���  �          @�����z�@��Ϳ����CxR��z�@��
?�AEC:�                                    Bxw��P  "          @�����@�{�8Q����\C��@�?ٙ�AJ=qCQ�                                    Bxw���  "          @�{���@�=q��\�s33C=q���@��H@z�A�G�C�
                                    Bxw��  $          @�=q��@�{>��R@�C	����@�33@5A��C��                                    Bxw��B  �          @�G����@��
�������C����@�  ?�\)AUp�C��                                    Bxw��  �          @����<��@�
=������
=B�=�<��@����|(�B��                                    Bxw��  V          @���� ��@�G�������B�(�� ��@�33�333��(�Bٽq                                    Bxw�#4  �          @��\��R@�(��qG����HBޅ��R@�����c�
B�
=                                    Bxw�1�  �          @��H�-p�@��
������G�B�R�-p�@�\���H�fffBܔ{                                    Bxw�@�  T          @�(��,(�@��H������p�B㞸�,(�@��H������B�33                                    Bxw�O&  
�          @���{@����\)�=qB݀ �{@���(��33B�B�                                    Bxw�]�  	�          @�ff�  @�Q������RBޮ�  @�
=��  �.{BՅ                                    Bxw�lr  �          @����@������� ��Bس3��@�Q���tz�B�                                      Bxw�{  "          @���p�@��ÿ�z��aCk���p�@�
=?�Q�A�C��                                    Bxw���  "          @�����{@��H������C�
��{@��R?��A?�
C	��                                    Bxw��d  T          @�(���ff@����#33��C�=��ff@�33>\@333C��                                    Bxw��
  "          @��\�q�@����vff��{B����q�@�녿���ffB�{                                    Bxw���  "          @�=q�~�R@��H�����z�C��~�R@���� ���n�RB�3                                    Bxw��V  T          @�G����@���\(���p�B�Ǯ���@��;�{�"�\B�Ǯ                                    Bxw���  "          @�����\)@�p����pQ�C �H��\)@��?���A��B��                                    Bxw��  "          @�(���G�@�  �"�\����B��
��G�@�{?Y��@�
=B�Q�                                    Bxw��H  �          @�(���=q@�
=����A��B�\)��=q@�{?޸RAM�B�\                                    Bxw���  �          @��\(�@��
�������B��f�\(�@�
=?���AG�B�                                     Bxw��  �          @���J�H@����R��ffB����J�H@���?��A"�HB�\                                    Bxw�:  
�          A   �c�
@��H�33�nffB��c�
@�
=?�ffA3�B�#�                                    Bxw�*�  T          @�{�>{@ڏ\�.�R���B��f�>{@陚?s33@��B�8R                                    Bxw�9�  
Z          @�ff�X��@�Q�����ffB�=q�X��@�\?�
=A��B��                                    Bxw�H,  T          @�\)�Vff@أ���H��=qB虚�Vff@��H?�Q�A	�B�                                    Bxw�V�  T          @�33�r�\@�ff�G����HB����r�\@׮?�A	�B�                                    Bxw�ex  �          @��\�7�@�
=�/\)��z�B�=q�7�@�ff?aG�@�
=B�p�                                    Bxw�t  �          @�=q�+�@��H�j=q�߮B��+�@�=q��G��G�B�.                                    Bxw���  �          @�=q���@�=q��\)��
C���@�G�@A�ffCǮ                                    Bxw��j  �          @�����@����H�J�\B�8R��@�\)?�G�A2�HB�                                    Bxw��  T          @�Q���  @�����
�VffB��f��  @�33?���A!�B��H                                    Bxw���  T          @�����z�@Å��G��R�HB�W
��z�@�ff?�A)�B��{                                    Bxw��\  �          @�����\@˅��G��G�B�����\@�?�p�Aj�HB��                                    Bxw��  T          @�=q�vff@�z�n{��{B�8R�vff@�Q�@��A��\B�.                                    Bxw�ڨ  �          @����I��@�
=�������B�G��I��@�p�?���A$z�B�                                    Bxw��N  "          @����c33@ҏ\���R�4z�B�3�c33@�\)?��Ad  B�p�                                    Bxw���  �          @���}p�@ʏ\���IB���}p�@�33?˅A@z�B��                                    Bxw��  "          @�\)�s�
@��
���~�RB�R�s�
@�33?�(�AB��                                    Bxw�@  "          @���ff@��H���\��B�#���ff@�ff?�=qAaG�B�k�                                    Bxw�#�  "          @���Q�@b�\@G�A�{Cff��Q�?���@�{Bp�C$u�                                    Bxw�2�  �          @�����  @H��@z=qA�G�CG���  ?^�R@�{B&(�C*�q                                    Bxw�A2  �          @�(���G�@>{@�B��Cٚ��G�?
=@��HB)�RC-��                                    Bxw�O�  �          @����
=@333@��B  C�f��
=?�\@��B"  C.�                                    Bxw�^~  �          @�{���?��@�Q�B#�RC ����#�
@��\B1G�C;{                                    Bxw�m$  T          @���(�@p�@�Q�B��CT{��(���@��B8��C5W
                                    Bxw�{�  �          @�\)���\@3�
@�Q�B6�CxR���\�8Q�@�{BU�
C65�                                    Bxw��p  �          A��\)@G
=@�=qB7(�C����\)�u@��HBYffC4�3                                    Bxw��  T          @�=q����@Tz�@�=qB0  C�q����>�p�@�Q�B[33C/=q                                    Bxw���  �          @�=q�{�@`  @�B<C
Q��{�>�Q�@��Bmp�C.��                                    Bxw��b  "          @��\��H@u@ʏ\BYQ�B���H>�33@�z�B�W
C+�                                     Bxw��  
(          @��X��@Q�@�
=BS(�C��X�ý��
@���B�� C5^�                                    Bxw�Ӯ  �          A z���
=@s�
@���B333C	���
=?(��@���Bg�C++�                                    Bxw��T  "          A�
����@|��@��B:�HC������?�R@���Bq=qC+E                                    Bxw���  �          A�����\@��H@���B0��C�����\?W
=@���Bg�\C(��                                    Bxw���  �          AQ��r�\@tz�@�{BC�
C�q�r�\>�
=@�Q�By(�C-��                                    Bxw�F  
�          AQ��h��@a�@�  BP(�C�{�h��<��
@���BG�C3�                                    Bxw��  T          AG��HQ�@+�@�
=Bq
=Ck��HQ쿇�@�(�B��CF�                                    Bxw�+�  �          A�R�s33?�Q�@�\)Bm�
C�q�s33��(�@�G�BpCLaH                                    Bxw�:8  �          A33���@B�\@�{BS�C+���녾��H@�\BsG�C:�{                                    Bxw�H�  �          A���{@(��@�33BX��C����{�k�@�G�Bn  C@c�                                    Bxw�W�  �          Az����R@�R@�p�BM��CB����R�xQ�@ᙚB^�RC?��                                    Bxw�f*  T          A	G����@R�\@�(�B@��C�����@�Bd�C5p�                                    Bxw�t�  T          A����{@k�@��\B-��CO\��{>��H@�(�BX�C.J=                                    Bxw��v  "          A33��z�@�p�@��HB8(�CǮ��z�?L��@��HBp�C)                                      Bxw��  �          A���s33@���@��B5��C�)�s33?��@�G�Bv  C#��                                    Bxw���  �          @��=q@��{���C�{�=q@{?L��A[\)C	B�                                    Bxw��h  "          @�{�.{�h����z�¡\)Ci:�.{@(���  z�B�                                    Bxw��  "          @�p�?   @g
=�����]�B�{?   @�p��>{��Q�B���                                    Bxw�̴  T          @陚��@����G��;�RB�LͿ�@�\)��H��  Bǳ3                                    Bxw��Z  T          @����9��@�����b  C@ �9��@��w��G�B�u�                                    Bxw��   
�          @���P  ?��R�����[{C� �P  @�G��u���B�W
                                    Bxw���  T          @У�����=u�����L�HC35�����@������,�CE                                    Bxw�L  
�          @ۅ���ÿ�z���(��7Q�CEQ�����?��
��\)�;�C'#�                                    Bxw��  
�          @����!G������(��d��C\T{�!G�?   ��G�33C(�R                                    Bxw�$�  "          @�  �u��6ff@���B1
=CX�f�u���{@2�\AŮCf                                    Bxw�3>  �          @�=q�|(��c33@�p�B%\)C^
=�|(���=q@{A�  CixR                                    Bxw�A�  �          @޸R�.�R�b�\@���BGz�ChL��.�R��33@C33A�z�Cs��                                    Bxw�P�  T          @׮���l(�@��BOCr�\�����@?\)A�=qC{��                                    Bxw�_0  T          @�
=�   �=p�@���Bd=qCk���   ��{@`��B{Cx��                                    Bxw�m�  �          @�
=?���O\)@�(�Bd
=C�\?�����@P��A���C���                                    Bxw�||  T          @�ff?B�\�
=@�33B�B�C��3?B�\��{@w�BC��q                                    Bxw��"  �          @��R=�G����@��B��
C�^�=�G���33@h��Bp�C��
                                    Bxw���  
�          @���� ��?�  ?�(�B\)C8R� ��>��R@
=B7�C+Q�                                    Bxw��n  T          @���z�@�  ����z�B�z��z�@��H=#�
>\Bօ                                    Bxw��  "          @��׿�=q@�(�� ����=qB�G���=q@�\)?z�@��HB�8R                                    Bxw�ź  T          @�(���33@��>\)?�G�B�G���33@���@��Aי�B�R                                    Bxw��`  T          @�Q쿔z�?���@�  B�aHC�R��z��G�@���B�8RCp�                                    Bxw��  
�          @��
�4z�@S�
@ϮB`�RCh��4z�L��@�G�B��3C4�                                    Bxw��  
�          A   �(�@�\)@���BD�B�Ǯ�(�?�
=@���B���C޸                                    Bxw� R  R          @��(��@�=q@ǮBP�B��f�(��?@  @�p�B�aHC$�                                    Bxw��  �          A ���@��@tz�@�p�BS�C @ �@��>�ff@�ffB���C+��                                    Bxw��  V          @�\)�A�@>�R@�Q�BgQ�C���A녾�ff@��
B���C<ff                                    Bxw�,D  
�          A���c�
@O\)@ϮBU�
C	�f�c�
��Q�@�Q�B�C5h�                                    Bxw�:�  T          @�ff�Tz�@y��@�\)BE�C^��Tz�?=p�@�33B��)C's3                                    Bxw�I�  T          A   �j=q@z�H@��B={C�j=q?W
=@޸RBv�\C'
                                    Bxw�X6  T          A�(Q�@qG�@ӅB\�RB�Ǯ�(Q�>�{@�B��=C,�{                                    Bxw�f�  T          A��H��@��
@�ffB�B�=q�H��@>{@�\BjffC�=                                    Bxw�u�  T          A
=���@�=q@�(�B�
B��
���@o\)@�BgffB�W
                                    Bxw��(  T          A���R�\@���@�Q�B(=qB�33�R�\@�@�  Bw  C��                                    Bxw���  
�          A  �_\)@��@��B1�B�=q�_\)?��@�z�Bwz�C.                                    Bxw��t  
�          A�����@n�R@���B@�\C	u����?�@�p�Bq{C,�                                    Bxw��  
�          AQ��q�@���@�\)B9�C\)�q�?�\)@���Bv=qC#�                                     Bxw���  T          A�����@���@�\)B$(�C\)���?�@߮Bc�\C��                                    Bxw��f  
�          A�����@�{@��RBp�C����@�
@�(�B_�HC��                                    Bxw��  �          A\)��33@�{@��
BQ�C����33@ff@ٙ�B_��C޸                                    Bxw��  
�          Aff����@�p�@�Q�Bp�C�����@G�@�33BM��C�\                                    Bxw��X  T          A���
@��\@�33B
(�C�q���
@\)@�G�BJ�C�                                    Bxw��  �          A���\)@���@�G�A�33C����\)@@��@���B@�C#�                                    Bxw��  �          A�H��p�@�z�@�  A��C����p�@AG�@��B<ffC
                                    Bxw�%J  
�          AQ����
@��H@�G�B�CE���
@ ��@ʏ\BI33C}q                                    Bxw�3�  
(          Aff��  @�\)@�{B1\)Ch���  ?�p�@�\)Bl33C"�H                                    Bxw�B�  "          Aff����@���@���B/��CY�����?��@޸RBj�
C"B�                                    Bxw�Q<  W          A\)����@��R@�p�B�\Cz�����?�
=@��
BWz�C�R                                    Bxw�_�  �          A�H����@��H@��B�RCT{����?�@�{BP33C�\                                    Bxw�n�  �          A�R���\@��@�B=qCff���\?˅@ҏ\BVp�C ��                                    Bxw�}.  
�          A����@��@���B,Q�C�����?�33@ۅBhz�C �\                                    Bxw���  �          Ap��0  @h��@�p�B^p�B���0  >���@��HB���C-�
                                    Bxw��z  
�          A���N{@�Q�@�p�BH��C ���N{?Y��@�=qB�{C%!H                                    Bxw��   �          A��(�@�Q�@�33B$�
C�\��(�?�Q�@�B]�\C!                                    Bxw���  �          A �����@vff@���B(�C�H���?�ff@љ�BY�RC'�                                    Bxw��l  
(          A������@u�@��B��C������?�z�@�=qBM�HC&��                                    Bxw��  "          A����@xQ�@�G�B6�C�H���?k�@��Bj�C'c�                                    Bxw��  T          Aff��
=@�Q�@��\B
=C���
=?���@ʏ\BKp�C$�                                     Bxw��^  �          A����@���@�p�B{Cu����?�ff@�
=B>�C#}q                                    Bxw�  �          A�����@W�@��B33Cz����?\(�@�z�B:��C+�                                    Bxw��  �          A{����@G
=@��\B$(�Cp�����>�ff@��BEG�C/�                                    Bxw�P  "          A ����G�@(��@�
=B+
=Cn��G���@�=qBC�
C4G�                                    Bxw�,�  
Z          A z���33@H��@��HB033C���33>�p�@���BS�C/�)                                    Bxw�;�  T          @������@B�\@��HB;�C{����>8Q�@ҏ\B]��C1Ǯ                                    Bxw�JB  "          A (���z�@N�R@��B4�C���z�>�(�@�  BY�\C.��                                    Bxw�X�  �          A{��(�@n{@���B:�HC
���(�?E�@�Bk(�C)}q                                    Bxw�g�  �          A=q��\)@xQ�@���B4��C	}q��\)?xQ�@�z�Bg{C'�                                    Bxw�v4  �          A{���
@~{@�=qB-z�C	� ���
?��@׮B`�
C%Y�                                    Bxw���  �          A ���fff@tz�@\BD�HCT{�fff?J=q@�(�BzC'��                                    Bxw���  
�          A�g
=@���@���B@{C���g
=?��\@�{By�RC$&f                                    Bxw��&  
Z          Ap��u@z=q@�ffB=Q�C���u?p��@ᙚBr��C&L�                                    Bxw���  �          Ap��x��@�(�@���B6  CE�x��?�(�@�  Bo{C"�{                                    Bxw��r  �          A ���s33@|��@�(�B<�C�f�s33?�G�@�Q�Br�
C%�                                    Bxw��  T          A Q��fff@��
@��B;�HC!H�fff?�
=@�=qBw33C!�\                                    Bxw�ܾ  
�          A   �hQ�@u�@���BC{Ch��hQ�?\(�@�\By
=C&��                                    Bxw��d  �          @���\(�@i��@ƸRBL33CO\�\(�?�R@�B�
=C)�{                                    Bxw��
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw�V  �          @���!G�@P��@�Bi\)B��\�!G�=�\)@�B�z�C2\)                                    Bxw�%�  
�          @��\�>{@Vff@��B\\)C�
�>{>�=q@�\)B�#�C.��                                    Bxw�4�  W          @��\�@  @Z�H@˅BY��C5��@  >�p�@�RB��=C,�                                    Bxw�CH  
�          @��9��@~{@�
=BN��B�=q�9��?n{@�=qB���C".                                    Bxw�Q�  �          @�{�2�\@�33@�{BM=qB��=�2�\?��@�33B�C0�                                    Bxw�`�  
�          @�p��>�R@�
=@���B<{B�p��>�R?�=q@��
B�B�C�                                    Bxw�o:  
�          @��H�q�@~�R@�=qB5�C���q�?�p�@�\)Bm�C!�                                    Bxw�}�  
�          @�33�Tz�@��@���B833B�B��Tz�?�  @�ffBy33C�3                                    Bxw���  �          @�G��]p�@|(�@���B<p�CB��]p�?�@�G�Bv�
C!G�                                    Bxw��,  "          @�\)�`  @W
=@�\)BK��C+��`  ?   @��HBzC+��                                    Bxw���  
Z          @�Q��C�
@��H@�(�B:�RB�W
�C�
?�ff@�{B~�
C+�                                    Bxw��x  
�          @�  �(��@�p�@���B@�B��(��?���@�33B�8RC@                                     Bxw��  Q          @�p��@  @�  @���BD��B����@  ?�@�ffB�33C��                                    Bxw���  "          @��
�^�R@dz�@���BD�C:��^�R?L��@�  Bx�C'                                    Bxw��j  �          @�
=�a�@vff@�{B=��C}q�a�?���@���Bu�C"�R                                    Bxw��  �          @�
=�a�@U@��BK�
C�\�a�?�\@��HBy�C+�                                     Bxw��  �          @���`��@G
=@���BP�RC
}q�`��>�\)@�G�BzG�C/^�                                    Bxw�\  
�          @�
=�|(�@G�@�33BEQ�C���|(�>\@��
Bl�C.u�                                    Bxw�  �          @��R����@B�\@��\BD�\C�f����>��
@�=qBiC/n                                    Bxw�-�  
�          @���k�@S33@�ffBIG�C
&f�k�?�\@�G�Bu=qC,.                                    Bxw�<N  �          @�
=�s33@XQ�@���BB�C
L��s33?(��@�Bp{C*5�                                    Bxw�J�  �          @�  �k�@O\)@�\)BJ�\C
���k�>�@�G�Buz�C,��                                    Bxw�Y�  "          @�
=�>�R@�p�@��B((�B��f�>�R@G�@ƸRBo�C�
                                    Bxw�h@  
�          @����L(�@E@��BH\)C�
�L(�?(�@��
Bx=qC)8R                                    Bxw�v�  	�          @�G��*�H@`  @��\BD\)B��3�*�H?�=q@\B�G�C�                                    Bxw���  �          @�33���@�G�@��HB?p�B�.���?���@ə�B��\C�                                    Bxw��2  �          @�G��
=q@��@VffA�\B�B��
=q@\(�@��BP�HB��                                    Bxw���  �          @�녿��
@�Q�@Mp�A���B����
@���@���BH��B�ff                                    Bxw��~  �          @�G��z�@�z�@`��A�B����z�@l(�@���BN��B�p�                                    Bxw��$  �          @���\@�@fffA��
B��f��\@l(�@��
BQQ�B�                                    Bxw���  
�          @�p��Q�@�  @c33A��
B�{�Q�@q�@�33BM�RB�=                                    Bxw��p  �          @�(��@��@h��A�(�B���@j=q@���BQ�B�.                                    Bxw��  �          @��
��
=@���@w�B�Bٳ3��
=@dz�@��HB\�B���                                    Bxw���  �          @���   @�{@�=qB�B��   @/\)@�
=Bh�\Cu�                                    Bxw�	b  �          @�G���Q�@�Q�@<��A���Bɀ ��Q�@�  @�  BA�Bъ=                                    Bxw�  �          @�Q쿢�\@θR@�\A�p�B�.���\@��R@�\)B*�
BУ�                                    Bxw�&�  �          @�(����\@�Q�?�
=A�ffB��Ϳ��\@��@��B   B�\)                                    Bxw�5T  �          @�Q�&ff@�p�?8Q�@ÅB���&ff@���@W�A�  B�                                    Bxw�C�  �          @�G��W
=@�ff?8Q�@�G�B�\)�W
=@��\@XQ�A�z�B�u�                                    Bxw�R�  �          @����z�@��@�A��BԀ ��z�@���@��B�
Bܽq                                    Bxw�aF  "          @޸R��z�@�(�@	��A�=qB�8R��z�@��R@���B#�HB�
=                                    Bxw�o�  T          @޸R�:�H@�=q@P  A�{B��q�:�H@��@�BN�BǙ�                                    Bxw�~�  "          @�\)��p�@Å@U�A�z�B��H��p�@��@�Q�BQG�B��f                                    Bxw��8  "          @�G�����@�ff@Q�A���B�Q����@�33@�  BN=qB�k�                                    Bxw���  �          @�=q>W
=@���@�A���B�ff>W
=@���@���B.�B�L�                                    Bxw���  �          @�z�>�33@�{@u�B(�B�Q�>�33@y��@���Ba�RB��
                                    Bxw��*  "          @�=�\)@ҏ\@��A��B�=�\)@��
@��RB*��B�ff                                    Bxw���  �          @��
?�=q@�{@\(�A�B�Ǯ?�=q@�=q@���BLz�B��\                                    Bxw��v  T          @��H���
@���@�
A��B�\)���
@�z�@�33B!�HB��R                                    Bxw��  �          @��L��@��@8��A���B�G��L��@�
=@��RB=��B�L�                                    Bxw���  T          @�녾W
=@�33@�HA�G�B���W
=@��H@��B.�\B�                                    Bxw�h  "          @�\��@���@�A��B�33��@�{@��B)�B��                                     Bxw�  
�          @�33��ff@Ϯ@333A�=qB���ff@��H@���B:�B���                                    Bxw��  
�          @�(���{@�G�@W�A�RB��ÿ�{@�ff@��BJ�\B�
=                                    Bxw�.Z  T          @��G�@��\@b�\A�RB�8R�G�@|��@��BM��B�(�                                    Bxw�=   
�          @�Q��p�@���?�33A3
=Bܞ��p�@���@z�HBQ�B���                                    Bxw�K�  �          @�Q��A�@�Q�.{���B���A�@���@=qA��HB�8R                                    Bxw�ZL  "          @�{�ff@�=q>�ff@eB�#��ff@��H@FffA͙�B���                                    Bxw�h�  �          @��@�\)����  B����@���?#�
@�(�B�\)                                    Bxw�w�  �          @�=q��
=@���x���\)B�33��
=@ٙ������-p�Bǣ�                                    Bxw��>  "          @�
=�fff@�
=��Q��!�\B�z�fff@�=q�ff��ffBÅ                                    Bxw���  T          @ۅ�p��@Ǯ������B�.�p��@�\)?333@�(�Bď\                                    Bxw���  �          @޸R��@�p�>�
=@\(�B�W
��@�ff@EA�
=B�ff                                    Bxw��0  �          @��;aG�@�ff?�A?\)B��aG�@��@z�HB�RB�z�                                    Bxw���  
�          @�ff�p��@\@&ffA�33Bř��p��@��H@���B6  B�33                                    Bxw��|  �          @׮���@��
@@  AӅBΊ=���@��@���BB�B�.                                    Bxw��"  T          @�\)��{@���@VffA��HB�(���{@r�\@�\)BL\)B�8R                                    Bxw���  T          @��z�@��H@��B/Q�B����z�@p�@�B{��C�                                    Bxw��n  "          @�p��1�@�ff@��\B"��B�
=�1�@(�@��
Bg�C�q                                    Bxw�
  T          @�  �{@��@hQ�Bz�B��H�{@S�
@��\BT�HB���                                    Bxw��  T          @��H����@�=q@z�A���B��f����@���@�33B�
B�8R                                    Bxw�'`  "          @ڏ\�5@���@\��A�  B�3�5@U�@��
BF  Cp�                                    Bxw�6  
Z          @��H�HQ�@�Q�@X��A�
=B����HQ�@Tz�@��B?�
CQ�                                    Bxw�D�  T          @����7
=@��@@  A�33B��7
=@z=q@�(�B4��B�G�                                    Bxw�SR  �          @�z��HQ�@�
=@.�RA�G�B�HQ�@�Q�@��
B(��C 
=                                    Bxw�a�  �          @��H�W
=@�33@\)A�  B�Q��W
=@\)@�33B33C!H                                    Bxw�p�  "          @�G��.{@�  @HQ�A�\)B����.{@i��@���B<\)B�W
                                    Bxw�D  "          @أ��C�
@�p�@:�HA�
=B�L��C�
@j�H@�p�B1G�C�H                                    Bxw���  
�          @أ�� ��@�\)@=p�A�G�B�=q� ��@|(�@�=qB7��B�
=                                    Bxw���  �          @�z��#�
@���@%�A��HB�ff�#�
@E@��B5=qC��                                    Bxw��6  �          @�Q�Ǯ@��R@q�B33B؏\�Ǯ@:�H@���Bg��B�
=                                    Bxw���  
�          @����ff@i��@K�A�C�{��ff@�@��\B$�C�                                    Bxw�Ȃ  �          @��
��z�@���@N{A�\)C	����z�@   @�  B&CO\                                    Bxw��(  T          @�\�(�@�ff@p  B�\B�B��(�@Y��@�{BQ��B�L�                                    Bxw���  T          @�  �-p�@���@l��A�=qB�8R�-p�@n�R@�Q�BHz�B�
=                                    Bxw��t  �          @���P��@�
=@u�A��B�  �P��@X��@�Q�BF(�C�H                                    Bxw�  
�          @��@  @�@j=qA�z�B��@  @j=q@�BC�HCQ�                                    Bxw��  �          @�R�p  @�
=@`��A��B�#��p  @R�\@��B6C
                                    Bxw� f  "          @�z���=q@�
=@~{BffC���=q@�H@�Q�B?�
C@                                     Bxw�/  
�          @ᙚ���H@��H@|��B	\)C����H@z�@�{B?ffCxR                                    Bxw�=�  "          @ᙚ�4z�@�=q@�ffB��B�ff�4z�@%�@��\Bc{C	�{                                    Bxw�LX  �          @����/\)@�\)@�z�B%��B��
�/\)@�@�
=Bi��C
^�                                    Bxw�Z�  "          @�Q��}p�@�  @�Q�BQ�C�R�}p�@Q�@�{BI�\C�3                                    Bxw�i�  �          @�  �z=q@'
=@��HB=G�CG��z=q?�@�ffB]=qC,@                                     Bxw�xJ  
�          @ᙚ�dz�@H��@�(�B<��C
���dz�?��\@�BgC$                                    Bxw���  �          @������
@��@�p�B2��C\)���
>���@��RBL�C.Ǯ                                    Bxw���  "          @�����\)@�@�ffB)p�C���\)>#�
@��B<��C2�                                    Bxw��<  �          @�
=��\)@�@���B'=qCG���\)>��R@�(�B={C0E                                    Bxw���  T          @�\���@�R@��HBz�C�����?8Q�@�ffB1C+ٚ                                    Bxw���  "          @���G�?�Q�@���B)��C$\��G���
=@��B2�\C8�=                                    Bxw��.  T          @�����\@��@�ffB�C����\?&ff@�G�B7�
C,W
                                    Bxw���  
�          @�Q��`  @�  @�ffB�
C=q�`  @@��BW�
C+�                                    Bxw��z  �          @�  �p  @�G�@�G�B��C�H�p  @(�@�
=BN(�C��                                    Bxw��   �          @�  ��?��
@�p�B7z�C!H���\)@�\)BF�HC5�                                    Bxw�
�  T          @�Q��tz�@#�
@�  BC(�C(��tz�>��@�=qBb�C,��                                    Bxw�l  �          @�Q��z�H@333@��\B:{Cu��z�H?B�\@�  B\�HC(��                                    Bxw�(  �          @�\)�|(�@5�@��B1=qCL��|(�?k�@�{BUp�C&�{                                    Bxw�6�  	c          @�Q��^�R@��?�Q�A�{B��=�^�R@��H@p��BG�C ��                                    Bxw�E^  �          @�ff����@{�@]p�A��C  ����@�H@���B(�C޸                                    Bxw�T  T          @޸R��  @}p�@S�
A�G�C
����  @   @���B%{C��                                    Bxw�b�  
Z          @����[�@��
@n�RB ffB�\)�[�@L��@��B@��C	                                    Bxw�qP  
�          @޸R�y��@��@g
=A�{Cu��y��@4z�@�ffB7Q�C�                                    Bxw��  T          @�
=��=q@9��@>{A�z�C����=q?��@s�
B�C#�
                                    Bxw���  T          @�ff���@��@\)A���C O\���?�G�@8Q�AĸRC(u�                                    Bxw��B  �          @�G����@Dz�@1G�A�  C�����?��@k�A�\)C!�f                                    Bxw���  
Z          @�z���{@?\)@i��BQ�C����{?��R@�\)B#�C#0�                                    Bxw���  "          @�\)����@8Q�@r�\B�
C!H����?���@��B%=qC$�q                                    Bxw��4  T          @�\)����@�@�z�B'=qC������>�  @��B:�\C0��                                    Bxw���  T          @�{���?��@��B8G�C#Ǯ�����\@�z�B?��C:\)                                    Bxw��  T          @�����p�?�\)@��
B;z�C&z���p��B�\@�{B>��C=L�                                    Bxw��&  T          @�\)����>�p�@��B:z�C/�)���׿���@��B1=qCD�)                                    Bxw��  	�          @�p����
�#�
@��
B3�C5�����
��\)@�  B#(�CH��                                    Bxw�r  	�          @�
=��?�\)@�
=B6=qC#�f����@�(�B>�C9�                                    Bxw�!  
�          @�p����?�\@�\)B8�
C-�)������\@��HB2Q�CC                                      Bxw�/�  %          @�����ff@&ff@���B!�
C����ff?c�
@���B?�C(��                                    Bxw�>d  Q          @�ff���@��H@j�HB G�B�8R���@n{@���BH�RB�u�                                    Bxw�M
  T          @�\)��33@ə�@�A��B�W
��33@��\@�p�Bp�B���                                    Bxw�[�  %          @�ff���H@���@7
=A�\)B��
���H@�z�@��
B3  B�W
                                    Bxw�jV  
�          @��
��(�@���@<��A���B�ff��(�@��@�ffB1=qBոR                                    Bxw�x�  
�          @�Q쿓33@�p�@ ��A��RB�Q쿓33@�z�@��HB${B�8R                                    Bxw���  T          @�\)��(�@��
@\(�A�G�B�z��(�@vff@��HBI��B�(�                                    Bxw��H  �          @�
=�Q�@q�@��B/�\C���Q�?��@��\Bd
=C0�                                    Bxw���  �          @�
=�G
=@�p�@��B'\)B�� �G
=@33@��BaG�C�                                     Bxw���  �          @�Q��g
=@U@�Q�B6�C	@ �g
=?�z�@�33Ba��C�R                                    Bxw��:  	�          @�ff�U@X��@��B;  C���U?�Q�@�p�Bi��C�3                                    Bxw���  "          @߮�U@fff@��RB5{C���U?�
=@�z�Bfz�CG�                                    Bxw�߆  
�          @����e�@o\)@�\)B)�C�=�e�?��@�
=BZC+�                                    Bxw��,  "          @߮�tz�@�\@��
BI=qC��tz�>��R@��\Bb�\C/T{                                    Bxw���  T          @�  �XQ�@   @�Q�B^�
CxR�XQ콸Q�@�33Bs�
C5�\                                    Bxw�x  
Z          @��e@��@��BM�Cc��e>���@���Bj(�C-�                                    Bxw�  
�          @��L��@b�\@���B9��C\�L��?�\)@�BkG�C�                                    Bxw�(�  
�          @�p��U@b�\@��B533CW
�U?�z�@�=qBe�C�{                                    Bxw�7j  
(          @�z��\(�@&ff@��BL�RC���\(�?!G�@�p�Bm�RC)��                                    Bxw�F  T          @�p��]p�@>�R@�{BB�C@ �]p�?�ff@�z�Bi��C#�                                    Bxw�T�  "          @�z��L��@e�@�p�B6�C���L��?ٙ�@��HBh�C��                                    Bxw�c\  "          @�z��N�R@�(�@�p�B ffB����N�R@��@���BX=qCT{                                    Bxw�r  
�          @�(��L��@��\@�G�B�B��H�L��@(��@�  BS�HC}q                                    Bxw���  
�          @�{�2�\@��H@�=qBQ�B��H�2�\@L(�@�ffBP�C�                                    Bxw��N  �          @�
=�~{@G
=@�p�B$�HC�H�~{?���@�ffBKC��                                    Bxw���  
�          @����@2�\@�  B-C.���?��@���BN33C%�3                                    Bxw���  �          @�{���@@�(�B3�C}q���?z�@�(�BLQ�C,aH                                    Bxw��@  
�          @�p�����@33@�
=B9\)C�����?�@�ffBR33C,�f                                    Bxw���  �          @�  ���@��@�z�B(�C{���?J=q@�B3
=C*�
                                    Bxw�،  "          @�
=��@3�
@Mp�A�C�)��?˅@}p�B{C#�3                                    Bxw��2  
�          @�ff����@�@Y��A��C0�����?���@~�RBC(��                                    Bxw���  
�          @�ff��\)?�Q�@��B)G�C��\)>�z�@���B:�
C0�                                    Bxw�~  T          @����@*=q@�ffB�\C����?�\)@��HB1�C&�f                                    Bxw�$  
�          @޸R��p�@  @�z�B'z�C=q��p�?��@�(�B=�
C,��                                    Bxw�!�  
�          @�\)���@�\@�Q�B7��C����>�\)@���BK
=C0T{                                    Bxw�0p  
Z          @�{���@�R@��B%(�C� ���?W
=@��
B?=qC)�                                     Bxw�?  	�          @�ff��ff@
=@��B(ffC����ff>��@��HB<�\C.B�                                    Bxw�M�  &          @�����{?�
=@��
B�C����{>Ǯ@�Q�B/�C/n                                    Bxw�\b            @�(�����?�\@���BQ�C!� ����>�Q�@�(�B\)C0�                                    Bxw�k  
�          @�����?\@�=qB  C$B�����=�G�@�33BffC2��                                    Bxw�y�  	�          @���?Ǯ@e�A�{C$�)��>��
@x��B	p�C0��                                    Bxw��T  
Z          @��
���H>�ff@�  B��C/33���H�s33@��B{C>)                                    Bxw���  T          @�33��(��k�@J�HA�{C<���(���33@,��A���CE�f                                    Bxw���  �          @����7
=@-p�@�
=BbG�C�{�7
=?#�
@���B���C'xR                                    Bxw��F  
�          @���'�@O\)@�ffBW\)C��'�?�p�@�{B��{C��                                    Bxw���  
�          @�33�  @fff@�{BU�B�  �  ?�=q@љ�B�u�C�                                    Bxw�ђ  �          @�zῷ
=@�G�@�{BT
=B�  ��
=@   @�B���B�#�                                    Bxw��8  P          @�33��G�@~�R@�
=BW��B�(���G�?�
=@�{B��B�(�                                    Bxw���  �          @�=q�qG�?ٙ�@�33BKC��qG����
@��
B[�RC4T{                                    Bxw���  "          @�(���
=?\@���B/�\C"#���
=���
@�Q�B:C4��                                    Bxw�*  T          @�p����?�p�@��B-=qC"���녽�G�@�
=B7�C5L�                                    Bxw��  
�          @޸R����?�p�@�z�B2�C%�������Ǯ@���B8��C8��                                    Bxw�)v  
�          @߮��p�?���@�
=B5�HC"� ��p��L��@�B?��C6xR                                    Bxw�8  
Z          @�
=��ff?�(�@�B433C"�)��ff�.{@�z�B>Q�C6�                                    Bxw�F�  
,          @���z�?�G�@�p�B4C"
=��z��@�z�B?�C5��                                    Bxw�Uh  �          @�p���  @   @��\B1�C���  >�p�@�
=BD�C/J=                                    Bxw�d  
�          @�{����@   @��HB4�Cn����?E�@��
BN
=C*�                                    Bxw�r�  
�          @�p�����@(�@��RB/ffC�����?B�\@�
=BG�\C*��                                    Bxw��Z  
�          @����{@(�@�p�B9(�C���{>�ff@�33BM�
C.&f                                    Bxw��   
�          @�R���R@�R@��
B,(�C����R?
=@�=qBA  C,�\                                    Bxw���  
�          @�G����@O\)@��HB0�C� ���?��R@�33BT��C �                                    Bxw��L  
�          @�
=�\)@+�@�(�B@�
C
�\)?^�R@�ffB^{C'��                                    Bxw���  
Z          @�
=���@  @��BF
=CT{���>�(�@��B\33C.�                                    Bxw�ʘ  
�          @����@G�@��HB>z�CB����?   @���BT\)C-ff                                    Bxw��>  "          @�R����@  @�G�BI  C������>�
=@��RB_p�C.                                    Bxw���  
�          @���(�@\)@���B@�
C�3��(�?333@���BZz�C*k�                                    Bxw���  
�          @�  �}p�@)��@�BB�C33�}p�?Y��@��B_��C'�                                    Bxw�0  
�          @�  ��  @%@���B:��C����  ?W
=@��BU33C(�{                                    Bxw��  �          @陚��
=@'
=@��B3�C�=��
=?fff@��RBMffC(�                                    Bxw�"|  �          @����  @"�\@��B2�HC�\��  ?Y��@�z�BK�HC)Q�                                    Bxw�1"  "          @�G���z�@=q@�G�B:33C33��z�?.{@���BQ�RC+5�                                    Bxw�?�  �          @������
@
=q@��B?�RC�����
>��@��BS��C.�f                                    Bxw�Nn  �          @�  ����@��@�p�BA�C�R����?#�
@�z�BZ
=C+L�                                    Bxw�]  
�          @�R��z�@:=q@��
B5=qC�H��z�?�(�@�Q�BT��C#�)                                    Bxw�k�  
Z          @�  ����@�@��\B==qC�����?#�
@���BTz�C+xR                                    Bxw�z`  P          @�R���
?�=q@��B=�
C!&f���
��G�@�G�BH�
C5Y�                                    Bxw��  "          @����?�\)@�p�BDQ�C�������Q�@��BP=qC533                                    Bxw���  �          @���z�?�\@�z�BBC���z�=u@�p�BP��C333                                    Bxw��R  �          @�{��  ?�33@��
BN  C!�{��  ��Q�@���BVQ�C8��                                    Bxw���  �          @����?�z�@�BE��C���>B�\@�  BV=qC1p�                                    Bxw�Þ  �          @�ff���@��@���B>  C�����>�ff@�ffBQ�C.!H                                    Bxw��D  "          @�\��(�?���@��RB=�C��(�>���@��BN�
C0�                                    Bxw���  �          @ᙚ��z�?�z�@��\BD��C"+���zᾀ  @�Q�BM�C75�                                    Bxw��  �          @����@Q�@�(�BBz�CO\���>�(�@���BVQ�C.:�                                    Bxw��6  "          @������@	��@�  BC\)C}q����>��@�z�BV�C.��                                    Bxw��  �          @�G����@G�@�  B8��C33���?(�@�{BM�HC,Q�                                    Bxw��  
�          @�p���\)@
=@��\B4p�C.��\)?@  @���BJ��C*}q                                    Bxw�*(  	�          @��
����@�H@�B/\)C�\����?\(�@�BF�RC)B�                                    Bxw�8�  
�          @������@#33@��B6Q�CaH���?n{@�z�BO�
C'ٚ                                    Bxw�Gt  "          @�33�}p�@p�@��RBI33C�{�}p�?   @��
B_(�C,                                    Bxw�V  �          @��x��?�Q�@��BT�CxR�x�ý#�
@���Bbz�C4��                                    Bxw�d�  T          @�Q��y��?��@�p�BW�HC!
�y���Ǯ@��B`  C9��                                    Bxw�sf  T          @޸R�s33?���@�z�BY�RC c��s33��Q�@�G�BbffC9aH                                    Bxw��  �          @ᙚ���?�ff@�  BM�\CG������@�
=BY{C5�3                                    Bxw���  "          @�(�����?���@�\)B=�\C!H����>�{@�=qBNz�C/�H                                    Bxw��X  
�          @�=q���R@�\@�{B2G�C�����R?@  @���BH{C*ff                                    Bxw���  
�          @����@33@�Q�B6��C(���?�\@�z�BIp�C-s3                                    Bxw���  �          @�G����\?�\)@�Q�B5\)C�)���\>���@��\BE  C/ٚ                                    Bxw��J  "          @��H��G�@G�@�G�B*p�C���G�?��@�p�B;�
C-k�                                    Bxw���  "          @�Q����?޸R@�{B?�\C.���>.{@�\)BM�
C1                                    Bxw��  
^          @ᙚ���?��@��B.z�CW
���>���@�ffB>�C/+�                                    Bxw��<  
�          @����{@�@�z�B��Cz���{?�=q@���B%
=C(L�                                    Bxw��  "          @�{��{?�  @�=qB/�C����{>�z�@��
B=�
C0h�                                    Bxw��  �          @�(���  @�G�@N�RA�
=C
{��  @<(�@�Q�Bp�C��                                    Bxw�#.  �          @�(��|��@�p�@p��B�
Cn�|��@:�H@���B1G�C�                                     Bxw�1�  
�          @��
�~�R@}p�@y��BG�C&f�~�R@+�@��
B5C
=                                    Bxw�@z  	�          @�(�����@e�@j=qB�C�{����@��@���B%p�CQ�                                    Bxw�O   �          @�(�����@@��@W
=A�G�C������?�
=@��\B\)Cff                                    Bxw�]�  
�          @�����\@`��@���B�HC�q���\@{@�(�B4=qC�f                                    Bxw�ll  
Z          @���u@c33@�  B"�HC	B��u@Q�@��\BI�HC                                      Bxw�{  
(          @�����\@\(�@���B  C�
���\@�
@�{BB��C:�                                    Bxw���  T          @��tz�@Q�@�  B-p�CQ��tz�?�@�\)BQ��C��                                    Bxw��^  
(          @��
��=q@
=q@�33B�\C���=q?k�@�G�B%�C)�q                                    Bxw��  
Z          @�p���G�@\)@��B\)C���G�?�
=@��HB1�C&(�                                    Bxw���  �          @������\@��@}p�B�C�=���\?�Q�@�
=B"Q�C&�)                                    Bxw��P  
�          @��
���R@�
@�{B ��C�f���R?z�H@��B6(�C(O\                                    Bxw���  	�          @����z�@%�@���B'33C����z�?��H@��HB@�
C$��                                    Bxw��  �          @����
=@!�@�=qB�C���
=?��@��B'C%h�                                    Bxw��B  
�          @�=q��33?fff@�33B(�C*h���33��=q@�{B  C6�f                                    Bxw���  T          @��H��ff?�\@S33A�33C/)��ff��(�@S�
A�{C8�                                    Bxw��  
Z          @��H���?�Q�@QG�A���C(�����>u@^{A�\C1��                                    Bxw�4  T          @�����H?��@i��B�HC&�����H>�  @w�B
��C1s3                                    Bxw�*�  "          @�33��Q�@   @��B�C0���Q�?G�@���B'�C++�                                    Bxw�9�  "          @�33���?#�
@�=qB4��C,B�����5@��B4ffC<�                                     Bxw�H&  T          @�33����=�Q�@���B<�RC2�����Ϳ��@�33B4�\CC��                                    Bxw�V�  
�          @��H��Q��@��B7�C5�{��Q쿼(�@�B,��CE5�                                    Bxw�er  �          @ڏ\��{?k�@\)Bp�C*\)��{�L��@��HB��C6&f                                    Bxw�t  �          @��
���H?���@n�RB�C&+����H>���@~{B\)C0�f                                    Bxw���  
V          @�z����?n{@���B��C*���녾�=q@��
B�RC6�3                                    Bxw��d  
�          @ۅ���\?��R@�G�Bp�C$\)���\>�{@�G�B(�C0c�                                    Bxw��
  
�          @�(���ff?���@q�B	�HC ��ff?8Q�@�(�B�
C,!H                                    Bxw���  
�          @�{���H@G
=@Dz�A�33C�q���H@
=q@s33B��C�                                    Bxw��V  T          @�p���z�@!G�@^{A�Q�C�3��z�?�p�@���B{C$��                                    Bxw���  T          @�ff��ff@0  @i��A�ffC(���ff?�z�@�Q�B�RC"O\                                    Bxw�ڢ  
�          @���?�33@�z�B��C���?#�
@��B.�\C,��                                    Bxw��H  T          @�
=���@0��@���B33C�{���?�(�@�\)B8p�C")                                    Bxw���  "          @�
=��z�@�H@��B�C����z�?�@���B.�\C&�\                                    Bxw��  	�          @���ff@
�H@�G�B�RCY���ff?p��@��RB,\)C)E                                    Bxw�:  
Z          @�p���@�
@|��Bp�C�R��?�@�B33C'L�                                    Bxw�#�  "          @�  ���R@Q�@�33BQ�CxR���R?���@��\B"p�C'\                                    Bxw�2�  �          @�
=����@   @��HB�CaH����?B�\@�
=B+G�C+\)                                    Bxw�A,  "          @޸R��
=?��R@�33B�RC$���
=>�\)@��\B%  C0�R                                    Bxw�O�  �          @�\��\)@#�
@xQ�B�\C�H��\)?�Q�@�p�B��C$�{                                    Bxw�^x  "          @�G���p�@Y��@VffA�33C����p�@��@�(�B��C=q                                    Bxw�m  
�          @������@P��@N�RA��
Ck�����@�@~�RB
p�C�                                    Bxw�{�  
�          @�G���@>�R@UA��C=q��?��R@�Q�B
�Cٚ                                    Bxw��j  
�          @�\��=q@3�
@S�
Aߙ�CG���=q?�=q@{�B\)C!��                                    Bxw��  
Z          @��H����@Y��@S33A�=qC5�����@=q@��\B��CxR                                    Bxw���  �          @�����@b�\@?\)Aȏ\CJ=���@(Q�@s�
B��C��                                    Bxw��\  "          @�ff���
@�33@0��A��C�)���
@O\)@o\)Bp�CaH                                    Bxw��  "          @�{��  @z=q@+�A�33C��  @Dz�@g
=A���Cz�                                    Bxw�Ө  T          @�  ���>�Q�@n�RB(�C0^�����!G�@l(�B�C:ff                                    Bxw��N  �          @�  ��  ?333@s�
B
  C,�R��  ���
@w
=B�C7G�                                    Bxw���  T          @�Q�����?5@tz�B	��C,�����þ��R@w�B��C7:�                                    Bxw���  T          @�����!G�@l(�Bp�C:G����Ǯ@Y��A�{CCaH                                    Bxw�@  �          @ٙ����R�B�\@j�HB�HC5�f���R���@`  A��C?=q                                    Bxw��  �          @�  ���R���
@dz�A��
C78R���R���R@W�AC@33                                    Bxw�+�  �          @�
=��\)>k�@w
=B��C1����\)�G�@r�\B	��C<�                                    Bxw�:2  "          @�������>�
=@���B=qC/�����Ϳ#�
@���B
=C:��                                    Bxw�H�  �          @��
��Q�?.{@��B�
C,�H��Q���@�z�B
=C9�                                    Bxw�W~  �          @ڏ\��?fff@�z�B+
=C)�������
@��RB.�C7�q                                    Bxw�f$  �          @�ff��=q��@5�A��C4G���=q�@  @.�RA�=qC;{                                    Bxw�t�  
�          @ָR��G����@:=qA�C5s3��G��c�
@1�A�p�C<h�                                    Bxw��p  "          @���
=�W
=@\(�A�  C6)��
=��=q@QG�A��C>�R                                    Bxw��  �          @������R>aG�@hQ�B�RC1�\���R�5@c�
A�=qC;!H                                    Bxw���  T          @�=q�����@j=qB(�C4T{����u@a�A�{C=�=                                    Bxw��b  �          @�����=q>B�\@^{A�(�C2&f��=q�333@Y��A���C:ٚ                                    Bxw��  �          @�=q��
=>k�@l��BffC1�3��
=�8Q�@hQ�B �HC;!H                                    Bxw�̮  "          @�G�����?\)@w�BQ�C.:����þ�G�@xQ�B��C8�=                                    Bxw��T  �          @����녽��
@xQ�B�RC4ٚ��녿��@o\)B�RC>��                                    Bxw���  �          @����
=�L��@���B�C6���
=���H@vffB
  C@��                                    Bxw���  T          @�G���(�>�ff@��B{C/.��(����@��HBQ�C:T{                                    Bxw�F  T          @أ�����?Y��@�z�B�C*�����׾aG�@�
=B��C6\)                                    Bxw��  �          @�����Q�?��@j=qB�C-��Q쾮{@l(�B�C7z�                                    Bxw�$�  �          @�z����=�\)@j�HB
=C3L�����\(�@dz�B�HC<ٚ                                    Bxw�38  �          @���ff    @uB��C3����ff�u@n{B�\C=��                                    Bxw�A�  
^          @�(���=q?�@�\)B�
C.��=q�
=q@�\)BC:)                                    Bxw�P�  
Z          @��
��p�>8Q�@q�Bp�C2���p��E�@l��BQ�C<�                                    Bxw�_*  �          @ҏ\���
>�33@U�A���C0n���
���H@S�
A�C8��                                    Bxw�m�  "          @�G���G�>���@[�A���C0����G��\)@Y��A��HC9�                                    Bxw�|v  �          @����
==��
@J�HA��
C3=q��
=�5@EA�p�C;�                                    Bxw��  �          @�p���  >k�@i��B(�C1�\��  �.{@eB�
C;�                                    Bxw���  �          @���\)?�(�@b�\B �
C'����\)>���@n�RB�C0�
                                    Bxw��h  T          @�=q��
=�:�H@!�A�ffC:����
=��=q@G�A���C@�\                                    Bxw��  �          @����
=�z�@)��A��\C9���
=���H@�HA���C?z�                                    Bxw�Ŵ  "          @�=q���
�E�@	��A�{C;#����
���\?�33A�G�C?                                    Bxw��Z  
(          @�(���z�   @��A�p�C8����zΉ�@  A���C>
=                                    Bxw��   
Z          @�p��Å���@(Q�A��C9���Å��p�@��A���C?W
                                    Bxw��  �          @�
=��
=�G�@A�(�C;#���
=��=q@�A��C@\                                    Bxw� L  �          @�Q����
���@S33A�=qC8�����
���\@EAٙ�C@.                                    Bxw��  
Z          @׮��ff�L��@ ��A��HC;Y���ff���@\)A�33C@�)                                    Bxw��  T          @׮�Ǯ�O\)@=qA��
C;c��Ǯ��\)@��A�ffC@ff                                    Bxw�,>  �          @�  ����O\)@?\)A��HC;�R����\@-p�A�G�CB33                                    Bxw�:�  
�          @أ���
=�h��@AG�A���C<����
=��{@-p�A��CC!H                                    Bxw�I�  �          @أ���
=�s33@_\)A�ffC=k���
=��\@I��A�z�CE(�                                    Bxw�X0  �          @�Q���{��
=@?\)A�z�C?5���{��\)@'�A���CEs3                                    Bxw�f�  �          @�\)��G��fff@5�A���C<����G��Ǯ@!�A�Q�CB}q                                    Bxw�u|  �          @�ff��Q�#�
@5A��C:���Q쿦ff@&ffA�G�C@:�                                    Bxw��"  �          @�\)��{�0��@7
=A�ffC:����{��{@'
=A�\)C@��                                    Bxw���  �          @����
==�\)@�G�BC3:���
=�u@�B  C>xR                                    Bxw��n  �          @�33���R>�@�p�BDG�C-����R�G�@�z�BB\)C=�                                    Bxw��  �          @��H��33=�\)@w�B
ffC3B���33�\(�@qG�B�C<��                                    Bxw���  "          @��
��  >���@���B�RC0�{��  �&ff@\)B�C:��                                    Bxw��`  
�          @�(����>�@�G�B�RC2������fff@�ffB��C=�                                    Bxw��  �          @�����G�����@�B�C5)��G���z�@���B(�C@Q�                                    Bxw��  
�          @�{��Q�>\)@���B#  C2u���Q�p��@�B��C>(�                                    Bxw��R  "          @�z����׽u@�B 33C4�����׿�{@�G�B�C?�                                    Bxw��  �          @�z���
=��\)@�\)B"33C7���
=��=q@���B�
CBQ�                                    Bxw��  T          @�p����þ�  @�ffB z�C6� ���ÿ�ff@�Q�Bp�CA�)                                    Bxw�%D  �          @�����׾��@���B��C6Ǯ���׿��\@��B��CA��                                    Bxw�3�  
�          @�33���
�k�@�G�B&(�C6�����
��ff@�33B
=CB=q                                    Bxw�B�  
�          @�����33�u@��B@{C4����33��G�@��RB8\)CCW
                                    Bxw�Q6  "          @ۅ��33?(��@�
=B/�HC,8R��33���@��B0��C9��                                    Bxw�_�  T          @�ff��G���{@q�B	\)CD����G����@R�\A�CL޸                                    Bxw�n�  �          @�ff��33����@l��B��CD����33��H@N{A噚CLW
                                    Bxw�}(  �          @ٙ����þaG�@�BffC6aH���ÿ�Q�@�Q�B33C@�                                     Bxw���  T          @ۅ����=���@���B*�HC2�����ÿ}p�@�G�B&{C?.                                    Bxw��t  T          @������?�\@���B��C.�=�����\@���B��C9n                                    Bxw��  �          @ۅ��G�>W
=@�z�B*G�C1�H��G��aG�@��B&�RC=��                                    Bxw���  �          @ۅ��{>L��@�  B/\)C1����{�h��@�p�B+�C>xR                                    Bxw��f  "          @�=q��\)�aG�@���B,{C6}q��\)���@�
=B$  CB��                                    Bxw��  �          @ڏ\���R�(�@�G�B)
=C:�����R���@���B�CFY�                                    Bxw��  �          @�{���ÿ�(�@?\)A��HCB:�������@#�
A���CH#�                                    Bxw��X  �          @�\)��zῸQ�@:=qA�G�CAǮ��z���@   A�ffCGu�                                    Bxw� �  "          @�Q���
=���\@7�A�\)C@���
=��33@   A�CE�                                    Bxw��  �          @׮��ff��  @6ffA���C?����ff���@�RA��CE�                                    Bxw�J  T          @�\)���\����@@��A��HCA�����\�
=@%A���CG�)                                    Bxw�,�  T          @�ff��Q��  @:=qAΣ�CD޸��Q���@�A���CJaH                                    Bxw�;�  T          @�\)���
��@<��A�ffCA�H���
�z�@#33A��CGaH                                    Bxw�J<  "          @׮��ff��{@>{A��C>����ff��\@(��A�  CD�=                                    Bxw�X�  T          @�ff��\)�s33@Tz�A�Q�C=^���\)��
=@AG�A�z�CDc�                                    Bxw�g�  "          @�{��{����@VffA�Q�C>����{��@@��A�{CE��                                    Bxw�v.  �          @�{��녿�z�@EA�{C?J=��녿�@/\)A��CE�=                                    Bxw���  �          @�p��������@B�\AمCB+�����
=@(Q�A�{CH(�                                    Bxw��z  �          @�ff��=q�z�H@J�HA�=qC=���=q��
=@7�Aʣ�CD�                                    Bxw��   �          @�(�����n{@Y��A��RC=aH�����
=@FffA���CD�f                                    Bxw���  T          @љ���33��G�@N�RA��C>+���33��(�@:�HAӮCE�                                    Bxw��l  T          @������333@S33A�C:�������
=@C33A�z�CB�                                    Bxw��  �          @�(���G��=p�@H��A�  C;Q���G���Q�@8Q�A�ffCA�3                                    Bxw�ܸ  T          @�=q���׿z�@EA��C9�q���׿��\@8Q�A�  C@c�                                    Bxw��^  T          @�(���33�\)@A�A�(�C9z���33���R@5�A�=qC?��                                    Bxw��  �          @�{���8Q�@AG�AָRC:�3�����@1�A�Q�CA0�                                    Bxw��  T          @�ff��p��Ǯ@FffA���C7�\��p����@;�A�C>ff                                    Bxw�P  �          @�����Q�k�@Tz�A�C6O\��Q�z�H@K�A���C=��                                    Bxw�%�  �          @Ӆ��녾��@HQ�A�=qC5p���녿Y��@@��A�G�C<O\                                    Bxw�4�  �          @�z���(�<��
@a�B Q�C3����(��G�@\(�A���C;�H                                    Bxw�CB  T          @��������@s�
B
�RC7h������
=@hQ�B��C@�                                    Bxw�Q�  T          @�{���Ϳ�ff@.�RA�33CB�������@z�A�=qCGǮ                                    Bxw�`�  �          @�ff��������@Dz�A�z�CA!H����� ��@,(�A�p�CG)                                    Bxw�o4  �          @�
=��(��u@FffA�C=8R��(��У�@3�
A�p�CCs3                                    Bxw�}�  �          @ָR���R��z�@4z�A�33C?����R��G�@\)A�ffCDz�                                    Bxw���  �          @�
=��\)���H@+�A�ffCA��\)��@�\A�G�CF��                                    Bxw��&  �          @�����Ϳ��H@%�A�(�CD.�����  @Q�A�33CH�
                                    Bxw���  �          @��H���ÿ��H@H��A�(�CE+�������@+�A�Q�CK0�                                    Bxw��r  �          @Ӆ���H��
=@W�A���CG�����H�(��@6ffA��HCNL�                                    Bxw��  �          @�p���{��\@QG�A���CH�
��{�.{@.�RA�p�CN�{                                    Bxw�վ  �          @�����Q�@\(�A��CL�f���E�@5�A���CR�                                    Bxw��d  T          @����ff���@_\)A��HCK����ff�>�R@9��A�z�CQ�{                                    Bxw��
  �          @���
=�Q�@q�B	�HCM����
=�J=q@J=qAᙚCTz�                                    Bxw��  "          @�(����
�!�@l��B
=COz����
�R�\@C33A�G�CV\                                    Bxw�V  "          @��
��
=�@`  A�33CI���
=�3�
@<��Aә�CPJ=                                    Bxw��  �          @ҏ\��33��@G
=A�(�CB33��33�z�@.{AÙ�CHL�                                    Bxw�-�  
�          @�����׿8Q�@0��A��C:�=���׿��@!�A��HC@J=                                    Bxw�<H  �          @�z���G���p�@.�RA�ffC7}q��G��u@%�A�33C=�                                    Bxw�J�  �          @�33�\���@\)A���C9���\���@33A�
=C>�)                                    Bxw�Y�  
�          @�Q���\)�
=@ ��A�33C9�H��\)����@z�A���C>��                                    Bxw�h:  "          @�
=����Tz�@G�A���C;�f������@�\A���C@\)                                    Bxw�v�  �          @Ϯ��(���{@
=A�
=CA  ��(�����@ ��A�(�CEk�                                    Bxw���  �          @�p����
���@��A���C@�{���
���
?�z�A��RCD�{                                    Bxw��,  �          @�\)��
=��
=@�RA��HC?0���
=���?�z�A�p�CCc�                                    Bxw���  �          @�p���G���\)@��A��CAT{��G���\)@�\A�z�CE޸                                    Bxw��x  T          @��
����
=@Q�A�
=CDp����
=q?���A�z�CH��                                    Bxw��  �          @�(���{��=q@   A�Q�C>Q���{���R?��HAyp�CB�                                    Bxw���  �          @�����z��  ?�(�A�CBQ���z��33?˅Ag
=CE�H                                    Bxw��j  
�          @�(���p����H?��AmCD���p���?�(�A0��CF��                                    Bxw��  T          @���(���p�?��HA��CD\)��(��
=?��
A\Q�CG                                    Bxw���  
�          @�{��{��  @ffA���C?޸��{��
=?��
A�ffCC�                                    Bxw�	\  T          @�=q��  �E�?�{A�z�C;G���  ��33?��Ap��C>��                                    Bxw�  	�          @�����H���?�33A�=qC8p����H�aG�?�  A~{C<0�                                    Bxw�&�  
�          @�  ��G��
=?��RAT��C9Q���G��fff?�=qA=�C<�                                    Bxw�5N  "          @У��ʏ\��(�?�  ATz�C7�H�ʏ\�=p�?�\)AAC:�R                                    Bxw�C�  
Z          @Ϯ�˅=���?��
A6�RC3\�˅�8Q�?��
A5��C5��                                    Bxw�R�  
(          @θR���
��?��
A
=C4@ ���
��  ?�  A�\C6B�                                    Bxw�a@  �          @����H=���?��Ap�C3
���H��?��A��C5(�                                    Bxw�o�  �          @Ϯ���þ�z�?��
A[\)C6�����ÿ�R?�
=AL��C9��                                    Bxw�~�  "          @�
=���
�J=q?��HA���C;O\���
��Q�?޸RAy�C?                                    Bxw��2  �          @У���\)���?�A�G�C8E��\)�Y��?ٙ�AqG�C;��                                    Bxw���  �          @�\)�ȣ׾�z�?�=qAb{C6���ȣ׿!G�?�p�AS\)C9�R                                    Bxw��~  �          @����{�k�?��Al��C6)��{�z�?�ffA_�C9T{                                    Bxw��$  
Z          @θR�ȣ�>W
=?�  AV�RC2��ȣ׽�?�G�AX(�C5)                                    Bxw���  �          @θR���;�=q?�\A�z�C6�=���Ϳ&ff?�Aq�C:
=                                    Bxw��p  
�          @�  ���(�?��A�G�C9������  ?��HAt��C=33                                    Bxw��  �          @�G���=q����?��Ah  C6�q��=q�&ff?��
AX��C9޸                                    Bxw��  "          @θR���þL��?\AY�C5�\���ÿ�?�Q�ANffC8                                    Bxw�b  
�          @�ff��Q�B�\?��
A[\)C5� ��Q��?���AP  C8�R                                    Bxw�  
�          @θR�ȣ׿\)?��HAP��C9
�ȣ׿\(�?�ffA:=qC;��                                    Bxw��  T          @�\)�ʏ\��=q?���A@��C6s3�ʏ\�\)?�G�A3�C9                                    Bxw�.T  �          @�\)��  �
=q?�=qAb�RC8����  �^�R?�
=AL(�C;��                                    Bxw�<�  T          @�����
=��=q?���AQG�C6����
=�z�?���AC�C9T{                                    Bxw�K�  �          @��
��
=��Q�?���AC�C4�
��
=�\?�ffA<  C7}q                                    Bxw�ZF  T          @�33��ff�B�\?�\)AF�RC5� ��ff��?��A<  C8k�                                    Bxw�h�  
�          @��
��{��\)?��HAS�
C6�H��{�
=?�{AEC9u�                                    Bxw�w�  "          @�z���\)��
=?�=qA@z�C7�f��\)�333?��HA.�\C:k�                                    Bxw��8  T          @��
��\)���R?��A;�C6�H��\)�z�?���A-�C9\)                                    Bxw���  �          @�z��Ǯ�Ǯ?��A:�\C7�)�Ǯ�(��?�
=A)C:�                                    Bxw���  "          @��H��z�\?\A]C7����z�333?�33ALQ�C:}q                                    Bxw��*  "          @�p���{�k�@p�A�p�C<Ǯ��{��\)?��HA���CA                                      Bxw���  T          @�{��G���G�@   A�G�C@G���G���\@�A�{CD�q                                    Bxw��v  
�          @���ff����@�A��\C>Ǯ��ff�Ǯ?�=qA��\CB�R                                    Bxw��  �          @�p����H�u?��A�\)C<����H����?��A^=qC@8R                                    Bxw���  �          @�ff�\��\)?��A��C>n�\��  ?���Ae�CA�
                                   Bxw��h  "          @�ff���ÿ���@G�A��
C>T{���ÿ�G�?޸RAyCB�                                   Bxw�
  "          @�\)��Q쿚�H@ffA�p�C?Y���Q�У�?��A���CC+�                                    Bxw��  
�          @У���p�����@ffA�
=C@޸��p�����@   A�G�CE�                                    Bxw�'Z  
�          @�Q���p����H@  A�Q�CA����p���33?��A�33CE��                                    Bxw�6   �          @�ff��(���p�?��A8Q�CA����(���(�?p��A=qCC�                                    Bxw�D�  �          @���p���ff?��RA2{C?���p����?n{A��CA�q                                    Bxw�SL  
�          @�p������z�?���AB�RC>�������?���A{C@�q                                    Bxw�a�  
Z          @����{�Y��?���AQ��C;�\��{���?��RA2{C>p�                                    Bxw�p�  T          @�(���p��O\)?�33AJ�\C;z���p����?�Q�A,z�C>�                                    Bxw�>  
�          @�(�����ff?�
=A*�HC=�������
?n{A=qC?�                                    Bxw���  "          @�����
=�^�R?�Q�A+�C;�R��
=��{?z�HAz�C>�                                    Bxw���  T          @�=q�Å�xQ�?�G�A8Q�C=  �Å���H?��\Ap�C?=q                                    Bxw��0  �          @�����33�fff?�z�A)�C<aH��33����?p��A	p�C>s3                                    Bxw���  
�          @Ǯ��G����?�A,Q�C=���G����
?h��A�\C?�q                                    Bxw��|  �          @����z�xQ�?��
A�HC=  ��z῕?L��@陚C>��                                    Bxw��"  
�          @˅�ƸR�u?}p�A33C<���ƸR��33?B�\@��
C>�                                     Bxw���  �          @˅�ƸR����?aG�@�{C=���ƸR���R?!G�@�p�C?E                                    Bxw��n  T          @�33�����?n{A=qC=�q����G�?+�@\C?�=                                    Bxw�  T          @�ff��=q��z�?�z�AJffCCT{��=q��
=?��
A=qCE��                                    Bxw��  �          @Ϯ�\���?�z�AH��CDaH�\�33?�  A�CF��                                    Bxw� `  �          @�\)��(��
�H?�33Al��CHG���(��{?�33A#�
CJ�
                                    Bxw�/  �          @�p���p�� ��?�p�AT��CF���p���?�G�Ap�CI{                                    Bxw�=�  �          @θR���R��?��HAP��CF�
���R�33?}p�AG�CI
                                    Bxw�LR  �          @θR��p��
�H?�AJffCH)��p���H?k�A�HCJ:�                                    Bxw�Z�  �          @�  ��
=��?���A@(�CH
��
=��H?\(�@��CJ{                                    Bxw�i�  �          @�{��z��\)?��A;�CHٚ��z��{?L��@�(�CJ                                    Bxw�xD  �          @�ff��=q��H?��A?�CJ�)��=q�)��?J=q@���CL��                                    Bxw���  �          @θR����
=q?��
AffCG�\�����?
=q@��CI:�                                    Bxw���  �          @�ff������?���A��CE�������
=?(��@�z�CG8R                                    Bxw��6  �          @�ff��33��{?n{A��CD�3��33� ��?�\@�=qCFG�                                    Bxw���  �          @�(���녿�?8Q�@θRCD޸��녿���>��R@1G�CE��                                    Bxw���  �          @ə����׿�
=?@  @ۅCC�H���׿�>\@Z�HCD�3                                    Bxw��(  �          @����녿˅?@  @ڏ\CB� ��녿�(�>Ǯ@dz�CC�
                                    Bxw���  �          @�(������(�?B�\@�=qCA^�����˅>�(�@uCB}q                                    Bxw��t  �          @�(�������?&ff@�(�CA+�����ff>���@>{CB�                                    Bxw��  �          @�z�����Q�?333@�=qCA#����Ǯ>\@Y��CB(�                                    Bxw�
�  �          @����\)��=q?333@�G�C@��\)��Q�>���@fffCA\                                    Bxw�f  �          @�33���Ϳ��H?��@�z�CAT{���Ϳ�ff>�=q@��CB(�                                    Bxw�(  �          @�ff��G���(�?@  @ָRC?  ��G�����>��@�
=C@(�                                    Bxw�6�  �          @�
=�ƸR���
?s33A�CA�{�ƸR�ٙ�?��@��CCG�                                    Bxw�EX  �          @θR���Ϳ�z�?�ffA��CC����Ϳ���?+�@�ffCD�R                                    Bxw�S�  �          @�ff�Å��Q�?��A#33CCz��Å��33?@  @ָRCE=q                                    Bxw�b�  �          @θR��p��ٙ�?J=q@�Q�CCc���p���=q>��@fffCD�                                     Bxw�qJ  �          @�p��Ǯ��z�?c�
@�ffC>�\�Ǯ��=q?�R@���C@                                      Bxw��  �          @�ff�\���
?�(�AR�\CB{�\��ff?�{A{CD�                                    Bxw���  �          @�ff���׿�{?���AAG�CE&f�����ff?k�A33CGB�                                    Bxw��<  �          @�  ���Ϳ��?ǮA_\)C?�����Ϳ�{?�  A1��CB��                                    Bxw���  T          @�ff���ͿO\)?��HAv�RC;xR���Ϳ�33?�  AW\)C>�H                                    Bxw���  �          @�ff���H��=q?��A��CD�R���H� ��?!G�@��\CFE                                    Bxw��.  �          @θR������
=?���Az�CE�f������?(��@�33CGB�                                    Bxw���  �          @�����=q�G�?��AA��CIO\��=q� ��?Q�@�(�CKG�                                    Bxw��z  �          @�p���(���?�AMG�CG�
��(��Q�?p��A=qCJ�                                    Bxw��   �          @����p�����?n{A�\C@\)��p��\?�R@��HCAٚ                                    Bxw��  �          @�(���{����?5@���C@��{��Q�>��@l��CA
                                    Bxw�l  �          @�(���\)��{?L��@�ffC>#���\)��G�?
=q@��
C?n                                    Bxw�!  T          @�����z��{?
=@���CB���z�ٙ�>k�@�CCs3                                    Bxw�/�  T          @���˅�.{>Ǯ@_\)C:��˅�@  >u@
=C:�3                                    Bxw�>^  
�          @������H�+�?\)@�G�C:\���H�G�>���@g�C;                                      Bxw�M  �          @������H���?��@�33C8�3���H�(��>�@��C9��                                    Bxw�[�  �          @�����
��ff?
=q@�(�C8
=���
�\)>�(�@x��C9                                      Bxw�jP  �          @�(���33��(�>�(�@y��C7�H��33��\>���@>{C8�H                                    Bxw�x�  �          @��
���H��>�33@L(�C80����H��>�  @p�C8�                                    Bxw���  �          @��H��녿\)>L��?��C9{��녿
==�\)?(��C9W
                                    Bxw��B  �          @��H��녿\)>L��?���C9
=��녿
==��
?8Q�C9Q�                                    Bxw���  �          @˅���H��G�    <#�
C7�q���H��(���Q�Y��C7�f                                    Bxw���  �          @��
��
=�Tz�?��\A\)C;�H��
=���
?O\)@�33C=n                                    Bxw��4  �          @�33��\)�5?��\A33C:xR��\)�h��?W
=@��HC<L�                                    Bxw���  �          @˅��  �Tz�?O\)@�G�C;�)��  �}p�?(�@�\)C=                                      Bxw�߀  �          @�=q��\)�W
=?+�@�G�C;����\)�xQ�>��@��C<�
                                    Bxw��&  T          @�=q��\)�J=q?#�
@���C;33��\)�h��>�ff@��C<G�                                    Bxw���  T          @�G���p��&ff?��A(�C:���p��\(�?^�R@�\)C;�                                    Bxw�r  �          @ȣ����ͿTz�?h��A��C;�����Ϳ�G�?5@ϮC=O\                                    Bxw�  �          @�Q����E�?+�@�p�C;{���c�
>��H@��C<@                                     Bxw�(�  �          @�G���{�p��?��@���C<����{���>�{@C�
C=�=                                    Bxw�7d  �          @�  ���
��ff?.{@�\)C=����
��
=>�(�@~{C>�H                                    Bxw�F
  �          @������
����?@  @�z�C>n���
��G�>��H@�\)C?��                                    Bxw�T�  �          @ə���{��  ?
=q@�C=(���{���>��R@5�C>�                                    Bxw�cV  �          @����ƸR�J=q?�@�Q�C;:��ƸR�aG�>�{@E�C<
                                    Bxw�q�  �          @����ƸR�333?!G�@�
=C:s3�ƸR�Q�>�@�C;�=                                    Bxw���  �          @�����
=��\?!G�@�  C8�R��
=�#�
?   @��HC9�)                                    Bxw��H  �          @�G���\)�(�?!G�@�ffC9�)��\)�:�H>�@�33C:�R                                    Bxw���  �          @�Q����Ϳ�  ?   @�=qC=G����Ϳ��>�=q@��C>
=                                    Bxw���  T          @ȣ���p��c�
?
=@�C<5���p���  >\@aG�C=0�                                    Bxw��:  T          @��������ff?�@��C=�3������>�\)@%�C>��                                    Bxw���  �          @�=q����
=?�@�
=C>�{�����\>�  @  C?�
                                    Bxw�؆  �          @������(�>�
=@u�C?(������
>\)?��C?�R                                    Bxw��,  �          @�=q��p�����>��@p  C@���p�����=�G�?��C@�
                                    Bxw���  �          @�=q���Ϳ�\)>��H@�ffC@�����Ϳ�Q�>8Q�?�{CA33                                    Bxw�x  �          @�33��
=����?�@�
=C>���
=���>�  @�RC?��                                    Bxw�  �          @�33�Ǯ���>�@�C=���Ǯ��>W
=?�z�C>��                                    Bxw�!�  �          @�=q�Ǯ�h��>Ǯ@c�
C<L��Ǯ�xQ�>8Q�?�z�C<�H                                    Bxw�0j  �          @ʏ\��
=����>�(�@w�C=��
=���>8Q�?��C>^�                                    Bxw�?  �          @�=q��\)�xQ�>�
=@s�
C<�\��\)���
>L��?��
C=n                                    Bxw�M�  �          @��H�ƸR��\)?�\@�(�C>:��ƸR���H>�  @�C>�q                                    Bxw�\\  �          @ə������Q�?�@��\C>�f������\>��@z�C?�                                    Bxw�k  �          @�G������?�@��RC>� ������\>���@.�RC?��                                    Bxw�y�  �          @�
=���H��33?
=@�
=C>�����H��  >��
@?\)C?��                                    Bxw��N  �          @Ǯ��33��?
=@�ffC>����33���
>��
@:�HC?�\                                    Bxw���  �          @�\)�\��  >��H@���C?�H�\��=q>L��?�{C@T{                                    Bxw���  �          @�  ��=q��>��H@�\)CA5���=q��  >#�
?�(�CA�)                                    Bxw��@  �          @�
=��
=�ٙ�>\@b�\CC���
=��      ��CDO\                                    Bxw���  �          @�\)�����33>�(�@~�RCC^�����ٙ�=u?�\CCٚ                                    Bxw�ь  �          @�Q���zῗ
==�?��C>޸��zῗ
=�\)��ffC>ٚ                                    Bxw��2  �          @ȣ������33>L��?�=qC>�
������u�\)C>��                                    Bxw���  �          @Ǯ���Ϳ��>#�
?��RC=�=���Ϳ��ý��
�8Q�C=�H                                    Bxw��~  �          @�
=��=q���>W
=?��HC?�3��=q��ff���
�=p�C@
                                    Bxw�$  �          @�  ��zῑ�>�  @��C>� ��z῔z�#�
��C>�q                                    Bxw��  �          @ȣ��Å���>B�\?�p�C@�\�Å��33����33C@޸                                    Bxw�)p  T          @ə����
��  <#�
=�\)CAǮ���
���H�����?\)CAu�                                    Bxw�8  �          @�\)�\��녽L�;�ffC@�H�\�����Q��S33C@u�                                    Bxw�F�  �          @�  ��\)�k��.{���C6!H��\)�B�\�aG���p�C5��                                    Bxw�Ub  �          @ƸR����    ���
C9E����;�����C9!H                                    Bxw�d  �          @�Q����u���
�:�HC<�{���k���z��,(�C<p�                                    Bxw�r�  �          @�\)���Ϳu�\)��ffC<�
���Ϳfff��33�P  C<T{                                    Bxw��T  �          @ƸR���H���׾�����C>xR���H�����\��{C=�                                    Bxw���  �          @ƸR���H��zᾅ����C>�=���H���ÿ����C=�R                                    Bxw���  �          @�����ÿ�p������C?�����ÿ�녿����RC>�R                                    Bxw��F  T          @����
=���Ǯ�g
=CAff��
=��ff�333�У�C@B�                                    Bxw���  �          @������\)�   ����CE������ٙ��h����
CD8R                                    Bxw�ʒ  �          @�p���z�����(��\)CD�H��z��녿Q���p�CC�
                                    Bxw��8  �          @�����Ϳ�33�z����CC�{���Ϳ�(��n{��CB�                                    Bxw���  �          @�ff��p��ٙ��(�����HCD
=��p���G����
���CBJ=                                    Bxw���  
�          @�{��(����
�333��  CD�\��(����ÿ���!CB��                                    Bxw�*  �          @���p���(��z���z�CD=q��p���ff�u��RCB��                                    Bxw��  �          @�  ��p����������
C6^���p����������C5#�                                    Bxw�"v  �          @�����ff�#�
�.{��Q�C9���ff��׿O\)��{C8W
                                    Bxw�1  �          @�33��G���  �5��{C6=q��G����
�@  ����C4�3                                    Bxw�?�  T          @����  ��\�.{��ffC8�f��  ��{�G���33C7�                                    Bxw�Nh  �          @�G���
=�&ff�#�
��33C9���
=���E��ᙚC8p�                                    Bxw�]  �          @�33��녾�=q�
=���\C6k���녽��!G����C5�                                    Bxw�k�  �          @���ȣ׾\�(������C7xR�ȣ׾aG��:�H����C5�q                                    Bxw�zZ  �          @����G��\)�����RC5@ ��G��#�
�
=q���
C4!H                                    Bxw��   �          @����ȣ׽#�
��(��|(�C4^��ȣ�=u��(��z�HC3u�                                    Bxw���  T          @�  �ƸR��(���G����HC8  �ƸR���
������C6�3                                    Bxw��L  �          @�����=q���Ϳs33��HC4���=q=��s33��\C2�{                                    Bxw���  �          @��Å��\)�fff��C4���Å>\)�c�
���C2��                                    Bxw�Ø  �          @�z���G�>#�
�}p��p�C2���G�>\�n{��
C0aH                                    Bxw��>  �          @�����G�>���\)�(z�C2�=��G�>\����\)C0\)                                    Bxw���  �          @�{�\�������C5&f�\=������C2�{                                    Bxw��  �          @������H>L�ͿB�\��=qC2
���H>�p��0����C0}q                                    Bxw��0  T          @�(����=#�
�p���C3�H���>���h���z�C1�\                                    Bxw��  �          @�z���=q��Q�n{�  C4�{��=q>��n{�33C2�                                     Bxw�|  �          @ƸR��z�=u�c�
�z�C3k���z�>�=q�Y�����C1z�                                    Bxw�*"  �          @�
=��z�>\)�}p���\C2�3��z�>�p��n{�	��C0�{                                    Bxw�8�  �          @�\)��p�>�z�:�H�أ�C1Y���p�>�ff�&ff���RC/�{                                    Bxw�Gn  �          @����\>�z�^�R�C1E�\>��H�G���C/p�                                    Bxw�V  T          @�����
=?(�ÿ(����Q�C-����
=?J=q���H���C,p�                                    Bxw�d�  �          @�\)��(�?u�����C*����(�?����z��/\)C)�=                                    Bxw�s`  �          @��R���>������=qC/�����?z��(����
C.p�                                    Bxw��  �          @�
=��>�����  C/�{��?녾����tz�C.��                                    Bxw���  �          @�
=��p�>��\)���C/c���p�?����(����
C.@                                     Bxw��R  �          @�p����\?Y����R��  C+�q���\?xQ�����xQ�C*�\                                    Bxw���  �          @������>�Q�L����G�C0n���?
=q�0����
=C.�                                    Bxw���  T          @���Q��׿�ff�J=qC8����Q�����\)�V=qC5�                                    Bxw��D  �          @�
=���׾��Ϳ�p��e�C7�3���׽#�
���
�mp�C4Y�                                    Bxw���  �          @�ff��G��8Q쿧��L(�C5�\��G�>\)�����M�C2��                                    Bxw��  �          @�Q���z�>Ǯ�����$��C0=q��z�?!G��s33���C-޸                                    Bxw��6  �          @��R��33>����� ��C/Q���33?5�c�
�
{C-
=                                    Bxw��  �          @�ff��(�>aG��c�
�	�C1�H��(�>�
=�O\)��=qC/�H                                    Bxw��  �          @�(����R>����\�H(�C/:����R?E���\)�/�
C,W
                                    Bxw�#(  �          @�����G�>�  ��\)�/�C1����G�?�\���
�!�C.��                                    Bxw�1�  �          @�\)���>��H��\)�-G�C/.���?@  �xQ����C,�3                                    Bxw�@t  �          @�ff��G�=u��{�S�C3h���G�>Ǯ����J�HC0.                                    Bxw�O  �          @�z������G���(����C8xR����#�
��\��ffC4�                                    Bxw�]�  �          @�(������#�
����G�C:�)�����B�\��z����C5�                                    Bxw�lf  �          @��H���H�L�Ϳ˅�}C6����H>B�\�˅�}�C2
=                                    Bxw�{  �          @��
��ff?��׿333���HC(����ff?�G������|(�C'��                                    Bxw���  �          @��R���?��
�����C)�3���?�녾����8��C(�                                    Bxw��X  �          @�����?�z�h����C!������@z��
=�w�C Y�                                    Bxw���  �          @���  ?�(��E����
C(}q��  ?�\)��G���(�C'!H                                    Bxw���  �          @Å��{?W
=�O\)��p�C+�q��{?�G��z���\)C*^�                                    Bxw��J  T          @��H��
=?��H�#�
���C!\��
=@�
�\)��=qC 0�                                    Bxw���  T          @��H��z�@
�H�G���=qC����z�@�\�aG���C��                                    Bxw��  �          @�G�����@33�z���\)C   ����@Q�L�Ϳ�\CL�                                    Bxw��<  �          @�\)��(�?(�ÿ\)��z�C-�{��(�?G��\�h��C,u�                                    Bxw���  �          @����\)�������(Q�C8����\)�8Q쿓33�4��C5�=                                    Bxw��  �          @�=q��{������'�C9T{��{��  ��33�733C6��                                    Bxw�.  �          @��H���R�\)�}p���C9�)���R���������.=qC6�q                                    Bxw�*�  �          @�=q��\)���}p��z�C58R��\)>��}p��z�C2��                                    Bxw�9z  T          @��\���R�����33�5G�C6�f���R<���
=�:{C3��                                    Bxw�H   �          @�Q����Ϳ��s33�(�C9J=���;�=q��ff�(Q�C6�q                                    Bxw�V�  �          @�Q����;�׿�  ��C8�q���;W
=��=q�-p�C6{                                    Bxw�el  �          @�G���p���(�����%G�C8T{��p��#�
��{�1p�C5�
                                    Bxw�t  T          @����  �Tz῀  ���C<=q��  �\)��
=�8z�C9�                                    Bxw���  T          @�33��p���G���  ���C>���p��:�H��(��@(�C;J=                                    Bxw��^  
�          @�����p���(������'�
C@���p��h�ÿ���R�HC=
                                    Bxw��  �          @��
���
�����  �E��C?  ���
�=p����R�k\)C;�                                    Bxw���  �          @��
��  �#�
�����,��C5����  =������-�C2�{                                    Bxw��P  �          @������H�h�ÿ�=q�,z�C=:����H�(����\�L(�C:33                                    Bxw���  �          @�=q���;aG�����Mp�C6=q����=�G���ff�O�
C2�)                                    Bxw�ڜ  �          @�G���녿O\)��ff�P(�C<@ ��녾�ff���H�j�RC8��                                    Bxw��B  �          @�=q��p�>�p����H�@��C0@ ��p�?+���=q�+33C-J=                                    Bxw���  �          @������
?��\����L(�C)�R���
?���}p��  C&�
                                    Bxw��  �          @����@(��W
=��HC
=��@������RC                                    Bxw�4  �          @��R��=q@�Ϳ�Q��`Q�C�\��=q@\)�Tz�� z�C��                                    Bxw�#�  �          @����z�@����z���{C���z�@/\)�}p���C�3                                    Bxw�2�  �          @��R����@
=��
=�_
=C�
����@)���G�����CJ=                                    Bxw�A&  �          @�{��33@G���p��?�
C  ��33@ �׿(�����Cٚ                                    Bxw�O�  �          @�p����?�(�����P��C   ���@  �G����
Cs3                                    Bxw�^r  �          @����
?�z��\)�}G�C k����
@�׿����&ffC(�                                    Bxw�m  �          @�p���(�@녿���L��CW
��(�@33�=p���ffC�H                                    Bxw�{�  �          @�z����
@ �׿�(��?�Cu����
@�׿(����p�C0�                                    Bxw��d  �          @��H��?�Ϳ��R��  C.(���?�ff��\��Q�C)
=                                    Bxw��
  �          @�33��=q?�ff��=q�S�C)L���=q?�{��G�� Q�C&B�                                    Bxw���  �          @��\��
=?J=q�˅���HC+�{��
=?�����W
=C'��                                    Bxw��V  �          @�33��
=?�ff��R��{C&���
=?�����33C ��                                    Bxw���  �          @�=q��{?��������C-�R��{?�ff�������\C)\                                    Bxw�Ӣ  �          @��
��  ?Tz�Q����C+����  ?�G������\C*�                                    Bxw��H  �          @�33���?u��p��g
=C*xR���?��
���Ϳ�  C)޸                                    Bxw���  �          @���G�?���<#�
=��
C(O\��G�?�33>��
@HQ�C(��                                    Bxw���  �          @����z�=#�
��  ���C3�{��z�=�G��k����C2�R                                    Bxw�:  �          @�(���{���
��Q��d  C4����{>�����z��_33C0��                                    Bxw��  �          @��H��p��zῘQ��=p�C9޸��p���  ����O\)C6}q                                    Bxw�+�  �          @�����׿n{��  �o�C=�)���׿   ��Q����
C98R                                    Bxw�:,  �          @�����  ����\��ffC8����  <#�
�����C3�                                    Bxw�H�  �          @��H���\�������z�\C9h����\��Q��33��(�C4�3                                    Bxw�Wx  �          @�������\��G���z�C7������=��Ϳ�ff����C2�                                    Bxw�f  T          @�����+���  ��\)C:������8Q��{����C5ٚ                                    Bxw�t�  T          @����׿����\)���HC>�{���׿�����z�C9^�                                    Bxw��j  �          @�
=�����G����
��G�CC^�����z�H�����HC>{                                    Bxw��  �          @�ff��  ��p����H��(�CH�H��  ������H��(�CB��                                    Bxw���  T          @��R��
=��{��G�����CD\)��
=����������C?)                                    Bxw��\  �          @�p����ÿ��ÿ��H��
=CG����ÿ�(��Q����
CA�                                    Bxw��  �          @�z�����\)��p����CK}q�����Q��G���  CF&f                                    Bxw�̨  �          @��\�����  ��\��  CL\������Q���
���CF�                                    Bxw��N  �          @�������{�޸R���
CN��녿�33�ff���CH��                                    Bxw���  �          @����z��R�\�B�\��z�CW^���z��G���=q�.�\CU�f                                    Bxw���  �          @����p��:=q�.{���CS���p��'������r=qCQB�                                    Bxw�@  �          @��H��=q��Ϳ@  ��
=CKz���=q��zῨ���[33CH��                                    Bxw��  �          @�z������R��  �$  CN�=����
=�����  CKG�                                    Bxw�$�  �          @�{���\@3�
��(����C�=���\@O\)��\)�.�RC�                                    Bxw�32  �          @�=q���@qG������
=C
\)���@��R���\�
=C33                                    Bxw�A�  �          @���~�R@{�����{CT{�~�R@��
�u�\)C\)                                    Bxw�P~  �          @�\)�\)@|�Ϳ��R��  CE�\)@��H�Q���{C�=                                    Bxw�_$  T          @�Q����@aG��Q����HCE���@��������J�RC\)                                    Bxw�m�  �          @�33��G�@Vff�$z��иRCY���G�@{�����u�C��                                    Bxw�|p  
�          @�G����@A��4z����HCJ=���@l(���\)����C	��                                    Bxw��  �          @�����\)@=q�E��33CJ=��\)@J�H��\��33C�                                    Bxw���  �          @�����?�  �dz��{C#����@(��E��C�                                    Bxw��b  �          @������H?�z���H�Ə\C'8R���H?��
�   ���\C ��                                    Bxw��  �          @����33?O\)�%���C*�\��33?�  ������
C"ٚ                                    Bxw�Ů  �          @�������@S�
�����{CǮ����@k��L������C�
                                    Bxw��T  �          @����\)@c�
�����G�C���\)@|(��Q���C��                                    Bxw���  �          @�\)�[�@�{�
=����B�B��[�@�?.{@��B�ff                                    Bxw��  �          @���h��@��\��Q�����C� �h��@�z������C��                                    Bxw� F  �          @�=q�'
=@���=q�1�B����'
=@���?���A:�HB�
=                                    Bxw��  �          @��
�{@��>8Q�?�(�B����{@�G�?�A�z�B�G�                                    Bxw��  �          @���3�
@���   ��{B��R�3�
@�G�?uA�B�\)                                    Bxw�,8  �          @��H�W
=@�����
�!B��{�W
=@�Q�>�{@VffB�p�                                    Bxw�:�  �          @�=q�|(�@|(�������  C���|(�@�\)�������C�                                    Bxw�I�  �          @�=q�`  @�{�����W�C L��`  @�z�����
B��                                    Bxw�X*  �          @��H�i��@��ÿ�ff�w33Cp��i��@�����=q�'�C �q                                    Bxw�f�  �          @��H�k�@�z��G���
=C�)�k�@�
=�   ��ffCp�                                    Bxw�uv  �          @���z=q@vff��\��33Cs3�z=q@��׿Y���CxR                                    Bxw��  �          @�(��~�R@l�������{C	&f�~�R@�p�����#�
C��                                    Bxw���  �          @���
=@S�
� ����Q�C���
=@x�ÿ����bffC	\)                                    Bxw��h  �          @����B�\@�33��z�����B�=q�B�\@�{��(����\B�q                                    Bxw��  �          @����b�\@���!����
CǮ�b�\@����p��=�B�u�                                    Bxw���  �          @�\)�n�R@u��,(��ՅC:��n�R@�{���H�bffC�q                                    Bxw��Z  �          @�
=�u�@e�5���=qC�
�u�@�Q��z���G�C��                                    Bxw��   �          @�  ���
@0  �C33����C�{���
@aG������C(�                                    Bxw��  �          @�����
=@Q��P���(�C���
=@N�R��H���C!H                                    Bxw��L  �          @���  @*�H�E���(�C����  @]p��
=q��Q�C޸                                    Bxw��  �          @�����  @Z=q�   ��
=CL���  @\)����V�\C�{                                    Bxw��  �          @�������@C�
�3�
���Cn����@p  ������
C
                                    Bxw�%>  �          @��R��z�@33�R�\�\)C\)��z�@J�H�{��=qC5�                                    Bxw�3�  �          @�z����@	���c�
�{CY����@G
=�1G���{C��                                    Bxw�B�  �          @��R��33@0  �U����C)��33@g�����p�C
��                                    Bxw�Q0  �          @��y��@6ff�QG��G�C�{�y��@l(������
C�{                                    Bxw�_�  �          @�p��g�@P  �U�	�C
��g�@�33�{���HC}q                                    Bxw�n|  �          @�Q��u@+��(Q�����C#��u@U��(���G�C                                    Bxw�}"  �          @�z���G�>�p������33C/���G�?����R��G�C'�{                                    Bxw���  �          @�(�����?Y���G
=��\C)k�����?޸R�.{��
=C
                                    Bxw��n  �          @�33����?�\�6ff���C-�����?���$z�����C$Y�                                    Bxw��  �          @�33��=q����1���
=C9����=q>Ǯ�333��=qC/h�                                    Bxw���  �          @�
=�I��@QG��S33�
=C�f�I��@���
�H���B��H                                    Bxw��`  �          @��׿��@����:�H��
=Bή���@��R����X��B˔{                                    Bxw��  �          @����0��@�=q�HQ���B�8R�0��@��ÿ��
�uG�B�#�                                    Bxw��  �          @��ÿ�{@����u��'
=B��H��{@�������33B�k�                                    Bxw��R  �          @����(�@vff�tz��(�B�33��(�@��\��R��=qB�G�                                    Bxw� �  �          @�����=q@��\�N{�
=B�Q쿪=q@��H��
=���B��                                    Bxw��  �          @�����@���S�
��HB����@�����\)����B�\                                    Bxw�D              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw�,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw�;�   ]          @�z��W
=@`  �U�
Q�C޸�W
=@�33�
=��=qB�\)                                    Bxw�J6  �          @��\�p  @!��j�H�(�C��p  @a��.{��ffC�R                                    Bxw�X�  �          @����^{@  ��G��133C
=�^{@XQ��I���{C�                                     Bxw�g�  �          @�
=�U�@33��p��;p�Ck��U�@N�R�U��C�)                                    Bxw�v(  �          @���K�?�33�����DffC33�K�@G��^�R���C��                                    Bxw���  �          @����!G�?�ff�����kffCs3�!G�@���~{�C=qC��                                    Bxw��t  �          @�zῙ����ff���\CHh�����?���G�(�C��                                    Bxw��  �          @�=q����{��ff�~�RC]J=��    ��ff{C3��                                    Bxw���  �          @��H�Ǯ@�������|B�33�Ǯ@vff�y���4�
B���                                    Bxw��f  �          @�zῡG�@P  ���O{B�p���G�@�
=�L(��

=Bӊ=                                    Bxw��  �          @�{��
=@^�R��z��;�
B�p���
=@�33�5��33B�\                                    Bxw�ܲ  �          @�(��p�@C�
��=q�<z�B����p�@��9����z�B��)                                    Bxw��X  �          @�p��<��@N{�`���33Cn�<��@���z����B��{                                    Bxw���  �          @����=q?�(��>�R� �RC%
��=q@�(���\)Ch�                                    Bxw��  �          @��z=q@��Z=q�ffC���z=q@I���#33���C(�                                    Bxw�J  �          @�ff����?�=q�O\)�=qC�)����@0���   �иRC5�                                    Bxw�%�  �          @�{��G�?��
�?\)���C!ff��G�@���ff��33C.                                    Bxw�4�  �          @�33�o\)?�
=�u�1ffC"u��o\)@�QG����C�                                    Bxw�C<  �          @�z��
=@P  �{��1Q�B��
=@��\�,(����HB�B�                                    Bxw�Q�  �          @���R�\?������;�C8R�R�\@Dz��P  �G�C��                                    Bxw�`�  �          @��\�;�?��
����TCaH�;�@7
=�l���(\)C��                                    Bxw�o.  �          @�z��!�?�Q���{�c��C8R�!�@Fff�y���1  C5�                                    Bxw�}�  �          @����>�R?s33�����]�C"W
�>�R@ff�~�R�9Q�C�\                                    Bxw��z  �          @����#33��\��(��w��C?33�#33?�������oz�C�3                                    Bxw��   �          @��
�7�@����
=�D��C�q�7�@`  �QG���\CT{                                    Bxw���  �          @��\�]p�?��H��=q�=(�C5��]p�@,(��XQ����C+�                                    Bxw��l  �          @�=q�QG�=L�����R�W=qC3��QG�?��R���R�F(�C�                                     Bxw��  �          @�G��U��\)����Q��C8�{�U?����
=�H�C!�                                    Bxw�ո  �          @����j�H�������=�C<.�j�H?O\)�\)�:��C'�
                                    Bxw��^  [          @�\)��33����_\)� z�C<J=��33?��_\)� �RC,
                                    Bxw��            @�{�U��Ǯ�QG��&��CM
=�U�����fff�<  C:��                                    Bxw��  �          @��
�3�
?�{�y���D�Cff�3�
@A��E���C                                    Bxw�P  �          @�\)� ������vff�B��C]�f� �׿^�R��(��jCG(�                                    Bxw��  �          @�  �k����R�Z=q�%(�CF�f�k��u�hQ��2ffC4�H                                    Bxw�-�  �          @�zῂ�\��{��ǮCm�����\<��
��ffC2��                                    Bxw�<B  �          @�(���׿���{ǮCC޸���?�����C�                                    Bxw�J�  �          @�
=�q�?�Q���
���HCٚ�q�@
=��\)��C                                    Bxw�Y�  �          @��\�g
=@W��5�{C���g
=@Z=q>�ff@�z�C��                                    Bxw�h4  �          @�
=�z�H@`�׽#�
��(�C
��z�H@U�?���AE��C�f                                    Bxw�v�  �          @�\)��=q@`�׿!G���z�C5���=q@aG�?�@��HC#�                                    Bxw���  �          @�z��Y��@~{?�G�A/33C�\�Y��@Z=q@G�A�(�C��                                    Bxw��&  �          @��R�E@���?���A`(�B��
�E@e@*=qA�RC�                                    Bxw���  �          @�������@aG�?�  AR=qC
� ����@9��@ffA���C!H                                    Bxw��r  �          @�=q��p�@Q�@%A�{C=q��p�?�33@I��B
Q�C%u�                                    Bxw��  �          @�33���?�@O\)B=qC-�����z�@N�RB�
C;�                                     Bxw�ξ  �          @�����H@   >���@p��Cff���H?��
?z�HA4��C��                                    Bxw��d  �          @��
�o\)@��׿����EC�3�o\)@�{>aG�@{C�                                    Bxw��
  �          @��
���?�Q쿣�
�hQ�C����@�׿����
=C�H                                    Bxw���  �          @�ff��(��L�Ϳ��
�]��C<����(����
��p���=qC7��                                    Bxw�	V  �          @�G���z�?h�ÿJ=q��C*p���z�?�\)��G���G�C(@                                     Bxw��  �          @������?n{��  �&�RC*.���?��H�!G���  C'B�                                    Bxw�&�  �          @�����R@�ÿE����RCL����R@ ��=�Q�?c�
C0�                                    Bxw�5H  �          @�����\)@1G������C����\)@2�\>�G�@�33Cu�                                    Bxw�C�  �          @�  ��z�@5�&ff��G�C����z�@8��>�Q�@qG�C
                                    Bxw�R�  �          @�  ��\)@@  �c�
�=qC.��\)@G�>B�\?��HC�                                    Bxw�a:  �          @����G�@,(��G����C���G�@333>B�\?�
=C��                                    Bxw�o�  �          @�=q���@>{�B�\��{C5����@C33>���@G�C}q                                    Bxw�~�  �          @����z�@7
=�z�H�#�CY���z�@AG�=u?�RC��                                    Bxw��,  �          @��\����@333�aG��z�C������@-p�?B�\@��\Cs3                                    Bxw���  �          @�z����@,(�?
=@�Q�CE���@z�?�p�AtQ�C�                                    Bxw��x  �          @�z�����@w�?   @�33C\)����@\��?�A��C��                                    Bxw��  �          @��
���@Vff>B�\?�Q�CG����@E�?��A]�C�{                                    Bxw���  �          @�{���@q녾����A�C
T{���@i��?��
A&�HCT{                                    Bxw��j  �          @�p����\@aG��B�\����C޸���\@dz�?�\@�C��                                    Bxw��  �          @�p����\@E��(��F�\C����\@Tz�u�!G�C{                                    Bxw��  �          @������@+��^�R��RC����@4z�=�?�(�C��                                    Bxw�\  �          @��\���\?��þ��
�S33C#�����\?�=q>�  @#�
C#}q                                    Bxw�  �          @�  ��ff�������g
=C5����ff>��ÿ���`��C0T{                                    Bxw��  �          @�
=���Tz��
=���C=�)���#�
�G���C4u�                                    Bxw�.N  �          @�Q���  �5�
=q��G�C<!H��  =�Q��G��Ù�C2�                                    Bxw�<�  �          @�\)���׿��Ϳ��R��CC����׿=p���\)���C<^�                                    Bxw�K�  �          @����ff����\)�Ə\CK���ff�z�H�0�����C@h�                                    Bxw�Z@  �          @��
���Ϳ�Q��,(���CM)���Ϳ\(��Mp��z�C?                                    Bxw�h�  �          @�(���ff��G��'
=��(�CF����ff���>{�=qC:.                                    Bxw�w�  �          @�z�������<����CE�=���W
=�O\)��RC6�\                                    Bxw��2  �          @�z���=q��p��?\)�33CK
=��=q����Z=q�C;��                                    Bxw���  �          @�Q��dz��3�
�%��33CZ#��dz�ٙ��Z�H�#G�CMk�                                    Bxw��~  �          @�\)�`  ��������RCPc��`  �fff�<(��z�CBn                                    Bxw��$  �          @�����\)?�
=�{��{C&���\)?�녿�33���
C0�                                    Bxw���  �          @�(���G�?�Ϳ
=q���HC.)��G�?333���R�VffC,h�                                    Bxw��p  �          @�G��w
=�B�\�Z=q�#��C?��w
=?��\���&33C,Y�                                    Bxw��  �          @�{��G�>8Q��=q����C1�q��G�?Q녿�����RC*��                                    Bxw��  �          @�p���(�����\���\C9k���(�>����z���\)C0n                                    Bxw��b  [          @�G���Q�=#�
��33�E��C3�\��Q�>�ff����5p�C/�                                    Bxw�
  �          @��R��{���ÿp����
CBB���{�c�
��{�g\)C=�3                                    Bxw��  �          @��R�E��@  �=q��z�C`:��E����U�*p�CS��                                    Bxw�'T  �          @��\�^�R�P  ��  ����C_��^�R����3�
�p�CV�                                    Bxw�5�  �          @����j=q�:=q��(��i�CZ}q�j=q�G��(����CSǮ                                    Bxw�D�  �          @��H��p����R��\)��ffCMu���p����{��\)CC��                                    Bxw�SF  �          @�z����H�  ��33���CP޸���H��  �
=��CH
                                    Bxw�a�  �          @�{��(�����=q��\)CA^���(����
�(Q���p�C5)                                    Bxw�p�  �          @��R@�\@X���&ff��B_��@�\@�����_�
Bs�\                                    Bxw�8  �          @��?���@o\)�/\)�	�
B�u�?���@�����_�B�                                    Bxw���  �          @��H@�\@��\�'���\B}�
@�\@�Q�n{�"ffB���                                    Bxw���  �          @��R?�
=@dz��L(���
Bv=q?�
=@����33��33B�z�                                    Bxw��*  �          @���?�=q@P  �xQ��?�B�G�?�=q@�ff������HB���                                    Bxw���  �          @���?��@Fff���\�L�B�.?��@�z��(Q����B�Ǯ                                    Bxw��v  �          @�Q�@Q�?�\)���H�cffB$�R@Q�@U�U� =qBe�H                                    Bxw��  �          @���?�(�@���W33BGG�?�(�@n{�@����BxQ�                                    Bxw���  �          @�녿�(�@��hQ��G�CB���(�@`���   ����B�{                                    Bxw��h  �          @�33�a�?Ǯ�hQ��-  C.�a�@333�1���C��                                   Bxw�  w          @��\?(��@U���33�I��B��=?(��@��
�#�
���B��R                                    Bxw��  �          @��
�L��@3�
��{�b�BӸR�L��@����C�
��
B�.                                    Bxw� Z  �          @��_\)@�H�L�����CO\�_\)@\(���
��(�Ck�                                    Bxw�/   "          @�p��O\)@(Q��U��=qC��O\)@l(�����RCJ=                                    Bxw�=�  �          @��R�e�?�p��`  �"33C�R�e�@J=q�\)����C
��                                    Bxw�LL  �          @��
��
=�����{���\CQ�f��
=��\)�����\)CH�R                                    Bxw�Z�  �          @�����R�333�����CU�H���R�����<(����CKk�                                    Bxw�i�  �          @�������\�7���(�CN&f����L���[����C>�                                    Bxw�x>  �          @���u���\)�XQ��\)CJ�)�u��W
=�o\)�1  C7.                                    Bxw���  �          @��\?�?W
=��ff�{A��R?�@{�r�\�J�BPp�                                    Bxw���  �          @�ff���R��(��a��\�CX�
���R���
�u��zffC4xR                                    Bxw��0  �          @�(�?�Ϳ����\)�C��?��?\)��z�¡�3B7z�                                    Bxw���  �          @�=q���
��p������3C��ὣ�
������{±
=Cg�R                                    Bxw��|  �          @���!G���=q���¤�CK)�!G�?�p����
��B���                                    Bxw��"  �          @�ff��z�>�=q��(���C&���z�@������t�HB�.                                    Bxw���  �          @�ff�u?�{��{33B�(��u@:�H��(��Z�RB�G�                                    Bxw��n  �          @��У�?��������C��У�@@���r�\�?B�Ǯ                                    Bxw��  �          @�z�5@z�����k�Bٽq�5@j=q�XQ��(�HB��                                    Bxw�
�  �          @��>��?�����\��B�G�>��@Z�H�a��6��B�.                                    Bxw�`  �          @��?��Ǯ�����Q�C�&f?�>aG���=q
=A�\                                    Bxw�(  �          @�33?�������o\)�L��C��3?��ÿB�\��(��{C���                                    Bxw�6�  �          @�?�(���������_G�C�O\?�(����R�����3C���                                    Bxw�ER  �          @��@녿����Q��i�C�8R@�>����p��y(�A9��                                    Bxw�S�  �          @�{@�\���
�o\)�[��C�h�@�\>�\)�|(��nz�@�z�                                    Bxw�b�  �          @��\@Q�?�G��P���(=qA�\)@Q�@*=q�=q��B�                                    Bxw�qD  �          @��
@G�@   �E��RB�\@G�@`�׿�\)��p�BA33                                    Bxw��  �          @�(�@$z�@$z��
=q���B4
=@$z�@Mp��z�H�UG�BM=q                                    Bxw���  �          @��R@Q�@��H����IBo  @Q�@�?!G�@�
=Bq{                                    Bxw��6  �          @��H��Q���
�~{�`Q�Cf�H��Q쾮{���R�3C?B�                                    Bxw���  �          @�ff�  �c33�1G����Cm�{�  �
�H�|(��NQ�C_�                                    Bxw���  �          @�  ���i���A���Cp�3���
=q��
=�[�HCb�=                                    Bxw��(  �          @�{�\)�u�XQ��${C��)�\)�(����
��C��                                    Bxw���  �          @�ff�k��{��Fff�(�C�n�k��Q������o��Cx�                                    Bxw��t  �          @���
=q>k���33C.�
=q@�
���R�Zz�CQ�                                    Bxw��  �          @�p��ff?@  ��=q  C c��ff@   �y���H(�C�                                    Bxw��  �          @�{�Y��?J=q��u�C	{�Y��@,(����R�`�RB�(�                                    Bxw�f  �          @��
��?�33�^{�DQ�C�
��@G������B�u�                                    Bxw�!  �          @�=q�>�R@N{>#�
@z�C�R�>�R@8��?�
=A�Q�C�{                                    Bxw�/�  �          @�  � ��@z�@�G�B_�
C�)� ��>��R@��HB�� C+33                                    Bxw�>X  �          @�Q��#�
@G
=@n�RB+p�Cz��#�
?���@�p�Bh��C+�                                    Bxw�L�  �          @�Q��@��@R�\@Mp�B�HCxR�@��?�p�@�Q�BK33C(�                                    Bxw�[�  T          @�
=�
=@L(�@j�HB+(�B�(��
=?�Q�@�z�Bl�RC�H                                    Bxw�jJ  T          @�����@Z�H@fffB#p�B�k���?�
=@�p�Bhp�C��                                    Bxw�x�  �          @����   @y��@U�B�RB�G��   @�R@��
Bd
=C�H                                    Bxw���  �          @��\���@w
=@Y��Bp�B��f���@
=q@�p�Bc��C�                                    Bxw��<  �          @���8Q�@z�H@0  A��B��\�8Q�@{@�33B<�HCQ�                                    Bxw���  �          @���e@w�?��A5�C�=�e@G
=@"�\A��C�                                    Bxw���  �          @��\��@��>�  @(Q�C���\��@l��?�z�A��C�                                    Bxw��.  �          @�{�`��@��\��33�l��C�3�`��@w�?��Af{C5�                                    Bxw���  �          @�ff�W
=@�  ������C L��W
=@�G�?�\)Aip�C�3                                    Bxw��z  �          @�(��:=q@�(���p����B�p��:=q@�
==���?�ffB��                                    Bxw��   �          @���@��@�=q�����H��B�
=�@��@�p�?!G�@��B���                                    Bxw���  �          @�33�3�
@��H���R�W�B����3�
@�
=?#�
@޸RB�L�                                    Bxw�l  �          @�  ��\@�
=����x  B�=q��\@���?\)@�p�B�p�                                    Bxw�  �          @�zῦff@�  �fff�%p�Bҏ\��ff@�ff?���AP��B��)                                    Bxw�(�  
�          @��.{@e�J�H�%�RB�k��.{@�=q�����B�B�                                    Bxw�7^  �          @�=q?z�H@k��5��ffB�G�?z�H@��׿���O33B���                                    Bxw�F  �          @��R?��
@I���c�
�933B���?��
@��H��Q���{B�#�                                    Bxw�T�  �          @���@p�@g��*�H� Bj�@p�@��Ϳp���,��B}{                                    Bxw�cP  �          @��@,��@������_�
Bk��@,��@�(�?+�@�33Bo                                      Bxw�q�  �          @��H@#33@�33�Q���{Bh��@#33@��
�u�$z�Bt\)                                    Bxw���  �          @�\)@:=q@u��.{�z�BS33@:=q@c�
?�
=A�Q�BJ��                                    Bxw��B  �          @��@1�@z=q�Ǯ��  BZ�@1�@�ff>W
=@�Ba�H                                    Bxw���  �          @�z�@H��@`�׿������B@@H��@�  ���R�_\)BO�                                    Bxw���  �          @���@,��@.{�A��\)B5{@,��@o\)��\)����BX                                    Bxw��4  �          @�
=@��?�=q����d�HA�ff@��@.{�U��)(�B?�
                                    Bxw���  �          @�z�@p�@'
=�+��=qBF�
@p�@_\)�����  Bf�                                    Bxw�؀  �          @��R���
@G
=�l(��G\)B��ͽ��
@��
�33���HB�                                      Bxw��&  
�          @�\)��\)@)���|���S�B��)��\)@���{��  B�ff                                    Bxw���  �          @�ff�#�
@n{�3�
��\B�.�#�
@����}p��D��B�(�                                    Bxw�r  �          @�  ?8Q�@�\)�����\B��R?8Q�@�33�Ǯ��Q�B��                                    Bxw�  �          @���?�@s33�1����B��)?�@��
�k��/
=B���                                    Bxw�!�  �          @�=q��R@5����Y��B̽q��R@�G��\)��
=Bą                                    Bxw�0d  �          @�33����?�G���=q�m��CJ=����@Z�H�J�H��B�                                    Bxw�?
  �          @�\)�!G�?G���p��l�C"�q�!G�@%��j�H�6
=C^�                                    Bxw�M�  �          @��H�9����\�w
=�=�CW+��9���W
=����ap�C8.                                    Bxw�\V  �          @�����n{�7
=��HCnu����	������S��C_\)                                    Bxw�j�  �          @�(��,(��E�8Q��ffCd�R�,(������z=q�N
=CRW
                                    Bxw�y�  �          @�z��>{��\)�j�H�=(�CP���>{=�Q���Q��U�
C2@                                     Bxw��H  T          @�  �H�ÿz�H�vff�E�CE\)�H��?@  �y���I�\C&�                                    Bxw���  �          @�=q�3�
��R�L(��!��C]k��3�
�aG��|(��T��CEk�                                    Bxw���  �          @�33�g����H�Fff�  CJ�g�    �[��-��C3�3                                    Bxw��:  �          @�33�l�Ϳ�ff�J=q�z�CGG��l��>8Q��Z=q�*�C1.                                    Bxw���  �          @�(��6ff>�p��`  �JG�C,�{�6ff?�=q�@  �&
=C@                                     Bxw�ц  �          @��R�ff@J=q�aG��+�\B�33�ff@��������G�B�\)                                    Bxw��,  �          @�{�<��@33�c�
�.Q�C\�<��@g
=��R��Q�CG�                                    Bxw���  �          @�녿���@(���e��E��B�aH����@z�H�ff��33B߅                                    Bxw��x  �          @��Ϳ��@,(����R�]�
B�\���@���(������B�                                    Bxw�  �          @��
�H@ ���b�\�;�C� �
�H@s33�
=��=qB�ff                                    Bxw��  �          @�{����@hQ��%���B������@�z�B�\���B�=q                                    Bxw�)j  �          @�G���{@�p���p��z�HB�aH��{@�=q?L��AffB�8R                                    Bxw�8  �          @�Q쿬��@�
=��ff�0��B�\����@��?��A]�B�aH                                    Bxw�F�  T          @��\��G�@�����H�S\)B�#׿�G�@�ff?��A6�\B��f                                    Bxw�U\  �          @�ff��G�@�33���
�^�RB�Q��G�@�33@  A�
=B��\                                    Bxw�d  �          @�\)?�\)@�Q������B�?�\)@��R?�A�\)B��                                    Bxw�r�  �          @�Q�>.{@�z��\���B�Q�>.{@�=q?��A��B�{                                    Bxw��N  �          @�  ?��
@��    �#�
B��?��
@���@z�AȸRB�                                    Bxw���  �          @���?c�
@����(���G�B��?c�
@�Q�?���A�  B���                                    Bxw���  �          @�\)?\(�@����}p��&ffB���?\(�@�ff?�
=As�B�=q                                    Bxw��@  �          @��;��@�녿z���G�B�#׾��@���?�ffA��B�Q�                                    Bxw���  �          @��
��@�ff?5A�
BոR��@z=q@,��B�HB�                                    Bxw�ʌ  �          @�z��K�@W
=@,(�A��CxR�K�?�@x��B:z�C                                    Bxw��2  �          @��H��(�@o\)@H��BG�B���(�?�p�@��RBg�RC�3                                    Bxw���  �          @�(����H@�G�@��A�\)B׽q���H@AG�@���BL(�B�R                                    Bxw��~  �          @�z��2�\@z=q@Aϙ�B�\�2�\@�R@u�B6�RC
h�                                    Bxw�$  �          @��\�H��@P��@)��A�\C���H��?�  @s�
B:�C�{                                    Bxw��  �          @�(��g
=@vff?�@�p�C+��g
=@N{@
�HA�p�C
B�                                    Bxw�"p  �          @�  ��@Dz�z�H�%G�CE��@L(�>�@�Q�CB�                                    Bxw�1  �          @��
�vff@w���ff�U�C޸�vff@�G�?��@�z�C�)                                    Bxw�?�  �          @����\��@w
=��{���C�=�\��@���=L��?�\C ٚ                                    Bxw�Nb  �          @���{@G��R�\�(�C Q��{@�Q�Ǯ��Q�B�33                                    Bxw�]  �          @�(��8Q�u��G�¤u�C8��8Q�@33��Q��|�RB�                                    Bxw�k�  �          @�녿���u���H(�C]O\���?�z������B�p�                                    Bxw�zT  �          @��\���;\)����«Q�CH{����@ff����aHB�(�                                    Bxw���  �          @�33>���    ����«C��
>���@�R��
=�|�
B���                                    Bxw���  �          @�>�{�k�����C�*=>�{?�z���z�B�                                    Bxw��F  �          @���>��
��R���\¥�qC��R>��
?�������  B�\)                                    Bxw���  �          @��>.{>8Q����H®�B5�
>.{@*=q����v  B��                                    Bxw�Ò  T          @��
�8Q�>aG�����®{C �=�8Q�@,(���33�s�
B���                                    Bxw��8  �          @�=q=L�Ϳ�����¨p�C��3=L��?�(�����{B��                                    Bxw���  �          @��\�0��=����¥.C)���0��@!���\)�t=qB�ff                                    Bxw��  �          @�Q쿀  ?Q������C�=��  @HQ����
�Tp�B�aH                                    Bxw��*  �          @��
��Q�?\(������{C�R��Q�@>�R�\)�E{B�33                                    Bxw��  �          @�=q��ff?�{����Q�C ^���ff?�33>�Q�@p  C�R                                    Bxw�v  �          @��H���?��Y���33C � ���@   =u?��C��                                    Bxw�*  �          @����
=?޸R��Q���Q�C ����
=@z�8Q���
=C                                    Bxw�8�  �          @�33��{@p��0����z�C�=��{@   ?   @�z�C)                                    Bxw�Gh  �          @�Q�����@�H���H�L��C�����@-p�<��
>B�\Cff                                    Bxw�V  �          @�Q�����@
�H�#�
��
=C�f����@{>��@���C)                                    Bxw�d�  �          @�����{@z�5��p�C����{@��>�
=@��C+�                                    Bxw�sZ  �          @�33�xQ�@�ÿ��
���HCL��xQ�@<(��   ���C��                                    Bxw��   T          @����Q�@�H����_��B�=��Q�@����&ff��33B�(�                                    Bxw���  �          @�{?��@��{�qBb�
?��@��H�U��ffB���                                    Bxw��L  �          @�
=?���@ ����  �uG�B=�\?���@�=q�aG����B�#�                                    Bxw���  �          @�(�?�=q?�{��z��=B6�?�=q@j�H�g
=�(�\B�p�                                    Bxw���  �          @�Q�?��?�p�����k�B��?��@u�H�����B��3                                    Bxw��>  �          @��
?\(�@@  �\)�O�RB�33?\(�@�  �
�H��z�B�u�                                    Bxw���  �          @���?n{@333��G��i�\B�33?n{@��>{��
=B��                                    Bxw��  �          @�z�?��@?\)����M�Blp�?��@���   �ԸRB�                                      Bxw��0  �          @�(�@.�R?��������]33A���@.�R@_\)�Tz��\)BO��                                    Bxw��  �          @�33@K�?L������W33Ab�H@K�@1��g
=�"
=B$�\                                    Bxw�|  �          @��?�ff?���(��v�\B:�?�ff@|���\(����B��f                                    Bxw�#"  �          @���
=@^{�vff�9��B٨���
=@�33�������
B�aH                                    Bxw�1�  �          @�
=��@R�\�xQ��2�
B����@��R�����HB�=                                    Bxw�@n  �          @�z���\@r�\�9����B�Q���\@��R�Q��ffB���                                    Bxw�O  �          @����	��@�z��QG���B����	��@�{�z�H�ffB�                                      Bxw�]�  �          @�p����
@��R�HQ��(�B�𤿣�
@���5���BΏ\                                    Bxw�l`  �          @�=q�#�
@33���R�B��)�#�
@��H�Tz���RB�.                                    Bxw�{  �          @�(�>Ǯ?˅����� B��{>Ǯ@w��|(��5�B��                                    Bxw���  �          @��
>�G�?�  ��G�B�Ǯ>�G�@Z=q���H�Nz�B�ff                                    Bxw��R  �          @��ÿTz�?�����  B➸�Tz�@�p��s33�'=qB�p�                                    Bxw���  �          @��H����@(Q����
�b��B��{����@�33�E���B�L�                                    Bxw���  �          @����@=p����\�L��B���@�  �*�H��=qB�G�                                    Bxw��D  �          @��#33@<(����B�C��#33@�p��#33�̏\B�=q                                    Bxw���  �          @���@3�
��  �VQ�B�B��@�ff�8Q����B��f                                    Bxw��  �          @��@,(����\�[33B���@�z��@������B�                                     Bxw��6  �          @���{@Fff��
=�;33C ���{@��R�G���33B�Q�                                    Bxw���  T          @�Q쿪=q@ff��(�G�B��ÿ�=q@�Q��c33��HB���                                    Bxw��  �          @���Ǯ@w
=�����0�B�\�Ǯ@����޸R��  B��)                                    Bxw�(  �          @�{�-p�@R�\�����-33Cz��-p�@�G��   ����B�                                      Bxw�*�  �          @�
=�(�@o\)���R��  B�=q�(�@�\)<#�
=�\)B��H                                    Bxw�9t  �          @��33@�\)?޸RA�{B޸R�33@r�\@�  B+�\B�3                                    Bxw�H  �          @��\���@��\?�  AJffB�#׾��@��\@o\)B#{B�\                                    Bxw�V�  �          @��=u@���<#�
=uB��
=u@�\)@*�HA�B��                                    Bxw�ef  �          @����I��@��?z�HA�B�Q��I��@mp�@FffB�CaH                                    Bxw�t  �          @�
=��p�@����  ��33B���p�@�Q�?333@�z�B�
=                                    Bxw�  �          @����(Q�@�녿�33���
B�W
�(Q�@���?Q�Az�B�W
                                    BxwX  �          @���>{@�����
�x��B��=�>{@�G�?W
=A�
B��                                    Bxw�  �          @���K�@�p��
=��{B��R�K�@�G����Ϳ��B�.                                    Bxw®�  �          @�ff�G
=@xQ��1����C �=�G
=@�\)�(�����B��3                                    Bxw½J  �          @����@  @j=q�?\)�(�Cc��@  @�z�c�
��HB��
                                    Bxw���  �          @���   @g��`���Q�B�G��   @�33��{�`��B�                                    Bxw�ږ  �          @�
=��Q�@l(��!G���HB���Q�@�{������B�(�                                    Bxw��<  �          @��?\)@�(��
=��\)B��?\)@�\)<#�
=�Q�B�=q                                    Bxw���  �          @���z�@�p��  ����B�3�z�@���>�p�@j�HB�
=                                    Bxw��  �          @����)��@��ÿ�����p�B���)��@�=q?.{@�G�B�#�                                    Bxw�.  �          @����.�R@��ÿ�Q����RB�p��.�R@�Q�?L��@��RB�(�                                    Bxw�#�  �          @��
�Fff@�
=�˅�|z�B��\�Fff@��?\(�A=qB�                                    Bxw�2z  �          @���6ff@��׿�{��Q�B��6ff@��\?&ff@��HB�\                                    Bxw�A   �          @�Q��7�@��\��������B�(��7�@�ff>�@�G�B�(�                                    Bxw�O�  �          @�p��U�@�{������Q�C � �U�@��>Ǯ@{�B�G�                                    Bxw�^l  T          @��q�@�Q쿥��RffCE�q�@�z�?L��Ap�Cc�                                    Bxw�m  �          @�Q��J=q@tz��Q�����C���J=q@��>�33@s33B�u�                                    Bxw�{�  �          @�{�w�@L������Cff�w�@s�
�����H��Ch�                                    BxwÊ^  �          @��
����?�p�������C#�����@�Ϳ����YG�C��                                    BxwÙ  �          @�ff����@.{���H��  C33����@Tz�����(�C�f                                    Bxwç�  �          @��H��G�@"�\��Q���C�q��G�@J=q����p�C�                                    BxwöP  �          @�
=��=q?�\)�˅���C%�{��=q?��H�:�H��C��                                    Bxw���  �          @�Q���
=?�G���
=�f{C)� ��
=?�=q�B�\��33C#�                                    Bxw�Ӝ  �          @��
���H@���0����{Ch����H@P�׿�{�X��C��                                    Bxw��B  �          @�
=�`  @H���^�R�33C
��`  @����
�r�HC G�                                    Bxw���  �          @�녿��@hQ�����K�B�
=���@��z����Bɽq                                    Bxw���  T          @����@��R��ff�9Q�B�����@��H������=qB��
                                    Bxw�4  T          @��H>��
@����W��p�B�Ǯ>��
@������H���B���                                    Bxw��  S          @���?0��@�(��A���z�B��?0��@��R�#�
��  B�Ǯ                                    Bxw�+�  �          @�p�?��@��H��\)�-ffB��
?��@�33�˅�n�HB��f                                    Bxw�:&  �          @��
?�=q@�����\)�/=qB���?�=q@�G���\)�u�B��H                                    Bxw�H�  �          @�p�?z�@7
=�����o(�B�33?z�@�{�?\)��{B���                                    Bxw�Wr  �          @�ff?c�
@0  ���R�s�
B���?c�
@��Mp��z�B�z�                                    Bxw�f  �          @��>���@ ����  �{B���>���@�  �W��
=B��                                    Bxw�t�  �          @�����?}p����R
=B�\���@i����  �JffBƽq                                    Bxwăd  �          @��
�Ǯ=��
��=q¬8RC'�;Ǯ@:=q��G��o
=B�#�                                    BxwĒ
  �          @����\)?^�R��=q¢�)B�{��\)@fff�����P�B�33                                    BxwĠ�  �          @��H>�Q�?c�
��\) �HB���>�Q�@e���=q�N��B�aH                                    BxwįV  �          @��?}p�?Ǯ���H\)Bf�?}p�@��H����1(�B���                                    BxwĽ�  �          @���?��@0  �����lp�B|�?��@���H������B�aH                                    Bxw�̢  �          @��?p��@z�����8RB���?p��@�33�]p��z�B���                                    Bxw��H  �          @�Q�>B�\@2�\�����v�B�W
>B�\@�ff�G�� �HB��                                    Bxw���  �          @�  ��@�\)��33�<��B�LͿ�@�ff��
=���B�33                                    Bxw���  �          @�  ����@�����R�6�B�����@���޸R����B�aH                                    Bxw�:  �          @�����@�
=��  �.{B��{����@�\)��G��bffB��                                    Bxw��  �          @Å�h��@�  ���R�=�\B��h��@�p������B�L�                                    Bxw�$�  �          @������@s�
��ff�E\)B�  ����@�  �   ���B�=q                                    Bxw�3,  �          @���?c�
@~�R���=p�B��)?c�
@�z������HB��                                    Bxw�A�  �          @�Q�?�(�@~�R��=q�833B�  ?�(�@�33��ff���B�p�                                    Bxw�Px  �          @���?�ff@��H��G��5�RB�(�?�ff@�p����H��Q�B�                                      Bxw�_  �          @�z�?E�@b�\���\�N��B�k�?E�@��\�\)��=qB��\                                    Bxw�m�  �          @�G����
?�33����Q�B��ü��
@u��y���6\)B��R                                    Bxw�|j  �          @���?c�
?�
=��  ��B�ff?c�
@����g
=��\B�p�                                    Bxwŋ  �          @�G�?���@�������d�BRz�?���@��
�7���\)B�                                    Bxwř�  T          @��?�@����p���B�=q?�@�G��i���\)B��3                                    BxwŨ\  �          @�?}p�>k����H� AU��?}p�@5�����a\)B�Ǯ                                    Bxwŷ  �          @�?�ff?Y����G��3B{?�ff@O\)�}p��B\)B�=q                                    Bxw�Ũ  �          @�{?�\)@Tz������Gz�Br��?�\)@��
����p�B���                                    Bxw��N  �          @��?�p�@>{���\�]�
B}�H?�p�@�
=�.{��
=B�Ǯ                                    Bxw���  �          @��?�p�@j�H�����B�\B�?�p�@��
�   ��G�B�=q                                    Bxw��  �          @�=q?333@!������|  B�G�?333@�  �Mp���B�\)                                    Bxw� @  �          @�=q?���@dz���ff�G�RB���?���@�����Q�B�
=                                    Bxw��  �          @�\)?\@����xQ��"ffB��H?\@�zῠ  �@��B��H                                    Bxw��  �          @�z�?�=q@��\�\)�+�B��H?�=q@��׿�
=�aB��f                                    Bxw�,2  �          @�ff?�{@���y���#��B��\?�{@�\)���H�;�B�#�                                    Bxw�:�  �          @�z�>\@xQ�����?\)B��3>\@��׿�ff���\B�
=                                    Bxw�I~  �          @��>B�\@�  �S33��B��\>B�\@�\)��p��`��B�G�                                    Bxw�X$  �          @�{?�  @�\)�Fff��G�B�G�?�  @���W
=��p�B���                                    Bxw�f�  �          @��@1�@b�\�Z=q��BO\)@1�@��׿�Q��B=qBn��                                    Bxw�up  �          @��@,��@mp��^{�z�BX  @,��@�ff��z��9p�Bu�                                    BxwƄ  �          @�p�@\)@����������Bt
=@\)@��?��@���B{�                                    Bxwƒ�  �          @��?��R@���+��У�B�#�?��R@��H@
�HA�
=B�G�                                    Bxwơb  �          @�{@�@��
��Q��9�B�Ǯ@�@�  ?�33A�B��
                                    Bxwư  �          @��\@.�R@��R=�?�(�Bt�@.�R@��@%A�\)Bd                                    Bxwƾ�  �          @���@/\)@�33�B�\��33Bq��@/\)@�G�@G�A�  Be�\                                    Bxw��T  �          @�=q>��@�\)�}p��+�\B�\)>��@��Ϳ�ff�O\)B��                                     Bxw���  �          @��>���@c�
��(��QffB��=>���@����{���RB��                                    Bxw��  �          @���>aG�@Q���\)�\�RB�Q�>aG�@��R�������B�(�                                    Bxw��F  �          @�?p��@H�����
�Z�RB���?p��@�G���H�ʏ\B���                                    Bxw��  �          @���?�@p�����B�B���?�@�  �U��\B��                                     Bxw��  �          @��׾.{@8Q����R�o(�B�Ǯ�.{@�\)�5��\)B��                                    Bxw�%8  �          @�
=��?����\)#�BЮ��@�Q��dz��=qB��                                    Bxw�3�  �          @���?c�
@;�����d�B�G�?c�
@���'����B��=                                    Bxw�B�  �          @�33?���@i�����H�@{B��3?���@��H��z�����B�{                                    Bxw�Q*  �          @��R��z὏\)����­��C@����z�@.�R��=q�q=qB�=q                                    Bxw�_�  �          @�Q��?:�H���¤�)B�  ��@]p���\)�Q�B�G�                                    Bxw�nv  �          @�(�?J=q@~{��ff�8G�B�� ?J=q@����У����B���                                    Bxw�}  �          @��?(��@�Q��Vff�\)B�B�?(��@��H��R���B�u�                                    Bxwǋ�  �          @���?Q�@���u�(�HB��?Q�@��ÿ�(��E��B��                                    Bxwǚh  �          @���W
=@QG���(��T��B�Q�W
=@�G��Q���
=B�                                    Bxwǩ  T          @�ff@ff@'
=���
�L�B@(�@ff@��R�����By                                      BxwǷ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw��Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw��               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxw��            @���@p�@���vff�=qBu�H@p�@�ff���\�B�HB��f                                    Bxw��L  T          @�\)��=q@��У����RB�ff��=q@�Q�?��AR�RB���                                    Bxw� �  �          @�Q쾔z�@���O\)�z�B��{��z�@�Q�u�  B��=                                    Bxw��  "          @���>�  @��R�j�H�p�B�  >�  @�p��G���RB�8R                                    Bxw�>  T          @\?n{@�Q��{��!B���?n{@�(������'\)B���                                    Bxw�,�  T          @�\)>�{@�33�'����HB�aH>�{@�\)>���@R�\B�33                                    Bxw�;�  "          @�z῜(�@��R=�?�{B�8R��(�@�=q@G�A��B��                                    Bxw�J0  
�          @�=q�!G�@�(��n{��HB�Q�!G�@�G�@�A���B�\                                    Bxw�X�  �          @�(��5�@�p�=L��>�B���5�@���@333AٮB��                                    Bxw�g|  "          @��
�+�@�  ���
�J=qB�  �+�@���@.{A�
=B                                    Bxw�v"  
�          @\�(�@�ff��ff���HB�(��(�@�p�@Q�A�33B��                                    BxwȄ�  T          @���?��@�33?�p�A=B�Q�?��@�ff@y��B#p�B�L�                                    Bxwȓn  �          @�G�?k�@�p�=�\)?#�
B�ff?k�@��@Dz�A�Q�B�z�                                    BxwȢ  T          @�?���@�Q�?
=@��RB�aH?���@�(�@c�
BQ�B�L�                                    BxwȰ�  T          @Ǯ?!G�@�z�@G�A��B��
?!G�@�G�@��BD�B�W
                                    Bxwȿ`  �          @��?�@��@AG�A���B��?�@{@�
=Bh�BUp�                                    Bxw��  �          @�\)?���@�z�@HQ�B	�RB��?���?��H@���Bp33B4�
                                    Bxw�ܬ  }          @���?�
=@�ff@6ffA�p�B�p�?�
=@3�
@�
=B^=qB^33                                    Bxw��R  �          @�=q?�
=@�=q>\)?���B�(�?�
=@�\)@=p�A�33B�B�                                    Bxw���  T          @�Q쾅�@�������)��B�(����@�G�@G�A�{B�ff                                    Bxw��  T          @���Tz�@��,����33B�z�Tz�@���>�ff@�ffBøR                                    Bxw�D  
�          @�������@��e���BУ׿���@���0����33B�{                                    Bxw�%�  "          @�녿��R@�(��c33�z�Bר����R@����0���ҏ\B��
                                    Bxw�4�  	�          @�33��@�p��dz���B�3��@���s33��HBٞ�                                    Bxw�C6  
Z          @�=q@�@��?L��AG�B��f@�@��H@W
=Bp�Bzp�                                    Bxw�Q�  T          @��H?��R@�
=�u�z�B��?��R@�\)@/\)A���B�z�                                    Bxw�`�  
�          @�  ?У�@�zῊ=q�'\)B��?У�@���?���A�ffB�W
                                    Bxw�o(  �          @�(�>�G�@�
=�#33��33B��3>�G�@�Q�?(�@��B��                                    Bxw�}�  T          @�z�>�z�@����z=q�&(�B�aH>�z�@��ÿ�{�-��B�33                                    BxwɌt  	�          @�\)>.{@�p��~�R�'�
B���>.{@��\��z��3�B���                                    Bxwɛ            @�
=��p�@|�����@��B��q��p�@��������
B��                                     Bxwɩ�  
Z          @�Q�<#�
@z�H����A\)B���<#�
@�(����
���\B��q                                    Bxwɸf  
�          @��=u@q�����DQ�B�(�=u@�\)����ffB��R                                    Bxw��  "          @������@%����|�B�uþ���@��
�HQ���B�L�                                    Bxw�ղ  
�          @�(�>��@�(�����4�B���>��@�p����H�e�B��                                     Bxw��X  
Z          @���{@I������fG�B�uþ�{@�Q��*�H����B�z�                                    Bxw���  �          @�G��B�\@3�
���R�n\)B�=q�B�\@�ff�4z����HB�k�                                    Bxw��  "          @�=q��=q@#�
���
�dp�B�녿�=q@�ff�7���
=Bހ                                     Bxw�J  T          @��?k�@���� ����{B��?k�@��H?Y��AB�\)                                    Bxw��  
�          @�����Q�@J=q����b��B�ff��Q�@�� ����33B�\                                    Bxw�-�  
�          @��?���@�  �S�
�Q�B�z�?���@��H�(�����
B��3                                    Bxw�<<  �          @�Q�?�Q�@G���p��_��BE�?�Q�@�
=�(Q����
B��                                    Bxw�J�  �          @�{��@��@\)A���B�z���@#33@�z�BN  C.                                    Bxw�Y�  "          @�Q��4z�@�G�@>{A��\B��)�4z�?�Q�@��
BU��Cn                                    Bxw�h.  T          @���\@���@8��A��B�Ǯ��\@=q@���BhffB���                                    Bxw�v�  
�          @�{?=p�@�p�������B�Ǯ?=p�@��H?8Q�@�\)B�                                      Bxwʅz  �          @���?8Q�@�
=�\)��z�B�G�?8Q�@�(�?B�\@�\B�k�                                    Bxwʔ   �          @��R?�@��׿�z��h(�B�?�@�  ?�  Au�B���                                    Bxwʢ�  	�          @��?ٙ�@��
��G��)��B�B�?ٙ�@��?�G�A�  B���                                    Bxwʱl  
Z          @��?�33@��R���H��(�B�aH?�33@�@��A�33B�
=                                    Bxw��  "          @�z�?�@�  �Q����RB���?�@�\)?(��@�{B���                                    Bxw�θ  
Z          @�z�?�G�@��� ����G�B��?�G�@���?
=q@�G�B��                                    Bxw��^  �          @��R>���@����O\)��\B��f>���@��
����   B�{                                    Bxw��  �          @�
=?�\)@��Ϳ\�q��B���?�\)@��?�(�Ai��B��R                                    Bxw���  
Z          @��
@HQ�@�z�333��(�Be�@HQ�@��?�A���B^                                      Bxw�	P  �          @�z�@Dz�@�p�>�33@Z�HBg��@Dz�@�G�@5�A���BR�                                    Bxw��  
�          @�{@<(�@�G�=��
?B�\Bo(�@<(�@���@*=qA���B^G�                                    Bxw�&�  T          @�
=@�\@���.{��p�B�G�@�\@��@\)A�ffB�#�                                    Bxw�5B  T          @��?�{@�33���H��{B�p�?�{@��R?��AK�B��                                    Bxw�C�  T          @���@�
@��\���
��Q�B���@�
@��?�(�A<  B��f                                    Bxw�R�  
Z          @���@Q�@�p������up�B�
=@Q�@�
=?�AY��B�aH                                    Bxw�a4  
�          @�?�@���
��{B�W
?�@�(�?0��@�  B��3                                    Bxw�o�  
�          @�  �A�@��
��p��f�HB��
�A�@�?��RAA��B�(�                                    Bxw�~�  T          @�(�=L��@�p��33��p�B���=L��@��?fffA��B�
=                                    Bxwˍ&  �          @����=q@AG���G��f�B�
=��=q@���#33�ՙ�B�                                      Bxw˛�  
�          @��R��(�@�H�����g��B򙚿�(�@���(����\B��                                    Bxw˪r  
�          @�{�>�R?
=��Q��e�HC(ٚ�>�R@=p��q��'��C:�                                    Bxw˹  "          @��
�Mp�=��������Z�RC2@ �Mp�@�H�u��.��C�q                                    Bxw�Ǿ  �          @�z��G�?Tz����H�[��C%��G�@E��`����CY�                                    Bxw��d  �          @�33�Q�@,(��~{�,G�C���Q�@�33� ����(�B�                                    Bxw��
  
�          @�z���H@q��qG�� 
=B�33��H@����
�I�B�
=                                    Bxw��  
Z          @�\)�G�@N�R�|(��7�HB�33�G�@�G����H���
B�                                    Bxw�V  T          @����G
=?�G��}p��F�C��G
=@G��/\)��\)C��                                    Bxw��  T          @�G��u@P�������T�B��H�u@�(��p���p�B�=q                                    Bxw��  T          @��ÿ\(�@�33�w
=�*��B�Ǯ�\(�@�  �����BffB���                                    Bxw�.H  �          @�{@6ff@���\��
=Be
=@6ff@�33?
=q@�\)Bn33                                    Bxw�<�  
Z          @�=q@!�@���z���\)Bu��@!�@�{>��@�=qB�                                    Bxw�K�  "          @��
@!�@��R��z����HB{�H@!�@��?���A4  B~�\                                    Bxw�Z:  
�          @�=q@G�@�{�����B���@G�@�
=?fffA\)B�
=                                    Bxw�h�  
�          @��
?�  @�ff�˅��G�B�Ǯ?�  @���?���AU��B�G�                                    Bxw�w�  �          @��\?�@�녿����-�B�
=?�@��?���A��\B��=                                    Bxw̆,  T          @��>�=q@�G��]p��p�B��f>�=q@��R�@  ���B�k�                                    Bxw̔�            @�{?p��@xQ��x���1(�B��?p��@��\����]B��                                    Bxẉx  �          @�ff?�\)@j�H���=�B�\?�\)@�����Q���=qB��                                    Bxw̲  �          @���?�G�@j=q���\�B�HB�
=?�G�@��
������\)B���                                    Bxw���  �          @��H?�{@���333����B��{?�{@�G����
�Q�B�(�                                    Bxw��j  T          @���?�ff@tz��i���*�
B�L�?�ff@��Ϳ�z��G33B�                                    Bxw��  �          @��H>�(�@.{��33�q�\B�(�>�(�@��\�0  ��ffB��f                                    Bxw��  �          @��\?!G�@<(���p��d�B��?!G�@��\)�ԣ�B��                                     Bxw��\  �          @�(�?u@n{�����9{B�?u@��׿��
�~ffB�G�                                    Bxw�
  T          @���?�33@����L(��
��B��3?�33@�������Q�B�                                      Bxw��  �          @�=q?��\@\)���
�|�B�#�?��\@��@���
=B�{                                    Bxw�'N  �          @�=q?+�@  �����)B��?+�@����H���

=B�.                                    Bxw�5�  �          @��R?�{@�
�����k33B\z�?�{@�(��2�\��\)B���                                    Bxw�D�  �          @�  ?�p�@(����(��S�BTG�?�p�@����������B���                                    Bxw�S@  �          @��?��\@!������m=qB}ff?��\@�(��2�\���HB�L�                                    Bxw�a�  
�          @�(�>\)@Dz����H�`�
B��f>\)@�  �ff�ɅB��H                                    Bxw�p�  �          @�  ?n{@s33�u��1�\B�\)?n{@�\)�����_�B��q                                    Bxw�2  �          @��?��@�ff�;���33B���?��@��\�aG��  B��R                                    Bxw͍�  �          @�  �z�@�������HB�\)�z�@�G�?�Q�A@  B��                                    Bxw͜~  
�          @�
=>�@�p��%���
=B�
=>�@���>���@X��B�aH                                    Bxwͫ$  
�          @��
>�\)@mp���ff�A��B�� >�\)@��H��Q���=qB��                                    Bxw͹�  �          @�G�?333@3�
���i  B��?333@�=q�#�
���B��                                    Bxw��p  �          @���=u@}p��vff�0B�.=u@�(���G��R�RB���                                    Bxw��  �          @����G�@���33��=qB��{��G�@�G�?p��A=qB�
=                                    Bxw��  �          @����@�z��N�R�G�B�ff���@�z�\�uB�B�                                    Bxw��b  �          @�\)����@���9��� �B������@�ff��G���\)B�#�                                    Bxw�  �          @�G����
@����C33��HB������
@�\)��
=����B�
=                                    Bxw��  T          @�z�8Q�@��;����B��\�8Q�@�녾k�� ��B���                                    Bxw� T  �          @��?#�
@�  �U����B��?#�
@��ͿJ=q�33B��f                                    Bxw�.�  �          @����p�@��\�E����B�G���p�@��\�
=q���HB�=q                                    Bxw�=�  �          @�z�u@~{�=p��B�B��u@����H���
B��                                    Bxw�LF  �          @��H��  @hQ��B�\�Q�B�uÿ�  @��=p��33B�                                      Bxw�Z�  
�          @��H�#�
@w��<(��p�BƮ�#�
@��\����{B�                                      Bxw�i�  �          @�  �l(�@Fff������C�R�l(�@h�ýL�Ϳ
=qC\)                                    Bxw�x8  �          @�
=�^�R@Q녿�Q���  C���^�R@k�>u@+�CW
                                    BxwΆ�  �          @��Ϳ.{@�
=�:�H�	�B�8R�.{@��
��{�n{B�\                                    BxwΕ�  �          @�
=>8Q�@]p���
=�Jp�B�.>8Q�@�(�����\)B�{                                    BxwΤ*  �          @�G��k�@XQ���33�P33B��f�k�@�z���R��
=B�33                                    Bxwβ�  �          @��׿��@G����R�Z��B�.���@�
=�{��  B��q                                    Bxw��v  �          @����\)@K���z��X  B�𤾏\)@�\)�Q�����B�W
                                    Bxw��  �          @���>��H@7
=��p��h�RB��>��H@�33�"�\���B��\                                    Bxw���  �          @���>��@HQ����R�Zz�B�>��@�\)�{���B�B�                                    Bxw��h  �          @�=q?�\)@k��}p��7Q�B��?�\)@�{��G��}�B���                                    Bxw��  �          @�=q?˅@G���p��H  B{�
?˅@�=q��p���=qB�p�                                    Bxw�
�  �          @��?�(�@W������;�RB|�?�(�@�
=�޸R��ffB��                                    Bxw�Z  
�          @���?u@~{�mp��)�B��q?u@�녿�33�@  B�u�                                    Bxw�(   
�          @�ff?@  @Tz���  �M=qB��3?@  @��ÿ�Q����B�#�                                    Bxw�6�  �          @�=q?��@U��z��Qz�B�\)?��@��
�33���
B�Ǯ                                    Bxw�EL  �          @�
=?���@����*�H���B�G�?���@�  <�>�z�B�B�                                    Bxw�S�  �          @��?�33@��R�B�\�\)B��f?�33@�z�?��A���B�33                                    Bxw�b�  �          @��?��@�  >8Q�@	��B�� ?��@~�R@%B��B�                                    Bxw�q>  �          @��þ�@�  @QG�B  B�����?�G�@��B�(�B�W
                                    Bxw��  �          @�
=�#�
@�Q�>�33@���B�B��#�
@��
@7�B�B�Q�                                    Bxwώ�  �          @�논#�
@���@ ��A�\B�#׼#�
@��@��Bt�B�=q                                    Bxwϝ0  �          @�G��h��@��\@{A�p�B�p��h��@8Q�@��HBb�RB�8R                                    Bxwϫ�  �          @��R����@��?xQ�AffB�aH����@���@g
=B��B�B�                                    BxwϺ|  �          @�G��   @�z�?G�@��\B�\�   @�{@c�
B
=B��q                                    Bxw��"  �          @��H?�\@�=q�#�
��
=B�
=?�\@���@%A��
B�u�                                    Bxw���  
�          @�G�?xQ�@�
==�G�?�
=B���?xQ�@�@1G�A�33B�L�                                    Bxw��n  ,          @��\?��@������0(�B�� ?��@�?ٙ�A�G�B�(�                                    Bxw��  J          @��H?�
=@��R�^�R��B�k�?�
=@��R?��A��HB�\)                                    Bxw��  "          @���?���@�p��xQ��&=qB��f?���@�{?�A���B��R                                    Bxw�`  T          @�Q�?���@��Ϳ��
�Z=qB�k�?���@��\?��A�  B��                                    Bxw�!  �          @���?���@���>\)?�\)B��=?���@�@<��A�  B��                                     Bxw�/�  T          @�
=?5@�Q�?���AU�B��?5@��@�(�B/G�B���                                    Bxw�>R  
�          @��R=L��@�
=?�
=A��HB���=L��@��@��
B<ffB��                                    Bxw�L�  �          @�p���\)@�ff@��A�{B�k���\)@a�@�
=BT��B�.                                    Bxw�[�  �          @�=q�0��@��H@G�A��\B�uÿ0��@N{@���BXp�B�8R                                    Bxw�jD  
�          @����=q@�p�@@  A��B�녿�=q@�R@��RBq��B�=                                    Bxw�x�  �          @�p����@�
=@VffB
p�Bᙚ���@��@��Bt�CY�                                    BxwЇ�  "          @��33@p  @\)B(B��33?�Q�@��HB�G�C��                                    BxwЖ6  
�          @�{��\@~�R@xQ�B#�HB�(���\?�Q�@��
B�(�C��                                    BxwФ�  
Z          @��Ϳ��\@y��@��B9�
B�p����\?�Q�@�(�B��)C�3                                    Bxwг�  �          @��
��@hQ�@�Q�B;ffB�G���?s33@���B���CE                                    Bxw��(  �          @���
=@q�@u�B"��B�\�
=?�ff@�
=Bz��C+�                                    Bxw���  �          @�>L��@N{@�
=Bc�HB�Ǯ>L��>u@�p�B��)BH�                                    Bxw��t  
�          @�p�>�(�@G
=@��BgffB�W
>�(�>�@�(�B�G�A��\                                    Bxw��  T          @�p�=�Q�@@  @��BjQ�B�k�=�Q�=��
@�Q�B��B#�R                                    