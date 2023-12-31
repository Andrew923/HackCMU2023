CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230227000000_e20230227235959_p20230228021620_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-28T02:16:20.184Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-27T00:00:00.000Z   time_coverage_end         2023-02-27T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxi�@  
Z          AA����  ��\)�&ffC=���@5��33�
=C"=q                                    Bxi��  	`          A7�
�\)��������ޏ\C:)�\)?�\��ff���C)��                                    Bxiь  
�          A:=q�3�
�p���p��D��C8�=�3�
>��%�O�C1��                                    Bxi�2  	�          A2�H����Q���e��C_&f������������CS޸                                    Bxi��  �          A/\)>#�
����Az�BpffC���>#�
�	��@���A�Q�C���                                    Bxi�~  
n          A/
=�@���z�@�RAip�C{�\�@���(��"�\�o
=C{��                                    Bxi$  
x          A$Q���z���p���  �I�Ca0���z�=��
�\)�|�C2�                                    Bxi�  "          A4Q����׿���$(��RCD)����@�p��\)�c�C	O\                                    Bxi)p  
�          A6ff����>�33�ff�n\)C0k�����@����G��9�C�R                                    Bxi8  "          A8���C�
?����.�\�
C@ �C�
@�{��R�<�B�(�                                    BxiF�  �          A2ff�_\)@.{��.C{�_\)@�
=�����#B�\                                    BxiUb  �          A9���{@:�H�/\)��B����{A=q��z��,=qBҊ=                                    Bxid  
�          A7�
��ff@AG��$(��|�CO\��ff@��
���=qB�33                                    Bxir�  
�          A=���\)@z=q�33�f(�C�f��\)A��У��G�B��                                    Bxi�T  
�          A?
=�љ�@��
���?  CǮ�љ�A  ��{��Q�B�.                                    Bxi��  
�          A>ff��\)@�{����� �
C���\)A	���p����(�C�                                    Bxi��  "          A*ff�ָR@�����  CT{�ָR@��'��g�Cff                                    Bxi�F  T          A����=q@��ƸR�p�C
xR��=q@�p��2�\��  B�                                    Bxi��  
�          A6ff��Q�@{��
=�SffC� ��Q�@�
=�������B�\)                                    Bxiʒ  
�          A'��#�
�˅��
��CS���#�
@h����HB�B�#�                                    Bxi�8  
�          A1���H���\�	��M�C`�3���H>���p��{C0��                                    Bxi��  
Z          A)p��?\)�
=��=q��{C}:��?\)��
=�����(�Cyh�                                    Bxi��  
�          A\)��z������z��1�Cz�Ϳ�z��  ��ffC^��                                    Bxi*  
�          A�\�����
=�����(�C������!�����#�Cv�)                                    Bxi�  
�          A(��h���Q���p���(�C��)�h�����R�  �oG�C���                                    Bxi"v  
(          A (�?+���
=��\)�+p�C���?+��0  ���qC��q                                    Bxi1  
Z          AC33@����?�p�@��HC��\@������a���  C���                                    Bxi?�  
(          AN=q@��H�=q@�B�RC�{@��H�.{?�\)Ap�C���                                    BxiNh  	�          AN�\@�\)��@�p�B�HC��R@�\)�-�?�33A	�C�xR                                    Bxi]  	B          AAG�@��H����A
ffB7=qC�H@��H��@��\A�z�C���                                    Bxik�  �          AB=q@�ff��z�A\)BE�\C��@�ff��\@��A�  C�k�                                    BxizZ  	L          AJ{@�����  A�BB��C���@����=q@��\A�Q�C�f                                    Bxi�   	          AK�
@����vffA!BS
=C���@�����@�
=A��C��                                    Bxi��  	B          ARff@�{�/\)A:=qBy�C��H@�{���A	��B#�C��)                                    Bxi�L  	`          AX��@��Ϳ
=qA7
=Bg�RC��3@������A��B4=qC�G�                                    Bxi��  	`          AF�R@ᙚ>\)A!��B\=q?�\)@ᙚ��G�A(�B5�C�:�                                    BxiØ  �          AG�@θR@~{A��BQ�A���@θR�˅A'\)Bf(�C�
                                    Bxi�>  	B          AA�@^{@�\A{B>z�B�\)@^{?�\A6{B�8RA�ff                                    Bxi��  	~          A2=q@�{@��@�\)B9
=B2G�@�{?
=qAp�BqQ�@�                                    Bxi�  
�          A8���QG��!p���Q����C|��QG���z��	���E{Cs�                                    Bxi�0  	�          A@Q��N{�-��|(����C}}q�N{��33��H�@(�CvT{                                    Bxi�  
�          AAG��W
=�(Q���(�����C|G��W
=�������Q\)Cr�f                                    Bxi|  
�          A<���   �.{��G�����C�Ǯ�   �陚�Q��G�HC~��                                    Bxi*"  	�          A>�H����3�
�hQ���ffC���������\����@p�C�Q�                                    Bxi8�  T          A;��h���'\)�����̸RC��f�h����ff�ff�_��C�H                                    BxiGn  
(          A:=q�\)�����
��RC��R�\)�����&�H�~�C�<)                                    BxiV  T          AB{>�p��"ff���
�\)C�
=>�p���(��-�~\)C���                                    Bxid�  �          ADQ�?W
=�1��33����C�*=?W
=��\)� ���[��C�p�                                    Bxis`  
�          ADz�@���3���z���Q�C�c�@�������K�C��                                    Bxi�  
�          ABff?�33�:�\�333�T��C��H?�33�
{�G��-(�C�7
                                    Bxi��  	`          A@z�@E��7\)�����\)C��@E��\)��(���C�}q                                    Bxi�R  
�          A@��@7
=�9���
=� z�C���@7
=�{��=q��C���                                    Bxi��  �          AC�
@$z��=�\���C��@$z��33��=q�{C��)                                    Bxi��  
�          AJff@.�R�E����H��RC�@ @.�R�&{������RC�`                                     Bxi�D  	�          AE��@5�?��8Q�Tz�C��=@5�#�
��
=��\C���                                    Bxi��  	�          AE��@?\)�?
=�#�
�@  C��@?\)�#�������C�'�                                    Bxi�  
�          A^�R@�G��O�>u?xQ�C�!H@�G��4����33���HC�xR                                    Bxi�6  	`          A^�R@tz��T��=L��>L��C�f@tz��8  ��p�����C�1�                                    Bxi�  
�          A\  ?ٙ��W�
?z�@��C��R?ٙ��?
=�������C�                                    Bxi�  
�          A[��8Q��Z{=L��>L��C����8Q��<�����H���HC��\                                    Bxi#(  	.          A\  @   �W�
?�  @�Q�C�4{@   �Dz���
=�ŅC��q                                    Bxi1�  	�          A^�\@fff�O\)@_\)Ah��C��H@fff�O33�aG��j�\C�                                    Bxi@t  	t          A_\)@�ff�8��@��A�33C�"�@�ff�H�Ϳ�33���C�5�                                    BxiO  
F          A[\)@���0  @���A�\)C��\@���K��#�
��C�t{                                    Bxi]�  �          AZ{@�ff�$��@�ffBQ�C���@�ff�J�\?�  @�Q�C��\                                    Bxilf  
�          A[33@���G�A ��B{C��H@���H��@ ��A�HC���                                    Bxi{  
(          Ab�R@��)G�@���A�=qC�#�@��>�H�
=q�33C���                                    Bxi��  	�          Ag�@�33�Mp�@.�RA.�HC�AH@�33�G����H��ffC���                                    Bxi�X  	.          A`��@u��F�R@�\)A�  C��3@u��O�
����\C�8R                                    Bxi��  
Z          Ab�H?��]���33���C�p�?��-G���H���C�b�                                    Bxi��  
�          A_���=q�M�������33C�����=q���)�IffC�xR                                    Bxi�J  �          AY���@  �Å��
=C�8R����=q�4���b��C�xR                                    Bxi��  
�          AX��=�\)�QG��J�H�Y��C�&f=�\)��
����.��C�33                                    Bxi�  	          AF=q@Y���=�?Tz�@w�C�@Y���*ff��ff��p�C�ٚ                                    Bxi�<  	�          AC�
@�33� ��@<��Aa��C�� @�33�"�H�{�<(�C�w
                                    Bxi��  �          A@Q�@�33����@�=qAʸRC���@�33�Q�?   @�C�5�                                    Bxi�  *          A?
=@ڏ\���R@��A���C�Ǯ@ڏ\���?�\)@�ffC���                                    Bxi.  
x          AD��@W��;
=?�p�@���C��@W��+�
������C��
                                    Bxi*�  	�          AC�@.�R�8��@
�HA%p�C��@.�R�1���x����Q�C���                                    Bxi9z  
�          AD(�@���<  ?�ff@���C��@���+\)��ff���
C�s3                                    BxiH   
(          AEG�@0���>{��R�:=qC���@0����
�θR��\)C���                                    BxiV�  
�          AD  @��=G�����C�޸@��p����Q�C�"�                                    Bxiel  	�          AG�@\���;��\(���  C�0�@\���  �ҏ\��HC��q                                    Bxit  
�          AO�@�\)�8��?E�@Y��C�33@�\)�&ff���\��33C�n                                    Bxi��  	B          AS33A
=��@33A"{C�w
A
=�p��5�Hz�C���                                    Bxi�^  	L          AT��@�=q�'�@1�ABffC���@�=q�'��2�\�B�RC��                                    Bxi�  
(          AL��@U�C�
�.{�E�C�� @U�(����\)����C�Ǯ                                    Bxi��  
(          AM��@��\�?�=�?�C��\@��\�'\)���H��Q�C�=q                                    Bxi�P  
�          AQ��@���D�׾u����C��@���(���ə���33C�9�                                    Bxi��  
Z          APz�@0  �G
=���\)C�=q@0  ��� ���(�C��f                                    Bxiڜ  
Z          AH��@(��@���33�33C��R@(��33������C�9�                                    Bxi�B  	�          AL��?�
=�C��I���f�\C���?�
=�������/z�C��
                                    Bxi��  
�          AK
=@z��D�Ϳ�  �ڏ\C�Y�@z��33��(��\)C���                                    Bxi�  
(          AE��@z=q�8�ÿ�  ��RC�U�@z=q�
=��\)��C���                                    Bxi4  
�          AE@(��=p��Q���C��3@(��  ��{�=qC�e                                    Bxi#�  
Z          AH��@HQ��&�\��Q���C�aH@HQ����H�%�b{C���                                    Bxi2�  
(          AG�@�p��0Q��\����C��@�p���z��
=�,{C���                                    BxiA&  
Z          AB�H@G��4  �aG���Q�C��
@G�� ���	���7{C��H                                    BxiO�  
�          A@(��s33�1���
=��33C��=�s33��{�G��L�HC�Y�                                    Bxi^r  z          A@z�@K��*{��\)�p�C�Q�@K��G��ۅ�33C�o\                                    Bxim  �          A?�
@�
=�z�8Q�\(�C��3@�
=�����H���HC���                                    Bxi{�  �          A;\)@mp��)p�����
=C��@mp��
ff��
=��
C��R                                    Bxi�d  T          A;\)@2�\�.{�&ff�PQ�C�.@2�\�\)��33�$��C�`                                     Bxi�
  "          A<z�@XQ��/�
�(��?\)C��@XQ��=q��Q���
C��)                                    Bxi��  �          A@(�@N{�2�H���H��z�C�f@N{�����  �z�C���                                    Bxi�V  T          A@�׿h���(�����\��=qC��Ϳh����\)� ���c��C�                                    Bxi��  �          A@  ?�  �#���
=��
=C�+�?�  ��  �Q��b�C���                                    BxiӢ  "          AAp�@@���$��������z�C�!H@@����G���R�RffC�XR                                    Bxi�H  �          AD��?h���)���p����C�q�?h����  �"{�d(�C���                                    Bxi��  
�          AH  ?�33�������C�u�?�33��=q�2�\=qC�K�                                    Bxi��  
�          AG�����   �����Cl����G��-��rQ�CR�                                    Bxi	:  
�          AG��Y���	G���
=�#Q�Cxff�Y���g
=�2=qCb�                                     Bxi	�  "          AD��>\�"ff�����C�3>\��(��,(��y=qC��\                                    Bxi	+�  �          AC�
?�G��)����(���(�C��?�G�����$���e��C��                                     Bxi	:,  "          AE?�  �&ff�Å��G�C�
?�  ��(��&=q�j��C���                                    Bxi	H�  
�          AB{?�33��R����RC�:�?�33��p��0(�=qC�Z�                                    Bxi	Wx  �          A>ff@p��G���ff���C��@p����H�#
=�l=qC��)                                    Bxi	f  
�          AB�R@{��
���
�C�|)@{��Q��'�
�{C���                                    Bxi	t�  
�          AK33@p����
=�!��C��@p���\)�;\)��C���                                    Bxi	�j  
Z          ABff?�(��������C��?�(���p��+�
�|z�C�9�                                    Bxi	�  	�          AB=q?��%p���{��  C���?�������VffC��                                    Bxi	��  
�          AD��@x���4(��/\)�O\)C��f@x��������H�z�C�=q                                    Bxi	�\  T          A?33@��H�(z�0���VffC�o\@��H�=q�����
C�|)                                    Bxi	�  T          A>{@�ff�*�R?�33A�C�� @�ff�"ff�l(���=qC�j=                                    Bxi	̨  "          A@  @���"{@	��A&ffC��@����\�@  �iG�C�L�                                    Bxi	�N  "          AD  @�=q��ý�G��\)C��@�=q����\���HC�
=                                    Bxi	��  �          AH��@��ff����0��C�Ǯ@��=q������  C�)                                    Bxi	��  T          AJ{@����G�>\?�(�C�c�@��������z�����C���                                    Bxi
@  
�          AK
=@������?��\@�  C�]q@����  �g
=��=qC�>�                                    Bxi
�  �          AQp�A�
��H@��A�C�1�A�
�=q��H�*�HC�B�                                    Bxi
$�  
�          ANffA����@S�
An�\C���A���zΐ33��z�C��)                                    Bxi
32  
�          AK�A�R��@g
=A��RC���A�R�=q��  ����C��                                    Bxi
A�  �          AP(�A��ff@��A�ffC�p�A���\��zῢ�\C���                                    Bxi
P~  "          AT��A
=�=q@��A�33C�<)A
=��?\)@=qC���                                    Bxi
_$  
Z          APQ�A���
@���A��C�xRA��>�Q�?���C�XR                                    Bxi
m�  @          AS�@����@�ffA�z�C���@��.ff�\)��HC�AH                                    Bxi
|p  T          AK
=@�ff��
@�(�A�p�C��{@�ff�#�<#�
=L��C��                                    Bxi
�  T          AG
=@�{��
@��A���C�˅@�{�p��333�QG�C��=                                    Bxi
��  
P          AL��@�\)��{@�33A�z�C��@�\)�{?˅@�
=C�                                    Bxi
�b  T          AH(�A33��Q�@�z�A؏\C�7
A33���
@,��AIp�C���                                    Bxi
�  T          AJ�RA$����Q�@��HA�(�C�u�A$���ᙚ?��RA��C���                                    Bxi
Ů  "          AJ�RA(�����\@��\Aƣ�C�C�A(�����@!G�A733C��                                    Bxi
�T  T          AH(�Az�����@���A�33C���Az���
=@(�A4  C�~�                                    Bxi
��            AB�\A����@�=qA�C�qA����R@0��AS
=C�~�                                    Bxi
�            A3�@�
=��R@��A�33C�ff@�
=���?�(�AC�.                                    Bxi F  
�          A(  @У��Ǯ@�33A�C�)@У����?�ffA	p�C��)                                    Bxi�  
�          A (�@�����@���A��HC�@����Q�?��
A$  C��                                    Bxi�  �          A!G�@�\)��Q�@�A���C��3@�\)��33?�33A.=qC��                                    Bxi,8  
�          AG�@�  ��
=@���A���C���@�  ���H@��AhQ�C���                                    Bxi:�  T          A=q@����@�Q�A�
=C��\@����?�p�A&�RC��q                                    BxiI�  �          A!@���z�@�\)A�
=C�� @���  ?���@��
C��)                                    BxiX*  �          A$��@�G��j=q@��B�C�ff@�G���33@L��A�\)C��                                    Bxif�  
�          A"�R@�33���
@��\A��C�,�@�33����@AV=qC���                                    Bxiuv  T          A#�
A Q���  @_\)A�(�C�b�A Q�����?!G�@a�C���                                    Bxi�  
�          A�@�R��z�@p  A�
=C��@�R���?�33@�z�C���                                    Bxi��  "          A=q@��\�qG�@��HB�
C��R@��\��ff@K�A���C���                                    Bxi�h  
�          A�
@ȣ����@�Q�B�C�e@ȣ���{@Y��A��C�!H                                    Bxi�  
�          A�@ʏ\�HQ�@�\)B ��C���@ʏ\���@u�AÅC�\)                                    Bxi��  
�          @�G�@�����@�
=BffC��@�����33@9��A�
=C�
=                                    Bxi�Z  "          A@�  �Z�H@��B{C�t{@�  ��G�@-p�A�C�ff                                    Bxi�   T          A�þ�Q���
@N{A��RC��q��Q���ͿxQ����C���                                    Bxi�  �          A(��?�Q��@�33A�  C�}q?�Q��#33����Y��C��                                    Bxi�L  
�          A�\@����z�@���B{C��f@��� (�@z�AM��C�                                      Bxi�  
�          A
=@o\)��\)@�z�B$33C��@o\)�=q@+�A�
=C�U�                                    Bxi�  T          A��@e����
@�(�B  C�S3@e���R?ٙ�A ��C��                                    Bxi%>  T          A�@����33@���B(�C��@����
=@8Q�A�{C���                                    Bxi3�  �          A&�\@.�R����@��B
=C��R@.�R�  ?��HA	p�C�:�                                    BxiB�  T          A"=q@�=q��  @�  B �C�b�@�=q��@@  A���C�
=                                    BxiQ0  
�          A(  @�\)��@�z�A�RC�ff@�\)�Q�?���@�G�C���                                    Bxi_�  
�          A-�@����\@�A�ffC�+�@���(�?!G�@Tz�C�@                                     Bxin|  
�          A*�\@�(��Q�@a�A�
=C��=@�(���\�c�
���C���                                    Bxi}"  
�          A$(�@B�\�@���A�{C�t{@B�\�����H�/\)C���                                    Bxi��  
�          A%��@XQ���@tz�A�  C�S3@XQ����5�|��C���                                    Bxi�n  �          A&=q@l(��  @��A�33C�l�@l(��Q���8Q�C�U�                                    Bxi�  T          A%�@�{��@�A[33C��q@�{�\)��\)�'\)C��                                    Bxi��  �          A)G�@�R���\@�A�\C�� @�R��  ?���A0(�C���                                    Bxi�`  6          A\)@�z���=q?�33A*=qC�T{@�z���Q�Y����ffC���                                    Bxi�  "          A ��@�{���
?��A=p�C�]q@�{�
=?E�@�\)C�P�                                    Bxi�            A{@�(�>��H@���B\)@y��@�(���p�@�{B  C�<)                                    Bxi�R  �          A,z�@(����A�HB���C�y�@(��ҏ\@�Q�B;�RC�0�                                    Bxi �  
�          A&�\>\)�.{A z�B���C�~�>\)��
=@��RB?ffC���                                    Bxi�  T          A#\)?�=q��A��B�Q�C�S3?�=q��@��BMz�C��                                    BxiD  �          A%?��H�w�A�HB�p�C�t{?��H��@�{BQ�C�<)                                    Bxi,�  "          A"ff@����:�HA��Bm�C�p�@�����
=@�=qB�C�Q�                                    Bxi;�  �          A�@�����Q�@@��A���C�/\@������?B�\@�G�C���                                    BxiJ6  T          A\)@�p����
>�=q?�33C�33@�p���
=���H�B�\C�H�                                    BxiX�  T          A	�@�p������H��C�T{@�p�>W
=�����{?��
                                    Bxig�  
�          @�  @ڏ\=�G��Z=q��z�?s33@ڏ\?���@  ���HAW�
                                    Bxiv(  T          A@���G
=�U���C�33@��������
��  C�Z�                                    Bxi��  
Z          A��@��H��ff�Q���p�C��@��H������ 33C�^�                                    Bxi�t  r          A�
@�ff������H��=qC���@�ff�S33��G��,�C�w
                                    Bxi�  �          A
=@�������G���G�C�� @���7��Ǯ�6(�C�                                    Bxi��  
�          A��@��H��  ���H��\)C��@��H��H�����%�RC�.                                    Bxi�f  
�          A  @�\)���������=qC�\@�\)�����ٙ��7(�C�)                                    Bxi�  
Z          A�R@�z���\)��
=��Q�C��)@�z��l������&ffC���                                    Bxiܲ  "          A33@�������G����HC�#�@����5��\)�)(�C�AH                                    Bxi�X  
�          A$(�@�\�����ff�p�C���@�\��ff��z��2G�C���                                    Bxi��  
2          A#
=@��H����������C�4{@��H���=q�.z�C���                                    Bxi�  
�          A�@��H��
=���
� 33C�h�@��H�޸R��=q�+��C�޸                                    BxiJ  
�          A��@�  �l(�������HC�H�@�  ����������C�/\                                    Bxi%�  �          A)A���:=q�5��v�HC��)A�ÿ�{�tz�����C��                                    Bxi4�  "          A$��A�\�ff�W
=����C��qA�\��R������(�C�)                                    BxiC<  "          A�A{�#33�33�k
=C���A{���\�K����RC���                                    BxiQ�  	�          A@�\)���R���\�G�C�f@�\)?\)��p����@�33                                    Bxi`�  "          A*=qA�\� ����33��Q�C�  A�\=�������G�?&ff                                    Bxio.  
�          A6ffA���O\)�����p�C��A���\(������ffC�~�                                    Bxi}�  T          A7�
Aff�XQ���ff��C��{Aff�xQ����R���HC�33                                    Bxi�z  T          A;\)A'
=�(����\����C�l�A'
=�B�\����33C�}q                                    Bxi�   �          A/\)A"�H�Q��N{���HC�o\A"�H�:�H�{����\C���                                    Bxi��  "          A.�RA%p��:�H��\)�p�C�qA%p����=p��{�C��                                    Bxi�l  �          A0��A((��R�\��Q��C�Q�A((��:�H�\��p�C�@                                     Bxi�  T          A3�A#��A��Q��4��C���A#���ff�O\)���RC��                                    Bxiո  �          A6�\A\)�.{��\)�ĸRC�^�A\)�Ǯ��{����C��                                    Bxi�^  	�          A6�\A(���.�R��R�8��C��qA(�׿��R�L(����C���                                    Bxi�  T          A6�RA0���,��>�{?�(�C�%A0���%��\(���33C�o\                                    Bxi�  
�          A;\)A6�\���?aG�@���C�
A6�\�!녾�  ����C��q                                    BxiP  
�          A=p�A7��1�?aG�@�  C�1�A7��8Q��녿��RC���                                    Bxi�  
�          A8��A333������E�C�� A333�Q쿘Q����RC��                                     Bxi-�  	�          A6ffA%p�������ff��ffC��\A%p�>�����H��\)?Q�                                    Bxi<B  
�          A6�HA���]p����H� �RC���A�Ϳ   �߮���C�n                                    BxiJ�  
�          A9��A�\�������
=C���A�\������\)��C��R                                    BxiY�  "          A=A�������
���C��A���33���,��C�z�                                    Bxih4  T          A>{A	p���  ���H� �HC�EA	p��G�����(��C�aH                                    Bxiv�  
�          A4(�A*�R�A녿Y����(�C�{A*�R���z��+33C��q                                    Bxi��  
�          A6�RA&{��=q�	���.=qC�H�A&{�0���k���p�C���                                    Bxi�&  �          A;33A$z������`������C�AHA$z��
=q������(�C��                                    Bxi��  T          A<  Ap��������33C�aHAp��\)�ȣ��ffC��{                                    Bxi�r  "          A?�
A�H����  �ڸRC���A�H��z����
��HC�                                      Bxi�  
�          A=�Ap�����������C�G�Ap��  ��ff���\C���                                    Bxiξ            A=G�A  ��G������G�C�S3A  �   �љ��33C�Ф                                    Bxi�d  �          A?�@������
=���C�P�@��`������E�\C��{                                    Bxi�
  
P          A>�R@�=q��33�љ���C��@�=q�0  �(��7{C�L�                                    Bxi��  �          A;�
AG��ə������C��AG��X����z��$p�C��                                     Bxi	V  
Z          AD  A�R���R��p����C�ǮA�R�����{��C�,�                                    Bxi�  
�          A:�\A Q��#�
��G�����C��RA Q쾙����p���33C�'�                                    Bxi&�  �          A9p�A#
=�=q��z���Q�C�W
A#
=����ff��33C���                                    Bxi5H  
�          A5��AQ���
��=q��Q�C��AQ�>�p���{��@
�H                                    BxiC�  �          A8��A(���
��
=�	(�C��qA(�?Y����\)���@�                                      BxiR�  �          A733A
=q�7
=���
=C���A
=q>������#�\@G�                                    Bxia:  
�          A5G�A
=�2�\��p���C��A
=>����\)���?��H                                    Bxio�  �          A'\)Aff�#�
��\)��HC�L�Aff>8Q���  �(�?�                                      Bxi~�  �          A�H@�{���������C�@�{?u�����(�@�z�                                    Bxi�,  �          A  @�\)�����z��{C�f@�\)?\(�����  @��
                                    Bxi��  @          A=q@�Q��0  ����+�HC���@�Q�=u�׮�A33?�                                    Bxi�x  �          A
=@�ff����Q��FffC���@�ff��(������ӅC��                                     Bxi�  
�          A(�@�p��ƸR�)����33C��R@�p�����������C�3                                    Bxi��  �          Az���33����@��A�ffCk����33�G�?��@�  CoǮ                                    Bxi�j  
�          A�R��p��
�\@z�HA�C�\)��p��(��#�
�L��C���                                    Bxi�  T          A!��&ff��@��\B  C��׿&ff��H@\)A[
=C��)                                    Bxi�  �          A"ff�{�љ�@�
=B7��CyY��{�
=@xQ�A��C~��                                    Bxi\  �          A�R��\)���HAz�B��3Cfk���\)����@�ffBO��C}L�                                    Bxi  T          A=q@`���7
=@�  Bdp�C�b�@`�����@��RB�C�J=                                    Bxi�  T          A ��@E���G�@��BQ�C�  @E����?�  A%p�C�g�                                    Bxi.N  
�          A4��@mp����@�G�A��
C��)@mp��33>B�\?�=qC��
                                    Bxi<�  "          A0Q�@��\�Q�@�
A1G�C��\@��\�p����H�ffC���                                    BxiK�  
�          A/�@�ff�p�?��\@���C��
@�ff���#�
�XQ�C�                                      BxiZ@  
Z          A.�H@��H��?޸RA�RC�9�@��H������,  C�Q�                                    Bxih�  
�          A0z�@�G��G�?�
=A��C���@�G��  ��
�-G�C�޸                                    Bxiw�  �          A3\)@�G������3\)C��=@�G���Q���
=��z�C���                                    Bxi�2  T          A0Q�@�G��������p�C���@�G��j=q����.�C�E                                    Bxi��  T          A/
=@������H��Q����C�R@����n{�  �L�C���                                    Bxi�~  �          A{@��H�θR@��AmC���@��H���;�p���RC��)                                    Bxi�$  T          A
{@�=q��=q@�A|z�C��@�=q��논#�
��Q�C���                                    Bxi��  
�          A/33Aff���?�
=A"�HC��HAff��ff������\C�C�                                    Bxi�p  
�          A6=qA�R��33��z���p�C�o\A�R��ff������C��3                                    Bxi�  "          A5�Ap���p��*=q�Yp�C��Ap����\�����  C���                                    Bxi�  �          ADQ�A (����Fff�l  C�&fA (���p���=q��G�C��                                    Bxi�b  

          A)G�A
=���Ϳ�p��ָRC��
A
=��z��e����C�S3                                    Bxi
  
�          A$��A(�������ff�
�\C�q�A(���
=�h�����
C�&f                                    Bxi�  �          A)�A33��33�33�J{C��fA33��ff��ff��C�                                      Bxi'T  �          AF=qA)���33�6ff�U�C�
A)���  ��33���C���                                    Bxi5�  T          AK\)A4  ��p��<(��Up�C���A4  �dz���  ��G�C�1�                                    BxiD�  
Z          ALz�A6{����Fff�`z�C�S3A6{�Q���G���=qC���                                    BxiSF  �          AK33A7���G��S33�qp�C���A7��(Q���\)���C��\                                    Bxia�  
�          AJ�RA9�j�H�^�R�33C�:�A9���R������=qC�"�                                    Bxip�  
�          AD��A9녿����[���=qC��=A9녾��
�w����C�7
                                    Bxi8  
�          AAG�A3�@C33�:=q�`��As
=A3�@|(���(���G�A�
=                                    Bxi��  �          AF�\A8  ?�G���(���{@�  A8  @)���hQ�����AO�                                    Bxi��  T          AHQ�A8  ?�����
=���
@�
=A8  @7��z=q��  A`                                      Bxi�*  "          ARffAEp������������C��qAEp�?�(���z����@��                                    Bxi��  T          AO33A:{@J�H��ff���\At(�A:{@�Q��E��\��A�(�                                    Bxi�v  "          ABffA)�@;���
=�˅AxQ�A)�@���i����{A�\)                                    Bxi�  T          A?
=A�@��������A��A�@���G��r{A�\)                                    Bxi��  �          AG�A'\)@u����\�̣�A���A'\)@�33�Z�H���\A�p�                                    Bxi�h  �          AN�RA<��?W
=�������@��\A<��@*�H������p�AK�                                    Bxi  
�          AN�RA8  ?�33��p���(�AffA8  @n{���H��=qA�p�                                    Bxi�  
�          ALQ�A7�
?�{�����@�ffA7�
@^�R��(���  A���                                    Bxi Z  T          AS\)AC�
@!�������A:�HAC�
@~�R�;��MG�A��                                    Bxi/   �          AJ�RA1�@��������
=A���A1�@����+\)A�
=                                    Bxi=�  �          A@��A(z�@P  ���H��A�\)A(z�@��
�Mp��w�AƏ\                                    BxiLL  
Z          AC�A(��@��
�k���33A�Q�A(��@�ff��(���A���                                    BxiZ�  "          A:{A"�R@h��������ffA��A"�R@�p���;
=A�z�                                    Bxii�  �          AAp�A  ��\)�ٙ��
33C�G�A  ?�{������\@���                                    Bxix>  "          AB{A
=�	����H�)��C�\A
=?�33���.��@���                                    Bxi��  �          ADQ�A
�H�����	���1z�C��{A
�H@�\�\)�-��AS\)                                    Bxi��  T          AD��A�Ϳ�  ��33���C��fA��?�G����H���A�                                    Bxi�0  
�          AB�\A
=���H��Q����C���A
=?�G���  ��A                                    Bxi��  "          AF=qAz�.{�\)�,=qC��)Az�@%�G��"�
A�(�                                    Bxi�|  T          APz�A{?&ff�
=�4z�@��\A{@��H�  ��A��                                    Bxi�"  
�          AMp�A��?�p��{�/��A�
A��@�������A��
                                    Bxi��  "          AMA?��H�\)�0��A�A@������
=A�                                      Bxi�n  
�          AL  A��?�G��  �3�A4Q�A��@�����p����A���                                    Bxi�  
�          AH��A	p�@\)��3�
Aip�A	p�@������
B �
                                    Bxi
�  
Z          AC�A   @���R�<
=A
=A   @�
=������B	�\                                    Bxi`  T          AK�
A?z��G��033@i��A@w
=��
=�ffA�p�                                    Bxi(  T          AEG�A�������=q���C�z�A�?У���\)�  A                                      Bxi6�  
(          AEp�Az�(����(����C�3Az�@	�����H�=qAF�\                                    BxiER  �          AIG�Az�@J�H��(��
=A��Az�@�33�����A�z�                                    BxiS�  �          AI�A ��?����ff�ff@��HA ��@�����p����A��\                                    Bxib�  T          AMp�A!p�?�p��������A2{A!p�@�{�˅��=qA�p�                                    BxiqD  �          AK�A�\@�R��z��AJffA�\@����ȣ���Q�Aҏ\                                    Bxi�  "          AK33A�@�\����z�A<  A�@�\)��33���A�                                      Bxi��  
�          AN�RA"�H@O\)�����	�A�G�A"�H@�ff�����  A�(�                                    Bxi�6  
�          AO�A"�\@[���ff��\A�
=A"�\@�33������A�G�                                    Bxi��  T          AJ{A33@5��
=��HA���A33@����������
A��                                    Bxi��  
�          AHz�A�
@�\)��
=�=qB \)A�
@����
=��(�B$��                                    Bxi�(  T          AG�
A	�@�33�Ӆ� �B=qA	�A���y����{B,��                                    Bxi��  
�          AE�A�\@Y���љ��   A�\)A�\@����33���RA�Q�                                    Bxi�t  �          A[�
A'\)�\(������  C��fA'\)��33�  �Q�C��                                    Bxi�  
�          AX(�A�R�׮���
�	�C�s3A�R�`  ����3C�K�                                    Bxi�  @          AZ�HA(�����\���C���A(���������1��C��                                    Bxif  
�          A]�>�G��Bff���R��G�C�
=>�G���
��
�9�C�]q                                    Bxi!  "          A`  ��  �D  �X���b�RCu�\��  �"�H���G�Cq��                                    Bxi/�  T          Ah���P���L��������{C�3�P����
�$  �3�RC{}q                                    Bxi>X  �          AZ�R��{�D(��u���{Cy���{� ���   �Q�Ct�3                                    BxiL�  �          Abff�_\)�Q��G��O33C
�_\)�1�����33C|�=                                    Bxi[�  
�          AF�H�#33��\)>�=q?�  CVaH�#33�Ӆ����(��CT�                                    BxijJ  
�          A@��� �����ÿ#�
�E�CS�R� ����(��6ff�^�\CQ:�                                    Bxix�  
�          AB{�ָR�\)>�p�?�G�Cl  �ָR��R�N�R�y��Cj��                                    Bxi��  �          AA��	G��33��
=� ��C_�R�	G������Q�����C[p�                                    Bxi�<  T          AD�����  ��p�����C^���������33��z�CZ�q                                    Bxi��  
�          ADz��$(����R?E�@~�RCJQ��$(����R�G�����CJO\                                    Bxi��  
Z          AJ�R���
��(������{Ce
=���
��
=��=q��HC[��                                    Bxi�.  
�          A�G��S33�Bff�   ��HC~�=�S33��G��Z�R�h�Cw
=                                    Bxi��  �          A�p��aG��Bff�'
=��C}�
�aG�����aG��l\)CuL�                                    Bxi�z  "          A��H����T����R��RC�q����p��X(��_�C���                                    Bxi�   �          A���?0���D���*ff�#Q�C��R?0������d���u��C���                                    Bxi��  �          A��R=��<Q��5��/�C�J==�����l  {C�|)                                    Bxil  
Z          A��׿xQ��G��!��G�C���xQ��{�]��mffC��
                                    Bxi  "          A�ff����<���-p��*
=C�������  �d���|Q�C�b�                                    Bxi(�  �          A�?��
�333�2�R�2Q�C���?��
��=q�f=qffC��3                                    Bxi7^  
(          A�G�@�����9�8=qC�!H@�����p��eG��z�RC��q                                    BxiF  
�          A���@�ff�-p��6ff�1�RC���@�ff��p��g��z�
C��                                    BxiT�  T          A�=q@����/33�0���+(�C�Ǯ@�������c
=�rp�C�Y�                                    BxicP  
Z          A�p�@�p��)��.�H�*�\C���@�p���33�^�H�n�\C��                                    Bxiq�  �          A�
=@��\�4z��&{� ��C��{@��\��
=�Zff�h�C�Y�                                    Bxi��  
Z          A�33@��\�*�H�-G��'��C��=@��\��Q��]��kQ�C��R                                    Bxi�B  �          A�Q�@Ϯ�(���)p��"=qC�Ф@Ϯ��
=�Yp��b
=C��                                    Bxi��  "          A�
@��
�2=q����\C�aH@��
��Q��S\)�_�C�Z�                                    Bxi��  
�          A�\)@x���-G��5�2�\C��@x����  �fff�{�RC���                                    Bxi�4  �          A�(�@����6�\�,  �%�\C�U�@�������`Q��nQ�C���                                    Bxi��  T          A���@�  �;�
���{C���@�  ��33�Vff�_G�C��\                                    Bxi؀  "          A�p�@��
�;��=q��C��=@��
��z��T���]p�C��                                    Bxi�&  �          A�33@��
�9G�� (��ffC���@��
��ff�U��_�RC�=q                                    Bxi��  �          A��H@����1�!G����C��q@�����  �T(��]�C��                                    Bxir  �          A�G�@�  �
=�/��+{C��
@�  ��33�[\)�g�HC���                                    Bxi  	�          A�ff@����ff�2�H�1�C��H@�������Y��i=qC���                                    Bxi!�  �          A|��@�\)�+\)�+
=�*�\C�Z�@�\)��ff�[
=�p��C�g�                                    Bxi0d  
�          A|��@��.{�&{�$��C��f@���
=�W33�j�C�q�                                    Bxi?
  
�          Aw�A�
��  ���!�
C�3A�
�l(��>�H�KffC�ٚ                                    BxiM�  �          Aup�@��ə��:�R�H  C���@����R�Q�m�\C���                                    Bxi\V  
�          A{�
A3���(������p�C�xRA3������H��C�e                                    Bxij�  
(          A|  A+\)���� \)C�� A+\)����(z��(33C�>�                                    Bxiy�  
Z          A{�A#���R����Q�C���A#��y���4Q��7Q�C���                                    Bxi�H  �          A|��A�H���)���(�C��A�H���@���G(�C���                                    Bxi��  �          Ax(�Aff�����$���&��C�9�Aff�%�=���Gp�C�|)                                    Bxi��  
�          Ax��A�\���
�(����C��HA�\��Q��8  �>�
C���                                    Bxi�:  �          Ax(�A�\���� ���!G�C�aHA�\�Z=q�=p��G(�C�
=                                    Bxi��  
�          AzffA�H��Q��Q����C�C�A�H���\�<���C�RC�L�                                    Bxiц  �          Aw�
A���G���R��C��A����=q�6�\�=�RC�N                                    Bxi�,  
(          A|Q�A\)� z��
ff�Q�C�y�A\)�Ӆ�7��9�C�g�                                    Bxi��  
�          Ay�Ap��p�������C�U�Ap���\)�3��8Q�C�g�                                    Bxi�x  T          Ay�A��\)�
=���C�\A���33�2�\�6�C�~�                                    Bxi  j          AuA�
�������ffC�7
A�
����/\)�6Q�C���                                    Bxi�  >          Av�RA����R����(�C�!HA������2�\�:  C�K�                                    Bxi)j  
Z          Ap(�A{��z��
=q�p�C��=A{��z��*�R�4�HC��                                     Bxi8  �          Aj�\Az��	���������C�/\Az���33�"{�.z�C�{                                    BxiF�            Aj�HA{� ���  �
=qC��A{��p��&�R�7{C���                                    BxiU\  "          Ao33A:�H�fff�Q����C�o\A:�H�!G����(�C�t{                                    Bxid  �          A�AH���7����=qC���AH��>��������H?�G�                                    Bxir�  T          A�ffATz��
=���	C�~�ATz�?}p���\��@���                                    Bxi�N  �          A}p�A\(��8Q����\��33C��qA\(�@
=��
=��G�A�                                    Bxi��  
�          AuAR�R����������C�g�AR�R?�����  ��R@�
=                                    Bxi��  
�          At��A.�R����p��(�C�)A.�R�У��"�\�*p�C��H                                    Bxi�@  T          AvffAff��� z��#  C��Aff�e��<���H�
C�Q�                                    Bxi��  "          Aip�@�����z���C�P�@�������7\)�NG�C���                                    Bxiʌ  �          Ah��@����ff�Q�C�33@�����9��Sz�C��q                                    Bxi�2  T          A_\)@��=q���p�C�<)@����
�%G��?�C���                                    Bxi��  T          AiA�H�G���ff����C�� A�H��z��$z��2�
C��                                    Bxi�~  
�          AG�@�
=�  �ʏ\��  C�O\@�
=��  �33�7G�C���                                    Bxi$  �          AI��@�Q��Q��θR��G�C�` @�Q��׮����8��C��f                                    Bxi�  T          AH��@�\)����G����C�q�@�\)��  ���6��C�b�                                    Bxi"p  �          AH  @��R������R��\)C�q�@��R��  ��H�)�RC�q                                    Bxi1  
Z          AH��@��R�
=������(�C���@��R������,{C�H�                                    Bxi?�  �          AF�H@��H�33����33C���@��H��33�	���0Q�C�xR                                    BxiNb  �          AK��u�I��333�K�C�s3�u�;\)������C�h�                                    Bxi]  T          AJ{?=p��G33��33���C��3?=p��6{�����G�C��)                                    Bxik�  
�          AM��>��K�
��33��p�C�K�>��;�
���\��G�C�Q�                                    BxizT  T          AM@*�H�E����
�%��C��@*�H�0(���������C�Ф                                    Bxi��  �          AK�
@��$����=q��  C�Z�@��{�(��"\)C�B�                                    Bxi��  �          AL��@�p��0(����H���C��@�p��
=����33C�
                                    Bxi�F  =          AJ�\@��R�-��{��  C�&f@��R�����\�  C�U�                                    Bxi��  �          AE�@�R�+\)������\)C��@�R�
=��z��{C��                                    BxiÒ  �          AC33��
=�9�=L��>��C�G���
=�0���b�\��  C��                                    Bxi�8            AD��?O\)�@�׿�G���(�C��\?O\)�/�
������z�C��                                    Bxi��  �          AF�H?�(��;�
�0���QG�C��?�(��$���Ǯ���C�j=                                    Bxi�  "          AE��@
�H�:�R�8���Z�RC�B�@
�H�#��ʏ\���
C��q                                    Bxi�*  �          AFff@J=q�1p��������C��@J=q��\�����\C��H                                    Bxi�  T          AE��@r�\�-��������\C��=@r�\��\��(���HC���                                    Bxiv  
�          AG
=@��\�'����R����C��H@��\�{��\�$��C��
                                    Bxi*  
�          AI��@p���&=q�����C��{@p���p����2�RC�y�                                    Bxi8�  �          AH��@���'\)������
=C�Ǯ@����H����Q�C�5�                                    BxiGh  �          AI�@��H������R���HC�N@��H��33��\�(�C��                                    BxiV  
�          AK�
@�����������\)C��q@�����p���C�5�                                    Bxid�  �          AH��@ƸR�(�������\C���@ƸR��ff�
{�/��C���                                    BxisZ  
�          AIp�@ҏ\�(���\)��ffC��)@ҏ\�����
�0z�C�s3                                    Bxi�             AK�
A{��  ��  �p�C�
A{������ff��C�H�                                    Bxi��  �          AM��A)�qG���p���C�<)A)�޸R����z�C�Z�                                    Bxi�L  �          AN�RA(���Z=q�����C��A(�׿���������HC�l�                                    Bxi��  �          AO
=A3��:=q��\)��z�C���A3�������H���HC�J=                                    Bxi��  T          AP  A6�H�%����̏\C��HA6�H�J=q��(�����C�                                    Bxi�>  T          AN{A4���@�����\��Q�C���A4�׿��
�����Q�C���                                    Bxi��  T          AM��A8(��?\)��  ��(�C���A8(���Q���p���ffC�p�                                    Bxi�  "          AMA4���)���������C�eA4�׿aG���33��ffC��f                                    Bxi�0  T          ALQ�A5���=p�������p�C��{A5����\)�����\)C��\                                    Bxi�  "          ALQ�A6�R�6ff���
��C�  A6�R���
�����Q�C���                                    Bxi|  T          AK�A3
=�������хC�nA3
=��ff��Q���C��R                                    Bxi#"  
�          AJ�HA0���G�������
=C�5�A0�;���{���C���                                    Bxi1�  
�          AK\)A1���������=qC�/\A1�������33C��                                     Bxi@n  
�          AIA%p��5����
����C�Y�A%p��B�\����
�C���                                    BxiO  T          AG�
A�`����
=��\C�/\A���
�
ff�0p�C��\                                    Bxi]�  
�          AHz�A
=�����{��RC���A
=�J�H���R���C�:�                                    Bxil`  �          AMG�AQ��ٙ�����љ�C�� AQ���33��z����C�/\                                    Bxi{  �          AJ=q@����陚?��A
=C�k�@������&ff�^{C�*=                                    Bxi��            AN=q@����
=@�z�A���C��@�����@�A  C�H�                                    Bxi�R  
n          AN�HAQ���?+�@@  C���AQ���׿��H�C��H                                    Bxi��  �          AL��A(��  �������C��\A(��33�\)��33C�f                                    Bxi��            AK�A��z��Q��-G�C��\A���
=�������C�ff                                    Bxi�D  
�          AG
=@����\��33��  C��q@����\)��\)��C���                                    Bxi��            AFffA   �Ǯ��z��ָRC��A   ��  �ָR��C�O\                                    Bxi�  �          AC�@ۅ��z������*�C��
@ۅ�)������Kz�C�k�                                    Bxi�6  
�          AG\)@�33��p����/33C�
@�33��\�z��L\)C�Y�                                    Bxi��  "          AL��A33�ƸR��z���C���A33�u��.��C��                                    Bxi�  
q          AY�A
{���H�(��,ffC�Z�A
{�   �$  �D�RC�xR                                    Bxi(  Q          APz�A���/\)��R�2��C�b�A�����
���=  C���                                    Bxi*�  
�          AO�A z��E�ff�A(�C�}qA z�aG��"{�N\)C�8R                                    Bxi9t  
�          AP��A���R����C�
C�k�A�?#�
� z��J(�@�                                    BxiH  T          AP��A�\�33���(
=C�  A�\>���Q��/�?�=q                                    BxiV�  �          A\  A녿���.�R�Q(�C�NA�@�\�,z��Mz�A[
=                                    Bxief  
�          AUp�A��*=q����233C���A�=�\)��R�;p�>�
=                                    Bxit  
�          AZ�RA
=������1�HC��A
=?�Q����/�A�\                                    Bxi��  �          AU�Aff���Q��)G�C���Aff@z�����$
=A=G�                                    Bxi�X  "          Ab�RA"ff�ٙ���R�,�HC�@ A"ff?�=q�(��.�H@��                                    Bxi��  T          AfffA(��B�\�&�R�8�
C�!HA(��#�
�-���C{C��
                                    Bxi��  "          AU��A
=�j�H�ff�'�C��A
=������7�C��                                    Bxi�J  T          AT��A��������
�*ffC��A�׿������>�
C��{                                    Bxi��  
�          AT��A�\�N{�Q��1�\C�{A�\��\����>�RC�Y�                                    Bxiږ  
�          AN{A(��e��\)�0��C���A(��s33����A�C��\                                    Bxi�<  	�          AM��A z��_\)���;33C�@ A z�G���R�K��C�=q                                    Bxi��  	�          AP(�@�
=���R�\)�:C���@�
=��
=�%��RQ�C��)                                    Bxi�  
�          APQ�@����tz��ff�F{C��\@��ÿp���)��Y33C�n                                    Bxi.  �          AQG�@�=q������
�A
=C�H�@�=q��
=�(���W\)C��=                                    Bxi#�  �          AQ�@�����=q��H�?G�C�.@��׿�G��&�H�SC�h�                                    Bxi2z  �          AU�A������
=�A  C�"�A��>���#��HQ�@0                                      BxiA   "          AUG�A��S33��
�=�C�<)A녿�\�$Q��K�C�@                                     BxiO�            AD��@��H�ҏ\�	p��3\)C�*=@��H�\)� ���^�RC�=q                                    Bxi^l  �          AF�\@�����H���\�
=C���@����{�\)�I
=C���                                    Bxim  	�          AB�R@�p��z���Q��p�C��=@�p���{�(��3G�C�ٚ                                    Bxi{�  
�          AH��@���{�ff�(p�C��R@��������R�HC�Ff                                    Bxi�^  
�          AL(�@����Q��ff�1
=C�E@���w
=�%G��XffC�z�                                    Bxi�  
�          AIG�@ָR���
�	�.  C��@ָR�c�
�33�R�\C�H                                    Bxi��            AK�@��H��{��)�HC��\@��H�������C(�C��
                                    Bxi�P  �          AK�@XQ�� ����R�6�C�` @XQ���33�,(��n33C�                                      Bxi��  
�          AN=q?�33�#����H��C�33?�33����� ���O�C�.                                    BxiӜ  
�          AO
=�J=q�\)��\��HC����J=q����(z��Z�HC���                                    Bxi�B  �          AL��?\)���-��ip�C�aH?\)�aG��D(��{C�|)                                    Bxi��  T          AK�?�ff��(��2�H�z�RC�� ?�ff���Dz���C�
=                                    Bxi��            AI�@�33���
�33�F{C�h�@�33�w
=�-��tG�C��                                    Bxi4  "          AK
=@Ϯ��Q�����Q{C�"�@Ϯ���R�)��h  C��3                                    Bxi�  |          AMG�@�ff��ff�=q�$33C�Ǯ@�ff�AG�����A�C���                                    Bxi+�  
�          AN{@�����
=���%�C���@����AG����B�RC���                                    Bxi:&  
�          AN=q@�p���
=���%33C��R@�p��\)�=q�IQ�C�q�                                    BxiH�  T          AO\)@��
����G����C��\@��
��\)����DQ�C�{                                    BxiWr  r          AO33@׮�������RC��@׮��{�{�0�C��
                                    Bxif  T          AL��@�\)�&=q���
��=qC�^�@�\)�	��p����C��f                                    Bxit�  
�          Aap�@�=q�����33�ffC��@�=q��������/�
C�
                                    Bxi�d  
�          ArffA���Q��G���\C��A�������3�
�@=qC���                                    Bxi�
  "          Aq�A33��33�(�� ��C��\A33�����7��E�C��                                    Bxi��  �          Ak�
Ap�����33�$��C�b�Ap������4Q��G��C��                                    Bxi�V  
P          Ak\)A=q��(���R�$Q�C���A=q�xQ��0���C
=C�4{                                    Bxi��  �          Ak\)A����
=����#z�C��RA���QG��-��>(�C�`                                     Bxi̢  �          As�
A'\)���H�
=��C�xRA'\)�(���,  �3�\C��=                                    Bxi�H  �          Aj�RA�H��(��33�(�\C���A�H�x���1p��HQ�C��)                                    Bxi��  �          Ag�
@�z���33�&{�733C��
@�z��k��;��X�\C�                                      Bxi��  �          Al  AG���\)�"{�-�\C��fAG��x���8Q��M��C�}q                                    Bxi:  "          Al��A
{����&�\�2�HC��)A
{�G
=�9���N�C��                                    Bxi�  �          Ah  A����$Q��4C��A��5�6=q�Op�C��q                                    Bxi$�  "          Aa�AG��u�-G��IC�O\AG��k��733�Zp�C���                                    Bxi3,  |          Aa�A��g
=�-��H(�C�FfA��5�6{�W  C��\                                    BxiA�  �          Aa�A33�E�#
=�9��C��
A33�\�*=q�D�\C��{                                    BxiPx  �          Ai��A���^�R�((��7��C���A�׿.{�0���D�C��R                                    Bxi_  �          A`z�A)�� ���33�  C���A)�>�{��\�#��?�{                                    Bxim�  �          Al(�A&�H�AG���
�*ffC��\A&�H��p��&�H�3�
C���                                    Bxi|j  �          AnffA*�\�L���ff�&��C���A*�\�z��&=q�0�HC�q�                                    Bxi�  �          As33A.�H�fff����!�
C���A.�H�}p��&�\�-�HC�j=                                    Bxi��  "          Az�RA>�H���
�=q�
ffC��A>�H�p��p����C�.                                    Bxi�\  �          AX  A����Q�����C���A���.�R���033C���                                    Bxi�  �          AX��A{������\��RC��3A{�#�
����)�C���                                    BxiŨ  T          AW�
A
=���=q�  C�S3A
=��
=��H�+�\C�]q                                    Bxi�N  �          A[�A �������\)�33C�.A ���(�����%\)C�ٚ                                    Bxi��  
�          AUG�A#
=�|(���(��C�l�A#
=��33�{���C���                                    Bxi�  
�          AO�
A!G��G
=��=q���C�l�A!G���{���\C�ٚ                                    Bxi  @  
�          Ac
=A0�׿�p��ff��C���A0��>��R��
=?У�                                    Bxi �  �          As\)A6�\�fff� (��$�C�A6�\?����R�"��A�\                                    Bxi �  T          Ap��A2ff�����\)�'
=C��HA2ff@����\�!�AA                                    Bxi ,2  �          A`��A,��?����H���@�G�A,��@U����{A�G�                                    Bxi :�  �          Ay�A;\)?��\�#33�#�R@�  A;\)@a�����RA�(�                                    Bxi I~  "          Ab�HA/\)?\�����@�(�A/\)@o\)�\)��A��\                                    Bxi X$  �          AG33A�@e����
�ffA���A�@�\)��Q���Q�A�\)                                    Bxi f�  �          AEA�@r�\����ffA�
=A�@�������(�A�33                                    Bxi up  �          AD��A��@:=q��(����A��A��@�������  A�ff                                    Bxi �  T          AA�AQ�?�z������RA"{AQ�@[��أ��	��A��\                                    Bxi ��  
�          A>{A{@n{�����"(�A�z�A{@���Ϯ�
=B
=                                    Bxi �b  "          A?33@��@�����\�ffB(�@��@�R������B2p�                                    Bxi �  �          AAG�@���@�G���R���Bp�@���@�z���(����
B-��                                    Bxi ��  "          A?\)@��H@�{��(���z�B5�\@��HA	p��hQ����\BF                                      Bxi �T  T          A=p�@�ff@�����\�ܣ�BK(�@�ffA��N�R��z�BY�                                    Bxi ��  @          A>�R@J�H@��333�B
G�@J�H@���%G��r��Bc�                                    Bxi �  �          A=p�@E@x���*=q��BN33@E@��
���SB�G�                                    Bxi �F  "          A9G��\@{��+
=p�B�#׾\@�p���\�^��B��
                                    Bxi!�  T          A6ff@G
=@�=q�
=�cp�Bn@G
=@�R����0�HB��R                                    Bxi!�  �          A9�@�33@�����S\)BX�@�33@�\)��=q�#p�Bx{                                    Bxi!%8  T          A4(�@�
=@�=q����[  B6G�@�
=@��
��p��/=qB_�
                                    Bxi!3�  "          A8(�?c�
@��H�(��k33B���?c�
A z�� ���3��B�L�                                    Bxi!B�  
�          AqG�A
=@z��/\)�<A<  A
=@���"=q�*�HA���                                    Bxi!Q*  �          At��A+�?�Q��,���3�
@�z�A+�@�  �"{�&�A�                                    Bxi!_�  
�          AvffA,��?aG��/\)�5ff@�A,��@`  �&�R�*Q�A�                                    Bxi!nv  �          Af�\A!�?s33�#�
�5p�@��A!�@XQ��\)�)A��\                                    Bxi!}  T          A_\)A&=q?�z���
�%z�@�  A&=q@e�
{�z�A��\                                    Bxi!��  "          AX��A5p���z���\)���C�:�A5p��8Q���\)�
=C��                                    Bxi!�h  �          AT  A���{�߮����C��3A��fff�p��p�C���                                    Bxi!�  �          AO\)A�����H��{��{C�&fA�����
��p���HC��                                    Bxi!��  
�          AV�HA�����\)���RC��qA���{������C�                                      Bxi!�Z  �          AVffA
=��������
C���A
=��p���z���(�C�(�                                    Bxi!�   �          AW33A�����e��w�C���A��
=��ff����C�Ф                                    Bxi!�  T          AT(�A����\������C��3A��ʏ\��=q�(�C��H                                    Bxi!�L  
�          AM��A  �\)������
=C���A  ��z��θR���C��                                    Bxi" �  "          AO\)A\)��p���G���C�Y�A\)������"�
C���                                    Bxi"�  
�          A[�A�
��33����Q�C��{A�
���
�R�\)C�޸                                    Bxi">  �          Ae�A�
��=q��	�
C���A�
��
=�{�#\)C���                                    Bxi",�  "          Ag
=A#���������z�C��=A#��Vff���'G�C���                                    Bxi";�  "          Ae�A ����\)��\��C��qA ���(��33�0Q�C�ٚ                                    Bxi"J0  "          AaA�H����  ���C�FfA�H�z��(��/��C�                                      Bxi"X�  
�          Ab{A{���
�G�� =qC��A{�
=��133C��R                                    Bxi"g|  T          Aa�A!���=q�  �p�C�˅A!�����G��+33C�U�                                    Bxi"v"  �          A\��A"=q��������	\)C���A"=q�L(��(��
=C�C�                                    Bxi"��  T          A[�AQ����
���� C���AQ��
�H�{�2\)C���                                    Bxi"�n  �          A[
=Ap����\����
C��Ap��?\)�z��*�
C�W
                                    Bxi"�  �          AW�
AG���z��
�R� (�C��qAG��.{���5  C���                                    Bxi"��  	�          AW�A��������z�C�*=A��5���-�C��3                                    Bxi"�`  �          AX��A"�H�������H��\C�޸A"�H�J�H�����C�XR                                    Bxi"�  
�          AX  A�����R���
��C��{A���N�R���%Q�C���                                    Bxi"ܬ  �          AX��A\)�������R�
=C���A\)�$z�����"(�C���                                    Bxi"�R  "          AY�A(�������R�
�\C�� A(��^{�  � ��C�7
                                    Bxi"��  
�          A\Q�Az�������H�
=C��Az��G
=�ff�&�C�/\                                    Bxi#�  �          A_�
A"ff���H�=q��\C���A"ff�Dz�����"�\C��
                                    Bxi#D  T          A\(�A"�\���R��H���C�#�A"�\�����
�"�HC�9�                                    Bxi#%�  
�          A]�A ����ff���p�C�xRA ���)����H�%��C��q                                    Bxi#4�  
�          A^�HAz���=q�z��"  C�1�Az��
=�p��433C��                                    Bxi#C6  T          A\��A=q��33��R�&=qC��A=q�&ff� ���:Q�C�                                    Bxi#Q�  �          A]��A!�������p���RC�xRA!���L���ff� {C�7
                                    Bxi#`�  �          Aa�A#\)��Q��33�z�C��HA#\)�,(����%�RC��H                                    Bxi#o(  �          A\(�A�R��Q��
�H�z�C��HA�R������.Q�C�                                    Bxi#}�  �          A\z�A\)��
=�p��%
=C��qA\)�G����5��C��)                                    Bxi#�t  �          A[�A��{��	p���
C�@ A������Q��)��C��)                                    Bxi#�  
�          A_\)A&=q�dz��	���(�C��fA&=q��  �
=�$�RC��H                                    Bxi#��  T          A`Q�A)G��Vff�	���RC�8RA)G���ff��"{C��                                     Bxi#�f  �          Ac33A+��`  �	���\C��{A+���Q��33�!p�C�.                                    Bxi#�  
�          Aa�A*=q�XQ��Q��ffC�1�A*=q������ �
C�g�                                    Bxi#ղ  �          A^ffA*{�s33�{�
=C�,�A*{���z���\C��                                    Bxi#�X  "          Ab�RA/33�b�\�  ��C��A/33�����p��z�C��                                    Bxi#��  "          Ac33A/��AG��(��(�C�O\A/���  �����C�c�                                    Bxi$�  T          AaG�A-���>{����HC�\)A-���u��R�=qC�w
                                    Bxi$J  �          A^�RA(z��l�������\C�Q�A(z��(���H���C�^�                                    Bxi$�  �          AQG�A(�Ϳ˅��Q��z�C���A(��>����p��{?Tz�                                    Bxi$-�  �          Ag�A8�׿�z��=q��\C�K�A8��=#�
�	�����>L��                                    Bxi$<<  �          Ai��A;����ff�C�޸A;��#�
�
�\��
C���                                    Bxi$J�  �          Ah��A5������z��p�C��A5����������C�                                      Bxi$Y�  T          Ai�A733�+��	��C�j=A733�(���  �z�C�XR                                    Bxi$h.  �          Ah  A2{�Q����p�C��3A2{�L������C��q                                    Bxi$v�  �          Ah��A1G���p��Q��\)C��\A1G�>�����R�!z�@                                    Bxi$�z  �          Aj{A4Q��p���\�{C�� A4Q�>�p�����33?�33                                    Bxi$�   �          Ai�A2ff�����H��C�k�A2ff���
�
=� ��C��                                    Bxi$��  �          Ae��A1G���z���\���C��)A1G�>Ǯ������?��R                                    Bxi$�l  �          Aep�A4Q쿾�R�\)��C�:�A4Q�?�\�G��33@$z�                                    Bxi$�  
�          AfffA7�����	G��\)C���A7�?��
�R�=q@5                                    Bxi$θ  T          AdQ�A3\)���H�33���C���A3\)?E��  ���@|��                                    Bxi$�^  T          Ad  A/
=��
=��\�\)C��A/
=>�Q�����z�?�z�                                    Bxi$�  "          Ad(�A/\)������p�C��
A/\)?:�H�����
@tz�                                    Bxi$��  "          Ad  A(��>�z��G��)  ?�=qA(��@��Q��"ffAO�                                    Bxi%	P  T          Ac�
A)G�>�33�(��'�?��A)G�@{�
=� �
AQ�                                    Bxi%�  	�          Ac�A*=q>aG���\�%��?�z�A*=q@�
�{��HADQ�                                    Bxi%&�  
�          Aa�A)p����R�
=�#�RC�(�A)p�?�  �Q�� =qA{                                    Bxi%5B  "          Ac�A,�Ϳ
=��H�!\)C�p�A,��?�p��G��=q@���                                    Bxi%C�  T          Aap�A)G�=L�����$=q>�z�A)G�@
=���  A4Q�                                    Bxi%R�  �          A`��A'�
?#�
���&=q@^{A'�
@,���
=�(�Af�R                                    Bxi%a4  
�          A^�RA*�R��(��(��C�� A*�R?B�\�����
@��H                                    Bxi%o�  T          AW�
A"{�P  ���C��A"{�����
�"�C�:�                                    Bxi%~�  �          AW�A%��  ��  �C�@ A%�1������C�|)                                    Bxi%�&  
�          AX��A!G��X������C���A!G����H��$�C�޸                                    Bxi%��  
�          AZffA$(��33�
�R�  C���A$(����
�\)�$z�C��                                    Bxi%�r  
�          AZ�RA$Q��P  ��\��C�4{A$Q쿧���R�#
=C�^�                                    Bxi%�  
�          A_�A"�\����\�)p�C�� A"�\>����p��-Q�?�                                    Bxi%Ǿ  
(          A_�A!����Q���
�+Q�C�=qA!��>�ff�{�.p�@#�
                                    Bxi%�d  T          A]p�A&=q�z��Q���C�Z�A&=q���
�Q��#��C�Ǯ                                    Bxi%�
  
�          A`  A(  ��=q��
�!
=C���A(  >�ff��#@��                                    Bxi%�  
�          A_�
A(Q쿷
=�\)� ��C�"�A(Q�?
=����"�@Mp�                                    Bxi&V            A\��A)G��<���=q�(�C�5�A)G���=q�	p���RC�
                                    Bxi&�  <          AX��A"ff�!G��(��33C�A"ff�#�
�	�!
=C�4{                                    Bxi&�  
�          A]p�A)����33�  �=qC��=A)��?B�\����33@��H                                    Bxi&.H  
�          A�33Aff���
�H���N�
C��\Aff@/\)�C��G�Ax                                      Bxi&<�  �          A�z�@���@ƸR����u�B'�H@���A���r�\�N�RB[G�                                    Bxi&K�  �          A�Q�AG�@������H�`G�AƸRAG�@���k\)�E
=B\)                                    Bxi&Z:  T          A��A������v�\�e{C���A��@Q��t(��a�AX��                                    Bxi&h�  T          A�{A�R���\�d���U��C��\A�R��=q�nff�cffC�Ф                                    Bxi&w�  �          A���A{@����d  �M�B)z�A{A)���C��(�BN��                                    Bxi&�,  �          A��A+33@�33�w��T(�A�(�A+33@�z��ap��:��B�                                    Bxi&��  �          A�G�A=@%�|Q��Q��AD��A=@���l���@p�A�Q�                                    Bxi&�x  �          A���A&�R?�������j\)@�G�A&�R@�(�����[\)Aљ�                                    Bxi&�  �          A�z�A���Q���Q��p  C���A�@\�������hQ�A��R                                    Bxi&��  
�          A�33A ���w
=�����d
=C��HA �ͽ���(��mC���                                    Bxi&�j  T          A�
=A��z=q���\�r
=C�\A��u��{�}  C��=                                    Bxi&�  �          A�p�A���(���{�s�C�Q�A���  ��  �=qC�0�                                    Bxi&�  �          A��RA����=q���\�b(�C�� A���(����z��yp�C�j=                                    Bxi&�\  �          A�33@�Q��6�H�S��5�C�XR@�Q��  �vff�`\)C��{                                    Bxi'
  |          A�z�A33��\�_��DffC�P�A33��(��y���e�
C���                                    Bxi'�  �          A��HA33��z��eG��Y�C���A33�'
=�tQ��o�C�n                                    Bxi''N  �          A��
A{?�z��q�g�H@��A{@�{�g�
�Y�\A�z�                                    Bxi'5�  �          A��\A)p�@��c��HQ�A�
=A)p�A�\�K
=�,�RB��                                    Bxi'D�  �          A���A�@+��u�ep�A|��A�@�
=�fff�P�\B                                      Bxi'S@  T          A��@�
=?�ff��(���A��@�
=@����vff�}B=q                                    Bxi'a�  �          A�=qAQ�@����hQ��O=qA��AQ�A
=�M��0B)                                    Bxi'p�  
�          A��A,��@�  �_33�G�A���A,��@�{�J{�/�\B
{                                    Bxi'2  �          A�(�@��@5���z�A�  @��@�(��x���f{B ��                                    Bxi'��  
�          A���@�33@��R��=q�x��A�@�33@���n�\�X\)B<Q�                                    Bxi'�~  �          A���@����b�\����
C��{@���=�Q����\\?fff                                    Bxi'�$  	�          A��HAff�У��:=q�:33C���Aff�s�
�Lz��Sp�C�e                                    Bxi'��  "          A��HA*ff���b{�R{C��qA*ff?�{�c�
�T=q@��R                                    Bxi'�p  	�          A���AK\)@333�3��#�AF�RAK\)@�{�%p��  A��                                    Bxi'�  �          A�p�AR�\@�Q��!��\)A��AR�\@��
�  ��z�AǙ�                                    Bxi'�  4          A�\)AmG���=q��R�Ə\C�t{AmG���G��
�\��\)C�|)                                    Bxi'�b  	�          A�(�A4  ��Q��$���z�C��HA4  ��33�8z��.C�p�                                    Bxi(  �          A���@�\)�4Q��#
=�33C���@�\)�Q��Fff�?��C�AH                                    Bxi(�  "          A�(�@�  �]��=q��p�C���@�  �;33�3��'G�C��f                                    Bxi( T  |          A��
Aff?G��a��fG�@�  Aff@u��X���YA�=q                                    Bxi(.�  @          A�G�?��&{�A��E
=C���?����`���w\)C�,�                                    Bxi(=�  
�          A��@������w��v�C��@��+���{(�C�XR                                    Bxi(LF  "          A��HA\)���p(��c�RC�˅A\)?fff�r�H�g@�{                                    Bxi(Z�  
Z          A���@��R���\��33�~33C�p�@��R��{���
.C�P�                                    Bxi(i�  �          A�ff@�����\)��G��C�˅@��Ϳ.{���
aHC��                                    Bxi(x8  "          A��H@����\)����=qC���@��;�Q����C���                                    Bxi(��  T          A��\@�  �,  �`(��I�
C���@�  ��{��(��wffC��                                     Bxi(��  
�          A�ff@�33��p��o\)�_�C�\)@�33�_\)�����|��C�K�                                    Bxi(�*            A���@K���Q���33C�@K��7���{�qC���                                    Bxi(��            A�(�@XQ������~=q��C��H@XQ�������C�9�                                    Bxi(�v  
�          A��HA
{���R�l���mz�C��A
{?�\�l(��l�A9p�                                    Bxi(�  
�          A�{@ٙ��=p��p��(�C��@ٙ�@#�
�m�A�\)                                    Bxi(��  
�          A��@l(�@����i���u�HBy
=@l(�A ���K��G��B���                                    Bxi(�h  �          A��@�HAR{����B�#�@�HAo�
���
���HB��3                                    Bxi(�            A�ff?���@q��k�
��B��f?���@���X���y�B�ff                                    Bxi)
�  
�          A��@���@ə��a���wp�Bb��@���A��F=q�K
=B���                                    Bxi)Z  
�          A33@љ�@b�\�V�H�t  A�33@љ�@�z��EG��UB1(�                                    Bxi)(   
�          Aqp�@�ff� z�����-z�C�˅@�ff��
=�!�X�RC��                                    Bxi)6�  �          Aw�@�33�4z��T����C��@�33>���Yp��
?�
=                                    Bxi)EL  �          A�@���@�(��k�
�(�B��@���@�  �V�H�[33BQ\)                                    Bxi)S�  |          A�Q�@���@#�
�pQ��)A��@���@����a���qB9G�                                    Bxi)b�  �          A��R@���@�ff�[33�kG�B��@���@��\�D���Hp�BF�                                    Bxi)q>  
x          A~�H@��r�\�Y� C���@��:�H�aQ�C��                                    Bxi)�  
�          Aep�@��
@$z��?�
�~Q�A�(�@��
@�(��2=q�a33B.�\                                    Bxi)��  T          Ab�R@�33=�G��@���j�?\(�@�33@0  �<  �a  A�                                    Bxi)�0  
Z          Ak33@�33@���LQ��rz�A��\@�33@�{�?��[Q�BG�                                    Bxi)��  �          Aj{@���@N�R�S33�
Bp�@���@�G��B�H�f��BN�
                                    Bxi)�|  6          A�{@333@�\)�}��)BwQ�@333A	��d���g  B�Ǯ                                    Bxi)�"  h          A��
?�@Dz��v{u�Bh  ?�@��
�eG�8RB�k�                                    Bxi)��  
<          A\z�@�{�Y���љ��2C��3@�{�����z��H  C���                                    Bxi)�n  
�          AN{@�33�&�\�_\)���HC�/\@�33�ff����33C��H                                    Bxi)�  
�          AU��A\)������
=��C��3A\)��{��\�,  C�(�                                    Bxi*�  
�          A[\)@љ��L(��9G��gQ�C��@љ����@  �u{C��                                    Bxi*`            A[�@�>�33�8  �g�@2�\@�@7
=�2=q�\��A��                                    Bxi*!  �          A\Q�@��
�33�%��Qp�C��q@��
<��
�)��X�
>8Q�                                    Bxi*/�  
�          A[�@����33�:ff�z��C�}q@��?�
=�9��yp�AZ{                                    Bxi*>R  T          A`��A\)��{���G�C��
A\)��
=��
�1��C�l�                                    Bxi*L�  
F          AU��@�����G��D(�
=C�]q@���@���@��u�Aə�                                    Bxi*[�  �          AW�@�\)=����C
=��?�  @�\)@1G��=��z��A���                                    Bxi*jD            A]G�@�ff�u�6=q�iffC��@�ff?����5��g=qAK
=                                    Bxi*x�  	�          AZ�RA#�������z�����C�A#��y����{��
C���                                    Bxi*��  
�          A[33A$���|(���  ��RC��fA$���33��ff�z�C���                                    Bxi*�6  
          A[�@�  ��(��-��X�C���@�  ?����-��XA��                                    Bxi*��  6          AV�\AG��  �%p��K�
C�5�AG�=��
�)p��R�\?z�                                    Bxi*��  r          AZ�H@�G��L���<���v�HC�e@�G�?����:�R�r�
A�33                                    Bxi*�(  
�          AU��@������>�RC���@��?�{�=��u�A�
=                                    Bxi*��  
n          AJ�R@�=q@��G��_  B=��@�=q@�Q���(��6z�Bd                                      Bxi*�t  "          AQG�@P����=q�7�#�C�U�@P�׿�Q��AG�.C��                                    Bxi*�  h          AN�H@���@�{����K��B#��@���@У����&�BI�                                    Bxi*��            AT��@љ�@33�4(��i\)A���@љ�@�\)�'��QffBQ�                                    Bxi+f            AT(�@�z�@K��4���mz�A��@�z�@���%G��O{B.Q�                                    Bxi+  �          AL��@�ff@��\���IffB��@�ff@ٙ���H�&p�B>�                                    Bxi+(�  r          A+�
@�A�\��
=�.{By=q@�A�=�Q�>��HB{ff                                    Bxi+7X  
�          A<(�@��RA\)��  ��Bj�@��RA���8���f=qBt
=                                    Bxi+E�  
x          A)@;�A���U�����B�B�@;�A��p���
=B�
=                                    Bxi+T�  J          AAG�@�Q�A  ������p�Bez�@�Q�Az��=q�?�
Bm��                                    Bxi+cJ  T          AI�@�G�A����
=��  BZ��@�G�A$�Ϳ��R�ffBbp�                                    Bxi+q�  T          AD  @�33A	p��r�\��Q�BE��@�33A�����
{BM�R                                    Bxi+��  6          A*�R@�G�@�녾��333B@�R@�G�@�?�z�@أ�B?�                                    Bxi+�<  �          A8z�@�(�A�@�A(Q�BcQ�@�(�A
�R@��A�\)B[{                                    Bxi+��  �          A.�H@�(����p��W�\C�y�@�(�?�ff�   �T33AF�R                                    Bxi+��  	�          A7�@��H@G������RA��
@��H@��
��{��A�{                                    Bxi+�.  
Z          A>�RA33@�33��G���B	(�A33@޸R�C�
�qB�                                    Bxi+��  �          A4��@�Q�@�ff�p����33BO�@�Q�A	p�����!BW��                                    Bxi+�z            A{@��@e���=q��Bp�@��@���y����B-�
                                    Bxi+�   
�          A1G�@��@vff��{�&  A�  @��@�33�����

=B33                                    Bxi+��  
�          A1G�@��@�p���p���=qB�
@��@�  �;��x  B&{                                    Bxi,l  �          A2ff@ȣ�A�����Q�B\Q�@ȣ�A�R>.{?c�
B^��                                    Bxi,  h          A,  @�(�@�\)��z���{B:33@�(�@�33�E���z�BH��                                    Bxi,!�  
-          A,(�?ٙ��Dz������C���?ٙ��J=q�$  B�C��
                                    Bxi,0^  ;          A6�R@��׾���((��)C��@���?�33�%��u�A�ff                                    Bxi,?            A5�@�@�\)�r�\����Bj  @�@�z��z��N{Br��                                    Bxi,M�  T          A/�@/\)��33��H�C�xR@/\)?333�(�Ad��                                    Bxi,\P  	�          A1p�?��\�����#��{C��?��\����-���C���                                    Bxi,j�  
Z          A3\)@��W��'�ffC�)@��k��/�.C�B�                                    Bxi,y�  "          A!p�@@  ���R��{C��@@  ?.{��aHAM�                                    Bxi,�B  �          @�@�?s33����>ffA�z�@�?��
��  ��p�A�p�                                    Bxi,��  
�          @�Q�?k����@أ�B�8RC�Q�?k��j�H@�33BhC��                                    Bxi,��  �          Aff�   =�Q�A  B�L�C)n�   ���A��B�B�C�                                    Bxi,�4  "          A#
=�
=�Mp�A��B��\C���
=���A{BgQ�C��                                    Bxi,��  T          A Q�?�{��@���A��C���?�{���@333A�33C�@                                     Bxi,р  �          A��@1���{@�33BZ  C��@1����@���B+��C�q�                                    Bxi,�&  T          A�?+�?�@�{B�{B���?+�>��
@�(�B��
A�33                                    Bxi,��  
�          A��?
=qAQ�@�AۅB��q?
=q@�G�@�  B"�B��                                    Bxi,�r  �          AQ�=uAz�?�z�A>�RB�B�=uA@z=qA�z�B�8R                                    Bxi-  T          A  �8Q�A=q����G�B�G��8Q�A�R?p��@��B�B�                                    Bxi-�  
�          A,z�@�G�>�ffA(�B�@�p�@�G���
=A�B|C�W
                                    Bxi-)d            AUG�@����Q�A+33B\33C���@�����A�B3�RC��=                                    Bxi-8
  
�          A`  @�=q���A9�B_(�C��f@�=q� ��A   B6p�C�T{                                    Bxi-F�  
�          AP��@vff��(�A Q�BI�C��f@vff��A�B{C�\                                    Bxi-UV  �          AM�@�G��\A#33BS33C�!H@�G��  A��B'Q�C�{                                    Bxi-c�  �          AV�R@��
�ȣ�A*=qBR�C��)@��
���A�RB'p�C�xR                                    Bxi-r�  	�          AZ{@�  ��Q�A0Q�BXffC�` @�  ��z�AQ�B1�C�                                    Bxi-�H  "          AdQ�@�z���ADz�Bo=qC�c�@�z���z�A8Q�BX��C��                                    Bxi-��  
�          Ao�
@����ffA7�
BI�\C���@����)�A��B�C�p�                                    Bxi-��  m          Ar{@�{��RA0��B<
=C��q@�{�;\)A	��B
��C��
                                    Bxi-�:  m          As
=@�{��ffAS
=BqC���@�{���
A=�BN{C���                                    Bxi-��  
�          Av�H@�33���HA]G�B�
C�� @�33���AQ�Bi�HC���                                    Bxi-ʆ  �          Ay@�zὸQ�A]Bz�C��f@�z��Q�AW\)Bn�HC���                                    Bxi-�,  "          Ay�A�R?(��AT��Bi\)@�(�A�R��AQp�Bc�C���                                    Bxi-��  T          Ax��A\)?��\AP��Bb��A��A\)���
AO�Ba\)C��                                    Bxi-�x  ;          AyAff@%�AIp�BV�\A�p�Aff����AMp�B]  C��{                                    Bxi.  
s          A�=qA(�@c33AI��BO(�A��A(�?
=qAQG�BZ��@S�
                                    Bxi.�  
�          Ap  A	@�  A6�HBF��A���A	?�(�AB�\BXA5G�                                    Bxi."j  	          Af�\@�\@�z�A8Q�BU�
A�\)@�\?˅AC�Bi��AC�
                                    Bxi.1  ;          Af�\A=q@.�RA4  BR�A�(�A=q<�A9G�B[��>k�                                    Bxi.?�  T          Ag�Aff@}p�A'\)B9�A�\)Aff?�A1��BH�A��                                    Bxi.N\  �          Amp�AG�@4z�A.ffB>
=A��AG�>L��A4(�BFff?�
=                                    Bxi.]  �          Ap��A!����(�A-G�B;{C���A!���n{A#�
B.G�C��                                    Bxi.k�  T          Ac�A�
���@��RB�\C�fA�
��
@ƸRAϙ�C���                                    Bxi.zN  �          A`��Az����RA�\B(�C���Az�����A ��B  C��                                    Bxi.��  �          AaA=q��(�Ap�B0��C�4{A=q���A��BC�*=                                    Bxi.��            A`��A"=q�g
=A�B(�C�7
A"=q���@��
B�C���                                    Bxi.�@  �          A]�A��Tz�A�B,Q�C�k�A����RA�RB��C�&f                                    Bxi.��  �          A^�\A	����A�
B7�HC���A	��ҏ\A
=BC�P�                                    Bxi.Ì  �          Ab�RA����33A%p�B;  C�33A����z�A��B33C�"�                                    Bxi.�2  �          A^{Az�����A"ffB<33C��)Az����A��B\)C�Ff                                    Bxi.��  �          AX(�A��,��A{B7�C���A�����A�B!�
C���                                    Bxi.�~  �          Ac�
Aff�=p�A((�B>(�C��Aff���A(�B'�C��\                                    Bxi.�$  �          Aq��A(��l(�A.ffB9  C��A(���p�A\)B Q�C���                                    Bxi/�  �          Ao33A%���.�RA%B0(�C���A%�����A�\B�RC��)                                    Bxi/p  �          Au�A&ff�{A0��B8�C��RA&ff��G�A#�B'  C��                                    Bxi/*  
�          ApQ�A%����A,��B8G�C��A%�g
=A#�
B,33C�ff                                    Bxi/8�  �          Ak33A{�.{A1��BD(�C�}qA{�3�
A+�
B;C�޸                                    Bxi/Gb  �          Ac33A�@   A2�\BP33AT(�A��8Q�A4��BTQ�C��                                    Bxi/V  �          A`(�A�׿&ffA#33B;\)C��A���Dz�A  B0��C��                                    Bxi/d�  T          A_�A�?�Q�A�B8  A(�A녿s33A{B9p�C�q                                    Bxi/sT  �          AQp�A�H@�AG�B>�Ah(�A�H�8Q�A�BEp�C�ff                                    Bxi/��  �          A>=qA  �l(�@���B �C�!HA  ���H@��RA�(�C��R                                    Bxi/��  T          A-��A���p�@�G�B  C��A��L(�@�{B33C��H                                    Bxi/�F  �          @�  @<�Ϳz�@S�
B?=qC�p�@<�Ϳ�{@E�B-�HC���                                    Bxi/��  �          A z�@��H�0��@\BGQ�C�'�@��H�Q�@�\)B8z�C���                                    Bxi/��  �          A{@�  �Ǯ@�(�B
=C�=q@�  �$z�@xQ�A�{C�1�                                    Bxi/�8  �          A�@�p��-p�@��
B p�C���@�p��k�@eA���C�{                                    Bxi/��  �          A  @�z��\��@�  A�RC�Ff@�z���(�@R�\A��C�O\                                    Bxi/�  �          @�
=@����N{@j=qA��C�p�@�����Q�@1G�A�
=C���                                    Bxi/�*  �          @��@��
�h��@b�\A�p�C��@��
��z�@#�
A��
C���                                    Bxi0�  �          @��H@��R�J�H@z=qBffC���@��R����@A�Aۙ�C�(�                                    Bxi0v  �          @�\)>���?O\)@��HB�� B�
=>�����=q@��B���C�                                    Bxi0#  T          @��@zᾏ\)@ƸRB��HC�
@z���@��RB��C���                                    Bxi01�  �          A*�R@׮��z�@%�Ac�C��{@׮� ��?#�
@_\)C��)                                    Bxi0@h  �          A/\)@������  ��=qC�@���� ����\�AC���                                    Bxi0O  �          A%p�@ȣ�� z��\)�Q�C���@ȣ���z��a���Q�C�(�                                    Bxi0]�  �          A  @�p�������p��C��)@�p���\)��p��י�C�\)                                    Bxi0lZ  �          @�\@J=q�Ǯ�*�H���C�p�@J=q��z����
���C�1�                                    Bxi0{   �          @�(�?���G��'����\C�#�?���ff���H��C�O\                                    Bxi0��  �          @�\)?}p��У��X�����C�P�?}p���\)�����%G�C�q                                    Bxi0�L  �          @��u��p���{�ffC����u��z���Q��L��C���                                    Bxi0��  |          AB�\@�Q��
ff?�  A	G�C��{@�Q���þ�G����C��\                                    Bxi0��  �          A>�\@���R����5C�y�@����&ff�K33C���                                    Bxi0�>  �          A>=q@��R�(�þ���RC��=@��R�#33�/\)�U��C�3                                    Bxi0��  �          A9�@�����;�{��
=C��@�����\�0���\��C��R                                    Bxi0�  �          A8��@���=q=#�
>B�\C��q@���p��
=�=G�C�                                    Bxi0�0  �          A0  @�p����xQ����C�B�@�p��33�J�H���HC��                                    Bxi0��  �          AAp�@��Q���G���{C�*=@�����G���G�C���                                    Bxi1|  �          A (�@Ǯ����E�����C��)@Ǯ�����  ���HC��                                    Bxi1"  �          AQ�@�p���G��\)�x��C�B�@�p����\)�|(�C��q                                    Bxi1*�  �          A ��@��
� (�������  C��@��
��ff�J�H��=qC��H                                    Bxi19n  T          @�  @0���g�������
=C���@0���B�\�,���\)C�#�                                    Bxi1H  �          @��@p  ��G�?��HA
=C�s3@p  �����z��
=C�0�                                    Bxi1V�  �          AQ�@��������	���j{C��{@�����G��b�\��{C�                                    Bxi1e`  �          A�@������R��33��(�C�!H@����qG���
=�p�C��                                    Bxi1t  �          @��@�(��B�\�^�R��\)C�@�(�?0���Z=q��(�@�Q�                                    Bxi1��  �          @�=q@�Q�Q��\(���\C�C�@�Q�=��
�b�\��?333                                    Bxi1�R  �          @�\@�ff��\)�>{��z�C��
@�ff�O\)�Q���Q�C�k�                                    Bxi1��  �          @�33@أ�����Q���  C��q@أ׿�ff�o\)��C���                                    Bxi1��  �          A�@�\)��U��\)C��q@�\)��z��u���{C�K�                                    Bxi1�D  �          A(�@����8���l(���\)C�Y�@��Ϳ��������C��q                                    Bxi1��  T          A
�R@�G��7��l����=qC�@ @�G���=q�����RC��3                                    Bxi1ڐ  �          A�\@�{�5�e��=qC�&f@�{��=q��ff���
C���                                    Bxi1�6  |          A#33A���QG��P  ����C��A�����~�R��C�Ф                                    Bxi1��  �          A(z�A{�c33�I����\)C��A{�'��|�����RC�33                                    Bxi2�  T          A5�A#�����/\)�_33C��A#��P���l����(�C�&f                                    Bxi2(  �          A$��A��G��
=�U�C��A�����E��(�C��                                    Bxi2#�  �          A7�
A((��~�R�\)�4Q�C���A((��QG��L(���\)C�\)                                    Bxi22t  �          A6�\@أ�@��H����ϮB=(�@أ�A���333�ip�BK                                    Bxi2A  �          AG�@�p�A����R��(�BO33@�p�A\)��Q��  BX��                                    Bxi2O�  �          AM�@�RA33�������BN��@�RA$Q�����BX                                      Bxi2^f  �          AT  @�33A�������33BM
=@�33A(�Ϳ����{BU\)                                    Bxi2m  �          A`��A z�A,����=q���\BU\)A z�A7���\)����B\{                                    Bxi2{�  �          Ae�A�@�33?��@��A��HA�@�(�@!G�AP��A�p�                                    Bxi2�X  �          A[�A8(�<��
@�B\)=��
A8(���
=@�p�A��RC�@                                     Bxi2��  �          AR�HA)p�?��@��HB�@O\)A)p���p�@�\)BffC��                                    Bxi2��  �          AUp�A�>��RA=qB.�R?�A��ffAffB)
=C��                                    Bxi2�J  �          A3�@�  ��AQ�BF{C��@�  �0  AG�B8�RC��)                                    Bxi2��  �          A ��@��
�(�@��RBQ=qC�,�@��
�+�@��BA�C�,�                                    Bxi2Ӗ  �          A=q@�
=�\(�AG�Bfp�C�P�@�
=�A�@��BQQ�C��                                    Bxi2�<  �          A$��@�33���A  Bg(�C��3@�33�g
=Ap�BM��C���                                    Bxi2��  �          A�@��R?\)@�G�Bj�R@���@��R�\@��Bd��C��                                    Bxi2��  |          A/\)@AG�A�
@�  A�(�B��H@AG�@�33@�ffB(��B�Q�                                    Bxi3.  T          ADz�@!G�A2�\@�\)A�
=B��{@!G�A�@���BffB��                                    Bxi3�  �          A:�\@��A+�@z�HA�p�B�=q@��A��@�(�B  B��3                                    Bxi3+z  T          A8  @$z�A$(�@��\A�p�B��
@$z�A
�R@���BffB��                                    Bxi3:   �          A*�\@]p�A\)@�ffA�33B��3@]p�@�z�@ۅB#p�B�L�                                    Bxi3H�  �          A)p�@N�R@���@�z�B(p�B���@N�R@��A�
B^�RB`Q�                                    Bxi3Wl  �          A"=q@S�
@ۅ@�ffB%ffB��@S�
@��A��B[p�B^                                      Bxi3f  �          A"�H@�G�@��H@�G�A��
Bz��@�G�@�ff@޸RB,��Bc��                                    Bxi3t�  �          A�H@�\)@�  @�z�B �Bk�@�\)@��@��HB4=qBN�                                    Bxi3�^  �          Aff@z�H@ָR@�B�Bn��@z�H@�@���BD�BN
=                                    Bxi3�  �          A@���@��H@�p�B1�B.��@���@!G�@�BWG�A�p�                                    Bxi3��  �          A=q@^{@��H@���A݅B|Q�@^{@��@�
=B%�Bf�
                                    Bxi3�P  �          A=q?�p�@�ff@\)A���B�33?�p�@׮@�\)B�B�L�                                    Bxi3��  �          A=q?���A(�@	��A_�
B���?���@�33@�(�A�=qB�u�                                    Bxi3̜  �          A�?��A��@�AYp�B��H?��@�p�@��HA�  B�B�                                    Bxi3�B  �          A�?0��A�@ffAp��B���?0��@�Q�@�z�A�z�B��)                                    Bxi3��  �          A���(�@�Q�@��A�z�B��(�@�=q@�p�BffB�G�                                    Bxi3��  �          A
�\��z�A�?���AQG�B��Ϳ�z�@�(�@�(�A���B�Q�                                    Bxi44  �          A{��=q@��@!G�A�{BÅ��=q@�p�@��HB
=BŸR                                    Bxi4�  �          A���z�A�H@%AyB�ff��z�A�@�Q�A�\)B�B�                                    Bxi4$�  �          A(�׿8Q�A$Q�@�
A5G�B�  �8Q�A��@��A�{B���                                    Bxi43&  T          A-����A)@ ��A*�RB������Aff@��AԸRB���                                    Bxi4A�  �          A/��}p�A+�?��RA'�B��=�}p�A(�@�(�AӅB��{                                    Bxi4Pr  �          A/
=>�A,��?ٙ�A33B��\>�Aff@��
A�(�B�p�                                    Bxi4_  �          A/
=>B�\A-p�?�Q�@�=qB�
=>B�\A Q�@�z�A�G�B��H                                    Bxi4m�  |          A4(�=�A2ff?���A   B���=�A$z�@�33A�
=B��R                                    Bxi4|d  �          A4��>�A2�H?��H@�B�G�>�A%G�@�Q�A���B��f                                    Bxi4�
  �          A:�\?W
=A7
=?�z�A�B���?W
=A(z�@�G�A�
=B��                                    Bxi4��  �          A$��?uA�@33AN�\B�  ?uA�H@��A�z�B��R                                    Bxi4�V  |          AK33���
AH�ÿaG���Q�B�W
���
AD��@*�HAC33B��=                                    Bxi4��  �          AM녿˅AG
=��R�!�B{�˅AI�?���@��HB�k�                                    Bxi4Ţ  �          AC��*=qA,Q����H��G�BϽq�*=qA<Q��{�Q�B�u�                                    Bxi4�H  �          AHQ��C�
A)���ff��Q�B�#��C�
A=��.{�H  B��                                    Bxi4��  �          AAG����@����
ff�:  C���A (������
B��f                                    Bxi4�  �          A=��@��
����*=qC�R��A����
��(�B���                                    Bxi5 :  �          A/�
���
@������/z�CT{���
@�{���H����B���                                    Bxi5�  �          A0����p�@�����z��!�B��q��p�A�\���R�ՅB�8R                                    Bxi5�  �          A\)�Mp�@ָR�ҏ\�&{B�8R�Mp�A������ҸRB�Q�                                    Bxi5,,  h          A6�\��Q�A
=?W
=@���B�zῸQ�A��@hQ�A��BŞ�                                    Bxi5:�  |          A5�@W���G�A$  B��{C�R@W��A�A��B�\)C��                                    Bxi5Ix  �          A[\)@��\��\)A9Bg\)C�z�@��\�p�A33B4{C���                                    Bxi5X  �          Ab{@���θRA:ffB]G�C�U�@���z�A��B$Q�C�ff                                    Bxi5f�  �          Ad��@|����A?
=Bc�C�O\@|����AQ�B'��C���                                    Bxi5uj  �          Ab=q@�(��\)A$��B;��C��\@�(��4��@�A���C��                                    Bxi5�  �          Aap�@�
=��RAp�B+��C�� @�
=�8��@�G�A�\)C�                                      Bxi5��  �          A_33@���
=AQ�B�
C�1�@���;�
@��A�
=C�f                                    Bxi5�\  �          Ah��@��<��@�p�A뙚C��\@��U�@\��AZ�HC���                                    Bxi5�  �          Al��@�33�Q��@��
A�G�C�.@�33�_33?\(�@W
=C���                                    Bxi5��  �          AYG�@
�H�T��?���@���C�� @
�H�P���5�B{C��
                                    Bxi5�N  �          AIp�?�  �G
=?z�@(Q�C�H?�  �A��Dz��a�C�                                      Bxi5��  �          ALz�?��H�Hz�^�R�|(�C��H?��H�;���\)��C�#�                                    Bxi5�  �          A>{�L���;33?�G�@�C�޸�L���:{��(��{C�޸                                    Bxi5�@  �          A*�R���H��G��5���=qC~녿��H��z����H�z�C|E                                    Bxi6�  �          A��hQ���(���z��뙚CpaH�hQ���(����\�/33Ciff                                    Bxi6�  �          A'\)�O\)�������ZffCkǮ�O\)�
�H��\��CU�
                                    Bxi6%2  �          A?�
�ᙚ?xQ�����UG�C,+��ᙚ@o\)�p��?C�                                    Bxi63�  �          A733�θR@�����T�RC!(��θR@���� ���433Ch�                                    Bxi6B~  �          A#��ƸR>�z�� ���Qz�C1Y��ƸR@*=q���
�AC�{                                    Bxi6Q$  �          A����33>L���ff�b��C1����33@&ff��\)�Q��C{                                    Bxi6_�  �          A\)��z��R�\)�t33C;����z�?��(��j�
C�                                     Bxi6np  T          A/�
��\)�@  �(��lffCU����\)�B�\�  p�C6ff                                    Bxi6}  �          A3���Q����
�\)�U�
Cc0���Q��p��   ��CK�3                                    Bxi6��  �          A4z���(��޸R�	�J  CB8R��(�?fff���N33C,�                                     Bxi6�b  �          A%���z��&ff����?��CJ���z�B�\�G��N��C5��                                    Bxi6�  �          A8Q����
�W
=�ff�;p�C5�\���
@�
�G��2G�C#��                                    Bxi6��  �          A2=q��p��(��G��<p�CD����p�>�����EQ�C1T{                                    Bxi6�T  �          A'\)�أ�?�z�����>�C&8R�أ�@x���ٙ��$33C&f                                    Bxi6��  �          A0z��ۅ?���   �E�C/u��ۅ@;���R�3��C�H                                    Bxi6�  �          A0���ۅ�p���(��KffC;��ۅ?�
=�=q�G��C&E                                    Bxi6�F  �          A6�H��p�?�=q���W{C$���p�@�  �=q�8(�C�R                                    Bxi7 �  �          AB=q��(�@�����C
=Ch���(�@����R�CL�                                    Bxi7�  �          AH  ��@�p���
�?\)C)��@����R�
=CY�                                    Bxi78  T          A.ff�{�?�p���H��C"���{�@����	�c
=C33                                    Bxi7,�  T          A4z����R?�
=�=q�|�C�����R@����	p��R�HC                                    Bxi7;�  �          A>ff���H@�  �z��5��C����H@�33��{�33C �3                                    Bxi7J*  �          A8����@�ff����/  C���@��
��{��=qC{                                    Bxi7X�  �          A/���(�@&ff����i(�C���(�@��\� ���=�
C�f                                    Bxi7gv  �          A8  ��33@1����Y{C����33@�  � (��1�C	�3                                    Bxi7v  �          A4z���  @�p�����P=qC޸��  @�{��z����Cn                                    Bxi7��  �          A:{���\@��H����C
=C:����\@�����\)�G�B��                                    Bxi7�h  �          A5�����\@�\)����U�C�
���\@����p�B�u�                                    Bxi7�  �          A?33����@�G��Q��>=qB��
����A�H��=q� \)B�                                      Bxi7��  �          A;\)��Q�@�\)� ���/=qB�
=��Q�A{�����B��f                                    Bxi7�Z  �          A8  ���\@��
����)G�C �
���\A33�����ۅB�                                      Bxi7�   T          A3���(�@\��p��'Q�C��(�Ap���������B���                                    Bxi7ܦ  �          A)��=q@C�
�(��M�CE��=q@�\)��z��#
=C�3                                    Bxi7�L  �          A,Q�����@Mp��G��_�Ch�����@����(��/z�C�{                                    Bxi7��  �          A���z�=��
��{ffC2\�z�@#�
��Q��{C(�                                    Bxi8�  �          A�Ϳ�z��R�\������Cz�)��z�.{��\¡�
CR��                                    Bxi8>  �          @�(�?�����(��qG���HC�\?����k���ff�V
=C�T{                                    Bxi8%�  �          @��
����(����
��C^n���>.{��ff=qC0�                                    Bxi84�  �          A  �@  �(����� C?z��@  ?�z�����33C��                                    Bxi8C0  �          A���
=>�z���{\C,5��
=@-p���ff
=B���                                    Bxi8Q�  �          @�Q쿼(��R�\���o{Cu�׿�(���z������CZ.                                    Bxi8`|  �          A����������ff�1��C5������?˅��
=�(�C$�                                    Bxi8o"  �          A���H��33��=q�*G�CCǮ���H>W
=����3Q�C1�                                    Bxi8}�  �          A
=��(��[����
�B�C[�q��(���ff��z��gz�CE}q                                    Bxi8�n  
�          A��y����\���\�r��CO���y��?�����
C+�                                    Bxi8�  �          A  ����!G�����hQ�CS+����=��33�|\)C2ff                                    Bxi8��  �          Aff�^�R�7
=��=q�i��C[xR�^�R��Q���33C9�                                    Bxi8�`  �          A녿��ڏ\��{��HC�q����
=���H�^�HCzxR                                    Bxi8�  �          A  ��
=����:�H��Q�C�%��
=�޸R���R�{C�4{                                    Bxi8լ  �          A  ?s33�  �7
=��{C�H�?s33��(����
���C���                                    Bxi8�R  �          Az�@
=�z��.{��G�C�%@
=�޸R��  �ffC�o\                                    Bxi8��  �          A�R@<����
���R�
=C���@<��������\��
=C���                                    Bxi9�  �          A=q@}p���\�u�љ�C�,�@}p����
�R�\���\C�l�                                    Bxi9D  �          A
�\@i��������(��8Q�C���@i����\)�<�����C�c�                                    Bxi9�  �          A��@����?n{@�(�C�0�@�����
=��G��2�\C�c�                                    Bxi9-�  �          A��@�  ��?�(�A
=C�AH@�  ����R����C�1�                                    Bxi9<6  �          AQ�@�����{?ǮA�HC��f@�����G��u����C���                                    Bxi9J�  �          A$��@�ff��R?�@�C�|)@�ff��׿�������C�b�                                    Bxi9Y�  �          A��@ə���@�\A>�\C�H�@ə����
���H�4z�C��=                                    Bxi9h(  �          A��@��\��33@-p�A��C�)@��\��=u>�Q�C�g�                                    Bxi9v�  �          A�@�ff��R@G�A�  C�j=@�ff��>��H@@��C�p�                                    Bxi9�t  �          A�H@����љ�@g�A��C�e@������
?�  @��C���                                    Bxi9�  �          Az�@��\���H@hQ�A�=qC��)@��\��ff?�z�A��C��q                                    Bxi9��  �          A��@�33���@��A�\)C���@�33���@	��AV�\C�B�                                    Bxi9�f  �          A��@Å��p�@n{A��C��3@Å��=q?�ffAG�C���                                    Bxi9�  �          A=q@����
=@c33A�{C�7
@�����?s33@��C���                                    Bxi9β  �          A
{�˅���>W
=?�p�C�p��˅���R�   ��
=C�*=                                    Bxi9�X  �          A��/\)���E�����C}���/\)�z��u����C|+�                                    Bxi9��  �          A Q��S�
��Ϳ����(�Cy��S�
�����  ��G�Cw��                                    Bxi9��  �          A���(�����R�P��CfxR��(���  ��33�ݮCa��                                    Bxi:	J  T          A�H�Ǯ��R���R�8��Cf
�Ǯ��33��(���p�Ca��                                    Bxi:�  �          A�
���
���ÿ����\)Cc�����
��=q�}p�����C_�f                                    Bxi:&�  �          A"ff�\�����z�� Q�Ch�q�\��G�������G�Ce5�                                    Bxi:5<  �          A#�
��=q�(����
�	�Cj�
��=q��Q���\)���
CgE                                    Bxi:C�  �          A$������(���\)�'
=Cjp������z�����љ�Cfff                                    Bxi:R�  �          A(�@=q��?��@dz�C�B�@=q���&ff�y�C���                                    Bxi:a.  �          A%�������p��_33Cs�������Q����\���
Co�=                                    Bxi:o�  �          A-G���
=�
{�1��p(�Cls3��
=��p�������p�Cgff                                    Bxi:~z  �          A.�\���������H�'33C_�f������  ��33��=qC[B�                                    Bxi:�   �          A+
=���H�33�����/33Cr�����H�����z���  CoQ�                                    Bxi:��  �          A*�R=L���)?^�R@�C�"�=L���$(��4z��uC�#�                                    Bxi:�l  �          A+
=?����&ff?��HA*=qC��?����'33��p��G�C���                                    Bxi:�  �          A.�\?.{�+
=@�A+�C��{?.{�+�
��\���C��3                                    Bxi:Ǹ  �          A-��&ff�&�H��G����C�9��&ff�=q��  ��{C�{                                    Bxi:�^  �          A*�\�l(�����Q���p�Cw���l(���G��ۅ�#�CrO\                                    Bxi:�  �          A/���z����
�����Q�CZ���z���  ����CQ�                                    Bxi:�  �          A1�������\���
� Q�CRǮ��׿˅��H�>��C@^�                                    Bxi;P  �          A8����G���}p���  Ck� ��G���{��G���Cd�q                                    Bxi;�  �          A5���ff����  �8��CpO\��ff��H����ClQ�                                    Bxi;�  �          A8����=q�p��ff�2�HCw����=q��
������RCtc�                                    Bxi;.B  �          A4Q����
�*=q����G�CGٚ���
����(��#=qC7�)                                    Bxi;<�  �          A.ff�k��)?8Q�@xQ�C��׿k��#33�A���{C�j=                                    Bxi;K�  �          A$(���G���� ���>=qCn0���G�������\)��Ci޸                                    Bxi;Z4  �          A��
=�����G����Cc  ��
=������p��=qC[^�                                    Bxi;h�  �          A���
����-p�����CWc����
�|����(���33CP+�                                    Bxi;w�  
�          A���33������(���\)CT.��33�AG�������\CI��                                    Bxi;�&  �          A����������������
CT� �����>{��
=�  CI�                                    Bxi;��  �          A ����(���
=��z���\)CRY���(��(Q���{�CF�q                                    Bxi;�r  �          A!����w
=��G����CNL���녿����\���CA�                                    Bxi;�  �          A���\)�Q�����G�CK���\)��
=��(��p�C<�R                                    Bxi;��  �          A�H�����Q�����"G�C^�����=q�����PG�CKc�                                    Bxi;�d  �          A ����{�\)��(��O�
C\^���{�k��z��u�C?�                                    Bxi;�
  �          A�����#33�
{�p��CS����?333����C*�                                    Bxi;�  �          A#��Q���R���H���CB���Q�.{������\C5&f                                    Bxi;�V  �          A$  �����R��
=��=qC@�{�����
��33��{C4�                                     Bxi<	�  �          A"=q���\��p���(�C=����>�{��z���C1                                    Bxi<�  �          A#
=��Ϳ�(���{���HC=33���>k���p���ffC2��                                    Bxi<'H  �          A$�����^�R��p���  C9c���?G���{����C/+�                                    Bxi<5�  �          A$Q����L����
=��Q�C8����?G���\)�\C/=q                                    Bxi<D�  �          A"{�������
=��ffC=J=��?
=����(�C/�q                                    Bxi<S:  �          A"�R�ff�\��  �(�C6�f�ff?�����
=�	�C'k�                                    Bxi<a�  �          A'
=��{?333��Q��(�C.�{��{@L(������{CxR                                    Bxi<p�  �          A%���>aG���{�7��C2G���@9����33�%��C��                                    Bxi<,  �          A%G���
=?�33�ָR�%  C'����
=@~�R������
C�                                    Bxi<��  �          A%p�����@!G���=q���C"�����@�����
=��  C�                                    Bxi<�x  �          A$����\@33��p��)
=C$^���\@�{��{�Q�C��                                    Bxi<�  �          Az���
=��z����R�G�CCW
��
=>��R��Q��#{C1n                                    Bxi<��  �          A���?�R���\)C/����@{��33�癚C"��                                    Bxi<�j  �          A�
�p�<#�
��G���33C3�f�p�?�\��ff�ܣ�C'�q                                    Bxi<�  �          A��׮��33��\�ZffC`n�׮���������(�CZ+�                                    Bxi<�  �          A�����  ����� Q�Ci���ָR���
����Ce�R                                    Bxi<�\  �          A���
=�ڏ\��g�
Cd�)��
=��\)��ff����C^��                                    Bxi=  �          A�������
=��Q��((�C_
=��������w��ř�CY��                                    Bxi=�  �          A����  ��Q�0����p�CWn��  ����,(�����CT)                                    Bxi= N  �          A�������������CZG�����(��{�p  CW�                                    Bxi=.�  �          AQ���R����?��\@ʏ\CW���R���Ϳ�����HCW��                                    Bxi==�  �          A���
=��G�?�A;\)C\�q��
=��녾�ff�6ffC^�                                    Bxi=L@  �          A������������
�<  CP�������H���L(�����CJ�f                                    Bxi=Z�  �          A���������ÿ�  ��CTY������n�R�6ff����CO�{                                    Bxi=i�  �          A
�\��{�qG��z�H��G�CN
��{�G��z��y�CJ)                                    Bxi=x2  �          A	p��\)��\�\)�q�CC���\)��z῰�����CA
                                    Bxi=��  �          A	��z��
�H���`  CB���z����ff��C@T{                                    Bxi=�~  �          A	���׿��R�z��w
=CA���׿У׿��
�
=qC?�                                    Bxi=�$  �          A	��
���R���\(�C<J=��
���Ϳ
=�z=qC;^�                                    Bxi=��  �          A  �=q�s33<��
>\)C:s3�=q�c�
��{�33C:                                    Bxi=�p  �          A
=�{���R?8Q�@��C6)�{�\)?��@o\)C7�\                                    Bxi=�  �          A���=q��(�?��\@�33C<J=�=q�\>�@N{C>B�                                    Bxi=޼  �          A=q�
=�.{?�=qA.=qC8�
=��p�?���AQ�C<��                                    Bxi=�b  �          A{� �þ#�
@
=qAo�
C5&f� �ÿxQ�?�Q�AV�\C:޸                                    Bxi=�  �          Ap���z����?z�H@�\CB��z���>.{?��RCDff                                    Bxi>
�  �          A�������þ�p��#�
CW���������
=q�t  CT�)                                    Bxi>T  �          A����ff���
��z����C_c���ff����]p��ģ�CZ�                                    Bxi>'�  �          A���
��Q����33Cj�����
��z���33���Cd�                                    Bxi>6�  �          A�dz���z��@  ���HCr�f�dz���
=����"Q�Ck��                                    Bxi>EF  T          A���L(��߮�C�
���Cuz��L(����������'G�Cnٚ                                    Bxi>S�  �          AQ��\)�Ӆ�(�����HCn�)�\)���H��\)�ffCgٚ                                    Bxi>b�  �          A�����\�W
=��p�Ce���������"�\���HCbO\                                    Bxi>q8  �          Ap���  �x�ÿ333���\CP=q��  �Tz��	���p��CL��                                    Bxi>�  �          A{�أ���(�?��@�=qCS\)�أ���G���ff��CR�\                                    Bxi>��  �          A=q���H����=�?n{Ch�{���H��(���R���RCf��                                    Bxi>�*  �          A(�����fff@�(�Bp�CT
�������@(��A��\C]&f                                    Bxi>��  �          A  ���H��33?fff@�=qCT}q���H����^�R��33CT�                                    Bxi>�v  �          A33���H��Q���h��CWٚ���H������
=CTk�                                    Bxi>�  �          A������
=�.{��z�Cd=q�����ff�����C[s3                                    Bxi>��  �          A����������p��-�CeY���������i����G�C_�                                     Bxi>�h  �          A����p��r�\?B�\@��\CO�f��p��r�\�J=q��=qCO�)                                    Bxi>�  �          A���Q���z��p��\��Cb����Q���z���(����C\)                                    Bxi?�  �          @�
=�Z�H��=q�k��ޣ�Cp�H�Z�H��z������<=qCfu�                                    Bxi?Z  �          @��
�������R���i��Ch^�������=q�*�H��p�Ce�                                    Bxi?!   �          @�\��G����\=��
?(�CV��G��q녿���B�RCT                                    Bxi?/�  �          @�33��  ���H�
=���Ca�{��  �Q��~�R�G�CX#�                                    Bxi?>L  �          @�G���Q���Q��
=��C[  ��Q��>�R�vff�  CQ�                                    Bxi?L�  �          @�{��ff�*=q������CKB���ff�����AG���Q�CBs3                                    Bxi?[�  �          @�
=�5���G���G��Cop��5��&ff��=q�f��C^��                                    Bxi?j>  �          @���z=q��ff�)�����
Ck�
�z=q����ff��\Cb�                                    Bxi?x�  �          @�
=�XQ����R�XQ���  CoaH�XQ��x�����
�8Ce
=                                    Bxi?��  �          A���=q��p��^�R��(�Cb� ��=q���\�J=q���C^aH                                    Bxi?�0  �          A33������
=���
��\Ccٚ��������U����
C_�                                    Bxi?��  �          Ap���  ���>���@Q�Ck�\��  ��(��ff���Ci��                                    Bxi?�|  �          A����{��p�����N=qCkT{��{��p����R����CeQ�                                    Bxi?�"  �          A �������\)��(��(��Ce8R�����z��p  ��=qC_xR                                    Bxi?��  T          A (���p���p����H�(��Cd�
��p����\�n{�ݙ�C_
=                                    Bxi?�n  �          A������  �����Q�Cg\)�������g
=��(�Cb\)                                    Bxi?�  �          A Q�������z�=p���Q�Cc}q�������H�C33���\C_T{                                    Bxi?��  �          @�p��xQ���Q쿓33��\Cp#��xQ���
=�q���Ck�
                                    Bxi@`  �          @�{����׮�@  ��p�Cn��������
�Z=q���
Ck�                                    Bxi@  �          @��������׿����ffCh�\�����=q�Y����\)Cc�                                     Bxi@(�  �          @����p�������p��s�Ce���p�����������C^u�                                    Bxi@7R  �          @�z����������ff��(�Ckk������������
�  Cd=q                                    Bxi@E�  �          @�Q��aG��ȣ��z���=qCp���aG���z���{��\CjJ=                                    Bxi@T�  �          @��"�\�ָR�������CyE�"�\��(���
=�$  Cs��                                    Bxi@cD  �          @�Q��!G���ff�33��{Cz{�!G�������Q�Cu�                                    Bxi@q�  �          @����Vff����z�\Cs\)�Vff�����z����Cmff                                    Bxi@��  
�          @�  �.{�Ӆ��
����Cw���.{�����H�G�CrG�                                    Bxi@�6  �          @�Q��p�����&ff���RCyxR�p���p�����*p�Csu�                                    Bxi@��  �          @��H�:�H���
�z���Cv33�:�H����33��Cp��                                    Bxi@��  �          @�ff�>{�����z���p�Cu{�>{��(�������Cn�f                                    Bxi@�(  �          @�p��k���33����l  Cn�3�k���G���
=��HCh�                                     Bxi@��  �          @�{�p�����
��G��D��C���p����z����R��C�H�                                    Bxi@�t  �          @��>�p���\�^{�У�C��H>�p��������H�I�RC�                                      Bxi@�  �          @��H�l��>L��>�
=@�G�C0�
�l��<#�
>��@�Q�C3�
                                    Bxi@��  �          @�33���
@k�@��RB!C
8R���
?�33@��BT�C!@                                     BxiAf  �          @陚��  @r�\@y��B�
C^���  ?�=q@�(�B4�
C��                                    BxiA  �          @����(�@-p�@eA�{C(���(�?xQ�@�z�B G�C)E                                    BxiA!�  �          @���ff��=q@8Q�A��HCF�=��ff�7
=?�{A�ffCO��                                    BxiA0X  �          @�����Q���z�@ffA�(�C]8R��Q����
=��
?+�C`+�                                    BxiA>�  �          @Ӆ�������?�Q�A�G�C\)������=�\)?#�
C_
=                                    BxiAM�  T          @�{�����?�  A>�RC|=q�������p��:ffC|@                                     BxiA\J  �          @�Q��\��(�?���AQ�C����\�Ǯ�ٙ��r�\C��                                    BxiAj�  �          @���=u��  ?0��@��C�Y�=u���\���H����C�]q                                    BxiAy�  �          @��W
=������
���C�lͿW
=��  �{��  C��{                                    BxiA�<  �          @��Ϳ�z�����>�  @�HCz���z����
�ff��(�Cy��                                    BxiA��  >          @��\��H��ff?�(�A��Cq}q��H����
=q���Cr�\                                    BxiA��  �          @�z��<������?}p�A,z�Ckp��<�������h���{Ck�\                                    BxiA�.  �          @�����(��tz�?��ANffC^��(���Q��G���p�C`!H                                    BxiA��  �          @�  ��  �N{?���A��
CT�\��  �l(�>�\)@&ffCXk�                                    BxiA�z  �          @�z���(��Tz�?�p�A���CS����(��vff>\@Tz�CW��                                    BxiA�   �          @�p������>{@33A�G�CQc������c�
?
=@��CU�q                                    BxiA��  �          @�\�����@,(�A���CI�
����U?�33A7
=CP�)                                    BxiA�l  �          @�(��Å���@Dz�AͅCE+��Å�AG�?�p�A��CNG�                                    BxiB  T          @����Å��33@U�A�{CC
�Å�:=q@�
A��HCMu�                                    BxiB�  �          @����녿�  @K�A��CA�����.{@\)A���CL#�                                    BxiB)^  �          @����ȣ׿�\)@7
=A�(�C@^��ȣ��p�@   A�=qCIu�                                    BxiB8  �          @ڏ\��p����@P  A�C7�q��p���G�@0��A��RCD�
                                    BxiBF�  �          @�ff���H>Ǯ@n{B�C05����H���R@a�A�z�C@�                                    BxiBUP  �          @��
��Q�?��@���B\)C.W
��Q쿞�R@xQ�B
  C@��                                    BxiBc�  �          @ۅ����?���@��B�C$Q����Ϳ�@�ffB"�
C9��                                    BxiBr�  �          @أ����?��H@���B  C!������z�@��HB*Q�C7Y�                                    BxiB�B  �          @�G����R?�ff@�z�B)�\C���R��=q@��B:�RC7h�                                    BxiB��  �          @��
��  ���
@^{A��C>0���  ��@,(�A�(�CJ�f                                    BxiB��  �          @�{��ff���H@VffA�\)C?� ��ff�"�\@   A�33CK�                                    BxiB�4  �          @���\�h��@XQ�A�(�C<��\��@)��A���CH�\                                    BxiB��  T          @�z���{��Q�@O\)A�=qCA\��{�,��@33A�z�CK��                                    BxiBʀ  
�          @�  �Å��R@#33A��CH�Å�G
=?�=qA/�CN��                                    BxiB�&  �          @��
��  ��ff@J=qAҏ\C?Ǯ��  �"�\@�A�G�CJ(�                                    BxiB��  �          @���33��\)@]p�A�
=C>h���33� ��@(Q�A��CJaH                                    BxiB�r  �          @�p�������  @XQ�A�  CD(������C33@�A���CN                                    BxiC  �          @��
���R��R@L(�A�
=CH�\���R�Y��?�33Aw33CQ��                                    BxiC�  �          @�\)����2�\@>{A��CO�����s33?��HAICW�H                                    BxiC"d  T          @�ff����u�?�=qAt��CV޸���������W
=CY�                                    BxiC1
  �          @�p���=q��p�?�  AIC`���=q����0������Ca:�                                    BxiC?�  �          A ���ָR�Tz�@<��A�=qCNL��ָR����?���A	p�CT�                                     BxiCNV  �          A���p��R�\@G
=A��
CL����p���=q?���A{CS{                                    BxiC\�  �          A����R�U�@EA�{CLǮ��R��33?��A�CS
                                    BxiCk�  T          A�
��(��^�R@HQ�A�(�CM=q��(���  ?��
A33CSc�                                    BxiCzH  
�          Az������\(�@N{A�ffCL�3������Q�?���AQ�CS^�                                    BxiC��  �          AG���{�x��@C33A�Q�CR&f��{���H?��
@��HCWٚ                                    BxiC��  �          A����
=�QG�@G�A�z�CM!H��
=���?�{A=qCS�q                                    BxiC�:  �          Aff��G�����@I��A�33C@���G��333@Q�AtQ�CH�q                                    BxiC��  �          @�ff�߮��
=@K�A�G�CCn�߮�H��@   Aip�CL+�                                    BxiCÆ  �          A�R��33�Z�H@X��A�=qCOff��33��=q?\A,  CV��                                    BxiC�,  �          A���ff�.�R@J�HA�  CIp���ff�vff?У�A9G�CQ                                      BxiC��  �          A�\��{�9��@J=qA��\CJ�R��{��Q�?��A.�\CR                                    BxiC�x  �          @�
=��G��}p�@#33A�\)CU=q��G���?�@xQ�CY                                    BxiC�  T          @���^�R�E@��A�p�C]�
�^�R�qG�?&ff@�RCcQ�                                    BxiD�  T          @��\�`  ��\)>�
=@��Cf^��`  �~{��G��~=qCd�)                                    BxiDj  T          @�z��Mp����>�p�@h��Clٚ�Mp���녿�{����Cj�H                                    BxiD*  �          @�(��'
=���=���?z�HCs�=�'
=��\)�  ��33Cq!H                                    BxiD8�  �          @�z��
=��G�?:�H@�{Cv��
=���ÿ�����{Cu��                                    BxiDG\  �          @ҏ\�\)�Å?h��@�Cy޸�\)���
��
=���\Cy�                                    BxiDV  �          @�p���z���
=����  C����z���33�%�иRC~�
                                    BxiDd�  �          @���^�R���ͿQ��
=C�8R�^�R��\)�C33��HC�5�                                    BxiDsN  �          @�ff�B�\��p������p�C�ٚ�B�\�����  �5�
C��                                     BxiD��  �          @�\)�z������H���C����z�����=q�6z�C�Ǯ                                    BxiD��  �          @��H��Q�����:=q��Q�C�uþ�Q���(���33�I33C���                                    BxiD�@  �          @�
=�޸R��녿˅�K\)C��޸R�������33C|�                                    BxiD��  �          @���G���ff��
=�ZffCG���G���
=������
C{W
                                    BxiD��  �          @�(���p���=q�z���=qC��׿�p���������4\)C|�)                                    BxiD�2  �          @�ff���
��
=� ����ffC�W
���
�����
=�+\)C|}q                                    BxiD��  S          @�=q��33��������RC��쿓33��(���33�*33C�b�                                    BxiD�~  �          @ָR��z����
��(��nffC�Ф��z���z����%z�C�L�                                    BxiD�$  �          @��H�h������p���z�C��Ϳh����33��=q�0��C�e                                    BxiE�  �          @�녾�����{������C��;�����G������3��C�8R                                    BxiEp  �          @�\?   ����
��33C�?   ������R�5C�Ǯ                                    BxiE#  �          @�\)?�ff��(��L(���(�C���?�ff��
=��  �S��C��                                    BxiE1�  �          @�p������(���p���z�C��������33���H�9�C��
                                    BxiE@b  �          @�
=�z���Q����Q�C��ÿz��`  �����F{C�N                                    BxiEO  �          @�=q�:�H��(��   ���\C��R�:�H�i����(��?�HC�]q                                    BxiE]�  �          @�녾���=q��(���=qC��f���u���ff�=�RC���                                    BxiElT  �          @�z��(�����ff�ap�C}�
��(������33� Q�CyW
                                    BxiEz�  �          @����{���
��G��v�RC}���{���
��(��%=qCx�                                    BxiE��  T          @��>8Q�����@����=qC���>8Q����H��Q��U��C�AH                                    BxiE�F  
�          @��?���\)����G�C���?��<���ƸR�=C��
                                    BxiE��  �          @��
?���(������0�RC�U�?��Q��Ϯu�C��                                     BxiE��  �          @�\?����(�����0�RC��f?����
�ӅffC�=q                                    BxiE�8  �          @�33@c33����:=q�¸RC�@c33�Z=q����7{C�{                                    BxiE��  �          @��H@3�
��Q��dz��z�C�@ @3�
�$z����
�Z�RC��H                                    BxiE�  �          @�z�@{���������RC���@{�(����H�w�\C���                                    BxiE�*  	�          @У�@6ff��=q����p�C�R@6ff�e����
�4�C�>�                                    BxiE��  T          @ҏ\@+���Q��8������C�~�@+��S�
��  �FQ�C�w
                                    BxiFv  
�          @��
@4z���\)�^�R����C�*=@4z��AG���Q��T��C��=                                    BxiF  
�          @�G�@K���G��fff� G�C�˅@K��$z���p��S�HC���                                    BxiF*�  �          @ڏ\@Tz�����mp����C�@Tz������ff�TG�C�q                                    BxiF9h  
Z          @ڏ\@����e��|(���HC�H@��׿�������D{C�Q�                                    BxiFH  
Z          @��
@�Q��l���z=q�
�C�~�@�Q쿹����{�DG�C���                                    BxiFV�  �          @�ff@��
�c33�������C�>�@��
������C{C�\                                    BxiFeZ  
�          @��
@����'
=�������C�q@��;�z������+�C�q�                                    BxiFt   
�          @�  @�����(���33�	C��@���?�R������@Å                                    BxiF��  �          @��@�=q���H�~�R�(�C�)@�=q?J=q���\�33@�                                      BxiF�L  
�          @�{@���
=�L�����C�#�@���z��{��Q�C���                                    BxiF��  
�          @��H@����#33�2�\��Q�C��{@����p���j�H��C���                                    BxiF��  �          @ə�@���L(��Q����
C�` @���ٙ��fff��C�b�                                    BxiF�>  �          @�
=@�33�r�\�.{�ϙ�C���@�33�z����)G�C���                                    BxiF��  
�          @ə�@�z��w��/\)��Q�C�q�@�z��Q�����)ffC�e                                    BxiFڊ  "          @�z�@q��^{�z���\)C��{@q���]p��C��f                                    BxiF�0  T          @�\)����{���|��C�׾����c�
� �C��=                                    BxiF��  T          @�=q>#�
��z῎{�=qC��)>#�
�����  �=qC��                                    BxiG|  
�          @�z�?�Q������
���RC�� ?�Q���{���H�7�C�q�                                    BxiG"  
(          @ƸR?�{��������C��?�{��(�����0C���                                    BxiG#�  T          @�ff@��p�������(�C��@��G������+{C�J=                                    BxiG2n  �          @�
=@	����ff��\)�C
=C���@	�����H�������C��\                                    BxiGA  �          @�=q?����=q�)����ffC���?���[�����VQ�C��H                                    BxiGO�  
�          @޸R�^{��p��\)��Co�H�^{��z��;��ʣ�Ck�q                                    BxiG^`  "          @ᙚ�S33��33��z����Cq���S33��\)�J=q��z�Cm�                                    BxiGm  "          @޸R�
=q��
=��p���{Cz�)�
=q�������R�,�RCt�                                     BxiG{�  �          @�z����(�=u?\)C��{������Fff��ffC�g�                                    BxiG�R  �          @�Q�>�\)��{��  �j�RC�5�>�\)������\)�,z�C���                                    BxiG��  �          @��.{�ڏ\�#33����C�(��.{��G���  �B�C���                                    BxiG��  
�          @�p�?5��p��\)����C��?5��p����
�B��C�U�                                    BxiG�D  �          @�=q>�  ��33������RC�{>�  ��(�����C�C��=                                    BxiG��  �          @�\)?���ڏ\�	������C�N?�����R����7�RC�+�                                    BxiGӐ  
�          @�>#�
��\)�J=q��  C���>#�
��p���(��Zz�C�3                                    BxiG�6  �          @�{?L����G�=�?��\C�~�?L�������B�\��33C��\                                    BxiG��  "          @���J=q��=q�\)��33C���J=q����i���33C���                                    BxiG��  "          @ָR@j�H��{�?\)��G�C�˅@j�H�p���  �<\)C��                                    BxiH(  �          @ڏ\@(������G���\)C�h�@(��~{��Q��9��C�w
                                    BxiH�  �          @�Q�?�33��33�g
=��G�C�Z�?�33�J�H����g{C�q�                                    BxiH+t  "          @��
?   ��{������HC��\?   ������6G�C��                                    BxiH:  "          @�ff@I�����R����{C�9�@I��������cC��{                                    BxiHH�  T          @�z�@a���(���(���RC�k�@a녿�(���ff�bp�C��q                                    BxiHWf  
�          @��
@qG��g���(����C�@qG�������H�\
=C�,�                                    BxiHf  �          Az�@����
�H��Q��A(�C�@���?�G�����NffA9�                                    BxiHt�  �          @��@�=q�}p������ ffC��@�=q��33��33�]{C�
                                    BxiH�X  �          @�  @`������b�\��
=C��@`���p������O=qC��f                                    BxiH��  �          @�p�@�Q������=q���C��\@�Q�>aG���=q�=q@�                                    BxiH��  T          A z�@AG��������C��@AG���H�ڏ\�q��C��                                     BxiH�J  T          @��R@p���  �����G�C���@p��U���(��op�C��                                    BxiH��  T          A33@�(��L(������ffC��@�(��
=��Q��+�C�'�                                    BxiH̖  T          A�\@   ���9��� (�C��f@   �\)��=q�b�C�ٚ                                    BxiH�<  �          A�\��33���@   A��Cb���33�\���mp�Cd��                                    BxiH��  
�          A�����R��ff@Adz�Cj�H���R���
��Q��G�Ck�=                                    BxiH��  
�          A	���(���ff@FffA�G�Cc�H��(��ָR�B�\���\CgE                                    BxiI.  "          A�
��
=��(�@?\)A�C\�H��
=������
��C`��                                    BxiI�  
�          A(�������
=?�(�A%�Ci�������˅����TQ�Ci+�                                    BxiI$z  
�          A녾����{�У��4��C��{�����ə������"(�C�}q                                    BxiI3   "          A�R@'
=��G��S33��C���@'
=���R����F
=C��)                                    BxiIA�  �          A   @�G������p���=qC���@�G����H��R�:{C�T{                                    BxiIPl  T          A%��@�33�������G�C���@�33��
=����:Q�C�&f                                    BxiI_  �          A'33@����b�\��Q���C��@��;����Q��"�C�
=                                    BxiIm�  T          A#�@��L(���  �  C�f@���Q��˅��C��\                                    BxiI|^  
�          A$Q�@���?\)��p��z�C�� @��>8Q�������H?�ff                                    BxiI�  
�          A#�A\)�\)����ffC��A\)?���  �z�@j=q                                    BxiI��  �          A(��A�
���
��{��C���A�
?��
�������A�H                                    BxiI�P  �          A%A����
��{���HC�c�A�?aG���p�����@��                                    BxiI��  �          A(�A33�33�|�����HC�b�A33>.{��ff�؏\?��                                    BxiIŜ  �          A�R@���B�\������C���@����������C�"�                                    BxiI�B  "          A�R@�(�����������C��@�(���(���=q�1  C�˅                                    BxiI��  
�          A(�@��a����R���
C��=@��}p�������C�T{                                    BxiI�  
�          A33A���(��z=q��C�� A��<���\)��R>L��                                    BxiJ 4  	�          A�R@�=q�}p��w��¸RC�1�@�=q���
��=q�
33C�G�                                    BxiJ�  
�          A(�@������H�u��Ù�C�}q@�����
��
=��RC�                                    BxiJ�  
�          A�
@׮��  ������  C�'�@׮��p������#��C�Ф                                    BxiJ,&  T          Aff@�ff������\)�܏\C��f@�ff��{���!�
C�=q                                    BxiJ:�  �          A��@������  ����C�k�@��Fff��(��6G�C�"�                                    BxiJIr  
�          Aff@�(���=q�mp����
C�O\@�(��#�
������C�u�                                    BxiJX  T          AG�@�{�����x������C�B�@�{�p������
C�޸                                    BxiJf�  �          A�@�=q��{������z�C�4{@�=q�У����H�\)C�G�                                    BxiJud  q          A=q@�����
�~{��(�C�h�@�녿���������C�b�                                    BxiJ�
            AQ�@�(������n�R���
C���@�(��'
=����� Q�C��H                                    BxiJ��  �          A��@ٙ������^{���C�c�@ٙ��(������(�C�&f                                    BxiJ�V  "          A��@�(�����Vff����C�J=@�(��
=���\�	Q�C��                                    BxiJ��  �          A(�@�(��>�R�s33��ffC��H@�(��0����G���\)C��                                     BxiJ��  �          A\)A����o\)��p�C��A�=u�����ߙ�>�ff                                    BxiJ�H  	`          AG�@��H�u��\)���RC��f@��H@�
�l���ԸRA��
                                    BxiJ��  �          A��@ᙚ��p�������C�z�@ᙚ?��x�����Af�R                                    BxiJ�  �          A
ff@��H?k���
=���@�(�@��H@I���hQ��ȣ�A�                                      BxiJ�:  �          A	�@�{@�\��=q� ��A�
=@�{@����;���Q�A�{                                    BxiK�  �          A��@���@AG����
�	�\A�ff@���@�p��\)��B��                                    BxiK�            A�
@�{��
=�\)��Q�C���@�{�fff��ff�B(�C�@                                     BxiK%,  �          A�\@�=q������(����C�f@�=q�I�����H�?\)C��                                    BxiK3�  T          A�@��������z�����C��q@���,����33�7
=C��                                    BxiKBx  �          A�@����Q��������C��@�녿�z���{�0��C�C�                                    BxiKQ  �          A�@�\)�I����=q��C��@�\)�8Q���
=�*�C�4{                                    BxiK_�  �          A=q@�p��tz���p�� 
=C���@�p��s33���R�*�C�Ǯ                                    BxiKnj  T          A	��@�
=��=q��=q�p�C��=@�
=?8Q����
�{@��
                                    BxiK}  �          A��@��<�����{>u@��@\)��
=��ffA�p�                                    BxiK��  |          A33A�H?�p�����ә�A��A�H@HQ��<(���Q�A��                                    BxiK�\  �          A�HA ��>�33���
��H@\)A ��@@�����R���
A�                                      BxiK�  �          AG�A  ?�G������ԸR@ٙ�A  @A��L(���A��                                    BxiK��  �          Az�A  >�Q�����z�@ ��A  @<�����H�ҸRA��                                    BxiK�N  �          A{AG�>�����=q���\@�AG�@,(�������33A��                                    BxiK��  �          Ap�AG�=�Q������?(�AG�@+�������HA��\                                    BxiK�  �          AG�A�?#�
��G����@�
=A�@E��������A���                                    BxiK�@  �          A��A (�?&ff��p����@��A (�@J�H��(�����A���                                    BxiL �  �          A��@�\)>8Q���p��  ?�ff@�\)@AG�������z�A��\                                    BxiL�  �          A��A (�>���Q��{@P��A (�@C�
����ՙ�A��                                    BxiL2  �          A  @��H?Y����(��z�@�z�@��H@e���p���=qA�                                      BxiL,�  
�          A  @��?!G���Q���H@��@��@Mp���
=��G�A�                                    BxiL;~  �          Az�A�R?(�����H���R@���A�R@A��u��Q�A��R                                    BxiLJ$  �          A  @��
?�33��\)���A	p�@��
@qG���z��ϮA�z�                                    BxiLX�  T          Aff@�\?O\)��(��"@�Q�@�\@s33���
��A�                                      BxiLgp  �          A�A�
?k���Q���R@��
A�
@Mp��hQ�����A�{                                    BxiLv  �          A
=A  ?�Q���{��
=A�A  @a��i�����HA��                                    BxiL��  �          A\)A��?�G���{���@�  A��@HQ��S�
����A��\                                    BxiL�b  �          A�RA
�H?E������Q�@���A
�H@1��L������A��                                    BxiL�  �          A=qAQ�?B�\������Q�@���AQ�@I���u���G�A��R                                    BxiL��  �          A�A33?����33��\)@��HA33@XQ��h�����\A�33                                    BxiL�T  �          A�
A ��?}p���p���@�\)A ��@Vff�n�R��G�A���                                    BxiL��  �          A��A ��?�G������p�A)��A ��@s33�\����A�                                      BxiLܠ  �          A  @��H������(��(�C��@��H@%��Q��   A��\                                    BxiL�F  
�          A=q@�����
����C�C�@��?�ff��G���AS
=                                    BxiL��  �          A�A�?=p���(����@�(�A�@P  ��G���z�A��R                                    BxiM�  �          AQ�A�H?\(�������=q@���A�H@L(��l(���(�A���                                    BxiM8  �          A�A (�>�����z�?��A (�@@  ��G��߅A�=q                                    BxiM%�  �          AQ�A\)>\��Q��z�@+�A\)@AG����\�ҏ\A�p�                                    BxiM4�  �          A�HA�?\(������  @�z�A�@G��dz���ffA��                                    BxiMC*  �          A�\A�?�{����ӮA'�A�@h���>{����A�G�                                    BxiMQ�  �          A"�HA�?�ff��
=���
@�=qA�@Tz��`�����A�
=                                    BxiM`v  �          A#\)A
�R>���33��p�?aG�A
�R@6ff������33A��                                    BxiMo  �          A&�RA  <��
�����z�>\)A  @(��hQ���p�AO�                                    BxiM}�  
�          A(Q�Az�?�\)�����z�AffAz�@q��J�H��z�A�33                                    BxiM�h  �          A(��A@(���(���{AR�\A@��
�(���g�A�=q                                    BxiM�  �          A)G�AG�@(��x������AN=qAG�@xQ��p��Ap�A�(�                                    BxiM��  T          A)p�A(�@(���G���33Ar{A(�@w���=q��ffA��H                                    BxiM�Z  �          A)G�A   @'���R�C
=Aj�RA   @X�ÿ��G
=A��                                    BxiM�   �          A)G�A"=q@�R���0��AF=qA"=q@<�Ϳ���N�RA��
                                    BxiMզ  T          A(��A!p�@p�����z�A[�A!p�@A녾�=q��p�A��                                    BxiM�L  T          A'\)Az�@*=q�"�\�`��As�
Az�@e�O\)��{A�33                                    BxiM��  �          A'�A�@C33��Q��+\)A��
A�@g
=���Ϳ�A��                                    BxiN�  �          A'�A��@S�
��Q�����A��RA��@\��?.{@o\)A��\                                    BxiN>  �          A%�A(�@W
=�Tz����A�{A(�@S�
?�G�@��
A�                                      BxiN�  �          A'\)A"{@�R�(��Tz�A\(�A"{@(�?B�\@��AX��                                    BxiN-�  T          A'�A (�@.{���H�   As
=A (�@E�>��?O\)A��H                                    BxiN<0  �          A&�RA=q@�p���G���A�p�A=q@�\)?!G�@_\)A�                                      BxiNJ�  �          A'
=A  @`�׿�ff����A�ffA  @c33?h��@�Q�A�                                      BxiNY|  �          A'
=A Q�@2�\�=p���G�AyG�A Q�@1G�?O\)@�ffAw�                                    BxiNh"  
�          A'�
A
=@R�\�+��l(�A���A
=@J�H?�\)@��
A��                                    BxiNv�  �          A'33A�H@J�H�\)�E�A�p�A�H@@  ?�33@ʏ\A�z�                                    BxiN�n  �          A=qAz�@B�\@3�
A�{A���Az�?�33@~�RA�z�@��R                                    BxiN�  �          A  @��
@XQ�@�=qA�z�A��@��
>�\)@��HBp�@Q�                                    BxiN��  �          A=qA ��@`  @z=qA�{A�=qA ��?z�H@��B�
@�\)                                    BxiN�`  �          A!A\)@��@Dz�A�G�A�(�A\)?�p�@���A��AL��                                    BxiN�  �          A"�RAp�@Vff@5A��HA�  Ap�?�33@�p�A�
=Az�                                    BxiNά  �          A'�
Az�@�G��(��W
=A��Az�@\)?�Q�A��A�                                    BxiN�R  �          A((�A�H@�p��
=�;�A͙�A�H@��
>��@&ffA�                                    BxiN��  �          A)A33@��R�b�\���
A�{A33@��Ϳ8Q��z=qB{                                    BxiN��  �          A(��A{@�33�g����HA��HA{@�33�Y����(�B �                                    BxiO	D  �          A(Q�A�
@���QG���33A��HA�
@�(����G
=A��                                    BxiO�  T          A'\)A@p  �.�R�r=qA���A@�(��Ǯ�
�HA�Q�                                    BxiO&�  
�          A'\)A@z�H�\)�\z�A�A@�z���Ϳ\)A�
=                                    BxiO56  �          A(��A��@����G����A�(�A��@�G���G���HA�G�                                    BxiOC�  �          A&ffA  @5��C�
��ffA�Q�A  @�Q쿓33��Q�A�p�                                    BxiOR�  �          A   @�z�?\)��ff�(p�@���@�z�@|����ff�{A�R                                    BxiOa(  �          A(�AG�?��������G�AK�AG�@����J�H���RA��                                    BxiOo�  �          A!G�A��@�
=�8Q�����Aܣ�A��@�=q����Y��A���                                    BxiO~t  �          A%G�A��@�z��33�MA�\A��@��
?�@J=qA�\)                                    BxiO�  �          A(Q�A{@�녿aG���=qB G�A{@��R@Q�A:�HA�p�                                    BxiO��  �          A&�\Az�@�=q��Q���\)A�p�Az�@�G�?\A33A��\                                    BxiO�f  �          A%p�A33@��AG���=qA�{A33@�=q�#�
�c�
Bz�                                    BxiO�  �          A$��A�H@�Q��,(��r=qA��A�H@�z�>�(�@��B��                                    BxiOǲ  �          A"{AG�@�
=�E����Bz�AG�@��=��
>�G�Bz�                                    BxiO�X  T          A!�@�@�  �tz�����B	ff@�@�
=����G
=B z�                                    BxiO��  �          A"=q@��\@����333��ffB�\@��\@�(�?��@Z=qB                                    BxiO�  �          A$z�Az�@�G��Mp���ffA�ffAz�@�Q쾅����HBQ�                                    BxiPJ  �          A$  A=q@���\)��p�AٮA=q@�Q쿕��=qB	�H                                    BxiP�  �          A$��A	@��R�dz����A��HA	@���:�H��=qBG�                                    BxiP�  �          A$��A
�H@�33�E���A�G�A
�H@��þ�  ��33B{                                    BxiP.<  �          A$��A=q@���������=qA�=qA=q@�녿�����z�B
�H                                    BxiP<�  �          A'
=A  @S33��ff�˙�A��A  @�(���:=qA���                                    BxiPK�  �          A&�HA
ff@W
=�����z�A��A
ff@��H�33�L(�A�\)                                    BxiPZ.  �          A%��A(�@}p�����¸RAǅA(�@��\�����(�B(�                                    BxiPh�  �          A#\)A
=@�z���
=��A�ffA
=@�����ffBG�                                    BxiPwz  �          A"{@�\)@s�
��{��p�A�  @�\)@�=q�\)�LQ�Bff                                    BxiP�   
�          A"�\@�z�@�����\)���HA�{@�z�@����
=q�D��Bz�                                    BxiP��  �          A"=q@�
=@�����H��(�A@�
=@θR�z��S�
B#G�                                    BxiP�l  �          A"=q@�33@I���ƸR�\)A��
@�33@�p��j=q��B�                                    BxiP�  �          A!@�ff@+��ᙚ�1p�A�Q�@�ff@�ff��(��ڏ\B&�\                                    BxiP��  �          A ��@�p�@1���{�7(�A�p�@�p�@��
��{���B.z�                                    BxiP�^  �          A ��@�p�@*�H��ff��A�Q�@�p�@�Q��y����G�BQ�                                    BxiP�  �          A
=@��@G���G��=qAtQ�@��@�  ��=q��(�B{                                    BxiP�  �          A
=@�z�?(����ff�^��@�G�@�z�@��R��{�({B$��                                    BxiP�P  �          A�@��?��
��=q�Zp�AT��@��@����Q��33B.�R                                    BxiQ	�  �          A�@��R?�\)��p��J
=AP  @��R@�G����
�
=B �H                                    BxiQ�  �          A�@�  @���H�G�A�
=@�  @�=q������p�B5\)                                    BxiQ'B  �          A=q@�
=@:=q��ff�9\)A��
@�
=@�33��p����B6��                                    BxiQ5�  T          A{@�@3�
��G��<{A���@�@����G��ᙚB6��                                    BxiQD�  �          A�@�=q@8Q�����7=qA�33@�=q@��������33B3�
                                    BxiQS4  �          Aff@�p�@+���(��6��A�@�p�@��
��\)���B.\)                                    BxiQa�            A=q@��?�33��  �  A-@��@�=q��z��ۙ�A��                                    BxiQp�  T          A{@��H@�
������A��@��H@��R�l����G�B=q                                    BxiQ&  "          A=q@�p�@\)���\��A�(�@�p�@����^{���Bz�                                    BxiQ��  �          A=q@�Q�@)�����	(�A�\)@�Q�@��H�P  ���
Bz�                                    BxiQ�r  |          A�@�ff@5���p�� �RA���@�ff@���;����RB=q                                    BxiQ�  �          A!�A@6ff�u����A��RA@����
�#�Aԣ�                                    BxiQ��  �          A\)@�33@dz���(���=qAÙ�@�33@��H��
�W�B�H                                    BxiQ�d  T          Aff@�  @K������z�A�@�  @��H�C33����B��                                    BxiQ�
  �          A�@�Q�@\(������z�A��@�Q�@�(��'
=�s�B�H                                    BxiQ�  
�          A ��@�\@�����\)�Ǚ�B\)@�\@�  �}p�����B"�                                    BxiQ�V  T          A!G�A Q�@�33�mp�����A�33A Q�@�=q�#�
�h��B�                                    BxiR�  �          A�R@�\@�  �Mp����B��@�\@��
=�?8Q�B {                                    BxiR�  
�          A�\@�=q@����333��
=Bp�@�=q@�z�?�R@g
=B �                                    BxiR H  �          A�\@�
=?�����(���HA�H@�
=@�����
��=qA��                                    BxiR.�  �          A�@���?�  ���R���A,z�@���@�(��w����
AܸR                                    BxiR=�  �          A��A\)?�(����R��\)AX(�A\)@����L�����RA�                                    BxiRL:  �          AA\)@c33�Vff���\A�  A\)@��ÿ�G���{A뙚                                    BxiRZ�  �          A�@��
?����{�	  A�R@��
@~�R�|(����\A֏\                                    BxiRi�  
�          A�H@�R?8Q�������\@���@�R@qG�������(�A�z�                                    BxiRx,  T          A�A
=@	����{��Aj�HA
=@����8Q����\A�z�                                    BxiR��  
�          A�
A�@B�\�l(���=qA���A�@��׿����  A�=q                                    BxiR�x  �          A�A   @$z���
=��ffA�=qA   @����,(���p�A�G�                                    BxiR�  
�          Aff@��?��������HAi��@��@�  ������ffB                                    BxiR��  "          A�A�?������
� G�A��A�@s33�k����Aɮ                                    BxiR�j  
�          A�A�?����G����@���A�@c33�n�R���RA���                                    BxiR�  
�          A33@�
=?������\�{A�@�
=@p  �}p���A�G�                                    BxiR޶  T          A\)A��?��������A&�RA��@n�R�J=q����A���                                    BxiR�\  �          A{@�
=?�{��33��RA33@�
=@w���ff��p�A���                                    BxiR�  
�          A=q@���?�\)���H���A@���@|���u���AԸR                                    BxiS
�  �          AG�@�p�?z�H��\)�p�@��@�p�@}p����H����A��\                                    BxiSN  T          A�@�
=?�  ��z����A��@�
=@�Q����H��p�A�z�                                    BxiS'�  �          A�@�z�?Q�����-�\@أ�@�z�@��
��p��(�A���                                    BxiS6�  T          A{@�G�?�ff�����\)A�@�G�@�33��ff��z�A��H                                    BxiSE@  �          A�@��H?�\)�ʏ\�$��A/\)@��H@�
=����p�B �
                                    BxiSS�  T          A{@��
?�������(�A6�R@��
@�Q�������{A�                                    BxiSb�  T          A��@�z�?�  ���(�AQ�@�z�@x���~�R��\)A�                                      BxiSq2  �          AG�@��?�����	{ATz�@��@��H�hQ����A�{                                    BxiS�  
�          A��@�33?�33�����  A,  @�33@�Q������
=A���                                    BxiS�~  "          AG�@�=q?�Q���\)��HA1��@�=q@��H������HA�\)                                    BxiS�$  �          A��@��?����Å�Af�R@��@�\)������33BG�                                    BxiS��  "          A(�@�ff?�33��{�"Atz�@�ff@�33������33B�\                                    BxiS�p  !          A��@�(�?�(��ȣ��$A
=@�(�@�{���\��Q�B�                                    BxiS�  
Z          A�@�?��R����z�Aw\)@�@�33��z���=qB(�                                    BxiS׼  �          A{@�R?��H�\��\As�@�R@��H��p���33Bz�                                    BxiS�b  �          A��@�=q?��R��(��
=As
=@�=q@�Q��~�R��(�B
=                                    BxiS�  �          A�\@��@�R����Q�A��
@��@�=q�hQ���
=B��                                    BxiT�  
�          A\)@�G�@'���  �
33A�
=@�G�@��H�U�����B(�                                    BxiTT  �          A��@���@����H�  A��H@���@���S33��=qA��
                                    BxiT �  �          A
=@�=q@8Q������=qA�Q�@�=q@�ff�@����
=B	�H                                    BxiT/�  �          A�H@�@Y�������Q�A�z�@�@����=p���=qB33                                    BxiT>F  �          A�@��@$z���z��
  A��@��@�  �QG�����B
=                                    BxiTL�  �          A�RA��?�����{���
AN=qA��@����P  ��p�A��                                    BxiT[�  
�          A33A�?�����ff��AIA�@��
�Q���z�A�G�                                    BxiTj8  �          A(�Ap�@����{��Ay�Ap�@�  �C33���A�ff                                    BxiTx�  �          Az�A Q�@��������A��
A Q�@��
�A���
=A                                    BxiT��  �          A��AQ�@
=q��Q�����AiAQ�@�=q�<����=qA�ff                                    BxiT�*  
�          A�A�R@aG��e����A�33A�R@��
���R��  A�(�                                    BxiT��  T          AG�A��@��
�L����(�A��HA��@��
=�^{A��                                    BxiT�v  
�          A��Aff@|(��N{����A�33Aff@�G��333��33A�                                    BxiT�  �          A   Az�@j=q�Z�H���A�Az�@��Ϳ��\��p�A�
=                                    BxiT��  T          A Q�AG�@N{�\������A�ffAG�@�G����\��G�A�                                    BxiT�h  T          A\)A
=q@Z�H�fff���A���A
=q@�G��������A�{                                    BxiT�  �          AffA
{@XQ��^{��(�A��A
{@�{���H����A�(�                                    BxiT��  "          AQ�A�\@hQ��X������A���A�\@�����\���A�(�                                    BxiUZ  
�          A�\A(�@fff�Z=q��A�ffA(�@�33��ff�ǮA�33                                    