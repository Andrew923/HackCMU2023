CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230220000000_e20230220235959_p20230221021629_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-21T02:16:29.122Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-20T00:00:00.000Z   time_coverage_end         2023-02-20T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxf��   T          @���x��@�\)?(��@��Cs3�x��@p��@<(�A�(�C�R                                    Bxf���  "          @�p��fff@��R?+�@ȣ�B���fff@}p�@Dz�A��CL�                                    Bxf�	L  T          @��
�dz�@�
=?@  @�G�B����dz�@�z�@P��A���C��                                    Bxf��  �          @˅�qG�@���?z�HA�B����qG�@w
=@W
=A��CW
                                    Bxf�&�  T          @�z��q�@���?��
A�
B��\�q�@w
=@Z�HBG�C^�                                    Bxf�5>  
�          @θR�w
=@��
?#�
@��B��w
=@��@G�A癚C0�                                    Bxf�C�  �          @˅���@��������?\)C)���@��?��RA�(�C	�                                    Bxf�R�  "          @�������@�\)<�>��CO\����@��\@��A���C��                                    Bxf�a0  
�          @�p���{@�(�>B�\?޸RC�=��{@z=q@�RA�(�C
��                                    Bxf�o�  �          @θR���\@���>�\)@�RC8R���\@�  @(Q�A�=qC	O\                                    Bxf�~|  �          @�p���\)@��?:�H@љ�CǮ��\)@o\)@@  A߮C
�=                                    Bxf"  "          @θR��=q@�z�?�z�A%��C(���=q@N{@L(�A��
C��                                    Bxf�  "          @�Q����@���?xQ�A	G�C
=���@j=q@N�RA��
C�                                    Bxfªn  "          @�������@��H?^�R@��C�f����@p��@J�HA�(�C
�H                                    Bxf¹  
�          @�����=q@�Q�?s33Ap�C{��=q@w�@Tz�A���Cn                                   Bxf�Ǻ  
�          @У����@��H?��\Az�C8R���@k�@S33A�C                                     Bxf��`  "          @���z=q@�Q�?=p�@��
B���z=q@~{@H��A���C��                                    Bxf��  
Z          @�G���p�@��R?h��@�\)C���p�@u@P��A��C	O\                                    Bxf��  �          @�G����\@���?�A%p�Cc����\@U�@QG�A��C��                                    Bxf�R  �          @����Q�@���?�p�AU�C�H��Q�@1G�@S33A�\)C�=                                    Bxf��  �          @љ���(�@\)?޸RAvffC�R��(�@&ff@_\)B(�C��                                    Bxf��  "          @�Q����@���?�z�AH��C�R���@1�@N{A��Cz�                                    Bxf�.D  �          @�Q����R@~{?��AE��CO\���R@0  @K�A�z�C��                                    Bxf�<�  �          @�{��p�@��?��A�\)C
�q��p�@(Q�@dz�B�
C��                                    Bxf�K�  
�          @��
��z�@��?�ffA<��C
���z�@=p�@L��A�G�Ck�                                    Bxf�Z6  
�          @�33��  @��\?��A=qCW
��  @AG�@;�AۮC��                                    Bxf�h�  "          @�{��@���?Y��@�33C����@E@0  A�z�C��                                    Bxf�w�  T          @�33��ff@��?�33A'�C
ٚ��ff@?\)@B�\A�
=C�                                    BxfÆ(  "          @�ff��{@�Q�?�  A�C���{@>�R@7
=A�33C޸                                    BxfÔ�  "          @�  ����@|��?���A ��C�
����@7�@<��A�CO\                                    Bxfãt  �          @ҏ\��p�@~{?s33A�Cu���p�@>{@333A�
=C�                                    Bxfò  �          @ҏ\��=q@�  ?�(�A+33C��=q@7�@B�\A܏\C��                                    Bxf���  �          @�G����@mp�?�p�A-�C�3���@'
=@:=qA�p�C�                                    Bxf��f  �          @��
��33@Vff?��Ap  C����33@ff@E�A陚C�{                                    Bxf��  "          @Ǯ���H@J=q?��HA\)C0����H?��@C33A��C�
                                    Bxf��  T          @Ǯ���@Q�?�A���C�����?���@L(�A�z�C��                                    Bxf��X  "          @ə���\)@]p�?�p�A~ffC(���\)@	��@N{A�G�C�
                                    Bxf�	�  �          @�\)��(�@\��?��HA\)C�R��(�@
=q@L��A�G�C.                                    Bxf��  �          @����=q@�Q�?�=qAK�
C	���=q@5�@H��A��\C�=                                    Bxf�'J  �          @�Q��XQ�@�>�Q�@Y��B��H�XQ�@��@0  A�Q�Cp�                                    Bxf�5�  T          @Å����@J�H@�
A��Cn����?޸R@W
=B�\C�R                                    Bxf�D�  �          @������@�?�=qA(��C�����@Fff@@  A���CaH                                    Bxf�S<  T          @��x��@�p�?E�@���C�R�x��@P  @.�RA�(�C
                                    Bxf�a�  T          @�ff�g�@��?�33Ac
=C�g�@;�@QG�B(�C                                      Bxf�p�  �          @��   @��>.{?�Bۣ��   @�  @8��A�=qB�q                                    Bxf�.  �          @�
=��ff@�
=>�
=@��HB�\)��ff@���@L(�BB�\                                    Bxfč�  "          @���\@�  >��@ ��BѨ��\@��
@C�
A�\)B֮                                    BxfĜz  �          @�
=��{@�  ?(��@��
Bγ3��{@�@Y��BBԀ                                     Bxfī   "          @��R����@��\��=q�$z�B��H����@��R@(��Aң�B�p�                                    BxfĹ�  "          @��Ϳ}p�@���>B�\?���B�z�}p�@�{@@��A��RBʸR                                    Bxf��l  
�          @�z��W�@w
=>\)?\C��W�@R�\@G�A�\)C��                                    Bxf��  "          @�G���=q@��>W
=@C}q��=q?���?��\AK
=C!#�                                    Bxf��  
�          @�����z�>L��?�=qA-G�C1�R��zᾸQ�?��A&�\C7�)                                    Bxf��^  �          @�33��\)?
=q?}p�A{C.����\)�#�
?���A+�C4�                                    Bxf�  �          @�����H?��?G�@��C.ٚ���H=�Q�?s33A=qC3)                                    Bxf��  �          @�ff��z�>�(�?333@�ffC/����z�=#�
?Q�@�33C3��                                    Bxf� P  �          @ƸR��{���
>�ff@�(�C40���{�u>\@`  C60�                                    Bxf�.�  T          @�{��p�>\)>�Q�@U�C2�R��p��u>\@a�C4��                                    Bxf�=�  �          @ȣ��Ǯ>�\)?\)@��HC1u��Ǯ��?�R@�C4G�                                    Bxf�LB  �          @�\)��?�>��@s33C.����>���?#�
@���C1@                                     Bxf�Z�  "          @�
=��=q?�
=?(�@�p�C)��=q?:�H?�{A$��C-+�                                    Bxf�i�  
�          @ə��Å?���?&ff@�p�C'B��Å?c�
?��RA5�C+��                                    Bxf�x4  T          @˅�\?�33?\(�@���C$��\?��\?ǮAc33C*p�                                    Bxfņ�  "          @�(�����?�z�?G�@���C"u�����?��
?�{Aj=qC'�q                                    Bxfŕ�  T          @˅�\?�(�?8Q�@�  C$:��\?�33?�(�AUG�C)O\                                    BxfŤ&  �          @�33���?�p�?E�@�
=C$����?��?\A]p�C)\)                                    BxfŲ�  �          @�=q��\)?�{?=p�@�Q�C"�R��\)?�G�?�ffAdQ�C({                                    Bxf��r  T          @ə���  ?��
?�
=A,��C%����  ?B�\?��
A���C,                                    Bxf��  �          @�Q���
=?\?��A'33C%�q��
=?E�?�p�A�Q�C,�H                                    Bxf�޾  �          @�  ��?�\)?�\)A%�C$����?^�R?�\A�\)C+�f                                    Bxf��d  "          @�\)��=q?�?��
A=�C"� ��=q?u@   A��C*��                                    Bxf��
  
�          @�z����?�\)?��A  C$ff���?h��?�Q�A�z�C+�                                    Bxf�
�  �          @�����?��?��HA5p�C*
=��>��
?ǮAk�C0޸                                    Bxf�V  �          @����p�?�  ?��A*�\C(���p�?
=q?���Aq��C.�                                     Bxf�'�  T          @��
��(�?���?���A ��C'^���(�?!G�?���An=qC-�
                                    Bxf�6�  
�          @�(����\?Ǯ?��A$(�C%  ���\?W
=?��HA��C+�{                                    Bxf�EH  �          @�������?��H?��HA6{C#�����?h��?��A�33C+�                                    Bxf�S�  "          @����H?��
?��AC�
C%L����H?5?��A�Q�C-�                                    Bxf�b�  �          @�z���z�?��
?�A/�C'���z�?��?��Ax  C.�f                                    Bxf�q:  �          @�z���?���?��A+
=C)+���>�G�?�ffAi�C/�q                                    Bxf��  T          @�z�����?Tz�?Tz�@�Q�C,�����>���?���A)p�C0�)                                    BxfƎ�  �          @�����
=?xQ�?�=qA!�C*�q��
=>���?�AT��C0��                                    BxfƝ,  T          @\��ff?�?�ffAl  C&���ff?   @�\A���C/�                                    Bxfƫ�  �          @��H��  ?�=q?�  Ad(�C&���  >�ff?��HA�  C/�\                                    Bxfƺx  �          @�(����H?��?���A��\C#�����H?��@=qA��C.h�                                    Bxf��  �          @\��(�?��?��A�33C&)��(�>�Q�@\)A�33C0Y�                                    Bxf���  �          @�����z�?��R?���AuC%=q��z�?�@�A�{C.�)                                    Bxf��j  �          @�����ff?�=q?�G�Ag33C&ٚ��ff>�G�?��HA��C/�\                                    Bxf��  �          @����?�z�?���Ao�
C&���>�@33A�Q�C/!H                                    Bxf��  �          @�33��z�?���?ٙ�A�p�C$+���z�?
=@��A��
C.�                                    Bxf�\  �          @�����?��
?�Q�A�C$�R���?�@{A�  C.�\                                    Bxf�!  �          @�=q���?���?�p�A`��C&�q���>�?�Q�A�33C/:�                                    Bxf�/�  �          @Å��  ?��?��
Ag�
C&u���  >�@ ��A�G�C/0�                                    Bxf�>N  �          @\��z�?:�H?�ffAE�C,����z�<��
?��RAbffC3�{                                    Bxf�L�  �          @�33��(�?B�\?���AR{C,�)��(�<#�
?�=qAp��C3��                                    Bxf�[�  T          @�33��{?#�
?�(�A8  C-�
��{���
?�\)APz�C4+�                                    Bxf�j@  �          @�p���
=?k�?��HA5�C+B���
=>k�?�  Aap�C1��                                    Bxf�x�  T          @������
?�33?�z�ATQ�C(޸���
>�{?��A��C0��                                    BxfǇ�  �          @�����\)?fff?��A"�HC+k���\)>�\)?���APz�C1\)                                    Bxfǖ2  T          @�\)��  ?�(�?�33A*�RC(� ��  ?�?���Amp�C/�                                    BxfǤ�  �          @�(���ff?z�H?�ffAG�C*����ff>�p�?��AQp�C0z�                                    Bxfǳ~  �          @��
��Q�?@  ?=p�@�{C,�f��Q�>��R?�G�A�C1\                                    Bxf��$  �          @������H?�  ?���A3�C%�{���H?@  ?�\A���C,�f                                    Bxf���  �          @���33?�{?�Q�A1��C$�{��33?Y��?�A�z�C+��                                    Bxf��p  �          @��
����?�p�?�
=A2�HC ������?�
=?�p�A���C((�                                    Bxf��  �          @��
����?�ff?��AJffC$������?:�H?�z�A�C,�\                                    Bxf���  �          @�=q���?��?���Ap��C(�q���>��?�
=A�Q�C1c�                                    Bxf�b  �          @�����ff?�z�?��RA>ffC&!H��ff?&ff?�  A�{C-p�                                    Bxf�  �          @�  ����@
=q>�(�@��HC�R����?ٙ�?�33AXz�C"�                                    Bxf�(�  �          @�  ��Q�@3�
����
=Cٚ��Q�@0  ?Tz�@�p�Cc�                                    Bxf�7T  �          @�=q��=q@P  �u�(�CW
��=q@9��?�p�Ac33CE                                    Bxf�E�  �          @�33��G�@X��>�ff@�
=C\��G�@1�@   A��HC�                                    Bxf�T�  �          @Å��G�@X��?z�@�
=C{��G�@.{@
=A�C��                                    Bxf�cF  �          @��
����@O\)?
=q@��HC������@'
=@ ��A��C+�                                    Bxf�q�  �          @�����  @U>�@���CG���  @.{?��RA�=qCs3                                    BxfȀ�  �          @�����p�@^{>��?���C�
��p�@?\)?�G�A�=qC��                                    Bxfȏ8  �          @�Q�����@h�ü#�
�uC�����@Mp�?ٙ�A��
C�                                    Bxfȝ�  �          @�{���@Z=q>��?�(�C�����@<(�?�p�A���C��                                    BxfȬ�  �          @�����  @0��?}p�A��C#���  ?�(�@
=qA�G�CxR                                    BxfȻ*  �          @�{���@�?���A-��C�����?˅@33A�G�C"�{                                    Bxf���  �          @��
����?���?Tz�A��C#:�����?��\?�p�Au�C)
                                    Bxf��v  �          @�=q���?ٙ�?z�HA$  C!�����?�ff?�z�A�z�C(��                                    Bxf��  �          @��
����?��?ǮA�33Cs3����?p��@��A�Q�C)ff                                    Bxf���  �          @�����z�?�?��A-C!����z�?�  ?ٙ�A��C)�                                    Bxf�h  �          @�����\)?��?�=qA]p�C&���\)?�?��
A��C.@                                     Bxf�  �          @������?�33?�
=AqG�C!�H����?J=q@�A�C+
=                                    Bxf�!�  �          @��R��(�@%�?c�
A�C�H��(�?�\)?��RA��
C�                                    Bxf�0Z  �          @�����p��k�@   A��
C6����p���{@
=A�  CC��                                    Bxf�?   �          @�ff�����ٙ�?aG�A��CG��������z�#�
��CI                                    Bxf�M�  �          @���p���(�?p��A�
CAE��p���G�>��@333CDT{                                    Bxf�\L  �          @��R��33>B�\?�A��HC1�H��33�8Q�?��A�33C<�                                    Bxf�j�  �          @������?�@
=A�ffC-����녾��@��A���C9Q�                                    Bxf�y�  �          @�����33?�  @�
A�(�C%�{��33=#�
@'�A�p�C3�\                                    BxfɈ>  �          @��R���
?���@AG�BG�C&f���
>aG�@aG�B!��C0�                                    Bxfɖ�  �          @������?���@%A�
=C&�f����B�\@4z�A�z�C6:�                                    Bxfɥ�  �          @�{��=q>��@=p�A��
C/+���=q��  @4z�A�(�C?�                                    Bxfɴ0  �          @��H��p�@L��?�\)A��CxR��p�@ff@:�HA��CO\                                    Bxf���  �          @�ff��
=@S33?�p�A���C����
=@Q�@C�
B��C.                                    Bxf��|  �          @��\��p�?�p�?�G�A�\)C���p�?s33@�RA��C)�{                                    Bxf��"  �          @������@(�?���A�  C  ����?�{@'�A�p�C'�)                                    Bxf���  �          @��R��=q@_\)@�A��\C���=q@��@\��B\)C�
                                    Bxf��n  �          @�p�����@L��@33A��\C������?�33@R�\B33C=q                                    Bxf�  �          @�����Q�@S33?��A�ffC�=��Q�@�@K�B�CE                                    Bxf��  T          @�33���@@��?�A��HC{���?�@AG�A�z�Cp�                                    Bxf�)`  �          @�33���R@7
=?�A���C�3���R?�@AG�A��HC xR                                    Bxf�8  �          @������R@-p�?�
=A�(�CW
���R?�33@.�RA�z�C!��                                    Bxf�F�  �          @�p����
@W
=@A�(�Cu����
@�\@X��Bp�C�                                    Bxf�UR  �          @�p����H@R�\@�\A���C�{���H?��@b�\B(�C��                                    Bxf�c�  �          @�����\@N�R@1G�A�ffC�3���\?�{@|(�B'�Cu�                                    Bxf�r�  �          @��
��{@4z�?���AQG�CQ���{?�z�@p�A�
=C�H                                    BxfʁD  
�          @����@HQ�>�ff@���C�)���@%?�=qA�  C�
                                    Bxfʏ�  T          @�=q�x��@\��@33A��HC
h��x��@�@g
=B�Cc�                                    Bxfʞ�  "          @���xQ�@b�\@%AхC	���xQ�?��R@z=qB'p�C�f                                    Bxfʭ6  
�          @�z�����@\(�@
=qA�{C#�����@@^�RB�C��                                    Bxfʻ�  "          @�����@W�?��Axz�C�����@�
@9��A��
CO\                                    Bxf�ʂ  �          @���s33@N{@.�RA�  C�s33?��@x��B,�HC��                                    Bxf��(  �          @�����\@�p�?z�HAQ�C����\@Q�@0��A��C޸                                    Bxf���  T          @��
����@�p��333�ʏ\C� ����@��?�(�AV{C�f                                    Bxf��t  �          @�z��e�@���{�PQ�B����e�@�G�?c�
A
=B��                                     Bxf�  �          @����:�H@A���z��2=qC���:�H@�=q�����B��                                    Bxf��  �          @�����@P���tz��-(�B��q���@�(��������
B��H                                    Bxf�"f  T          @��R�>{@%�����>��C��>{@���,����z�B�                                      Bxf�1  T          @�z��7�@�(����
�q�B��f�7�@���?Q�A (�B�8R                                    Bxf�?�  �          @�p��%@���(����B�ff�%@�G�>�ff@��HB�#�                                    Bxf�NX  �          @�33�@��@�G�� ���̣�B�\�@��@��R�����O\)B�z�                                    Bxf�\�  �          @��H�+�@����(���p�B�Q��+�@�z�.{��
=B��                                    Bxf�k�  
�          @�(��#33@�{��{��Q�B��#33@�  ?�@��RB�3                                    Bxf�zJ  �          @�
=�>{@�  �p���{B��R�>{@�(��L�Ϳ�z�B�{                                    Bxfˈ�  
�          @�\)�>�R@�33�����z�\B���>�R@��?:�H@��B�                                    Bxf˗�  �          @�p��XQ�@��׿aG��Q�B����XQ�@��?�=qAPQ�B��f                                    Bxf˦<  T          @����h��@�Q쿧��D��B����h��@�33?p��A��B���                                    Bxf˴�  �          @�
=�q�@���xQ���HB��q�@���?��
A>ffC O\                                    Bxf�È  �          @ȣ��~�R@�(���33�O33C���~�R@���?J=q@�C��                                    Bxf��.  T          @�33�~{@��\�u��C^��~{@�Q�?��\A9�C��                                    Bxf���  	�          @�z��}p�@�zῐ���#
=C�}p�@�z�?��A$z�C
=                                    Bxf��z  T          @�=q�|��@��׿��H�0Q�C�H�|��@��?��\A��C^�                                    Bxf��   T          @�z��{�@��\��{�&=qC���{�@�33?��\AG�C�                                    Bxf��  
(          @��|��@�=q�aG��(�Ch��|��@���?�{A-p�CǮ                                    Bxf�l  �          @�
=����@����#�
��\)C�=����@mp�?У�A���C	L�                                    Bxf�*  T          @�������@����������C������@x��?��
AN=qC�                                    Bxf�8�  �          @�ff��=q@|(��(����C���=q@s�
?�A>�\C��                                    Bxf�G^  
�          @���u�@�������\Cff�u�@l��?��A���C�                                    Bxf�V  T          @�G��aG�@�Q쿽p��pz�C���aG�@�\)?�@�  C 5�                                    Bxf�d�  �          @����i��@��\��Q��\��C �=�i��@�  ?333@ָRB�{                                    Bxf�sP  
�          @�z��c33@�G������-�C ��c33@�=q?}p�A�RB��q                                    Bxf́�  
(          @�z��_\)@�33����2�RB�z��_\)@�z�?xQ�AQ�B�                                    Bxf̐�  "          @���hQ�@�G��}p��ffC ���hQ�@���?�=qA)G�C ��                                    Bxf̟B  �          @����`  @�녾�33�aG�B�  �`  @���?�z�A�  Cc�                                    Bxf̭�  �          @�{�Z�H@�zΐ33�:ffB��f�Z�H@�ff?^�RA��B��                                    Bxf̼�  "          @�G��Tz�@��H�Y����B��
�Tz�@���?�{A:{B���                                    Bxf��4  T          @�Q��Y��@�  ��33�8(�B�.�Y��@���?h��AG�B��                                    Bxf���  �          @�Q��J�H@�{��G��"=qB���J�H@�p�?�{A2=qB�\)                                    Bxf��  "          @�(��2�\@�Q쿞�R�C33B�(��2�\@��?��A#�B��q                                    Bxf��&  T          @�G��-p�@�  �fff���B� �-p�@�p�?�p�AM�B�L�                                    Bxf��  �          @���%@�
=�p����\B��%@�p�?�
=AG\)B�{                                    Bxf�r  �          @�p����H@��������v=qB�Ǯ���H@�p�?O\)A  B���                                    Bxf�#  "          @��
��p�@�G���=q�ep�B�LͿ�p�@�(�?uA%B��H                                    Bxf�1�  �          @���B�\@�{� �����B�{�B�\@�z�����RB��{                                    Bxf�@d  �          @��
@�\?����k��TB(@�\@O\)�!���Bg\)                                    Bxf�O
  
�          @�(�?��@
=q�b�\�W�Be�?��@^{�G����RB�L�                                    Bxf�]�  �          @�(�?��\@$z��j�H�T  B��{?��\@x���{�癚B��                                    Bxf�lV  �          @���>���@*�H���H�b�B�B�>���@����#�
��
=B���                                    Bxf�z�  
�          @�  ?   @1G��n{�S��B�� ?   @��H�(���  B�
=                                    Bxf͉�  �          @�p��!G�@~�R���
��\)B���!G�@o\)?�33A���B��                                    Bxf͘H  �          @���>\@G��1��&(�B���>\@�  �������RB�\                                    Bxfͦ�  �          @�p��!�@w��У���B�L��!�@�{>#�
?���B��                                    Bxf͵�  
�          @�����@�p������Q�B� ���@�{>�p�@�p�B�\                                    Bxf��:  �          @�33����@�\)�
=q�ȣ�B�.����@�  �.{��
=Bޅ                                    Bxf���  T          @����.�R@s�
��{����B�.�.�R@�����
�uB�z�                                    Bxf��  �          @���Vff@_\)������Cٚ�Vff@tz�=L��?�C:�                                    Bxf��,  �          @��H�1�@x�ÿ޸R���B��1�@�Q�=u?333B�.                                    Bxf���  "          @�\)�A�@}p��Ǯ��
=B��3�A�@�  >�  @-p�B��)                                    Bxf�x  
�          @�G��333@c33�(�����C @ �333@��Ϳ   ��
=B��                                    Bxf�  �          @����ff@i���"�\��z�B��)�ff@�(��G��G�B�L�                                    Bxf�*�  �          @��H� ��@�p�����
=B�#�� ��@��
>�@�B���                                    Bxf�9j  "          @���?\)@\���
�H�˅C���?\)@�G������B�                                      Bxf�H  
�          @�(��N�R@333�Q��{C\�N�R@z=q������C�                                    Bxf�V�  
�          @��Z=q@<���AG��Q�C.�Z=q@|(��\��=qC��                                    Bxf�e\  T          @��R�U@L(��9��� �Cc��U@�33�����_33C.                                    Bxf�t  T          @�  �h��@C�
�0  ��=qC��h��@{���p��O33C�
                                    Bxf΂�  �          @�ff�XQ�@^{����G�CG��XQ�@�p��G���
C
                                    BxfΑN  �          @�ff�-p�@����z�H�(z�B��-p�@���?}p�A(��B�=                                    BxfΟ�  "          @�ff�5@�녿W
=�B��)�5@��?�\)A>=qB�=                                    Bxfή�  "          @����S�
@i����
=����C:��S�
@��
��  �.�RC ��                                    Bxfν@  
�          @��\�p��@C�
�(��£�C�
�p��@l�Ϳ=p����RC�                                    Bxf���  �          @��q�@8���'���RC�H�q�@mp������Lz�C��                                    Bxf�ڌ  �          @�(��x��@1G��(��؏\C���x��@a녿���;\)C	��                                    Bxf��2  
�          @����X��@��׿�{�8z�C p��X��@�33?:�H@��HB��
                                    Bxf���  
�          @��H�W�@��H���
�*=qB��q�W�@�(�?Tz�A��B�33                                    Bxf�~  "          @���I��@�  �aG����B���I��@�
=?��A,Q�B�W
                                    Bxf�$  �          @�G��Mp�@��R�.{����B�aH�Mp�@�33?���AHz�B�                                    Bxf�#�  	.          @����J=q@�G���p��w
=B����J=q@���?�G�A|��B���                                    Bxf�2p  �          @����QG�@��Ϳ���B���QG�@�\)?��A\��B�G�                                    Bxf�A  
�          @����N{@�G���p��|��B�Ǯ�N{@�=q?�33Ar{C Y�                                    Bxf�O�  �          @�z��^�R@��\=u?&ffCp��^�R@mp�?��HA��RC0�                                    Bxf�^b  
(          @���l��@p�׿5����C}q�l��@n�R?^�RA\)C�                                     Bxf�m  �          @�G��hQ�@s�
�����
=C�
�hQ�@mp�?��A4��C^�                                    Bxf�{�  
]          @���mp�@qG���
=��Q�C�{�mp�@g�?��AF{C                                    BxfϊT            @�G��w
=@g
=��  �0  C��w
=@Y��?�(�AU��C
�{                                    BxfϘ�  T          @��
����@^�R=L��?
=qC8R����@J=q?�Q�A{�
C�H                                    Bxfϧ�  	�          @���fff@��
��z��@  C#��fff@x��?��Ak\)C�=                                    Bxf϶F            @����O\)@��R����˅B�
=�O\)@�z�?�33A���C                                     Bxf���  �          @�ff�;�@�=q>��@��HB�p��;�@���@��A��B�=q                                    Bxf�Ӓ  "          @�=q�@��@��?
=q@��\B����@��@��@
=A�Q�B�{                                    Bxf��8  "          @�p��?\)@���>��
@W�B����?\)@�Q�@A���B�W
                                    Bxf���  �          @��\�G
=@�33����-p�B�
=�G
=@�=q?˅A�
=B�\)                                    Bxf���  T          @��\�:�H@�\)�z���p�B�Q��:�H@�=q?���A`  B��                                    Bxf�*  �          @����#�
@��
��  �%B�z��#�
@�=q?ٙ�A�(�B�=                                    Bxf��  T          @����(Q�@��\��G�����B�B��(Q�@�33?�G�A~�\B�\                                    Bxf�+v  "          @��R�Q�@��?!G�@�z�B����Q�@mp�@�A�  C�                                    Bxf�:  T          @�Q��J=q@�\)>k�@=qB�k��J=q@���?�p�A�G�C &f                                    Bxf�H�  "          @��H�U�@u�?�=qAk33C�q�U�@?\)@.�RA�33C
                                    Bxf�Wh  
(          @����@  @q�@�\A�\)C p��@  @*�H@W�BC
Q�                                    Bxf�f  �          @�{�Fff@b�\@Q�A���C5��Fff@=q@VffB!C
=                                    Bxf�t�  "          @�\)�G�@vff?��A��RC��G�@9��@AG�Bp�C	)                                    BxfЃZ  �          @�\)�L��@n{?�A��C�q�L��@-p�@G�BffC�R                                    BxfВ   T          @�G��N{@s33?�p�A��C=q�N{@4z�@E�B�
C
�R                                    BxfР�  
�          @��
�?\)@�{?�p�A���B�(��?\)@Q�@@  B33Cn                                    BxfЯL  Q          @���0��@���?s33A!p�B�3�0��@s�
@(��A�(�B��                                    Bxfн�  �          @����)��@�>\)?�  B�
=�)��@�Q�?���A�  B���                                    Bxf�̘  "          @�ff�=q@��׿h���\)B�=�=q@�Q�?�G�A+�B�R                                    Bxf��>  �          @��\���@�  �����\)B�W
���@�������
=B���                                    Bxf���  
�          @�
=��Q�@�z���\����B�B���Q�@�(�����B���                                    Bxf���  "          @�  ��z�@���#�
���
B�LͿ�z�@���z��,(�B�                                    Bxf�0  �          @���� ��@�G�����Bݞ�� ��@��>.{?˅B��                                    Bxf��  
]          @�{��
@�  ��z����B�(���
@��>�Q�@S33B��                                    Bxf�$|  
�          @ȣ��5�@��������F=qB�#��5�@���?Q�@�z�B�=q                                    Bxf�3"  
�          @�
=�tz�@�=q�\)���C c��tz�@�p�?�=qAF�RCG�                                    Bxf�A�  "          @�
=���H@�zᾳ33�P  Ch����H@�p�?���AX��C�=                                    Bxf�Pn  
�          @�p��w�@��R?#�
@�\)Cff�w�@�33@��A���CG�                                    Bxf�_  
�          @Ǯ�?\)@�p�=��
?:�HB��?\)@�  @�\A�  B�Q�                                    Bxf�m�  �          @�녿˅@���>8Q�?��Bх�˅@�  @p�A��B�.                                    Bxf�|`  "          @˅��
@�(�?+�@���B����
@�{@6ffA�G�B�
=                                    Bxfы  �          @�����@���?���A5�B۔{��@���@P��B z�B�(�                                    Bxfљ�  
�          @����
�H@��\�Tz���
B�ff�
�H@��?�  AF{B�                                    BxfѨR  T          @�ff���H@��\�N�R���B�B����H@��R��  �C33B�p�                                    BxfѶ�  �          @�33���R@�  �G��z�B㙚���R@�33��Q��<z�Bܽq                                    Bxf�Ş  "          @�=q�u@��R�HQ��	�B�.�u@�녿�(��K33B�ff                                    Bxf��D  �          @�p���=q�
=����
=C�T{��=q�#�
��
=­�CR�                                    Bxf���  T          @���>���2�\����k�C�}q>���\(���=q¡p�C�Ф                                    Bxf��  �          @��>#�
��33��Q�ffC�n>#�
=����¯�RB�H                                    Bxf� 6  T          @�  >.{���G��fC�E>.{�L����ff°\)C�t{                                    Bxf��  T          @�G��z�z���¢ǮCaͿz�?��R��Q�33Bި�                                    Bxf��  %          @��׿�>���~�R ��C&f��?�\)�b�\�u�B���                                    Bxf�,(  
�          @���Q�@�33�#33��=qB�B��Q�@�ff����B�(�                                    Bxf�:�  
�          @��
�{@�p��
=���B����{@��H�#�
����B㞸                                    Bxf�It  �          @Ǯ�'
=@�������z�B���'
=@���#�
���B�                                     Bxf�X  �          @�
=��H@�33�n�R�ffB�(���H@�p���{����B��                                    Bxf�f�  �          @�ff��@�p��N�R���B�\��@��ÿ��\�>�\B�                                     Bxf�uf  	�          @�=q�   @��<(���Q�B�8R�   @���G��{B�aH                                    Bxf҄  �          @���33@���>{��G�B枸�33@��
�z�H�{B��=                                    BxfҒ�  T          @�=q��R@���#33�ƸRB� ��R@�  �����B�p�                                    BxfҡX  �          @Ǯ�7�@�Q��Q�����B���7�@��þ\�]p�B��H                                    Bxfү�  T          @�  �7
=@���	�����B��7
=@�G������HB��                                    BxfҾ�  �          @ȣ���@���HQ����B��)��@�
=��\)�$��Bߏ\                                    Bxf��J  �          @ȣ�� ��@�G��J=q��ffB�aH� ��@�������0��B�B�                                    Bxf���  �          @��H�<��@�  �>{���
B���<��@�Q쿅��33B�W
                                    Bxf��  	�          @���X��@���Dz�����B��X��@�����p��0��B�(�                                    Bxf��<  �          @θR�@��@�33�A���B�{�@��@��
�����Q�B�G�                                    Bxf��  
Z          @�{�8��@��R�8���ՙ�B�aH�8��@�p��fff� ��B���                                    Bxf��  "          @ʏ\�C33@���.{��ffB��C33@�
=�L����=qB�Q�                                    Bxf�%.  "          @�  �:=q@��
�Vff�=qB�p��:=q@��ÿ�ff�g\)B�Ǯ                                    Bxf�3�  �          @�p��0  @�G��_\)�	��B�u��0  @�  ��(���p�B�Q�                                    Bxf�Bz  	`          @�ff�0  @���Z�H�33B��0  @����У��t(�B��                                    Bxf�Q   �          @����ff@��
�x���\)B�u��ff@���
=q����B�\)                                    Bxf�_�  �          @��
�
�H@����j=q��
B�z��
�H@�p�������ffBߨ�                                    Bxf�nl  
�          @��R�(�@��H�Z=q�ffB�{�(�@��׿�����\B��                                    Bxf�}  	�          @��R��Q�@���i����B����Q�@�Q��33���B�p�                                    BxfӋ�  
�          @�녿�G�@p����(��6Q�B���G�@�\)�!���G�BոR                                    BxfӚ^  "          @�=q�޸R@xQ��|���+p�B�LͿ޸R@����z���(�B�.                                    Bxfө  
�          @��ÿ���@a����HffB֣׿���@��
�8����z�B̀                                     Bxfӷ�  �          @��R�+�@Z=q��Q��Q�B�.�+�@����AG�� {B��f                                    Bxf��P  
�          @���\@|(�����6Q�B���\@��
�=q�ȣ�B�aH                                    Bxf���  
Z          @��
�n{@�Q��tz��+p�B�B��n{@��H�
�H����B���                                    Bxf��  "          @�Q쿑�@o\)�w��2��B��)���@����
�ƸRB�\)                                    Bxf��B  "          @��Ϳ5?�z�������B�3�5@N{�x���F�
B�                                      Bxf� �  �          @�  �=p�?!G����H(�C���=p�@�\����sp�B�\                                    Bxf��  	�          @��H�n{?��H���\B�  �n{@5���(��X�RB�aH                                    Bxf�4  �          @��ÿ@  ?�z���  �3B�.�@  @>�R�}p��PB�p�                                    Bxf�,�  �          @�  �#�
?��H��\)��B�aH�#�
@@���{��OB���                                    Bxf�;�  "          @�ff�B�\?������B�R�B�\@I���l(��C
=B�
=                                    Bxf�J&  �          @��
=?��H��  (�B�.�
=@J=q�hQ��A��B�8R                                    Bxf�X�  T          @��\��G�?�p���ff�}��B����G�@^�R�n�R�4B��f                                    Bxf�gr  �          @��%@�\)���R��Q�B�.�%@�zᾨ���W�B��                                    Bxf�v  T          @�p��@��@��\�8Q���HB��H�@��@�=q?�z�A{�
B�ff                                    BxfԄ�  
(          @Å�Dz�@�
=�G���33B��
�Dz�@�?��A�
B�Q�                                    Bxfԓd  "          @�z��E�@��H��{�)�B�Q��E�@��?333@���B�                                    BxfԢ
  �          @���3�
@�����)p�B�33�3�
@�p�?Q�@�\B�3                                    Bxf԰�  "          @�Q��R@j=q���
�L�B�LͿ�R@�Q��E���
B�#�                                    BxfԿV  T          @�녿��@�
=�P���
=B�����@������H�b�HB�                                      Bxf���  "          @�
=�G�@�
=��p���p�B�\)�G�@��H���
�O\)B��
                                    Bxf�ܢ  
�          @�����=q@�Q������z�B�Q��=q@�ff��z��0  Bה{                                    Bxf��H  
�          @�Q���
@��H���\�D��B�Ǯ��
@�ff?(�@�p�B��f                                    Bxf���  T          @�
=���@�ff����0��B��
���@�Q�?B�\@�B�\)                                    Bxf��  T          @�\)��R@�(����\�D��B�  ��R@�  ?�R@�ffB�.                                    Bxf�:  
�          @��R��\@�G���p��f�\B��)��\@�\)>\@l(�B�ff                                    Bxf�%�  
�          @�Q����@�녿�33�Z=qB�����@�
=>�@�
=B��H                                    Bxf�4�  T          @�Q����@�=q?�z�A��RB�k����@��
@e�B��B�Q�                                    Bxf�C,  
�          @�ff�	��@�ff@A�A���B���	��@i��@�Q�B;=qB���                                    Bxf�Q�  "          @�=q�\)@�  @.{AˮB�.�\)@�G�@��B,{B��                                    Bxf�`x  T          @�33�
=@�ff@2�\A�{B�q�
=@~�R@��B-=qB�G�                                    Bxf�o  �          @ƸR�\)@�@4z�A�Q�B陚�\)@mp�@�G�B/ffB��3                                    Bxf�}�  �          @�G��'�@��H@?\)A�Q�B��
�'�@dz�@�p�B3�HB�p�                                    BxfՌj  
�          @Ϯ�:�H@��R@W�A�Q�B��:�H@U�@��B;��CG�                                    Bxf՛  �          @�Q��?\)@�
=@UA��HB����?\)@Vff@��RB9\)CǮ                                    Bxfթ�  
�          @љ��!G�@��@S�
A�RB��!G�@l(�@�G�B;��B�Ǯ                                    Bxfո\  T          @Ϯ�H��@�G�@Z=qA��
B�u��H��@J=q@��RB:Q�C�{                                    Bxf��  T          @У��Mp�@�G�@X��A�33B��=�Mp�@K�@�{B8p�CQ�                                    Bxf�ը  
�          @���L��@��@U�A�(�B�  �L��@S�
@�p�B5��C�                                    Bxf��N  T          @ҏ\�Z=q@���@K�A���B����Z=q@Vff@���B-�C��                                    Bxf���  T          @�G��c�
@�33@@  A�ffB�k��c�
@W�@��\B&  C��                                    Bxf��  T          @�(��e�@�@B�\AڸRB��H�e�@[�@���B&G�C33                                    Bxf�@  
�          @��
�\(�@�{@J=qA�Q�B��{�\(�@Z=q@�Q�B+�HCJ=                                    Bxf��  "          @ҏ\�]p�@���@QG�A�z�B�� �]p�@O\)@�=qB/C�
                                    Bxf�-�  "          @���W�@��@W�A�B�Ǯ�W�@J=q@�z�B4z�C�{                                    Bxf�<2  �          @���b�\@�=q@Z�HA���CW
�b�\@?\)@�(�B3��C�
                                    Bxf�J�  "          @�G��Tz�@��H@7
=A��B����Tz�@j=q@�Q�B#�C:�                                    Bxf�Y~  
�          @�  �S33@���@{A��RB����S33@~{@|��B��C                                    Bxf�h$  T          @�Q��b�\@�(�@8��A�
=B��)�b�\@]p�@�
=B!��C�                                    Bxf�v�  T          @љ��p  @�{@@  A�(�C33�p  @P  @�Q�B"�C)                                    Bxfօp  
�          @ҏ\�u@�p�@@��AمC���u@N�R@�Q�B!\)C�                                    Bxf֔  �          @�=q�w
=@�p�@=p�A��C
�w
=@P  @��RB\)C�)                                    Bxf֢�  
�          @�
=�y��@�  @9��A�\)C��y��@G
=@�33B��Cff                                    Bxfֱb  �          @�\)�u�@hQ�@VffB��C�=�u�@Q�@��\B/=qC!H                                    Bxf��  �          @�z��>�R@AG�?�33A�Q�C���>�R@�H@�\A�  C�3                                    Bxf�ή  �          @x��=�G�?���0  �^�B�p�=�G�@*=q�G����B�W
                                    Bxf��T  "          @�
=?xQ�?�\�Z=q�m��Bt��?xQ�@0  �+��*\)B��                                    Bxf���  �          @�  �\@��\�HQ����B���\@�녿��H�j=qB��f                                    Bxf���  
Z          @�(�?�@�  � �����B���?�@�녿xQ��-G�B�{                                    Bxf�	F  T          @�33@ ��?�������b�\B��@ ��@>�R�s�
�1z�BG��                                    Bxf��  �          @���@U�>L����  �a33@\(�@U�?�  ��{�M{A��                                    Bxf�&�  �          @��@Z=q?E���ff�Z�
AJ�H@Z=q@G����R�=��Bz�                                    Bxf�58  T          @Å@^{?Tz���ff�Y  AV=q@^{@z���ff�;Q�BG�                                    Bxf�C�  T          @��@S�
?�ff���
�P��A�Q�@S�
@N{��=q�%z�B0�
                                    Bxf�R�  
�          @�
=@L(�@
�H���
�Np�B	G�@L(�@dz��}p��G�BA(�                                    Bxf�a*  �          @�\)@l��?\(���(��P\)AQ�@l��@���(��4�B ��                                    Bxf�o�  
�          @�  @aG�@������;�B	�@aG�@l(��fff�  B9\)                                    Bxf�~v  
�          @��@n�R?�{���R�>�\A�ff@n�R@9���p  ���BQ�                                    Bxf׍  
�          @�G�@r�\@ff���3=qA���@r�\@fff�`  ��B.{                                    Bxfכ�  
�          @�G�@\)@$z���(��$�B=q@\)@n{�I����  B+                                    Bxfתh  
�          @�G�@s33@g��Z�H�\)B.ff@s33@���	�����RBG                                      Bxf׹  �          @ə�@�
=@QG��Vff� G�B�@�
=@�(��(�����B1z�                                    Bxf�Ǵ  T          @�G�@n�R@n{�Z�H���B3�@n�R@��\�Q���z�BKQ�                                    Bxf��Z  
�          @ʏ\@fff@p  �g
=�
�B8��@fff@�p��33��Q�BQQ�                                    Bxf��   "          @���@U�@qG��p���=qBB\)@U�@�\)�����\)B[��                                    Bxf��  �          @�Q�@L(�@p���u��
BF��@L(�@���!���
=B`p�                                    Bxf�L  
�          @ƸR@:=q@l(���G��"BO
=@:=q@�\)�/\)��  Bi��                                    Bxf��  
�          @��
@-p�@c�
��p��+��BR��@-p�@����9������Bn��                                    Bxf��  
�          @�z�@&ff@�
=�g����Bip�@&ff@���(���33B|(�                                    Bxf�.>  �          @��
@$z�@���p���\)Bh
=@$z�@����
=���B|33                                    Bxf�<�  T          @��
@G�@���l(���By  @G�@�
=��R���B���                                    Bxf�K�  
(          @Å@�R@�
=�j�H�ffBn33@�R@��
�\)��=qB�L�                                    Bxf�Z0  X          @�
=@E@��?
=q@��Bjff@E@�p�@G�A�Q�Bb(�                                    Bxf�h�  
�          @�{@C�
@��R�����~{Bc@C�
@�
=�����\Bi\)                                    Bxf�w|  T          @��@>{@���%��У�B_33@>{@���
=�8��Bk�                                    Bxf؆"  
�          @��@8Q�@���
=��\)Bf�
@8Q�@����h����\BqG�                                    Bxfؔ�  T          @��@,��@�G������
BrG�@,��@�{�z���z�Bz                                      Bxfأn  
^          @��R@%@��H��
=����B{�@%@�33��G����\B�L�                                    Bxfز  
�          @�ff@!G�@����!G����Bx�@!G�@�����G���B�(�                                    Bxf���  
�          @�\)@6ff@����8������Bdz�@6ff@�����(��dQ�Br�                                    Bxf��`  T          @�Q�@9��@��J�H���B\��@9��@�p�������
=Bm�
                                    Bxf��  T          @��R@,(�@���0����RBjz�@,(�@��ÿ����UBwG�                                    Bxf��  T          @�ff@�H@�Q���\��\)B�u�@�H@�z�   ��=qB���                                    Bxf��R  �          @�p�@��@��Ϳ�(���  B��@��@�  �����u�B��\                                    Bxf�	�  
Z          @�{@
=q@�������B���@
=q@�Q쾮{�P  B�G�                                    Bxf��  
�          @�ff@��@��\�
=��ffB��@��@�\)�����33B�8R                                    Bxf�'D  T          @�ff@�@�p�� ����{B�p�@�@��׾�(���p�B�{                                    Bxf�5�  �          @�@�\@�  ������G�B��=@�\@�녾��� ��B��3                                    Bxf�D�  
�          @�@(�@�
=�Ǯ�v�\B�aH@(�@�ff    <�B��                                    Bxf�S6  
V          @��@{@��R��G�����ByG�@{@�Q쾮{�`  B(�                                    Bxf�a�  
�          @���@��@���
=��B��H@��@�  �\)��=qB�                                    Bxf�p�            @���@G�@��Z�H�	�B{\)@G�@�
=�G����B���                                    Bxf�(  
�          @���@  @����c�
�p�By
=@  @���(�����B��                                     Bxfٍ�  "          @��@
=q@��R�0  ��B��@
=q@��ÿ�p��;\)B�aH                                    Bxfٜt  �          @\?+�@�p�?�33A.�\B��?+�@�33@1�A�  B�                                    Bxf٫  T          @\?�\)@�33?B�\@�  B��R?�\)@�z�@Q�A���B���                                    Bxfٹ�  T          @�=q?���@���>��?�33B��?���@��
?�A���B�k�                                    Bxf��f  �          @�Q�?�@��R>�33@W
=B�G�?�@�(�?�Q�A���B�=q                                    Bxf��  �          @��
?�\@��H<�>�  B��H?�\@��?˅A|(�B�u�                                    Bxf��  
�          @��@  @�=q>.{?�p�B�{@  @��?�z�A�G�B���                                    Bxf��X  
�          @�z�@�@���333��Q�B��@�@�
=?J=q@�(�B���                                    Bxf��  �          @���@'
=@�  ��Q��`��B~(�@'
=@���?�\)A/\)B|=q                                    Bxf��  
�          @��H@Z=q@�ff��=q��z�BR�@Z=q@��ÿ�����BZ                                      Bxf� J            @�ff@(�@�  ����
=B�z�@(�@�{?z�HA�
B�
=                                    Bxf�.�  "          @��@1G�@�����
�E�Bv�@1G�@�Q�>B�\?�=qBx��                                    Bxf�=�  T          @�
=@Tz�@�  ��G��Dz�B\ff@Tz�@�p�=���?xQ�B`                                      Bxf�L<  
�          @�33@7
=@�33��G���z�Bwff@7
=@�Q�?�ffA\)Bu�
                                    Bxf�Z�  
�          @�Q�@$z�@�ff�L�Ϳ�=qB�ff@$z�@�G�?���AK33B�#�                                    Bxf�i�  �          @�z�@�@�ff�}p��
=B���@�@�Q�?��@�\)B��                                    Bxf�x.  �          @�@z�@�Q��   ����B�@z�@���u�33B�W
                                    Bxfچ�  "          @�\)@��@�  �ٙ���z�B��@��@�  ����p�B�=q                                    Bxfڕz  
�          @��@��@�ff�L�Ϳ��B��H@��@�G�?�{AM�B�                                    Bxfڤ   "          @�G�@+�@�(�>���@HQ�B~
=@+�@��H?��
A���Bx�H                                    Bxfڲ�  �          @���@��@�p�>�=q@(��B�W
@��@���?޸RA��B�
=                                    Bxf��l  �          @�33@*�H@��?�{A'�B}�@*�H@��@!G�A�G�Bt�                                    Bxf��  "          @��@(��@�(�?O\)@���Bz�@(��@��R@�RA�
=Bw��                                    Bxf�޸  T          @��?���@ff���
�X{BN?���@W��U�#�Bt�R                                    Bxf��^  
�          @�
=?��?�Q����#�Buz�?��@Q������R��B�(�                                    Bxf��  "          @���?z�H@   ��z��y��B�8R?z�H@r�\�����>G�B�                                      Bxf�
�  &          @��R?��@p  �����3p�B�� ?��@�ff�8Q���{B���                                    Bxf�P  �          @\?�p�@���^{���B�W
?�p�@�p��ff���
B���                                    Bxf�'�  
�          @���?ٙ�@����w�� p�B�z�?ٙ�@����%��z�B�\)                                    Bxf�6�  
�          @Å>#�
@:�H���H�p��B���>#�
@����(��3  B��=                                    Bxf�EB  
�          @�{?�G�@����{�*(�B�u�?�G�@����8�����HB�\                                    Bxf�S�            @�33?�  @�{�r�\�B���?�  @�����R���B�\)                                    Bxf�b�  
�          @��
@�\@�ff�j�H�G�B���@�\@����
=��p�B��                                    Bxf�q4  T          @�z�@��@�(��:=q��p�B�\)@��@�\)�\�fffB�33                                    Bxf��  
�          @���@p�@�(���33�yp�B�Ǯ@p�@��
�8Q��33B��                                    Bxfێ�  "          @��H?�Q�@��>�ff@�G�B�{?�Q�@���?�A�=qB��                                    Bxf۝&  �          @�(�?�
=@��\?�=qA0��B�?�
=@��@�A�\)B�G�                                    Bxf۫�  
Z          @�=q?�\)@�{?��Ag�B�k�?�\)@���@,��A��B�(�                                    Bxfۺr  T          @�
=?�z�@���?���A�
=B�(�?�z�@��@J=qBffB��R                                    Bxf��  �          @���?�(�@���?�{A�z�B��H?�(�@�33@J�HB�B���                                    Bxf�׾  
Z          @�z�?��R@�G�?���AN�RB���?��R@�Q�@-p�A�\)B��                                    Bxf��d  �          @��H?�  @�p�?��
As\)B���?�  @��H@8Q�A�
=B�u�                                    Bxf��
  �          @�ff?�ff@�G�?���AlQ�B�\)?�ff@�\)@1G�A���B�                                    Bxf��  	`          @���?У�@��?�  AE�B��H?У�@�\)@(Q�A��HB��R                                    Bxf�V  "          @���?��@�
=?�p�A:{B�8R?��@��R@*=qA�z�B�
=                                    Bxf� �  �          @�\)?�  @�z�=�?�Q�B���?�  @��?У�A}B��)                                    Bxf�/�  	�          @\?���@��>aG�@�
B�Q�?���@���?�(�A�=qB�8R                                    Bxf�>H            @���?��
@�(���G���ffB�p�?��
@��R?�33AV�HB��q                                    Bxf�L�  
�          @��?�=q@��?s33AffB�\)?�=q@�@ffA�(�B�=q                                    Bxf�[�  T          @�Q�?�(�@�G�?p��AffB�(�?�(�@��
@z�A�Q�B�.                                    Bxf�j:  T          @�  ?���@�(�?�
=AfffB��)?���@��H@0  A�\B��                                    Bxf�x�  �          @���?�\)@�{?���AiG�B��q?�\)@���@1�A�B�=q                                    Bxf܇�  
^          @�
=?���@��@G�A��B��?���@��@P��B=qB�z�                                    Bxfܖ,  
�          @��H?�33@��?�Q�A�G�B�� ?�33@�  @K�B�B�(�                                    Bxfܤ�  
�          @�\)?�G�@���@A�33B�� ?�G�@��H@R�\BG�B���                                    Bxfܳx  �          @�\)?�
=@�@p�A�p�B�=q?�
=@�
=@XQ�Bp�B�Ǯ                                    Bxf��  
�          @�Q�@�@�33?��RAr=qB���@�@��@.�RA�B�=q                                    Bxf���  T          @�  ?�{@��H?��\A#�B��?�{@��@z�A���B�p�                                    Bxf��j  T          @�
=?�  @���?��HA��B�p�?�  @��@<��A��HB���                                    Bxf��  
�          @�(�?��
@���?�\A���B��?��
@�@>{A���B�Q�                                    Bxf���  &          @��?��@�  ?�Q�A���B���?��@��
@C�
B
ffB���                                    Bxf�\            @�p�?��@�@4z�A�33B�\?��@e@u�B4G�B��                                    Bxf�  T          @�G�?�ff@�Q������p�B��
?�ff@�  ������p�B�                                    Bxf�(�  
(          @�p�?�p�@�z�+���z�B��?�p�@���?�R@�(�B���                                    Bxf�7N  �          @��?�(�@���?��\A\z�B��R?�(�@���@{A�
=B�
=                                    Bxf�E�  "          @��>�ff@��
?�G�A�ffB�\)>�ff@�G�@:=qB��B��                                    Bxf�T�  T          @�>�p�@�33?�p�A���B�33>�p�@���@7�B
=B�                                    Bxf�c@  "          @�33<��
@�  ?��A���B��=<��
@�p�@9��BG�B�u�                                    Bxf�q�  "          @��\�c�
@�p�@%A�Q�B�\�c�
@Z=q@a�B4\)B�8R                                    Bxf݀�  �          @��
@P  @��R>#�
?�\BQQ�@P  @���?�Q�ANffBL��                                    Bxfݏ2  "          @��@K�@�G�>�  @&ffB\{@K�@��H?���Aap�BW=q                                    Bxfݝ�  
�          @��@l��@�ff���
�Q�BB�\@l��@�33?p��A\)B?��                                    Bxfݬ~  �          @��H@hQ�@�  =�Q�?fffBE��@hQ�@�33?�\)A8z�BA�                                    Bxfݻ$  
�          @�(�@s33@��ͼ#�
��\)B>33@s33@���?�  A#�B:                                    Bxf���  
�          @�Q�@x��@��>�Q�@g
=B=�@x��@���?���A]G�B8                                      Bxf��p  "          @��R@�Q�@���>.{?�p�B4��@�Q�@w�?��A8��B/��                                    Bxf��  "          @���@��@~�R>.{?�p�B.�H@��@tz�?���A4��B*=q                                    Bxf���  �          @��\@s�
@�  ?&ff@�p�B9�\@s�
@n�R?˅A��RB1p�                                    Bxf�b  �          @��H@c33@��?��AV=qBD@c33@k�@\)A�\)B8(�                                    Bxf�  
�          @�z�@l��@{�?У�A��RB;  @l��@[�@!�A���B+=q                                    Bxf�!�  
�          @�z�@�\)@E?���AZ{B
z�@�\)@,(�@G�A�33A��                                    Bxf�0T  �          @��@�{@%?s33A�\A��
@�{@�\?��A|��A�\)                                    Bxf�>�  �          @��
@��@�R?333@��A�Q�@��@   ?��HAF�HA���                                    Bxf�M�  
Z          @�(�@�=q?�>�=q@1�A�=q@�=q?�Q�?333@�33A�33                                    Bxf�\F  "          @�Q�@�Q�?�\)�L���G�A�33@�Q�?�\)>B�\?�\)A�G�                                    Bxf�j�  
�          @��H@�(�?��R�k��{AmG�@�(�?�  >�?��An�H                                    Bxf�y�  
�          @�{@���?��<��
>#�
A��R@���?޸R>�G�@�  A��R                                    Bxfވ8  �          @�{@�G�?��
���
�Dz�AH  @�G�?��ü#�
�uAM�                                    Bxfޖ�  �          @�33@�?���<��
>8Q�AU��@�?��>���@Q�AO
=                                    Bxfޥ�  
Z          @��@��\?�G�>�{@[�AK�@��\?��?!G�@ə�A9G�                                    Bxf޴*  T          @���@��?�=q<#�
=�G�AU�@��?��>��
@N{AO�
                                    Bxf���  T          @���@��?�녾�����A6=q@��?��H�\)��{A@z�                                    Bxf��v  
�          @��
@���?fff�����w�Ap�@���?xQ�8Q���HA�
                                    Bxf��  T          @��\@�\)?O\)�0���أ�A (�@�\)?p�׾���
=A                                    Bxf���  T          @��H@�{?p�׿aG���A{@�{?�\)�#�
���A2{                                    Bxf��h  
Z          @�33@�33?�=q�xQ��Q�AU��@�33?\�!G���ffArff                                    Bxf�  �          @�p�@��R?��ÿ���*�\A)�@��R?���Q���{AK�                                    Bxf��  �          @�@�G�?G��fff�  @��@�G�?xQ�0����  A\)                                    Bxf�)Z  �          @��@���?���z��8z�A=@���?�33�\(��(�Ab�R                                    Bxf�8   T          @�{@�p�?���(��G
=AB�H@�p�?��k��G�Aj�H                                    Bxf�F�  �          @��@���?�=q��(��B=qAX  @���?��ÿaG��  A~{                                    Bxf�UL  T          @���@�
=?�=q��\)�[33A[33@�
=?�{���
�#�A�p�                                    Bxf�c�  
�          @��@��R?�\)��
=�c�Aa�@��R?�zῊ=q�*�RA�33                                    Bxf�r�  
�          @��@���?��׿��R�D  A_�
@���?У׿c�
�Q�A���                                    Bxf߁>  �          @�(�@�  ?����\)�XQ�Az�H@�  ?��ÿ}p��ffA�=q                                    Bxfߏ�  "          @��\@���?��R�˅�}G�Aw
=@���?����H�?\)A��\                                    Bxfߞ�  �          @��H@�z�?�
=��(�����An{@�z�?�������U��A��\                                    Bxf߭0  �          @��@��?��Ϳ����A_�@��?�(����H�fffA�G�                                    Bxf߻�  
�          @��@�33?����z���\)AI��@�33?У׿޸R��z�A��
                                    Bxf��|  "          @��H@�{?�  ���AXQ�@�{?�  �   ����A��R                                    Bxf��"  
�          @�z�@�\)?�=q�����\)A��\@�\)@�\������A�(�                                    Bxf���  "          @�  @�p�?��z���=qA��R@�p�@�ÿ�=q��{A�ff                                    Bxf��n  �          @��@���@z���\��{A�Q�@���@1녿��H���HA�p�                                    Bxf�  �          @�  @�{@&ff�{���A�@�{@A녿�=q�tz�A�{                                    Bxf��  T          @��@��?�{�ff��{A��R@��@�\��Q���p�A��R                                    Bxf�"`  T          @�p�@�\)?�  �z����A��@�\)@
�H�У�����A��\                                    Bxf�1  �          @�\)@��\?�p��Q����HA�ff@��\?��޸R��z�A�\)                                    Bxf�?�  �          @�{@���?�z��
�H��{A�G�@���?�{��ff����A�                                    Bxf�NR  T          @�Q�@�(�?��H������A�G�@�(�?�녿ٙ���{A�
=                                    Bxf�\�  �          @�  @�(�?�Q����=qA�\)@�(�?�\)���H���A�\)                                    Bxf�k�  T          @��@�33?��   ����AN�\@�33?�=q��Q���(�A��                                    Bxf�zD  "          @�ff@�\)?����
=��ffA=��@�\)?�����33��p�A���                                    Bxf���  
�          @��@��R?�G�����ffAd��@��R?��Ϳ����h  A�33                                    Bxf���  �          @���@�ff?�G���G��~ffA/\)@�ff?��ÿ�  �QAc
=                                    Bxf�6  T          @��@���?&ff����<��@�p�@���?c�
�u��
A��                                    Bxf��  
�          @��\@�Q�?\(��������A(�@�Q�?�����z��j�\AMG�                                    Bxf�Â  
�          @��@��
?��
��33��Q�A5@��
?��У����\AxQ�                                    Bxf��(  T          @�p�@�  ?�{��ff��=qA@  @�  ?�p���G��w�A|Q�                                    Bxf���  �          @��@�?��
�ٙ���
=A^ff@�?�\)��\)�c33A��                                    Bxf��t  T          @��H@�{?��\��z����
A[\)@�{?��Ϳ���]A��H                                    Bxf��  
�          @��
@�  ?�=q��p��t��Ac
=@�  ?�\)��z��>=qA�
=                                    Bxf��  "          @��@���?s33���H��  A&�\@���?��Ϳٙ���  Ak
=                                    Bxf�f  
Y          @�z�@���?�{��\)��  A���@���?���p��JffA��
                                    Bxf�*  �          @��@�{?�(��ٙ���AT  @�{?Ǯ��33�g\)A�{                                    Bxf�8�  
�          @��@��?�z��{��{AL��@��?���Ǯ��z�A��
                                    Bxf�GX  T          @�G�@��H?�z������=qALz�@��H?��
�Ǯ����A��                                    Bxf�U�  �          @���@�\)?�z�����
=AQp�@�\)?�=q���
��z�A���                                    Bxf�d�  �          @���@�
=?�
=����p�AVff@�
=?�{���
��ffA�G�                                    Bxf�sJ  T          @���@���?��
��(����A8z�@���?�
=���H���A}�                                    Bxf��  �          @�G�@�=q?n{�G�����A'
=@�=q?��Ϳ�\���Am�                                    Bxfᐖ  T          @���@�=q?��\��ff��(�AaG�@�=q?У׿�p��yp�A�z�                                    Bxf�<  T          @��@��
?�Q��ff���ARff@��
?�  ��G��T��A��R                                    Bxf��  
�          @���@�G�?u��  �R=qA$��@�G�?��H���\�*=qAM�                                    BxfἈ  T          @���@��?�{��G��|z�Ak�
@��?�33��
=�D��A�                                    Bxf��.  T          @��R@�Q�?�Q�������RA�(�@�Q�?�  ��G��Up�A��
                                    Bxf���  
�          @�33@c33@*=q�����Y�B\)@c33@5�������B�                                    Bxf��z  �          @��
@!G�@s�
�0���	G�Bb{@!G�@w�>\)?�(�Bc�R                                    Bxf��   T          @���?��@�33�=p���RB�ff?��@��>��?�(�B��                                    Bxf��  T          @�z�?�Q�@~{�B�\� ��B�  ?�Q�@�G�=���?��\B��q                                    Bxf�l  �          @��׿�G�@��H�k��=p�B�33��G�@�G�?333A�B�z�                                    Bxf�#  �          @�녾�@��׾.{�(�B��=��@�ff?J=qA�B��R                                    Bxf�1�  "          @���>��
@�Q쾅��R�\B�Ǯ>��
@��R?333A=qB��3                                    Bxf�@^  T          @�
=?Y��@��ÿ(�� Q�B�p�?Y��@��>���@|��B���                                    Bxf�O  
�          @�p�?}p�@�Q�
=���B��
?}p�@�G�>��
@��RB�                                    Bxf�]�  T          @��?��
@�ff�W
=�,��B�.?��
@���=�\)?\(�B�Ǯ                                    Bxf�lP  T          @�ff?�ff@�G��
=��p�B��?�ff@��>\@�B�\                                    Bxf�z�  
�          @�=q?�{@�z�
=��RB��?�{@��>�{@��B��
                                    Bxf≜  T          @��?���@�G��5�
=B�\?���@��H>W
=@(Q�B�u�                                    Bxf�B  �          @�?��H@�  �E��!��B��{?��H@�=q=��
?�ffB�W
                                    Bxf��  T          @�G�?5@��׿xQ��I��B�(�?5@��
�L�Ϳ+�B���                                    Bxfⵎ  
�          @�  ?�ff@��ÿ��ָRB�u�?�ff@�G�>Ǯ@���B��=                                    Bxf��4  
�          @�=q?s33@�z᾽p����RB��\?s33@��
?\)@�\B�p�                                    Bxf���  �          @�{@��@��׿�ff�O�Buff@��@��;B�\�ffBxz�                                    Bxf��  
�          @�p�>��@�z�aG��"�\B�W
>��@��R>#�
?�=qB�z�                                    Bxf��&  
Z          @�G���@�{���H����B��ÿ�@�?(�@���B�
=                                    Bxf���  "          @�Q��
=@�ff�����\)B��Ϳ�
=@�ff?��@�ffB���                                    Bxf�r  
�          @��H��(�@��ÿ���,z�B����(�@�(�=��
?G�BҊ=                                    Bxf�  "          @�\)��G�@�p��xQ��#�BԞ���G�@�Q�=�?��\B�{                                    Bxf�*�  �          @��Ϳ}p�@���Q���B�aH�}p�@�G�>�z�@B�\B�(�                                    Bxf�9d  	�          @�ff����@�ff��z��DQ�B̊=����@�=q�u�
=B���                                    Bxf�H
  "          @��þ�=q@�=q��G��UG�B��쾊=q@��R����z�B���                                    Bxf�V�  "          @�(�>�33@�G��Y���G�B��>�33@�33>�p�@dz�B�.                                    Bxf�eV  T          @�\)?��@����z��4z�B�k�?��@���<�>�z�B���                                    Bxf�s�  �          @�=q?+�@�����
�\z�B�B�?+�@�녾B�\��p�B��                                    Bxfア  �          @�\)?8Q�@��\�u�*{B���?8Q�@�p�=���?��B�8R                                    Bxf�H  
�          @��>��@�p��+���p�B�L�>��@�ff>���@�B�W
                                    Bxf��  
�          @�\)>�@���333��z�B���>�@�{>�
=@��B��)                                    Bxf㮔  �          @��?:�H@�녿@  ���HB��?:�H@�33>��@���B�33                                    Bxf�:  "          @��H?�R@��׿�R�ʏ\B�(�?�R@���?�@�B�.                                    Bxf���  |          A33>���A������ ��B�G�>���A�R?��@p  B�L�                                    Bxf�چ  
�          AQ녿��AK
=��p���(�B��{���AK�?�33@��
B��=                                    Bxf��,  h          A[33��Q�AQ�?@  @G�B�����Q�AK
=@Z=qAf=qB�\                                    Bxf���  �          AV=q�FffAO\)�\��\)B��H�FffAL(�@�Ap�B�G�                                    Bxf�x  �          AT(��5�AM��33��Q�B��
�5�AM�?��R@ϮB��f                                    Bxf�  "          AU���XQ�AMG�?(�@'�Bя\�XQ�AF�H@Mp�A_�
B�u�                                    Bxf�#�  
�          AZ�\�k�AP�׿������\Bӊ=�k�AP��?�ff@�\)BӅ                                    Bxf�2j  
�          AYp���z�AN�H�\��\)B׀ ��z�AK�
@��A��B�                                      Bxf�A  
�          AYp�����AM녽���\B�������AJ{@   A)��Bٞ�                                    Bxf�O�  
Z          AUG���G�AG
=>��H@B����G�AAG�@@  AP(�B�=q                                    Bxf�^\  �          AS33����AE��>�{?�(�B�u�����A@z�@5AG
=B�u�                                    Bxf�m  T          AR�H���HAG�
>��?�\)B�=q���HAB�R@1�AC�B��                                    Bxf�{�  
�          AR�\��33A<Q�@�A�B��
��33A1�@��A���B�\                                    Bxf�N  
�          AO�����AD�þ��33B�.����ABff@�\A��B؞�                                    Bxf��  
(          AP����  AEp�������B�����  AB=q@\)A=qBڔ{                                    Bxf䧚  T          AU�����AG\)��G����B������AC�@=qA'33Bݣ�                                    Bxf�@  T          ATQ���z�AB�R?���@��HB�����z�A:=q@q�A�Q�B��                                    Bxf���  
�          AN�\����A5��?�Q�A�B�(�����A+�@�{A���B�                                    Bxf�ӌ  
�          ATz�����A.�H@j=qA�Q�B�3����A�@�G�A�\)B�p�                                    Bxf��2  �          AO
=��{A%�@uA�{B�����{A=q@�33A��B��                                    Bxf���  �          AJ�H��G�A(��@'
=A>{B���G�A��@�A�G�B�ff                                    Bxf��~  
�          AK��ȣ�A/�
?�\)@�{B�k��ȣ�A'�
@c�
A�B�                                    Bxf�$  
�          AZ{��G�AD��?\(�@g
=B�k���G�A=�@S�
Aa�B�                                      Bxf��  
�          AYG�����AIp������Bߔ{����AG
=?�(�A�B�                                    Bxf�+p            AP(����HAB=q��p����B�aH���HAC�
?=p�@P��B�{                                    Bxf�:  T          AR�H��Q�AG\)������
=Bמ���Q�AHz�?n{@��B�p�                                    Bxf�H�  T          AS33�aG�AH���ff�=qB�\)�aG�AK33>�@   B���                                    Bxf�Wb  
�          AV=q�9��AIp��N{�_�
B����9��AO��!G��.{B�33                                    Bxf�f  ,          AZ�H���
A<��@#�
A-�B����
A0��@�(�A�\)B��H                                    Bxf�t�  Y          A\  ��Q�A;
=@   A((�B�\��Q�A/33@�G�A�p�B�\)                                    Bxf�T  
          A`z���(�A;�@7
=A<Q�B���(�A.�\@���A�Q�B�Q�                                    Bxf��  �          Ad���љ�AI�?�@�B��)�љ�A@z�@���A���B��                                    Bxf堠  "          Ad  ��
=AO\)��  ����B�\��
=AP��?^�R@g�B���                                    Bxf�F  �          AX��?��\AA������ˮB�\?��\APz��QG��`  B��f                                    Bxf��  �          A_����RAS
=��z����\B��)���RA\�Ϳ����
B�L�                                    Bxf�̒  
�          AV�R���AF�H�   �	��Bۮ���AIG�?�@\)B�=q                                    Bxf��8  �          AU����AC33���ÅB��
���AC�?��@��B�3                                    Bxf���  "          AX(���Q�AEG���{���HB�(���Q�ABff@�Az�B���                                    Bxf���  
�          AK�
��ffA�@%A<z�C ff��ffA(�@�
=A�\)C�=                                    Bxf�*  
�          A�H��
=�fff@�33BF�C<����
=�33@���B:��CI\                                    Bxf��  �          Aff��p��(Q�@�\)B<�CK{��p�����@��B&�CU\                                    Bxf�$v  T          A{��ff�!�@�ffBD(�CK  ��ff�}p�@ٙ�B.=qCU��                                    Bxf�3  �          A���p�����@�\BAG�C@���p��,��@�B3�CK��                                    Bxf�A�  �          A$���ָR�#�
@�{BC  C9xR�ָR�
=q@��B9��CE�{                                    Bxf�Ph  �          A-��Q�>�@���B=p�C0=q��Q쿌��@��\B;p�C<�H                                    Bxf�_  �          Ap���\)@7
=@�B<(�Cu���\)?��@�G�BKp�C&�3                                    Bxf�m�  
-          A\)��@!G�@�\B6��C����?��@�{BC33C*�f                                    Bxf�|Z  O          A"ff��@{@��HB33C"+���?��H@θRB�HC+
=                                    Bxf�   T          A*�\��G�@�
=@��HA��HC���G�@���@�=qBffC
�f                                    Bxf晦  "          A#�
��
=@�(�@�\)B
=C#���
=@��H@�B+��C��                                    Bxf�L  �          A*�R���@�\)@��HB�
C�����@HQ�@�33B&��C��                                    Bxf��  �          A*=q��z�@�z�@��B�\C����z�@/\)@�33B/ffC                                    Bxf�Ř  �          A%p����@N{@�B=qC8R���?��@�{B*Q�C%��                                    Bxf��>  "          A$(�����@�33@�ffA���C�����@�33@�
=B  C5�                                    Bxf���  �          Az���Q�A\)?#�
@qG�B�R��Q�A�R@33A[
=B�z�                                    Bxf��  
�          A���G�A�R?(��@��B�k���G�@�(�@G�Ac
=B�(�                                    Bxf� 0  �          A�H�h��AG�@Q�AS\)B�8R�h��@��@o\)A���B�aH                                    Bxf��  �          A���QG�A�?��
@���B���QG�AG�@5A�(�Bܳ3                                    Bxf�|  
�          A!p��P��A�H?��@�G�B�.�P��A(�@?\)A�z�B���                                    Bxf�,"  �          A����p�@�(�@8Q�A��B�\��p�@���@�z�A�z�B�ff                                    Bxf�:�  �          A��Q�@��@���A��
C�R��Q�@��@��\BQ�C��                                    Bxf�In  "          A���{@ʏ\@XQ�A�
=C^���{@���@��A�Q�C
5�                                    Bxf�X  T          A���{@�Q�@���A�=qB�z���{@��@��BC �                                    Bxf�f�  "          A{��33@ٙ�@a�A�p�C 5���33@��R@���A�p�C�f                                    Bxf�u`  
�          A=q��(�@���@   Ai�B����(�@�z�@K�A�z�Ch�                                    Bxf�  T          @��H���\@���?�G�@�  B�p����\@�ff@�A��B�k�                                    Bxf璬  "          @�
=���H@�ff�ff����CG����H@��\��(��3�C�H                                    Bxf�R  
�          @����Q�@�G�>W
=?�{C	����Q�@���?���AQ�C
W
                                    Bxf��  T          @�33���@�z�h����z�C
���@�
==�?xQ�C��                                    Bxf羞  
Z          @�
=��33@�z�Q���\)C:���33@��R>#�
?���Cٚ                                    Bxf��D  �          @������@��R���]p�C�����@�
=>�(�@P��C�q                                    Bxf���  
�          A�\��
=?�G�@�G�BG�C%O\��
=?!G�@���B"G�C.��                                    Bxf��  �          A ���G�@��?�(�A4Q�C@ �G�@��@@  A�{C�q                                    Bxf��6  �          A(z��Q�@��H@B�\A��RC��Q�@�(�@�G�A���C}q                                    Bxf��  �          Az����R@��\@>�RA��C�f���R@�z�@{�A���CaH                                    Bxf��  �          Az����H@���@#33A�G�CQ����H@c�
@Tz�A��RC�
                                    Bxf�%(  �          A33�  @:�H@;�A�z�C!�  @�@\��A�\)C$��                                    Bxf�3�  �          A�H���@AG�@�Au��C ����@�R@>�RA���C#^�                                    Bxf�Bt  �          A=q����@Z�H@q�A�p�C�\����@'
=@�(�A�z�C J=                                    Bxf�Q  "          @�{��  @Q�@���B
�\C�=��  ?�p�@�{B{C%�{                                    Bxf�_�  �          @��R��  @{@n{A��
CxR��  ?�
=@�(�BQ�C$�3                                    Bxf�nf  
�          @����p�@(��@��B��C�q��p�?�G�@�33B�\C#n                                    Bxf�}  
�          @�z���  @!G�@�B
=CY���  ?˅@��HB%�RC#!H                                    Bxf苲  "          Az���p�?u@�=qB^p�C)���p���\@�B`p�C9��                                    Bxf�X  
�          AG���(�?��@�(�BJC&�q��(���G�@�Q�BO�C5�                                    Bxf��  "          Aff��?��@�ffB?{C(\������@�\BC�C4�                                    Bxf跤  �          Aff��(�?��\@�=qBZ�HC&��(����R@��B^�C7J=                                    Bxf��J  	�          A((���@I��A\)B`p�Cp���?�{A�\BsQ�C$�)                                    Bxf���  �          A'���Q�@�(�A	p�B]\)C� ��Q�@%A��B{33C33                                    Bxf��  �          A=q���\>�@��BDC.�{���\�8Q�@�(�BC�\C<�                                    Bxf��<  �          A����\)�'
=@�  B�\CJ��\)�`��@�z�A��CQ^�                                    Bxf� �  
�          A���=q��z�@�  BG�C6� ��=q��p�@�33B

=C>�\                                    Bxf��  
�          A�
��\)@2�\@��\B{C�
��\)?У�@���B,ffC%Y�                                    Bxf�.  �          AM��R@���@�G�A�
=C{��R@���@�{B��CJ=                                    Bxf�,�  �          A`  �#33@ҏ\@��A�z�C(��#33@�33@��B
=Cs3                                    Bxf�;z  
�          Ao�
��ffAS�
>�p�?�Q�B�ff��ffANff@@  A9G�B��                                    Bxf�J   T          Apz��(�A?�>�  ?s33B��H�(�A;33@(Q�A!��B�8R                                    Bxf�X�  
�          Ah���.�R@��H@�33A�  C
�.�R@���@�
=AʸRC)                                    Bxf�gl  
�          Ad���>=q@x��@�{A��C!���>=q@!�@�\A��
C(                                      Bxf�v  T          Alz��@��@�z�@׮A�33C���@��@\��@�\A�\)C$�                                    Bxf鄸  T          Ap���9A{@���A{\)C(��9@��@��
A��Cp�                                    Bxf�^  T          Ak��/�A
�H@���A��HC�3�/�@�=q@���A�Ch�                                    Bxf�  
�          Ae��%�A(�@��A�Q�C�\�%�@�
=@���AǮCY�                                    Bxf鰪  �          Axz��C\)@�ff@�\A�{C�C\)@�p�A�B {C )                                    Bxf�P  "          A
=�M�@C33A{B�\C&���M�?��HA�B�C.��                                    Bxf���  
�          A~=q�IG�@���A
�RBffC!)�IG�@�RA��Bp�C(�
                                    Bxf�ܜ  �          A\)�F=q@��HA33B��C!���F=q@
�HAG�B�C*{                                    Bxf��B  
Z          A\)�I�@�Ap�A�=qC�)�I�@Y��A33B
=C$ٚ                                    Bxf���  T          A|(��I@�\)A�HA�=qCs3�I@?\)A\)B
�C&�f                                    Bxf��  T          A|(��F�R@���A  A���C�H�F�R@R�\Ap�B�C%&f                                    Bxf�4  "          Avff�>=q@�p�@��A��C�>=q@�Q�@�p�A�C33                                    Bxf�%�  �          Avff�Fff@���@��
A�  C��Fff@���@�{A�Q�C�                                    Bxf�4�  �          Ar=q�8z�@��@�p�A��C�f�8z�@�Q�@��
A�{C��                                    Bxf�C&  S          Ar�\�-��A\)@���A�C���-��A (�@���AՅC�{                                    Bxf�Q�  �          Ar{�@(�@�
=@�(�A��C�H�@(�@���@��A�RC�{                                    Bxf�`r  
�          Aq���F�\@��@�
=A£�C�=�F�\@�G�@���A���C�                                    Bxf�o  �          An�\�G33@���@�33A��HC�R�G33@��R@��HA��HC!Q�                                    Bxf�}�  �          Al(��7�AQ�@�G�A~�HC=q�7�@�G�@��
A��C�)                                    Bxf�d  T          Ap(��:=qA\)@333A,��Cff�:=qA\)@���A��C�=                                    Bxf�
  �          Ap���<  A\)@���A��
C\�<  @�33@ƸRA���Cٚ                                    Bxfꩰ  "          Ap���=�AG�@�  A�33C�=�@�  @�G�A�\)Cu�                                    Bxf�V  �          Ap���<��A\)@�\)A�ffC#��<��@�z�@�G�A�p�C�\                                    Bxf���  �          Ao
=�9�@�{@�{A�  CB��9�@��H@��A�CT{                                    Bxf�բ  T          Al���@��@���@�(�Aď\C���@��@�G�@�ffA�CY�                                    Bxf��H  
Z          Am���C33@�G�@��HA���C���C33@u@���A�C"�                                    Bxf���  �          Ap���D��@��H@�z�A�G�C�\�D��@�(�@�(�A�C!s3                                    Bxf��  �          Aq��E��@�=q@ָRA��HC�R�E��@��H@�ffA�
=C!��                                    Bxf�:  T          Atz��G�@�
=@�\)A��C\)�G�@�\)@�Q�A��C!@                                     Bxf��  �          Ao��C\)@���@���A��C�)�C\)@�=q@�z�A��C!�{                                    Bxf�-�  �          Ao\)�G
=@�33@�G�Aϙ�C���G
=@j=q@�{A���C#��                                    Bxf�<,  
�          At(��LQ�@��
@�p�A���CQ��LQ�@�G�@߮A�(�C n                                    Bxf�J�  T          Au��F�\@�Q�@���A�Q�C��F�\@�{@�\A�\)C^�                                    Bxf�Yx  �          As��J�R@���@�{A��HC���J�R@�@߮A�(�C �R                                    Bxf�h  �          Al���O�
@n�R@���A�(�C#�q�O�
@�H@��A��C)s3                                    Bxf�v�  T          Ah  �L��@1�@�  A�Q�C'���L��?���@�{A�C-�=                                    Bxf�j  T          Ae��Hz�@1�@θRA�C'}q�Hz�?�33@�z�A�G�C-�H                                    Bxf�  �          An{�O�
?��H@߮A�{C+h��O�
>��H@�A���C1�
                                    Bxf뢶  "          Ar=q�B�\@���@��HA�\)Ch��B�\@��@���A��HC�                                    Bxf�\  
�          At  �'
=A0(�?}p�@n{Cz��'
=A)�@N�RAC�
C��                                    Bxf��  T          Ap(���HAE�?}p�@s33B��q��HA=��@`  AX��B���                                    Bxf�Ψ  �          Ar{�(�AB{��\)��ffB�=q�(�A>�H@{A\)B�(�                                    Bxf��N  �          Ar=q�Q�AHz�?�@�G�B��)�Q�A=@��RA�
=B�                                    Bxf���  "          As\)� (�AMp�?�  @p��B��f� (�AE@g�A\��B��H                                    Bxf���  �          Ar�\�z�AHQ�?�?���B���z�ABff@EA<(�B�(�                                    Bxf�	@  �          Apz���
A<��?�  @u�C 
=��
A5��@Z�HAR�\C(�                                    Bxf��  
�          Ap����RA=p�?�33@��B�� ��RA4��@tz�Ak�C�                                    Bxf�&�  @          AuG��.�RA&�\@-p�A#
=C\)�.�RA{@��A�=qC
��                                    Bxf�52  
�          Au�4  A ��@<(�A0Q�C
:��4  A�@�ffA�z�C��                                    Bxf�C�  T          At���=�A33@r�\AeG�C�)�=�@�
=@��A�z�C�                                    Bxf�R~  "          Av�H�C�A z�@��RA��C�3�C�@�=q@У�A�C�{                                    Bxf�a$  
�          Au���D  @�(�@�{A�=qC��D  @�{@�A�p�C
=                                    Bxf�o�  
T          Av�\�=��AQ�@�p�A�Q�C{�=��@�=q@���AʸRC0�                                    Bxf�~p  J          AtQ���\A9G�?��R@��C����\A.�\@��A�G�C��                                    Bxf�  �          Aq�*�RA%p�@%A{C���*�RA�@�{A��HC
)                                    Bxf웼  
�          At����A:{?�@ٙ�C���A0  @�ffA\)C��                                    Bxf�b  @          As\)���A;
=?���@�Q�C:����A333@c33AX��Cp�                                    Bxf�  
�          Ar�H�+\)A%p�@8��A/�C��+\)A(�@��A��C
h�                                    Bxf�Ǯ  
�          Apz��,(�A"=q@G�@�Q�C���,(�A�
@�33A
=C
�{                                    Bxf��T  
�          Ap���BffA33@)��A"�RC0��Bff@�ff@��
A��C�H                                    Bxf���  "          Ao��A��A33@)��A"�\C\�A��@��R@��A�\)C�                                     Bxf��  
�          Aj=q�   A(��?�ff@��C���   A�R@~�RA|(�C=q                                    Bxf�F  "          Ah���"�HA"�R@�\A�C��"�HA(�@�z�A�Q�C�3                                    Bxf��  
(          Am�(��A$z�?�(�@�(�C�q�(��A{@�33A�{C	��                                    Bxf��  �          Ak��(��A ��@�A=qC\)�(��A{@��A�p�C
Q�                                    Bxf�.8  �          Af�R�%��A?�{@��Cff�%��A  @y��A{\)C
=q                                    Bxf�<�  T          An{�3
=A{@
=A�HC�3
=A
�R@�G�A�
=C:�                                    Bxf�K�  "          Ap(��4(�A\)@33@�33C8R�4(�A��@��A{\)C0�                                    Bxf�Z*  �          Anff�0  AG�@�
@��RC
33�0  A�R@�33A�  C.                                    Bxf�h�  
�          Ao��'33A(��?��H@��C�q�'33A=q@��A�
=C�
                                    Bxf�wv  �          Amp��%p�A'�
?�@��C�{�%p�A@�Q�Az�RCW
                                    Bxf�  
�          Ak33�'33A"ff?�(�@��RC�
�'33A�
@��HA�G�C	�q                                    Bxf��  �          Aj�H�+�A{?�p�@���C	Y��+�Az�@s33ApQ�C!H                                    Bxf��h  "          Ajff�.�RA��?�(�@���C
�{�.�RA(�@^�RA\��CxR                                    Bxf��  �          AdQ��9G�A
=?��@�{C�q�9G�@�p�@P��AS33C}q                                    Bxf���  T          AeG��5p�A	�?�(�@�z�C� �5p�A@R�\AT(�Cp�                                    Bxf��Z  
�          Ab�H�'33A�?��RA ��C
)�'33A�@}p�A���C#�                                    Bxf��   T          A^�H�#33A��?�{@���C	� �#33Az�@eAn�RCG�                                    Bxf��  
�          A_\)�$��AQ�?�  @�
=C	�q�$��A
�R@mp�Av�\C޸                                    Bxf��L  
�          A\  �  Aff?xQ�@�G�CL��  A\)@A�AK�C�H                                    Bxf�	�  "          A\z����A ��?��R@���C�����A��@XQ�Ac33Cn                                    Bxf��  
�          AO33�ə�A2�H>��?(��B����ə�A.=q@$z�A8z�B�#�                                    Bxf�'>  �          AM���p�A%��>W
=?uB�L���p�A ��@��A1p�B��)                                    Bxf�5�  |          A>{�ÅA"=q�   ���B���ÅA (�?�(�AG�B�Ǯ                                    Bxf�D�  �          AB�H��{Aff>�33?�33B�G���{A��@(�A8��C �=                                    Bxf�S0  |          AAp���RA33��z῰��C L���RAz�?�\A�C �=                                    Bxf�a�  T          A1���ϮA�H��G���B�  �ϮA�
?�Ap�B�.                                    Bxf�p|  h          A0��@`  A33�{���{B�@`  A�����R�8z�B���                                    Bxf�"  |          A0  @g
=@�33�B�\��
=BQ�@g
=@��\��{�\)B��                                    Bxf��  �          A:=qA�H@�  ��(����A�=qA�H@�
=�u���ffA�                                    Bxf�n  �          A9��A-��?Q��Z�H���@��A-��?�ff�I����z�A�                                    Bxf�  �          A:ffA((�@8����z���=qAu��A((�@n�R�Y�����A�{                                    Bxf  �          A<Q�A3
=@
=��G��(�A>=qA3
=@,(�������ffAX                                      Bxf��`  �          A=��A9p�?�ff����ff@��A9p�?�
=������A(�                                    Bxf��  T          A:�RA2�R?�{��p���HA
=A2�R@  �\��A6{                                    Bxf��  �          AD��A?
=��
����z�C���A?
=��=q��/�C���                                    Bxf��R  �          AF�\A@���#�
�������C��A@����R�޸R� ��C��                                    Bxf��  �          AG33A@�ÿ��R����'\)C�Q�A@�ÿ�Q��(���DQ�C���                                    Bxf��  �          ADz�A<�׿h���P  �u�C���A<�׾#�
�W��33C���                                    Bxf� D  �          A=�A5p��n{�L���{�C��=A5p��B�\�U����HC��                                    Bxf�.�  �          AF{A<�ÿz�H�aG���ffC�� A<�þ.{�i����p�C���                                    Bxf�=�  �          AG\)A>{�333�j=q����C�P�A>{>��n{���?&ff                                    Bxf�L6  �          AG\)A>{��Q��b�\��=qC�"�A>{��p��n{��
=C�)                                    Bxf�Z�  �          AFffA?\)��\)�Q��t��C�S3A?\)>��P���s�@��                                    Bxf�i�  �          A>ffA9�<��0  �V{>\)A9�?#�
�*�H�P  @L(�                                    Bxf�x(  �          AA��A;33��ff�9���^=qC�nA;33��p��Dz��j�HC�
                                    Bxf��  T          A8z�A1���Q��!��K33C�A1������9���i��C��
                                    Bxf�t  �          A4��A,z��\)�#33�P��C�!HA,z��{�?\)�u��C�                                    Bxf�  �          A1p�A*ff��\�z��,Q�C��A*ff��G��"�\�S�C�O\                                    Bxf��  �          A1�A((��N�R��z��	�C�w
A((��1G���B�RC��                                     Bxf��f  �          A0��A\)����������C���A\)�����H�J=qC�˅                                    Bxf��  �          A(��A\)?����5���
A,z�A\)@(���RffAb{                                    Bxf�޲  �          A$��A�@n�R�AG����
A��\A�@�=q��=G�A͙�                                    Bxf��X  r          A�@��@������c�
B.\)@��@�z�!G���  B4                                    Bxf���  �          A33���A����H��33B��Ϳ��AQ�?aG�@�Q�B��q                                    Bxf�
�  �          A{�\(�A�?�\@Dz�B�ff�\(�A�@*�HA��B���                                    Bxf�J  T          A �ÿ�G�A  @{AO
=B��ÿ�G�A�@��HAϙ�Bʽq                                    Bxf�'�  �          A���A(�?��H@��B�
=���A�@W
=A��
Bυ                                    Bxf�6�  �          A���<(�A	����p���BٸR�<(�A\)?
=q@W�B�G�                                    Bxf�E<  �          A*�\���A@p�AY�B��H���A��@���AͮB��
                                    Bxf�S�  �          A1��{A
=�c�
��p�B�(���{A{?�
=@�Q�B�p�                                    Bxf�b�  h          A8(���{A(z����ffB���{A%��?�p�A\)B�u�                                    Bxf�q.  ,          A?���=qA2ff��
=��B����=qA1�?�
=@��B�33                                    Bxf��  �          A<���K�A2{��33�G�B��f�K�A4Q�?:�H@e�Bӊ=                                    Bxf��z  �          A>{��Q�A9G���G����Bď\��Q�A:�R?z�H@��B�p�                                    Bxf�   �          A:�R>W
=A5�!��I�B��>W
=A:=q>8Q�?fffB���                                    Bxf��  �          A6�H��ffA-���O\)��{B��3��ffA4�ÿ�R�G
=B�#�                                    Bxf�l  �          A2{��z�A"{�I����ffBƏ\��z�A)p��0���l(�B�Ǯ                                    Bxf��  |          A7�
@>{A"ff�z=q��z�B�Q�@>{A,zῷ
=����B�#�                                    Bxf�׸  �          A9p�@h��A$���\(����B�{@h��A-G��k���(�B���                                    Bxf��^  �          A6{���A�R�8Q�h��C�3���A�?�ffA�RC�)                                    Bxf��  T          A6�\��
=Aff�8Q�h��B� ��
=A�R@ffA*ffB��q                                    Bxf��  �          A5����\A{������B�R���\A�?E�@z�HB�=q                                    Bxf�P  |          A9����A$�ÿ�����  B�=q����A$Q�?�{@�G�B�aH                                    Bxf� �  �          A9G���HA/
=�#�
�R�\B�����HA,��?�(�A!G�B�Q�                                    Bxf�/�  |          A<z��z�A2�R�333�^�\B���z�A8(���\)����B�p�                                    Bxf�>B  �          A8�Ϳ�A(Q������z�B���A4�׿�����B�aH                                    Bxf�L�  �          A9?J=qA�
��p����B�z�?J=qA,  �~�R��  B���                                    Bxf�[�  �          A9G�@
=A���{��ffB��@
=A)p��@  �xz�B�p�                                    Bxf�j4  �          A4(��c�
AG���\)���B��f�c�
A&�H�U��p�B��q                                    Bxf�x�  �          A4Q��z�A&�H�X����
=B���z�A/
=�L�����\B���                                    Bxf�  �          A=p��h��A Q���Q���p�B���h��A.�H�p��@��B���                                    Bxf�&  �          AC\)�w
=A���ff�   B��f�w
=A-p��}p����
B�.                                    Bxf��  �          AE����A�\������\B����A1��]p�����Bܳ3                                    Bxf�r  �          A;����
A�@z�AAG�B�G����
A�@�=qA�=qB���                                    Bxf��  �          A8z����A=q?�
=@�p�B�#����A  @q�A�{B�z�                                    Bxf�о  �          A7
=��(�A33?�{@�=qB�z���(�AG�@j�HA�B��H                                    Bxf��d  �          A733��{A{�p�����B����{AG�?�{@ۅB�8R                                    Bxf��
  h          A.{���A
=��\)���B�Q����A��33�0  B��                                    Bxf���  �          A/���
=A
{�H����33B�aH��
=A=q�p����p�B�\)                                    Bxf�V  �          A0�����RA\)�?\)�|��B��R���RA�H�E���Q�B���                                    Bxf��  h          A8z���ffA�
�H���}��B�B���ffA�
�W
=���RB�aH                                    Bxf�(�  |          A@(���G�A
�R��{��p�C���G�A�
���'
=B��                                    Bxf�7H  �          A:�\��
=A(���z�����B�\��
=A���\� ��B��                                    Bxf�E�  �          A:�H��A
=�R�\����B�\��A'33�=p��i��B�                                    Bxf�T�  �          A<(�� ��A33���?�Cs3� ��A�׾�{��33CE                                    Bxf�c:  �          A<�����A�S�
��(�CE���A
�H������=qCW
                                    Bxf�q�  |          A;���G�A���a���=qB�����G�A�\��
=��Q�B�33                                    Bxf�  �          A<(���\)A z��vff��33B�z���\)A*�R���R���B���                                    Bxf�,  �          A<Q�����A#��]p���p�B�  ����A,(��O\)�}p�B��)                                    Bxf��  �          AD�����A�׿�ff�G�C	�����A�>���?�G�C�H                                    Bxf�x  �          AFff�z�A Q�У���C0��z�A�R>�(�?�(�C
��                                    Bxf�  h          A@z���33A
=�����
B�ff��33A�>��@�\B�Q�                                    Bxf���  �          A>{�$��@���?^�R@�
=CE�$��@�  @  A0z�C                                      Bxf��j  �          A<Q����@�G�?�{@�=qCff���@��\@!�ALz�Cs3                                    Bxf��  �          A8���\)@�?�A=qC޸�\)@��@Z=qA�z�C�)                                    Bxf���  �          A8(��	G�@�
=?�(�@��
C
���	G�@ۅ@S�
A�{C^�                                    Bxf�\  �          A8  ���A�þW
=����C����A	G�@G�A"�\Cn                                    Bxf�  �          A<���   A\)?�{@���C^��   @���@mp�A�
=C�=                                    Bxf�!�  �          AA����{Az�?�=q@ə�CL���{Aff@e�A���C^�                                    Bxf�0N  �          A@������A�?k�@��
C�=����A	G�@Mp�Aw33C@                                     Bxf�>�  h          A>�\��A=q��z῵B�G���A�\@
=A#�
C Y�                                    Bxf�M�            A@����\)A녿�Q��p�B����\)A�?Q�@{�B�L�                                    Bxf�\@  �          AG
=��\A�Ϳ�p���C=q��\A��?�Q�@�  C:�                                    Bxf�j�  {          AH(���{A��.�R�I��B�#���{A#�
��G��   B�#�                                    Bxf�y�  �          AB�H��G�A���!��?�B�� ��G�A!=�\)>�{B�Ǯ                                    Bxf�2  �          AC����A���
=���B�#����A�R?h��@�Q�B��{                                    Bxf��  h          AG�
��z�A�ÿ�(���z�C����z�A��?��
@�p�C��                                    Bxf�~  �          AE�У�A ���Dz��ep�B��f�У�A((���Q��B��                                    Bxf�$  �          AIp����A#
=����B�ff���A&�\?�@�HB�G�                                    Bxf���  �          AJ�H��A�R��Q���Q�CxR��A�?��@���CQ�                                    Bxf��p  �          AJ�H�=qA�?0��@G�C���=qA�@HQ�Ad��C)                                    Bxf��  �          AN{�
=A
=?���@�p�C�f�
=A�@aG�A~{C�{                                    Bxf��  �          AN�\��HA�R?ٙ�@�\C=q��HA�\@�G�A�C��                                    Bxf��b  �          AV�H��
A�@��A$(�C	T{��
A ��@��A�(�CxR                                    Bxf�  �          A]G�� ��A{?�33@ۅC��� ��A	@��A�Q�CaH                                    Bxf��  T          AW���Azᾮ{��Q�C�\��A��@�AG�C�\                                    Bxf�)T  �          AR{�p�A	�    �#�
C
�=�p�A��@G�A�C�\                                    Bxf�7�  �          AN=q�Q�A	>��
?�
=C	޸�Q�A�@%�A9��C0�                                    Bxf�F�  �          AR�R�p�A�R��(����C	
�p�A�R?�
=@��C	�                                    Bxf�UF  �          AT  �'�A �ÿTz��fffCh��'�A   ?��@���C�f                                    Bxf�c�  �          AZ�H�.�\A=q���
��Q�CG��.�\@�33@��A��CB�                                    Bxf�r�  �          Aa��*�HA��Q���
CO\�*�HA�>���?�p�C��                                    Bxf�8  �          A\(��(Q�A\)�$z��,Q�C8R�(Q�AG��aG��k�C�q                                    Bxf��  |          AZ�\�
=A33�aG��o33C	L��
=A�ÿ�  ���RCc�                                    Bxf���  �          AX����A�H�p  ����C����A!p�������
C��                                    Bxf��*  �          AV{�
ffA�R�|�����C���
ffA"{���\���RC�                                     Bxf���  �          AQG���{A{��(�����C ����{A'33�У���p�B���                                    Bxf��v  �          AM���Az��|�����HC{���A#�
���R����B�#�                                    Bxf��  T          AL�����HA�R�p  ��=qC����HA!G���������B��q                                    Bxf���  �          AO
=���RA\)�?\)�Up�B�u����RA&ff�8Q�L��B��                                    Bxf��h  �          AQ���{A!���E��ZffB��{��{A(�þk���G�B�#�                                    Bxf�  �          AP�����A
=�c33�|z�C�����A$�׿G��[�B���                                    Bxf��  �          AR=q�أ�A,  �J�H�`Q�B�k��أ�A333�����B�L�                                    Bxf�"Z  �          AK\)��A���B�\�]p�C)��Az�Ǯ��  C��                                    Bxf�1   �          AQ��=qA'\)�B�\�VffB�����=qA.=q��\)��\)B�Ǯ                                    Bxf�?�  �          AV=q��
=A/��N{�^�HB��H��
=A6�H��Q���B�q                                    Bxf�NL  T          AT����\)A+�
�?\)�O�
B����\)A2ff=��
>���B���                                    Bxf�\�  �          AV{���
A/33�x�����B�L����
A9���8Q��EB�p�                                    Bxf�k�  �          AS\)��Q�A1���U��i�B��)��Q�A9p��\)�
=B�Ǯ                                    Bxf�z>  �          APQ����HA.�H�g����B�B����HA8(����G�B���                                    Bxf���  �          AP  ��(�A((��s33���B�p���(�A2ff�=p��P  B�k�                                    Bxf���  �          AS���
=A0���o\)��\)B���
=A:ff���G�B�33                                    Bxf��0  T          AJ�\����A+�
�����B�L�����A7\)�u���B�k�                                    Bxf���  �          AJ�H��{A)��33��Q�B�.��{A7����H��(�B�q                                    Bxf��|  �          AG�
��Q�A   ������\)B�  ��Q�A,�Ϳ�����33B�.                                    Bxf��"  �          AG\)�ڏ\A���=q��  B�#��ڏ\A$z��G���B�.                                    Bxf���  �          AL���陚AG���p�����C�
�陚A ���>{�W
=B��                                    Bxf��n  �          AE��Tz�A+���
=����B�\)�Tz�A;33��  �\)BӨ�                                    Bxf��  �          AL(��w
=A/\)��{���\B����w
=A?�������B׽q                                    Bxf��  �          AM��w�A(��������  B�L��w�A>{�C�
�[�
B�\                                    Bxf�`  �          AI���G�Aff��\)� ��B�p���G�A2�\�n{��p�B�Q�                                    Bxf�*  �          AE��A-p��C�
�hz�B�R��A4(�>\)?&ffB�#�                                    Bxf�8�  �          APz����RA6�H�XQ��p��B�  ���RA>�R<#�
<��
B�8R                                    Bxf�GR  �          AL������A4���(���?
=B�������A9p�?0��@FffB��H                                    Bxf�U�  �          ATQ�����A7\)�1��B{B�ff����A<z�?��@'
=B��                                    Bxf�d�  �          AZ=q�ə�A;
=�G
=�R�RB���ə�AAp�>�Q�?��
B�
=                                    Bxf�sD  �          A]����A@���b�\�l��B�33����AH��=u>uB�W
                                    Bxf���  �          A^�H����AD���N�R�U�B�aH����AK33>�G�?��B���                                    Bxf���  �          A`������AF�H�@  �E�B�������ALQ�?333@6ffB��                                    Bxf��6  �          AZ{��A?
=�G
=�S�B��
��AEG�>��?�(�B�p�                                    Bxf���  �          AH����p�A3��b�\���B�Ǯ��p�A<Q�\)�&ffB�
=                                    Bxf���  �          A=���p�A,�����H��  Bͮ�p�A8z�J=q�s�
B��                                    Bxf��(  �          A=p����RA&�H��R�0Q�B�����RA)�?c�
@�z�B��)                                    Bxf���  |          A:{���A��AG��r�\B�����A
=�\)�333B�u�                                    Bxf��t  �          A;
=�j=qA'��j�H���\Bڔ{�j=qA1G����G�Bؔ{                                    Bxf��  �          A:�R�>�RA*ff�k���33B�G��>�RA4(�����G�BѨ�                                    Bxf��  �          A:�H���A-G��qG���z�B�\���A7\)��G��
�HB�L�                                    Bxf�f  �          A=����A((���������Bî����A7
=��p���G�B�ff                                    Bxf�#  �          A<  ��Q�A)����\���HB�zῸQ�A8Q��G����B�8R                                    Bxf�1�  �          A;\)�	��A.ff�l����p�B�B��	��A8(����ÿ�{B��                                    Bxf�@X  �          A?���  A0���������B�  ��  A<(��(��:�HB��                                    Bxf�N�  �          AB�\��A6{�x������B�aH��A@Q쾳33�У�B�ff                                    Bxf�]�  �          A?�
�
=qA1���tz���G�B����
=qA;���Q��(�B���                                    Bxf�lJ  �          AAp���RA1���������B�uÿ�RA?�
���
�\B���                                    Bxf�z�  �          A?�
�=qA1��n�R��33B�u��=qA;���  ���HB�=q                                    Bxf���  �          A@zῇ�A7��U�����B������A?
=>�  ?�B�.                                    Bxf��<  T          A@��>�
=A/
=����(�B��\>�
=A>�\��Q�����B��                                    Bxf���  �          A?33>L��A,�����
���B��
>L��A=���z���
=B�
=                                    Bxf���  �          A@�ÿk�A,������Q�B��q�k�A=녿ٙ��=qB��H                                    Bxf��.  �          AF{��33A6=q�����B�� ��33ADQ쿃�
����B��                                    Bxf���  �          AB{��=qA-p��������
B��f��=qA>�H��\�33B���                                    Bxf��z  �          AC�?���A'33�ƸR��ffB��?���A=p��0���Q�B�u�                                    Bxf��   �          A?�
?=p�A'���Q���  B��?=p�A;��z��2�RB�Ǯ                                    Bxf���  �          A>=q?#�
A*�R��33��{B�.?#�
A;��У���z�B�Ǯ                                    Bxf�l  �          A>�R?��A!��  ��  B���?��A7\)�(���MG�B�ff                                    Bxf�  �          A@  @�A�����
��RB�{@�A2�H�h����33B�
=                                    Bxf�*�  �          A=G�@@��A	����R�(�B�z�@@��A'���\)����B���                                    Bxf�9^  �          A;
=?�{Aff��
=�ffB�.?�{A1p��>�R�mG�B���                                    Bxf�H  T          A8z�?�
=A(������p�B�W
?�
=A1���(���T(�B�Q�                                    Bxf�V�  �          A6�R@�A  ��ff��33B�33@�A-��0  �^�RB�\)                                    Bxf�eP  �          A6{@6ffA
ff��=q�Q�B�z�@6ffA%G��u���\)B��                                    Bxf�s�  |          A7�@0  A
=��\)���B���@0  A*�\�E�{
=B�                                    Bxf���  �          A4��?��A!G�����ffB��f?��A0zΎ��ڏ\B��                                    Bxf��B  �          A4��?޸RA$  �������B�Ǯ?޸RA1��G���z�B�=q                                    Bxf���  �          A7\)@Q�A#����R��ffB�ff@Q�A2�H����ҏ\B�aH                                    Bxf���  �          A5?��\A�\��33��33B�k�?��\A1p��   �"�RB��                                    Bxf��4  �          A8��@R�\A����G����B�
=@R�\A.�H��(��	p�B�z�                                    Bxf���  �          A:ff@_\)Aff��G���\)B�33@_\)A/��ٙ���\B�Ǯ                                    Bxf�ڀ  �          A;�
@J=qA'�
��{���B�� @J=qA4Q�333�\(�B��R                                    Bxf��&  �          A<z�@(�A0  �e���B��@(�A9�=#�
>aG�B���                                    Bxf���  �          A<��?��A5�4z��]�B��\?��A:�R?c�
@�=qB���                                    Bxf�r  �          A<��?��A6�\�+��R{B�#�?��A:�R?�ff@��\B�p�                                    Bxf�  �          A=?��A5��0  �V�RB��?��A:ff?z�H@�
=B��{                                    Bxf�#�  T          A>�R@  A5��,���R=qB���@  A:=q?��
@�{B��                                    Bxf�2d  �          A=�@(�A1��i����p�B���@(�A:ff=#�
>L��B��q                                    Bxf�A
  �          A>�R@"�\A3\)�I���up�B�p�@"�\A:{?\)@.{B�W
                                    Bxf�O�  �          A>=q@ffA2{�e���RB��{@ffA;
=>�?#�
B���                                    Bxf�^V  �          A@(�?�{A5���Z=q��(�B�W
?�{A=p�>\?���B��                                    Bxf�l�  �          A>{?ǮA4���N�R�|��B�W
?ǮA;�
?��@(��B��                                    Bxf�{�  �          A>{?��
A6�R�-p��S33B�=q?��
A;
=?�=q@��B���                                    Bxf��H  �          A=��?�(�A5��C33�o
=B��3?�(�A;33?:�H@c33B�=q                                    Bxf���  �          A=p�?��RA.=q��z����
B�B�?��RA:=q��
=��B��{                                    Bxf���  �          A?
=@.�RA4���Q��&=qB���@.�RA6=q?�\)@��
B�                                      Bxf��:  h          A=p�?��\A;33>���?�Q�B���?��\A/33@���A��B�                                    Bxf���  |          AG�>��A@�׿z�H��p�B��>��A;\)@=p�Ab�RB���                                    Bxf�ӆ  T          AX(��#�
AL��������B�Q�#�
AX  =���>�
=B�L�                                    Bxf��,  �          AX��=�\)AQG��`���p(�B�k�=�\)AX(�?n{@z�HB�p�                                    Bxf���  �          AW33>ǮAR�R�(Q��4Q�B���>ǮAT��?�ff@�{B���                                    Bxf��x  �          AIp�>k�AI��u���B��H>k�A?�@w�A�G�B�Ǯ                                    Bxf�  �          AK
=�\)AJ�H���
���HB�B��\)AAp�@u�A��RB�Q�                                    Bxf��  �          AB�R��A7\)@Q�A|  B�{��A(�@�33B
  B�z�                                    Bxf�+j  h          AA���RA>�R?c�
@�
=B�G����RA/�@��A��BÀ                                     Bxf�:  |          A=p��Y��A;33>L��?}p�B�B��Y��A/�@���A���B���                                    Bxf�H�  �          AB�\���AAp�>.{?L��B��f���A5@�z�A�  B�#�                                    Bxf�W\  �          AA�>�z�A:�\�!��D(�B�(�>�z�A=p�?���@߮B�33                                    Bxf�f  �          AAp�>���A6{��Q���33B��>���AA�����B��                                    Bxf�t�  �          AD��?���A2�H��G����B�\?���AB=q�Q��u�B�L�                                    Bxf��N  �          AA�?�ffA1���33����B�#�?�ffA@�Ϳh����G�B�                                    Bxf���  T          A>�\?�Q�A (������RB���?�Q�A8  � ���B�RB�=q                                    Bxf���  �          A;
=?�ffA{��
=��\B�8R?�ffA1G��N{��B��                                     Bxf��@  �          A=�@=qA�R��G��
=B��
@=qA-���y�����\B�                                      Bxf���  �          AA�@��\@���
�,�
BjG�@��\A�������υB�                                      Bxf�̌  �          A>=q@8��AQ����z�B�k�@8��A.{�j�H���\B�33                                    Bxf��2  �          A8  @���@��
��33���Bu�@���A���������
B���                                    Bxf���  �          A8z�?���A=q���H��B��\?���A.�\�W���ffB�33                                    Bxf��~  �          A>�R?���A���z��33B���?���A7��<(��dz�B�B�                                    Bxf�$  �          AJ=q?�
=A$(������	��B��R?�
=AA��QG��o\)B��)                                    Bxf��  �          AK
=?�ffA0(����H����B�u�?�ffAEp���\�G�B�k�                                    Bxf�$p  �          AJ�\?��AB=q�P���o\)B�k�?��AHz�?��@��B���                                    Bxf�3  �          AN{?�(�AE��X���tz�B�ff?�(�AK�
?}p�@�B��
                                    Bxf�A�  �          AL��?�\)AC
=�mp�����B�#�?�\)AK�?&ff@:�HB��                                    Bxf�Pb  �          AK33?z�HABff�a����B�Ǯ?z�HAJ{?Q�@n�RB��                                    Bxf�_  T          AI?:�HA9���������RB���?:�HAI��#�
�:=qB�aH                                    Bxf�m�  �          AG�?#�
A2�R��  �љ�B�z�?#�
AF{������
=B��                                    Bxf�|T  �          AG�?   A/
=��{���B��R?   AE���=q�\)B�Q�                                    Bxf���  �          AK33?
=qA8  ��{��(�B���?
=qAIG��n{��ffB��                                    Bxf���  �          AM녾���AC33�s�
���\B��q����ALQ�?!G�@333B��{                                    Bxf��F  �          AP�ý��
AA���H��z�B��q���
APz᾽p����B��                                    Bxf���  �          APQ��RAC�
������z�B���RAO�>L��?^�RB�p�                                    Bxf�Œ  �          AP�Ϳ.{AD����Q����\B�LͿ.{APz�>��?�33B��                                    Bxf��8  �          AQ���O\)AF�H������{B�uÿO\)AQ�?�\@\)B��                                    Bxf���  �          AR=q�0��AF�\��{���B�Q�0��AQ��>\?�B�                                      Bxf��  �          APz�.{AEp���33���RB�W
�.{AP  >�@ ��B�                                    Bxf� *  �          AO\)��RAC�
��p���=qB��Ϳ�RAN�H>�Q�?�{B�z�                                    Bxf��  �          AN=q�Q�AC33������{B���Q�AMp�>�@	��B�L�                                    Bxf�v  �          AO�
��{AF{�o\)��B����{ANff?Tz�@j=qB�                                    Bxf�,  �          AJ�R�fffA:�R������B��
�fffAI�Ǯ�޸RB�33                                    Bxf�:�  �          AH�ͼ�A;\)�������RB�=q��AH�ͽ�G���B�8R                                    Bxf�Ih  �          AH��>�  A;�������33B��\>�  AH�ý����B��q                                    Bxf�X  �          AD(��ǮA1��ff��{B�\�ǮAB=q�8Q��W�B��R                                    Bxf�f�  �          AB�H��A/33������p�B���AA�����Q�B��=                                    Bxf�uZ  �          A?����A'���G��ܣ�B��Ϳ��A<  ��(���=qB��H                                    Bxf��   �          A<Q��;�A  ��z��	z�B�(��;�A0(��:�H�fffB���                                    Bxf���  �          A9�@��A����(��Q�Bٙ��@��A+��Q���p�B�ff                                    Bxf��L  �          A:�R���A�
��\)���B�����A0z��?\)�m��B̏\                                    Bxf���  �          A:{��RA(��陚��BО���RA,���l(���Q�B�k�                                    Bxf���  �          A:{�#33A
�\��\�Q�B��)�#33A+33�p  ����B���                                    Bxf��>  
�          A?�
�=qAz���ff��\B�{�=qA4  �Y�����B�.                                    Bxf���  �          A?�
����A����ff�G�B��쿹��A8���/\)�S�
B�L�                                    Bxf��  �          A<��>��A  ���H�=qB��f>��A7��)���P(�B�\)                                    Bxf��0  �          A;�
�n{A=q���
�	p�B�
=�n{A6{�-p��Up�B�\)                                    Bxf��  �          A<(��E�A  ������HB�{�E�A7
=�$z��J=qB��q                                    Bxf�|  �          A<(��   Aff�����
B��R�   A8(��33�4��B��                                    Bxf�%"  �          A>{>��A"ff��{����B��3>��A;33��"=qB�k�                                    Bxf�3�  �          A<(�?z�HA$�����
��=qB��?z�HA9녿�  ��33B�\)                                    Bxf�Bn  �          A8z�?���A'���p���G�B���?���A733����.{B��                                    Bxf�Q  �          A8��?�{A(����{���\B���?�{A7
=��zΌ��B���                                    Bxf�_�  �          A8��?B�\A&{��\)�̸RB���?B�\A7��\(���Q�B�ff                                    Bxf�n`  �          A:�\>���A%�����ظRB��R>���A9����33��B�{                                    Bxf�}  �          A=�?�A((����
��=qB�aH?�A<  ��33���HB���                                    Bxf���  �          A=p���p�A'���{��p�B�  ��p�A;�
��(���{B��\                                    Bxf��R  �          A:ff��A'33�����=qB�W
��A9�u���B�L�                                    Bxf���  �          A:�R?�RA#���(���=qB�{?�RA9G���(���\)B��f                                    Bxf���  �          A:�H>�p�A%������B��>�p�A9p����R��(�B�\)                                    Bxf��D  �          A9p���Q�A'�������HB�𤽸Q�A8�׿0���\(�B��)                                    Bxf���  �          A-G�?���A��=q��=qB�?���A)���  ��p�B��                                     Bxf��  �          A33?E�A�\��G�B��3?E�AQ�?�ffA33B���                                    Bxf��6  �          AG�?��Az��33�@��B���?��A��?��A��B�{                                    Bxf� �  �          AG�>W
=A{��z���B�B�>W
=Az�
=q�I��B��                                    Bxf��  �          A�>��
Ap����;�B�{>��
Ap�@A�A�\)B��)                                    Bxf�(  �          AQ�?@  A
=q�P  ��z�B�?@  A�>W
=?��\B���                                    Bxf�,�  �          AG�?�
=A�\�mp���(�B�?�
=A
=��������B�\)                                    Bxf�;t  T          A��?�=qA�
����ң�B��\?�=qA\)�@  ��  B�                                    Bxf�J  �          A�?xQ�A
�\��=q����B�B�?xQ�A=q�B�\���RB��\                                    Bxf�X�  �          A�?
=A(����
��
=B�B�?
=A�R���FffB���                                    Bxf�gf  �          A?�\)A����
��Q�B��?�\)A  �Y����33B��=                                    Bxf�v  �          Az�?���A�������B��q?���A녿fff����B��                                    Bxf���  �          A��?s33A (�������\)B�p�?s33A=q��Q���B��                                    Bxf��X  �          Az�@UA(���z���B���@UA�H?���A:ffB��\                                    Bxf���  �          A
�H@1�@�{�   �Up�B�k�@1�A?�z�@�\)B�(�                                    Bxf���  �          AQ�@   @�(��5��B���@   A{>�=q?�B�Ǯ                                    Bxf��J  �          A�R?�ff@�33��Q���Q�B�33?�ffA
=���R�(�B��f                                    Bxf���  �          A�\?�\@�{�~{���
B�aH?�\A�\�p�����B��\                                    Bxf�ܖ  �          A33?�(�@�ff�}p���B�z�?�(�A�R�k���
=B���                                    Bxf��<  �          A(�@
�H@��H�mp��θRB�@
�HA33��R��{B�Q�                                    Bxf���  T          A�@.{@����Vff��ffB�{@.{A (�������
B��                                    Bxf��  �          A33@;�@�
=�J=q���B�p�@;�@��;�  �޸RB��                                    Bxf�.  �          A Q�@��H@���N�R����B_�H@��H@ڏ\�(�����RBlff                                    Bxf�%�  �          A Q�@q�@���n{��  Bf�
@q�@޸R�����\Bu�H                                    Bxf�4z  �          A ��@\)@�ff�R�\��(�B�W
@\)@���Q��$z�B��\                                    Bxf�C   �          Aff?��@��H�z�H��\)B��?��@�=q�z�H��p�B���                                    Bxf�Q�  �          Az�?���@����
=�
=B��?���A (����R�&=qB�u�                                    Bxf�`l  �          A  ?Q�@��������
B���?Q�Ap���\)��=qB�\)                                    Bxf�o  �          A
=?\(�@�
=�����
=B��)?\(�A �׿�����ffB��R                                    Bxf�}�  �          A(�?:�H@Ӆ��(��33B�W
?:�H@�\)��
=�X��B��                                     Bxf��^  �          A�H?n{@ۅ���H� Q�B��\?n{A (���=q���B��q                                    Bxf��  �          A�?\(�@�(��\)��z�B�B�?\(�A{�fff��Q�B��                                    Bxf���  �          A�H���
@�
=�s33��=qB�aH���
A{�.{����B�Q�                                    Bxf��P  �          A33>�ff@�ff�U���Q�B��>�ffA�\�����ffB��R                                    Bxf���  �          A33>���@�  �P  ��p�B�>���A�H�L�;\B�p�                                    Bxf�՜  �          Az�?��@��j=q��
=B���?��A\)���VffB���                                    Bxf��B  �          A  ?��
@��|����Q�B���?��
Aff�O\)��33B��\                                    Bxf���  �          AG�?h��@��������{B���?h��A(��Tz���\)B�z�                                    Bxg �  �          Az�?�=q@��H���R�Q�B��{?�=q@��������
=B��                                    Bxg 4  �          A=q?�(�@�{��{�2{B�#�?�(�@��E���z�B��=                                    Bxg �  �          A@G
=@�(����
�9�HB`=q@G
=@�=q�c�
����B�L�                                    Bxg -�  �          A ��@X��@�=q��G��C{BI  @X��@�(��{����Bt\)                                    Bxg <&  �          @�{@i��@;������W\)B  @i��@�
=���R��B\\)                                    Bxg J�  �          @�{@^�R@/\)���H�`G�B�@^�R@�(����R��B_G�                                    Bxg Yr  �          AG�@B�\@#33����r{B�
@B�\@�(����)�Bmff                                    Bxg h  �          A@XQ�?�����R�{�HA��@XQ�@�ff��G��@�HBL��                                    Bxg v�  T          @�@W�?�z�����z
=A��
@W�@�33���
�?��BJ=q                                    Bxg �d  �          @�G�@N�R@=q��p��k�\B��@N�R@��
�����%��Ba�                                    Bxg �
  �          @�  @XQ�?�Q���  �s{A��H@XQ�@�  ��Q��5��BN33                                    Bxg ��  �          @�\)@x��?(����{�q�A��@x��@X�����H�G
=B$(�                                    Bxg �V  �          @�ff@p  ?z�H�����rAj{@p  @j�H���A��B1�                                    Bxg ��  �          @���@O\)@�R��ff�n=qB
(�@O\)@����\)�)p�B^z�                                    Bxg ΢  �          @�ff@@��@{��
=�k�HBff@@��@��
���"�Bi
=                                    Bxg �H  �          @�
=@?\)@
=��\)�u�B33@?\)@�
=���,p�Bl
=                                    Bxg ��  �          A�R@2�\?�����\)��A��@2�\@�=q��{�I
=Bj(�                                    Bxg ��  �          AQ�@@  ?�������HA�G�@@  @�G��޸R�T33B[�                                    Bxg	:  �          A�H@7�@.�R����t{B.
=@7�@�����{�&�\Bw�
                                    Bxg�  �          A33@.�R@G
=��=q�r{BB�@.�R@����� B��3                                    Bxg&�  T          Aff@}p�?���{�v��A�33@}p�@�{�љ��Bz�B:ff                                    Bxg5,  �          A�R@b�\?��H�
=�~��A�@b�\@��
���<  B]p�                                    BxgC�  �          A�@(��@C33��
=�|�RBDp�@(��@��������*
=B�                                    BxgRx  �          A�R����@���Y����=qC������@�
=?޸RA7�C�)                                    Bxga  �          A\)���
@��H?�
=@�=qC
33���
@���@qG�A�=qC�                                    Bxgo�  �          Az���@���@�AD��C����@��@��
A�33C��                                    Bxg~j  |          A,Q���
@�G���  ��Q�Bӣ���
@ᙚ@�A|��Bԣ�                                    Bxg�  �          A3\)��R@�\)?�(�A�C	����R@Å@�z�AÙ�C33                                    Bxg��  �          A4������A ��?��R@ʏ\C)����@�=q@�ffA��C
�{                                    Bxg�\  �          A5G���\)A
=?W
=@�Q�C� ��\)@�=q@�33A�z�C��                                    Bxg�  �          A4z���=qA	G�?�\@%Cs3��=q@�@�Q�A�=qC��                                    BxgǨ  T          A4Q���A\)>�
=@	��CQ���@���@xQ�A��C�f                                    Bxg�N  �          A4(���ffA
=q>�@(�C����ff@�p�@�Q�A��C33                                    Bxg��  �          A7\)��p�A(����.{C��p�@�Q�@`  A�(�C��                                    Bxg�  T          A7\)��A(���  ���RC{��@��H@S�
A�{Ch�                                    Bxg@  �          A6�\����A�>�z�?���B�� ����A{@���A��
CO\                                    Bxg�  �          A7
=�љ�A�>�=q?�=qB��f�љ�A�\@�z�A��B��
                                    Bxg�  �          A8������A{?���@�ffB�������A z�@�  A�
=C�H                                    Bxg.2  �          A9���{A�H?#�
@J�HB����{A��@���A�C ٚ                                    Bxg<�  �          A:=q��ffA�@�RA1�B��H��ff@��@�33A�
=B��                                    BxgK~  �          A;
=��G�A\)@��A=G�B���G�@�=q@ǮB(�B�aH                                    BxgZ$  �          A5G����RA�@{A5�B������R@�=q@�{A�G�C :�                                    Bxgh�  �          A5����A��?�{@��B������@��@��A�=qC0�                                    Bxgwp  �          A/33�θRA�?��@�ffB�\�θR@�\)@��A��C�{                                    Bxg�  �          A)G���p�@�����
��C���p�@�  @UA�p�C��                                    Bxg��  �          A*{���
AG����R��Cc����
@�R@HQ�A�
=C��                                    Bxg�b  �          A)���(�A�׾.{�n{C �f��(�@��H@UA��HC#�                                    Bxg�  �          A%���33A{���8Q�B����33@���@Z�HA�Q�C �{                                    Bxg��  �          A+33��{A(��#�
�[�B�8R��{@�\)@AG�A�\)C ��                                    Bxg�T  �          A+
=�ƸRA	���p�����B��)�ƸRA��@ ��AZffB��R                                    Bxg��  �          A,Q���
=A�ÿ#�
�[�B����
=A z�@C33A�  C �)                                    Bxg�  �          A/�����AG�>k�?�z�B�\)����A�@��A���C =q                                    Bxg�F  �          A<Q���A  >��?��B�=q��A�
@���A���C@                                     Bxg	�  �          A0z���z�A��?@  @x��B�#���z�@�(�@�p�A��HC                                    Bxg�  �          A)�����
A��?��
@�=qB�����
@��@��\A�p�C�3                                    Bxg'8  �          A*{�ə�A(�?:�H@|��B�{�ə�@�(�@���A�
=C�                                     Bxg5�  �          A*{�ǮA	��>�Q�?�
=B��H�Ǯ@��
@�  A�
=CJ=                                    BxgD�  T          A$  ���A������=qB����A\)@��A]G�B���                                    BxgS*  �          A#���=qA	녿�����B�L���=qA=q@(�A]G�B��q                                    Bxga�  �          A%����
A33�W
=���B�\���
A
=@EA��HB�z�                                    Bxgpv  �          A(z���z�A녾aG����HB��f��z�A��@g�A��HB��                                    Bxg  �          A"�H���\A33��G��!G�B�q���\A{@p  A�33B�z�                                    Bxg��  �          A!����A��Ǯ��RB�����AQ�@`��A��\B��)                                    Bxg�h  �          A!p���{AQ�.{�xQ�B�p���{A�H@Q�A��B�\                                    Bxg�  T          A ������A�׿B�\���HB�������A�@N�RA��B�R                                    Bxg��  �          A$����z�Az�.{�vffB�B���z�A\)@L(�A��B�                                    Bxg�Z  �          A$(����A�\����Y��B�k����@��H@G
=A���B�.                                    Bxg�   �          A#��j=qA�H�(��^{B�u��j=qAG�?��
A!p�B���                                    Bxg�  �          A$���<��Az���R�4Q�B�p��<��A
=@ffAUBָR                                    Bxg�L  �          A&=q�L��A����T��B�=q�L��A�?��RA2�HB��                                    Bxg�  �          A&�R�[�A�H�}p���  B����[�Aff@U�A�z�B�{                                    Bxg�  �          A%������A{�����B�
=����A\)@�=qA��B�                                    Bxg >  �          A&{��
=Ap�=�\)>���B����
=A�@���A���B�#�                                    Bxg.�  �          A%�z=qA�׾�����z�B��{�z=qA\)@y��A��
B�W
                                    Bxg=�  �          A'33���
@�G�?xQ�@�=qCO\���
@�(�@��A���C=q                                    BxgL0  �          A'��
ff@�?���@�z�C�R�
ff@�(�@\��A��HC�\                                    BxgZ�  �          A'\)��G�@�\)?���A
=C33��G�@��@��
A�Q�C��                                    Bxgi|  �          A%p���ff@�Q�?��@�\)C
�R��ff@��R@�
=A���C��                                    Bxgx"  �          A&ff����@���?�(�@�Q�C������@���@�{A��\C�                                     Bxg��  �          A%G���ff@�p�?���A%�C	)��ff@���@��RA��HC�                                    Bxg�n  �          A&�H��@�=q?�ffA\)Ch���@���@��A�ffC8R                                    Bxg�  �          A'
=��{@���@�A333C�f��{@��@�p�A��C�q                                    Bxg��  �          A'�
��=q@ָR?��A&ffC^���=q@�@��A�\)Cu�                                    Bxg�`  
�          A'33��\@�{@��A<��C����\@�33@��Aә�C��                                    Bxg�  �          A!�����R@���@Tz�A�=qB�{���R@�  @�B�RC                                    Bxgެ  �          A �����@��R@w
=A��B�� ���@�ff@�(�B.{C5�                                    Bxg�R  �          A
=���@��@r�\A��B��R���@�@ٙ�B/
=C#�                                    Bxg��  �          A!���أ�@ۅ@33A@(�C���أ�@�\)@�z�A��
CO\                                    Bxg
�  �          A z���G�@��@@��A���C����G�@��
@��B
Q�C�                                    BxgD  +          A!p���z�@�z�@>{A�  C�
��z�@��@��HB�C��                                    Bxg'�  	W          A!G���{@�  @��A�p�C� ��{@}p�@��
B$��Ch�                                    Bxg6�  �          A  ��  @�@�Q�A�33C}q��  @R�\@�{B-C8R                                    BxgE6  T          A���z�@�=q@�p�A؏\C	����z�@N�R@��B)�C+�                                    BxgS�  �          A�����@��@��B��CJ=���?�{@��HB-�\C(E                                    Bxgb�  T          A����{@E�@���B(�Cs3��{>�=q@ʏ\B1�HC1�{                                    Bxgq(  T          A�����H@R�\@��\B��C�{���H?(��@�Q�B-Q�C.
=                                    Bxg�  
�          Ap����H@QG�@�p�B�C�)���H?5@��B&
=C-ٚ                                    Bxg�t  "          A�
��
=@O\)@�Q�B��C=q��
=?#�
@�{B%Q�C.�=                                    Bxg�  
�          A���=q@_\)@�Q�B�RC���=q?^�R@�G�B)p�C,�                                     Bxg��  �          A�
�Ϯ@c�
@��HB
{CB��Ϯ?aG�@���B,C,B�                                    Bxg�f  �          A���@^�R@���B��C�H��?(�@ϮB933C.W
                                    Bxg�  T          Az���
=@O\)@��B��Cu���
=>�{@�Q�B9(�C0��                                    Bxgײ  
�          A�R���R@4z�@�{B+�C�3���R�u@�33BE(�C4��                                    Bxg�X  	�          A���{@(Q�@�\)B(�
C0���{�\)@\B@�C5n                                    Bxg��  
�          A���@Q�@g�A��HCn��?��@��RBp�C)�{                                    Bxg�  
�          A��љ�@[�@{�A�  CY��љ�?��
@��B�RC(�q                                    BxgJ  �          A��߮@u@���A�Q�C=q�߮?��R@���B!(�C)��                                    Bxg �  T          AQ�����@S�
@�G�B�C������?��@�
=B4��C.�                                    Bxg/�  	�          @�����@(Q�@���B�HC�����>.{@�ffB4��C2.                                    Bxg><  
�          A(��Å@Q�@��B{C�3�Å���@���B(�C5p�                                    BxgL�  �          A�H��p�@/\)@�ffB��C33��p�>8Q�@��B.�C2J=                                    Bxg[�  �          A\)��p�@(Q�@�(�B��C\��p���@��B5\)C5(�                                    Bxgj.  T          A������?�33@��
B)z�C"�����Ϳ}p�@�33B1z�C=&f                                    Bxgx�  T          @�ff����@33@]p�B	33C�����>�@�(�B(��C.0�                                    Bxg�z  
�          @'��Y��>aG�>��HA�RC%}q�Y����?
=qB �C5��                                    Bxg�   �          @��H��\>�z�@9��BZ�\C+ٚ��\����@.{BF�RCO��                                    Bxg��  �          A
=���?�=q@���B0��C%=q�����{@�=qB2C@�\                                    Bxg�l  T          A
=q��G�@7
=@��\B=qC����G�>W
=@�=qB+�C2�                                    Bxg�  T          AQ�����@@  @��BQ�Cu�����>aG�@��HB0z�C1�3                                    Bxgи  �          A ����Q�@ff@�{B*�RC+���Q�#�
@�G�B8�C:�H                                    Bxg�^  �          @�  ���H@ ��@��B$�Cp����H���
@�
=B6�HC7�                                     Bxg�  �          @�����=q@p�@��RB�\Cn��=q<��
@�  B'�C3                                    Bxg��  
�          @������R@  @j�HBp�C�����R>���@�G�B�C0��                                    BxgP  
�          A�
���@/\)@�z�B �HC�����>���@��B�C0u�                                    Bxg�  T          @�{��ff@9��@Z�HAԣ�C���ff?}p�@�(�B�C*�                                    Bxg(�  �          A��ٙ�@.�R@n{A�\)C��ٙ�?333@��B�HC.#�                                    Bxg7B  T          @��\�QG�?�  ?���A��C\�QG�>��H?�  A�p�C+aH                                    BxgE�  T          @��R@!��E������\C��@!녿�{�e��HQ�C��=                                    BxgT�  T          @���@N�R��=q�:�H��33C���@N�R�W
=����=ffC���                                    Bxgc4  �          @�=q@0�������z�����C���@0���p  ���
�&
=C�+�                                    Bxgq�  �          @�@�����R�s33��C�g�@�������XQ���  C���                                    Bxg��  T          @�@��H��p������G�C���@��H��z��1G���ffC�O\                                    Bxg�&  T          A�H@����  ��33��C��@���j�H�N�R��  C�e                                    Bxg��  
�          A
=@�����������\C���@����z��o\)����C�Z�                                    Bxg�r  
�          A
=@�{��  ��ff��\C�ff@�{��G��fff����C��
                                    Bxg�  T          A=q@ҏ\��{�k���Q�C�AH@ҏ\�n{�@������C�C�                                    Bxgɾ  
Z          A33@�Q����\�0������C��@�Q���p��>�R����C��H                                    Bxg�d  T          A	��@�\)���׾B�\���RC�'�@�\)���H� �����C��                                    Bxg�
  "          A��@�G����R=�Q�?
=C�Ff@�G��~�R� ���YC��                                    Bxg��  �          A
ff@��
��
=>�=q?�ffC�e@��
����p��m�C���                                    BxgV  "          A�
@�z���
=?\)@j�HC���@�z����Ϳ��B�HC�w
                                    Bxg�  
�          AQ�@�{��
=?@  @���C��=@�{�����
=�0(�C�K�                                    Bxg!�  T          Aff@�G���33?���@�(�C�1�@�G���G�������C�W
                                    Bxg0H  "          Az�@�33���R?�=q@���C�)@�33�����G�� ��C�:�                                    Bxg>�  �          A  A ���e?�A<��C��)A �����ý�\)��
=C���                                    BxgM�  �          Az�A z���  ?k�@��C��HA z��}p��������HC��q                                    Bxg\:  T          A
�R@�����\)�\�\)C�T{@����c�
���z{C�]q                                    Bxgj�  �          A
{@�G���z�.{���C���@�G��\)�:=q��Q�C�<)                                    Bxgy�  T          A�@�Q��tz�?xQ�@���C�E@�Q��u��s33��ffC�@                                     Bxg�,  
�          A33@�z��7
=@\(�A�
=C��@�z�����?�
=A8Q�C��                                    Bxg��  
�          @���@�p��+�@S33AׅC��@�p��{�?��AO�
C�7
                                    Bxg�x  
�          @���@b�\�p��@���B]G�C���@b�\�U@�
=B#��C�W
                                    Bxg�  "          @�
=?333��@�Q�B��C�E?333�5@��B[Q�C��H                                    Bxg��  "          @�zΌ���k�@��\B�G�CTG������U@�  BE�HCvz�                                    Bxg�j  T          @�p���Ϳ�@�ffBd\)CXk�����tz�@Q�B\)Cm\)                                    Bxg�  �          @�\)�^�R�ff@u�B)�\CV��^�R�x��@��A���Cd.                                    Bxg�  �          @���b�\�`��@;�A�C`�q�b�\��\)?h��A��Cg��                                    Bxg�\  
�          @��
���
�n{@{B \)Cz����
��{>���@�p�C}޸                                    Bxg	  "          @`  ���R�	��?.{AR=qCc@ ���R�p���
=���Cd)                                    Bxg	�  �          @_\)�  �!G��\���
Cd@ �  �녿��
��z�C^#�                                    Bxg	)N  
�          @��
�U����=q����CPff�U�Tz�������CA�q                                    Bxg	7�  
�          @��\��녿�\)�/\)��CCxR���>��
�<(��ffC/z�                                    Bxg	F�  T          @���u>�Q��1���C.���u?���G���(�C�{                                    Bxg	U@  	�          @����33�J=q�:�H�  C>����33?333�<���33C*@                                     Bxg	c�  
�          @�ff���
�Y���\)���C<�
���
>�ff�%��p�C/xR                                    Bxg	r�  
�          @�Q���(�����z���p�C5z���(�?.{�\��
=C,s3                                    Bxg	�2  
�          @�33��\)?@  ��  ���RC,�
��\)?��R��
=�3�
C%ff                                    Bxg	��  
�          @��R����@A녿z���
=Ch�����@<(�?p��@�33C�                                    Bxg	�~  
�          Az���@�33=��
?\)CL���@c33@�\AdQ�C��                                    Bxg	�$  
�          A����=q@���?(��@��
CE��=q@k�@1G�A�\)C��                                    Bxg	��  T          A  ��G�@�=q?��HA;33C5���G�@g�@|(�A�ffC�                                    Bxg	�p  
�          A{��Q�@��\?���A�C��Q�@~{@uA�Cu�                                    Bxg	�  "          A�����@ə�@ ��AmG�C
B����@��@�\)B(�C�{                                    Bxg	�  �          A(  ��G�@�
=@0  As�
CJ=��G�@���@���B�C��                                    Bxg	�b  "          A%���R@��
@ffAP��C	Y���R@�z�@��
A��\C�                                    Bxg
  	`          A-����@��?��HA'�C�=���@�p�@�\)A�C�q                                    Bxg
�  �          A)����@��
?�Q�A(�Cu���@�p�@�p�A��C)                                    Bxg
"T  �          A&{��{@�Q�@J�HA�  C�)��{@���@��B33C�                                    Bxg
0�  �          A'\)����@�\@S�
A��
C0�����@���@�=qB�\C�
                                    Bxg
?�  �          A'\)�˅@�(�@S33A��C�
�˅@��\@ҏ\B�
Ck�                                    Bxg
NF  �          A$�����@�?�  ACǮ���@�
=@�(�A��C��                                    Bxg
\�  �          A"�H���@�  ?^�R@��C:����@�z�@�z�A��HC	��                                    Bxg
k�  �          A%���@�?�33A
=C�{���@�33@�z�A�{CQ�                                    Bxg
z8  �          A&=q�Ӆ@�33?�{A�RC
�Ӆ@\@�
=A�33C	^�                                    Bxg
��  �          A%G��ҏ\@��\?��
A�
C��ҏ\@Å@�z�A�ffC	�                                    Bxg
��  �          A%G����
@���?�  A��Ck����
@��@��\A�=qC	�                                     Bxg
�*  �          A%p���\)@��R?�p�A\)C#���\)@���@�G�A�33C
8R                                    Bxg
��  �          A%���Q�@�
=?�(�A{C5���Q�@���@�G�A�\C
E                                    Bxg
�v  �          A$  ��=q@�\)?�@�ffC^���=q@��@�  A�C	O\                                    Bxg
�  �          A%���z�A z�?��@��C ����z�@θR@���A���C��                                    Bxg
��  �          A(���ٙ�@�  @
=A8��C@ �ٙ�@��@��
B{C��                                    Bxg
�h  �          A,z���\)@�z�@�\AC�
C����\)@���@��HBQ�Ck�                                    Bxg
�  T          A,Q���z�A (�@ffA3\)C�R��z�@�
=@��B �RC�                                    Bxg�  T          A+
=��G�@��@
=A6ffCY���G�@�{@�  B  C
�\                                    BxgZ  �          A(Q����H@�
=?�{A#\)C�
���H@�G�@���A��C	}q                                    Bxg*   �          A)���
=A  ?�\)@�C ���
=@���@�\)A���C�q                                    Bxg8�  �          A'33��A�\?��\@��C =q��@��
@�(�A��C.                                    BxgGL  �          A(z���33@�ff@�
A4��C���33@�p�@�{B�
C

                                    BxgU�  �          A)���˅A33@�A=p�B����˅@��H@��B��CE                                    Bxgd�  �          A)������A{@z�AJffC 33����@�ff@�Q�B
{C	
                                    Bxgs>  �          A)����A�\?�(�A+�C (����@���@�
=B
=C33                                    Bxg��  �          A*ff�\A�@=p�A��B���\@�
=@��
B��CǮ                                    Bxg��  �          A)p���Q�A�\@E�A�\)B��q��Q�@��H@�{B�RC	
=                                    Bxg�0  �          A(����G�A ��@G
=A�p�B���G�@�\)@��B��C	                                    Bxg��  �          A&�R���R@�ff@B�\A�  B��R���R@�@љ�BffC	��                                    Bxg�|  �          A%���p�A��?��A'
=B���p�@�G�@��RB�HC8R                                    Bxg�"  �          A'���@�
=@5A{\)B��\��@�G�@�z�B\)C
�                                    Bxg��  �          A%�����A ��@1G�AxQ�B�
=���@���@��
B=qC�                                    Bxg�n  �          A%�����H@��R@Dz�A�Q�B��\���H@��@��HBz�C	.                                    Bxg�  �          A%p�����@�z�@5A\)B�Ǯ����@��R@˅B  C	�\                                    Bxg�  T          A$  ��@��@E�A�=qB����@�
=@�  BC
�                                    Bxg`  
�          A$(���@��\@4z�A�{B�L���@�p�@��B��C	�{                                    Bxg#  �          A%p����@���?��R@�z�C�3���@�Q�@�\)A�(�CW
                                    Bxg1�  �          A'���\)AG�?�{@�ffC ����\)@˅@�p�A�G�C��                                    Bxg@R  �          A#���A (�?�p�A��B�L���@�\)@�  A�
=C                                    BxgN�  �          A"{��{A z�?�A�B����{@��@�p�BffC�                                    Bxg]�  T          A#
=��(�@�\)?���A��B�  ��(�@���@��HA���C�f                                    BxglD  �          A&=q��33@�{@.�RAtQ�B����33@�G�@ə�B��C	�                                     Bxgz�  T          A"=q���HA
=@�AA�B������H@�=q@���B=qC��                                    Bxg��  �          A$(��Å@��@:=qA�=qC �\�Å@�
=@�=qB��C��                                    Bxg�6  �          A#\)��\)@��H@0��Az�HCaH��\)@�
=@���B�C                                      Bxg��  �          A!�Ϯ@�ff@6ffA��C�Ϯ@��H@���B
=CG�                                    Bxg��  T          A���p�@�=q@5A��C)��p�@�ff@�33Bp�C@                                     Bxg�(  T          A�\��{@�z�@:=qA���C ��{@�
=@�ffB�C�                                    Bxg��  �          A���33@�@EA�
=C ���33@�  @���B=qC�                                    Bxg�t  �          A��Å@�  @C�
A��C&f�Å@���@�Q�B��C
=                                    Bxg�  �          A   ���@��@@��A�ffC �R���@�@ə�B33C�=                                    Bxg��  �          A��ƸR@�@@  A�{C�
�ƸR@��@�p�B33C��                                    Bxgf  �          A�H��(�@�Q�@7
=A�(�C.��(�@��
@��HB�C�                                    Bxg  �          A33��
=@���@C�
A�\)B�u���
=@�Q�@���B Q�C
�=                                    Bxg*�  S          A{��(�@���@O\)A�
=B�����(�@���@�{B*�C�
                                    Bxg9X  �          A��z�@�{@@  A��B�
=��z�@�p�@�B"��C+�                                    BxgG�  �          Az���{@�z�@/\)A�ffB����{@�Q�@�B
=C��                                    BxgV�  �          AG���Q�@��@X��A�G�B��
��Q�@���@�G�B/  C�                                    BxgeJ  �          A�\��\)@�  @QG�A�(�B�8R��\)@��@�ffB'z�C��                                    Bxgs�  �          A�
��p�@�p�@?\)A�\)B����p�@�
=@�p�B"��C
�                                    Bxg��  �          A����H@��@�A^=qB�����H@��@�33B��C�q                                    Bxg�<  �          A���G�@��
@�A
=B����G�@��@���B"�C��                                    Bxg��  �          A
ff?z�@�\@�G�A��B�
=?z�@��R@�(�Bp(�B�u�                                    Bxg��  T          A=q@�\)@�@p��A���B-33@�\)@��@�
=B3Q�AĸR                                    Bxg�.  �          @�ff@�G�@�=q@o\)A���BF{@�G�@/\)@���B@
=A�
=                                    Bxg��  �          @��@�33@�\)@n�RA�B+�@�33@�R@�33B4��A�\)                                    Bxg�z  �          @��
@�
=@�
=@��B\)B!�@�
=?���@�BC��A�Q�                                    Bxg�   �          @�p�@���@~{@��BB�
@���?�G�@�Q�B:�AVff                                    Bxg��  �          @��R@�  @\)@s�
A�ffB
=@�  ?�ff@���B&�AqG�                                    Bxgl  �          @��R@�G�@w�@Y��A���B�@�G�?�33@�(�BAs�
                                    Bxg  T          A��@��\@��@_\)A�=qBff@��\@z�@�\)B �
A�=q                                    Bxg#�  �          @��@��@��@n{A��
B�R@��?�\)@��
B0�A���                                    Bxg2^  �          @�\)@�(�@��@6ffA���B7@�(�@J�H@��
B!\)A�33                                    BxgA  T          @�p�@y��@�=q@*=qA�BiQ�@y��@�(�@�
=B/��B:                                    BxgO�  T          @�
=@љ�@`  @�\Aup�A�
=@љ�@ ��@aG�AمA�ff                                    Bxg^P  �          @�
=@�G�@g�?�
=Ah(�A�@�G�@
=q@_\)A���A�ff                                    Bxgl�  �          @��R@ۅ@G�?�33AEG�A�@ۅ?�\)@?\)A��\Atz�                                    Bxg{�  �          @�=q@�\@�?Y��@�Q�A���@�\?�?��HAq�AT��                                    Bxg�B  �          @�(�@��@:�H?�{A<(�A�\)@��?��H@7
=A��AW�                                    Bxg��  �          @��H@˅@\)@  A�33B Q�@˅@z�@|(�A�A�ff                                    Bxg��  �          @�=q@�\)@w�@(�A�p�B�\@�\)@�@���B\)A�                                      Bxg�4  �          @�{@�ff@��
@n{A�ffBJ�\@�ff@#�
@�G�BF�A���                                    Bxg��  ,          A{?���@�G�@Z�HẠ�B��3?���@��
@ϮB\ffB�W
                                    BxgӀ  
�          A (�@   @�\@*�HA�z�B��@   @���@��B=�\By�                                    Bxg�&  �          A��@<(�@׮@VffA��
B��@<(�@���@�G�BL33BZ                                    Bxg��  �          A�\@Y��@��@]p�Aȏ\Bzp�@Y��@{�@���BI�BD��                                    Bxg�r  �          A=q@J=q@�33@@  A��B�k�@J=q@��@�G�B@=qBY\)                                    Bxg  "          A�@p�@�\)@7�A�Q�B�@p�@���@��
BD�B���                                    Bxg�  �          A��?�z�@�p�@)��A��B��\?�z�@���@���B?�RB�(�                                    Bxg+d  T          AQ�@3�
@�\)@Tz�A�=qB��@3�
@��@�(�BK�Bd                                    Bxg:
  �          A  @g�@�@'
=A��B}{@g�@�z�@��
B/��BV
=                                    BxgH�  �          A�\?Q�@���@"�\A�z�B���?Q�@�=q@���BAG�B��=                                    BxgWV  T          @�
=���\@�(�?�G�AS�
BǊ=���\@��@��B+B�{                                    Bxge�  
Z          @���{@�  ?��Az�Bٞ��{@��@�Q�B\)B�z�                                    Bxgt�  T          @�\)�h��@��?=p�@��HB�\�h��@�
=@�ffA�{B��                                    Bxg�H  
�          A����@ٙ���\)���HC �f���@��@W
=A��C޸                                    Bxg��  
(          A33���H@�����^�RCY����H@���@1G�A�
=C33                                    Bxg��  	�          A\)��ff@�G��!G����C\��ff@�
=@0  A���C�                                    Bxg�:  �          A�
���@�\)�����^{B�k����@ҏ\?˅A4Q�B���                                    Bxg��  �          A�
����@�p���33�UG�B�\)����@�  ?���A3�B��                                    Bxg̆  T          A	����@�녾�
=�333C������@��@4z�A��C�                                    Bxg�,  �          Az���(�@�z῎{��{C ���(�@�=q@G�A~�HC0�                                    Bxg��  �          A���G�@����N�\B�\)��G�@�
=?�z�A:=qB���                                    Bxg�x  "          A�
���@�{�n{��
=B�3���@θR@,��A���B�p�                                    Bxg  �          @�G���@\�����p�B����@θR?�G�@�B�Ǯ                                    Bxg�  �          A33���@����(����Q�B�  ���@�  ?c�
@�  B�{                                    Bxg$j  �          A�����@���p����B�=q���@�R?�\)A�HB�                                      Bxg3  �          A�R���
@�\)�
�H�aG�B��\���
@��
?�33A*�RB�z�                                    BxgA�  "          A�R���@�33�#�
��G�C\)���@߮?���@���B��=                                    BxgP\  "          AQ�����@�(��(��|  B��H����@�p�?�\)A�
B��{                                    Bxg_  �          A=q>���Ap�?#�
@���B��>���@陚@���B	��B�L�                                    Bxgm�  �          A��?���A?�Q�Ap�B��f?���@�(�@��HB  B�                                      Bxg|N  �          A  @�
A=q@(��A���B��@�
@��H@أ�B5�
B�
=                                    Bxg��  �          A  @(�A
�R@?\)A���B��{@(�@�
=@޸RB<�RB��\                                    Bxg��  �          A��@%�A��@/\)A��\B�u�@%�@��@�{B6�B�u�                                    Bxg�@  �          A
=@��Az�@UA�  B�
=@��@�p�@�{BG{B��=                                    Bxg��  �          A33@\)A�
@X��A�G�B��{@\)@���@�33BJ�B��f                                    BxgŌ  T          Ap�@�H@�33@xQ�A�z�B��q@�H@��@�=qBV�RB}(�                                    Bxg�2  �          AG�@��@�@�A�B��3@��@���@�z�BeQ�Br��                                    Bxg��  	�          A
{@,��@�@��A�
=B��@,��@q�@��Be�
BY�H                                    Bxg�~  �          @�p�@0  @�{@]p�A���B��@0  @e�@��
BVG�BR�                                    Bxg $  �          @�z�@1G�@�\)@5�A���B��R@1G�@��@�
=BCffBaQ�                                    Bxg�  T          @�(�@p�@�=q@.�RA�p�B�z�@p�@�=q@��BE�RBkff                                    Bxgp  
�          @���@&ff@�\)@-p�A���B~\)@&ff@I��@�\)BJBI�R                                    Bxg,  �          @ָR@K�@���@=qA�  Bm�@K�@\(�@��B6�B<�H                                    Bxg:�  �          @�33?�33@�
=@XQ�A�z�B��f?�33@2�\@���BjB^                                    BxgIb  �          @�Q�@��@���@HQ�A�\)B���@��@K�@�  BZ�BZ�                                    BxgX  �          @ڏ\?h��@�z�@r�\B�B�?h��@!G�@��
B�B�ff                                    Bxgf�  �          @�(�?��@��@{A�=qB�?��@p��@�=qBO�\B�8R                                    BxguT  �          @�=q?޸R@�=q@)��A�  B�\?޸R@x��@�33BM�
B��                                    Bxg��  "          @�Q�@.{@��
@W
=A��B�G�@.{@H��@��BXp�BDQ�                                    Bxg��  �          @��
��\@�ff@|��B�\B�\)��\@Vff@љ�Bzz�B�=q                                    Bxg�F  �          @�
=�   @�z�@�
=B:�\B�\�   ?�@�=qB�G�B�G�                                    Bxg��  �          @�p���33@��@�=qB8G�B�녾�33?�(�@�B��=B�Q�                                    Bxg��  �          @��ÿ�G�@�\)@�p�B3ffB�LͿ�G�?�\@�=qB��3C�\                                    Bxg�8  �          @��\��G�@�ff@�z�BD33BУ׿�G�?��@�\B�8RCn                                    Bxg��  �          @�?�ff@j=q@�G�B_��Bff?�ff=���@�=qB�aH@P��                                    Bxg�  
�          @��@��\?��@w�B��AJ�R@��\�Tz�@�  B�C��R                                    Bxg�*  �          @���@��H?\)@ ��At  @�33@��H��@ ��At��C��                                    Bxg�  
�          @�ff@��þ�=q@�A�
=C��@��ÿ�
=?�z�An�RC�AH                                    Bxgv  �          @�
=@�z���@Dz�AӮC��{@�z����@=qA��C�,�                                    Bxg%  T          @�(�@k�?\(�@���BN��AQ�@k���
=@��BA��C���                                    Bxg3�  T          @��?ٙ�?�Q�@�Q�B�ǮB3p�?ٙ�����@�B��C��                                     BxgBh  �          @�33��ff?�p�������C ��ff?�(�?z�HA33C"p�                                    BxgQ  �          @���\)?�{�p  ���HC)�)��\)@7
=�*�H��{CaH                                    Bxg_�  �          @�Q�����?�33�\����
=C)�)����@/\)�����  Cp�                                    BxgnZ  "          @����?���q���Q�C(�f���@AG��'
=��C�                                     Bxg}   �          @�����
?�p��n{���
C'c����
@J=q�p���=qC}q                                    Bxg��  �          @ʏ\���R?�p��<����ffC'Q����R@#33������C�3                                    Bxg�L  �          @ۅ���?���Q����C%+����@>�R�33��33C��                                    Bxg��  "          @�=q���?�=q�aG�����C${���@W
=�
=��Q�CT{                                    Bxg��  
�          @���\)?ٙ��fff���\C$)��\)@R�\�\)����C0�                                    Bxg�>  "          @����H?����s�
� ��C$�����H@S�
��R��=qCu�                                    Bxg��  �          @�Q�����>��~{�2�C233����@��W
=��C33                                    Bxg�  
�          @��h��>B�\����@Q�C1  �h��@p��Y����C�f                                    Bxg�0  �          @ə���Q�?G��g
=�  C+&f��Q�@   �.{�ϮCu�                                    Bxg �  �          @����?��l����\C%0����@S�
�ff��{CO\                                    Bxg|  T          @ҏ\���?�����
�C-���@#�
�QG���C33                                    Bxg"  T          @ə���
=>�  ��z��$C0����
=@�
�\�����C�R                                    Bxg,�  �          @����k��G�����LQ�C?�q�k�?�
=����>  Cu�                                    Bxg;n  
�          @��H�P  �\�����GCM��P  ?Y����ff�S�
C%n                                    BxgJ  T          @�ff�333��{�|���RCI�=�333?���~{�T
=C��                                    BxgX�  �          @\��g
=�����2��Cm
=��u��Q���CJY�                                    Bxgg`  
�          @�(��L(��2�\�����4��C])�L(���  ���H�gG�C8xR                                    Bxgv  T          @��
���\=�����G��%C2�q���\@Q��\(��	
=C\                                    Bxg��  �          @������s33���3p�C?�H���?�ff�����+�RC!u�                                    Bxg�R  �          @�{��(�>�  �Y�����C0�3��(�?�
=�3�
��  Cc�                                    Bxg��  �          @����\)���
�y���$G�C4B���\)?����XQ��
�Cp�                                    Bxg��  T          @�  �u?W
=�����(�C'���u?�
=������RCL�                                    Bxg�D  T          @����p�>���tz��#{C0����p�@	���J�H�(�C{                                    Bxg��  �          @�z���녿^�R��  �=qC=T{���?������Q�C%Q�                                    Bxgܐ  �          @��
�HQ��33����g{CXW
�HQ�?�\)��\)�{ffC O\                                    Bxg�6  �          @�p���\�Vff���\�\�Ck�3��\=L����
=p�C2ٚ                                    Bxg��  T          @����W����R�_33CpY�������(��3C5                                    Bxg�  �          @�\)�aG��y�������Z�C��=�aG������H£\CR��                                    Bxg(  T          @陚�������
��33�;z�C�3������G���=q�Cgz�                                    Bxg%�  �          @陚��p�������\)�M�
C~k���p��h����p��CX��                                    Bxg4t  �          @���z���33���R�S�RC���z�B�\���H��CU+�                                    BxgC  "          @�=q��\)�x����
=�U��C}�H��\)�(���{\)CPO\                                    