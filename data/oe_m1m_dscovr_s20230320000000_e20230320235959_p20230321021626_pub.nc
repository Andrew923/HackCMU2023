CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230320000000_e20230320235959_p20230321021626_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-21T02:16:26.678Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-20T00:00:00.000Z   time_coverage_end         2023-03-20T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxo�   T          @���1�?�(�@��BOffCY��1녿G�@�Bc�\CC�R                                    Bxo��  T          @����9��?�(�@��
BQp�Cff�9���c�
@��Bb�CE!H                                    Bxo�*L  
Z          @�z��Mp�@�@�p�BEQ�C���Mp��&ff@�=qB_=qC?��                                    Bxo�8�  �          @�ff�aG�?�z�@�z�B>��CxR�aG��=p�@��BS(�C?ٚ                                    Bxo�G�  �          @��R���@(�?�  A��
C�����?�\)@2�\A�ffC'Q�                                    Bxo�V>  T          @������
@"�\?�
=A�p�C�����
?�{@?\)A��C'��                                    Bxo�d�  �          @�  ���\@8Q�@G
=A�p�C�{���\?E�@�p�B'��C*�                                     Bxo�s�  
�          @�(���=q@C�
@'�A�  C:���=q?�
=@vffB��C%�=                                    Bxoł0  
�          @�=q��  @��>���@�z�C޸��  ?�{?�33A�C�H                                    BxoŐ�  T          @��
��ff@:�H?W
=A
{C(���ff@G�@G�A��C�                                    Bxoş|  T          @��H����@L(��aG���\C������@4z�?\A}G�CE                                    BxoŮ"  T          @�Q���
=@5������RC���
=@=q?��RAzffC�                                    Bxoż�  �          @�=q��{@>{�u�+�C����{@#33?��
A�=qC�                                     Bxo��n  �          @�(���p�@G
=����CW
��p�@7�?��\AQ�Cc�                                    Bxo��  
�          @�p����R@J=q>�G�@��\C�=���R@�@�
A��C��                                    Bxo��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxo��`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxo�  -          @����O\)@����,����(�B���O\)@����#�
��B�p�                                    Bxo��  T          @\�J�H@���7
=��p�B����J�H@�p�����Q�B���                                    Bxo�#R  �          @��
�}p�@{������\C8R�}p�@��
=#�
>���C�
                                    Bxo�1�  "          @��H����@`�׿�
=����C�R����@w
=>�G�@�ffCE                                    Bxo�@�  �          @����=q@G
=�5��RCB���=q@��Ϳ@  ��z�C0�                                    Bxo�OD  
�          @�33�z=q@H���P����CE�z=q@�zΉ��(  C�3                                    Bxo�]�  �          @\�O\)@��
�0����=qC 5��O\)@��R�����B�aH                                    Bxo�l�  	�          @����j=q@�G���
=�\(�C ���j=q@�33?��HA9�C ��                                    Bxo�{6  
Z          @���vff@�G�������C�f�vff@��\?E�@�G�C                                    BxoƉ�  
�          @�\)��ff@%�%���p�C.��ff@c�
�O\)���\Cٚ                                    BxoƘ�  T          @�Q���(�?�G���R����C"�)��(�@%��ff�S\)C�                                    BxoƧ(  �          @�z��N{@����Q�����B�(��N{@���?333@��B���                                    BxoƵ�  
Z          @���`��@s33�@����\)C�=�`��@�녿����B�B�                                    Bxo��t  
�          @Å�e�@�p��(Q��̣�C�)�e�@�{���
���B��f                                    Bxo��  
Z          @�33�;�@��!��ĸRB���;�@��>�{@Mp�B���                                    Bxo���  �          @��
�hQ�@�p��!��ď\C��hQ�@�(�=�\)?!G�B�\)                                    Bxo��f  �          @\�Q�@�Q��C33��=qCW
�Q�@��׾����ffB�k�                                    Bxo��  "          @�G���=q@z=q��p����RC)��=q@��
>Ǯ@p��C��                                    Bxo��  "          @�{�p��@<���[���C޸�p��@�=q�����S�
C
=                                    Bxo�X  �          @���4z�@s�
�1G����HB�  �4z�@�ff��{�b�\B��H                                    Bxo�*�  T          @\�mp�@HQ��b�\���C�
�mp�@��ÿ����PQ�CL�                                    Bxo�9�  �          @�=q�^�R@(Q���ff�/�C��^�R@��R����ffC                                       Bxo�HJ  
�          @�=q���R@G��B�\��
=CxR���R@�  �n{�C�3                                    Bxo�V�  
(          @�(���z�@<(��8Q�����C8R��z�@��׿c�
���C	��                                    Bxo�e�  �          @\���\@0  �8Q����C� ���\@-p�?W
=A (�C�                                    Bxo�t<  �          @Å��Q�@'��.{�˅C�=��Q�@z�?��RA;�C0�                                    Bxoǂ�  "          @˅����?�ff?fffA�HC#aH����?�ff?��HAz�HC*)                                    BxoǑ�  "          @�ff��Q�@   ?��HA-��C!����Q�?��@�A��HC*�                                    BxoǠ.  
(          @Ϯ��
=@   ?��A[\)C!����
=?c�
@
=A��RC+�=                                    BxoǮ�  T          @�G���G�?�\)?�Al��C"�\��G�?5@��A�z�C-O\                                    Bxoǽz  �          @����(�?��
?��RAQp�C#����(�?:�H@��A��HC-.                                    Bxo��   
�          @�G���{?�?��A8Q�C$���{?:�H?��RA��C-E                                    Bxo���  "          @�=q���?���?��HAUC"u����?O\)@p�A�=qC,&f                                    Bxo��l  T          @�(�����@
�H?�A�ffC)����?Tz�@1�Aי�C+8R                                    Bxo��  "          @ƸR��Q�@p�@Q�A�
=C�f��Q�?z�H@FffA��C)xR                                    Bxo��  
�          @�G����@�\?��HA~�\C�����?��@*=qA�G�C)^�                                    Bxo�^  "          @�  �Ǯ?�\>�z�@%C$33�Ǯ?���?�
=A'
=C'��                                    Bxo�$  "          @���33?��H>�{@@��C"5���33?��R?��A=�C&E                                    Bxo�2�  �          @У����@�>W
=?�=qC!^����?�z�?��\A3�C$�                                    Bxo�AP  T          @Ϯ��\)?�G���=q���C$8R��\)?��?0��@�(�C%=q                                    Bxo�O�  [          @�ff��z�?�Q�<#�
=��
C"z���z�?�33?��\A=qC$�q                                    Bxo�^�            @������@�R>���@7�CL����?޸R?�
=AZffC"�q                                    Bxo�mB  T          @����\)@.{?(�@��RC�{��\)@   ?���A��C�3                                    Bxo�{�  
(          @ȣ����@�\>���@2�\C�����?��?��HAW\)C"�H                                    BxoȊ�  �          @�����@  �u��CaH���@33?z�HAQ�C!&f                                    Bxoș4  �          @ȣ����@
=�J=q��\)C�f���@�H?\)@�z�C)                                    Bxoȧ�  �          @�z���\)?��R�B�\��C �{��\)@ff>Ǯ@h��C��                                    Bxoȶ�  T          @����
=?����U���C$���
=@:�H�Q���(�C�)                                    Bxo��&  
�          @�����{?��������HC%33��{@�ÿ�z��fffC+�                                    Bxo���  
�          @�\)��z�?��
�=p����
C#33��z�?�33>�z�@.{C"
                                    Bxo��r  �          @�  ��=q?���>u@p�C'5���=q?�ff?k�AQ�C*+�                                    Bxo��  
Z          @ƸR���\?�p�?�33A��C%#����\>�\)@��A�G�C1(�                                    Bxo���  �          @�Q���ff?�@ffA���C ����ff>�
=@@  A��C/�H                                    Bxo�d  
Z          @�G���=q?�@Q�A��
C#^���=q>B�\@9��Aۙ�C2{                                    Bxo�
  �          @�G���33?�(�@3�
A���C"33��33<#�
@S33A��C3�                                    Bxo�+�  
�          @�����?Ǯ@<(�A�  C#�q����B�\@Tz�A�=qC6                                    Bxo�:V  
�          @�  ����?��\@>�RA㙚C)W
���Ϳ.{@Dz�A�C;!H                                    Bxo�H�  "          @�G����\?z�@J=qA�Q�C-����\���@@��A�
=C@�                                    Bxo�W�  T          @ʏ\��G�?��
@W�B  C ����G��aG�@s�
B33C6�=                                    Bxo�fH  �          @�(���ff?^�R@H��A�\)C*�H��ff�aG�@H��A�33C=.                                    Bxo�t�  
�          @θR��ff@Q�@c33Bp�C�q��ff���
@�z�B�\C4.                                    BxoɃ�  �          @�z���  ?��R@p  B
=CB���  �k�@��B&�RC6��                                    Bxoɒ:  �          @������@�@g�B	C�����    @�\)B%\)C4�                                    Bxoɠ�  �          @��
��\)@=q@aG�B{C���\)>�  @�  B'�RC1�                                    Bxoɯ�  �          @�33��z�@.�R@VffA��C����z�?(�@���B)C,��                                    Bxoɾ,  T          @�33��  @E@N{A�=qC����  ?}p�@��B-z�C'��                                    Bxo���  �          @˅��(�@>�R@L(�A�33C=q��(�?fff@���B(C(�                                    Bxo��x  T          @�p���ff@,��@(Q�A���C�\��ff?u@i��B
ffC)�\                                    Bxo��  �          @�Q����H@I��@!G�A�
=CE���H?�33@q�B�\C$��                                    Bxo���  "          @�����Q�@J�H@A�  C�f��Q�?�33@Z�HA��\C"��                                    Bxo�j  �          @�Q����@H��@p�A�p�C����?�ff@`��B�C#�                                     Bxo�  T          @ҏ\��33@-p�@�RA��C&f��33?�ff@aG�BQ�C(��                                    Bxo�$�  T          @�=q��\)?�ff@k�B�C#� ��\)�\)@|��BC:!H                                    Bxo�3\  
�          @��H��
=?���@dz�Bz�C �{��
=��=q@�  B��C7�                                    Bxo�B  T          @�G���z�?У�@k�B	Q�C"^���z���H@�  B{C9ff                                    Bxo�P�  
�          @У���ff?�
=@h��B��C$����ff�!G�@w
=Bz�C:�                                    Bxo�_N  "          @�Q����?��\@b�\B�C&����녿8Q�@l��B
G�C;�                                     Bxo�m�  
�          @У���  ?�G�@i��B33C&���  �J=q@q�BC<�                                     Bxo�|�  "          @У���\)?�\)@n�RBz�C'޸��\)�p��@q�B�C>.                                    Bxoʋ@  
�          @�  ��{?��H@n�RB��C&޸��{�\(�@tz�B{C=u�                                    Bxoʙ�  "          @Ϯ���?��
@g
=B=qC&E����=p�@p��B�\C<�                                    Bxoʨ�  
(          @�ff��\)?��@`  B��C%�
��\)�!G�@l(�B{C:�{                                    Bxoʷ2  
(          @������?�\)@K�A�\C"����녾aG�@dz�Bz�C6h�                                    Bxo���  "          @�
=���
?�
=@K�A뙚C"�
���
�.{@eB
=C5��                                    Bxo��~  "          @�  ����?��R@^{BffC$8R���þ�@p  B{C9=q                                    Bxo��$  
�          @�\)���R?���@dz�B��C$z����R��@s�
B�C:@                                     Bxo���  �          @�p����?���@n�RB��C%Y���녿@  @xQ�B��C<}q                                    Bxo� p  �          @������?�p�@j�HB	�C&Ǯ����L��@r�\B(�C<��                                    Bxo�  
�          @�����
=?��@k�B	p�C%�H��
=�8Q�@vffB�RC;��                                    Bxo��  �          @�=q���H?��\@fffB�C&�����H�:�H@o\)B{C;Ǯ                                    Bxo�,b  �          @�=q��33?�ff@uB
=C#
��33�(�@�33BG�C:��                                    Bxo�;  T          @�
=���?�(�@n�RB33C!  ������@��HB
=C8�                                    Bxo�I�  
�          @Ϯ���\?@  @eB��C+�����\��@^�RB
=C@W
                                    Bxo�XT  
�          @�
=���\?L��@c33Bp�C+p����\����@]p�B  C?��                                    Bxo�f�  
�          @�ff����?k�@J=qA�\)C*�����ÿL��@L(�A��C<E                                    Bxo�u�  �          @Ϯ��z�>��H@I��A�z�C/\��z῕@=p�A�C?�                                    Bxo˄F  �          @�  ��p�=�G�@`  BffC2�
��p���@E�A�  CE�                                    Bxo˒�  
Z          @�=q��녾�Q�@\(�A�p�C7�������\@2�\A��CH#�                                    Bxoˡ�  T          @�����H���@�ffB-  CK�
���H�j=q@.�RAӮC]�=                                    Bxo˰8  T          @�
=�n�R�!�@�G�B.p�CV��n�R���H@(�A���CeW
                                    Bxo˾�  
(          @Ǯ�dz��*�H@��B1\)CX�=�dz���  @�A�33Cg�\                                    Bxo�̈́  "          @����W��3�
@��B-��C[�\�W�����@(�A��CiB�                                    Bxo��*  "          @����XQ��Q�@��B:33CW!H�XQ���  @%�A˅Cgz�                                    Bxo���  �          @����aG��
=@�p�B<Q�CR���aG���G�@0��A�Q�Cd�f                                    Bxo��v  T          @�G��G���H@���BD  CY�\�G����
@.{A���Cjz�                                    Bxo�  
�          @���k��8��@s33B=qCZ!H�k�����?�A�z�Cf�                                    Bxo��  �          @�33�X���Tz�@s33B�\C`k��X������?У�AxQ�Cj��                                    Bxo�%h  �          @���fff�B�\@r�\Bp�C\.�fff����?�\A�(�Cg�                                    Bxo�4  
�          @����`  �<��@xQ�B!33C\��`  ���?��A�
=Ch{                                    Bxo�B�  �          @�  �>�R�W�@|��B%G�Cdz��>�R��z�?�  A��
Cn�H                                    Bxo�QZ  
�          @�����(���(�@|(�B"��CM�=��(��k�@(�A�=qC]��                                    Bxo�`   �          @�G���(���ff@vffB��CGxR��(��P��@$z�AɮCX��                                    Bxo�n�  
�          @�Q�������@|��B$��CH�����S33@*=qAң�CY�                                    Bxo�}L  "          @�Q����H���\@i��B��CC�����H�:�H@"�\A�  CTz�                                    Bxő�  �          @�����
=�333@U�B=qC<
=��
=���@"�\A�CLxR                                    Bxo̚�  �          @�G���ff���
@L(�A�p�CBs3��ff�,(�@��A�  CP�                                     Bxo̩>  �          @\��z����@G
=A���COL���z��g�?\Ag�CY�q                                    Bxo̷�  T          @�G���녿\@P��Bp�CE�����;�@�A�{CS^�                                    Bxo�Ɗ  
Z          @�
=���R�s33@`  BffC?c����R�#�
@$z�A�\)CP}q                                    Bxo��0  T          @�{������@Dz�A�p�C9�)������(�@=qA�(�CIJ=                                    Bxo���  T          @����=q�!G�@R�\B�C;c���=q�
�H@#�
A�z�CL=q                                    Bxo��|  "          @��H������
@aG�Bp�C@�����(Q�@#33A���CR^�                                    Bxo�"  T          @�����z����@l��B�\CH����z��L(�@(�Aȣ�CY�H                                    Bxo��  �          @��
�����
=@e�B�\CB�����2�\@"�\A��HCS��                                    Bxo�n  T          @�Q���33�}p�@o\)B�HC@(���33�,��@1G�A�p�CRn                                    Bxo�-  "          @�����33���H@j�HB��CB� ��33�6ff@&ffA�G�CSٚ                                    Bxo�;�  �          @����R�:�H@Tz�B�C<�=���R���@"�\A�33CM�                                    Bxo�J`  
�          @�����  ���H@Q�B��CGǮ��  �Fff@G�A��\CU�                                    Bxo�Y  
�          @�{�����z�@=p�A��CC�����*�H?��A�Q�CO�                                    Bxo�g�  T          @ʏ\��\)�@^{B=qCNc���\)�o\)?��A�Q�CZ\)                                    Bxo�vR  T          @�ff���
�\)@o\)BCPO\���
��  @�\A��
C\޸                                    Bxö́�  "          @љ���  �<(�@mp�B
Q�CU#���  ��(�?�ffA�
C`.                                    Bxo͓�  T          @љ�������R@W�B �COff�����s33?�p�A}�CZ��                                    Bxo͢D  
�          @�G����?��H@\)A��C ����?   @HQ�A�ffC.��                                    BxoͰ�  "          @�(���ff?�ff@2�\A�ffC"}q��ff>W
=@S�
A�\)C1�H                                    BxoͿ�  "          @�33��
=?��?�(�AN�\C%8R��
=?8Q�@A�p�C-n                                    Bxo��6  �          @�z����H@{?Y��@�\C�����H?�  ?�Q�A�Q�C#�3                                    Bxo���  �          @�33��(�?�p�?��A5G�C"\��(�?�{@A�z�C)�                                     Bxo��  �          @ҏ\�Ǯ?�R?�Q�A�\)C.^��Ǯ����?��RA�p�C7�                                    Bxo��(  T          @�����\��z�@33A�33CD�)���\���?}p�A\)CK�3                                    Bxo��  �          @�33��z�@��
>�ff@z=qC���z�@�G�@0  A�C	W
                                    Bxo�t  �          @�������@��?(�@�=qC�)����@Z=q@%�A�ffC�\                                    Bxo�&  
�          @����p�@U?h��@�z�C}q��p�@\)@=qA�Q�CT{                                    Bxo�4�  �          @�����33@!�?J=q@��Cz���33?�=q?�z�A���C#B�                                    Bxo�Cf  
�          @���ff@<(�?0��@�C�q��ff@\)@G�A��HC^�                                    Bxo�R  
�          @�ff���@J=q?:�H@�Q�C�����@=q@
=qA�z�C��                                    Bxo�`�  �          @׮��
=@R�\?�\)A;�C#���
=@{@1�A���CǮ                                    Bxo�oX  �          @׮��33@[�?�  AN{C�=��33@�\@=p�A���C��                                    Bxo�}�  �          @�  ��\)@W�?�z�A�\C�=��\)@��@(��A��CG�                                    BxoΌ�  �          @�\)���
@Z=q?��A>�HC�R���
@�@7
=A��Cz�                                    BxoΛJ  T          @�p���\)@O\)?�A�ffC^���\)?�
=@O\)A�G�C �)                                    BxoΩ�  
�          @أ�����@G
=@'�A�z�C\����?�p�@r�\B\)C$��                                    Bxoθ�  �          @�����@Y��@(�A��CQ�����?�Q�@c33B ��C�f                                    Bxo��<  
�          @�{��(�@X��@ ��A�{C�\��(�@G�@XQ�A�Cn                                    Bxo���  T          @�\)��p�@fff?�  A-�Cn��p�@$z�@3�
A�
=C��                                    Bxo��  
�          @�ff��\)@s�
>��@�  C33��\)@HQ�@{A�\)CJ=                                    Bxo��.  "          @����@x��>��
@0  CB����@QG�@Q�A�33C�{                                    Bxo��  �          @�ff���H@j=q>�Q�@G
=C�{���H@B�\@�
A�33Cp�                                    Bxo�z  "          @ָR��{@x��>\@QG�Cz���{@O\)@�A�  C@                                     Bxo�   "          @�{��ff@��?�R@��\CT{��ff@W�@!G�A��C\                                    Bxo�-�  T          @�\)����@��H?E�@ӅC:�����@y��@>{A��HCh�                                    Bxo�<l  "          @�ff����@�Q�?5@�33C\)����@w
=@8Q�AˮCY�                                    Bxo�K  T          @�
=���@��H?W
=@�{Cc����@Z�H@2�\A�ffC��                                    Bxo�Y�  T          @ָR��p�@��R?&ff@���Cٚ��p�@Z=q@#�
A�p�C�H                                    Bxo�h^  
(          @ָR��33@xQ�?�  A	�C���33@<��@-p�A��C!H                                    Bxo�w  �          @׮��Q�@o\)?�33A�G�C����Q�@��@\(�A�CxR                                    Bxoυ�  	�          @ָR����@.�R@Z=qA�ffC� ����?L��@�G�B �\C*��                                    BxoϔP  
�          @�p���
=@.�R@Dz�A�{CY���
=?xQ�@\)B��C)s3                                    BxoϢ�  �          @����G�@Tz�@�
A�G�C����G�?���@fffB=qC�
                                    Bxoϱ�  �          @�p���(�@mp�?��RA+�C^���(�@,��@5�AɮCT{                                    Bxo��B  "          @�(����\@@  ?�
=A�33C����\?�  @FffAߙ�C"�H                                    Bxo���  
�          @�z����
@P��?���A?�C޸���
@\)@/\)A�33CG�                                    Bxo�ݎ  
�          @��H��ff@Z=q?�z�AE��C���ff@
=@5A�z�C�{                                    Bxo��4  T          @�=q��  @Vff?���A9�C����  @@.�RA�=qC��                                    Bxo���  �          @�(����@l(�?�Q�A%��CxR���@-p�@1G�A�{C.                                    Bxo�	�  T          @����z�@n�R?�=qA�CL���z�@333@,��A�p�C��                                    Bxo�&  
�          @�z���G�@^{?�\)A\)C�R��G�@#33@&ffA���CJ=                                    Bxo�&�  �          @������@O\)?��
A2ffC#�����@�@(��A���C\                                    Bxo�5r  
�          @�p���
=@E�?�(�AL  C����
=@33@.�RA��
C E                                    Bxo�D  
�          @Ӆ��\)@9��?���A=C.��\)?���@"�\A�  C!B�                                    Bxo�R�  T          @�p���=q@K�?��
Ax��CJ=��=q?��R@C33A�Q�C O\                                    Bxo�ad  
�          @����z�@�\?�z�A�\)C����z�?�33@0  A�ffC(�                                     Bxo�p
  
�          @�(���
=@�R?�(�Aw33C����
=?�33@*�HA�G�C&5�                                    Bxo�~�  
Z          @���33@
=@��A��C
=��33?�{@>�RA��HC)B�                                    BxoЍV  T          @����@0  @
=A��C���?�(�@H��A�\)C%s3                                    BxoЛ�  T          @Ӆ��\)?�=q@s�
B�HC�
��\)��@�\)B!G�C5}q                                    BxoЪ�  �          @��
���
@\)@��B'�C�f���
�#�
@�{BB
=C4z�                                    BxoйH            @�(���p�?�@��
B%�C O\��p���@���B2��C:W
                                    Bxo���  �          @���z�?�Q�@z�HB��C!�
��zᾙ��@�  B=qC7J=                                    Bxo�֔  ]          @ָR��  ?�(�@�(�B��C!  ��  ��Q�@��RB&�C8\                                    Bxo��:  
Q          @���ff@�\@|(�BffC����ff<#�
@�{B'p�C3ٚ                                    Bxo���  �          @�\)���\@ff@FffA�CY����\>�@n{B\)C/Q�                                    Bxo��  T          @�
=���
@:=q@j=qB=qC����
?n{@��\B0�
C(��                                    Bxo�,  T          @�ff����@2�\@8Q�Aʏ\C������?�
=@u�B  C'��                                    Bxo��  �          @����ff?��@5�A�  C!�H��ff>���@XQ�A�
=C0                                      Bxo�.x  �          @�ff��p�?�=q@Q�A��HC&Ǯ��p�����@aG�A��\C7
=                                    Bxo�=  �          @���{?�\)@4z�A�=qC)Y���{��\)@AG�AׅC6�q                                    Bxo�K�  T          @ָR��ff=L��@��B�HC3n��ff��\)@s�
B
=qCG�{                                    Bxo�Zj  
�          @�  ��G�>\@�ffB%ffC/����G���@�z�B�RCFaH                                    Bxo�i  
�          @������ͽ�G�@��B.33C5T{�����p�@��B�\CL=q                                    Bxo�w�  �          @�G����׾���@���B?C9������'�@��B��CR#�                                    Bxoц\  T          @�G���=q��(�@��\BB�CC�\��=q�XQ�@\)B  CZ
=                                    Bxoѕ  
�          @ٙ��|(��@���B>��CR�R�|(���33@\(�A���Ccٚ                                    Bxoѣ�  
�          @����(���z�@��\B1�HCG����(��j=q@dz�A�p�CZE                                    BxoѲN  
Z          @�����
=���
@��B0G�CF  ��
=�a�@g
=A��CX                                    Bxo���  
�          @�ff���!G�@��B0(�C;=q���-p�@���B
=CP��                                    Bxo�Ϛ  "          @�ff���?@  @w�B�\C,xR������@s33B��C>��                                    Bxo��@  
�          @�  ��
=?��@s�
B��C)����
=�B�\@xQ�BC;��                                    Bxo���  
�          @߮���>8Q�@o\)B�C2E��녿�G�@Z�HA�p�CB��                                    Bxo���  T          @�G����\>�p�@��\B-�HC/�)���\����@�\)B��CG�q                                    Bxo�
2  �          @�\���<��
@���BO�\C3����p�@�\)B3�\CQh�                                    Bxo��  T          @����H��@�p�B;{C:����H�0��@���B\)CQ��                                    Bxo�'~  T          @�33������33@�B%�\C7� �������@�G�B33CLB�                                    Bxo�6$  	�          @��H��{>#�
@��\B+�HC2:���{�   @��B�\CI{                                    Bxo�D�  "          @�=q��>aG�@���B+{C1�����@���B�CHY�                                    Bxo�Sp  
�          @�Q���  ?!G�@��B/�C,�=��  ��=q@�z�B&
=CE�\                                    Bxo�b  
�          @�����=q>���@��HB.\)C/}q��=q���
@���B Q�CGQ�                                    Bxo�p�  "          @�  ��>�@��RB4C2p����z�@�Q�B �RCJ�                                     Bxo�b  �          @����{���@�
=B4��C5�=��{��@���BQ�CM5�                                    BxoҎ  "          @߮���;#�
@�\)B5�HC5�f������@���B=qCM}q                                    BxoҜ�  �          @�Q���>�p�@��RB4z�C/��������@��
B%(�CH��                                    BxoҫT  
�          @�33��ff>��@�=qB6G�C/G���ff��{@��B'\)CH��                                    Bxoҹ�  "          @�33��G�>�  @��B2�
C1+���G����H@�33B!�\CIE                                    Bxo�Ƞ  
�          @�33��33>#�
@�{B0=qC2+���33� ��@�Q�B��CI��                                    Bxo��F  T          @����p��#�
@��B+��C4k���p��Q�@��B�\CJp�                                    Bxo���  �          @�=q���\�k�@���B/��C6�\���\�@��BffCL��                                    Bxo���  "          @ᙚ��  ��G�@�p�B=33C9B���  �)��@�
=B�\CQ�                                    Bxo�8  �          @����{�k�@��RB4Q�C6���{�
=@��
B=qCM�{                                    Bxo��  "          @߮���׾�p�@�=qB/
=C80�������H@�B�\CM                                    Bxo� �  T          @�p�����?Y��@�33Bp�C*�����ÿ�\)@�G�B�RC@�                                    Bxo�/*  �          @�{��@*�H@UA���C�{��?u@�p�B(�C)��                                    Bxo�=�  �          @�
=����@(��@2�\A��C�\����?�@i��A��C(��                                    Bxo�Lv  "          @߮��z�@6ff@{A��C33��z�?�  @]p�A�33C%��                                    Bxo�[  T          @ڏ\��p�@ff@7
=AʸRC����p�?(��@_\)A��C-T{                                    Bxo�i�  
�          @�{��=q@u�>��@c�
C:���=q@Q�@G�A�
=CO\                                    Bxo�xh  �          @�Q���{@���4z���=qC���{@��
����z�C�\                                    BxoӇ  T          @޸R��{@�33�A���\)C
޸��{@�  �z�H�C.                                    Bxoӕ�  "          @޸R����@�{��33�8Q�C޸����@{�?�p�AD��C�{                                    BxoӤZ  �          @�
=���R@z�H�
=��z�C����R@s33?��A33C^�                                    Bxoӳ   "          @�  ��@j=q���xQ�C\)��@W
=?��HA@��C��                                    Bxo���  
�          @޸R����@L��?���AC�����@�H@��A��HC:�                                    Bxo��L  �          @������@e�?�
=Ab{C�H����@!G�@C33A��
C�                                    Bxo���  
�          @ۅ��ff@E?�z�A>�\C����ff@��@%A��C�q                                    Bxo��  
�          @�33����?^�R@Z�HB�C)�������(��@]p�B{C;ٚ                                    Bxo��>  
�          @ڏ\�^�R��33@�{Bbz�CI޸�^�R�k�@��B({Cb�{                                    Bxo�
�  
�          @ڏ\�l�Ϳ��H@�33B\CF&f�l���^{@��B'�
C_(�                                    Bxo��  �          @�=q��z�B�\@��HBO
=C>W
��z��<(�@���B&�CWc�                                    Bxo�(0  �          @��H��ff�u@��
BO�C7E��ff�\)@���B1Q�CR�
                                    Bxo�6�  
�          @�����ÿG�@�BG�
C>aH�����8��@��
B ��CV\                                    Bxo�E|  "          @أ��{�?˅@�=qB<�\C
=�{���@��BJ(�C;                                    Bxo�T"  T          @�ff�C�
@�{@��B�B���C�
@��@��\Bdz�CO\                                    Bxo�b�  �          @�ff�*�H@r�\@�(�B?�B�aH�*�H?��@ǮB�.C@                                     Bxo�qn  
�          @��Ϳ���@o\)@��\BS  B�uÿ���?�z�@��B�#�Ck�                                    BxoԀ  T          @�z��(�@i��@�(�B:(�B��\�(�?���@�\)B��C��                                    BxoԎ�  "          @��������@�Q�B.33C9c������H@�(�B�
CMٚ                                    Bxoԝ`  
�          @����G��Q�@���BU\)C?���G��C33@�{B+33CY!H                                    BxoԬ  
�          @�(��\�Ϳ�\)@��\Bhz�CE���\���\(�@��\B3{C`�3                                    BxoԺ�  �          @��H��{��G�@�Q�BMC5���{��\@��B3�CP�H                                    Bxo��R  �          @��
���׿���@��RBE\)CE!H�����W
=@�{B�CZ(�                                    Bxo���  "          @�=q�A���\@���B`G�CY)�A���@�{B��Ck��                                    Bxo��  
Z          @�33�:�H��@�=qBz�CIٚ�:�H�dz�@�G�B>(�Cf�=                                    Bxo��D  
�          @�{�:=q��G�@��Bm=qCS0��:=q�~{@�  B)�Ci�                                    Bxo��  
�          @�z��{�}p�@��HB�u�CI��{�aG�@��
BM33Cj��                                    Bxo��  �          @�Q��
=��ff@��
B��C@��
=�H��@��HBdz�Cl)                                    Bxo�!6  "          @�
=����?�@���B�B�C%&f�������@���B��Ce@                                     Bxo�/�  �          @޸R�G�    @��Bz
=C3�R�G��p�@��RBX
=CZ@                                     Bxo�>�  �          @�
=�hQ���@��
Bh
=C<���hQ��:=q@�(�B?Q�CZ��                                    Bxo�M(  �          @�\)��(��8Q�@�  BR��C=���(��:=q@�
=B,\)CW0�                                    Bxo�[�  T          @�  �|�Ϳ�@�  B]33C;s3�|���4z�@�G�B8Q�CW�=                                    Bxo�jt  
�          @�
=�q�>�=q@�\)Bb=qC/���q��ff@��HBL33CQ�                                    Bxo�y  
�          @�ff��
=?�\)@�ffBA��C#���
=���@�Q�BQ�C9�{                                    BxoՇ�  
�          @����(�@z�@��RB?  C�R��(�    @�ffBXffC4�                                    BxoՖf  �          @����\)?�33@�z�BGC����\)�.{@�33BRffC=.                                    Bxoե  T          @ᙚ�\)���
@�ffB\  C5!H�\)�Q�@�B@ffCR�{                                    Bxoճ�  �          @�\)�xQ쾊=q@�G�B`G�C7��xQ��%�@�B@
=CU�f                                    Bxo��X  T          @޸R���;�=q@�(�BJ�RC7xR������H@�=qB/=qCP�=                                    Bxo���  T          @��
���\>L��@��BF�HC1�=���\�   @���B4�CK�)                                    Bxo�ߤ  T          @�\)���
?���@��B7��C&�f���
�xQ�@�B8�C?ٚ                                    Bxo��J  �          @޸R����?��
@y��B	�C&�R�����Ǯ@��HBQ�C8\                                    Bxo���  T          @�p�����?�@��HBp�C"� ������G�@�p�B33C533                                    Bxo��  T          @���
=?˅@z�HB
�C#����
=��G�@�\)BC533                                    Bxo�<  T          @�p����@G�@k�B ��C���>\@�B�HC0                                      Bxo�(�  T          @�z���(�@��@S33A�(�C5���(�?#�
@xQ�B	�RC-�                                    Bxo�7�  �          @׮���\?���@��
B��C"� ���\�.{@�p�B#�HC5�3                                    Bxo�F.  �          @������?�@�=qBffC 0����=�\)@��RB$��C3=q                                    Bxo�T�  +          @����?�z�@��BffC�\���=�Q�@��\B-G�C2�H                                    Bxo�cz  
�          @�����  ?�\)@�ffB(�C#�)��  ��\@�(�B0��C:
                                    Bxo�r   
�          @�����=q?���@��B)Q�C$c���=q��@�
=B0��C:��                                    Bxoր�  
�          @�=q��33?��@�z�B �
C�\��33�#�
@���B2ffC4#�                                    Bxo֏l  �          @�33��p�@�\@�G�B33Cu���p�>B�\@�  B/�C1�
                                    Bxo֞  "          @ۅ��Q�@Mp�@h��Bz�C�q��Q�?\@��B*��C"Q�                                    Bxo֬�  
Z          @ۅ�qG�@�p�@�A��
B�(��qG�@u@�p�B=qCu�                                    Bxoֻ^  �          @���l(�@�?�p�A�ffB�  �l(�@�ff@���B33C��                                    Bxo��  T          @ᙚ�xQ�@�
=?�Q�A��B�ff�xQ�@�G�@UA�C
=                                    Bxo�ت  �          @�G��8��@�(���{�1�B�R�8��@�Q�@(�A�33B�aH                                    Bxo��P  �          @���B�\@��
@{A��RB�q�B�\@�G�@��HB��B���                                    Bxo���  
�          @�{�fff@�33@�A���B��f�fff@~�R@�=qB\)C&f                                    Bxo��  T          @���{@>{@XQ�B33CB���{?�@���B)�
C"L�                                    Bxo�B  �          @�������?��@�\)B-\)C'}q���ͿW
=@���B/��C=��                                    Bxo�!�  T          @�����G�?�R@�G�Bp�C-L���G����@�ffB\)C?�                                    Bxo�0�  T          @����Q쾀  @��\B��C6����Q��z�@x��BG�CG��                                    Bxo�?4  
�          @�
=��33?�{@�=qB.�
C$Y���33�(�@�
=B5��C;+�                                    Bxo�M�  �          @�
=���R?�\)@�\)B7\)C&�����R�c�
@���B9z�C>                                    Bxo�\�  T          @�{���?�z�@���B#��C�
���<#�
@��B5�C3�{                                    Bxo�k&  T          @�p���(�@ ��@�ffB�
C����(�?333@�33B2
=C+޸                                    Bxo�y�  T          @��
��@@�p�B �Cff��>�@�\)B:�RC.\)                                    Bxo׈r  
�          @�(���z�@�@���B$�\C�)��z�>�p�@���B=(�C/n                                    Bxoח  �          @ۅ��=q@ ��@�p�B�HC���=q?8Q�@��B2p�C+}q                                    Bxoץ�  T          @��
��p�?���@�
=B5�RC!޸��p����@�p�B?�C:�                                    Bxo״d  "          @��
���׾B�\@��B7�C6B������33@��RB"CK8R                                    Bxo��
  T          @��H��\)�L��@�p�B8z�C6c���\)��
@�
=B#��CK��                                    Bxo�Ѱ  �          @��H����?!G�@�p�B-z�C,�R���Ϳ���@���B(G�CA�R                                    Bxo��V  �          @ۅ���
��Q�@��B2��C5����
��z�@�p�B CIk�                                    Bxo���  T          @ڏ\��{=L��@�B-C3aH��{��p�@�33B��CG=q                                    Bxo���  �          @�33���<#�
@��\B(�\C3���녿�p�@�  B�CF�
                                    Bxo�H  T          @ۅ��Q�>��@�(�B�HC/����Q쿣�
@��RB�\CA��                                    Bxo��  �          @��H���\>���@���B��C0�q���\��=q@�=qB(�CB�                                    Bxo�)�  "          @ڏ\���H>L��@�  B�C1�
���H��z�@�Q�B��CB�\                                    Bxo�8:  T          @�=q��\)>���@�G�B33C/�=��\)���@x��B=qC?Ǯ                                    Bxo�F�  T          @����z�=�Q�@�z�B��C3���zῼ(�@w�B
�
CC33                                    Bxo�U�  �          @�=q��G�>�z�@}p�B  C0����G����H@q�BC@O\                                    Bxo�d,  �          @��H����?�@}p�B�
C.!H���ÿs33@x��B
C=Ǯ                                    Bxo�r�  
�          @�����?@  @��BG�C,B�����W
=@��HB�C<��                                    Bxo؁x  �          @�p�����?k�@��
B�C*!H���ÿE�@�z�B�HC<W
                                    Bxoؐ  T          @�p���\)?u@��B�C*
��\)�#�
@�p�Bp�C:��                                    Bxo؞�  T          @�p���?B�\@w
=B��C,W
���8Q�@w
=B�C;@                                     Bxoحj  �          @����R?O\)@u�BQ�C+�3���R�+�@w
=BffC:�                                    Bxoؼ  �          @�����{?�z�@a�A��\C&���{    @s33B  C3��                                    Bxo�ʶ  "          @�\)��Q�?Y��@=p�Aȣ�C,=q��Q쾙��@C�
A�Q�C6�q                                    Bxo��\  
�          @�ff��(�?��\@@  A���C(B���(�=��
@P��A��C3E                                    Bxo��  �          @�{���
?Ǯ@3�
A��C%�q���
>��@L(�AۅC0#�                                    Bxo���  T          @�ff���R@,(�@<(�A�p�C�\���R?�@n{B{C&�                                    Bxo�N  �          @߮��@*�H@$z�A��\C� ��?��@W�A�\C%k�                                    Bxo��  T          @�\)����@0  @�A��
C������?��@?\)Aʣ�C#��                                    Bxo�"�  "          @߮��
=@p�@L(�A�  C�f��
=?c�
@q�B�C+.                                    Bxo�1@  
Z          @�Q����@��@`��A��\CB����?8Q�@���BQ�C,Ǯ                                    Bxo�?�  "          @�  ���
@\)@Tz�A���C����
?�{@�  B�C(��                                    Bxo�N�  
�          @�\)���H@%�@Q�A�(�C0����H?�(�@\)B��C'��                                    Bxo�]2  �          @�
=��  @0  @P��A�33Ch���  ?��@�G�B�C%޸                                    Bxo�k�  
�          @�\)����@C�
@L(�A�\)C������?�Q�@��\B  C"�
                                    Bxo�z~  
P          @�{��  @G
=@6ffA\C����  ?�{@q�B\)C!G�                                    Bxoى$  	�          @�p�����@qG�@
=A�{Cu�����@*=q@c�
A���C8R                                    Bxoٗ�  �          @�z����@�{?�z�A���C	޸���@\��@XQ�A��C޸                                    Bxo٦p  "          @���p�@��
@�A���C
ff��p�@Tz�@c33A��C�                                    Bxoٵ  "          @���Q�@��@z�A�\)C����Q�@@  @h��A�33C�                                    Bxo�ü  T          @�p���{@�z�@#33A�G�C��{@Z�H@�G�B=qCn                                    Bxo��b  T          @ڏ\�p��@�{?���AYB�B��p��@�Q�@\(�A��C�\                                    Bxo��  T          @�(��X��@���?��A4(�B��X��@�{@Tz�A��
B���                                    Bxo��  T          @��
�K�@��
?˅AV=qB���K�@�p�@e�A�G�B���                                    Bxo��T  "          @���@��@���?�Q�A@��B����@��@��
@_\)A�p�B�Ǯ                                    Bxo��  �          @�(��Fff@�\)?�=qA{B����Fff@��R@HQ�A��HB�=                                    Bxo��  �          @��H�P  @��H?�A@(�B�.�P  @��R@Y��A��B�z�                                    Bxo�*F  �          @ڏ\�9��@��R?�ffAR�HB���9��@���@c�
A�=qB�                                    Bxo�8�  
�          @���Q�@���?��A<z�B���Q�@�p�@UA뙚B�W
                                    Bxo�G�  �          @�33�z�H@���?:�H@��B�ff�z�H@�{@(��A�{C n                                    Bxo�V8  "          @ۅ���@�ff>���@W
=B�W
���@��R@�\A�\)CG�                                    Bxo�d�  T          @ۅ��=q@���������C:���=q@���?�{AZ=qC��                                    Bxo�s�  "          @�(����@������Z=qC����@���?��HAD  C�                                    Bxoڂ*  �          @�33��ff@�ff�k�����C����ff@�ff?���AXQ�C�                                    Bxoڐ�  
�          @ٙ����\@��R>��@���C����\@�  @A�{C
�H                                    Bxoڟv  
�          @ۅ���\@�Q�?8Q�@���C
aH���\@~�R@  A��C�                                    Bxoڮ  �          @������@��
?:�H@�C������@��@Q�A�z�C	.                                    Bxoڼ�  
�          @��H��(�@�
=?�@��C T{��(�@�
=@�
A��C33                                    Bxo��h  
�          @�=q�K�@�
=�.{��  B�(��K�@��H?�
=AB�\B�=q                                    Bxo��  
�          @���_\)@�녿����RB����_\)@�(�?�G�AN{B�                                    Bxo��  "          @�G��\(�@�녿#�
���B�G��\(�@�p�?�z�A?�B�                                     Bxo��Z  T          @ڏ\�r�\@�33�ٙ��hz�B��R�r�\@��>�  @
=B�.                                    Bxo�   
�          @�=q�o\)@�=q��(��K
=B�=q�o\)@�  >�@{�B�u�                                    Bxo��  
�          @׮�@��@����
=�h��B�aH�@��@�
=>Ǯ@W
=B�=                                    Bxo�#L  "          @�ff�b�\@�������\B�p��b�\@�(�������B�Q�                                    Bxo�1�  
�          @׮�c33@�=q�
=���B�aH�c33@�
=�#�
���B�                                    Bxo�@�  T          @ָR�tz�@�z���R��{B�Q��tz�@�Q��G��xQ�B��=                                    Bxo�O>  �          @�{��  @�  ����?\)CxR��  @�{>��
@.{C\)                                    Bxo�]�  
�          @�����@�z������
=C\)����@�\)���
�.{C xR                                    Bxo�l�  
�          @����r�\@��H��z���  B�Q��r�\@�{��\)�\)B��q                                    Bxo�{0  T          @��
�Mp�@�\)�Q����B�
=�Mp�@�\)������B�                                     Bxoۉ�  "          @�(��@��@���-p���\)B���@��@�33�G���\)B�aH                                    Bxoۘ|  
�          @���� ��@����:=q��Q�B�Ǯ� ��@�=q�h������B��H                                    Bxoۧ"  T          @�p��z�@�=q�Mp��癚B�\�z�@�33��p��*�\Bݣ�                                    Bxo۵�  T          @�
=��
=?���>�z�@G�C�)��
=?�\)?c�
A�C!�3                                    Bxo��n  
(          @�{��p���  @AG�A�=qC6� ��p���  @0��A˙�C@n                                    Bxo��  ,          @�����(���@2�\A���C5=q��(����@%�A�
=C>.                                    Bxo��  
�          @�Q���>�{?�(�A�
=C0�)����{?�(�A�
=C7&f                                    Bxo��`  
Z          @�
=�Ǯ?!G�?�AK�
C.B��Ǯ=�?�ffA^�\C2�                                    Bxo��  
�          @Ϯ���
?fff@*�HA��C+=q���
�#�
@4z�A�G�C4c�                                    Bxo��  �          @�G����H?��?�\)A��C'�
���H?(�@��A�{C.=q                                    Bxo�R  
(          @�G��\?�z�?�z�A��
C&���\?333@G�A���C-p�                                    Bxo�*�  �          @�G��Å?���?���A�
=C(޸�Å>��H@\)A�ffC/p�                                    Bxo�9�  �          @љ���G�?���@	��A�G�C'����G�?�@p�A��HC.�R                                    Bxo�HD  "          @�=q��{?�33?���A��HC)����{>�@A�33C/��                                    Bxo�V�  �          @�G���ff@]p�?=p�@�ffC{��ff@AG�?�A��\C��                                    Bxo�e�  
�          @�����\)@���?h��A�C(���\)@p��@�A�ffC�q                                    Bxo�t6  J          @˅��@w
=?uA  C���@U�@�A�
=C��                                    Bxo܂�  
�          @�����\@�?�A���CǮ���\?�ff@�RA��C$z�                                    Bxoܑ�  T          @�33��=q@#�
?�\)Al��CQ���=q?��@�A��C!J=                                    Bxoܠ(  "          @�����@L(�?��A��CaH���@*�H@�\A��C�\                                    Bxoܮ�  	�          @Ӆ��@l��?\)@��C��@S�
?޸RAt(�C�H                                    Bxoܽt  �          @�(���G�@dz�?�@���C+���G�@Mp�?�z�Ahz�C�                                    Bxo��  
�          @�p����R@h��?��A�CW
���R@Fff@
=qA�  Ch�                                    Bxo���  �          @������@q�?0��@�ffC����@Vff?��A���C5�                                    Bxo��f  T          @�z���  @~{?333@���C���  @a�?���A���C�                                    Bxo��  �          @�{��(�@z=q?�R@��\C��(�@`  ?���A��RC�q                                    Bxo��  �          @�Q�����@w
=>�33@=p�C\����@b�\?�=qAXQ�CY�                                    Bxo�X  T          @������R@e>�(�@i��C�\���R@P��?���AVffCB�                                    Bxo�#�  �          @�����H@Y��?!G�@���C�\���H@@��?�Q�AfffC��                                    Bxo�2�  �          @�Q�����@U?uA33C  ����@6ff?�p�A��HC�                                     Bxo�AJ  
�          @�Q��R�\@�\)�����HB���R�\@��H�#�
��33B�Ǯ                                    Bxo�O�  
�          @ڏ\�g
=@���z�����B�aH�g
=@��׾��H����B�                                      Bxo�^�  �          @�33�^{@��
�����z�B��R�^{@���\)��p�B�G�                                    Bxo�m<  
�          @ڏ\�h��@�녿�=q�U��B�\)�h��@���>�\)@ffB�ff                                    Bxo�{�  T          @�33�\)@�Q��(��i��B�u��\)@���=#�
>�p�B���                                    Bxo݊�  
�          @������\@�����R�H(�C���\@���#�
��G�C	z�                                    Bxoݙ.  
�          @���8Q�@ƸR�.{����B����8Q�@�{?�At��B�q                                    Bxoݧ�  
�          @ۅ�<��@�33?Q�@���B�=�<��@���@-p�A�33B��                                    Bxoݶz  ,          @��
�S33@�
=?.{@�p�B��)�S33@�ff@!�A�G�B�aH                                    Bxo��   
O          @��H�}p�@�Q�>�{@9��B�aH�}p�@��
@z�A���B�ff                                    Bxo���  �          @�(���
=@��>Ǯ@R�\C 33��
=@�
=@�
A�z�CY�                                    Bxo��l  �          @��H���@�
=?+�@�(�C����@�  @�A�Q�C:�                                    Bxo��  �          @�p���(�@��
<��
>\)C���(�@��
?��AO
=C�)                                    Bxo���  
�          @�z����R@�33�:�H��33C����R@��?.{@�C
=                                    Bxo�^  �          @�p�����@�{������C�����@���?fff@�\)C	:�                                    Bxo�  
�          @�ff����@�{��33�9��C33����@��=u?�\C	޸                                    Bxo�+�  
�          @��
���
@�{�����C����
@�(�?aG�@��CxR                                    Bxo�:P  �          @ۅ����@�33���
�.{C������@���?��
A+�
C��                                    Bxo�H�  "          @�33����@�{?aG�@��
CG�����@�@�\A��\C�                                    Bxo�W�  �          @��
����@���?�  AJ{C	B�����@u@5�A�33C�f                                    Bxo�fB  �          @�(����@��R?޸RAk
=C	���@l��@B�\A�{C�q                                    Bxo�t�  �          @��
��Q�@�z�?���ATz�C
Ǯ��Q�@l(�@6ffA���C��                                    Bxoރ�  �          @�z�����@vff��(��q�C������@s33?G�@ٙ�C\)                                    Bxoޒ4  �          @��
����@��w
=��HCu�����@J�H�A��ѮC�                                    Bxoޠ�  "          @ڏ\���@	���hQ�� ��C&f���@J=q�2�\��p�CxR                                    Bxoޯ�  
�          @��H��\)@ ���]p����\CL���\)@]p��!G���\)C��                                    Bxo޾&  
�          @����(�@AG��5��p�C����(�@o\)���
�rffC33                                    Bxo���  �          @�G�����@&ff��G��s�
C�����@'
=>\@R�\C�)                                    Bxo��r  �          @�Q�����@S�
@Mp�A�
=CB�����@�@��B
=C��                                    Bxo��  �          @�
=���
=q?���AA�C8�����p��?�33A z�C<J=                                    Bxo���  "          @�33��=q@	��@)��A��
C���=q?��R@K�A��HC&G�                                    Bxo�d  �          @׮��G�>W
=@33A��RC2{��G��   @��A�p�C8�                                    Bxo�
  �          @׮���\@!�@G�A���C�\���\?�(�@o\)BQ�C$��                                    Bxo�$�  
�          @�G���=q@,(�@8Q�A�Q�C�{��=q?�Q�@c�
Bp�C �f                                    Bxo�3V  �          @�(���ff@����9������Ch���ff@�
=��ff�Q��C�f                                    Bxo�A�  
�          @�Q���=q@vff�C�
�؏\C����=q@��H��  �q�C޸                                    Bxo�P�  �          @�Q���
=@i���
=����C���
=@��
�k���33C��                                    Bxo�_H  
�          @�\)���R@Z�H�Vff���C����R@����
=q��ffC	��                                    Bxo�m�  "          @���=q@^�R�/\)��C.��=q@�z�����]CY�                                    Bxo�|�  �          @�{���
@�33��H����C
h����
@�zῊ=q���C�f                                    Bxoߋ:  "          @���  @u�)����p�C���  @��R�����?
=C��                                    Bxoߙ�  
�          @�z���z�@Y���:=q��z�C=q��z�@����\�w33C�R                                    Bxoߨ�  
Z          @љ����H@E�H����Cu����H@x������(�CB�                                    Bxo߷,  �          @����  @33�:�H���
C�
��  @4z������C��                                    Bxo���  T          @љ�����@%��R�\��(�C������@\(��Q����HC�{                                    Bxo��x  �          @�  ���R?����.�R�ǅC&�R���R@ff�p���p�Cٚ                                    Bxo��  �          @�\)��\)?c�
�����C+�{��\)?\����(�C%�R                                    Bxo���  "          @�(���33@�\�������HC����33@!녿�G��9p�C��                                    Bxo� j  �          @�
=���H?�ff��G��Y��C'�q���H?�
=����  C$�\                                    Bxo�  �          @�
=��
=>��ÿ���]C0�3��
=?E���\)�Ep�C,�3                                    Bxo��  �          @�����ff�c�
��G��YG�C<+���ff��녿��H�u�C7�                                     Bxo�,\  �          @�����?=p���z��t(�C,�R����?�����{�Hz�C(�                                    Bxo�;  
�          @���
=?�\)�p���
=C!�3��
=@��Ǯ�aC                                      Bxo�I�  	�          @��H����?�ff��33���RC#O\����@녿���5p�C=q                                    Bxo�XN  T          @��H���H@?\)��G��4  C�)���H@N�R�����<��C                                      Bxo�f�  
Z          @�33��p�@(��?J=q@�{C���p�@33?��
AX  CǮ                                    Bxo�u�  "          @�
=��  @8��@333AՅC�q��  ?���@`��B	�C�R                                    Bxo��@  
�          @�����@3�
?�(�AV�\C�����@��@{A�=qC�)                                    Bxo���  �          @�z���G�@a�?�(�A�
=C+���G�@8Q�@*�HA�
=C��                                    Bxoࡌ  �          @�p�����@?\)?�{A���C�{����@�@)��A��HC��                                    Bxo�2  �          @�\)��{@#33>��
@7
=C�R��{@ff?�ffA�Cc�                                    Bxo��  �          @��
���@=p��#�
�\C�R���@6ff?O\)@�C��                                    Bxo��~  
�          @�  ��=q?��@QG�A�C)����=q=L��@[�A�{C3}q                                    Bxo��$  "          @����  ?J=q@U�A�Q�C,����  �L��@Z=qA���C5�                                    Bxo���  �          @�G����H?�R@p  B�RC-�3���H��@qG�Bz�C8��                                    Bxo��p  "          @أ���?n{@c�
A��C*� ����@j�HB�C5W
                                    Bxo�  
Z          @ָR��33?�\@/\)A���C/33��33��z�@1G�A�G�C6                                    Bxo��  �          @�\)��(�?��R@6ffA�33C%Ǯ��(�?(�@I��A��C.
=                                    Bxo�%b  �          @�Q���33?˅@>{AЏ\C$Ǯ��33?.{@S33A��C-^�                                    Bxo�4  �          @�  ��p�?�{@9��A�z�C'
��p�>��@J�HA���C/n                                    Bxo�B�  �          @��H��{?}p�@N{A���C*����{=#�
@W�A�ffC3�f                                    Bxo�QT  �          @����z�?�ff@(�A�z�C%�{��z�?J=q@1�A�\)C,�H                                    Bxo�_�  
�          @ڏ\�Å?��
@(�A���C#���Å?��\@5A�33C*��                                    Bxo�n�  T          @�33���H?�p�@0  A�p�C&Y����H?#�
@C�
A�{C.                                      Bxo�}F  �          @ۅ�ə�?�ff@��A���C(aH�ə�?�@*=qA�(�C.ٚ                                    Bxo��  
�          @�G���ff?�33?�{A[
=C'����ff?c�
?�Q�A���C,!H                                    Bxoᚒ  
Z          @����љ�?��?�z�A@��C+��љ�?��?�33Aa�C.��                                    Bxo�8  
�          @أ��У�?��?�Q�ADz�C*� �У�?#�
?�Q�Ag33C.aH                                    Bxo��  �          @ڏ\��p�?}p�?�  A�C+����p�?+�?��RA(  C.E                                    Bxo�Ƅ  
�          @�=q���
?u@��A��\C+k����
>��R@Q�A��C1=q                                    Bxo��*  
�          @�33���?fff@   A���C+�����>#�
@)��A��\C2��                                    Bxo���  �          @�(���Q�?Y��@+�A��
C,B���Q�=�\)@3�
A�\)C3aH                                    Bxo��v  �          @ۅ����?aG�@EA�=qC+�������
@Mp�A�p�C40�                                    Bxo�  
�          @��
���?���@A�AѮC)�����>8Q�@Mp�A�
=C2B�                                    Bxo��  "          @�33�Å?��\@9��A���C*�=�Å>.{@Dz�A�\)C2ff                                    Bxo�h  �          @�����\)�333?��HAt��C:aH��\)��33?���AN�\C>k�                                    Bxo�-  �          @׮��(����?z�HACDn��(���>�
=@c33CF�                                    Bxo�;�  T          @�����׿��
?ٙ�As33CDxR������?��A"=qCG�                                    Bxo�JZ  T          @Ӆ��{��@O\)A�CFp���{�(��@'
=A���CM��                                    Bxo�Y   �          @��H��(���
=@p  B{CF&f��(��)��@H��A�(�COT{                                    Bxo�g�  
�          @�(���ff��(�@eBG�CH���ff�8Q�@9��A�(�CQ                                    Bxo�vL  
�          @љ���z���H@1G�A�\)CD�{��z��=q@(�A�\)CK{                                    Bxo��  
�          @�\)��\)��p�@4z�A�G�CE� ��\)�(�@\)A��CK��                                    Bxoⓘ            @�\)���Ϳ��?�Q�A�G�C9�����Ϳ�{?ٙ�As\)C>33                                    Bxo�>  "          @������>��@�
A��
C/�����ͽ�G�@
=A��C5�                                    Bxo��  �          @׮��{�J=q@>{A�
=C;����{��=q@(��A�{CB޸                                    Bxo⿊  T          @ָR����h��@C33AٮC<��������H@+�A��CD@                                     Bxo��0  T          @�p����
=u@e�B��C3k����
�k�@]p�A�(�C=Q�                                    Bxo���  �          @�ff��(��\(�@AG�A��C<O\��(���33@+�A�\)CC��                                    Bxo��|  
�          @׮��p�?B�\@A�A�\)C,����p���Q�@HQ�A�z�C4�                                    Bxo��"  �          @�  ���
��
=@eB(�CEk����
�&ff@?\)A�p�CM��                                    Bxo��  
�          @�  ��  ��  @VffA�\C=�
��  ��\)@<��A�  CF                                      Bxo�n  
�          @�G���{>8Q�@�A��C2^���{�Ǯ@33A��C7p�                                    Bxo�&  	�          @�G����
��@��A�=qC9
���
����?�z�A��C>�                                    Bxo�4�  	�          @�33��  �+�@(��A�  C:)��  ��{@
=A��
C@@                                     Bxo�C`  "          @�G��\��@=p�A��HC8޸�\��ff@-p�A�Q�C@
=                                    Bxo�R  
�          @�z����Ϳ.{@]p�A��
C:�)���Ϳ˅@I��A�{CC�                                    Bxo�`�  "          @�z���\)�(�@U�A�C9޸��\)���R@B�\A�CA��                                    Bxo�oR  "          @��
��ff���@7
=A�G�C9����ff����@%A�C@L�                                    Bxo�}�  
�          @ۅ�θR�u@{A��C6(��θR�Tz�@�A�G�C;T{                                    Bxo㌞  �          @�=q��  �z�H?�  AnffC<�)��  ��?�A@��C@L�                                    Bxo�D  �          @�����ff����@9��A�G�C@���ff��@�A���CG:�                                    Bxo��  
�          @�G���Q쿞�R@��A�  C?:���Q��ff?���A|��CD\                                    Bxo㸐  
�          @ڏ\�����@  A��RCGO\����)��?�{AZ=qCK��                                    Bxo��6  
�          @�=q����=q@�A�{CJh�����=p�?��A`Q�CN�\                                    Bxo���  "          @ٙ���=q��(�@2�\A�33CB33��=q�
=q@�\A�(�CHaH                                    Bxo��  
�          @�p����\�p��@W�A�(�C=�{���\��@@  A؏\CE�                                    Bxo��(  
Z          @ָR���׿�  @��B)G�CB���������@�  BQ�CN�f                                    Bxo��  �          @�p����׿�Q�@�Q�B��CF�������,��@Z�HA��
CPB�                                    Bxo�t  T          @�ff��{�h��@qG�B	Q�C=u���{��\)@X��A�{CG                                      Bxo�  
�          @�
=���
����@~�RBz�CC!H���
���@^{A�ffCM                                      Bxo�-�  �          @�\)��\)�W
=@^{A��
C6���\)��\)@R�\A�C?�                                    Bxo�<f  �          @�p���녿z�@N�RA�C9���녿�z�@>{A���CA�f                                    Bxo�K  T          @������R��p�@p  B	�C7�����R����@aG�B (�CA�)                                    Bxo�Y�  �          @�����
�Ǯ@��A���CBٚ���
�ff?�\A|��CG�f                                    Bxo�hX  T          @�{���H�ff?���A&�HCI\���H�%�?
=q@�{CJ�3                                    Bxo�v�  
�          @�����  ����@��A�{C@:���  ��\)?��HAj�HCD�f                                    Bxo䅤  �          @ڏ\��=q��G�@{A���C={��=q�Ǯ?�\)A�z�CA�{                                    Bxo�J  
�          @ٙ�������ff@Q�A�
=CD�=������\?�=qA[�
CH�                                     Bxo��  
�          @�=q��(��@!G�A�\)CI����(��:�H?�Aw33CNaH                                    Bxo䱖  
�          @��
��Q쿰��@{A�ffC@u���Q��p�@G�A��RCE�\                                    Bxo��<  �          @��
�ȣ׿�@(�A�33CB��ȣ��(�?�
=Ab�RCG:�                                    Bxo���  �          @�33��{��33@FffA�\)CAG���{�
=q@'�A�z�CH                                      Bxo�݈  
�          @����Q쿵@�RA�
=C@����Q��G�@G�A���CE��                                    Bxo��.  T          @�z�����n{@8��A��HC<�)�����33@"�\A�  CC�                                    Bxo���  �          @�����=q�B�\@)��A���C:�)��=q��@
=A�(�C@��                                    Bxo�	z  H          @��
���
�@  @@  A�C:�3���
��  @-p�A��CAǮ                                    Bxo�   �          @ٙ���\)���R@%A��RC6޸��\)�xQ�@�HA�=qC<�f                                    Bxo�&�  
�          @�ff����n{@\)A�Q�C<=q������@
=qA�p�CA��                                    Bxo�5l  
�          @�����z�Tz�@>�RA�\)C;����z����@*=qA�(�CBaH                                    Bxo�D  �          @޸R��Q��(�@��A�G�C7Ǯ��Q쿃�
@��A��
C=�                                    Bxo�R�  
�          @�{��p���G�@A�Q�C<���p���=q@   A�z�CA��                                    Bxo�a^  
�          @���ff��\)@\)A��HC=޸��ff��z�?�\)A{
=CBp�                                    Bxo�p  �          @�G���z��G�@/\)A��
CBaH��z��
�H@��A�{CH+�                                    Bxo�~�  �          @���  �ٙ�@e�B�RCE�3��  �$z�@AG�A�ffCN�                                    Bxo�P  "          @أ���{����@Q�A�G�CBs3��{�?�33Ab{CF�f                                    Bxo��  �          @ڏ\��{��=q@��A�G�C@���{��z�@�A���CE+�                                   Bxo媜  �          @��ə��޸R@�A��CCs3�ə��  ?�z�A_33CG��                                   Bxo�B  
(          @��
����p�@Q�A��CAxR����\?�A�33CFB�                                    Bxo���  T          @��
����G�?c�
@�
=CG�������H>�\)@�CH��                                    Bxo�֎  
�          @�(�������@33A�
=C@=q�����\)?У�A[33CDB�                                    Bxo��4  �          @�����Ϳ\(�@(��A��\C;�����Ϳ�G�@A�  CA��                                    Bxo���  "          @���ȣ��(�>�(�@l(�CIG��ȣ��{�L�Ϳ޸RCI�                                    Bxo��  
�          @أ��Ǯ� ��?8Q�@�=qCI�f�Ǯ�'
==L��>�(�CJ��                                    Bxo�&  �          @�ff��  �\(�?�ffAW33CT
=��  �n{?�R@��CV�                                    Bxo��  
�          @��
���H��Q�?��@��C@�����H��ff>W
=?�CA�3                                    Bxo�.r  
�          @����녿��>�33@@��C=8R��녿�\)=�\)?z�C=��                                    Bxo�=  T          @Ӆ��  �c�
��G��p��C;��  �Tzᾨ���7�C;E                                    Bxo�K�  	�          @ָR��33�fff�u�33C;�R��33�O\)���|(�C:��                                    Bxo�Zd  
�          @�p���녿녿:�H�ə�C8�R��녾�p��Y����C733                                    Bxo�i
  
(          @�p���33=�Q�Tz���ffC3.��33>����G���Q�C1^�                                    Bxo�w�  �          @�����  ?�  �k��   C+G���  ?����#�
����C)��                                    Bxo�V  "          @�(���?���p����C(����?�  ����\)C&ٚ                                    Bxo��  
�          @�����G�>��u���C/�q��G�?0�׿O\)���C.�                                    Bxo棢  T          @�33��z�?�{��G��33C#!H��z�@�
��\��\)C!z�                                    Bxo�H  �          @�(����H@�J=q��C�R���H@{�8Q���
C��                                    Bxo���  
�          @�p�����?s33������C+ff����?��׿\�T��C'��                                    Bxo�ϔ  �          @�{���R?L���������C,\)���R?�����R��{C'J=                                    Bxo��:  T          @�  ��  @(��A��ՙ�C)��  @G��z�����Cp�                                    Bxo���  T          @�\)��p�@���C�
���CxR��p�@5����
=CW
                                    Bxo���  T          @ٙ���G�@:�H�^{��=qC���G�@l���'���{C��                                    Bxo�
,  "          @ٙ�@"�\@���j�H���B}z�@"�\@�=q����B��H                                    Bxo��  
�          @ٙ�@
=@���j�H��B��H@
=@�������=qB��=                                    Bxo�'x  "          @��H@
�H@����p����B�Q�@
�H@�����
��z�B��\                                    Bxo�6  
�          @��H@$z�@�z��S�
��(�B�  @$z�@�G������|Q�B��                                    Bxo�D�  "          @�G�@L(�@�
=�8����ffBjG�@L(�@��ÿ��R�K�
Bt\)                                    Bxo�Sj  T          @أ�@c33@��\��
���HBa\)@c33@��(������Bg�                                    Bxo�b  "          @�Q�@I��@����j�H��{Bgp�@I��@����33���Buz�                                    Bxo�p�  
�          @�z�@K�@�z��i�����Bc�
@K�@�����
���RBrp�                                    Bxo�\  T          @�(�?�z�@�(����� �\B�B�?�z�@�=q�C�
��{B��R                                    Bxo�  
(          @�p�?xQ�@�33�z�H�
�B�k�?xQ�@���������HB��
                                    Bxo眨  	�          @��?@  @��\�c�
��p�B�\)?@  @�G����R����B��                                    Bxo�N  �          @�ff>�Q�@�(��g����RB�  >�Q�@�33��\��p�B�                                    Bxo��  
�          @��
��(�@����N{���B�=q��(�@�z��\)�[
=B�z�                                    Bxo�Ț  T          @���>�ff@�p��R�\��  B�=q>�ff@љ����H�i�B�{                                    Bxo��@  �          @�{?��\@�������"  B���?��\@Å�C�
�ҏ\B�
=                                    Bxo���  �          @�\)?!G�@\�Tz���(�B��?!G�@ָR�ٙ��c
=B�=q                                    Bxo��  �          @�Q쾞�R@����l��� (�B�\���R@�(�����\)B�aH                                    Bxo�2  T          @��
���@��R����� z�B�����@�z��>{��=qB���                                    Bxo��  T          @�=q�\)@�
=��=q�p�B��\)@�(��8�����B��\                                    Bxo� ~  �          @�p��z�@�����Q��U�B�.�z�@�G����\��B�k�                                    Bxo�/$  T          @ٙ�?��@����r�\�)=qB��R?��@��\�*�H����B���                                    Bxo�=�  
�          @��H@��@�ff��(����\B ff@��@�녿^�R��=qB)�                                    Bxo�Lp  T          @ٙ�@QG�@��H�7��י�BZ=q@QG�@������pQ�Bf��                                    Bxo�[  T          @ٙ�����@�=q��\)�%��B�Q����@����E�����B���                                    Bxo�i�  .          @��þ�\)@��\�����K�B��þ�\)@��R�xQ���RB�=q                                    Bxo�xb  H          @�\)��R@�  ����({B��)��R@���@  ��ffB�z�                                    Bxo�  �          @ָR��p�@�Q��e����B����p�@�
=�
=��33B�Q�                                    Bxo蕮  �          @�G�����@�p��e����B�p�����@�z��Q���
=B��                                    Bxo�T  �          @У׿=p�@���Y�����B�녿=p�@�\)��p���B�#�                                    Bxo��  �          @�녿+�@E����
�s  B̳3�+�@��
��{�:BŅ                                    Bxo���  T          @�ff�^�R@�����p��"�\BȮ�^�R@���7
=��ffBŅ                                    Bxo��F  T          @Ӆ�#�
@�\)�/\)�ģ�B�#׼#�
@�
=��
=�$��B��                                    Bxo���  �          @�G��z�@���������B���z�@��O\)��(�B�L�                                    Bxo��  T          @�  ��33@�=q�Q���B�33��33@ƸR����B�u�                                    Bxo��8  
�          @�\)��z�@����`����
B׽q��z�@���Q���z�BӨ�                                    Bxo�
�  T          @�G��!G�@�{�����(�B�33�!G�@����4z���
=B��                                    Bxo��  �          @�����@����e����B���@������B�z�                                    Bxo�(*  �          @�
=���R@�(��a���\B��ÿ��R@�=q����ffB��                                    Bxo�6�  �          @�{���R@�Q��e�����B۽q���R@�
=�����B�z�                                    Bxo�Ev  
(          @����;�@���
=q��{B�33�;�@�{�Y����=qB�3                                    Bxo�T  
(          @����;�@7���ff�N�C�=�;�@�����\�#z�B��\                                    Bxo�b�  �          @�{�7
=?�
=���R�op�C��7
=@A����
�Kz�CJ=                                    Bxo�qh  �          @�(��:�H@7
=���
� B�z�:�H@�����\)�IQ�B�B�                                    Bxo�  �          @�33��
=?�p���(�8RC�3��
=@4z����
�}�B�=q                                    Bxo鎴  �          @�
=�   @Tz���ff�Zp�B���   @�����
=�'��B�\)                                    Bxo�Z  �          @��H�>{@E������K��C���>{@���������B��                                    Bxo�   
�          @��?\)?�����H�y�C��?\)@#33��z��\p�C}q                                    Bxo麦  "          @޸R�K�=#�
�����z��C3W
�K�?����ff�m(�C)                                    Bxo��L  "          @޸R�L�ͽ�\)��33�yG�C5Q��L��?����m��C�q                                    Bxo���  
�          @��@����\����d�CV.�@�׿����\)�{\)C?0�                                    Bxo��  �          @�(��S�
��\)��
=�t  C5@ �S�
?�������i
=C(�                                    Bxo��>  
�          @��ÿ�(�����
={CEE��(�?��\����B�C@                                     Bxo��  T          @�
=��녿�ff�\�C_�ÿ�녾�����H�=C;��                                    Bxo��  
�          @�G��333��z���  �y�\CN���333=������
=C1�)                                    Bxo�!0  �          @��H�N{���
��(��m�HCI���N{>L����Q��w{C0u�                                    Bxo�/�  
�          @�(��J=q���H��Q��kp�CL���J=q�#�
��{�x
=C4�                                    Bxo�>|  T          @�33����?��������RC%@ ����@��c�
����C&f                                    Bxo�M"  T          @������
?�
=�|(��p�C'�����
@
=�a����C��                                    Bxo�[�  T          @ۅ���H��������I(�C8ff���H?u��p��E�C'��                                    Bxo�jn  
�          @�\)�s33���
��\)�\�C8�)�s33?��
��z��W\)C$�H                                    Bxo�y  "          @�G��b�\��������b33CF���b�\>k������i��C0Y�                                    Bxoꇺ  �          @��H�i�������ff�S��COn�i���   ��  �ez�C;�{                                    Bxo�`  �          @��H�S�
�z���=q�[�RCS���S�
�#�
�����p�C>��                                    Bxo�  "          @ڏ\�3�
�����
�o��CVn�3�
��(�����C<�                                    Bxo곬  
(          @ۅ�c�
�
=q�أ�¢�=CS  �c�
?����{��C \                                    Bxo��R  �          @��
�\)�B�\��33�C8\)�\)?�\)�ƸR��C#�                                    Bxo���  T          @����hQ��ff�����g�C;#��hQ�?z�H����c{C$�f                                    Bxo�ߞ  �          @�(��G
=>\)��
=�y�C1���G
=?˅��  �k  C޸                                    Bxo��D  "          @ڏ\�
�H@�H���R�uz�C��
�H@s33���G33B�z�                                    Bxo���  �          @ڏ\�0  @+���z��_
=C�=�0  @~{����3z�B�aH                                    Bxo��  "          @�33�A�@I������F�
C���A�@�=q��\)��HB�#�                                    Bxo�6  �          @�ff�Z=q@Y����33�.�RC��Z=q@�p��hQ��  B�B�                                    Bxo�(�  
�          @�Q��O\)@S�
��Q��B�\C^��O\)@����G����B��                                    Bxo�7�  
Z          @ᙚ�HQ�@`  ��p��9Q�C�q�HQ�@�33�z=q�z�B�ff                                    Bxo�F(  "          @�����@��\�r�\�
=C	0�����@�(��*�H���C�                                    Bxo�T�  T          @��
�x��@k������#C�
�x��@��R�fff��ffC�{                                    Bxo�ct  
�          @�G��\��@j�H��(��0G�C5��\��@�Q��u��p�B��                                    Bxo�r  S          @�33�N�R@fff��ff�<p�C��N�R@�Q�������B�L�                                    Bxo��  
�          @��
�^�R@aG�����7��C�R�^�R@�����H�z�B��=                                    Bxo�f  
(          @����^�R@U���Q��>=qC@ �^�R@�����G���
B�.                                    Bxo�  T          @�=q�G�@����Q��6\)B��G�@�ff�r�\��HB�.                                    Bxo묲  
�          @�z��{@��������@�\B����{@�ff��(���B���                                    Bxo�X  "          @�z�<#�
��33�}p���C�"�<#�
�Ǯ��©Q�C�z�                                    Bxo���  
�          @�
=?��
�����Q����C���?��
���H��\)�D�\C�1�                                    Bxo�ؤ  "          @���?c�
���\�����BC��?c�
�>{��{�y�HC�N                                    Bxo��J  
�          @�G�?�33��p������-z�C��
?�33�fff���\�dG�C��                                     Bxo���  T          @�\)?�Q����
������RC�'�?�Q��x�������Q�
C�Ǯ                                    Bxo��  �          @�33@dz��@  ���\�1��C���@dz������QQ�C���                                    Bxo�<  T          @ָR@�=q�P  �����
C���@�=q�#33�I����ffC���                                    Bxo�!�  �          @�(���
=���
��p��4C�Ff��
=�I����33�m=qC�:�                                    Bxo�0�  T          @�G����H�"�\���R�&=qCRaH���H��\)��{�=�\CE�=                                    Bxo�?.  �          @Ӆ@U��p���{�L��C�Ф@U?+������J\)A4(�                                    Bxo�M�  �          @��H@�\)�5�~{��C�S3@�\)>�z�������H@@��                                    Bxo�\z  �          @��@��ÿ   ���R�7��C���@���?:�H���6G�A33                                    Bxo�k   �          @��H@�Q�
=���\�@z�C�B�@�Q�?+���=q�@  A�                                    Bxo�y�  T          @��H@�
=�
=��=q�M(�C��@�
=?=p������L33A{                                    Bxo�l  
�          @��@�  ���R��G���
C�h�@�  �Ǯ�����p�C���                                    Bxo�  �          @�\@���33��{�&Q�C�t{@��0����Q��4��C�                                    Bxo쥸  
�          @�Q�@��
�����=q�-�C��
@��
�c�
��ff�?\)C��3                                    Bxo�^  "          @�  @y���Q����\�HC��\@y���@  ��{�\\)C���                                    Bxo��  �          @ۅ�8Q�>����{«ffBݞ��8Q�@ff�˅Q�B��3                                    Bxo�Ѫ  "          @�(��O\)����=q£��CQ�
�O\)?���\).B�u�                                    Bxo��P  "          @��7�@����\)�p�B��H�7�@��
�,(����HB�\                                    Bxo���  �          @�=q?���\��  ��C��f?��u��  G�C�AH                                    Bxo���  
�          @���8Q�@~{�����0�HB����8Q�@����i��� �RB�\                                    Bxo�B  �          @�=q�>�R@��H��Q����B�  �>�R@�  �@  ���B��f                                    Bxo��  "          @����N{@�\)�h������B�(��N{@�
=��R��B��                                    Bxo�)�  "          @�{�L��@�{�>{��B�Ǯ�L��@�  ��(��=B�Q�                                    Bxo�84  �          @��H��H@��\(�����B�.��H@�33��
=�}p�BݸR                                    Bxo�F�  T          @ᙚ�-p�@���c�
���HB�\�-p�@\����\)B�
=                                    Bxo�U�  
Z          @�=q�b�\@���=�Q�?8Q�B��
�b�\@�?�AS�B�\                                    Bxo�d&  �          @�33�k�@��H��\�|(�B�33�k�@ȣ�?�=qA33B�Ǯ                                    Bxo�r�  
�          @�\)����@�ff��p����C(�����@��
��Q�.{C#�                                    Bxo�r  �          @���z�@�ff���tz�CaH��z�@���?c�
@�C�f                                    Bxo�  �          @����ff@�
=����ffB�Q���ff@�=q?���A*ffB��q                                    Bxoힾ  T          @���p  @�\)�+���\)B�{�p  @�ff?fff@��HB�Q�                                    Bxo��d  "          @�=q�P��@�\)�^�R��33B�\)�P��@Ϯ?B�\@�  B�B�                                    Bxo��
  
�          @陚�'
=@�\)����G�B�W
�'
=@���?&ff@��HB�
=                                    Bxo�ʰ  T          @陚�l(�@Ǯ�Ǯ�C�
B�#��l(�@�z�?�
=A�B��                                    Bxo��V  �          @���c33@ȣ׿k�����B�{�c33@ə�?+�@���B���                                    Bxo���  
          @���"�\@�녿����k�B�W
�"�\@�녾#�
��G�B��H                                    Bxo���  �          @����(Q�@�Q��33�r=qB��(Q�@��þaG���(�B�p�                                    Bxo�H  �          @��5�@�p���G��`z�B㞸�5�@�����ͿJ=qB��                                    Bxo��  �          @�ff�Dz�@�G���33�S�B����Dz�@�  ���W
=B�                                     Bxo�"�  �          @�Q��AG�@�{��\)�/
=B�G��AG�@�=q>�=q@
�HB�W
                                    Bxo�1:  �          @����'
=@�G�<#�
=���B�{�'
=@�=q?��HAY�B�W
                                    Bxo�?�  �          @�  �:=q@�(�>�  ?�Q�B�\)�:=q@˅?��ArffB�(�                                    Bxo�N�  �          @�
=���@�\)<#�
=�Q�B������@�Q�?�Q�AZ�RB�33                                    Bxo�],  �          @���-p�@��H�
=q���HB�q�-p�@У�?���A��B�33                                    Bxo�k�  �          @�녿ٙ�@�G���������B�{�ٙ�@�=q��  �z�B��                                    Bxo�zx  �          @��ÿ�
=@�  �33���B�
=��
=@�녾�33�8��B���                                    Bxo�  �          @��� ��@θR��  �hz�B�z�� ��@�ff���
�#�
B�W
                                    Bxo��  �          @�\��\@љ���{��Bڊ=��\@��
?�@��B�(�                                    Bxo�j  �          @��H� ��@ҏ\�B�\��(�Bݽq� ��@��?k�@�
=B��)                                    Bxo�  �          @�33��@�
=��p��Ap�B�#���@�z�>B�\?ǮB�33                                    Bxo�ö  �          @�zῘQ�@���?�ffAK�B��f��Q�@�p�@Mp�A�ffB���                                    Bxo��\  
�          @�����@��?�=qAQ�Bƙ�����@��@1�A��B�                                    Bxo��  �          @�����@�33=#�
>�{B��Ϳ���@��
?�\Af�RB���                                    Bxo��  �          @�\��z�@أ׾�
=�\(�Bӏ\��z�@��?�ffA)��B�\                                    Bxo��N  �          @����.�R@�(�����{\)B����.�R@����\)�33B��                                    Bxo��  �          @��H�<��@��У��V�RB�\�<��@��ͼ��
�.{B�\                                    Bxo��  �          @����Z�H@����{���\B�u��Z�H@�
=�}p���B�\                                    Bxo�*@  �          @�ff�a�@����z����HB���a�@�{�h������B�3                                    Bxo�8�  �          @����@n�R�qG���C�\���@���-p�����C
=                                    Bxo�G�  �          @�ff��G�@O\)���
�Cz���G�@��J=q���C	aH                                    Bxo�V2  �          @߮��\)@s�
�S33��=qC0���\)@�����R��ffCQ�                                    Bxo�d�  �          @����z�@���!���  CW
��z�@��
��33�7\)C
\                                    Bxo�s~  �          @ᙚ��\)@�33��=q�qG�CE��\)@��;���w
=C��                                    Bxo�$  �          @����p��@��ÿ���,(�B��p��@�p�>L��?�33B��q                                    Bxo��  �          @�  �@��@�
=�s33���B癚�@��@�Q�?+�@�\)B�W
                                    Bxo�p  �          @߮��
@�\)�n{��
=B�G���
@�Q�?@  @�{B�#�                                    Bxo�  �          @޸R���H@ָR�����/�B�\)���H@ڏ\>�(�@dz�B�                                    Bxo＼  �          @�  ��@�\)��G��ip�B�Ǯ��@޸R<��
>#�
B��                                     Bxo��b  �          @�Q�=#�
@أ׿��
�k�
B�aH=#�
@�  <#�
=uB�ff                                    Bxo��  �          @��ÿ��@���R��{B�  ���@Ӆ�Q���G�B��f                                    Bxo��  �          @�\)��@�z��1���ffB�p���@��Ϳ�{�33B�8R                                    Bxo��T  �          @�ff�(��@�(����~�\B�p��(��@��;�=q��
B�                                     Bxo��  �          @�p��#�
@�  ��Q����B����#�
@�녾����_\)B���                                    Bxo��  T          @�p���p�@�zῪ=q�;�CxR��p�@�=q����z�Cu�                                    Bxo�#F  �          @��
�L��@�  �����
B�\)�L��@��H>�Q�@J�HB�                                     Bxo�1�  �          @�  �L(�@�  �\)����B�=q�L(�@��R?^�R@��
B�                                    Bxo�@�  �          @�p����
@Y��@(�A��
C����
@.{@?\)AᙚC�                                    Bxo�O8  y          @����Vff@�Q켣�
���B��\�Vff@�33?��
AC�B�W
                                    Bxo�]�  �          @�p��l��@���   ����B�aH�l��@��?W
=@��
B��)                                    Bxo�l�  �          @Ϯ�hQ�@��R���
����B�\)�hQ�@�Q��(��y��B�G�                                    Bxo�{*  �          @У��hQ�@�{�����@  B���hQ�@��
=#�
>�p�B�33                                    Bxo���  �          @�33�.{@���>�p�@VffB�W
�.{@��?���A��\B�R                                    Bxo�v  �          @��N{@�(��#�
�L��B����N{@�
=?�Q�AC�B�u�                                    Bxo�  �          @�p���\@��R�J=q��\)B�  ��\@�\)?&ff@��
B��H                                    Bxo��  �          @��!G�@�\)��33�DQ�B���!G�@�33>�  @+�B��                                     Bxo��h  �          @�p��\@��H���H��p�B؊=�\@�(���ff��=qB֊=                                    Bxo��  �          @��
��\@�33��33�QB�k���\@�  <�>��RB��)                                    Bxo��  �          @�\)�z=q@R�\����׮C���z=q@U>�\)@Dz�C�                                    Bxo��Z  �          @{��G�@>{����(�B�aH�G�@?\)>���@��B�
=                                    Bxo��   �          @{��\)>�  @]p�B���B�L;\)�+�@Y��B�ffC��                                    Bxo��            @��?&ff�K��0���Ap�C��3?&ff�8Q��  ��G�C�ff                                    Bxo�L  �          @�p���������?xQ�A?�
Cy�������(��8Q����Cy�R                                    Bxo�*�  �          @��׾�G���z�?�\)A���C�(���G����>�
=@�=qC�S3                                    Bxo�9�  �          @��R�u���H?�z�A���C����u��G�=�Q�?�  C���                                    Bxo�H>  �          @����$z��\)?��AH��Cm0��$z���(�������Cn�                                    Bxo�V�  �          @��ͿTz���  ?޸RA��HC�\�Tz���G�>�(�@�=qC�W
                                    Bxo�e�  �          @��
��  ���H?��AM��C���  ��
=�.{����C�H                                    Bxo�t0  T          @�z��=q��  ?�G�A}G�Cy�3��=q��
=>\)?�Cz��                                    Bxo��  �          @��R�������@�A��Cff�����{?E�@���C�5�                                    Bxo�|  �          @�G���������?�@�=qC��쿬����ff����"=qC��q                                    Bxo�"  �          @\���
��{>.{?��C�����
���׿�33�U��C���                                    Bxo��  �          @ٙ���G����H?�@��C��=��G���Q쿙���"�RC�u�                                    Bxo�n  �          @��H���H��{?E�@ȣ�C�����H��zΉ��{C��                                    Bxo��  �          @��
���R��ff?k�@���C��Ϳ��R��{�u����C��                                    Bxo�ں  �          @�p��h����33>�z�@33C�Uÿh����p���{�O�
C�=q                                    Bxo��`  �          @��Ϳ@  ���>��@tz�C���@  �����8��C���                                    Bxo��  �          @��Ϳ:�H���H>��
@&ffC�3�:�H��p���=q�L��C�H                                    Bxo��  �          @�{�8Q���?�@���C��8Q��߮��33�4(�C�                                    Bxo�R  �          @���G���=q?(��@��C�{��G���\)�����$  C��                                    Bxo�#�  �          @�p��=p���\?.{@�G�C�!H�=p���  ����"�\C�R                                    Bxo�2�  T          @�\)�B�\����?#�
@�(�C���B�\�陚��\)�)p�C�f                                    Bxo�AD  �          @��ÿh����ff>�G�@VffC��׿h����G���=q�A��C�p�                                    Bxo�O�  �          @��Ϳ���\)>�G�@Z=qC�lͿ���\�\�>=qC�P�                                    Bxo�^�  �          @�33���H��ff=��
?(��C�G����H��ff��{�j=qC��                                    Bxo�m6  �          @���(���׽�G��aG�C�E��(���
=�z�����C��                                    Bxo�{�  �          @�  ���H��33>\)?�=qC�33���H���
���
�d  C��                                    Bxo�  �          @�
=��33��=q�Ǯ�@��C��쿳33��{�
=��33C�H�                                    Bxo�(  �          @�ff��Q���=q�0�����C�` ��Q��ۅ�(����ffC��                                    Bxo��  �          @�녿�p����ͿY����
=C�Ff��p������5���C��\                                    Bxo�t  �          @�zῪ=q�����z��(�C��=��=q��=q�HQ���  C�|)                                    Bxo��  �          @����
����  ��\C�!H���
��(��?\)���HC��                                     Bxo���  �          @�ff������׿c�
��z�C�uÿ����߮�9����C�!H                                    Bxo��f  �          @�p���33�񙚿����z�C��=��33���
�%��(�C�g�                                    Bxo��  �          @���{��׾�33�'
=C��H��{��z������p�C��)                                    Bxo���  �          @��ÿ���>#�
?��HC��H����녿�z��dQ�C��                                    Bxo�X  �          A �ÿ�Q���G��W���=qC~����Q���33��p��   C{��                                    Bxo��  �          A Q����{�:=q��
=C�����
������C|�=                                    Bxo�+�  �          @�z���H���H�>�R����C�4{���H��  ����z�C~\                                    Bxo�:J  �          @�����R��\)�|����(�C�����R��(������6��C��H                                    Bxo�H�  T          @�G��
=�����(�� G�C�h��
=��p������?�HC��R                                    Bxo�W�  �          @��þB�\�Ϯ������C�(��B�\��������E�C���                                    Bxo�f<  �          @��׾�Q���\)��Q��=qC�c׾�Q�������z��E=qC��                                    Bxo�t�  �          @����}p���������!��C�,Ϳ}p���ff���`��C�`                                     Bxo�  �          @��ÿ����������:z�C~�쿵�XQ��ۅ�wffCw5�                                    Bxo�.  �          @�����������33�=��C�������QG��أ��{�
C{aH                                    Bxo��  �          @񙚿��
�u����iQ�C~�ÿ��
������)Cqu�                                    Bxo�z  �          @�=q�=p��0���߮��C�=p��333��\)¤�C_��                                    Bxo�   �          @�(��+��b�\��p��r��C��f�+���{�����{Cwu�                                    Bxo���  �          @�=q�(����������[��C�w
�(���z���p���C~                                    Bxo��l  �          @�G���=q������ff�M�HC�H���=q�0���׮L�C�/\                                    Bxo��  
�          @�G���z��Z=q��p��w�RC�����z´p����
�C�`                                     Bxo���  �          @�{>��8����=q� C��>��\(���¤��C��                                    Bxo�^  �          @��>B�\�I����p�L�C��H>B�\��33��G�¡�qC��H                                    Bxo�  �          @�R=L���w��˅�j�C�e=L�Ϳ�
=��{  C���                                    Bxo�$�  �          @�ff����g
=�Ϯ�r��C�P���׿У���  ��C~\                                    Bxo�3P  �          @��E��tz�����i33C�G��E������(��Cw��                                    Bxo�A�  �          @�33�����l(��ʏ\�n�C�s3���ÿ�G�����C��                                    Bxo�P�  T          @�G��333�w���(��e�C�� �333��p���
=u�Cz��                                    Bxo�_B  �          @��
�+��]p��θR�up�C��H�+����R��Q�Cu�R                                    Bxo�m�  �          @�Q�>��5���z�C�E>��n{��33¤\)C�ٚ                                    Bxo�|�  �          @߮�(��QG������t��C��
�(����׮33Cv�                                    Bxo�4  �          @��Ϳ.{�]p���
=�q�HC�t{�.{�����ff�RCvL�                                    Bxo���  �          @�p��Q��������I{C���Q��/\)�љ���C}B�                                    Bxo���  �          @�p��k����H��33�,��C��H�k��_\)��33�m�C=q                                    Bxo��&  �          @�ff@8Q�<��
��  �}��>�
=@8Q�?�z���  �kp�A�ff                                    Bxo���  �          @߮?Y���e����h  C���?Y����  ��{�{C���                                    Bxo��r  �          @�
=>������z��?33C�Ф>��?\)��Q��C�E                                    Bxo��  �          @޸R�����\)�u��p�C�������z���=q�H��C��                                    Bxo��  �          @�p���  ��Q��(�����C~ff��  ����y���
�\C{��                                    Bxo� d  T          @޸R��{����N�R��C|  ��{�����  �/��Cw�                                    Bxo�
  �          @�׾�p������
�  C����p��|(���  �]��C�K�                                    Bxo��  �          @���>���G��e��C��H>�������>(�C���                                    Bxo�,V  �          @�녿�33��G�����\)C|�f��33�tz�����V�Cv�{                                    Bxo�:�  �          @�p���=q��33��p��#  Cz8R��=q�aG���ff�a
=Cr��                                    Bxo�I�  �          @���\������z��7��Cr����\�.{��ff�p��Ce�H                                    Bxo�XH  
�          @����������]p����C~�=�����Q���=q�4Cz�q                                    Bxo�f�  �          @��
������p���G��
��C�H������������K�RC{)                                    Bxo�u�  �          @�z��{��(���z���\Cv���{�g���{�U(�Cn��                                    Bxo��:  �          @�녿���{�qG���C}������H�����BG�Cy                                      Bxo���  �          @�G���G���Q����R��HC�E��G��n�R����_�C{Q�                                    Bxo���  �          @�Q������G���33�/ffC{������J=q�����n�Cs+�                                    Bxo��,  �          @��H��z���������C�1쿔z���ff���H�P\)C~��                                    Bxo���  �          @�녿��R�������\�z�C  ���R��33�����O  Cz�                                    Bxo��x  �          @�33���H�������\�!{C������H�k����c\)C{�\                                    Bxo��  �          @�33�Y����������533C���Y���L����Q��xz�C\                                    Bxo���  �          @���G���ff�����A�C{@ ��G��+���(�u�Cp�)                                    Bxo��j  �          @�33�ٙ����������:�
Cy�\�ٙ��3�
�ȣ��yffCn�                                    Bxo�  �          @�׿��
��  ��z��&\)C|�3���
�X������g=qCu�                                    Bxo��  �          @����33��\)��(��%ffCt
�33�H�����\�a�Ci�=                                    Bxo�%\  �          @�=q��33��z������33C�33��33��z���z��VffC��                                    Bxo�4  �          @�G��˅��(��r�\���C~33�˅��  �����E��Cyh�                                    Bxo�B�  �          @�  ���R���
�^�R���C�q���R������\�:�C{�H                                    Bxo�QN  �          @߮���R��(���p����C~�=���R�xQ����H�U\)Cy
=                                    Bxo�_�  �          @�Q������\)��G���CzǮ�����l������WCs                                    Bxo�n�  �          @�����������
�HCy� ���~{��
=�K(�Cr�                                    Bxo�}@  �          @߮�7��fff����3  Cgk��7���
���c��CW�)                                    Bxo���  �          @�33����  ��Q��J��CAk���?#�
����M��C+Y�                                    Bxo���  �          @�33�q��J�H����+  C[�R�q녿������P��CKxR                                    Bxo��2  �          @�z��K��\����ff�:  CcE�K������Q��fp�CQO\                                    Bxo���  �          @�Q������=q�����{Cz5ÿ����_\)��\)�^
=Crp�                                    Bxo��~  �          @�G��2�\�������
��RCpL��2�\�Z=q��p��KffCf�3                                    Bxo��$  T          @�=q�'
=�����u��=qCs���'
=�w
=��  �A��Ck��                                    Bxo���  �          @��2�\��z����\�+z�Cm���2�\�-p���{�c(�C`8R                                    Bxo��p  �          @��Dz����R����� 33CkxR�Dz��7
=��{�V�\C^�3                                    Bxo�  �          @�(��[�����������Cg��[��)����33�Q
=CY�3                                    Bxo��  �          @�{�r�\��\��
=�WQ�CM��r�\=�\)����f��C2�                                    Bxo�b  �          @�R�~�R�5���G��=(�CWz��~�R���������\p�CC
                                    Bxo�-  �          @�  ����|(���p��(�C`+������
����ICQ��                                    Bxo�;�  �          @��
�������~{��C\�����'
=�����,p�CQ�                                    Bxo�JT  �          @�z�����o\)��\)���CZQ����������2p�CMxR                                    Bxo�X�  �          @�z�����c�
��=q��CY�������(���ff�=�\CK+�                                    Bxo�g�  �          @��r�\�c33�����0ffC_\�r�\����(��Y��CM��                                    Bxo�vF  �          @��
�P���e���G��CffCc�3�P�׿�Q���(��p\)COW
                                    Bxo���  �          @���33�.�R��=q�,�HCT+���33������{�I��CB.                                    Bxo���  T          @��H��33�J=q�vff� �C;ff��33>�ff�y���(�C/��                                    Bxo��8  �          @�(��˅>���l����C1���˅?�z��\(��ޏ\C'�                                    Bxo���  �          @�{��
=    ��  ��C4  ��
=?�  �s33��G�C(��                                    Bxo���  �          @�p����׿�����p���CC�)���׼����'{C4B�                                    Bxo��*  �          @陚��\)��z���ff�Q�C6���\)?�\)��=q�{C(�                                    Bxo���  �          @�=q���
?�33�j�H��(�C$T{���
@-p��=p�����C+�                                    Bxo��v  �          @��
�Ǯ?����N�R�ң�C"���Ǯ@7������p�CG�                                    Bxo��  �          @�(���(�?��R�<�����HC"����(�@4z������RC(�                                    Bxo��  T          @����33?��R�7���G�C)W
��33@����
=C"u�                                    Bxo�h  �          @�p����>k��z�����C2(����?^�R��33�n�HC,��                                    Bxo�&  �          @�\)��  ��Q�������C?����  �+��!G����
C9z�                                    Bxo�4�  �          @�\��  ���33�{�CD&f��  ���\�%���C>G�                                    Bxo�CZ  T          @����zῸQ��   �uC?k���z�:�H�
=��\)C9�
                                    Bxo�R   
�          @�����?���vff�  C'������@���O\)���HCn                                    Bxo�`�  �          @�\)��@Q���33�!p�CG���@s�
�u���Q�C��                                    Bxo�oL  �          @�  ��Q�@z������p�C33��Q�@n�R�s33��Q�C�H                                    Bxo�}�  �          @��H��{?������!�C �)��{@W
=�����\)CE                                    Bxo���  �          @�p���z�?�������C%���z�@5�e���\)C.                                    Bxo��>  �          @�33��G�?�����
�p�C*h���G�@��c�
�ߙ�C��                                    Bxo���  �          @�=q�ə�?������� �RC*+��ə�@��]p����
C�3                                    Bxo���  T          @���=q?�33������C"�)��=q@G
=�N{��(�C�f                                    Bxo��0  �          @�Q���{?���s33��C#\��{@AG��?\)���C
=                                    Bxo���  �          @������R@{��Q�� �C�\���R@Y���Dz���
=CJ=                                    Bxo��|  �          @������
@)���w����C�R���
@p���333��
=CY�                                    Bxo��"  �          @�����@W��e��=qCW
��@�33�33���\C��                                   Bxo��  �          @�  ����@e�e���Q�C�q����@���{����Cu�                                   Bxo�n  
�          @�����R@�\)�l������C����R@��R����\)C�
                                    Bxo�  �          @�  ��G�@�=q��Q��33C����G�@���?333@�(�CE                                    Bxo�-�  �          @�z����
@�(���\)�K
=C�
���
@��.{����C�                                    Bxo�<`  �          @��
���H@e��c�
�޸RC�����H@j�H>���@'
=C�                                    Bxo�K  �          @�
=��ff@!�?��\@�=qC����ff@�
?��A^=qC#��                                    Bxo�Y�  �          @�����@%�?�Q�A(�CE���@�\?�(�Aw�
C#O\                                    Bxo�hR  �          @�Q�����@���E��̏\C������@�����=q�+
=B�p�                                    Bxo�v�  �          @���33@����*=q��(�C ���33@���O\)�љ�B�\                                    Bxo���  �          @�z��~�R@��R�e����C:��~�R@�(������lz�B��{                                    Bxo��D  �          @��c33@�p���p����B��\�c33@�G�������HB��                                    Bxo���  �          @�z��c�
@���p  � G�B�\�c�
@��ÿ�Q��~�HB�G�                                    Bxo���  �          @��H����@�ff�O\)�ڸRCn����@�Q쿾�R�C33B��                                    Bxo��6  S          @���\)@�{�c�
����C�
��\)@��
��\)�t��C 33                                    Bxo���  �          @�33����@����s33�C�����@��(���Q�C��                                    Bxo�݂  �          @�\�i��@����e��=qC (��i��@��\��=q�u�B�ff                                    Bxo��(  �          @�=q�y��@���k���  Cz��y��@��ÿ�
=�
=B�L�                                    Bxo���  �          @�p��N�R@����{��33B�z��N�R@��� �����\B�33                                    Bxo�	t  �          @�녿���@`  ���
�e\)B�33����@�����33�ffB�u�                                    Bxo�  �          @�(�>�(�?��H�����B�8R>�(�@j=q��z��g��B���                                    Bxo�&�  T          @�G��A�@L(����
��G�C�\�A�@aG���Q���ffC                                    Bxo�5f  �          @����
=@�{?ǮAC�Cc���
=@xQ�@C�
A�  C��                                    Bxo�D  �          @�\���H@���?�G�@��
CT{���H@�(�@/\)A�G�C)                                    Bxo�R�  �          @�������@��?�p�A�C
Y�����@�ff@5�A�p�C�q                                    Bxo�aX  �          @�p����
@�G�@J�HA�C	  ���
@L(�@���Bz�C                                    Bxo�o�  �          @�
=��33@�\)?���A'�
CL���33@�ff@C�
A�(�C
�H                                    Bxo�~�  �          @�z���(�@Å>��?�Q�B����(�@�{@�RA���B���                                    Bxo��J  �          @�(���G�@\�.{����B�8R��G�@���?�As�B�                                      Bxo���  �          @������@�=q=L��>ǮB������@�ff@�\A�{C ��                                    Bxo���  �          @�����@�(�=�G�?fffB�p����@��@Q�A�C &f                                    Bxo��<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxo���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxo�ֈ   �          @�z��i��@�����Q�� �CJ=�i��@�{�?\)���RB�ff                                   Bxo��.  �          @��9��@�����R�4G�B��H�9��@�33�Y�����B�Ǯ                                   Bxo���  �          @�\��z�@��\�����E��B���z�@�
=�k����B�aH                                   Bxo�z  T          @�  ��@'�����|��B�k���@���33�5
=B��=                                   Bxo�   �          @�
=� ��?�
=��z��\C&f� ��@��
����J
=B���                                   Bxo��  �          @�ff�G�?����{�=C��G�@w
=��ff�RQ�B�#�                                   Bxo�.l  �          @���Q�?�
=��33z�C=q��Q�@_\)��Q��d�B�                                    Bxo�=  �          @���\@XQ��Z=q���B�#���\@���G���{B�W
                                    Bxo�K�  �          @�(��`��@ƸR?�Ac�B����`��@�@w�A�B�8R                                    Bxo�Z^  �          @�z��|��@�33?�{A
�\B��H�|��@��\@K�A��HB�.                                    Bxo�i  �          @��
��(�@\>�
=@S33B�\)��(�@�G�@!�A��B�\)                                    Bxo�w�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxo��P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxo���   �          @������@�(���{�	G�B������@�(�?���Az�B��\                                   Bxo���  �          @�G��R�\@ƸR�޸R�^�RB��)�R�\@�?�@�Q�B�G�                                   Bxo��B  �          @�\��@���@G
=A��HB�p���@�z�@��HB4  B�L�                                   Bxo���  1          @�{�"�\@��@33A��\Bݽq�"�\@��@�G�B33B噚                                   Bxo�ώ  w          @�\)�ff@���?�@���B����ff@�z�@?\)A��
B�p�                                   Bxo��4  
�          @��(�@�?
=@���B����(�@�@EA��B�z�                                   Bxo���  �          @���33@��?Tz�@�33B���33@���@U�A�33B׊=                                   Bxo���  �          @����G�@�=q?E�@�=qB�녿�G�@ʏ\@P��A���B�\                                    Bxo�
&  �          @�׿\(�@�ff�)����G�B��\(�@�R��\)���B���                                    Bxo��  �          @��aG�@���S33���B��aG�@�p��G���ffB��                                    Bxo�'r  �          @陚���@���>�R���HB�ff���@�ff����P��B���                                    Bxo�6  �          @��ÿ�@�
=��Q��[�B���@�(�?W
=@ٙ�B�Q�                                    Bxo�D�  �          @�33��
=@�=q����{BҞ���
=@�G�@G�A�B���                                    Bxo�Sd  T          @�R��R@޸R�u����B�#׿�R@��H?У�AUG�B�Q�                                    Bxo�b
  �          @�R���@�\)�����)G�B��ÿ��@�\)?�ffA'
=B���                                    Bxo�p�  �          @�R�   @�p���z��v�HB�G��   @�z�?:�H@��B�                                    Bxo�V  �          @��ü#�
@�z��\)�r�\B���#�
@�33?E�@�p�B��                                    Bxo���  �          @�G����@�33?�\@�=qB׊=���@�ff@>{A¸RB�
=                                    Bxo���  T          @��ÿ��H@�33?�(�A�
B����H@�@h��A�Bؔ{                                    Bxo��H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxo���  �          @�G���@�\)?�Q�A6�RB�(���@��@s�
A��\B�.                                   Bxo�Ȕ  �          @陚�'
=@׮?\(�@��B�k��'
=@�ff@Q�A�  B�k�                                   Bxo��:  �          @��0  @�G�?�@�\)B�
=�0  @�33@A�A���B�p�                                   Bxo���  �          @����5�@׮?p��@陚B�{�5�@�p�@W
=A�z�B��                                   Bxo��  �          @���Q�@�Q�@��A���CL���Q�@�G�@�(�BQ�Cs3                                   Bxo�,  �          @�p���(�@�=q?�\)AR�\B����(�@��\@dz�A�Q�C\                                   Bxo��  �          @�{�Z=q@�Q쿐����
B�=q�Z=q@�
=?�{A(��B�\                                    Bxo� x  �          @�ff�8��@�
=��Q��  B�z��8��@�?��A,z�B�R                                    Bxo�/  �          @�\)�33@��H����  B�#��33@��@ ��A�G�Bڙ�                                    Bxo�=�  �          @��>�R@�=q�&ff���B�#��>�R@�=q?�
=Ap(�B�3                                    Bxo�Lj  �          @�\)�I��@�z���
����B��I��@�ff>��@h��B�ff                                    Bxo�[              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxo�i�            @�Q��J=q@�=q��G��Yp�B�p��J=q@�  ?Q�@ə�B�8R                                   Bxo�x\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxo��   �          @���(�@y���c�
���C{��(�@�ff��=q�`��C	h�                                    Bxo���  �          @�����\@�33�S33���HCn���\@�G���G��:ffC�)                                    Bxo��N  �          @�G��[�@�
=�@�����B�q�[�@���
=q��z�B�p�                                    Bxo���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxo���   �          @�(��L��@ٙ�����
{B�k��L��@�\)?�G�A7�
B��f                                    Bxo��@  �          @�G���=q@�ff����h��CL���=q@���>aG�?ٙ�C�\                                    Bxo���  �          @�����@�G����\��C�����@��?n{@�33C+�                                    Bxo��  �          @�����@�G�����$��C +�����@��?u@�(�B���                                    Bxo��2  �          @�=q����@�\)�����$��B�������@���?��
@��HB�{                                    Bxo�
�  �          @�33���@��
���H�=qC u����@�z�?�\)A(�C aH                                    Bxo�~  T          @�\���@����)����G�C	=q���@�p��!G���G�C�
                                    Bxo�($  �          @��H��p�@Fff�����Q�C���p�@�33�AG�����C
J=                                    Bxo�6�  �          @�  ���@@���Tz���{CǮ���@�����z��v�RC(�                                    Bxo�Ep  �          @�ff���@U?z�HA�Cs3���@-p�@(�A���C��                                    Bxo�T  �          @�{���@xQ�@�G�B��C
�����?�=q@�BGCh�                                    Bxo�b�  �          @�{��
=?G�@���B#\)C+����
=��\)@��RB �\C?�{                                    Bxo�qb  �          @����=#�
@���B3\)C3�
����   @�(�B"�CH��                                    Bxo��  �          @�z������Q�@���BJ�CM@ ����z�H@��HB�C_�f                                    Bxo���  
�          @�ff��ff��\)@��
B:�
CI����ff�s33@�\)B�HCZ��                                    Bxo��T  �          @�{��=q�u@�(�B/��C6�H��=q�p�@�(�BG�CJ�)                                    Bxo���  �          @����>�\)@�ffB)=qC1{����@�p�B��CE�                                    Bxo���  �          @�ff���H?�z�@��
B;\)C#�����H�J=q@�  BA(�C=G�                                    Bxo��F  �          @��
����?��@l(�BffC"#�����>W
=@�33B�C1�)                                    Bxo���  T          @�=q��\)?��
@VffA�z�C$���\)>���@qG�A�\)C0��                                    Bxo��  �          @�(���{@0  @
=qA�p�C����{?�
=@Dz�A��C%\)                                    Bxo��8  �          @�����\)��ff@~{B��C>����\)�!G�@O\)A��
CL��                                    Bxo��  �          @����ƸR���@P��A��CA�f�ƸR�,(�@��A�CKn                                    Bxo��  �          @�z�������@�A�{C<�=�����?�
=AR{CC0�                                    Bxo�!*  �          @�\)��녾u?�(�A5��C5ٚ��녿J=q?�G�A33C:5�                                    Bxo�/�  �          @�G���z�L��?��HA��C:(���zῚ�H?L��@��
C=Q�                                    Bxo�>v  �          @���(���G�=#�
>�{C;Ǯ��(��p�׾�Q��0��C;J=                                    Bxo�M  �          @�����@  ?�\@w�C9� ���c�
>.{?��C:�\                                    Bxo�[�  �          @�����=�Q�>u?�{C3O\��׼#�
>��?��RC4�                                    Bxo�jh  �          @�����\)?�R=��
?!G�C/J=��\)?�>��
@(�C/�                                    Bxo�y  T          @�ff��33>�G�<��
>8Q�C0����33>���>L��?��C0��                                    Bxo���  �          @����33��\�����%G�CD���33��=q�33�}�C>�H                                    Bxo��Z  �          @��H����
�H�ff���CF�����33�2�\����C=�                                    Bxo��   �          @����33�4z��G����CK���33���
�>�R��=qCC�H                                    Bxo���  �          @���1G�@�{?�{AW�B� �1G�@\)@N{B\)B��\                                    Bxo��L  �          @�ff�\(�@�z�@
=A�z�B�ff�\(�@��@�
=B-�RB���                                    Bxo���  �          @�p�����@\@0��A�B�����@��R@�
=B9�B�Ǯ                                    Bxo�ߘ  �          @�p��8Q�@θR@,(�A��B���8Q�@��\@��B7��B��f                                    Bxo��>  �          @�33��@�{@@  A�=qB�\��@�@��HB<(�B�33                                    Bxo���  �          @�
=���H@���@fffA��\B�33���H@�=q@�
=BS�RB�u�                                    Bxp �              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp 0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp (�   �          @���@�@A�Q�B�녿�@�@�  B(G�B��                                    Bxp 7|  T          @�{=�\)@У�@�  A�Q�B�Ǯ=�\)@��\@��B^
=B�(�                                    Bxp F"  �          @�z�<��
@�  @�(�B  B���<��
@{�@�G�Bl  B�u�                                    Bxp T�  �          @�33>k�@���@���B �B���>k�@��R@���B`��B���                                    Bxp cn  �          @�녿.{@���@��
B
��B�33�.{@l��@�p�BjQ�B��H                                    Bxp r  �          @�녾8Q�@ȣ�@k�A�
=B�W
�8Q�@�
=@�p�BZ  B���                                    Bxp ��  �          @�p����
@���@w
=A��\B�p����
@���@��HB^��B�.                                    Bxp �`  �          @�����H@�(�@A�A�\)B�\���H@�=q@�(�B?{B�\                                    Bxp �  �          @�����H@��H@�RA�z�B�L���H@�(�@���B#B�\                                    Bxp ��  �          @��=�Q�@߮@N{AŅB�� =�Q�@���@��BC�B��                                    Bxp �R  �          @�  ��ff@�=q@
=qA�Q�B�ff��ff@���@�\)B!=qBȮ                                    Bxp ��  �          @��ÿ���@���@�\A��B�  ����@�{@��HB$=qBή                                    Bxp ؞  �          @����ff@�\?��@���B���ff@Ǯ@\)A�  B�(�                                    Bxp �D  �          @��ÿ���@��H���i��B�ff����@��
@&ffA��\B��)                                    Bxp ��  �          A (��N�R@��
?�{A:ffB����N�R@��@�p�Bp�B�#�                                    Bxp�  �          AG��p  @��
@j�HA��HB���p  @�G�@�(�B;p�C��                                    Bxp6  �          A ���b�\@�\)@i��A�{B�G��b�\@�z�@��B==qC�                                    Bxp!�  �          A (��O\)@�z�@|��A��HB�{�O\)@|(�@���BIG�Cff                                    Bxp0�  �          A ���b�\@�@�
=B  B��b�\@S33@�BTG�C�R                                    Bxp?(  �          A��G
=@�G�@|��A��B��G
=@��@�
=BJ33B��                                    BxpM�  �          A z��=q@��
@�BB���=q@mp�@ҏ\B`p�B���                                    Bxp\t  �          A (���@�p�@w�A癚B�Q���@�@��BP\)B�33                                    Bxpk  �          A ���,��@�{@�33B
=B���,��@s33@�G�BZ33B��
                                    Bxpy�  �          A����@�  @�{B�B�L����@W�@�  BpffB�L�                                    Bxp�f  �          A�R�  @�p�@�p�B9�B�{�  @(�@�B�ǮC�q                                    Bxp�  �          A{�%@�=q@�p�BFQ�B�33�%?��@��B�B�CG�                                    Bxp��  2          A �׾��H@�G�@'�A��HB�z���H@�\)@��B0{B��
                                    Bxp�X  v          A �ÿ���@�{@33Al��B��
����@�33@��B{B��H                                    Bxp��  �          @�
=�(��@���@33A��B�
=�(��@�33@���B'�B��                                    BxpѤ  �          @��(��@�33>u?�p�B����(��@�Q�@c33A��
B��q                                    Bxp�J  �          @�
=��ff@�33@<(�A�(�B̙���ff@�@�B9��B�G�                                    Bxp��  �          AG����@�  @�  B$ffB֏\���@>{@�ffB��{B��                                    Bxp��  �          Ap��\)@�33@���A�  B�z��\)@�Q�@�ffBU33B�z�                                    Bxp<  T          A��*�H@�{@C33A�=qB���*�H@�\)@�
=B5ffB�p�                                    Bxp�  �          Ap��Mp�@��@��A�=qB��H�Mp�@��@��BG�B�L�                                    Bxp)�  �          A �׿��@ڏ\?�\)Al��B��H���@��
@��
B�Bڮ                                    Bxp8.  �          A   �L(�@�{@33A�(�B�Q��L(�@�G�@��BG�B�.                                    BxpF�  �          @�
=�b�\@�(�?�AB�\B�k��b�\@�  @��RB	(�B��                                    BxpUz  �          A Q���(�@�33?B�\@��B�(���(�@�(�@a�A�Q�B�                                    Bxpd   
�          A ������@�?�\)A�C������@��
@dz�AӅC	B�                                    Bxpr�  �          A z�����@��H@33AnffCc�����@��@�Q�BQ�CǮ                                    Bxp�l  �          A Q��B�\@��
@q�A��RB��f�B�\@\(�@��\BO(�C��                                    Bxp�  �          A��6ff@ȣ�@vffA陚B��)�6ff@\)@�BN\)B�
=                                    Bxp��  �          A��w
=@��@O\)A�33B��H�w
=@���@���B1�C�q                                    Bxp�^  �          A ���B�\@��H@9��A�z�B�u��B�\@�@�
=B1�
B��                                    Bxp�  �          @��R�33@��
@Tz�A��HB�#��33@�  @�\)BD�\B♚                                    Bxpʪ  �          @�p���
=@�Q�@S�
AƸRB����
=@��
@���BG��B���                                    Bxp�P  �          @�G���=q@ۅ@G
=A��
B�녿�=q@�=q@�G�BD��B���                                    Bxp��  �          @�녿aG�@�{@s�
A�\)B��aG�@��@��HB[B��H                                    Bxp��  �          @��
����@�@1�A�(�B�����@���@��B9
=B��                                    BxpB  �          @���W
=@��H@/\)A��HB��W
=@���@�p�B8G�BŞ�                                    Bxp�  �          @�p�����@�  ?�@r�\BŞ�����@׮@w
=A�p�B�33                                    Bxp"�  
(          @�p���z�@�
=?�G�Az�B�Q쾔z�@�(�@���B{B�8R                                    Bxp14  �          @�p�?�@�@ ��AlQ�B��?�@�ff@���B#  B��                                    Bxp?�  �          @���?�ff@�=q@W�A�{B�z�?�ff@���@���BI��B��3                                    BxpN�  �          A ��@w�@Å@XQ�A�ffBf�R@w�@�  @�\)B7Q�B7�
                                    Bxp]&  �          A   @n{@�p�@�A~�\Bsff@n{@���@�z�B=qBV                                      Bxpk�  �          @�ff@>{@�?��
A2ffB��@>{@���@���B�B{�                                    Bxpzr  �          @�{?ٙ�@�R?�=qAW�B�ff?ٙ�@�(�@�{B�B��)                                    Bxp�  �          @�p�?�ff@��R?���A ��B�p�?�ff@ə�@���B33B�                                    Bxp��  �          @��?L��@�Q�?���@��HB�33?L��@�\)@���B
=B��H                                    Bxp�d  �          A �;Ǯ@���?�z�A!G�B����Ǯ@�ff@���B��B��)                                    Bxp�
  �          A z��\@�z�?��A
=B�ff��\@��@���B
{B��H                                    Bxpð  T          A   �333@�@�A���B����333@�{@��
B,�\B��                                    Bxp�V  �          @���33@�\?��
APz�B����33@���@��B(�B�k�                                    Bxp��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp�   �          A ���~�R@��R@�G�A���B��q�~�R@S33@��BH
=CO\                                    Bxp�H  �          A ���z=q@���@�G�BffB����z=q@AG�@ə�BO�CQ�                                    Bxp�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp*:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp8�   �          A (����?   @�p�A�\)C/�{��녿�
=@|��A�\C?��                                   BxpG�  �          A���{��\@���B  C8W
��{�\)@s�
A�RCHn                                   BxpV,  �          A�����H��@P��A��RCD  ���H�Q�?��RAd��CL�)                                    Bxpd�  �          A �����Ϳ��\@^{A�Q�C>
�����,��@!G�A�{CH��                                    Bxpsx  �          A ����
=?W
=@���A��
C,�)��
=��\)@��\A�G�C=��                                    Bxp�  �          A (���
=?�33@~�RA�33C&:���
=����@�G�B(�C6�
                                    Bxp��  �          A   ��p�@ff@���B��C!�H��p����
@���B�C4��                                    Bxp�j              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp�  �          A Q���Q�@5@j=qA��Ck���Q�?^�R@���B
�
C,ff                                   Bxp��  �          @�
=��  @n�R@hQ�A�\)C&f��  ?�@�B\)C$xR                                   Bxp�\  �          @�\)����@y��@r�\A��C�R����?޸R@���B"�C#:�                                   Bxp�  �          @����(�@P  @\��AͮC�q��(�?���@���B��C(O\                                   Bxp�  �          @�����@7
=@�z�B�C�\���>�
=@�B,
=C/�)                                    Bxp�N  �          A   ��@$z�@�A���C #���?�@UA���C*�f                                    Bxp�  �          A ����\@\)@�A��\C#��\?aG�@K�A��\C-+�                                    Bxp�  �          Ap����@(��@l��A�ffC^����?&ff@�  B��C.n                                    Bxp#@  �          A�H��\)@3�
@}p�A�(�C����\)?.{@���B��C.
=                                    Bxp1�  T          A�
��Q�@C�
@}p�A�\C�\��Q�?fff@�B\)C,#�                                    Bxp@�  �          A(���z�@$z�@���B�
C)��z�>8Q�@�
=B  C2\)                                    BxpO2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp]�            A\)���@l(�@��B.z�C�����?(��@�
=B\{C+�R                                   Bxpl~  �          A�
��{@Fff@�  B�
CT{��{>��
@\B@��C0��                                   Bxp{$  �          Az���(�@(�@�Q�BE�C޸��(��u@��BSG�C?�                                   Bxp��  �          A����@Vff@�33B@
=C}q���>\)@޸RBgC25�                                    Bxp�p              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp�  
          Ap���\)@^�R@��B$z�C���\)?�R@�=qBN�C,�f                                   Bxp��  �          AG���z�?�33@�p�B*�
C#����zῇ�@�=qB0��C>�f                                   Bxp�b  �          Ap���\)?^�R@�\)BK��C*\��\)�Q�@�p�B>33CK&f                                   Bxp�  �          A���
=>.{@�(�BF33C2���
=�,��@�  B,\)COaH                                   Bxp�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp�T  �          @�\���\@��
@{A�ffC:����\@;�@�33BQ�C�R                                   Bxp��  �          @����
@��R���H� 33B������
@�
=��  �B�\B�L�                                   Bxp�  
�          @�G�?���@�33���R�3�B�W
?���@�ff�p����
B��                                    BxpF  �          @�@%?����ٙ���A��@%@����z��:��Bl�                                    Bxp*�  �          @�=q�   @�p���(��B܅�   @޸R����@Q�B�{                                    Bxp9�  �          @����@�����=q�>p�B��H���@�(��1�����Bٮ                                    BxpH8  �          @�>�{@&ff��33k�B�{>�{@�z���G��'  B�{                                    BxpV�  �          @��Ϳ!G�?Y����(�£�RB�aH�!G�@�
=�Å�\=qB���                                    Bxpe�  �          @�G�?(�@��;\�7�B��H?(�@�@QG�A�p�B��                                    Bxpt*  �          @�녾�G�@�녿�(��L��B��q��G�@�\)@33Au�B���                                    Bxp��  �          @���?+�@��H�����RB��?+�@�R@(Q�A�p�B�ff                                    Bxp�v  �          @���?#�
@�G��\)��z�B���?#�
@�33?�  Ap�B�\)                                    Bxp�  �          @��þB�\@�  �:=q���\B�\�B�\@�?@  @�ffB��
                                    Bxp��  �          @���?:�H@�{�(Q����B�\)?:�H@�=q?���A
=B��                                    Bxp�h  �          @����.{@�=q����  B����.{@޸R@ ��A��RB�                                    Bxp�  �          @��\�\)@�33�'���G�B�W
�\)@��?s33@�z�B���                                    Bxpڴ  �          @���<��@�ff>��@^�RB�z��<��@Å@vffA�Q�B�z�                                    Bxp�Z  �          @��
�c33@��@>{A�z�B��c33@�z�@��
B7�\C�H                                    Bxp�   �          @�p��X��@�ff@�A�  B�R�X��@���@��B'\)B�Ǯ                                    Bxp�  �          @��H�Tz�@��@�A���B����Tz�@�{@�=qB+33B��{                                    BxpL  �          @��
�G�@���@C33A���B�\�G�@���@���B>Q�B�(�                                    Bxp#�  �          @�ff�y��@�33@#33A�33B��y��@��@���B(\)CǮ                                    Bxp2�  �          A   �p�@�Q�>���@   Bـ �p�@ƸR@qG�A�p�B�B�                                    BxpA>  �          A �ÿ�ff@�  ?�p�AL  B�k���ff@�  @��
B"��B�=q                                    BxpO�  �          A �Ϳ
=@�33>�Q�@'�B����
=@�
=@��\A��B�
=                                    Bxp^�  �          A�>#�
@��R?���A%G�B��>#�
@���@�33B(�B�\                                    Bxpm0  T          Aff?h��@�\)    �#�
B�?h��@�  @uA�Q�B�8R                                    Bxp{�  �          A(�����@��@W�A�33B��\����@�
=@ҏ\BS�RB�\)                                    Bxp�|  �          A�
�L��@�@j=qAӮB��L��@�@�  B]�B�.                                    Bxp�"  �          A\)�!G�@�\@���A�(�B�33�!G�@�Q�@�ffBhB��H                                    Bxp��  �          A��G�@�ff@3�
A�Q�B��ÿG�@�p�@��BD�
B�(�                                    Bxp�n  �          A(��L��A�H���H�XQ�B�.�L��@�z�@a�A��B�\)                                    Bxp�  �          A�����A�H?�ffA+\)B�����@���@��B�\B��3                                    BxpӺ  �          A�\>��RAQ�?���AQ�B��R>��R@���@��B�B��                                    Bxp�`  �          A�\?B�\A�\?�Q�A:ffB�L�?B�\@�=q@�{B"
=B�33                                    Bxp�  �          A�\?�Q�@���@'
=A�G�B�p�?�Q�@���@��B9Q�B���                                    Bxp��  �          A�?�\)@�@'�A��RB���?�\)@�{@���B9Q�B�{                                    BxpR  �          A��?���@�z����>ffB��q?���@�
=@�A�
=B�=q                                    Bxp�  �          A�?�G�A=�Q�?(�B��{?�G�@�G�@���A�=qB�                                      Bxp+�  T          Az�@!G�@�33?!G�@�33B�aH@!G�@љ�@�(�A���B��H                                    Bxp:D  �          A�>�=qA �ÿ��M�B�8R>�=q@�{@�A��\B�(�                                    BxpH�  �          A��>��RA�\��  �&{B��{>��R@�33@+�A��RB�k�                                    BxpW�  �          A?\(�A  �G���(�B��?\(�@�\@VffA��
B�{                                    Bxpf6  �          A(�@{@��>���@�B���@{@׮@��A��HB�z�                                    Bxpt�  �          AQ�@  @�z�?s33@ҏ\B��@  @��@�{B
ffB�B�                                    Bxp��  �          A��?��A �ÿ�����B�k�?��@�G�@AG�A�  B��                                    Bxp�(  �          A��?�{AG����W
=B�(�?�{@�33@w
=A�  B��R                                    Bxp��  �          A��@�A ��=u>���B��3@�@�\)@�Q�A���B���                                    Bxp�t  �          AQ�@�@��R>��?�B�.@�@ٙ�@�z�A�B�aH                                    Bxp�  �          A��>�A(�?��@�B�L�>�@�z�@��HBp�B���                                    Bxp��  �          A��?+�A  ?   @]p�B��q?+�@�@�  Bz�B���                                    Bxp�f  �          Az�?�@�
=?L��@�33B�p�?�@�G�@�(�B	33B��                                    Bxp�  
�          A��@?\)@�
=?5@��B��@?\)@˅@�B �HB��3                                    Bxp��  �          Ap�@��@ٙ�?�\AE�Bd
=@��@�G�@���BBC{                                    Bxp	X  �          A�@���@ۅ?�Q�AW�
Bg��@���@�  @��\Bz�BEff                                    Bxp	�  �          A{@�=q@أ�?���AL(�B_@�=q@��R@��RB(�B=\)                                    Bxp	$�  �          A��@��H@޸R?�G�A'
=Bh33@��H@���@�Q�B=qBJ�                                    Bxp	3J  �          Ap�@��@�=q?�z�A(�Ba
=@��@�\)@��Bz�BC��                                    Bxp	A�  �          A�@�z�@�G�?�=qA.�RBT��@�z�@���@��
B��B4{                                    Bxp	P�  �          A��@���@��?�\)AO�BE�@���@��@�B�B�\                                    Bxp	_<  �          A�@��@�p�?��APz�BE�@��@�p�@�ffB	  B��                                    Bxp	m�  �          A�@��@Ϯ?�
=AV�\BT�R@��@�p�@��BG�B/ff                                    Bxp	|�  �          A�\@��H@ָR@�
Ac
=B^p�@��H@�G�@�(�B�B8�                                    Bxp	�.  �          A�R@,��@�Q�?�z�A�
B���@,��@���@��HB�\B��3                                    Bxp	��  �          A=q?�(�A(�>Ǯ@,��B�=q?�(�@޸R@��RB�B�(�                                    Bxp	�z  
�          A�H?��
A��=���?&ffB���?��
@�@�  A���B��R                                    Bxp	�   �          A
=>�G�A=q>L��?�=qB�  >�G�@�p�@��
A�=qB���                                    Bxp	��  �          A33?uAG�?�@w
=B���?u@�@��B��B�(�                                    Bxp	�l  �          A�R?�Aff>��@N{B��{?�@�=q@��BB���                                    Bxp	�  �          A33?���Az�?Y��@�=qB��?���@�\)@�z�B�B��f                                    Bxp	�  �          A�
?���A
=?��
A
�HB�u�?���@�{@�
=B��B��)                                    Bxp
 ^  �          A��@z�A\)?�(�A33B���@z�@Ϯ@�B��B��                                    Bxp
  �          A��@�A\)?�G�A�B�B�@�@�
=@�
=B\)B�(�                                    Bxp
�  �          A��@(��@�?�  A=��B�(�@(��@�
=@�  B �B�L�                                    Bxp
,P  �          A	�@3�
@��?��AK\)B��@3�
@�(�@��
B#
=B���                                    Bxp
:�  �          A
{?޸RAG�?���@�ffB�\)?޸R@���@��
B�RB���                                    Bxp
I�  �          A	�@B�\@��H?��
A$��B���@B�\@�  @���B\)B|��                                    Bxp
XB  �          A��@k�@�G�?���A*{B�  @k�@��R@��B�Be                                      Bxp
f�  �          A	p�@Y��@��?(�@�33B���@Y��@�@�
=A��\Bx��                                    Bxp
u�  �          A	��@:�HA�>#�
?��B�.@:�H@�z�@��RA��B���                                    Bxp
�4  �          A
{@z�A�?.{@��B��@z�@�  @�  B�\B�{                                    Bxp
��  �          A
ff@%A�\�aG�����B�Ǯ@%@�@W�A�Q�B���                                    Bxp
��  �          A
=q?��
@�(�@8Q�A��B�(�?��
@��H@�
=BG�HB�=q                                    Bxp
�&  �          A33@�@���?!G�@�ffB�(�@�@�(�@��B�\B�Q�                                    Bxp
��  �          A33@��@�Q��33�/�B](�@��@�(�@Q�Ac\)B[(�                                    Bxp
�r  �          A\)@e�@��þ����*�HB��{@e�@���@fffA�
=Bz\)                                    Bxp
�  �          A�@,��A\)�z��w
=B��{@,��@�@j�HA�z�B��R                                    Bxp
�  �          A
�R@��A�H�+���\)B�8R@��@�(�@e�A�=qB��                                    Bxp
�d  �          A��@+�A Q�^�R���B�  @+�@�33@U�A�ffB��H                                    Bxp
  �          Aff@.{@�
=�h���˅B��@.{@�33@I��A��B�
=                                    Bxp�  �          A
=q@(�AQ콏\)��B�(�@(�@���@��A�\B�\)                                    Bxp%V  �          A
=q?��A(�?��
A	B�L�?��@�ff@�=qBffB�p�                                    Bxp3�  �          A
�R?�  A�R?�ffA
�RB��q?�  @ҏ\@�p�B�B�8R                                    BxpB�  �          A33?z�HA�H?�=qAC
=B��q?z�H@��@�(�B*��B�W
                                    BxpQH  �          A
=?E�Az�@�RA��HB�\)?E�@��H@˅B<�RB���                                    Bxp_�  �          A33�@  A=q@�AYG�B�=q�@  @�p�@�G�B0B��H                                    Bxpn�  �          A\)���Az�?��RA�B�\)���@�{@�BB�\)                                    Bxp}:  �          A��n{A?��HAQp�B��n{@�@��RB.�B�8R                                    Bxp��  �          AQ쿐��A�?�AA�B������@��H@�p�B*�B�#�                                    Bxp��  �          Ap��z�HA	p�?��
A9�B�\�z�H@θR@�B(��B�L�                                    Bxp�,  �          A녿��HA
�\?��
A��B��f���H@أ�@�G�BffB�=q                                    Bxp��  �          A��33@�{@�ffA�G�B��Ϳ�33@s33@�p�Br(�B�z�                                    Bxp�x  �          Az���@��
@���Bm�B�Q��녾���A	G�B���C?�                                    Bxp�  �          A�
�#�
@�=q@��HBPz�B�.�#�
?��A	��B�(�B��f                                    Bxp��  T          A�
?�
=@�{@�p�BffB�?�
=@<(�Ap�B�8RB�\)                                    Bxp�j  �          A�?�G�@��H@�B�HB��?�G�@L��@�
=B�#�B�u�                                    Bxp  �          A33?��@��@�p�A�ffB���?��@�ff@���Bp�B��)                                    Bxp�  �          A
�R��G�@��H@hQ�A��HB��\��G�@��H@���B_��B��                                    Bxp\  �          A
=q��ff@��R@P  A��B�z��ff@���@�z�BT��B�                                    Bxp-  �          A
�\��  @�ff@�{B%�Bә���  @
=AG�B���B��                                    Bxp;�  �          A���G�@��H@��HB�RB�aH��G�@FffAB��
B�=q                                    BxpJN  �          A�?��@��@ƸRB6ffB��q?��?�ffA��B�z�B>�R                                    BxpX�  �          A�?^�R@Ϯ@��RB$ffB�?^�R@&ffA  B�B�B�                                      Bxpg�  �          A
�\?޸R@\@��\B*�RB�\?޸R@�A=qB�(�BM�                                    Bxpv@  �          A	��?�Q�@��H@��BQ�B�(�?�Q�@U�@��RB��\B��                                    Bxp��  �          A(�@@  @׮@�A�\B���@@  @e@�=qBb  BH\)                                    Bxp��  �          Az�@fff@�G�@��RA�
=Bp��@fff@C�
@�\Ba�B!ff                                    Bxp�2  �          AQ�@`  @���@�(�A��BuQ�@`  @L��@��B`�B)�                                    Bxp��  �          A��@dz�@�G�@P��A��B|ff@dz�@�=q@�{BC�HBI�R                                    Bxp�~  �          A	G�@4z�@�(�@.{A��B�k�@4z�@�33@�G�B<�BtG�                                    Bxp�$  �          A	G�@^{@��@4z�A��B�G�@^{@���@ƸRB9�
BX33                                    Bxp��  �          A
ff���@�ff�u���33B��)���@�\�.{��
=B���                                    Bxp�p  �          A	����
@����l(���Q�C�����
@�ff�Ǯ�'�C
                                    Bxp�  �          A\)���H@���p����(�C
�����H@˅���\��C�=                                    Bxp�  �          A
=q�.{@�ff��G��	�\B�3�.{A   �8Q�����Bم                                    Bxpb  �          A	��@�������HC^���@�������B�\)                                    Bxp&  �          A	G���{@s33��G��4�C�q��{@�z��J�H��=qB��                                    Bxp4�  �          Az����@j=q��(��/p�CL����@�{�G
=����C ff                                    BxpCT  �          A�����@�����\)���C�����@���� �����\C Y�                                    BxpQ�  �          A  ��33@�p������+��C	�
��33@�=q�/\)���HB�                                    Bxp`�  �          AQ����
@���=��
?z�CT{���
@���@G
=A��HC#�                                    BxpoF  �          A�����@�����
�}p�C	T{���@�p�?k�@ǮC\)                                    Bxp}�  �          A	���
=@�z��)����z�C�f��
=@���>��R@�C\)                                    Bxp��  T          AQ��љ�@�33�Q���
=Cn�љ�@�p���\�\��Cc�                                    Bxp�8  T          A����ff@�33�w
=���C���ff@�\)��  �أ�C	E                                    Bxp��  �          A����{@�Q��;����C
xR��{@���>�=q?�=qC��                                    Bxp��  �          A	�����@�����
=���C�����@�
=���w33B��q                                    Bxp�*  �          A�����@~�R��ff��\Cc����@��
����
C8R                                    Bxp��  "          Az���=q@����XQ���p�Cn��=q@��R�@  ����C�{                                    Bxp�v  �          A  ���@^{�&ff���\C�{���@�=q�Ǯ�'
=Cff                                    Bxp�  �          Az����
@��\�AG���
=C�=���
@��׾�z��Q�Cu�                                    Bxp�  �          A�����@�  �j�H���C�f���@�
=��R��Ch�                                    Bxph  �          A����Q�@���K���\)C����Q�@�p��
=q�h��C�\                                    Bxp  �          A(����@�  ������\C�����@��
��=q�p�C�=                                    Bxp-�  �          Az���ff@�(������HC����ff@�z῎{���C�f                                    Bxp<Z  �          A���p�@��H�.{���Ch���p�@�?!G�@��C��                                    BxpK   �          A
=����@�(��C�
��  C�����@�Q���8Q�Cc�                                    BxpY�  �          A�
��  @o\)��p��C���  @�z���R�Y��C�                                    BxphL  �          A\)��=q@l(���
=�C�R��=q@���У��2=qC��                                    Bxpv�  �          A  ���@A������(�C�����@�33��
�`��C��                                    Bxp��  �          A�
����@1���33��
C������@�G��(���  C�q                                    Bxp�>  �          A�\�O\)@�(�������B��O\)@�ff?�{A�B�#�                                    Bxp��  �          AG��z�A(�>��H@XQ�B��z�@�  @���B�B���                                    Bxp��  T          A{����A=q�aG���  B�#׿���@��@��
A���B�                                      Bxp�0  �          Ap��6ff@�
=��ff�  B܏\�6ff@�@>�RA�B�                                    Bxp��  �          A��8��@����G��^{B�u��8��@��@A��HB��f                                    Bxp�|  �          A�\�.�R@�  �:=q����B�L��.�R@�{?�\)A��B��                                    Bxp�"  �          A33�-p�@�?�33A7
=B��
�-p�@�z�@�
=B$�B�L�                                    Bxp��  �          A
=�z�H@�
=@�RA�G�B���z�H@�{@ʏ\BCz�B�\)                                    Bxp	n  �          A  �+�@�ff@}p�A�\)B�8R�+�@��R@�=qBo  B���                                    Bxp  �          A���;�@�p����H�:�\B�G��;�@���{��33B�#�                                    Bxp&�  �          A��3�
@�
=����BffB�Q��3�
@�
=�,����(�B��                                    Bxp5`  �          A��]p�@p  ��
=�J�C�3�]p�@�\)�Q���=qB�33                                    BxpD  �          A(��!�@�ff@��B�HB����!�@Q�@�{B}�C��                                    BxpR�  �          A�׿�33@�\@>�RA��B�\��33@�(�@�p�BPQ�B�                                    BxpaR  �          A�
���\@�33@G�Ae�B����\@�=q@�z�B8p�Bȳ3                                    Bxpo�  �          A  ��@߮�Y����ffB���@��
@@  A�  B��\                                    Bxp~�  �          A�\��z�@�G���\)��\B�����z�@�@P  A�ffB��                                    Bxp�D  �          A����ff@�p����
�\)B�ff��ff@�{@_\)A�(�C �                                    Bxp��  �          A Q���@ȣ�@�A�33B�k���@�z�@�B%�\CG�                                    Bxp��  �          @�p��3�
@У�@ffA��RB�\�3�
@���@�
=B;�
B��                                    Bxp�6  �          @�z��\(�@θR�У��H��B��\(�@��H@z�A�
B���                                    Bxp��  �          Az���{@�z����\�  C�\��{@�33��Q��=B��f                                    Bxpւ  �          Ap���\)?��R���H�7  C$��\)@�(���z���=qCh�                                    Bxp�(  �          AG����=L����z��?G�C3u����@Tz�����=qC+�                                    Bxp��  �          AG���G�?&ff��z��H��C,�3��G�@~{�����C�                                    Bxpt  �          Ap���p���=q�����D�\C6�\��p�@HQ���ff�$(�C�                                    Bxp  �          A{��
@+����W
C��
@�p���  �{B�\                                    Bxp�  �          A(��fff@��G��Bݽq�fff@�=q����$(�B�G�                                    Bxp.f  �          A{�
=q<#�
����RC3�3�
=q@�����  �Z\)B�{                                    Bxp=  �          A�H���L������C5J=��@�����z��\
=B�                                    BxpK�  �          A\)����@G���Q��5�\CT{����@�33�Z=q���RC�\                                    BxpZX  �          AQ���{@Dz���G��I�Cff��{@��H�x����(�B�8R                                    Bxph�  �          A���I��?p�����Hu�C#W
�I��@�  ���
�8  B�=                                    Bxpw�  �          A
=q�=q�O\)�33��CF� �=q@mp����k��B��f                                    Bxp�J  �          A
�\�33����G�C@���33@�Q�����fB�Ǯ                                    Bxp��  �          A
=q��Ϳ
=���=qCB�����@|(�����i(�B�\)                                    Bxp��  �          A
=q�;��L����
=C7���;�@����p��VB�Q�                                    Bxp�<  �          A	p��(��>�Q�����\C,:��(��@�����z��L�RB�=q                                    Bxp��  �          A(����H�%�����
Cp}q���H@�
� ��33B��q                                    Bxpψ  �          A	p��\)?�{��p���CT{�\)@�=q��{�)z�B�#�                                    Bxp�.  �          A	���'
=@{�����=C�f�'
=@ȣ���Q��{B�B�                                    Bxp��  �          A���5�@%��  �{�RC	s3�5�@�
=��{��\B��f                                    Bxp�z  �          A���c�
@|(���  �J�C\�c�
@���W
=���B�#�                                    Bxp
   �          A	p��xQ�@�G����H�6ffC���xQ�@�33�-p���  B�L�                                    Bxp�  �          A	��  @n{���H�mG�B�W
�  @�\��{��G�B�L�                                    Bxp'l  �          A���(Q�@�����YB�ff�(Q�@��e��Ǚ�B���                                    Bxp6  �          A(��\��@����У��Ip�C��\��@�  �QG�����B�                                    BxpD�  �          A  �^�R@ ����  �mz�CE�^�R@�Q������	�
B�33                                    BxpS^  �          A��޸R@�p������\)B�
=�޸R@�33�z�H��(�BΔ{                                    Bxpb  �          A	���\A ���(Q���ffB�uÿ\A
=@
=Ac33B�{                                    Bxpp�  �          A
{�Q�@�G��3�
��=qB���Q�AG�?���AC33B�Ǯ                                    BxpP  �          A	���*=q@�\�A���p�Bڣ��*=qA Q�?\A#33Bب�                                    Bxp��  �          AQ��{@����H����z�B�33�{A  ?��A!�B�L�                                    Bxp��  �          AQ�(��@�\������Q�B��(��A
=>�@J=qB��                                    Bxp�B  �          A�?��@�{��33�p�B�u�?��AQ�h����
=B�                                      Bxp��  �          Ap�?���@أ������B��?���A
�\�z�H��(�B�.                                    BxpȎ  �          Ap�?�{@�ff��z��{B��?�{A
{�
=�u�B��H                                    Bxp�4  �          Aff@
�H@��H�����"�
B�=q@
�HA33��Q��=qB�33                                    Bxp��  �          A�H@�H@��
�����B  B�� @�HA���-p���Q�B�Ǯ                                    Bxp�  �          A
=?˅@�ff��{�933B�{?˅A��{�g
=B�                                    Bxp&  �          A
=���
@�
=�\)�ԏ\B��\���
A?n{@���B��                                    Bxp�  �          A\)��p�@�\)�����
B�����p�Aff���
��B��R                                    Bxp r  �          A�H?s33@�  ��{�\)B�  ?s33A  ��33��RB���                                    Bxp/  �          A
=?5@�(���
=�\)B�
=?5Aff�W
=���B��f                                    Bxp=�  �          A33?n{@��H�����
=B��\?n{A��.{���B���                                    BxpLd  �          A�?8Q�@����
��Q�B��?8Q�A��>��?z�HB��{                                    Bxp[
  �          A�>aG�@����{���B��\>aG�A33��(��1G�B�33                                    Bxpi�  �          A(��uA (����H��ffB�\)�uA33?aG�@��HB�                                    BxpxV  �          A�׿�=q@��
��ff� G�B��H��=qA
=�L�;��RB�=q                                    Bxp��  �          A{�z�@������H�G�BԽq�z�A�Ϳ333��B�k�                                    Bxp��  �          A��(�A���AG���{B�8R��(�A
�\?��HAH��B�ff                                    Bxp�H  �          A�R��(�A�R�U����B�uÿ�(�A=q?�p�A.�\BÙ�                                    Bxp��  �          A�H��\)A�\� �����\B��)��\)A
=@�A���B��
                                    Bxp��  �          Aff@*=qA33?\(�@�(�B���@*=q@ۅ@�p�Bp�B��{                                    Bxp�:  �          A33@�A�?xQ�@�=qB�aH@�@�{@�33Bp�B�#�                                    Bxp��  �          A(�?ǮA>\@�B��?Ǯ@�R@��B
  B�Q�                                    Bxp�  �          A
=@�A�H>��?У�B��q@�@��
@�G�B\)B�u�                                    Bxp�,  �          A\)?�{Ap�=u>�p�B��?�{@��
@��RB��B��3                                    Bxp
�  �          A�;�\)A�
<#�
=��
B�� ��\)@���@�  B  B�(�                                    Bxpx  �          A�>\)A�H?���A:�HB�8R>\)@У�@�(�B1z�B��\                                    Bxp(  �          A���A�?��HAJ=qBϸR�@�=q@�
=B0ffB�                                      Bxp6�  �          A��?z�A�@B�\A�G�B�W
?z�@�ff@��BO  B�L�                                    BxpEj  �          Ap�?
=qAQ�>k�?�(�B��?
=q@�
=@�=qBp�B��3                                    BxpT  �          A�?�  A(��
=q�VffB�aH?�  A  @C33A��HB�                                      Bxpb�  �          Az�?��A�H���YG�B�p�?��A
=@@  A���B�
=                                    Bxpq\  �          A�?���A��!G��}p�B��?���@�z�@�=qA�RB�(�                                    Bxp�  �          A\)��p�A ���U���{B�aH��p�A	�?��A!�B�W
                                    Bxp��  �          A\)?��A�?fff@�{B�aH?��@�=q@�33BG�B�W
                                    Bxp�N  �          A
=?L��A녿G����B��q?L��@�(�@�z�AܸRB�u�                                    Bxp��  �          A�@xQ�@��R@ ��ALQ�B�
=@xQ�@��
@��B%(�B]ff                                    Bxp��  �          AG��333A  <��
>��B����333@���@���B{B�B�                                    Bxp�@  �          A���p�A\)�%���B��
��p�A(�@��Axz�BǸR                                    Bxp��  �          A?
=A�=��
>�B���?
=@��R@���Bz�B�8R                                    Bxp�  �          A���ǮA33���B�\B��H�Ǯ@�Q�@�ffB  B�Ǯ                                    Bxp�2  �          A녿���Ap�?��HA�BǙ�����@ۅ@�z�B$\)B�Ǯ                                    Bxp�  �          A�\��A��?���@�33B��)��@޸R@���B=qB�33                                    Bxp~  �          Aff���HA��?\(�@�  B�����H@��
@�  B��B��=                                    Bxp!$  �          A{��ffAff?�G�@�B�8R��ff@���@�G�B��B�33                                    Bxp/�  �          A��(�A�?�ff@�  B�aH�(�@߮@��BffB�Ǯ                                    Bxp>p  �          A{����A{?\AffB�𤿙��@ۅ@�
=B'{B���                                    BxpM  �          A{���
AQ�?�G�A-��B�LͿ��
@�(�@˅B+�B���                                    Bxp[�  �          Ap��   AG�@�
Ad��B�z��   @�@ָRB7B���                                    Bxpjb  �          A��A�?�33A<��B���@���@��
B-p�Bخ                                    Bxpy  �          AG���\)A��?��A"�\B�=q��\)@׮@���B)��B�\                                    Bxp��  T          AG��aG�A��?�G�A.{B��aG�@ָR@���B-��B��                                    Bxp�T  �          A�Ϳ�  A��?�A7
=B����  @��
@�{B/�\B�B�                                    Bxp��  T          AG����A�@{Au�B��쿧�@�(�@��
B>ffB�(�                                    Bxp��  T          A���{A ��@ffAj{B��
��{@�z�@��
B=33B��                                    Bxp�F  �          A?�ffA녿   �C�
B���?�ff@�{@�Q�A��
B�k�                                    Bxp��  �          A?uA33��=q��B��?uA�@��AθRB��)                                    Bxpߒ  �          A녿˅A�@p  A�=qBɏ\�˅@�Q�@�B^\)B�G�                                    Bxp�8  �          A
=����Aff@�=qA͙�B�zῈ��@��
@��RBg�B̽q                                    Bxp��  �          A�\���
A�\@:=qA�z�B�
=���
@�@�Q�BK  B��\                                    Bxp�  �          A
==���Ap�@!G�Ax  B�=���@�G�@���B@��B�8R                                    Bxp*  �          A\)>��A��@'�A��HB��>��@ƸR@�33BC33B�=q                                    Bxp(�  �          A��n{A�R@;�A�z�B��
�n{@�@���BJ  BŽq                                    Bxp7v  �          A�H��A\)@L��A�{B�\)��@�33@���BQ
=Bˮ                                    BxpF  �          A
=��
=AQ�@G�Ac\)B����
=@���@���B7�B���                                    BxpT�  T          A33�,��A�?�\)A733B�
=�,��@�p�@��HB)33B��                                    Bxpch  �          A33�EAQ�@X��A�=qB����E@�(�@�=qBJ�B��                                    Bxpr  �          Aff���H@Ӆ@�\)A�
=B�B����H@Q�@�(�BF�C:�                                    Bxp��  �          Aff��Q�@��@`��A�G�C�)��Q�@Z�H@˅B*\)C\)                                    Bxp�Z  �          A{����@���@��A�p�C������@*=q@�  BKG�C5�                                    Bxp�   �          A=q��@��H@�\)Bz�C&f��?��H@���B^��CE                                    Bxp��  �          A=q�`  @�
=@���B��B�Ǯ�`  @�AB}33CO\                                    Bxp�L  �          Ap��"�\@��@R�\A�B��"�\@��H@�Q�BP33B�G�                                    Bxp��  �          A���0  @�33@�(�B�
B�ff�0  @E�AB~C�=                                    Bxpؘ  	N          Aff�$z�@�@�
=B��B�L��$z�@FffA�B�=qC��                                    Bxp�>  
�          A��<(�@߮@�(�B=qBី�<(�@7
=A(�B��=C�
                                    Bxp��  "          Az���33@�  @���A���B���33@P��@���Ba��C�                                    Bxp�  �          Az��mp�@�33@���B=qB���mp�@ffA  B}p�Ch�                                    Bxp0  
�          Az����\@�p�@��
A�z�C�{���\@,��@�
=BIz�C�                                    Bxp!�  �          A�����@��@fffA�C����@P  @˅B'\)C�                                     Bxp0|  �          A  ��\@�G�@]p�A��C�=��\@Q�@�(�B�C!k�                                    Bxp?"  �          A(�� ��@xQ�@O\)A�p�CO\� ��?\@�=qA�C)T{                                    BxpM�  "          A�� z�@�{@ ��Atz�C�� z�@��@�  A�(�C#h�                                    Bxp\n  �          A����@p�?�z�A	p�C&O\���?���@Q�Aip�C-{                                    Bxpk  �          A���˅@�G�@��
BCxR�˅?�=q@�Q�B9=qC*h�                                    Bxpy�  
�          AG���G�A Q�?�(�A�\B���G�@�
=@���B��C ��                                    Bxp�`  �          A�R�`  @�
=@�
=B�B��`  @
=qA�RB}{CQ�                                    Bxp�  �          Aff����@���A�\BrB�Ǯ�����fffAffB�B�CL�\                                    Bxp��  T          A�E@��\@�\)Ba\)B�\�E�&ffA��B��{C?ٚ                                    Bxp�R  
�          A��U�@�  @��BF\)B����U�?   A�B��)C+��                                    Bxp��  
�          A���  @�{@�{B3  B����  ?}p�A	�B���C&&f                                    Bxpў  �          A(�����@�(�@��B?CO\������z�A=qBm��C7}q                                    Bxp�D  
�          AQ���z�@�\)@��
B0�C  ��z�?Y��AffBw�HC)�                                    Bxp��  T          Az����@~{@߮B=�\C�
��녾��A z�Bf�HC8�
                                    Bxp��  
�          A�����@��
@�=qB-p�C�R����u@�  BXz�C4�{                                    Bxp6  
�          A(�����@���@�p�B*�HC^����ͼ�@��HBUQ�C4T{                                    Bxp�  t          A���33@�@��
B�C	T{��33?�  @�33BY��C)\)                                    Bxp)�  
(          Aff�`  A�
@8Q�A�\)B��)�`  @��
@߮B:33B��                                    Bxp8(  
�          A\)�j�HA�R@C�
A��B�(��j�H@��R@�33B<��B��f                                    BxpF�  "          A���*=qA   ������
B��*=q@��@Dz�A�Q�B��f                                    BxpUt  T          A�R��Q�AG���{��B�  ��Q�A�>.{?}p�B�(�                                    Bxpd  "          A��\(�A��mp���\)B���\(�A(�?�=q@�=qB�Ǯ                                    